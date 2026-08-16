#!/usr/bin/env python3
"""
Explore and visualize the 2018 IEEE GRSS Data Fusion Challenge dataset.

Extracts metadata, class distribution, and alignment info from GT, RGB, HSI, and LiDAR.
Saves visualizations to output/grss_exploration/.

Usage:
    conda activate flow  # or: pip install -r requirements_grss_explore.txt
    python scripts/explore_grss_dataset.py --root ImageryAndTrainingGT --out output/grss_exploration

Dependencies: numpy, matplotlib, PIL (Pillow)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Use Agg backend for headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# =============================================================================
# ENVI format helpers
# =============================================================================

ENVI_DTYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def parse_envi_hdr(hdr_path: str) -> Dict[str, Any]:
    """Parse ENVI .hdr file into a dict."""
    meta: Dict[str, Any] = {}
    with open(hdr_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        # Handle multi-line values (e.g. description = { ... })
        if "=" in line:
            key, _, rest = line.partition("=")
            key = key.strip().lower().replace(" ", "_")
            val = rest.strip()
            if val.startswith("{"):
                # Multi-line: collect until closing }
                val = val[1:]
                while "}" not in val and i + 1 < len(lines):
                    i += 1
                    val += " " + lines[i].strip()
                val = val.split("}")[0].strip()
            else:
                val = val.strip("{}").strip()
            # Store with original key names for known fields
            if key == "samples":
                meta["samples"] = int(val) if val.isdigit() else 0
            elif key == "lines":
                meta["lines"] = int(val) if val.isdigit() else 0
            elif key == "bands":
                meta["bands"] = int(val) if val.isdigit() else 0
            elif key == "data_type":
                meta["data_type"] = int(val) if val.isdigit() else 4
            elif key == "interleave":
                meta["interleave"] = val.lower()
            elif key == "header_offset":
                meta["header_offset"] = int(val) if val.isdigit() else 0
            elif key == "map_info":
                meta["map_info"] = val
                parts = [p.strip() for p in val.split(",")]
                if len(parts) >= 6:
                    try:
                        meta["utm_x_ul"] = float(parts[3])
                        meta["utm_y_ul"] = float(parts[4])
                        meta["pixel_size_x"] = float(parts[5])
                        meta["pixel_size_y"] = float(parts[6])
                    except (ValueError, IndexError):
                        pass
            elif key == "x_start":
                meta["x_start"] = int(val) if val.isdigit() else 0
            elif key == "y_start":
                meta["y_start"] = int(val) if val.isdigit() else 0
            elif key == "wavelength":
                meta["wavelength"] = val
        i += 1

    return meta


def load_envi(data_path: str, hdr_path: Optional[str] = None, roi: Optional[Tuple[slice, slice]] = None) -> np.ndarray:
    """
    Load ENVI binary file. If roi=(rows, cols) slice is given, load only that region
    (for large files like HSI).
    """
    if hdr_path is None:
        hdr_path = data_path + ".hdr" if not data_path.endswith(".hdr") else data_path[:-4] + ".hdr"
    meta = parse_envi_hdr(hdr_path)
    samples, lines, bands = meta["samples"], meta["lines"], meta["bands"]
    dtype = ENVI_DTYPES.get(meta["data_type"], np.float32)
    interleave = meta.get("interleave", "bsq")
    offset = meta.get("header_offset", 0)
    el_size = np.dtype(dtype).itemsize

    with open(data_path, "rb") as f:
        f.seek(offset)
        if interleave == "bip":
            # band interleaved by pixel: (lines, samples, bands)
            if roi is None:
                data = np.fromfile(f, dtype=dtype, count=lines * samples * bands)
                return data.reshape(lines, samples, bands)
            row_slice, col_slice = roi
            start_row = row_slice.start or 0
            end_row = row_slice.stop or lines
            start_col = col_slice.start or 0
            end_col = col_slice.stop or samples
            n_rows, n_cols = end_row - start_row, end_col - start_col
            row_size = samples * bands
            f.seek(offset + start_row * row_size * el_size)
            data = np.fromfile(f, dtype=dtype, count=n_rows * row_size)
            arr = data.reshape(n_rows, samples, bands)
            return arr[:, start_col:end_col, :].copy()
        elif interleave == "bsq":
            # band sequential: (bands, lines, samples)
            if roi is None:
                data = np.fromfile(f, dtype=dtype, count=bands * lines * samples)
                arr = data.reshape(bands, lines, samples)
                return np.transpose(arr, (1, 2, 0))
            row_slice, col_slice = roi
            data = np.fromfile(f, dtype=dtype, count=bands * lines * samples)
            arr = data.reshape(bands, lines, samples)
            out = arr[:, row_slice, col_slice]
            return np.transpose(out, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported interleave: {interleave}")


# =============================================================================
# GeoTIFF / World file helpers (PIL-based, no rasterio)
# =============================================================================

def parse_tfw(tfw_path: str) -> Dict[str, float]:
    """Parse .tfw world file. Returns pixel_size_x, pixel_size_y, x_ul, y_ul."""
    with open(tfw_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    return {
        "pixel_size_x": float(lines[0]),
        "rotation_1": float(lines[1]),
        "rotation_2": float(lines[2]),
        "pixel_size_y": float(lines[3]),
        "x_ul": float(lines[4]),
        "y_ul": float(lines[5]),
    }


def load_geotiff(path: str, crop: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Load GeoTIFF via PIL. crop = (x0, y0, x1, y1) in pixels, or None for full.
    Returns (array, tfw_meta or None).
    """
    base = path.rsplit(".tif", 1)[0].rsplit(".tiff", 1)[0]
    tfw_path = base + ".tfw"
    meta = parse_tfw(tfw_path) if os.path.exists(tfw_path) else None

    # Avoid DecompressionBombWarning for large images
    Image.MAX_IMAGE_PIXELS = None

    im = Image.open(path)
    arr = np.array(im)
    if crop is not None:
        x0, y0, x1, y1 = crop
        arr = arr[y0:y1, x0:x1]
    return arr, meta


# =============================================================================
# Dataset paths and loading
# =============================================================================

def find_dataset_paths(root: str) -> Dict[str, Any]:
    """Discover paths for GT, RGB tiles, HSI, LiDAR."""
    root = Path(root)
    phase2 = root / "2018IEEE_Contest" / "Phase2"

    paths = {
        "gt_tif": phase2 / "TrainingGT" / "2018_IEEE_GRSS_DFC_GT_TR.tif",
        "gt_tfw": phase2 / "TrainingGT" / "2018_IEEE_GRSS_DFC_GT_TR.tfw",
        "hsi_pix": phase2 / "FullHSIDataset" / "20170218_UH_CASI_S4_NAD83.pix",
        "hsi_hdr": phase2 / "FullHSIDataset" / "20170218_UH_CASI_S4_NAD83.hdr",
    }

    rgb_dir = phase2 / "Final RGB HR Imagery"
    paths["rgb_tiles"] = sorted(rgb_dir.glob("*.tif")) if rgb_dir.exists() else []

    lidar_dir = phase2 / "Lidar GeoTiff Rasters"
    dem_dir = lidar_dir / "DEM_C123_3msr"
    dsm_dir = lidar_dir / "DSM_C12"
    intensity_dir = lidar_dir / "Intensity_C1"
    paths["lidar_dem"] = list(dem_dir.glob("*.tif"))[0] if dem_dir.exists() else None
    paths["lidar_dsm"] = list(dsm_dir.glob("*.tif"))[0] if dsm_dir.exists() else None
    paths["lidar_intensity"] = list(intensity_dir.glob("*.tif"))[0] if intensity_dir.exists() else None

    return {k: str(v) if isinstance(v, Path) else [str(p) for p in v] if isinstance(v, list) else v
            for k, v in paths.items()}


def load_gt(paths: Dict) -> Tuple[np.ndarray, Dict]:
    """Load ground truth labels."""
    arr, meta = load_geotiff(paths["gt_tif"])
    return arr, meta or parse_tfw(paths["gt_tfw"])


def load_hsi_crop(paths: Dict, max_size: int = 1200) -> Tuple[np.ndarray, Dict]:
    """Load HSI - full if small enough, else center crop for memory."""
    meta = parse_envi_hdr(paths["hsi_hdr"])
    lines, samples, bands = meta["lines"], meta["samples"], meta["bands"]

    if lines <= max_size and samples <= max_size:
        arr = load_envi(paths["hsi_pix"], paths["hsi_hdr"])
    else:
        # Center crop
        r0 = (lines - max_size) // 2
        c0 = (samples - max_size) // 2
        arr = load_envi(paths["hsi_pix"], paths["hsi_hdr"], roi=(slice(r0, r0 + max_size), slice(c0, c0 + max_size)))
    return arr, meta


def _find_overlapping_rgb_tile(tiles: List[str], gt_meta: Dict) -> Optional[int]:
    """Find RGB tile whose top-left is closest to the GT top-left (best overlap)."""
    gt_x = gt_meta.get("x_ul")
    gt_y = gt_meta.get("y_ul")
    if gt_x is None or gt_y is None:
        return 0
    best_i, best_d = 0, float("inf")
    for i, tpath in enumerate(tiles):
        tfw = tpath.rsplit(".tif", 1)[0] + ".tfw"
        if not os.path.exists(tfw):
            continue
        tm = parse_tfw(tfw)
        tx, ty = tm.get("x_ul"), tm.get("y_ul")
        if tx is None or ty is None:
            continue
        d = (tx - gt_x) ** 2 + (ty - gt_y) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def load_rgb_crop(paths: Dict, tile_idx: Optional[int] = None, max_pixels: int = 4000,
                  gt_meta: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """Load an RGB tile. If gt_meta is provided, pick the tile that overlaps the GT extent."""
    tiles = paths.get("rgb_tiles", [])
    if not tiles:
        return np.zeros((100, 100, 3), dtype=np.uint8), {}

    if tile_idx is None and gt_meta:
        tile_idx = _find_overlapping_rgb_tile(tiles, gt_meta)
    if tile_idx is None:
        tile_idx = 0

    path = tiles[min(tile_idx, len(tiles) - 1)]
    im = Image.open(path)
    w_orig, h_orig = im.size
    if w_orig * h_orig > max_pixels * max_pixels:
        scale = (max_pixels * max_pixels / (w_orig * h_orig)) ** 0.5
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
    arr = np.array(im)
    tfw_path = path.rsplit(".tif", 1)[0] + ".tfw"
    meta = parse_tfw(tfw_path) if os.path.exists(tfw_path) else {}
    # Store extent in m so overlay can compute effective pixel size after resize
    psx = abs(meta.get("pixel_size_x", 0.05))
    psy = abs(meta.get("pixel_size_y", 0.05))
    meta["extent_x_m"] = w_orig * psx
    meta["extent_y_m"] = h_orig * psy
    return arr, meta


def load_lidar_dem(paths: Dict) -> Tuple[np.ndarray, Dict]:
    """Load LiDAR DEM."""
    path = paths.get("lidar_dem")
    if not path or not os.path.exists(path):
        return np.zeros((100, 100), dtype=np.float32), {}
    arr, meta = load_geotiff(path)
    # Clip no-data (often very large values)
    if arr.dtype == np.float32:
        valid = np.isfinite(arr) & (arr < 1e10) & (arr > -1e4)
        arr = np.where(valid, arr, np.nan)
    return arr, meta or {}


# =============================================================================
# Extraction
# =============================================================================

def extract_dataset_info(paths: Dict) -> Dict[str, Any]:
    """Extract metadata, class stats, and alignment info."""
    info: Dict[str, Any] = {"paths": paths}

    # GT
    gt_arr, gt_meta = load_gt(paths)
    info["gt"] = {
        "shape": list(gt_arr.shape),
        "dtype": str(gt_arr.dtype),
        "unique_classes": [int(x) for x in np.unique(gt_arr.ravel())],
        "meta": gt_meta,
    }
    unique, counts = np.unique(gt_arr.ravel(), return_counts=True)
    info["gt"]["class_counts"] = {int(u): int(c) for u, c in zip(unique, counts)}
    info["gt"]["total_pixels"] = int(gt_arr.size)

    # HSI
    try:
        hsi_arr, hsi_meta = load_hsi_crop(paths)
        info["hsi"] = {
            "shape": list(hsi_arr.shape),
            "dtype": str(hsi_arr.dtype),
            "bands": hsi_meta.get("bands", hsi_arr.shape[-1]),
            "wavelength_nm": hsi_meta.get("wavelength", ""),
            "meta": {k: v for k, v in hsi_meta.items() if k in ("samples", "lines", "utm_x_ul", "utm_y_ul", "pixel_size_x")},
        }
    except Exception as e:
        info["hsi"] = {"error": str(e)}

    # RGB (pick tile overlapping GT for overlay)
    try:
        rgb_arr, rgb_meta = load_rgb_crop(paths, gt_meta=info["gt"]["meta"])
        info["rgb"] = {
            "shape": list(rgb_arr.shape),
            "dtype": str(rgb_arr.dtype),
            "num_tiles": len(paths.get("rgb_tiles", [])),
            "meta": rgb_meta,
        }
    except Exception as e:
        info["rgb"] = {"error": str(e)}

    # LiDAR
    try:
        dem_arr, dem_meta = load_lidar_dem(paths)
        info["lidar"] = {
            "shape": list(dem_arr.shape),
            "dtype": str(dem_arr.dtype),
            "meta": dem_meta,
        }
    except Exception as e:
        info["lidar"] = {"error": str(e)}

    return info


# =============================================================================
# Visualizations
# =============================================================================

def vis_gt_labels(gt: np.ndarray, out_path: str, class_names: Optional[Dict[int, str]] = None) -> None:
    """Save GT label map with colormap."""
    fig, ax = plt.subplots(figsize=(14, 6))
    n_classes = int(gt.max()) + 1
    try:
        cmap = matplotlib.colormaps["tab20"].resampled(max(20, n_classes))
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20", max(20, n_classes))
    im = ax.imshow(gt, cmap=cmap, vmin=0, vmax=max(19, n_classes - 1), interpolation="nearest")
    labels = [(class_names or {}).get(i, f"Class {i}") for i in range(n_classes)]
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_classes), label="Class")
    cbar.ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Ground Truth Labels (2018 IEEE GRSS DFC)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Line")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def vis_class_distribution(class_counts: Dict[int, int], out_path: str) -> None:
    """Bar chart of pixel counts per class."""
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([str(c) for c in classes], counts, color=plt.cm.tab20(np.linspace(0, 1, len(classes))))
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Pixel count")
    ax.set_title("Class Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def vis_rgb_sample(rgb: np.ndarray, out_path: str) -> None:
    """Save RGB composite."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb)
    ax.set_title("RGB Imagery (sample tile)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def vis_gt_over_rgb(gt: np.ndarray, rgb: np.ndarray, out_path: str,
                    gt_meta: Optional[Dict] = None, rgb_meta: Optional[Dict] = None) -> None:
    """
    Overlay GT on RGB. Crop both to overlapping UTM region so alignment is correct.
    """
    if gt_meta and rgb_meta:
        # Compute overlapping region in UTM, crop both to overlap
        gt_x, gt_y = gt_meta.get("x_ul"), gt_meta.get("y_ul")
        rgb_x, rgb_y = rgb_meta.get("x_ul"), rgb_meta.get("y_ul")
        gt_psx, gt_psy = abs(gt_meta.get("pixel_size_x", 0.5)), abs(gt_meta.get("pixel_size_y", 0.5))
        h_gt, w_gt = gt.shape
        h_rgb, w_rgb = rgb.shape[:2]
        # RGB may be downsampled; use extent in m to get effective pixel size
        ext_x = rgb_meta.get("extent_x_m", w_rgb * 0.05)
        ext_y = rgb_meta.get("extent_y_m", h_rgb * 0.05)
        rgb_psx, rgb_psy = ext_x / w_rgb, ext_y / h_rgb
        # Overlap in UTM: x from max(gt_x, rgb_x) to min(gt_x+w_gt*gt_psx, rgb_x+w_rgb*rgb_psx)
        gt_x_max = gt_x + w_gt * gt_psx
        gt_y_min = gt_y - h_gt * gt_psy
        rgb_x_max = rgb_x + w_rgb * rgb_psx
        rgb_y_min = rgb_y - h_rgb * rgb_psy
        x_min = max(gt_x, rgb_x)
        x_max = min(gt_x_max, rgb_x_max)
        y_min = max(gt_y_min, rgb_y_min)
        y_max = min(gt_y, rgb_y)
        if x_max > x_min and y_max > y_min:
            # Crop GT and RGB to overlap
            gt_c0 = int((x_min - gt_x) / gt_psx)
            gt_c1 = int((x_max - gt_x) / gt_psx)
            gt_r0 = int((gt_y - y_max) / gt_psy)
            gt_r1 = int((gt_y - y_min) / gt_psy)
            rgb_c0 = int((x_min - rgb_x) / rgb_psx)
            rgb_c1 = int((x_max - rgb_x) / rgb_psx)
            rgb_r0 = int((rgb_y - y_max) / rgb_psy)
            rgb_r1 = int((rgb_y - y_min) / rgb_psy)
            gt_crop = gt[gt_r0:gt_r1, gt_c0:gt_c1]
            rgb_crop = rgb[rgb_r0:rgb_r1, rgb_c0:rgb_c1]
            rgb_resized = np.array(Image.fromarray(rgb_crop).resize(
                (gt_crop.shape[1], gt_crop.shape[0]), Image.Resampling.BILINEAR))
        else:
            gt_crop = gt
            rgb_resized = np.array(Image.fromarray(rgb).resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR))
    else:
        gt_crop = gt
        rgb_resized = np.array(Image.fromarray(rgb).resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(rgb_resized)
    # Overlay GT as semi-transparent
    try:
        cmap = matplotlib.colormaps["tab20"].resampled(20)
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20", 20)
    gt_overlay = np.ma.masked_where(gt_crop == 0, gt_crop)
    ax.imshow(gt_overlay, cmap=cmap, vmin=0, vmax=19, alpha=0.4, interpolation="nearest")
    ax.set_title("GT Labels Overlaid on RGB")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def vis_hsi_falsecolor(hsi: np.ndarray, hdr_meta: Dict, out_path: str) -> None:
    """False color: R=NIR-like, G=red, B=green. Bands 0-47 are spectral; use 31, 16, 8 as proxy."""
    n_bands = hsi.shape[-1]
    r_band = min(31, n_bands - 1)  # ~818 nm
    g_band = min(16, n_bands - 1)  # ~589 nm
    b_band = min(8, n_bands - 1)   # ~474 nm
    r = np.clip(hsi[:, :, r_band].astype(float) / (np.nanpercentile(hsi[:, :, r_band], 99) + 1e-6), 0, 1)
    g = np.clip(hsi[:, :, g_band].astype(float) / (np.nanpercentile(hsi[:, :, g_band], 99) + 1e-6), 0, 1)
    b = np.clip(hsi[:, :, b_band].astype(float) / (np.nanpercentile(hsi[:, :, b_band], 99) + 1e-6), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(rgb)
    ax.set_title(f"HSI False Color (R: band {r_band}, G: band {g_band}, B: band {b_band})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def vis_lidar_dem(dem: np.ndarray, out_path: str) -> None:
    """DEM as elevation colormap."""
    valid = np.isfinite(dem) & (dem > -100) & (dem < 1e6)
    vmin, vmax = np.nanpercentile(dem[valid], 1), np.nanpercentile(dem[valid], 99)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(np.where(valid, dem, np.nan), cmap="terrain", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Elevation (m)")
    ax.set_title("LiDAR DEM")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Line")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def vis_alignment_grid(gt: np.ndarray, rgb: np.ndarray, hsi: np.ndarray, dem: np.ndarray, out_path: str) -> None:
    """2x2 grid: GT, RGB, HSI falsecolor, LiDAR DEM."""
    h_gt, w_gt = gt.shape

    def _safe_rgb(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.zeros((h_gt, w_gt, 3), dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        r = np.clip(arr.astype(float) / max(arr.max(), 1), 0, 1) * 255
        return np.array(Image.fromarray(r.astype(np.uint8)).resize((w_gt, h_gt), Image.Resampling.BILINEAR))

    rgb_small = _safe_rgb(rgb) if rgb.size > 0 else np.zeros((h_gt, w_gt, 3), dtype=np.uint8)
    if hsi.size > 0 and hsi.ndim == 3:
        r_band = min(31, hsi.shape[-1] - 1)
        g_band = min(16, hsi.shape[-1] - 1)
        b_band = min(8, hsi.shape[-1] - 1)
        hsi_rgb = np.stack([hsi[:, :, r_band], hsi[:, :, g_band], hsi[:, :, b_band]], axis=-1)
        hsi_rgb = np.clip(hsi_rgb.astype(float) / (np.nanpercentile(hsi_rgb, 99) + 1e-6), 0, 1) * 255
        hsi_small = np.array(Image.fromarray(hsi_rgb.astype(np.uint8)).resize((w_gt, h_gt), Image.Resampling.BILINEAR))
    else:
        hsi_small = np.zeros((h_gt, w_gt, 3), dtype=np.uint8)
    if dem.size > 0:
        valid = np.isfinite(dem) & (dem > -100) & (dem < 1e6)
        vmin, vmax = np.nanpercentile(dem[valid], 1), np.nanpercentile(dem[valid], 99)
        dem_norm = np.where(valid, (dem - vmin) / (vmax - vmin + 1e-9), 0)
        dem_small = np.array(Image.fromarray((dem_norm * 255).astype(np.uint8)).resize((w_gt, h_gt), Image.Resampling.BILINEAR))
    else:
        dem_small = np.zeros((h_gt, w_gt), dtype=np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes[0, 0].imshow(gt, cmap="tab20", vmin=0, vmax=19)
    axes[0, 0].set_title("GT Labels")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_small)
    axes[0, 1].set_title("RGB")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(hsi_small)
    axes[1, 0].set_title("HSI False Color")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(dem_small, cmap="terrain")
    axes[1, 1].set_title("LiDAR DEM")
    axes[1, 1].axis("off")

    plt.suptitle("Alignment Check (resampled to GT grid)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def vis_spectral_signatures(hsi: np.ndarray, gt: np.ndarray, hsi_meta: Dict, out_path: str, max_classes: int = 10) -> None:
    """
    Mean spectral signature per class. HSI and GT must overlap - we use a crop where both exist.
    """
    h_hsi, w_hsi, n_bands = hsi.shape
    h_gt, w_gt = gt.shape

    # Use overlapping region
    h_min = min(h_hsi, h_gt)
    w_min = min(w_hsi, w_gt)
    gt_crop = gt[:h_min, :w_min]
    # HSI may have different extent - assume same origin for central overlap
    hsi_crop = hsi[:h_min, :w_min, :] if hsi.shape[0] >= h_min and hsi.shape[1] >= w_min else hsi[:h_min, :w_min, :]

    classes = np.unique(gt_crop.ravel())
    classes = classes[classes > 0][:max_classes]  # skip 0 (unclass), limit for clarity

    fig, ax = plt.subplots(figsize=(10, 5))
    wavelengths = list(range(n_bands))  # placeholder; could parse from hsi_meta

    for c in classes:
        mask = gt_crop == c
        if mask.sum() < 10:
            continue
        spec = hsi_crop[mask, :].mean(axis=0)
        ax.plot(wavelengths, spec, label=f"Class {c}", alpha=0.8)

    ax.set_xlabel("Band index")
    ax.set_ylabel("Mean reflectance (DN)")
    ax.set_title("Spectral Signatures per Class (sample)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Explore 2018 IEEE GRSS Data Fusion Challenge dataset")
    parser.add_argument("--root", type=str, default="ImageryAndTrainingGT", help="Root path to ImageryAndTrainingGT")
    parser.add_argument("--out", type=str, default="output/grss_exploration", help="Output directory")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = Path.cwd() / root
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve path - may be at repo root or inside 2018IEEE_Contest
    if (root / "2018IEEE_Contest").exists():
        pass
    elif (root.parent / "2018IEEE_Contest").exists():
        root = root.parent
    else:
        # Try from cwd
        alt = Path.cwd() / "ImageryAndTrainingGT"
        if (alt / "2018IEEE_Contest").exists():
            root = alt

    paths = find_dataset_paths(str(root))
    print("Paths:", json.dumps({k: v if isinstance(v, str) else f"<{len(v) if isinstance(v, list) else 0} items>" for k, v in paths.items()}, indent=2))

    # Extract info
    print("Extracting dataset info...")
    info = extract_dataset_info(paths)
    def _json_serializer(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    summary_path = out_dir / "dataset_summary.json"
    dumpable = {k: v for k, v in info.items() if k != "paths"}
    with open(summary_path, "w") as f:
        json.dump(dumpable, f, indent=2, default=_json_serializer)
    print(f"Saved {summary_path}")

    # Load data for viz (use RGB tile that overlaps GT for correct overlay)
    gt_arr, gt_meta = load_gt(paths)
    rgb_arr, rgb_meta = load_rgb_crop(paths, gt_meta=gt_meta)
    hsi_arr, hsi_meta = load_hsi_crop(paths)
    dem_arr, _ = load_lidar_dem(paths)

    # Visualizations
    print("Generating visualizations...")
    vis_gt_labels(gt_arr, str(out_dir / "01_gt_labels.png"))
    vis_class_distribution(info["gt"]["class_counts"], str(out_dir / "02_class_distribution.png"))
    vis_rgb_sample(rgb_arr, str(out_dir / "03_rgb_sample.png"))
    vis_gt_over_rgb(gt_arr, rgb_arr, str(out_dir / "04_gt_over_rgb.png"),
                    gt_meta=gt_meta, rgb_meta=rgb_meta)
    vis_hsi_falsecolor(hsi_arr, hsi_meta, str(out_dir / "05_hsi_falsecolor.png"))
    vis_lidar_dem(dem_arr, str(out_dir / "06_lidar_dem.png"))
    vis_alignment_grid(gt_arr, rgb_arr, hsi_arr, dem_arr, str(out_dir / "07_alignment_grid.png"))

    # Spectral signatures - need overlapping HSI and GT; HSI has different extent
    try:
        vis_spectral_signatures(hsi_arr, gt_arr, hsi_meta, str(out_dir / "08_spectral_signatures.png"))
    except Exception as e:
        print(f"Skipping spectral signatures: {e}")

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
