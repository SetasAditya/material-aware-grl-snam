from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, sobel

from .ontology import RellisOntology


@dataclass(frozen=True)
class BevConfig:
    x_min: float = -5.0
    x_max: float = 45.0
    y_min: float = -25.0
    y_max: float = 25.0
    resolution: float = 0.5
    min_points_per_cell: int = 1
    risk_sigma_cells: float = 1.0
    hard_inflate_cells: int = 1
    unknown_inflate_cells: int = 0

    @property
    def rows(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    @property
    def cols(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


def xy_to_rc(x: float, y: float, cfg: BevConfig) -> Tuple[int, int]:
    c = int(np.floor((float(x) - cfg.x_min) / cfg.resolution))
    r = int(np.floor((cfg.y_max - float(y)) / cfg.resolution))
    return r, c


def rc_to_xy(r: float, c: float, cfg: BevConfig) -> Tuple[float, float]:
    x = cfg.x_min + (float(c) + 0.5) * cfg.resolution
    y = cfg.y_max - (float(r) + 0.5) * cfg.resolution
    return x, y


def _modal_labels(labels: np.ndarray, linear: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    flat_size = rows * cols
    z = np.zeros(flat_size, dtype=np.uint16)
    for cell in np.unique(linear):
        idx = linear == cell
        vals, counts = np.unique(labels[idx], return_counts=True)
        z[int(cell)] = vals[np.argmax(counts)]
    return z.reshape(rows, cols)


def build_bev_maps(
    points: np.ndarray,
    labels: np.ndarray,
    ontology: RellisOntology,
    cfg: BevConfig,
) -> Dict[str, np.ndarray]:
    rows, cols = cfg.rows, cfg.cols
    x = points[:, 0]
    y = points[:, 1]
    valid = (
        (x >= cfg.x_min)
        & (x < cfg.x_max)
        & (y >= cfg.y_min)
        & (y < cfg.y_max)
    )
    x = x[valid]
    y = y[valid]
    lab = labels[valid]

    c = np.floor((x - cfg.x_min) / cfg.resolution).astype(np.int64)
    r = np.floor((cfg.y_max - y) / cfg.resolution).astype(np.int64)
    in_grid = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    r = r[in_grid]
    c = c[in_grid]
    lab = lab[in_grid]
    linear = r * cols + c

    point_count = np.bincount(linear, minlength=rows * cols).reshape(rows, cols).astype(np.int32)
    observed = point_count >= int(cfg.min_points_per_cell)
    z2_labels = _modal_labels(lab, linear, (rows, cols)) if linear.size else np.zeros((rows, cols), np.uint16)

    rho_lookup = np.full(65536, ontology.void_risk, dtype=np.float32)
    for idx, rho in ontology.rho_by_id.items():
        rho_lookup[int(idx)] = float(rho)
    risk_raw = rho_lookup[z2_labels]
    risk_raw = np.where(observed, risk_raw, ontology.void_risk).astype(np.float32)
    risk_map = gaussian_filter(risk_raw, sigma=float(cfg.risk_sigma_cells)).astype(np.float32)
    risk_map = np.clip(risk_map, 0.0, 1.0)

    hard_lookup = np.zeros(65536, dtype=bool)
    for idx in ontology.hard_ids:
        hard_lookup[int(idx)] = True
    hard_mask = hard_lookup[z2_labels] & observed
    unknown_mask = ~observed if ontology.unknown_is_obstacle else np.zeros_like(observed, dtype=bool)
    if cfg.unknown_inflate_cells > 0:
        unknown_mask = binary_dilation(unknown_mask, iterations=int(cfg.unknown_inflate_cells))
    hard_mask = hard_mask | unknown_mask
    if cfg.hard_inflate_cells > 0:
        hard_mask = binary_dilation(hard_mask, iterations=int(cfg.hard_inflate_cells))
    hard_mask = hard_mask.astype(np.uint8)

    soft_lookup = np.zeros(65536, dtype=bool)
    for idx in ontology.soft_ids:
        soft_lookup[int(idx)] = True
    soft_mask = (soft_lookup[z2_labels] & observed).astype(np.uint8)

    low_lookup = np.zeros(65536, dtype=bool)
    for idx in ontology.low_ids:
        low_lookup[int(idx)] = True
    low_mask = (low_lookup[z2_labels] & observed).astype(np.uint8)

    sdf_hard = (distance_transform_edt(~hard_mask.astype(bool)) * cfg.resolution).astype(np.float32)
    grad_row = (sobel(risk_map, axis=0) / (2.0 * cfg.resolution)).astype(np.float32)
    grad_col = (sobel(risk_map, axis=1) / (2.0 * cfg.resolution)).astype(np.float32)
    sdf_grad_row = (sobel(sdf_hard, axis=0) / (2.0 * cfg.resolution)).astype(np.float32)
    sdf_grad_col = (sobel(sdf_hard, axis=1) / (2.0 * cfg.resolution)).astype(np.float32)

    return {
        "z2_labels": z2_labels.astype(np.uint16),
        "risk_map": risk_map.astype(np.float32),
        "hard_mask": hard_mask.astype(np.uint8),
        "soft_mask": soft_mask.astype(np.uint8),
        "low_mask": low_mask.astype(np.uint8),
        "geom_occ": hard_mask.astype(np.uint8),
        "observed_mask": observed.astype(np.uint8),
        "point_count": point_count,
        "sdf_hard": sdf_hard,
        "grad_row": grad_row,
        "grad_col": grad_col,
        "sdf_grad_row": sdf_grad_row,
        "sdf_grad_col": sdf_grad_col,
    }
