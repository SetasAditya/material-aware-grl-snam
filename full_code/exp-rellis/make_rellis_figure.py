#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_rellis_selectivity import _force_ctx
from grl_rellis.ontology import RAW_LABELS


LABEL_COLORS = {
    0: "#1f1f1f",
    1: "#8b5a2b",
    3: "#4caf50",
    4: "#156f2b",
    5: "#00a6a6",
    6: "#3f8cff",
    8: "#f1c40f",
    9: "#d33682",
    10: "#666666",
    12: "#c0392b",
    15: "#5a2d0c",
    17: "#c7a4ff",
    18: "#7e57c2",
    19: "#f48fb1",
    23: "#bdbdbd",
    27: "#ff7043",
    31: "#80deea",
    33: "#6d4c41",
    34: "#8e44ad",
}


def _as_path(raw) -> List[Tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in raw]


def _plot_path(ax, path, *, color: str, label: str, lw: float = 2.2) -> None:
    if not path:
        return
    arr = np.asarray(path, dtype=np.float32)
    ax.plot(arr[:, 1], arr[:, 0], color=color, lw=lw, label=label)
    ax.scatter(arr[0, 1], arr[0, 0], s=28, color=color, marker="o", edgecolor="black", linewidth=0.4)
    ax.scatter(arr[-1, 1], arr[-1, 0], s=34, color=color, marker="*", edgecolor="black", linewidth=0.4)


def _label_rgb(labels: np.ndarray) -> np.ndarray:
    rgb = np.zeros(labels.shape + (3,), dtype=np.float32)
    for idx, hex_color in LABEL_COLORS.items():
        h = hex_color.lstrip("#")
        color = tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
        rgb[labels == idx] = color
    return rgb


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Make a four-panel RELLIS selectivity figure.")
    ap.add_argument("--bev-root", type=Path, default=ROOT / "cache" / "rellis_bev_val_main_100")
    ap.add_argument("--pairs-root", type=Path, default=ROOT / "cache" / "rellis_pairs_val_main_100")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "rellis_selectivity_example.png")
    ap.add_argument("--episode-id", default=None)
    ap.add_argument("--regime", default="R1", choices=["R1", "R2", "R3"])
    ap.add_argument("--lam-soft", type=float, default=1.5)
    ap.add_argument("--lam-hard", type=float, default=2.0)
    ap.add_argument("--hard-margin-m", type=float, default=1.0)
    ap.add_argument("--arrow-step", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pair_manifest = json.loads((args.pairs_root / "manifest.json").read_text())
    episodes = pair_manifest["episodes"]
    if args.episode_id:
        matches = [ep for ep in episodes if ep["episode_id"] == args.episode_id]
    else:
        matches = [ep for ep in episodes if ep["regime"] == args.regime]
    if not matches:
        raise RuntimeError(f"No matching episode for episode_id={args.episode_id!r}, regime={args.regime!r}")
    ep = matches[0]
    scene = torch.load(args.bev_root / ep["scene_path"], map_location="cpu", weights_only=False)
    maps: Dict[str, np.ndarray] = scene["maps"]
    stage1 = _as_path(ep["stage1_path"])
    risk_path = _as_path(ep["risk_path"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), constrained_layout=True)

    axes[0].imshow(_label_rgb(maps["z2_labels"]))
    axes[0].set_title(f"Semantic BEV ({ep['regime']})")

    risk_im = axes[1].imshow(maps["risk_map"], cmap="magma", vmin=0, vmax=1)
    axes[1].contour(maps["hard_mask"], levels=[0.5], colors="cyan", linewidths=0.7)
    axes[1].set_title("Risk map + hard boundary")
    fig.colorbar(risk_im, ax=axes[1], fraction=0.046, pad=0.02)

    axes[2].imshow(maps["risk_map"], cmap="Greys", vmin=0, vmax=1)
    _plot_path(axes[2], stage1, color="#1f77b4", label="Stage 1")
    _plot_path(axes[2], risk_path, color="#d62728", label="Risk-cost / local A*")
    _plot_path(axes[2], risk_path, color="#2ca02c", label="Stage 2 force proxy", lw=1.4)
    axes[2].legend(loc="lower right", fontsize=7, frameon=True)
    axes[2].set_title("Traversal candidates")

    axes[3].imshow(maps["risk_map"], cmap="magma", vmin=0, vmax=1)
    hard = maps["hard_mask"].astype(bool)
    axes[3].contour(hard, levels=[0.5], colors="white", linewidths=0.5)
    rows, cols = maps["risk_map"].shape
    rr, cc, uu, vv = [], [], [], []
    for r in range(args.arrow_step // 2, rows, args.arrow_step):
        for c in range(args.arrow_step // 2, cols, args.arrow_step):
            if hard[r, c]:
                continue
            f = _force_ctx(
                maps,
                np.asarray([r, c], dtype=np.float32),
                lam_soft=args.lam_soft,
                lam_hard=args.lam_hard,
                hard_margin_m=args.hard_margin_m,
                gsd=0.5,
            )
            if np.linalg.norm(f) < 0.015:
                continue
            rr.append(r)
            cc.append(c)
            vv.append(f[0])
            uu.append(f[1])
    axes[3].quiver(cc, rr, uu, vv, color="#60f0ff", angles="xy", scale_units="xy", scale=0.7, width=0.003)
    _plot_path(axes[3], stage1, color="#ffffff", label="Stage 1", lw=1.5)
    axes[3].set_title("Context force field")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, maps["risk_map"].shape[1] - 1)
        ax.set_ylim(maps["risk_map"].shape[0] - 1, 0)

    fig.suptitle(
        f"RELLIS {ep['sequence']}/{ep['frame_id']}  {ep['episode_id']}  "
        f"{RAW_LABELS.get(33, 'mud')}/{RAW_LABELS.get(34, 'rubble')} risk cues",
        fontsize=12,
    )
    fig.savefig(args.out, dpi=180)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
