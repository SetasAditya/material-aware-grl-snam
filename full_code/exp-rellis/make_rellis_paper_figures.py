#!/usr/bin/env python3
"""Paper-facing RELLIS figures that make the selectivity comparison explicit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_rellis_selectivity import _force_ctx
from make_rellis_figure import _label_rgb, _plot_path
from make_rellis_final_artifacts import (
    _as_path,
    _build_point,
    _head_force,
    _legacy_nonroute_x,
    _load_head,
    _load_scene,
    _path_dir,
    _route_context,
)


FIXED_EXAMPLES = {
    "R1": ("00001", "002198", "455"),
    "R2": ("00000", "002145", "17"),
    "R3": ("00000", "000531", "2"),
}


def _fixed_examples(args: argparse.Namespace) -> Dict[str, dict]:
    manifest = json.loads((args.pairs_root / "manifest.json").read_text())
    episodes = manifest["episodes"]
    selected: Dict[str, dict] = {}
    for regime, (seq, frame_id, episode_id) in FIXED_EXAMPLES.items():
        for ep in episodes:
            if (
                str(ep["regime"]) == regime
                and str(ep["sequence"]) == seq
                and str(ep["frame_id"]) == frame_id
                and str(ep["episode_id"]) == episode_id
            ):
                selected[regime] = ep
                break
        if regime not in selected:
            raise RuntimeError(f"Could not find fixed {regime} example {seq}/{frame_id}/{episode_id}")
    return selected


def _force_arrows(
    maps: Dict[str, np.ndarray],
    path: List[Tuple[int, int]],
    *,
    method: str,
    model=None,
    threshold: Optional[float] = None,
    route: Optional[Dict[str, np.ndarray]] = None,
    ep: Optional[dict] = None,
    args: argparse.Namespace,
) -> Tuple[List[int], List[int], List[float], List[float], float, float]:
    rr: List[int] = []
    cc: List[int] = []
    uu: List[float] = []
    vv: List[float] = []
    norms: List[float] = []
    active = 0
    total = 0
    for idx in range(0, len(path) - 1, max(args.arrow_path_step, 1)):
        p = np.asarray(path[idx], dtype=np.float32)
        force = np.zeros(2, dtype=np.float32)
        if method == "scalar":
            force = _force_ctx(
                maps,
                p,
                lam_soft=args.lam_soft,
                lam_hard=args.lam_hard,
                hard_margin_m=args.hard_margin_m,
                gsd=args.gsd,
            )
        elif method == "nonroute":
            x = _legacy_nonroute_x(
                maps,
                path,
                idx,
                horizon_cells=args.horizon_cells,
                long_horizon_cells=args.horizon_cells,
                hard_margin_m=args.hard_margin_m,
            )
            if x is not None:
                force = _head_force(model, threshold, x, device=args.device)
        elif method == "route":
            row = _build_point(
                maps,
                path,
                idx,
                regime=str(ep["regime"]),
                episode_id=str(ep["episode_id"]),
                horizon_cells=args.horizon_cells,
                long_horizon_cells=args.long_horizon_cells,
                hard_margin_m=args.hard_margin_m,
                improvement_margin=args.improvement_margin,
                route=route,
                route_max_ratio=args.route_max_ratio,
            )
            if row is not None:
                force = _head_force(model, threshold, np.asarray(row["x"]), device=args.device)
        norm = float(np.linalg.norm(force))
        norms.append(norm)
        total += 1
        if norm > args.force_eps:
            active += 1
            r, c = path[idx]
            rr.append(r)
            cc.append(c)
            vv.append(float(force[0]))
            uu.append(float(force[1]))
    active_frac = active / max(1, total)
    mean_norm = float(np.mean(norms)) if norms else 0.0
    return rr, cc, uu, vv, active_frac, mean_norm


def _plot_force_panel(
    ax,
    maps: Dict[str, np.ndarray],
    path: List[Tuple[int, int]],
    arrows: Tuple[List[int], List[int], List[float], List[float], float, float],
    *,
    title: str,
    desired: str,
) -> None:
    ax.imshow(maps["risk_map"], cmap="magma", vmin=0, vmax=1)
    ax.contour(maps["hard_mask"], levels=[0.5], colors="white", linewidths=0.45)
    rr, cc, uu, vv, active_frac, mean_norm = arrows
    if rr:
        ax.quiver(cc, rr, uu, vv, color="#58f3ff", angles="xy", scale_units="xy", scale=0.28, width=0.0045)
    _plot_path(ax, path, color="#ffffff", label="Stage 1", lw=1.3)
    ax.set_title(title, fontsize=9)
    ax.text(
        0.02,
        0.98,
        f"{desired}\nactive={active_frac:.2f}, |F|={mean_norm:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="white",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2.0),
    )


def make_selectivity_comparison(args: argparse.Namespace) -> Path:
    examples = _fixed_examples(args)
    rows = [("R1", "should activate"), ("R2", "should suppress"), ("R3", "should stay quiet")]
    fig, axes = plt.subplots(len(rows), 5, figsize=(16, 9.2), constrained_layout=True)
    for row_idx, (regime, desired) in enumerate(rows):
        ep = examples[regime]
        scene = _load_scene(args.bev_root, ep["scene_path"])
        maps = scene["maps"]
        path = _as_path(ep["stage1_path"])
        risk_path = _as_path(ep["risk_path"])
        goal = tuple(int(x) for x in ep["goal_rc"])
        route = _route_context(maps, goal, risk_weight=args.route_risk_weight)
        nonroute_model, nonroute_threshold = _load_head(
            args.runs_root / f"rellis_directional_loso_aw050_cal_far020_{ep['sequence']}" / "best.pt",
            device=args.device,
        )
        route_model, route_threshold = _load_head(
            args.runs_root / f"rellis_directional_routeaware_aw050_far020_{ep['sequence']}" / "best.pt",
            device=args.device,
        )

        ax = axes[row_idx, 0]
        ax.imshow(_label_rgb(maps["z2_labels"]))
        ax.set_title(f"{regime}: semantic material BEV", fontsize=9)
        ax.text(
            0.02,
            0.98,
            f"{ep['sequence']}/{ep['frame_id']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="white",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2.0),
        )

        ax = axes[row_idx, 1]
        ax.imshow(maps["risk_map"], cmap="Greys", vmin=0, vmax=1)
        _plot_path(ax, path, color="#1f77b4", label="Stage 1", lw=2.0)
        _plot_path(ax, risk_path, color="#d62728", label="Risk-cost A*", lw=1.6)
        if row_idx == 0:
            ax.legend(loc="lower right", fontsize=7)
        ax.set_title("candidate paths", fontsize=9)

        scalar = _force_arrows(maps, path, method="scalar", args=args)
        nonroute = _force_arrows(
            maps,
            path,
            method="nonroute",
            model=nonroute_model,
            threshold=nonroute_threshold,
            args=args,
        )
        routeaware = _force_arrows(
            maps,
            path,
            method="route",
            model=route_model,
            threshold=route_threshold,
            route=route,
            ep=ep,
            args=args,
        )
        _plot_force_panel(axes[row_idx, 2], maps, path, scalar, title="scalar risk-gradient force", desired=desired)
        _plot_force_panel(axes[row_idx, 3], maps, path, nonroute, title="non-route directional force", desired=desired)
        _plot_force_panel(axes[row_idx, 4], maps, path, routeaware, title="route-aware Stage 2 force", desired=desired)

        for col in range(5):
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])
            axes[row_idx, col].set_xlim(0, maps["risk_map"].shape[1] - 1)
            axes[row_idx, col].set_ylim(maps["risk_map"].shape[0] - 1, 0)

    fig.suptitle(
        "RELLIS selectivity diagnostic: route-aware Stage 2 activates in R1 and suppresses force in R2/R3",
        fontsize=13,
    )
    out = args.out / "rellis_selectivity_comparison.png"
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def make_material_context(args: argparse.Namespace) -> Path:
    examples = _fixed_examples(args)
    ep = examples["R1"]
    scene = _load_scene(args.bev_root, ep["scene_path"])
    maps = scene["maps"]
    rgb_path = ROOT / "RELLIS-3D" / "utils" / "example" / "frame000104-1581624663_149.jpg"
    rgb = mpimg.imread(rgb_path)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("representative RELLIS RGB scene", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(_label_rgb(maps["z2_labels"]))
    axes[1].set_title("semantic LiDAR BEV", fontsize=10)

    im = axes[2].imshow(maps["risk_map"], cmap="magma", vmin=0, vmax=1)
    axes[2].contour(maps["hard_mask"], levels=[0.5], colors="cyan", linewidths=0.7)
    axes[2].set_title("soft risk + hard hazards", fontsize=10)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)

    axes[3].axis("off")
    axes[3].set_title("material-risk mapping", fontsize=10)
    legend_rows = [
        ("low risk", "road, dirt road, compact ground", "#2ca25f"),
        ("medium risk", "grass, rough terrain, light vegetation", "#fdae61"),
        ("high soft risk", "mud, puddles, dense vegetation, rubble", "#d7191c"),
        ("hard hazard", "vehicle, pole, fence, person, building", "#222222"),
    ]
    y = 0.84
    for name, desc, color in legend_rows:
        axes[3].add_patch(mpatches.Rectangle((0.02, y - 0.035), 0.08, 0.05, color=color, transform=axes[3].transAxes))
        axes[3].text(0.13, y, name, transform=axes[3].transAxes, fontsize=9, fontweight="bold", va="center")
        axes[3].text(0.13, y - 0.075, desc, transform=axes[3].transAxes, fontsize=8, va="center", wrap=True)
        y -= 0.215
    for ax in axes[1:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    out = args.out / "rellis_material_context.png"
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create paper-facing RELLIS explanatory figures.")
    ap.add_argument("--bev-root", type=Path, default=ROOT / "cache" / "rellis_bev_all_seqbalanced_2500")
    ap.add_argument("--pairs-root", type=Path, default=ROOT / "cache" / "rellis_pairs_all_seqbalanced_2500_loso")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rellis_paper_figures")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--arrow-path-step", type=int, default=6)
    ap.add_argument("--horizon-cells", type=int, default=8)
    ap.add_argument("--long-horizon-cells", type=int, default=8)
    ap.add_argument("--hard-margin-m", type=float, default=1.0)
    ap.add_argument("--improvement-margin", type=float, default=0.1)
    ap.add_argument("--route-risk-weight", type=float, default=12.0)
    ap.add_argument("--route-max-ratio", type=float, default=2.2)
    ap.add_argument("--force-eps", type=float, default=1e-3)
    ap.add_argument("--lam-soft", type=float, default=1.5)
    ap.add_argument("--lam-hard", type=float, default=2.0)
    ap.add_argument("--gsd", type=float, default=0.5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    material = make_material_context(args)
    comparison = make_selectivity_comparison(args)
    print(f"Wrote {material}")
    print(f"Wrote {comparison}")


if __name__ == "__main__":
    main()
