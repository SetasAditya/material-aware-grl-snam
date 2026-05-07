#!/usr/bin/env python3
"""Final RELLIS table and qualitative route-aware selectivity figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_rellis_selectivity import _force_ctx
from grl_rellis import BevConfig
from make_rellis_figure import _label_rgb, _plot_path
from train_rellis_directional_force import (
    DIRS_16,
    DirectionalForceHead,
    _build_point,
    _direction_stats,
    _route_context,
    _sample_grid,
)


SEQUENCES = ["00000", "00001", "00002", "00003", "00004"]


@dataclass
class MethodStats:
    method: str
    fold: str
    car: float
    far: float
    ratio: float
    auprc: float
    n: int


def _as_path(raw: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in raw]


def _load_scene(bev_root: Path, rel_path: str) -> Dict:
    return torch.load(bev_root / rel_path, map_location="cpu", weights_only=False)


def _load_head(path: Path, *, device: str) -> tuple[DirectionalForceHead, Optional[float]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hidden = int(ckpt.get("summary", {}).get("config", {}).get("hidden", 128))
    model = DirectionalForceHead(int(ckpt["in_dim"]), hidden, 1 + len(DIRS_16)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    threshold = ckpt.get("activation_threshold", ckpt.get("summary", {}).get("activation_threshold"))
    return model, None if threshold is None else float(threshold)


@torch.no_grad()
def _head_force(
    model: DirectionalForceHead,
    threshold: Optional[float],
    x: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    logits = model(torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    if threshold is None:
        cls = int(np.argmax(probs))
    else:
        active_idx = int(np.argmax(probs[1:]))
        score = float(probs[active_idx + 1] - probs[0])
        cls = active_idx + 1 if score >= threshold else 0
    if cls == 0:
        return np.zeros(2, dtype=np.float32)
    return DIRS_16[cls - 1].astype(np.float32)


def _path_dir(path: List[Tuple[int, int]], idx: int) -> np.ndarray:
    p = np.asarray(path[idx], dtype=np.float32)
    q = np.asarray(path[min(idx + 1, len(path) - 1)], dtype=np.float32)
    v = q - p
    n = float(np.linalg.norm(v))
    return np.zeros(2, dtype=np.float32) if n < 1e-8 else (v / n).astype(np.float32)


def _legacy_nonroute_x(
    maps: Dict[str, np.ndarray],
    path: List[Tuple[int, int]],
    idx: int,
    *,
    horizon_cells: int,
    long_horizon_cells: int,
    hard_margin_m: float,
) -> Optional[np.ndarray]:
    if len(path) < 3 or idx >= len(path) - 1:
        return None
    p = np.asarray(path[idx], dtype=np.float32)
    scaffold_dir = _path_dir(path, idx)
    if float(np.linalg.norm(scaffold_dir)) < 1e-8:
        return None
    scaffold_risk, scaffold_feasible, scaffold_min_sdf, scaffold_blocked = _direction_stats(
        maps, p, scaffold_dir, horizon_cells=horizon_cells, hard_margin_m=hard_margin_m
    )
    scaffold_risk_long, scaffold_feasible_long, scaffold_min_sdf_long, scaffold_blocked_long = _direction_stats(
        maps, p, scaffold_dir, horizon_cells=long_horizon_cells, hard_margin_m=hard_margin_m
    )
    cand_features: List[List[float]] = []
    feasible_count = 0
    for d in DIRS_16:
        cand_risk, feasible, min_sdf, blocked_frac = _direction_stats(
            maps, p, d, horizon_cells=horizon_cells, hard_margin_m=hard_margin_m
        )
        cand_risk_long, feasible_long, min_sdf_long, blocked_frac_long = _direction_stats(
            maps, p, d, horizon_cells=long_horizon_cells, hard_margin_m=hard_margin_m
        )
        feasible_count += int(feasible)
        cand_features.append(
            [
                cand_risk / max(1.0, float(horizon_cells)),
                1.0 if feasible else 0.0,
                (scaffold_risk - cand_risk) / max(1.0, float(horizon_cells)),
                min_sdf / 10.0,
                blocked_frac,
                cand_risk_long / max(1.0, float(long_horizon_cells)),
                1.0 if feasible_long else 0.0,
                (scaffold_risk_long - cand_risk_long) / max(1.0, float(long_horizon_cells)),
                min_sdf_long / 10.0,
                blocked_frac_long,
                float(np.dot(scaffold_dir, d)),
            ]
        )
    local = [
        scaffold_risk / max(1.0, float(horizon_cells)),
        1.0 if scaffold_feasible else 0.0,
        scaffold_min_sdf / 10.0,
        scaffold_blocked,
        scaffold_risk_long / max(1.0, float(long_horizon_cells)),
        1.0 if scaffold_feasible_long else 0.0,
        scaffold_min_sdf_long / 10.0,
        scaffold_blocked_long,
        float(feasible_count) / float(len(DIRS_16)),
        _sample_grid(maps["risk_map"], p[0], p[1]),
        _sample_grid(maps["sdf_hard"], p[0], p[1]) / 10.0,
        _sample_grid(maps["grad_row"], p[0], p[1]),
        _sample_grid(maps["grad_col"], p[0], p[1]),
        float(idx) / max(1.0, float(len(path) - 1)),
    ]
    return np.asarray(local + np.asarray(cand_features, dtype=np.float32).reshape(-1).tolist(), dtype=np.float32)


def _update_metrics(
    buckets: Dict[str, List[dict]],
    method: str,
    *,
    force: np.ndarray,
    scaffold_dir: np.ndarray,
    true_label: int,
    best_idx: int,
    regime: str,
    eps: float,
) -> None:
    safe = DIRS_16[best_idx]
    perp = force - float(np.dot(force, scaffold_dir)) * scaffold_dir
    buckets[method].append(
        {
            "active": true_label > 0,
            "regime": regime,
            "car_hit": float(np.dot(force, safe)) > eps,
            "far_hit": float(np.linalg.norm(perp)) > eps,
            "force_norm": float(np.linalg.norm(force)),
            "activation_score": max(0.0, float(np.dot(force, safe))) if true_label > 0 else float(np.linalg.norm(perp)),
        }
    )


def _average_precision(rows: List[dict]) -> float:
    if not rows:
        return float("nan")
    y = np.asarray([1 if r["active"] else 0 for r in rows], dtype=np.int32)
    s = np.asarray([float(r["activation_score"]) for r in rows], dtype=np.float64)
    if int(y.sum()) == 0:
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1.0)
    return float((precision * y).sum() / max(1, int(y.sum())))


def _summarize_bucket(method: str, fold: str, rows: List[dict]) -> MethodStats:
    active = [r for r in rows if r["active"]]
    far_pool = [r for r in rows if r["regime"] in ("R2", "R3")]
    r1 = [r["force_norm"] for r in rows if r["regime"] == "R1"]
    r2 = [r["force_norm"] for r in rows if r["regime"] == "R2"]
    car = float(np.mean([r["car_hit"] for r in active])) if active else float("nan")
    far = float(np.mean([r["far_hit"] for r in far_pool])) if far_pool else float("nan")
    ratio = float(np.mean(r1) / max(float(np.mean(r2)), 1e-8)) if r1 and r2 else float("nan")
    return MethodStats(method=method, fold=fold, car=car, far=far, ratio=ratio, auprc=_average_precision(rows), n=len(rows))


def evaluate_final(args: argparse.Namespace) -> tuple[List[MethodStats], Dict[str, dict]]:
    pair_manifest = json.loads((args.pairs_root / "manifest.json").read_text())
    episodes_by_seq: Dict[str, List[dict]] = defaultdict(list)
    for ep in pair_manifest["episodes"]:
        episodes_by_seq[str(ep["sequence"])].append(ep)

    fold_stats: List[MethodStats] = []
    scene_cache: Dict[str, Dict] = {}
    route_cache: Dict[Tuple[str, Tuple[int, int]], Dict[str, np.ndarray]] = {}

    for seq in SEQUENCES:
        nonroute_model, nonroute_threshold = _load_head(
            args.runs_root / f"rellis_directional_loso_aw050_cal_far020_{seq}" / "best.pt",
            device=args.device,
        )
        route_model, route_threshold = _load_head(
            args.runs_root / f"rellis_directional_routeaware_aw050_far020_{seq}" / "best.pt",
            device=args.device,
        )
        buckets: Dict[str, List[dict]] = defaultdict(list)
        for ep in episodes_by_seq[seq]:
            scene_path = str(ep["scene_path"])
            if scene_path not in scene_cache:
                scene_cache[scene_path] = _load_scene(args.bev_root, scene_path)
            maps = scene_cache[scene_path]["maps"]
            goal = tuple(int(x) for x in ep["goal_rc"])
            route_key = (scene_path, goal)
            if route_key not in route_cache:
                route_cache[route_key] = _route_context(maps, goal, risk_weight=args.route_risk_weight)
            route = route_cache[route_key]
            path = _as_path(ep["stage1_path"])
            if len(path) < 3:
                continue
            for idx in range(0, len(path) - 1, args.stride):
                route_row = _build_point(
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
                local_x = _legacy_nonroute_x(
                    maps,
                    path,
                    idx,
                    horizon_cells=args.horizon_cells,
                    long_horizon_cells=args.horizon_cells,
                    hard_margin_m=args.hard_margin_m,
                )
                if route_row is None or local_x is None:
                    continue
                p = np.asarray(path[idx], dtype=np.float32)
                scaffold_dir = _path_dir(path, idx)
                true_label = int(route_row["label"])
                best_idx = int(route_row["best_idx"])
                regime = str(ep["regime"])
                _update_metrics(
                    buckets,
                    "Stage 1 scaffold",
                    force=np.zeros(2, dtype=np.float32),
                    scaffold_dir=scaffold_dir,
                    true_label=true_label,
                    best_idx=best_idx,
                    regime=regime,
                    eps=args.force_eps,
                )
                _update_metrics(
                    buckets,
                    "Scalar Stage 2",
                    force=_force_ctx(
                        maps,
                        p,
                        lam_soft=args.lam_soft,
                        lam_hard=args.lam_hard,
                        hard_margin_m=args.hard_margin_m,
                        gsd=args.gsd,
                    ),
                    scaffold_dir=scaffold_dir,
                    true_label=true_label,
                    best_idx=best_idx,
                    regime=regime,
                    eps=args.force_eps,
                )
                _update_metrics(
                    buckets,
                    "Non-route directional Stage 2",
                    force=_head_force(nonroute_model, nonroute_threshold, local_x, device=args.device),
                    scaffold_dir=scaffold_dir,
                    true_label=true_label,
                    best_idx=best_idx,
                    regime=regime,
                    eps=args.force_eps,
                )
                _update_metrics(
                    buckets,
                    "Route-aware Stage 2",
                    force=_head_force(route_model, route_threshold, np.asarray(route_row["x"]), device=args.device),
                    scaffold_dir=scaffold_dir,
                    true_label=true_label,
                    best_idx=best_idx,
                    regime=regime,
                    eps=args.force_eps,
                )
        for method, rows in buckets.items():
            fold_stats.append(_summarize_bucket(method, seq, rows))

    summary: Dict[str, dict] = {}
    for method in sorted({s.method for s in fold_stats}):
        rows = [s for s in fold_stats if s.method == method]
        summary[method] = {
            "folds": len(rows),
            "CAR_mean": mean(s.car for s in rows),
            "CAR_std": pstdev(s.car for s in rows),
            "FAR_mean": mean(s.far for s in rows),
            "FAR_std": pstdev(s.far for s in rows),
            "selectivity_ratio_mean": mean(s.ratio for s in rows),
            "selectivity_ratio_std": pstdev(s.ratio for s in rows),
            "AUPRC_mean": mean(s.auprc for s in rows),
            "AUPRC_std": pstdev(s.auprc for s in rows),
            "n_mean": mean(s.n for s in rows),
        }
    return fold_stats, summary


def write_final_table(out: Path, fold_stats: List[MethodStats], summary: Dict[str, dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "final_rellis_folds.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "heldout_sequence", "CAR", "FAR", "selectivity_ratio", "AUPRC", "n"])
        for s in fold_stats:
            writer.writerow([s.method, s.fold, s.car, s.far, s.ratio, s.auprc, s.n])
    order = ["Stage 1 scaffold", "Scalar Stage 2", "Non-route directional Stage 2", "Route-aware Stage 2"]
    with (out / "final_rellis_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "CAR_mean", "CAR_std", "FAR_mean", "FAR_std", "selectivity_ratio_mean", "selectivity_ratio_std", "AUPRC_mean", "AUPRC_std"])
        for method in order:
            row = summary[method]
            writer.writerow([
                method,
                row["CAR_mean"],
                row["CAR_std"],
                row["FAR_mean"],
                row["FAR_std"],
                row["selectivity_ratio_mean"],
                row["selectivity_ratio_std"],
                row["AUPRC_mean"],
                row["AUPRC_std"],
            ])
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & CAR $\uparrow$ & FAR $\downarrow$ & Selectivity ratio $\uparrow$ & AUPRC $\uparrow$ \\",
        r"\midrule",
    ]
    for method in order:
        row = summary[method]
        lines.append(
            f"{method} & {row['CAR_mean']:.3f} $\\pm$ {row['CAR_std']:.3f} & "
            f"{row['FAR_mean']:.3f} $\\pm$ {row['FAR_std']:.3f} & "
            f"{row['selectivity_ratio_mean']:.3f} $\\pm$ {row['selectivity_ratio_std']:.3f} & "
            f"{row['AUPRC_mean']:.3f} $\\pm$ {row['AUPRC_std']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (out / "final_rellis_table.tex").write_text("\n".join(lines))
    md = [
        "# Final RELLIS Selectivity Artifacts",
        "",
        "All methods are evaluated on the same five-sequence leave-one-sequence-out benchmark and the same route-aware safe-alternative oracle.",
        "",
        "| Method | CAR up | FAR down | Selectivity ratio | AUPRC up |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in order:
        row = summary[method]
        md.append(
            f"| {method} | {row['CAR_mean']:.3f} +/- {row['CAR_std']:.3f} | "
            f"{row['FAR_mean']:.3f} +/- {row['FAR_std']:.3f} | "
            f"{row['selectivity_ratio_mean']:.3f} +/- {row['selectivity_ratio_std']:.3f} | "
            f"{row['AUPRC_mean']:.3f} +/- {row['AUPRC_std']:.3f} |"
        )
    md.extend(
        [
            "",
            "Qualitative figures:",
            "",
            "- `qual_r1_routeaware.png`: route-aware forces activate along a feasible safer route.",
            "- `qual_r2_routeaware.png`: forces are reduced relative to scalar risk-gradient behavior in blocked/ambiguous terrain.",
            "- `qual_r3_routeaware.png`: forces remain mostly quiet in risk-neutral terrain.",
        ]
    )
    (out / "README.md").write_text("\n".join(md))


def choose_examples(args: argparse.Namespace) -> Dict[str, dict]:
    pair_manifest = json.loads((args.pairs_root / "manifest.json").read_text())
    best: Dict[str, tuple[float, dict]] = {}
    scene_cache: Dict[str, Dict] = {}
    route_cache: Dict[Tuple[str, Tuple[int, int]], Dict[str, np.ndarray]] = {}
    model_cache: Dict[str, tuple[DirectionalForceHead, Optional[float]]] = {}
    for ep in pair_manifest["episodes"]:
        regime = str(ep["regime"])
        seq = str(ep["sequence"])
        if seq not in model_cache:
            model_cache[seq] = _load_head(
                args.runs_root / f"rellis_directional_routeaware_aw050_far020_{seq}" / "best.pt",
                device=args.device,
            )
        model, threshold = model_cache[seq]
        scene_path = str(ep["scene_path"])
        if scene_path not in scene_cache:
            scene_cache[scene_path] = _load_scene(args.bev_root, scene_path)
        maps = scene_cache[scene_path]["maps"]
        goal = tuple(int(x) for x in ep["goal_rc"])
        route_key = (scene_path, goal)
        if route_key not in route_cache:
            route_cache[route_key] = _route_context(maps, goal, risk_weight=args.route_risk_weight)
        route = route_cache[route_key]
        path = _as_path(ep["stage1_path"])
        active = 0
        total = 0
        for idx in range(0, len(path) - 1, max(args.stride * 2, 1)):
            row = _build_point(
                maps,
                path,
                idx,
                regime=regime,
                episode_id=str(ep["episode_id"]),
                horizon_cells=args.horizon_cells,
                long_horizon_cells=args.long_horizon_cells,
                hard_margin_m=args.hard_margin_m,
                improvement_margin=args.improvement_margin,
                route=route,
                route_max_ratio=args.route_max_ratio,
            )
            if row is None:
                continue
            force = _head_force(model, threshold, np.asarray(row["x"]), device=args.device)
            total += 1
            active += int(float(np.linalg.norm(force)) > args.force_eps)
        if total == 0:
            continue
        frac = active / total
        score = frac if regime == "R1" else 1.0 - frac
        if regime not in best or score > best[regime][0]:
            best[regime] = (score, ep)
    return {regime: ep for regime, (_, ep) in best.items()}


def plot_example(args: argparse.Namespace, ep: dict, out_path: Path) -> None:
    model, threshold = _load_head(
        args.runs_root / f"rellis_directional_routeaware_aw050_far020_{ep['sequence']}" / "best.pt",
        device=args.device,
    )
    scene = _load_scene(args.bev_root, ep["scene_path"])
    maps = scene["maps"]
    path = _as_path(ep["stage1_path"])
    risk_path = _as_path(ep["risk_path"])
    goal = tuple(int(x) for x in ep["goal_rc"])
    route = _route_context(maps, goal, risk_weight=args.route_risk_weight)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
    axes[0].imshow(_label_rgb(maps["z2_labels"]))
    axes[0].set_title(f"Semantic BEV ({ep['regime']})")
    im = axes[1].imshow(maps["risk_map"], cmap="magma", vmin=0, vmax=1)
    axes[1].contour(maps["hard_mask"], levels=[0.5], colors="cyan", linewidths=0.6)
    axes[1].set_title("Risk + hard hazards")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    axes[2].imshow(maps["risk_map"], cmap="Greys", vmin=0, vmax=1)
    _plot_path(axes[2], path, color="#1f77b4", label="Stage 1", lw=2.0)
    _plot_path(axes[2], risk_path, color="#d62728", label="Risk-cost A*", lw=1.6)
    axes[2].legend(loc="lower right", fontsize=7)
    axes[2].set_title("Candidate paths")
    axes[3].imshow(maps["risk_map"], cmap="magma", vmin=0, vmax=1)
    axes[3].contour(maps["hard_mask"], levels=[0.5], colors="white", linewidths=0.5)
    rr, cc, uu, vv = [], [], [], []
    for idx in range(0, len(path) - 1, max(args.arrow_path_step, 1)):
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
        if row is None:
            continue
        force = _head_force(model, threshold, np.asarray(row["x"]), device=args.device)
        if float(np.linalg.norm(force)) <= args.force_eps:
            continue
        p = path[idx]
        rr.append(p[0])
        cc.append(p[1])
        vv.append(force[0])
        uu.append(force[1])
    if rr:
        axes[3].quiver(cc, rr, uu, vv, color="#60f0ff", angles="xy", scale_units="xy", scale=0.26, width=0.004)
    _plot_path(axes[3], path, color="#ffffff", label="Stage 1", lw=1.4)
    axes[3].set_title("Route-aware Stage 2 force")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, maps["risk_map"].shape[1] - 1)
        ax.set_ylim(maps["risk_map"].shape[0] - 1, 0)
    fig.suptitle(f"RELLIS {ep['sequence']}/{ep['frame_id']}  {ep['episode_id']}  {ep['regime']}", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create final RELLIS comparison table and qualitative figures.")
    ap.add_argument("--bev-root", type=Path, default=ROOT / "cache" / "rellis_bev_all_seqbalanced_2500")
    ap.add_argument("--pairs-root", type=Path, default=ROOT / "cache" / "rellis_pairs_all_seqbalanced_2500_loso")
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rellis_final_artifacts")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    args.out.mkdir(parents=True, exist_ok=True)
    fold_stats, summary = evaluate_final(args)
    write_final_table(args.out, fold_stats, summary)
    examples = choose_examples(args)
    for regime in ("R1", "R2", "R3"):
        if regime in examples:
            plot_example(args, examples[regime], args.out / f"qual_{regime.lower()}_routeaware.png")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
