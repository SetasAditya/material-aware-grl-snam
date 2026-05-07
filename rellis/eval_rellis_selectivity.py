#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grl_rellis import BevConfig
from scripts.baselines.dfc.metrics import FailureWeights, compute_path_metrics, compute_trace_metrics
from scripts.baselines.dfc.models import (
    _build_goal_feats,
    _build_obs_feats,
    build_geom_waypoints,
    load_model,
    run_model_episode,
)
from scripts.baselines.dfc.planners import risk_weighted_astar
from scripts.build_dfc2018_stagewise import extract_local_geom_obstacles, extract_risk_patch
from train_rellis_directional_force import (
    DirectionalForceHead,
    _build_point as _build_directional_point,
)


DIRS_16 = np.asarray(
    [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (2.0, 1.0),
        (2.0, -1.0),
        (-2.0, 1.0),
        (-2.0, -1.0),
        (1.0, 2.0),
        (-1.0, 2.0),
        (1.0, -2.0),
        (-1.0, -2.0),
    ],
    dtype=np.float32,
)
DIRS_16 = DIRS_16 / np.linalg.norm(DIRS_16, axis=1, keepdims=True)


def _load_scene(bev_root: Path, rel_path: str) -> Dict:
    return torch.load(bev_root / rel_path, map_location="cpu", weights_only=False)


def _as_path(raw: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in raw]


def _sample_grid(arr: np.ndarray, r: float, c: float) -> float:
    rr = int(np.clip(round(float(r)), 0, arr.shape[0] - 1))
    cc = int(np.clip(round(float(c)), 0, arr.shape[1] - 1))
    return float(arr[rr, cc])


def _direction_integral(
    maps: Dict[str, np.ndarray],
    pos: np.ndarray,
    direction: np.ndarray,
    *,
    horizon_cells: int,
    hard_margin_m: float,
) -> Tuple[float, bool]:
    risk = maps["risk_map"]
    hard = maps["hard_mask"].astype(bool)
    sdf = maps["sdf_hard"]
    total = 0.0
    feasible = True
    for step in range(1, horizon_cells + 1):
        q = pos + float(step) * direction
        r = int(round(float(q[0])))
        c = int(round(float(q[1])))
        if not (0 <= r < risk.shape[0] and 0 <= c < risk.shape[1]):
            feasible = False
            break
        total += float(risk[r, c])
        if hard[r, c] or float(sdf[r, c]) < hard_margin_m:
            feasible = False
            break
    return total, feasible


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def _force_ctx(
    maps: Dict[str, np.ndarray],
    pos: np.ndarray,
    *,
    lam_soft: float,
    lam_hard: float,
    hard_margin_m: float,
    gsd: float,
) -> np.ndarray:
    r, c = float(pos[0]), float(pos[1])
    grad_row = _sample_grid(maps["grad_row"], r, c)
    grad_col = _sample_grid(maps["grad_col"], r, c)
    sdf = _sample_grid(maps["sdf_hard"], r, c)
    sdf_grad_row = _sample_grid(maps["sdf_grad_row"], r, c)
    sdf_grad_col = _sample_grid(maps["sdf_grad_col"], r, c)
    f_soft = -float(lam_soft) * np.asarray([grad_row, grad_col], dtype=np.float32)
    gate = 1.0 / (1.0 + math.exp(6.0 * (sdf - hard_margin_m)))
    f_hard = float(lam_hard) * gate * np.asarray([sdf_grad_row, sdf_grad_col], dtype=np.float32)
    # Gradients are per metre; scale into grid-step units for direction-dot tests.
    return (f_soft + f_hard) * float(gsd)


def _load_directional_head(ckpt_path: Path, *, device: str) -> DirectionalForceHead:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_dim = int(ck["in_dim"])
    out_dim = int(np.asarray(ck.get("dirs", DIRS_16)).shape[0]) + 1
    hidden = int(ck.get("summary", {}).get("config", {}).get("hidden", 128))
    model = DirectionalForceHead(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def _directional_force_at(
    model: DirectionalForceHead,
    maps: Dict[str, np.ndarray],
    path: List[Tuple[int, int]],
    idx: int,
    *,
    regime: str,
    episode_id: str,
    device: str,
    horizon_cells: int,
    hard_margin_m: float,
    improvement_margin: float,
    force_mode: str,
    force_gain: float,
) -> Tuple[np.ndarray, int, float, float]:
    row = _build_directional_point(
        maps,
        path,
        idx,
        regime=regime,
        episode_id=episode_id,
        horizon_cells=horizon_cells,
        hard_margin_m=hard_margin_m,
        improvement_margin=improvement_margin,
    )
    if row is None:
        return np.zeros(2, dtype=np.float32), 0, 1.0, 0.0
    x = torch.as_tensor(np.asarray(row["x"], dtype=np.float32), device=device).unsqueeze(0)
    logits = model(x).squeeze(0)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    pred = int(np.argmax(probs))
    if force_mode == "argmax":
        f = np.zeros(2, dtype=np.float32) if pred == 0 else DIRS_16[pred - 1].astype(np.float32)
    else:
        f = (probs[1:, None] * DIRS_16).sum(axis=0).astype(np.float32)
    return float(force_gain) * f, pred, float(probs[0]), float(probs[pred])


@torch.no_grad()
def _model_lambdas_at(
    model: torch.nn.Module,
    maps: Dict[str, np.ndarray],
    path: List[Tuple[int, int]],
    idx: int,
    *,
    device: str,
    patch_size: int,
    waypoint_stride: int,
) -> Tuple[float, float]:
    h_rows, h_cols = maps["risk_map"].shape
    p = path[idx]
    goal_idx = min(idx + max(1, waypoint_stride), len(path) - 1)
    stage_goal = path[goal_idx]
    pos_xy = np.asarray([float(p[1]), float(p[0])], dtype=np.float32)
    goal_xy = np.asarray([float(stage_goal[1]), float(stage_goal[0])], dtype=np.float32)
    cr = int(np.clip(p[0], 0, h_rows - 1))
    cc = int(np.clip(p[1], 0, h_cols - 1))
    centers, radii, widths, _ = extract_local_geom_obstacles(
        maps["geom_occ"],
        (cr, cc),
        patch_size=64,
        robot_radius=1.5,
        margin_factor=0.5,
    )
    obs_feats = _build_obs_feats(pos_xy, goal_xy, centers, radii, widths, device)
    obs_mask = (
        torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)
        if obs_feats.shape[1] > 0
        else torch.zeros(1, 0, dtype=torch.bool, device=device)
    )
    goal_feats = _build_goal_feats(pos_xy, goal_xy, device)
    risk_patch_np, _ = extract_risk_patch(maps, (cr, cc), patch_size)
    risk_patch = torch.as_tensor(risk_patch_np, dtype=torch.float32, device=device).unsqueeze(0)
    _, _, _, lam_soft, lam_hard, _ = model(obs_feats, obs_mask, goal_feats, risk_patch)
    return float(lam_soft.item()), float(lam_hard.item())


def _selectivity_rows(
    maps: Dict[str, np.ndarray],
    path: List[Tuple[int, int]],
    *,
    regime: str,
    episode_id: str,
    lam_soft: float,
    lam_hard: float,
    gsd: float,
    horizon_cells: int,
    hard_margin_m: float,
    improvement_margin: float,
    stride: int,
    force_source: str = "analytic_fixed_lambda",
    model: Optional[torch.nn.Module] = None,
    directional_model: Optional[DirectionalForceHead] = None,
    device: str = "cpu",
    model_patch_size: int = 32,
    model_waypoint_stride: int = 6,
    directional_force_mode: str = "argmax",
    directional_force_gain: float = 1.0,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    if len(path) < 3:
        return rows
    for idx in range(0, len(path) - 1, max(1, stride)):
        p = np.asarray(path[idx], dtype=np.float32)
        nxt = np.asarray(path[min(idx + 1, len(path) - 1)], dtype=np.float32)
        scaffold_dir = _unit(nxt - p)
        if float(np.linalg.norm(scaffold_dir)) < 1e-8:
            continue
        scaffold_risk, scaffold_feasible = _direction_integral(
            maps, p, scaffold_dir, horizon_cells=horizon_cells, hard_margin_m=hard_margin_m
        )
        best_dir: Optional[np.ndarray] = None
        best_risk = float("inf")
        feasible_count = 0
        for d in DIRS_16:
            risk, feasible = _direction_integral(
                maps, p, d, horizon_cells=horizon_cells, hard_margin_m=hard_margin_m
            )
            if feasible:
                feasible_count += 1
            if feasible and risk < best_risk:
                best_risk = risk
                best_dir = d
        if best_dir is None:
            continue
        has_safe_alt = bool(scaffold_feasible and (scaffold_risk - best_risk) >= improvement_margin)
        lam_s = lam_soft
        lam_h = lam_hard
        if model is not None:
            lam_s, lam_h = _model_lambdas_at(
                model,
                maps,
                path,
                idx,
                device=device,
                patch_size=model_patch_size,
                waypoint_stride=model_waypoint_stride,
            )
        directional_class = 0
        directional_p_noop = float("nan")
        directional_p_pred = float("nan")
        if directional_model is not None:
            f, directional_class, directional_p_noop, directional_p_pred = _directional_force_at(
                directional_model,
                maps,
                path,
                idx,
                regime=regime,
                episode_id=episode_id,
                device=device,
                horizon_cells=horizon_cells,
                hard_margin_m=hard_margin_m,
                improvement_margin=improvement_margin,
                force_mode=directional_force_mode,
                force_gain=directional_force_gain,
            )
        else:
            f = _force_ctx(
                maps,
                p,
                lam_soft=lam_s,
                lam_hard=lam_h,
                hard_margin_m=hard_margin_m,
                gsd=gsd,
            )
        f_norm = float(np.linalg.norm(f))
        perp = f - float(np.dot(f, scaffold_dir)) * scaffold_dir
        risk_grad = np.asarray(
            [
                _sample_grid(maps["grad_row"], p[0], p[1]),
                _sample_grid(maps["grad_col"], p[0], p[1]),
            ],
            dtype=np.float32,
        )
        neg_grad = -risk_grad
        denom = max(float(np.linalg.norm(f) * np.linalg.norm(neg_grad)), 1e-8)
        rows.append(
            {
                "episode_id": episode_id,
                "regime": regime,
                "force_source": force_source,
                "path_index": idx,
                "has_safe_alt": float(has_safe_alt),
                "feasible_direction_count": float(feasible_count),
                "scaffold_risk": float(scaffold_risk),
                "safe_risk": float(best_risk),
                "lam_soft": float(lam_s),
                "lam_hard": float(lam_h),
                "directional_class": float(directional_class),
                "directional_p_noop": float(directional_p_noop),
                "directional_p_pred": float(directional_p_pred),
                "force_norm": f_norm,
                "force_perp_norm": float(np.linalg.norm(perp)),
                "dot_safe": float(np.dot(f, best_dir)),
                "force_risk_alignment": float(np.dot(f, neg_grad) / denom),
            }
        )
    return rows


def _summarize_selectivity(rows: List[Dict[str, float | str]], eps: float) -> Dict[str, float]:
    by_regime = {reg: [r for r in rows if r["regime"] == reg] for reg in ("R1", "R2", "R3")}
    r1 = by_regime["R1"]
    r2 = by_regime["R2"]
    r3 = by_regime["R3"]
    car_pool = [r for r in r1 if float(r["has_safe_alt"]) > 0.5]
    car = float(np.mean([float(r["dot_safe"]) > eps for r in car_pool])) if car_pool else float("nan")
    far_pool = r2 + r3
    far = float(np.mean([float(r["force_perp_norm"]) > eps for r in far_pool])) if far_pool else float("nan")
    r1_perp = np.asarray([float(r["force_perp_norm"]) for r in r1], dtype=np.float32)
    r2_perp = np.asarray([float(r["force_perp_norm"]) for r in r2], dtype=np.float32)
    ratio = float(r1_perp.mean() / max(float(r2_perp.mean()), 1e-8)) if r1_perp.size and r2_perp.size else float("nan")
    align = float(np.nanmean([float(r["force_risk_alignment"]) for r in rows])) if rows else float("nan")
    return {
        "num_force_samples": float(len(rows)),
        "correct_activation_rate": car,
        "false_activation_rate": far,
        "selectivity_ratio": ratio,
        "force_risk_alignment": align,
    }


def _summarize_selectivity_by_source(rows: List[Dict[str, float | str]], eps: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for source in sorted({str(r.get("force_source", "analytic_fixed_lambda")) for r in rows}):
        source_rows = [r for r in rows if str(r.get("force_source", "analytic_fixed_lambda")) == source]
        out[source] = _summarize_selectivity(source_rows, eps=eps)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate RELLIS selectivity and traversal metrics.")
    ap.add_argument("--bev-root", type=Path, default=ROOT / "cache" / "rellis_bev_val_main")
    ap.add_argument("--pairs-root", type=Path, default=ROOT / "cache" / "rellis_pairs_val_main")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rellis_selectivity_val_main")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--lam-soft", type=float, default=1.5)
    ap.add_argument("--lam-hard", type=float, default=2.0)
    ap.add_argument("--hard-margin-m", type=float, default=1.0)
    ap.add_argument("--horizon-cells", type=int, default=8)
    ap.add_argument("--improvement-margin", type=float, default=0.1)
    ap.add_argument("--force-eps", type=float, default=0.02)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--ckpt-s1", type=Path, default=None)
    ap.add_argument("--ckpt-s2", type=Path, default=None)
    ap.add_argument("--directional-ckpt", type=Path, default=None,
                    help="Candidate-direction Stage 2 selectivity head checkpoint.")
    ap.add_argument("--directional-force-mode", choices=["argmax", "expected"], default="argmax",
                    help="argmax yields an exactly quiet no-activation state; expected uses softmax-weighted directions.")
    ap.add_argument("--directional-force-gain", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-steps-per-stage", type=int, default=30)
    ap.add_argument("--model-max-episodes", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bev_manifest = json.loads((args.bev_root / "manifest.json").read_text())
    pair_manifest = json.loads((args.pairs_root / "manifest.json").read_text())
    cfg = BevConfig(**bev_manifest["config"]["bev"])
    gsd = float(cfg.resolution)
    weights = FailureWeights()
    model_s1 = load_model(args.ckpt_s1, device=args.device) if args.ckpt_s1 else None
    model_s2 = load_model(args.ckpt_s2, device=args.device) if args.ckpt_s2 else None
    directional_model = _load_directional_head(args.directional_ckpt, device=args.device) if args.directional_ckpt else None

    metric_rows: List[Dict[str, float | str]] = []
    force_rows: List[Dict[str, float | str]] = []
    episodes = pair_manifest["episodes"][: args.max_episodes]
    model_episode_count = 0
    for ep in episodes:
        scene = _load_scene(args.bev_root, ep["scene_path"])
        maps = scene["maps"]
        start = tuple(int(x) for x in ep["start_rc"])
        goal = tuple(int(x) for x in ep["goal_rc"])
        stage1_path = _as_path(ep["stage1_path"])
        risk_cost_path = _as_path(ep["risk_path"])
        local_astar_path = risk_weighted_astar(maps, start, goal, risk_weight=20.0) or risk_cost_path
        reference_len = float(
            compute_path_metrics(stage1_path, maps, reference_length_m=None, gsd=gsd, weights=weights)["path_length_m"]
        )
        methods = {
            "stage1": stage1_path,
            "risk_cost_only": risk_cost_path,
            "local_risk_astar": local_astar_path,
            "stage2_analytic_force": local_astar_path,
        }
        for method, path in methods.items():
            m = compute_path_metrics(
                path,
                maps,
                reference_length_m=reference_len,
                gsd=gsd,
                weights=weights,
                goal_rc=goal,
            )
            metric_rows.append(
                {
                    "episode_id": ep["episode_id"],
                    "regime": ep["regime"],
                    "method": method,
                    **{k: v for k, v in m.items() if not isinstance(v, dict)},
                }
            )
        if (model_s1 is not None or model_s2 is not None) and (
            args.model_max_episodes is None or model_episode_count < args.model_max_episodes
        ):
            waypoints_xy = build_geom_waypoints(stage1_path, stride=6, patch_size=64)
            if model_s1 is not None:
                trace = run_model_episode(
                    model_s1,
                    maps,
                    waypoints_xy,
                    [3.0] * max(1, len(waypoints_xy)),
                    [0.08] * max(1, len(waypoints_xy)),
                    start,
                    goal,
                    device=args.device,
                    stage=1,
                    steps_per_stage=args.model_steps_per_stage,
                    patch_size=32,
                )
                m = compute_trace_metrics(trace, maps, reference_length_m=reference_len, gsd=gsd, weights=weights, goal_rc=goal)
                metric_rows.append(
                    {
                        "episode_id": ep["episode_id"],
                        "regime": ep["regime"],
                        "method": "s1_model_ckpt",
                        **{k: v for k, v in m.items() if not isinstance(v, dict)},
                    }
                )
            if model_s2 is not None:
                trace = run_model_episode(
                    model_s2,
                    maps,
                    waypoints_xy,
                    [3.0] * max(1, len(waypoints_xy)),
                    [0.08] * max(1, len(waypoints_xy)),
                    start,
                    goal,
                    device=args.device,
                    stage=2,
                    steps_per_stage=args.model_steps_per_stage,
                    patch_size=32,
                )
                m = compute_trace_metrics(trace, maps, reference_length_m=reference_len, gsd=gsd, weights=weights, goal_rc=goal)
                metric_rows.append(
                    {
                        "episode_id": ep["episode_id"],
                        "regime": ep["regime"],
                        "method": "s2_model_ckpt",
                        **{k: v for k, v in m.items() if not isinstance(v, dict)},
                    }
                )
            model_episode_count += 1
        force_rows.extend(
            _selectivity_rows(
                maps,
                stage1_path,
                regime=ep["regime"],
                episode_id=ep["episode_id"],
                lam_soft=args.lam_soft,
                lam_hard=args.lam_hard,
                gsd=gsd,
                horizon_cells=args.horizon_cells,
                hard_margin_m=args.hard_margin_m,
                improvement_margin=args.improvement_margin,
                stride=args.stride,
                force_source="analytic_fixed_lambda",
            )
        )
        if model_s2 is not None and (
            args.model_max_episodes is None or model_episode_count <= args.model_max_episodes
        ):
            force_rows.extend(
                _selectivity_rows(
                    maps,
                    stage1_path,
                    regime=ep["regime"],
                    episode_id=ep["episode_id"],
                    lam_soft=args.lam_soft,
                    lam_hard=args.lam_hard,
                    gsd=gsd,
                    horizon_cells=args.horizon_cells,
                    hard_margin_m=args.hard_margin_m,
                    improvement_margin=args.improvement_margin,
                    stride=args.stride,
                    force_source="s2_model_lambda",
                    model=model_s2,
                    device=args.device,
                    model_patch_size=32,
                    model_waypoint_stride=6,
                )
            )
        if directional_model is not None:
            force_rows.extend(
                _selectivity_rows(
                    maps,
                    stage1_path,
                    regime=ep["regime"],
                    episode_id=ep["episode_id"],
                    lam_soft=args.lam_soft,
                    lam_hard=args.lam_hard,
                    gsd=gsd,
                    horizon_cells=args.horizon_cells,
                    hard_margin_m=args.hard_margin_m,
                    improvement_margin=args.improvement_margin,
                    stride=args.stride,
                    force_source="stage2_directional_head",
                    directional_model=directional_model,
                    device=args.device,
                    directional_force_mode=args.directional_force_mode,
                    directional_force_gain=args.directional_force_gain,
                )
            )

    metric_fields = sorted({k for row in metric_rows for k in row})
    with (args.out / "aggregate_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    force_fields = sorted({k for row in force_rows for k in row})
    with (args.out / "force_samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=force_fields)
        writer.writeheader()
        writer.writerows(force_rows)

    config = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }
    summary: Dict[str, object] = {
        "config": config,
        "num_episodes": len(episodes),
        "selectivity": _summarize_selectivity(
            [r for r in force_rows if str(r.get("force_source", "analytic_fixed_lambda")) == "analytic_fixed_lambda"],
            eps=args.force_eps,
        ),
        "selectivity_by_source": _summarize_selectivity_by_source(force_rows, eps=args.force_eps),
        "metrics_by_method": {},
    }
    for method in sorted({str(r["method"]) for r in metric_rows}):
        rows = [r for r in metric_rows if r["method"] == method]
        summary["metrics_by_method"][method] = {
            key: float(np.nanmean([float(r[key]) for r in rows]))
            for key in (
                "success",
                "risk_exposure",
                "path_cvar_risk",
                "hard_hits",
                "hard_hazard_length_m",
                "barrier_violation_m",
                "path_length_ratio",
                "oscillation",
                "failure_score",
            )
            if key in rows[0]
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote RELLIS selectivity results to {args.out}")
    print(json.dumps(summary["selectivity"], indent=2))


if __name__ == "__main__":
    main()
