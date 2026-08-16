#!/usr/bin/env python3
"""One-factor ablation of the Stage-2 soft coefficient.

The three paired arms change only ``lam_soft``::

    zero:    lam_soft_used = gate * 0
    learned: lam_soft_used = gate * lam_soft_learned
    fixed:   lam_soft_used = gate * fixed_value

``lam_hard`` and every other network output are the checkpoint predictions in
all arms.  RELLIS behavioral rollouts use validation sequence 00003/R1 only;
R2/R3 are evaluated only at common reference-path points for selectivity and
coefficient-distribution statistics.  DFC uses its held-out test split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-exp9")

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DEFAULT_SOURCE = Path("/mnt/data/adityas/GRL-SNAM")
DEFAULT_RELLIS_CKPT = DEFAULT_SOURCE / "exp-rellis/checkpoints/rellis_stage2_decision_mid_ep12/best.pt"
DEFAULT_RELLIS_BEV = DEFAULT_SOURCE / "exp-rellis/cache/rellis_bev_all_seqbalanced_2500"
DEFAULT_RELLIS_PAIRS = DEFAULT_SOURCE / "exp-rellis/cache/rellis_pairs_all_seqbalanced_2500_seq00003"
DEFAULT_DFC_CKPT = DEFAULT_SOURCE / "checkpoints/s2/best.pt"
DEFAULT_DFC_ROOT = DEFAULT_SOURCE / "data/dfc2018_stagewise"
ARMS = ("zero", "learned", "fixed")

sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "full_code"))
sys.path.insert(0, str(WORKSPACE / "full_code/exp-rellis"))

from grl_rellis import BevConfig  # noqa: E402
from scripts.baselines.dfc.metrics import FailureWeights, compute_path_metrics, compute_trace_metrics  # noqa: E402
from scripts.baselines.dfc.models import (  # noqa: E402
    _build_goal_feats,
    _build_obs_feats,
    astar_geom_only,
    build_episode_waypoints,
    build_geom_waypoints,
    load_model,
)
from scripts.build_dfc2018_stagewise import (  # noqa: E402
    extract_local_geom_obstacles,
    extract_risk_patch,
    extract_rollout_patch,
)
from train_material import integrate_surrogate_material  # noqa: E402
from rebuttal_experiments.exp1_gate_ablation import primitive_feasibility_gate  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def ci95(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(1.96 * array.std(ddof=0) / math.sqrt(array.size)) if array.size else float("nan")


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector.astype(np.float32) / norm if norm > 1e-8 else np.zeros_like(vector, dtype=np.float32)


def clip_rc(point_rc: np.ndarray, shape: Tuple[int, int]) -> Tuple[int, int]:
    return (
        int(np.clip(round(float(point_rc[0])), 0, shape[0] - 1)),
        int(np.clip(round(float(point_rc[1])), 0, shape[1] - 1)),
    )


def arm_lambda(arm: str, learned: torch.Tensor, fixed: float, gate: bool) -> torch.Tensor:
    if arm == "zero":
        base = torch.zeros_like(learned)
    elif arm == "learned":
        base = learned
    elif arm == "fixed":
        base = torch.full_like(learned, float(fixed))
    else:
        raise ValueError(f"unknown arm: {arm}")
    return base * float(gate)


@torch.no_grad()
def predict_coefficients(
    model: torch.nn.Module,
    maps: Mapping[str, np.ndarray],
    position_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    device: str,
    patch_size: int,
    obstacle_patch_size: int,
    robot_radius: float,
    margin_factor: float,
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[np.ndarray, np.ndarray, np.ndarray, float], Tuple[int, int]]:
    shape = maps["risk_map"].shape
    center_rc = clip_rc(position_xy[::-1], shape)
    centers, radii, widths, d_hat = extract_local_geom_obstacles(
        maps["geom_occ"],
        center_rc,
        patch_size=obstacle_patch_size,
        robot_radius=robot_radius,
        margin_factor=margin_factor,
    )
    risk_patch_np, _ = extract_risk_patch(maps, center_rc, patch_size)
    obs_feats = _build_obs_feats(position_xy, goal_xy, centers, radii, widths, device)
    obs_mask = torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)
    goal_feats = _build_goal_feats(position_xy, goal_xy, device)
    risk_patch = torch.as_tensor(risk_patch_np, dtype=torch.float32, device=device).unsqueeze(0)
    outputs = model(obs_feats, obs_mask, goal_feats, risk_patch)
    return outputs, (centers, radii, widths, float(d_hat)), center_rc


@torch.no_grad()
def rollout(
    *,
    arm: str,
    fixed_lambda: float,
    model: torch.nn.Module,
    maps: Mapping[str, np.ndarray],
    waypoints_xy: Sequence[Tuple[float, float]],
    d_hats: Sequence[float],
    dts: Sequence[float],
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    device: str,
    steps_per_stage: int,
    patch_size: int,
    obstacle_patch_size: int,
    robot_radius: float,
    margin_factor: float,
    d_hat_sdf: float,
    goal_tol_cells: float,
    primitive_count: int,
    primitive_horizon: int,
    hard_margin: float,
    improvement_margin: float,
    material_trigger: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    shape = maps["risk_map"].shape
    position = np.asarray([start_rc[1], start_rc[0]], dtype=np.float32)
    velocity = np.zeros(2, dtype=np.float32)
    trace_xy = [position.copy()]
    coefficients: List[Dict[str, Any]] = []
    final_goal_xy = np.asarray([goal_rc[1], goal_rc[0]], dtype=np.float32)

    for stage_index, waypoint in enumerate(waypoints_xy):
        stage_goal = np.asarray(waypoint, dtype=np.float32)
        gate = primitive_feasibility_gate(
            maps,
            position,
            stage_goal,
            primitive_count=primitive_count,
            horizon_cells=primitive_horizon,
            hard_margin_m=hard_margin,
            improvement_margin=improvement_margin,
            material_trigger=material_trigger,
        )
        outputs, obstacle_data, center_rc = predict_coefficients(
            model,
            maps,
            position,
            stage_goal,
            device=device,
            patch_size=patch_size,
            obstacle_patch_size=obstacle_patch_size,
            robot_radius=robot_radius,
            margin_factor=margin_factor,
        )
        alphas, beta, gamma, learned_soft, learned_hard, _ = outputs
        used_soft = arm_lambda(arm, learned_soft, fixed_lambda, gate.active)
        centers, radii, _, extracted_d_hat = obstacle_data
        d_hat = float(d_hats[stage_index]) if stage_index < len(d_hats) else extracted_d_hat
        dt = float(dts[stage_index]) if stage_index < len(dts) else 0.04
        coefficients.append(
            {
                "stage": stage_index,
                "arm": arm,
                "gate_active": int(gate.active),
                "lambda_soft_learned": float(learned_soft.item()),
                "lambda_soft_used": float(used_soft.item()),
                "lambda_hard_learned": float(learned_hard.item()),
                "nominal_risk": gate.nominal_risk,
                "best_risk": gate.best_risk,
            }
        )

        centers_t = torch.as_tensor(centers, dtype=torch.float32, device=device).unsqueeze(0)
        radii_t = torch.as_tensor(radii, dtype=torch.float32, device=device).unsqueeze(0)
        obstacle_mask = torch.ones(1, centers.shape[0], dtype=torch.bool, device=device)
        goal_t = torch.as_tensor(stage_goal, dtype=torch.float32, device=device).unsqueeze(0)
        position_t = torch.as_tensor(position, dtype=torch.float32, device=device).unsqueeze(0)
        velocity_t = torch.as_tensor(velocity, dtype=torch.float32, device=device).unsqueeze(0)
        for _ in range(steps_per_stage):
            center_rc = clip_rc(position_t[0].cpu().numpy()[::-1], shape)
            rollout_patch_np = extract_rollout_patch(maps, center_rc, patch_size)
            rollout_patch = torch.as_tensor(rollout_patch_np, dtype=torch.float32, device=device).unsqueeze(0)
            position_t, velocity_t, _, _, _, _ = integrate_surrogate_material(
                o0=position_t.clone(),
                v0=velocity_t.clone(),
                goal=goal_t,
                C=centers_t,
                R=radii_t,
                mask=obstacle_mask,
                alphas=alphas,
                beta=beta,
                gamma=gamma,
                lam_soft=used_soft,
                lam_hard=learned_hard,
                rollout_patch=rollout_patch,
                d_hat=torch.tensor([d_hat], dtype=torch.float32, device=device),
                dt=torch.tensor([dt], dtype=torch.float32, device=device),
                H=torch.ones(1, dtype=torch.long, device=device),
                robot_radius=torch.tensor([robot_radius], dtype=torch.float32, device=device),
                margin_factor=margin_factor,
                d_hat_sdf=d_hat_sdf,
            )
            point = position_t[0].cpu().numpy()
            if not np.all(np.isfinite(point)):
                break
            trace_xy.append(point.copy())
            if float(np.linalg.norm(point - stage_goal)) < goal_tol_cells:
                break
        position = position_t[0].cpu().numpy()
        velocity = velocity_t[0].cpu().numpy()
        if not np.all(np.isfinite(position)) or float(np.linalg.norm(position - final_goal_xy)) < 2 * goal_tol_cells:
            break

    trace_xy_array = np.asarray(trace_xy, dtype=np.float32)
    return np.stack([trace_xy_array[:, 1], trace_xy_array[:, 0]], axis=-1), coefficients


@torch.no_grad()
def common_point_rows(
    *,
    dataset: str,
    episode_id: str,
    regime: str,
    model: torch.nn.Module,
    maps: Mapping[str, np.ndarray],
    path_rc: Sequence[Sequence[int]],
    device: str,
    fixed_lambda: float,
    stride: int,
    waypoint_stride: int,
    patch_size: int,
    obstacle_patch_size: int,
    robot_radius: float,
    margin_factor: float,
    primitive_count: int,
    primitive_horizon: int,
    hard_margin: float,
    improvement_margin: float,
    material_trigger: float,
    force_epsilon: float,
) -> List[Dict[str, Any]]:
    path = np.asarray(path_rc, dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    for index in range(0, max(0, len(path) - 1), max(1, stride)):
        position_rc = path[index]
        next_rc = path[min(index + 1, len(path) - 1)]
        scaffold = unit(next_rc - position_rc)
        if not np.any(scaffold):
            continue
        target_rc = path[min(index + waypoint_stride, len(path) - 1)]
        position_xy = position_rc[::-1].copy()
        target_xy = target_rc[::-1].copy()
        gate = primitive_feasibility_gate(
            maps,
            position_xy,
            target_xy,
            primitive_count=primitive_count,
            horizon_cells=primitive_horizon,
            hard_margin_m=hard_margin,
            improvement_margin=improvement_margin,
            material_trigger=material_trigger,
        )
        outputs, _, _ = predict_coefficients(
            model,
            maps,
            position_xy,
            target_xy,
            device=device,
            patch_size=patch_size,
            obstacle_patch_size=obstacle_patch_size,
            robot_radius=robot_radius,
            margin_factor=margin_factor,
        )
        learned_soft = outputs[3]
        learned_hard = outputs[4]
        rr, cc = clip_rc(position_rc, maps["risk_map"].shape)
        negative_gradient = -np.asarray([maps["grad_row"][rr, cc], maps["grad_col"][rr, cc]], dtype=np.float32)
        safe_direction = np.asarray(gate.selected_direction_rc, dtype=np.float32)
        for arm in ARMS:
            raw = float(arm_lambda(arm, learned_soft, fixed_lambda, True).item())
            used = raw * float(gate.active)
            raw_soft_force = raw * negative_gradient
            soft_force = used * negative_gradient
            perpendicular = soft_force - float(np.dot(soft_force, scaffold)) * scaffold
            raw_perpendicular = raw_soft_force - float(np.dot(raw_soft_force, scaffold)) * scaffold
            dot_safe = float(np.dot(raw_soft_force, safe_direction))
            rows.append(
                {
                    "dataset": dataset,
                    "episode_id": episode_id,
                    "regime": regime,
                    "path_index": index,
                    "arm": arm,
                    "gate_active": int(gate.active),
                    "lambda_soft_learned": float(learned_soft.item()),
                    "lambda_soft_used": used,
                    "lambda_hard_learned": float(learned_hard.item()),
                    "soft_force_norm": float(np.linalg.norm(soft_force)),
                    "soft_force_perp_norm": float(np.linalg.norm(perpendicular)),
                    "raw_soft_force_norm": float(np.linalg.norm(raw_soft_force)),
                    "raw_soft_force_perp_norm": float(np.linalg.norm(raw_perpendicular)),
                    "dot_safe": dot_safe,
                    "correct_activation": int(gate.active and dot_safe > force_epsilon),
                    "false_activation": int((not gate.active) and float(np.linalg.norm(perpendicular)) > force_epsilon),
                }
            )
    return rows


def summarize_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        result[dataset] = {}
        subset = [row for row in rows if row["dataset"] == dataset]
        for arm in ARMS:
            arm_rows = [row for row in subset if row["arm"] == arm]
            result[dataset][arm] = {
                key: {"mean": finite_mean(float(r[key]) for r in arm_rows), "ci95": ci95(float(r[key]) for r in arm_rows)}
                for key in ("success", "path_length_ratio", "risk_exposure", "mean_rho")
            }
    return result


def summarize_selectivity(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        result[dataset] = {}
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for arm in ARMS:
            arm_rows = [row for row in dataset_rows if row["arm"] == arm]
            if dataset == "RELLIS-3D":
                positive = [r for r in arm_rows if r["regime"] == "R1" and int(r["gate_active"])]
                numerator = [float(r["raw_soft_force_perp_norm"]) for r in arm_rows if r["regime"] == "R1"]
                denominator = [float(r["raw_soft_force_perp_norm"]) for r in arm_rows if r["regime"] == "R2"]
            else:
                positive = [r for r in arm_rows if int(r["gate_active"])]
                numerator = [float(r["raw_soft_force_perp_norm"]) for r in arm_rows if int(r["gate_active"])]
                denominator = [float(r["raw_soft_force_perp_norm"]) for r in arm_rows if not int(r["gate_active"])]
            car = finite_mean(float(r["correct_activation"]) for r in positive)
            num_mean = finite_mean(numerator)
            den_mean = finite_mean(denominator)
            result[dataset][arm] = {
                "correct_activation_rate": car,
                "selectivity_ratio": num_mean / max(den_mean, 1e-12) if np.isfinite(num_mean) else float("nan"),
                "positive_points": len(positive),
                "all_points": len(arm_rows),
            }
    return result


def distribution_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    learned_rows = [row for row in rows if row["arm"] == "learned"]
    groups: Dict[str, List[float]] = defaultdict(list)
    for row in learned_rows:
        groups[f'{row["dataset"]}:{row["regime"]}'].append(float(row["lambda_soft_learned"]))
    for group, values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        result[group] = {
            "n": int(array.size),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
            "median": float(np.median(array)),
            "q05": float(np.quantile(array, 0.05)),
            "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75)),
            "q95": float(np.quantile(array, 0.95)),
        }
    return result


def load_delayed_distribution(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    values: List[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("arm") == "gate_on":
                values.append(float(row["lam_soft_learned"]))
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "source": str(path),
    }


def paired_differences(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keyed = {(r["dataset"], r["episode_id"], r["arm"]): r for r in rows}
    output: List[Dict[str, Any]] = []
    for dataset, episode, arm in sorted(keyed):
        if arm != "learned":
            continue
        learned = keyed[(dataset, episode, "learned")]
        for baseline in ("zero", "fixed"):
            other = keyed.get((dataset, episode, baseline))
            if other is None:
                continue
            row: Dict[str, Any] = {"dataset": dataset, "episode_id": episode, "comparison": f"learned_minus_{baseline}"}
            for metric in ("success", "path_length_ratio", "risk_exposure", "mean_rho"):
                row[f"delta_{metric}"] = float(learned[metric]) - float(other[metric])
            output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--rellis-checkpoint", type=Path, default=DEFAULT_RELLIS_CKPT)
    parser.add_argument("--rellis-bev", type=Path, default=DEFAULT_RELLIS_BEV)
    parser.add_argument("--rellis-pairs", type=Path, default=DEFAULT_RELLIS_PAIRS)
    parser.add_argument("--dfc-checkpoint", type=Path, default=DEFAULT_DFC_CKPT)
    parser.add_argument("--dfc-root", type=Path, default=DEFAULT_DFC_ROOT)
    parser.add_argument("--out", type=Path, default=HERE / "results/exp9_soft_coefficient_isolation")
    parser.add_argument("--datasets", nargs="+", choices=["rellis", "dfc"], default=["rellis", "dfc"])
    parser.add_argument("--fixed-lambda", type=float, default=1.5)
    parser.add_argument("--max-rellis-r1", type=int, default=None)
    parser.add_argument("--max-dfc", type=int, default=None)
    parser.add_argument("--steps-per-stage", type=int, default=40)
    parser.add_argument("--point-stride", type=int, default=3)
    parser.add_argument("--waypoint-stride", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--obstacle-patch-size", type=int, default=64)
    parser.add_argument("--robot-radius", type=float, default=1.5)
    parser.add_argument("--margin-factor", type=float, default=0.5)
    parser.add_argument("--d-hat-sdf", type=float, default=3.0)
    parser.add_argument("--goal-tol-cells", type=float, default=3.0)
    parser.add_argument("--primitive-count", type=int, default=16)
    parser.add_argument("--primitive-horizon", type=int, default=12)
    parser.add_argument("--hard-margin", type=float, default=1.0)
    # The established evaluator uses a 0.1 summed-risk improvement over eight
    # cells.  The gate uses mean ray risk, so the equivalent margin is 0.0125.
    parser.add_argument("--improvement-margin", type=float, default=0.0125)
    parser.add_argument("--material-trigger", type=float, default=0.0)
    parser.add_argument("--force-epsilon", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=27370)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--distribution-only", action="store_true")
    parser.add_argument(
        "--delayed-trace",
        type=Path,
        default=HERE / "results/exp1_gate_ablation_100/step_traces.csv",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    weights = FailureWeights()
    metric_rows: List[Dict[str, Any]] = []
    point_rows: List[Dict[str, Any]] = []
    coefficient_rows: List[Dict[str, Any]] = []

    if "rellis" in args.datasets:
        model = load_model(args.rellis_checkpoint, device=args.device, patch_size=args.patch_size)
        bev_manifest = json.loads((args.rellis_bev / "manifest.json").read_text())
        pair_manifest = json.loads((args.rellis_pairs / "manifest.json").read_text())
        gsd = float(BevConfig(**bev_manifest["config"]["bev"]).resolution)
        episodes = pair_manifest["episodes"]
        scene_cache: Dict[str, Mapping[str, Any]] = {}
        r1_seen = 0
        for ep in episodes:
            regime = str(ep["regime"])
            if regime == "R1" and args.max_rellis_r1 is not None and r1_seen >= args.max_rellis_r1:
                continue
            if regime == "R1":
                r1_seen += 1
            scene_path = str(ep["scene_path"])
            if scene_path not in scene_cache:
                scene_cache[scene_path] = torch.load(args.rellis_bev / scene_path, map_location="cpu", weights_only=False)
            maps = scene_cache[scene_path]["maps"]
            path = [(int(p[0]), int(p[1])) for p in ep["stage1_path"]]
            point_rows.extend(
                common_point_rows(
                    dataset="RELLIS-3D",
                    episode_id=str(ep["episode_id"]),
                    regime=regime,
                    model=model,
                    maps=maps,
                    path_rc=path,
                    device=args.device,
                    fixed_lambda=args.fixed_lambda,
                    stride=args.point_stride,
                    waypoint_stride=args.waypoint_stride,
                    patch_size=args.patch_size,
                    obstacle_patch_size=args.obstacle_patch_size,
                    robot_radius=args.robot_radius,
                    margin_factor=args.margin_factor,
                    primitive_count=args.primitive_count,
                    primitive_horizon=args.primitive_horizon,
                    hard_margin=args.hard_margin,
                    improvement_margin=args.improvement_margin,
                    material_trigger=args.material_trigger,
                    force_epsilon=args.force_epsilon,
                )
            )
            if regime != "R1" or args.distribution_only:
                continue
            start = tuple(int(x) for x in ep["start_rc"])
            goal = tuple(int(x) for x in ep["goal_rc"])
            reference = compute_path_metrics(path, maps, reference_length_m=None, gsd=gsd, weights=weights, goal_rc=goal)
            waypoints = build_geom_waypoints(path, stride=args.waypoint_stride, patch_size=64)
            for arm in ARMS:
                trace, coeffs = rollout(
                    arm=arm,
                    fixed_lambda=args.fixed_lambda,
                    model=model,
                    maps=maps,
                    waypoints_xy=waypoints,
                    d_hats=[3.0] * len(waypoints),
                    dts=[0.08] * len(waypoints),
                    start_rc=start,
                    goal_rc=goal,
                    device=args.device,
                    steps_per_stage=args.steps_per_stage,
                    patch_size=args.patch_size,
                    obstacle_patch_size=args.obstacle_patch_size,
                    robot_radius=args.robot_radius,
                    margin_factor=args.margin_factor,
                    d_hat_sdf=args.d_hat_sdf,
                    goal_tol_cells=args.goal_tol_cells,
                    primitive_count=args.primitive_count,
                    primitive_horizon=args.primitive_horizon,
                    hard_margin=args.hard_margin,
                    improvement_margin=args.improvement_margin,
                    material_trigger=args.material_trigger,
                )
                metrics = compute_trace_metrics(trace, maps, reference_length_m=float(reference["path_length_m"]), gsd=gsd, weights=weights, goal_rc=goal)
                metric_rows.append({"dataset": "RELLIS-3D", "episode_id": str(ep["episode_id"]), "regime": regime, "arm": arm, **{k: v for k, v in metrics.items() if not isinstance(v, dict)}})
                coefficient_rows.extend({"dataset": "RELLIS-3D", "episode_id": str(ep["episode_id"]), "regime": regime, **row} for row in coeffs)

    if "dfc" in args.datasets:
        model = load_model(args.dfc_checkpoint, device=args.device, patch_size=args.patch_size)
        manifest = json.loads((args.dfc_root / "manifest.json").read_text())
        test_entries = [entry for entry in manifest if entry["split"] == "test"]
        if args.max_dfc is not None:
            test_entries = test_entries[: args.max_dfc]
        scene_cache = {}
        for entry in test_entries:
            episode_path = args.source_root / str(entry["path"])
            episode = torch.load(episode_path, map_location="cpu", weights_only=False)
            scene_id = str(entry["scene_id"])
            if scene_id not in scene_cache:
                scene_cache[scene_id] = torch.load(args.dfc_root / f"scene_{scene_id}.pt", map_location="cpu", weights_only=False)
            maps = scene_cache[scene_id]["maps"]
            start = tuple(int(x) for x in entry["start_rc"])
            goal = tuple(int(x) for x in entry["goal_rc"])
            path = astar_geom_only(maps, start, goal) or [start, goal]
            point_rows.extend(
                common_point_rows(
                    dataset="DFC2018",
                    episode_id=str(entry["episode_id"]),
                    regime="soft-dominant-test",
                    model=model,
                    maps=maps,
                    path_rc=path,
                    device=args.device,
                    fixed_lambda=args.fixed_lambda,
                    stride=max(args.point_stride, 6),
                    waypoint_stride=args.waypoint_stride,
                    patch_size=args.patch_size,
                    obstacle_patch_size=args.obstacle_patch_size,
                    robot_radius=args.robot_radius,
                    margin_factor=args.margin_factor,
                    primitive_count=args.primitive_count,
                    primitive_horizon=args.primitive_horizon,
                    hard_margin=args.hard_margin,
                    improvement_margin=args.improvement_margin,
                    material_trigger=args.material_trigger,
                    force_epsilon=args.force_epsilon,
                )
            )
            if args.distribution_only:
                continue
            checkpoints_path = args.source_root / str(episode["logs"]["checkpoints_jsonl"])
            with checkpoints_path.open() as handle:
                checkpoints = [json.loads(line) for line in handle]
            # Match the paper evaluator's end-to-end waypoint construction.
            proxy = dict(episode)
            proxy["logs"] = {"checkpoints_jsonl": str(checkpoints_path)}
            waypoints, d_hats, dts = build_episode_waypoints(proxy, maps, start, goal, checkpoints, eval_mode="endtoend", patch_size=64)
            gsd = float(episode["meta"]["gsd"])
            reference_length = float(entry["path_length_m"])
            for arm in ARMS:
                trace, coeffs = rollout(
                    arm=arm,
                    fixed_lambda=args.fixed_lambda,
                    model=model,
                    maps=maps,
                    waypoints_xy=waypoints,
                    d_hats=d_hats,
                    dts=dts,
                    start_rc=start,
                    goal_rc=goal,
                    device=args.device,
                    steps_per_stage=args.steps_per_stage,
                    patch_size=args.patch_size,
                    obstacle_patch_size=args.obstacle_patch_size,
                    robot_radius=args.robot_radius,
                    margin_factor=args.margin_factor,
                    d_hat_sdf=args.d_hat_sdf,
                    goal_tol_cells=args.goal_tol_cells,
                    primitive_count=args.primitive_count,
                    primitive_horizon=args.primitive_horizon,
                    hard_margin=args.hard_margin,
                    improvement_margin=args.improvement_margin,
                    material_trigger=args.material_trigger,
                )
                metrics = compute_trace_metrics(trace, maps, reference_length_m=reference_length, gsd=gsd, weights=weights, goal_rc=goal)
                metric_rows.append({"dataset": "DFC2018", "episode_id": str(entry["episode_id"]), "regime": "soft-dominant-test", "arm": arm, **{k: v for k, v in metrics.items() if not isinstance(v, dict)}})
                coefficient_rows.extend({"dataset": "DFC2018", "episode_id": str(entry["episode_id"]), "regime": "soft-dominant-test", **row} for row in coeffs)

    write_csv(args.out / "per_episode_metrics.csv", metric_rows)
    write_csv(args.out / "common_point_selectivity.csv", point_rows)
    write_csv(args.out / "rollout_coefficients.csv", coefficient_rows)
    differences = paired_differences(metric_rows)
    write_csv(args.out / "paired_differences.csv", differences)
    summary = {
        "design": {
            "arms": {
                "zero": "lambda_soft_used = gate * 0",
                "learned": "lambda_soft_used = gate * checkpoint prediction",
                "fixed": f"lambda_soft_used = gate * {args.fixed_lambda}",
            },
            "invariant": "lambda_hard and all other network outputs are unchanged",
            "rellis_behavior_split": "sequence 00003 validation, R1 only",
            "dfc_behavior_split": "held-out test",
        },
        "metrics": summarize_metrics(metric_rows),
        "selectivity": summarize_selectivity(point_rows),
        "lambda_soft_distribution": distribution_summary(point_rows),
        "delayed_escape_lambda_soft_distribution": load_delayed_distribution(args.delayed_trace),
        "paired_difference_means": {
            comparison: {
                key: finite_mean(float(r[key]) for r in differences if r["comparison"] == comparison)
                for key in ("delta_success", "delta_path_length_ratio", "delta_risk_exposure", "delta_mean_rho")
            }
            for comparison in ("learned_minus_zero", "learned_minus_fixed")
        },
        "provenance": {
            "rellis_checkpoint": str(args.rellis_checkpoint),
            "rellis_checkpoint_sha256": sha256(args.rellis_checkpoint) if "rellis" in args.datasets else None,
            "dfc_checkpoint": str(args.dfc_checkpoint),
            "dfc_checkpoint_sha256": sha256(args.dfc_checkpoint) if "dfc" in args.datasets else None,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
