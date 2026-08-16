#!/usr/bin/env python3
"""Experiment 1: paired, learned Stage-2 soft-gate ablation on RELLIS-Dyn.

This harness intentionally does *not* call ``eval_rellis_dyn._route_aware_step``.
That legacy method is a discrete route-following heuristic.  Here the canonical
Stage-2 ``CoefEnergyNetMaterial`` checkpoint drives
``integrate_surrogate_material`` directly.

The paired intervention is exactly:

    gate_on:  lam_soft_used = primitive_feasibility_gate * lam_soft_learned
    gate_off: lam_soft_used = 1.0                        * lam_soft_learned

The learned hard-material coefficient is never gated.  All other model
parameters, episode/event definitions, scaffold waypoints, and rollout
hyperparameters are shared by the two arms.
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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-exp1")

DEFAULT_SOURCE_ROOT = Path("/mnt/data/adityas/GRL-SNAM")
DEFAULT_CHECKPOINT = (
    DEFAULT_SOURCE_ROOT
    / "exp-rellis/checkpoints/rellis_stage2_decision_mid_ep12/best.pt"
)
DEFAULT_BEV_ROOT = (
    DEFAULT_SOURCE_ROOT / "exp-rellis/cache/rellis_bev_all_seqbalanced_2500"
)
DEFAULT_PAIRS_ROOT = (
    DEFAULT_SOURCE_ROOT
    / "exp-rellis/cache/rellis_pairs_all_seqbalanced_2500_loso"
)


@dataclass(frozen=True)
class GateDecision:
    active: bool
    nominal_risk: float
    best_risk: float
    feasible_count: int
    selected_direction_rc: Tuple[float, float]
    selected_endpoint_rc: Tuple[float, float]
    selected_min_clearance_m: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / norm).astype(np.float32)


def _clip_rc(point_rc: np.ndarray, shape: Tuple[int, int]) -> Tuple[int, int]:
    return (
        int(np.clip(round(float(point_rc[0])), 0, shape[0] - 1)),
        int(np.clip(round(float(point_rc[1])), 0, shape[1] - 1)),
    )


def _nearest_path_index(
    path_rc: np.ndarray, point_rc: np.ndarray, previous_index: int
) -> int:
    if len(path_rc) == 0:
        return 0
    lo = max(0, previous_index - 5)
    distances = np.linalg.norm(path_rc[lo:] - point_rc[None, :], axis=1)
    return lo + int(np.argmin(distances))


def _stage_goal_xy(
    stage1_path_rc: np.ndarray,
    position_xy: np.ndarray,
    previous_index: int,
    lookahead_cells: int,
) -> Tuple[np.ndarray, int]:
    position_rc = position_xy[::-1]
    nearest = _nearest_path_index(stage1_path_rc, position_rc, previous_index)
    target_index = min(nearest + lookahead_cells, len(stage1_path_rc) - 1)
    target_rc = stage1_path_rc[target_index]
    return target_rc[::-1].astype(np.float32), nearest


def _ray_cost(
    maps: Mapping[str, np.ndarray],
    position_rc: np.ndarray,
    direction_rc: np.ndarray,
    *,
    horizon_cells: int,
    hard_margin_m: float,
) -> Tuple[float, bool, float]:
    risk = maps["risk_map"]
    hard = maps["hard_mask"].astype(bool)
    sdf = maps["sdf_hard"]
    values: List[float] = []
    min_clearance = float("inf")
    feasible = True
    for distance in range(1, horizon_cells + 1):
        query = position_rc + float(distance) * direction_rc
        if not (
            0.0 <= query[0] < risk.shape[0]
            and 0.0 <= query[1] < risk.shape[1]
        ):
            feasible = False
            break
        cell = _clip_rc(query, risk.shape)
        values.append(float(risk[cell]))
        clearance = float(sdf[cell])
        min_clearance = min(min_clearance, clearance)
        if bool(hard[cell]) or clearance < hard_margin_m:
            feasible = False
            break
    mean_risk = float(np.mean(values)) if values else float("inf")
    return mean_risk, feasible, min_clearance


def primitive_feasibility_gate(
    maps: Mapping[str, np.ndarray],
    position_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    primitive_count: int,
    horizon_cells: int,
    hard_margin_m: float,
    improvement_margin: float,
    material_trigger: float,
) -> GateDecision:
    """Test whether a feasible, progress-making ray improves on the nominal ray.

    The gate is a local activation witness only; it does not choose the executed
    action.  Candidate rays are uniformly distributed over 360 degrees.  A ray
    is eligible only if every sample clears hard terrain and its endpoint makes
    progress toward the current stage goal.
    """
    position_rc = position_xy[::-1].astype(np.float32)
    goal_rc = goal_xy[::-1].astype(np.float32)
    nominal_direction = _unit(goal_rc - position_rc)
    nominal_risk, _, _ = _ray_cost(
        maps,
        position_rc,
        nominal_direction,
        horizon_cells=horizon_cells,
        hard_margin_m=hard_margin_m,
    )

    best_risk = float("inf")
    best_direction = np.zeros(2, dtype=np.float32)
    best_min_clearance = float("nan")
    feasible_count = 0
    current_goal_distance = float(np.linalg.norm(goal_rc - position_rc))
    for index in range(primitive_count):
        angle = 2.0 * math.pi * float(index) / float(primitive_count)
        direction = np.asarray([math.sin(angle), math.cos(angle)], dtype=np.float32)
        endpoint = position_rc + float(horizon_cells) * direction
        if float(np.linalg.norm(goal_rc - endpoint)) >= current_goal_distance - 0.5:
            continue
        candidate_risk, feasible, candidate_min_clearance = _ray_cost(
            maps,
            position_rc,
            direction,
            horizon_cells=horizon_cells,
            hard_margin_m=hard_margin_m,
        )
        if not feasible:
            continue
        feasible_count += 1
        if candidate_risk < best_risk:
            best_risk = candidate_risk
            best_direction = direction
            best_min_clearance = candidate_min_clearance

    active = (
        feasible_count > 0
        and nominal_risk >= material_trigger
        and nominal_risk - best_risk >= improvement_margin
    )
    return GateDecision(
        active=bool(active),
        nominal_risk=nominal_risk,
        best_risk=best_risk,
        feasible_count=feasible_count,
        selected_direction_rc=(float(best_direction[0]), float(best_direction[1])),
        selected_endpoint_rc=(
            float(position_rc[0] + horizon_cells * best_direction[0]),
            float(position_rc[1] + horizon_cells * best_direction[1]),
        ),
        selected_min_clearance_m=best_min_clearance,
    )


def _build_obs_feats(
    position_xy: np.ndarray,
    goal_xy: np.ndarray,
    centers_xy: np.ndarray,
    radii: np.ndarray,
    widths: np.ndarray,
    device: str,
) -> torch.Tensor:
    del position_xy
    if centers_xy.shape[0] == 0:
        return torch.zeros(1, 0, 6, dtype=torch.float32, device=device)
    centers = torch.as_tensor(centers_xy, dtype=torch.float32, device=device)
    radii_t = torch.as_tensor(radii, dtype=torch.float32, device=device)
    widths_t = torch.as_tensor(widths, dtype=torch.float32, device=device)
    goal = torch.as_tensor(goal_xy, dtype=torch.float32, device=device)
    goal_offsets = goal.unsqueeze(0) - centers
    return torch.cat(
        [centers, radii_t[:, None], widths_t[:, None], goal_offsets], dim=-1
    ).unsqueeze(0)


def _build_goal_feats(
    position_xy: np.ndarray, goal_xy: np.ndarray, device: str
) -> torch.Tensor:
    position = torch.as_tensor(position_xy, dtype=torch.float32, device=device)
    goal = torch.as_tensor(goal_xy, dtype=torch.float32, device=device)
    delta = goal - position
    return torch.cat(
        [delta, torch.linalg.norm(delta).view(1), torch.ones(1, device=device)]
    ).unsqueeze(0)


def _weighted_upper_tail(
    values: Sequence[float], weights: Sequence[float], tail_fraction: float = 0.2
) -> float:
    if not values:
        return 0.0
    val = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if float(weight.sum()) <= 1e-12:
        return float(np.max(val))
    order = np.argsort(val)[::-1]
    val = val[order]
    weight = weight[order]
    target = tail_fraction * float(weight.sum())
    used = 0.0
    total = 0.0
    for item, item_weight in zip(val, weight):
        take = min(float(item_weight), target - used)
        if take <= 0.0:
            break
        total += float(item) * take
        used += take
    return total / max(used, 1e-12)


def _load_model(
    checkpoint: Path, source_root: Path, device: str
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "exp-rellis"))
    from train_material import CoefEnergyNetMaterial

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = dict(payload.get("cfg", {}))
    model = CoefEnergyNetMaterial(
        patch_size=int(cfg.get("patch_size", 32)),
        lam_soft_max=float(cfg.get("lam_soft_max", 5.0)),
        lam_hard_max=float(cfg.get("lam_hard_max", 10.0)),
        mu_lat_max=float(cfg.get("mu_lat_max", 5.0)),
    )
    state = payload.get("model_state_dict", payload.get("model", payload))
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def _rollout_arm(
    *,
    gate_enabled: bool,
    model: torch.nn.Module,
    base_maps: Mapping[str, np.ndarray],
    spec: Any,
    episode: Mapping[str, Any],
    source_root: Path,
    device: str,
    gsd: float,
    max_steps: int,
    dt: float,
    stage_lookahead_cells: int,
    patch_size: int,
    obstacle_patch_size: int,
    robot_radius: float,
    margin_factor: float,
    d_hat_sdf: float,
    primitive_count: int,
    primitive_horizon_cells: int,
    hard_margin_m: float,
    improvement_margin: float,
    material_trigger: float,
    hard_violation_penalty: float,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from grl_rellis.dyn_events import apply_dynamic_event
    from scripts.build_dfc2018_stagewise import (
        extract_local_geom_obstacles,
        extract_risk_patch,
        extract_rollout_patch,
    )
    from train_material import integrate_surrogate_material

    del source_root
    start_rc = np.asarray(episode["start_rc"], dtype=np.float32)
    goal_rc = np.asarray(episode["goal_rc"], dtype=np.float32)
    goal_xy = goal_rc[::-1].copy()
    stage1_path = np.asarray(episode["stage1_path"], dtype=np.float32)
    position = start_rc[::-1].copy()
    velocity = np.zeros(2, dtype=np.float32)
    path_index = 0
    trace: List[Dict[str, Any]] = []
    risks: List[float] = []
    violations: List[float] = []
    movement_weights: List[float] = []
    path_length_m = 0.0
    risk_exposure = 0.0
    hard_length_m = 0.0
    hard_contacts = 0
    out_of_bounds = False
    start_time = time.perf_counter()

    for step in range(max_steps):
        maps = apply_dynamic_event(base_maps, spec, step, resolution=gsd)
        shape = maps["risk_map"].shape
        if not (
            0.0 <= position[0] < shape[1] and 0.0 <= position[1] < shape[0]
        ):
            out_of_bounds = True
            break
        stage_goal, path_index = _stage_goal_xy(
            stage1_path, position, path_index, stage_lookahead_cells
        )
        gate = primitive_feasibility_gate(
            maps,
            position,
            stage_goal,
            primitive_count=primitive_count,
            horizon_cells=primitive_horizon_cells,
            hard_margin_m=hard_margin_m,
            improvement_margin=improvement_margin,
            material_trigger=material_trigger,
        )

        center_rc = _clip_rc(position[::-1], shape)
        # The legacy extractor randomly thins dense boundaries.  Seeding it by
        # paired episode and time makes obstacle token construction reproducible.
        np.random.seed(seed + step)
        centers, radii, widths, d_hat_value = extract_local_geom_obstacles(
            maps["geom_occ"],
            center_rc,
            patch_size=obstacle_patch_size,
            robot_radius=robot_radius,
            margin_factor=margin_factor,
        )
        risk_patch_np, _ = extract_risk_patch(maps, center_rc, patch_size)
        obs_feats = _build_obs_feats(
            position, stage_goal, centers, radii, widths, device
        )
        obs_mask = torch.ones(
            1, obs_feats.shape[1], dtype=torch.bool, device=device
        )
        goal_feats = _build_goal_feats(position, stage_goal, device)
        risk_patch = torch.as_tensor(
            risk_patch_np, dtype=torch.float32, device=device
        ).unsqueeze(0)
        alphas, beta, gamma, lam_soft, lam_hard, mu_lat = model(
            obs_feats, obs_mask, goal_feats, risk_patch
        )
        del mu_lat

        multiplier = float(gate.active) if gate_enabled else 1.0
        lam_soft_used = lam_soft * multiplier
        rollout_patch_np = extract_rollout_patch(maps, center_rc, patch_size)
        rollout_patch = torch.as_tensor(
            rollout_patch_np, dtype=torch.float32, device=device
        ).unsqueeze(0)
        centers_t = torch.as_tensor(
            centers, dtype=torch.float32, device=device
        ).unsqueeze(0)
        radii_t = torch.as_tensor(
            radii, dtype=torch.float32, device=device
        ).unsqueeze(0)
        obstacle_mask = torch.ones(
            1, centers.shape[0], dtype=torch.bool, device=device
        )
        position_t = torch.as_tensor(
            position, dtype=torch.float32, device=device
        ).unsqueeze(0)
        velocity_t = torch.as_tensor(
            velocity, dtype=torch.float32, device=device
        ).unsqueeze(0)
        next_position_t, next_velocity_t, _, _, _, _ = (
            integrate_surrogate_material(
                o0=position_t,
                v0=velocity_t,
                goal=torch.as_tensor(
                    stage_goal, dtype=torch.float32, device=device
                ).unsqueeze(0),
                C=centers_t,
                R=radii_t,
                mask=obstacle_mask,
                alphas=alphas,
                beta=beta,
                gamma=gamma,
                lam_soft=lam_soft_used,
                lam_hard=lam_hard,
                rollout_patch=rollout_patch,
                d_hat=torch.tensor(
                    [d_hat_value], dtype=torch.float32, device=device
                ),
                dt=torch.tensor([dt], dtype=torch.float32, device=device),
                H=torch.ones(1, dtype=torch.long, device=device),
                robot_radius=torch.tensor(
                    [robot_radius], dtype=torch.float32, device=device
                ),
                margin_factor=margin_factor,
                d_hat_sdf=d_hat_sdf,
            )
        )
        next_position = next_position_t[0].cpu().numpy()
        next_velocity = next_velocity_t[0].cpu().numpy()
        if not np.all(np.isfinite(next_position)) or not np.all(
            np.isfinite(next_velocity)
        ):
            out_of_bounds = True
            break

        movement_m = gsd * float(np.linalg.norm(next_position - position))
        sample_rc = _clip_rc(next_position[::-1], shape)
        risk_value = float(maps["risk_map"][sample_rc])
        sdf_clearance = float(maps["sdf_hard"][sample_rc])
        hard_value = bool(maps["hard_mask"][sample_rc])
        violation = risk_value + hard_violation_penalty * float(hard_value)
        weight = max(movement_m, 1e-9)
        path_length_m += movement_m
        risk_exposure += movement_m * risk_value
        hard_length_m += movement_m * float(hard_value)
        hard_contacts += int(hard_value)
        risks.append(risk_value)
        violations.append(violation)
        movement_weights.append(weight)
        trace.append(
            {
                "episode_id": str(episode["episode_id"]),
                "arm": "gate_on" if gate_enabled else "gate_off",
                "step": step,
                "event_step": int(spec.event_step),
                "opening_step": int(spec.event_step + spec.open_delay),
                "position_x": float(position[0]),
                "position_y": float(position[1]),
                "next_x": float(next_position[0]),
                "next_y": float(next_position[1]),
                "speed": float(np.linalg.norm(next_velocity)),
                "stage_goal_x": float(stage_goal[0]),
                "stage_goal_y": float(stage_goal[1]),
                "gate_decision": int(gate.active),
                "soft_multiplier": multiplier,
                "nominal_primitive_risk": gate.nominal_risk,
                "best_feasible_primitive_risk": gate.best_risk,
                "feasible_primitive_count": gate.feasible_count,
                "selected_direction_row": gate.selected_direction_rc[0],
                "selected_direction_col": gate.selected_direction_rc[1],
                "selected_endpoint_row": gate.selected_endpoint_rc[0],
                "selected_endpoint_col": gate.selected_endpoint_rc[1],
                "selected_ray_min_clearance_m": gate.selected_min_clearance_m,
                "predicted_risk_improvement": (
                    gate.nominal_risk - gate.best_risk
                    if np.isfinite(gate.best_risk)
                    else float("nan")
                ),
                "lam_soft_learned": float(lam_soft.item()),
                "lam_soft_used": float(lam_soft_used.item()),
                "lam_hard_used": float(lam_hard.item()),
                "risk": risk_value,
                "sdf_clearance_m": sdf_clearance,
                "hard_contact": int(hard_value),
                "movement_m": movement_m,
            }
        )
        position = next_position
        velocity = next_velocity
        if gsd * float(np.linalg.norm(position - goal_xy)) <= 3.0:
            break

    compute_seconds = time.perf_counter() - start_time
    opening_step = int(spec.event_step + spec.open_delay)
    pre_window = [
        row
        for row in trace
        if int(spec.event_step) <= int(row["step"]) < opening_step
    ]
    post_window = [row for row in trace if int(row["step"]) >= opening_step]
    first_post_activation = next(
        (
            int(row["step"])
            for row in post_window
            if float(row["soft_multiplier"]) > 0.5
        ),
        None,
    )
    final_distance_m = gsd * float(np.linalg.norm(position - goal_xy))
    metric = {
        "episode_id": str(episode["episode_id"]),
        "scene_id": str(episode["scene_id"]),
        "sequence": str(episode["sequence"]),
        "regime": str(episode["regime"]),
        "event_type": str(spec.event_type),
        "arm": "gate_on" if gate_enabled else "gate_off",
        "steps": len(trace),
        "success": float(final_distance_m <= 3.0),
        "final_distance_m": final_distance_m,
        "path_length_m": path_length_m,
        "risk_exposure": risk_exposure,
        "mean_risk": float(np.average(risks, weights=movement_weights))
        if risks
        else 0.0,
        "cvar20_risk": _weighted_upper_tail(risks, movement_weights),
        "cvar20_violation": _weighted_upper_tail(violations, movement_weights),
        "hard_contacts": hard_contacts,
        "hard_hazard_length_m": hard_length_m,
        "gate_diagnostic_rate": float(
            np.mean([float(row["gate_decision"]) for row in trace])
        )
        if trace
        else 0.0,
        "effective_soft_rate": float(
            np.mean([float(row["soft_multiplier"]) > 0.5 for row in trace])
        )
        if trace
        else 0.0,
        "false_pre_activation_rate": float(
            np.mean([float(row["soft_multiplier"]) > 0.5 for row in pre_window])
        )
        if pre_window
        else 0.0,
        "post_open_activation_rate": float(
            np.mean([float(row["soft_multiplier"]) > 0.5 for row in post_window])
        )
        if post_window
        else 0.0,
        "post_open_activation_delay": float(
            first_post_activation - opening_step
            if first_post_activation is not None
            else max_steps - opening_step
        ),
        "mean_lam_soft_learned": float(
            np.mean([float(row["lam_soft_learned"]) for row in trace])
        )
        if trace
        else 0.0,
        "mean_lam_soft_used": float(
            np.mean([float(row["lam_soft_used"]) for row in trace])
        )
        if trace
        else 0.0,
        "mean_lam_hard_used": float(
            np.mean([float(row["lam_hard_used"]) for row in trace])
        )
        if trace
        else 0.0,
        "compute_ms_per_step": 1000.0 * compute_seconds / max(1, len(trace)),
        "out_of_bounds_or_nonfinite": int(out_of_bounds),
    }
    return metric, trace


def _paired_rows(metrics: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in metrics:
        grouped.setdefault(str(row["episode_id"]), {})[str(row["arm"])] = row
    result: List[Dict[str, Any]] = []
    delta_metrics = (
        "success",
        "final_distance_m",
        "path_length_m",
        "risk_exposure",
        "mean_risk",
        "cvar20_risk",
        "cvar20_violation",
        "hard_contacts",
        "hard_hazard_length_m",
        "false_pre_activation_rate",
        "post_open_activation_rate",
        "post_open_activation_delay",
    )
    for episode_id, arms in grouped.items():
        if set(arms) != {"gate_on", "gate_off"}:
            continue
        on = arms["gate_on"]
        off = arms["gate_off"]
        row: Dict[str, Any] = {
            "episode_id": episode_id,
            "scene_id": on["scene_id"],
            "regime": on["regime"],
        }
        for metric in delta_metrics:
            row[f"gate_on_{metric}"] = on[metric]
            row[f"gate_off_{metric}"] = off[metric]
            row[f"delta_on_minus_off_{metric}"] = float(on[metric]) - float(
                off[metric]
            )
        result.append(row)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired learned Stage-2 gate-on/off RELLIS-Dyn ablation."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bev-root", type=Path, default=DEFAULT_BEV_ROOT)
    parser.add_argument("--pairs-root", type=Path, default=DEFAULT_PAIRS_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rebuttal_experiments/results/exp1_gate_ablation"),
    )
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["gate_on", "gate_off"],
        default=["gate_on", "gate_off"],
        help="Rollout arms to execute; default preserves the paired ablation.",
    )
    parser.add_argument(
        "--event-type", choices=["delayed_required_escape"], default="delayed_required_escape"
    )
    parser.add_argument("--event-fraction", type=float, default=0.38)
    parser.add_argument("--event-duration", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=140)
    # The RELLIS-Dyn event clock advances once per controller update.  At 0.04
    # the continuous learned rollout finishes these episodes before the event
    # begins; 0.01 keeps the event timing comparable to the cached grid rollout.
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--stage-lookahead-cells", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--obstacle-patch-size", type=int, default=64)
    parser.add_argument("--robot-radius", type=float, default=1.5)
    parser.add_argument("--margin-factor", type=float, default=0.5)
    parser.add_argument("--d-hat-sdf", type=float, default=3.0)
    parser.add_argument("--primitive-count", type=int, default=16)
    parser.add_argument("--primitive-horizon-cells", type=int, default=12)
    parser.add_argument("--hard-margin-m", type=float, default=1.0)
    parser.add_argument("--improvement-margin", type=float, default=0.05)
    parser.add_argument("--material-trigger", type=float, default=0.45)
    parser.add_argument("--hard-violation-penalty", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=27370)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    sys.path.insert(0, str(args.source_root))
    sys.path.insert(0, str(args.source_root / "exp-rellis"))
    from grl_rellis import BevConfig
    from grl_rellis.dyn_events import make_event_spec

    model, checkpoint_cfg = _load_model(
        args.checkpoint, args.source_root, args.device
    )
    bev_manifest_path = args.bev_root / "manifest.json"
    pair_manifest_path = args.pairs_root / "manifest.json"
    bev_manifest = json.loads(bev_manifest_path.read_text())
    pair_manifest = json.loads(pair_manifest_path.read_text())
    gsd = float(BevConfig(**bev_manifest["config"]["bev"]).resolution)
    episodes = pair_manifest["episodes"][: args.max_episodes]

    config = {
        "experiment": "exp1_same_checkpoint_soft_gate_ablation",
        "intervention": {
            "gate_on": "lam_soft_used = primitive_feasibility_gate * lam_soft_learned",
            "gate_off": "lam_soft_used = 1.0 * lam_soft_learned",
            "lam_hard": "learned value unchanged in both arms",
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "provenance": {
            "checkpoint_sha256": _sha256(args.checkpoint),
            "bev_manifest_sha256": _sha256(bev_manifest_path),
            "pairs_manifest_sha256": _sha256(pair_manifest_path),
            "checkpoint_cfg": checkpoint_cfg,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "original_integrator": str(args.source_root / "train_material.py"),
            "original_event_builder": str(
                args.source_root / "exp-rellis/grl_rellis/dyn_events.py"
            ),
        },
        "pairing": {
            "episode_ids": [str(episode["episode_id"]) for episode in episodes],
            "arm_order": list(args.arms),
            "shared_seed": args.seed,
        },
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    metrics: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    event_specs: List[Dict[str, Any]] = []
    scene_cache: Dict[str, Dict[str, Any]] = {}
    for episode_index, episode in enumerate(episodes):
        scene_path = str(episode["scene_path"])
        if scene_path not in scene_cache:
            scene_cache[scene_path] = torch.load(
                args.bev_root / scene_path,
                map_location="cpu",
                weights_only=False,
            )
        base_maps = scene_cache[scene_path]["maps"]
        spec = make_event_spec(
            args.event_type,
            episode["stage1_path"],
            episode["risk_path"],
            episode["goal_rc"],
            event_fraction=args.event_fraction,
            duration=args.event_duration,
        )
        event_specs.append(
            {
                "episode_id": str(episode["episode_id"]),
                "scene_id": str(episode["scene_id"]),
                "regime": str(episode["regime"]),
                **spec.to_dict(),
            }
        )
        for arm in args.arms:
            gate_enabled = arm == "gate_on"
            metric, trace = _rollout_arm(
                gate_enabled=gate_enabled,
                model=model,
                base_maps=base_maps,
                spec=spec,
                episode=episode,
                source_root=args.source_root,
                device=args.device,
                gsd=gsd,
                max_steps=args.max_steps,
                dt=args.dt,
                stage_lookahead_cells=args.stage_lookahead_cells,
                patch_size=args.patch_size,
                obstacle_patch_size=args.obstacle_patch_size,
                robot_radius=args.robot_radius,
                margin_factor=args.margin_factor,
                d_hat_sdf=args.d_hat_sdf,
                primitive_count=args.primitive_count,
                primitive_horizon_cells=args.primitive_horizon_cells,
                hard_margin_m=args.hard_margin_m,
                improvement_margin=args.improvement_margin,
                material_trigger=args.material_trigger,
                hard_violation_penalty=args.hard_violation_penalty,
                seed=args.seed + 10000 * episode_index,
            )
            metrics.append(metric)
            traces.extend(trace)
            print(
                f"episode={episode['episode_id']} arm={metric['arm']} "
                f"steps={metric['steps']} success={metric['success']:.0f} "
                f"cvarV={metric['cvar20_violation']:.4f} "
                f"preAct={metric['false_pre_activation_rate']:.3f} "
                f"postAct={metric['post_open_activation_rate']:.3f}",
                flush=True,
            )

    paired = _paired_rows(metrics)
    _write_csv(args.out / "per_episode_metrics.csv", metrics)
    _write_csv(args.out / "paired_differences.csv", paired)
    _write_csv(args.out / "step_traces.csv", traces)
    _write_csv(args.out / "event_specs.csv", event_specs)
    summary = {
        "num_episodes": len(episodes),
        "num_rollouts": len(metrics),
        "arms": {},
        "mean_paired_delta_on_minus_off": {},
    }
    for arm in args.arms:
        pool = [row for row in metrics if row["arm"] == arm]
        summary["arms"][arm] = {
            key: float(np.mean([float(row[key]) for row in pool]))
            for key in (
                "success",
                "final_distance_m",
                "path_length_m",
                "risk_exposure",
                "cvar20_risk",
                "cvar20_violation",
                "hard_contacts",
                "false_pre_activation_rate",
                "post_open_activation_rate",
                "post_open_activation_delay",
            )
        }
    if paired:
        delta_keys = [
            key for key in paired[0] if key.startswith("delta_on_minus_off_")
        ]
        summary["mean_paired_delta_on_minus_off"] = {
            key.removeprefix("delta_on_minus_off_"): float(
                np.mean([float(row[key]) for row in paired])
            )
            for key in delta_keys
        }
    trace_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in traces:
        trace_groups.setdefault(
            (str(row["episode_id"]), str(row["arm"])), []
        ).append(row)
    paired_first_step_equal = []
    pair_invariants_evaluated = set(args.arms) == {"gate_on", "gate_off"}
    all_rollouts_cross_opening = True
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        episode_traces = [
            trace_groups[(episode_id, arm)] for arm in args.arms
        ]
        if pair_invariants_evaluated:
            on_trace = trace_groups[(episode_id, "gate_on")]
            off_trace = trace_groups[(episode_id, "gate_off")]
            paired_first_step_equal.append(
                bool(
                    np.isclose(
                        float(on_trace[0]["lam_soft_learned"]),
                        float(off_trace[0]["lam_soft_learned"]),
                        atol=1e-8,
                    )
                    and np.isclose(
                        float(on_trace[0]["lam_hard_used"]),
                        float(off_trace[0]["lam_hard_used"]),
                        atol=1e-8,
                    )
                )
            )
        all_rollouts_cross_opening = all_rollouts_cross_opening and all(
            int(arm_trace[-1]["step"]) >= int(arm_trace[-1]["opening_step"])
            for arm_trace in episode_traces
        )
    summary["smoke_validation"] = {
        "paired_episode_count": len(paired),
        "pair_invariants_evaluated": pair_invariants_evaluated,
        "paired_first_step_learned_coefficients_equal": (
            bool(all(paired_first_step_equal))
            if pair_invariants_evaluated
            else None
        ),
        "gate_off_multiplier_always_one": bool(
            all(
                np.isclose(float(row["soft_multiplier"]), 1.0)
                for row in traces
                if row["arm"] == "gate_off"
            )
        ),
        "gate_on_multiplier_binary": bool(
            all(
                float(row["soft_multiplier"]) in (0.0, 1.0)
                for row in traces
                if row["arm"] == "gate_on"
            )
        ),
        "learned_hard_force_active_both_arms": bool(
            all(float(row["lam_hard_used"]) > 0.0 for row in traces)
        ),
        "all_rollouts_reach_post_opening_window": bool(all_rollouts_cross_opening),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote isolated Experiment 1 artifacts to {args.out}")


if __name__ == "__main__":
    main()
