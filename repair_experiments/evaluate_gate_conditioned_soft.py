#!/usr/bin/env python3
"""Static R1 three-arm evaluation for the gate-conditioned soft repair.

This development evaluator keeps the canonical hard coefficient and all
geometry/goal coefficients fixed.  It recomputes them every control tick,
uses the sequence-00003-held-out directional evidence head only to predict a
context-conditioned soft magnitude, and normalizes/projects the soft direction
into the feasible primitive cone.  The sealed sequence 00004 is never read.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
FULL_CODE = WORKSPACE / "full_code"
EXP_RELLIS = FULL_CODE / "exp-rellis"
for root in (WORKSPACE, FULL_CODE, EXP_RELLIS):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gate-conditioned-soft")

from grl_rellis import BevConfig  # noqa: E402
from rebuttal_experiments.exp1_gate_ablation import (  # noqa: E402
    _build_goal_feats,
    _build_obs_feats,
    _clip_rc,
    _load_model,
    _nearest_path_index,
    primitive_feasibility_gate,
)
from repair_experiments.v1_controller import align_rollout_soft_force  # noqa: E402
from repair_experiments.v1_controller import (  # noqa: E402
    enumerate_stagewise_primitive_candidates,
    primitive_direction_xy,
    primitive_ray_is_hard_feasible,
)
from scripts.baselines.dfc.metrics import FailureWeights, compute_path_metrics, compute_trace_metrics  # noqa: E402
from scripts.build_dfc2018_stagewise import extract_local_geom_obstacles, extract_risk_patch, extract_rollout_patch  # noqa: E402
from train_material import integrate_surrogate_material  # noqa: E402
from train_rellis_directional_force import (  # noqa: E402
    DIRS_16,
    DirectionalForceHead,
    _build_point,
    _route_context,
)

ARMS = ("zero", "learned", "fixed")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_evidence_head(path: Path, device: str) -> Tuple[DirectionalForceHead, Dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    summary = dict(payload["summary"])
    hidden = int(summary["config"]["hidden"])
    model = DirectionalForceHead(
        in_dim=int(payload["in_dim"]),
        hidden=hidden,
        out_dim=int(np.asarray(payload["dirs"]).shape[0]) + 1,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, summary


def _synthetic_path(
    reference_path: np.ndarray,
    position_xy: np.ndarray,
    previous_index: int,
) -> Tuple[List[Tuple[int, int]], int]:
    current_rc = position_xy[::-1]
    nearest = _nearest_path_index(reference_path, current_rc, previous_index)
    current = (int(round(float(current_rc[0]))), int(round(float(current_rc[1]))))
    tail = [(int(p[0]), int(p[1])) for p in reference_path[nearest + 1 :]]
    if not tail:
        tail = [(int(reference_path[-1, 0]), int(reference_path[-1, 1]))]
    if tail[0] == current and len(tail) > 1:
        tail = tail[1:]
    return [current, *tail], nearest


@torch.no_grad()
def _evidence_prediction(
    evidence_head: DirectionalForceHead,
    maps: Mapping[str, np.ndarray],
    reference_path: np.ndarray,
    position_xy: np.ndarray,
    previous_index: int,
    *,
    episode_id: str,
    route: Mapping[str, np.ndarray],
    device: str,
    horizon_cells: int,
    long_horizon_cells: int,
    hard_margin: float,
    improvement_margin: float,
    activation_threshold: float,
    lambda_max: float,
) -> Tuple[float, float, Tuple[float, float], int, int]:
    path, nearest = _synthetic_path(reference_path, position_xy, previous_index)
    row = _build_point(
        maps,
        path,
        0,
        regime="R1",
        episode_id=episode_id,
        horizon_cells=horizon_cells,
        long_horizon_cells=long_horizon_cells,
        hard_margin_m=hard_margin,
        improvement_margin=improvement_margin,
        route=route,
        route_max_ratio=2.2,
    )
    if row is None:
        return 0.0, 0.0, (0.0, 0.0), 0, nearest
    x = torch.as_tensor(np.asarray(row["x"], dtype=np.float32), device=device).unsqueeze(0)
    probabilities = torch.softmax(evidence_head(x), dim=-1)[0]
    p_active = float(1.0 - probabilities[0].item())
    pred = int(torch.argmax(probabilities).item())
    # Calibrated continuous magnitude: exactly zero below the pre-existing
    # activation threshold and spans [0, lambda_max] above it.
    scaled = max(0.0, p_active - activation_threshold) / max(1.0 - activation_threshold, 1e-6)
    lam = float(lambda_max * min(1.0, scaled))
    direction = (
        (float(DIRS_16[pred - 1, 0]), float(DIRS_16[pred - 1, 1]))
        if pred > 0
        else (0.0, 0.0)
    )
    return lam, p_active, direction, pred, nearest


@torch.no_grad()
def rollout_arm(
    *,
    arm: str,
    base_model: torch.nn.Module,
    evidence_head: DirectionalForceHead,
    evidence_summary: Mapping[str, Any],
    maps: Mapping[str, np.ndarray],
    episode: Mapping[str, Any],
    route: Mapping[str, np.ndarray],
    device: str,
    fixed_lambda: float,
    lambda_max: float,
    max_steps: int,
    dt: float,
    stage_lookahead: int,
    patch_size: int,
    obstacle_patch_size: int,
    robot_radius: float,
    margin_factor: float,
    d_hat_sdf: float,
    primitive_count: int,
    primitive_horizon: int,
    hard_margin: float,
    improvement_margin: float,
    material_trigger: float,
    cone_half_angle: float,
    controller: str,
    commit_distance_m: float,
    commit_steps: int,
    mpc_horizon_steps: int,
    mpc_replan_steps: int,
    mpc_min_improvement: float,
    gsd: float,
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    reference = np.asarray(episode["stage1_path"], dtype=np.float32)
    start_rc = np.asarray(episode["start_rc"], dtype=np.float32)
    goal_rc = np.asarray(episode["goal_rc"], dtype=np.float32)
    position = start_rc[::-1].copy()
    velocity = np.zeros(2, dtype=np.float32)
    goal_xy = goal_rc[::-1].copy()
    path_index = 0
    trace_xy = [position.copy()]
    decisions: List[Dict[str, Any]] = []
    latched_direction: Tuple[float, float] = (0.0, 0.0)
    latched_waypoint = np.zeros(2, dtype=np.float32)
    latch_remaining = 0
    threshold = float(evidence_summary["activation_threshold"])
    config = evidence_summary["config"]

    for step in range(max_steps):
        shape = maps["risk_map"].shape
        if not (0 <= position[0] < shape[1] and 0 <= position[1] < shape[0]):
            break
        nearest = _nearest_path_index(reference, position[::-1], path_index)
        target_index = min(nearest + stage_lookahead, len(reference) - 1)
        stage_goal = reference[target_index][::-1].astype(np.float32)
        learned_lambda, p_active, predicted_direction, predicted_class, path_index = _evidence_prediction(
            evidence_head,
            maps,
            reference,
            position,
            path_index,
            episode_id=str(episode["episode_id"]),
            route=route,
            device=device,
            horizon_cells=int(config["horizon_cells"]),
            long_horizon_cells=int(config["long_horizon_cells"]),
            hard_margin=hard_margin,
            improvement_margin=float(config["improvement_margin"]),
            activation_threshold=threshold,
            lambda_max=lambda_max,
        )
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
        if arm == "zero":
            candidate_lambda = 0.0
        elif arm == "learned":
            candidate_lambda = learned_lambda
        elif arm == "fixed":
            candidate_lambda = fixed_lambda
        else:
            raise ValueError(arm)
        raw_activation = bool(gate.active and candidate_lambda > 1e-8)

        center_rc = _clip_rc(position[::-1], shape)
        np.random.seed(seed + step)
        centers, radii, widths, d_hat = extract_local_geom_obstacles(
            maps["geom_occ"],
            center_rc,
            patch_size=obstacle_patch_size,
            robot_radius=robot_radius,
            margin_factor=margin_factor,
        )
        risk_patch_np, _ = extract_risk_patch(maps, center_rc, patch_size)
        obs_feats = _build_obs_feats(position, stage_goal, centers, radii, widths, device)
        obs_mask = torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)
        goal_feats = _build_goal_feats(position, stage_goal, device)
        risk_patch = torch.as_tensor(risk_patch_np, dtype=torch.float32, device=device).unsqueeze(0)
        alphas, beta, gamma, _, lam_hard, _ = base_model(obs_feats, obs_mask, goal_feats, risk_patch)

        rollout_patch = torch.as_tensor(
            extract_rollout_patch(maps, center_rc, patch_size),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        def shoot(direction_rc: Tuple[float, float]) -> Tuple[float, float, float]:
            direction_xy = primitive_direction_xy(direction_rc)
            candidate_goal = position + (commit_distance_m / gsd) * direction_xy
            candidate_patch, _ = align_rollout_soft_force(
                rollout_patch,
                direction_rc,
                half_angle_degrees=cone_half_angle,
                gradient_confidence_threshold=1e-3,
                low_confidence_fallback_policy="selected_axis",
            )
            endpoint, _, min_clearance, cumulative_risk, hard_count, _ = integrate_surrogate_material(
                o0=torch.as_tensor(position, dtype=torch.float32, device=device).unsqueeze(0),
                v0=torch.as_tensor(velocity, dtype=torch.float32, device=device).unsqueeze(0),
                goal=torch.as_tensor(candidate_goal, dtype=torch.float32, device=device).unsqueeze(0),
                C=torch.as_tensor(centers, dtype=torch.float32, device=device).unsqueeze(0),
                R=torch.as_tensor(radii, dtype=torch.float32, device=device).unsqueeze(0),
                mask=torch.ones(1, centers.shape[0], dtype=torch.bool, device=device),
                alphas=alphas,
                beta=beta,
                gamma=gamma,
                lam_soft=torch.tensor([candidate_lambda], dtype=torch.float32, device=device),
                lam_hard=lam_hard,
                rollout_patch=candidate_patch,
                d_hat=torch.tensor([d_hat], dtype=torch.float32, device=device),
                dt=torch.tensor([dt], dtype=torch.float32, device=device),
                H=torch.tensor([mpc_horizon_steps], dtype=torch.long, device=device),
                robot_radius=torch.tensor([robot_radius], dtype=torch.float32, device=device),
                margin_factor=margin_factor,
                d_hat_sdf=d_hat_sdf,
            )
            progress = float(np.linalg.norm(position - stage_goal) - np.linalg.norm(endpoint[0].cpu().numpy() - stage_goal))
            return float(cumulative_risk.item()) / max(1, mpc_horizon_steps), float(min_clearance.item()), progress - 100.0 * float(hard_count.item())

        selected_predicted_risk = float("nan")
        nominal_predicted_risk = float("nan")
        mpc_candidate_count = 0
        if latch_remaining > 0:
            still_safe = primitive_ray_is_hard_feasible(
                maps,
                position,
                latched_direction,
                horizon_cells=max(1, int(math.ceil(commit_distance_m / gsd))),
                hard_margin_m=hard_margin,
            )
            reached = float(np.linalg.norm(position - latched_waypoint)) <= 0.5
            if not still_safe or reached:
                latch_remaining = 0

        if controller == "commit" and latch_remaining <= 0 and raw_activation:
            latched_direction = gate.selected_direction_rc
            latched_waypoint = position + (commit_distance_m / gsd) * primitive_direction_xy(latched_direction)
            latch_remaining = commit_steps
        elif controller == "mpc" and raw_activation and latch_remaining <= 0:
            enumeration = enumerate_stagewise_primitive_candidates(
                maps,
                position,
                stage_goal,
                primitive_count=primitive_count,
                horizon_cells=primitive_horizon,
                hard_margin_m=hard_margin,
                require_endpoint_progress=True,
            )
            nominal_predicted_risk, _, _ = shoot(enumeration.nominal_direction_rc)
            scored = []
            for candidate in enumeration.candidates:
                predicted_risk, predicted_clearance, predicted_progress = shoot(candidate.direction_rc)
                mpc_candidate_count += 1
                if predicted_clearance >= hard_margin and predicted_progress > 0.0:
                    scored.append((predicted_risk, -predicted_progress, -predicted_clearance, candidate.primitive_index, candidate.direction_rc))
            if scored:
                scored.sort()
                best = scored[0]
                selected_predicted_risk = float(best[0])
                if nominal_predicted_risk - selected_predicted_risk >= mpc_min_improvement:
                    latched_direction = best[-1]
                    latched_waypoint = position + (commit_distance_m / gsd) * primitive_direction_xy(latched_direction)
                    latch_remaining = mpc_replan_steps

        if controller == "per_step":
            effective_active = raw_activation
            direction = gate.selected_direction_rc
            goal_used = stage_goal
        else:
            effective_active = latch_remaining > 0
            direction = latched_direction if effective_active else (0.0, 0.0)
            goal_used = latched_waypoint if effective_active else stage_goal
        used_lambda = candidate_lambda * float(effective_active)
        if effective_active and np.linalg.norm(direction) > 1e-8:
            rollout_patch, _ = align_rollout_soft_force(
                rollout_patch,
                direction,
                half_angle_degrees=cone_half_angle,
                gradient_confidence_threshold=1e-3,
                low_confidence_fallback_policy="selected_axis",
            )
        centers_t = torch.as_tensor(centers, dtype=torch.float32, device=device).unsqueeze(0)
        radii_t = torch.as_tensor(radii, dtype=torch.float32, device=device).unsqueeze(0)
        next_position_t, next_velocity_t, *_ = integrate_surrogate_material(
            o0=torch.as_tensor(position, dtype=torch.float32, device=device).unsqueeze(0),
            v0=torch.as_tensor(velocity, dtype=torch.float32, device=device).unsqueeze(0),
            goal=torch.as_tensor(goal_used, dtype=torch.float32, device=device).unsqueeze(0),
            C=centers_t,
            R=radii_t,
            mask=torch.ones(1, centers.shape[0], dtype=torch.bool, device=device),
            alphas=alphas,
            beta=beta,
            gamma=gamma,
            lam_soft=torch.tensor([used_lambda], dtype=torch.float32, device=device),
            lam_hard=lam_hard,
            rollout_patch=rollout_patch,
            d_hat=torch.tensor([d_hat], dtype=torch.float32, device=device),
            dt=torch.tensor([dt], dtype=torch.float32, device=device),
            H=torch.ones(1, dtype=torch.long, device=device),
            robot_radius=torch.tensor([robot_radius], dtype=torch.float32, device=device),
            margin_factor=margin_factor,
            d_hat_sdf=d_hat_sdf,
        )
        next_position = next_position_t[0].cpu().numpy()
        next_velocity = next_velocity_t[0].cpu().numpy()
        if not np.all(np.isfinite(next_position)):
            break
        decisions.append(
            {
                "step": step,
                "arm": arm,
                "controller": controller,
                "gate_active": int(gate.active),
                "effective_active": int(effective_active),
                "latch_remaining": latch_remaining,
                "mpc_candidate_count": mpc_candidate_count,
                "mpc_nominal_predicted_risk": nominal_predicted_risk,
                "mpc_selected_predicted_risk": selected_predicted_risk,
                "p_active": p_active,
                "predicted_class": predicted_class,
                "predicted_direction_row": predicted_direction[0],
                "predicted_direction_col": predicted_direction[1],
                "lambda_soft_candidate": candidate_lambda,
                "lambda_soft_used": used_lambda,
                "lambda_hard": float(lam_hard.item()),
            }
        )
        trace_xy.append(next_position.copy())
        position, velocity = next_position, next_velocity
        if latch_remaining > 0:
            latch_remaining -= 1
        if float(np.linalg.norm(position - goal_xy)) < 6.0:
            break
    trace_xy = np.asarray(trace_xy, dtype=np.float32)
    return np.stack([trace_xy[:, 1], trace_xy[:, 0]], axis=-1), decisions


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        if not subset:
            continue
        output[arm] = {}
        for metric in ("success", "risk_exposure", "mean_rho", "path_length_ratio", "hard_hits"):
            values = np.asarray([float(row[metric]) for row in subset], dtype=np.float64)
            output[arm][metric] = {
                "mean": float(values.mean()),
                "ci95": float(1.96 * values.std(ddof=0) / math.sqrt(len(values))),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/data/adityas/GRL-SNAM"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("/mnt/data/adityas/GRL-SNAM/exp-rellis/checkpoints/rellis_stage2_decision_mid_ep12/best.pt"))
    parser.add_argument("--evidence-checkpoint", type=Path, default=Path("/mnt/data/adityas/GRL-SNAM/exp-rellis/runs/rellis_directional_routeaware_aw050_far020_00003/best.pt"))
    parser.add_argument("--bev-root", type=Path, default=Path("/mnt/data/adityas/GRL-SNAM/exp-rellis/cache/rellis_bev_all_seqbalanced_2500"))
    parser.add_argument("--pairs-root", type=Path, default=Path("/mnt/data/adityas/GRL-SNAM/exp-rellis/cache/rellis_pairs_all_seqbalanced_2500_seq00003"))
    parser.add_argument("--out", type=Path, default=HERE / "outputs/gate_conditioned_soft_smoke")
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--fixed-lambda", type=float, default=1.5)
    parser.add_argument("--lambda-max", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--stage-lookahead", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--obstacle-patch-size", type=int, default=64)
    parser.add_argument("--robot-radius", type=float, default=1.5)
    parser.add_argument("--margin-factor", type=float, default=0.5)
    parser.add_argument("--d-hat-sdf", type=float, default=3.0)
    parser.add_argument("--primitive-count", type=int, default=16)
    parser.add_argument("--primitive-horizon", type=int, default=8)
    parser.add_argument("--hard-margin", type=float, default=1.0)
    parser.add_argument("--improvement-margin", type=float, default=0.0125)
    parser.add_argument("--material-trigger", type=float, default=0.0)
    parser.add_argument("--cone-half-angle", type=float, default=35.0)
    parser.add_argument("--controller", choices=["per_step", "commit", "mpc"], default="per_step")
    parser.add_argument("--commit-distance-m", type=float, default=3.0)
    parser.add_argument("--commit-steps", type=int, default=25)
    parser.add_argument("--mpc-horizon-steps", type=int, default=12)
    parser.add_argument("--mpc-replan-steps", type=int, default=3)
    parser.add_argument("--mpc-min-improvement", type=float, default=0.001)
    parser.add_argument("--route-risk-weight", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=27370)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    base_model, base_cfg = _load_model(args.base_checkpoint, args.source_root, args.device)
    evidence_head, evidence_summary = _load_evidence_head(args.evidence_checkpoint, args.device)
    bev_manifest = json.loads((args.bev_root / "manifest.json").read_text())
    pair_manifest = json.loads((args.pairs_root / "manifest.json").read_text())
    gsd = float(BevConfig(**bev_manifest["config"]["bev"]).resolution)
    episodes = [ep for ep in pair_manifest["episodes"] if ep["regime"] == "R1"][: args.max_episodes]
    weights = FailureWeights()
    metrics: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    scene_cache: Dict[str, Any] = {}
    for number, episode in enumerate(episodes):
        scene_path = str(episode["scene_path"])
        if scene_path not in scene_cache:
            scene_cache[scene_path] = torch.load(args.bev_root / scene_path, map_location="cpu", weights_only=False)
        maps = scene_cache[scene_path]["maps"]
        goal = tuple(int(x) for x in episode["goal_rc"])
        route = _route_context(maps, goal, risk_weight=args.route_risk_weight)
        reference_path = [(int(p[0]), int(p[1])) for p in episode["stage1_path"]]
        reference = compute_path_metrics(reference_path, maps, reference_length_m=None, gsd=gsd, weights=weights, goal_rc=goal)
        for arm in args.arms:
            trace, arm_decisions = rollout_arm(
                arm=arm,
                base_model=base_model,
                evidence_head=evidence_head,
                evidence_summary=evidence_summary,
                maps=maps,
                episode=episode,
                route=route,
                device=args.device,
                fixed_lambda=args.fixed_lambda,
                lambda_max=args.lambda_max,
                max_steps=args.max_steps,
                dt=args.dt,
                stage_lookahead=args.stage_lookahead,
                patch_size=args.patch_size,
                obstacle_patch_size=args.obstacle_patch_size,
                robot_radius=args.robot_radius,
                margin_factor=args.margin_factor,
                d_hat_sdf=args.d_hat_sdf,
                primitive_count=args.primitive_count,
                primitive_horizon=args.primitive_horizon,
                hard_margin=args.hard_margin,
                improvement_margin=args.improvement_margin,
                material_trigger=args.material_trigger,
                cone_half_angle=args.cone_half_angle,
                controller=args.controller,
                commit_distance_m=args.commit_distance_m,
                commit_steps=args.commit_steps,
                mpc_horizon_steps=args.mpc_horizon_steps,
                mpc_replan_steps=args.mpc_replan_steps,
                mpc_min_improvement=args.mpc_min_improvement,
                gsd=gsd,
                seed=args.seed + 1000 * number,
            )
            result = compute_trace_metrics(trace, maps, reference_length_m=float(reference["path_length_m"]), gsd=gsd, weights=weights, goal_rc=goal)
            metrics.append({"episode_id": str(episode["episode_id"]), "arm": arm, **{key: value for key, value in result.items() if not isinstance(value, dict)}})
            decisions.extend({"episode_id": str(episode["episode_id"]), **row} for row in arm_decisions)
        print(f"completed {number + 1}/{len(episodes)}", flush=True)
    summary = {
        "status": "development_validation_only",
        "method": f"{args.controller} gate-evidence-conditioned magnitude with normalized/projected soft direction",
        "metrics": _summarize(metrics),
        "counts": {"episodes": len(episodes), "rollouts": len(metrics), "decisions": len(decisions)},
        "provenance": {
            "base_checkpoint": str(args.base_checkpoint),
            "evidence_checkpoint": str(args.evidence_checkpoint),
            "evidence_holdout_sequence": evidence_summary["config"].get("holdout_sequence"),
            "evaluated_sequence": "00003",
            "sealed_sequence_00004_loaded": False,
            "base_checkpoint_cfg": base_cfg,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    }
    _write_csv(args.out / "per_episode_metrics.csv", metrics)
    _write_csv(args.out / "per_step_decisions.csv", decisions)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
