#!/usr/bin/env python3
"""Run the v1 repaired learned controller on an unsealed RELLIS split.

This is intentionally separate from ``rebuttal_experiments`` and never calls
the historical route-following heuristic.  Every arm executes the same
``CoefEnergyNetMaterial`` checkpoint through
``integrate_surrogate_material``.  Arms differ only in gating/projection.

Development is restricted to the preregistered train and validation manifests.
The sealed test manifest and sequence 00004 cannot be selected by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rebuttal_experiments.exp1_gate_ablation import (
    DEFAULT_BEV_ROOT,
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_ROOT,
    _build_goal_feats,
    _build_obs_feats,
    _clip_rc,
    _load_model,
    _sha256,
    _stage_goal_xy,
    _weighted_upper_tail,
    primitive_feasibility_gate,
)
from repair_experiments.v1_controller import (
    CONTROLLER_VERSION,
    DIRECT_WAYPOINT_CONTROLLER_VERSION,
    DirectWaypointLatch,
    HysteresisConfig,
    ProjectionDiagnostics,
    StagewisePrimitiveCandidate,
    StatefulFeasibilityGate,
    TemporalObstacleReleaseGate,
    TemporalReleaseConfig,
    VelocityAwareSelectorConfig,
    WaypointLatchConfig,
    align_rollout_soft_force,
    direct_activation_admitted,
    enumerate_stagewise_primitive_candidates,
    primitive_ray_is_hard_feasible,
    receding_candidate_pool,
    velocity_tracking_direction_rc,
)
from repair_experiments.velocity_selector import (
    preserved_numpy_rng,
    rank_velocity_candidates,
    require_nominal_forecast_improvement,
    simulate_velocity_candidate,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-repair-v1")

ALLOWED_SPLITS = ("train", "validation")
MODES = (
    "direct_waypoint",
    "repaired",
    "stateful_unprojected",
    "stateless_projected",
    "stateless_unprojected",
    "gate_off",
    "geometry_only",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe_record(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_record(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(
        float(value)
    ):
        return None
    return value


def _null_projection(direction_rc: Tuple[float, float]) -> ProjectionDiagnostics:
    return ProjectionDiagnostics(
        selected_direction_x=float(direction_rc[1]),
        selected_direction_y=float(direction_rc[0]),
        center_soft_x_before=float("nan"),
        center_soft_y_before=float("nan"),
        center_soft_x_after=float("nan"),
        center_soft_y_after=float("nan"),
        center_alignment_before=float("nan"),
        center_alignment_after=float("nan"),
        projected_pixel_fraction=0.0,
        center_gradient_norm=float("nan"),
        center_low_confidence_fallback=False,
        low_confidence_pixel_fraction=0.0,
        gradient_confidence_threshold=float("nan"),
        low_confidence_fallback_policy="not_applied",
    )


@torch.no_grad()
def rollout(
    *,
    mode: str,
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
    hard_override_clearance_m: float,
    cone_half_angle_degrees: float,
    gradient_confidence_threshold: float,
    low_confidence_fallback_policy: str,
    lambda_active_threshold: float,
    gate_config: HysteresisConfig,
    hard_violation_penalty: float,
    seed: int,
    waypoint_config: WaypointLatchConfig = WaypointLatchConfig(),
    temporal_config: TemporalReleaseConfig = TemporalReleaseConfig(),
    selector_config: VelocityAwareSelectorConfig = (
        VelocityAwareSelectorConfig()
    ),
    direct_lambda_floor: float = 2.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "exp-rellis"))
    from grl_rellis.dyn_events import apply_dynamic_event
    from scripts.build_dfc2018_stagewise import (
        extract_local_geom_obstacles,
        extract_risk_patch,
        extract_rollout_patch,
    )
    from train_material import integrate_surrogate_material

    is_direct_waypoint = mode == "direct_waypoint"
    is_stateful = mode in {"repaired", "stateful_unprojected"}
    uses_projection = mode in {
        "direct_waypoint", "repaired", "stateless_projected"
    }
    stateful_gate = StatefulFeasibilityGate(gate_config)
    waypoint_latch = DirectWaypointLatch(waypoint_config, gsd=gsd)
    temporal_release_gate = TemporalObstacleReleaseGate(temporal_config)
    latched_direction_rc = (0.0, 0.0)

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
    weights: List[float] = []
    path_length_m = 0.0
    hard_contacts = 0
    start_time = time.perf_counter()

    for step in range(max_steps):
        maps = apply_dynamic_event(base_maps, spec, step, resolution=gsd)
        shape = maps["risk_map"].shape
        if not (0 <= position[0] < shape[1] and 0 <= position[1] < shape[0]):
            break
        stage_goal, path_index = _stage_goal_xy(
            stage1_path, position, path_index, stage_lookahead_cells
        )
        raw_gate = primitive_feasibility_gate(
            maps,
            position,
            stage_goal,
            primitive_count=primitive_count,
            horizon_cells=primitive_horizon_cells,
            hard_margin_m=hard_margin_m,
            improvement_margin=gate_config.on_improvement,
            material_trigger=gate_config.on_material_trigger,
        )
        if not is_direct_waypoint and raw_gate.feasible_count > 0 and np.linalg.norm(
            raw_gate.selected_direction_rc
        ) > 1e-8:
            latched_direction_rc = raw_gate.selected_direction_rc

        center_rc = _clip_rc(position[::-1], shape)
        current_sdf = float(maps["sdf_hard"][center_rc])
        current_hard = bool(maps["hard_mask"][center_rc])
        hard_override = bool(
            current_hard or current_sdf < hard_override_clearance_m
        )
        direct_state_update = None
        temporal_update = None
        if is_direct_waypoint:
            # Direct mode uses feasibility-only persistence/hysteresis.  The
            # learned coefficient remains a continuous force magnitude and is
            # never a second binary activation signal.
            direct_state_update = stateful_gate.update(
                nominal_risk=raw_gate.nominal_risk,
                best_risk=raw_gate.best_risk,
                feasible_count=raw_gate.feasible_count,
                hard_hazard_override=hard_override,
                magnitude_eligible=True,
            )
            temporal_update = temporal_release_gate.update(
                raw_activation_requested=bool(raw_gate.active),
                current_hard_mask=maps["hard_mask"],
                static_hard_mask=base_maps["hard_mask"],
                hard_hazard_override=hard_override,
            )
        waypoint_horizon_cells = max(
            1, int(math.ceil(waypoint_config.distance_m / gsd))
        )
        latched_direction_valid = bool(
            not waypoint_latch.active
            or primitive_ray_is_hard_feasible(
                maps,
                position,
                waypoint_latch.direction_rc,
                horizon_cells=waypoint_horizon_cells,
                hard_margin_m=hard_margin_m,
            )
        )
        selector_invoked = False
        selector_enumerated_count = 0
        selector_improvement_eligible_count = 0
        selector_diagnostics = None
        selector_nominal_record = None
        selector_no_safe_fallback = False
        selector_direction_rc = raw_gate.selected_direction_rc
        direct_admission = False
        if is_direct_waypoint:
            direct_admission = direct_activation_admitted(
                stateful_feasibility_active=direct_state_update.active,
                temporal_update=temporal_update,
                hard_hazard_override=hard_override,
            )
            should_select = bool(direct_admission)
            if should_select:
                selector_invoked = True
                enumeration = enumerate_stagewise_primitive_candidates(
                    maps,
                    position,
                    stage_goal,
                    primitive_count=primitive_count,
                    horizon_cells=primitive_horizon_cells,
                    hard_margin_m=hard_margin_m,
                    require_endpoint_progress=False,
                )
                selector_enumerated_count = len(enumeration.candidates)
                eligible_candidates = list(
                    receding_candidate_pool(enumeration)
                )
                selector_improvement_eligible_count = len(
                    enumeration.candidates
                )
                nominal_candidate = StagewisePrimitiveCandidate(
                    primitive_index=-1,
                    direction_rc=enumeration.nominal_direction_rc,
                    endpoint_rc=enumeration.nominal_endpoint_rc,
                    mean_risk=enumeration.nominal_mean_risk,
                    min_clearance_m=float("nan"),
                )
                forecast_records = []
                for candidate in [nominal_candidate, *eligible_candidates]:

                    def forecast_stage_step(
                        sim_position: np.ndarray,
                        sim_velocity: np.ndarray,
                        rolling_goal: np.ndarray,
                        direction_rc: Tuple[float, float],
                        stage_index: int,
                    ) -> Tuple[np.ndarray, np.ndarray]:
                        sim_center_rc = _clip_rc(
                            sim_position[::-1], shape
                        )
                        # The seed depends on real time and forecast stage,
                        # not candidate identity, so nominal and a candidate
                        # with the same state/direction receive identical
                        # extraction behavior.
                        forecast_seed = seed + 100003 * step + stage_index
                        with preserved_numpy_rng(forecast_seed):
                            (
                                sim_centers,
                                sim_radii,
                                sim_widths,
                                sim_d_hat,
                            ) = extract_local_geom_obstacles(
                                maps["geom_occ"],
                                sim_center_rc,
                                patch_size=obstacle_patch_size,
                                robot_radius=robot_radius,
                                margin_factor=margin_factor,
                            )
                        sim_risk_patch_np, _ = extract_risk_patch(
                            maps, sim_center_rc, patch_size
                        )
                        sim_obs_feats = _build_obs_feats(
                            sim_position,
                            rolling_goal,
                            sim_centers,
                            sim_radii,
                            sim_widths,
                            device,
                        )
                        sim_obs_mask = torch.ones(
                            1,
                            sim_obs_feats.shape[1],
                            dtype=torch.bool,
                            device=device,
                        )
                        sim_goal_feats = _build_goal_feats(
                            sim_position, rolling_goal, device
                        )
                        sim_risk_patch = torch.as_tensor(
                            sim_risk_patch_np,
                            dtype=torch.float32,
                            device=device,
                        ).unsqueeze(0)
                        (
                            sim_alphas,
                            sim_beta,
                            sim_gamma,
                            sim_lam_soft,
                            sim_lam_hard,
                            _,
                        ) = model(
                            sim_obs_feats,
                            sim_obs_mask,
                            sim_goal_feats,
                            sim_risk_patch,
                        )
                        sim_rollout_patch = torch.as_tensor(
                            extract_rollout_patch(
                                maps, sim_center_rc, patch_size
                            ),
                            dtype=torch.float32,
                            device=device,
                        ).unsqueeze(0)
                        sim_tracking_direction_rc = (
                            velocity_tracking_direction_rc(
                                sim_velocity,
                                direction_rc,
                            )
                        )
                        sim_rollout_patch, _ = align_rollout_soft_force(
                            sim_rollout_patch,
                            sim_tracking_direction_rc,
                            half_angle_degrees=cone_half_angle_degrees,
                            gradient_confidence_threshold=(
                                gradient_confidence_threshold
                            ),
                            low_confidence_fallback_policy=(
                                low_confidence_fallback_policy
                            ),
                        )
                        sim_centers_t = torch.as_tensor(
                            sim_centers,
                            dtype=torch.float32,
                            device=device,
                        ).unsqueeze(0)
                        sim_radii_t = torch.as_tensor(
                            sim_radii,
                            dtype=torch.float32,
                            device=device,
                        ).unsqueeze(0)
                        sim_next_position, sim_next_velocity, *_ = (
                            integrate_surrogate_material(
                                o0=torch.as_tensor(
                                    sim_position,
                                    dtype=torch.float32,
                                    device=device,
                                ).unsqueeze(0),
                                v0=torch.as_tensor(
                                    sim_velocity,
                                    dtype=torch.float32,
                                    device=device,
                                ).unsqueeze(0),
                                goal=torch.as_tensor(
                                    rolling_goal,
                                    dtype=torch.float32,
                                    device=device,
                                ).unsqueeze(0),
                                C=sim_centers_t,
                                R=sim_radii_t,
                                mask=torch.ones(
                                    1,
                                    sim_centers.shape[0],
                                    dtype=torch.bool,
                                    device=device,
                                ),
                                alphas=sim_alphas,
                                beta=sim_beta,
                                gamma=sim_gamma,
                                lam_soft=torch.clamp(
                                    sim_lam_soft,
                                    min=direct_lambda_floor,
                                ),
                                lam_hard=sim_lam_hard,
                                rollout_patch=sim_rollout_patch,
                                d_hat=torch.tensor(
                                    [sim_d_hat],
                                    dtype=torch.float32,
                                    device=device,
                                ),
                                dt=torch.tensor(
                                    [dt],
                                    dtype=torch.float32,
                                    device=device,
                                ),
                                H=torch.ones(
                                    1, dtype=torch.long, device=device
                                ),
                                robot_radius=torch.tensor(
                                    [robot_radius],
                                    dtype=torch.float32,
                                    device=device,
                                ),
                                margin_factor=margin_factor,
                                d_hat_sdf=d_hat_sdf,
                            )
                        )
                        return (
                            sim_next_position[0].cpu().numpy(),
                            sim_next_velocity[0].cpu().numpy(),
                        )

                    forecast_record = simulate_velocity_candidate(
                            candidate=candidate,
                            initial_position_xy=position,
                            initial_velocity_xy=velocity,
                            stage_goal_xy=stage_goal,
                            maps=maps,
                            gsd=gsd,
                            rolling_distance_m=waypoint_config.distance_m,
                            config=selector_config,
                            stage_step=forecast_stage_step,
                            enforce_safety=(
                                candidate.primitive_index != -1
                            ),
                        )
                    if candidate.primitive_index == -1:
                        selector_nominal_record = forecast_record
                    else:
                        forecast_records.append(forecast_record)
                if selector_nominal_record is None:
                    raise RuntimeError(
                        "nominal forecast was not produced"
                    )
                forecast_records = list(
                    require_nominal_forecast_improvement(
                        forecast_records,
                        nominal_path_weighted_mean_risk=(
                            selector_nominal_record.path_weighted_mean_risk
                        ),
                        improvement_margin=gate_config.on_improvement,
                    )
                )
                selector_diagnostics = rank_velocity_candidates(
                    forecast_records
                )
                if selector_diagnostics.selected_primitive_index is None:
                    selector_no_safe_fallback = True
                    direct_admission = False
                    selector_direction_rc = (0.0, 0.0)
                else:
                    selector_direction_rc = (
                        selector_diagnostics.selected_direction_rc
                    )
            waypoint_update = waypoint_latch.update_receding(
                position_xy=position,
                gate_active=direct_admission,
                selected_direction_rc=selector_direction_rc,
                selection_attempted=selector_invoked,
                latched_direction_valid=latched_direction_valid,
                hard_hazard_override=hard_override,
            )
            latched_direction_rc = waypoint_update.direction_rc
            goal_used = (
                np.asarray(waypoint_update.waypoint_xy, dtype=np.float32)
                if waypoint_update.active
                else stage_goal
            )
        else:
            waypoint_update = None
            goal_used = stage_goal
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
            position, goal_used, centers, radii, widths, device
        )
        obs_mask = torch.ones(
            1, obs_feats.shape[1], dtype=torch.bool, device=device
        )
        goal_feats = _build_goal_feats(position, goal_used, device)
        risk_patch = torch.as_tensor(
            risk_patch_np, dtype=torch.float32, device=device
        ).unsqueeze(0)
        alphas, beta, gamma, lam_soft, lam_hard, _ = model(
            obs_feats, obs_mask, goal_feats, risk_patch
        )
        learned_magnitude_active = bool(
            float(lam_soft.item()) >= lambda_active_threshold
        )
        # Magnitude eligibility is part of stateful evidence, rather than a
        # second frame-wise switch after hysteresis.  Otherwise a coefficient
        # oscillating around its threshold reintroduces the flicker that the
        # state machine is intended to remove.
        state_update = (
            direct_state_update
            if is_direct_waypoint
            else stateful_gate.update(
                nominal_risk=raw_gate.nominal_risk,
                best_risk=raw_gate.best_risk,
                feasible_count=raw_gate.feasible_count,
                hard_hazard_override=hard_override,
                magnitude_eligible=learned_magnitude_active,
            )
        )
        if is_direct_waypoint:
            feasibility_active = bool(state_update.active)
            effective_active = bool(waypoint_update.active)
            gate_reason = waypoint_update.reason
        elif mode == "gate_off":
            feasibility_active = True
            effective_active = learned_magnitude_active
            gate_reason = (
                "gate_off_control"
                if learned_magnitude_active
                else "gate_off_control|learned_magnitude_below_threshold"
            )
        elif mode == "geometry_only":
            feasibility_active = False
            effective_active = False
            gate_reason = "geometry_only_control"
        elif is_stateful:
            feasibility_active = state_update.active
            effective_active = state_update.active
            gate_reason = state_update.reason
        else:
            feasibility_active = bool(raw_gate.active)
            effective_active = bool(
                feasibility_active and learned_magnitude_active
            )
            gate_reason = "stateless_raw_gate"
            if feasibility_active and not learned_magnitude_active:
                gate_reason += "|learned_magnitude_below_threshold"
        soft_multiplier = 1.0 if effective_active else 0.0
        if is_direct_waypoint and effective_active:
            lam_soft_used = torch.clamp(lam_soft, min=direct_lambda_floor)
        else:
            lam_soft_used = lam_soft * soft_multiplier

        rollout_patch_np = extract_rollout_patch(maps, center_rc, patch_size)
        rollout_patch = torch.as_tensor(
            rollout_patch_np, dtype=torch.float32, device=device
        ).unsqueeze(0)
        projection = _null_projection(latched_direction_rc)
        projection_applied = bool(
            uses_projection
            and effective_active
            and np.linalg.norm(latched_direction_rc) > 1e-8
        )
        force_direction_rc = latched_direction_rc
        if is_direct_waypoint and projection_applied:
            force_direction_rc = velocity_tracking_direction_rc(
                velocity,
                latched_direction_rc,
            )
        if projection_applied:
            rollout_patch, projection = align_rollout_soft_force(
                rollout_patch,
                force_direction_rc,
                half_angle_degrees=cone_half_angle_degrees,
                gradient_confidence_threshold=gradient_confidence_threshold,
                low_confidence_fallback_policy=low_confidence_fallback_policy,
            )

        centers_t = torch.as_tensor(
            centers, dtype=torch.float32, device=device
        ).unsqueeze(0)
        radii_t = torch.as_tensor(
            radii, dtype=torch.float32, device=device
        ).unsqueeze(0)
        next_position_t, next_velocity_t, _, _, _, _ = (
            integrate_surrogate_material(
                o0=torch.as_tensor(
                    position, dtype=torch.float32, device=device
                ).unsqueeze(0),
                v0=torch.as_tensor(
                    velocity, dtype=torch.float32, device=device
                ).unsqueeze(0),
                goal=torch.as_tensor(
                    goal_used, dtype=torch.float32, device=device
                ).unsqueeze(0),
                C=centers_t,
                R=radii_t,
                mask=torch.ones(
                    1, centers.shape[0], dtype=torch.bool, device=device
                ),
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
        if not np.all(np.isfinite(next_position)):
            break

        movement_m = gsd * float(np.linalg.norm(next_position - position))
        sample_rc = _clip_rc(next_position[::-1], shape)
        risk_value = float(maps["risk_map"][sample_rc])
        hard_contact = bool(maps["hard_mask"][sample_rc])
        violation = risk_value + hard_violation_penalty * float(hard_contact)
        path_length_m += movement_m
        hard_contacts += int(hard_contact)
        risks.append(risk_value)
        violations.append(violation)
        weights.append(max(movement_m, 1e-9))
        displacement = next_position - position
        displacement_norm = float(np.linalg.norm(displacement))
        selected_xy = np.asarray(
            [latched_direction_rc[1], latched_direction_rc[0]], dtype=np.float32
        )
        displacement_alignment = (
            float(np.dot(displacement, selected_xy) / displacement_norm)
            if displacement_norm > 1e-8
            and float(np.linalg.norm(selected_xy)) > 1e-8
            else float("nan")
        )
        selector_rejection_counts = (
            dict(selector_diagnostics.rejection_counts)
            if selector_diagnostics is not None
            else {}
        )
        selector_selected_record = (
            next(
                (
                    record
                    for record in selector_diagnostics.candidate_diagnostics
                    if record.primitive_index
                    == selector_diagnostics.selected_primitive_index
                ),
                None,
            )
            if selector_diagnostics is not None
            else None
        )
        selector_candidate_records_json = (
            json.dumps(
                [
                    _json_safe_record(asdict(record))
                    for record in selector_diagnostics.candidate_diagnostics
                ],
                sort_keys=True,
                allow_nan=False,
            )
            if selector_diagnostics is not None
            else "[]"
        )
        selector_nominal_record_json = (
            json.dumps(
                _json_safe_record(asdict(selector_nominal_record)),
                sort_keys=True,
                allow_nan=False,
            )
            if selector_nominal_record is not None
            else "{}"
        )
        selector_dynamic_improvement = (
            selector_nominal_record.path_weighted_mean_risk
            - selector_selected_record.path_weighted_mean_risk
            if selector_nominal_record is not None
            and selector_selected_record is not None
            else float("nan")
        )
        trace.append(
            {
                "controller_version": (
                    DIRECT_WAYPOINT_CONTROLLER_VERSION
                    if is_direct_waypoint
                    else CONTROLLER_VERSION
                ),
                "episode_uid": str(episode["episode_uid"]),
                "scene_id": str(episode["scene_id"]),
                "sequence": str(episode["sequence"]),
                "regime": str(episode["regime"]),
                "event_type": str(spec.event_type),
                "mode": mode,
                "step": step,
                "event_step": int(spec.event_step),
                "opening_step": int(spec.event_step + spec.open_delay),
                "position_x": float(position[0]),
                "position_y": float(position[1]),
                "next_x": float(next_position[0]),
                "next_y": float(next_position[1]),
                "stage_goal_x": float(stage_goal[0]),
                "stage_goal_y": float(stage_goal[1]),
                "goal_used_x": float(goal_used[0]),
                "goal_used_y": float(goal_used[1]),
                "goal_is_waypoint": int(
                    is_direct_waypoint and waypoint_update.active
                ),
                "waypoint_latch_active": int(
                    waypoint_update.active if waypoint_update is not None else 0
                ),
                "waypoint_latch_transition": int(
                    waypoint_update.transitioned
                    if waypoint_update is not None
                    else 0
                ),
                "waypoint_latch_reason": (
                    waypoint_update.reason
                    if waypoint_update is not None
                    else "not_direct_waypoint"
                ),
                "waypoint_age_steps": (
                    waypoint_update.age_steps
                    if waypoint_update is not None
                    else 0
                ),
                "waypoint_hold_steps": (
                    waypoint_update.hold_steps
                    if waypoint_update is not None
                    else 0
                ),
                "waypoint_replan_due": int(
                    waypoint_update.replan_due
                    if waypoint_update is not None
                    else False
                ),
                "waypoint_forward_progress_delta_m": (
                    waypoint_update.forward_progress_delta_m
                    if waypoint_update is not None
                    else 0.0
                ),
                "waypoint_cumulative_forward_progress_m": (
                    waypoint_update.cumulative_forward_progress_m
                    if waypoint_update is not None
                    else 0.0
                ),
                "waypoint_active_steps": (
                    waypoint_update.active_steps
                    if waypoint_update is not None
                    else 0
                ),
                "waypoint_active_step_limit": (
                    waypoint_update.active_step_limit
                    if waypoint_update is not None
                    else waypoint_config.active_step_limit
                ),
                "waypoint_cumulative_forward_limit_m": (
                    waypoint_update.cumulative_forward_limit_m
                    if waypoint_update is not None
                    else waypoint_config.cumulative_forward_limit_m
                ),
                "waypoint_rearm_inactive_streak": (
                    waypoint_update.rearm_inactive_streak
                    if waypoint_update is not None
                    else 0
                ),
                "waypoint_rearm_inactive_required": (
                    waypoint_update.rearm_inactive_required
                    if waypoint_update is not None
                    else waypoint_config.rearm_inactive_steps
                ),
                "waypoint_armed": int(
                    waypoint_update.armed
                    if waypoint_update is not None
                    else False
                ),
                "waypoint_rearmed": int(
                    waypoint_update.rearmed
                    if waypoint_update is not None
                    else False
                ),
                "waypoint_activation_block_reason": (
                    waypoint_update.activation_block_reason
                    if waypoint_update is not None
                    else ""
                ),
                "temporal_release_state": (
                    temporal_update.state
                    if temporal_update is not None
                    else "NOT_DIRECT"
                ),
                "temporal_release_reason": (
                    temporal_update.reason
                    if temporal_update is not None
                    else "not_direct_waypoint"
                ),
                "temporal_suppress_activation": int(
                    temporal_update.suppress_activation
                    if temporal_update is not None
                    else False
                ),
                "temporal_released_snapshot_cell_count": (
                    temporal_update.released_snapshot_cell_count
                    if temporal_update is not None
                    else 0
                ),
                "temporal_wait_age_steps": (
                    temporal_update.wait_age_steps
                    if temporal_update is not None
                    else 0
                ),
                "temporal_release_credit_remaining": (
                    temporal_update.release_credit_remaining
                    if temporal_update is not None
                    else 0
                ),
                "temporal_immediate_activation_pulse": int(
                    temporal_update.immediate_activation_pulse
                    if temporal_update is not None
                    else False
                ),
                "temporal_added_hard_count": (
                    temporal_update.added_hard_count
                    if temporal_update is not None
                    else 0
                ),
                "temporal_snapshotted_hard_count": (
                    temporal_update.snapshotted_hard_count
                    if temporal_update is not None
                    else 0
                ),
                "waypoint_x": (
                    waypoint_update.waypoint_xy[0]
                    if waypoint_update is not None
                    else float("nan")
                ),
                "waypoint_y": (
                    waypoint_update.waypoint_xy[1]
                    if waypoint_update is not None
                    else float("nan")
                ),
                "waypoint_distance_m": waypoint_config.distance_m,
                "waypoint_replan_interval_steps": (
                    waypoint_config.replan_interval_steps
                ),
                "waypoint_reach_tolerance_m": (
                    waypoint_config.reach_tolerance_m
                ),
                "waypoint_minimum_hold_steps": (
                    waypoint_config.minimum_hold_steps
                ),
                "waypoint_maximum_hold_steps": (
                    waypoint_config.maximum_hold_steps
                ),
                "latched_direction_valid": int(latched_direction_valid),
                "raw_gate_active": int(raw_gate.active),
                "feasibility_gate_active": int(feasibility_active),
                "learned_magnitude_active": int(learned_magnitude_active),
                "lambda_active_threshold": lambda_active_threshold,
                "effective_gate_active": int(effective_active),
                "gate_transition": (
                    int(waypoint_update.transitioned)
                    if is_direct_waypoint
                    else (
                        int(state_update.transitioned) if is_stateful else 0
                    )
                ),
                "gate_reason": gate_reason,
                "gate_on_evidence": int(state_update.on_evidence),
                "gate_off_evidence": int(state_update.off_evidence),
                "gate_on_streak": state_update.on_streak,
                "gate_off_streak": state_update.off_streak,
                "gate_dwell_steps": state_update.dwell_steps,
                "hard_hazard_override": int(hard_override),
                "current_sdf_m": current_sdf,
                "current_hard": int(current_hard),
                "nominal_primitive_risk": raw_gate.nominal_risk,
                "best_primitive_risk": raw_gate.best_risk,
                "predicted_risk_improvement": state_update.improvement,
                "feasible_primitive_count": raw_gate.feasible_count,
                "selected_direction_row": latched_direction_rc[0],
                "selected_direction_col": latched_direction_rc[1],
                "selected_ray_min_clearance_m": raw_gate.selected_min_clearance_m,
                "selector_invoked": int(selector_invoked),
                "selector_enumerated_candidate_count": (
                    selector_enumerated_count
                ),
                "selector_improvement_eligible_count": (
                    selector_improvement_eligible_count
                ),
                "selector_all_ray_candidate_count": (
                    selector_improvement_eligible_count
                ),
                "selector_accepted_count": (
                    selector_diagnostics.accepted_count
                    if selector_diagnostics is not None
                    else 0
                ),
                "selector_no_safe_fallback": int(
                    selector_no_safe_fallback
                ),
                "selector_selected_primitive_index": (
                    selector_diagnostics.selected_primitive_index
                    if selector_diagnostics is not None
                    and selector_diagnostics.selected_primitive_index
                    is not None
                    else -1
                ),
                "selector_selected_direction_row": (
                    selector_direction_rc[0]
                    if selector_invoked
                    and not selector_no_safe_fallback
                    else float("nan")
                ),
                "selector_selected_direction_col": (
                    selector_direction_rc[1]
                    if selector_invoked
                    and not selector_no_safe_fallback
                    else float("nan")
                ),
                "selector_predicted_mean_risk": (
                    selector_selected_record.path_weighted_mean_risk
                    if selector_selected_record is not None
                    else float("nan")
                ),
                "selector_predicted_stage_goal_progress_m": (
                    selector_selected_record.stage_goal_progress_m
                    if selector_selected_record is not None
                    else float("nan")
                ),
                "selector_predicted_min_sdf_clearance_m": (
                    selector_selected_record.minimum_sdf_clearance_m
                    if selector_selected_record is not None
                    else float("nan")
                ),
                "selector_predicted_initial_velocity_cosine": (
                    selector_selected_record.initial_velocity_cosine
                    if selector_selected_record is not None
                    else float("nan")
                ),
                "selector_nominal_forecast_mean_risk": (
                    selector_nominal_record.path_weighted_mean_risk
                    if selector_nominal_record is not None
                    else float("nan")
                ),
                "selector_dynamic_risk_improvement": (
                    selector_dynamic_improvement
                ),
                "selector_replanned_this_step": int(
                    waypoint_update.replan_due
                    if waypoint_update is not None
                    else False
                ),
                "selector_direction_changed_on_replan": int(
                    selector_invoked
                    and waypoint_update.active
                    and trace
                    and bool(trace[-1]["effective_gate_active"])
                    and (
                        float(trace[-1]["selected_direction_row"])
                        != float(latched_direction_rc[0])
                        or float(trace[-1]["selected_direction_col"])
                        != float(latched_direction_rc[1])
                    )
                ),
                "selector_reject_invalid_nonfinite": (
                    selector_rejection_counts.get("invalid_nonfinite", 0)
                ),
                "selector_reject_out_of_bounds": (
                    selector_rejection_counts.get("out_of_bounds", 0)
                ),
                "selector_reject_hard_collision": (
                    selector_rejection_counts.get("hard_collision", 0)
                ),
                "selector_reject_clearance": (
                    selector_rejection_counts.get("clearance", 0)
                ),
                "selector_reject_insufficient_progress": (
                    selector_rejection_counts.get(
                        "insufficient_progress", 0
                    )
                ),
                "selector_reject_insufficient_dynamic_improvement": (
                    selector_rejection_counts.get(
                        "insufficient_dynamic_risk_improvement", 0
                    )
                ),
                "selector_reject_invalid_nominal_forecast": (
                    selector_rejection_counts.get(
                        "invalid_nominal_forecast", 0
                    )
                ),
                "selector_nominal_record_json": (
                    selector_nominal_record_json
                ),
                "selector_candidate_records_json": (
                    selector_candidate_records_json
                ),
                "projection_applied": int(projection_applied),
                **asdict(projection),
                "lam_soft_learned": float(lam_soft.item()),
                "direct_lambda_floor": direct_lambda_floor,
                "soft_multiplier": soft_multiplier,
                "lam_soft_used": float(lam_soft_used.item()),
                "lam_hard_used": float(lam_hard.item()),
                "displacement_alignment": displacement_alignment,
                "speed": float(np.linalg.norm(next_velocity)),
                "movement_m": movement_m,
                "risk": risk_value,
                "hard_contact": int(hard_contact),
            }
        )
        position = next_position
        velocity = next_velocity
        if gsd * float(np.linalg.norm(position - goal_xy)) <= 3.0:
            break

    elapsed = time.perf_counter() - start_time
    opening_step = int(spec.event_step + spec.open_delay)
    pre = [
        row
        for row in trace
        if int(spec.event_step) <= int(row["step"]) < opening_step
    ]
    post = [row for row in trace if int(row["step"]) >= opening_step]
    final_distance_m = gsd * float(np.linalg.norm(position - goal_xy))
    metric = {
        "controller_version": (
            DIRECT_WAYPOINT_CONTROLLER_VERSION
            if is_direct_waypoint
            else CONTROLLER_VERSION
        ),
        "episode_uid": str(episode["episode_uid"]),
        "scene_id": str(episode["scene_id"]),
        "sequence": str(episode["sequence"]),
        "regime": str(episode["regime"]),
        "event_type": str(spec.event_type),
        "mode": mode,
        "steps": len(trace),
        "success": int(final_distance_m <= 3.0),
        "final_distance_m": final_distance_m,
        "path_length_m": path_length_m,
        "cvar20_risk": _weighted_upper_tail(risks, weights),
        "cvar20_violation": _weighted_upper_tail(violations, weights),
        "hard_contacts": hard_contacts,
        "activation_rate": float(
            np.mean([row["effective_gate_active"] for row in trace])
        )
        if trace
        else 0.0,
        "false_pre_activation_rate": float(
            np.mean([row["effective_gate_active"] for row in pre])
        )
        if pre
        else 0.0,
        "post_open_activation_rate": float(
            np.mean([row["effective_gate_active"] for row in post])
        )
        if post
        else 0.0,
        "activation_transitions": int(
            sum(
                trace[index]["effective_gate_active"]
                != trace[index - 1]["effective_gate_active"]
                for index in range(1, len(trace))
            )
        ),
        "hard_override_steps": int(
            sum(row["hard_hazard_override"] for row in trace)
        ),
        "projection_steps": int(sum(row["projection_applied"] for row in trace)),
        "selector_invocations": int(
            sum(row["selector_invoked"] for row in trace)
        ),
        "selector_no_safe_fallbacks": int(
            sum(row["selector_no_safe_fallback"] for row in trace)
        ),
        "selector_candidates_simulated": int(
            sum(row["selector_improvement_eligible_count"] for row in trace)
        ),
        "selector_candidates_accepted": int(
            sum(row["selector_accepted_count"] for row in trace)
        ),
        "selector_reject_invalid_nonfinite": int(
            sum(row["selector_reject_invalid_nonfinite"] for row in trace)
        ),
        "selector_reject_out_of_bounds": int(
            sum(row["selector_reject_out_of_bounds"] for row in trace)
        ),
        "selector_reject_hard_collision": int(
            sum(row["selector_reject_hard_collision"] for row in trace)
        ),
        "selector_reject_clearance": int(
            sum(row["selector_reject_clearance"] for row in trace)
        ),
        "selector_reject_insufficient_progress": int(
            sum(row["selector_reject_insufficient_progress"] for row in trace)
        ),
        "selector_reject_insufficient_dynamic_improvement": int(
            sum(
                row[
                    "selector_reject_insufficient_dynamic_improvement"
                ]
                for row in trace
            )
        ),
        "selector_reject_invalid_nominal_forecast": int(
            sum(
                row["selector_reject_invalid_nominal_forecast"]
                for row in trace
            )
        ),
        "selector_replans": int(
            sum(row["selector_replanned_this_step"] for row in trace)
        ),
        "waypoint_replans": int(
            sum(
                row["waypoint_latch_reason"] == "replan_interval"
                for row in trace
            )
        ),
        "waypoint_releases": int(
            sum(
                bool(row["waypoint_latch_transition"])
                and not bool(row["waypoint_latch_active"])
                for row in trace
            )
        ),
        "mean_lam_soft_learned": float(
            np.mean([row["lam_soft_learned"] for row in trace])
        )
        if trace
        else 0.0,
        "compute_ms_per_step": 1000.0 * elapsed / max(1, len(trace)),
    }
    return metric, trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bev-root", type=Path, default=DEFAULT_BEV_ROOT)
    parser.add_argument(
        "--split", choices=ALLOWED_SPLITS, default="train",
        help="The sealed test split is deliberately unavailable.",
    )
    parser.add_argument(
        "--split-root", type=Path, default=Path("repair_experiments/splits")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("repair_experiments/results/v1_dynamic_smoke"),
    )
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=[
        "repaired", "stateless_unprojected", "gate_off"
    ])
    parser.add_argument("--event-type", default="delayed_required_escape")
    parser.add_argument("--event-fraction", type=float, default=0.38)
    parser.add_argument("--event-duration", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=140)
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
    parser.add_argument("--hard-override-clearance-m", type=float, default=0.5)
    parser.add_argument("--cone-half-angle-degrees", type=float, default=35.0)
    parser.add_argument(
        "--gradient-confidence-threshold", type=float, default=1e-3
    )
    parser.add_argument(
        "--low-confidence-fallback-policy",
        choices=["selected_axis", "zero"],
        default="selected_axis",
    )
    parser.add_argument(
        "--lambda-active-threshold",
        type=float,
        default=None,
        help=(
            "Override checkpoint repair_calibration.lambda_active_threshold. "
            "Historical checkpoints without metadata default to 0."
        ),
    )
    parser.add_argument("--on-improvement", type=float, default=0.05)
    parser.add_argument("--off-improvement", type=float, default=0.025)
    parser.add_argument("--on-material-trigger", type=float, default=0.45)
    parser.add_argument("--off-material-trigger", type=float, default=0.35)
    parser.add_argument("--on-persistence-steps", type=int, default=3)
    parser.add_argument("--off-persistence-steps", type=int, default=2)
    parser.add_argument("--minimum-dwell-steps", type=int, default=5)
    parser.add_argument("--waypoint-distance-m", type=float, default=1.0)
    parser.add_argument("--waypoint-active-step-limit", type=int, default=10)
    parser.add_argument(
        "--waypoint-cumulative-forward-limit-m", type=float, default=3.0
    )
    parser.add_argument(
        "--waypoint-rearm-inactive-steps", type=int, default=5
    )
    parser.add_argument("--selector-prediction-steps", type=int, default=6)
    parser.add_argument("--selector-progress-min-m", type=float, default=0.1)
    parser.add_argument(
        "--selector-swept-sample-spacing-m", type=float, default=0.25
    )
    parser.add_argument(
        "--selector-goal-direction-cosine-min", type=float, default=0.25
    )
    parser.add_argument(
        "--selector-velocity-direction-cosine-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--temporal-wait-timeout-steps", type=int, default=12)
    parser.add_argument(
        "--temporal-release-credit-steps", type=int, default=12
    )
    parser.add_argument(
        "--waypoint-replan-interval-steps", type=int, default=10
    )
    parser.add_argument(
        "--waypoint-reach-tolerance-m", type=float, default=0.25
    )
    parser.add_argument(
        "--waypoint-minimum-hold-steps", type=int, default=50
    )
    parser.add_argument(
        "--waypoint-maximum-hold-steps", type=int, default=100
    )
    parser.add_argument("--direct-lambda-floor", type=float, default=1.5)
    parser.add_argument("--hard-violation-penalty", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=27370)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "test" in args.split.lower():
        raise RuntimeError("sealed test evaluation is disabled in v1 development")
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

    split_path = args.split_root / f"{args.split}_static.json"
    split_manifest = json.loads(split_path.read_text())
    episodes = split_manifest["episodes"][: args.max_episodes]
    if any(str(episode["sequence"]) == "00004" for episode in episodes):
        raise RuntimeError("sealed sequence 00004 must not be loaded")
    model, checkpoint_cfg = _load_model(
        args.checkpoint, args.source_root, args.device
    )
    checkpoint_payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    repair_calibration = dict(checkpoint_payload.get("repair_calibration", {}))
    checkpoint_lambda_threshold = float(
        repair_calibration.get("lambda_active_threshold", 0.0)
    )
    lambda_active_threshold = (
        float(args.lambda_active_threshold)
        if args.lambda_active_threshold is not None
        else checkpoint_lambda_threshold
    )
    if lambda_active_threshold < 0.0:
        raise ValueError("lambda_active_threshold must be nonnegative")
    bev_manifest_path = args.bev_root / "manifest.json"
    bev_manifest = json.loads(bev_manifest_path.read_text())
    gsd = float(BevConfig(**bev_manifest["config"]["bev"]).resolution)
    gate_config = HysteresisConfig(
        on_improvement=args.on_improvement,
        off_improvement=args.off_improvement,
        on_material_trigger=args.on_material_trigger,
        off_material_trigger=args.off_material_trigger,
        on_persistence_steps=args.on_persistence_steps,
        off_persistence_steps=args.off_persistence_steps,
        minimum_dwell_steps=args.minimum_dwell_steps,
    )
    waypoint_config = WaypointLatchConfig(
        distance_m=args.waypoint_distance_m,
        active_step_limit=args.waypoint_active_step_limit,
        cumulative_forward_limit_m=(
            args.waypoint_cumulative_forward_limit_m
        ),
        rearm_inactive_steps=args.waypoint_rearm_inactive_steps,
        replan_interval_steps=args.waypoint_replan_interval_steps,
        reach_tolerance_m=args.waypoint_reach_tolerance_m,
        minimum_hold_steps=args.waypoint_minimum_hold_steps,
        maximum_hold_steps=args.waypoint_maximum_hold_steps,
    )
    temporal_config = TemporalReleaseConfig(
        wait_timeout_steps=args.temporal_wait_timeout_steps,
        release_credit_steps=args.temporal_release_credit_steps,
    )
    selector_config = VelocityAwareSelectorConfig(
        prediction_steps=args.selector_prediction_steps,
        progress_min_m=args.selector_progress_min_m,
        hard_margin_m=args.hard_margin_m,
        swept_sample_spacing_m=args.selector_swept_sample_spacing_m,
        goal_direction_cosine_min=(
            args.selector_goal_direction_cosine_min
        ),
        velocity_direction_cosine_min=(
            args.selector_velocity_direction_cosine_min
        ),
    )
    if args.direct_lambda_floor < 0.0:
        raise ValueError("direct_lambda_floor must be nonnegative")
    config = {
        "controller_version": CONTROLLER_VERSION,
        "data_policy": {
            "split": args.split,
            "manifest": str(split_path),
            "test_split_loaded": False,
            "sequence_00004_loaded": False,
        },
        "checkpoint_only_execution": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_cfg": checkpoint_cfg,
        "repair_calibration": repair_calibration,
        "lambda_active_threshold": {
            "effective": lambda_active_threshold,
            "checkpoint_value": checkpoint_lambda_threshold,
            "source": (
                "cli_override"
                if args.lambda_active_threshold is not None
                else (
                    "checkpoint_repair_calibration"
                    if "lambda_active_threshold" in repair_calibration
                    else "historical_checkpoint_default_zero"
                )
            ),
        },
        "split_manifest_sha256": _sha256(split_path),
        "bev_manifest_sha256": _sha256(bev_manifest_path),
        "hysteresis": asdict(gate_config),
        "direct_waypoint": {
            **asdict(waypoint_config),
            "lambda_floor": args.direct_lambda_floor,
            "activation_signal": (
                "stateful_feasibility_gate_with_magnitude_eligible_true"
            ),
            "one_shot_rearm": False,
            "rolling_target": True,
            "fixed_direction_per_activation": False,
            "replanning_transition_policy": (
                "safe active-to-active direction changes are not transitions; "
                "no-safe or hard-override steps deactivate actual soft control"
            ),
            "legacy_non_driving_fields": [
                "replan_interval_steps",
                "reach_tolerance_m",
                "minimum_hold_steps",
                "maximum_hold_steps",
            ],
            "temporal_release_rule": asdict(temporal_config),
            "temporal_inputs": (
                "current full hard mask, static hard-mask prior, current raw "
                "request, and current hard override only"
            ),
            "velocity_selector": {
                **asdict(selector_config),
                "ranking": [
                    "higher_stage_goal_progress",
                    "higher_minimum_sdf_clearance",
                    "lower_path_weighted_mean_risk",
                    "higher_initial_velocity_cosine",
                    "lower_primitive_index",
                ],
                "map_policy": "current_frozen_map_only",
                "candidate_filter": (
                    "all in-bounds stage1 hard-clearance-feasible rays without "
                    "static endpoint-progress or static-risk filtering; "
                    "accepted only when the short dynamic forecast passes "
                    "forward-direction/progress/safety checks and improves "
                    "over the matched dynamic nominal forecast by "
                    "on_improvement"
                ),
                "execution": (
                    "one real step then replan; velocity-error force direction "
                    "with a bounded learned-magnitude floor; short safe hold "
                    "when replanning temporarily finds no replacement"
                ),
            },
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    metrics: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
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
        for mode in args.modes:
            metric, trace = rollout(
                mode=mode,
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
                hard_override_clearance_m=args.hard_override_clearance_m,
                cone_half_angle_degrees=args.cone_half_angle_degrees,
                gradient_confidence_threshold=args.gradient_confidence_threshold,
                low_confidence_fallback_policy=args.low_confidence_fallback_policy,
                lambda_active_threshold=lambda_active_threshold,
                gate_config=gate_config,
                hard_violation_penalty=args.hard_violation_penalty,
                seed=args.seed + 10000 * episode_index,
                waypoint_config=waypoint_config,
                temporal_config=temporal_config,
                selector_config=selector_config,
                direct_lambda_floor=args.direct_lambda_floor,
            )
            metrics.append(metric)
            traces.extend(trace)
            print(
                f"{episode['episode_uid']} {mode}: success={metric['success']} "
                f"act={metric['activation_rate']:.3f} "
                f"transitions={metric['activation_transitions']} "
                f"lambda={metric['mean_lam_soft_learned']:.4f}",
                flush=True,
            )
    _write_csv(args.out / "per_episode_metrics.csv", metrics)
    _write_csv(args.out / "step_traces.csv", traces)
    summary: Dict[str, Any] = {
        "controller_version": CONTROLLER_VERSION,
        "split": args.split,
        "num_episodes": len(episodes),
        "num_rollouts": len(metrics),
        "modes": {},
    }
    for mode in args.modes:
        pool = [row for row in metrics if row["mode"] == mode]
        summary["modes"][mode] = {
            key: float(np.mean([float(row[key]) for row in pool]))
            for key in (
                "success",
                "final_distance_m",
                "cvar20_violation",
                "hard_contacts",
                "activation_rate",
                "false_pre_activation_rate",
                "post_open_activation_rate",
                "activation_transitions",
                "projection_steps",
                "selector_invocations",
                "selector_no_safe_fallbacks",
                "selector_candidates_simulated",
                "selector_candidates_accepted",
                "selector_reject_invalid_nonfinite",
                "selector_reject_out_of_bounds",
                "selector_reject_hard_collision",
                "selector_reject_clearance",
                "selector_reject_insufficient_progress",
                "selector_reject_insufficient_dynamic_improvement",
                "selector_reject_invalid_nominal_forecast",
                "selector_replans",
                "waypoint_replans",
                "waypoint_releases",
                "mean_lam_soft_learned",
            )
        }
    summary["smoke_checks"] = {
        "all_sequences_allowed": all(
            row["sequence"] in {"00000", "00001", "00002", "00003"}
            for row in metrics
        ),
        "checkpoint_executed_every_step": len(traces) > 0
        and all(np.isfinite(float(row["lam_soft_learned"])) for row in traces),
        "gate_off_multiplier_matches_learned_threshold": all(
            float(row["soft_multiplier"])
            == float(row["learned_magnitude_active"])
            for row in traces
            if row["mode"] == "gate_off"
        ),
        "geometry_only_multiplier_zero": all(
            float(row["soft_multiplier"]) == 0.0
            for row in traces
            if row["mode"] == "geometry_only"
        ),
        "hard_override_never_leaves_stateful_gate_active": all(
            not (
                bool(row["hard_hazard_override"])
                and bool(row["effective_gate_active"])
            )
            for row in traces
            if row["mode"] in {"repaired", "stateful_unprojected"}
        ),
        "direct_waypoint_ignores_learned_binary_threshold": all(
            bool(row["effective_gate_active"])
            == bool(row["waypoint_latch_active"])
            for row in traces
            if row["mode"] == "direct_waypoint"
        ),
        "direct_waypoint_uses_lambda_floor_when_active": all(
            (not bool(row["effective_gate_active"]))
            or float(row["lam_soft_used"]) + 1e-8
            >= float(row["direct_lambda_floor"])
            for row in traces
            if row["mode"] == "direct_waypoint"
        ),
        "hard_override_never_leaves_direct_waypoint_active": all(
            not (
                bool(row["hard_hazard_override"])
                and bool(row["waypoint_latch_active"])
            )
            for row in traces
            if row["mode"] == "direct_waypoint"
        ),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
