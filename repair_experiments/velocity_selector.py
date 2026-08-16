"""Deterministic short-horizon simulation and lexicographic candidate ranking."""
from __future__ import annotations

import contextlib
import math
from collections import Counter
from dataclasses import replace
from typing import Callable, Iterator, Mapping, Sequence, Tuple

import numpy as np

from repair_experiments.v1_controller import (
    StagewisePrimitiveCandidate,
    VelocityAwareCandidateDiagnostics,
    VelocityAwareSelectorConfig,
    VelocityAwareSelectorDiagnostics,
    primitive_direction_xy,
)


StageStep = Callable[
    [np.ndarray, np.ndarray, np.ndarray, Tuple[float, float], int],
    Tuple[np.ndarray, np.ndarray],
]


@contextlib.contextmanager
def preserved_numpy_rng(seed: int) -> Iterator[None]:
    """Temporarily seed legacy NumPy RNG without perturbing its caller."""

    state = np.random.get_state()
    try:
        np.random.seed(int(seed) % (2**32))
        yield
    finally:
        np.random.set_state(state)


def _sample_rc(point_xy: np.ndarray, shape: Tuple[int, int]) -> Tuple[int, int]:
    point_rc = np.asarray(point_xy, dtype=np.float64)[::-1]
    return (
        int(np.clip(round(float(point_rc[0])), 0, shape[0] - 1)),
        int(np.clip(round(float(point_rc[1])), 0, shape[1] - 1)),
    )


def _in_bounds(point_xy: np.ndarray, shape: Tuple[int, int]) -> bool:
    return bool(
        0.0 <= float(point_xy[0]) < shape[1]
        and 0.0 <= float(point_xy[1]) < shape[0]
    )


def _rejected(
    candidate: StagewisePrimitiveCandidate,
    reason: str,
    *,
    endpoint: np.ndarray,
    velocity: np.ndarray,
    stages: int,
    sample_count: int,
    weighted_risk: float,
    path_length_m: float,
    progress_m: float,
    min_clearance_m: float,
    velocity_cosine: float,
) -> VelocityAwareCandidateDiagnostics:
    return VelocityAwareCandidateDiagnostics(
        primitive_index=candidate.primitive_index,
        direction_rc=candidate.direction_rc,
        predicted_endpoint_xy=(float(endpoint[0]), float(endpoint[1])),
        predicted_terminal_velocity_xy=(
            float(velocity[0]),
            float(velocity[1]),
        ),
        accepted=False,
        rejection_reason=reason,
        simulated_stages=stages,
        swept_sample_count=sample_count,
        path_weighted_mean_risk=(
            weighted_risk / path_length_m
            if path_length_m > 1e-12
            else float("inf")
        ),
        stage_goal_progress_m=progress_m,
        minimum_sdf_clearance_m=min_clearance_m,
        initial_velocity_cosine=velocity_cosine,
    )


def simulate_velocity_candidate(
    *,
    candidate: StagewisePrimitiveCandidate,
    initial_position_xy: np.ndarray,
    initial_velocity_xy: np.ndarray,
    stage_goal_xy: np.ndarray,
    maps: Mapping[str, np.ndarray],
    gsd: float,
    rolling_distance_m: float,
    config: VelocityAwareSelectorConfig,
    stage_step: StageStep,
    enforce_safety: bool = True,
) -> VelocityAwareCandidateDiagnostics:
    """Forecast one fixed-direction candidate on one frozen map."""

    if gsd <= 0.0:
        raise ValueError("gsd must be positive")
    if rolling_distance_m <= 0.0:
        raise ValueError("rolling_distance_m must be positive")
    risk = np.asarray(maps["risk_map"])
    hard = np.asarray(maps["hard_mask"], dtype=bool)
    sdf = np.asarray(maps["sdf_hard"])
    if risk.shape != hard.shape or risk.shape != sdf.shape:
        raise ValueError("risk, hard, and SDF maps must have the same shape")

    position = np.asarray(initial_position_xy, dtype=np.float64).copy()
    velocity = np.asarray(initial_velocity_xy, dtype=np.float64).copy()
    stage_goal = np.asarray(stage_goal_xy, dtype=np.float64)
    direction_xy = primitive_direction_xy(candidate.direction_rc).astype(
        np.float64
    )
    velocity_cosine = 0.0
    min_clearance = float("inf")
    weighted_risk = 0.0
    path_length_m = 0.0
    sample_count = 0
    completed_stages = 0

    if (
        position.shape != (2,)
        or velocity.shape != (2,)
        or stage_goal.shape != (2,)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(velocity))
        or not np.all(np.isfinite(stage_goal))
    ):
        return _rejected(
            candidate,
            "invalid_nonfinite",
            endpoint=np.zeros(2) if position.shape != (2,) else position,
            velocity=np.zeros(2) if velocity.shape != (2,) else velocity,
            stages=0,
            sample_count=0,
            weighted_risk=0.0,
            path_length_m=0.0,
            progress_m=0.0,
            min_clearance_m=float("inf"),
            velocity_cosine=velocity_cosine,
        )
    speed = float(np.linalg.norm(velocity))
    velocity_cosine = (
        float(np.dot(velocity, direction_xy) / speed)
        if speed > 1e-8
        else 0.0
    )
    goal_delta = stage_goal - position
    goal_delta_norm = float(np.linalg.norm(goal_delta))
    goal_direction_cosine = (
        float(np.dot(direction_xy, goal_delta) / goal_delta_norm)
        if goal_delta_norm > 1e-8
        else 1.0
    )
    if (
        enforce_safety
        and goal_direction_cosine < config.goal_direction_cosine_min
    ):
        return _rejected(
            candidate,
            "nonforward_goal_direction",
            endpoint=position,
            velocity=velocity,
            stages=0,
            sample_count=0,
            weighted_risk=0.0,
            path_length_m=0.0,
            progress_m=0.0,
            min_clearance_m=float("inf"),
            velocity_cosine=velocity_cosine,
        )
    if (
        enforce_safety
        and speed > 1e-8
        and velocity_cosine < config.velocity_direction_cosine_min
    ):
        return _rejected(
            candidate,
            "nonforward_velocity_direction",
            endpoint=position,
            velocity=velocity,
            stages=0,
            sample_count=0,
            weighted_risk=0.0,
            path_length_m=0.0,
            progress_m=0.0,
            min_clearance_m=float("inf"),
            velocity_cosine=velocity_cosine,
        )
    initial_goal_distance_m = gsd * float(
        np.linalg.norm(stage_goal - position)
    )
    if enforce_safety and not _in_bounds(position, risk.shape):
        return _rejected(
            candidate,
            "out_of_bounds",
            endpoint=position,
            velocity=velocity,
            stages=0,
            sample_count=0,
            weighted_risk=0.0,
            path_length_m=0.0,
            progress_m=0.0,
            min_clearance_m=float("inf"),
            velocity_cosine=velocity_cosine,
        )
    start_cell = _sample_rc(position, risk.shape)
    start_clearance = float(sdf[start_cell])
    min_clearance = start_clearance
    sample_count = 1
    if enforce_safety and bool(hard[start_cell]):
        return _rejected(
            candidate,
            "hard_collision",
            endpoint=position,
            velocity=velocity,
            stages=0,
            sample_count=sample_count,
            weighted_risk=0.0,
            path_length_m=0.0,
            progress_m=0.0,
            min_clearance_m=min_clearance,
            velocity_cosine=velocity_cosine,
        )
    if enforce_safety and start_clearance < config.hard_margin_m:
        return _rejected(
            candidate,
            "clearance",
            endpoint=position,
            velocity=velocity,
            stages=0,
            sample_count=sample_count,
            weighted_risk=0.0,
            path_length_m=0.0,
            progress_m=0.0,
            min_clearance_m=min_clearance,
            velocity_cosine=velocity_cosine,
        )

    for stage_index in range(config.prediction_steps):
        rolling_goal = (
            position
            + (rolling_distance_m / gsd) * direction_xy
        )
        next_position, next_velocity = stage_step(
            position.astype(np.float32),
            velocity.astype(np.float32),
            rolling_goal.astype(np.float32),
            candidate.direction_rc,
            stage_index,
        )
        next_position = np.asarray(next_position, dtype=np.float64)
        next_velocity = np.asarray(next_velocity, dtype=np.float64)
        if (
            next_position.shape != (2,)
            or next_velocity.shape != (2,)
            or not np.all(np.isfinite(next_position))
            or not np.all(np.isfinite(next_velocity))
        ):
            return _rejected(
                candidate,
                "invalid_nonfinite",
                endpoint=position,
                velocity=velocity,
                stages=completed_stages,
                sample_count=sample_count,
                weighted_risk=weighted_risk,
                path_length_m=path_length_m,
                progress_m=0.0,
                min_clearance_m=min_clearance,
                velocity_cosine=velocity_cosine,
            )
        if enforce_safety and not _in_bounds(next_position, risk.shape):
            return _rejected(
                candidate,
                "out_of_bounds",
                endpoint=next_position,
                velocity=next_velocity,
                stages=completed_stages,
                sample_count=sample_count,
                weighted_risk=weighted_risk,
                path_length_m=path_length_m,
                progress_m=0.0,
                min_clearance_m=min_clearance,
                velocity_cosine=velocity_cosine,
            )

        segment = next_position - position
        segment_m = gsd * float(np.linalg.norm(segment))
        subdivisions = max(
            1,
            int(math.ceil(segment_m / config.swept_sample_spacing_m)),
        )
        subsegment_m = segment_m / subdivisions
        for sample_index in range(1, subdivisions + 1):
            fraction = float(sample_index) / float(subdivisions)
            sample_xy = position + fraction * segment
            if enforce_safety and not _in_bounds(sample_xy, risk.shape):
                return _rejected(
                    candidate,
                    "out_of_bounds",
                    endpoint=sample_xy,
                    velocity=next_velocity,
                    stages=completed_stages,
                    sample_count=sample_count,
                    weighted_risk=weighted_risk,
                    path_length_m=path_length_m,
                    progress_m=0.0,
                    min_clearance_m=min_clearance,
                    velocity_cosine=velocity_cosine,
                )
            cell = _sample_rc(sample_xy, risk.shape)
            sample_count += 1
            clearance = float(sdf[cell])
            min_clearance = min(min_clearance, clearance)
            weighted_risk += float(risk[cell]) * subsegment_m
            path_length_m += subsegment_m
            if enforce_safety and bool(hard[cell]):
                return _rejected(
                    candidate,
                    "hard_collision",
                    endpoint=sample_xy,
                    velocity=next_velocity,
                    stages=completed_stages,
                    sample_count=sample_count,
                    weighted_risk=weighted_risk,
                    path_length_m=path_length_m,
                    progress_m=0.0,
                    min_clearance_m=min_clearance,
                    velocity_cosine=velocity_cosine,
                )
            if enforce_safety and clearance < config.hard_margin_m:
                return _rejected(
                    candidate,
                    "clearance",
                    endpoint=sample_xy,
                    velocity=next_velocity,
                    stages=completed_stages,
                    sample_count=sample_count,
                    weighted_risk=weighted_risk,
                    path_length_m=path_length_m,
                    progress_m=0.0,
                    min_clearance_m=min_clearance,
                    velocity_cosine=velocity_cosine,
                )
        position = next_position
        velocity = next_velocity
        completed_stages += 1

    progress_m = initial_goal_distance_m - gsd * float(
        np.linalg.norm(stage_goal - position)
    )
    if progress_m < config.progress_min_m:
        return _rejected(
            candidate,
            "insufficient_progress",
            endpoint=position,
            velocity=velocity,
            stages=completed_stages,
            sample_count=sample_count,
            weighted_risk=weighted_risk,
            path_length_m=path_length_m,
            progress_m=progress_m,
            min_clearance_m=min_clearance,
            velocity_cosine=velocity_cosine,
        )
    return VelocityAwareCandidateDiagnostics(
        primitive_index=candidate.primitive_index,
        direction_rc=candidate.direction_rc,
        predicted_endpoint_xy=(float(position[0]), float(position[1])),
        predicted_terminal_velocity_xy=(
            float(velocity[0]),
            float(velocity[1]),
        ),
        accepted=True,
        rejection_reason="",
        simulated_stages=completed_stages,
        swept_sample_count=sample_count,
        path_weighted_mean_risk=(
            weighted_risk / path_length_m
            if path_length_m > 1e-12
            else float(risk[_sample_rc(position, risk.shape)])
        ),
        stage_goal_progress_m=progress_m,
        minimum_sdf_clearance_m=min_clearance,
        initial_velocity_cosine=velocity_cosine,
    )


def rank_velocity_candidates(
    diagnostics: Sequence[VelocityAwareCandidateDiagnostics],
) -> VelocityAwareSelectorDiagnostics:
    """Apply the declared lexicographic ordering to accepted candidates."""

    records = tuple(diagnostics)
    accepted = [record for record in records if record.accepted]
    accepted.sort(
        key=lambda item: (
            -item.stage_goal_progress_m,
            -item.minimum_sdf_clearance_m,
            item.path_weighted_mean_risk,
            -item.initial_velocity_cosine,
            item.primitive_index,
        )
    )
    selected = accepted[0] if accepted else None
    rejection_counts = Counter(
        record.rejection_reason for record in records if not record.accepted
    )
    return VelocityAwareSelectorDiagnostics(
        candidate_count=len(records),
        accepted_count=len(accepted),
        selected_primitive_index=(
            selected.primitive_index if selected is not None else None
        ),
        selected_direction_rc=(
            selected.direction_rc if selected is not None else (0.0, 0.0)
        ),
        rejection_counts=tuple(sorted(rejection_counts.items())),
        candidate_diagnostics=records,
    )


def require_nominal_forecast_improvement(
    diagnostics: Sequence[VelocityAwareCandidateDiagnostics],
    *,
    nominal_path_weighted_mean_risk: float,
    improvement_margin: float,
) -> Tuple[VelocityAwareCandidateDiagnostics, ...]:
    """Reject otherwise-safe candidates that do not beat forecast nominal."""

    if improvement_margin < 0.0:
        raise ValueError("improvement_margin must be nonnegative")
    output = []
    valid_nominal = math.isfinite(float(nominal_path_weighted_mean_risk))
    for record in diagnostics:
        if not record.accepted:
            output.append(record)
            continue
        improvement = (
            float(nominal_path_weighted_mean_risk)
            - record.path_weighted_mean_risk
        )
        if not valid_nominal:
            output.append(
                replace(
                    record,
                    accepted=False,
                    rejection_reason="invalid_nominal_forecast",
                )
            )
        elif improvement + 1e-12 < improvement_margin:
            output.append(
                replace(
                    record,
                    accepted=False,
                    rejection_reason=(
                        "insufficient_dynamic_risk_improvement"
                    ),
                )
            )
        else:
            output.append(record)
    return tuple(output)
