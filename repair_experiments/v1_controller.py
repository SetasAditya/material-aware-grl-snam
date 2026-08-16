"""Controller-side repair primitives for the learned material field.

Version ``v1`` makes two deliberately small changes around the frozen
``CoefEnergyNetMaterial`` model:

* a stateful gate replaces frame-wise switching; and
* when active, the learned soft force is projected into a cone centred on the
  feasible primitive selected by the gate.

The learned coefficient remains the force magnitude.  The primitive is only a
directional constraint, not a second controller or a waypoint follower.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np
import torch


CONTROLLER_VERSION = "checkpoint_projected_hysteresis_v1"
DIRECT_WAYPOINT_CONTROLLER_VERSION = (
    "checkpoint_direct_forward_velocity_tracking_v7"
)


@dataclass(frozen=True)
class TemporalReleaseConfig:
    wait_timeout_steps: int = 12
    release_credit_steps: int = 12

    def __post_init__(self) -> None:
        if self.wait_timeout_steps < 1:
            raise ValueError("wait_timeout_steps must be at least one")
        if self.release_credit_steps < 1:
            raise ValueError("release_credit_steps must be at least one")


@dataclass(frozen=True)
class TemporalReleaseUpdate:
    state: str
    reason: str
    suppress_activation: bool
    immediate_activation_pulse: bool
    added_hard_count: int
    snapshotted_hard_count: int
    released_snapshot_cell_count: int
    wait_age_steps: int
    release_credit_remaining: int


def direct_activation_admitted(
    *,
    stateful_feasibility_active: bool,
    temporal_update: TemporalReleaseUpdate,
    hard_hazard_override: bool,
) -> bool:
    """Combine normal feasibility and the one-shot temporal pulse."""

    if hard_hazard_override:
        return False
    return bool(
        (
            stateful_feasibility_active
            and not temporal_update.suppress_activation
        )
        or temporal_update.immediate_activation_pulse
    )


class TemporalObstacleReleaseGate:
    """Online delayed-obstacle release rule with no event-time information."""

    NORMAL = "NORMAL"
    WAIT_RELEASE = "WAIT_RELEASE"
    BYPASS_UNTIL_CLEAR = "BYPASS_UNTIL_CLEAR"
    RELEASE_CREDIT = "RELEASE_CREDIT"

    def __init__(self, config: TemporalReleaseConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.state = self.NORMAL
        self.snapshot_added_hard: Optional[np.ndarray] = None
        self.wait_age_steps = 0
        self.release_credit_remaining = 0

    def update(
        self,
        *,
        raw_activation_requested: bool,
        current_hard_mask: np.ndarray,
        static_hard_mask: np.ndarray,
        hard_hazard_override: bool,
    ) -> TemporalReleaseUpdate:
        current = np.asarray(current_hard_mask, dtype=bool)
        static = np.asarray(static_hard_mask, dtype=bool)
        if current.shape != static.shape:
            raise ValueError("current and static hard masks must have same shape")
        added = current & ~static
        added_count = int(np.count_nonzero(added))
        raw_hard_safe = bool(
            raw_activation_requested and not hard_hazard_override
        )
        released_count = (
            int(np.count_nonzero(self.snapshot_added_hard & ~current))
            if self.snapshot_added_hard is not None
            else 0
        )
        suppress = False
        pulse = False
        reason = "normal"

        if self.state == self.NORMAL:
            if raw_hard_safe and added_count > 0:
                self.snapshot_added_hard = added.copy()
                self.wait_age_steps = 0
                self.release_credit_remaining = 0
                self.state = self.WAIT_RELEASE
                suppress = True
                reason = "snapshot_added_hard_and_wait"
        elif self.state == self.WAIT_RELEASE:
            suppress = True
            if released_count > 0:
                self.snapshot_added_hard = None
                self.wait_age_steps = 0
                self.release_credit_remaining = (
                    self.config.release_credit_steps
                )
                self.state = self.RELEASE_CREDIT
                suppress = True
                reason = "snapshotted_cell_released"
                if raw_hard_safe:
                    pulse = True
                    suppress = False
                    self.release_credit_remaining = 0
                    self.state = self.NORMAL
                    reason = "release_credit_immediate_pulse"
            else:
                self.wait_age_steps += 1
                reason = "wait_for_snapshotted_release"
                if self.wait_age_steps >= self.config.wait_timeout_steps:
                    self.snapshot_added_hard = None
                    self.wait_age_steps = 0
                    self.state = self.BYPASS_UNTIL_CLEAR
                    suppress = False
                    reason = "wait_timeout_bypass"
        elif self.state == self.BYPASS_UNTIL_CLEAR:
            reason = "bypass_uses_normal_gate"
            if not raw_activation_requested:
                self.state = self.NORMAL
                reason = "bypass_cleared"
        elif self.state == self.RELEASE_CREDIT:
            suppress = True
            reason = "hold_release_credit"
            if raw_hard_safe:
                pulse = True
                suppress = False
                self.release_credit_remaining = 0
                self.state = self.NORMAL
                reason = "release_credit_immediate_pulse"
            else:
                self.release_credit_remaining = max(
                    0, self.release_credit_remaining - 1
                )
                if self.release_credit_remaining == 0:
                    self.state = self.NORMAL
                    suppress = False
                    reason = "release_credit_expired"
        else:
            raise RuntimeError(f"unknown temporal state {self.state!r}")

        return TemporalReleaseUpdate(
            state=self.state,
            reason=reason,
            suppress_activation=suppress,
            immediate_activation_pulse=pulse,
            added_hard_count=added_count,
            snapshotted_hard_count=(
                int(np.count_nonzero(self.snapshot_added_hard))
                if self.snapshot_added_hard is not None
                else 0
            ),
            released_snapshot_cell_count=released_count,
            wait_age_steps=self.wait_age_steps,
            release_credit_remaining=self.release_credit_remaining,
        )


@dataclass(frozen=True)
class WaypointLatchConfig:
    """Configuration for the fixed-direction rolling-target repair."""

    distance_m: float = 1.0
    active_step_limit: int = 10
    cumulative_forward_limit_m: float = 3.0
    rearm_inactive_steps: int = 5
    # Retained temporarily for call-site compatibility. These fixed-waypoint
    # parameters do not affect the rolling-target controller.
    replan_interval_steps: int = 10
    reach_tolerance_m: float = 0.25
    minimum_hold_steps: int = 50
    maximum_hold_steps: int = 100

    def __post_init__(self) -> None:
        if self.distance_m <= 0.0:
            raise ValueError("distance_m must be positive")
        if self.active_step_limit < 1:
            raise ValueError("active_step_limit must be at least one")
        if self.cumulative_forward_limit_m <= 0.0:
            raise ValueError("cumulative_forward_limit_m must be positive")
        if self.rearm_inactive_steps < 1:
            raise ValueError("rearm_inactive_steps must be at least one")
        if self.replan_interval_steps < 1:
            raise ValueError("replan_interval_steps must be at least one")
        if self.reach_tolerance_m < 0.0:
            raise ValueError("reach_tolerance_m must be nonnegative")
        if self.minimum_hold_steps < 1:
            raise ValueError("minimum_hold_steps must be at least one")
        if self.maximum_hold_steps < self.minimum_hold_steps:
            raise ValueError(
                "maximum_hold_steps must not be less than minimum_hold_steps"
            )


@dataclass(frozen=True)
class WaypointLatchUpdate:
    active: bool
    transitioned: bool
    reason: str
    waypoint_xy: Tuple[float, float]
    direction_rc: Tuple[float, float]
    age_steps: int
    hold_steps: int
    replan_due: bool
    armed: bool
    rearmed: bool
    activation_block_reason: str
    forward_progress_delta_m: float
    cumulative_forward_progress_m: float
    active_steps: int
    active_step_limit: int
    cumulative_forward_limit_m: float
    rearm_inactive_streak: int
    rearm_inactive_required: int


@dataclass(frozen=True)
class StagewisePrimitiveCandidate:
    """One progress-making, hard-feasible ray in raster coordinates."""

    primitive_index: int
    direction_rc: Tuple[float, float]
    endpoint_rc: Tuple[float, float]
    mean_risk: float
    min_clearance_m: float


@dataclass(frozen=True)
class StagewisePrimitiveEnumeration:
    """Candidate set plus the nominal-ray baseline used by the current gate."""

    nominal_direction_rc: Tuple[float, float]
    nominal_endpoint_rc: Tuple[float, float]
    nominal_mean_risk: float
    candidates: Tuple[StagewisePrimitiveCandidate, ...]


@dataclass(frozen=True)
class VelocityAwareSelectorConfig:
    """Lexicographic Stage 2 forecast settings."""

    prediction_steps: int = 6
    progress_min_m: float = 0.1
    hard_margin_m: float = 1.0
    swept_sample_spacing_m: float = 0.25
    goal_direction_cosine_min: float = 0.25
    velocity_direction_cosine_min: float = 0.0

    def __post_init__(self) -> None:
        if self.prediction_steps < 1:
            raise ValueError("prediction_steps must be at least one")
        for name in (
            "progress_min_m",
            "hard_margin_m",
            "swept_sample_spacing_m",
            "goal_direction_cosine_min",
            "velocity_direction_cosine_min",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.progress_min_m < 0.0:
            raise ValueError("progress_min_m must be nonnegative")
        if self.hard_margin_m < 0.0:
            raise ValueError("hard_margin_m must be nonnegative")
        if self.swept_sample_spacing_m <= 0.0:
            raise ValueError("swept_sample_spacing_m must be positive")
        for name in (
            "goal_direction_cosine_min",
            "velocity_direction_cosine_min",
        ):
            if not -1.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")


@dataclass(frozen=True)
class VelocityAwareCandidateDiagnostics:
    primitive_index: int
    direction_rc: Tuple[float, float]
    predicted_endpoint_xy: Tuple[float, float]
    predicted_terminal_velocity_xy: Tuple[float, float]
    accepted: bool
    rejection_reason: str
    simulated_stages: int
    swept_sample_count: int
    path_weighted_mean_risk: float
    stage_goal_progress_m: float
    minimum_sdf_clearance_m: float
    initial_velocity_cosine: float


@dataclass(frozen=True)
class VelocityAwareSelectorDiagnostics:
    candidate_count: int
    accepted_count: int
    selected_primitive_index: Optional[int]
    selected_direction_rc: Tuple[float, float]
    rejection_counts: Tuple[Tuple[str, int], ...]
    candidate_diagnostics: Tuple[
        VelocityAwareCandidateDiagnostics, ...
    ]


def receding_candidate_pool(
    enumeration: StagewisePrimitiveEnumeration,
) -> Tuple[StagewisePrimitiveCandidate, ...]:
    """Return every Stage 1 geometric candidate without a static-risk filter."""

    return enumeration.candidates


class DirectWaypointLatch:
    """Execute one gate-selected direction through a rolling local target.

    The direction is fixed for an activation, while the target remains
    ``distance_m`` ahead of the current position. Positive along-track motion
    is accumulated without allowing backward or lateral motion to subtract
    from it. The activation is bounded by both time and cumulative progress.
    """

    def __init__(self, config: WaypointLatchConfig, *, gsd: float):
        if gsd <= 0.0:
            raise ValueError("gsd must be positive")
        self.config = config
        self.gsd = float(gsd)
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.waypoint_xy = np.zeros(2, dtype=np.float32)
        self.previous_position_xy = np.zeros(2, dtype=np.float32)
        self.direction_rc = (0.0, 0.0)
        self.age_steps = 0
        self.hold_steps = 0
        self.active_steps = 0
        self.forward_progress_delta_m = 0.0
        self.cumulative_forward_progress_m = 0.0
        self.armed = True
        self.rearm_inactive_streak = 0

    def _activate(
        self,
        position_xy: np.ndarray,
        selected_direction_rc: Tuple[float, float],
        reason: str,
        *,
        transitioned: bool,
    ) -> WaypointLatchUpdate:
        direction_xy = primitive_direction_xy(selected_direction_rc)
        current_position = np.asarray(position_xy, dtype=np.float32)
        self.previous_position_xy = current_position.copy()
        self.waypoint_xy = (
            current_position
            + (self.config.distance_m / self.gsd) * direction_xy
        )
        self.direction_rc = (
            float(selected_direction_rc[0]),
            float(selected_direction_rc[1]),
        )
        self.active = True
        self.active_steps = 1
        self.age_steps = self.active_steps
        self.hold_steps = self.active_steps
        self.forward_progress_delta_m = 0.0
        self.cumulative_forward_progress_m = 0.0
        self.rearm_inactive_streak = 0
        return self._result(reason, transitioned=transitioned)

    def _release(self, reason: str) -> WaypointLatchUpdate:
        was_active = self.active
        self.active = False
        self.age_steps = 0
        self.hold_steps = 0
        self.armed = False
        self.rearm_inactive_streak = 0
        return self._result(reason, transitioned=was_active)

    def _result(
        self,
        reason: str,
        *,
        transitioned: bool,
        replan_due: bool = False,
        rearmed: bool = False,
        activation_block_reason: str = "",
    ) -> WaypointLatchUpdate:
        return WaypointLatchUpdate(
            active=self.active,
            transitioned=transitioned,
            reason=reason,
            waypoint_xy=(
                float(self.waypoint_xy[0]),
                float(self.waypoint_xy[1]),
            ),
            direction_rc=self.direction_rc,
            age_steps=self.age_steps,
            hold_steps=self.hold_steps,
            replan_due=replan_due,
            armed=self.armed,
            rearmed=rearmed,
            activation_block_reason=activation_block_reason,
            forward_progress_delta_m=self.forward_progress_delta_m,
            cumulative_forward_progress_m=(
                self.cumulative_forward_progress_m
            ),
            active_steps=self.active_steps,
            active_step_limit=self.config.active_step_limit,
            cumulative_forward_limit_m=(
                self.config.cumulative_forward_limit_m
            ),
            rearm_inactive_streak=self.rearm_inactive_streak,
            rearm_inactive_required=self.config.rearm_inactive_steps,
        )

    def update(
        self,
        *,
        position_xy: np.ndarray,
        gate_active: bool,
        selected_direction_rc: Tuple[float, float],
        latched_direction_valid: bool,
        hard_hazard_override: bool,
    ) -> WaypointLatchUpdate:
        """Advance the latch by one controller tick."""

        if hard_hazard_override:
            if not self.active:
                return self._result(
                    "inactive_hard_hazard_hold", transitioned=False
                )
            return self._release("hard_hazard_override")

        if self.active:
            current_position = np.asarray(position_xy, dtype=np.float32)
            direction_xy = primitive_direction_xy(self.direction_rc)
            self.forward_progress_delta_m = max(
                0.0,
                self.gsd
                * float(
                    np.dot(
                        current_position - self.previous_position_xy,
                        direction_xy,
                    )
                ),
            )
            self.cumulative_forward_progress_m += (
                self.forward_progress_delta_m
            )
            self.previous_position_xy = current_position.copy()
            if not latched_direction_valid:
                return self._release("latched_primitive_invalid")
            if (
                self.cumulative_forward_progress_m
                >= self.config.cumulative_forward_limit_m
            ):
                return self._release("cumulative_forward_limit_reached")
            if self.active_steps >= self.config.active_step_limit:
                return self._release("active_step_limit_reached")
            self.waypoint_xy = (
                current_position
                + (self.config.distance_m / self.gsd) * direction_xy
            )
            self.active_steps += 1
            self.age_steps = self.active_steps
            self.hold_steps = self.active_steps
            return self._result("hold_rolling_target", transitioned=False)

        selected_norm = float(np.linalg.norm(np.asarray(selected_direction_rc)))
        if not gate_active:
            if not self.armed:
                self.rearm_inactive_streak += 1
                if (
                    self.rearm_inactive_streak
                    >= self.config.rearm_inactive_steps
                ):
                    self.armed = True
                    return self._result(
                        "rearm_after_sustained_feasibility_inactive",
                        transitioned=False,
                        rearmed=True,
                    )
                return self._result(
                    "collect_rearm_inactive",
                    transitioned=False,
                    activation_block_reason="rearm_inactive_streak_incomplete",
                )
            return self._result("hold_inactive", transitioned=False)
        self.rearm_inactive_streak = 0
        if self.armed and selected_norm > 1e-8:
            # Consuming the arm makes this a one-shot activation for the
            # current stateful feasibility episode.
            self.armed = False
            return self._activate(
                position_xy,
                selected_direction_rc,
                "activate_from_feasibility_gate",
                transitioned=True,
            )
        return self._result(
            "activation_blocked_until_rearm",
            transitioned=False,
            activation_block_reason=(
                "feasibility_state_still_active"
                if not self.armed
                else "selected_direction_unavailable"
            ),
        )

    def update_receding(
        self,
        *,
        position_xy: np.ndarray,
        gate_active: bool,
        selected_direction_rc: Tuple[float, float],
        selection_attempted: bool,
        hard_hazard_override: bool,
        latched_direction_valid: bool = True,
    ) -> WaypointLatchUpdate:
        """Execute exactly one rolling-direction action before replanning.

        Consecutive safe selections remain one logically continuous active
        interval even when the selected direction changes.
        """

        previous_active = self.active
        if hard_hazard_override:
            self.active = False
            self.armed = True
            self.active_steps = 0
            self.age_steps = 0
            self.hold_steps = 0
            return self._result(
                "hard_hazard_override",
                transitioned=previous_active,
            )

        direction_norm = float(
            np.linalg.norm(np.asarray(selected_direction_rc))
        )
        if gate_active and direction_norm > 1e-8:
            direction_xy = primitive_direction_xy(selected_direction_rc)
            current = np.asarray(position_xy, dtype=np.float32)
            self.waypoint_xy = (
                current
                + (self.config.distance_m / self.gsd) * direction_xy
            )
            self.direction_rc = (
                float(selected_direction_rc[0]),
                float(selected_direction_rc[1]),
            )
            self.active = True
            self.armed = True
            self.active_steps = (
                self.active_steps + 1 if previous_active else 1
            )
            self.age_steps = self.active_steps
            self.hold_steps = self.active_steps
            self.forward_progress_delta_m = 0.0
            self.cumulative_forward_progress_m = 0.0
            return self._result(
                (
                    "replan_safe_direction"
                    if previous_active
                    else "activate_safe_direction"
                ),
                transitioned=not previous_active,
                replan_due=True,
            )

        if (
            previous_active
            and selection_attempted
            and latched_direction_valid
            and self.active_steps < self.config.active_step_limit
        ):
            direction_xy = primitive_direction_xy(self.direction_rc)
            current = np.asarray(position_xy, dtype=np.float32)
            self.waypoint_xy = (
                current
                + (self.config.distance_m / self.gsd) * direction_xy
            )
            self.active = True
            self.active_steps += 1
            self.age_steps = self.active_steps
            self.hold_steps = self.active_steps
            return self._result(
                "hold_last_safe_direction",
                transitioned=False,
            )

        self.active = False
        self.armed = True
        self.active_steps = 0
        self.age_steps = 0
        self.hold_steps = 0
        return self._result(
            (
                "no_safe_improving_candidate"
                if selection_attempted
                else "activation_not_admitted"
            ),
            transitioned=previous_active,
        )


def _stagewise_unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _stagewise_clip_rc(
    point_rc: np.ndarray, shape: Tuple[int, int]
) -> Tuple[int, int]:
    return (
        int(np.clip(round(float(point_rc[0])), 0, shape[0] - 1)),
        int(np.clip(round(float(point_rc[1])), 0, shape[1] - 1)),
    )


def _stagewise_ray_cost(
    maps: Mapping[str, np.ndarray],
    position_rc: np.ndarray,
    direction_rc: np.ndarray,
    *,
    horizon_cells: int,
    hard_margin_m: float,
) -> Tuple[float, bool, float]:
    """Exact local equivalent of the submitted primitive gate's ray cost."""

    risk = np.asarray(maps["risk_map"])
    hard = np.asarray(maps["hard_mask"], dtype=bool)
    sdf = np.asarray(maps["sdf_hard"])
    values = []
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
        cell = _stagewise_clip_rc(query, risk.shape)
        values.append(float(risk[cell]))
        clearance = float(sdf[cell])
        min_clearance = min(min_clearance, clearance)
        if bool(hard[cell]) or clearance < hard_margin_m:
            feasible = False
            break
    mean_risk = float(np.mean(values)) if values else float("inf")
    return mean_risk, feasible, min_clearance


def enumerate_stagewise_primitive_candidates(
    maps: Mapping[str, np.ndarray],
    position_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    primitive_count: int,
    horizon_cells: int,
    hard_margin_m: float,
    require_endpoint_progress: bool = True,
) -> StagewisePrimitiveEnumeration:
    """Enumerate the current gate's complete eligible candidate set.

    Ordering is the original uniform-ray index order. The default preserves
    the legacy endpoint-progress filter exactly. Receding v6 explicitly sets
    ``require_endpoint_progress=False`` and lets its dynamic forecast enforce
    progress instead. No scoring or selection is performed here.
    """

    if primitive_count < 1:
        raise ValueError("primitive_count must be at least one")
    if horizon_cells < 1:
        raise ValueError("horizon_cells must be at least one")
    if hard_margin_m < 0.0:
        raise ValueError("hard_margin_m must be nonnegative")
    required = ("risk_map", "hard_mask", "sdf_hard")
    missing = [key for key in required if key not in maps]
    if missing:
        raise KeyError(f"missing map fields: {missing}")
    shapes = {np.asarray(maps[key]).shape for key in required}
    if len(shapes) != 1:
        raise ValueError("risk, hard, and SDF maps must have the same shape")

    position_rc = np.asarray(position_xy, dtype=np.float32)[::-1]
    goal_rc = np.asarray(goal_xy, dtype=np.float32)[::-1]
    nominal_direction = _stagewise_unit(goal_rc - position_rc)
    nominal_risk, _, _ = _stagewise_ray_cost(
        maps,
        position_rc,
        nominal_direction,
        horizon_cells=horizon_cells,
        hard_margin_m=hard_margin_m,
    )
    nominal_endpoint = (
        position_rc + float(horizon_cells) * nominal_direction
    )

    current_goal_distance = float(np.linalg.norm(goal_rc - position_rc))
    candidates = []
    for index in range(primitive_count):
        angle = 2.0 * math.pi * float(index) / float(primitive_count)
        direction = np.asarray(
            [math.sin(angle), math.cos(angle)], dtype=np.float32
        )
        endpoint = position_rc + float(horizon_cells) * direction
        if require_endpoint_progress and (
            float(np.linalg.norm(goal_rc - endpoint))
            >= current_goal_distance - 0.5
        ):
            continue
        mean_risk, feasible, min_clearance = _stagewise_ray_cost(
            maps,
            position_rc,
            direction,
            horizon_cells=horizon_cells,
            hard_margin_m=hard_margin_m,
        )
        if feasible:
            candidates.append(
                StagewisePrimitiveCandidate(
                    primitive_index=index,
                    direction_rc=(
                        float(direction[0]),
                        float(direction[1]),
                    ),
                    endpoint_rc=(
                        float(endpoint[0]),
                        float(endpoint[1]),
                    ),
                    mean_risk=mean_risk,
                    min_clearance_m=min_clearance,
                )
            )

    return StagewisePrimitiveEnumeration(
        nominal_direction_rc=(
            float(nominal_direction[0]),
            float(nominal_direction[1]),
        ),
        nominal_endpoint_rc=(
            float(nominal_endpoint[0]),
            float(nominal_endpoint[1]),
        ),
        nominal_mean_risk=nominal_risk,
        candidates=tuple(candidates),
    )


def primitive_ray_is_hard_feasible(
    maps: Mapping[str, np.ndarray],
    position_xy: np.ndarray,
    direction_rc: Tuple[float, float],
    *,
    horizon_cells: int,
    hard_margin_m: float,
) -> bool:
    """Recheck only the hard-clearance validity of a latched primitive."""

    if horizon_cells < 1:
        raise ValueError("horizon_cells must be at least one")
    risk_shape = maps["hard_mask"].shape
    position_rc = np.asarray(position_xy, dtype=np.float32)[::-1]
    direction = np.asarray(direction_rc, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return False
    direction = direction / norm
    hard = np.asarray(maps["hard_mask"], dtype=bool)
    sdf = np.asarray(maps["sdf_hard"])
    for distance in range(1, horizon_cells + 1):
        query = position_rc + float(distance) * direction
        if not (
            0.0 <= query[0] < risk_shape[0]
            and 0.0 <= query[1] < risk_shape[1]
        ):
            return False
        cell = tuple(
            np.clip(np.rint(query).astype(int), [0, 0], np.asarray(risk_shape) - 1)
        )
        if bool(hard[cell]) or float(sdf[cell]) < hard_margin_m:
            return False
    return True


@dataclass(frozen=True)
class HysteresisConfig:
    """Thresholds and temporal conditions for the repaired soft-force gate."""

    on_improvement: float = 0.05
    off_improvement: float = 0.025
    on_material_trigger: float = 0.45
    off_material_trigger: float = 0.35
    on_persistence_steps: int = 3
    off_persistence_steps: int = 2
    minimum_dwell_steps: int = 5

    def __post_init__(self) -> None:
        if self.off_improvement > self.on_improvement:
            raise ValueError("off_improvement must not exceed on_improvement")
        if self.off_material_trigger > self.on_material_trigger:
            raise ValueError(
                "off_material_trigger must not exceed on_material_trigger"
            )
        for name in (
            "on_persistence_steps",
            "off_persistence_steps",
            "minimum_dwell_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")


@dataclass(frozen=True)
class GateUpdate:
    active: bool
    transitioned: bool
    reason: str
    on_evidence: bool
    off_evidence: bool
    on_streak: int
    off_streak: int
    dwell_steps: int
    improvement: float


class StatefulFeasibilityGate:
    """Hysteretic feasibility gate with an immediate hard-hazard override."""

    def __init__(self, config: HysteresisConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.on_streak = 0
        self.off_streak = 0
        self.dwell_steps = 0

    def update(
        self,
        *,
        nominal_risk: float,
        best_risk: float,
        feasible_count: int,
        hard_hazard_override: bool,
        magnitude_eligible: bool = True,
    ) -> GateUpdate:
        finite_best = bool(np.isfinite(best_risk))
        improvement = (
            float(nominal_risk - best_risk) if finite_best else float("-inf")
        )
        on_evidence = bool(
            feasible_count > 0
            and finite_best
            and magnitude_eligible
            and nominal_risk >= self.config.on_material_trigger
            and improvement >= self.config.on_improvement
        )
        # Staying active is easier than becoming active.  This is the
        # hysteresis band: the gate turns off only after evidence drops below
        # the lower thresholds (or there is no feasible primitive).
        off_evidence = bool(
            feasible_count <= 0
            or not finite_best
            or not magnitude_eligible
            or nominal_risk < self.config.off_material_trigger
            or improvement < self.config.off_improvement
        )

        previous = self.active
        reason = "hold_active" if previous else "hold_inactive"
        if hard_hazard_override:
            self.active = False
            self.on_streak = 0
            self.off_streak = 0
            self.dwell_steps = 0
            reason = "hard_hazard_override"
        elif not self.active:
            self.off_streak = 0
            self.on_streak = self.on_streak + 1 if on_evidence else 0
            if self.on_streak >= self.config.on_persistence_steps:
                self.active = True
                self.on_streak = 0
                self.dwell_steps = 1
                reason = "activate_after_persistence"
            elif on_evidence:
                reason = "collect_on_evidence"
        else:
            self.dwell_steps += 1
            self.on_streak = 0
            self.off_streak = self.off_streak + 1 if off_evidence else 0
            dwell_satisfied = (
                self.dwell_steps >= self.config.minimum_dwell_steps
            )
            if (
                dwell_satisfied
                and self.off_streak >= self.config.off_persistence_steps
            ):
                self.active = False
                self.off_streak = 0
                self.dwell_steps = 0
                reason = "deactivate_after_hysteresis"
            elif off_evidence and not dwell_satisfied:
                reason = "minimum_dwell"
            elif off_evidence:
                reason = "collect_off_evidence"

        return GateUpdate(
            active=self.active,
            transitioned=self.active != previous,
            reason=reason,
            on_evidence=on_evidence,
            off_evidence=off_evidence,
            on_streak=self.on_streak,
            off_streak=self.off_streak,
            dwell_steps=self.dwell_steps,
            improvement=improvement,
        )


def primitive_direction_xy(direction_rc: Tuple[float, float]) -> np.ndarray:
    """Convert a gate ray from raster ``(row,col)`` to dynamics ``(x,y)``."""

    direction = np.asarray(
        [float(direction_rc[1]), float(direction_rc[0])], dtype=np.float32
    )
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 1e-8 else np.zeros(2, dtype=np.float32)


def velocity_tracking_direction_rc(
    velocity_xy: np.ndarray,
    selected_direction_rc: Tuple[float, float],
) -> Tuple[float, float]:
    """Return the force direction that rotates velocity toward a primitive.

    The desired velocity preserves the current speed and replaces only its
    direction.  The returned vector is the normalized velocity error in raster
    ``(row, col)`` order, suitable for the existing force-alignment helper.
    When the robot is stationary or already aligned, the primitive direction
    is returned directly.
    """

    velocity = np.asarray(velocity_xy, dtype=np.float32)
    if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity_xy must be one finite 2-D vector")
    selected_xy = primitive_direction_xy(selected_direction_rc)
    selected_norm = float(np.linalg.norm(selected_xy))
    if selected_norm <= 1e-8:
        return (0.0, 0.0)
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-8:
        tracking_xy = selected_xy
    else:
        desired_velocity = speed * selected_xy
        error = desired_velocity - velocity
        error_norm = float(np.linalg.norm(error))
        tracking_xy = (
            error / error_norm if error_norm > 1e-8 else selected_xy
        )
    return (float(tracking_xy[1]), float(tracking_xy[0]))


def project_vectors_to_cone(
    vectors: torch.Tensor,
    axis: torch.Tensor,
    half_angle_degrees: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 2-D vectors to the closest ray of a circular cone.

    Args:
        vectors: ``(B,2,H,W)`` vectors.
        axis: ``(B,2)`` unit or non-unit cone axes.
        half_angle_degrees: cone half-angle in ``[0, 180]``.

    Returns:
        Projected vectors and a boolean ``(B,H,W)`` changed mask.  Magnitudes
        are preserved.  Zero vectors and batches with a zero axis are unchanged.
    """

    if vectors.ndim != 4 or vectors.shape[1] != 2:
        raise ValueError("vectors must have shape (B,2,H,W)")
    if axis.shape != (vectors.shape[0], 2):
        raise ValueError("axis must have shape (B,2)")
    if not 0.0 <= half_angle_degrees <= 180.0:
        raise ValueError("half_angle_degrees must be in [0, 180]")

    magnitude = torch.linalg.vector_norm(vectors, dim=1)
    axis_norm = torch.linalg.vector_norm(axis, dim=1)
    axis_unit = axis / axis_norm.clamp_min(eps).unsqueeze(-1)
    vector_unit = vectors / magnitude.clamp_min(eps).unsqueeze(1)
    cosine = (vector_unit * axis_unit[:, :, None, None]).sum(dim=1)
    half_angle = math.radians(float(half_angle_degrees))
    outside = cosine < math.cos(half_angle)
    valid = (magnitude > eps) & (axis_norm[:, None, None] > eps)
    changed = outside & valid

    perpendicular = torch.stack([-axis_unit[:, 1], axis_unit[:, 0]], dim=1)
    cross = (
        axis_unit[:, 0, None, None] * vector_unit[:, 1]
        - axis_unit[:, 1, None, None] * vector_unit[:, 0]
    )
    side = torch.where(cross < 0.0, -torch.ones_like(cross), torch.ones_like(cross))
    boundary = (
        math.cos(half_angle) * axis_unit[:, :, None, None]
        + math.sin(half_angle)
        * side[:, None]
        * perpendicular[:, :, None, None]
    )
    projected = boundary * magnitude[:, None]
    output = torch.where(changed[:, None], projected, vectors)
    return output, changed


@dataclass(frozen=True)
class ProjectionDiagnostics:
    selected_direction_x: float
    selected_direction_y: float
    center_soft_x_before: float
    center_soft_y_before: float
    center_soft_x_after: float
    center_soft_y_after: float
    center_alignment_before: float
    center_alignment_after: float
    projected_pixel_fraction: float
    center_gradient_norm: float
    center_low_confidence_fallback: bool
    low_confidence_pixel_fraction: float
    gradient_confidence_threshold: float
    low_confidence_fallback_policy: str


def align_rollout_soft_force(
    rollout_patch: torch.Tensor,
    direction_rc: Tuple[float, float],
    *,
    half_angle_degrees: float,
    gradient_confidence_threshold: float = 1e-3,
    low_confidence_fallback_policy: str = "selected_axis",
) -> Tuple[torch.Tensor, ProjectionDiagnostics]:
    """Constrain ``-risk_grad`` in a rollout patch to the feasible cone.

    Only channels 2/3 (``dr/dx``, ``dr/dy``) are transformed.  The risk,
    hard-hazard SDF, and true SDF-gradient channels are unchanged, so the
    learned hard force is unaffected.

    The transformed soft-force vectors have unit magnitude. Consequently,
    ``lam_soft`` in the canonical integrator is the actual soft-force
    magnitude, instead of being implicitly attenuated by ``||grad risk||``.
    A confident risk gradient supplies direction and is constrained to the
    feasible cone. At locally flat/uncertain pixels, ``selected_axis`` uses the
    gate-certified primitive direction; ``zero`` disables soft force there.
    """

    if rollout_patch.ndim != 4 or rollout_patch.shape[1] < 6:
        raise ValueError("rollout_patch must have shape (B,>=6,H,W)")
    if rollout_patch.shape[0] != 1:
        raise ValueError("v1 controller currently expects a batch of one")
    if gradient_confidence_threshold < 0.0:
        raise ValueError("gradient_confidence_threshold must be nonnegative")
    if low_confidence_fallback_policy not in {"selected_axis", "zero"}:
        raise ValueError(
            "low_confidence_fallback_policy must be selected_axis or zero"
        )
    direction_xy_np = primitive_direction_xy(direction_rc)
    direction_xy = torch.as_tensor(
        direction_xy_np,
        dtype=rollout_patch.dtype,
        device=rollout_patch.device,
    ).unsqueeze(0)
    soft_before = -rollout_patch[:, 2:4]
    projected, changed = project_vectors_to_cone(
        soft_before, direction_xy, half_angle_degrees
    )
    gradient_norm = torch.linalg.vector_norm(soft_before, dim=1)
    confident = gradient_norm >= float(gradient_confidence_threshold)
    projected_norm = torch.linalg.vector_norm(projected, dim=1)
    normalized = projected / projected_norm.clamp_min(1e-8).unsqueeze(1)
    if low_confidence_fallback_policy == "selected_axis":
        fallback = direction_xy[:, :, None, None].expand_as(normalized)
    else:
        fallback = torch.zeros_like(normalized)
    soft_after = torch.where(confident[:, None], normalized, fallback)
    result = rollout_patch.clone()
    result[:, 2:4] = -soft_after

    row = rollout_patch.shape[-2] // 2
    col = rollout_patch.shape[-1] // 2
    before = soft_before[0, :, row, col]
    after = soft_after[0, :, row, col]
    axis = direction_xy[0]

    def alignment(vector: torch.Tensor) -> float:
        denominator = torch.linalg.vector_norm(vector) * torch.linalg.vector_norm(axis)
        if float(denominator.item()) <= 1e-8:
            return float("nan")
        return float((torch.dot(vector, axis) / denominator).item())

    diagnostics = ProjectionDiagnostics(
        selected_direction_x=float(direction_xy_np[0]),
        selected_direction_y=float(direction_xy_np[1]),
        center_soft_x_before=float(before[0].item()),
        center_soft_y_before=float(before[1].item()),
        center_soft_x_after=float(after[0].item()),
        center_soft_y_after=float(after[1].item()),
        center_alignment_before=alignment(before),
        center_alignment_after=alignment(after),
        projected_pixel_fraction=float(changed.float().mean().item()),
        center_gradient_norm=float(gradient_norm[0, row, col].item()),
        center_low_confidence_fallback=bool(not confident[0, row, col].item()),
        low_confidence_pixel_fraction=float((~confident).float().mean().item()),
        gradient_confidence_threshold=float(gradient_confidence_threshold),
        low_confidence_fallback_policy=low_confidence_fallback_policy,
    )
    return result, diagnostics
