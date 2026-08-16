from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel


MAIN_EVENT_TYPES = (
    "mud_onset",
    "puddle_expansion",
    "corridor_closes",
    "corridor_opens",
    "crossing_obstacle",
    "moving_obstacle_blocks_detour",
    "mud_onset_detour_blocked",
    "delayed_escape_opens",
    "delayed_required_escape",
)


@dataclass(frozen=True)
class DynamicEventSpec:
    event_type: str
    event_step: int
    duration: int
    center_rc: Tuple[float, float]
    detour_rc: Tuple[float, float]
    goal_rc: Tuple[float, float]
    axis_rc: Tuple[float, float]
    radius_cells: float = 8.0
    hard_radius_cells: float = 4.0
    barrier_half_len_cells: float = 9.0
    barrier_half_width_cells: float = 2.0
    risk_value: float = 0.95
    low_risk_value: float = 0.18
    open_delay: int = 10

    def to_dict(self) -> Dict[str, object]:
        return {
            "event_type": self.event_type,
            "event_step": self.event_step,
            "duration": self.duration,
            "center_rc": list(self.center_rc),
            "detour_rc": list(self.detour_rc),
            "goal_rc": list(self.goal_rc),
            "axis_rc": list(self.axis_rc),
            "radius_cells": self.radius_cells,
            "hard_radius_cells": self.hard_radius_cells,
            "barrier_half_len_cells": self.barrier_half_len_cells,
            "barrier_half_width_cells": self.barrier_half_width_cells,
            "risk_value": self.risk_value,
            "low_risk_value": self.low_risk_value,
            "open_delay": self.open_delay,
        }


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.asarray([0.0, 1.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _path_point(path: Sequence[Sequence[int]], frac: float) -> np.ndarray:
    if not path:
        return np.asarray([50.0, 50.0], dtype=np.float32)
    idx = int(np.clip(round(frac * (len(path) - 1)), 0, len(path) - 1))
    return np.asarray(path[idx], dtype=np.float32)


def _path_direction(path: Sequence[Sequence[int]], idx: int) -> np.ndarray:
    if len(path) < 2:
        return np.asarray([0.0, 1.0], dtype=np.float32)
    idx = int(np.clip(idx, 0, len(path) - 2))
    p = np.asarray(path[idx], dtype=np.float32)
    q = np.asarray(path[idx + 1], dtype=np.float32)
    return _unit(q - p)


def _detour_point(
    stage1_path: Sequence[Sequence[int]],
    risk_path: Sequence[Sequence[int]],
    center: np.ndarray,
    direction: np.ndarray,
    shape: Tuple[int, int],
) -> np.ndarray:
    if risk_path:
        risk = np.asarray(risk_path, dtype=np.float32)
        stage = np.asarray(stage1_path, dtype=np.float32) if stage1_path else risk
        # Prefer the point on the risk-aware route that is farthest from the
        # Stage-1 route; this approximates the local escape corridor.
        if len(stage) > 0:
            dists = []
            for p in risk:
                dists.append(float(np.min(np.linalg.norm(stage - p[None, :], axis=1))))
            idx = int(np.argmax(dists))
            if dists[idx] >= 3.0:
                return risk[idx].astype(np.float32)
    perp = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    candidate = center + 12.0 * perp
    candidate[0] = np.clip(candidate[0], 3, shape[0] - 4)
    candidate[1] = np.clip(candidate[1], 3, shape[1] - 4)
    return candidate.astype(np.float32)


def make_event_spec(
    event_type: str,
    stage1_path: Sequence[Sequence[int]],
    risk_path: Sequence[Sequence[int]],
    goal_rc: Sequence[int],
    *,
    event_fraction: float = 0.38,
    duration: int = 80,
) -> DynamicEventSpec:
    if event_type not in MAIN_EVENT_TYPES:
        raise ValueError(f"Unknown RELLIS-Dyn event type: {event_type}")
    if not stage1_path:
        raise ValueError("stage1_path is required to place a dynamic event")
    event_idx = int(np.clip(round(event_fraction * (len(stage1_path) - 1)), 2, len(stage1_path) - 2))
    center = _path_point(stage1_path, event_fraction)
    direction = _path_direction(stage1_path, event_idx)
    detour = _detour_point(stage1_path, risk_path, center, direction, shape=(100, 100))
    return DynamicEventSpec(
        event_type=event_type,
        event_step=max(4, event_idx),
        duration=int(duration),
        center_rc=(float(center[0]), float(center[1])),
        detour_rc=(float(detour[0]), float(detour[1])),
        goal_rc=(float(goal_rc[0]), float(goal_rc[1])),
        axis_rc=(float(direction[0]), float(direction[1])),
    )


def _rr_cc(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    rr, cc = np.indices(shape, dtype=np.float32)
    return rr, cc


def _ellipse_mask(shape: Tuple[int, int], center: np.ndarray, major: float, minor: float, axis: np.ndarray) -> np.ndarray:
    rr, cc = _rr_cc(shape)
    dr = rr - float(center[0])
    dc = cc - float(center[1])
    axis = _unit(axis)
    perp = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    along = dr * axis[0] + dc * axis[1]
    across = dr * perp[0] + dc * perp[1]
    return (along / max(major, 1e-6)) ** 2 + (across / max(minor, 1e-6)) ** 2 <= 1.0


def _circle_mask(shape: Tuple[int, int], center: np.ndarray, radius: float) -> np.ndarray:
    rr, cc = _rr_cc(shape)
    return (rr - float(center[0])) ** 2 + (cc - float(center[1])) ** 2 <= float(radius) ** 2


def _barrier_mask(shape: Tuple[int, int], center: np.ndarray, axis: np.ndarray, half_len: float, half_width: float) -> np.ndarray:
    # The barrier is placed perpendicular to the local route direction so it
    # blocks the corridor without filling the whole patch.
    rr, cc = _rr_cc(shape)
    axis = _unit(axis)
    perp = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    dr = rr - float(center[0])
    dc = cc - float(center[1])
    along_barrier = dr * perp[0] + dc * perp[1]
    across_barrier = dr * axis[0] + dc * axis[1]
    return (np.abs(along_barrier) <= half_len) & (np.abs(across_barrier) <= half_width)


def _moving_center(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return ((1.0 - alpha) * start + alpha * end).astype(np.float32)


def _recompute_fields(maps: Dict[str, np.ndarray], *, resolution: float = 0.5, sigma: float = 0.75) -> Dict[str, np.ndarray]:
    out = dict(maps)
    out["risk_map"] = gaussian_filter(np.clip(out["risk_map"], 0.0, 1.0), sigma=sigma).astype(np.float32)
    hard = out["hard_mask"].astype(bool)
    out["hard_mask"] = hard.astype(np.uint8)
    out["geom_occ"] = hard.astype(np.uint8)
    out["sdf_hard"] = (distance_transform_edt(~hard) * float(resolution)).astype(np.float32)
    out["grad_row"] = (sobel(out["risk_map"], axis=0) / (2.0 * float(resolution))).astype(np.float32)
    out["grad_col"] = (sobel(out["risk_map"], axis=1) / (2.0 * float(resolution))).astype(np.float32)
    out["sdf_grad_row"] = (sobel(out["sdf_hard"], axis=0) / (2.0 * float(resolution))).astype(np.float32)
    out["sdf_grad_col"] = (sobel(out["sdf_hard"], axis=1) / (2.0 * float(resolution))).astype(np.float32)
    return out


def _copy_dynamic_base(maps: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in maps.items():
        out[key] = np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    return out


def apply_dynamic_event(
    maps: Mapping[str, np.ndarray],
    spec: DynamicEventSpec,
    step: int,
    *,
    resolution: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Return a BEV map modified by a nonstationary RELLIS-Dyn event.

    The function is deterministic and stateless: callers may request the map at
    any time index without storing the previous map. This keeps all methods on
    the same observed local patch.
    """

    out = _copy_dynamic_base(maps)
    risk = out["risk_map"].astype(np.float32, copy=True)
    hard = out["hard_mask"].astype(bool, copy=True)
    shape = risk.shape
    t0 = int(spec.event_step)
    dt = int(step) - t0
    active = dt >= 0
    progress = 0.0 if not active else float(np.clip(dt / max(1, spec.duration), 0.0, 1.0))

    center = np.asarray(spec.center_rc, dtype=np.float32)
    detour = np.asarray(spec.detour_rc, dtype=np.float32)
    goal = np.asarray(spec.goal_rc, dtype=np.float32)
    axis = np.asarray(spec.axis_rc, dtype=np.float32)
    perp = np.asarray([-axis[1], axis[0]], dtype=np.float32)

    et = spec.event_type
    if et == "mud_onset":
        if active:
            mask = _ellipse_mask(shape, center, spec.radius_cells * 1.25, spec.radius_cells * 0.7, axis)
            risk[mask] = np.maximum(risk[mask], spec.risk_value)
    elif et == "puddle_expansion":
        if active:
            radius = 2.0 + progress * spec.radius_cells * 1.5
            mask = _circle_mask(shape, center + 2.0 * axis, radius)
            risk[mask] = np.maximum(risk[mask], spec.risk_value)
    elif et == "corridor_closes":
        if active:
            block = _barrier_mask(shape, detour, axis, spec.barrier_half_len_cells, spec.barrier_half_width_cells)
            hard[block] = True
    elif et == "corridor_opens":
        block = _barrier_mask(shape, detour, axis, spec.barrier_half_len_cells, spec.barrier_half_width_cells)
        if not active:
            hard[block] = True
        else:
            hard[block] = False
            risk[block] = np.minimum(risk[block], spec.low_risk_value)
    elif et == "crossing_obstacle":
        if active:
            start = center - 16.0 * perp
            end = center + 16.0 * perp
            moving = _moving_center(start, end, progress)
            hard[_circle_mask(shape, moving, spec.hard_radius_cells)] = True
    elif et == "moving_obstacle_blocks_detour":
        if active:
            start = detour - 14.0 * perp
            end = detour
            moving = _moving_center(start, end, min(1.0, progress * 1.5))
            hard[_circle_mask(shape, moving, spec.hard_radius_cells)] = True
    elif et == "mud_onset_detour_blocked":
        if active:
            mud = _ellipse_mask(shape, center, spec.radius_cells * 1.25, spec.radius_cells * 0.7, axis)
            block = _barrier_mask(shape, detour, axis, spec.barrier_half_len_cells, spec.barrier_half_width_cells)
            risk[mud] = np.maximum(risk[mud], spec.risk_value)
            hard[block] = True
    elif et == "delayed_escape_opens":
        mud = _ellipse_mask(shape, center, spec.radius_cells * 1.25, spec.radius_cells * 0.7, axis)
        block = _barrier_mask(shape, detour, axis, spec.barrier_half_len_cells, spec.barrier_half_width_cells)
        if active:
            risk[mud] = np.maximum(risk[mud], spec.risk_value)
        if int(step) < t0 + int(spec.open_delay):
            hard[block] = True
        else:
            hard[block] = False
            risk[block] = np.minimum(risk[block], spec.low_risk_value)
            # Slightly lower the revealed corridor toward the goal so it is a
            # genuine feasible escape, not merely an obstacle deletion.
            escape = _ellipse_mask(shape, (detour + goal) / 2.0, spec.radius_cells * 1.2, spec.radius_cells * 0.45, _unit(goal - detour))
            risk[escape] = np.minimum(risk[escape], spec.low_risk_value)
    elif et == "delayed_required_escape":
        block = _barrier_mask(shape, detour, axis, spec.barrier_half_len_cells, spec.barrier_half_width_cells)
        escape = _ellipse_mask(shape, (detour + goal) / 2.0, spec.radius_cells * 1.45, spec.radius_cells * 0.55, _unit(goal - detour))
        if int(step) < t0 + int(spec.open_delay):
            # The escape corridor is visible as low-risk terrain but is not
            # feasible yet; selective policies should not pre-activate.
            risk[escape] = np.minimum(risk[escape], spec.low_risk_value)
            hard[block] = True
        else:
            # At the same instant the escape becomes feasible, the nominal
            # scaffold is made unsafe. This makes post-escape activation a
            # required behavior rather than an optional shortcut.
            hard[block] = False
            risk[block] = np.minimum(risk[block], spec.low_risk_value)
            risk[escape] = np.minimum(risk[escape], spec.low_risk_value)
            closure_center = center + float(spec.open_delay + 3) * axis
            mud = _ellipse_mask(shape, closure_center, spec.radius_cells * 1.8, spec.radius_cells * 1.0, axis)
            closure = _barrier_mask(
                shape,
                closure_center,
                axis,
                spec.barrier_half_len_cells * 1.15,
                spec.barrier_half_width_cells * 1.15,
            )
            risk[mud] = np.maximum(risk[mud], spec.risk_value)
            hard[closure] = True
    else:  # pragma: no cover - guarded above.
        raise ValueError(f"Unhandled RELLIS-Dyn event type: {et}")

    out["risk_map"] = risk
    out["hard_mask"] = hard.astype(np.uint8)
    return _recompute_fields(out, resolution=resolution)


def make_event_specs_for_episode(
    event_types: Iterable[str],
    stage1_path: Sequence[Sequence[int]],
    risk_path: Sequence[Sequence[int]],
    goal_rc: Sequence[int],
    *,
    event_fraction: float = 0.38,
    duration: int = 80,
) -> List[DynamicEventSpec]:
    return [
        make_event_spec(
            event_type,
            stage1_path,
            risk_path,
            goal_rc,
            event_fraction=event_fraction,
            duration=duration,
        )
        for event_type in event_types
    ]
