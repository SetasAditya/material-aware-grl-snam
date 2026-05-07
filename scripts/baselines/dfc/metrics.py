from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.baselines.dfc.planners import STEP, path_length_m


@dataclass(frozen=True)
class FailureWeights:
    hard_hits: float = 100.0
    hard_hazard_length: float = 50.0
    risk_exposure: float = 1.0
    barrier_violation: float = 10.0
    path_ratio_excess: float = 25.0
    oscillation: float = 0.5
    catastrophic_failure: float = 250.0
    no_path_penalty: float = 500.0


def _turning_oscillation(path: List[Tuple[int, int]]) -> float:
    if len(path) < 3:
        return 0.0
    headings: List[float] = []
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        headings.append(float(np.arctan2(r1 - r0, c1 - c0)))
    total = 0.0
    for h0, h1 in zip(headings[:-1], headings[1:]):
        d = np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0))
        total += abs(float(d))
    return total


def _empty_metrics(weights: FailureWeights) -> Dict[str, float | Dict[str, float]]:
    return {
        "success": 0.0,
        "catastrophic_failure": 1.0,
        "path_length_m": 0.0,
        "path_length_ratio": float("nan"),
        "excess_length_m": float("nan"),
        "risk_exposure": 0.0,
        "mean_rho": 0.0,
        "max_risk": 0.0,
        "path_cvar_risk": 0.0,
        "soft_risk_violation_length_m": 0.0,
        "hard_hits": 0.0,
        "hard_hazard_length_m": 0.0,
        "max_violation_severity_m": 0.0,
        "barrier_violation_m": 0.0,
        "min_hard_distance_m": 0.0,
        "mean_safety_margin_m": 0.0,
        "low_margin_length_m": 0.0,
        "oscillation": 0.0,
        "curvature_energy": 0.0,
        "backtracking_ratio": 0.0,
        "revisit_count": 0.0,
        "material_exposure_m": {},
        "failure_score": weights.no_path_penalty,
    }


def _weighted_tail_mean(values: List[float], weights: List[float], q: float) -> float:
    if not values or not weights:
        return 0.0
    q = float(np.clip(q, 0.0, 1.0))
    if q <= 0.0:
        return 0.0
    vals = np.asarray(values, dtype=np.float64)
    wts = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    vals = vals[valid]
    wts = wts[valid]
    if vals.size == 0:
        return 0.0
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    wts = wts[order]
    target = q * float(wts.sum())
    kept_v = []
    kept_w = []
    used = 0.0
    for val, wt in zip(vals, wts):
        take = min(float(wt), max(0.0, target - used))
        if take <= 0:
            break
        kept_v.append(float(val))
        kept_w.append(take)
        used += take
    if not kept_w:
        return float(vals[0])
    return float(np.average(np.asarray(kept_v), weights=np.asarray(kept_w)))


def _heading_stats(headings: List[float]) -> Tuple[float, float]:
    oscillation = 0.0
    curvature_energy = 0.0
    for h0, h1 in zip(headings[:-1], headings[1:]):
        d = np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0))
        oscillation += abs(float(d))
        curvature_energy += float(d) ** 2
    return float(oscillation), float(curvature_energy)


def _score(
    *,
    weights: FailureWeights,
    hard_hits: int,
    hard_hazard_length_m: float,
    risk_exposure: float,
    barrier_violation: float,
    ratio_excess: float,
    oscillation: float,
    catastrophic_failure: float,
) -> float:
    return float(
        weights.hard_hits * hard_hits
        + weights.hard_hazard_length * hard_hazard_length_m
        + weights.risk_exposure * risk_exposure
        + weights.barrier_violation * barrier_violation
        + weights.path_ratio_excess * ratio_excess
        + weights.oscillation * oscillation
        + weights.catastrophic_failure * catastrophic_failure
    )


def cumulative_risk_curve(
    path: Optional[List[Tuple[int, int]]],
    maps: Dict[str, np.ndarray],
    *,
    gsd: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if not path or len(path) < 2:
        return np.array([]), np.array([])
    risk = maps["risk_map"]
    ds = [0.0]
    cr = [0.0]
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        step = gsd * STEP[(r1 - r0, c1 - c0)]
        ds.append(ds[-1] + step)
        cr.append(cr[-1] + step * float(risk[r1, c1]))
    return np.asarray(ds, dtype=np.float32), np.asarray(cr, dtype=np.float32)


def cumulative_risk_curve_trace(
    trace_rc: Optional[Sequence[Sequence[float]]],
    maps: Dict[str, np.ndarray],
    *,
    gsd: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if trace_rc is None or len(trace_rc) < 2:
        return np.array([]), np.array([])
    risk = maps["risk_map"]
    h_rows, h_cols = risk.shape
    ds = [0.0]
    cr = [0.0]
    for p0, p1 in zip(trace_rc[:-1], trace_rc[1:]):
        r0, c0 = float(p0[0]), float(p0[1])
        r1, c1 = float(p1[0]), float(p1[1])
        step = gsd * float(np.linalg.norm([r1 - r0, c1 - c0]))
        mr = int(np.clip((r0 + r1) / 2.0, 0, h_rows - 1))
        mc = int(np.clip((c0 + c1) / 2.0, 0, h_cols - 1))
        ds.append(ds[-1] + step)
        cr.append(cr[-1] + step * float(risk[mr, mc]))
    return np.asarray(ds, dtype=np.float32), np.asarray(cr, dtype=np.float32)


def compute_path_metrics(
    path: Optional[List[Tuple[int, int]]],
    maps: Dict[str, np.ndarray],
    *,
    reference_length_m: Optional[float],
    gsd: float,
    weights: FailureWeights,
    hazard_margin_m: float = 0.5,
    low_margin_m: float = 1.0,
    soft_risk_threshold: float = 0.25,
    cvar_top_q: float = 0.1,
    goal_rc: Optional[Tuple[int, int]] = None,
    goal_tolerance_m: float = 3.0,
) -> Dict[str, float | Dict[str, float]]:
    if not path:
        return _empty_metrics(weights)

    risk = maps["risk_map"]
    hard = maps["hard_mask"].astype(bool)
    sdf = maps["sdf_hard"]
    labels = maps["z2_labels"]

    length_m = 0.0
    risk_exposure = 0.0
    hard_hits = 0
    hard_hazard_length_m = 0.0
    barrier_violation = 0.0
    max_violation = 0.0
    min_hard_distance = float("inf")
    weighted_margin = 0.0
    low_margin_length_m = 0.0
    soft_risk_violation_length_m = 0.0
    risk_values: List[float] = []
    risk_weights: List[float] = []
    material_exposure: Dict[str, float] = {}
    goal = np.asarray(goal_rc, dtype=np.float64) if goal_rc is not None else None
    backtracking_m = 0.0
    visited = set()
    revisit_count = 0
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        step = gsd * STEP[(r1 - r0, c1 - c0)]
        length_m += step
        rho = float(risk[r1, c1])
        risk_exposure += step * rho
        risk_values.append(rho)
        risk_weights.append(step)
        if hard[r1, c1]:
            hard_hits += 1
            hard_hazard_length_m += step
        dist = float(sdf[r1, c1])
        violation = max(0.0, hazard_margin_m - dist)
        barrier_violation += step * violation
        max_violation = max(max_violation, violation)
        min_hard_distance = min(min_hard_distance, dist)
        weighted_margin += step * dist
        if dist < low_margin_m:
            low_margin_length_m += step
        if rho > soft_risk_threshold:
            soft_risk_violation_length_m += step
        label = str(int(labels[r1, c1]))
        material_exposure[label] = material_exposure.get(label, 0.0) + step
        if goal is not None:
            d0 = np.linalg.norm(np.asarray([r0, c0], dtype=np.float64) - goal)
            d1 = np.linalg.norm(np.asarray([r1, c1], dtype=np.float64) - goal)
            if d1 > d0:
                backtracking_m += step
        cell = (int(r1), int(c1))
        if cell in visited:
            revisit_count += 1
        visited.add(cell)

    mean_rho = risk_exposure / max(length_m, 1e-6)
    headings = [
        float(np.arctan2(r1 - r0, c1 - c0))
        for (r0, c0), (r1, c1) in zip(path[:-1], path[1:])
    ]
    oscillation, curvature_energy = _heading_stats(headings)
    catastrophic_failure = 1.0 if hard_hits > 0 else 0.0
    max_risk = max(risk_values) if risk_values else 0.0
    path_cvar_risk = _weighted_tail_mean(risk_values, risk_weights, cvar_top_q)
    mean_safety_margin = weighted_margin / max(length_m, 1e-6)
    backtracking_ratio = backtracking_m / max(length_m, 1e-6)
    if goal_rc is None:
        success = 1.0
    else:
        final_dist_m = gsd * float(np.linalg.norm(np.asarray(path[-1], dtype=np.float64) - goal))
        success = 1.0 if final_dist_m <= goal_tolerance_m else 0.0

    ratio = float("nan")
    excess_length_m = float("nan")
    ratio_excess = 0.0
    if reference_length_m is not None and reference_length_m > 0:
        ratio = length_m / reference_length_m
        excess_length_m = length_m - reference_length_m
        ratio_excess = max(0.0, ratio - 1.0)

    score = _score(
        weights=weights,
        hard_hits=hard_hits,
        hard_hazard_length_m=hard_hazard_length_m,
        risk_exposure=risk_exposure,
        barrier_violation=barrier_violation,
        ratio_excess=ratio_excess,
        oscillation=oscillation,
        catastrophic_failure=catastrophic_failure,
    )
    return {
        "success": success,
        "catastrophic_failure": catastrophic_failure,
        "path_length_m": float(length_m),
        "path_length_ratio": float(ratio),
        "excess_length_m": float(excess_length_m),
        "risk_exposure": float(risk_exposure),
        "mean_rho": float(mean_rho),
        "max_risk": float(max_risk),
        "path_cvar_risk": float(path_cvar_risk),
        "soft_risk_violation_length_m": float(soft_risk_violation_length_m),
        "hard_hits": float(hard_hits),
        "hard_hazard_length_m": float(hard_hazard_length_m),
        "max_violation_severity_m": float(max_violation),
        "barrier_violation_m": float(barrier_violation),
        "min_hard_distance_m": float(0.0 if not np.isfinite(min_hard_distance) else min_hard_distance),
        "mean_safety_margin_m": float(mean_safety_margin),
        "low_margin_length_m": float(low_margin_length_m),
        "oscillation": float(oscillation),
        "curvature_energy": float(curvature_energy),
        "backtracking_ratio": float(backtracking_ratio),
        "revisit_count": float(revisit_count),
        "material_exposure_m": {k: float(v) for k, v in sorted(material_exposure.items(), key=lambda kv: int(kv[0]))},
        "failure_score": float(score),
    }


def compute_trace_metrics(
    trace_rc: Optional[Sequence[Sequence[float]]],
    maps: Dict[str, np.ndarray],
    *,
    reference_length_m: Optional[float],
    gsd: float,
    weights: FailureWeights,
    hazard_margin_m: float = 0.5,
    low_margin_m: float = 1.0,
    soft_risk_threshold: float = 0.25,
    cvar_top_q: float = 0.1,
    goal_rc: Optional[Tuple[int, int]] = None,
    goal_tolerance_m: float = 3.0,
) -> Dict[str, float | Dict[str, float]]:
    if trace_rc is None or len(trace_rc) < 2:
        return _empty_metrics(weights)

    risk = maps["risk_map"]
    hard = maps["hard_mask"].astype(bool)
    sdf = maps["sdf_hard"]
    labels = maps["z2_labels"]
    h_rows, h_cols = risk.shape

    length_m = 0.0
    risk_exposure = 0.0
    hard_hits = 0
    hard_hazard_length_m = 0.0
    barrier_violation = 0.0
    max_violation = 0.0
    min_hard_distance = float("inf")
    weighted_margin = 0.0
    low_margin_length_m = 0.0
    soft_risk_violation_length_m = 0.0
    headings: List[float] = []
    risk_values: List[float] = []
    risk_weights: List[float] = []
    material_exposure: Dict[str, float] = {}
    goal = np.asarray(goal_rc, dtype=np.float64) if goal_rc is not None else None
    backtracking_m = 0.0
    visited = set()
    revisit_count = 0
    for p0, p1 in zip(trace_rc[:-1], trace_rc[1:]):
        r0, c0 = float(p0[0]), float(p0[1])
        r1, c1 = float(p1[0]), float(p1[1])
        step = gsd * float(np.linalg.norm([r1 - r0, c1 - c0]))
        if step <= 0:
            continue
        length_m += step
        mr = int(np.clip((r0 + r1) / 2.0, 0, h_rows - 1))
        mc = int(np.clip((c0 + c1) / 2.0, 0, h_cols - 1))
        rho = float(risk[mr, mc])
        risk_exposure += step * rho
        risk_values.append(rho)
        risk_weights.append(step)
        if hard[mr, mc]:
            hard_hits += 1
            hard_hazard_length_m += step
        dist = float(sdf[mr, mc])
        violation = max(0.0, hazard_margin_m - dist)
        barrier_violation += step * violation
        max_violation = max(max_violation, violation)
        min_hard_distance = min(min_hard_distance, dist)
        weighted_margin += step * dist
        if dist < low_margin_m:
            low_margin_length_m += step
        if rho > soft_risk_threshold:
            soft_risk_violation_length_m += step
        label = str(int(labels[mr, mc]))
        material_exposure[label] = material_exposure.get(label, 0.0) + step
        if goal is not None:
            d0 = np.linalg.norm(np.asarray([r0, c0], dtype=np.float64) - goal)
            d1 = np.linalg.norm(np.asarray([r1, c1], dtype=np.float64) - goal)
            if d1 > d0:
                backtracking_m += step
        cell = (mr, mc)
        if cell in visited:
            revisit_count += 1
        visited.add(cell)
        headings.append(float(np.arctan2(r1 - r0, c1 - c0)))

    oscillation, curvature_energy = _heading_stats(headings)

    mean_rho = risk_exposure / max(length_m, 1e-6)
    catastrophic_failure = 1.0 if hard_hits > 0 else 0.0
    max_risk = max(risk_values) if risk_values else 0.0
    path_cvar_risk = _weighted_tail_mean(risk_values, risk_weights, cvar_top_q)
    mean_safety_margin = weighted_margin / max(length_m, 1e-6)
    backtracking_ratio = backtracking_m / max(length_m, 1e-6)
    if goal_rc is None:
        success = 1.0
    else:
        final = np.asarray(trace_rc[-1], dtype=np.float64)
        final_dist_m = gsd * float(np.linalg.norm(final - goal))
        success = 1.0 if final_dist_m <= goal_tolerance_m else 0.0

    ratio = float("nan")
    excess_length_m = float("nan")
    ratio_excess = 0.0
    if reference_length_m is not None and reference_length_m > 0:
        ratio = length_m / reference_length_m
        excess_length_m = length_m - reference_length_m
        ratio_excess = max(0.0, ratio - 1.0)
    score = _score(
        weights=weights,
        hard_hits=hard_hits,
        hard_hazard_length_m=hard_hazard_length_m,
        risk_exposure=risk_exposure,
        barrier_violation=barrier_violation,
        ratio_excess=ratio_excess,
        oscillation=oscillation,
        catastrophic_failure=catastrophic_failure,
    )
    return {
        "success": success,
        "catastrophic_failure": catastrophic_failure,
        "path_length_m": float(length_m),
        "path_length_ratio": float(ratio),
        "excess_length_m": float(excess_length_m),
        "risk_exposure": float(risk_exposure),
        "mean_rho": float(mean_rho),
        "max_risk": float(max_risk),
        "path_cvar_risk": float(path_cvar_risk),
        "soft_risk_violation_length_m": float(soft_risk_violation_length_m),
        "hard_hits": float(hard_hits),
        "hard_hazard_length_m": float(hard_hazard_length_m),
        "max_violation_severity_m": float(max_violation),
        "barrier_violation_m": float(barrier_violation),
        "min_hard_distance_m": float(0.0 if not np.isfinite(min_hard_distance) else min_hard_distance),
        "mean_safety_margin_m": float(mean_safety_margin),
        "low_margin_length_m": float(low_margin_length_m),
        "oscillation": float(oscillation),
        "curvature_energy": float(curvature_energy),
        "backtracking_ratio": float(backtracking_ratio),
        "revisit_count": float(revisit_count),
        "material_exposure_m": {k: float(v) for k, v in sorted(material_exposure.items(), key=lambda kv: int(kv[0]))},
        "failure_score": float(score),
    }
