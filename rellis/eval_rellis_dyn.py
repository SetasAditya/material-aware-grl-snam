#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
from grl_rellis.dyn_events import MAIN_EVENT_TYPES, DynamicEventSpec, apply_dynamic_event, make_event_spec
from scripts.baselines.dfc.planners import STEP, chance_constrained_mpc, path_length_m, risk_weighted_astar


DIRS_8 = np.asarray(
    [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ],
    dtype=np.int32,
)


METHODS = (
    "stage1",
    "risk_loss_only",
    "blackbox_cvar",
    "stage2_expected_cost",
    "fixed_coeff_stage2",
    "scalar_stage2",
    "non_route_directional_stage2",
    "neural_potential_field",
    "dwa_semantic",
    "cbf_safety_filter",
    "route_aware_stage2",
    "local_astar_budget",
    "mpc_budget",
    "oracle_replanner",
)


def _load_scene(bev_root: Path, rel_path: str) -> Dict:
    return torch.load(bev_root / rel_path, map_location="cpu", weights_only=False)


def _as_path(raw: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in raw]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def _clip_cell(p: Sequence[float], shape: Tuple[int, int]) -> Tuple[int, int]:
    return (
        int(np.clip(round(float(p[0])), 0, shape[0] - 1)),
        int(np.clip(round(float(p[1])), 0, shape[1] - 1)),
    )


def _nearest_path_index(path: Sequence[Tuple[int, int]], cur: Tuple[int, int], last_idx: int) -> int:
    if not path:
        return 0
    lo = max(0, last_idx - 8)
    hi = len(path)
    pts = np.asarray(path[lo:hi], dtype=np.float32)
    d = np.linalg.norm(pts - np.asarray(cur, dtype=np.float32)[None, :], axis=1)
    return lo + int(np.argmin(d))


def _path_follower_next(path: Sequence[Tuple[int, int]], cur: Tuple[int, int], last_idx: int) -> Tuple[Tuple[int, int], int]:
    if not path:
        return cur, last_idx
    idx = _nearest_path_index(path, cur, last_idx)
    nxt_idx = min(idx + 1, len(path) - 1)
    return path[nxt_idx], nxt_idx


def _one_cell_toward(cur: Tuple[int, int], target: Tuple[int, int], shape: Tuple[int, int]) -> Tuple[int, int]:
    d = np.asarray(target, dtype=np.float32) - np.asarray(cur, dtype=np.float32)
    return _step_from_direction(cur, d, shape)


def _in_bounds(p: Tuple[int, int], shape: Tuple[int, int]) -> bool:
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _goal_dist(p: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return float(np.linalg.norm(np.asarray(p, dtype=np.float32) - np.asarray(goal, dtype=np.float32)))


def _direction_integral(
    maps: Mapping[str, np.ndarray],
    cur: Tuple[int, int],
    d: np.ndarray,
    *,
    horizon: int,
    hard_margin_m: float,
) -> Tuple[float, bool]:
    risk = maps["risk_map"]
    hard = maps["hard_mask"].astype(bool)
    sdf = maps["sdf_hard"]
    total = 0.0
    feasible = True
    p = np.asarray(cur, dtype=np.float32)
    for step in range(1, horizon + 1):
        q = p + float(step) * d
        r, c = _clip_cell(q, risk.shape)
        if not _in_bounds((r, c), risk.shape):
            feasible = False
            break
        total += float(risk[r, c])
        if hard[r, c] or float(sdf[r, c]) < hard_margin_m:
            feasible = False
            break
    return total, feasible


def _route_segment_eval(
    maps: Mapping[str, np.ndarray],
    path: Sequence[Tuple[int, int]],
    cur: Tuple[int, int],
    last_idx: int,
    *,
    horizon: int,
    hard_margin_m: float,
) -> Tuple[Tuple[int, int], int, float, bool]:
    if not path:
        return cur, last_idx, float("inf"), False
    idx = _nearest_path_index(path, cur, last_idx)
    nxt_idx = min(idx + 1, len(path) - 1)
    hi = min(len(path), idx + horizon + 1)
    risk = maps["risk_map"]
    hard = maps["hard_mask"].astype(bool)
    sdf = maps["sdf_hard"]
    total = 0.0
    feasible = True
    for p in path[nxt_idx:hi]:
        if not _in_bounds(p, risk.shape):
            feasible = False
            continue
        total += float(risk[p])
        feasible = feasible and not bool(hard[p]) and float(sdf[p]) >= 0.5 * hard_margin_m
    return path[nxt_idx], nxt_idx, total, feasible


def _step_from_direction(cur: Tuple[int, int], d: np.ndarray, shape: Tuple[int, int]) -> Tuple[int, int]:
    if float(np.linalg.norm(d)) < 1e-8:
        return cur
    step = np.sign(d).astype(np.int32)
    if step[0] == 0 and abs(d[0]) > 0.35:
        step[0] = 1 if d[0] > 0 else -1
    if step[1] == 0 and abs(d[1]) > 0.35:
        step[1] = 1 if d[1] > 0 else -1
    if step[0] == 0 and step[1] == 0:
        axis = int(np.argmax(np.abs(d)))
        step[axis] = 1 if d[axis] > 0 else -1
    nxt = (int(cur[0] + step[0]), int(cur[1] + step[1]))
    return _clip_cell(nxt, shape)


def _greedy_scalar_step(
    maps: Mapping[str, np.ndarray],
    cur: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    hard_margin_m: float,
) -> Tuple[int, int]:
    risk = maps["risk_map"]
    sdf = maps["sdf_hard"]
    hard = maps["hard_mask"].astype(bool)
    best = cur
    best_score = float("inf")
    for dr, dc in DIRS_8:
        nxt = (cur[0] + int(dr), cur[1] + int(dc))
        if not _in_bounds(nxt, risk.shape):
            continue
        step = STEP.get((int(dr), int(dc)), 1.0)
        penalty = 12.0 * float(hard[nxt]) + 5.0 * max(0.0, hard_margin_m - float(sdf[nxt]))
        score = _goal_dist(nxt, goal) + 12.0 * float(risk[nxt]) + penalty + 0.05 * step
        if score < best_score:
            best_score = score
            best = nxt
    return best


def _candidate_step(
    maps: Mapping[str, np.ndarray],
    cur: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    previous: Optional[Tuple[int, int]],
    risk_weight: float,
    clearance_weight: float,
    heading_weight: float,
    hard_margin_m: float,
) -> Tuple[int, int]:
    risk = maps["risk_map"]
    sdf = maps["sdf_hard"]
    hard = maps["hard_mask"].astype(bool)
    prev_dir = np.zeros(2, dtype=np.float32)
    if previous is not None:
        prev_dir = _unit(np.asarray(cur, dtype=np.float32) - np.asarray(previous, dtype=np.float32))
    best = cur
    best_score = float("inf")
    for dr, dc in DIRS_8:
        nxt = (cur[0] + int(dr), cur[1] + int(dc))
        if not _in_bounds(nxt, risk.shape):
            continue
        move = _unit(np.asarray([dr, dc], dtype=np.float32))
        clearance = max(0.0, hard_margin_m - float(sdf[nxt]))
        turn = 0.0 if previous is None else 1.0 - float(np.dot(prev_dir, move))
        score = (
            _goal_dist(nxt, goal)
            + risk_weight * float(risk[nxt])
            + clearance_weight * clearance
            + 100.0 * float(hard[nxt])
            + heading_weight * turn
        )
        if score < best_score:
            best_score = score
            best = nxt
    return best


def _non_route_directional_step(
    maps: Mapping[str, np.ndarray],
    cur: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    horizon: int,
    hard_margin_m: float,
    improvement_margin: float,
) -> Tuple[int, int]:
    scaffold = _unit(np.asarray(goal, dtype=np.float32) - np.asarray(cur, dtype=np.float32))
    scaffold_risk, scaffold_feasible = _direction_integral(
        maps, cur, scaffold, horizon=horizon, hard_margin_m=min(0.25, hard_margin_m)
    )
    best_dir: Optional[np.ndarray] = None
    best_risk = float("inf")
    for raw in DIRS_8.astype(np.float32):
        d = _unit(raw)
        risk, feasible = _direction_integral(
            maps, cur, d, horizon=horizon, hard_margin_m=min(0.25, hard_margin_m)
        )
        progress = float(np.dot(d, scaffold))
        if feasible and progress > -0.2 and risk < best_risk:
            best_risk = risk
            best_dir = d
    if best_dir is not None and (not scaffold_feasible or (scaffold_risk - best_risk) >= improvement_margin):
        return _step_from_direction(cur, best_dir, maps["risk_map"].shape)
    candidate = _step_from_direction(cur, scaffold, maps["risk_map"].shape)
    if not maps["hard_mask"][candidate]:
        return candidate
    return _candidate_step(
        maps,
        cur,
        goal,
        previous=None,
        risk_weight=2.0,
        clearance_weight=4.0,
        heading_weight=0.0,
        hard_margin_m=hard_margin_m,
    )


def _potential_field_step(
    maps: Mapping[str, np.ndarray],
    cur: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    hard_margin_m: float,
) -> Tuple[int, int]:
    # Unstructured learned-potential proxy: descend a local scalar energy that
    # mixes goal attraction, semantic risk, and barrier potential, but has no
    # route feasibility latch. This keeps it credible without giving it the
    # route-aware mechanism.
    risk = maps["risk_map"]
    sdf = maps["sdf_hard"]
    hard = maps["hard_mask"].astype(bool)
    if (
        _in_bounds(goal, risk.shape)
        and not hard[goal]
        and float(sdf[goal]) >= 0.5 * hard_margin_m
        and float(risk[goal]) < 0.72
    ):
        return goal
    best = cur
    best_score = float("inf")
    cur_goal_dist = _goal_dist(cur, goal)
    for dr, dc in DIRS_8:
        nxt = (cur[0] + int(dr), cur[1] + int(dc))
        if not _in_bounds(nxt, risk.shape) or hard[nxt]:
            continue
        clearance = max(0.0, hard_margin_m - float(sdf[nxt]))
        progress_bonus = max(0.0, cur_goal_dist - _goal_dist(nxt, goal))
        score = (
            2.5 * _goal_dist(nxt, goal)
            + 2.0 * float(risk[nxt])
            + 6.0 * clearance
            - 0.8 * progress_bonus
        )
        if score < best_score:
            best_score = score
            best = nxt
    return best


def _blackbox_cvar_step(
    maps: Mapping[str, np.ndarray],
    cur: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    previous: Optional[Tuple[int, int]],
    hard_margin_m: float,
    horizon: int,
) -> Tuple[int, int]:
    # A black-box local-policy proxy: it scores local actions from the same BEV
    # risk/SDF patch and goal direction, but it has no Hamiltonian force channel
    # and no route-aware latch. The tail term mimics CVaR-style training by
    # penalizing the worst risk/hard samples along each candidate direction.
    risk = maps["risk_map"]
    sdf = maps["sdf_hard"]
    hard = maps["hard_mask"].astype(bool)
    prev_dir = np.zeros(2, dtype=np.float32)
    if previous is not None:
        prev_dir = _unit(np.asarray(cur, dtype=np.float32) - np.asarray(previous, dtype=np.float32))
    goal_vec = _unit(np.asarray(goal, dtype=np.float32) - np.asarray(cur, dtype=np.float32))
    best = cur
    best_score = float("inf")
    for dr, dc in DIRS_8:
        d = _unit(np.asarray([dr, dc], dtype=np.float32))
        nxt = (cur[0] + int(dr), cur[1] + int(dc))
        if not _in_bounds(nxt, risk.shape):
            continue
        samples: List[float] = []
        hard_seen = 0.0
        p = np.asarray(cur, dtype=np.float32)
        for step in range(1, horizon + 1):
            q = p + float(step) * d
            rr, cc = _clip_cell(q, risk.shape)
            samples.append(float(risk[rr, cc]))
            hard_seen = max(hard_seen, float(hard[rr, cc]))
        tail = float(np.mean(sorted(samples, reverse=True)[: max(1, horizon // 4)]))
        clearance = max(0.0, hard_margin_m - float(sdf[nxt]))
        turn = 0.0 if previous is None else 1.0 - float(np.dot(prev_dir, d))
        progress = float(np.dot(d, goal_vec))
        score = (
            _goal_dist(nxt, goal)
            + 4.0 * float(risk[nxt])
            + 6.0 * tail
            + 10.0 * clearance
            + 75.0 * hard_seen
            + 0.7 * turn
            - 1.2 * progress
        )
        if score < best_score:
            best_score = score
            best = nxt
    return best


def _cbf_filter_step(
    maps: Mapping[str, np.ndarray],
    nominal: Tuple[int, int],
    cur: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    hard_margin_m: float,
) -> Tuple[int, int]:
    if (
        _in_bounds(nominal, maps["risk_map"].shape)
        and not maps["hard_mask"][nominal]
        and float(maps["sdf_hard"][nominal]) >= 0.5 * hard_margin_m
    ):
        return nominal
    return _candidate_step(
        maps,
        cur,
        goal,
        previous=None,
        risk_weight=0.5,
        clearance_weight=8.0,
        heading_weight=0.0,
        hard_margin_m=hard_margin_m,
    )


def _local_step_toward_target(
    maps: Mapping[str, np.ndarray],
    cur: Tuple[int, int],
    target: Tuple[int, int],
    *,
    hard_margin_m: float,
) -> Tuple[int, int]:
    risk = maps["risk_map"]
    sdf = maps["sdf_hard"]
    hard = maps["hard_mask"].astype(bool)
    if cur == target:
        return cur
    margin = 14
    r0 = max(0, min(cur[0], target[0]) - margin)
    r1 = min(risk.shape[0] - 1, max(cur[0], target[0]) + margin)
    c0 = max(0, min(cur[1], target[1]) - margin)
    c1 = min(risk.shape[1] - 1, max(cur[1], target[1]) + margin)
    heap: List[Tuple[float, float, Tuple[int, int]]] = []
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    best_g: Dict[Tuple[int, int], float] = {cur: 0.0}
    heapq.heappush(heap, (_goal_dist(cur, target), 0.0, cur))
    expanded = 0
    while heap and expanded < 700:
        _, g, node = heapq.heappop(heap)
        if g > best_g.get(node, float("inf")) + 1e-8:
            continue
        expanded += 1
        if node == target or _goal_dist(node, target) <= 1.0:
            step = node
            while parent.get(step) is not None and parent[step] != cur:
                step = parent[step]
            return step
        for dr, dc in DIRS_8:
            nxt = (node[0] + int(dr), node[1] + int(dc))
            if not (r0 <= nxt[0] <= r1 and c0 <= nxt[1] <= c1):
                continue
            if hard[nxt]:
                continue
            step_len = STEP.get((int(dr), int(dc)), 1.0)
            clearance = max(0.0, 0.5 * hard_margin_m - float(sdf[nxt]))
            ng = g + step_len + 0.8 * float(risk[nxt]) + 2.0 * clearance
            if ng < best_g.get(nxt, float("inf")):
                best_g[nxt] = ng
                parent[nxt] = node
                h = _goal_dist(nxt, target)
                heapq.heappush(heap, (ng + h, ng, nxt))

    best = cur
    best_score = float("inf")
    for dr, dc in DIRS_8:
        nxt = (cur[0] + int(dr), cur[1] + int(dc))
        if not _in_bounds(nxt, risk.shape) or hard[nxt]:
            continue
        clearance = max(0.0, 0.5 * hard_margin_m - float(sdf[nxt]))
        score = _goal_dist(nxt, target) + 1.5 * float(risk[nxt]) + 4.0 * clearance
        if score < best_score:
            best_score = score
            best = nxt
    return best


def _route_aware_step(
    maps: Mapping[str, np.ndarray],
    cur: Tuple[int, int],
    goal: Tuple[int, int],
    stage1_path: Sequence[Tuple[int, int]],
    risk_path: Sequence[Tuple[int, int]],
    idx_stage1: int,
    idx_risk: int,
    detour_cooldown: int,
    *,
    horizon: int,
    hard_margin_m: float,
    improvement_margin: float,
) -> Tuple[Tuple[int, int], int, int, int]:
    shape = maps["risk_map"].shape
    stage_target, next_stage_idx, stage_risk, stage_feasible = _route_segment_eval(
        maps, stage1_path, cur, idx_stage1, horizon=horizon, hard_margin_m=hard_margin_m
    )
    risk_target, next_risk_idx, detour_risk, detour_feasible = _route_segment_eval(
        maps, risk_path, cur, idx_risk, horizon=horizon, hard_margin_m=hard_margin_m
    )

    stage_next = _one_cell_toward(cur, stage_target, shape)
    risk_next = _one_cell_toward(cur, risk_target, shape)
    if not _in_bounds(stage_next, shape):
        stage_next = cur
    if not _in_bounds(risk_next, shape):
        risk_next = cur

    stage_avg_risk = stage_risk / max(1, horizon)
    material_trigger = stage_avg_risk >= 0.45
    risk_improves = (stage_risk - detour_risk) >= improvement_margin
    next_cooldown = max(0, detour_cooldown - 1)
    detour_progress_ok = _goal_dist(risk_target, goal) <= _goal_dist(cur, goal) - 0.5

    if idx_risk > 0 and detour_feasible and not maps["hard_mask"][risk_next]:
        return risk_next, idx_stage1, next_risk_idx, next_cooldown
    blocked_active_detour = idx_risk > 0 and not detour_feasible
    if (
        detour_cooldown <= 0
        and detour_feasible
        and detour_progress_ok
        and (material_trigger or not stage_feasible)
        and (risk_improves or not stage_feasible)
    ):
        if not maps["hard_mask"][risk_next]:
            return risk_next, idx_stage1, next_risk_idx, next_cooldown
        return (
            _local_step_toward_target(maps, cur, risk_target, hard_margin_m=hard_margin_m),
            idx_stage1,
            next_risk_idx,
            next_cooldown,
        )
    if stage_feasible:
        if not maps["hard_mask"][stage_next]:
            return stage_next, next_stage_idx, 0, 8 if blocked_active_detour else next_cooldown
        return (
            _local_step_toward_target(maps, cur, stage_target, hard_margin_m=hard_margin_m),
            next_stage_idx,
            0,
            8 if blocked_active_detour else next_cooldown,
        )

    fallback = _local_step_toward_target(maps, cur, goal, hard_margin_m=hard_margin_m)
    if fallback == cur:
        fallback = _greedy_scalar_step(maps, cur, goal, hard_margin_m=hard_margin_m)
    return fallback, idx_stage1, 0, 8 if blocked_active_detour else next_cooldown


def _planner_next(
    maps: Mapping[str, np.ndarray],
    cur: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    planner: str,
    cached_plan: Optional[List[Tuple[int, int]]],
    plan_idx: int,
    risk_weight: float,
) -> Tuple[Tuple[int, int], Optional[List[Tuple[int, int]]], int, bool, float]:
    start_time = time.perf_counter()
    maps_dict = dict(maps)
    if planner == "astar":
        plan = risk_weighted_astar(maps_dict, cur, goal, risk_weight=risk_weight)
    elif planner == "mpc":
        plan = chance_constrained_mpc(maps_dict, cur, goal, risk_weight=max(2.0, 0.5 * risk_weight), horizon=10, beam_width=16)
    else:
        raise ValueError(planner)
    elapsed = time.perf_counter() - start_time
    if not plan or len(plan) < 2:
        plan = cached_plan
    if not plan or len(plan) < 2:
        return _greedy_scalar_step(maps, cur, goal, hard_margin_m=0.5), plan, 0, True, elapsed
    return plan[1], plan, 1, True, elapsed


def _risk_tail(values: List[float], weights: List[float], q: float = 0.2) -> float:
    if not values:
        return 0.0
    vals = np.asarray(values, dtype=np.float64)
    wts = np.asarray(weights, dtype=np.float64)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    wts = wts[order]
    target = q * max(float(wts.sum()), 1e-8)
    used = 0.0
    kept_v: List[float] = []
    kept_w: List[float] = []
    for v, w in zip(vals, wts):
        take = min(float(w), target - used)
        if take <= 0:
            break
        kept_v.append(float(v))
        kept_w.append(take)
        used += take
    if not kept_w:
        return float(vals[0])
    return float(np.average(kept_v, weights=kept_w))


def _nearest_distance_to_path(path: Sequence[Tuple[int, int]], point: Tuple[int, int], lo: int = 0) -> float:
    if not path:
        return float("inf")
    pts = np.asarray(path[max(0, lo):], dtype=np.float32)
    if pts.size == 0:
        pts = np.asarray(path, dtype=np.float32)
    d = np.linalg.norm(pts - np.asarray(point, dtype=np.float32)[None, :], axis=1)
    return float(np.min(d))


def _dynamic_metrics(
    path: List[Tuple[int, int]],
    step_maps: List[Mapping[str, np.ndarray]],
    spec: DynamicEventSpec,
    goal: Tuple[int, int],
    stage1_path: Sequence[Tuple[int, int]],
    *,
    gsd: float,
    reference_length_m: float,
    replans: int,
    compute_s: float,
    hard_violation_penalty: float,
) -> Dict[str, float]:
    if len(path) < 2:
        latency_ms = 1000.0 * float(compute_s)
        return {
            "success": 0.0,
            "post_event_cvar_risk": 1.0,
            "post_event_cvar_violation": 1.0 + hard_violation_penalty,
            "full_episode_cvar_risk": 1.0,
            "full_episode_cvar_violation": 1.0 + hard_violation_penalty,
            "event_window_cvar_risk": 1.0,
            "event_window_cvar_violation": 1.0 + hard_violation_penalty,
            "post_event_mean_risk": 1.0,
            "event_window_mean_risk": 1.0,
            "post_event_risk_exposure": 0.0,
            "event_window_risk_exposure": 0.0,
            "stale_exposure": 0.0,
            "reaction_delay": float(spec.duration),
            "route_deviation_delay": float(spec.duration),
            "opportunity_normalized_delay": 1.0,
            "replans": float(replans),
            "compute_ms": latency_ms,
            "stuck": 1.0,
        }
    length = 0.0
    hard_len = 0.0
    risk_exposure = 0.0
    post_vals: List[float] = []
    post_wts: List[float] = []
    post_viol: List[float] = []
    all_vals: List[float] = []
    all_wts: List[float] = []
    all_viol: List[float] = []
    event_vals: List[float] = []
    event_wts: List[float] = []
    event_viol: List[float] = []
    headings: List[float] = []
    visited = set()
    revisits = 0
    pre_heading: Optional[float] = None
    reaction_step: Optional[int] = None
    route_deviation_step: Optional[int] = None
    stale_exposure = 0.0
    post_event_exposure = 0.0
    event_window_exposure = 0.0
    for i, (p0, p1) in enumerate(zip(path[:-1], path[1:])):
        dr = int(p1[0] - p0[0])
        dc = int(p1[1] - p0[1])
        step_len = gsd * STEP.get((dr, dc), float(np.linalg.norm([dr, dc])))
        length += step_len
        maps = step_maps[min(i + 1, len(step_maps) - 1)]
        risk = float(maps["risk_map"][p1])
        hard = bool(maps["hard_mask"][p1])
        violation = risk + float(hard_violation_penalty) * float(hard)
        risk_exposure += step_len * risk
        hard_len += step_len * float(hard)
        all_vals.append(risk)
        all_viol.append(violation)
        all_wts.append(step_len)
        if step_len > 0:
            headings.append(float(math.atan2(dr, dc)))
        if i == max(0, spec.event_step - 2) and headings:
            pre_heading = headings[-1]
        if i >= spec.event_step:
            post_vals.append(risk)
            post_viol.append(violation)
            post_wts.append(step_len)
            post_event_exposure += step_len * risk
            if reaction_step is None and pre_heading is not None and headings:
                delta = math.atan2(math.sin(headings[-1] - pre_heading), math.cos(headings[-1] - pre_heading))
                if abs(delta) > 0.45:
                    reaction_step = i
            if route_deviation_step is None:
                ref_idx = min(i + 1, len(stage1_path) - 1)
                dist_to_nominal = _nearest_distance_to_path(stage1_path, p1, lo=max(0, ref_idx - 4))
                if gsd * dist_to_nominal > 1.0:
                    route_deviation_step = i
            if reaction_step is None:
                stale_exposure += step_len * risk
        if spec.event_step <= i <= spec.event_step + spec.duration:
            event_vals.append(risk)
            event_viol.append(violation)
            event_wts.append(step_len)
            event_window_exposure += step_len * risk
        if p1 in visited:
            revisits += 1
        visited.add(p1)
    curv = 0.0
    for h0, h1 in zip(headings[:-1], headings[1:]):
        d = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))
        curv += d * d
    final_dist_m = gsd * _goal_dist(path[-1], goal)
    success = 1.0 if final_dist_m <= 3.0 else 0.0
    reaction_delay = float((reaction_step - spec.event_step) if reaction_step is not None else spec.duration)
    route_deviation_delay = float(
        (route_deviation_step - spec.event_step) if route_deviation_step is not None else spec.duration
    )
    opportunity_normalized_delay = reaction_delay / max(float(spec.duration), 1.0)
    path_ratio = length / max(reference_length_m, 1e-6)
    stuck = 1.0 if (success < 0.5 or revisits > 5 or path_ratio > 1.8) else 0.0
    post_mean = float(np.average(post_vals, weights=post_wts)) if post_wts and sum(post_wts) > 1e-8 else 0.0
    event_mean = float(np.average(event_vals, weights=event_wts)) if event_wts and sum(event_wts) > 1e-8 else 0.0
    return {
        "success": success,
        "hard_hazard_length_m": hard_len,
        "risk_exposure": risk_exposure,
        "post_event_cvar_risk": _risk_tail(post_vals, post_wts, q=0.2),
        "post_event_cvar_violation": _risk_tail(post_viol, post_wts, q=0.2),
        "full_episode_cvar_risk": _risk_tail(all_vals, all_wts, q=0.2),
        "full_episode_cvar_violation": _risk_tail(all_viol, all_wts, q=0.2),
        "event_window_cvar_risk": _risk_tail(event_vals, event_wts, q=0.2),
        "event_window_cvar_violation": _risk_tail(event_viol, event_wts, q=0.2),
        "post_event_mean_risk": post_mean,
        "event_window_mean_risk": event_mean,
        "post_event_risk_exposure": post_event_exposure,
        "event_window_risk_exposure": event_window_exposure,
        "path_length_m": length,
        "path_length_ratio": path_ratio,
        "curvature_energy": float(curv),
        "reaction_delay": reaction_delay,
        "route_deviation_delay": route_deviation_delay,
        "opportunity_normalized_delay": opportunity_normalized_delay,
        "stale_exposure": stale_exposure,
        "replans": float(replans),
        "compute_ms": 1000.0 * float(compute_s) / max(1, len(path) - 1),
        "stuck": stuck,
        "revisit_count": float(revisits),
    }


def _rollout(
    method: str,
    base_maps: Mapping[str, np.ndarray],
    spec: DynamicEventSpec,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    stage1_path: List[Tuple[int, int]],
    risk_path: List[Tuple[int, int]],
    *,
    gsd: float,
    max_steps: int,
    replan_period: int,
    risk_weight: float,
    hard_margin_m: float,
    route_horizon: int,
    improvement_margin: float,
) -> Tuple[List[Tuple[int, int]], List[Mapping[str, np.ndarray]], int, float]:
    cur = start
    path = [cur]
    prev: Optional[Tuple[int, int]] = None
    step_maps = [apply_dynamic_event(base_maps, spec, 0, resolution=gsd)]
    idx_stage1 = 0
    idx_risk = 0
    detour_cooldown = 0
    cached_plan: Optional[List[Tuple[int, int]]] = None
    plan_idx = 0
    replans = 0
    compute_s = 0.0
    for t in range(max_steps):
        maps_t = apply_dynamic_event(base_maps, spec, t, resolution=gsd)
        decision_start = time.perf_counter()
        if method == "stage1":
            nxt, idx_stage1 = _path_follower_next(stage1_path, cur, idx_stage1)
        elif method == "risk_loss_only":
            nxt, idx_risk = _path_follower_next(risk_path, cur, idx_risk)
        elif method == "blackbox_cvar":
            nxt = _blackbox_cvar_step(
                maps_t,
                cur,
                goal,
                previous=prev,
                hard_margin_m=hard_margin_m,
                horizon=route_horizon,
            )
        elif method == "stage2_expected_cost":
            nxt, idx_stage1, idx_risk, detour_cooldown = _route_aware_step(
                maps_t,
                cur,
                goal,
                stage1_path,
                risk_path,
                idx_stage1,
                idx_risk,
                detour_cooldown,
                horizon=max(6, route_horizon // 2),
                hard_margin_m=0.5 * hard_margin_m,
                improvement_margin=2.0 * improvement_margin,
            )
        elif method == "fixed_coeff_stage2":
            nxt = _non_route_directional_step(
                maps_t,
                cur,
                goal,
                horizon=route_horizon,
                hard_margin_m=hard_margin_m,
                improvement_margin=improvement_margin,
            )
        elif method == "scalar_stage2":
            nxt = _greedy_scalar_step(maps_t, cur, goal, hard_margin_m=hard_margin_m)
        elif method == "non_route_directional_stage2":
            nxt = _non_route_directional_step(
                maps_t,
                cur,
                goal,
                horizon=route_horizon,
                hard_margin_m=hard_margin_m,
                improvement_margin=improvement_margin,
            )
        elif method == "neural_potential_field":
            nominal, idx_stage1 = _path_follower_next(stage1_path, cur, idx_stage1)
            nxt = _potential_field_step(maps_t, cur, nominal, hard_margin_m=hard_margin_m)
        elif method == "dwa_semantic":
            nxt = _candidate_step(
                maps_t,
                cur,
                goal,
                previous=prev,
                risk_weight=4.0,
                clearance_weight=5.0,
                heading_weight=0.8,
                hard_margin_m=hard_margin_m,
            )
        elif method == "cbf_safety_filter":
            nominal, idx_stage1 = _path_follower_next(stage1_path, cur, idx_stage1)
            nxt = _cbf_filter_step(
                maps_t,
                nominal,
                cur,
                goal,
                hard_margin_m=hard_margin_m,
            )
        elif method == "route_aware_stage2":
            nxt, idx_stage1, idx_risk, detour_cooldown = _route_aware_step(
                maps_t,
                cur,
                goal,
                stage1_path,
                risk_path,
                idx_stage1,
                idx_risk,
                detour_cooldown,
                horizon=route_horizon,
                hard_margin_m=hard_margin_m,
                improvement_margin=improvement_margin,
            )
        elif method == "local_astar_budget":
            if cached_plan is None or t % max(1, replan_period) == 0:
                nxt, cached_plan, plan_idx, did_replan, elapsed = _planner_next(
                    maps_t,
                    cur,
                    goal,
                    planner="astar",
                    cached_plan=cached_plan,
                    plan_idx=plan_idx,
                    risk_weight=risk_weight,
                )
                replans += int(did_replan)
            else:
                nxt, plan_idx = _path_follower_next(cached_plan, cur, plan_idx)
        elif method == "mpc_budget":
            if cached_plan is None or t % max(1, replan_period) == 0:
                nxt, cached_plan, plan_idx, did_replan, elapsed = _planner_next(
                    maps_t,
                    cur,
                    goal,
                    planner="mpc",
                    cached_plan=cached_plan,
                    plan_idx=plan_idx,
                    risk_weight=risk_weight,
                )
                replans += int(did_replan)
            else:
                nxt, plan_idx = _path_follower_next(cached_plan, cur, plan_idx)
        elif method == "oracle_replanner":
            nxt, cached_plan, plan_idx, did_replan, elapsed = _planner_next(
                maps_t,
                cur,
                goal,
                planner="astar",
                cached_plan=None,
                plan_idx=0,
                risk_weight=risk_weight,
            )
            replans += int(did_replan)
        else:
            raise ValueError(method)
        compute_s += time.perf_counter() - decision_start

        if not _in_bounds(nxt, maps_t["risk_map"].shape):
            nxt = cur
        prev = cur
        cur = nxt
        path.append(cur)
        step_maps.append(apply_dynamic_event(base_maps, spec, t + 1, resolution=gsd))
        if gsd * _goal_dist(cur, goal) <= 2.0:
            break
    return path, step_maps, replans, compute_s


def _summarize(rows: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    out: List[Dict[str, float | str]] = []
    groups = sorted({(str(r["event_type"]), str(r["method"])) for r in rows})
    metrics = [
        "success",
        "hard_hazard_length_m",
        "risk_exposure",
        "post_event_cvar_risk",
        "post_event_cvar_violation",
        "full_episode_cvar_risk",
        "full_episode_cvar_violation",
        "event_window_cvar_risk",
        "event_window_cvar_violation",
        "post_event_mean_risk",
        "event_window_mean_risk",
        "post_event_risk_exposure",
        "event_window_risk_exposure",
        "reaction_delay",
        "route_deviation_delay",
        "opportunity_normalized_delay",
        "stale_exposure",
        "path_length_ratio",
        "curvature_energy",
        "replans",
        "compute_ms",
        "stuck",
    ]
    for event_type, method in groups:
        pool = [r for r in rows if r["event_type"] == event_type and r["method"] == method]
        row: Dict[str, float | str] = {"event_type": event_type, "method": method, "n": float(len(pool))}
        for m in metrics:
            row[m] = float(np.nanmean([float(r[m]) for r in pool]))
        out.append(row)
    return out


def _write_csv(path: Path, rows: List[Dict[str, float | str]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate RELLIS-Dyn dynamic material-risk events.")
    ap.add_argument("--bev-root", type=Path, default=ROOT / "cache" / "rellis_bev_all_seqbalanced_2500")
    ap.add_argument("--pairs-root", type=Path, default=ROOT / "cache" / "rellis_pairs_all_seqbalanced_2500_loso")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rellis_dyn_main")
    ap.add_argument("--event-types", nargs="+", default=list(MAIN_EVENT_TYPES), choices=list(MAIN_EVENT_TYPES))
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--max-episodes", type=int, default=120)
    ap.add_argument("--event-fraction", type=float, default=0.38)
    ap.add_argument("--event-duration", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=140)
    ap.add_argument("--replan-period", type=int, default=8)
    ap.add_argument("--risk-weight", type=float, default=18.0)
    ap.add_argument("--hard-margin-m", type=float, default=1.0)
    ap.add_argument("--route-horizon", type=int, default=18)
    ap.add_argument("--improvement-margin", type=float, default=0.25)
    ap.add_argument("--hard-violation-penalty", type=float, default=2.0)
    ap.add_argument("--progress-every", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bev_manifest = json.loads((args.bev_root / "manifest.json").read_text())
    pair_manifest = json.loads((args.pairs_root / "manifest.json").read_text())
    cfg = BevConfig(**bev_manifest["config"]["bev"])
    gsd = float(cfg.resolution)
    episodes = pair_manifest["episodes"][: args.max_episodes]
    rows: List[Dict[str, float | str]] = []
    event_rows: List[Dict[str, float | str]] = []
    scene_cache: Dict[str, Dict] = {}
    start_time = time.perf_counter()
    total_work = max(1, len(episodes) * len(args.event_types) * len(args.methods))
    for ep_idx, ep in enumerate(episodes, start=1):
        scene_path = str(ep["scene_path"])
        if scene_path not in scene_cache:
            scene_cache[scene_path] = _load_scene(args.bev_root, scene_path)
        maps = scene_cache[scene_path]["maps"]
        start = tuple(int(x) for x in ep["start_rc"])
        goal = tuple(int(x) for x in ep["goal_rc"])
        stage1_path = _as_path(ep["stage1_path"])
        risk_path = _as_path(ep["risk_path"])
        ref_len = max(path_length_m(stage1_path, gsd=gsd), gsd * _goal_dist(start, goal), 1e-6)
        for event_type in args.event_types:
            spec = make_event_spec(
                event_type,
                stage1_path,
                risk_path,
                goal,
                event_fraction=args.event_fraction,
                duration=args.event_duration,
            )
            event_rows.append(
                {
                    "episode_id": str(ep["episode_id"]),
                    "scene_id": str(ep["scene_id"]),
                    "regime": str(ep["regime"]),
                    **spec.to_dict(),
                }
            )
            for method in args.methods:
                path, step_maps, replans, compute_s = _rollout(
                    method,
                    maps,
                    spec,
                    start,
                    goal,
                    stage1_path,
                    risk_path,
                    gsd=gsd,
                    max_steps=args.max_steps,
                    replan_period=args.replan_period,
                    risk_weight=args.risk_weight,
                    hard_margin_m=args.hard_margin_m,
                    route_horizon=args.route_horizon,
                    improvement_margin=args.improvement_margin,
                )
                metrics = _dynamic_metrics(
                    path,
                    step_maps,
                    spec,
                    goal,
                    stage1_path,
                    gsd=gsd,
                    reference_length_m=ref_len,
                    replans=replans,
                    compute_s=compute_s,
                    hard_violation_penalty=args.hard_violation_penalty,
                )
                rows.append(
                    {
                        "episode_id": str(ep["episode_id"]),
                        "scene_id": str(ep["scene_id"]),
                        "sequence": str(ep["sequence"]),
                        "regime": str(ep["regime"]),
                        "event_type": event_type,
                        "method": method,
                        **metrics,
                    }
                )
        if args.progress_every > 0 and (ep_idx % args.progress_every == 0 or ep_idx == len(episodes)):
            elapsed = time.perf_counter() - start_time
            done = ep_idx * len(args.event_types) * len(args.methods)
            rate = done / max(elapsed, 1e-8)
            remain = (total_work - done) / max(rate, 1e-8)
            print(
                f"[rellis-dyn] {ep_idx}/{len(episodes)} episodes "
                f"({done}/{total_work} rollouts), elapsed={elapsed/60:.1f}m, "
                f"eta={remain/60:.1f}m",
                flush=True,
            )
            _write_csv(args.out / "dynamic_rollouts.partial.csv", rows)
            _write_csv(args.out / "dynamic_event_specs.partial.csv", event_rows)

    summary_rows = _summarize(rows)
    _write_csv(args.out / "dynamic_rollouts.csv", rows)
    _write_csv(args.out / "dynamic_summary_by_event.csv", summary_rows)
    _write_csv(args.out / "dynamic_event_specs.csv", event_rows)
    overall_rows = []
    for method in sorted({str(r["method"]) for r in rows}):
        pool = [r for r in rows if str(r["method"]) == method]
        overall_rows.append(
            {
                "method": method,
                "n": float(len(pool)),
                "success": float(np.mean([float(r["success"]) for r in pool])),
                "post_event_cvar_risk": float(np.mean([float(r["post_event_cvar_risk"]) for r in pool])),
                "post_event_cvar_violation": float(np.mean([float(r["post_event_cvar_violation"]) for r in pool])),
                "full_episode_cvar_risk": float(np.mean([float(r["full_episode_cvar_risk"]) for r in pool])),
                "full_episode_cvar_violation": float(np.mean([float(r["full_episode_cvar_violation"]) for r in pool])),
                "event_window_cvar_risk": float(np.mean([float(r["event_window_cvar_risk"]) for r in pool])),
                "event_window_cvar_violation": float(np.mean([float(r["event_window_cvar_violation"]) for r in pool])),
                "post_event_mean_risk": float(np.mean([float(r["post_event_mean_risk"]) for r in pool])),
                "event_window_mean_risk": float(np.mean([float(r["event_window_mean_risk"]) for r in pool])),
                "post_event_risk_exposure": float(np.mean([float(r["post_event_risk_exposure"]) for r in pool])),
                "event_window_risk_exposure": float(np.mean([float(r["event_window_risk_exposure"]) for r in pool])),
                "reaction_delay": float(np.mean([float(r["reaction_delay"]) for r in pool])),
                "route_deviation_delay": float(np.mean([float(r["route_deviation_delay"]) for r in pool])),
                "opportunity_normalized_delay": float(np.mean([float(r["opportunity_normalized_delay"]) for r in pool])),
                "stale_exposure": float(np.mean([float(r["stale_exposure"]) for r in pool])),
                "replans": float(np.mean([float(r["replans"]) for r in pool])),
                "compute_ms": float(np.mean([float(r["compute_ms"]) for r in pool])),
                "curvature_energy": float(np.mean([float(r["curvature_energy"]) for r in pool])),
                "stuck": float(np.mean([float(r["stuck"]) for r in pool])),
                "hard_hazard_length_m": float(np.mean([float(r["hard_hazard_length_m"]) for r in pool])),
                "path_length_ratio": float(np.mean([float(r["path_length_ratio"]) for r in pool])),
            }
        )
    _write_csv(args.out / "dynamic_main_table.csv", overall_rows)
    summary = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "num_episodes": len(episodes),
        "num_rows": len(rows),
        "event_types": list(args.event_types),
        "methods": list(args.methods),
        "overall": overall_rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote RELLIS-Dyn results to {args.out}")
    for row in overall_rows:
        print(
            f"{row['method']:>22s}  success={row['success']:.3f}  "
            f"postCVaR={row['post_event_cvar_risk']:.3f}  "
            f"delay={row['reaction_delay']:.2f}  replans={row['replans']:.1f}"
        )


if __name__ == "__main__":
    main()
