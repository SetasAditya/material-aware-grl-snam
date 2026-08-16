from __future__ import annotations

import heapq
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from scripts.build_dfc2018_stagewise import GT_GSD
from scripts.baselines.dfc.rl import ppo_lagrangian_plan


DIAGONALS = True
DIRS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
] if DIAGONALS else [(-1, 0), (1, 0), (0, -1), (0, 1)]
STEP = {(dr, dc): 1.4142 if abs(dr) + abs(dc) == 2 else 1.0 for dr, dc in DIRS}

FAST_PLANNERS = (
    "blind_dijkstra",
    "geometry_astar",
    "risk_weighted_astar",
    "oracle_astar",
)

HEAVY_PLANNERS = (
    "cvar_costmap_astar",
    "chance_constrained_mpc",
    "ppo_lagrangian",
)

ALL_PLANNERS = FAST_PLANNERS + HEAVY_PLANNERS
DEFAULT_PLANNERS = FAST_PLANNERS


def _octile_distance(goal: Tuple[int, int]) -> Callable[[int, int], float]:
    gr, gc = goal

    def h(r: int, c: int) -> float:
        dr = abs(r - gr)
        dc = abs(c - gc)
        return max(dr, dc) + (1.4142 - 1.0) * min(dr, dc)

    return h


def _search_grid(
    maps: Dict[str, np.ndarray],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    blocked: np.ndarray,
    cell_cost: Callable[[int, int], float],
    heuristic: bool,
) -> Optional[List[Tuple[int, int]]]:
    h_rows, h_cols = maps["z2_labels"].shape
    h_fn = _octile_distance(goal)

    gscore = {start: 0.0}
    parent = {start: None}
    heap = [(h_fn(*start) if heuristic else 0.0, 0.0, start[0], start[1])]
    while heap:
        _, g, r, c = heapq.heappop(heap)
        node = (r, c)
        if node == goal:
            path: List[Tuple[int, int]] = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        if g > gscore.get(node, 1e18) + 1e-9:
            continue
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h_rows and 0 <= nc < h_cols):
                continue
            if blocked[nr, nc]:
                continue
            cost = cell_cost(nr, nc)
            if not np.isfinite(cost):
                continue
            ng = g + STEP[(dr, dc)] * cost
            nxt = (nr, nc)
            if ng < gscore.get(nxt, 1e18):
                gscore[nxt] = ng
                parent[nxt] = node
                priority = ng + (h_fn(nr, nc) if heuristic else 0.0)
                heapq.heappush(heap, (priority, ng, nr, nc))
    return None


def blind_dijkstra(
    maps: Dict[str, np.ndarray],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    blocked = np.zeros_like(maps["z2_labels"], dtype=bool)
    return _search_grid(
        maps, start, goal,
        blocked=blocked,
        cell_cost=lambda *_: 1.0,
        heuristic=False,
    )


def geometry_astar(
    maps: Dict[str, np.ndarray],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    blocked = maps["geom_occ"].astype(bool)
    return _search_grid(
        maps, start, goal,
        blocked=blocked,
        cell_cost=lambda *_: 1.0,
        heuristic=True,
    )


def risk_weighted_astar(
    maps: Dict[str, np.ndarray],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    risk_weight: float = 10.0,
) -> Optional[List[Tuple[int, int]]]:
    blocked = maps["geom_occ"].astype(bool)
    risk = maps["risk_map"]
    return _search_grid(
        maps, start, goal,
        blocked=blocked,
        cell_cost=lambda r, c: 1.0 + risk_weight * float(risk[r, c]),
        heuristic=True,
    )


def oracle_astar(
    maps: Dict[str, np.ndarray],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    risk_weight: float = 10.0,
) -> Optional[List[Tuple[int, int]]]:
    blocked = maps["hard_mask"].astype(bool)
    risk = maps["risk_map"]
    return _search_grid(
        maps, start, goal,
        blocked=blocked,
        cell_cost=lambda r, c: 1.0 + risk_weight * float(risk[r, c]),
        heuristic=True,
    )


def cvar_costmap_astar(
    maps: Dict[str, np.ndarray],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    risk_weight: float = 10.0,
    cvar_alpha: float = 0.9,
    patch_radius: int = 3,
    hard_penalty: float = 8.0,
) -> Optional[List[Tuple[int, int]]]:
    blocked = maps["geom_occ"].astype(bool)
    risk = maps["risk_map"]
    hard = maps["hard_mask"].astype(bool)

    @lru_cache(maxsize=200000)
    def local_cvar(r: int, c: int) -> float:
        r0 = max(0, r - patch_radius)
        r1 = min(risk.shape[0], r + patch_radius + 1)
        c0 = max(0, c - patch_radius)
        c1 = min(risk.shape[1], c + patch_radius + 1)
        vals = risk[r0:r1, c0:c1].reshape(-1)
        q = float(np.quantile(vals, cvar_alpha))
        tail = vals[vals >= q]
        cvar = float(tail.mean()) if tail.size else float(vals.max())
        if hard[r, c]:
            cvar = max(cvar, 1.0)
        return cvar

    return _search_grid(
        maps,
        start,
        goal,
        blocked=blocked,
        cell_cost=lambda r, c: (
            1.0
            + risk_weight * (0.45 * float(risk[r, c]) + 0.55 * local_cvar(r, c))
            + hard_penalty * float(hard[r, c])
        ),
        heuristic=True,
    )


def _beam_sequence(
    maps: Dict[str, np.ndarray],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    risk_weight: float,
    horizon: int,
    beam_width: int,
    safety_margin_m: float,
    tail_bias: float,
    turn_penalty: float,
    prev_dir: Optional[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    geom = maps["geom_occ"].astype(bool)
    hard = maps["hard_mask"].astype(bool)
    risk = maps["risk_map"]
    sdf = maps["sdf_hard"]
    h_fn = _octile_distance(goal)

    @lru_cache(maxsize=4096)
    def local_tail(r: int, c: int) -> float:
        r0 = max(0, r - 2)
        r1 = min(risk.shape[0], r + 3)
        c0 = max(0, c - 2)
        c1 = min(risk.shape[1], c + 3)
        vals = risk[r0:r1, c0:c1]
        return float(np.quantile(vals, 0.9))

    beam = [(0.0, start, [], prev_dir)]
    best_goal = None
    for _ in range(horizon):
        candidates = []
        for base_cost, node, seq, last_dir in beam:
            r, c = node
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < risk.shape[0] and 0 <= nc < risk.shape[1]):
                    continue
                if geom[nr, nc]:
                    continue
                if float(sdf[nr, nc]) < safety_margin_m:
                    continue
                step = STEP[(dr, dc)]
                turn = turn_penalty if last_dir is not None and last_dir != (dr, dc) else 0.0
                cost = (
                    base_cost
                    + step
                    * (
                        1.0
                        + risk_weight * float(risk[nr, nc])
                        + tail_bias * local_tail(nr, nc)
                        + 12.0 * float(hard[nr, nc])
                        + turn
                    )
                )
                new_seq = seq + [(nr, nc)]
                if (nr, nc) == goal:
                    best_goal = new_seq
                    break
                priority = cost + 1.2 * h_fn(nr, nc)
                candidates.append((priority, cost, (nr, nc), new_seq, (dr, dc)))
            if best_goal is not None:
                break
        if best_goal is not None:
            return best_goal
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        beam = [(cost, node, seq, d) for _, cost, node, seq, d in candidates[:beam_width]]
    if not beam:
        return None
    beam.sort(key=lambda x: x[0] + h_fn(*x[1]))
    return beam[0][2]


def chance_constrained_mpc(
    maps: Dict[str, np.ndarray],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    risk_weight: float = 8.0,
    horizon: int = 14,
    beam_width: int = 28,
    safety_margin_m: float = 0.35,
    tail_bias: float = 2.5,
    turn_penalty: float = 0.12,
) -> Optional[List[Tuple[int, int]]]:
    guide_path = cvar_costmap_astar(
        maps,
        start,
        goal,
        risk_weight=max(3.0, risk_weight),
    )
    if not guide_path:
        return None
    path = [start]
    cur = start
    prev_dir: Optional[Tuple[int, int]] = None
    max_steps = max(50, int(2.5 * _octile_distance(goal)(*start)))
    visited = {start: 1}

    for _ in range(max_steps):
        if cur == goal:
            return path
        seq = _beam_sequence(
            maps,
            cur,
            goal,
            risk_weight=risk_weight,
            horizon=horizon,
            beam_width=beam_width,
            safety_margin_m=safety_margin_m,
            tail_bias=tail_bias,
            turn_penalty=turn_penalty,
            prev_dir=prev_dir,
        )
        if not seq:
            return guide_path
        nxt = seq[0]
        prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
        cur = nxt
        path.append(cur)
        visited[cur] = visited.get(cur, 0) + 1
        if visited[cur] > 4:
            return guide_path
    return path if path[-1] == goal else guide_path


def plan_path(
    planner_name: str,
    maps: Dict[str, np.ndarray],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    risk_weight: float = 10.0,
) -> Optional[List[Tuple[int, int]]]:
    if planner_name == "blind_dijkstra":
        return blind_dijkstra(maps, start, goal)
    if planner_name == "geometry_astar":
        return geometry_astar(maps, start, goal)
    if planner_name == "risk_weighted_astar":
        return risk_weighted_astar(maps, start, goal, risk_weight=risk_weight)
    if planner_name == "oracle_astar":
        return oracle_astar(maps, start, goal, risk_weight=risk_weight)
    if planner_name == "cvar_costmap_astar":
        return cvar_costmap_astar(maps, start, goal, risk_weight=risk_weight)
    if planner_name == "chance_constrained_mpc":
        return chance_constrained_mpc(maps, start, goal, risk_weight=max(2.0, risk_weight / 2.0))
    if planner_name == "ppo_lagrangian":
        return ppo_lagrangian_plan(maps, start, goal, risk_weight=risk_weight)
    raise KeyError(f"Unknown planner: {planner_name}")


def path_length_m(path: Optional[List[Tuple[int, int]]], *, gsd: float = GT_GSD) -> float:
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        total += gsd * STEP[(r1 - r0, c1 - c0)]
    return float(total)
