from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from .common import (
    HighwayBaseline,
    current_lane_index,
    get_lane_count,
    get_lane_width,
    lane_id_from_index,
    lane_keep_action,
)


@dataclass
class CandidatePlan:
    lane_id: int
    target_speed: float
    cost: float


class RiskAwareMPCBaseline(HighwayBaseline):
    name = "risk_aware_mpc"

    def __init__(
        self,
        horizon: int = 8,
        dt: float = 0.1,
        target_speed: float = 24.0,
        safe_clearance_m: float = 8.0,
        ttc_safe_s: float = 3.0,
        lane_change_penalty: float = 3.0,
    ):
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.target_speed = float(target_speed)
        self.safe_clearance_m = float(safe_clearance_m)
        self.ttc_safe_s = float(ttc_safe_s)
        self.lane_change_penalty = float(lane_change_penalty)

    def _candidate_lanes(self, env: Any) -> List[int]:
        uenv = env.unwrapped
        cur = current_lane_index(env)
        if cur is None:
            return [0]
        base_id = lane_id_from_index(cur)
        candidates = {base_id}
        for lane_idx in uenv.road.network.side_lanes(cur):
            if uenv.road.network.get_lane(lane_idx).is_reachable_from(uenv.vehicle.position):
                candidates.add(lane_id_from_index(lane_idx))
        return sorted(candidates)

    def _simulate_plan(self, obs: dict, *, lane_id: int, target_speed: float, lane_width: float, lane_count: int) -> float:
        lane_id = 0 if lane_id is None else int(lane_id)
        x = float(obs["o0"][0])
        y = float(obs["o0"][1])
        vx = float(obs["v0"][0])
        vy = float(obs["v0"][1])
        v = max(1e-3, float(np.linalg.norm(obs["v0"])))
        goal_y = lane_width * lane_id
        road_y_min = -0.5 * lane_width
        road_y_max = (lane_count - 0.5) * lane_width

        C = np.asarray(obs["C"], dtype=np.float64).copy()
        V = np.asarray(obs.get("V_neighbors", np.zeros_like(C)), dtype=np.float64).copy()
        R = np.asarray(obs["R"], dtype=np.float64).copy()
        mask = np.asarray(obs["mask"], dtype=bool)

        cost = 0.0
        for _ in range(self.horizon):
            speed_err = float(target_speed - v)
            accel = float(np.clip(0.8 * speed_err, -6.0, 4.0))
            lat_err = goal_y - y
            lat_rate = float(np.clip(0.9 * lat_err, -3.0, 3.0))

            v = max(0.0, v + accel * self.dt)
            x = x + v * self.dt
            y = y + lat_rate * self.dt
            vx = v
            vy = lat_rate

            if mask.any():
                C[mask] = C[mask] + V[mask] * self.dt
                rel = C[mask] - np.array([x, y], dtype=np.float64)
                dists = np.linalg.norm(rel, axis=-1) - R[mask]
                min_clear = float(dists.min()) if dists.size else float("inf")
                cost += 2000.0 * max(0.0, 0.5 - min_clear)
                cost += 30.0 * max(0.0, self.safe_clearance_m - min_clear)

                same_lane = np.abs(rel[:, 1]) <= (0.5 * lane_width)
                ahead = rel[:, 0] > 0.0
                if np.any(same_lane & ahead):
                    ahead_rel = rel[same_lane & ahead]
                    front = ahead_rel[np.argmin(ahead_rel[:, 0])]
                    gap = float(front[0])
                    front_speed = float(V[mask][same_lane & ahead][np.argmin(ahead_rel[:, 0]), 0])
                    closing = max(0.0, vx - front_speed)
                    if closing > 1e-6:
                        ttc = gap / closing
                        cost += 40.0 * max(0.0, self.ttc_safe_s - ttc)

            if y < road_y_min or y > road_y_max:
                cost += 500.0 + 150.0 * min(abs(y - road_y_min), abs(y - road_y_max))

            cost += 0.05 * (accel ** 2) + 0.02 * (lat_rate ** 2)
            cost += 0.2 * abs(goal_y - y)
            cost += 0.05 * max(0.0, self.target_speed - v)
            cost -= 0.1 * vx

        return float(cost)

    def act(self, env: Any, observer: Any) -> np.ndarray:
        obs = observer.build(env)
        lane_width = get_lane_width(env)
        lane_count = get_lane_count(env)
        cur_lane_id = lane_id_from_index(current_lane_index(env))
        cur_lane_id = 0 if cur_lane_id is None else int(cur_lane_id)
        speed = float(np.linalg.norm(obs["v0"]))

        target_speeds = [
            0.0,
            max(2.0, speed - 4.0),
            min(self.target_speed, speed + 0.0),
            min(self.target_speed, speed + 3.0),
        ]
        best = CandidatePlan(lane_id=cur_lane_id, target_speed=self.target_speed, cost=float("inf"))
        for lane_id in self._candidate_lanes(env):
            lane_id = 0 if lane_id is None else int(lane_id)
            for tgt_speed in target_speeds:
                cost = self._simulate_plan(
                    obs,
                    lane_id=lane_id,
                    target_speed=float(tgt_speed),
                    lane_width=lane_width,
                    lane_count=lane_count,
                )
                if lane_id != cur_lane_id:
                    cost += self.lane_change_penalty
                if cost < best.cost:
                    best = CandidatePlan(lane_id=lane_id, target_speed=float(tgt_speed), cost=cost)

        cur = current_lane_index(env)
        if cur is not None:
            target_lane_index = (cur[0], cur[1], 0 if best.lane_id is None else int(best.lane_id))
        else:
            target_lane_index = None
        return lane_keep_action(env, target_speed=best.target_speed, target_lane_index=target_lane_index)

    def describe(self):
        return {
            "name": self.name,
            "horizon": self.horizon,
            "dt": self.dt,
            "target_speed": self.target_speed,
        }


class ChanceConstrainedMPCBaseline(RiskAwareMPCBaseline):
    name = "chance_constrained_mpc"

    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.1,
        target_speed: float = 22.0,
        safe_clearance_m: float = 10.0,
        ttc_safe_s: float = 4.0,
        lane_change_penalty: float = 4.5,
        chance_margin_m: float = 3.0,
        tail_risk_weight: float = 2.0,
    ):
        super().__init__(
            horizon=horizon,
            dt=dt,
            target_speed=target_speed,
            safe_clearance_m=safe_clearance_m,
            ttc_safe_s=ttc_safe_s,
            lane_change_penalty=lane_change_penalty,
        )
        self.chance_margin_m = float(chance_margin_m)
        self.tail_risk_weight = float(tail_risk_weight)

    def _simulate_plan(self, obs: dict, *, lane_id: int, target_speed: float, lane_width: float, lane_count: int) -> float:
        base = super()._simulate_plan(
            obs,
            lane_id=lane_id,
            target_speed=target_speed,
            lane_width=lane_width,
            lane_count=lane_count,
        )
        C = np.asarray(obs["C"], dtype=np.float64)
        V = np.asarray(obs.get("V_neighbors", np.zeros_like(C)), dtype=np.float64)
        R = np.asarray(obs["R"], dtype=np.float64)
        mask = np.asarray(obs["mask"], dtype=bool)
        if not mask.any():
            return base

        ego = np.asarray(obs["o0"], dtype=np.float64)
        rel = C[mask] - ego[None, :]
        dists = np.linalg.norm(rel, axis=-1) - R[mask]
        shortfall = np.maximum(0.0, self.safe_clearance_m + self.chance_margin_m - dists)
        base += 45.0 * float(shortfall.mean()) if shortfall.size else 0.0

        same_lane = np.abs(rel[:, 1] - lane_width * (lane_id - round(float(ego[1] / max(lane_width, 1e-6))))) <= (0.75 * lane_width)
        ahead = rel[:, 0] > 0.0
        if np.any(same_lane & ahead):
            ahead_rel = rel[same_lane & ahead]
            ahead_v = V[mask][same_lane & ahead]
            order = np.argsort(ahead_rel[:, 0])
            gaps = ahead_rel[order, 0]
            speeds = ahead_v[order, 0]
            ego_speed = float(np.linalg.norm(obs["v0"]))
            closings = np.maximum(0.0, ego_speed - speeds)
            ttcs = gaps / np.maximum(closings, 1e-6)
            finite = ttcs[np.isfinite(ttcs)]
            if finite.size:
                tail = np.sort(finite)[: max(1, int(np.ceil(0.3 * finite.size)))]
                base += self.tail_risk_weight * 30.0 * float(np.maximum(0.0, self.ttc_safe_s - tail).mean())
        return float(base)

    def describe(self):
        out = super().describe()
        out.update(
            {
                "chance_margin_m": self.chance_margin_m,
                "tail_risk_weight": self.tail_risk_weight,
            }
        )
        return out
