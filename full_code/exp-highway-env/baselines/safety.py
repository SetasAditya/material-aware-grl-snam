from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .common import (
    HighwayBaseline,
    clip_action_physical,
    current_lane_index,
    front_vehicle_info,
    get_lane_count,
    get_lane_width,
    lane_center_y,
    lane_keep_action,
    normalize_action,
)


class CBFQPFilterBaseline(HighwayBaseline):
    """Lightweight CBF-QP-style safety filter on top of a lane-keeping nominal policy.

    This is a diagonal-QP approximation: minimize deviation from a nominal
    (accel, steer) subject to simple forward-gap and road-boundary barrier
    constraints. The resulting optimizer reduces to clipping and steering
    projection, which keeps the implementation dependable for benchmarking.
    """

    name = "cbf_qp_filter"

    def __init__(
        self,
        nominal_speed: float = 24.0,
        safe_time_headway_s: float = 1.8,
        min_gap_m: float = 8.0,
        gamma_long: float = 0.8,
        gamma_lat: float = 1.0,
        dt: float = 0.1,
    ):
        self.nominal_speed = float(nominal_speed)
        self.safe_time_headway_s = float(safe_time_headway_s)
        self.min_gap_m = float(min_gap_m)
        self.gamma_long = float(gamma_long)
        self.gamma_lat = float(gamma_lat)
        self.dt = float(dt)

    def _nominal_action(self, env: Any) -> np.ndarray:
        return lane_keep_action(env, target_speed=self.nominal_speed)

    def _longitudinal_filter(self, env: Any, accel_nom: float) -> float:
        uenv = env.unwrapped
        ego = uenv.vehicle
        info = front_vehicle_info(env)
        if info["exists"] < 0.5:
            return accel_nom

        ego_speed = float(getattr(ego, "speed", np.linalg.norm(ego.velocity)))
        d_safe = self.min_gap_m + self.safe_time_headway_s * ego_speed
        h = float(info["gap"] - d_safe)
        front_speed = float(info["front_speed"])
        # Discrete-time CBF surrogate:
        # h_next - h + gamma*h >= 0
        # h_next ≈ gap + dt*(v_front - (v_ego + a*dt)) - d_safe
        # => a <= (v_front - v_ego + gamma*h) / dt
        a_max = (front_speed - ego_speed + self.gamma_long * h) / max(self.dt, 1e-6)
        return min(accel_nom, float(a_max))

    def _lateral_filter(self, env: Any, steer_nom: float) -> float:
        uenv = env.unwrapped
        ego = uenv.vehicle
        y = float(ego.position[1])
        lc_y = float(lane_center_y(env))
        lane_width = float(get_lane_width(env))
        n_lanes = int(get_lane_count(env))
        road_low = -0.5 * lane_width
        road_high = (n_lanes - 0.5) * lane_width
        h_low = y - road_low
        h_high = road_high - y

        steer = float(steer_nom)
        # If we are close to the boundary, project the steering toward the road interior.
        boundary_margin = 0.35 * lane_width
        center_correction = -self.gamma_lat * (y - lc_y) / max(lane_width, 1e-6)
        if h_low < boundary_margin:
            steer = max(steer, center_correction)
        if h_high < boundary_margin:
            steer = min(steer, center_correction)
        return steer

    def act(self, env: Any, observer: Any) -> np.ndarray:
        nominal = self._nominal_action(env)
        accel_nom = 0.5 * (float(nominal[0]) + 1.0) * (8.0 - (-6.0)) - 6.0
        steer_nom = 0.5 * (float(nominal[1]) + 1.0) * (np.pi / 4 - (-np.pi / 4)) - (np.pi / 4)
        accel = self._longitudinal_filter(env, accel_nom)
        steer = self._lateral_filter(env, steer_nom)
        accel, steer = clip_action_physical(accel, steer)
        return normalize_action(accel, steer)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nominal_speed": self.nominal_speed,
            "safe_time_headway_s": self.safe_time_headway_s,
            "min_gap_m": self.min_gap_m,
        }
