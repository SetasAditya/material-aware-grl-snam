from __future__ import annotations

from typing import Any

import numpy as np

from .common import (
    HighwayBaseline,
    front_vehicle_info,
    idle_continuous_action,
    lane_keep_action,
)

from data_collect_idm import replace_ego_with_idm  # noqa: E402
from highway_env.vehicle.behavior import IDMVehicle  # noqa: E402


class ConstantVelocityBaseline(HighwayBaseline):
    name = "constant_velocity"

    def __init__(self, target_speed: float = 25.0):
        self.target_speed = float(target_speed)

    def act(self, env: Any, observer: Any) -> np.ndarray:
        return lane_keep_action(env, target_speed=self.target_speed)

    def describe(self):
        return {"name": self.name, "target_speed": self.target_speed}


class SafeStopBaseline(HighwayBaseline):
    name = "safe_stop"

    def __init__(
        self,
        cruise_speed: float = 20.0,
        stop_gap_m: float = 18.0,
        caution_gap_m: float = 30.0,
        min_follow_speed: float = 2.0,
    ):
        self.cruise_speed = float(cruise_speed)
        self.stop_gap_m = float(stop_gap_m)
        self.caution_gap_m = float(caution_gap_m)
        self.min_follow_speed = float(min_follow_speed)

    def act(self, env: Any, observer: Any) -> np.ndarray:
        front = front_vehicle_info(env)
        target_speed = self.cruise_speed
        if front["exists"] > 0.5:
            if front["gap"] <= self.stop_gap_m:
                target_speed = 0.0
            elif front["gap"] <= self.caution_gap_m:
                frac = (front["gap"] - self.stop_gap_m) / max(
                    self.caution_gap_m - self.stop_gap_m, 1e-6
                )
                target_speed = min(
                    self.cruise_speed * max(0.0, frac),
                    max(self.min_follow_speed, front["front_speed"]),
                )
        return lane_keep_action(env, target_speed=target_speed)

    def describe(self):
        return {
            "name": self.name,
            "cruise_speed": self.cruise_speed,
            "stop_gap_m": self.stop_gap_m,
            "caution_gap_m": self.caution_gap_m,
        }


class _IDMBase(HighwayBaseline):
    enable_lane_change = True

    def reset(self, env: Any) -> None:
        uenv = env.unwrapped
        cur = uenv.controlled_vehicles[0]
        try:
            ego = replace_ego_with_idm(env, IDMVehicle)
        except Exception:
            ego = IDMVehicle(
                uenv.road,
                cur.position,
                heading=float(getattr(cur, "heading", 0.0)),
                speed=float(getattr(cur, "speed", np.linalg.norm(cur.velocity))),
                target_lane_index=getattr(cur, "lane_index", None),
                target_speed=float(getattr(cur, "speed", np.linalg.norm(cur.velocity))),
                route=getattr(cur, "route", None),
                enable_lane_change=bool(self.enable_lane_change),
            )
            try:
                idx = uenv.road.vehicles.index(cur)
                uenv.road.vehicles[idx] = ego
            except ValueError:
                uenv.road.vehicles.append(ego)
            uenv.controlled_vehicles[0] = ego
            uenv.vehicle = ego
        ego.enable_lane_change = bool(self.enable_lane_change)
        ego.target_lane_index = ego.lane_index

    def act(self, env: Any, observer: Any) -> np.ndarray:
        return idle_continuous_action()


class IDMMOBILBaseline(_IDMBase):
    name = "idm_mobil"
    enable_lane_change = True


class IDMFollowOnlyBaseline(_IDMBase):
    name = "idm_follow_only"
    enable_lane_change = False


class IDMBaseline(IDMFollowOnlyBaseline):
    name = "idm"


class MOBILIDMBaseline(IDMMOBILBaseline):
    name = "mobil_idm"
