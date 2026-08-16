from __future__ import annotations

import abc
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCAL_HIGHWAY_ENV = ROOT / "HighwayEnv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if LOCAL_HIGHWAY_ENV.exists() and str(LOCAL_HIGHWAY_ENV) not in sys.path:
    sys.path.insert(0, str(LOCAL_HIGHWAY_ENV))

from bicycle_surrogate import ACCEL_RANGE, STEER_RANGE  # noqa: E402


def clip_action_physical(accel_phys: float, steer_phys: float) -> Tuple[float, float]:
    accel = float(np.clip(accel_phys, ACCEL_RANGE[0], ACCEL_RANGE[1]))
    steer = float(np.clip(steer_phys, STEER_RANGE[0], STEER_RANGE[1]))
    return accel, steer


def normalize_action(accel_phys: float, steer_phys: float) -> np.ndarray:
    accel, steer = clip_action_physical(accel_phys, steer_phys)
    a_lo, a_hi = ACCEL_RANGE
    s_lo, s_hi = STEER_RANGE
    a = float(np.clip(2.0 * (accel - a_lo) / (a_hi - a_lo) - 1.0, -1.0, 1.0))
    s = float(np.clip(2.0 * (steer - s_lo) / (s_hi - s_lo) - 1.0, -1.0, 1.0))
    return np.array([a, s], dtype=np.float32)


def current_lane_index(env: Any):
    return getattr(env.unwrapped.vehicle, "lane_index", None)


def lane_center_y(env: Any, lane_index=None) -> float:
    uenv = env.unwrapped
    if lane_index is None:
        lane_index = current_lane_index(env)
    if lane_index is None:
        pos = np.asarray(uenv.vehicle.position, dtype=np.float64)
        return float(round(pos[1] / 4.0) * 4.0)
    lane = uenv.road.network.get_lane(lane_index)
    pos = np.asarray(uenv.vehicle.position, dtype=np.float64)
    s, _ = lane.local_coordinates(pos)
    center = lane.position(s, 0.0)
    return float(center[1])


def lane_keep_action(env: Any, *, target_speed: float, target_lane_index=None) -> np.ndarray:
    uenv = env.unwrapped
    vehicle = uenv.vehicle
    lane_idx = target_lane_index if target_lane_index is not None else current_lane_index(env)
    if lane_idx is None:
        lane_idx = getattr(vehicle, "target_lane_index", None)
    if lane_idx is None:
        return normalize_action(0.0, 0.0)

    lane = uenv.road.network.get_lane(lane_idx)
    pos = np.asarray(vehicle.position, dtype=np.float64)
    speed = float(getattr(vehicle, "speed", np.linalg.norm(vehicle.velocity)))
    heading = float(getattr(vehicle, "heading", 0.0))

    # Mirror highway-env's ControlledVehicle lane-centering controller so these
    # baselines behave consistently even when the live ego vehicle class does
    # not expose steering_control()/speed_control().
    tau_heading = 0.2
    tau_lateral = 0.6
    tau_pursuit = 0.5 * tau_heading
    kp_acc = 1.0 / 0.6
    kp_heading = 1.0 / tau_heading
    kp_lateral = 1.0 / tau_lateral
    max_steer = np.pi / 3.0
    vehicle_length = float(getattr(vehicle, "LENGTH", 5.0))

    lane_coords = lane.local_coordinates(pos)
    lane_next_coords = lane_coords[0] + speed * tau_pursuit
    lane_future_heading = lane.heading_at(lane_next_coords)
    lateral_speed_command = -kp_lateral * lane_coords[1]
    heading_command = np.arcsin(
        np.clip(lateral_speed_command / max(speed, 1e-3), -1.0, 1.0)
    )
    heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4.0, np.pi / 4.0)
    heading_rate_command = kp_heading * _wrap_to_pi(heading_ref - heading)
    slip_angle = np.arcsin(
        np.clip(vehicle_length / 2.0 / max(speed, 1e-3) * heading_rate_command, -1.0, 1.0)
    )
    steer = float(np.clip(np.arctan(2.0 * np.tan(slip_angle)), -max_steer, max_steer))

    accel = float(kp_acc * (float(target_speed) - speed))
    return normalize_action(accel, steer)


def idle_continuous_action() -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def get_lane_width(env: Any) -> float:
    lane_idx = current_lane_index(env)
    if lane_idx is None:
        return 4.0
    lane = env.unwrapped.road.network.get_lane(lane_idx)
    pos = np.asarray(env.unwrapped.vehicle.position, dtype=np.float64)
    try:
        s, _ = lane.local_coordinates(pos)
        return float(lane.width_at(s))
    except Exception:
        return 4.0


def get_lane_count(env: Any) -> int:
    lane_idx = current_lane_index(env)
    if lane_idx is None:
        return int(env.unwrapped.config.get("lanes_count", 4))
    _from, _to, _id = lane_idx
    try:
        return len(env.unwrapped.road.network.graph[_from][_to])
    except Exception:
        return int(env.unwrapped.config.get("lanes_count", 4))


def lane_id_from_index(lane_index: Any) -> int:
    if lane_index is None:
        return 0


def _wrap_to_pi(x: float) -> float:
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)
    try:
        return int(lane_index[2])
    except Exception:
        return 0


def front_vehicle_info(env: Any, lane_index=None) -> Dict[str, float]:
    uenv = env.unwrapped
    ego = uenv.vehicle
    lane_idx = lane_index if lane_index is not None else current_lane_index(env)
    front, _ = uenv.road.neighbour_vehicles(ego, lane_idx)
    if front is None:
        return {
            "exists": 0.0,
            "gap": float("inf"),
            "front_speed": float("inf"),
            "closing_speed": 0.0,
            "ttc": float("inf"),
        }

    gap = float(ego.lane_distance_to(front))
    front_speed = float(getattr(front, "speed", np.linalg.norm(front.velocity)))
    ego_speed = float(getattr(ego, "speed", np.linalg.norm(ego.velocity)))
    closing_speed = max(0.0, ego_speed - front_speed)
    ttc = float(gap / max(closing_speed, 1e-6)) if closing_speed > 1e-6 else float("inf")
    return {
        "exists": 1.0,
        "gap": gap,
        "front_speed": front_speed,
        "closing_speed": closing_speed,
        "ttc": ttc,
    }


class HighwayBaseline(abc.ABC):
    """Shared interface for highway-env baselines."""

    name: str = "baseline"

    def env_config_overrides(self) -> Dict[str, Any]:
        return {}

    def reset(self, env: Any) -> None:
        pass

    @abc.abstractmethod
    def act(self, env: Any, observer: Any) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name}


def merge_env_config(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not override:
        return dict(base)
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            merged = dict(out[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return out
