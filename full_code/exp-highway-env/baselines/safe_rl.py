from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np

from .common import front_vehicle_info, get_lane_width
from .learned import SB3PPOBaseline


def default_env_config(
    *,
    vehicles_count: int,
    lanes_count: int,
    duration: int,
    policy_frequency: int,
    simulation_frequency: int,
) -> Dict[str, Any]:
    config = {
        "vehicles_count": int(vehicles_count),
        "lanes_count": int(lanes_count),
        "duration": int(duration),
        "policy_frequency": int(policy_frequency),
        "simulation_frequency": int(simulation_frequency),
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
        },
    }
    config.update(SB3PPOBaseline.observation_config())
    return config


@dataclass
class SafetyCostConfig:
    collision_weight: float = 10.0
    offroad_weight: float = 8.0
    clearance_weight: float = 1.0
    ttc_weight: float = 1.0
    lane_center_weight: float = 0.25
    safe_clearance_m: float = 8.0
    ttc_safe_s: float = 3.0


@dataclass
class EvaluationStats:
    reward_mean: float
    reward_std: float
    episode_cost_mean: float
    episode_cost_std: float
    collision_rate: float
    offroad_rate: float
    mean_steps: float


def _lane_center_error(env: Any) -> float:
    uenv = env.unwrapped
    ego = uenv.vehicle
    lane_index = getattr(ego, "lane_index", None)
    if lane_index is None:
        return 0.0
    lane = uenv.road.network.get_lane(lane_index)
    pos = np.asarray(ego.position, dtype=np.float64)
    try:
        _, lat = lane.local_coordinates(pos)
        lane_width = max(1e-3, get_lane_width(env))
        return float(abs(lat) / (0.5 * lane_width))
    except Exception:
        return 0.0


def compute_safety_cost(env: Any, cfg: SafetyCostConfig) -> Dict[str, float]:
    uenv = env.unwrapped
    ego = uenv.vehicle
    crashed = bool(getattr(ego, "crashed", False))
    on_road = bool(getattr(ego, "on_road", True))

    front = front_vehicle_info(env)
    gap = float(front["gap"])
    ttc = float(front["ttc"])

    clearance_violation = max(0.0, float(cfg.safe_clearance_m) - gap) if np.isfinite(gap) else 0.0
    ttc_violation = max(0.0, float(cfg.ttc_safe_s) - ttc) if np.isfinite(ttc) else 0.0
    lane_error = _lane_center_error(env)

    collision_cost = float(cfg.collision_weight) if crashed else 0.0
    offroad_cost = float(cfg.offroad_weight) if not on_road else 0.0
    clearance_cost = float(cfg.clearance_weight) * clearance_violation
    ttc_cost = float(cfg.ttc_weight) * ttc_violation
    lane_cost = float(cfg.lane_center_weight) * lane_error

    total = collision_cost + offroad_cost + clearance_cost + ttc_cost + lane_cost
    return {
        "cost": float(total),
        "collision_cost": float(collision_cost),
        "offroad_cost": float(offroad_cost),
        "clearance_cost": float(clearance_cost),
        "ttc_cost": float(ttc_cost),
        "lane_cost": float(lane_cost),
        "gap": float(gap) if np.isfinite(gap) else float("inf"),
        "ttc": float(ttc) if np.isfinite(ttc) else float("inf"),
        "on_road": 1.0 if on_road else 0.0,
        "crashed": 1.0 if crashed else 0.0,
    }


class SafeDrivingCostWrapper(gym.Wrapper):
    """Gymnasium wrapper that attaches a safety cost and optional penalty shaping."""

    def __init__(
        self,
        env: gym.Env,
        *,
        cost_cfg: Optional[SafetyCostConfig] = None,
        penalty_weight: float = 0.0,
    ):
        super().__init__(env)
        self.cost_cfg = cost_cfg or SafetyCostConfig()
        self.penalty_weight = float(penalty_weight)
        self._episode_cost = 0.0
        self._episode_reward_raw = 0.0
        self._episode_steps = 0

    def set_penalty_weight(self, value: float) -> None:
        self.penalty_weight = max(0.0, float(value))

    def get_penalty_weight(self) -> float:
        return float(self.penalty_weight)

    def reset(self, *args, **kwargs):
        self._episode_cost = 0.0
        self._episode_reward_raw = 0.0
        self._episode_steps = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_steps += 1
        self._episode_reward_raw += float(reward)

        cost_info = compute_safety_cost(self.env, self.cost_cfg)
        cost = float(cost_info["cost"])
        self._episode_cost += cost

        shaped_reward = float(reward) - float(self.penalty_weight) * cost
        info = dict(info)
        info["cost"] = cost
        info["reward_raw"] = float(reward)
        info["reward_shaped"] = float(shaped_reward)
        info["penalty_weight"] = float(self.penalty_weight)
        info["cost_breakdown"] = cost_info
        if terminated or truncated:
            info["episode_cost"] = float(self._episode_cost)
            info["episode_reward_raw"] = float(self._episode_reward_raw)
            info["episode_steps"] = int(self._episode_steps)
        return obs, shaped_reward, terminated, truncated, info


@dataclass
class SafeEnvFactory:
    env_id: str
    config: Dict[str, Any]
    cost_cfg: SafetyCostConfig
    penalty_weight: float = 0.0

    def __call__(self) -> gym.Env:
        env = gym.make(self.env_id, config=self.config)
        return SafeDrivingCostWrapper(
            env,
            cost_cfg=self.cost_cfg,
            penalty_weight=self.penalty_weight,
        )


def make_safe_env_factory(
    *,
    env_id: str,
    config: Dict[str, Any],
    cost_cfg: SafetyCostConfig,
    penalty_weight: float = 0.0,
) -> Callable[[], Any]:
    return SafeEnvFactory(
        env_id=env_id,
        config=dict(config),
        cost_cfg=cost_cfg,
        penalty_weight=float(penalty_weight),
    )


def evaluate_safe_policy(
    *,
    gym,
    env_id: str,
    config: Dict[str, Any],
    model: Any,
    cost_cfg: SafetyCostConfig,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> EvaluationStats:
    rewards = []
    costs = []
    collisions = 0
    offroads = 0
    steps = []

    for ep in range(int(episodes)):
        env = SafeDrivingCostWrapper(gym.make(env_id, config=config), cost_cfg=cost_cfg, penalty_weight=0.0)
        try:
            obs, _info = env.reset(seed=int(seed) + ep)
            done = False
            trunc = False
            ep_reward = 0.0
            ep_cost = 0.0
            ep_steps = 0
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, trunc, info = env.step(action)
                ep_reward += float(info.get("reward_raw", reward))
                ep_cost += float(info.get("cost", 0.0))
                ep_steps += 1
            rewards.append(ep_reward)
            costs.append(ep_cost)
            steps.append(ep_steps)
            breakdown = dict(info.get("cost_breakdown", {}))
            collisions += int(bool(breakdown.get("crashed", 0.0)))
            offroads += int(not bool(breakdown.get("on_road", 1.0)))
        finally:
            env.close()

    rewards_np = np.asarray(rewards, dtype=np.float64) if rewards else np.zeros(1, dtype=np.float64)
    costs_np = np.asarray(costs, dtype=np.float64) if costs else np.zeros(1, dtype=np.float64)
    steps_np = np.asarray(steps, dtype=np.float64) if steps else np.zeros(1, dtype=np.float64)
    n = max(1, len(rewards))
    return EvaluationStats(
        reward_mean=float(rewards_np.mean()),
        reward_std=float(rewards_np.std()),
        episode_cost_mean=float(costs_np.mean()),
        episode_cost_std=float(costs_np.std()),
        collision_rate=float(collisions) / float(n),
        offroad_rate=float(offroads) / float(n),
        mean_steps=float(steps_np.mean()),
    )


class SafeProgressCallback:
    def __init__(self, base_callback_cls, *, tag: str, total_timesteps: int, log_every: int):
        self.tag = tag
        self.total_timesteps = max(1, int(total_timesteps))
        self.log_every = max(1, int(log_every))
        self._callback_cls = base_callback_cls

    def build(self):
        tag = self.tag
        total_timesteps = self.total_timesteps
        log_every = self.log_every

        class _Callback(self._callback_cls):
            def __init__(self):
                super().__init__()
                self._last_print = 0
                self._start_time = 0.0

            def _on_training_start(self) -> None:
                self._start_time = time.time()
                print(
                    f"[{tag}] training started: total_timesteps={total_timesteps:,}, "
                    f"log_every={log_every:,}",
                    flush=True,
                )

            def _on_step(self) -> bool:
                if (self.num_timesteps - self._last_print) < log_every:
                    return True
                self._last_print = self.num_timesteps
                elapsed = max(time.time() - self._start_time, 1e-6)
                pct = 100.0 * min(1.0, self.num_timesteps / total_timesteps)
                fps = self.num_timesteps / elapsed
                print(
                    f"[{tag}] progress: {self.num_timesteps:,}/{total_timesteps:,} "
                    f"steps ({pct:5.1f}%) | elapsed={elapsed:7.1f}s | fps={fps:7.1f}",
                    flush=True,
                )
                return True

            def _on_training_end(self) -> None:
                elapsed = max(time.time() - self._start_time, 1e-6)
                print(
                    f"[{tag}] training finished in {elapsed:.1f}s "
                    f"at {self.num_timesteps:,} steps",
                    flush=True,
                )

        return _Callback()


def save_history_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_eval_row(history: list[Dict[str, Any]], *, chunk: int, penalty_weight: float, stats: EvaluationStats) -> None:
    row = {
        "chunk": int(chunk),
        "penalty_weight": float(penalty_weight),
        **asdict(stats),
    }
    history.append(row)
