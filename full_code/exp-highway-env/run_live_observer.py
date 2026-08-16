#!/usr/bin/env python3
"""
Run Step 1 + Step 2 against a real highway-env environment.

This is a smoke test for the live highway-env API surface before Step 3
touches actions or training. It imports a real env, builds the DFC-format
observation dict at every policy step, and prints the checks reviewers will
ask about: ego/world frame alignment, lane-center goal behavior, vehicle-list
conversion, risk-patch ranges, and rollout-patch gradient channels.

Examples
--------
python exp-highway-env/run_live_observer.py --env highway-v0 --steps 10
python exp-highway-env/run_live_observer.py --env merge-v0 --steps 20 --action random
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
LOCAL_HIGHWAY_ENV = HERE / "HighwayEnv"
if LOCAL_HIGHWAY_ENV.exists():
    sys.path.insert(0, str(LOCAL_HIGHWAY_ENV))
sys.path.insert(0, str(HERE))

from env_wrapper import (  # noqa: E402
    HighwayMaterialObservation,
    WrapperConfig,
    _ego_lane_center_y,
)


def _import_gymnasium():
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401  Registers highway-env ids.

        return gym
    except ImportError as exc:
        raise SystemExit(
            "Could not import gymnasium/highway_env. Install the local checkout first:\n"
            "  python -m pip install -e exp-highway-env/HighwayEnv\n"
            f"Original import error: {exc}"
        ) from exc


def _reset(env: Any, seed: int):
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def _step(env: Any, action: Any):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated), bool(truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), False, info


def _idle_action(env: Any) -> int:
    action_type = getattr(env.unwrapped, "action_type", None)
    indexes = getattr(action_type, "actions_indexes", None)
    if indexes and "IDLE" in indexes:
        return int(indexes["IDLE"])
    return 0


def _choose_action(env: Any, mode: str, step_idx: int) -> Any:
    if mode == "random":
        return env.action_space.sample()
    if mode == "cycle":
        n = getattr(env.action_space, "n", 1)
        return int(step_idx % max(int(n), 1))
    return _idle_action(env)


def _format_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(x):7.3f}" for x in v) + "]"


def _print_header(env: Any, args: argparse.Namespace, cfg: WrapperConfig) -> None:
    uenv = env.unwrapped
    print("Live highway-env observer smoke test")
    print(f"  env={args.env} seed={args.seed} steps={args.steps} action={args.action}")
    print(
        "  surrogate="
        f"dt={cfg.dt_surrogate:.3f}s H={cfg.horizon_surrogate} "
        f"goal_lookahead={cfg.goal_lookahead_m:.1f}m "
        f"n_max={cfg.n_max_vehicles}"
    )
    print(
        "  env_config="
        f"sim_freq={uenv.config.get('simulation_frequency')}Hz "
        f"policy_freq={uenv.config.get('policy_frequency')}Hz "
        f"vehicles={uenv.config.get('vehicles_count', 'n/a')} "
        f"lanes={uenv.config.get('lanes_count', 'n/a')}"
    )
    print()


def _check_observation(obs: dict[str, np.ndarray], cfg: WrapperConfig) -> None:
    expected_shapes = {
        "o0": (2,),
        "v0": (2,),
        "goal": (2,),
        "C": (cfg.n_max_vehicles, 2),
        "R": (cfg.n_max_vehicles,),
        "W": (cfg.n_max_vehicles,),
        "mask": (cfg.n_max_vehicles,),
        "risk_patch": (2, cfg.risk_field.patch_lat_cells, cfg.risk_field.patch_lon_cells),
        "rollout_patch": (6, cfg.risk_field.patch_lat_cells, cfg.risk_field.patch_lon_cells),
    }
    for key, shape in expected_shapes.items():
        if obs[key].shape != shape:
            raise AssertionError(f"{key}: expected shape {shape}, got {obs[key].shape}")

    if obs["risk_patch"].dtype != np.float32:
        raise AssertionError(f"risk_patch dtype must be float32, got {obs['risk_patch'].dtype}")
    if obs["rollout_patch"].dtype != np.float32:
        raise AssertionError(
            f"rollout_patch dtype must be float32, got {obs['rollout_patch'].dtype}"
        )
    if obs["mask"].dtype != np.bool_:
        raise AssertionError(f"mask dtype must be bool, got {obs['mask'].dtype}")

    if not np.allclose(obs["risk_patch"][0], obs["rollout_patch"][0]):
        raise AssertionError("risk_patch[0] must equal rollout_patch risk channel")
    expected_hard = (obs["rollout_patch"][1] <= 0.0).astype(np.float32)
    if not np.allclose(obs["risk_patch"][1], expected_hard):
        raise AssertionError("risk_patch[1] must be hard mask derived from phi")


def _print_step(env: Any, step_idx: int, action: Any, obs: dict[str, np.ndarray]) -> None:
    ego = env.unwrapped.vehicle
    lane_y, lane_source = _ego_lane_center_y(env, obs["o0"], return_source=True)

    patch = obs["rollout_patch"]
    risk = obs["risk_patch"][0]
    hard = obs["risk_patch"][1]
    phi = patch[1]
    ego_r = obs["rollout_patch"].shape[1] // 2
    ego_c = max(0, int(round(0.05 * obs["rollout_patch"].shape[2])))
    grad_risk_ego = patch[2:4, ego_r, ego_c]
    grad_phi_ego = patch[4:6, ego_r, ego_c]

    print(
        f"step={step_idx:03d} action={action!s:>5} "
        f"ego={_format_vec(obs['o0'])} "
        f"heading={float(ego.heading): .4f} lane={getattr(ego, 'lane_index', None)} "
        f"goal={_format_vec(obs['goal'])} lane_y={lane_y:7.3f}({lane_source})"
    )
    print(
        f"          vehicles={len(env.unwrapped.road.vehicles):3d} "
        f"valid_obs={int(obs['mask'].sum()):2d} "
        f"risk=[{float(risk.min()):.4f},{float(risk.max()):.4f}] "
        f"hard_px={int(hard.sum()):4d} "
        f"phi=[{float(phi.min()):7.3f},{float(phi.max()):7.3f}] "
        f"grad_risk@ego={_format_vec(grad_risk_ego)} "
        f"grad_phi@ego={_format_vec(grad_phi_ego)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="highway-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--action", choices=["idle", "random", "cycle"], default="idle")
    parser.add_argument("--vehicles-count", type=int, default=None)
    parser.add_argument("--lanes-count", type=int, default=None)
    parser.add_argument("--policy-frequency", type=int, default=None)
    parser.add_argument("--simulation-frequency", type=int, default=None)
    parser.add_argument("--goal-lookahead-m", type=float, default=30.0)
    parser.add_argument("--n-max-vehicles", type=int, default=15)
    parser.add_argument("--sensing-radius-m", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gym = _import_gymnasium()

    env_config: dict[str, Any] = {}
    if args.vehicles_count is not None:
        env_config["vehicles_count"] = args.vehicles_count
    if args.lanes_count is not None:
        env_config["lanes_count"] = args.lanes_count
    if args.policy_frequency is not None:
        env_config["policy_frequency"] = args.policy_frequency
    if args.simulation_frequency is not None:
        env_config["simulation_frequency"] = args.simulation_frequency

    env = gym.make(args.env, config=env_config or None)
    cfg = WrapperConfig(
        n_max_vehicles=args.n_max_vehicles,
        sensing_radius_m=args.sensing_radius_m,
        goal_lookahead_m=args.goal_lookahead_m,
    )
    cfg.risk_field.sensing_radius_m = args.sensing_radius_m
    observer = HighwayMaterialObservation(cfg)

    try:
        _reset(env, args.seed)
        _print_header(env, args, cfg)
        for step_idx in range(args.steps):
            obs_dict = observer.build(env)
            _check_observation(obs_dict, cfg)
            action = _choose_action(env, args.action, step_idx)
            _print_step(env, step_idx, action, obs_dict)
            _, _, terminated, truncated, _ = _step(env, action)
            if terminated or truncated:
                print(f"\nEpisode ended: terminated={terminated} truncated={truncated}")
                break
        print("\nLive observer smoke test passed.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
