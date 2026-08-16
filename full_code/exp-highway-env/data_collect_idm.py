#!/usr/bin/env python3
"""
data_collect_idm.py — Step 5, component 1.

Collects IDM-driven trajectories from highway-env scenarios and dumps a torch
dataset suitable for Stage 1 imitation training.

Locked design choices (per handoff §5–6):
    H_target  = 20 surrogate steps
    dt        = 0.1 s
    so o_tgt, v_tgt are the IDM ego state 2.0 s ahead of the observation.
    env_id    = configurable; highway-v0 is the default Stage 1 source.

To make the env step granularity match the surrogate dt, we run highway-env
at policy_frequency=10 Hz (default is 1 Hz). Each env.step() then advances
exactly 0.1 s of real time, so the imitation target at offset H is the
straightforward t+H entry in the episode log.

The "ego" we observe is forced to be an IDMVehicle (not the default
MDPVehicle): we call IDMVehicle.create_from on the controlled vehicle
right after reset and swap it into road.vehicles + controlled_vehicles +
env.unwrapped.vehicle. IDM then drives itself; we pass IDLE actions to
env.step which IDM silently ignores during road.act().

Dataset format
--------------
Per episode: ep_{idx:05d}.pt
    {
      "episode_idx": int,
      "seed":        int,
      "env_id":      str,
      "policy_frequency": 10,
      "samples": [
         {
           "step_idx": int,
           "obs":      {... 12 keys, all torch.float32 (or bool / int32) ...},
           "o_tgt":    torch.Tensor (2,) float32,
           "v_tgt":    torch.Tensor (2,) float32,
           "o_next":   torch.Tensor (2,) float32,
           "v_next":   torch.Tensor (2,) float32,
           "accel_tgt": torch.Tensor () float32,
           "steer_tgt": torch.Tensor () float32,
           "has_action_tgt": torch.Tensor () float32 in {0, 1},
         },
         ...
      ],
    }

Plus manifest.json at out_dir root:
    [{"path": "ep_00000.pt", "episode_idx": 0, "n_samples": 195,
      "split": "train", ...}, ...]

Usage
-----
    python data_collect_idm.py --out runs/stage1_data --episodes 100
    python data_collect_idm.py --env highway-slow-leader-v0 --out runs/stage1_data_slow_leader --episodes 20
    python data_collect_idm.py --smoke-test   # 2 quick episodes, prints sanity checks
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LOCAL_HIGHWAY_ENV = HERE / "HighwayEnv"
if LOCAL_HIGHWAY_ENV.exists():
    sys.path.insert(0, str(LOCAL_HIGHWAY_ENV))
sys.path.insert(0, str(HERE))

from env_wrapper import HighwayMaterialObservation, WrapperConfig  # noqa: E402
from bicycle_surrogate import ACCEL_RANGE, STEER_RANGE, force_to_action  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Constants (these are LOCKED — do not change without re-deriving the target.)
# ─────────────────────────────────────────────────────────────────────────────

H_TARGET = 20            # surrogate horizon, in env-steps (= obs steps)
DT_TARGET = 0.1          # seconds per env-step
POLICY_FREQ = 10         # Hz — chosen so each env.step is exactly DT_TARGET
SIM_FREQ = 30            # Hz — 3 sub-steps per policy step, smooth IDM dynamics


# ─────────────────────────────────────────────────────────────────────────────
# Imports that need highway-env to be importable
# ─────────────────────────────────────────────────────────────────────────────

def _import_highway_env():
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401  registers env ids
        from highway_env.vehicle.behavior import IDMVehicle
        return gym, IDMVehicle
    except ImportError as exc:
        raise SystemExit(
            "highway_env / gymnasium not importable. Either pip install or:\n"
            "    python -m pip install -e exp-highway-env/HighwayEnv\n"
            f"Original error: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Env construction and IDM swap
# ─────────────────────────────────────────────────────────────────────────────

def make_env(gym: Any, env_id: str = "highway-v0", *, vehicles_count: int = 50,
             lanes_count: int = 4, render_mode: str | None = None):
    """Construct a highway-env tuned for 0.1s-resolution data collection."""
    config = {
        "policy_frequency":     POLICY_FREQ,
        "simulation_frequency": SIM_FREQ,
        "vehicles_count":       vehicles_count,
        "lanes_count":          lanes_count,
        # Long enough to give us ~200 obs samples / episode after target offset:
        "duration":             40,
        # Default ContinuousAction would be wrong here; the IDM-driven ego
        # ignores actions anyway, but DiscreteMetaAction is what env.step
        # expects shape-wise.
        "action": {"type": "DiscreteMetaAction"},
    }
    return gym.make(env_id, config=config, render_mode=render_mode)


def _idle_action(env: Any) -> int:
    action_type = getattr(env.unwrapped, "action_type", None)
    indexes = getattr(action_type, "actions_indexes", None)
    if indexes and "IDLE" in indexes:
        return int(indexes["IDLE"])
    return 0


def replace_ego_with_idm(env: Any, IDMVehicle) -> Any:
    """In-place: swap the controlled vehicle for an IDMVehicle.

    After this call, env.step()'s action is silently ignored by the ego
    (IDMVehicle decides for itself during road.act()), and observations
    via env.unwrapped.vehicle return the IDM driver's state.
    """
    uenv = env.unwrapped
    ego = uenv.controlled_vehicles[0]
    idm = IDMVehicle.create_from(ego)

    # Replace in road.vehicles
    try:
        idx = uenv.road.vehicles.index(ego)
        uenv.road.vehicles[idx] = idm
    except ValueError:
        # Old ego wasn't in the list (shouldn't happen, but be defensive)
        uenv.road.vehicles.append(idm)

    # Replace in controlled-vehicles list and env.unwrapped.vehicle
    uenv.controlled_vehicles[0] = idm
    uenv.vehicle = idm
    return idm


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode collection
# ─────────────────────────────────────────────────────────────────────────────

def _obs_to_torch(obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Convert numpy obs_dict to torch tensors, preserving dtypes."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                out[k] = torch.from_numpy(v.copy()).bool()
            elif v.dtype == np.int32 or v.dtype == np.int64:
                out[k] = torch.from_numpy(v.copy()).long()
            else:
                out[k] = torch.from_numpy(v.astype(np.float32, copy=False))
        elif isinstance(v, (np.floating, float)):
            out[k] = torch.tensor(float(v), dtype=torch.float32)
        elif isinstance(v, (np.integer, int)):
            out[k] = torch.tensor(int(v), dtype=torch.long)
        else:
            out[k] = v  # pass-through; should not happen for the obs dict
    return out


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


def _first_step_action_target(
    v0_np: np.ndarray, v1_np: np.ndarray, dt: float = DT_TARGET,
) -> Dict[str, torch.Tensor]:
    """Approximate IDM's first deploy action from adjacent velocities."""
    v0 = torch.as_tensor(v0_np, dtype=torch.float32).view(1, 2)
    v1 = torch.as_tensor(v1_np, dtype=torch.float32).view(1, 2)
    F_tgt = (v1 - v0) / float(dt)
    speed0 = torch.linalg.norm(v0, dim=-1).clamp_min(1e-3)
    heading0 = torch.atan2(v0[:, 1], v0[:, 0])
    accel, steer = force_to_action(F_tgt, heading0, speed0)
    return {
        "accel_tgt": accel.squeeze(0).clamp(*ACCEL_RANGE),
        "steer_tgt": steer.squeeze(0).clamp(*STEER_RANGE),
        "has_action_tgt": torch.tensor(1.0, dtype=torch.float32),
    }


def collect_episode(
    env: Any, IDMVehicle, observer: HighwayMaterialObservation,
    *, seed: int, max_steps: int,
) -> List[Dict[str, Any]]:
    """Run one IDM-driven episode and return a list of (obs, o_tgt, v_tgt) dicts.

    The list has length up to (effective_T - H_TARGET); the last H_TARGET
    samples are dropped because their target lies past episode end.
    """
    _reset(env, seed)
    ego = replace_ego_with_idm(env, IDMVehicle)

    idle = _idle_action(env)

    # Phase 1: roll the episode and record (obs_dict, ego_pos, ego_vel) per step.
    log: List[Dict[str, Any]] = []
    for step_idx in range(max_steps):
        obs_dict = observer.build(env)
        ego_pos = np.asarray(env.unwrapped.vehicle.position, dtype=np.float32).copy()
        ego_vel = np.asarray(env.unwrapped.vehicle.velocity, dtype=np.float32).copy()
        log.append({"step_idx": step_idx, "obs": obs_dict,
                    "ego_pos": ego_pos, "ego_vel": ego_vel})
        _, _, term, trunc, _ = _step(env, idle)
        if term or trunc:
            break

    # Phase 2: build H-step targets plus the first-step deploy target.
    samples: List[Dict[str, Any]] = []
    for t in range(len(log) - H_TARGET):
        s = log[t]
        nxt = log[t + 1]
        tgt = log[t + H_TARGET]
        action_tgt = _first_step_action_target(s["ego_vel"], nxt["ego_vel"])
        samples.append({
            "step_idx": s["step_idx"],
            "obs": _obs_to_torch(s["obs"]),
            "o_tgt": torch.from_numpy(tgt["ego_pos"]),
            "v_tgt": torch.from_numpy(tgt["ego_vel"]),
            "o_next": torch.from_numpy(nxt["ego_pos"]),
            "v_next": torch.from_numpy(nxt["ego_vel"]),
            **action_tgt,
        })
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def _split_for(idx: int, n_total: int) -> str:
    # 80/10/10 train/val/test — interleaved by episode_idx for reproducibility
    # under partial collection runs (early-stopped run still covers all splits).
    r = idx % 10
    if r < 8:
        return "train"
    if r == 8:
        return "val"
    return "test"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",          type=str, default="runs/stage1_data")
    ap.add_argument("--env",          type=str, default="highway-v0",
                    help="Gymnasium env id to collect from.")
    ap.add_argument("--episodes",     type=int, default=100)
    ap.add_argument("--max-steps",    type=int, default=200,
                    help="Per-episode env step cap (each step is 0.1s).")
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--vehicles-count", type=int, default=50)
    ap.add_argument("--lanes-count",  type=int, default=4)
    ap.add_argument("--n-max-vehicles", type=int, default=15)
    ap.add_argument("--smoke-test",   action="store_true",
                    help="Run 2 short episodes and print sanity checks.")
    args = ap.parse_args()

    if args.smoke_test:
        args.episodes = 2
        args.max_steps = 60
        args.out = args.out.rstrip("/") + "_smoke"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gym, IDMVehicle = _import_highway_env()
    env = make_env(gym, args.env, vehicles_count=args.vehicles_count,
                    lanes_count=args.lanes_count)

    # Wrapper config: surrogate dt/H must match what we're actually collecting.
    cfg = WrapperConfig(
        n_max_vehicles=args.n_max_vehicles,
        dt_surrogate=DT_TARGET,
        horizon_surrogate=H_TARGET,
    )
    observer = HighwayMaterialObservation(cfg)

    print(f"Collecting {args.episodes} episodes from {args.env} → {out_dir}")
    print(f"  policy_frequency={POLICY_FREQ}Hz  sim_frequency={SIM_FREQ}Hz")
    print(f"  H_TARGET={H_TARGET} steps × dt={DT_TARGET}s = "
          f"{H_TARGET*DT_TARGET:.1f}s lookahead")

    manifest: List[Dict[str, Any]] = []
    t_start = time.time()

    for ep in range(args.episodes):
        seed = args.seed + ep
        try:
            samples = collect_episode(
                env, IDMVehicle, observer,
                seed=seed, max_steps=args.max_steps,
            )
        except Exception as exc:
            print(f"  ep {ep:5d} seed={seed} FAILED: {exc}")
            continue

        if not samples:
            print(f"  ep {ep:5d} seed={seed} produced 0 samples (episode too short)")
            continue

        path = out_dir / f"ep_{ep:05d}.pt"
        torch.save({
            "episode_idx": ep,
            "seed": seed,
            "env_id": args.env,
            "policy_frequency": POLICY_FREQ,
            "h_target": H_TARGET,
            "dt": DT_TARGET,
            "samples": samples,
        }, path)

        record = {
            "path": str(path.relative_to(out_dir)),
            "episode_idx": ep,
            "seed": seed,
            "env_id": args.env,
            "n_samples": len(samples),
            "split": _split_for(ep, args.episodes),
        }
        manifest.append(record)

        if args.smoke_test or ep % 10 == 0 or ep == args.episodes - 1:
            elapsed = time.time() - t_start
            print(f"  ep {ep:5d} seed={seed} samples={len(samples):3d} "
                  f"split={record['split']} elapsed={elapsed:6.1f}s")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_total_samples = sum(r["n_samples"] for r in manifest)
    n_train = sum(r["n_samples"] for r in manifest if r["split"] == "train")
    print(f"\nDone. {len(manifest)} episodes, {n_total_samples} samples total "
          f"({n_train} train).")
    print(f"Manifest: {out_dir / 'manifest.json'}")

    # Smoke test sanity checks
    if args.smoke_test and manifest:
        print("\n── Smoke test sanity checks ──")
        first = torch.load(out_dir / manifest[0]["path"], weights_only=False)
        s0 = first["samples"][0]
        obs = s0["obs"]
        expected_keys = {"o0", "v0", "goal", "C", "R", "W", "mask",
                          "risk_patch", "rollout_patch", "d_hat", "dt", "H"}
        assert set(obs.keys()) >= expected_keys, \
            f"Missing obs keys: {expected_keys - set(obs.keys())}"
        assert obs["risk_patch"].shape[0] == 2, "risk_patch must be 2-channel"
        assert obs["rollout_patch"].shape[0] == 6, "rollout_patch must be 6-channel"
        assert obs["mask"].dtype == torch.bool
        assert s0["o_tgt"].shape == (2,) and s0["v_tgt"].shape == (2,)
        assert s0["o_next"].shape == (2,) and s0["v_next"].shape == (2,)
        assert s0["accel_tgt"].ndim == 0 and s0["steer_tgt"].ndim == 0
        # Target distance: at highway speeds (~25 m/s) over 2.0s, ego should
        # have moved ~50m. Sanity-check we're getting non-trivial displacement.
        disp = (s0["o_tgt"] - obs["o0"]).norm().item()
        print(f"  ep0 sample0:")
        print(f"    o0={obs['o0'].tolist()}  o_tgt={s0['o_tgt'].tolist()}")
        print(f"    v0={obs['v0'].tolist()}  v_tgt={s0['v_tgt'].tolist()}")
        print(f"    first-step target: accel={float(s0['accel_tgt']):+.3f} "
              f"steer={float(s0['steer_tgt']):+.4f}")
        print(f"    |o_tgt - o0| = {disp:.2f} m  (expect ~30–60m at highway speed)")
        if disp < 5.0:
            print("    WARNING: target displacement is very small — check that "
                  "policy_frequency is taking effect and IDM is actually moving.")
        print(f"    risk_patch range = [{obs['risk_patch'].min():.4f}, "
              f"{obs['risk_patch'].max():.4f}]")
        print(f"    valid neighbours = {int(obs['mask'].sum())}/"
              f"{obs['mask'].numel()}")
        print("  All shape/dtype checks passed.")

    env.close()


if __name__ == "__main__":
    main()
