#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .learned import SB3PPOBaseline


def main() -> None:
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as exc:
        raise SystemExit(
            "train_ppo.py requires gymnasium, highway_env, and stable_baselines3."
        ) from exc

    class ProgressCallback(BaseCallback):
        def __init__(self, total_timesteps: int, log_every: int):
            super().__init__()
            self.total_timesteps = max(1, int(total_timesteps))
            self.log_every = max(1, int(log_every))
            self._last_print = 0
            self._start_time = 0.0

        def _on_training_start(self) -> None:
            self._start_time = time.time()
            print(
                f"[ppo] training started: total_timesteps={self.total_timesteps:,}, "
                f"log_every={self.log_every:,}",
                flush=True,
            )

        def _on_step(self) -> bool:
            if (self.num_timesteps - self._last_print) < self.log_every:
                return True
            self._last_print = self.num_timesteps
            elapsed = max(time.time() - self._start_time, 1e-6)
            pct = 100.0 * min(1.0, self.num_timesteps / self.total_timesteps)
            fps = self.num_timesteps / elapsed
            print(
                f"[ppo] progress: {self.num_timesteps:,}/{self.total_timesteps:,} "
                f"steps ({pct:5.1f}%) | elapsed={elapsed:7.1f}s | fps={fps:7.1f}",
                flush=True,
            )
            return True

        def _on_training_end(self) -> None:
            elapsed = max(time.time() - self._start_time, 1e-6)
            print(
                f"[ppo] training finished in {elapsed:.1f}s "
                f"at {self.num_timesteps:,} steps",
                flush=True,
            )

    ap = argparse.ArgumentParser(description="Train an SB3 PPO highway baseline.")
    ap.add_argument("--env-id", type=str, default="highway-fast-v0")
    ap.add_argument("--vehicles-count", type=int, default=15)
    ap.add_argument("--lanes-count", type=int, default=4)
    ap.add_argument("--total-timesteps", type=int, default=200000)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--duration", type=int, default=40)
    ap.add_argument("--policy-frequency", type=int, default=10)
    ap.add_argument("--simulation-frequency", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument(
        "--rollout-steps",
        type=int,
        default=128,
        help="PPO rollout steps per environment before each update.",
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="checkpoints/highway_ppo_baseline")
    ap.add_argument("--log-every", type=int, default=10000)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "vehicles_count": args.vehicles_count,
        "lanes_count": args.lanes_count,
        "duration": args.duration,
        "policy_frequency": args.policy_frequency,
        "simulation_frequency": args.simulation_frequency,
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
        },
    }
    config.update(SB3PPOBaseline.observation_config())

    def make_env():
        return gym.make(args.env_id, config=config)

    print("[ppo] config", flush=True)
    print(f"  env_id: {args.env_id}", flush=True)
    print(f"  out: {out_dir}", flush=True)
    print(f"  total_timesteps: {args.total_timesteps:,}", flush=True)
    print(f"  n_envs: {args.n_envs}", flush=True)
    print(f"  seed: {args.seed}", flush=True)
    print(f"  device: {args.device}", flush=True)
    print(
        f"  vehicles_count: {args.vehicles_count}  lanes_count: {args.lanes_count}",
        flush=True,
    )
    print(
        f"  duration: {args.duration}  policy_frequency: {args.policy_frequency}  "
        f"simulation_frequency: {args.simulation_frequency}",
        flush=True,
    )
    print(
        f"  batch_size: {args.batch_size}  n_epochs: {args.n_epochs}  "
        f"learning_rate: {args.learning_rate:g}  gamma: {args.gamma:g}",
        flush=True,
    )
    print(f"  rollout_steps_per_env: {args.rollout_steps}", flush=True)
    print("[ppo] building vectorized environment...", flush=True)
    vec_env_kwargs = {}
    if args.n_envs > 1:
        vec_env_kwargs["vec_env_cls"] = SubprocVecEnv
    vec_env = make_vec_env(
        make_env,
        n_envs=args.n_envs,
        seed=args.seed,
        **vec_env_kwargs,
    )
    print("[ppo] creating PPO model...", flush=True)
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        n_steps=max(8, int(args.rollout_steps)),
        batch_size=max(8, int(args.batch_size)),
        n_epochs=max(1, int(args.n_epochs)),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        verbose=2,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(out_dir / "tb"),
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=ProgressCallback(
            total_timesteps=args.total_timesteps,
            log_every=args.log_every,
        ),
    )
    print("[ppo] saving checkpoint...", flush=True)
    model.save(out_dir / "model")
    print(f"Saved PPO model to {out_dir / 'model'}")


if __name__ == "__main__":
    main()
