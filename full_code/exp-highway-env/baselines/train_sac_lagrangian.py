#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .safe_rl import (
    SafetyCostConfig,
    default_env_config,
    evaluate_safe_policy,
    make_safe_env_factory,
    save_eval_row,
    save_history_json,
)


def main() -> None:
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as exc:
        raise SystemExit(
            "train_sac_lagrangian.py requires gymnasium, highway_env, and stable_baselines3."
        ) from exc

    from .safe_rl import SafeProgressCallback

    ap = argparse.ArgumentParser(description="Train a SAC-Lagrangian highway baseline.")
    ap.add_argument("--env-id", type=str, default="highway-fast-v0")
    ap.add_argument("--vehicles-count", type=int, default=15)
    ap.add_argument("--lanes-count", type=int, default=4)
    ap.add_argument("--duration", type=int, default=40)
    ap.add_argument("--policy-frequency", type=int, default=10)
    ap.add_argument("--simulation-frequency", type=int, default=30)
    ap.add_argument("--total-timesteps", type=int, default=300000)
    ap.add_argument("--chunk-timesteps", type=int, default=30000)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--buffer-size", type=int, default=100000)
    ap.add_argument("--learning-starts", type=int, default=5000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--cost-limit", type=float, default=6.0)
    ap.add_argument("--lambda-lr", type=float, default=0.05)
    ap.add_argument("--init-lambda", type=float, default=0.0)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5000)
    ap.add_argument("--out", type=str, default="checkpoints/highway_sac_lagrangian_baseline")
    ap.add_argument("--collision-weight", type=float, default=10.0)
    ap.add_argument("--offroad-weight", type=float, default=8.0)
    ap.add_argument("--clearance-weight", type=float, default=1.0)
    ap.add_argument("--ttc-weight", type=float, default=1.0)
    ap.add_argument("--lane-center-weight", type=float, default=0.25)
    ap.add_argument("--safe-clearance-m", type=float, default=8.0)
    ap.add_argument("--ttc-safe-s", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = default_env_config(
        vehicles_count=args.vehicles_count,
        lanes_count=args.lanes_count,
        duration=args.duration,
        policy_frequency=args.policy_frequency,
        simulation_frequency=args.simulation_frequency,
    )
    cost_cfg = SafetyCostConfig(
        collision_weight=args.collision_weight,
        offroad_weight=args.offroad_weight,
        clearance_weight=args.clearance_weight,
        ttc_weight=args.ttc_weight,
        lane_center_weight=args.lane_center_weight,
        safe_clearance_m=args.safe_clearance_m,
        ttc_safe_s=args.ttc_safe_s,
    )
    make_env = make_safe_env_factory(
        env_id=args.env_id,
        config=config,
        cost_cfg=cost_cfg,
        penalty_weight=args.init_lambda,
    )

    print("[sac_lagrangian] config", flush=True)
    print(f"  env_id: {args.env_id}", flush=True)
    print(f"  out: {out_dir}", flush=True)
    print(f"  total_timesteps: {args.total_timesteps:,}", flush=True)
    print(f"  chunk_timesteps: {args.chunk_timesteps:,}", flush=True)
    print(f"  n_envs: {args.n_envs}", flush=True)
    print(f"  device: {args.device}", flush=True)
    print(
        f"  cost_limit: {args.cost_limit:.3f}  lambda_lr: {args.lambda_lr:.4f}  "
        f"init_lambda: {args.init_lambda:.4f}",
        flush=True,
    )
    print("[sac_lagrangian] building vectorized environment...", flush=True)
    vec_env_kwargs = {}
    if args.n_envs > 1:
        vec_env_kwargs["vec_env_cls"] = SubprocVecEnv
    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed, **vec_env_kwargs)

    print("[sac_lagrangian] creating SAC model...", flush=True)
    model = SAC(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        batch_size=max(8, int(args.batch_size)),
        learning_rate=float(args.learning_rate),
        buffer_size=max(1000, int(args.buffer_size)),
        learning_starts=max(0, int(args.learning_starts)),
        gamma=float(args.gamma),
        verbose=2,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(out_dir / "tb"),
    )

    total_done = 0
    penalty_weight = max(0.0, float(args.init_lambda))
    best_score = float("-inf")
    best_cost = float("inf")
    best_feasible_found = False
    history: list[dict[str, float]] = []
    chunk_idx = 0
    while total_done < int(args.total_timesteps):
        chunk_idx += 1
        chunk_steps = min(int(args.chunk_timesteps), int(args.total_timesteps) - total_done)
        vec_env.env_method("set_penalty_weight", penalty_weight)
        print(
            f"[sac_lagrangian] chunk {chunk_idx}: lambda={penalty_weight:.4f} "
            f"timesteps={chunk_steps:,}",
            flush=True,
        )
        callback = SafeProgressCallback(
            BaseCallback,
            tag="sac_lagrangian",
            total_timesteps=total_done + chunk_steps,
            log_every=args.log_every,
        ).build()
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False, callback=callback)
        total_done += chunk_steps

        stats = evaluate_safe_policy(
            gym=gym,
            env_id=args.env_id,
            config=config,
            model=model,
            cost_cfg=cost_cfg,
            episodes=args.eval_episodes,
            seed=args.seed + 10_000 + chunk_idx * 100,
            deterministic=True,
        )
        save_eval_row(history, chunk=chunk_idx, penalty_weight=penalty_weight, stats=stats)
        print(
            f"[sac_lagrangian] eval: reward={stats.reward_mean:8.2f} "
            f"cost={stats.episode_cost_mean:7.3f} crash={stats.collision_rate:6.1%} "
            f"offroad={stats.offroad_rate:6.1%}",
            flush=True,
        )

        feasible = stats.episode_cost_mean <= float(args.cost_limit)
        if feasible:
            score = stats.reward_mean
            if score > best_score or not best_feasible_found:
                model.save(out_dir / "best_model")
                best_score = score
                best_feasible_found = True
        elif (not best_feasible_found) and stats.episode_cost_mean < best_cost:
            model.save(out_dir / "best_model")

        if stats.episode_cost_mean < best_cost:
            best_cost = stats.episode_cost_mean
            if not best_feasible_found:
                model.save(out_dir / "best_model")

        penalty_weight = max(
            0.0,
            penalty_weight + float(args.lambda_lr) * (stats.episode_cost_mean - float(args.cost_limit)),
        )
        model.save(out_dir / "latest_model")

    print("[sac_lagrangian] saving final checkpoint...", flush=True)
    model.save(out_dir / "model")
    save_history_json(
        out_dir / "history.json",
        {
            "trainer": "sac_lagrangian",
            "config": vars(args),
            "cost_cfg": vars(cost_cfg),
            "history": history,
        },
    )
    print(f"Saved SAC-Lagrangian model to {out_dir / 'model.zip'}", flush=True)


if __name__ == "__main__":
    main()
