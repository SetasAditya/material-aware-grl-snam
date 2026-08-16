#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List
import time

import numpy as np

from env_wrapper import HighwayMaterialObservation, WrapperConfig  # noqa: E402
from eval_stage2 import (  # noqa: E402
    DT_TARGET,
    H_TARGET,
    POLICY_FREQ,
    SIM_FREQ,
    SCENARIOS,
    ScenarioConfig,
    _agg,
    _detect_lane_change,
    _print_summary,
    _reset,
    _step,
    _ego_lane_center_y,
    _import_gym,
)

from .common import merge_env_config
from .registry import BASELINE_NAMES, create_baseline


@dataclass
class StepRecord:
    speed: float
    accel_phys: float
    steer_phys: float
    dmin: float
    risk_val: float
    lat_y: float
    lane_y: float
    on_road: bool


@dataclass
class EpisodeResult:
    seed: int
    scenario: str
    steps: int
    collided: bool
    truncated: bool
    distance_m: float
    mean_speed: float
    speed_std: float
    lateral_accel_mean_abs: float
    lateral_accel_p95_abs: float
    lateral_pos_std: float
    lane_keep_err_mean: float
    lane_keep_err_p95: float
    lane_keep_err_max: float
    lane_changes: int
    on_road_fraction: float
    offroad_steps: int
    went_offroad: bool
    ended_offroad: bool
    cum_risk_eval: float
    cvar_step_risk: float
    min_clearance: float
    lam_soft_mean: float = 0.0
    lam_hard_mean: float = 0.0
    F_norm_mean: float = 0.0


def make_env(gym, scenario: ScenarioConfig, *, config_override: Dict[str, Any] | None = None, offroad_terminal: bool = False):
    config = {
        "policy_frequency": POLICY_FREQ,
        "simulation_frequency": SIM_FREQ,
        "vehicles_count": scenario.vehicles_count,
        "lanes_count": scenario.lanes_count,
        "ego_spacing": scenario.ego_spacing,
        "duration": 40,
        "offroad_terminal": bool(offroad_terminal),
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
        },
    }
    config = merge_env_config(config, config_override)
    if scenario.initial_lane_id is not None:
        config["initial_lane_id"] = scenario.initial_lane_id
    return gym.make(scenario.env_id, config=config)


def _risk_at_ego(obs_np: Dict[str, np.ndarray]) -> float:
    patch = np.asarray(obs_np["rollout_patch"], dtype=np.float32)
    h = patch.shape[1] // 2
    w = patch.shape[2] // 2
    return float(patch[0, h, w])


def _min_clearance(obs_np: Dict[str, np.ndarray]) -> float:
    C = np.asarray(obs_np["C"], dtype=np.float64)
    R = np.asarray(obs_np["R"], dtype=np.float64)
    mask = np.asarray(obs_np["mask"], dtype=bool)
    o0 = np.asarray(obs_np["o0"], dtype=np.float64)
    if not mask.any():
        return float("inf")
    rel = C[mask] - o0[None, :]
    d = np.linalg.norm(rel, axis=-1) - R[mask]
    return float(d.min()) if d.size else float("inf")


def _action_phys(action_norm: np.ndarray) -> tuple[float, float]:
    accel = float(np.clip(action_norm[0], -1.0, 1.0))
    steer = float(np.clip(action_norm[1], -1.0, 1.0))
    accel_phys = 0.5 * (accel + 1.0) * (8.0 - (-6.0)) - 6.0
    steer_phys = 0.5 * (steer + 1.0) * (np.pi / 4 - (-np.pi / 4)) - (np.pi / 4)
    return accel_phys, steer_phys


def run_episode(baseline, observer, env, *, seed: int, scenario_name: str, max_steps: int) -> EpisodeResult:
    _reset(env, seed)
    baseline.reset(env)
    uenv = env.unwrapped
    o_start = np.array(uenv.vehicle.position, dtype=np.float64).copy()
    prev_lane = getattr(uenv.vehicle, "lane_index", None)
    lane_changes = 0
    collided = False
    truncated = False
    diags: List[StepRecord] = []

    for _ in range(max_steps):
        obs_np = observer.build(env)
        action = np.asarray(baseline.act(env, observer), dtype=np.float32).reshape(2)
        accel_phys, steer_phys = _action_phys(action)
        ego_pos = np.array(uenv.vehicle.position, dtype=np.float64)
        speed = float(getattr(uenv.vehicle, "speed", np.linalg.norm(uenv.vehicle.velocity)))
        try:
            lane_y = float(_ego_lane_center_y(env, ego_pos))
        except Exception:
            lane_y = float("nan")
        diags.append(
            StepRecord(
                speed=speed,
                accel_phys=accel_phys,
                steer_phys=steer_phys,
                dmin=_min_clearance(obs_np),
                risk_val=_risk_at_ego(obs_np),
                lat_y=float(ego_pos[1]),
                lane_y=lane_y,
                on_road=bool(getattr(uenv.vehicle, "on_road", True)),
            )
        )

        _, term, trunc, info = _step(env, action)
        cur_lane = getattr(uenv.vehicle, "lane_index", None)
        if _detect_lane_change(prev_lane, cur_lane):
            lane_changes += 1
        prev_lane = cur_lane
        if term:
            collided = bool(info.get("crashed", True))
            break
        if trunc:
            truncated = True
            break

    o_end = np.array(uenv.vehicle.position, dtype=np.float64).copy()
    final_on_road = bool(getattr(uenv.vehicle, "on_road", True))
    if not diags:
        return EpisodeResult(
            seed=seed,
            scenario=scenario_name,
            steps=0,
            collided=collided,
            truncated=truncated,
            distance_m=0.0,
            mean_speed=0.0,
            speed_std=0.0,
            lateral_accel_mean_abs=0.0,
            lateral_accel_p95_abs=0.0,
            lateral_pos_std=0.0,
            lane_keep_err_mean=float("nan"),
            lane_keep_err_p95=float("nan"),
            lane_keep_err_max=float("nan"),
            lane_changes=0,
            on_road_fraction=1.0 if final_on_road else 0.0,
            offroad_steps=0 if final_on_road else 1,
            went_offroad=not final_on_road,
            ended_offroad=not final_on_road,
            cum_risk_eval=0.0,
            cvar_step_risk=0.0,
            min_clearance=float("inf"),
        )

    speeds = np.asarray([d.speed for d in diags], dtype=np.float64)
    steers = np.asarray([d.steer_phys for d in diags], dtype=np.float64)
    risks = np.asarray([d.risk_val for d in diags], dtype=np.float64)
    dmins = np.asarray([d.dmin for d in diags], dtype=np.float64)
    lat_pos = np.asarray([d.lat_y for d in diags], dtype=np.float64)
    lane_y = np.asarray([d.lane_y for d in diags], dtype=np.float64)
    lke = np.abs(lat_pos - lane_y)
    lke = lke[~np.isnan(lke)]
    on_road = np.asarray([d.on_road for d in diags], dtype=np.bool_)
    on_road_with_final = np.concatenate([on_road, np.asarray([final_on_road], dtype=np.bool_)])
    offroad_steps = int((~on_road_with_final).sum())
    lat_accel = (speeds ** 2) * steers / 5.0
    abs_lat_accel = np.abs(lat_accel)

    if risks.size >= 20:
        risks_sorted = np.sort(risks)[::-1]
        n_tail = max(1, int(0.05 * risks.size))
        cvar_step = float(np.mean(risks_sorted[:n_tail]))
    else:
        cvar_step = float(risks.max()) if risks.size else 0.0

    return EpisodeResult(
        seed=seed,
        scenario=scenario_name,
        steps=len(diags),
        collided=collided,
        truncated=truncated,
        distance_m=float(np.linalg.norm(o_end - o_start)),
        mean_speed=float(speeds.mean()),
        speed_std=float(speeds.std()),
        lateral_accel_mean_abs=float(abs_lat_accel.mean()),
        lateral_accel_p95_abs=float(np.percentile(abs_lat_accel, 95)) if abs_lat_accel.size else 0.0,
        lateral_pos_std=float(lat_pos.std()),
        lane_keep_err_mean=float(lke.mean()) if lke.size else float("nan"),
        lane_keep_err_p95=float(np.percentile(lke, 95)) if lke.size else float("nan"),
        lane_keep_err_max=float(lke.max()) if lke.size else float("nan"),
        lane_changes=lane_changes,
        on_road_fraction=float(on_road_with_final.mean()) if on_road_with_final.size else 1.0,
        offroad_steps=offroad_steps,
        went_offroad=bool(offroad_steps > 0),
        ended_offroad=not final_on_road,
        cum_risk_eval=float(risks.sum()),
        cvar_step_risk=cvar_step,
        min_clearance=float(dmins.min()) if dmins.size else float("inf"),
    )


def _print_episode_progress(
    *,
    baseline_name: str,
    scenario_name: str,
    episode_idx: int,
    episodes_total: int,
    result: EpisodeResult,
    started_at: float,
) -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    status = "collision" if result.collided else "truncated" if result.truncated else "ok"
    print(
        f"[{baseline_name} | {scenario_name}] episode "
        f"{episode_idx}/{episodes_total} finished in {elapsed:.1f}s "
        f"| steps={result.steps:3d} | status={status:9s} "
        f"| speed={result.mean_speed:5.2f} | dmin={result.min_clearance:5.2f}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate baselines on highway-env scenarios.")
    ap.add_argument("--baselines", nargs="+", choices=BASELINE_NAMES, required=True)
    ap.add_argument("--scenarios", nargs="+", default=["default", "authored_slow_leader", "authored_slow_leader_boxed"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--s1-ckpt", type=str, default="")
    ap.add_argument("--s2-ckpt", type=str, default="")
    ap.add_argument("--ppo-ckpt", type=str, default="")
    ap.add_argument("--sac-ckpt", type=str, default="")
    ap.add_argument("--ppo-lagrangian-ckpt", type=str, default="")
    ap.add_argument("--sac-lagrangian-ckpt", type=str, default="")
    ap.add_argument("--cpo-ckpt", type=str, default="")
    ap.add_argument("--dfc-root", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    gym = _import_gym()
    observer = HighwayMaterialObservation(
        WrapperConfig(
            n_max_vehicles=15,
            dt_surrogate=DT_TARGET,
            horizon_surrogate=H_TARGET,
        )
    )

    all_results: Dict[str, Dict[str, Any]] = {}
    for baseline_name in args.baselines:
        baseline_started = time.time()
        baseline = create_baseline(
            baseline_name,
            device=args.device,
            s1_ckpt=args.s1_ckpt,
            s2_ckpt=args.s2_ckpt,
            ppo_ckpt=args.ppo_ckpt,
            sac_ckpt=args.sac_ckpt,
            ppo_lagrangian_ckpt=args.ppo_lagrangian_ckpt,
            sac_lagrangian_ckpt=args.sac_lagrangian_ckpt,
            cpo_ckpt=args.cpo_ckpt,
            dfc_root=args.dfc_root,
        )
        print(f"\n=== {baseline_name} ===")
        per_scenario: Dict[str, Any] = {}
        try:
            for scenario_name in args.scenarios:
                scenario_started = time.time()
                print(
                    f"[{baseline_name}] starting scenario '{scenario_name}' "
                    f"for {args.episodes} episodes",
                    flush=True,
                )
                scenario = SCENARIOS[scenario_name]
                env = make_env(
                    gym,
                    scenario,
                    config_override=baseline.env_config_overrides(),
                    offroad_terminal="boxed" in scenario_name,
                )
                try:
                    eps: List[EpisodeResult] = []
                    for ep in range(args.episodes):
                        seed = args.base_seed + ep
                        episode_started = time.time()
                        result = run_episode(
                            baseline,
                            observer,
                            env,
                            seed=seed,
                            scenario_name=scenario_name,
                            max_steps=args.max_steps,
                        )
                        eps.append(result)
                        _print_episode_progress(
                            baseline_name=baseline_name,
                            scenario_name=scenario_name,
                            episode_idx=ep + 1,
                            episodes_total=args.episodes,
                            result=result,
                            started_at=episode_started,
                        )
                    agg = _agg(eps)
                    _print_summary(f"{baseline_name} / {scenario_name}", agg)
                    print(
                        f"[{baseline_name}] scenario '{scenario_name}' complete "
                        f"in {time.time() - scenario_started:.1f}s",
                        flush=True,
                    )
                    per_scenario[scenario_name] = {
                        "episodes": [asdict(e) for e in eps],
                        "aggregate": agg,
                    }
                finally:
                    env.close()
        finally:
            baseline.close()
            print(
                f"[{baseline_name}] all requested scenarios complete "
                f"in {time.time() - baseline_started:.1f}s",
                flush=True,
            )
        all_results[baseline_name] = per_scenario

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
