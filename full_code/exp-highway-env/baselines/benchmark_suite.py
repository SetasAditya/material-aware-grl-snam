#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time

from env_wrapper import HighwayMaterialObservation, WrapperConfig  # noqa: E402
from eval_stage2 import DT_TARGET, H_TARGET, SCENARIOS, _agg, _import_gym, _print_summary  # noqa: E402

from .evaluate import EpisodeResult, make_env, run_episode
from .registry import BASELINE_NAMES, create_baseline


HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent


def _preferred_ckpt(*candidates: Path) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


DEFAULT_BASELINES: Tuple[str, ...] = (
    "constant_velocity",
    "idm",
    "mobil_idm",
    "ppo",
    "sac",
    "ppo_lagrangian",
    "sac_lagrangian",
    "cpo",
    "risk_aware_mpc",
    "chance_constrained_mpc",
    "cbf_qp_filter",
    "s1_model",
    "s2_model",
)

DEFAULT_S1_CKPT = EXP_ROOT / "checkpoints" / "highway_stage1_default_slow_x4" / "best.pt"
DEFAULT_S2_CKPT = EXP_ROOT / "checkpoints" / "highway_stage2_mu_lat" / "best.pt"
DEFAULT_PPO_CKPT = _preferred_ckpt(
    EXP_ROOT / "checkpoints" / "highway_ppo_baseline" / "model.zip",
)
DEFAULT_SAC_CKPT = _preferred_ckpt(
    EXP_ROOT / "checkpoints" / "highway_sac_baseline" / "model.zip",
)
DEFAULT_PPO_LAGRANGIAN_CKPT = _preferred_ckpt(
    EXP_ROOT / "checkpoints" / "highway_ppo_lagrangian_baseline" / "best_model.zip",
    EXP_ROOT / "checkpoints" / "highway_ppo_lagrangian_baseline" / "model.zip",
)
DEFAULT_SAC_LAGRANGIAN_CKPT = _preferred_ckpt(
    EXP_ROOT / "checkpoints" / "highway_sac_lagrangian_baseline" / "best_model.zip",
    EXP_ROOT / "checkpoints" / "highway_sac_lagrangian_baseline" / "model.zip",
)
DEFAULT_CPO_CKPT = _preferred_ckpt(
    EXP_ROOT / "checkpoints" / "highway_cpo_baseline" / "best_model.zip",
    EXP_ROOT / "checkpoints" / "highway_cpo_baseline" / "model.zip",
)

LEARNED_BASELINE_TO_ARG = {
    "ppo": "ppo_ckpt",
    "sac": "sac_ckpt",
    "ppo_lagrangian": "ppo_lagrangian_ckpt",
    "sac_lagrangian": "sac_lagrangian_ckpt",
    "cpo": "cpo_ckpt",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark the full highway baseline suite against the best paper-facing GRL checkpoints."
    )
    ap.add_argument(
        "--baselines",
        nargs="+",
        choices=BASELINE_NAMES,
        default=list(DEFAULT_BASELINES),
        help="Baseline names to evaluate. Defaults to the full benchmark suite.",
    )
    ap.add_argument(
        "--scenarios",
        nargs="+",
        default=["default", "authored_slow_leader", "authored_slow_leader_boxed"],
        choices=list(SCENARIOS.keys()),
        help="Scenario sweep to benchmark on.",
    )
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--s1-ckpt", type=Path, default=DEFAULT_S1_CKPT)
    ap.add_argument("--s2-ckpt", type=Path, default=DEFAULT_S2_CKPT)
    ap.add_argument("--ppo-ckpt", type=Path, default=DEFAULT_PPO_CKPT)
    ap.add_argument("--sac-ckpt", type=Path, default=DEFAULT_SAC_CKPT)
    ap.add_argument("--ppo-lagrangian-ckpt", type=Path, default=DEFAULT_PPO_LAGRANGIAN_CKPT)
    ap.add_argument("--sac-lagrangian-ckpt", type=Path, default=DEFAULT_SAC_LAGRANGIAN_CKPT)
    ap.add_argument("--cpo-ckpt", type=Path, default=DEFAULT_CPO_CKPT)
    ap.add_argument(
        "--skip-missing-learned",
        action="store_true",
        help="Skip learned-policy baselines whose checkpoints are not provided instead of failing.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=EXP_ROOT / "runs" / "baseline_benchmark_suite.json",
        help="Where to save the full benchmark JSON.",
    )
    return ap.parse_args()


def _ckpt_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "device": args.device,
        "s1_ckpt": str(args.s1_ckpt),
        "s2_ckpt": str(args.s2_ckpt),
        "ppo_ckpt": str(args.ppo_ckpt) if args.ppo_ckpt else "",
        "sac_ckpt": str(args.sac_ckpt) if args.sac_ckpt else "",
        "ppo_lagrangian_ckpt": str(args.ppo_lagrangian_ckpt) if args.ppo_lagrangian_ckpt else "",
        "sac_lagrangian_ckpt": str(args.sac_lagrangian_ckpt) if args.sac_lagrangian_ckpt else "",
        "cpo_ckpt": str(args.cpo_ckpt) if args.cpo_ckpt else "",
        "dfc_root": "",
    }


def _resolve_baselines(args: argparse.Namespace) -> Tuple[List[str], Dict[str, str]]:
    runnable: List[str] = []
    skipped: Dict[str, str] = {}
    ckpt_kwargs = _ckpt_kwargs(args)

    for name in args.baselines:
        learned_arg = LEARNED_BASELINE_TO_ARG.get(name)
        if learned_arg is not None:
            ckpt_value = ckpt_kwargs.get(learned_arg, "")
            if not ckpt_value:
                msg = f"missing required checkpoint argument --{learned_arg.replace('_', '-')}"
                if args.skip_missing_learned:
                    skipped[name] = msg
                    continue
                raise SystemExit(f"Cannot run baseline '{name}': {msg}")
            if not Path(ckpt_value).exists():
                msg = f"checkpoint does not exist: {ckpt_value}"
                if args.skip_missing_learned:
                    skipped[name] = msg
                    continue
                raise SystemExit(f"Cannot run baseline '{name}': {msg}")
        runnable.append(name)
    return runnable, skipped


def _run_baseline(
    baseline_name: str,
    *,
    args: argparse.Namespace,
    gym,
    observer: HighwayMaterialObservation,
) -> Dict[str, Any]:
    baseline_started = time.time()
    baseline = create_baseline(baseline_name, **_ckpt_kwargs(args))
    print(f"\n=== {baseline_name} ===")
    results: Dict[str, Any] = {}
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
                    elapsed = max(time.time() - episode_started, 1e-6)
                    status = "collision" if result.collided else "truncated" if result.truncated else "ok"
                    print(
                        f"[{baseline_name} | {scenario_name}] episode "
                        f"{ep + 1}/{args.episodes} finished in {elapsed:.1f}s "
                        f"| steps={result.steps:3d} | status={status:9s} "
                        f"| speed={result.mean_speed:5.2f} | dmin={result.min_clearance:5.2f}",
                        flush=True,
                    )
                agg = _agg(eps)
                _print_summary(f"{baseline_name} / {scenario_name}", agg)
                print(
                    f"[{baseline_name}] scenario '{scenario_name}' complete "
                    f"in {time.time() - scenario_started:.1f}s",
                    flush=True,
                )
                results[scenario_name] = {
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
    return results


def _print_cross_baseline_table(
    benchmark: Dict[str, Dict[str, Any]],
    scenarios: List[str],
) -> None:
    for scenario_name in scenarios:
        print(f"\n=== Scenario: {scenario_name} ===")
        print(
            f"{'baseline':<26} {'crash%':>8} {'offroad%':>9} {'speed':>8} "
            f"{'clear(m)':>10} {'cum_risk':>10}"
        )
        ranked = []
        for baseline_name, payload in benchmark.items():
            agg = payload[scenario_name]["aggregate"]
            ranked.append(
                (
                    float(agg.get("collision_rate", 1.0)),
                    float(agg.get("offroad_rate", 1.0)),
                    float(agg.get("cum_risk_eval_mean", 1e9)),
                    -float(agg.get("mean_speed_mean", 0.0)),
                    baseline_name,
                    agg,
                )
            )
        ranked.sort()
        for _, _, _, _, baseline_name, agg in ranked:
            print(
                f"{baseline_name:<26} "
                f"{agg.get('collision_rate', float('nan')):>7.1%} "
                f"{agg.get('offroad_rate', float('nan')):>8.1%} "
                f"{agg.get('mean_speed_mean', float('nan')):>8.2f} "
                f"{agg.get('min_clearance_mean', float('nan')):>10.2f} "
                f"{agg.get('cum_risk_eval_mean', float('nan')):>10.2f}"
            )


def _rank_baseline(agg: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(agg.get("collision_rate", 1.0)),
        float(agg.get("offroad_rate", 1.0)),
        float(agg.get("cum_risk_eval_mean", 1e9)),
        -float(agg.get("mean_speed_mean", 0.0)),
    )


def _compute_winners(
    benchmark: Dict[str, Dict[str, Any]],
    scenarios: List[str],
) -> Dict[str, Any]:
    winners: Dict[str, Any] = {"per_scenario": {}, "overall": {}}
    overall_rows = []
    for baseline_name, payload in benchmark.items():
        ranks = [_rank_baseline(payload[scenario]["aggregate"]) for scenario in scenarios]
        mean_rank = tuple(float(sum(items[i] for items in ranks)) / max(1, len(ranks)) for i in range(4))
        overall_rows.append((mean_rank, baseline_name))

    for scenario_name in scenarios:
        ranked = sorted(
            (
                _rank_baseline(payload[scenario_name]["aggregate"]),
                baseline_name,
            )
            for baseline_name, payload in benchmark.items()
        )
        best_rank, best_name = ranked[0]
        winners["per_scenario"][scenario_name] = {
            "baseline": best_name,
            "rank_key": list(best_rank),
        }

    overall_rows.sort()
    winners["overall"] = {
        "baseline": overall_rows[0][1],
        "rank_key": list(overall_rows[0][0]),
    }
    return winners


def main() -> None:
    args = _parse_args()
    if not args.s1_ckpt.exists():
        raise SystemExit(f"Missing Stage 1 checkpoint: {args.s1_ckpt}")
    if not args.s2_ckpt.exists():
        raise SystemExit(f"Missing Stage 2 checkpoint: {args.s2_ckpt}")

    runnable, skipped = _resolve_baselines(args)
    if not runnable:
        raise SystemExit("No runnable baselines after checkpoint resolution.")

    gym = _import_gym()
    observer = HighwayMaterialObservation(
        WrapperConfig(
            n_max_vehicles=15,
            dt_surrogate=DT_TARGET,
            horizon_surrogate=H_TARGET,
        )
    )

    print("Benchmark suite")
    print(f"  baselines: {', '.join(runnable)}")
    if skipped:
        print("  skipped:")
        for name, reason in skipped.items():
            print(f"    {name}: {reason}")
    print(f"  Stage 1 ckpt: {args.s1_ckpt}")
    print(f"  Stage 2 ckpt: {args.s2_ckpt}")
    print(f"  scenarios: {', '.join(args.scenarios)}")
    print(f"  episodes: {args.episodes}  max_steps: {args.max_steps}")

    benchmark: Dict[str, Dict[str, Any]] = {}
    for baseline_name in runnable:
        benchmark[baseline_name] = _run_baseline(
            baseline_name,
            args=args,
            gym=gym,
            observer=observer,
        )

    _print_cross_baseline_table(benchmark, args.scenarios)
    winners = _compute_winners(benchmark, args.scenarios)
    print("\nWinners")
    for scenario_name, payload in winners["per_scenario"].items():
        print(f"  {scenario_name}: {payload['baseline']}")
    print(f"  overall: {winners['overall']['baseline']}")

    out = {
        "config": {
            "requested_baselines": list(args.baselines),
            "executed_baselines": runnable,
            "skipped_baselines": skipped,
            "scenarios": list(args.scenarios),
            "episodes": int(args.episodes),
            "max_steps": int(args.max_steps),
            "base_seed": int(args.base_seed),
            "device": args.device,
            "s1_ckpt": str(args.s1_ckpt),
            "s2_ckpt": str(args.s2_ckpt),
            "ppo_ckpt": str(args.ppo_ckpt) if args.ppo_ckpt else "",
            "sac_ckpt": str(args.sac_ckpt) if args.sac_ckpt else "",
            "ppo_lagrangian_ckpt": str(args.ppo_lagrangian_ckpt) if args.ppo_lagrangian_ckpt else "",
            "sac_lagrangian_ckpt": str(args.sac_lagrangian_ckpt) if args.sac_lagrangian_ckpt else "",
            "cpo_ckpt": str(args.cpo_ckpt) if args.cpo_ckpt else "",
        },
        "winners": winners,
        "results": benchmark,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved benchmark suite to {args.out}")


if __name__ == "__main__":
    main()
