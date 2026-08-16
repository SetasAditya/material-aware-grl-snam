#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


HERE = Path(__file__).resolve().parent


def _resolve_device(device: str) -> str:
    if str(device).lower() != "auto":
        return str(device)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass(frozen=True)
class TtcPreset:
    name: str
    ttc_gain: float
    ttc_threshold_s: float
    ttc_softness_s: float
    w_road: float

    @property
    def slug(self) -> str:
        return (
            f"{self.name}_g{self.ttc_gain:g}_t{self.ttc_threshold_s:g}"
            f"_s{self.ttc_softness_s:g}_wr{self.w_road:g}"
        ).replace(".", "p")


PRESETS: Sequence[TtcPreset] = (
    # Anchor near the current TTC checkpoint family.
    TtcPreset("ttc_anchor", 8.0, 4.0, 0.50, 0.25),
    # Lower threshold to reduce over-braking in boxed.
    TtcPreset("ttc_less_timid_a", 8.0, 3.5, 0.50, 0.25),
    TtcPreset("ttc_less_timid_b", 8.0, 3.0, 0.75, 0.25),
    # Softer transition around TTC threshold.
    TtcPreset("ttc_soft_gate_a", 8.0, 3.5, 0.75, 0.25),
    TtcPreset("ttc_soft_gate_b", 8.0, 3.0, 1.00, 0.25),
    # Slightly smaller longitudinal pressure.
    TtcPreset("ttc_gentler_gain", 6.0, 3.0, 0.75, 0.25),
    # Stronger offroad discouragement while staying TTC-based.
    TtcPreset("ttc_road_guard_a", 8.0, 3.5, 0.75, 0.50),
    TtcPreset("ttc_road_guard_b", 6.0, 3.0, 1.00, 0.50),
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Sweep TTC-based Stage 2 retrains and automatically select the safest "
            "low-risk checkpoint that still meets a boxed speed floor."
        )
    )
    ap.add_argument(
        "--stage1-ckpt",
        type=Path,
        default=HERE / "checkpoints" / "highway_stage1_default_slow_x4" / "best.pt",
    )
    ap.add_argument(
        "--idm-data",
        type=Path,
        default=HERE / "runs" / "stage1_data",
    )
    ap.add_argument(
        "--base-out",
        type=Path,
        default=HERE / "checkpoints" / "ttc_sweep_candidates",
    )
    ap.add_argument(
        "--presets",
        nargs="+",
        default=[],
        help="Optional subset of preset names to run. Default runs the full preset grid.",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.30)
    ap.add_argument("--collect-episodes", type=int, default=5)
    ap.add_argument("--collect-max-steps", type=int, default=120)
    ap.add_argument("--closed-loop-every", type=int, default=5)
    ap.add_argument("--closed-loop-episodes", type=int, default=10)
    ap.add_argument("--best-val-ltraj-guard", type=float, default=10.0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--collect-envs",
        type=str,
        default="highway-v0,highway-slow-leader-v0,highway-slow-leader-boxed-v0",
    )
    ap.add_argument(
        "--best-eval-envs",
        type=str,
        default="highway-v0,highway-slow-leader-v0,highway-slow-leader-boxed-v0",
    )
    ap.add_argument(
        "--eval-scenarios",
        nargs="+",
        default=["default", "authored_slow_leader", "authored_slow_leader_boxed"],
    )
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--eval-max-steps", type=int, default=120)
    ap.add_argument(
        "--boxed-speed-floor",
        type=float,
        default=8.0,
        help="Minimum mean speed required in authored_slow_leader_boxed for a candidate to count as 'not too conservative'.",
    )
    ap.add_argument(
        "--overall-speed-floor",
        type=float,
        default=15.0,
        help="Minimum mean speed averaged across eval scenarios for a candidate to count as 'not too conservative'.",
    )
    ap.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even if the candidate best.pt already exists.",
    )
    ap.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-evaluate even if the eval JSON already exists.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    return ap.parse_args()


def _select_presets(names: Sequence[str]) -> List[TtcPreset]:
    if not names:
        return list(PRESETS)
    wanted = set(names)
    chosen = [p for p in PRESETS if p.name in wanted or p.slug in wanted]
    missing = sorted(wanted - {p.name for p in chosen} - {p.slug for p in chosen})
    if missing:
        raise SystemExit(
            f"Unknown preset(s): {missing}. Available: {[p.name for p in PRESETS]}"
        )
    return chosen


def _run(cmd: Sequence[str], *, cwd: Path, dry_run: bool) -> None:
    rendered = shlex.join([str(x) for x in cmd])
    print(f"\n$ {rendered}", flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def _candidate_dir(base_out: Path, preset: TtcPreset) -> Path:
    return base_out / preset.slug


def _build_train_cmd(args: argparse.Namespace, preset: TtcPreset, out_dir: Path) -> List[str]:
    return [
        sys.executable,
        "train_stage2.py",
        "--stage1-ckpt", str(args.stage1_ckpt),
        "--idm-data", str(args.idm_data),
        "--out", str(out_dir),
        "--epochs", str(args.epochs),
        "--bs", str(args.bs),
        "--lr", str(args.lr),
        "--warmup-frac", str(args.warmup_frac),
        "--collect-episodes", str(args.collect_episodes),
        "--collect-max-steps", str(args.collect_max_steps),
        "--closed-loop-every", str(args.closed_loop_every),
        "--closed-loop-episodes", str(args.closed_loop_episodes),
        "--best-val-ltraj-guard", str(args.best_val_ltraj_guard),
        "--device", str(args.device),
        "--seed", str(args.seed),
        "--collect-envs", str(args.collect_envs),
        "--best-eval-envs", str(args.best_eval_envs),
        "--freeze-geometry",
        "--stress-offroad-terminal",
        "--w-road", str(preset.w_road),
        "--ttc-gain", str(preset.ttc_gain),
        "--ttc-threshold-s", str(preset.ttc_threshold_s),
        "--ttc-softness-s", str(preset.ttc_softness_s),
    ]


def _build_eval_cmd(
    args: argparse.Namespace,
    *,
    ckpt: Path,
    out_json: Path,
) -> List[str]:
    return [
        sys.executable,
        "eval_stage2.py",
        "--ckpt", str(ckpt),
        "--stage1-ckpt", str(args.stage1_ckpt),
        "--scenarios", *list(args.eval_scenarios),
        "--episodes", str(args.eval_episodes),
        "--max-steps", str(args.eval_max_steps),
        "--device", str(args.device),
        "--seed", str(args.seed),
        "--offroad-terminal",
        "--out", str(out_json),
    ]


def _load_eval_summary(eval_json: Path) -> Dict[str, Any]:
    obj = json.loads(eval_json.read_text())
    aggs = obj["aggregates"]
    scenario_rows: Dict[str, Dict[str, float]] = {}
    mean_risk_terms = []
    mean_speed_terms = []
    max_collision = 0.0
    max_offroad = 0.0
    for scenario_name, agg in aggs.items():
        # `eval_stage2.py` stores per-scenario metrics under `stage1` and
        # `stage2`; select the Stage 2 view for sweep ranking. Keep a flat-path
        # fallback so older eval outputs still parse.
        if "stage2" in agg and isinstance(agg["stage2"], dict):
            agg = agg["stage2"]
        row = {
            "collision_rate": float(agg["collision_rate"]),
            "offroad_rate": float(agg["offroad_rate"]),
            "mean_speed_mean": float(agg["mean_speed_mean"]),
            "cum_risk_eval_mean": float(agg["cum_risk_eval_mean"]),
            "min_clearance_mean": float(agg["min_clearance_mean"]),
        }
        scenario_rows[scenario_name] = row
        mean_risk_terms.append(row["cum_risk_eval_mean"])
        mean_speed_terms.append(row["mean_speed_mean"])
        max_collision = max(max_collision, row["collision_rate"])
        max_offroad = max(max_offroad, row["offroad_rate"])
    return {
        "scenario_metrics": scenario_rows,
        "mean_cum_risk": statistics.fmean(mean_risk_terms) if mean_risk_terms else float("inf"),
        "mean_speed": statistics.fmean(mean_speed_terms) if mean_speed_terms else 0.0,
        "max_collision_rate": max_collision,
        "max_offroad_rate": max_offroad,
    }


def _select_best(rows: List[Dict[str, Any]], *, boxed_speed_floor: float, overall_speed_floor: float) -> Dict[str, Any]:
    safe_rows = []
    for row in rows:
        boxed = row["summary"]["scenario_metrics"].get("authored_slow_leader_boxed", {})
        boxed_speed = float(boxed.get("mean_speed_mean", 0.0))
        row["boxed_speed"] = boxed_speed
        row["safe_enough"] = (
            row["summary"]["max_collision_rate"] <= 0.0
            and row["summary"]["max_offroad_rate"] <= 0.0
            and boxed_speed >= float(boxed_speed_floor)
            and row["summary"]["mean_speed"] >= float(overall_speed_floor)
        )
        if row["safe_enough"]:
            safe_rows.append(row)

    def safe_key(row: Dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(row["summary"]["mean_cum_risk"]),
            -float(row["summary"]["mean_speed"]),
            -float(row["boxed_speed"]),
        )

    def fallback_key(row: Dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row["summary"]["max_collision_rate"]),
            float(row["summary"]["max_offroad_rate"]),
            float(row["summary"]["mean_cum_risk"]),
            -float(row["summary"]["mean_speed"]),
        )

    selected_pool = safe_rows if safe_rows else rows
    winner = min(selected_pool, key=safe_key if safe_rows else fallback_key)
    return {
        "winner_slug": winner["preset"]["slug"],
        "winner_out_dir": winner["out_dir"],
        "winner_ckpt": winner["best_ckpt"],
        "winner_eval_json": winner["eval_json"],
        "winner_safe_enough": winner["safe_enough"],
        "selection_mode": "safe_then_low_risk" if safe_rows else "fallback_rank",
        "boxed_speed_floor": float(boxed_speed_floor),
        "overall_speed_floor": float(overall_speed_floor),
    }


def main() -> None:
    args = _parse_args()
    args.device = _resolve_device(args.device)
    if not args.stage1_ckpt.exists():
        raise SystemExit(f"Missing Stage 1 checkpoint: {args.stage1_ckpt}")
    if not args.idm_data.exists():
        raise SystemExit(f"Missing IDM data directory: {args.idm_data}")

    presets = _select_presets(args.presets)
    args.base_out.mkdir(parents=True, exist_ok=True)

    print("TTC sweep")
    print(f"  stage1_ckpt: {args.stage1_ckpt}")
    print(f"  idm_data:    {args.idm_data}")
    print(f"  base_out:    {args.base_out}")
    print(f"  presets:     {[p.slug for p in presets]}")
    print(f"  eval_scenarios: {args.eval_scenarios}")
    print(
        f"  speed floors: boxed>={args.boxed_speed_floor:.2f}, "
        f"overall>={args.overall_speed_floor:.2f}",
    )

    rows: List[Dict[str, Any]] = []
    for preset in presets:
        out_dir = _candidate_dir(args.base_out, preset)
        best_ckpt = out_dir / "best.pt"
        eval_json = out_dir / f"eval_ep{args.eval_episodes}.json"

        if args.force_train or not best_ckpt.exists():
            train_cmd = _build_train_cmd(args, preset, out_dir)
            _run(train_cmd, cwd=HERE, dry_run=args.dry_run)
        else:
            print(f"\n[skip train] {best_ckpt} exists", flush=True)

        if args.force_eval or not eval_json.exists():
            eval_cmd = _build_eval_cmd(args, ckpt=best_ckpt, out_json=eval_json)
            _run(eval_cmd, cwd=HERE, dry_run=args.dry_run)
        else:
            print(f"[skip eval]  {eval_json} exists", flush=True)

        if args.dry_run:
            continue
        if not best_ckpt.exists():
            raise SystemExit(f"Training did not produce {best_ckpt}")
        if not eval_json.exists():
            raise SystemExit(f"Evaluation did not produce {eval_json}")

        summary = _load_eval_summary(eval_json)
        rows.append(
            {
                "preset": {
                    **asdict(preset),
                    "slug": preset.slug,
                },
                "out_dir": str(out_dir),
                "best_ckpt": str(best_ckpt),
                "eval_json": str(eval_json),
                "summary": summary,
            }
        )

    if args.dry_run:
        print("\nDry run only; no selection file written.")
        return

    selection = _select_best(
        rows,
        boxed_speed_floor=args.boxed_speed_floor,
        overall_speed_floor=args.overall_speed_floor,
    )

    manifest = {
        "config": {
            "stage1_ckpt": str(args.stage1_ckpt),
            "idm_data": str(args.idm_data),
            "base_out": str(args.base_out),
            "eval_scenarios": list(args.eval_scenarios),
            "eval_episodes": int(args.eval_episodes),
            "boxed_speed_floor": float(args.boxed_speed_floor),
            "overall_speed_floor": float(args.overall_speed_floor),
            "device": args.device,
            "seed": int(args.seed),
        },
        "candidates": rows,
        "selection": selection,
    }

    manifest_path = args.base_out / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nSelection summary")
    for row in sorted(
        rows,
        key=lambda r: (
            r["summary"]["max_collision_rate"],
            r["summary"]["max_offroad_rate"],
            r["summary"]["mean_cum_risk"],
            -r["summary"]["mean_speed"],
        ),
    ):
        boxed = row["summary"]["scenario_metrics"]["authored_slow_leader_boxed"]
        print(
            f"  {row['preset']['slug']:<36} "
            f"safe={str(row.get('safe_enough', False)):<5} "
            f"risk={row['summary']['mean_cum_risk']:6.2f} "
            f"speed={row['summary']['mean_speed']:5.2f} "
            f"boxed_speed={boxed['mean_speed_mean']:5.2f} "
            f"crash={row['summary']['max_collision_rate']:.1%} "
            f"offroad={row['summary']['max_offroad_rate']:.1%}"
        )

    print("\nWinner")
    print(f"  mode:       {selection['selection_mode']}")
    print(f"  slug:       {selection['winner_slug']}")
    print(f"  ckpt:       {selection['winner_ckpt']}")
    print(f"  eval_json:  {selection['winner_eval_json']}")
    print(f"  safe_enough:{selection['winner_safe_enough']}")
    print(f"  manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
