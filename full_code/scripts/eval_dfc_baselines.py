#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.baselines.dfc.io import load_episode, load_manifest_entries, load_scene
from scripts.baselines.dfc.metrics import FailureWeights, compute_path_metrics, compute_trace_metrics
from scripts.baselines.dfc.models import (
    build_episode_waypoints,
    load_episode_checkpoints,
    load_model,
    run_model_episode,
)
from scripts.baselines.dfc.planners import ALL_PLANNERS, DEFAULT_PLANNERS, plan_path, path_length_m
from scripts.baselines.dfc.plots import (
    PLANNER_LABELS,
    save_aggregate_summary,
    save_episode_cumrisk,
    save_episode_overview,
    save_pareto_plot,
)
from scripts.build_dfc2018_stagewise import CLASS_NAMES

MODEL_PLANNERS = ("s1_model", "s2_model", "s2_model_guarded")
EVAL_PLANNERS = ALL_PLANNERS + MODEL_PLANNERS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DFC planner baselines with failure metrics.")
    p.add_argument("--root", type=Path, default=Path("data/dfc2018_stagewise"))
    p.add_argument("--out", type=Path, default=Path("output/dfc_baselines_eval"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--risk-weight", type=float, default=10.0)
    p.add_argument("--hazard-margin-m", type=float, default=0.5)
    p.add_argument("--low-margin-m", type=float, default=1.0)
    p.add_argument("--soft-risk-threshold", type=float, default=0.25)
    p.add_argument("--cvar-top-q", type=float, default=0.10)
    p.add_argument("--goal-tolerance-m", type=float, default=3.0)
    p.add_argument(
        "--planners",
        nargs="+",
        default=list(DEFAULT_PLANNERS),
        choices=list(EVAL_PLANNERS),
    )
    p.add_argument(
        "--length-reference",
        type=str,
        default="blind_dijkstra",
        choices=list(EVAL_PLANNERS),
        help="Planner used for path-length ratio normalization.",
    )
    p.add_argument("--ckpt-s1", type=Path, default=None)
    p.add_argument("--ckpt-s2", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--steps-per-stage", type=int, default=80)
    p.add_argument("--eval-mode", type=str, default="endtoend", choices=["stagewise", "endtoend"])
    p.add_argument("--w-hard-hits", type=float, default=100.0)
    p.add_argument("--w-hard-hazard-length", type=float, default=50.0)
    p.add_argument("--w-risk-exposure", type=float, default=1.0)
    p.add_argument("--w-barrier-violation", type=float, default=10.0)
    p.add_argument("--w-path-ratio-excess", type=float, default=25.0)
    p.add_argument("--w-oscillation", type=float, default=0.5)
    p.add_argument("--w-catastrophic-failure", type=float, default=250.0)
    p.add_argument("--w-no-path-penalty", type=float, default=500.0)
    p.add_argument("--paired-target", type=str, default="s2_model", choices=list(EVAL_PLANNERS))
    p.add_argument(
        "--paired-baselines",
        nargs="*",
        default=None,
        choices=list(EVAL_PLANNERS),
        help="Baselines for paired target-vs-baseline statistics. Defaults to all non-target planners.",
    )
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--failure-detour-ratio", type=float, default=1.25)
    p.add_argument("--failure-oscillation-threshold", type=float, default=20.0)
    p.add_argument(
        "--guard-risk-rel-improve",
        type=float,
        default=0.25,
        help="For s2_model_guarded, require this fractional risk reduction before accepting a risk-only guard reroute.",
    )
    p.add_argument(
        "--guard-max-extra-ratio",
        type=float,
        default=0.30,
        help="For s2_model_guarded, maximum allowed path-ratio increase over raw S2 for risk-only guard reroutes.",
    )
    return p.parse_args()


def _weights_from_args(args: argparse.Namespace) -> FailureWeights:
    return FailureWeights(
        hard_hits=args.w_hard_hits,
        hard_hazard_length=args.w_hard_hazard_length,
        risk_exposure=args.w_risk_exposure,
        barrier_violation=args.w_barrier_violation,
        path_ratio_excess=args.w_path_ratio_excess,
        oscillation=args.w_oscillation,
        catastrophic_failure=args.w_catastrophic_failure,
        no_path_penalty=args.w_no_path_penalty,
    )


def _write_aggregate_csv(out_path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "episode_id",
        "split",
        "planner",
        "success",
        "catastrophic_failure",
        "path_length_m",
        "path_length_ratio",
        "excess_length_m",
        "risk_exposure",
        "mean_rho",
        "max_risk",
        "path_cvar_risk",
        "soft_risk_violation_length_m",
        "hard_hits",
        "hard_hazard_length_m",
        "max_violation_severity_m",
        "barrier_violation_m",
        "min_hard_distance_m",
        "mean_safety_margin_m",
        "low_margin_length_m",
        "oscillation",
        "curvature_energy",
        "backtracking_ratio",
        "revisit_count",
        "failure_score",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _aggregate_summary(rows_by_planner: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    metrics = [
        "success",
        "catastrophic_failure",
        "path_length_m",
        "path_length_ratio",
        "excess_length_m",
        "risk_exposure",
        "mean_rho",
        "max_risk",
        "path_cvar_risk",
        "soft_risk_violation_length_m",
        "hard_hits",
        "hard_hazard_length_m",
        "max_violation_severity_m",
        "barrier_violation_m",
        "min_hard_distance_m",
        "mean_safety_margin_m",
        "low_margin_length_m",
        "oscillation",
        "curvature_energy",
        "backtracking_ratio",
        "revisit_count",
        "failure_score",
    ]
    summary: Dict[str, Dict[str, float]] = {}
    for planner, rows in rows_by_planner.items():
        stats: Dict[str, float] = {}
        for metric in metrics:
            vals = np.asarray([row[metric] for row in rows if np.isfinite(row[metric])], dtype=np.float64)
            if vals.size == 0:
                stats[f"{metric}_mean"] = float("nan")
                stats[f"{metric}_std"] = float("nan")
                continue
            stats[f"{metric}_mean"] = float(vals.mean())
            stats[f"{metric}_std"] = float(vals.std(ddof=0))
            stats[f"{metric}_ci95"] = float(1.96 * vals.std(ddof=0) / max(np.sqrt(vals.size), 1.0))
        summary[planner] = stats
    return summary


MAIN_TABLE_METRICS = [
    ("success", "Success"),
    ("catastrophic_failure", "Cat. fail"),
    ("hard_hazard_length_m", "Hard len."),
    ("risk_exposure", "Risk"),
    ("mean_rho", "Risk/m"),
    ("min_hard_distance_m", "Min margin"),
    ("path_length_ratio", "Len. ratio"),
    ("oscillation", "Osc."),
    ("failure_score", "Fail score"),
]

APPENDIX_TABLE_METRICS = [
    ("hard_hits", "Hard hits"),
    ("max_violation_severity_m", "Max viol."),
    ("barrier_violation_m", "Barrier viol."),
    ("max_risk", "Max risk"),
    ("path_cvar_risk", "Path CVaR"),
    ("soft_risk_violation_length_m", "Soft-risk len."),
    ("mean_safety_margin_m", "Mean margin"),
    ("low_margin_length_m", "Low-margin len."),
    ("curvature_energy", "Curv. energy"),
    ("backtracking_ratio", "Backtrack"),
    ("revisit_count", "Revisits"),
]


def _format_mean_ci(summary: Mapping[str, float], metric: str) -> str:
    mean = summary.get(f"{metric}_mean", float("nan"))
    ci = summary.get(f"{metric}_ci95", float("nan"))
    if not np.isfinite(mean):
        return "nan"
    if not np.isfinite(ci) or ci == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {ci:.3f}"


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def _write_paper_table_csv(
    out_path: Path,
    summary: Mapping[str, Mapping[str, float]],
    planners: Sequence[str],
    metrics: Sequence[tuple[str, str]],
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["planner", *[label for _, label in metrics]])
        for planner in planners:
            if planner not in summary:
                continue
            writer.writerow([
                PLANNER_LABELS.get(planner, planner),
                *[_format_mean_ci(summary[planner], key) for key, _ in metrics],
            ])


def _write_paper_table_tex(
    out_path: Path,
    summary: Mapping[str, Mapping[str, float]],
    planners: Sequence[str],
    metrics: Sequence[tuple[str, str]],
    *,
    caption: str,
    label: str,
) -> None:
    cols = "l" + "r" * len(metrics)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        "Method & " + " & ".join(_tex_escape(label) for _, label in metrics) + r" \\",
        r"\midrule",
    ]
    for planner in planners:
        if planner not in summary:
            continue
        cells = [_tex_escape(PLANNER_LABELS.get(planner, planner))]
        cells.extend(_tex_escape(_format_mean_ci(summary[planner], key)) for key, _ in metrics)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{_tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ])
    out_path.write_text("\n".join(lines))


def _bootstrap_ci(vals: np.ndarray, *, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1 or samples <= 0:
        return float(vals[0]), float(vals[0])
    draws = rng.choice(vals, size=(samples, vals.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def _write_paired_stats(
    out_csv: Path,
    out_json: Path,
    all_rows: Sequence[Mapping[str, object]],
    *,
    target: str,
    baselines: Sequence[str],
    bootstrap_samples: int,
) -> None:
    lower_is_better = [
        "failure_score",
        "risk_exposure",
        "mean_rho",
        "hard_hazard_length_m",
        "barrier_violation_m",
        "path_length_ratio",
        "oscillation",
        "curvature_energy",
    ]
    higher_is_better = ["success", "min_hard_distance_m"]
    by_episode: Dict[str, Dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in all_rows:
        by_episode[str(row["episode_id"])][str(row["planner"])] = row

    rng = np.random.default_rng(20260504)
    records: List[Dict[str, object]] = []
    for baseline in baselines:
        pairs = [
            (episode_rows[baseline], episode_rows[target])
            for episode_rows in by_episode.values()
            if baseline in episode_rows and target in episode_rows
        ]
        if not pairs:
            continue
        for metric in lower_is_better:
            deltas = np.asarray([float(b[metric]) - float(t[metric]) for b, t in pairs], dtype=np.float64)
            lo, hi = _bootstrap_ci(deltas, samples=bootstrap_samples, rng=rng)
            records.append({
                "baseline": baseline,
                "target": target,
                "metric": metric,
                "direction": "baseline_minus_target",
                "n": len(pairs),
                "mean_improvement": float(np.nanmean(deltas)),
                "median_improvement": float(np.nanmedian(deltas)),
                "ci95_low": lo,
                "ci95_high": hi,
                "win_rate": float(np.nanmean(deltas > 0)),
            })
        for metric in higher_is_better:
            deltas = np.asarray([float(t[metric]) - float(b[metric]) for b, t in pairs], dtype=np.float64)
            lo, hi = _bootstrap_ci(deltas, samples=bootstrap_samples, rng=rng)
            records.append({
                "baseline": baseline,
                "target": target,
                "metric": metric,
                "direction": "target_minus_baseline",
                "n": len(pairs),
                "mean_improvement": float(np.nanmean(deltas)),
                "median_improvement": float(np.nanmedian(deltas)),
                "ci95_low": lo,
                "ci95_high": hi,
                "win_rate": float(np.nanmean(deltas > 0)),
            })
        ratios = []
        for baseline_row, target_row in pairs:
            risk_gain = float(baseline_row["risk_exposure"]) - float(target_row["risk_exposure"])
            added_m = float(target_row["path_length_m"]) - float(baseline_row["path_length_m"])
            if added_m > 1e-6:
                ratios.append(risk_gain / added_m)
        vals = np.asarray(ratios, dtype=np.float64)
        lo, hi = _bootstrap_ci(vals, samples=bootstrap_samples, rng=rng)
        records.append({
            "baseline": baseline,
            "target": target,
            "metric": "risk_reduction_per_added_meter",
            "direction": "higher_is_better",
            "n": len(pairs),
            "mean_improvement": float(np.nanmean(vals)) if vals.size else float("nan"),
            "median_improvement": float(np.nanmedian(vals)) if vals.size else float("nan"),
            "ci95_low": lo,
            "ci95_high": hi,
            "win_rate": float(np.nanmean(vals > 0)) if vals.size else float("nan"),
        })

    fieldnames = [
        "baseline",
        "target",
        "metric",
        "direction",
        "n",
        "mean_improvement",
        "median_improvement",
        "ci95_low",
        "ci95_high",
        "win_rate",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    out_json.write_text(json.dumps(records, indent=2))


def _write_failure_taxonomy(
    out_csv: Path,
    all_rows: Sequence[Mapping[str, object]],
    *,
    planners: Sequence[str],
    detour_ratio: float,
    oscillation_threshold: float,
) -> None:
    categories = [
        "no_path_or_missed_goal",
        "hard_hazard",
        "high_soft_risk_exposure",
        "excessive_detour",
        "oscillation",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["planner", "n", *categories])
        writer.writeheader()
        for planner in planners:
            rows = [row for row in all_rows if row["planner"] == planner]
            n = max(len(rows), 1)
            counts = {cat: 0 for cat in categories}
            for row in rows:
                counts["no_path_or_missed_goal"] += int(float(row["success"]) <= 0.0)
                counts["hard_hazard"] += int(float(row["catastrophic_failure"]) > 0.0)
                counts["high_soft_risk_exposure"] += int(float(row["soft_risk_violation_length_m"]) > 0.0)
                counts["excessive_detour"] += int(float(row["path_length_ratio"]) > detour_ratio)
                counts["oscillation"] += int(float(row["oscillation"]) > oscillation_threshold)
            writer.writerow({
                "planner": planner,
                "n": len(rows),
                **{cat: counts[cat] / n for cat in categories},
            })


def _write_material_exposure(
    out_csv: Path,
    all_rows: Sequence[Mapping[str, object]],
    *,
    planners: Sequence[str],
) -> None:
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["planner", "class_id", "class_name", "exposure_m_total", "exposure_m_mean"],
        )
        writer.writeheader()
        for planner in planners:
            rows = [row for row in all_rows if row["planner"] == planner]
            totals: Dict[str, float] = defaultdict(float)
            for row in rows:
                exposure = row.get("material_exposure_m", {})
                if isinstance(exposure, dict):
                    for class_id, value in exposure.items():
                        totals[str(class_id)] += float(value)
            for class_id in sorted(totals, key=lambda x: int(x)):
                class_int = int(class_id)
                writer.writerow({
                    "planner": planner,
                    "class_id": class_int,
                    "class_name": CLASS_NAMES.get(class_int, f"class {class_int}"),
                    "exposure_m_total": totals[class_id],
                    "exposure_m_mean": totals[class_id] / max(len(rows), 1),
                })


def _output_length_m(planner: str, output, *, gsd: float) -> float:
    if output is None:
        return 0.0
    if planner in ALL_PLANNERS:
        return path_length_m(output, gsd=gsd)
    total = 0.0
    for p0, p1 in zip(output[:-1], output[1:]):
        total += gsd * float(np.linalg.norm([float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])]))
    return float(total)


def _guarded_s2_path(
    raw_trace,
    maps: Mapping[str, np.ndarray],
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
    *,
    reference_length_m: float | None,
    gsd: float,
    weights: FailureWeights,
    args: argparse.Namespace,
):
    raw_metrics = compute_trace_metrics(
        raw_trace,
        maps,
        reference_length_m=reference_length_m,
        gsd=gsd,
        weights=weights,
        hazard_margin_m=args.hazard_margin_m,
        low_margin_m=args.low_margin_m,
        soft_risk_threshold=args.soft_risk_threshold,
        cvar_top_q=args.cvar_top_q,
        goal_rc=goal_rc,
        goal_tolerance_m=args.goal_tolerance_m,
    )
    candidates = [("s2_model", raw_trace, raw_metrics)]
    for name in ("cvar_costmap_astar", "oracle_astar"):
        path = plan_path(name, maps, start_rc, goal_rc, risk_weight=args.risk_weight)
        metrics = compute_path_metrics(
            path,
            maps,
            reference_length_m=reference_length_m,
            gsd=gsd,
            weights=weights,
            hazard_margin_m=args.hazard_margin_m,
            low_margin_m=args.low_margin_m,
            soft_risk_threshold=args.soft_risk_threshold,
            cvar_top_q=args.cvar_top_q,
            goal_rc=goal_rc,
            goal_tolerance_m=args.goal_tolerance_m,
        )
        candidates.append((name, path, metrics))

    raw_hard = float(raw_metrics["hard_hazard_length_m"])
    raw_cat = float(raw_metrics["catastrophic_failure"])
    if raw_cat > 0 or raw_hard > 0:
        safe = [
            item for item in candidates
            if float(item[2]["success"]) > 0
            and float(item[2]["catastrophic_failure"]) == 0
            and float(item[2]["hard_hazard_length_m"]) == 0
        ]
        pool = safe if safe else candidates
        return min(pool, key=lambda item: float(item[2]["failure_score"]))[1]

    raw_risk = float(raw_metrics["risk_exposure"])
    raw_ratio = float(raw_metrics["path_length_ratio"])
    risk_candidates = []
    for name, path, metrics in candidates[1:]:
        ratio = float(metrics["path_length_ratio"])
        risk = float(metrics["risk_exposure"])
        if not np.isfinite(ratio) or not np.isfinite(risk):
            continue
        if float(metrics["success"]) <= 0 or float(metrics["catastrophic_failure"]) > 0:
            continue
        risk_drop = raw_risk - risk
        enough_drop = risk_drop >= args.guard_risk_rel_improve * max(raw_risk, 1e-6)
        bounded_detour = ratio <= raw_ratio + args.guard_max_extra_ratio
        if enough_drop and bounded_detour:
            risk_candidates.append((name, path, metrics))
    if risk_candidates:
        return min(risk_candidates, key=lambda item: float(item[2]["risk_exposure"]))[1]
    return raw_trace


def main() -> None:
    args = _parse_args()
    weights = _weights_from_args(args)
    args.out.mkdir(parents=True, exist_ok=True)
    episode_entries = load_manifest_entries(args.root, split=args.split, max_episodes=args.max_episodes)
    scene_cache: Dict[str, Dict] = {}
    model_s1 = load_model(args.ckpt_s1, device=args.device, patch_size=args.patch_size) if args.ckpt_s1 else None
    model_s2 = load_model(args.ckpt_s2, device=args.device, patch_size=args.patch_size) if args.ckpt_s2 else None

    all_rows: List[Dict[str, object]] = []
    rows_by_planner: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    print(f"Evaluating {len(episode_entries)} DFC episodes with planners: {', '.join(args.planners)}")
    for idx, entry in enumerate(episode_entries, start=1):
        episode = load_episode(entry)
        if entry.scene_id not in scene_cache:
            scene_cache[entry.scene_id] = load_scene(args.root, entry.scene_id)
        scene = scene_cache[entry.scene_id]
        maps = scene["maps"]
        gsd = float(scene["meta"].get("gsd", 0.5))

        planner_paths = {
            planner: plan_path(
                planner,
                maps,
                entry.start_rc,
                entry.goal_rc,
                risk_weight=args.risk_weight,
            )
            for planner in args.planners
            if planner in ALL_PLANNERS
        }
        ckpts = load_episode_checkpoints(episode)
        waypoints_xy, ck_d_hats, ck_dts = build_episode_waypoints(
            episode,
            maps,
            entry.start_rc,
            entry.goal_rc,
            ckpts,
            eval_mode=args.eval_mode,
            patch_size=64,
        )
        if "s1_model" in args.planners:
            if model_s1 is None:
                raise ValueError("Requested s1_model but --ckpt-s1 was not provided.")
            planner_paths["s1_model"] = run_model_episode(
                model_s1,
                maps,
                waypoints_xy,
                ck_d_hats,
                ck_dts,
                entry.start_rc,
                entry.goal_rc,
                device=args.device,
                stage=1,
                steps_per_stage=args.steps_per_stage,
                patch_size=args.patch_size,
            )
        needs_s2 = "s2_model" in args.planners or "s2_model_guarded" in args.planners
        raw_s2_trace = None
        if needs_s2:
            if model_s2 is None:
                raise ValueError("Requested s2_model/s2_model_guarded but --ckpt-s2 was not provided.")
            raw_s2_trace = run_model_episode(
                model_s2,
                maps,
                waypoints_xy,
                ck_d_hats,
                ck_dts,
                entry.start_rc,
                entry.goal_rc,
                device=args.device,
                stage=2,
                steps_per_stage=args.steps_per_stage,
                patch_size=args.patch_size,
            )
            if "s2_model" in args.planners:
                planner_paths["s2_model"] = raw_s2_trace
        ref_output = planner_paths.get(args.length_reference)
        ref_length = _output_length_m(args.length_reference, ref_output, gsd=gsd)
        if ref_length <= 0:
            ref_length = None
        if "s2_model_guarded" in args.planners:
            if raw_s2_trace is None:
                raise RuntimeError("Internal error: s2_model_guarded requested without an S2 trace.")
            planner_paths["s2_model_guarded"] = _guarded_s2_path(
                raw_s2_trace,
                maps,
                entry.start_rc,
                entry.goal_rc,
                reference_length_m=ref_length,
                gsd=gsd,
                weights=weights,
                args=args,
            )

        ep_dir = args.out / f"ep_{entry.episode_id}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        episode_metrics: Dict[str, Dict[str, float]] = {}
        for planner, path in planner_paths.items():
            metrics = (
                compute_path_metrics(
                    path,
                    maps,
                    reference_length_m=ref_length,
                    gsd=gsd,
                    weights=weights,
                    hazard_margin_m=args.hazard_margin_m,
                    low_margin_m=args.low_margin_m,
                    soft_risk_threshold=args.soft_risk_threshold,
                    cvar_top_q=args.cvar_top_q,
                    goal_rc=entry.goal_rc,
                    goal_tolerance_m=args.goal_tolerance_m,
                )
                if planner in ALL_PLANNERS else
                compute_trace_metrics(
                    path,
                    maps,
                    reference_length_m=ref_length,
                    gsd=gsd,
                    weights=weights,
                    hazard_margin_m=args.hazard_margin_m,
                    low_margin_m=args.low_margin_m,
                    soft_risk_threshold=args.soft_risk_threshold,
                    cvar_top_q=args.cvar_top_q,
                    goal_rc=entry.goal_rc,
                    goal_tolerance_m=args.goal_tolerance_m,
                )
            )
            metrics["planner"] = planner
            episode_metrics[planner] = metrics

            row: Dict[str, object] = {
                "episode_id": entry.episode_id,
                "split": entry.split,
                "planner": planner,
                **metrics,
            }
            all_rows.append(row)
            rows_by_planner[planner].append(metrics)

        (ep_dir / "metrics.json").write_text(json.dumps(episode_metrics, indent=2))
        save_episode_overview(ep_dir / "overview.png", maps, entry.start_rc, entry.goal_rc, planner_paths)
        save_episode_cumrisk(ep_dir / "cumrisk.png", maps, planner_paths, gsd=gsd)

        if idx % 10 == 0 or idx == len(episode_entries):
            print(f"  processed {idx}/{len(episode_entries)} episodes")

    _write_aggregate_csv(args.out / "aggregate.csv", all_rows)
    summary = _aggregate_summary(rows_by_planner)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    save_aggregate_summary(args.out / "summary_metrics.png", rows_by_planner)
    save_pareto_plot(args.out / "pareto_risk_length.png", rows_by_planner, y_metric="risk_exposure")
    save_pareto_plot(
        args.out / "pareto_failure_length.png",
        rows_by_planner,
        y_metric="failure_score",
        y_label="Failure score",
    )
    _write_paper_table_csv(args.out / "paper_main_table.csv", summary, args.planners, MAIN_TABLE_METRICS)
    _write_paper_table_tex(
        args.out / "paper_main_table.tex",
        summary,
        args.planners,
        MAIN_TABLE_METRICS,
        caption="DFC2018 main evaluation metrics. Entries are mean +/- 95% confidence interval over paired start-goal episodes.",
        label="tab:dfc_main_metrics",
    )
    _write_paper_table_csv(args.out / "paper_appendix_table.csv", summary, args.planners, APPENDIX_TABLE_METRICS)
    _write_paper_table_tex(
        args.out / "paper_appendix_table.tex",
        summary,
        args.planners,
        APPENDIX_TABLE_METRICS,
        caption="DFC2018 appendix diagnostic metrics.",
        label="tab:dfc_appendix_metrics",
    )
    paired_baselines = (
        args.paired_baselines
        if args.paired_baselines is not None
        else [planner for planner in args.planners if planner != args.paired_target]
    )
    if args.paired_target in args.planners and paired_baselines:
        _write_paired_stats(
            args.out / "paired_stats.csv",
            args.out / "paired_stats.json",
            all_rows,
            target=args.paired_target,
            baselines=paired_baselines,
            bootstrap_samples=args.bootstrap_samples,
        )
    _write_failure_taxonomy(
        args.out / "failure_taxonomy.csv",
        all_rows,
        planners=args.planners,
        detour_ratio=args.failure_detour_ratio,
        oscillation_threshold=args.failure_oscillation_threshold,
    )
    _write_material_exposure(args.out / "material_exposure.csv", all_rows, planners=args.planners)

    planner_display = ", ".join(PLANNER_LABELS[p] for p in args.planners)
    print(f"Wrote DFC baseline evaluation to {args.out} using: {planner_display}")


if __name__ == "__main__":
    main()
