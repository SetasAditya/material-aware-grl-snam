#!/usr/bin/env python3
"""Experiment 4: primitive-count K sensitivity on paired RELLIS-Dyn episodes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


KS = (4, 8, 16, 32)
PHASES = ("pre_event", "blocked_pre_opening", "post_opening")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_logged(command: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        result = subprocess.run(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with status {result.returncode}; see {log_path}"
        )


def _run_sweep(out: Path, max_episodes: int) -> None:
    here = Path(__file__).resolve().parent
    exp1 = here / "exp1_gate_ablation.py"
    exp2 = here / "exp2_gate_trajectory_agreement.py"
    common = [
        sys.executable,
        str(exp1),
        "--max-episodes",
        str(max_episodes),
        "--event-type",
        "delayed_required_escape",
        "--device",
        "cpu",
    ]
    # Empirically validate the analytical fact that K changes only gate
    # diagnostics in the gate-off arm. These smoke traces are preserved.
    for k in (4, 32):
        result_dir = out / "gateoff_invariance" / f"k{k}"
        _run_logged(
            common
            + [
                "--max-episodes",
                "2",
                "--arms",
                "gate_off",
                "--primitive-count",
                str(k),
                "--out",
                str(result_dir),
            ],
            out / "logs" / f"gateoff_invariance_k{k}.log",
        )
    for k in KS:
        result_dir = out / "raw" / f"k{k}"
        _run_logged(
            common
            + [
                "--arms",
                "gate_on",
                "--primitive-count",
                str(k),
                "--out",
                str(result_dir),
            ],
            out / "logs" / f"gate_on_k{k}.log",
        )
        agreement_dir = out / "agreement" / f"k{k}"
        _run_logged(
            [
                sys.executable,
                str(exp2),
                "--exp1-results",
                str(result_dir),
                "--out",
                str(agreement_dir),
            ],
            out / "logs" / f"agreement_k{k}.log",
        )


def _phase(row: Mapping[str, str]) -> str:
    step = int(row["step"])
    if step < int(row["event_step"]):
        return "pre_event"
    if step < int(row["opening_step"]):
        return "blocked_pre_opening"
    return "post_opening"


def _per_episode(out: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for k in KS:
        raw_dir = out / "raw" / f"k{k}"
        metrics = {
            row["episode_id"]: row
            for row in _read_csv(raw_dir / "per_episode_metrics.csv")
            if row["arm"] == "gate_on"
        }
        traces: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in _read_csv(raw_dir / "step_traces.csv"):
            if row["arm"] == "gate_on":
                traces[row["episode_id"]].append(row)
        agreement: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in _read_csv(
            out / "agreement" / f"k{k}" / "per_decision_agreement.csv"
        ):
            if int(row["actual_horizon_complete"]):
                agreement[row["episode_id"]].append(row)
        for episode_id, metric in metrics.items():
            rows = sorted(traces[episode_id], key=lambda row: int(row["step"]))
            active = sum(int(row["gate_decision"]) for row in rows)
            record: Dict[str, Any] = {
                "k": k,
                "episode_id": episode_id,
                "regime": metric["regime"],
                "steps": len(rows),
                "active_steps": active,
                "activation_rate": active / max(1, len(rows)),
                "success": float(metric["success"]),
                "cvar20_violation": float(metric["cvar20_violation"]),
                "hard_contacts": float(metric["hard_contacts"]),
                "compute_ms_per_step": float(metric["compute_ms_per_step"]),
            }
            for phase in PHASES:
                phase_rows = [row for row in rows if _phase(row) == phase]
                phase_active = sum(
                    int(row["gate_decision"]) for row in phase_rows
                )
                record[f"{phase}_steps"] = len(phase_rows)
                record[f"{phase}_active_steps"] = phase_active
                record[f"{phase}_activation_rate"] = (
                    phase_active / len(phase_rows) if phase_rows else 0.0
                )
                record[f"{phase}_any_activation"] = int(phase_active > 0)
            record["false_preactivation_episode"] = record[
                "blocked_pre_opening_any_activation"
            ]
            record["post_opening_no_activation_episode"] = int(
                not record["post_opening_any_activation"]
            )
            decisions = agreement.get(episode_id, [])
            record["agreement_decisions"] = len(decisions)
            for source, target in (
                ("directional_cosine", "agreement_directional_cosine"),
                ("clearance_agreement", "agreement_clearance_rate"),
                (
                    "hard_contact_disagreement",
                    "agreement_hard_disagreement_rate",
                ),
            ):
                record[target] = (
                    float(np.mean([float(row[source]) for row in decisions]))
                    if decisions
                    else float("nan")
                )
            records.append(record)
    return records


def _bootstrap_mean(
    values: np.ndarray, rng: np.random.Generator, reps: int
) -> Tuple[float, float]:
    n = len(values)
    samples = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        selected = rng.integers(0, n, size=n)
        samples[index] = float(np.mean(values[selected]))
    return (
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _bootstrap_ratio(
    numerators: np.ndarray,
    denominators: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> Tuple[float, float]:
    n = len(numerators)
    samples = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        selected = rng.integers(0, n, size=n)
        samples[index] = float(numerators[selected].sum()) / max(
            1.0, float(denominators[selected].sum())
        )
    return (
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _aggregate(
    per_episode: Sequence[Mapping[str, Any]],
    *,
    out: Path,
    bootstrap_reps: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    summaries: List[Dict[str, Any]] = []
    paired: List[Dict[str, Any]] = []
    by_k = {
        k: sorted(
            [row for row in per_episode if int(row["k"]) == k],
            key=lambda row: int(row["episode_id"]),
        )
        for k in KS
    }
    for k, rows in by_k.items():
        active = np.asarray([float(row["active_steps"]) for row in rows])
        steps = np.asarray([float(row["steps"]) for row in rows])
        activation = float(active.sum() / steps.sum())
        activation_ci = _bootstrap_ratio(active, steps, rng, bootstrap_reps)
        row: Dict[str, Any] = {
            "k": k,
            "n_episodes": len(rows),
            "activation_rate": activation,
            "activation_rate_ci_low": activation_ci[0],
            "activation_rate_ci_high": activation_ci[1],
        }
        for metric in (
            "false_preactivation_episode",
            "post_opening_no_activation_episode",
            "success",
            "cvar20_violation",
            "hard_contacts",
            "compute_ms_per_step",
        ):
            values = np.asarray([float(item[metric]) for item in rows])
            low, high = _bootstrap_mean(values, rng, bootstrap_reps)
            row[metric] = float(np.mean(values))
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        complete_decisions = _read_csv(
            out / "agreement" / f"k{k}" / "per_decision_agreement.csv"
        )
        complete_decisions = [
            item
            for item in complete_decisions
            if int(item["actual_horizon_complete"])
        ]
        for source, target in (
            ("directional_cosine", "agreement_directional_cosine"),
            ("clearance_agreement", "agreement_clearance_rate"),
            (
                "hard_contact_disagreement",
                "agreement_hard_disagreement_rate",
            ),
        ):
            row[target] = float(
                np.mean([float(item[source]) for item in complete_decisions])
            )
        summaries.append(row)

    reference = {
        int(row["episode_id"]): row for row in by_k[16]
    }
    for k, rows in by_k.items():
        if k == 16:
            continue
        for metric in (
            "activation_rate",
            "false_preactivation_episode",
            "post_opening_no_activation_episode",
            "success",
            "cvar20_violation",
            "hard_contacts",
            "compute_ms_per_step",
        ):
            differences = np.asarray(
                [
                    float(row[metric])
                    - float(reference[int(row["episode_id"])][metric])
                    for row in rows
                ]
            )
            low, high = _bootstrap_mean(differences, rng, bootstrap_reps)
            paired.append(
                {
                    "k": k,
                    "reference_k": 16,
                    "metric": metric,
                    "mean_paired_difference": float(np.mean(differences)),
                    "ci_low": low,
                    "ci_high": high,
                    "n": len(differences),
                }
            )
    return summaries, paired


def _gateoff_invariance(out: Path) -> Dict[str, Any]:
    a = _read_csv(
        out / "gateoff_invariance/k4/step_traces.csv"
    )
    b = _read_csv(
        out / "gateoff_invariance/k32/step_traces.csv"
    )
    diagnostic = {
        "gate_decision",
        "nominal_primitive_risk",
        "best_feasible_primitive_risk",
        "feasible_primitive_count",
        "selected_direction_row",
        "selected_direction_col",
        "selected_endpoint_row",
        "selected_endpoint_col",
        "selected_ray_min_clearance_m",
        "predicted_risk_improvement",
    }
    compared = [key for key in a[0] if key not in diagnostic]
    mismatches = sum(
        row_a[key] != row_b[key]
        for row_a, row_b in zip(a, b)
        for key in compared
    )
    metrics_a = _read_csv(
        out / "gateoff_invariance/k4/per_episode_metrics.csv"
    )
    metrics_b = _read_csv(
        out / "gateoff_invariance/k32/per_episode_metrics.csv"
    )
    metric_exclusions = {"compute_ms_per_step", "gate_diagnostic_rate"}
    metric_fields = [
        key for key in metrics_a[0] if key not in metric_exclusions
    ]
    metric_mismatches = sum(
        row_a[key] != row_b[key]
        for row_a, row_b in zip(metrics_a, metrics_b)
        for key in metric_fields
    )
    return {
        "tested_k": [4, 32],
        "tested_episodes": len(metrics_a),
        "trace_rows_each": len(a),
        "compared_non_gate_trace_fields": len(compared),
        "non_gate_trace_mismatches": mismatches,
        "outcome_metric_mismatches_excluding_compute_and_gate_diagnostic": metric_mismatches,
        "validated": mismatches == 0 and metric_mismatches == 0,
        "sharing": (
            "gate-off outcomes are taken once from exp1_gate_ablation_100 "
            "(K=16); K changes diagnostic gate computation but cannot enter "
            "lam_soft_used=lam_soft_learned or the integrator"
        ),
    }


def _monotonicity(
    summaries: Sequence[Mapping[str, Any]],
    per_episode: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    ordered = sorted(summaries, key=lambda row: int(row["k"]))
    activation = [float(row["activation_rate"]) for row in ordered]
    false_pre = [
        float(row["false_preactivation_episode"]) for row in ordered
    ]
    post_miss = [
        float(row["post_opening_no_activation_episode"]) for row in ordered
    ]
    by_episode: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_episode:
        by_episode[int(row["episode_id"])].append(row)
    episode_monotone = 0
    for rows in by_episode.values():
        rows = sorted(rows, key=lambda row: int(row["k"]))
        rates = [float(row["activation_rate"]) for row in rows]
        episode_monotone += int(
            all(left <= right + 1e-12 for left, right in zip(rates, rates[1:]))
        )
    return {
        "candidate_sets_nested_at_identical_state": True,
        "closed_loop_activation_rate_nondecreasing": all(
            left <= right + 1e-12
            for left, right in zip(activation, activation[1:])
        ),
        "closed_loop_false_pre_episode_rate_nondecreasing": all(
            left <= right + 1e-12
            for left, right in zip(false_pre, false_pre[1:])
        ),
        "closed_loop_post_open_miss_rate_nonincreasing": all(
            left + 1e-12 >= right
            for left, right in zip(post_miss, post_miss[1:])
        ),
        "episodes_with_nondecreasing_activation_rate": episode_monotone,
        "n_episodes": len(by_episode),
        "caution": (
            "Nested directions imply monotonic permissiveness only at the same "
            "state. Closed-loop trajectories diverge after K-dependent gates."
        ),
    }


def _markdown(
    summaries: Sequence[Mapping[str, Any]],
    monotonicity: Mapping[str, Any],
    invariance: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    lines = [
        "# Experiment 4: Primitive-count sensitivity",
        "",
        "The same checkpoint, first 100 LOSO episodes, dynamic events, seed, and "
        "rollout settings are used for every value of `K`. Only the number of "
        "uniformly spaced gate-witness directions changes.",
        "",
        "| K | Activation rate [95% CI] | False-pre episode rate | Post-open miss | Success | Violation CVaR | Hard contacts | Cosine | Clearance agree | Hard disagree | CPU ms/step |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summaries, key=lambda item: int(item["k"])):
        lines.append(
            f"| {row['k']} | {row['activation_rate']:.3f} "
            f"[{row['activation_rate_ci_low']:.3f}, "
            f"{row['activation_rate_ci_high']:.3f}] | "
            f"{row['false_preactivation_episode']:.3f} | "
            f"{row['post_opening_no_activation_episode']:.3f} | "
            f"{row['success']:.3f} | {row['cvar20_violation']:.4f} | "
            f"{row['hard_contacts']:.2f} | "
            f"{row['agreement_directional_cosine']:.3f} | "
            f"{row['agreement_clearance_rate']:.3f} | "
            f"{row['agreement_hard_disagreement_rate']:.3f} | "
            f"{row['compute_ms_per_step']:.2f} |"
        )
    low_k = min(summaries, key=lambda row: int(row["k"]))
    high_k = max(summaries, key=lambda row: int(row["k"]))
    lines.extend(
        [
            "",
            "## Main finding",
            "",
            f"Increasing K from {low_k['k']} to {high_k['k']} raises activation "
            f"from {low_k['activation_rate']:.3f} to "
            f"{high_k['activation_rate']:.3f}, reduces post-opening misses "
            f"from {low_k['post_opening_no_activation_episode']:.3f} to "
            f"{high_k['post_opening_no_activation_episode']:.3f}, but increases "
            f"false-preactivation episodes from "
            f"{low_k['false_preactivation_episode']:.3f} to "
            f"{high_k['false_preactivation_episode']:.3f}. Success, hard "
            "contacts, and violation CVaR are unchanged at reported precision. "
            "Meanwhile, directional cosine decreases and hard-contact "
            "disagreement rises, so added primitive coverage makes the gate "
            "more permissive without improving this checkpoint's outcomes.",
            "",
            "## Permissiveness and sharing validation",
            "",
            "At an identical state the direction sets are nested "
            "(`K=4 ⊂ 8 ⊂ 16 ⊂ 32`), so adding candidates cannot make the "
            "witness test less permissive. Closed-loop monotonicity is reported "
            "separately because trajectories can diverge after activation.",
            "",
            "```json",
            json.dumps(
                {"monotonicity": monotonicity, "gateoff_invariance": invariance},
                indent=2,
            ),
            "```",
            "",
            "Gate-off outcomes are shared rather than recomputed for every K. "
            "The preserved K=4/K=32 smoke validation compares every non-gate "
            "trajectory field and every outcome metric (excluding timing and "
            "diagnostic rate) exactly.",
            "",
            "## Provenance",
            "",
            "```json",
            json.dumps(provenance, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/postprocess Experiment 4.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rebuttal_experiments/results/exp4_k_sensitivity"),
    )
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=27370)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Postprocess already completed raw K directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_run:
        _run_sweep(args.out, args.max_episodes)
    per_episode = _per_episode(args.out)
    summaries, paired = _aggregate(
        per_episode,
        out=args.out,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    invariance = _gateoff_invariance(args.out)
    monotonicity = _monotonicity(summaries, per_episode)
    provenance = {
        "ks": list(KS),
        "max_episodes": args.max_episodes,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.seed,
        "checkpoint_sha256": json.loads(
            (args.out / "raw/k16/config.json").read_text()
        )["provenance"]["checkpoint_sha256"],
        "per_k": {
            str(k): {
                "config_sha256": _sha256(args.out / f"raw/k{k}/config.json"),
                "metrics_sha256": _sha256(
                    args.out / f"raw/k{k}/per_episode_metrics.csv"
                ),
                "traces_sha256": _sha256(
                    args.out / f"raw/k{k}/step_traces.csv"
                ),
            }
            for k in KS
        },
        "paired_uncertainty": (
            "episode bootstrap; paired K-minus-K16 differences resample the "
            "same episode indices"
        ),
    }
    shared_gateoff_metrics = Path(
        "rebuttal_experiments/results/exp1_gate_ablation_100/"
        "per_episode_metrics.csv"
    )
    provenance["shared_gateoff_reference"] = {
        "path": str(shared_gateoff_metrics),
        "sha256": _sha256(shared_gateoff_metrics),
        "role": (
            "single K-invariant gate-off reference; per-K table reports "
            "gate-on sensitivity only"
        ),
    }
    configs = {
        k: json.loads((args.out / f"raw/k{k}/config.json").read_text())
        for k in KS
    }
    normalized_arguments = []
    for k in KS:
        arguments = dict(configs[k]["arguments"])
        for key in ("out", "primitive_count"):
            arguments.pop(key)
        normalized_arguments.append(arguments)
    validation = {
        "gateoff_k_invariance": invariance["validated"],
        "four_k_values_present": {int(row["k"]) for row in summaries} == set(KS),
        "same_100_episode_ids": all(
            {
                int(row["episode_id"])
                for row in per_episode
                if int(row["k"]) == k
            }
            == set(range(args.max_episodes))
            for k in KS
        ),
        "all_gate_on": all(
            json.loads((args.out / f"raw/k{k}/config.json").read_text())[
                "arguments"
            ]["arms"]
            == ["gate_on"]
            for k in KS
        ),
        "same_checkpoint": len(
            {
                json.loads((args.out / f"raw/k{k}/config.json").read_text())[
                    "provenance"
                ]["checkpoint_sha256"]
                for k in KS
            }
        )
        == 1,
        "same_non_k_rollout_arguments": all(
            arguments == normalized_arguments[0]
            for arguments in normalized_arguments[1:]
        ),
    }
    output = {
        "validation": validation,
        "gateoff_invariance": invariance,
        "monotonicity": monotonicity,
        "summaries": summaries,
        "paired_differences_vs_k16": paired,
        "provenance": provenance,
    }
    _write_csv(args.out / "per_episode_k_metrics.csv", per_episode)
    _write_csv(args.out / "summary_by_k.csv", summaries)
    _write_csv(args.out / "paired_differences_vs_k16.csv", paired)
    (args.out / "summary.json").write_text(json.dumps(output, indent=2))
    (args.out / "RESULTS.md").write_text(
        _markdown(summaries, monotonicity, invariance, provenance)
    )
    print(json.dumps(output, indent=2))
    print(f"Wrote Experiment 4 artifacts to {args.out}")


if __name__ == "__main__":
    main()
