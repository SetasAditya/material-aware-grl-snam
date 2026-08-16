#!/usr/bin/env python3
"""Experiment 3: gate activation frequency, persistence, and flicker analysis.

This is a read-only postprocessor over the gate-on arm of Experiment 1.  The
new checkpoint-driven harness has no cooldown, latch, or hysteresis state:
every binary gate decision is recomputed from the current dynamic map.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


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


def _phase(row: Mapping[str, str]) -> str:
    step = int(row["step"])
    event_step = int(row["event_step"])
    opening_step = int(row["opening_step"])
    if step < event_step:
        return "pre_event"
    if step < opening_step:
        return "blocked_pre_opening"
    return "post_opening"


def _activation_runs(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Return active runs plus off→on and on→off transition counts."""
    if not rows:
        return [], 0, 0
    runs: List[Dict[str, Any]] = []
    off_to_on = 0
    on_to_off = 0
    previous = int(rows[0]["active"])
    run_start = 0 if previous else None
    for index in range(1, len(rows)):
        current = int(rows[index]["active"])
        if current == previous:
            continue
        if previous == 0:
            off_to_on += 1
            run_start = index
        else:
            on_to_off += 1
            assert run_start is not None
            runs.append(
                {
                    "start_index": run_start,
                    "end_index": index - 1,
                    "length_steps": index - run_start,
                }
            )
            run_start = None
        previous = current
    if previous == 1:
        assert run_start is not None
        runs.append(
            {
                "start_index": run_start,
                "end_index": len(rows) - 1,
                "length_steps": len(rows) - run_start,
            }
        )
    return runs, off_to_on, on_to_off


def _decorate_trace(
    raw_trace: Sequence[Mapping[str, str]],
    regimes: Mapping[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for raw in raw_trace:
        if raw["arm"] != "gate_on":
            continue
        episode_id = raw["episode_id"]
        by_episode[episode_id].append(
            {
                **raw,
                "step": int(raw["step"]),
                "active": int(raw["gate_decision"]),
                "phase": _phase(raw),
                "regime": regimes[episode_id],
                "lam_soft_learned": float(raw["lam_soft_learned"]),
                "lam_hard_used": float(raw["lam_hard_used"]),
            }
        )
    for rows in by_episode.values():
        rows.sort(key=lambda row: int(row["step"]))
    return by_episode


def _episode_record(
    episode_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    active = np.asarray([int(row["active"]) for row in rows], dtype=np.int64)
    runs, off_to_on, on_to_off = _activation_runs(rows)
    run_records: List[Dict[str, Any]] = []
    for run_index, run in enumerate(runs):
        start = rows[int(run["start_index"])]
        end = rows[int(run["end_index"])]
        phases = {str(rows[i]["phase"]) for i in range(
            int(run["start_index"]), int(run["end_index"]) + 1
        )}
        run_records.append(
            {
                "episode_id": episode_id,
                "regime": rows[0]["regime"],
                "run_index": run_index,
                "start_step": int(start["step"]),
                "end_step": int(end["step"]),
                "length_steps": int(run["length_steps"]),
                "isolated_one_step": int(int(run["length_steps"]) == 1),
                "start_phase": start["phase"],
                "end_phase": end["phase"],
                "crosses_phase_boundary": int(len(phases) > 1),
            }
        )
    lengths = [int(run["length_steps"]) for run in runs]
    record: Dict[str, Any] = {
        "episode_id": episode_id,
        "regime": rows[0]["regime"],
        "steps": len(rows),
        "active_steps": int(active.sum()),
        "activation_rate": float(active.mean()) if len(active) else 0.0,
        "any_activation": int(bool(active.any())),
        "activation_runs": len(runs),
        "mean_run_length_steps": float(np.mean(lengths)) if lengths else 0.0,
        "median_run_length_steps": float(np.median(lengths)) if lengths else 0.0,
        "max_run_length_steps": max(lengths, default=0),
        "isolated_one_step_runs": sum(length == 1 for length in lengths),
        "isolated_run_flicker_rate": (
            sum(length == 1 for length in lengths) / len(lengths)
            if lengths
            else 0.0
        ),
        "isolated_active_step_rate": (
            sum(length == 1 for length in lengths) / max(1, int(active.sum()))
        ),
        "off_to_on_transitions": off_to_on,
        "on_to_off_transitions": on_to_off,
        "total_transitions": off_to_on + on_to_off,
        "transitions_per_100_steps": (
            100.0 * (off_to_on + on_to_off) / max(1, len(rows) - 1)
        ),
    }
    for phase in PHASES:
        phase_rows = [row for row in rows if row["phase"] == phase]
        phase_active = sum(int(row["active"]) for row in phase_rows)
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
    return record, run_records


def _coefficient_distribution(
    rows: Sequence[Mapping[str, Any]], key: str
) -> Dict[str, Any]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def _group_summary(
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["episode_id"])].append(row)
    episode_stats: List[Dict[str, Any]] = []
    for episode_rows in grouped.values():
        episode_rows.sort(key=lambda row: int(row["step"]))
        runs, off_to_on, on_to_off = _activation_runs(episode_rows)
        lengths = [int(run["length_steps"]) for run in runs]
        episode_stats.append(
            {
                "any_activation": int(any(int(row["active"]) for row in episode_rows)),
                "runs": len(runs),
                "mean_run_length": float(np.mean(lengths)) if lengths else 0.0,
                "transitions": off_to_on + on_to_off,
                "transitions_per_100": (
                    100.0 * (off_to_on + on_to_off)
                    / max(1, len(episode_rows) - 1)
                ),
                "isolated_runs": sum(length == 1 for length in lengths),
                "isolated_rate": (
                    sum(length == 1 for length in lengths) / len(lengths)
                    if lengths
                    else 0.0
                ),
            }
        )
    active_rows = [row for row in rows if int(row["active"]) == 1]
    inactive_rows = [row for row in rows if int(row["active"]) == 0]
    return {
        "group": label,
        "n_episodes": len(grouped),
        "n_steps": len(rows),
        "active_steps": len(active_rows),
        "activation_rate": len(active_rows) / max(1, len(rows)),
        "episodes_with_any_activation": sum(
            int(record["any_activation"]) for record in episode_stats
        ),
        "episode_any_activation_rate": (
            np.mean([int(record["any_activation"]) for record in episode_stats])
            if episode_stats
            else 0.0
        ),
        "activation_runs": sum(int(record["runs"]) for record in episode_stats),
        "mean_episode_run_length_steps": (
            np.mean([float(record["mean_run_length"]) for record in episode_stats])
            if episode_stats
            else 0.0
        ),
        "total_transitions": sum(
            int(record["transitions"]) for record in episode_stats
        ),
        "mean_transitions_per_episode": (
            np.mean([int(record["transitions"]) for record in episode_stats])
            if episode_stats
            else 0.0
        ),
        "mean_transitions_per_100_steps": (
            np.mean([float(record["transitions_per_100"]) for record in episode_stats])
            if episode_stats
            else 0.0
        ),
        "isolated_one_step_runs": sum(
            int(record["isolated_runs"]) for record in episode_stats
        ),
        "pooled_mean_run_length_steps": (
            len(active_rows)
            / max(1, sum(int(record["runs"]) for record in episode_stats))
        ),
        "pooled_median_run_length_steps": (
            float(
                np.median(
                    [
                        int(run["length_steps"])
                        for episode_rows in grouped.values()
                        for run in _activation_runs(episode_rows)[0]
                    ]
                )
            )
            if active_rows
            else 0.0
        ),
        "pooled_isolated_run_flicker_rate": (
            sum(int(record["isolated_runs"]) for record in episode_stats)
            / max(1, sum(int(record["runs"]) for record in episode_stats))
        ),
        "mean_episode_isolated_run_flicker_rate": (
            np.mean(
                [float(record["isolated_rate"]) for record in episode_stats]
            )
            if episode_stats
            else 0.0
        ),
        "mean_lam_soft_when_active": (
            float(np.mean([float(row["lam_soft_learned"]) for row in active_rows]))
            if active_rows
            else None
        ),
        "mean_lam_soft_when_inactive": (
            float(np.mean([float(row["lam_soft_learned"]) for row in inactive_rows]))
            if inactive_rows
            else None
        ),
        "mean_lam_hard_when_active": (
            float(np.mean([float(row["lam_hard_used"]) for row in active_rows]))
            if active_rows
            else None
        ),
        "mean_lam_hard_when_inactive": (
            float(np.mean([float(row["lam_hard_used"]) for row in inactive_rows]))
            if inactive_rows
            else None
        ),
    }


def _summaries(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    groups: List[Tuple[str, List[Mapping[str, Any]]]] = [("overall", list(rows))]
    for regime in ("R1", "R2", "R3"):
        groups.append(
            (
                f"regime={regime}",
                [row for row in rows if row["regime"] == regime],
            )
        )
    for phase in PHASES:
        groups.append(
            (
                f"phase={phase}",
                [row for row in rows if row["phase"] == phase],
            )
        )
    for regime in ("R1", "R2", "R3"):
        for phase in PHASES:
            groups.append(
                (
                    f"regime={regime},phase={phase}",
                    [
                        row
                        for row in rows
                        if row["regime"] == regime and row["phase"] == phase
                    ],
                )
            )
    return [
        _group_summary(pool, label) for label, pool in groups
    ]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _markdown(
    summaries: Sequence[Mapping[str, Any]],
    headline: Mapping[str, Any],
    coefficients: Mapping[str, Any],
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    lines = [
        "# Experiment 3: Gate activation analysis",
        "",
        "This analysis uses the `gate_on` learned-controller trajectories. The "
        "gate is recomputed independently at every step; this harness contains "
        "**no cooldown, latch, or hysteresis state**.",
        "",
        "The blocked pre-opening phase is `event_step ≤ step < opening_step`. "
        "Activation there is counted as false pre-activation. Ordinary behavior "
        "before event onset is reported separately.",
        "",
        "## Headline episode rates",
        "",
        f"- Episodes with any activation: {headline['episodes_with_any_activation']}/"
        f"{headline['n_episodes']} "
        f"({_fmt(headline['episode_any_activation_rate'])})",
        f"- False-preactivation episodes: "
        f"{headline['false_preactivation_episodes']}/{headline['n_episodes']} "
        f"({_fmt(headline['false_preactivation_episode_rate'])})",
        f"- Post-opening no-activation episodes: "
        f"{headline['post_opening_no_activation_episodes']}/"
        f"{headline['n_episodes']} "
        f"({_fmt(headline['post_opening_no_activation_episode_rate'])})",
        f"- Activation runs: {headline['activation_runs']} "
        f"(mean/median/max length "
        f"{_fmt(headline['mean_activation_run_length_steps'])}/"
        f"{_fmt(headline['median_activation_run_length_steps'])}/"
        f"{headline['max_activation_run_length_steps']} steps)",
        f"- Isolated one-step runs: {headline['isolated_one_step_runs']}/"
        f"{headline['activation_runs']} "
        f"({_fmt(headline['isolated_run_flicker_rate'])})",
        f"- Gate transitions: {headline['total_transitions']} "
        f"({headline['off_to_on_transitions']} off→on, "
        f"{headline['on_to_off_transitions']} on→off)",
        "",
        "## Frequency and temporal stability",
        "",
        "| Group | Episodes | Steps | Active | Activation rate | Episodes active | Runs | Transitions | Trans./100 steps | One-step runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {group} | {episodes} | {steps} | {active} | {rate} | "
            "{ep_active} | {runs} | {transitions} | {transition_rate} | "
            "{flickers} |".format(
                group=row["group"],
                episodes=row["n_episodes"],
                steps=row["n_steps"],
                active=row["active_steps"],
                rate=_fmt(row["activation_rate"]),
                ep_active=row["episodes_with_any_activation"],
                runs=row["activation_runs"],
                transitions=row["total_transitions"],
                transition_rate=_fmt(row["mean_transitions_per_100_steps"]),
                flickers=row["isolated_one_step_runs"],
            )
        )
    lines.extend(
        [
            "",
            "## Learned coefficient distributions by gate state",
            "",
            "The gate changes only the multiplier applied to learned `lam_soft`; "
            "`lam_hard` remains active in both states.",
            "",
            "| Gate state | Coefficient | n | Mean | Std | P10 | Median | P90 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for state in ("active", "inactive"):
        for key, label in (("lam_soft_learned", "lam_soft"), ("lam_hard_used", "lam_hard")):
            dist = coefficients[state][key]
            lines.append(
                f"| {state} | {label} | {dist['n']} | {_fmt(dist['mean'], 5)} | "
                f"{_fmt(dist['std'], 5)} | {_fmt(dist['p10'], 5)} | "
                f"{_fmt(dist['median'], 5)} | {_fmt(dist['p90'], 5)} |"
            )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "```json",
            json.dumps(validation, indent=2),
            "```",
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


def _self_test_runs() -> bool:
    synthetic = [{"active": value} for value in (0, 1, 1, 0, 1, 0, 0, 1)]
    runs, off_to_on, on_to_off = _activation_runs(synthetic)
    return (
        [run["length_steps"] for run in runs] == [2, 1, 1]
        and off_to_on == 3
        and on_to_off == 2
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze activation frequency and flicker in Experiment 1."
    )
    parser.add_argument(
        "--exp1-results",
        type=Path,
        default=Path("rebuttal_experiments/results/exp1_gate_ablation_100"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rebuttal_experiments/results/exp3_gate_activation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trace_path = args.exp1_results / "step_traces.csv"
    metrics_path = args.exp1_results / "per_episode_metrics.csv"
    config_path = args.exp1_results / "config.json"
    metrics = _read_csv(metrics_path)
    regimes = {
        row["episode_id"]: row["regime"]
        for row in metrics
        if row["arm"] == "gate_on"
    }
    raw_trace = _read_csv(trace_path)
    by_episode = _decorate_trace(raw_trace, regimes)

    episode_records: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []
    for episode_id, rows in sorted(
        by_episode.items(), key=lambda item: int(item[0])
    ):
        record, runs = _episode_record(episode_id, rows)
        episode_records.append(record)
        run_records.extend(runs)
    all_rows = [
        row
        for episode_id in sorted(by_episode, key=int)
        for row in by_episode[episode_id]
    ]
    summaries = _summaries(all_rows)
    active_rows = [row for row in all_rows if int(row["active"]) == 1]
    inactive_rows = [row for row in all_rows if int(row["active"]) == 0]
    coefficients = {
        state: {
            key: _coefficient_distribution(pool, key)
            for key in ("lam_soft_learned", "lam_hard_used")
        }
        for state, pool in (("active", active_rows), ("inactive", inactive_rows))
    }
    headline = {
        "n_episodes": len(episode_records),
        "episodes_with_any_activation": sum(
            int(row["any_activation"]) for row in episode_records
        ),
        "episode_any_activation_rate": np.mean(
            [int(row["any_activation"]) for row in episode_records]
        ),
        "false_preactivation_episodes": sum(
            int(row["false_preactivation_episode"]) for row in episode_records
        ),
        "false_preactivation_episode_rate": np.mean(
            [int(row["false_preactivation_episode"]) for row in episode_records]
        ),
        "post_opening_no_activation_episodes": sum(
            int(row["post_opening_no_activation_episode"])
            for row in episode_records
        ),
        "post_opening_no_activation_episode_rate": np.mean(
            [
                int(row["post_opening_no_activation_episode"])
                for row in episode_records
            ]
        ),
        "activation_runs": len(run_records),
        "mean_activation_run_length_steps": float(
            np.mean([int(row["length_steps"]) for row in run_records])
        ),
        "median_activation_run_length_steps": float(
            np.median([int(row["length_steps"]) for row in run_records])
        ),
        "max_activation_run_length_steps": max(
            int(row["length_steps"]) for row in run_records
        ),
        "isolated_one_step_runs": sum(
            int(row["isolated_one_step"]) for row in run_records
        ),
        "isolated_run_flicker_rate": np.mean(
            [int(row["isolated_one_step"]) for row in run_records]
        ),
        "off_to_on_transitions": sum(
            int(row["off_to_on_transitions"]) for row in episode_records
        ),
        "on_to_off_transitions": sum(
            int(row["on_to_off_transitions"]) for row in episode_records
        ),
        "total_transitions": sum(
            int(row["total_transitions"]) for row in episode_records
        ),
    }

    phases_total = sum(
        sum(int(row[f"{phase}_steps"]) for phase in PHASES)
        for row in episode_records
    )
    active_from_runs = sum(int(row["length_steps"]) for row in run_records)
    validation = {
        "synthetic_run_segmentation_test": _self_test_runs(),
        "input_contains_both_exp1_arms": {row["arm"] for row in raw_trace}
        == {"gate_on", "gate_off"},
        "analysis_uses_gate_on_only": all(
            row["arm"] == "gate_on" for row in all_rows
        ),
        "binary_gate_values": all(int(row["active"]) in (0, 1) for row in all_rows),
        "unique_contiguous_steps_per_episode": all(
            [int(row["step"]) for row in rows] == list(range(len(rows)))
            for rows in by_episode.values()
        ),
        "phase_partition_matches_total_steps": phases_total == len(all_rows),
        "run_lengths_sum_to_active_steps": active_from_runs == len(active_rows),
        "episode_count_matches_exp1_gate_on": len(episode_records)
        == sum(row["arm"] == "gate_on" for row in metrics),
        "all_coefficients_finite": all(
            math.isfinite(float(row[key]))
            for row in all_rows
            for key in ("lam_soft_learned", "lam_hard_used")
        ),
        "cooldown_or_latch_present": False,
    }
    provenance = {
        "input_results": str(args.exp1_results),
        "step_traces_sha256": _sha256(trace_path),
        "per_episode_metrics_sha256": _sha256(metrics_path),
        "config_sha256": _sha256(config_path),
        "analyzed_arm": "gate_on",
        "gate_semantics": "binary decision recomputed independently every step",
        "cooldown_latch_hysteresis": "none in checkpoint-driven harness",
        "phase_definitions": {
            "pre_event": "step < event_step",
            "blocked_pre_opening": "event_step <= step < opening_step",
            "post_opening": "step >= opening_step",
        },
        "isolated_flicker_definition": "active run of exactly one controller step",
    }
    output = {
        "headline": headline,
        "coefficient_distributions": coefficients,
        "validation": validation,
        "provenance": provenance,
        "summaries": summaries,
    }
    _write_csv(args.out / "per_episode_activation.csv", episode_records)
    _write_csv(args.out / "activation_runs.csv", run_records)
    _write_csv(args.out / "summary_by_regime_phase.csv", summaries)
    (args.out / "summary.json").write_text(json.dumps(output, indent=2))
    (args.out / "RESULTS.md").write_text(
        _markdown(summaries, headline, coefficients, validation, provenance)
    )
    print(json.dumps(output, indent=2))
    print(f"Wrote Experiment 3 artifacts to {args.out}")


if __name__ == "__main__":
    main()
