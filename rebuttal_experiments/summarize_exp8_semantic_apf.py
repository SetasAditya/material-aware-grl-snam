#!/usr/bin/env python3
"""Summarize Experiment 8 without modifying any source run or cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


ALIASES = {"neural_potential_field": "semantic_apf"}
METHODS = ("semantic_apf", "route_aware_stage2", "dwa_semantic")
METHOD_LABELS = {
    "semantic_apf": "Semantic APF",
    "route_aware_stage2": "Route-aware Stage 2",
    "dwa_semantic": "Semantic DWA",
}
METRICS = (
    "success",
    "hard_hazard_length_m",
    "post_event_cvar_violation",
    "reaction_delay",
    "route_deviation_delay",
    "path_length_ratio",
    "stuck",
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if "method" in row:
            row["method"] = ALIASES.get(row["method"], row["method"])
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> Tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def _index(rows: Iterable[Mapping[str, str]]) -> Dict[str, Dict[str, Mapping[str, str]]]:
    out: Dict[str, Dict[str, Mapping[str, str]]] = {}
    for row in rows:
        method = row["method"]
        if method in METHODS:
            out.setdefault(row["episode_id"], {})[method] = row
    return out


def _derived(row: Mapping[str, str], metric: str) -> float:
    if metric == "pre_open_deviation_proxy":
        return float(float(row["route_deviation_delay"]) < 10.0)
    return float(row[metric])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delayed-run", type=Path, required=True)
    ap.add_argument("--historical-8event-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=27370)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    delayed_path = args.delayed_run / "dynamic_rollouts.csv"
    historical_path = args.historical_8event_run / "dynamic_summary_by_event.csv"
    rows = _read_csv(delayed_path)
    paired = {
        episode_id: methods
        for episode_id, methods in _index(rows).items()
        if all(method in methods for method in METHODS)
    }
    if not paired:
        raise RuntimeError("No complete paired episodes found")

    rng = np.random.default_rng(args.seed)
    metrics = (*METRICS, "pre_open_deviation_proxy")
    aggregate_rows: List[Dict[str, object]] = []
    for method in METHODS:
        for metric in metrics:
            vals = np.asarray(
                [_derived(method_rows[method], metric) for method_rows in paired.values()],
                dtype=np.float64,
            )
            mean, lo, hi = _mean_ci(vals, rng, args.bootstrap_samples)
            aggregate_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "metric": metric,
                    "n_paired": len(vals),
                    "mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    _write_csv(args.out / "aggregate_metrics.csv", aggregate_rows)

    difference_rows: List[Dict[str, object]] = []
    for comparator in ("route_aware_stage2", "dwa_semantic"):
        for metric in metrics:
            diffs = np.asarray(
                [
                    _derived(method_rows[comparator], metric)
                    - _derived(method_rows["semantic_apf"], metric)
                    for method_rows in paired.values()
                ],
                dtype=np.float64,
            )
            mean, lo, hi = _mean_ci(diffs, rng, args.bootstrap_samples)
            difference_rows.append(
                {
                    "difference": f"{comparator} - semantic_apf",
                    "metric": metric,
                    "n_paired": len(diffs),
                    "mean_difference": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    _write_csv(args.out / "paired_differences_vs_semantic_apf.csv", difference_rows)

    historical_rows = [
        row for row in _read_csv(historical_path) if row.get("method") in METHODS
    ]
    _write_csv(args.out / "historical_8event_preliminary.csv", historical_rows)

    summarizer_source = Path(__file__).resolve()
    source = summarizer_source.parents[1] / "full_code" / "exp-rellis" / "eval_rellis_dyn.py"
    historical_source = args.historical_8event_run.parents[1] / "eval_rellis_dyn.py"
    delayed_summary = json.loads((args.delayed_run / "summary.json").read_text())
    bev_root = Path(delayed_summary["config"]["bev_root"])
    pairs_root = Path(delayed_summary["config"]["pairs_root"])
    inputs = [
        delayed_path,
        args.delayed_run / "summary.json",
        historical_path,
        args.historical_8event_run / "summary.json",
        bev_root / "manifest.json",
        pairs_root / "manifest.json",
        source,
        summarizer_source,
    ]
    if historical_source.exists():
        inputs.append(historical_source)
    provenance = {
        "experiment": "exp8_semantic_apf",
        "canonical_baseline": "semantic_apf",
        "deprecated_input_alias": "neural_potential_field",
        "baseline_is_learned": False,
        "description": (
            "Fixed one-step, 8-connected semantic artificial potential field "
            "using goal attraction, semantic risk, hard clearance, and progress."
        ),
        "paired_episode_count": len(paired),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "delayed_run_config": delayed_summary["config"],
        "historical_8event_is_preliminary": True,
        "sha256": {str(path): _sha256(path) for path in inputs},
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    lookup = {(r["method"], r["metric"]): r for r in aggregate_rows}
    report = [
        "# Experiment 8: fixed semantic APF baseline",
        "",
        "The former `neural_potential_field` baseline is now accurately named "
        "`semantic_apf`. It contains no neural network, learned weights, or "
        "training. It performs one-step descent over eight neighboring cells "
        "using fixed goal, semantic-risk, clearance, and progress terms.",
        "",
        f"Delayed-required-escape results use {len(paired)} paired episodes. "
        "Intervals are episode bootstrap 95% confidence intervals "
        f"({args.bootstrap_samples:,} resamples; seed {args.seed}).",
        "",
        "| Method | Success | Pre-open deviation proxy | Violation CVaR | Hard length (m) | Path ratio | Stuck |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        vals = []
        for metric in (
            "success",
            "pre_open_deviation_proxy",
            "post_event_cvar_violation",
            "hard_hazard_length_m",
            "path_length_ratio",
            "stuck",
        ):
            row = lookup[(method, metric)]
            vals.append(
                f"{float(row['mean']):.3f} "
                f"[{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}]"
            )
        report.append(f"| {METHOD_LABELS[method]} | " + " | ".join(vals) + " |")
    report.extend(
        [
            "",
            "The pre-open deviation measure is only a trajectory proxy: whether "
            "the rollout deviates more than 1 m from the nominal route before "
            "the escape opens. The APF and DWA have no internal activation gate.",
            "",
            "The historical eight-event sweep is included only as a preliminary "
            "descriptive comparison. It was generated before the rename, but the "
            "controller implementation is the same fixed baseline; copied rows "
            "are canonicalized to `semantic_apf`.",
            "",
            "Limitations: this is a lightweight grid APF comparison, not an RMP "
            "or Geometric Fabric implementation; its weights were not tuned in a "
            "nested validation protocol; and it follows the next Stage-1 waypoint, "
            "so it is not independent of the nominal scaffold.",
            "",
            "Machine-readable results: `aggregate_metrics.csv`, "
            "`paired_differences_vs_semantic_apf.csv`, "
            "`historical_8event_preliminary.csv`, and `provenance.json`.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python full_code/exp-rellis/eval_rellis_dyn.py \\",
            "  --bev-root /mnt/data/adityas/GRL-SNAM/exp-rellis/cache/rellis_bev_all_seqbalanced_2500 \\",
            "  --pairs-root /mnt/data/adityas/GRL-SNAM/exp-rellis/cache/rellis_pairs_all_seqbalanced_2500_loso \\",
            "  --out rebuttal_experiments/results/exp8_semantic_apf_delayed \\",
            "  --event-types delayed_required_escape \\",
            "  --methods semantic_apf route_aware_stage2 dwa_semantic \\",
            "  --max-episodes 100 --progress-every 10",
            "",
            "python rebuttal_experiments/summarize_exp8_semantic_apf.py \\",
            "  --delayed-run rebuttal_experiments/results/exp8_semantic_apf_delayed \\",
            "  --historical-8event-run /mnt/data/adityas/GRL-SNAM/exp-rellis/runs/rellis_dyn_8events_fast_100 \\",
            "  --out rebuttal_experiments/results/exp8_semantic_apf_delayed \\",
            "  --bootstrap-samples 10000 --seed 27370",
            "```",
        ]
    )
    (args.out / "REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
