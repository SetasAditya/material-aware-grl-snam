#!/usr/bin/env python3
"""Experiment 2: compare gate activation witnesses to learned-field motion.

The selected primitive is an *activation witness*, not a trajectory command.
This postprocessor asks whether a gate-positive witness is nevertheless
consistent with the subsequently executed CoefEnergyNetMaterial trajectory.

Input is the instrumented ``step_traces.csv`` from Experiment 1.  For each
gate-positive decision in the gate-on arm, the actual trajectory is followed
until it accumulates the primitive's horizon in grid-cell arc length.  This
    arc-length matching avoids comparing a 12-cell primitive with an arbitrary
number of controller updates. The first observed controller endpoint at or
beyond the target arc length is used so all risk and clearance samples remain
actual map observations rather than interpolated estimates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


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


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    values_np = np.asarray(values, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    if len(values_np) == 0:
        return float("nan")
    if float(weights_np.sum()) <= 1e-12:
        return float(np.mean(values_np))
    return float(np.average(values_np, weights=weights_np))


def _future_motion(
    rows: Sequence[Mapping[str, str]],
    decision_index: int,
    target_arc_cells: float,
) -> Dict[str, Any]:
    decision = rows[decision_index]
    start_rc = np.asarray(
        [float(decision["position_y"]), float(decision["position_x"])],
        dtype=np.float64,
    )
    points = [start_rc]
    risks: List[float] = []
    weights: List[float] = []
    clearances: List[float] = []
    hard_contacts: List[int] = []
    accumulated = 0.0
    controller_steps = 0
    final_step = int(decision["step"])
    for future in rows[decision_index:]:
        p0 = points[-1]
        raw_p1 = np.asarray(
            [float(future["next_y"]), float(future["next_x"])],
            dtype=np.float64,
        )
        segment = raw_p1 - p0
        segment_length = float(np.linalg.norm(segment))
        p1 = raw_p1
        used_length = segment_length
        points.append(p1)
        risks.append(float(future["risk"]))
        weights.append(max(used_length, 1e-12))
        clearances.append(float(future["sdf_clearance_m"]))
        hard_contacts.append(int(future["hard_contact"]))
        accumulated += used_length
        controller_steps += 1
        final_step = int(future["step"])
        if accumulated >= target_arc_cells - 1e-8:
            break
    return {
        "start_rc": start_rc,
        "points_rc": np.asarray(points, dtype=np.float64),
        "endpoint_rc": np.asarray(points[-1], dtype=np.float64),
        "arc_cells": accumulated,
        "controller_steps": controller_steps,
        "final_step": final_step,
        "mean_risk": _weighted_mean(risks, weights),
        "min_clearance_m": min(clearances) if clearances else float("nan"),
        "any_hard_contact": bool(any(hard_contacts)),
        "complete": bool(accumulated >= 0.95 * target_arc_cells),
    }


def _decision_record(
    decision: Mapping[str, str],
    future: Mapping[str, Any],
    *,
    regime: str,
    gsd: float,
    hard_margin_m: float,
    horizon_cells: int,
) -> Dict[str, Any]:
    start_rc = np.asarray(future["start_rc"], dtype=np.float64)
    selected_direction = np.asarray(
        [
            float(decision["selected_direction_row"]),
            float(decision["selected_direction_col"]),
        ],
        dtype=np.float64,
    )
    selected_direction /= max(float(np.linalg.norm(selected_direction)), 1e-12)
    selected_endpoint = np.asarray(
        [
            float(decision["selected_endpoint_row"]),
            float(decision["selected_endpoint_col"]),
        ],
        dtype=np.float64,
    )
    actual_endpoint = np.asarray(future["endpoint_rc"], dtype=np.float64)
    actual_displacement = actual_endpoint - start_rc
    actual_distance = float(np.linalg.norm(actual_displacement))
    if actual_distance > 1e-12:
        directional_cosine = float(
            np.dot(selected_direction, actual_displacement / actual_distance)
        )
    else:
        directional_cosine = float("nan")

    offsets = np.asarray(future["points_rc"], dtype=np.float64) - start_rc[None, :]
    # In 2-D, |cross(unit_direction, offset)| is perpendicular distance to
    # the infinite selected ray.  Negative along-ray points are separately
    # penalized through directional cosine.
    cross_track_cells = np.abs(
        selected_direction[0] * offsets[:, 1]
        - selected_direction[1] * offsets[:, 0]
    )
    mean_cross_track_m = gsd * float(np.mean(cross_track_cells))
    endpoint_deviation_m = gsd * float(
        np.linalg.norm(actual_endpoint - selected_endpoint)
    )
    selected_min_clearance = float(decision["selected_ray_min_clearance_m"])
    actual_min_clearance = float(future["min_clearance_m"])
    selected_clear = selected_min_clearance >= hard_margin_m
    actual_clear = actual_min_clearance >= hard_margin_m
    nominal_risk = float(decision["nominal_primitive_risk"])
    predicted_improvement = float(decision["predicted_risk_improvement"])
    realized_improvement = nominal_risk - float(future["mean_risk"])
    step = int(decision["step"])
    event_step = int(decision["event_step"])
    opening_step = int(decision["opening_step"])
    if step < event_step:
        phase = "before_event"
    elif step < opening_step:
        phase = "pre_opening"
    else:
        phase = "post_opening"
    return {
        "episode_id": decision["episode_id"],
        "source_gate_decision": int(decision["gate_decision"]),
        "regime": regime,
        "phase": phase,
        "decision_step": step,
        "event_step": event_step,
        "opening_step": opening_step,
        "actual_horizon_final_step": int(future["final_step"]),
        "horizon_crosses_opening": int(
            step < opening_step <= int(future["final_step"])
        ),
        "primitive_horizon_cells": horizon_cells,
        "actual_horizon_complete": int(bool(future["complete"])),
        "actual_controller_steps": int(future["controller_steps"]),
        "actual_arc_length_cells": float(future["arc_cells"]),
        "selected_direction_row": float(selected_direction[0]),
        "selected_direction_col": float(selected_direction[1]),
        "selected_endpoint_row": float(selected_endpoint[0]),
        "selected_endpoint_col": float(selected_endpoint[1]),
        "actual_endpoint_row": float(actual_endpoint[0]),
        "actual_endpoint_col": float(actual_endpoint[1]),
        "actual_displacement_row": float(actual_displacement[0]),
        "actual_displacement_col": float(actual_displacement[1]),
        "directional_cosine": directional_cosine,
        "endpoint_deviation_m": endpoint_deviation_m,
        "mean_cross_track_deviation_m": mean_cross_track_m,
        "selected_ray_min_clearance_m": selected_min_clearance,
        "actual_path_min_clearance_m": actual_min_clearance,
        "clearance_threshold_m": hard_margin_m,
        "clearance_agreement": int(selected_clear == actual_clear),
        "actual_clearance_safe": int(actual_clear),
        "hard_contact_disagreement": int(bool(future["any_hard_contact"])),
        "nominal_primitive_risk": nominal_risk,
        "selected_primitive_risk": float(
            decision["best_feasible_primitive_risk"]
        ),
        "executed_path_mean_risk": float(future["mean_risk"]),
        "predicted_risk_improvement": predicted_improvement,
        "realized_risk_improvement": realized_improvement,
        "improvement_sign_agreement": int(realized_improvement > 0.0),
        "absolute_improvement_error": abs(
            predicted_improvement - realized_improvement
        ),
    }


def _finite(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
    return values[np.isfinite(values)]


def _summarize_group(
    records: Sequence[Mapping[str, Any]], label: str
) -> Dict[str, Any]:
    complete = [record for record in records if int(record["actual_horizon_complete"])]
    pool = complete
    result: Dict[str, Any] = {
        "group": label,
        "n_gate_positive": len(records),
        "n_complete_horizon": len(complete),
        "complete_horizon_rate": len(complete) / max(1, len(records)),
    }
    if not pool:
        return result
    for key in (
        "directional_cosine",
        "endpoint_deviation_m",
        "mean_cross_track_deviation_m",
        "selected_ray_min_clearance_m",
        "actual_path_min_clearance_m",
        "predicted_risk_improvement",
        "realized_risk_improvement",
        "absolute_improvement_error",
    ):
        values = _finite(pool, key)
        result[f"mean_{key}"] = float(np.mean(values)) if len(values) else None
        result[f"median_{key}"] = (
            float(np.median(values)) if len(values) else None
        )
    for key in (
        "clearance_agreement",
        "actual_clearance_safe",
        "hard_contact_disagreement",
        "improvement_sign_agreement",
        "horizon_crosses_opening",
    ):
        result[f"{key}_rate"] = float(
            np.mean([float(record[key]) for record in pool])
        )
    predicted = _finite(pool, "predicted_risk_improvement")
    realized = _finite(pool, "realized_risk_improvement")
    if (
        len(predicted) >= 2
        and len(realized) == len(predicted)
        and float(np.std(predicted)) > 1e-12
        and float(np.std(realized)) > 1e-12
    ):
        result["predicted_realized_pearson_r"] = float(
            np.corrcoef(predicted, realized)[0, 1]
        )
    else:
        result["predicted_realized_pearson_r"] = None
    return result


def _all_summaries(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Tuple[str, List[Mapping[str, Any]]]] = [("overall", list(records))]
    regimes = sorted({str(record["regime"]) for record in records})
    phases = ("before_event", "pre_opening", "post_opening")
    groups.extend(
        (
            f"regime={regime}",
            [record for record in records if record["regime"] == regime],
        )
        for regime in regimes
    )
    groups.extend(
        (
            f"phase={phase}",
            [record for record in records if record["phase"] == phase],
        )
        for phase in phases
    )
    groups.extend(
        (
            f"regime={regime},phase={phase}",
            [
                record
                for record in records
                if record["regime"] == regime and record["phase"] == phase
            ],
        )
        for regime in regimes
        for phase in phases
    )
    # A witness formed just before opening may be evaluated against a future
    # map that changes within its 12-cell horizon. Keep those decisions
    # separate so environment change is not misreported as execution mismatch.
    groups.extend(
        (
            f"phase=pre_opening,crosses_opening={crosses}",
            [
                record
                for record in records
                if record["phase"] == "pre_opening"
                and int(record["horizon_crosses_opening"]) == crosses
            ],
        )
        for crosses in (0, 1)
    )
    return [_summarize_group(pool, label) for label, pool in groups]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _markdown(
    summaries: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    lines = [
        "# Experiment 2: Gate witness–trajectory agreement",
        "",
        "The sampled primitive is an **activation witness**, not a trajectory "
        "command. The learned Hamiltonian field still determines the executed "
        "motion. This audit measures whether those two objects agree over an "
        "arc-length-matched short horizon.",
        "",
        "Only gate-positive decisions from the learned `gate_on` rollouts are "
        "included. Aggregate metrics use decisions with at least 95% of the "
        "12-cell horizon observed.",
        "",
        "| Group | Gate + | Complete | Cosine ↑ | Endpoint dev. m ↓ | Cross-track m ↓ | Clearance agree ↑ | Hard disagreement ↓ | Pred. Δrisk | Realized Δrisk | Sign agree ↑ | Pearson r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {group} | {n} | {complete} | {cosine} | {endpoint} | "
            "{cross} | {clear} | {hard} | {pred} | {realized} | "
            "{sign} | {corr} |".format(
                group=row["group"],
                n=row["n_gate_positive"],
                complete=row["n_complete_horizon"],
                cosine=_fmt(row.get("mean_directional_cosine")),
                endpoint=_fmt(row.get("mean_endpoint_deviation_m")),
                cross=_fmt(row.get("mean_mean_cross_track_deviation_m")),
                clear=_fmt(row.get("clearance_agreement_rate")),
                hard=_fmt(row.get("hard_contact_disagreement_rate")),
                pred=_fmt(row.get("mean_predicted_risk_improvement"), 4),
                realized=_fmt(row.get("mean_realized_risk_improvement"), 4),
                sign=_fmt(row.get("improvement_sign_agreement_rate")),
                corr=_fmt(row.get("predicted_realized_pearson_r")),
            )
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
            "Interpretation caution: agreement indicates that a local witness "
            "and the resulting field motion are directionally compatible. It "
            "does not turn the witness into a safety certificate or prove that "
            "the integrator executes the sampled primitive.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess Experiment 1 gate-positive trajectory agreement."
    )
    parser.add_argument(
        "--exp1-results",
        type=Path,
        default=Path("rebuttal_experiments/results/exp1_gate_ablation_100"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rebuttal_experiments/results/exp2_gate_trajectory_agreement"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trace_path = args.exp1_results / "step_traces.csv"
    metrics_path = args.exp1_results / "per_episode_metrics.csv"
    config_path = args.exp1_results / "config.json"
    config = json.loads(config_path.read_text())
    bev_manifest_path = (
        Path(config["arguments"]["bev_root"]) / "manifest.json"
    )
    bev_manifest = json.loads(bev_manifest_path.read_text())
    gsd = float(bev_manifest["config"]["bev"]["resolution"])
    horizon_cells = int(config["arguments"]["primitive_horizon_cells"])
    hard_margin_m = float(config["arguments"]["hard_margin_m"])
    improvement_margin = float(config["arguments"]["improvement_margin"])

    metrics = _read_csv(metrics_path)
    regimes = {
        row["episode_id"]: row["regime"]
        for row in metrics
        if row["arm"] == "gate_on"
    }
    raw_trace = [
        row
        for row in _read_csv(trace_path)
        if row["arm"] == "gate_on"
    ]
    by_episode: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in raw_trace:
        by_episode[row["episode_id"]].append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda row: int(row["step"]))

    decisions: List[Dict[str, Any]] = []
    for episode_id, rows in sorted(
        by_episode.items(), key=lambda item: int(item[0])
    ):
        for index, decision in enumerate(rows):
            if int(decision["gate_decision"]) != 1:
                continue
            future = _future_motion(rows, index, float(horizon_cells))
            decisions.append(
                _decision_record(
                    decision,
                    future,
                    regime=regimes[episode_id],
                    gsd=gsd,
                    hard_margin_m=hard_margin_m,
                    horizon_cells=horizon_cells,
                )
            )

    summaries = _all_summaries(decisions)
    complete = [
        record for record in decisions if int(record["actual_horizon_complete"])
    ]
    validation = {
        "all_input_rows_gate_on": all(row["arm"] == "gate_on" for row in raw_trace),
        "all_decisions_gate_positive": all(
            int(record["source_gate_decision"]) == 1
            for record in decisions
        ),
        "selected_primitives_meet_clearance_threshold": all(
            float(record["selected_ray_min_clearance_m"]) + 1e-8
            >= hard_margin_m
            for record in decisions
        ),
        "predicted_improvements_meet_gate_margin": all(
            float(record["predicted_risk_improvement"]) + 1e-8
            >= improvement_margin
            for record in decisions
        ),
        "complete_alignment_in_unit_interval": all(
            -1.0 - 1e-8 <= float(record["directional_cosine"]) <= 1.0 + 1e-8
            for record in complete
            if math.isfinite(float(record["directional_cosine"]))
        ),
        "nonnegative_deviations": all(
            float(record["endpoint_deviation_m"]) >= 0.0
            and float(record["mean_cross_track_deviation_m"]) >= 0.0
            for record in decisions
        ),
        "num_gate_positive_decisions": len(decisions),
        "num_complete_horizon_decisions": len(complete),
        "num_episodes_with_gate_positive": len(
            {record["episode_id"] for record in decisions}
        ),
    }
    provenance = {
        "input_results": str(args.exp1_results),
        "step_traces_sha256": _sha256(trace_path),
        "per_episode_metrics_sha256": _sha256(metrics_path),
        "config_sha256": _sha256(config_path),
        "bev_manifest_sha256": _sha256(bev_manifest_path),
        "grid_resolution_m_per_cell": gsd,
        "primitive_horizon_cells": horizon_cells,
        "actual_horizon_definition": (
            "subsequent learned-field path through the first observed "
            "controller endpoint reaching/exceeding equal grid-cell arc length"
        ),
        "complete_horizon_threshold": 0.95,
        "actual_clearance_sampling": (
            "minimum dynamic-map SDF sampled at executed controller endpoints"
        ),
        "phase_definition": {
            "before_event": "decision_step < event_step",
            "pre_opening": "event_step <= decision_step < opening_step",
            "post_opening": "decision_step >= opening_step",
        },
        "primitive_semantics": "activation witness, not trajectory command",
    }
    output = {
        "provenance": provenance,
        "validation": validation,
        "summaries": summaries,
    }
    _write_csv(args.out / "per_decision_agreement.csv", decisions)
    _write_csv(args.out / "summary_by_regime_phase.csv", summaries)
    (args.out / "summary.json").write_text(json.dumps(output, indent=2))
    (args.out / "RESULTS.md").write_text(
        _markdown(summaries, validation, provenance)
    )
    print(json.dumps(output, indent=2))
    print(f"Wrote Experiment 2 artifacts to {args.out}")


if __name__ == "__main__":
    main()
