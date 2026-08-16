#!/usr/bin/env python3
"""Close the gate-ablation and regime/phase reporting gaps.

This analysis intentionally consumes the frozen raw artifacts from Experiments
1--3.  It does not rerun the controller or alter the evaluation set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(values: Iterable[object]) -> np.ndarray:
    out = np.asarray([float(value) for value in values], dtype=float)
    return out[np.isfinite(out)]


def paired_bootstrap(
    rows: Sequence[dict[str, str]],
    *,
    metric: str,
    regime: str,
    reps: int,
    seed: int,
) -> dict[str, object]:
    by_episode: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["regime"] != regime:
            continue
        value = float(row[metric])
        if np.isfinite(value):
            by_episode[row["episode_id"]][row["arm"]] = value
    paired = sorted(
        episode for episode, arms in by_episode.items()
        if "gate_on" in arms and "gate_off" in arms
    )
    if not paired:
        raise ValueError(f"No pairs for {regime}/{metric}")
    on = np.asarray([by_episode[e]["gate_on"] for e in paired], dtype=float)
    off = np.asarray([by_episode[e]["gate_off"] for e in paired], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=float)
    for start in range(0, reps, 2_000):
        stop = min(start + 2_000, reps)
        idx = rng.integers(0, len(paired), size=(stop - start, len(paired)))
        boot[start:stop] = (on[idx] - off[idx]).mean(axis=1)
    return {
        "regime": regime,
        "metric": metric,
        "n_pairs": len(paired),
        "gate_on_mean": float(on.mean()),
        "gate_off_mean": float(off.mean()),
        "difference_on_minus_off": float((on - off).mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_unit": "paired_episode",
        "bootstrap_reps": reps,
    }


def selected_groups(rows: Sequence[dict[str, str]], prefixes: Sequence[str]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if any(row["group"] == prefix or row["group"].startswith(prefix) for prefix in prefixes)
    ]


def fmt(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp1",
        type=Path,
        default=ROOT / "rebuttal_experiments/results/exp1_gate_ablation_100",
    )
    parser.add_argument(
        "--exp2",
        type=Path,
        default=ROOT / "rebuttal_experiments/results/exp2_gate_trajectory_agreement",
    )
    parser.add_argument(
        "--exp3",
        type=Path,
        default=ROOT / "rebuttal_experiments/results/exp3_gate_activation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "rebuttal_experiments/results/gaps1_2_gate_analysis",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=27_370)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    exp1_path = args.exp1 / "per_episode_metrics.csv"
    exp2_path = args.exp2 / "summary_by_regime_phase.csv"
    exp3_path = args.exp3 / "summary_by_regime_phase.csv"
    exp1 = read_csv(exp1_path)
    exp2 = read_csv(exp2_path)
    exp3 = read_csv(exp3_path)

    # These are dynamic exposure diagnostics, not static directional CAR/FAR:
    # R1 activation is beneficial exposure and R2 activation is false exposure.
    metrics = (
        "effective_soft_rate",
        "success",
        "false_pre_activation_rate",
        "post_open_activation_rate",
        "cvar20_violation",
        "hard_contacts",
        "path_length_m",
    )
    paired_rows = [
        paired_bootstrap(
            exp1,
            metric=metric,
            regime=regime,
            reps=args.bootstrap_reps,
            seed=args.seed + metric_index + 100 * regime_index,
        )
        for regime_index, regime in enumerate(("R1", "R2", "R3"))
        for metric_index, metric in enumerate(metrics)
    ]
    write_csv(args.out / "gate_ablation_by_regime_paired_ci.csv", paired_rows)

    exp2_selected = selected_groups(
        exp2,
        ("overall", "regime=R1", "regime=R2", "regime=R3", "phase="),
    )
    exp3_selected = selected_groups(
        exp3,
        ("overall", "regime=R1", "regime=R2", "regime=R3", "phase="),
    )
    write_csv(args.out / "witness_execution_by_regime_phase.csv", exp2_selected)
    write_csv(args.out / "activation_by_regime_phase.csv", exp3_selected)

    by_key = {(row["regime"], row["metric"]): row for row in paired_rows}
    lines = [
        "# Gaps 1–2 — gate ablation and regime/phase analysis",
        "",
        "This report uses the frozen 100-episode, same-checkpoint gate ablation and "
        "the corresponding decision traces. No controller parameter, episode, map, "
        "event, or seed differs between gate-on and gate-off.",
        "",
        "## Gap 1: same-model gate-on/off by regime",
        "",
        "The first row below is a **dynamic soft-channel exposure rate**. It is the "
        "closest rollout analogue of CAR/FAR, but it is not relabeled as static "
        "directional CAR/FAR because the trace does not contain a direction-correctness "
        "label at every step.",
        "",
        "| Regime | Pairs | Gate-on exposure | Gate-off exposure | Paired Δ (95% CI) | "
        "Success on/off | Violation CVaR on/off |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for regime in ("R1", "R2", "R3"):
        exposure = by_key[(regime, "effective_soft_rate")]
        success = by_key[(regime, "success")]
        cvar = by_key[(regime, "cvar20_violation")]
        lines.append(
            f"| {regime} | {exposure['n_pairs']} | {fmt(exposure['gate_on_mean'])} | "
            f"{fmt(exposure['gate_off_mean'])} | "
            f"{fmt(exposure['difference_on_minus_off'])} "
            f"[{fmt(exposure['ci95_low'])}, {fmt(exposure['ci95_high'])}] | "
            f"{fmt(success['gate_on_mean'])}/{fmt(success['gate_off_mean'])} | "
            f"{fmt(cvar['gate_on_mean'])}/{fmt(cvar['gate_off_mean'])} |"
        )
    lines += [
        "",
        "The gate sharply reduces force exposure in every regime, including R2/R3 "
        "where exposure is undesirable. It does not create a resolved outcome benefit: "
        "success is identical, and the violation-CVaR changes are tiny relative to their "
        "paired intervals. This supports a mechanism/suppression claim, not an efficacy claim.",
        "",
        "## Gap 2: activation and gate–execution mismatch by regime",
        "",
        "| Regime | Step activation | Episodes with activation | Direction cosine | "
        "Clearance agreement | Hard disagreement | Risk-sign agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    activation = {row["group"]: row for row in exp3 if row["group"].startswith("regime=")}
    agreement = {row["group"]: row for row in exp2 if row["group"].startswith("regime=")}
    for regime in ("R1", "R2", "R3"):
        group = f"regime={regime}"
        a = activation[group]
        w = agreement[group]
        lines.append(
            f"| {regime} | {fmt(a['activation_rate'])} | "
            f"{a['episodes_with_any_activation']}/{a['n_episodes']} | "
            f"{fmt(w['mean_directional_cosine'])} | "
            f"{fmt(w['clearance_agreement_rate'])} | "
            f"{fmt(w['hard_contact_disagreement_rate'])} | "
            f"{fmt(w['improvement_sign_agreement_rate'])} |"
        )
    lines += [
        "",
        "The mismatch is not confined to one regime. R3 is worst on clearance agreement "
        "and hard-contact disagreement; R2 has the strongest predicted/realized risk "
        "correlation. The primitive remains evidence for activation, not a guarantee of "
        "the executed trajectory.",
        "",
        "## Phase breakdown",
        "",
        "| Phase | Activation | Direction cosine | Clearance agreement | "
        "Hard disagreement | Realized risk improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    phase_activation = {
        row["group"]: row for row in exp3
        if row["group"] in ("phase=pre_event", "phase=blocked_pre_opening", "phase=post_opening")
    }
    phase_agreement = {
        row["group"]: row for row in exp2
        if row["group"] in ("phase=before_event", "phase=pre_opening", "phase=post_opening")
    }
    phase_pairs = (
        ("pre-event", "phase=pre_event", "phase=before_event"),
        ("blocked/pre-opening", "phase=blocked_pre_opening", "phase=pre_opening"),
        ("post-opening", "phase=post_opening", "phase=post_opening"),
    )
    for label, activation_key, agreement_key in phase_pairs:
        a = phase_activation[activation_key]
        w = phase_agreement[agreement_key]
        lines.append(
            f"| {label} | {fmt(a['activation_rate'])} | "
            f"{fmt(w['mean_directional_cosine'])} | "
            f"{fmt(w['clearance_agreement_rate'])} | "
            f"{fmt(w['hard_contact_disagreement_rate'])} | "
            f"{fmt(w['mean_realized_risk_improvement'])} |"
        )
    lines += [
        "",
        "Pre-opening mismatch is strongly affected by horizons that cross the opening "
        "event (50/57 decisions). Post-opening is the cleanest phase, but endpoint and "
        "risk agreement remain incomplete.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python rebuttal_experiments/analyze_gate_gaps.py",
        "```",
    ]
    (args.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    provenance = {
        "experiment": "gaps1_2_gate_analysis",
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "input_hashes": {
            str(exp1_path): sha256(exp1_path),
            str(exp2_path): sha256(exp2_path),
            str(exp3_path): sha256(exp3_path),
        },
        "interpretation_guardrail": (
            "effective_soft_rate is a dynamic exposure diagnostic and is not "
            "reported as static directional CAR/FAR"
        ),
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote gate-gap analysis to {args.out}")


if __name__ == "__main__":
    main()
