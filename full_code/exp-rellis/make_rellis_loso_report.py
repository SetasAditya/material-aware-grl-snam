#!/usr/bin/env python3
"""Summarize five-sequence RELLIS leave-one-sequence-out selectivity runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np


RUN_ROOT = Path("exp-rellis/runs")
OUT_DIR = RUN_ROOT / "rellis_loso_report"
SEQUENCES = ["00000", "00001", "00002", "00003", "00004"]


@dataclass
class Row:
    method: str
    fold: str
    car: float
    far: float
    ratio: float
    accuracy: float
    threshold: float | None
    n: float


def load_row(method: str, fold: str, run_name: str) -> Row:
    summary = json.loads((RUN_ROOT / run_name / "summary.json").read_text())
    metrics = summary["val_metrics"]
    return Row(
        method=method,
        fold=fold,
        car=float(metrics["correct_activation_rate"]),
        far=float(metrics["false_activation_rate"]),
        ratio=float(metrics["selectivity_ratio"]),
        accuracy=float(metrics["accuracy"]),
        threshold=summary.get("activation_threshold"),
        n=float(metrics["n"]),
    )


def aggregate(rows: list[Row], method: str) -> dict[str, float | str]:
    subset = [r for r in rows if r.method == method]
    return {
        "method": method,
        "folds": len(subset),
        "CAR_mean": mean(r.car for r in subset),
        "CAR_std": pstdev(r.car for r in subset),
        "FAR_mean": mean(r.far for r in subset),
        "FAR_std": pstdev(r.far for r in subset),
        "selectivity_ratio_mean": mean(r.ratio for r in subset),
        "selectivity_ratio_std": pstdev(r.ratio for r in subset),
        "accuracy_mean": mean(r.accuracy for r in subset),
        "accuracy_std": pstdev(r.accuracy for r in subset),
    }


def collect() -> tuple[list[Row], list[dict[str, float | str]]]:
    specs = [
        ("Raw directional, AW=3", "rellis_directional_loso_{seq}"),
        ("Calibrated FAR20, AW=3", "rellis_directional_loso_cal_far020_{seq}"),
        ("Long-horizon FAR20, AW=3", "rellis_directional_loso_longh24_cal_far020_{seq}"),
        ("Calibrated FAR20, AW=0.5", "rellis_directional_loso_aw050_cal_far020_{seq}"),
        ("Calibrated FAR30, AW=0.5", "rellis_directional_loso_aw050_cal_far030_{seq}"),
        ("Route-aware FAR20, AW=0.5", "rellis_directional_routeaware_aw050_far020_{seq}"),
    ]
    rows: list[Row] = []
    for method, pattern in specs:
        for seq in SEQUENCES:
            path = RUN_ROOT / pattern.format(seq=seq) / "summary.json"
            if path.exists():
                rows.append(load_row(method, seq, pattern.format(seq=seq)))
    methods = [method for method, _ in specs if any(r.method == method for r in rows)]
    return rows, [aggregate(rows, method) for method in methods]


def write_csv(rows: list[Row], aggregates: list[dict[str, float | str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "loso_folds.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "heldout_sequence", "CAR", "FAR", "selectivity_ratio", "accuracy", "threshold", "n"])
        for r in rows:
            writer.writerow([r.method, r.fold, r.car, r.far, r.ratio, r.accuracy, r.threshold, r.n])
    with (OUT_DIR / "loso_summary.csv").open("w", newline="") as f:
        fields = list(aggregates[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregates)


def write_latex(aggregates: list[dict[str, float | str]]) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & CAR $\uparrow$ & FAR $\downarrow$ & Sel. ratio $\uparrow$ & Acc. $\uparrow$ \\",
        r"\midrule",
    ]
    for row in aggregates:
        lines.append(
            f"{row['method']} & "
            f"{row['CAR_mean']:.3f} $\\pm$ {row['CAR_std']:.3f} & "
            f"{row['FAR_mean']:.3f} $\\pm$ {row['FAR_std']:.3f} & "
            f"{row['selectivity_ratio_mean']:.3f} $\\pm$ {row['selectivity_ratio_std']:.3f} & "
            f"{row['accuracy_mean']:.3f} $\\pm$ {row['accuracy_std']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (OUT_DIR / "loso_summary.tex").write_text("\n".join(lines))


def write_readme(aggregates: list[dict[str, float | str]]) -> None:
    lines = [
        "# RELLIS Five-Sequence LOSO Selectivity",
        "",
        "| Method | CAR up | FAR down | Selectivity ratio | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['method']} | {row['CAR_mean']:.3f} +/- {row['CAR_std']:.3f} | "
            f"{row['FAR_mean']:.3f} +/- {row['FAR_std']:.3f} | "
            f"{row['selectivity_ratio_mean']:.3f} +/- {row['selectivity_ratio_std']:.3f} | "
            f"{row['accuracy_mean']:.3f} +/- {row['accuracy_std']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The five-sequence benchmark removes the earlier two-sequence artifact and gives 2,250 balanced episodes.",
            "- Raw AW=3 keeps high CAR but false activation is too high for a selectivity claim.",
            "- AW=0.5 with train-calibrated thresholds gives the best honest tradeoff so far.",
            "- Route-aware candidate-to-go features recover high CAR while keeping FAR near the train-calibrated target.",
            "- This supports the diagnosis that R2 is primarily a route feasibility problem, not just a local risk-gradient problem.",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines))


def plot(aggregates: list[dict[str, float | str]]) -> None:
    labels = [str(r["method"]).replace(", ", "\n") for r in aggregates]
    car = np.asarray([float(r["CAR_mean"]) for r in aggregates])
    far = np.asarray([float(r["FAR_mean"]) for r in aggregates])
    car_std = np.asarray([float(r["CAR_std"]) for r in aggregates])
    far_std = np.asarray([float(r["FAR_std"]) for r in aggregates])
    x = np.arange(len(labels))
    width = 0.38
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width / 2, car, width, yerr=car_std, label="CAR up", color="#2f6f73", capsize=3)
    ax.bar(x + width / 2, far, width, yerr=far_std, label="FAR down", color="#c7634d", capsize=3)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Held-out sequence rate")
    ax.set_title("RELLIS five-sequence leave-one-sequence-out selectivity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loso_selectivity_tradeoff.png", dpi=220)
    plt.close(fig)


def main() -> None:
    rows, aggregates = collect()
    write_csv(rows, aggregates)
    write_latex(aggregates)
    write_readme(aggregates)
    plot(aggregates)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
