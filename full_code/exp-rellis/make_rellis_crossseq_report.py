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
OUT_DIR = RUN_ROOT / "rellis_crossseq_report"
SEQS = ["00000", "00001", "00002", "00003", "00004"]


@dataclass
class SummaryRow:
    method: str
    car: float
    car_std: float
    far: float
    far_std: float
    ratio: float
    ratio_std: float
    acc: float
    acc_std: float
    note: str


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())["val_metrics"]


def summarize(method: str, paths: list[Path], note: str) -> tuple[SummaryRow, list[tuple[str, dict]]]:
    fold_metrics = [(seq, load_metrics(path)) for seq, path in zip(SEQS, paths)]
    cars = [m["correct_activation_rate"] for _, m in fold_metrics]
    fars = [m["false_activation_rate"] for _, m in fold_metrics]
    ratios = [m["selectivity_ratio"] for _, m in fold_metrics]
    accs = [m["accuracy"] for _, m in fold_metrics]
    row = SummaryRow(
        method=method,
        car=mean(cars),
        car_std=pstdev(cars),
        far=mean(fars),
        far_std=pstdev(fars),
        ratio=mean(ratios),
        ratio_std=pstdev(ratios),
        acc=mean(accs),
        acc_std=pstdev(accs),
        note=note,
    )
    return row, fold_metrics


def fmt(mu: float, sigma: float) -> str:
    return f"{mu:.3f} +/- {sigma:.3f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = [
        (
            "Raw directional head",
            [RUN_ROOT / f"rellis_directional_loso_{seq}" / "summary.json" for seq in SEQS],
            "High activation recall, but false activation remains too high for the selectivity claim.",
        ),
        (
            "Train-calibrated FAR<=0.20",
            [RUN_ROOT / f"rellis_directional_loso_cal_far020_{seq}" / "summary.json" for seq in SEQS],
            "Confidence gate calibrated on train split; FAR controlled, CAR too conservative.",
        ),
        (
            "Conservative loss",
            [
                RUN_ROOT / "rellis_directional_loso_w03_na3_00000" / "summary.json",
                RUN_ROOT / "rellis_directional_loso_w03_na3_00001" / "summary.json",
                RUN_ROOT / "rellis_directional_loso_w03_na3_00002" / "summary.json",
                RUN_ROOT / "rellis_directional_loso_probe_w03_na3_00003" / "summary.json",
                RUN_ROOT / "rellis_directional_loso_w03_na3_00004" / "summary.json",
            ],
            "Best current tradeoff: lower FAR and higher selectivity ratio, but CAR is only moderate.",
        ),
    ]
    rows: list[SummaryRow] = []
    folds: dict[str, list[tuple[str, dict]]] = {}
    for method, paths, note in configs:
        row, fold_metrics = summarize(method, paths, note)
        rows.append(row)
        folds[method] = fold_metrics

    with (OUT_DIR / "crossseq_main_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "CAR", "CAR_std", "FAR", "FAR_std", "selectivity_ratio", "ratio_std", "accuracy", "accuracy_std", "note"])
        for row in rows:
            writer.writerow([row.method, row.car, row.car_std, row.far, row.far_std, row.ratio, row.ratio_std, row.acc, row.acc_std, row.note])

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & CAR $\uparrow$ & FAR $\downarrow$ & Sel. ratio $\uparrow$ & Accuracy $\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(f"{row.method} & {fmt(row.car, row.car_std)} & {fmt(row.far, row.far_std)} & {fmt(row.ratio, row.ratio_std)} & {fmt(row.acc, row.acc_std)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (OUT_DIR / "crossseq_main_table.tex").write_text("\n".join(lines))

    md = [
        "# RELLIS Cross-Sequence Selectivity Report",
        "",
        "| Method | CAR up | FAR down | Selectivity ratio | Accuracy | Note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md.append(f"| {row.method} | {fmt(row.car, row.car_std)} | {fmt(row.far, row.far_std)} | {fmt(row.ratio, row.ratio_std)} | {fmt(row.acc, row.acc_std)} | {row.note} |")
    md.extend(
        [
            "",
            "Fold details:",
            "",
            "| Method | Holdout seq | CAR | FAR | Selectivity ratio | Accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, fold_metrics in folds.items():
        for seq, metrics in fold_metrics:
            md.append(
                f"| {method} | {seq} | {metrics['correct_activation_rate']:.3f} | "
                f"{metrics['false_activation_rate']:.3f} | {metrics['selectivity_ratio']:.3f} | {metrics['accuracy']:.3f} |"
            )
    md.extend(
        [
            "",
            "Current conclusion: not paper-ready for a strong cross-sequence generalization claim yet. The five-sequence benchmark is healthy, but the model still trades off CAR and FAR too sharply.",
            "Most likely missing piece: route-aware feasibility/cost-to-go gating that is efficient enough to run at scale.",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(md))

    labels = [row.method.replace(" ", "\n") for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - width / 2, [r.car for r in rows], width, yerr=[r.car_std for r in rows], label="CAR up", color="#2f6f73", capsize=3)
    ax.bar(x + width / 2, [r.far for r in rows], width, yerr=[r.far_std for r in rows], label="FAR down", color="#c7634d", capsize=3)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("RELLIS five-sequence leave-one-sequence-out selectivity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "crossseq_car_far.png", dpi=220)
    plt.close(fig)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
