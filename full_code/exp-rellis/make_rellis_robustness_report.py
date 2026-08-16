#!/usr/bin/env python3
"""Build the main RELLIS selectivity table and robustness figure."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


RUN_ROOT = Path("exp-rellis/runs")
OUT_DIR = RUN_ROOT / "rellis_robustness_report"


@dataclass
class Row:
    evaluation: str
    method: str
    car: float
    far: float
    selectivity_ratio: float | None = None
    alignment: float | None = None
    accuracy: float | None = None
    n: float | None = None
    car_std: float | None = None
    far_std: float | None = None
    note: str = ""


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def fmt(value: float | None, std: float | None = None) -> str:
    if value is None:
        return "--"
    if std is None:
        return f"{value:.3f}"
    return f"{value:.3f} +/- {std:.3f}"


def source_row(evaluation: str, method: str, metrics: dict, note: str = "") -> Row:
    return Row(
        evaluation=evaluation,
        method=method,
        car=metrics["correct_activation_rate"],
        far=metrics["false_activation_rate"],
        selectivity_ratio=metrics.get("selectivity_ratio"),
        alignment=metrics.get("force_risk_alignment"),
        accuracy=metrics.get("accuracy"),
        n=metrics.get("num_force_samples", metrics.get("n")),
        note=note,
    )


def aggregate_rows(evaluation: str, method: str, summaries: Iterable[dict], note: str = "") -> Row:
    metrics = [s["val_metrics"] for s in summaries]
    cars = [m["correct_activation_rate"] for m in metrics]
    fars = [m["false_activation_rate"] for m in metrics]
    ratios = [m["selectivity_ratio"] for m in metrics]
    accs = [m["accuracy"] for m in metrics]
    ns = [m["n"] for m in metrics]
    return Row(
        evaluation=evaluation,
        method=method,
        car=mean(cars),
        far=mean(fars),
        selectivity_ratio=mean(ratios),
        accuracy=mean(accs),
        n=mean(ns),
        car_std=pstdev(cars),
        far_std=pstdev(fars),
        note=note,
    )


def collect_rows() -> list[Row]:
    rows: list[Row] = []

    argmax = load_json(RUN_ROOT / "rellis_selectivity_val1500_with_directional_head" / "summary.json")
    expected = load_json(RUN_ROOT / "rellis_selectivity_val1500_with_directional_expected" / "summary.json")

    rows.append(
        source_row(
            "All episodes",
            "Stage 2 scalar lambda",
            argmax["selectivity_by_source"]["s2_model_lambda"],
            "Original scalar force gate.",
        )
    )
    rows.append(
        source_row(
            "All episodes",
            "Directional head (argmax)",
            argmax["selectivity_by_source"]["stage2_directional_head"],
            "Candidate-direction force with argmax activation.",
        )
    )
    rows.append(
        source_row(
            "All episodes",
            "Directional head (expected)",
            expected["selectivity_by_source"]["stage2_directional_head"],
            "Expected direction over candidate probabilities.",
        )
    )

    seed_summaries = [
        load_json(RUN_ROOT / f"rellis_directional_seed{i}" / "summary.json")
        for i in range(3)
    ]
    rows.append(
        aggregate_rows(
            "Episode held-out seeds",
            "Directional head",
            seed_summaries,
            "Mean/std over seeds 0, 1, 2 on episode-level validation splits.",
        )
    )
    for i, summary in enumerate(seed_summaries):
        rows.append(
            source_row(
                f"Seed {i}",
                "Directional head",
                summary["val_metrics"],
                "Single seeded episode split.",
            )
        )

    holdout_00001 = load_json(RUN_ROOT / "rellis_directional_holdout_00001" / "summary.json")
    holdout_00000 = load_json(RUN_ROOT / "rellis_directional_holdout_00000" / "summary.json")
    sequence_summaries = [holdout_00001, holdout_00000]
    rows.append(
        aggregate_rows(
            "Sequence held-out",
            "Directional head",
            sequence_summaries,
            "Mean/std over train 00000 -> test 00001 and train 00001 -> test 00000.",
        )
    )
    rows.append(
        source_row(
            "Train 00000, test 00001",
            "Directional head",
            holdout_00001["val_metrics"],
            "Asymmetric failure: model mostly suppresses activation on held-out 00001.",
        )
    )
    rows.append(
        source_row(
            "Train 00001, test 00000",
            "Directional head",
            holdout_00000["val_metrics"],
            "More usable transfer direction, but FAR remains higher than seeded splits.",
        )
    )
    return rows


def write_csv(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "evaluation",
                "method",
                "CAR",
                "CAR_std",
                "FAR",
                "FAR_std",
                "selectivity_ratio",
                "force_risk_alignment",
                "accuracy",
                "n",
                "note",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.evaluation,
                    row.method,
                    f"{row.car:.6f}",
                    "" if row.car_std is None else f"{row.car_std:.6f}",
                    f"{row.far:.6f}",
                    "" if row.far_std is None else f"{row.far_std:.6f}",
                    "" if row.selectivity_ratio is None else f"{row.selectivity_ratio:.6f}",
                    "" if row.alignment is None else f"{row.alignment:.6f}",
                    "" if row.accuracy is None else f"{row.accuracy:.6f}",
                    "" if row.n is None else f"{row.n:.1f}",
                    row.note,
                ]
            )


def write_latex(rows: list[Row], path: Path) -> None:
    main_rows = [
        r
        for r in rows
        if r.evaluation
        in {
            "All episodes",
            "Episode held-out seeds",
            "Sequence held-out",
            "Train 00000, test 00001",
            "Train 00001, test 00000",
        }
    ]
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Evaluation & Method & CAR $\uparrow$ & FAR $\downarrow$ & Sel. ratio $\uparrow$ & Align. $\uparrow$ \\",
        r"\midrule",
    ]
    for row in main_rows:
        lines.append(
            f"{row.evaluation} & {row.method} & "
            f"{fmt(row.car, row.car_std)} & {fmt(row.far, row.far_std)} & "
            f"{fmt(row.selectivity_ratio)} & {fmt(row.alignment)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))


def write_markdown(rows: list[Row], path: Path) -> None:
    lines = [
        "# RELLIS Selectivity Robustness Report",
        "",
        "| Evaluation | Method | CAR up | FAR down | Selectivity ratio | Alignment | Note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.evaluation} | {row.method} | {fmt(row.car, row.car_std)} | "
            f"{fmt(row.far, row.far_std)} | {fmt(row.selectivity_ratio)} | "
            f"{fmt(row.alignment)} | {row.note} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The directional head is much more selective than the scalar lambda force on the in-distribution all-episode benchmark.",
            "- Seeded episode splits are reasonably stable: CAR remains high and FAR remains low-to-moderate.",
            "- Sequence-held-out transfer is not paper-ready yet because train 00000 -> test 00001 collapses to near-zero CAR.",
            "- The next fix should target sequence/domain imbalance, activation-label sparsity, and calibration rather than another figure pass.",
        ]
    )
    path.write_text("\n".join(lines))


def plot(rows: list[Row], path: Path) -> None:
    wanted = [
        ("Stage 2\nscalar", "All episodes", "Stage 2 scalar lambda"),
        ("Dir.\nargmax", "All episodes", "Directional head (argmax)"),
        ("Dir.\nexpected", "All episodes", "Directional head (expected)"),
        ("Seed\nmean", "Episode held-out seeds", "Directional head"),
        ("Seq.\nmean", "Sequence held-out", "Directional head"),
        ("00000->\n00001", "Train 00000, test 00001", "Directional head"),
        ("00001->\n00000", "Train 00001, test 00000", "Directional head"),
    ]
    lookup = {(r.evaluation, r.method): r for r in rows}
    selected = [lookup[(evaluation, method)] for _, evaluation, method in wanted]
    labels = [label for label, _, _ in wanted]
    car = np.array([r.car for r in selected])
    far = np.array([r.far for r in selected])
    car_err = np.array([0.0 if r.car_std is None else r.car_std for r in selected])
    far_err = np.array([0.0 if r.far_std is None else r.far_std for r in selected])

    x = np.arange(len(selected))
    width = 0.38

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width / 2, car, width, yerr=car_err, label="CAR up", color="#2f6f73", capsize=3)
    ax.bar(x + width / 2, far, width, yerr=far_err, label="FAR down", color="#c7634d", capsize=3)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Rate")
    ax.set_title("RELLIS material-risk selectivity: in-distribution vs robustness checks")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.axhline(0.5, color="#222222", linewidth=0.8, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    write_csv(rows, OUT_DIR / "rellis_selectivity_main_table.csv")
    write_latex(rows, OUT_DIR / "rellis_selectivity_main_table.tex")
    write_markdown(rows, OUT_DIR / "README.md")
    plot(rows, OUT_DIR / "rellis_selectivity_robustness.png")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
