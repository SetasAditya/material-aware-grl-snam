"""RQ5: matched semantic-corruption degradation and conservative failure mode.

The decision-fate panel tracks all 52,113 decision points that are matched
across every corruption level, rather than sampling a handful of them: it
compares what happens to activations that were *correct* when clean against
those that were *false* when clean. The two decay together, which is what makes
the failure mode conservative rather than selective.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import COLORS, DEFAULT_OUTPUT, DEFAULT_RESULTS, f, panel_label, rows, save_figure, setup_style


def _truthy(value: str) -> bool:
    return value in ("1", "True", "true", "1.0")


def matched_fate(raw: list[dict[str, str]]) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, int]]]:
    """Retention of clean-correct and clean-false activations under corruption.

    Decisions are matched on ``(episode_uid, path_index)`` so the same decision
    point is followed across every corruption level.
    """
    by_point: dict[tuple[str, str], dict[float, dict[str, str]]] = collections.defaultdict(dict)
    for row in raw:
        by_point[(row["episode_uid"], row["path_index"])][f(row, "corruption_probability")] = row

    levels = np.array(sorted({f(r, "corruption_probability") for r in raw}))
    clean = levels[0]
    populations = {
        "correct": [v for v in by_point.values()
                    if clean in v and _truthy(v[clean]["correct_activation"])],
        "false": [v for v in by_point.values()
                  if clean in v and _truthy(v[clean]["active"])
                  and not _truthy(v[clean]["correct_activation"])],
    }
    out = {}
    for name, population in populations.items():
        retained = np.array([
            sum(1 for v in population if level in v and _truthy(v[level]["active"])) / len(population)
            for level in levels
        ])
        out[name] = (retained, len(population))
    return levels * 100.0, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS / "exp7_semantic_corruption")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    setup_style()
    summary_path, raw_path = args.results / "summary_metrics.csv", args.results / "raw_predictions.csv"
    summary, raw = rows(summary_path), rows(raw_path)
    p = np.array([100 * f(r, "corruption_probability") for r in summary])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)

    # A: the rates themselves.
    ax = axes[0]
    panel_label(ax, "A")
    for key, label, color in [("CAR", "Correct activation (CAR)", COLORS["material"]),
                              ("FAR", "False activation (FAR)", COLORS["risk"]),
                              ("activation_rate", "Overall activation", COLORS["safe"])]:
        ax.plot(p, [f(r, key) for r in summary], "o-", lw=2, label=label, color=color)
    ax.set(xlabel="Corrupted semantic labels (%)", ylabel="Rate", ylim=(0, 0.8),
           title="Both correct and false activation fall\nas semantics degrade")
    ax.legend(fontsize=8, loc="upper right")

    # B: does discrimination survive? CAR/FAR is the selectivity ratio.
    ax = axes[1]
    panel_label(ax, "B")
    ratio = np.array([f(r, "SR") for r in summary])
    ax.plot(p, ratio, "o-", lw=2, color=COLORS["material"])
    ax.axhline(1.0, color=COLORS["risk"], ls="--", lw=1.2)
    ax.text(p[-1], 1.03, "no discrimination", ha="right", fontsize=7.5, color=COLORS["risk"])
    for x, y in zip(p, ratio):
        ax.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=8)
    ax.set(xlabel="Corrupted semantic labels (%)", ylabel="Selectivity ratio (CAR / FAR)",
           ylim=(0.9, 2.1),
           title="Discrimination degrades but\ndoes not collapse")

    # C: matched decision fate -- the failure mode itself.
    ax = axes[2]
    panel_label(ax, "C")
    levels, fate = matched_fate(raw)
    correct, n_correct = fate["correct"]
    false, n_false = fate["false"]
    ax.plot(levels, correct, "o-", lw=2, color=COLORS["material"],
            label=f"was correct when clean (n={n_correct:,})")
    ax.plot(levels, false, "s--", lw=2, color=COLORS["risk"],
            label=f"was false when clean (n={n_false:,})")
    ax.fill_between(levels, correct, false, color=COLORS["muted"], alpha=0.55, zorder=0)
    gap = abs(correct[-1] - false[-1])
    ax.annotate(f"gap at 30% corruption: {gap:.3f}",
                xy=(levels[-1], (correct[-1] + false[-1]) / 2),
                xytext=(levels[-1] - 2, 0.72), ha="right", fontsize=8,
                color=COLORS["hazard"],
                arrowprops={"arrowstyle": "->", "color": COLORS["geometry"], "lw": 0.9})
    ax.set(xlabel="Corrupted semantic labels (%)",
           ylabel="Fraction still activating", ylim=(0, 1.05),
           title="Suppression is indiscriminate:\nmatched decisions decay together")
    ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("RQ5 — Corruption makes the controller conservative, not selectively safer", weight="bold")
    save_figure(fig, args.output, "rq5_perception_robustness", [summary_path, raw_path])


if __name__ == "__main__":
    main()
