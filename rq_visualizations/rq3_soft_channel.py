"""RQ3: paired one-factor soft-channel intervention."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import COLORS, DEFAULT_OUTPUT, DEFAULT_RESULTS, f, grouped, panel_label, rows, save_figure, setup_style


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS / "exp9_soft_coefficient_isolation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); setup_style()
    metrics_path, coeff_path = args.results / "per_episode_metrics.csv", args.results / "rollout_coefficients.csv"
    metrics, coeffs = rows(metrics_path), rows(coeff_path)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), constrained_layout=True)
    datasets = ["DFC2018", "RELLIS-3D"]
    for col, dataset in enumerate(datasets):
        ax = axes[0, col]; panel_label(ax, "A" if col == 0 else "B")
        by_episode = grouped([r for r in metrics if r["dataset"] == dataset], "episode_id")
        deltas = []
        for values in by_episode.values():
            arm = {r["arm"]: f(r, "risk_exposure") for r in values}
            if "zero" in arm and "learned" in arm: deltas.append(arm["learned"] - arm["zero"])
        ax.axvline(0, color=COLORS["geometry"], lw=1)
        ax.hist(deltas, bins=18, color=COLORS["material"], alpha=.8)
        ax.axvline(np.mean(deltas), color=COLORS["risk"], lw=1.8, label=f"mean {np.mean(deltas):+.4f}")
        outcome = "MEASURABLE REDUCTION" if dataset == "DFC2018" else "NO DETECTABLE EFFECT"
        ax.set(title=f"{dataset}: {outcome}", xlabel="Learned minus soft-off cumulative risk", ylabel="Episodes")
        ax.legend()

        ax = axes[1, col]; panel_label(ax, "C" if col == 0 else "D")
        values = {}
        for arm in ["zero", "learned", "fixed"]:
            values[arm] = np.array([f(r, "risk_exposure") for r in metrics if r["dataset"] == dataset and r["arm"] == arm])
        means = [values[a].mean() for a in values]
        colors = [COLORS["zero"], COLORS["material"], COLORS["fixed"]]
        deltas_from_zero = np.asarray(means) - means[0]
        bars = ax.bar(range(3), deltas_from_zero, color=colors, alpha=.9)
        baseline = means[0]
        span = max(0.004, np.max(np.abs(deltas_from_zero)) * 1.45)
        ax.set_ylim(-span, span)
        ax.axhline(0, color=COLORS["geometry"], lw=1)
        ax.set(xticks=range(3), xticklabels=["$\\lambda_s=0$", "learned", "fixed 1.5"],
               ylabel="Change from soft-off", title="One-factor intervention")
        for bar, delta in zip(bars, deltas_from_zero):
            va = "bottom" if delta >= 0 else "top"
            ax.text(bar.get_x()+bar.get_width()/2, delta, f"{delta:+.3f}",
                    ha="center", va=va, fontsize=10, weight="bold")
        ax.text(.03, .04, f"SUCCESS = 1.00 in every arm\nSoft-off mean = {baseline:.3f}",
                transform=ax.transAxes, fontsize=9, weight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#E8F5E9",
                      "edgecolor": COLORS["safe"]})
    fig.suptitle("RQ3 — The isolated soft channel changes risk only where the field is informative",
                 fontsize=14, weight="bold")
    save_figure(fig, args.output, "rq3_isolated_soft_channel", [metrics_path, coeff_path])


if __name__ == "__main__":
    main()
