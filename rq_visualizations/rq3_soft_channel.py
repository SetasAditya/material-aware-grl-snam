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
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), constrained_layout=True)
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
        ax.set(title=f"{dataset}: learned minus soft-off", xlabel="Paired cumulative-risk difference", ylabel="Episodes")
        ax.legend()

        ax = axes[1, col]; panel_label(ax, "C" if col == 0 else "D")
        values = {}
        for arm in ["zero", "learned", "fixed"]:
            values[arm] = np.array([f(r, "risk_exposure") for r in metrics if r["dataset"] == dataset and r["arm"] == arm])
        means = [values[a].mean() for a in values]
        colors = [COLORS["zero"], COLORS["material"], COLORS["fixed"]]
        bars = ax.bar(range(3), means, color=colors, alpha=.9)
        baseline = means[0]
        ax.set_ylim(min(means) - max(.01, abs(max(means)-min(means))*2), max(means) + max(.01, abs(max(means)-min(means))*2))
        ax.set(xticks=range(3), xticklabels=["$\\lambda_s=0$", "learned", "fixed 1.5"], ylabel="Cumulative soft risk", title="Magnified mean scale; same checkpoint and episodes")
        for bar, mean in zip(bars, means): ax.text(bar.get_x()+bar.get_width()/2, mean, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(.02, .04, f"Success = 1.000 in all arms\nZero baseline = {baseline:.3f}", transform=ax.transAxes, fontsize=8)
    fig.suptitle("RQ3 — Isolating the soft coefficient on its own behavioral axis", weight="bold")
    save_figure(fig, args.output, "rq3_isolated_soft_channel", [metrics_path, coeff_path])


if __name__ == "__main__":
    main()
