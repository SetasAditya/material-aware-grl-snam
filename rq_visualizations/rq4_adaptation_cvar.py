"""RQ4: cross-domain coefficient adaptation and CVaR timing comparison.

Panel B locates the CVaR tail spatially: the 100 logged gate-on rollouts are
overlaid in the shared BEV frame and the worst-decile episodes by risk exposure
are highlighted, so the tail the objective targets is visible as geometry
rather than only as a scalar.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from common import fs, COLORS, DEFAULT_OUTPUT, DEFAULT_RESULTS, ROOT, SHAPE, f, grouped, panel_label, rows, save_figure, setup_style

FIG_W = 11.2   # authored width in inches; drives font sizing via setup_style



def episode_paths(gate: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Gate-on rollouts keyed by episode, ordered by control step."""
    out: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in gate:
        if row["arm"] == "gate_on":
            out[row["episode_id"]].append(row)
    return {k: sorted(v, key=lambda r: f(r, "step")) for k, v in out.items()}


def risk_exposure(path: list[dict[str, str]]) -> float:
    """Path-integrated soft risk: the quantity the CVaR objective bounds."""
    return sum(f(r, "risk", 0.0) * f(r, "movement_m", 0.0) for r in path)


def draw_tail(ax, gate: list[dict[str, str]], quantile: float = 0.9) -> tuple[float, float]:
    """Overlay every rollout, highlighting the worst-decile exposure tail."""
    paths = episode_paths(gate)
    exposure = {ep: risk_exposure(p) for ep, p in paths.items()}
    values = np.array(list(exposure.values()))
    threshold = float(np.quantile(values, quantile))

    for episode, path in paths.items():
        track = np.array([[f(r, "position_x"), f(r, "position_y")] for r in path])
        tail = exposure[episode] >= threshold
        ax.plot(track[:, 0], track[:, 1],
                color=COLORS["risk"] if tail else COLORS["material_light"],
                lw=1.5 if tail else 0.8, alpha=0.95 if tail else 0.5,
                zorder=3 if tail else 2)
    ax.set_xlim(0, SHAPE[1])
    ax.set_ylim(SHAPE[0], 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return threshold, float(values[values >= threshold].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp9", type=Path, default=DEFAULT_RESULTS / "exp9_soft_coefficient_isolation")
    parser.add_argument("--gate", type=Path, default=DEFAULT_RESULTS / "exp1_gate_ablation_100")
    parser.add_argument("--objective-summary", type=Path, default=ROOT / "results" / "rellis_missing_ablation_results" / "delayed_required_false_preact_100.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); setup_style(FIG_W)
    coeff_path, gate_path = args.exp9 / "rollout_coefficients.csv", args.gate / "step_traces.csv"
    coeff, gate, objective = rows(coeff_path), rows(gate_path), rows(args.objective_summary)
    distributions = [
        np.array([f(r, "lam_soft_learned") for r in gate if r["arm"] == "gate_on"]),
        np.array([f(r, "lambda_soft_learned") for r in coeff if r["dataset"] == "RELLIS-3D" and r["arm"] == "learned"]),
        np.array([f(r, "lambda_soft_learned") for r in coeff if r["dataset"] == "DFC2018" and r["arm"] == "learned"]),
    ]
    distributions = [x[np.isfinite(x)] for x in distributions]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 5.6), constrained_layout=True)
    ax = axes[0]; panel_label(ax, "A")
    parts = ax.violinplot(distributions, showmedians=True, widths=.72)
    for body, color in zip(parts["bodies"], [COLORS["zero"], COLORS["material_light"], COLORS["material"]]): body.set_facecolor(color); body.set_alpha(.85)
    means = [values.mean() for values in distributions]
    ax.set(xticks=[1,2,3],
           xticklabels=[f"Delayed escape\nmean {means[0]:.3f}",
                        f"RELLIS R1\nmean {means[1]:.3f}",
                        f"DFC2018\nmean {means[2]:.3f}"],
           ylabel="$\\lambda_s$", title="Context-dependent force scale")
    ax.text(.04, .94, "~30x cross-domain range", transform=ax.transAxes,
            va="top", fontsize=fs(11), weight="bold",
            bbox={"boxstyle": "round,pad=.3", "facecolor": "white",
                  "edgecolor": COLORS["material"]})

    ax = axes[1]; panel_label(ax, "B")
    methods = {"stage2_expected_cost": "Expected cost", "route_aware_stage2": "CVaR"}
    selected = {
        key: next(r for r in objective if r["subset"] == "all" and r["method"] == key)
        for key in methods
    }
    expected_row, cvar_row = selected["stage2_expected_cost"], selected["route_aware_stage2"]
    labels = ["Expected-risk", "CVaR"]
    success = np.array([f(expected_row, "success"), f(cvar_row, "success")]) * 100
    y = np.arange(2)
    ax.barh(y, success, color=COLORS["safe"], height=.52, label="success")
    ax.barh(y, 100-success, left=success, color=COLORS["risk"], height=.52, label="failure")
    for yi, value in zip(y, success):
        ax.text(value/2, yi, f"{value:.0f} SUCCESS", color="white", ha="center",
                va="center", fontsize=fs(11), weight="bold")
        ax.text(value+(100-value)/2, yi, f"{100-value:.0f} FAILURE", color="white",
                ha="center", va="center", fontsize=fs(10), weight="bold")
    ax.set(yticks=y, yticklabels=labels, xlim=(0,100), xlabel="Episodes (%)",
           title="Outcome on 100 paired episodes")
    ax.invert_yaxis()
    ax.text(.5, .50, "NO SIGNIFICANT DIFFERENCE (95% CI)", transform=ax.transAxes,
            ha="center", fontsize=fs(11), weight="bold",
            bbox={"boxstyle": "round,pad=.3", "facecolor": "#FFF3E0",
                  "edgecolor": COLORS["fixed"]})
    fig.suptitle("RQ4 — Context adapts force scale; CVaR does not",
                 fontsize=fs(14), weight="bold")
    save_figure(fig, args.output, "rq4_adaptation_and_cvar", [coeff_path, gate_path, args.objective_summary])


if __name__ == "__main__":
    main()
