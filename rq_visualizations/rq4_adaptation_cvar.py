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

from common import COLORS, DEFAULT_OUTPUT, DEFAULT_RESULTS, ROOT, SHAPE, f, grouped, panel_label, rows, save_figure, setup_style


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
    args = parser.parse_args(); setup_style()
    coeff_path, gate_path = args.exp9 / "rollout_coefficients.csv", args.gate / "step_traces.csv"
    coeff, gate, objective = rows(coeff_path), rows(gate_path), rows(args.objective_summary)
    distributions = [
        np.array([f(r, "lam_soft_learned") for r in gate if r["arm"] == "gate_on"]),
        np.array([f(r, "lambda_soft_learned") for r in coeff if r["dataset"] == "RELLIS-3D" and r["arm"] == "learned"]),
        np.array([f(r, "lambda_soft_learned") for r in coeff if r["dataset"] == "DFC2018" and r["arm"] == "learned"]),
    ]
    distributions = [x[np.isfinite(x)] for x in distributions]

    fig, axes = plt.subplots(1, 4, figsize=(14.6, 4.0), constrained_layout=True)
    ax = axes[0]; panel_label(ax, "A")
    parts = ax.violinplot(distributions, showmedians=True, widths=.72)
    for body, color in zip(parts["bodies"], [COLORS["zero"], COLORS["material_light"], COLORS["material"]]): body.set_facecolor(color); body.set_alpha(.85)
    ax.set(xticks=[1,2,3], xticklabels=["Delayed\nescape", "RELLIS R1", "DFC2018"], ylabel="$\\lambda_s$", title="Recovered coefficient distributions")
    for i, values in enumerate(distributions, 1): ax.text(i, np.percentile(values, 96), f"mean {values.mean():.3f}", ha="center", fontsize=8)

    ax = axes[1]; panel_label(ax, "B")
    threshold, tail_mean = draw_tail(ax, gate)
    all_exposure = np.array([risk_exposure(p) for p in episode_paths(gate).values()])
    ax.legend(handles=[
        Line2D([], [], color=COLORS["risk"], lw=1.5, label=f"worst decile (CVaR$_{{90}}$ = {tail_mean:.1f})"),
        Line2D([], [], color=COLORS["material_light"], lw=0.8, label=f"remaining (mean = {all_exposure.mean():.1f})"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7)
    ax.set_title("Where the CVaR tail lives\n100 rollouts, 13 scenes, shared BEV frame", fontsize=9)

    ax = axes[2]; panel_label(ax, "C")
    methods = {"stage2_expected_cost": "Expected cost", "route_aware_stage2": "CVaR"}
    selected = {
        key: next(r for r in objective if r["subset"] == "all" and r["method"] == key)
        for key in methods
    }
    metrics = ["success", "stuck", "violation_cvar"]
    labels = ["Success ↑", "Stuck ↓", "Violation CVaR ↓"]
    x = np.arange(3); width=.35
    for j, (method, label) in enumerate(methods.items()):
        means = [f(selected[method], m) for m in metrics]
        ax.bar(x + (j-.5)*width, means, width, label=label, color=COLORS["expected"] if j==0 else COLORS["material"])
    ax.set(xticks=x, xticklabels=labels, ylim=(0,1.05), title="Auditable objective intervention")
    ax.legend()

    ax = axes[3]; panel_label(ax, "D")
    labels = ["False pre-activation", "Suppression", "Success", "Violation CVaR"]
    expected_row, cvar_row = selected["stage2_expected_cost"], selected["route_aware_stage2"]
    expected = [f(expected_row,"false_pre_activation_rate"), f(expected_row,"suppression_rate"), f(expected_row,"success"), f(expected_row,"violation_cvar")]
    cvar = [f(cvar_row,"false_pre_activation_rate"), f(cvar_row,"suppression_rate"), f(cvar_row,"success"), f(cvar_row,"violation_cvar")]
    y=np.arange(4)
    for yi, a, b in zip(y, expected, cvar):
        ax.plot([a,b],[yi,yi], color=COLORS["muted"], lw=3); ax.scatter(a,yi,color=COLORS["expected"],s=35); ax.scatter(b,yi,color=COLORS["material"],s=35)
    ax.set(yticks=y, yticklabels=labels, xlim=(0,1), xlabel="Rate / CVaR", title="Expected cost → CVaR")
    ax.invert_yaxis()
    fig.suptitle("RQ4 — Context adapts force scale; the auditable CVaR effect is small", weight="bold")
    save_figure(fig, args.output, "rq4_adaptation_and_cvar", [coeff_path, gate_path, args.objective_summary])


if __name__ == "__main__":
    main()
