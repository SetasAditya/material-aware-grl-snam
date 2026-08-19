#!/usr/bin/env python3
"""Generate the paper figures revised in response to CB/Yi comments.

The RELLIS comparison reads the checked-in eight-event summary.  The force and
highway repairs re-layout the original paper images so no numerical trajectory
or channel information is invented when the underlying rollout cache is absent.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from common import COLORS, DEFAULT_OUTPUT, DEFAULT_RESULTS, ROOT, draw_field, event_field, f, rows, setup_style


EVENT_GROUP = {
    "mud_onset": "A — soft material",
    "puddle_expansion": "A — soft material",
    "corridor_closes": "B — hard boundary",
    "corridor_opens": "B — hard boundary",
    "crossing_obstacle": "C — dynamic obstacle",
    "moving_obstacle_blocks_detour": "C — dynamic obstacle",
    "mud_onset_detour_blocked": "D — compound",
    "delayed_escape_opens": "D — compound",
}


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def _box(ax, title: str, lines: list[str], color: str) -> None:
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.03, 0.04), 0.94, 0.90,
                                boxstyle="round,pad=0.025", facecolor=color + "12",
                                edgecolor=color, linewidth=1.6,
                                transform=ax.transAxes))
    ax.text(0.08, 0.86, title, transform=ax.transAxes, color=color,
            fontsize=12, weight="bold", va="top")
    y = 0.68
    for line in lines:
        ax.text(0.08, y, line, transform=ax.transAxes, fontsize=10.5, va="top")
        y -= 0.19


def make_overview(out: Path, traces: Path) -> None:
    trace = rows(traces / "step_traces.csv")
    specs = rows(traces / "event_specs.csv")
    selected = [r for r in trace if r["arm"] == "gate_on" and f(r, "gate_decision", 0) > 0.5]
    point = selected[len(selected) // 2]
    spec = next(r for r in specs if r["episode_id"] == point["episode_id"])

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.8), constrained_layout=True)
    ax = axes[0, 0]
    draw_field(ax, event_field(spec, f(point, "step")))
    pos = np.array([f(point, "position_x"), f(point, "position_y")])
    end = np.array([f(point, "selected_endpoint_col"), f(point, "selected_endpoint_row")])
    ax.plot([pos[0], end[0]], [pos[1], end[1]], ":", color="#0F6B3D", lw=2.2)
    ax.scatter(*pos, s=70, c="white", edgecolor="black", zorder=5)
    ax.scatter(*end, s=55, marker="x", c="#0F6B3D", zorder=5)
    ax.set(xlim=(max(0, pos[0]-22), min(100, pos[0]+22)),
           ylim=(min(100, pos[1]+22), max(0, pos[1]-22)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title("1  Local BEV percept (code output)", weight="bold", color="#3C69AF")
    ax.text(0.03, 0.04, "risk • hard mask • goal", transform=ax.transAxes,
            fontsize=10, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})

    _box(axes[0, 1], "2  Feasibility witness", [
        r"sample $K$ short maneuvers", r"traversable + clearance", r"risk improvement $\Rightarrow g$"], "#3C8C50")
    _box(axes[0, 2], "3  Additive energies", [
        r"$H=H_{\rm kin}+H_{\rm geom}$", r"$\quad+\lambda_s\tilde r+\lambda_h b(\phi)$",
        r"heads predict $\lambda_s,\lambda_h$"], "#B8871C")
    _box(axes[1, 0], "4  Force channels", [
        r"$f_{\rm mat}=-\nabla\tilde r$", r"$f_{\rm haz}=-b'(\phi)\nabla\phi$",
        r"$F=f_{\rm geom}+g\lambda_sf_{\rm mat}+\lambda_hf_{\rm haz}$"], "#8050A3")
    _box(axes[1, 1], "5  Closed-loop update", [
        r"semi-implicit $p_{t+1},q_{t+1}$", r"gate affects material exposure only",
        r"$g=0$: geometry-only soft channel"], "#466EAA")
    _box(axes[1, 2], "6  Tail-sensitive learning", [
        r"roll out trajectory cost $J$", r"optimize empirical $\mathrm{CVaR}_\alpha(J)$",
        r"focus updates on worst rollouts"], "#B74242")
    fig.suptitle("Material-aware port-Hamiltonian navigation", fontsize=15, weight="bold")
    _save(fig, out, "overview_pipeline")


def make_rellis_tradeoff(out: Path, summary: Path) -> None:
    data = list(csv.DictReader(summary.open()))
    methods = [("dwa_semantic", "Semantic DWA", COLORS["dwa"], "o"),
               ("route_aware_stage2", "Material-aware", COLORS["material"], "s")]
    groups = list(dict.fromkeys(EVENT_GROUP.values()))
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), constrained_layout=True,
                             sharex=True, sharey=True)
    for label, ax in zip(groups, axes.ravel()):
        events = [e for e, g in EVENT_GROUP.items() if g == label]
        for method, name, color, marker in methods:
            pool = [r for r in data if r["event_type"] in events and r["method"] == method]
            delay = np.mean([f(r, "reaction_delay") for r in pool])
            success = np.mean([f(r, "success") for r in pool])
            ax.scatter(delay, success, s=115, marker=marker, color=color, label=name, zorder=3)
            ax.annotate(name, (delay, success), xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.set_title(label, weight="bold")
        ax.grid(alpha=.25)
    for ax in axes[1]: ax.set_xlabel("Reaction delay (steps)  ↓")
    for ax in axes[:, 0]: ax.set_ylabel("Episode success  ↑")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("RELLIS-Dyn: responsiveness–completion trade-off", fontsize=14, weight="bold")
    _save(fig, out, "rellis_dyn_8event_group_pareto")


def _bar_tops_from_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Digitize the two bar series from the original Matplotlib artifact."""
    image = plt.imread(path)
    if image.dtype != np.uint8:
        image = (image[..., :3] * 255).round().astype(np.uint8)
    else:
        image = image[..., :3]
    # Original plot coordinates: baseline=831 px; y=102 px corresponds to 1.0.
    baseline, one = 831, 102
    colors = [np.array([253, 174, 97]), np.array([116, 173, 209])]
    series = []
    for series_index, color in enumerate(colors):
        mask = np.max(np.abs(image.astype(int) - color), axis=2) < 8
        components = []
        first = 204 + 79 * series_index
        for x0, x1 in [(first + 218*i, first + 77 + 218*i) for i in range(8)]:
            if series_index == 0 and x0 == first:
                x0 += 8  # avoid the legend swatch, which ends at x=208
            ys = np.where(mask[:, x0:x1])[0]
            components.append((baseline - ys.min()) / (baseline - one))
        series.append(np.asarray(components))
    return series[0], series[1]


def make_force_decomposition(out: Path, source: Path) -> None:
    soft, hard = _bar_tops_from_image(source)
    labels = ["Mud onset", "Puddle expansion", "Corridor closes", "Corridor opens",
              "Crossing obstacle", "Moving obstacle blocks", "Mud + blocked detour",
              "Delayed escape opens"]
    y = np.arange(len(labels)); h = .34
    fig, ax = plt.subplots(figsize=(10.2, 6.0), constrained_layout=True)
    ax.barh(y-h/2, soft, h, color="#FDAE61", label=r"material proxy $\|\nabla\tilde r\|$")
    ax.barh(y+h/2, hard, h, color="#74ADD1", label=r"hazard proxy $w(\phi)\|\nabla\phi\|$")
    ax.set(yticks=y, yticklabels=labels, xlabel="Mean channel-proxy magnitude")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.25)
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2)
    ax.set_title("RELLIS-Dyn force channels respond to distinct field structure", weight="bold")
    _save(fig, out, "rellis_dyn_force_decomposition")


def make_highway(out: Path, source: Path) -> None:
    img = plt.imread(source)
    # Data-panel bounds in the original 2719x2056 artifact. Titles and the
    # footer are deliberately excluded; only the recorded trajectory panels remain.
    xs = [(138, 1365), (1475, 2702)]
    ys = [(230, 670), (836, 1277), (1440, 1881)]
    row_names = ["Default traffic", "Open adjacent lane", "Blocked adjacent lanes"]
    outcomes = [[("OFF-ROAD", "#B42318"), ("SUCCESS", "#16794B")],
                [("COLLISION", "#B42318"), ("SUCCESS", "#16794B")],
                [("OFF-ROAD", "#B42318"), ("SUCCESS", "#16794B")]]
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 8.0), constrained_layout=True)
    for i, (y0, y1) in enumerate(ys):
        for j, (x0, x1) in enumerate(xs):
            ax = axes[i, j]
            ax.imshow(img[y0:y1, x0:x1]); ax.set_axis_off()
            status, color = outcomes[i][j]
            ax.text(.98, .96, status, transform=ax.transAxes, ha="right", va="top",
                    color="white", fontsize=12, weight="bold",
                    bbox={"boxstyle": "round,pad=.3", "facecolor": color, "edgecolor": color})
            if j == 0:
                ax.text(-.02, .5, row_names[i], transform=ax.transAxes, ha="right", va="center",
                        rotation=90, fontsize=11, weight="bold")
    axes[0, 0].text(.5, 1.03, "Geometry-only", transform=axes[0, 0].transAxes,
                    ha="center", fontsize=13, weight="bold", color="#2878B5")
    axes[0, 1].text(.5, 1.03, "Material-aware", transform=axes[0, 1].transAxes,
                    ha="center", fontsize=13, weight="bold", color="#D62828")
    _save(fig, out, "highway_scenario_path_panels")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overleaf-figures", type=Path, required=True)
    args = parser.parse_args(); setup_style()
    make_overview(args.output, DEFAULT_RESULTS / "exp1_gate_ablation_100")
    make_rellis_tradeoff(args.output,
        DEFAULT_RESULTS / "exp8_semantic_apf_delayed" / "historical_8event_preliminary.csv")
    make_force_decomposition(args.output, args.overleaf_figures / "rellis_dyn_force_decomposition.png")
    make_highway(args.output, args.overleaf_figures / "highway_scenario_path_panels.png")


if __name__ == "__main__":
    main()
