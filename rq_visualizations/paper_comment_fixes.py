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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

from common import COLORS, DEFAULT_OUTPUT, DEFAULT_RESULTS, ROOT, draw_field, event_field, f, fs, rows, setup_style


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


W_OVERVIEW, W_TRADEOFF, W_FORCE, W_HIGHWAY = 12.2, 9.6, 10.2, 11.8
FRAC_TRADEOFF, FRAC_FORCE = 0.88, 0.90


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=320)
    plt.close(fig)


def _box(ax, title: str, lines: list[str], color: str) -> None:
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.03, 0.04), 0.94, 0.90,
                                boxstyle="round,pad=0.025", facecolor=color + "12",
                                edgecolor=color, linewidth=1.6,
                                transform=ax.transAxes))
    ax.text(0.08, 0.86, title, transform=ax.transAxes, color=color,
            fontsize=fs(11), weight="bold", va="top")
    y = 0.68
    for line in lines:
        ax.text(0.075, y, line, transform=ax.transAxes, fontsize=fs(9.5), va="top")
        y -= 0.19


def make_overview(out: Path, traces: Path) -> None:
    setup_style(W_OVERVIEW)
    trace = rows(traces / "step_traces.csv")
    specs = rows(traces / "event_specs.csv")
    selected = [r for r in trace if r["arm"] == "gate_on" and f(r, "gate_decision", 0) > 0.5]
    point = selected[len(selected) // 2]
    spec = next(r for r in specs if r["episode_id"] == point["episode_id"])

    fig, axes = plt.subplots(2, 3, figsize=(W_OVERVIEW, 6.8), constrained_layout=True)
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
    ax.set_title("1  Local BEV percept", weight="bold", color="#3C69AF")
    ax.text(0.03, 0.04, "risk • hard mask • goal", transform=ax.transAxes,
            fontsize=fs(10), bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})

    _box(axes[0, 1], "2  Feasibility witness", [
        r"sample $K$ short maneuvers", r"traversable + clearance", r"risk improvement $\Rightarrow g$"], "#3C8C50")
    _box(axes[0, 2], "3  Additive energies", [
        r"$H=H_{\rm kin}+H_{\rm geom}$", r"$\quad+\lambda_s\tilde r+\lambda_h b(\phi)$",
        r"heads predict $\lambda_s,\lambda_h$"], "#B8871C")
    _box(axes[1, 0], "4  Force channels", [
        r"$f_{\rm mat}=-\nabla\tilde r$", r"$f_{\rm haz}=-b'(\phi)\nabla\phi$",
        r"$F=f_{\rm geom}+g\lambda_sf_{\rm mat}+\lambda_hf_{\rm haz}$"], "#8050A3")
    _box(axes[1, 1], "5  Closed-loop update", [
        r"semi-implicit $p_{t+1},q_{t+1}$", r"gate affects material exposure",
        r"$g=0$ recovers geometry-only"], "#466EAA")
    _box(axes[1, 2], "6  Tail-sensitive learning", [
        r"roll out trajectory cost $J$", r"optimize $\mathrm{CVaR}_\alpha(J)$",
        r"focus updates on worst rollouts"], "#B74242")
    fig.suptitle("Material-aware port-Hamiltonian navigation", fontsize=fs(13.5), weight="bold")
    _save(fig, out, "overview_pipeline")


def make_rellis_tradeoff(out: Path, summary: Path) -> None:
    setup_style(W_TRADEOFF, frac=FRAC_TRADEOFF)
    data = list(csv.DictReader(summary.open()))
    methods = [("dwa_semantic", "Semantic DWA", COLORS["dwa"], "o"),
               ("route_aware_stage2", "Material-aware", COLORS["material"], "s")]
    groups = list(dict.fromkeys(EVENT_GROUP.values()))
    fig, axes = plt.subplots(2, 2, figsize=(W_TRADEOFF, 7.6), constrained_layout=True,
                             sharex=True, sharey=True)
    for label, ax in zip(groups, axes.ravel()):
        events = [e for e, g in EVENT_GROUP.items() if g == label]
        for method, name, color, marker in methods:
            pool = [r for r in data if r["event_type"] in events and r["method"] == method]
            delay = np.mean([f(r, "reaction_delay") for r in pool])
            success = np.mean([f(r, "success") for r in pool])
            ax.scatter(delay, success, s=115, marker=marker, color=color, label=name, zorder=3)
            # place the label inboard for points near the right edge, or it
            # runs past the axes and gets cut off at the figure boundary
            right = delay > np.mean(ax.get_xlim()) if ax.get_xlim()[1] > ax.get_xlim()[0] else False
            ax.annotate(name, (delay, success), xytext=(-6 if right else 6, 6),
                        textcoords="offset points", fontsize=fs(8.5),
                        ha="right" if right else "left")
        ax.set_title(label, weight="bold")
        ax.grid(alpha=.25)
    for ax in axes[1]: ax.set_xlabel("Reaction delay (steps)  ↓")
    for ax in axes[:, 0]: ax.set_ylabel("Episode success  ↑")
    axes[0, 0].legend(loc="lower right", fontsize=fs(8.5))
    fig.suptitle("RELLIS-Dyn: responsiveness–completion trade-off", fontsize=fs(14), weight="bold")
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
    setup_style(W_FORCE, frac=FRAC_FORCE)
    soft, hard = _bar_tops_from_image(source)
    labels = ["Mud onset", "Puddle expansion", "Corridor closes", "Corridor opens",
              "Crossing obstacle", "Moving obstacle blocks", "Mud + blocked detour",
              "Delayed escape opens"]
    y = np.arange(len(labels)); h = .34
    fig, ax = plt.subplots(figsize=(W_FORCE, 6.0), constrained_layout=True)
    ax.barh(y-h/2, soft, h, color="#FDAE61", label=r"material proxy $\|\nabla\tilde r\|$")
    ax.barh(y+h/2, hard, h, color="#74ADD1", label=r"hazard proxy $w(\phi)\|\nabla\phi\|$")
    ax.set(yticks=y, yticklabels=labels, xlabel="Mean channel-proxy magnitude")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.25)
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2)
    ax.set_title("RELLIS-Dyn force channels respond to distinct field structure", weight="bold")
    _save(fig, out, "rellis_dyn_force_decomposition")


def _trace_from_panel(img: np.ndarray, bounds: tuple[int, int, int, int],
                      rgb: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Recover the plotted trace and road edges from the original raster panel."""
    x0, x1, y0, y1 = bounds
    panel = img[y0:y1, x0:x1, :3]
    if panel.dtype != np.uint8:
        panel = (panel * 255).round().astype(np.uint8)
    distance = np.linalg.norm(panel.astype(float) - np.asarray(rgb), axis=2)
    mask = distance < 35
    mask[:25] = False; mask[-25:] = False; mask[:, :25] = False; mask[:, -25:] = False
    xs, medians = [], []
    for x in np.where(mask.sum(0) > 0)[0]:
        yy = np.where(mask[:, x])[0]
        if len(yy):
            xs.append(x); medians.append(float(np.median(yy)))
    xs = np.asarray(xs, float); medians = np.asarray(medians, float)
    order = np.argsort(xs); xs, medians = xs[order], medians[order]
    # One point per x coordinate; median filtering removes arrowheads and markers.
    if len(medians) > 9:
        from scipy.ndimage import median_filter
        medians = median_filter(medians, size=9, mode="nearest")
    # Preserve the longitudinal extent shown by the original shared panel
    # scale; normalizing each trace independently would make an early collision
    # look like a full-length rollout.
    xx = xs / panel.shape[1]
    yy = 1.0 - medians / panel.shape[0]

    dark = panel.mean(2) < 55
    row_strength = dark[:, 30:-30].mean(1)
    candidate = np.where(row_strength > .35)[0]
    groups = np.split(candidate, np.where(np.diff(candidate) > 1)[0] + 1) if len(candidate) else []
    edges = [float(np.mean(g)) for g in groups if len(g)]
    road = (1.0 - max(edges) / panel.shape[0], 1.0 - min(edges) / panel.shape[0])
    return xx, yy, road


def make_highway(out: Path, source: Path) -> None:
    setup_style(W_HIGHWAY)
    """Large, clean redraw of the six measured traces in the original artifact."""
    img = plt.imread(source)
    xs = [(138, 1365), (1475, 2702)]
    ys = [(230, 670), (836, 1277), (1440, 1881)]
    row_names = ["Default\ntraffic", "Open\nadjacent lane", "Blocked\nadjacent lanes"]
    outcomes = [[("FAILURE: OFF-ROAD", "#B42318"), ("SUCCESS: ON-ROAD", "#16794B")],
                [("FAILURE: COLLISION", "#B42318"), ("SUCCESS: SAFE PASS", "#16794B")],
                [("FAILURE: OFF-ROAD", "#B42318"), ("SUCCESS: SAFE WAIT", "#16794B")]]
    trace_colors = [(31, 119, 180), (214, 39, 40)]
    fig, axes = plt.subplots(3, 2, figsize=(W_HIGHWAY, 6.6), constrained_layout=True)
    for i, (y0, y1) in enumerate(ys):
        for j, (x0, x1) in enumerate(xs):
            ax = axes[i, j]
            xx, yy, road = _trace_from_panel(img, (x0, x1, y0, y1), trace_colors[j])
            ax.axhspan(0, road[0], color="#FCE8E8", zorder=0)
            ax.axhspan(road[1], 1, color="#FCE8E8", zorder=0)
            ax.axhspan(road[0], road[1], color="#F7F8F8", zorder=0)
            for lane in np.linspace(road[0], road[1], 5)[1:-1]:
                ax.axhline(lane, color="#A5A5A5", lw=1.1, ls=(0, (5, 5)), zorder=1)
            ax.axhline(road[0], color="black", lw=1.4); ax.axhline(road[1], color="black", lw=1.4)
            # Minimal traffic context: leader plus adjacent-lane traffic/blockers.
            lane_centres = np.linspace(road[0], road[1], 5)[:-1] + (road[1]-road[0])/8
            cars = [(0.42, lane_centres[-1]), (0.68, lane_centres[1])]
            if i >= 1:
                cars.append((0.19, yy[0]))  # slow leader
            if i == 2:
                cars.extend([(0.22, lane_centres[0]), (0.22, lane_centres[2])])
            for cx, cy in cars:
                ax.add_patch(Rectangle((cx-.025, cy-.035), .05, .07,
                                       facecolor="#C9C9C9", edgecolor="#555555", lw=.9, zorder=2))
            color = "#2878B5" if j == 0 else "#D62828"
            ax.plot(xx, yy, color="white", lw=7, solid_capstyle="round", zorder=3)
            ax.plot(xx, yy, color=color, lw=4.2, solid_capstyle="round", zorder=4)
            ax.scatter(xx[0], yy[0], s=75, facecolor="white", edgecolor="black", lw=1.6, zorder=5)
            ax.scatter(xx[-1], yy[-1], s=85, facecolor=color, edgecolor="black", lw=1.6, zorder=5)
            status, badge = outcomes[i][j]
            ax.text(.98, .94, status, transform=ax.transAxes, ha="right", va="top",
                    color="white", fontsize=fs(11), weight="bold",
                    bbox={"boxstyle": "round,pad=.3", "facecolor": badge, "edgecolor": badge})
            ax.set(xlim=(-.03, 1.03), ylim=(0, 1)); ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if j == 0:
                ax.set_ylabel(row_names[i], fontsize=fs(8.5), weight="bold", rotation=90,
                              labelpad=6, ha="center", va="center")
    axes[0, 0].set_title("Geometry-only", fontsize=fs(14), weight="bold", color="#2878B5")
    axes[0, 1].set_title("Material-aware", fontsize=fs(14), weight="bold", color="#D62828")
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
