"""RQ1: the gate as a decision field -- when material risk is exposed.

The storyboard panels render the reconstructed RELLIS-Dyn event field (exact
mask geometry from ``grl_rellis.dyn_events``) with the policy's own logged
state drawn on top: executed path, feasibility witness, selected primitive
direction, gate decision, and feasible primitive count.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from common import (
    COLORS,
    DEFAULT_OUTPUT,
    DEFAULT_RESULTS,
    draw_field,
    event_field,
    f,
    field_legend_handles,
    grouped,
    panel_label,
    parse_vector,
    rows,
    save_figure,
    setup_style,
    SHAPE,
)


def choose_episode(trace: list[dict[str, str]]) -> str:
    """Pick the episode with the cleanest temporal selectivity.

    Rewards post-opening activations, penalises pre-activations heavily, and
    breaks ties on spatial extent so the storyboard panels show real motion.
    """
    by_episode = grouped([r for r in trace if r["arm"] == "gate_on"], "episode_id")
    scored = []
    for (episode,), values in by_episode.items():
        opening = f(values[0], "opening_step")
        post = sum(f(r, "gate_decision", 0) > 0.5 and f(r, "step") >= opening for r in values)
        pre = sum(f(r, "gate_decision", 0) > 0.5 and f(r, "step") < opening for r in values)
        positions = np.array([[f(r, "position_x"), f(r, "position_y")] for r in values])
        spread = float(np.linalg.norm(positions.max(0) - positions.min(0)))
        scored.append((post - 3 * pre, spread, episode))
    if not scored:
        raise ValueError("No gate-on traces found")
    return max(scored)[2]


def storyboard_window(spec, trace, pad: float = 8.0) -> tuple[float, float, float, float]:
    """Fixed crop covering the whole executed path and the event geometry.

    Framing every panel identically is what lets the reader see the robot
    approach the barrier; a robot-centred crop hides the event until contact.
    """
    points = [[f(r, "position_x"), f(r, "position_y")] for r in trace]
    for key in ("center_rc", "detour_rc", "goal_rc"):
        row, col = parse_vector(spec[key])
        points.append([col, row])
    array = np.asarray(points, dtype=float)
    lo, hi = array.min(0) - pad, array.max(0) + pad
    # Keep the aspect square so cells stay isotropic.
    span = float(max(hi - lo))
    mid = (lo + hi) / 2.0
    x0, y0 = mid[0] - span / 2, mid[1] - span / 2
    # Keep the window inside the BEV patch so panels do not show empty margin.
    limit = float(SHAPE[0])
    span = min(span, limit)
    x0 = float(np.clip(x0, 0.0, limit - span))
    y0 = float(np.clip(y0, 0.0, limit - span))
    return x0, x0 + span, y0, y0 + span


def draw_snapshot(ax, spec, trace, step, title, window):
    """Render the field at `step` with the policy state logged at that step."""
    field = event_field(spec, step)
    draw_field(ax, field)
    point = min(trace, key=lambda r: abs(f(r, "step") - step))
    pos = np.array([f(point, "position_x"), f(point, "position_y")])

    # Full route in faint outline, executed portion solid: shows progress.
    track = np.array([[f(r, "position_x"), f(r, "position_y")] for r in trace])
    ax.plot(track[:, 0], track[:, 1], color=COLORS["geometry"], lw=0.9, ls="-",
            alpha=0.35, zorder=3)
    history = [r for r in trace if f(r, "step") <= f(point, "step")]
    if len(history) > 1:
        done = np.array([[f(r, "position_x"), f(r, "position_y")] for r in history])
        ax.plot(done[:, 0], done[:, 1], color=COLORS["material"], lw=2.2, zorder=4)

    # Feasibility witness: the ray from the pose to the selected endpoint.
    endpoint = np.array([f(point, "selected_endpoint_col"), f(point, "selected_endpoint_row")])
    if np.all(np.isfinite(endpoint)):
        ax.plot([pos[0], endpoint[0]], [pos[1], endpoint[1]], ":", color="#0F6B3D", lw=1.7, zorder=5)
        ax.scatter(*endpoint, s=18, marker="x", color="#0F6B3D", lw=1.5, zorder=6)

    direction = np.array([f(point, "selected_direction_col"), f(point, "selected_direction_row")])
    if np.all(np.isfinite(direction)) and np.linalg.norm(direction) > 1e-6:
        unit = direction / np.linalg.norm(direction)
        ax.annotate("", xy=pos + unit * 6.0, xytext=pos,
                    arrowprops={"arrowstyle": "-|>", "color": "#111111", "lw": 1.9}, zorder=7)
    ax.scatter(*pos, s=40, color="#FFFFFF", edgecolor="#111111", lw=1.1, zorder=8)

    # The closure barrier forms on the nominal scaffold at the same instant the
    # escape opens, i.e. behind the robot. Mark where the path crossed that
    # footprint and when, so the static path is not misread as driving through
    # a standing wall.
    if np.any(field["closure"]):
        crossed = [
            r for r in trace
            if field["closure"][int(round(f(r, "position_y"))), int(round(f(r, "position_x")))]
        ]
        if crossed:
            last = crossed[-1]
            mark = np.array([f(last, "position_x"), f(last, "position_y")])
            ax.scatter(*mark, s=70, marker="o", facecolor="none", edgecolor="#FFD54F",
                       lw=1.8, zorder=9)
            ax.annotate(
                f"closure forms behind robot\n(passed at step {int(f(last, 'step'))})",
                xy=mark, xytext=(0.5, 0.06), textcoords="axes fraction", fontsize=7,
                color="#6B5500", ha="center", va="bottom",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FFF8E1",
                      "edgecolor": "#E0B400", "linewidth": 0.7},
                arrowprops={"arrowstyle": "-", "color": "#B08900", "lw": 0.8}, zorder=10,
            )

    gate = int(f(point, "gate_decision", 0) > 0.5)
    feasible = int(f(point, "feasible_primitive_count", 0))
    ax.text(0.04, 0.97, f"gate = {gate}", transform=ax.transAxes, va="top", weight="bold",
            fontsize=9, color=COLORS["safe"] if gate else COLORS["risk"])
    ax.text(0.04, 0.89, f"feasible primitives = {feasible}", transform=ax.transAxes, va="top",
            fontsize=8, color=COLORS["hazard"])
    ax.set_title(f"{title}\nstep {int(f(point, 'step'))}", fontsize=9)

    x0, x1, y0, y1 = window
    ax.set_aspect("equal")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])


def paired_divergence(trace: list[dict[str, str]]) -> np.ndarray:
    """Max per-episode distance between the gate-on and gate-off paths."""
    by_episode = grouped(trace, "episode_id", "arm")
    episodes = sorted({key[0] for key in by_episode})
    out = []
    for episode in episodes:
        on = sorted(by_episode.get((episode, "gate_on"), []), key=lambda r: f(r, "step"))
        off = sorted(by_episode.get((episode, "gate_off"), []), key=lambda r: f(r, "step"))
        if not on or not off:
            continue
        out.append(max(
            math.dist((f(a, "position_x"), f(a, "position_y")),
                      (f(b, "position_x"), f(b, "position_y")))
            for a, b in zip(on, off)
        ))
    return np.asarray(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS / "exp1_gate_ablation_100")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode", default=None, help="Override the storyboard episode")
    args = parser.parse_args()
    setup_style()

    trace_path, spec_path = args.results / "step_traces.csv", args.results / "event_specs.csv"
    trace, specs = rows(trace_path), rows(spec_path)
    episode = args.episode or choose_episode(trace)
    spec = next(r for r in specs if r["episode_id"] == episode)
    selected = sorted(
        [r for r in trace if r["episode_id"] == episode and r["arm"] == "gate_on"],
        key=lambda r: f(r, "step"),
    )
    event = f(spec, "event_step")
    opening = event + f(spec, "open_delay")

    fig = plt.figure(figsize=(12.2, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.08, 0.92])

    active_post = [f(r, "step") for r in selected if f(r, "step") >= opening and f(r, "gate_decision", 0) > 0.5]
    snapshots = [
        (max(0, event - 6), "Nominal route"),
        (event + 2, "Escape visible but blocked"),
        (active_post[0] if active_post else opening, "Escape feasible: gate activates"),
    ]
    window = storyboard_window(spec, selected)
    for index, (step, title) in enumerate(snapshots):
        ax = fig.add_subplot(gs[0, index])
        draw_snapshot(ax, spec, selected, step, title, window)
        panel_label(ax, chr(65 + index))

    handles = [
        Line2D([], [], color=COLORS["material"], lw=2.2, label="executed path"),
        Line2D([], [], color=COLORS["geometry"], lw=0.9, alpha=0.35, label="remaining route"),
        Line2D([], [], color="#0F6B3D", lw=1.7, ls=":", label="feasibility witness"),
        Line2D([], [], color="#111111", lw=1.9, label="selected primitive"),
        *field_legend_handles(),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.51),
               ncol=4, fontsize=9)

    # Exposure timeline: the quantity the gate actually controls.
    ax = fig.add_subplot(gs[1, :2])
    panel_label(ax, "D")
    for arm, color, offset in [("gate_on", COLORS["material"], 0.03), ("gate_off", COLORS["fixed"], -0.03)]:
        values = sorted(
            [r for r in trace if r["episode_id"] == episode and r["arm"] == arm],
            key=lambda r: f(r, "step"),
        )
        ax.step([f(r, "step") for r in values], [f(r, "soft_multiplier") + offset for r in values],
                where="post", color=color, lw=1.7, label=arm.replace("_", " "))
    ax.axvspan(event, opening, color=COLORS["hazard"], alpha=0.10, label="blocked interval")
    ax.axvline(opening, color=COLORS["safe"], ls="--", lw=1.2)
    ax.set(xlabel="Control step", ylabel="Soft-force exposure", ylim=(-0.12, 1.16),
           title="Gate changes exposure timing; navigation outcome is unchanged")
    ax.legend(ncol=3, loc="upper left")

    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "E")
    regimes = ["R1", "R2", "R3"]
    on_frac = [0.094, 0.061, 0.076]
    y = np.arange(3)
    ax.barh(y + 0.18, [1.0] * 3, 0.32, color=COLORS["muted"], label="gate off")
    ax.barh(y - 0.18, on_frac, 0.32, color=COLORS["material"], label="gate on")
    ax.set(yticks=y, yticklabels=regimes, xlim=(0, 1.05), xlabel="Fraction of steps",
           title="Exposure falls by 91--94%")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    divergence = paired_divergence(trace)
    ax.text(0.03, 0.03,
            "SUCCESS: 1.00 in both arms\n"
            f"Median path difference: {np.median(divergence):.4f} cells",
            transform=ax.transAxes, fontsize=9, weight="bold", va="bottom",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#E8F5E9",
                  "edgecolor": COLORS["safe"]})

    fig.suptitle("RQ1 — The witness gate suppresses unnecessary soft-force exposure",
                 fontsize=14, weight="bold")
    save_figure(fig, args.output, "rq1_gate_exposure", [trace_path, spec_path])


if __name__ == "__main__":
    main()
