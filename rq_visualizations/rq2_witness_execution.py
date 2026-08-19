"""RQ2: witness/path separation and safety-property correspondence.

The geometry panel shows the logged witness rays and executed displacements at
gate-positive steps, so the witness/path distinction is drawn from measured
decisions rather than illustrated with a sketch.
"""

from __future__ import annotations

import argparse
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
    panel_label,
    rows,
    save_figure,
    setup_style,
)


def witness_pairs(trace: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Unit witness directions and unit executed displacements, gate-positive.

    ``selected_direction_(col,row)`` is already expressed as ``(x, y)`` and is
    colinear with the ray to ``selected_endpoint_*``; the executed step is the
    logged pose difference.
    """
    witness, executed = [], []
    for row in trace:
        if row["arm"] != "gate_on" or f(row, "gate_decision", 0) <= 0.5:
            continue
        direction = np.array([f(row, "selected_direction_col"), f(row, "selected_direction_row")])
        step = np.array([f(row, "next_x") - f(row, "position_x"),
                         f(row, "next_y") - f(row, "position_y")])
        if not (np.all(np.isfinite(direction)) and np.all(np.isfinite(step))):
            continue
        if np.linalg.norm(direction) < 1e-6 or np.linalg.norm(step) < 1e-6:
            continue
        witness.append(direction / np.linalg.norm(direction))
        executed.append(step / np.linalg.norm(step))
    return np.asarray(witness), np.asarray(executed)


def draw_geometry(ax, spec, trace) -> float:
    """Witness ray vs executed step at each gate-positive pose, on the field.

    Returns the step whose field is drawn: the overlay is nonstationary, so a
    single slice is shown while the decisions span the episode.
    """
    active = [r for r in trace if f(r, "gate_decision", 0) > 0.5]
    reference = active[len(active) // 2] if active else trace[len(trace) // 2]
    field_step = f(reference, "step")
    draw_field(ax, event_field(spec, field_step))

    track = np.array([[f(r, "position_x"), f(r, "position_y")] for r in trace])
    ax.plot(track[:, 0], track[:, 1], color=COLORS["material"], lw=2.0, zorder=3,
            label="executed path")

    for row in active:
        pos = np.array([f(row, "position_x"), f(row, "position_y")])
        endpoint = np.array([f(row, "selected_endpoint_col"), f(row, "selected_endpoint_row")])
        step = np.array([f(row, "next_x"), f(row, "next_y")])
        if np.all(np.isfinite(endpoint)):
            ax.plot([pos[0], endpoint[0]], [pos[1], endpoint[1]], ":", color="#0F6B3D",
                    lw=1.3, alpha=0.9, zorder=4)
            ax.scatter(*endpoint, s=10, marker="x", color="#0F6B3D", lw=1.1, zorder=5)
        # Executed displacement is short; scale it so the direction is legible.
        delta = (step - pos)
        if np.linalg.norm(delta) > 1e-6:
            unit = delta / np.linalg.norm(delta)
            ax.annotate("", xy=pos + unit * 4.0, xytext=pos,
                        arrowprops={"arrowstyle": "-|>", "color": "#111111", "lw": 1.4}, zorder=6)
        ax.scatter(*pos, s=12, color="#FFFFFF", edgecolor="#111111", lw=0.7, zorder=7)

    pad = 6.0
    lo = track.min(0) - pad
    hi = track.max(0) + pad
    span = float(max(hi - lo))
    mid = (lo + hi) / 2.0
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] + span / 2, mid[1] - span / 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return field_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS / "gaps1_2_gate_analysis")
    parser.add_argument("--traces", type=Path, default=DEFAULT_RESULTS / "exp1_gate_ablation_100")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode", default=None)
    args = parser.parse_args()
    setup_style()

    source = args.results / "witness_execution_by_regime_phase.csv"
    trace_path = args.traces / "step_traces.csv"
    spec_path = args.traces / "event_specs.csv"
    data = rows(source)
    overall = next(r for r in data if r["group"] == "overall")
    post = next((r for r in data if "post" in r["group"].lower()), None)
    if post is None:
        post = max(data, key=lambda r: float(r.get("clearance_agreement_rate", 0) or 0))

    trace, specs = rows(trace_path), rows(spec_path)
    if args.episode is None:
        counts: dict[str, int] = {}
        for row in trace:
            if row["arm"] == "gate_on" and f(row, "gate_decision", 0) > 0.5:
                counts[row["episode_id"]] = counts.get(row["episode_id"], 0) + 1
        episode = max(counts, key=counts.get)
    else:
        episode = args.episode
    spec = next(r for r in specs if r["episode_id"] == episode)
    selected = sorted(
        [r for r in trace if r["episode_id"] == episode and r["arm"] == "gate_on"],
        key=lambda r: f(r, "step"),
    )

    fig = plt.figure(figsize=(10.2, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    field_step = draw_geometry(ax, spec, selected)
    ax.set_title(f"Logged geometry, episode {episode}\n(gate-positive steps; field at step {int(field_step)})", fontsize=9)
    ax.legend(handles=[
        Line2D([], [], color=COLORS["material"], lw=2.0, label="executed path"),
        Line2D([], [], color="#0F6B3D", lw=1.3, ls=":", label="witness ray"),
        Line2D([], [], color="#111111", lw=1.4, label="executed step"),
        *field_legend_handles(),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    witness, executed = witness_pairs(trace)
    cosines = np.sum(witness * executed, axis=1)
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    ax.hist(angles, bins=np.arange(0, 181, 7.5), color=COLORS["material_light"],
            edgecolor=COLORS["material"], lw=0.5)
    median = float(np.median(angles))
    ax.axvline(median, color=COLORS["risk"], ls="--", lw=1.3)
    ax.text(median + 4, 0.92, f"median {median:.0f}°", transform=ax.get_xaxis_transform(),
            fontsize=8, color=COLORS["risk"])
    ax.set(xlabel="Angle between witness ray\nand executed step (deg)", ylabel="Gate-positive steps",
           xlim=(0, 180), xticks=[0, 45, 90, 135, 180],
           title=f"Paths diverge in direction\n(n={len(angles)} decisions)")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "C")
    names = ["Direction\ncosine", "Clearance\nagreement", "Risk-sign\nagreement"]
    values = [float(overall["median_directional_cosine"]),
              float(overall["clearance_agreement_rate"]),
              float(overall["improvement_sign_agreement_rate"])]
    bars = ax.bar(names, values, color=[COLORS["material_light"], COLORS["safe"], COLORS["material_light"]])
    ax.axhline(0.5, color=COLORS["geometry"], ls="--", lw=1)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)
    ax.set(ylim=(0, 1), ylabel="Agreement",
           title=f"Partial property agreement\n(n={overall['n_gate_positive']} decisions)")

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "D")
    clearance = float(post["clearance_agreement_rate"])
    contact_agree = 1.0 - float(post["hard_contact_disagreement_rate"])
    bars = ax.bar(["Clearance\nagreement", "Hard-contact\nagreement"], [clearance, contact_agree],
                  color=[COLORS["safe"], COLORS["safe"]])
    for bar, value in zip(bars, [clearance, contact_agree]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)
    ax.set(ylim=(0, 1), ylabel="Agreement", title="Least-confounded\npost-opening phase")

    ax.text(0.03, 0.04, "WITNESS $\u2260$ EXECUTED PATH",
            transform=ax.transAxes, fontsize=10, weight="bold",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF3E0",
                  "edgecolor": COLORS["fixed"]})
    fig.suptitle("RQ2 — The primitive is a feasibility witness, not a tracked reference",
                 fontsize=14, weight="bold")
    save_figure(fig, args.output, "rq2_witness_execution", [source, trace_path, spec_path])


if __name__ == "__main__":
    main()
