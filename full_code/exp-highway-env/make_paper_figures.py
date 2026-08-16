#!/usr/bin/env python3
"""
make_paper_figures.py — generates Figures 2, 3, 4 from existing JSONs.

Reads:
    - eval_paired.json          (Figures 3-4 input; Table 2 input)
    - force_diagnostic.json     (Figure 2 input)
Writes:
    - figures/figure2_paired_rollout.{pdf,png}
    - figures/figure3_force_decomposition.{pdf,png}
    - figures/figure4_lateral_factors.{pdf,png}

Figure 2: paired rollout snapshot (3 scenarios × 3 timepoints × stages).
    Shows what each model does at key moments. Built from the per-step
    diagnostic data (we trace ego_x, ego_y over time, with overlays).

Figure 3: outcome summary across scenarios.
    Crash/off-road rates, progress, and lane changes. This is the main
    paper-facing evidence: Stage 2 fixes the default scaffold, passes the
    solvable slow-leader case, and does not invent an escape in boxed.

Figure 4: paired per-seed distance transitions.
    Shows each paired seed moving from Stage 1 to Stage 2, with endpoint
    status markers. This makes the "only when the affordance exists" story
    visible without leaning on force internals.

All figures save as both .pdf (vector, paper) and .png (raster, slides).
No seaborn dependency.

Usage
-----
    python make_paper_figures.py \\
        --force-diagnostic runs/paper_data/force_diagnostic.json \\
        --paired-eval runs/paper_data/eval_paired_full.json \\
        --out figures/

    # Re-render only one figure:
    python make_paper_figures.py --force-diagnostic <path> --only fig3
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Paper style — minimal, no seaborn
# ─────────────────────────────────────────────────────────────────────────────

PAPER_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# Distinct, colorblind-friendly palette
PAL = {
    "stage1":   "#1f77b4",   # blue
    "stage2":   "#d62728",   # red
    "clean":    "#2ca25f",   # green
    "offroad":  "#f28e2b",   # orange
    "crash":    "#d62728",   # red
    "neutral":  "#6b7280",   # gray
    "F_goal":   "#2ca02c",   # green
    "F_geom":   "#ff7f0e",   # orange
    "F_soft":   "#17becf",   # teal
    "F_hard":   "#bcbd22",   # olive
    "F_lat":    "#9467bd",   # purple — the spotlight star
    "mu_lat":   "#9467bd",
    "side":     "#e377c2",   # pink
    "Flat_mag": "#9467bd",
    "neighbor": "#aaaaaa",
    "ego_path": "#000000",
    "lane":     "#bbbbbb",
}

# Scenario display order + pretty names
SCENARIO_DISPLAY = [
    ("default",                     "Default"),
    ("authored_slow_leader",        "Slow leader (passable)"),
    ("authored_slow_leader_boxed",  "Boxed (no escape)"),
]

LATERAL_LANE_WIDTH = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Data loading + reshaping
# ─────────────────────────────────────────────────────────────────────────────

def load_force_diagnostic(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_paired_eval(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        print(f"  paired eval not found: {path}; skipping paired-eval figures")
        return None
    with open(path) as f:
        return json.load(f)


def gather_window_steps(scn_data: Dict[str, List], stage: str,
                          *, scenario_name: str = "") -> List[Dict[str, Any]]:
    """Pool all window_steps across episodes for a scenario × stage.

    For default scenario where most episodes have no trigger window,
    fall back to all_steps (per the user's spec on default handling).
    """
    use_all = scenario_name == "default"
    pooled = []
    for ep in scn_data[stage]:
        steps = ep["all_steps"] if use_all else ep["window_steps"]
        if not steps and ep["all_steps"]:
            steps = ep["all_steps"]
        pooled.extend(steps)
    return pooled


def force_magnitude(F: List[float]) -> float:
    return math.sqrt(F[0]**2 + F[1]**2)


def paired_scenarios(paired: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [(name, pretty) for name, pretty in SCENARIO_DISPLAY
            if name in paired.get("results_by_scenario", {})]


def agg_value(paired: Dict[str, Any], scenario: str, stage: str, key: str,
              default: float = 0.0) -> float:
    return float(paired.get("aggregates", {})
                 .get(scenario, {})
                 .get(stage, {})
                 .get(key, default))


def episode_status(ep: Dict[str, Any]) -> str:
    if ep.get("collided", False):
        return "crash"
    if ep.get("went_offroad", False) or ep.get("ended_offroad", False) or ep.get("offroad_steps", 0):
        return "offroad"
    return "clean"


def status_label(status: str) -> str:
    return {"clean": "clean", "offroad": "off-road", "crash": "crash"}[status]


def annotate_bar(ax, bars, *, percent: bool = False, fmt: str = "{:.0f}",
                 dy_frac: float = 0.025):
    y0, y1 = ax.get_ylim()
    dy = (y1 - y0) * dy_frac
    for bar in bars:
        h = bar.get_height()
        label = f"{h:.0f}%" if percent else fmt.format(h)
        ax.text(bar.get_x() + bar.get_width() / 2, h + dy, label,
                ha="center", va="bottom", fontsize=7)


def draw_vehicle(ax, x: float, y: float, *, color: str, edge: str = "black",
                 label: str = "", alpha: float = 1.0):
    length = 8.0
    width = 2.0
    patch = Rectangle((x - length / 2, y - width / 2), length, width,
                      facecolor=color, edgecolor=edge, linewidth=0.8,
                      alpha=alpha, zorder=4)
    ax.add_patch(patch)
    if label:
        ax.text(x, y + 1.7, label, ha="center", va="bottom", fontsize=6.5,
                color=edge, zorder=5)


def draw_scenario_snapshot(ax, scenario: str):
    """Compact authored-scenario schematic for Figure 4.

    These are not simulated screenshots; they mirror the environment setup:
    ego in lane 1, slow leader ahead, and boxed blockers in adjacent lanes.
    """
    lane_centers = [0, 4, 8, 12]
    x_lo, x_hi = 82, 166
    for y in lane_centers:
        ax.plot([x_lo, x_hi], [y, y], color=PAL["lane"], linewidth=0.7,
                linestyle=(0, (4, 4)), zorder=1)
    ax.plot([x_lo, x_hi], [-2, -2], color="black", linewidth=0.8, zorder=1)
    ax.plot([x_lo, x_hi], [14, 14], color="black", linewidth=0.8, zorder=1)

    ego_x, ego_y = 100, 4
    draw_vehicle(ax, ego_x, ego_y, color="white", edge=PAL["stage2"], label="ego")

    if scenario == "default":
        draw_vehicle(ax, 135, 0, color="#d9d9d9", edge="#777777", label="traffic",
                     alpha=0.9)
        draw_vehicle(ax, 153, 8, color="#d9d9d9", edge="#777777", alpha=0.9)
        ax.annotate("no fixed blocker", xy=(124, 4), xytext=(128, 10.5),
                    arrowprops=dict(arrowstyle="->", linewidth=0.8,
                                    color=PAL["neutral"]),
                    fontsize=7, ha="left", color=PAL["neutral"])
    else:
        leader_x = 130 if scenario == "authored_slow_leader" else 128
        draw_vehicle(ax, leader_x, ego_y, color="#f6c85f", edge="#8a5a00",
                     label="slow leader")
        ax.annotate("", xy=(leader_x - 6, ego_y), xytext=(ego_x + 6, ego_y),
                    arrowprops=dict(arrowstyle="<->", linewidth=0.8,
                                    color=PAL["neutral"]))
        ax.text((ego_x + leader_x) / 2, ego_y - 1.7, "28-30 m",
                ha="center", va="top", fontsize=6.5, color=PAL["neutral"])

        if scenario == "authored_slow_leader":
            ax.annotate("open adjacent lane", xy=(122, 8), xytext=(132, 11.5),
                        arrowprops=dict(arrowstyle="->", linewidth=0.8,
                                        color=PAL["clean"]),
                        fontsize=7, ha="left", color=PAL["clean"])
            ax.add_patch(Rectangle((104, 6.3), 48, 3.4, facecolor=PAL["clean"],
                                   edgecolor="none", alpha=0.08, zorder=0))
        else:
            for lane_y in [0, 8]:
                draw_vehicle(ax, 120, lane_y, color="#bdbdbd", edge="#555555",
                             label=("blockers" if lane_y == 8 else ""))
                draw_vehicle(ax, 154, lane_y, color="#bdbdbd", edge="#555555")
            ax.annotate("adjacent lanes blocked", xy=(120, 8), xytext=(132, 11.5),
                        arrowprops=dict(arrowstyle="->", linewidth=0.8,
                                        color=PAL["crash"]),
                        fontsize=7, ha="left", color=PAL["crash"])

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-3, 15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: paired rollout snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _leader_center_x_for_step(step_idx: int) -> float:
    """Approximate the authored slow leader center from the scenario spec."""
    # The authored slow leader starts about 30 m ahead and moves at 10 m/s.
    # With dt = 0.1 s, that is ~1 m per step.
    return 130.0 + float(step_idx)


def _draw_slowleader_storyboard_panel(ax, steps: List[Dict[str, Any]], idx: int,
                                      *, stage: str, note: str = ""):
    step = steps[idx]
    color = PAL["stage1" if stage == "stage1" else "stage2"]
    ego_x = float(step["ego_x"])
    ego_y = float(step["ego_y"])
    leader_x = _leader_center_x_for_step(int(step["step"]))
    leader_y = 4.0

    x_lo, x_hi = ego_x - 35.0, ego_x + 35.0
    for y in [0, 4, 8, 12]:
        ax.plot([x_lo, x_hi], [y, y], color=PAL["lane"], linewidth=0.7,
                linestyle=(0, (4, 4)), zorder=1)
    ax.plot([x_lo, x_hi], [-2, -2], color="black", linewidth=0.8, zorder=1)
    ax.plot([x_lo, x_hi], [14, 14], color="black", linewidth=0.8, zorder=1)

    trail_lo = max(0, idx - 8)
    trail = steps[trail_lo:idx + 1]
    ax.plot([float(s["ego_x"]) for s in trail], [float(s["ego_y"]) for s in trail],
            color=color, linewidth=2.2, alpha=0.9, zorder=3)

    draw_vehicle(ax, leader_x, leader_y, color="#f6c85f", edge="#8a5a00",
                 label=("slow leader" if abs(leader_x - ego_x) < 26 else ""))
    draw_vehicle(ax, ego_x, ego_y, color="white", edge=color, label="ego")

    if idx > 0:
        prev = steps[idx - 1]
        arr = FancyArrowPatch((float(prev["ego_x"]), float(prev["ego_y"])),
                              (ego_x, ego_y), arrowstyle="->",
                              mutation_scale=8, linewidth=1.2,
                              color=color, zorder=4)
        ax.add_patch(arr)

    if stage == "stage2":
        flat = step.get("F_lat", [0.0, 0.0])
        if len(flat) == 2:
            fy = float(flat[1])
            if abs(fy) > 1e-3:
                scale = 0.9
                arr = FancyArrowPatch((ego_x + 5.0, ego_y),
                                      (ego_x + 5.0, ego_y + fy * scale),
                                      arrowstyle="-|>", mutation_scale=10,
                                      linewidth=1.4, color=PAL["F_lat"],
                                      alpha=0.85, zorder=4)
                ax.add_patch(arr)

    dmin = float(step.get("dmin", float("nan")))
    if math.isfinite(dmin):
        ax.text(0.02, 0.94, f"d_min={dmin:.1f} m",
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                color=PAL["neutral"])
    ax.text(0.98, 0.94, f"t={0.1 * int(step['step']):.1f}s",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=PAL["neutral"])
    if note:
        ax.text(0.5, 0.06, note, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-3, 15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure2_paired_rollout(diag: Dict[str, Any], out_dir: Path,
                              *, episode_idx: int = 0):
    """Slow-leader interaction storyboard + time-series.

    The current paper story lives or dies on the passable slow-leader case.
    A multi-scenario top-down trace grid hides the decision. This figure
    instead shows the interaction window explicitly: four snapshots for each
    stage, followed by lane-position and clearance traces over time.
    """
    scn = diag["scenarios"].get("authored_slow_leader")
    if scn is None:
        print("  authored_slow_leader missing; skipping Figure 2")
        return
    if (not scn.get("stage1")) or (not scn.get("stage2")):
        print("  missing stage data for authored_slow_leader; skipping Figure 2")
        return
    if episode_idx >= len(scn["stage1"]) or episode_idx >= len(scn["stage2"]):
        print("  requested episode index out of range; skipping Figure 2")
        return

    ep1 = scn["stage1"][episode_idx]
    ep2 = scn["stage2"][episode_idx]
    s1 = ep1["all_steps"]
    s2 = ep2["all_steps"]
    if not s1 or not s2:
        print("  empty rollout for authored_slow_leader; skipping Figure 2")
        return

    s1_trigger = int(ep1["trigger_window"][0]) if ep1.get("trigger_window") else min(10, len(s1) - 1)
    s2_trigger = int(ep2["trigger_window"][0]) if ep2.get("trigger_window") else min(10, len(s2) - 1)
    s1_close = next((i for i, st in enumerate(s1) if float(st.get("dmin", 999.0)) < 8.0),
                    min(len(s1) - 2, s1_trigger + 5))
    s2_commit = next((i for i, st in enumerate(s2)
                      if abs(float(st["ego_y"]) - float(s2[0]["ego_y"])) > 1.0),
                     min(len(s2) - 2, s2_trigger + 6))
    s2_clear = next((i for i in range(s2_commit + 1, len(s2))
                     if float(s2[i]["ego_x"]) > _leader_center_x_for_step(int(s2[i]["step"])) + 8.0),
                    min(len(s2) - 1, s2_commit + 18))

    frame_labels = ["Start", "Trigger", "Decision", "Outcome"]
    stage1_idx = [0, s1_trigger, s1_close, len(s1) - 1]
    stage2_idx = [0, s2_trigger, s2_commit, s2_clear]
    stage1_notes = ["same lane", "closing on leader", "no lateral response", "crash"]
    stage2_notes = ["same lane", "lateral force engages", "lane change starts", "leader passed"]

    fig = plt.figure(figsize=(8.6, 5.8))
    gs = fig.add_gridspec(nrows=3, ncols=4, height_ratios=[1.0, 1.0, 1.1],
                          hspace=0.18, wspace=0.12)

    for c, label in enumerate(frame_labels):
        ax1 = fig.add_subplot(gs[0, c])
        ax2 = fig.add_subplot(gs[1, c])
        _draw_slowleader_storyboard_panel(ax1, s1, stage1_idx[c], stage="stage1",
                                          note=stage1_notes[c])
        _draw_slowleader_storyboard_panel(ax2, s2, stage2_idx[c], stage="stage2",
                                          note=stage2_notes[c])
        ax1.set_title(label, fontsize=9, pad=3)
        if c == 0:
            ax1.text(-0.10, 0.5, "Stage 1", transform=ax1.transAxes,
                     ha="right", va="center", fontsize=9, weight="bold",
                     color=PAL["stage1"])
            ax2.text(-0.10, 0.5, "Stage 2", transform=ax2.transAxes,
                     ha="right", va="center", fontsize=9, weight="bold",
                     color=PAL["stage2"])

    ax_lane = fig.add_subplot(gs[2, 0:2])
    ax_dmin = fig.add_subplot(gs[2, 2:4])

    t1 = np.arange(len(s1)) * 0.1
    t2 = np.arange(len(s2)) * 0.1
    y1 = np.array([float(st["ego_y"]) for st in s1], dtype=np.float64)
    y2 = np.array([float(st["ego_y"]) for st in s2], dtype=np.float64)
    d1 = np.array([float(st.get("dmin", np.nan)) for st in s1], dtype=np.float64)
    d2 = np.array([float(st.get("dmin", np.nan)) for st in s2], dtype=np.float64)

    ax_lane.plot(t1, y1, color=PAL["stage1"], linewidth=2.0,
                 linestyle=(0, (4, 2)), label="Stage 1")
    ax_lane.plot(t2, y2, color=PAL["stage2"], linewidth=2.0, label="Stage 2")
    ax_lane.axhline(4.0, color=PAL["lane"], linewidth=0.8, linestyle=(0, (4, 4)))
    ax_lane.axhline(0.0, color=PAL["lane"], linewidth=0.8, linestyle=(0, (4, 4)))
    for idx in stage1_idx[1:]:
        ax_lane.scatter([0.1 * idx], [y1[idx]], color=PAL["stage1"], s=22, zorder=4)
    for idx in stage2_idx[1:]:
        ax_lane.scatter([0.1 * idx], [y2[idx]], color=PAL["stage2"], s=22, zorder=4)
    ax_lane.set_title("Ego lateral position", fontsize=9)
    ax_lane.set_xlabel("time (s)")
    ax_lane.set_ylabel("y (m)")
    ax_lane.grid(True)
    ax_lane.legend(frameon=False, loc="upper right")

    ax_dmin.plot(t1, d1, color=PAL["stage1"], linewidth=2.0,
                 linestyle=(0, (4, 2)), label="Stage 1")
    ax_dmin.plot(t2, d2, color=PAL["stage2"], linewidth=2.0, label="Stage 2")
    ax_dmin.axhline(8.0, color=PAL["neutral"], linewidth=0.9, linestyle=(0, (3, 3)))
    ax_dmin.text(0.02, 0.06, "trigger threshold", transform=ax_dmin.transAxes,
                 fontsize=7, color=PAL["neutral"])
    for idx in stage1_idx[1:]:
        ax_dmin.scatter([0.1 * idx], [d1[idx]], color=PAL["stage1"], s=22, zorder=4)
    for idx in stage2_idx[1:]:
        ax_dmin.scatter([0.1 * idx], [d2[idx]], color=PAL["stage2"], s=22, zorder=4)
    ax_dmin.set_title("Minimum clearance", fontsize=9)
    ax_dmin.set_xlabel("time (s)")
    ax_dmin.set_ylabel("d_min (m)")
    ax_dmin.grid(True)

    fig.suptitle("Slow-leader interaction window: Stage 2 initiates the pass where Stage 1 crashes",
                 y=0.99)
    fig.text(0.5, 0.94,
             "Same seed, same local information, different outcome only when an adjacent-lane affordance exists.",
             ha="center", va="center", fontsize=8.2, color=PAL["neutral"])
    plt.tight_layout(rect=[0.02, 0.02, 1.0, 0.95])

    pdf_path = out_dir / "figure2_paired_rollout.pdf"
    png_path = out_dir / "figure2_paired_rollout.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {pdf_path} and {png_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: outcome summary
# ─────────────────────────────────────────────────────────────────────────────

def figure3_outcome_summary(paired: Dict[str, Any], out_dir: Path):
    """Outcome-level summary for the main paper.

    The plot makes the paper claim visible without asking the reader to decode
    force components: Stage 2 eliminates default off-road behavior, passes the
    solvable slow-leader case, and does not hallucinate a lane change in boxed.
    """
    scenarios = paired_scenarios(paired)
    if not scenarios:
        print("  no recognized scenarios found in paired eval; skipping Figure 3")
        return

    labels = [pretty.replace(" (", "\n(") for _, pretty in scenarios]
    x = np.arange(len(scenarios))
    w = 0.34
    s1 = "stage1"
    s2 = "stage2"

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4))
    axes = axes.ravel()

    panels = [
        ("Crash rate", "collision_rate", "% episodes", True, 100.0, (0, 110), "{:.0f}"),
        ("Went off-road", "offroad_rate", "% episodes", True, 100.0, (0, 110), "{:.0f}"),
        ("Progress", "distance_m_mean", "distance traveled (m)", False, 1.0, (0, 345), "{:.0f}"),
        ("Lane changes", "lane_changes_mean", "changes / episode", False, 1.0, (0, 1.25), "{:.2g}"),
    ]

    for ax, (title, key, ylabel, is_pct, scale, ylim, fmt) in zip(axes, panels):
        vals1 = [agg_value(paired, n, s1, key) * scale for n, _ in scenarios]
        vals2 = [agg_value(paired, n, s2, key) * scale for n, _ in scenarios]
        b1 = ax.bar(x - w / 2, vals1, w, color=PAL["stage1"],
                    edgecolor="black", linewidth=0.5, label="Stage 1")
        b2 = ax.bar(x + w / 2, vals2, w, color=PAL["stage2"],
                    edgecolor="black", linewidth=0.5, label="Stage 2")
        ax.set_title(title, weight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        annotate_bar(ax, b1, percent=is_pct, fmt=fmt)
        annotate_bar(ax, b2, percent=is_pct, fmt=fmt)

    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Stage 2 improves solvable cases and stays selective when escape is blocked",
                 y=1.00, weight="bold")
    plt.tight_layout()

    pdf_path = out_dir / "figure3_outcome_summary.pdf"
    png_path = out_dir / "figure3_outcome_summary.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {pdf_path} and {png_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: paired per-seed transitions
# ─────────────────────────────────────────────────────────────────────────────

def figure4_paired_transitions(paired: Dict[str, Any], out_dir: Path):
    scenarios = paired_scenarios(paired)
    if not scenarios:
        print("  no recognized scenarios found in paired eval; skipping Figure 4")
        return

    fig, axes = plt.subplots(2, len(scenarios),
                              figsize=(3.25 * len(scenarios), 5.2),
                              sharex=False,
                              gridspec_kw={"height_ratios": [1.05, 3.0],
                                           "hspace": 0.08, "wspace": 0.08})
    if len(scenarios) == 1:
        axes = axes.reshape(2, -1)

    marker_for_stage = {"stage1": "s", "stage2": "o"}

    for col, (scn_name, scn_pretty) in enumerate(scenarios):
        snap_ax = axes[0, col]
        ax = axes[1, col]
        draw_scenario_snapshot(snap_ax, scn_name)
        snap_ax.set_title(scn_pretty, weight="bold", pad=1.5)

        scn = paired["results_by_scenario"][scn_name]
        stage1_eps = sorted(scn["stage1"], key=lambda e: e["seed"])
        stage2_eps = sorted(scn["stage2"], key=lambda e: e["seed"])
        by_seed_2 = {e["seed"]: e for e in stage2_eps}
        pairs = [(e1, by_seed_2[e1["seed"]]) for e1 in stage1_eps
                 if e1["seed"] in by_seed_2]
        if not pairs:
            ax.set_visible(False)
            continue

        for idx, (ep1, ep2) in enumerate(pairs):
            y = idx + 1
            x1 = ep1["distance_m"]
            x2 = ep2["distance_m"]
            ax.plot([x1, x2], [y, y], color=PAL["neutral"],
                    linewidth=0.7, alpha=0.35, zorder=1)
            for stage, ep, xval in [("stage1", ep1, x1), ("stage2", ep2, x2)]:
                status = episode_status(ep)
                face = PAL[status] if stage == "stage2" else "white"
                ax.scatter(xval, y, marker=marker_for_stage[stage], s=34,
                           facecolors=face, edgecolors=PAL[status],
                           linewidths=1.2, zorder=3)

        s1_dist = agg_value(paired, scn_name, "stage1", "distance_m_mean")
        s2_dist = agg_value(paired, scn_name, "stage2", "distance_m_mean")
        s1_fail = agg_value(paired, scn_name, "stage1", "collision_rate") * 100
        s2_fail = agg_value(paired, scn_name, "stage2", "collision_rate") * 100
        s2_lc = agg_value(paired, scn_name, "stage2", "lane_changes_mean")
        ax.text(0.02, 0.96,
                f"mean distance: {s1_dist:.0f} -> {s2_dist:.0f} m\n"
                f"crash: {s1_fail:.0f}% -> {s2_fail:.0f}%\n"
                f"Stage 2 lane changes: {s2_lc:.2g}/ep",
                transform=ax.transAxes, va="top", ha="left", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0))
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_xlabel("episode distance (m)")

    trans_axes = [axes[1, c] for c in range(len(scenarios))]
    trans_axes[0].set_ylabel("paired seed")
    trans_axes[0].set_ylim(0.2, max(len(paired["results_by_scenario"][n]["stage1"])
                                    for n, _ in scenarios) + 0.8)
    trans_axes[0].set_yticks([1, 5, 10, 15, 20])
    for ax in trans_axes:
        ax.set_xlim(0, 340)
    for ax in trans_axes[1:]:
        ax.set_yticklabels([])

    legend_elements = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=PAL["neutral"], markersize=6, label="Stage 1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAL["neutral"],
               markeredgecolor=PAL["neutral"], markersize=6, label="Stage 2"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAL["clean"],
               markeredgecolor=PAL["clean"], markersize=6, label="clean"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAL["offroad"],
               markeredgecolor=PAL["offroad"], markersize=6, label="off-road"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAL["crash"],
               markeredgecolor=PAL["crash"], markersize=6, label="crash"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Environment context explains the paired result: pass when open, no escape when boxed",
                 y=1.02, weight="bold")
    plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.98])

    pdf_path = out_dir / "figure4_paired_transitions.pdf"
    png_path = out_dir / "figure4_paired_transitions.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {pdf_path} and {png_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-diagnostic", type=str, required=True,
                    help="Path to force_diagnostic.json")
    ap.add_argument("--paired-eval", type=str, default="",
                    help="Path to eval_paired_full.json for Figures 3-4")
    ap.add_argument("--out", type=str, default="figures/")
    ap.add_argument("--only", type=str, default="all",
                    choices=["all", "fig2", "fig3", "fig4"])
    ap.add_argument("--episode-idx", type=int, default=0,
                    help="Which episode to show in Figure 2 panels")
    args = ap.parse_args()

    plt.rcParams.update(PAPER_STYLE)

    diag = load_force_diagnostic(Path(args.force_diagnostic))
    paired_path = Path(args.paired_eval) if args.paired_eval else None
    paired = load_paired_eval(paired_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {args.force_diagnostic}")
    print(f"  scenarios: {list(diag['scenarios'].keys())}")
    if paired is not None:
        print(f"Loaded {paired_path}")
        print(f"  paired scenarios: {list(paired.get('results_by_scenario', {}).keys())}")

    if args.only in ("all", "fig2"):
        print("\nFigure 2: paired rollout snapshot")
        figure2_paired_rollout(diag, out_dir, episode_idx=args.episode_idx)

    if args.only in ("all", "fig3"):
        print("\nFigure 3: outcome summary")
        if paired is None:
            print("  --paired-eval is required for Figure 3")
        else:
            figure3_outcome_summary(paired, out_dir)

    if args.only in ("all", "fig4"):
        print("\nFigure 4: paired per-seed transitions")
        if paired is None:
            print("  --paired-eval is required for Figure 4")
        else:
            figure4_paired_transitions(paired, out_dir)

    print(f"\nDone. Figures in {out_dir}/")


if __name__ == "__main__":
    main()
