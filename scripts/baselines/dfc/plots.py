from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from scripts.build_dfc2018_stagewise import CLASS_COLORS
from scripts.baselines.dfc.metrics import cumulative_risk_curve, cumulative_risk_curve_trace


PLANNER_COLORS = {
    "blind_dijkstra": "#00e5ff",
    "geometry_astar": "#a8ff78",
    "risk_weighted_astar": "#ff6b00",
    "oracle_astar": "#cc00ff",
    "cvar_costmap_astar": "#f6bd16",
    "chance_constrained_mpc": "#1890ff",
    "ppo_lagrangian": "#13c2c2",
    "s1_model": "#7cb305",
    "s2_model": "#722ed1",
    "s2_model_guarded": "#eb2f96",
}

PLANNER_LABELS = {
    "blind_dijkstra": "Blind Dijkstra",
    "geometry_astar": "Geometry-only A*",
    "risk_weighted_astar": "Risk-weighted A*",
    "oracle_astar": "Oracle A*",
    "cvar_costmap_astar": "CVaR Costmap A*",
    "chance_constrained_mpc": "Chance-constrained MPC",
    "ppo_lagrangian": "PPO-Lagrangian",
    "s1_model": "Ours S1",
    "s2_model": "Ours S2",
    "s2_model_guarded": "Ours S2 + Guard",
}


def _make_cmap():
    return ListedColormap(CLASS_COLORS), BoundaryNorm(range(22), 21)


def _plot_trace(ax, trace_rc, color, *, lw=2.0, ls="-", label=None):
    if trace_rc is None or len(trace_rc) == 0:
        return
    rs = [float(p[0]) for p in trace_rc]
    cs = [float(p[1]) for p in trace_rc]
    ax.plot(
        cs, rs, color=color, lw=lw, ls=ls, label=label,
        path_effects=[pe.Stroke(linewidth=lw + 1.5, foreground="black", alpha=0.45), pe.Normal()],
    )


def save_episode_overview(
    out_path: Path,
    maps: Dict[str, np.ndarray],
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    planner_paths: Mapping[str, Optional[List[Tuple[int, int]]]],
):
    cmap, norm = _make_cmap()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    axes[0].imshow(maps["z2_labels"], cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].set_title("Labels")

    im = axes[1].imshow(maps["risk_map"], cmap="magma", interpolation="nearest")
    axes[1].set_title("Risk field")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(maps["z2_labels"], cmap=cmap, norm=norm, interpolation="nearest", alpha=0.7)
    axes[2].imshow(np.ma.masked_where(maps["hard_mask"] == 0, maps["hard_mask"]), cmap="Reds", alpha=0.35)
    axes[2].set_title("Planner overlays")

    for ax in axes:
        ax.scatter(start_rc[1], start_rc[0], marker="o", s=40, c="white", edgecolors="black")
        ax.scatter(goal_rc[1], goal_rc[0], marker="*", s=80, c="yellow", edgecolors="black")
        ax.set_xticks([])
        ax.set_yticks([])

    for name, path in planner_paths.items():
        _plot_trace(axes[2], path, PLANNER_COLORS[name], label=PLANNER_LABELS[name], ls="--" if name == "blind_dijkstra" else "-")
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(loc="lower right", fontsize=8, frameon=True)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_episode_cumrisk(
    out_path: Path,
    maps: Dict[str, np.ndarray],
    planner_paths: Mapping[str, Optional[List[Tuple[int, int]]]],
    *,
    gsd: float,
):
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for name, path in planner_paths.items():
        if path is None:
            ds, cr = np.array([]), np.array([])
        else:
            is_grid = isinstance(path, list) and path and isinstance(path[0][0], (int, np.integer))
            ds, cr = (
                cumulative_risk_curve(path, maps, gsd=gsd)
                if is_grid else
                cumulative_risk_curve_trace(path, maps, gsd=gsd)
            )
        if ds.size == 0:
            continue
        ax.plot(ds, cr, lw=2.0, color=PLANNER_COLORS[name], label=PLANNER_LABELS[name])
    ax.set_xlabel("Path length (m)")
    ax.set_ylabel("Cumulative risk exposure")
    ax.set_title("Cumulative risk along path")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_aggregate_summary(
    out_path: Path,
    planner_rows: Mapping[str, List[Dict[str, float]]],
):
    metrics = [
        ("failure_score", "Failure score"),
        ("risk_exposure", "Risk exposure"),
        ("mean_rho", "Mean risk / m"),
        ("min_hard_distance_m", "Min hard distance"),
        ("barrier_violation_m", "Barrier violation"),
        ("hard_hazard_length_m", "Hard hazard length"),
        ("path_length_ratio", "Path length ratio"),
        ("oscillation", "Oscillation"),
    ]
    planners = list(planner_rows.keys())
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, metrics):
        means = []
        cis = []
        for planner in planners:
            vals = np.asarray([
                row[key] for row in planner_rows[planner]
                if np.isfinite(row[key])
            ], dtype=np.float64)
            if vals.size == 0:
                means.append(np.nan)
                cis.append(0.0)
            else:
                means.append(float(vals.mean()))
                cis.append(float(1.96 * vals.std(ddof=0) / max(np.sqrt(vals.size), 1.0)))
        x = np.arange(len(planners))
        ax.bar(x, means, yerr=cis, color=[PLANNER_COLORS[p] for p in planners], alpha=0.9, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([PLANNER_LABELS[p] for p in planners], rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.2)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_pareto_plot(
    out_path: Path,
    planner_rows: Mapping[str, List[Dict[str, float]]],
    *,
    y_metric: str = "risk_exposure",
    y_label: str = "Cumulative risk exposure",
):
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for planner, rows in planner_rows.items():
        pairs = [
            (float(row["path_length_ratio"]), float(row[y_metric]))
            for row in rows
            if np.isfinite(row.get("path_length_ratio", np.nan)) and np.isfinite(row.get(y_metric, np.nan))
        ]
        if not pairs:
            continue
        x_vals = np.asarray([p[0] for p in pairs], dtype=np.float64)
        y_vals = np.asarray([p[1] for p in pairs], dtype=np.float64)
        x = float(x_vals.mean())
        y = float(y_vals.mean())
        x_ci = float(1.96 * x_vals.std(ddof=0) / max(np.sqrt(x_vals.size), 1.0))
        y_ci = float(1.96 * y_vals.std(ddof=0) / max(np.sqrt(y_vals.size), 1.0))
        ax.errorbar(
            x,
            y,
            xerr=x_ci,
            yerr=y_ci,
            fmt="o",
            ms=8,
            capsize=4,
            color=PLANNER_COLORS[planner],
            label=PLANNER_LABELS[planner],
        )
        ax.annotate(
            PLANNER_LABELS[planner],
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(1.0, color="black", lw=1, alpha=0.25)
    ax.set_xlabel("Path length ratio")
    ax.set_ylabel(y_label)
    ax.set_title("Risk-efficiency Pareto summary")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, loc="best")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
