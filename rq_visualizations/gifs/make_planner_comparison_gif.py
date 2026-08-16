#!/usr/bin/env python3
"""Four-way RELLIS-Dyn GIF: DWA, MPPI, budgeted MPC, and our field."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

METHODS = ["dwa_semantic", "mppi_semantic", "mpc_budget", "route_aware_stage2"]
META = {
    "dwa_semantic": ("Semantic DWA", "#377eb8", "reactive velocity search"),
    "mppi_semantic": ("Semantic MPPI", "#ff8c42", "sampled trajectory optimization"),
    "mpc_budget": ("Budgeted MPC", "#8e5bb7", "receding-horizon replanning"),
    "route_aware_stage2": ("Material-aware (ours)", "#18a568", "gated field; zero replans"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, default=Path("/mnt/data/adityas/GRL-SNAM/exp-rellis"))
    ap.add_argument("--episode", default="7")
    ap.add_argument("--event", default="moving_obstacle_blocks_detour")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "behavioral/planner_comparison.gif")
    ap.add_argument("--stride", type=int, default=2)
    ns = ap.parse_args()
    sys.path.insert(0, str(ns.source_root))
    import make_rellis_dyn_artifacts as api

    args = argparse.Namespace(
        bev_root=ns.source_root / "cache/rellis_bev_all_seqbalanced_2500",
        pairs_root=ns.source_root / "cache/rellis_pairs_all_seqbalanced_2500_loso",
        event_fraction=0.38, event_duration=80, max_steps=140, replan_period=8,
        risk_weight=18.0, hard_margin_m=1.0, route_horizon=18,
        improvement_margin=0.25,
    )
    ep, _, paths, maps, spec = api._roll_case(args, ns.episode, ns.event, METHODS)
    goal = np.asarray(ep["goal_rc"], dtype=float)
    all_rc = np.asarray([q for p in paths.values() for q in p], dtype=float)
    r0, c0 = np.min(all_rc, axis=0) - 7
    r1, c1 = np.max(all_rc, axis=0) + 7
    n = max(len(p) for p in paths.values())
    tids = list(range(0, n, max(1, ns.stride)))
    if tids[-1] != n - 1: tids.append(n - 1)
    frames = []
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    for fi, t in enumerate(tids):
        fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for ax, method in zip(axs.flat, METHODS):
            name, color, family = META[method]
            mt = min(t, len(maps[method]) - 1)
            dyn = maps[method][mt]
            ax.imshow(dyn["risk_map"], cmap="magma", vmin=0, vmax=1)
            ax.contour(dyn["hard_mask"], levels=[0.5], colors="white", linewidths=.75)
            p = paths[method]; upto = min(t + 1, len(p)); pp = np.asarray(p[:upto])
            ax.plot(pp[:, 1], pp[:, 0], color=color, lw=3)
            pos = np.asarray(p[min(t, len(p)-1)])
            ax.scatter([pos[1]], [pos[0]], s=65, c=color, edgecolors="white", zorder=8)
            ax.scatter([goal[1]], [goal[0]], marker="*", s=120, c="#00f5d4", edgecolors="black", zorder=8)
            # Evaluation uses a 3 m goal tolerance; RELLIS BEV resolution is
            # 0.5 m/cell, hence a six-cell endpoint tolerance here.
            reached = np.linalg.norm(np.asarray(p[-1], float)-goal) <= 6.0
            outcome = "SUCCESS" if reached else ("STUCK" if len(p) >= args.max_steps else "INCOMPLETE")
            if method == "mpc_budget":
                workload = f"explicit replans={max(1, (len(p)-1)//args.replan_period)}"
            elif method == "mppi_semantic":
                workload = "trajectory sampling every step"
            elif method == "dwa_semantic":
                workload = "velocity search every step"
            else:
                workload = "explicit replans=0"
            ax.set_title(f"{name} — {outcome}\n{family}; {workload}", fontsize=10, weight="bold")
            ax.set_xlim(c0, c1); ax.set_ylim(r1, r0); ax.set_xticks([]); ax.set_yticks([])
        phase = "before event" if t < spec.event_step else "moving obstacle active"
        fig.suptitle(f"Same local BEV and event • {phase} • step {t}\n"
                     "dark=lower soft risk; white=hard boundary; star=goal",
                     fontsize=13, weight="bold")
        png = ns.out.parent / f".planner_{fi:04d}.png"
        fig.savefig(png, dpi=105); plt.close(fig); frames.append(png)
    ims = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frames]
    ims[0].save(ns.out, save_all=True, append_images=ims[1:], duration=130,
                loop=0, disposal=2, optimize=False)
    for im in ims: im.close()
    for p in frames: p.unlink()
    print(ns.out)
    print({m: {"steps": len(paths[m]), "endpoint_goal_distance_cells": float(np.linalg.norm(np.asarray(paths[m][-1], float)-goal))} for m in METHODS})

if __name__ == "__main__":
    main()
