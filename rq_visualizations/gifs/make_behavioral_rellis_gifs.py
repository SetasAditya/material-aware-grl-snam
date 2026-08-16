#!/usr/bin/env python3
"""Render behaviorally distinct RELLIS-Dyn policy rollouts as paired GIFs.

Unlike the coefficient/gate diagnostic animations, these frames replay the
closed-loop trajectories of four policies.  The risk map and hard-hazard mask
are recomputed at every step, so the animation shows both the changing scene
and the resulting policy decisions.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


METHODS = ["stage1", "dwa_semantic", "cbf_safety_filter", "route_aware_stage2"]
META = {
    "stage1": ("Geometry scaffold", "#8c8c8c"),
    "dwa_semantic": ("DWA + semantic cost", "#377eb8"),
    "cbf_safety_filter": ("CBF-QP safety filter", "#984ea3"),
    "route_aware_stage2": ("Material-aware (ours)", "#20a464"),
}


def _csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _select_cases(run: Path, n: int):
    rows = _csv(run / "dynamic_rollouts.csv")
    by_ep = defaultdict(dict)
    for row in rows:
        if row["event_type"] == "delayed_required_escape":
            by_ep[row["episode_id"]][row["method"]] = row
    ranked = []
    for eid, m in by_ep.items():
        if not set(METHODS).issubset(m):
            continue
        ours, dwa, bb = m["route_aware_stage2"], m["dwa_semantic"], m["cbf_safety_filter"]
        if float(ours["success"]) < 0.5:
            continue
        def false_pre(row):
            return float(row["route_deviation_delay"]) < 10.0
        score = (
            12 * (1 - float(dwa["success"]))
            + 12 * (1 - float(bb["success"]))
            + 7 * false_pre(dwa) + 7 * false_pre(bb)
            + 5 * float(bb["stuck"])
            + 3 * (float(dwa["stale_exposure"]) - float(ours["stale_exposure"]))
            + 3 * (float(bb["stale_exposure"]) - float(ours["stale_exposure"]))
        )
        ranked.append((score, eid, m))
    ranked.sort(reverse=True)
    return ranked[:n]


def _pad(path, t):
    return path[min(t, len(path) - 1)]


def render_case(api, args, eid: str, rows, rank: int):
    ep, _, paths, step_maps, spec = api._roll_case(
        args, eid, "delayed_required_escape", METHODS)
    t_open = int(spec.event_step + spec.open_delay)
    max_t = max(len(p) for p in paths.values())
    stride = max(1, int(args.stride))
    indices = list(range(0, max_t, stride))
    if indices[-1] != max_t - 1:
        indices.append(max_t - 1)

    all_rc = np.asarray([q for p in paths.values() for q in p], dtype=float)
    r0, c0 = np.min(all_rc, axis=0) - 7
    r1, c1 = np.max(all_rc, axis=0) + 7
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = []

    for frame_i, t in enumerate(indices):
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.1), constrained_layout=True)
        for ax, method in zip(axes.flat, METHODS):
            label, color = META[method]
            mt = min(t, len(step_maps[method]) - 1)
            dyn = step_maps[method][mt]
            ax.imshow(dyn["risk_map"], cmap="magma", vmin=0, vmax=1)
            ax.contour(dyn["hard_mask"], levels=[0.5], colors="white", linewidths=0.75)
            path = paths[method]
            upto = min(t + 1, len(path))
            pp = np.asarray(path[:upto])
            ax.plot(pp[:, 1], pp[:, 0], color=color, lw=3.0, zorder=5)
            rr, cc = _pad(path, t)
            ax.scatter([cc], [rr], s=62, c=color, edgecolors="white", linewidths=1.2, zorder=7)
            ax.scatter([ep["start_rc"][1]], [ep["start_rc"][0]], s=30, c="white", edgecolors="black", zorder=7)
            ax.scatter([ep["goal_rc"][1]], [ep["goal_rc"][0]], marker="*", s=115,
                       c="#00f5d4", edgecolors="black", zorder=7)
            row = rows[method]
            outcome = "success" if float(row["success"]) > 0.5 else ("stuck" if float(row["stuck"]) > 0.5 else "failed")
            phase = "escape blocked" if t < t_open else "escape open"
            ax.set_title(f"{label}  |  {outcome}\n{phase}; step {t}", fontsize=10, weight="bold")
            ax.set_xlim(c0, c1)
            ax.set_ylim(r1, r0)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(
            "Delayed-required escape: suppress premature detours, then commit when feasible\n"
            "dark = lower soft risk; white boundary = hard hazard; star = goal",
            fontsize=13, weight="bold")
        png = out_dir / f".rellis_case{rank}_{frame_i:04d}.png"
        fig.savefig(png, dpi=105)
        plt.close(fig)
        pngs.append(png)

    images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in pngs]
    gif = out_dir / f"rellis_delayed_escape_example_{rank}.gif"
    images[0].save(gif, save_all=True, append_images=images[1:], duration=args.duration,
                   loop=0, optimize=False, disposal=2)
    for im in images:
        im.close()
    for p in pngs:
        p.unlink()
    return gif, t_open, {m: len(paths[m]) for m in METHODS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, default=Path("/mnt/data/adityas/GRL-SNAM/exp-rellis"))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "behavioral")
    ap.add_argument("--examples", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--duration", type=int, default=130)
    ns = ap.parse_args()

    sys.path.insert(0, str(ns.source_root))
    import make_rellis_dyn_artifacts as api

    args = argparse.Namespace(
        bev_root=ns.source_root / "cache/rellis_bev_all_seqbalanced_2500",
        pairs_root=ns.source_root / "cache/rellis_pairs_all_seqbalanced_2500_loso",
        fast_run=ns.source_root / "runs/rellis_dyn_8events_fast_100",
        planner_run=ns.source_root / "runs/rellis_dyn_8events_planners_20",
        delayed_required_run=ns.source_root / "runs/rellis_dyn_delayed_required_100_v3",
        event_fraction=0.38, event_duration=80, max_steps=140, replan_period=8,
        risk_weight=18.0, hard_margin_m=1.0, route_horizon=18,
        improvement_margin=0.25, out=ns.out, stride=ns.stride,
        duration=ns.duration,
    )
    selected = _select_cases(args.delayed_required_run, ns.examples)
    if not selected:
        raise SystemExit("No qualifying delayed-required-escape episodes found")
    for rank, (_, eid, rows) in enumerate(selected, 1):
        gif, t_open, lengths = render_case(api, args, eid, rows, rank)
        print(f"{gif}: episode={eid}, escape_step={t_open}, lengths={lengths}")


if __name__ == "__main__":
    main()
