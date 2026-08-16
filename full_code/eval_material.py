#!/usr/bin/env python3
"""
eval_material.py

Evaluate Setting 1 (geometry-only) and Setting 2 (material-aware) trained
models on DFC2018 test episodes.  Produces the same visualisations and
metrics as grss_risk_path_v2.py, extended with model-based trajectories.

Four planners compared
----------------------
  blind    : Dijkstra, uniform cost, no material knowledge  (S1 reference floor)
  oracle   : A* with oracle material map                    (S2 reference ceiling)
  s1_model : CoefEnergyNetMaterial, stage=1 checkpoint      (geometry-only model)
  s2_model : CoefEnergyNetMaterial, stage=2 checkpoint      (material-aware model)

Per-episode outputs (in --out/<episode_id>/)
  overview.png      — 4-column: labels | risk | SDF | hazard overlay, all 4 paths
  cumrisk.png       — cumulative risk + instantaneous risk profile
  metrics.json      — all numeric metrics for this episode

Aggregate outputs (in --out/)
  aggregate.csv     — one row per episode per planner
  summary.png       — bar chart across planners (mean ± std)
  summary.json      — aggregate statistics

Usage
-----
# Minimal (both model checkpoints required for S1/S2 comparison)
python eval_material.py \\
    --root   data/dfc2018_stagewise \\
    --ckpt_s1 checkpoints/s1/best.pt \\
    --ckpt_s2 checkpoints/s2/best.pt \\
    --out    eval_output/

# Evaluate only oracle vs blind (no models needed)
python eval_material.py \\
    --root   data/dfc2018_stagewise \\
    --out    eval_output/ \\
    --no_models

# Limit to first N test episodes
python eval_material.py \\
    --root data/dfc2018_stagewise \\
    --ckpt_s1 checkpoints/s1/best.pt \\
    --ckpt_s2 checkpoints/s2/best.pt \\
    --out eval_output/ --max_episodes 20
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────────────────────────────────────
# Import model and integrator from train_material.py
# ─────────────────────────────────────────────────────────────────────────────
try:
    from train_material import (
        CoefEnergyNetMaterial,
        bilinear_sample_patch,
        integrate_surrogate_material,
    )
    from train_coef_energy import ipc_piecewise
except ImportError as e:
    raise ImportError(f"train_material.py / train_coef_energy.py must be on path: {e}")

try:
    from scripts.build_dfc2018_stagewise import (
        extract_local_geom_obstacles,
        extract_risk_patch,
        extract_rollout_patch,
        HARD_CLASSES, SOFT_CLASSES, CLASS_NAMES, CLASS_COLORS,
        RHO, GT_GSD,
    )
except ImportError as e:
    raise ImportError(f"scripts/build_dfc2018_stagewise.py must be on path: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# Paper-reported rare catastrophic set (subset of HARD_CLASSES used for metrics)
HAZARD_PAPER = frozenset({7, 14, 15})   # water, highways, railways

RISK_WEIGHT   = 10.0
DIAGONALS     = True
_DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
_STEP = {(dr,dc): 1.4142 if abs(dr)+abs(dc)==2 else 1.0 for dr,dc in _DIRS}

PLANNER_COLORS = {
    "blind":   "#00e5ff",
    "oracle":  "#ff6b00",
    "s1_model":"#a8ff78",
    "s2_model":"#cc00ff",
}
PLANNER_LABELS = {
    "blind":   "S1: Blind Dijkstra",
    "oracle":  "S2: Oracle A*",
    "s1_model":"S1 Model (geom-only)",
    "s2_model":"S2 Model (material-aware)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Grid planners  (reference: blind + oracle)
# ─────────────────────────────────────────────────────────────────────────────

def dijkstra_blind(maps: Dict, start: Tuple[int,int],
                   goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    H,W = maps["z2_labels"].shape
    geom = maps["geom_occ"].astype(bool)
    dist = {start: 0.0}; prev = {start: None}
    heap = [(0.0, start)]
    while heap:
        d,u = heapq.heappop(heap)
        if u == goal:
            path=[]; node=goal
            while node: path.append(node); node=prev[node]
            return path[::-1]
        if d > dist.get(u,1e18)+1e-9: continue
        r,c = u
        for dr,dc in _DIRS:
            nr,nc = r+dr,c+dc
            if not(0<=nr<H and 0<=nc<W): continue
            if geom[nr,nc]: continue
            nd = d+_STEP[(dr,dc)]
            if nd < dist.get((nr,nc),1e18):
                dist[(nr,nc)]=nd; prev[(nr,nc)]=u
                heapq.heappush(heap,(nd,(nr,nc)))
    return None


def astar_oracle(maps: Dict, start: Tuple[int,int],
                 goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    H,W    = maps["z2_labels"].shape
    risk   = maps["risk_map"]
    hard   = (maps["hard_mask"]).astype(bool)
    gr,gc  = goal

    def h(r,c):
        dr,dc=abs(r-gr),abs(c-gc)
        return max(dr,dc)+(1.4142-1)*min(dr,dc)
    def cost(r,c):
        if not(0<=r<H and 0<=c<W): return 1e9
        if hard[r,c]: return 1e9
        return 1.0 + RISK_WEIGHT*float(risk[r,c])

    gs={start:0.0}; came={start:None}
    heap=[(h(*start),0.0,start[0],start[1])]
    while heap:
        _,g,r,c=heapq.heappop(heap)
        if(r,c)==goal:
            path=[]; node=goal
            while node: path.append(node); node=came[node]
            return path[::-1]
        if g>gs.get((r,c),1e18)+1e-9: continue
        for dr,dc in _DIRS:
            nr,nc=r+dr,c+dc; cc=cost(nr,nc)
            if cc>=1e8: continue
            ng=g+_STEP[(dr,dc)]*cc
            if ng<gs.get((nr,nc),1e18):
                gs[(nr,nc)]=ng; came[(nr,nc)]=(r,c)
                heapq.heappush(heap,(ng+h(nr,nc),ng,nr,nc))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics on continuous OR grid paths
# ─────────────────────────────────────────────────────────────────────────────

def path_metrics_grid(
    path: List[Tuple[int,int]],
    maps: Dict,
    gsd: float = GT_GSD,
) -> Dict[str,float]:
    """Metrics for a grid path (list of (row,col) tuples)."""
    if not path: return _empty_metrics()
    risk_map  = maps["risk_map"]
    hard      = maps["hard_mask"].astype(bool)
    paper_haz = np.isin(maps["z2_labels"], list(HAZARD_PAPER))
    H,W       = risk_map.shape

    length=0.; risk_acc=0.; hard_hits=0; paper_hits=0
    for i in range(len(path)-1):
        r0,c0=path[i]; r1,c1=path[i+1]
        step=gsd*_STEP[(r1-r0,c1-c0)]
        length+=step
        risk_acc+=step*float(risk_map[r1,c1])
        if hard[r1,c1]:     hard_hits+=1
        if paper_haz[r1,c1]: paper_hits+=1
    return dict(length_m=float(length), risk_exposure=float(risk_acc),
                hard_hits=int(hard_hits), paper_hard_hits=int(paper_hits),
                mean_rho=float(risk_acc/max(length,1e-6)))


def path_metrics_continuous(
    traj_xy: np.ndarray,   # (T,2)  continuous (col,row) positions
    maps: Dict,
    gsd: float = GT_GSD,
) -> Dict[str,float]:
    """Metrics for a continuous trajectory sampled at pixel resolution."""
    if len(traj_xy) < 2: return _empty_metrics()
    H,W      = maps["risk_map"].shape
    risk_map = maps["risk_map"]
    hard     = maps["hard_mask"].astype(bool)
    paper    = np.isin(maps["z2_labels"], list(HAZARD_PAPER))

    length=0.; risk_acc=0.; hard_hits=0; paper_hits=0
    for i in range(len(traj_xy)-1):
        x0,y0=traj_xy[i];   x1,y1=traj_xy[i+1]
        step=gsd*float(np.linalg.norm([x1-x0,y1-y0]))
        length+=step
        # sample at midpoint
        mx,my=int(np.clip((x0+x1)/2,0,W-1)), int(np.clip((y0+y1)/2,0,H-1))
        risk_acc+=step*float(risk_map[my,mx])
        if hard[my,mx]:  hard_hits+=1
        if paper[my,mx]: paper_hits+=1
    return dict(length_m=float(length), risk_exposure=float(risk_acc),
                hard_hits=int(hard_hits), paper_hard_hits=int(paper_hits),
                mean_rho=float(risk_acc/max(length,1e-6)))


def _empty_metrics():
    return dict(length_m=0.,risk_exposure=0.,hard_hits=0,paper_hard_hits=0,mean_rho=0.)


def cumulative_risk_grid(path, maps, gsd=GT_GSD):
    if not path: return np.array([]),np.array([])
    risk=maps["risk_map"]
    ds=[0.]; cr=[0.]
    for i in range(len(path)-1):
        r0,c0=path[i]; r1,c1=path[i+1]
        step=gsd*_STEP[(r1-r0,c1-c0)]
        ds.append(ds[-1]+step)
        cr.append(cr[-1]+step*float(risk[r1,c1]))
    return np.array(ds),np.array(cr)


def cumulative_risk_continuous(traj_xy, maps, gsd=GT_GSD):
    if len(traj_xy)<2: return np.array([]),np.array([])
    H,W=maps["risk_map"].shape; risk=maps["risk_map"]
    ds=[0.]; cr=[0.]
    for i in range(len(traj_xy)-1):
        x0,y0=traj_xy[i]; x1,y1=traj_xy[i+1]
        step=gsd*float(np.linalg.norm([x1-x0,y1-y0]))
        mx=int(np.clip((x0+x1)/2,0,W-1)); my=int(np.clip((y0+y1)/2,0,H-1))
        ds.append(ds[-1]+step)
        cr.append(cr[-1]+step*float(risk[my,mx]))
    return np.array(ds),np.array(cr)


def astar_geom_only(maps: Dict, start: Tuple[int,int],
                    goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    """
    A* on geometry occupancy only (buildings impassable, no material cost).
    Used in end-to-end eval mode to derive model stage waypoints without
    oracle material knowledge — gives both S1 and S2 models identical,
    fair waypoint scaffolding.
    """
    H,W  = maps["z2_labels"].shape
    geom = maps["geom_occ"].astype(bool)
    gr,gc = goal

    def h(r,c):
        dr,dc=abs(r-gr),abs(c-gc)
        return max(dr,dc)+(1.4142-1)*min(dr,dc)

    gs={start:0.0}; came={start:None}
    heap=[(h(*start),0.0,start[0],start[1])]
    while heap:
        _,g,r,c=heapq.heappop(heap)
        if(r,c)==goal:
            path=[]; node=goal
            while node: path.append(node); node=came[node]
            return path[::-1]
        if g>gs.get((r,c),1e18)+1e-9: continue
        for dr,dc in _DIRS:
            nr,nc=r+dr,c+dc
            if not(0<=nr<H and 0<=nc<W): continue
            if geom[nr,nc]: continue
            ng=g+_STEP[(dr,dc)]
            if ng<gs.get((nr,nc),1e18):
                gs[(nr,nc)]=ng; came[(nr,nc)]=(r,c)
                heapq.heappush(heap,(ng+h(nr,nc),ng,nr,nc))
    return None


def build_geom_waypoints(
    path_rc: List[Tuple[int,int]],
    stride: int = 6,
    max_stages: int = 256,
    patch_size: int = 64,
) -> List[Tuple[float,float]]:
    """
    Downsample a grid path to stage exit waypoints in (col,row)=(x,y) format.
    Mirrors the logic in build_dfc2018_stagewise.py so the model sees the
    same type of local goals it was trained on.
    """
    if not path_rc: return []
    T    = len(path_rc)
    idxs = list(range(0, T, stride))
    if idxs[-1] != T-1: idxs.append(T-1)
    if len(idxs) > max_stages:
        sel  = np.linspace(0, len(idxs)-1, max_stages, dtype=int)
        idxs = [idxs[i] for i in sel]

    # For each centre index, look ahead to near-perimeter of patch
    path_rc_np = np.array(path_rc, dtype=np.float32)
    waypoints  = []
    r_max = patch_size / 2.0 - 2.0
    for ci in idxs:
        last_good = ci
        for j in range(ci+3, min(ci+64, T)):
            delta = path_rc_np[j] - path_rc_np[ci]
            if float(np.linalg.norm(delta)) <= r_max:
                last_good = j
            else:
                break
        if last_good == ci:
            last_good = min(ci+3, T-1)
        er, ec = int(path_rc_np[last_good,0]), int(path_rc_np[last_good,1])
        waypoints.append((float(ec), float(er)))   # (col=x, row=y)
    return waypoints

def load_model(ckpt_path: str, device: str, patch_size: int = 32,
               stage: int = 2) -> CoefEnergyNetMaterial:
    ck  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {})
    model = CoefEnergyNetMaterial(
        patch_size   = cfg.get("patch_size",   patch_size),
        lam_soft_max = cfg.get("lam_soft_max", 5.0),
        lam_hard_max = cfg.get("lam_hard_max", 10.0),
    )
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    print(f"  Loaded {'S1' if stage==1 else 'S2'} model from {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Stagewise model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _build_obs_feats(pos_xy, goal_xy, C, R, W_arr, device):
    """Build (1,N,6) obstacle feature tensor matching CoefEnergyNetMaterial input."""
    N = C.shape[0]
    if N == 0:
        return torch.zeros(1, 0, 6, device=device)
    C_t  = torch.tensor(C, dtype=torch.float32, device=device)
    R_t  = torch.tensor(R, dtype=torch.float32, device=device)
    W_t  = torch.tensor(W_arr, dtype=torch.float32, device=device)
    g_t  = torch.tensor(goal_xy, dtype=torch.float32, device=device)
    dxdy = g_t.unsqueeze(0) - C_t    # (N,2)
    return torch.cat([C_t, R_t.unsqueeze(-1), W_t.unsqueeze(-1), dxdy],
                      dim=-1).unsqueeze(0)   # (1,N,6)


def _build_goal_feats(pos_xy, goal_xy, device):
    o = torch.tensor(pos_xy,  dtype=torch.float32, device=device)
    g = torch.tensor(goal_xy, dtype=torch.float32, device=device)
    dg   = g - o
    dist = torch.linalg.norm(dg).unsqueeze(0)
    return torch.cat([dg, dist, torch.ones(1, device=device)]).unsqueeze(0)  # (1,4)


@torch.no_grad()
def run_model_episode(
    model: CoefEnergyNetMaterial,
    maps: Dict,
    waypoints_xy: List[Tuple[float,float]],  # (col,row) stage goals — source depends on eval_mode
    ck_d_hats: List[float],                  # per-stage d_hat values from checkpoints
    ck_dts: List[float],                     # per-stage dt values from checkpoints
    start_rc: Tuple[int,int],
    goal_rc:  Tuple[int,int],
    device: str,
    stage: int      = 2,
    steps_per_stage: int = 80,
    robot_radius: float  = 1.5,
    margin_factor: float = 0.5,
    patch_size: int = 32,
    d_hat_sdf: float= 3.0,
    goal_tol: float = 3.0,
) -> np.ndarray:
    """
    Run a full episode using stagewise CoefEnergyNetMaterial inference.

    `waypoints_xy` drives the per-stage goals:
      - stagewise mode  → comes from checkpoint stage_exit fields
        (follows oracle dataset waypoints)
      - endtoend mode   → comes from build_geom_waypoints() on geom-only A*
        (fair: both S1 and S2 get identical geometry-derived goals; material
         learning expresses itself in λ_soft/λ_hard forces deflecting around
         hazards, not in which waypoints are chosen)

    dt and d_hat are read per-stage from checkpoint metadata, not hardcoded.
    """
    H, W     = maps["z2_labels"].shape
    pos      = np.array([float(start_rc[1]), float(start_rc[0])], dtype=np.float32)
    vel      = np.zeros(2, dtype=np.float32)
    traj     = [pos.copy()]
    geom_occ = maps["geom_occ"]

    # Zip waypoints with per-stage metadata; pad if lengths differ
    n_stages = len(waypoints_xy)

    for si in range(n_stages):
        stage_goal_xy = np.array(waypoints_xy[si], dtype=np.float32)
        dt     = float(ck_dts[si])    if si < len(ck_dts)    else 0.04
        d_hat_v= float(ck_d_hats[si]) if si < len(ck_d_hats) else 3.0

        cr, cc = int(np.clip(pos[1],0,H-1)), int(np.clip(pos[0],0,W-1))
        C, R_eff, W_arr, _ = extract_local_geom_obstacles(
            geom_occ, (cr, cc), patch_size=64,
            robot_radius=robot_radius, margin_factor=margin_factor)

        risk_patch_np, _ = extract_risk_patch(maps, (cr,cc), patch_size)

        obs_feats  = _build_obs_feats(pos, stage_goal_xy, C, R_eff, W_arr, device)
        obs_mask   = (torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)
                      if obs_feats.shape[1] > 0
                      else torch.zeros(1, 0, dtype=torch.bool, device=device))
        goal_feats = _build_goal_feats(pos, stage_goal_xy, device)
        risk_patch = torch.tensor(risk_patch_np, dtype=torch.float32,
                                   device=device).unsqueeze(0)

        alphas, beta, gamma, lam_soft, lam_hard = model(
            obs_feats, obs_mask, goal_feats, risk_patch)

        if stage == 1:
            lam_soft = torch.zeros_like(lam_soft)
            lam_hard = torch.zeros_like(lam_hard)

        def t1d(x):
            return torch.tensor([x], dtype=torch.float32, device=device)
        def t2d(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

        C_t   = torch.tensor(C,     dtype=torch.float32, device=device).unsqueeze(0)
        R_t   = torch.tensor(R_eff, dtype=torch.float32, device=device).unsqueeze(0)
        mask  = (torch.ones(1, C.shape[0], dtype=torch.bool, device=device)
                 if C.shape[0] > 0
                 else torch.zeros(1, 0, dtype=torch.bool, device=device))
        goal_t= t2d(stage_goal_xy)
        dt_t  = t1d(dt)
        d_hat_t=t1d(d_hat_v)
        rr_t  = t1d(robot_radius)

        o = t2d(pos); v = t2d(vel)
        stage_traj = []
        for s in range(steps_per_stage):
            cr_s = int(np.clip(o[0,1].item(), 0, H-1))
            cc_s = int(np.clip(o[0,0].item(), 0, W-1))
            rp_np = np.asarray(
                extract_rollout_patch(maps, (cr_s, cc_s), patch_size),
                dtype=np.float32,
            )
            rp_s = torch.as_tensor(rp_np, dtype=torch.float32, device=device).unsqueeze(0)

            oN, vN, _, _, _, _ = integrate_surrogate_material(
                o0=o.clone(), v0=v.clone(), goal=goal_t,
                C=C_t, R=R_t, mask=mask,
                alphas=alphas, beta=beta, gamma=gamma,
                lam_soft=lam_soft, lam_hard=lam_hard,
                rollout_patch=rp_s,
                d_hat=d_hat_t, dt=dt_t,
                H=torch.tensor([1], dtype=torch.long, device=device),
                robot_radius=rr_t,
                margin_factor=margin_factor,
                d_hat_sdf=d_hat_sdf,
            )
            o = oN; v = vN
            pos_s = o[0].cpu().numpy()
            stage_traj.append(pos_s.copy())
            if np.linalg.norm(pos_s - stage_goal_xy) < goal_tol:
                break

        traj.extend(stage_traj)
        pos = o[0].cpu().numpy()
        vel = v[0].cpu().numpy()

        goal_xy = np.array([float(goal_rc[1]), float(goal_rc[0])], dtype=np.float32)
        if np.linalg.norm(pos - goal_xy) < goal_tol * 2:
            break

    return np.array(traj, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers  (same palette as grss_risk_path_v2.py)
# ─────────────────────────────────────────────────────────────────────────────

def _make_cmap():
    return ListedColormap(CLASS_COLORS), BoundaryNorm(range(22), 21)


def _plot_grid_path(ax, path, color, lw=2.2, alpha=1.0, ls="-", label=None):
    if not path: return
    rs=[p[0] for p in path]; cs=[p[1] for p in path]
    ax.plot(cs, rs, color=color, lw=lw, alpha=alpha, ls=ls, label=label,
            path_effects=[pe.Stroke(linewidth=lw+1.8, foreground="black",
                                     alpha=alpha*0.55), pe.Normal()])


def _plot_cont_path(ax, traj_xy, color, lw=2.2, alpha=1.0, ls="-", label=None):
    """traj_xy: (T,2) col,row."""
    if traj_xy is None or len(traj_xy) < 2: return
    ax.plot(traj_xy[:,0], traj_xy[:,1], color=color, lw=lw, alpha=alpha, ls=ls,
            label=label,
            path_effects=[pe.Stroke(linewidth=lw+1.8, foreground="black",
                                     alpha=alpha*0.55), pe.Normal()])


def _marker(ax, rc, marker, color, ms=12, zorder=30):
    ax.plot(rc[1], rc[0], marker, ms=ms, color=color, zorder=zorder,
            path_effects=[pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()])


def render_overview(
    labels_crop, maps,
    paths_grid: Dict[str, Optional[List]],
    paths_cont: Dict[str, Optional[np.ndarray]],
    start_rc, goal_rc,
    episode_id: str = "",
) -> plt.Figure:
    """
    Four-panel overview.  Grid paths (blind, oracle) and continuous model
    trajectories (s1_model, s2_model) are all overlaid on the same panels.
    """
    cmap, norm = _make_cmap()
    risk_map   = maps["risk_map"]
    sdf_hard   = maps["sdf_hard"]
    hard_mask  = maps["hard_mask"].astype(bool)

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    def draw_all(ax, skip_cont=False):
        for key, path in paths_grid.items():
            if path:
                _plot_grid_path(ax, path, PLANNER_COLORS[key], lw=2.0,
                                 label=PLANNER_LABELS[key],
                                 ls="--" if key=="blind" else "-")
        if not skip_cont:
            for key, traj in paths_cont.items():
                if traj is not None and len(traj) > 1:
                    _plot_cont_path(ax, traj, PLANNER_COLORS[key], lw=2.5,
                                     label=PLANNER_LABELS[key])
        _marker(ax, start_rc, "^", "#ffffff")
        _marker(ax, goal_rc,  "*", "#ffd60a", ms=14)

    # Panel 1: GT labels
    ax = axes[0]
    ax.imshow(labels_crop, cmap=cmap, norm=norm, interpolation="nearest")
    draw_all(ax)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
    ax.set_title("GT Labels", fontweight="bold", fontsize=11)
    ax.axis("off")

    # Panel 2: Smoothed risk field
    ax = axes[1]
    im = ax.imshow(risk_map, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="bilinear")
    plt.colorbar(im, ax=ax, fraction=0.03, label="r̃(x)")
    draw_all(ax)
    ax.set_title("Smoothed Risk Field r̃(x)", fontweight="bold", fontsize=11)
    ax.axis("off")

    # Panel 3: SDF to hard hazards
    ax = axes[2]
    im = ax.imshow(sdf_hard, cmap="viridis", interpolation="bilinear")
    plt.colorbar(im, ax=ax, fraction=0.03, label="φ(x) [m]")
    draw_all(ax)
    ax.set_title("SDF to Hard Hazards φ(x)", fontweight="bold", fontsize=11)
    ax.axis("off")

    # Panel 4: Hazard overlay
    ax = axes[3]
    ax.imshow(labels_crop, cmap=cmap, norm=norm, interpolation="nearest", alpha=0.65)
    rgba_h = np.zeros((*labels_crop.shape, 4))
    rgba_h[hard_mask] = [0.9, 0.05, 0.05, 0.55]
    rgba_s = np.zeros((*labels_crop.shape, 4))
    rgba_s[np.isin(labels_crop, list(SOFT_CLASSES))] = [1.0, 0.55, 0.0, 0.35]
    ax.imshow(rgba_h, interpolation="nearest")
    ax.imshow(rgba_s, interpolation="nearest")
    draw_all(ax)
    handles = [mpatches.Patch(color=PLANNER_COLORS[k], label=PLANNER_LABELS[k])
               for k in PLANNER_COLORS]
    handles += [mpatches.Patch(color=(0.9,0.05,0.05,0.55), label="Hard hazard"),
                mpatches.Patch(color=(1.0,0.55,0.0,0.35), label="Soft hazard")]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.85)
    ax.set_title("Hazard Overlay + All Paths", fontweight="bold", fontsize=11)
    ax.axis("off")

    fig.suptitle(
        f"HAVSN Evaluation — Episode {episode_id}  |  "
        f"S1 Blind vs S2 Oracle vs S1/S2 Models",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def render_cumrisk(
    paths_grid:  Dict[str, Optional[List]],
    paths_cont:  Dict[str, Optional[np.ndarray]],
    maps: Dict,
    labels_crop: np.ndarray,
    gsd: float = GT_GSD,
) -> plt.Figure:
    """Two-panel: cumulative risk vs distance + instantaneous r̃ profile."""
    risk_map  = maps["risk_map"]
    hard_mask = maps["hard_mask"].astype(bool)
    paper_haz = np.isin(labels_crop, list(HAZARD_PAPER))
    H, W      = risk_map.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: cumulative risk
    for key, path in paths_grid.items():
        if not path: continue
        ds, cr = cumulative_risk_grid(path, maps, gsd)
        col = PLANNER_COLORS[key]
        ls  = "--" if key == "blind" else "-"
        ax1.plot(ds, cr, color=col, lw=2.2, ls=ls,
                 label=f"{PLANNER_LABELS[key]}  Σ={cr[-1]:.2f}",
                 path_effects=[pe.Stroke(linewidth=3.8, foreground="black", alpha=0.35),
                                pe.Normal()])
        ax1.fill_between(ds, cr, alpha=0.08, color=col)

    for key, traj in paths_cont.items():
        if traj is None or len(traj) < 2: continue
        ds, cr = cumulative_risk_continuous(traj, maps, gsd)
        col = PLANNER_COLORS[key]
        ax1.plot(ds, cr, color=col, lw=2.2,
                 label=f"{PLANNER_LABELS[key]}  Σ={cr[-1]:.2f}",
                 path_effects=[pe.Stroke(linewidth=3.8, foreground="black", alpha=0.35),
                                pe.Normal()])
        ax1.fill_between(ds, cr, alpha=0.08, color=col)

    ax1.set_xlabel("Distance along path (m)", fontsize=12)
    ax1.set_ylabel("Cumulative risk  Σ r̃(xₜ)·Δs", fontsize=12)
    ax1.set_title("Cumulative Risk Exposure", fontweight="bold", fontsize=12)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # Right: instantaneous r̃ profile
    for key, path in paths_grid.items():
        if not path: continue
        pct = np.linspace(0, 100, len(path))
        rho = [float(risk_map[r,c]) for r,c in path]
        col = PLANNER_COLORS[key]
        ls  = "--" if key == "blind" else "-"
        ax2.plot(pct, rho, color=col, lw=1.5, alpha=0.85, ls=ls,
                 label=PLANNER_LABELS[key])
        # shade paper hard hazard entries
        for i,(r,c) in enumerate(path):
            if paper_haz[r,c]:
                ax2.axvspan(pct[i], pct[min(i+1,len(pct)-1)],
                             color="red", alpha=0.25, lw=0)

    for key, traj in paths_cont.items():
        if traj is None or len(traj) < 2: continue
        pct = np.linspace(0, 100, len(traj))
        rho = [float(risk_map[int(np.clip(p[1],0,H-1)),
                               int(np.clip(p[0],0,W-1))]) for p in traj]
        ax2.plot(pct, rho, color=PLANNER_COLORS[key], lw=1.5, alpha=0.85,
                 label=PLANNER_LABELS[key])

    ax2.axhline(0.5, color="gray", lw=1, ls="--", label="τ=0.5")
    ax2.set_xlabel("Path progress (%)", fontsize=12)
    ax2.set_ylabel("Instantaneous r̃(x)", fontsize=12)
    ax2.set_title("Risk Profile  (red shading = paper hard-hazard entry)",
                  fontweight="bold", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3); ax2.set_ylim(0, 1.05)

    fig.suptitle("Risk Comparison Across All Planners",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def render_summary_bar(agg: Dict[str, Dict[str, List[float]]]) -> plt.Figure:
    """Aggregate bar chart — mean ± std across test episodes."""
    planners = list(PLANNER_LABELS.keys())
    metrics  = ["length_m", "risk_exposure", "paper_hard_hits", "mean_rho"]
    labels_m = ["Path length (m)", "Risk exposure\n(m·ρ)", "Paper hard\nhazard hits", "Mean ρ"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    x = np.arange(len(planners))
    w = 0.55

    for mi, (key, label_m) in enumerate(zip(metrics, labels_m)):
        ax = axes[mi]
        means = [np.mean(agg.get(p, {}).get(key, [0])) for p in planners]
        stds  = [np.std( agg.get(p, {}).get(key, [0])) for p in planners]
        bars  = ax.bar(x, means, w,
                       color=[PLANNER_COLORS[p] for p in planners],
                       edgecolor="black", linewidth=0.8,
                       yerr=stds, capsize=4, error_kw={"linewidth":1.2})
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(means)*0.01+s,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([PLANNER_LABELS[p] for p in planners],
                            fontsize=8, rotation=15, ha="right")
        ax.set_title(label_m, fontweight="bold", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Aggregate Metrics Across Test Episodes  (mean ± std)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    device = args.device

    # ── Load models ───────────────────────────────────────────────────────
    model_s1 = model_s2 = None
    if not args.no_models:
        if args.ckpt_s1:
            model_s1 = load_model(args.ckpt_s1, device, patch_size=args.patch_size,
                                    stage=1)
        if args.ckpt_s2:
            model_s2 = load_model(args.ckpt_s2, device, patch_size=args.patch_size,
                                    stage=2)

    # ── Load manifest ─────────────────────────────────────────────────────
    root = Path(args.root)
    with (root / "manifest.json").open() as f:
        records = json.load(f)

    test_recs = [r for r in records if r.get("split","train") == "test"]
    if args.max_episodes:
        test_recs = test_recs[:args.max_episodes]
    print(f"Evaluating {len(test_recs)} test episodes  "
          f"[eval_mode={args.eval_mode}] …")

    # Scene cache — one entry per scene_id (avoids reloading for multi-scene datasets)
    scene_cache: Dict[str, Dict] = {}

    def get_maps(scene_id: str) -> Dict:
        if scene_id not in scene_cache:
            scene = torch.load(root / f"scene_{scene_id}.pt",
                                map_location="cpu", weights_only=False)
            scene_cache[scene_id] = scene["maps"]
            print(f"  Loaded scene '{scene_id}'  "
                  f"shape={scene_cache[scene_id]['z2_labels'].shape}")
        return scene_cache[scene_id]

    # ── Aggregate accumulators ────────────────────────────────────────────
    agg: Dict[str, Dict[str, List]] = {p: {} for p in PLANNER_COLORS}
    all_rows: List[Dict] = []

    for ep_i, rec in enumerate(test_recs):
        ep_id = rec["episode_id"]
        print(f"  [{ep_i+1}/{len(test_recs)}] ep_{ep_id}", end="  ", flush=True)

        ep       = torch.load(rec["path"], map_location="cpu", weights_only=False)
        scene_id = ep["meta"]["scene_id"]
        maps     = get_maps(scene_id)          # ← correct scene per episode
        labels   = maps["z2_labels"]

        start_rc = tuple(ep["meta"]["start_rc"])
        goal_rc  = tuple(ep["meta"]["goal_rc"])

        # Load checkpoints — always needed for dt/d_hat metadata
        with open(ep["logs"]["checkpoints_jsonl"]) as f:
            checkpoints = [json.loads(l) for l in f]

        # Per-stage timing extracted from checkpoint metadata
        ck_dts    = [float(ck["dt"])                        for ck in checkpoints]
        ck_d_hats = [float(ck["barrier"]["barrier_d_hat"])  for ck in checkpoints]

        # Crop maps for visualisation
        ctx = 60
        r0 = max(0, min(start_rc[0],goal_rc[0])-ctx)
        r1 = min(labels.shape[0], max(start_rc[0],goal_rc[0])+ctx)
        c0 = max(0, min(start_rc[1],goal_rc[1])-ctx)
        c1 = min(labels.shape[1], max(start_rc[1],goal_rc[1])+ctx)
        labels_crop = labels[r0:r1, c0:c1]
        crop_maps   = {k: (v[r0:r1,c0:c1] if isinstance(v, np.ndarray) and v.ndim==2 else v)
                       for k,v in maps.items()}
        start_crop  = (start_rc[0]-r0, start_rc[1]-c0)
        goal_crop   = (goal_rc[0]-r0,  goal_rc[1]-c0)

        # ── Reference grid planners (on crop) ────────────────────────────
        path_blind  = dijkstra_blind(crop_maps, start_crop, goal_crop)
        path_oracle = astar_oracle(crop_maps,   start_crop, goal_crop)

        # ── Build stage waypoints for model evaluation ────────────────────
        if args.eval_mode == "stagewise":
            # Uses oracle dataset stage_exits — follows supervised waypoints.
            # Fair for: "how well does the learned controller execute the
            # benchmark's stage decomposition?"
            waypoints_xy = [ck["stage_exit"] for ck in checkpoints]
            mode_note    = "stagewise (oracle waypoints)"

        else:  # endtoend
            # Uses geometry-only A* on geom_occ to derive waypoints — no
            # oracle material knowledge baked into the route.
            # Both S1 and S2 get IDENTICAL waypoints; material learning
            # shows up purely in λ_soft/λ_hard deflecting hazards.
            # Fair for: "end-to-end navigation quality from start+goal only."
            geom_path = astar_geom_only(maps, start_rc, goal_rc)
            if geom_path is None:
                geom_path = [(start_rc[0], start_rc[1]), (goal_rc[0], goal_rc[1])]
            waypoints_xy = build_geom_waypoints(
                geom_path,
                stride     = ep["meta"].get("path_stride", 6),
                patch_size = 64,
            )
            # Pad ck_dts/d_hats if geom waypoints > checkpoints
            while len(ck_dts)    < len(waypoints_xy): ck_dts.append(ck_dts[-1])
            while len(ck_d_hats) < len(waypoints_xy): ck_d_hats.append(ck_d_hats[-1])
            mode_note = "endtoend (geom-only waypoints)"

        # ── Model trajectories ────────────────────────────────────────────
        traj_s1 = traj_s2 = None
        traj_s1_crop = traj_s2_crop = None

        def _run(model, stage_flag):
            return run_model_episode(
                model, maps, waypoints_xy,
                ck_d_hats, ck_dts,
                start_rc, goal_rc, device,
                stage           = stage_flag,
                steps_per_stage = args.steps_per_stage,
                patch_size      = args.patch_size,
            )

        if model_s1 is not None:
            traj_s1      = _run(model_s1, 1)
            traj_s1_crop = traj_s1 - np.array([c0, r0], np.float32)
        if model_s2 is not None:
            traj_s2      = _run(model_s2, 2)
            traj_s2_crop = traj_s2 - np.array([c0, r0], np.float32)

        # ── Compute metrics (on full-map coords for models, crop for grid) ─
        m_blind  = path_metrics_grid(path_blind,  crop_maps)
        m_oracle = path_metrics_grid(path_oracle, crop_maps)
        m_s1     = path_metrics_continuous(traj_s1, maps) if traj_s1 is not None \
                   else _empty_metrics()
        m_s2     = path_metrics_continuous(traj_s2, maps) if traj_s2 is not None \
                   else _empty_metrics()

        metrics_by_planner = {
            "blind":   m_blind,
            "oracle":  m_oracle,
            "s1_model":m_s1,
            "s2_model":m_s2,
        }

        # Accumulate
        for p, m in metrics_by_planner.items():
            for k, v in m.items():
                if k not in agg[p]: agg[p][k] = []
                agg[p][k].append(float(v))

        # CSV row per planner
        for p, m in metrics_by_planner.items():
            row = {"episode_id": ep_id, "planner": p}
            row.update(m)
            all_rows.append(row)

        summary = " | ".join(
            f"{p[:5]}:risk={m['risk_exposure']:.1f},hard={m['paper_hard_hits']}"
            for p, m in metrics_by_planner.items())
        print(summary)

        # ── Per-episode outputs ───────────────────────────────────────────
        ep_dir = out_root / f"ep_{ep_id}"
        ep_dir.mkdir(exist_ok=True)

        paths_grid = {"blind": path_blind, "oracle": path_oracle}
        paths_cont = {
            "s1_model": traj_s1_crop if traj_s1 is not None else None,
            "s2_model": traj_s2_crop if traj_s2 is not None else None,
        }

        # Overview PNG
        fig = render_overview(labels_crop, crop_maps, paths_grid, paths_cont,
                               start_crop, goal_crop, episode_id=ep_id)
        fig.savefig(ep_dir/"overview.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        # Cumulative risk PNG
        fig = render_cumrisk(paths_grid, paths_cont, crop_maps, labels_crop)
        fig.savefig(ep_dir/"cumrisk.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        # Metrics JSON
        with (ep_dir/"metrics.json").open("w") as f:
            json.dump({p: m for p,m in metrics_by_planner.items()}, f, indent=2)

    # ── Aggregate outputs ─────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"{'Planner':<14} {'Length(m)':>12} {'Risk':>10} {'PaperHits':>12} {'MeanRho':>10}")
    print("─"*72)
    for p in PLANNER_COLORS:
        d = agg[p]
        if not d: continue
        print(f"  {PLANNER_LABELS[p]:<20} "
              f"{np.mean(d.get('length_m',[0])):>10.2f}  "
              f"{np.mean(d.get('risk_exposure',[0])):>10.2f}  "
              f"{np.mean(d.get('paper_hard_hits',[0])):>10.2f}  "
              f"{np.mean(d.get('mean_rho',[0])):>10.4f}")

    # CSV
    csv_path = out_root / "aggregate.csv"
    fieldnames = ["episode_id","planner","length_m","risk_exposure",
                  "hard_hits","paper_hard_hits","mean_rho"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(all_rows)
    print(f"\n  CSV  → {csv_path}")

    # Summary JSON
    summary_stats = {}
    for p in PLANNER_COLORS:
        summary_stats[p] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                             for k, v in agg[p].items()}
    with (out_root/"summary.json").open("w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"  JSON → {out_root/'summary.json'}")

    # Summary bar chart
    fig = render_summary_bar(agg)
    fig.savefig(out_root/"summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_root/'summary.png'}")
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    required=True,
                    help="data/dfc2018_stagewise (must contain manifest.json + scene_*.pt)")
    ap.add_argument("--ckpt_s1", default=None, help="Stage 1 checkpoint .pt")
    ap.add_argument("--ckpt_s2", default=None, help="Stage 2 checkpoint .pt")
    ap.add_argument("--out",     default="eval_output")
    ap.add_argument("--no_models", action="store_true",
                    help="Run only blind/oracle planners (no model checkpoints needed)")
    ap.add_argument("--eval_mode", default="endtoend",
                    choices=["stagewise","endtoend"],
                    help="stagewise: use oracle checkpoint stage_exits as waypoints. "
                         "endtoend: derive waypoints from geometry-only A* (fair "
                         "comparison — both models get identical geometry-derived goals).")
    ap.add_argument("--max_episodes", type=int, default=None,
                    help="Limit to first N test episodes")
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patch_size",       type=int,   default=32)
    ap.add_argument("--steps_per_stage",  type=int,   default=80,
                    help="Integrator steps per stagewise waypoint")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
