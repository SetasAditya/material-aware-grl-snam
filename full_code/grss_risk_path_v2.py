#!/usr/bin/env python3
"""
Risk-aware vs. Material-blind path comparison for DFC2018 GT labels.

Two planners:
  - BLIND  : pure Dijkstra with uniform cost=1, no material awareness at all.
             This models GRL-SNAM (no Z2). Will happily cross buildings/water.
  - AWARE  : A* with cell cost = 1 + RISK_WEIGHT * r̃(cell), hard hazards
             treated as impassable. This models HAVSN oracle (with Z2).

Outputs:
  1. Four-panel overview PNG  (labels | risk field | SDF | overlay+paths)
  2. Animated GIF — agent walks AWARE path; BLIND path shown as ghost
  3. Cumulative risk comparison plot

Usage:
  python grss_risk_path_v2.py --gt /path/to/2018_IEEE_GRSS_DFC_GT_TR.tif
  python grss_risk_path_v2.py  # synthetic demo
  
  ppython -m scripts.build_dfc2018_stagewise --gt /mnt/data/adityas/GRL-SNAM/ImageryAndTrainingGT/2018IEEE_Contest/Phase2/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif --crop_r0 0 --crop_c0 0 --crop_h 1202 --crop_w 4768 --out data/dfc2018_stagewise --num_episodes 300
"""
from __future__ import annotations

import argparse, heapq, os
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel
from scipy.ndimage import map_coordinates

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
GT_GSD    = 0.5
GT_ORIGIN = np.array([272056.25, 3290289.75])

CLASS_NAMES = {
    0:'Unclassified',    1:'Healthy grass',    2:'Stressed grass',
    3:'Artificial turf', 4:'Evergreen trees',  5:'Deciduous trees',
    6:'Bare earth',      7:'Water',            8:'Residential bldg',
    9:'Non-res bldg',   10:'Roads',           11:'Sidewalks',
   12:'Crosswalks',     13:'Major thoroughfare',14:'Highways',
   15:'Railways',       16:'Paved parking',   17:'Unpaved parking',
   18:'Cars',           19:'Trains',          20:'Stadium seats',
}

RHO: Dict[int,float] = {
    0:0.00, 1:0.05, 2:0.30, 3:0.10, 4:0.10, 5:0.15,
    6:0.20, 7:0.95, 8:0.85, 9:0.90,10:0.10,11:0.05,
   12:0.20,13:0.25,14:0.95,15:0.95,16:0.10,17:0.30,
   18:0.25,19:0.95,20:0.15,
}

HAZARD_HARD = frozenset({7, 8, 9, 14, 15, 19})   # ρ ≥ 0.85, impassable for AWARE
HAZARD_SOFT = frozenset({2, 6, 12, 13, 17, 18})   # moderate risk

CLASS_COLORS = [
    "#2d2d2d","#55a630","#a7c957","#3d9970","#1b4332","#52b788",
    "#d4a373","#4895ef","#e63946","#c1121f","#adb5bd","#dee2e6",
    "#f4d35e","#e9c46a","#f77f00","#9b2226","#b7b7b7","#8d5524",
    "#023e8a","#780000","#7b2d8b",
]

RISK_WEIGHT = 10.0   # A* cost amplifier for risk
DIAGONALS   = True

_DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)] if DIAGONALS \
        else [(-1,0),(1,0),(0,-1),(0,1)]
_STEP = {(dr,dc): 1.4142 if abs(dr)+abs(dc)==2 else 1.0 for dr,dc in _DIRS}

# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_gt_labels(gt_path: str) -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = None
    return np.array(Image.open(gt_path)).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic demo scene
# ─────────────────────────────────────────────────────────────────────────────
def make_demo_scene(rows=240, cols=500) -> np.ndarray:
    rng = np.random.default_rng(99)
    gt  = np.ones((rows,cols), dtype=np.uint8)          # healthy grass bg

    # Major roads
    gt[rows//2-2:rows//2+2, :] = 10                      # H road
    gt[:, cols//3-2:cols//3+2] = 10                      # V road 1
    gt[:, 2*cols//3-2:2*cols//3+2] = 10                  # V road 2
    gt[rows//2-4:rows//2-2, :] = 11                      # sidewalks
    gt[rows//2+2:rows//2+4, :] = 11

    # Diagonal thoroughfare (class 13) — mimics the pink band in real data
    for i in range(cols):
        r = int(rows*0.15 + i*0.18)
        if 0 <= r < rows:
            gt[max(0,r-3):min(rows,r+3), i] = 13

    # Railway strip at bottom (class 15)
    gt[rows-20:rows-14, :] = 15

    # Highway strip at top (class 14)
    gt[:8, :] = 14

    # Non-res buildings (class 9) — hard obstacles
    for r0,c0,h,w in [(30,80,40,60),(30,220,50,70),(30,380,40,55),
                       (140,80,45,65),(140,250,40,60),(140,390,50,70),
                       (30,490,80,80),(170,490,50,70)]:
        if r0+h < rows and c0+w < cols:
            gt[r0:r0+h, c0:c0+w] = 9

    # Water body (class 7)
    gt[55:90, 155:215] = 7

    # Res buildings (class 8)
    for _ in range(12):
        r=rng.integers(10,rows-20); c=rng.integers(10,cols-20)
        gt[r:r+10, c:c+10] = 8

    # Stressed grass & bare earth
    for cls, count in [(2,6),(6,5)]:
        for _ in range(count):
            r=rng.integers(0,rows-15); c=rng.integers(0,cols-20)
            gt[r:r+12, c:c+18] = cls

    # Parking (class 16) — safe
    gt[10:35, cols//2-20:cols//2+20] = 16

    # Unclassified border
    gt[:3,:]=0; gt[-3:,:]=0; gt[:,:3]=0; gt[:,-3:]=0
    return gt

# ─────────────────────────────────────────────────────────────────────────────
# Risk field
# ─────────────────────────────────────────────────────────────────────────────
class RiskField:
    def __init__(self, labels: np.ndarray, gsd=GT_GSD, sigma=2.0):
        self.labels = labels
        self.gsd    = gsd
        rho_fn = np.vectorize(RHO.get)
        self.rho = rho_fn(labels).astype(np.float32)
        self.rho_smooth = gaussian_filter(self.rho, sigma=sigma).astype(np.float32)
        self.grad_ry = (sobel(self.rho_smooth,axis=0)/(2*gsd)).astype(np.float32)
        self.grad_rx = (sobel(self.rho_smooth,axis=1)/(2*gsd)).astype(np.float32)
        self.hard_mask = np.isin(labels, list(HAZARD_HARD))
        d_px = distance_transform_edt(~self.hard_mask)
        self.phi = (d_px * gsd).astype(np.float32)
        self.grad_py = (sobel(self.phi,axis=0)/(2*gsd)).astype(np.float32)
        self.grad_px = (sobel(self.phi,axis=1)/(2*gsd)).astype(np.float32)

    def aware_cost(self, r: int, c: int) -> float:
        H,W = self.labels.shape
        if not (0<=r<H and 0<=c<W): return 1e9
        if self.hard_mask[r,c]:     return 1e9
        return 1.0 + RISK_WEIGHT * float(self.rho_smooth[r,c])

# ─────────────────────────────────────────────────────────────────────────────
# Planners
# ─────────────────────────────────────────────────────────────────────────────
def dijkstra_blind(labels: np.ndarray,
                   start: Tuple[int,int],
                   goal:  Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    """
    Fully material-blind: uniform cost=1, NO obstacle avoidance whatsoever.
    Models GRL-SNAM with zero material knowledge.
    """
    H,W = labels.shape
    dist = {start: 0.0}
    prev: Dict[Tuple,Optional[Tuple]] = {start: None}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == goal:
            path=[]; node=goal
            while node: path.append(node); node=prev[node]
            return path[::-1]
        if d > dist.get(u, 1e18)+1e-9: continue
        r,c = u
        for dr,dc in _DIRS:
            nr,nc = r+dr, c+dc
            if not (0<=nr<H and 0<=nc<W): continue
            nd = d + _STEP[(dr,dc)]
            if nd < dist.get((nr,nc), 1e18):
                dist[(nr,nc)] = nd
                prev[(nr,nc)] = u
                heapq.heappush(heap, (nd,(nr,nc)))
    return None


def astar_aware(rf: RiskField,
                start: Tuple[int,int],
                goal:  Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    """
    Risk-aware A*: avoids hard hazards, penalises soft risk.
    Models HAVSN oracle (with Z2).
    """
    H,W = rf.labels.shape
    gr,gc = goal
    def h(r,c):
        dr,dc=abs(r-gr),abs(c-gc)
        return max(dr,dc)+(1.4142-1)*min(dr,dc)

    g_score = {start: 0.0}
    came_from: Dict[Tuple,Optional[Tuple]] = {start: None}
    heap = [(h(*start), 0.0, *start)]
    while heap:
        _,g,r,c = heapq.heappop(heap)
        if (r,c)==goal:
            path=[]; node=goal
            while node: path.append(node); node=came_from[node]
            return path[::-1]
        if g > g_score.get((r,c),1e18)+1e-9: continue
        for dr,dc in _DIRS:
            nr,nc=r+dr,c+dc
            step = _STEP[(dr,dc)]
            cell_c = rf.aware_cost(nr,nc)
            if cell_c >= 1e8: continue
            ng = g + step*cell_c
            if ng < g_score.get((nr,nc),1e18):
                g_score[(nr,nc)] = ng
                came_from[(nr,nc)] = (r,c)
                heapq.heappush(heap,(ng+h(nr,nc),ng,nr,nc))
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def path_metrics(path, rf: RiskField, gsd=GT_GSD) -> Dict:
    if not path: return {}
    length=0.; risk_acc=0.; hard_hits=0; rho_along=[]
    for i in range(len(path)-1):
        r0,c0=path[i]; r1,c1=path[i+1]
        step = gsd*(_STEP[(r1-r0,c1-c0)])
        length += step
        rho_here = float(rf.rho_smooth[r1,c1])
        risk_acc += step*rho_here
        rho_along.append(rho_here)
        if rf.hard_mask[r1,c1]: hard_hits+=1
    return dict(length_m=length, risk_exposure=risk_acc,
                hard_hits=hard_hits, mean_rho=risk_acc/max(length,1e-6),
                rho_along=rho_along)

def cumulative_risk(path, rf: RiskField, gsd=GT_GSD):
    """Returns (distances_m, cumulative_risk) arrays for plotting."""
    if not path: return np.array([]), np.array([])
    ds=[0.]; cr=[0.]
    for i in range(len(path)-1):
        r0,c0=path[i]; r1,c1=path[i+1]
        step=gsd*_STEP[(r1-r0,c1-c0)]
        ds.append(ds[-1]+step)
        cr.append(cr[-1]+step*float(rf.rho_smooth[r1,c1]))
    return np.array(ds), np.array(cr)

# ─────────────────────────────────────────────────────────────────────────────
# Auto region finder — picks best crop+start/goal from actual GT
# ─────────────────────────────────────────────────────────────────────────────
def find_demo_region(labels: np.ndarray, rf: RiskField,
                     crop_h=300, crop_w=600,
                     n_tries=200, seed=7) -> Tuple:
    """
    Scan the actual GT to find a (crop_r0,crop_c0,start,goal) where:
      - both start and goal are on safe cells (ρ < 0.3, not unclassified)
      - Euclidean distance ≥ 120 cells
      - direct line crosses ≥ 2 hard hazard cells (makes BLIND vs AWARE dramatic)
      - crop fits within the GT extent
    Returns (r0,c0,start_in_crop,goal_in_crop) — coords relative to crop.
    """
    rng = np.random.default_rng(seed)
    H,W = labels.shape

    # Build safe cell list within inset (avoid GT border unclassified strip)
    inset = 20
    safe_rc = np.argwhere(
        (labels[inset:H-inset, inset:W-inset] > 0) &
        (rf.rho_smooth[inset:H-inset, inset:W-inset] < 0.25)
    ) + inset

    if len(safe_rc) < 200:
        return (0, 0, (crop_h//4, crop_w//4), (3*crop_h//4, 3*crop_w//4))

    best_score = -1
    best = None

    for _ in range(n_tries):
        si = rng.integers(len(safe_rc)); gi = rng.integers(len(safe_rc))
        sr,sc = safe_rc[si]; gr,gc = safe_rc[gi]
        dist = ((sr-gr)**2+(sc-gc)**2)**0.5
        if dist < 120 or dist > 400: continue

        # Count hard hazard cells on straight line
        n_pts = max(int(dist),2)
        ts = np.linspace(0,1,n_pts)
        rs = np.clip((sr+ts*(gr-sr)).astype(int),0,H-1)
        cs = np.clip((sc+ts*(gc-sc)).astype(int),0,W-1)
        hard_on_line = int(rf.hard_mask[rs,cs].sum())
        soft_on_line = int(np.isin(labels[rs,cs], list(HAZARD_SOFT)).sum())

        score = hard_on_line*5 + soft_on_line
        if score < 8: continue   # need meaningful hazards on blind path

        # Try to fit a crop
        cr_min = min(sr,gr); cr_max = max(sr,gr)
        cc_min = min(sc,gc); cc_max = max(sc,gc)
        pad_r = max(40, (crop_h-(cr_max-cr_min))//2)
        pad_c = max(40, (crop_w-(cc_max-cc_min))//2)
        r0 = max(0, cr_min-pad_r); r1 = r0+crop_h
        c0 = max(0, cc_min-pad_c); c1 = c0+crop_w
        if r1 > H: r0=H-crop_h; r1=H
        if c1 > W: c0=W-crop_w; c1=W
        r0=max(0,r0); c0=max(0,c0)

        if score > best_score:
            best_score = score
            best = (r0, c0, (sr-r0, sc-c0), (gr-r0, gc-c0))

    if best is None:
        return (0, 0, (crop_h//4, crop_w//4), (3*crop_h//4, 3*crop_w//4))
    return best

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def _make_cmap():
    return ListedColormap(CLASS_COLORS), BoundaryNorm(range(22),21)

def _plot_path(ax, path, color, lw=2.5, alpha=1.0, ls="-", label=None):
    if not path: return
    rs=[p[0] for p in path]; cs=[p[1] for p in path]
    ax.plot(cs,rs,color=color,lw=lw,alpha=alpha,ls=ls,label=label,
            path_effects=[pe.Stroke(linewidth=lw+2,foreground="black",alpha=alpha*0.6),
                          pe.Normal()])

def _marker(ax, rc, marker, color, ms=13, zorder=30):
    ax.plot(rc[1],rc[0],marker,ms=ms,color=color,zorder=zorder,
            path_effects=[pe.Stroke(linewidth=2.5,foreground="black"),pe.Normal()])

def render_overview(labels_crop, rf_crop, path_blind, path_aware,
                    start, goal) -> plt.Figure:
    H,W = labels_crop.shape
    cmap,norm = _make_cmap()

    fig,axes = plt.subplots(1,4,figsize=(26,7))

    # ── 1. GT labels + both paths ─────────────────────────────────────────
    ax=axes[0]
    ax.imshow(labels_crop,cmap=cmap,norm=norm,interpolation="nearest")
    _plot_path(ax,path_blind,"#00e5ff",lw=2.0,alpha=0.85,label="Blind (GRL-SNAM)")
    _plot_path(ax,path_aware,"#ff6b00",lw=2.8,alpha=1.0,label="Risk-aware (HAVSN)")
    _marker(ax,start,"^","#00e5ff"); _marker(ax,goal,"*","#ff6b00",ms=15)
    ax.set_title("GT Labels",fontweight="bold",fontsize=12)
    ax.legend(loc="lower right",fontsize=8,framealpha=0.85)
    ax.axis("off")

    # ── 2. Smoothed risk field r̃ ─────────────────────────────────────────
    ax=axes[1]
    im=ax.imshow(rf_crop.rho_smooth,cmap="RdYlGn_r",vmin=0,vmax=1,interpolation="bilinear")
    plt.colorbar(im,ax=ax,fraction=0.03,label="r̃(x)")
    _plot_path(ax,path_blind,"#00e5ff",lw=2.0,alpha=0.7)
    _plot_path(ax,path_aware,"#ff6b00",lw=2.8)
    _marker(ax,start,"^","white"); _marker(ax,goal,"*","white")
    ax.set_title("Smoothed Risk Field r̃(x)",fontweight="bold",fontsize=12)
    ax.axis("off")

    # ── 3. SDF φ to hard hazards ──────────────────────────────────────────
    ax=axes[2]
    im=ax.imshow(rf_crop.phi,cmap="viridis",interpolation="bilinear")
    plt.colorbar(im,ax=ax,fraction=0.03,label="φ(x) [m]")
    _plot_path(ax,path_blind,"#00e5ff",lw=2.0,alpha=0.7)
    _plot_path(ax,path_aware,"#ff6b00",lw=2.8)
    _marker(ax,start,"^","white"); _marker(ax,goal,"*","white")
    ax.set_title("SDF to Hard Hazards φ(x)",fontweight="bold",fontsize=12)
    ax.axis("off")

    # ── 4. Hazard overlay with paths ──────────────────────────────────────
    ax=axes[3]
    ax.imshow(labels_crop,cmap=cmap,norm=norm,interpolation="nearest",alpha=0.65)
    hard_rgba=np.zeros((*labels_crop.shape,4))
    hard_rgba[rf_crop.hard_mask]=[0.9,0.05,0.05,0.6]
    soft_rgba=np.zeros((*labels_crop.shape,4))
    soft_rgba[np.isin(labels_crop,list(HAZARD_SOFT))]=[1.0,0.55,0.0,0.4]
    ax.imshow(hard_rgba,interpolation="nearest")
    ax.imshow(soft_rgba,interpolation="nearest")
    _plot_path(ax,path_blind,"#00e5ff",lw=2.0,alpha=0.9,label="Blind (GRL-SNAM)")
    _plot_path(ax,path_aware,"#ff6b00",lw=2.8,label="Risk-aware (HAVSN)")
    _marker(ax,start,"^","#00e5ff"); _marker(ax,goal,"*","#ff6b00",ms=15)
    hard_p=mpatches.Patch(color=(0.9,0.05,0.05,0.6),label="Hard hazard (impassable)")
    soft_p=mpatches.Patch(color=(1.0,0.55,0.0,0.4),label="Soft hazard (penalised)")
    ax.legend(handles=[hard_p,soft_p],loc="lower right",fontsize=8,framealpha=0.85)
    ax.set_title("Hazard Classes + Path Comparison",fontweight="bold",fontsize=12)
    ax.axis("off")

    fig.suptitle("HAVSN vs GRL-SNAM: Material-Aware vs Material-Blind Navigation",
                 fontsize=14,fontweight="bold",y=1.01)
    plt.tight_layout()
    return fig


def render_metrics(m_blind, m_aware, out_path: str):
    """Standalone comparison bar chart — clean figure for the paper."""
    metrics = ["Path length (m)","Risk exposure\n(m·ρ)","Hard hazard\ncell entries","Mean ρ\nalong path"]
    keys    = ["length_m","risk_exposure","hard_hits","mean_rho"]
    vals_b  = [m_blind.get(k,0) for k in keys]
    vals_a  = [m_aware.get(k,0) for k in keys]

    fig,ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(metrics))
    w = 0.35
    b1=ax.bar(x-w/2,vals_b,w,label="Material-blind (GRL-SNAM)",color="#00e5ff",
              edgecolor="black",linewidth=0.8)
    b2=ax.bar(x+w/2,vals_a,w,label="Risk-aware (HAVSN oracle)",color="#ff6b00",
              edgecolor="black",linewidth=0.8)
    for bar in list(b1)+list(b2):
        h=bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2., h+max(vals_b+vals_a)*0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(metrics,fontsize=11)
    ax.set_ylabel("Value",fontsize=11)
    ax.set_title("Navigation Metrics: Material-Blind vs Risk-Aware",
                 fontweight="bold",fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y",alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"  Metrics bar chart → {out_path}")


def render_cumulative_risk(path_blind, path_aware, rf: RiskField, out_path: str):
    """Cumulative risk exposure along each path — key paper figure."""
    db, cb = cumulative_risk(path_blind, rf)
    da, ca = cumulative_risk(path_aware, rf)

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))

    # Left: cumulative risk vs distance
    if len(cb) > 0:
        ax1.plot(db,cb,color="#00e5ff",lw=2.5,label=f"Blind  — total {cb[-1]:.2f}",
                 path_effects=[pe.Stroke(linewidth=4,foreground="black",alpha=0.4),pe.Normal()])
        ax1.fill_between(db,cb,alpha=0.12,color="#00e5ff")
    if len(ca) > 0:
        ax1.plot(da,ca,color="#ff6b00",lw=2.5,label=f"Aware  — total {ca[-1]:.2f}",
                 path_effects=[pe.Stroke(linewidth=4,foreground="black",alpha=0.4),pe.Normal()])
        ax1.fill_between(da,ca,alpha=0.12,color="#ff6b00")
    elif len(cb) > 0:
        ax1.text(0.02, 0.95, "Aware path unavailable", transform=ax1.transAxes,
                 fontsize=10, color="#ff6b00", fontweight="bold", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff6b00", alpha=0.85))
    ax1.set_xlabel("Distance along path (m)",fontsize=12)
    ax1.set_ylabel("Cumulative risk exposure  Σ r̃(xₜ)·Δs",fontsize=12)
    ax1.set_title("Cumulative Risk Exposure",fontweight="bold",fontsize=13)
    if ax1.lines:
        ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Right: instantaneous ρ along path (normalised to % complete)
    pct_b = np.linspace(0,100,len(path_blind)) if path_blind else np.array([])
    pct_a = np.linspace(0,100,len(path_aware)) if path_aware else np.array([])
    rho_b = [float(rf.rho_smooth[r,c]) for r,c in (path_blind or [])]
    rho_a = [float(rf.rho_smooth[r,c]) for r,c in (path_aware or [])]
    if rho_b: ax2.plot(pct_b,rho_b,color="#00e5ff",lw=1.5,alpha=0.8,label="Blind")
    if rho_a: ax2.plot(pct_a,rho_a,color="#ff6b00",lw=1.5,alpha=0.8,label="Aware")
    if rho_b: ax2.fill_between(pct_b,rho_b,alpha=0.1,color="#00e5ff")
    if rho_a: ax2.fill_between(pct_a,rho_a,alpha=0.1,color="#ff6b00")
    # Shade hard-hazard regions for blind path
    for i,(r,c) in enumerate(path_blind or []):
        if rf.hard_mask[r,c]:
            ax2.axvspan(pct_b[i],pct_b[min(i+1,len(pct_b)-1)],
                        color="red",alpha=0.3,lw=0)
    ax2.axhline(0.5,color="gray",lw=1,ls="--",label="τ=0.5 threshold")
    ax2.set_xlabel("Path progress (%)",fontsize=12)
    ax2.set_ylabel("Instantaneous r̃(x)",fontsize=12)
    ax2.set_title("Risk Profile Along Path  (red = hard hazard entry)",
                  fontweight="bold",fontsize=13)
    ax2.legend(fontsize=10); ax2.grid(alpha=0.3); ax2.set_ylim(0,1.05)

    fig.suptitle("Risk Comparison: Material-Blind (GRL-SNAM) vs Risk-Aware (HAVSN Oracle)",
                 fontsize=13,fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"  Cumulative risk plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# GIF
# ─────────────────────────────────────────────────────────────────────────────
def make_gif(labels_crop, rf_crop, path_blind, path_aware,
             start, goal, out_path, fps=15, skip=3):
    cmap,norm = _make_cmap()
    H,W = labels_crop.shape

    def subsamp(path,step):
        if not path: return []
        idx=list(range(0,len(path),step))
        if idx[-1]!=len(path)-1: idx.append(len(path)-1)
        return [path[i] for i in idx]

    frames_pts = subsamp(path_aware, skip)
    n = len(frames_pts)
    print(f"  Rendering {n} GIF frames …")

    def fig_to_arr(fig):
        fig.canvas.draw()
        w2,h2=fig.canvas.get_width_height()
        buf=np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8)
        return buf.reshape(h2,w2,4)[...,:3].copy()

    imgs=[]
    for fi,(pr,pc) in enumerate(frames_pts):
        fig,axes=plt.subplots(1,3,figsize=(21,6),dpi=88)

        done_aware = path_aware[:fi*skip+1] if path_aware else []
        ahead_aware= path_aware[fi*skip:]   if path_aware else []

        # ── Panel 1: label map ─────────────────────────────────────────
        ax=axes[0]
        ax.imshow(labels_crop,cmap=cmap,norm=norm,interpolation="nearest",alpha=0.82)
        # Hard/soft hazard overlay
        hard_rgba=np.zeros((*labels_crop.shape,4))
        hard_rgba[rf_crop.hard_mask]=[0.9,0.1,0.1,0.45]
        soft_rgba=np.zeros((*labels_crop.shape,4))
        soft_rgba[np.isin(labels_crop,list(HAZARD_SOFT))]=[1.0,0.55,0.0,0.28]
        ax.imshow(hard_rgba,interpolation="nearest")
        ax.imshow(soft_rgba,interpolation="nearest")
        # Blind path (ghost)
        if path_blind:
            brs=[p[0] for p in path_blind]; bcs=[p[1] for p in path_blind]
            ax.plot(bcs,brs,color="#00e5ff",lw=1.8,alpha=0.6,ls="--",
                    label="Blind path",
                    path_effects=[pe.Stroke(linewidth=3.5,foreground="black",alpha=0.3),pe.Normal()])
        # Aware path: done + ahead
        if done_aware:
            drs=[p[0] for p in done_aware]; dcs=[p[1] for p in done_aware]
            ax.plot(dcs,drs,"#ff6b00",lw=2.8,alpha=0.95,
                    path_effects=[pe.Stroke(linewidth=4.5,foreground="black",alpha=0.5),pe.Normal()])
        if ahead_aware:
            ars=[p[0] for p in ahead_aware]; acs=[p[1] for p in ahead_aware]
            ax.plot(acs,ars,color="#ff6b00",lw=1.5,alpha=0.30,ls=":")
        _marker(ax,start,"^","#00e5ff",ms=12)
        _marker(ax,goal,"*","#ffd60a",ms=14)
        # Agent
        ax.plot(pc,pr,"o",ms=15,color="#ff0054",zorder=50,
                path_effects=[pe.Stroke(linewidth=3,foreground="white"),pe.Normal()])
        ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis("off")
        bl_p=mpatches.Patch(color="#00e5ff",alpha=0.7,label="Blind (GRL-SNAM)")
        aw_p=mpatches.Patch(color="#ff6b00",label="Risk-aware (HAVSN)")
        ax.legend(handles=[bl_p,aw_p],loc="lower right",fontsize=8,framealpha=0.85)
        ax.set_title("GT Labels + Hazard Overlay",fontweight="bold")

        # ── Panel 2: risk field live readout ──────────────────────────
        ax=axes[1]
        im=ax.imshow(rf_crop.rho_smooth,cmap="RdYlGn_r",vmin=0,vmax=1,interpolation="bilinear")
        plt.colorbar(im,ax=ax,fraction=0.03,pad=0.01,label="r̃(x)")
        if done_aware:
            ax.plot([p[1] for p in done_aware],[p[0] for p in done_aware],
                    "#ff6b00",lw=2.5,alpha=0.9)
        ax.plot(pc,pr,"o",ms=12,color="white",zorder=50,
                path_effects=[pe.Stroke(linewidth=3,foreground="#ff0054"),pe.Normal()])
        _marker(ax,start,"^","white",ms=10); _marker(ax,goal,"*","white",ms=12)
        rho_here = float(rf_crop.rho_smooth[pr,pc])
        phi_here = float(rf_crop.phi[pr,pc])
        ax.set_title(f"Risk Field r̃(x)   [now: {rho_here:.3f}]",fontweight="bold")
        ax.axis("off")

        # ── Panel 3: cumulative risk race ─────────────────────────────
        ax=axes[2]
        # Compute cumulative at this frame
        db_full,cb_full = cumulative_risk(path_blind, rf_crop)
        da_full,ca_full = cumulative_risk(path_aware, rf_crop)
        # Current progress on aware path
        cur_step = min(fi*skip, len(path_aware)-1) if path_aware else 0
        da_now = da_full[:cur_step+1]; ca_now = ca_full[:cur_step+1]
        # Map blind path to same distance axis
        if len(db_full)>1 and len(da_full)>1:
            ax.plot(db_full,cb_full,color="#00e5ff",lw=2,alpha=0.45,ls="--",label="Blind (full)")
            ax.fill_between(db_full,cb_full,alpha=0.07,color="#00e5ff")
        if len(da_now)>1:
            ax.plot(da_now,ca_now,color="#ff6b00",lw=2.5,alpha=0.95,label="Aware (live)")
            ax.fill_between(da_now,ca_now,alpha=0.12,color="#ff6b00")
        if len(db_full)>0 and len(cb_full)>0:
            ax.axhline(cb_full[-1],color="#00e5ff",lw=1,ls=":",alpha=0.5,label=f"Blind total={cb_full[-1]:.1f}")
        ax.set_xlabel("Distance (m)"); ax.set_ylabel("Cumulative risk Σ r̃·Δs")
        ax.set_title("Risk Accumulation Race",fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        if len(db_full)>0:
            ax.set_xlim(0,max(db_full[-1],da_full[-1] if len(da_full)>0 else 1)*1.05)
            ax.set_ylim(0,max(cb_full[-1] if len(cb_full)>0 else 1,
                              ca_full[-1] if len(ca_full)>0 else 1)*1.15)

        pct = int(100*fi/max(n-1,1))
        label_id = int(labels_crop[pr,pc])
        fig.suptitle(
            f"Step {fi+1}/{n}  |  Progress {pct}%  |  "
            f"Class {label_id}: {CLASS_NAMES.get(label_id,'?')}  |  "
            f"r̃={rho_here:.3f}  φ={phi_here:.1f}m",
            fontsize=11,fontweight="bold"
        )
        plt.tight_layout()
        imgs.append(Image.fromarray(fig_to_arr(fig)))
        plt.close(fig)
        if (fi+1)%20==0: print(f"    {fi+1}/{n}")

    # Hold last frame
    imgs += [imgs[-1]]*int(fps*2)
    imgs[0].save(out_path,save_all=True,append_images=imgs[1:],
                 duration=int(1000/fps),loop=0,optimize=False)
    print(f"  GIF → {out_path}  ({len(imgs)} frames @ {fps}fps)")

# ─────────────────────────────────────────────────────────────────────────────
# Two-Maneuver Scene
# ─────────────────────────────────────────────────────────────────────────────
def make_two_maneuver_scene(rows: int = 120, cols: int = 625):
    """
    Two vertical hazard bands that CROSS the road (row 60).
    Detour corridors are on OPPOSITE sides:

    Zone A — Highway (class 14):  cols 148–157 (10 wide)
      Spans rows 8–119  (crosses road at row 60)
      North gap: rows 0–7  (8 clear rows)  → NORTH detour
      South gap: NONE (blocked to bottom)

    Zone B — Railway  (class 15):  cols 425–434 (10 wide)
      Spans rows 0–111  (crosses road at row 60)
      South gap: rows 112–119 (8 clear rows) → SOUTH detour
      North gap: NONE (blocked to top)

    Detour A (north, gap 8 rows, start row 60):
      Go from row 60 to row 7: ~53 diag-steps up
      Cross 10-wide Zone A: 10 steps east at row 7
      Return to row 60: ~53 diag-steps down
      Extra ≈ 2×53 + 10 - 10 ≈ 106 extra steps

    Detour B (south, gap 8 rows, start row 60):
      Same geometry but south: ~106 extra steps

    Blind path (straight at row 60):  ≈ 614 steps, crosses both zones
    Aware (detour A + detour B):      ≈ 614 + 106 + 106 = 826 steps
    Detour A only:                    ≈ 614 + 106 = 720 steps
    Detour B only:                    ≈ 614 + 106 = 720 steps

    With factor=0.55:
      deadline = 614 + (826-614)×0.55 = 614 + 117 = 731
      Detour A alone: 720 ≤ 731  ✓ feasible
      Detour B alone: 720 ≤ 731  ✓ feasible
      Both:           826 > 731  ✗ infeasible

    astar_timed at Zone B (progress ≈ 0.67, bucket ≈ 10.7):
      pressure ≈ sigmoid(4×(0.668-0.5)) = 0.67
      hard_cost = 1 + (1-0.67)×8 = 1 + 2.64 = 3.64  per cell
      Through 10 cells → +3.64×10 = 36.4 vs detour +106 steps
      → timed pushes through Zone B ✓
    """
    gt = np.ones((rows, cols), dtype=np.uint8)

    mid = rows // 2   # 60

    # ── Unclassified border FIRST (will be overwritten by zones where they overlap) ──
    gt[:3,  :] = 0; gt[-3:, :] = 0
    gt[:,  :3] = 0; gt[:, -3:] = 0

    # Road corridor
    gt[mid - 5 : mid + 5, :] = 10
    gt[mid - 7 : mid - 5, :] = 11
    gt[mid + 5 : mid + 7, :] = 11

    # ── Zone A: Highway — crosses road, blocked below row 8, north-gap rows 0-7 ──
    A_gap_rows = 8
    A_r0, A_r1 = A_gap_rows, rows   # rows 8–119 blocked (road is in here, so crossing)
    A_c0, A_c1 = 148, 158
    gt[A_r0:A_r1, A_c0:A_c1] = 14   # overwrites road cells in zone columns → blocks path

    # ── Zone B: Railway — crosses road, blocked above row 112, south-gap rows 112-119 ──
    B_gap_rows = 8
    B_r0, B_r1 = 0, rows - B_gap_rows   # rows 0–111 blocked
    B_c0, B_c1 = 425, 435
    gt[B_r0:B_r1, B_c0:B_c1] = 15   # overwrites border rows 0-2 → NO free north passage
    for dc in [2, 6]:
        if B_c0 + dc + 1 <= B_c1:
            gt[B_r0:B_r1, B_c0+dc:B_c0+dc+1] = 19

    # Landscape context
    rng = np.random.default_rng(42)
    for _ in range(6):
        r = rng.integers(3, 55); c = rng.integers(175, 415)
        h = rng.integers(6, 14); w = rng.integers(10, 22)
        if r+h >= rows-3 or c+w >= cols-3: continue
        gt[r:r+h, c:c+w] = 9
    for _ in range(6):
        r = rng.integers(65, 115); c = rng.integers(175, 415)
        h = rng.integers(6, 14); w = rng.integers(10, 22)
        if r+h >= rows-3 or c+w >= cols-3: continue
        gt[r:r+h, c:c+w] = 9
    for cls in [2, 6]:
        for _ in range(3):
            r = rng.integers(3, rows-10); c = rng.integers(3, cols-15)
            gt[r:r+6, c:c+10] = cls

    start = (mid, 5)
    goal  = (mid, cols - 6)

    return (gt, start, goal,
            (A_c0, A_c1), (B_c0, B_c1),
            (A_r0, A_r1), (B_r0, B_r1))


# ─────────────────────────────────────────────────────────────────────────────
# Time-Pressure Planner
# ─────────────────────────────────────────────────────────────────────────────
def astar_timed(rf: RiskField,
                start: Tuple[int, int],
                goal:  Tuple[int, int],
                deadline_dist: float,
                n_buckets: int = 16,
                kappa: float   = 4.0) -> Optional[List[Tuple[int, int]]]:
    """
    Time-augmented A*.  State = (r, c, dist_bucket).

    As the agent progresses toward the deadline, a sigmoid pressure function
    lowers the effective hard-hazard cost:

        pressure(bucket) = sigmoid(kappa × (progress − 0.5))

        progress = (bucket + 0.5) / n_buckets   ∈ [0, 1]

    Early (progress < 0.4): hard hazards have cost = 1e9  → avoided
    Late  (progress > 0.6): hard hazard cost diminishes   → may traverse

    This makes the agent detour Zone A (encountered at ~25% progress)
    and push through Zone B (encountered at ~75% progress).
    """
    H, W = rf.labels.shape
    gr, gc = goal
    bucket_size = deadline_dist / n_buckets

    def h(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return max(dr, dc) + (1.4142 - 1) * min(dr, dc)

    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

    def pressure(bkt):
        prog = (bkt + 0.5) / n_buckets
        return sigmoid(kappa * (prog - 0.5))

    def cell_cost(nr, nc, bkt):
        if not (0 <= nr < H and 0 <= nc < W): return 1e9
        p = pressure(bkt)
        if rf.hard_mask[nr, nc]:
            if p < 0.4:
                return 1e9           # Early: hard hazards impassable
            # Late: penalty coefficient = 8, falls to ~0 as p→1
            # At p=0.65: cost = 1 + 0.35×8 = 3.8 → 25 Zone-B cells cost 95
            # which is < detour cost of ~104 extra steps → agent pushes through
            penalty = (1.0 - p) * 8.0
            return 1.0 + penalty
        # Soft risk weight also fades with pressure
        eff_rw = RISK_WEIGHT * max(0.0, 1.0 - p)
        return 1.0 + eff_rw * float(rf.rho_smooth[nr, nc])

    start_state = (*start, 0)
    g_score:    Dict = {start_state: 0.0}
    d_score:    Dict = {start_state: 0.0}   # pure distance (no risk weighting)
    came_from:  Dict = {start_state: None}
    heap = [(h(*start), 0.0, 0.0, start[0], start[1], 0)]
    # heap tuple: (f, g_cost, dist_cells, r, c, bucket)

    while heap:
        _, g, dist, r, c, bkt = heapq.heappop(heap)
        state = (r, c, bkt)
        if g > g_score.get(state, 1e18) + 1e-9: continue
        if (r, c) == goal:
            path = []
            node = state
            while node is not None:
                path.append((node[0], node[1]))
                node = came_from.get(node)
            return path[::-1]
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W): continue
            step  = _STEP[(dr, dc)]
            nd    = dist + step
            nbkt  = min(int(nd / bucket_size), n_buckets - 1)
            cc    = cell_cost(nr, nc, nbkt)
            if cc >= 1e8: continue
            ng    = g + step * cc
            nst   = (nr, nc, nbkt)
            if ng < g_score.get(nst, 1e18):
                g_score[nst]   = ng
                d_score[nst]   = nd
                came_from[nst] = state
                heapq.heappush(heap, (ng + h(nr, nc), ng, nd, nr, nc, nbkt))

    return None   # no path found


def path_length_steps(path: list) -> float:
    """Diagonal-aware step count (sum of _STEP values). Used for deadline calibration."""
    if not path: return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        r0, c0 = path[i]; r1, c1 = path[i+1]
        total += _STEP[(r1-r0, c1-c0)]
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Deadline calibration
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_deadline(path_blind: list, path_aware_both: list,
                       factor: float = 0.45) -> float:
    """
    Set deadline (in diagonal-weighted step units) so that:
      detouring Zone A alone → FEASIBLE  (< deadline)
      detouring Zone B alone → INFEASIBLE (> deadline, since Zone B gap is very tight)
      detouring both         → INFEASIBLE

        deadline = L_blind_steps + (L_aware_steps − L_blind_steps) × factor

    astar_timed uses d_score in the same step units, so they are directly comparable.
    """
    L_b = path_length_steps(path_blind)
    L_a = path_length_steps(path_aware_both)
    return L_b + (L_a - L_b) * factor


# ─────────────────────────────────────────────────────────────────────────────
# Two-Maneuver Overview Figure
# ─────────────────────────────────────────────────────────────────────────────
def render_two_maneuver_overview(labels_crop, rf_crop,
                                  path_blind, path_aware, path_timed,
                                  start, goal,
                                  zone_a_cols, zone_b_cols,
                                  zone_a_rows, zone_b_rows,
                                  deadline_dist: float) -> plt.Figure:
    H, W = labels_crop.shape
    cmap, norm = _make_cmap()

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    A_COLS = zone_a_cols; B_COLS = zone_b_cols
    A_ROWS = zone_a_rows; B_ROWS = zone_b_rows

    COLORS = {
        "blind": "#00e5ff",
        "aware": "#ff6b00",
        "timed": "#cc00ff",
    }
    L_b = path_length_steps(path_blind or [])
    L_a = path_length_steps(path_aware or [])
    L_t = path_length_steps(path_timed or [])
    LABELS_AGENT = {
        "blind": f"S1: Material-blind (GRL-SNAM)  L={L_b:.0f}",
        "aware": f"S2 Base: Risk-aware, no deadline  L={L_a:.0f}  "
                 f"{'✓ MEETS' if L_a <= deadline_dist else '✗ MISSES'} deadline",
        "timed": f"S2 Timed: Risk-aware + deadline   L={L_t:.0f}  "
                 f"{'✓ MEETS' if L_t <= deadline_dist else '✗ MISSES'} deadline",
    }

    def annotate_zones(ax):
        """Draw zone outlines on any axis."""
        for (c0, c1), (r0, r1), label, color in [
            (A_COLS, A_ROWS, "Zone A\n(Highway)", "#f77f00"),
            (B_COLS, B_ROWS, "Zone B\n(Railway)", "#9b2226"),
        ]:
            rect = plt.Rectangle((c0, r0), c1 - c0, r1 - r0,
                                  linewidth=2, edgecolor=color,
                                  facecolor="none", linestyle="--", zorder=10)
            ax.add_patch(rect)
            ax.text((c0 + c1) / 2, r0 - 4, label, ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold", zorder=11)

    # ── Panel 1: GT labels + all 3 paths ────────────────────────────────────
    ax = axes[0]
    ax.imshow(labels_crop, cmap=cmap, norm=norm, interpolation="nearest", alpha=0.80)
    hard_rgba = np.zeros((*labels_crop.shape, 4))
    hard_rgba[rf_crop.hard_mask] = [0.9, 0.1, 0.1, 0.4]
    ax.imshow(hard_rgba, interpolation="nearest")
    for key, path in [("blind", path_blind), ("aware", path_aware), ("timed", path_timed)]:
        if path:
            rs = [p[0] for p in path]; cs = [p[1] for p in path]
            lw = 2.0 if key == "blind" else 2.8
            ls = "--" if key == "blind" else "-"
            ax.plot(cs, rs, color=COLORS[key], lw=lw, ls=ls, alpha=0.9,
                    label=LABELS_AGENT[key],
                    path_effects=[pe.Stroke(linewidth=lw+2, foreground="black", alpha=0.45),
                                  pe.Normal()])
    _marker(ax, start, "^", "white", ms=12); _marker(ax, goal, "*", "white", ms=14)
    annotate_zones(ax)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.88)
    ax.set_title("Three-Agent Path Comparison", fontweight="bold", fontsize=12)
    ax.axis("off")

    # ── Panel 2: Smoothed risk field + paths ─────────────────────────────────
    ax = axes[1]
    im = ax.imshow(rf_crop.rho_smooth, cmap="RdYlGn_r", vmin=0, vmax=1,
                   interpolation="bilinear")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="r̃(x)")
    for key, path in [("blind", path_blind), ("aware", path_aware), ("timed", path_timed)]:
        if path:
            rs = [p[0] for p in path]; cs = [p[1] for p in path]
            ax.plot(cs, rs, color=COLORS[key], lw=2.2, alpha=0.85)
    _marker(ax, start, "^", "white", ms=10); _marker(ax, goal, "*", "white", ms=12)
    annotate_zones(ax)
    ax.set_title("Risk Field r̃(x)  +  Paths", fontweight="bold", fontsize=12)
    ax.axis("off")

    # ── Panel 3: Cumulative risk vs path-progress/deadline ───────────────────
    ax = axes[2]
    for key, path in [("blind", path_blind), ("aware", path_aware), ("timed", path_timed)]:
        if not path: continue
        _, cr = cumulative_risk(path, rf_crop)
        steps = np.arange(len(cr))
        x = steps / deadline_dist   # normalise by deadline
        ax.plot(x, cr, color=COLORS[key], lw=2.5, label=LABELS_AGENT[key],
                path_effects=[pe.Stroke(linewidth=4, foreground="black", alpha=0.3),
                              pe.Normal()])
        ax.fill_between(x, cr, alpha=0.09, color=COLORS[key])
        # Mark where agent finishes (vertical tick at x=len/deadline)
        finish_x = (len(path) - 1) / deadline_dist
        ax.axvline(finish_x, color=COLORS[key], lw=1.0, ls=":", alpha=0.6)

    ax.axvline(1.0, color="red", lw=2, ls="--", label=f"Deadline (={int(deadline_dist)} cells)")
    ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100,
               xmin=0, xmax=1.0, alpha=0.0)   # placeholder for ylim
    ax.set_xlabel("Path progress / Deadline", fontsize=11)
    ax.set_ylabel("Cumulative risk  Σ r̃·Δs", fontsize=11)
    ax.set_title("Risk–Time Tradeoff", fontweight="bold", fontsize=12)
    ax.legend(fontsize=7, framealpha=0.88)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "HAVSN Two-Maneuver Experiment: Time-Dependent Risk Scheduling",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Two-Maneuver GIF  (3 agents, 3 panels)
# ─────────────────────────────────────────────────────────────────────────────
def make_two_maneuver_gif(labels_crop, rf_crop,
                           path_blind, path_aware, path_timed,
                           start, goal,
                           zone_a_cols, zone_b_cols,
                           zone_a_rows, zone_b_rows,
                           deadline_dist: float,
                           out_path: str,
                           fps: int = 12,
                           skip: int = 3):
    """
    Animated GIF with three simultaneous agents walking their paths.

    Panel 1 — Scene: GT labels + hazard overlay + 3 agent trails + deadline clock
    Panel 2 — Risk field: 3 coloured dots showing live positions
    Panel 3 — Risk-Time Race: x=steps/deadline, y=cumulative risk, red deadline line
    """
    cmap, norm = _make_cmap()
    H, W = labels_crop.shape

    COLORS  = {"blind": "#00e5ff", "aware": "#ff6b00", "timed": "#cc00ff"}
    MARKERS = {"blind": "o",       "aware": "s",        "timed": "D"}
    NAMES   = {"blind": "S1 Blind", "aware": "S2 Aware (no DL)", "timed": "S2 Timed"}
    paths   = {"blind": path_blind or [], "aware": path_aware or [], "timed": path_timed or []}

    # Precompute cumulative risk curves for all agents
    cr_data = {}
    for key, path in paths.items():
        _, cr = cumulative_risk(path, rf_crop)
        cr_data[key] = cr

    # Determine frame count from the longest path, sub-sampled by skip
    max_len   = max((len(p) for p in paths.values() if p), default=1)
    frame_pts = list(range(0, max_len, skip))
    if frame_pts[-1] != max_len - 1:
        frame_pts.append(max_len - 1)
    N = len(frame_pts)
    print(f"  Rendering {N} two-maneuver GIF frames …")

    def agent_pos_at_frame(key, frame_idx):
        """Get (r, c) of agent 'key' at animation frame frame_idx."""
        path = paths[key]
        if not path: return start
        # Map frame_idx to path index proportionally
        pi = int(frame_idx / (N - 1) * (len(path) - 1)) if N > 1 else len(path) - 1
        return path[pi]

    def agent_trail_at_frame(key, frame_idx):
        """Return path cells walked so far by agent 'key'."""
        path = paths[key]
        if not path: return []
        pi = int(frame_idx / (N - 1) * (len(path) - 1)) if N > 1 else len(path) - 1
        return path[:pi + 1]

    def agent_steps_at_frame(key, frame_idx):
        """Return the physical step index (≈ distance in cells) for deadline comparison."""
        path = paths[key]
        if not path: return 0
        return int(frame_idx / (N - 1) * (len(path) - 1)) if N > 1 else len(path) - 1

    def fig_to_arr(fig):
        fig.canvas.draw()
        w2, h2 = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf.reshape(h2, w2, 4)[..., :3].copy()

    def annotate_zones(ax, show_labels=True):
        for (c0, c1), (r0, r1), label, color in [
            (zone_a_cols, zone_a_rows, "A", "#f77f00"),
            (zone_b_cols, zone_b_rows, "B", "#9b2226"),
        ]:
            rect = plt.Rectangle((c0, r0), c1 - c0, r1 - r0,
                                  linewidth=1.5, edgecolor=color,
                                  facecolor="none", linestyle="--", zorder=10)
            ax.add_patch(rect)
            if show_labels:
                ax.text((c0 + c1) / 2, r0 - 3, f"Zone {label}",
                        ha="center", va="bottom", fontsize=7,
                        color=color, fontweight="bold", zorder=11)

    # Hard hazard overlay (precomputed, static)
    hard_rgba = np.zeros((*labels_crop.shape, 4))
    hard_rgba[rf_crop.hard_mask] = [0.9, 0.1, 0.1, 0.38]

    imgs = []
    for fi in range(N):
        fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=90)
        fig.patch.set_facecolor("#111111")

        # ── Panel 1: Scene ────────────────────────────────────────────────
        ax = axes[0]
        ax.set_facecolor("#111111")
        ax.imshow(labels_crop, cmap=cmap, norm=norm, interpolation="nearest", alpha=0.75)
        ax.imshow(hard_rgba, interpolation="nearest")

        # Full paths (ghost traces, faded)
        for key, path in paths.items():
            if path:
                ax.plot([p[1] for p in path], [p[0] for p in path],
                        color=COLORS[key], lw=1.2, alpha=0.18, ls="--")

        # Live trails + agent markers
        for key in ["blind", "aware", "timed"]:
            trail = agent_trail_at_frame(key, fi)
            if trail:
                ax.plot([p[1] for p in trail], [p[0] for p in trail],
                        color=COLORS[key], lw=2.2, alpha=0.9,
                        path_effects=[pe.Stroke(linewidth=3.8, foreground="black", alpha=0.4),
                                      pe.Normal()])
            pos = agent_pos_at_frame(key, fi)
            has_goal = (pos == goal)
            ax.plot(pos[1], pos[0], MARKERS[key], ms=11,
                    color=COLORS[key], zorder=50,
                    path_effects=[pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])

        _marker(ax, start, "^", "white", ms=11, zorder=40)
        _marker(ax, goal,  "*", "#ffd60a", ms=13, zorder=40)
        annotate_zones(ax)

        # Deadline clock bar
        steps_blind = agent_steps_at_frame("blind", fi)
        steps_aware = agent_steps_at_frame("aware", fi)
        steps_timed = agent_steps_at_frame("timed", fi)
        ref_steps   = max(steps_blind, steps_aware, steps_timed)
        clock_frac  = min(ref_steps / deadline_dist, 1.0)
        bar_x = 0.01; bar_y = 0.02; bar_w = 0.55; bar_h = 0.035
        ax.add_patch(plt.Rectangle((bar_x * W, (1 - bar_y - bar_h) * H),
                                    bar_w * W, bar_h * H,
                                    color="white", alpha=0.15, transform=ax.transData,
                                    zorder=60, clip_on=False))
        fill_color = "#ff4444" if clock_frac >= 0.95 else "#ffd60a" if clock_frac >= 0.7 else "#44ff88"
        ax.add_patch(plt.Rectangle((bar_x * W, (1 - bar_y - bar_h) * H),
                                    clock_frac * bar_w * W, bar_h * H,
                                    color=fill_color, alpha=0.75, transform=ax.transData,
                                    zorder=61, clip_on=False))
        ax.text(bar_x * W + 2, (1 - bar_y - bar_h * 0.5) * H,
                f"Deadline clock: {clock_frac*100:.0f}%",
                fontsize=8, color="white", va="center", zorder=62, fontweight="bold")

        # Legend
        legend_els = [mpatches.Patch(color=COLORS[k], label=NAMES[k]) for k in COLORS]
        ax.legend(handles=legend_els, loc="upper right", fontsize=7, framealpha=0.7,
                  facecolor="#222222", labelcolor="white")
        ax.set_title("Scene  +  Agent Trails", color="white", fontweight="bold", fontsize=11)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

        # ── Panel 2: Risk Field + Live Positions ─────────────────────────
        ax = axes[1]
        ax.set_facecolor("#111111")
        im = ax.imshow(rf_crop.rho_smooth, cmap="RdYlGn_r", vmin=0, vmax=1,
                       interpolation="bilinear")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="r̃(x)")

        for key in ["blind", "aware", "timed"]:
            trail = agent_trail_at_frame(key, fi)
            if trail:
                ax.plot([p[1] for p in trail], [p[0] for p in trail],
                        color=COLORS[key], lw=1.8, alpha=0.7)
            pos = agent_pos_at_frame(key, fi)
            rho_here = float(rf_crop.rho_smooth[pos[0], pos[1]])
            ax.plot(pos[1], pos[0], MARKERS[key], ms=10, color=COLORS[key], zorder=50,
                    path_effects=[pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])
            ax.text(pos[1] + 3, pos[0] - 3, f"{rho_here:.2f}",
                    fontsize=7, color=COLORS[key], fontweight="bold", zorder=51)

        _marker(ax, start, "^", "white", ms=9, zorder=40)
        _marker(ax, goal,  "*", "white", ms=11, zorder=40)
        annotate_zones(ax, show_labels=False)
        ax.set_title("Risk Field r̃(x)  (live ρ shown)", color="white",
                     fontweight="bold", fontsize=11)
        ax.axis("off")

        # ── Panel 3: Risk-Time Race Plot ──────────────────────────────────
        ax = axes[2]
        ax.set_facecolor("#1a1a2e")

        # Precomputed full curves (faded)
        for key, cr in cr_data.items():
            steps_full = np.arange(len(cr))
            x_full     = steps_full / deadline_dist
            ax.plot(x_full, cr, color=COLORS[key], lw=1.2, alpha=0.22, ls="--")

        # Live portions
        for key in ["blind", "aware", "timed"]:
            cr   = cr_data[key]
            stp  = agent_steps_at_frame(key, fi)
            stp  = min(stp, len(cr) - 1)
            x_live = np.arange(stp + 1) / deadline_dist
            ax.plot(x_live, cr[:stp + 1], color=COLORS[key], lw=2.5, alpha=0.95,
                    label=f"{NAMES[key]}  risk={cr[stp]:.1f}",
                    path_effects=[pe.Stroke(linewidth=4, foreground="black", alpha=0.3),
                                  pe.Normal()])
            ax.plot(x_live[-1], cr[stp], MARKERS[key], ms=9,
                    color=COLORS[key], zorder=50)

        # Deadline line
        ax.axvline(1.0, color="#ff4444", lw=2.5, ls="--", zorder=40,
                   label="Deadline")
        ax.axvspan(1.0, ax.get_xlim()[1] if ax.get_xlim()[1] > 1.05 else 1.4,
                   color="#ff4444", alpha=0.08)

        # Shade deadline-exceeded region dynamically
        ax.set_xlim(0, max(1.35, max((len(p) / deadline_dist for p in paths.values() if p),
                                      default=1.35)))
        ax.set_ylim(0, max((cr[-1] for cr in cr_data.values() if len(cr) > 0),
                            default=1.0) * 1.15)

        ax.set_xlabel("Steps elapsed / Deadline", fontsize=10, color="white")
        ax.set_ylabel("Cumulative risk  Σ r̃·Δs", fontsize=10, color="white")
        ax.set_title("Risk–Time Race  (past deadline line = MISSED)",
                     color="white", fontweight="bold", fontsize=11)
        ax.legend(fontsize=7, framealpha=0.6, facecolor="#222222",
                  labelcolor="white", loc="upper left")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#444444")
        ax.grid(alpha=0.2, color="white")

        pct = int(100 * fi / max(N - 1, 1))
        fig.suptitle(
            f"Two-Maneuver Experiment — Frame {fi+1}/{N}  ({pct}%)  "
            f"Deadline = {int(deadline_dist)} cells",
            color="white", fontsize=11, fontweight="bold"
        )
        fig.patch.set_facecolor("#111111")
        plt.tight_layout()
        imgs.append(Image.fromarray(fig_to_arr(fig)))
        plt.close(fig)
        if (fi + 1) % 20 == 0:
            print(f"    {fi+1}/{N}")

    # Hold last frame for 3 s
    imgs += [imgs[-1]] * int(fps * 3)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=False)
    print(f"  Two-maneuver GIF → {out_path}  ({len(imgs)} frames @ {fps}fps)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",     default="standard",
                    choices=["standard", "two_maneuver"],
                    help="standard: 2-agent blind vs aware.  "
                         "two_maneuver: 3-agent with deadline pressure.")
    ap.add_argument("--gt",       default=None)
    ap.add_argument("--start_rc", type=int, nargs=2, default=None, metavar=("R","C"))
    ap.add_argument("--goal_rc",  type=int, nargs=2, default=None, metavar=("R","C"))
    ap.add_argument("--crop_r0",  type=int, default=None)
    ap.add_argument("--crop_c0",  type=int, default=None)
    ap.add_argument("--crop_h",   type=int, default=300)
    ap.add_argument("--crop_w",   type=int, default=600)
    ap.add_argument("--sigma",    type=float, default=2.5)
    ap.add_argument("--risk_w",   type=float, default=10.0)
    ap.add_argument("--fps",      type=int, default=15)
    ap.add_argument("--skip",     type=int, default=3)
    ap.add_argument("--deadline_factor", type=float, default=0.55,
                    help="Fraction of (aware−blind) gap to add to blind length for deadline.")
    ap.add_argument("--kappa",    type=float, default=4.0,
                    help="Time-pressure sigmoid steepness for astar_timed.")
    ap.add_argument("--out",      default="output/risk_path.gif")
    ap.add_argument("--overview", default="output/risk_path_overview.png")
    ap.add_argument("--metrics",  default="output/risk_metrics.png")
    ap.add_argument("--cumrisk",  default="output/risk_cumulative.png")
    args = ap.parse_args()

    global RISK_WEIGHT
    RISK_WEIGHT = args.risk_w
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    #  TWO-MANEUVER MODE
    # ═══════════════════════════════════════════════════════════════════════
    if args.mode == "two_maneuver":
        print("=" * 60)
        print("  MODE: two_maneuver")
        print("=" * 60)

        # Build scene
        if args.gt and os.path.exists(args.gt):
            print(f"  [NOTE] --gt provided but two_maneuver uses synthetic scene.")
            print(f"         (real-data crop support can be added with --crop_r0 etc.)")

        print("Building two-maneuver synthetic scene …")
        (labels_crop, start, goal,
         zone_a_cols, zone_b_cols,
         zone_a_rows, zone_b_rows) = make_two_maneuver_scene()
        rf_crop = RiskField(labels_crop, sigma=args.sigma)
        H, W    = labels_crop.shape
        print(f"  Scene: {H}×{W}  start={start}  goal={goal}")
        print(f"  Zone A: cols {zone_a_cols}, rows {zone_a_rows}  (class 14 Highway)")
        print(f"  Zone B: cols {zone_b_cols}, rows {zone_b_rows}  (class 15 Railway)")

        # Plan: blind and aware-both first (for deadline calibration)
        print("\nPlanning paths …")
        print("  [1/3] Dijkstra blind (S1) …")
        path_blind = dijkstra_blind(labels_crop, start, goal)
        print(f"        length = {len(path_blind or [])} cells")

        print("  [2/3] A* risk-aware, no deadline (S2 Base) …")
        path_aware = astar_aware(rf_crop, start, goal)
        print(f"        length = {len(path_aware or [])} cells")

        # Calibrate deadline
        deadline = calibrate_deadline(path_blind or [], path_aware or [],
                                       factor=args.deadline_factor)
        print(f"\nDeadline calibrated: {deadline:.0f} cells")
        print(f"  (blind={len(path_blind or [])}, aware={len(path_aware or [])}, factor={args.deadline_factor})")

        print(f"  [3/3] A* timed (S2 Timed, κ={args.kappa}) …")
        path_timed = astar_timed(rf_crop, start, goal,
                                  deadline_dist=deadline, kappa=args.kappa)
        print(f"        length = {len(path_timed or [])} cells")

        # Print metric table
        mb = path_metrics(path_blind or [], rf_crop)
        ma = path_metrics(path_aware or [], rf_crop)
        mt = path_metrics(path_timed or [], rf_crop)

        print(f"\n{'Metric':<26} {'Blind':>14} {'Aware':>14} {'Timed':>14}  Deadline={deadline:.0f}")
        print("─" * 74)
        for k, label in [("length_m","length_m"), ("risk_exposure","risk_exp"),
                          ("hard_hits","hard_hits"), ("mean_rho","mean_ρ")]:
            row = f"  {label:<24} {mb.get(k,0):>14.2f} {ma.get(k,0):>14.2f} {mt.get(k,0):>14.2f}"
            print(row)

        print("\nDeadline status:")
        for name, path in [("Blind", path_blind), ("Aware", path_aware), ("Timed", path_timed)]:
            L = path_length_steps(path or [])
            status = "✓ MEETS" if L <= deadline else "✗ MISSES"
            print(f"  {name:<8}: steps={L:.1f}  deadline={deadline:.1f}  → {status}")

        # Render overview
        print("\nRendering overview …")
        fig = render_two_maneuver_overview(
            labels_crop, rf_crop,
            path_blind, path_aware, path_timed,
            start, goal,
            zone_a_cols, zone_b_cols,
            zone_a_rows, zone_b_rows,
            deadline
        )
        fig.savefig(args.overview, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Overview → {args.overview}")

        # Render GIF
        print("\nRendering two-maneuver GIF …")
        make_two_maneuver_gif(
            labels_crop, rf_crop,
            path_blind, path_aware, path_timed,
            start, goal,
            zone_a_cols, zone_b_cols,
            zone_a_rows, zone_b_rows,
            deadline,
            out_path=args.out,
            fps=args.fps,
            skip=args.skip,
        )
        print("\nDone!")
        return

    # ═══════════════════════════════════════════════════════════════════════
    #  STANDARD MODE  (existing v2 behaviour)
    # ═══════════════════════════════════════════════════════════════════════
    print("Mode: standard (blind vs. aware)")

    # ── Load labels ──────────────────────────────────────────────────────
    if args.gt and os.path.exists(args.gt):
        print(f"Loading GT: {args.gt}")
        labels_full = load_gt_labels(args.gt)
        print(f"  Shape: {labels_full.shape}")
    else:
        print("No GT found — using synthetic demo.")
        labels_full = make_demo_scene(240, 500)

    # ── Build risk field on full map first (for region search) ───────────
    print("Building full risk field …")
    rf_full = RiskField(labels_full, sigma=args.sigma)

    # ── Find or accept crop + start/goal ────────────────────────────────
    if args.crop_r0 is not None and args.start_rc is not None:
        r0=args.crop_r0; c0=args.crop_c0 or 0
        r1=r0+args.crop_h; c1=c0+args.crop_w
        start=tuple(args.start_rc); goal=tuple(args.goal_rc)
    else:
        print("Searching for dramatic demo region …")
        r0,c0,start,goal = find_demo_region(
            labels_full, rf_full,
            crop_h=args.crop_h, crop_w=args.crop_w
        )
        r1=r0+args.crop_h; c1=c0+args.crop_w
        print(f"  Crop: rows {r0}–{r1}, cols {c0}–{c1}")
        print(f"  Start (in crop): {start}  Goal: {goal}")
        print(f"  To re-run:  --crop_r0 {r0} --crop_c0 {c0} "
              f"--start_rc {start[0]} {start[1]} "
              f"--goal_rc {goal[0]} {goal[1]}")

    # ── Crop ─────────────────────────────────────────────────────────────
    labels_crop = labels_full[r0:r1, c0:c1].copy()
    rf_crop     = RiskField(labels_crop, sigma=args.sigma)
    H,W         = labels_crop.shape

    start=(int(np.clip(start[0],0,H-1)), int(np.clip(start[1],0,W-1)))
    goal =(int(np.clip(goal[0], 0,H-1)), int(np.clip(goal[1], 0,W-1)))
    print(f"  Start: {start} — {CLASS_NAMES.get(int(labels_crop[start]),'?')}")
    print(f"  Goal:  {goal}  — {CLASS_NAMES.get(int(labels_crop[goal]),'?')}")
    if labels_crop[start] == 0:
        print("  [WARN] Start lies on Unclassified terrain; aware planner may behave conservatively.")
    if labels_crop[goal] == 0:
        print("  [WARN] Goal lies on Unclassified terrain; aware planner may fail to find a path.")
    if rf_crop.hard_mask[start]:
        print("  [WARN] Start lies on a hard hazard; aware planner may fail to depart.")
    if rf_crop.hard_mask[goal]:
        print("  [WARN] Goal lies on a hard hazard; aware planner cannot terminate there.")

    # ── Plan ─────────────────────────────────────────────────────────────
    print("Dijkstra blind path …")
    path_blind = dijkstra_blind(labels_crop, start, goal)
    print("A* risk-aware path …")
    path_aware = astar_aware(rf_crop, start, goal)
    if path_aware is None:
        print("  [WARN] Aware path not found for the requested start/goal.")

    if path_blind is None:
        print("[WARN] Blind path failed — falling back to aware path")
        path_blind = path_aware

    # ── Print metrics ────────────────────────────────────────────────────
    mb = path_metrics(path_blind or [], rf_crop)
    ma = path_metrics(path_aware or [], rf_crop)
    print(f"\n{'Metric':<28} {'Blind (GRL-SNAM)':>18} {'Aware (HAVSN)':>15}")
    print("─" * 63)
    for k in ("length_m","risk_exposure","hard_hits","mean_rho"):
        print(f"  {k:<26} {mb.get(k,0):>18.3f} {ma.get(k,0):>15.3f}")
    if mb and ma:
        det = (ma['length_m'] / max(mb['length_m'], 1e-6) - 1) * 100
        red = (1 - ma['risk_exposure'] / max(mb['risk_exposure'], 1e-6)) * 100
        print(f"\n  Detour: {det:+.1f}%  |  Risk reduction: {red:.1f}%")
        print(f"  Hard hazard entries: blind={mb['hard_hits']}  aware={ma['hard_hits']}")

    # ── Render ───────────────────────────────────────────────────────────
    print("\nRendering overview …")
    fig = render_overview(labels_crop, rf_crop, path_blind, path_aware, start, goal)
    fig.savefig(args.overview, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overview → {args.overview}")

    render_metrics(mb, ma, args.metrics)
    render_cumulative_risk(path_blind or [], path_aware or [], rf_crop, args.cumrisk)

    print("\nRendering GIF …")
    if path_aware:
        make_gif(labels_crop, rf_crop, path_blind, path_aware, start, goal,
                 out_path=args.out, fps=args.fps, skip=args.skip)
    print("\nDone!")

if __name__=="__main__":
    main()
