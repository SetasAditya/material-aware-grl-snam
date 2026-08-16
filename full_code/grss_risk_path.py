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
    ax1.plot(db,cb,color="#00e5ff",lw=2.5,label=f"Blind  — total {cb[-1]:.2f}",
             path_effects=[pe.Stroke(linewidth=4,foreground="black",alpha=0.4),pe.Normal()])
    ax1.plot(da,ca,color="#ff6b00",lw=2.5,label=f"Aware  — total {ca[-1]:.2f}",
             path_effects=[pe.Stroke(linewidth=4,foreground="black",alpha=0.4),pe.Normal()])
    ax1.fill_between(db,cb,alpha=0.12,color="#00e5ff")
    ax1.fill_between(da,ca,alpha=0.12,color="#ff6b00")
    ax1.set_xlabel("Distance along path (m)",fontsize=12)
    ax1.set_ylabel("Cumulative risk exposure  Σ r̃(xₜ)·Δs",fontsize=12)
    ax1.set_title("Cumulative Risk Exposure",fontweight="bold",fontsize=13)
    ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gt",       default=None)
    ap.add_argument("--start_rc", type=int,nargs=2,default=None,metavar=("R","C"))
    ap.add_argument("--goal_rc",  type=int,nargs=2,default=None,metavar=("R","C"))
    ap.add_argument("--crop_r0",  type=int,default=None)
    ap.add_argument("--crop_c0",  type=int,default=None)
    ap.add_argument("--crop_h",   type=int,default=300)
    ap.add_argument("--crop_w",   type=int,default=600)
    ap.add_argument("--sigma",    type=float,default=2.5)
    ap.add_argument("--risk_w",   type=float,default=10.0)
    ap.add_argument("--fps",      type=int,default=15)
    ap.add_argument("--skip",     type=int,default=3)
    ap.add_argument("--out",      default="output/risk_path.gif")
    ap.add_argument("--overview", default="output/risk_path_overview.png")
    ap.add_argument("--metrics",  default="output/risk_metrics.png")
    ap.add_argument("--cumrisk",  default="output/risk_cumulative.png")
    args=ap.parse_args()

    global RISK_WEIGHT
    RISK_WEIGHT = args.risk_w

    Path(args.out).parent.mkdir(parents=True,exist_ok=True)

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
        # Print global coords for re-use
        print(f"  To re-run:  --crop_r0 {r0} --crop_c0 {c0} "
              f"--start_rc {start[0]} {start[1]} "
              f"--goal_rc {goal[0]} {goal[1]}")

    # ── Crop ─────────────────────────────────────────────────────────────
    labels_crop = labels_full[r0:r1, c0:c1].copy()
    rf_crop     = RiskField(labels_crop, sigma=args.sigma)
    H,W         = labels_crop.shape

    # Validate
    start=(int(np.clip(start[0],0,H-1)), int(np.clip(start[1],0,W-1)))
    goal =(int(np.clip(goal[0], 0,H-1)), int(np.clip(goal[1], 0,W-1)))
    print(f"  Start: {start} — {CLASS_NAMES.get(int(labels_crop[start]),'?')}")
    print(f"  Goal:  {goal}  — {CLASS_NAMES.get(int(labels_crop[goal]),'?')}")

    # ── Plan ─────────────────────────────────────────────────────────────
    print("Dijkstra blind path (no material knowledge) …")
    path_blind = dijkstra_blind(labels_crop, start, goal)
    print("A* risk-aware path …")
    path_aware = astar_aware(rf_crop, start, goal)

    if path_blind is None:
        print("[WARN] Blind path failed — trying with passable-only")
        path_blind = path_aware   # fallback

    # ── Print metrics ────────────────────────────────────────────────────
    mb = path_metrics(path_blind or [], rf_crop)
    ma = path_metrics(path_aware or [], rf_crop)
    print(f"\n{'Metric':<28} {'Blind (GRL-SNAM)':>18} {'Aware (HAVSN)':>15}")
    print("─"*63)
    for k in ("length_m","risk_exposure","hard_hits","mean_rho"):
        print(f"  {k:<26} {mb.get(k,0):>18.3f} {ma.get(k,0):>15.3f}")
    if mb and ma:
        det=(ma['length_m']/max(mb['length_m'],1e-6)-1)*100
        red=(1-ma['risk_exposure']/max(mb['risk_exposure'],1e-6))*100
        print(f"\n  Detour: {det:+.1f}%  |  Risk reduction: {red:.1f}%")
        print(f"  Hard hazard entries: blind={mb['hard_hits']}  aware={ma['hard_hits']}")

    # ── Render ───────────────────────────────────────────────────────────
    print("\nRendering overview …")
    fig=render_overview(labels_crop,rf_crop,path_blind,path_aware,start,goal)
    fig.savefig(args.overview,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"  Overview → {args.overview}")

    render_metrics(mb, ma, args.metrics)
    render_cumulative_risk(path_blind or [], path_aware or [], rf_crop, args.cumrisk)

    print("\nRendering GIF …")
    if path_aware:
        make_gif(labels_crop,rf_crop,path_blind,path_aware,start,goal,
                 out_path=args.out,fps=args.fps,skip=args.skip)
    print("\nDone!")

if __name__=="__main__":
    main()