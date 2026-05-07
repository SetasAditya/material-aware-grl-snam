#!/usr/bin/env python3
"""
scripts/build_dfc2018_stagewise.py

Generate stagewise navigation episodes for DFC2018 Houston dataset.
Fully compatible with the stagewise training pipeline (train_coef_energy.py,
energy_data_navmax.py) while adding material-aware fields for Setting 2.

Schema
------
data/dfc2018_stagewise/
├── manifest.json                        # episode index (train/val/test)
├── scene_<sid>.pt                       # one per crop — shared map data
│   ├── meta  {scene_id, crop, gsd, ...}
│   ├── maps  {z2_labels, risk_map, hard_mask, soft_mask, sdf_hard,
│   │          grad_row, grad_col, geom_occ}
│   └── risk_ontology  {rho, hard_classes, soft_classes}
└── episodes/
    └── ep_<eid>.pt                      # one per episode — lightweight
        ├── meta    {episode_id, scene_id, split, start_rc, goal_rc, dt, ...}
        ├── params  {d_hat, gamma_o, lambda_mat, risk_scale}
        ├── obstacles  {centers, radii, weights, d_hat_env}
        ├── frames      [{t, o, v_o, center, theta}]
        ├── frame_state [{t, stage_idx, bounds, entry_point, exit_point, center}]
        ├── episode_metrics  {risk_exposure_gt, hard_hits, mean_rho, path_length_m}
        ├── success   bool
        ├── final_center  [x, y]
        └── logs  {checkpoints_jsonl: "..."}
            └── logs/stagewise_checkpoints.jsonl
                  per-step: t, dt, stage_idx, stage_bounds, stage_entry,
                             stage_exit, center, theta, min_d, barrier,
                             obstacles_effective, risk_patch, risk_grad,
                             phaz_at_center, entropy_at_center, o_tgt, v_tgt

Usage
-----
# Synthetic demo
python -m scripts.build_dfc2018_stagewise --mode demo --out data/dfc2018_stagewise

# Real DFC2018
python -m scripts.build_dfc2018_stagewise \\
    --gt /path/to/2018_IEEE_GRSS_DFC_GT_TR.tif \\
    --out data/dfc2018_stagewise \\
    --num_episodes 300 --seed 42

# Specific crop (same coords as grss_risk_path_v2.py)
python -m scripts.build_dfc2018_stagewise \\
    --gt /path/to/2018_IEEE_GRSS_DFC_GT_TR.tif \\
    --crop_r0 0 --crop_c0 0 --crop_h 1202 --crop_w 4768 \\
    --out data/dfc2018_stagewise --num_episodes 300
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
from scipy.ndimage import (distance_transform_edt, gaussian_filter,
                           sobel, binary_dilation)

# ─────────────────────────────────────────────────────────────────────────────
# DFC2018 constants
# ─────────────────────────────────────────────────────────────────────────────
GT_GSD = 0.5   # metres per pixel

CLASS_NAMES = {
    0:'Unclassified',    1:'Healthy grass',    2:'Stressed grass',
    3:'Artificial turf', 4:'Evergreen trees',  5:'Deciduous trees',
    6:'Bare earth',      7:'Water',            8:'Residential bldg',
    9:'Non-res bldg',   10:'Roads',           11:'Sidewalks',
   12:'Crosswalks',     13:'Major thoroughfare',14:'Highways',
   15:'Railways',       16:'Paved parking',   17:'Unpaved parking',
   18:'Cars',           19:'Trains',          20:'Stadium seats',
}

RHO: Dict[int, float] = {
    0:0.00, 1:0.05, 2:0.30, 3:0.10, 4:0.10, 5:0.15,
    6:0.20, 7:0.95, 8:0.85, 9:0.90,10:0.10,11:0.05,
   12:0.20,13:0.25,14:0.95,15:0.95,16:0.10,17:0.30,
   18:0.25,19:0.95,20:0.15,
}

HARD_CLASSES  = frozenset({7, 8, 9, 14, 15, 19})
SOFT_CLASSES  = frozenset({2, 6, 12, 13, 17, 18})
BUILD_CLASSES = frozenset({8, 9})

CLASS_COLORS = [
    "#2d2d2d","#55a630","#a7c957","#3d9970","#1b4332","#52b788",
    "#d4a373","#4895ef","#e63946","#c1121f","#adb5bd","#dee2e6",
    "#f4d35e","#e9c46a","#f77f00","#9b2226","#b7b7b7","#8d5524",
    "#023e8a","#780000","#7b2d8b",
]

DIAGONALS = True
_DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)] \
        if DIAGONALS else [(-1,0),(1,0),(0,-1),(0,1)]
_STEP = {(dr,dc): 1.4142 if abs(dr)+abs(dc)==2 else 1.0 for dr,dc in _DIRS}

PATCH_SIZE  = 32
RISK_WEIGHT = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_labels(gt_path: str) -> np.ndarray:
    Image.MAX_IMAGE_PIXELS = None
    return np.array(Image.open(gt_path)).astype(np.uint8)


def make_demo_scene(rows: int = 300, cols: int = 600) -> np.ndarray:
    rng = np.random.default_rng(42)
    gt  = np.ones((rows, cols), dtype=np.uint8)
    gt[rows//2-2:rows//2+2, :] = 10
    gt[:, cols//3-2:cols//3+2] = 10
    gt[:, 2*cols//3-2:2*cols//3+2] = 10
    gt[rows//2-4:rows//2-2, :] = 11
    gt[rows//2+2:rows//2+4, :] = 11
    for i in range(cols):
        r = int(rows*0.15 + i*0.18)
        if 0 <= r < rows:
            gt[max(0,r-3):min(rows,r+3), i] = 13
    gt[rows-20:rows-14, :] = 15
    gt[:8, :] = 14
    for r0,c0,h,w in [(30,80,40,60),(30,220,50,70),(30,380,40,55),
                       (140,80,45,65),(140,250,40,60),(140,390,50,70)]:
        if r0+h < rows and c0+w < cols:
            gt[r0:r0+h, c0:c0+w] = 9
    gt[55:90, 155:215] = 7
    for _ in range(12):
        r = rng.integers(10, rows-20); c = rng.integers(10, cols-20)
        gt[r:r+10, c:c+10] = 8
    gt[:3,:]=0; gt[-3:,:]=0; gt[:,:3]=0; gt[:,-3:]=0
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# Scene-level map arrays  (computed once, stored in scene.pt)
# ─────────────────────────────────────────────────────────────────────────────

def build_scene_maps(
    labels: np.ndarray,
    sigma: float = 2.0,
    geom_inflate: int = 2,
    gsd: float = GT_GSD,
) -> Dict[str, np.ndarray]:
    """
    All derived arrays from oracle Z(2).  Stored once per scene crop.
    Keys: z2_labels, risk_map, hard_mask, soft_mask, geom_occ,
          sdf_hard, grad_row, grad_col
    """
    rho_fn   = np.vectorize(RHO.get)
    rho_raw  = rho_fn(labels).astype(np.float32)
    risk_map = gaussian_filter(rho_raw, sigma=sigma).astype(np.float32)

    hard_mask = np.isin(labels, list(HARD_CLASSES)).astype(np.uint8)
    soft_mask = np.isin(labels, list(SOFT_CLASSES)).astype(np.uint8)
    geom_occ  = np.isin(labels, list(BUILD_CLASSES)).astype(np.uint8)
    if geom_inflate > 0:
        geom_occ = binary_dilation(geom_occ, iterations=geom_inflate).astype(np.uint8)

    sdf_hard = (distance_transform_edt(~hard_mask.astype(bool)) * gsd).astype(np.float32)

    # Gradients of r̃ (risk field)
    grad_row = (sobel(risk_map, axis=0) / (2*gsd)).astype(np.float32)
    grad_col = (sobel(risk_map, axis=1) / (2*gsd)).astype(np.float32)

    # Gradients of φ (SDF to hard hazards)  — used in F_mat_hard
    # φ increases away from hazards, so ∇φ points toward open space.
    sdf_grad_row = (sobel(sdf_hard, axis=0) / (2*gsd)).astype(np.float32)
    sdf_grad_col = (sobel(sdf_hard, axis=1) / (2*gsd)).astype(np.float32)

    return dict(z2_labels=labels, risk_map=risk_map, hard_mask=hard_mask,
                soft_mask=soft_mask, geom_occ=geom_occ, sdf_hard=sdf_hard,
                grad_row=grad_row, grad_col=grad_col,
                sdf_grad_row=sdf_grad_row, sdf_grad_col=sdf_grad_col)


# ─────────────────────────────────────────────────────────────────────────────
# Planners
# ─────────────────────────────────────────────────────────────────────────────

def astar_aware(
    maps: Dict[str, np.ndarray],
    start: Tuple[int,int],
    goal:  Tuple[int,int],
    risk_weight: float = RISK_WEIGHT,
) -> Optional[List[Tuple[int,int]]]:
    """Setting 2 oracle planner: avoids hard hazards, penalises soft risk."""
    H, W = maps["z2_labels"].shape
    risk  = maps["risk_map"]
    hard  = maps["hard_mask"].astype(bool)
    gr,gc = goal

    def h(r,c):
        dr,dc=abs(r-gr),abs(c-gc)
        return max(dr,dc)+(1.4142-1)*min(dr,dc)

    def cost(r,c):
        if not(0<=r<H and 0<=c<W): return 1e9
        if hard[r,c]: return 1e9
        return 1.0 + risk_weight*float(risk[r,c])

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
            nr,nc=r+dr,c+dc; cc2=cost(nr,nc)
            if cc2>=1e8: continue
            ng=g+_STEP[(dr,dc)]*cc2
            if ng<gs.get((nr,nc),1e18):
                gs[(nr,nc)]=ng; came[(nr,nc)]=(r,c)
                heapq.heappush(heap,(ng+h(nr,nc),ng,nr,nc))
    return None


def dijkstra_geom(
    maps: Dict[str, np.ndarray],
    start: Tuple[int,int],
    goal:  Tuple[int,int],
) -> Optional[List[Tuple[int,int]]]:
    """Setting 1 geometry-only planner (baseline)."""
    H,W   = maps["z2_labels"].shape
    geom  = maps["geom_occ"].astype(bool)
    dist  = {start:0.0}; prev={start:None}
    heap  = [(0.0,start)]
    while heap:
        d,u=heapq.heappop(heap)
        if u==goal:
            path=[]; node=goal
            while node: path.append(node); node=prev[node]
            return path[::-1]
        if d>dist.get(u,1e18)+1e-9: continue
        r,c=u
        for dr,dc in _DIRS:
            nr,nc=r+dr,c+dc
            if not(0<=nr<H and 0<=nc<W): continue
            if geom[nr,nc]: continue
            nd=d+_STEP[(dr,dc)]
            if nd<dist.get((nr,nc),1e18):
                dist[(nr,nc)]=nd; prev[(nr,nc)]=u
                heapq.heappush(heap,(nd,(nr,nc)))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Local geometric obstacle extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_local_geom_obstacles(
    geom_occ: np.ndarray,
    center_rc: Tuple[int,int],
    patch_size: int = 64,
    robot_radius: float = 1.5,
    margin_factor: float = 0.5,
    stride: int = 2,
    max_points: int = 128,
) -> Tuple[np.ndarray,np.ndarray,np.ndarray,float]:
    """
    Boundary wall pixels in a local patch → obstacle disc list.
    Returns (C[N,2], R_eff[N], W[N], d_hat) in (col,row)=(x,y) coords.
    Mirrors extract_local_stage_obstacles from the dungeon pipeline.
    """
    H,W   = geom_occ.shape
    cr,cc = center_rc; half=patch_size//2
    r0,r1 = max(0,cr-half),min(H,cr+half)
    c0,c1 = max(0,cc-half),min(W,cc+half)
    patch = geom_occ[r0:r1,c0:c1]
    if patch.size==0:
        return (np.zeros((0,2),np.float32),np.zeros((0,),np.float32),
                np.zeros((0,),np.float32),robot_radius*(1+margin_factor))

    walls=patch>0; free=~walls
    border=np.zeros_like(walls)
    border[0,:]=True;border[-1,:]=True;border[:,0]=True;border[:,-1]=True
    up_f=np.zeros_like(free);up_f[:-1,:]=free[1:,:]
    dn_f=np.zeros_like(free);dn_f[1:,:]=free[:-1,:]
    lt_f=np.zeros_like(free);lt_f[:,:-1]=free[:,1:]
    rt_f=np.zeros_like(free);rt_f[:,1:]=free[:,:-1]
    boundary=walls&(border|up_f|dn_f|lt_f|rt_f)
    if stride>1:
        ms=np.zeros_like(boundary);ms[::stride,::stride]=True
        boundary=boundary&ms

    ys,xs=np.nonzero(boundary)
    if xs.size==0:
        return (np.zeros((0,2),np.float32),np.zeros((0,),np.float32),
                np.zeros((0,),np.float32),robot_radius*(1+margin_factor))
    if xs.size>max_points:
        idx=np.random.choice(xs.size,max_points,replace=False); xs,ys=xs[idx],ys[idx]

    xs_g=(c0+xs+0.5).astype(np.float32); ys_g=(r0+ys+0.5).astype(np.float32)
    C=np.stack([xs_g,ys_g],axis=-1)
    R_eff=np.full(xs.shape[0],robot_radius+0.5,dtype=np.float32)
    W_arr=np.ones(xs.shape[0],dtype=np.float32)
    d_hat=float(R_eff.max()+margin_factor*robot_radius)
    return C,R_eff,W_arr,d_hat


# ─────────────────────────────────────────────────────────────────────────────
# Local risk patch extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_risk_patch(
    maps: Dict[str,np.ndarray],
    center_rc: Tuple[int,int],
    patch_size: int = PATCH_SIZE,
) -> Tuple[np.ndarray,np.ndarray]:
    """
    (2,P,P) float32:  ch0=r̃(x),  ch1=hard hazard mask.
    Used by RiskPatchEncoder in CoefEnergyNetMaterial (model input).
    Zero-padded outside map bounds.
    """
    risk_map=maps["risk_map"]; hard_mask=maps["hard_mask"]
    H,W=risk_map.shape; cr,cc=center_rc; P=patch_size; half=P//2
    rp=np.zeros((P,P),np.float32); hp=np.zeros((P,P),np.float32)
    r0g=cr-half;r1g=cr-half+P;c0g=cc-half;c1g=cc-half+P
    r0s=max(0,r0g);r1s=min(H,r1g);c0s=max(0,c0g);c1s=min(W,c1g)
    r0d=r0s-r0g;r1d=r0d+(r1s-r0s);c0d=c0s-c0g;c1d=c0d+(c1s-c0s)
    if r1s>r0s and c1s>c0s:
        rp[r0d:r1d,c0d:c1d]=risk_map[r0s:r1s,c0s:c1s]
        hp[r0d:r1d,c0d:c1d]=hard_mask[r0s:r1s,c0s:c1s].astype(np.float32)
    return np.stack([rp,hp],axis=0), hp


def extract_rollout_patch(
    maps: Dict[str,np.ndarray],
    center_rc: Tuple[int,int],
    patch_size: int = PATCH_SIZE,
    sdf_clip: float = 50.0,
) -> np.ndarray:
    """
    (6,P,P) float32 rollout patch used by integrate_surrogate_material.

    Channels (all in col=x, row=y convention):
      ch0  r̃(x)       smoothed risk map            ∈ [0,1]
      ch1  φ(x)        SDF to hard hazard (metres)  ∈ [0, sdf_clip]
      ch2  ∂r̃/∂x      risk gradient, col direction
      ch3  ∂r̃/∂y      risk gradient, row direction
      ch4  ∂φ/∂x       SDF gradient, col direction  (∇φ points toward open space)
      ch5  ∂φ/∂y       SDF gradient, row direction

    Using true oracle sdf_hard and its precomputed Sobel gradients means
    F_mat_hard in the integrator is exact, not approximated via -∇r̃.
    Zero-padded outside map bounds.
    """
    H, W  = maps["risk_map"].shape
    cr, cc = center_rc
    P, half = patch_size, patch_size // 2

    def _crop(arr, fill=0.0):
        out = np.full((P, P), fill, dtype=np.float32)
        r0g = cr-half; r1g = cr-half+P
        c0g = cc-half; c1g = cc-half+P
        r0s = max(0, r0g); r1s = min(H, r1g)
        c0s = max(0, c0g); c1s = min(W, c1g)
        r0d = r0s-r0g;  r1d = r0d+(r1s-r0s)
        c0d = c0s-c0g;  c1d = c0d+(c1s-c0s)
        if r1s > r0s and c1s > c0s:
            out[r0d:r1d, c0d:c1d] = arr[r0s:r1s, c0s:c1s]
        return out

    r_tilde   = _crop(maps["risk_map"])
    sdf       = _crop(maps["sdf_hard"]).clip(0.0, sdf_clip)
    drdc      = _crop(maps["grad_col"])   # ∂r̃/∂x  (col = x)
    drdr      = _crop(maps["grad_row"])   # ∂r̃/∂y  (row = y)
    dsdf_dc   = _crop(maps["sdf_grad_col"])  # ∂φ/∂x
    dsdf_dr   = _crop(maps["sdf_grad_row"])  # ∂φ/∂y

    return np.stack([r_tilde, sdf, drdc, drdr, dsdf_dc, dsdf_dr], axis=0)  # (6,P,P)


# ─────────────────────────────────────────────────────────────────────────────
# Episode sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_episode_pairs(
    maps: Dict[str,np.ndarray],
    labels: np.ndarray,
    num_episodes: int,
    min_dist: int=80, max_dist: int=600,
    min_hazards_on_line: int=2,
    seed: int=42, n_tries: int=30000,
    val_frac: float=0.1, test_frac: float=0.1,
) -> List[Dict]:
    H,W      = labels.shape
    rng      = np.random.default_rng(seed)
    risk_map = maps["risk_map"]; hard=maps["hard_mask"].astype(bool)
    geom     = maps["geom_occ"].astype(bool)

    safe_rc=np.argwhere((labels>0)&(~geom)&(~hard)&(risk_map<0.25))
    if len(safe_rc)<100:
        raise RuntimeError("Too few safe cells.")

    pairs: List[Dict]=[]
    for _ in range(n_tries):
        if len(pairs)>=num_episodes: break
        i,j=rng.integers(len(safe_rc),size=2)
        sr,sc=safe_rc[i]; gr,gc=safe_rc[j]
        dist=((sr-gr)**2+(sc-gc)**2)**0.5
        if not(min_dist<=dist<=max_dist): continue
        n_pts=max(int(dist),2); ts=np.linspace(0,1,n_pts)
        rs=np.clip((sr+ts*(gr-sr)).astype(int),0,H-1)
        cs=np.clip((sc+ts*(gc-sc)).astype(int),0,W-1)
        if int(hard[rs,cs].sum())<min_hazards_on_line: continue
        pairs.append({"start_rc":[int(sr),int(sc)],"goal_rc":[int(gr),int(gc)]})

    if len(pairs)<num_episodes:
        print(f"[WARN] Only {len(pairs)} pairs with hazards; filling remainder.")
        for _ in range(n_tries):
            if len(pairs)>=num_episodes: break
            i,j=rng.integers(len(safe_rc),size=2)
            sr,sc=safe_rc[i]; gr,gc=safe_rc[j]
            dist=((sr-gr)**2+(sc-gc)**2)**0.5
            if not(min_dist<=dist<=max_dist): continue
            pairs.append({"start_rc":[int(sr),int(sc)],"goal_rc":[int(gr),int(gc)]})

    n=len(pairs[:num_episodes])
    n_val=max(1,int(n*val_frac)); n_test=max(1,int(n*test_frac))
    splits=(["val"]*n_val+["test"]*n_test+["train"]*(n-n_val-n_test))
    rng.shuffle(splits)
    for p,s in zip(pairs,splits): p["split"]=s
    return pairs[:num_episodes]


# ─────────────────────────────────────────────────────────────────────────────
# Episode metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_path_metrics(
    path: List[Tuple[int,int]],
    maps: Dict[str,np.ndarray],
    gsd: float=GT_GSD,
) -> Dict[str,float]:
    if not path:
        return dict(path_length_m=0.,risk_exposure_gt=0.,hard_hits=0,mean_rho=0.)
    risk_map=maps["risk_map"]; hard=maps["hard_mask"].astype(bool)
    length=0.;risk_acc=0.;hard_hits=0
    for i in range(len(path)-1):
        r0,c0=path[i];r1,c1=path[i+1]
        step=gsd*_STEP[(r1-r0,c1-c0)]
        length+=step; risk_acc+=step*float(risk_map[r1,c1])
        if hard[r1,c1]: hard_hits+=1
    return dict(path_length_m=float(length),risk_exposure_gt=float(risk_acc),
                hard_hits=int(hard_hits),mean_rho=float(risk_acc/max(length,1e-6)))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pick_local_goal_index(
    path_rc: np.ndarray, ci: int,
    patch_size: int=64, min_ahead: int=3, max_ahead: int=64,
) -> int:
    center=path_rc[ci]; r_max=patch_size/2.0-2.0; T=len(path_rc); last_good=ci
    for j in range(ci+min_ahead,min(ci+max_ahead,T)):
        if float(np.linalg.norm(path_rc[j]-center))<=r_max: last_good=j
        else: break
    if last_good==ci: last_good=min(ci+min_ahead,T-1)
    return last_good


def _path_to_obstacle_primitives(
    path_rc: np.ndarray,
    geom_occ: np.ndarray,
    robot_radius: float,
    patch_size: int=64,
) -> Dict:
    """
    Aggregate full-path geometric obstacles for episode.pt['obstacles'].
    Matches energy_data_navmax.py expected format:
      {centers, radii, weights, d_hat_env}
    """
    all_C=[]; all_R=[]
    stride=max(1,len(path_rc)//50)
    for ci in range(0,len(path_rc),stride):
        cr,cc=int(path_rc[ci,0]),int(path_rc[ci,1])
        C,R,_,_=extract_local_geom_obstacles(geom_occ,(cr,cc),patch_size,robot_radius)
        if C.shape[0]>0: all_C.append(C); all_R.append(R)
    if not all_C:
        return dict(centers=[],radii=[],weights=[],d_hat_env=robot_radius*1.5)
    C_cat=np.concatenate(all_C,axis=0); R_cat=np.concatenate(all_R,axis=0)
    keys=np.round(C_cat).astype(int); _,ui=np.unique(keys,axis=0,return_index=True)
    C_u=C_cat[ui]; R_u=R_cat[ui]
    return dict(centers=C_u.tolist(),radii=R_u.tolist(),
                weights=np.ones(len(ui),dtype=np.float32).tolist(),
                d_hat_env=float(R_u.max()+0.5*robot_radius) if len(R_u) else robot_radius*1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint writer  (stagewise_checkpoints.jsonl)
# ─────────────────────────────────────────────────────────────────────────────

def write_checkpoints_jsonl(
    ck_path: Path,
    path_rc: np.ndarray,
    maps: Dict[str,np.ndarray],
    geom_occ: np.ndarray,
    path_stride: int,
    max_stages: int,
    patch_size_geom: int,
    patch_size_risk: int,
    robot_radius: float,
    base_dt: float,
) -> Tuple[List[Tuple[int,int]],List[Tuple[int,int]]]:
    """
    Write stagewise_checkpoints.jsonl.

    Core stagewise fields:
      t, dt, stage_idx, stage_bounds, stage_entry, stage_exit,
      center, theta, min_d, barrier, obstacles_effective

    Setting 2 material fields (new):
      risk_patch, risk_grad, risk_at_center,
      phaz_at_center (float), entropy_at_center

    Surrogate training targets (new):
      o_tgt, v_tgt
    """
    T=len(path_rc)
    risk_map=maps["risk_map"]; hard_mask=maps["hard_mask"].astype(bool)
    grad_row=maps["grad_row"]; grad_col=maps["grad_col"]
    sdf_hard=maps["sdf_hard"]

    center_idxs=np.arange(0,T,path_stride,dtype=int)
    if center_idxs[-1]!=T-1: center_idxs=np.append(center_idxs,T-1)
    if len(center_idxs)>max_stages:
        sel=np.linspace(0,len(center_idxs)-1,max_stages,dtype=int)
        center_idxs=center_idxs[sel]

    stage_centers_rc: List[Tuple[int,int]]=[]
    stage_exits_rc:   List[Tuple[int,int]]=[]

    with ck_path.open("w") as f:
        for stage_idx,ci in enumerate(center_idxs):
            cr,cc=int(path_rc[ci,0]),int(path_rc[ci,1])
            gi=_pick_local_goal_index(path_rc,ci,patch_size_geom)
            er,ec=int(path_rc[gi,0]),int(path_rc[gi,1])

            half=patch_size_geom//2
            bounds=[max(0,cr-half),min(risk_map.shape[0],cr+half),
                    max(0,cc-half),min(risk_map.shape[1],cc+half)]

            C,R_eff,W_arr,d_hat=extract_local_geom_obstacles(
                geom_occ,(cr,cc),patch_size_geom,robot_radius)

            # min_d: minimum clearance to nearest geometric obstacle
            if C.shape[0]>0:
                dists=np.linalg.norm(C-np.array([float(cc),float(cr)]),axis=1)-R_eff
                min_d=float(dists.min())
            else:
                min_d=float(sdf_hard[cr,cc])

            risk_patch,_=extract_risk_patch(maps,(cr,cc),patch_size_risk)
            rollout_patch=extract_rollout_patch(maps,(cr,cc),patch_size_risk)

            # ∂r̃/∂x=col-direction, ∂r̃/∂y=row-direction
            risk_grad_xy=[float(grad_col[cr,cc]),float(grad_row[cr,cc])]

            # Surrogate targets K steps ahead along reference path
            ti=min(ci+path_stride,T-2)
            o_tgt_xy=[float(path_rc[ti,1]),  float(path_rc[ti,0])]   # (x,y)
            v_tgt_xy=[float((path_rc[ti+1,1]-path_rc[ti,1])/base_dt),
                      float((path_rc[ti+1,0]-path_rc[ti,0])/base_dt)]

            ck={
                # Core stagewise state.
                "t":            int(stage_idx),
                "dt":           float(base_dt),
                "stage_idx":    int(stage_idx),
                "stage_bounds": bounds,
                "stage_entry":  [float(cc),float(cr)],
                "stage_exit":   [float(ec),float(er)],
                "center":       [float(cc),float(cr)],
                "theta":        0.0,
                "min_d":        float(min_d),
                "barrier": {
                    "barrier_d_hat": float(d_hat),
                    "U_barrier":     0.0,
                    "U_bulk":        0.0,
                },
                "obstacles_effective": {
                    "C":     C.tolist(),
                    "R_eff": R_eff.tolist(),
                    "W":     W_arr.tolist(),
                },
                # ── Setting 2 material fields ─────────────────────────────
                "risk_patch":        risk_patch.tolist(),     # (2,P,P) — model encoder input
                "rollout_patch":     rollout_patch.tolist(),  # (6,P,P) — integrator resampling
                                                              # ch0=r̃  ch1=φ  ch2=∂r̃/∂x
                                                              # ch3=∂r̃/∂y  ch4=∂φ/∂x  ch5=∂φ/∂y
                "risk_grad":         risk_grad_xy,
                "risk_at_center":    float(risk_map[cr,cc]),
                "phaz_at_center":    float(hard_mask[cr,cc]),
                "entropy_at_center": 0.0,
                # ── Surrogate training targets ────────────────────────────
                "o_tgt": o_tgt_xy,   # (x,y)
                "v_tgt": v_tgt_xy,   # (vx,vy)
            }
            f.write(json.dumps(ck)+"\n")
            stage_centers_rc.append((cr,cc)); stage_exits_rc.append((er,ec))

    return stage_centers_rc,stage_exits_rc


# ─────────────────────────────────────────────────────────────────────────────
# Episode builder
# ─────────────────────────────────────────────────────────────────────────────

def build_episode(
    ep_dir: Path,
    episode_id: str,
    scene_id: str,
    split: str,
    maps: Dict[str,np.ndarray],
    path_rc: np.ndarray,
    start_rc: Tuple[int,int],
    goal_rc:  Tuple[int,int],
    path_stride: int=6,
    max_stages: int=256,
    patch_size_geom: int=64,
    patch_size_risk: int=PATCH_SIZE,
    base_dt: float=0.04,
    robot_radius: float=1.5,
    gamma_o: float=4.0,
    gsd: float=GT_GSD,
) -> Dict[str,Any]:
    """
    Write ep_<id>/episode.pt + ep_<id>/logs/stagewise_checkpoints.jsonl.
    Returns a manifest record dict.
    """
    ep_dir.mkdir(parents=True,exist_ok=True)
    logs_dir=ep_dir/"logs"; logs_dir.mkdir(exist_ok=True)
    T=len(path_rc)
    if T<3: raise ValueError(f"{episode_id}: path too short ({T})")

    geom_occ=maps["geom_occ"]
    path_list=[(int(r),int(c)) for r,c in path_rc]
    ep_metrics=compute_path_metrics(path_list,maps,gsd)
    obstacles=_path_to_obstacle_primitives(path_rc,geom_occ,robot_radius,patch_size_geom)

    # frames + frame_state  (energy_data_navmax.py compatibility)
    frames=[]; frame_state=[]; prev_ci=0; stage_ctr=0
    for fi,ci in enumerate(range(0,T,max(1,path_stride//2))):
        cr,cc=int(path_rc[ci,0]),int(path_rc[ci,1])
        if ci>0:
            vx=float((path_rc[ci,1]-path_rc[prev_ci,1])/base_dt)
            vy=float((path_rc[ci,0]-path_rc[prev_ci,0])/base_dt)
        else: vx=vy=0.0
        frames.append({"t":fi,"o":[float(cc),float(cr)],
                        "v_o":[vx,vy],"center":[float(cc),float(cr)],"theta":0.0})
        half=patch_size_geom//2
        frame_state.append({
            "t":fi,"stage_idx":stage_ctr,
            "bounds":[max(0,cr-half),min(maps["z2_labels"].shape[0],cr+half),
                      max(0,cc-half),min(maps["z2_labels"].shape[1],cc+half)],
            "entry_point":[float(cc),float(cr)],
            "exit_point": [float(cc),float(cr)],
            "center":     [float(cc),float(cr)],
        })
        stage_ctr=min(stage_ctr+1,max_stages-1); prev_ci=ci

    ck_path=logs_dir/"stagewise_checkpoints.jsonl"
    stage_centers_rc,stage_exits_rc=write_checkpoints_jsonl(
        ck_path,path_rc,maps,geom_occ,
        path_stride,max_stages,patch_size_geom,patch_size_risk,robot_radius,base_dt)

    ep_obj={
        "meta":{
            "episode_id":      episode_id,
            "scene_id":        scene_id,
            "split":           split,
            "start_rc":        list(start_rc),
            "goal_rc":         list(goal_rc),
            "start_xy":        [float(start_rc[1]),float(start_rc[0])],
            "goal_xy":         [float(goal_rc[1]), float(goal_rc[0])],
            "dt":              float(base_dt),
            "robot_radius":    float(robot_radius),
            "patch_size_geom": int(patch_size_geom),
            "patch_size_risk": int(patch_size_risk),
            "path_stride":     int(path_stride),
            "gsd":             float(gsd),
            "total_steps":     int(T),
        },
        "params":{
            "d_hat":      float(obstacles["d_hat_env"]),
            "gamma_o":    float(gamma_o),
            "lambda_mat": 1.0,       # placeholder; tuned during training
            "risk_scale": float(RISK_WEIGHT),
        },
        "obstacles":       obstacles,
        "frames":          frames,
        "frame_state":     frame_state,
        "episode_metrics": ep_metrics,
        "success":         True,
        "final_center":    [float(path_rc[-1,1]),float(path_rc[-1,0])],
        "logs":{"checkpoints_jsonl":str(ck_path)},
    }
    ep_path=ep_dir/"episode.pt"
    torch.save(ep_obj,ep_path)

    return {
        "path":       str(ep_path),
        "episode_id": episode_id,
        "scene_id":   scene_id,
        "split":      split,
        "start_rc":   list(start_rc),
        "goal_rc":    list(goal_rc),
        "success":    True,
        **ep_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scene builder  (scene_<sid>.pt  — stored once per crop)
# ─────────────────────────────────────────────────────────────────────────────

def build_scene(
    out_root: Path,
    scene_id: str,
    labels: np.ndarray,
    crop: Dict,
    maps: Dict[str,np.ndarray],
    gsd: float=GT_GSD,
) -> str:
    """Write scene_<sid>.pt with all shared map arrays."""
    out_root.mkdir(parents=True,exist_ok=True)
    scene_obj={
        "meta":{"scene_id":scene_id,"crop":crop,"gsd":gsd,"shape":list(labels.shape)},
        "maps":maps,
        "risk_ontology":{
            "rho":          {int(k):float(v) for k,v in RHO.items()},
            "hard_classes": sorted(HARD_CLASSES),
            "soft_classes": sorted(SOFT_CLASSES),
            "class_names":  {int(k):v for k,v in CLASS_NAMES.items()},
        },
    }
    scene_path=out_root/f"scene_{scene_id}.pt"
    torch.save(scene_obj,scene_path)
    print(f"  Scene → {scene_path}")
    return str(scene_path)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def render_episode_snapshot(
    labels: np.ndarray,
    maps: Dict[str,np.ndarray],
    path_rc: List[Tuple[int,int]],
    stage_centers_rc: List[Tuple[int,int]],
    stage_exits_rc:   List[Tuple[int,int]],
    start_rc: Tuple[int,int],
    goal_rc:  Tuple[int,int],
    out_path: str,
    episode_id: str="",
    crop_context: int=40,
):
    risk_map=maps["risk_map"]; hard_mask=maps["hard_mask"].astype(bool)
    if path_rc:
        rs=[p[0] for p in path_rc]; cs=[p[1] for p in path_rc]
        r0=max(0,min(rs)-crop_context); r1=min(labels.shape[0],max(rs)+crop_context)
        c0=max(0,min(cs)-crop_context); c1=min(labels.shape[1],max(cs)+crop_context)
    else:
        r0,r1,c0,c1=0,labels.shape[0],0,labels.shape[1]

    lc=labels[r0:r1,c0:c1]; rc2=risk_map[r0:r1,c0:c1]; hc=hard_mask[r0:r1,c0:c1]
    def tc(p): return(p[0]-r0,p[1]-c0)
    path_c=[tc(p) for p in path_rc]
    cent_c=[tc(p) for p in stage_centers_rc]
    exit_c=[tc(p) for p in stage_exits_rc]
    sc=tc(start_rc); gc2=tc(goal_rc)

    cmap=ListedColormap(CLASS_COLORS); norm=BoundaryNorm(range(22),21)
    fig,axes=plt.subplots(1,3,figsize=(21,7))

    def plot_path(ax):
        if not path_c: return
        ax.plot([p[1] for p in path_c],[p[0] for p in path_c],color="#ff6b00",lw=2.2,
                path_effects=[pe.Stroke(linewidth=3.8,foreground="black",alpha=0.45),pe.Normal()])
    def mark_sg(ax):
        ax.plot(sc[1],sc[0],"^",ms=11,color="#00e5ff",zorder=30,
                path_effects=[pe.Stroke(linewidth=2,foreground="black"),pe.Normal()])
        ax.plot(gc2[1],gc2[0],"*",ms=14,color="#ffd60a",zorder=30,
                path_effects=[pe.Stroke(linewidth=2,foreground="black"),pe.Normal()])

    # Panel 1 — GT labels + stages
    ax=axes[0]; ax.imshow(lc,cmap=cmap,norm=norm,interpolation="nearest")
    plot_path(ax); mark_sg(ax)
    stride=max(1,len(exit_c)//20)
    for i,(er,ec) in enumerate(exit_c):
        if i%stride==0: ax.plot(ec,er,"o",ms=4,color="white",alpha=0.7,zorder=20)
    for i,(cr2,cc2) in enumerate(cent_c):
        if i%stride==0: ax.plot(cc2,cr2,"+",ms=5,color="yellow",alpha=0.6,zorder=21)
    ax.set_title(f"GT Labels + Stages [{episode_id}]  ({len(stage_centers_rc)} stages)",
                 fontweight="bold"); ax.axis("off")

    # Panel 2 — risk field
    ax=axes[1]; im=ax.imshow(rc2,cmap="RdYlGn_r",vmin=0,vmax=1,interpolation="bilinear")
    plt.colorbar(im,ax=ax,fraction=0.03,label="r̃(x)")
    plot_path(ax); mark_sg(ax)
    ax.set_title("Smoothed Risk Field r̃(x)",fontweight="bold"); ax.axis("off")

    # Panel 3 — hard hazard overlay
    ax=axes[2]; ax.imshow(lc,cmap=cmap,norm=norm,interpolation="nearest",alpha=0.65)
    rgba=np.zeros((*lc.shape,4),np.float32); rgba[hc]=[0.9,0.05,0.05,0.55]
    ax.imshow(rgba,interpolation="nearest")
    plot_path(ax); mark_sg(ax)
    ax.legend(handles=[mpatches.Patch(color=(0.9,0.05,0.05,0.55),label="Hard hazard"),
                        mpatches.Patch(color="#ff6b00",label="Aware path")],
              fontsize=8,framealpha=0.85,loc="lower right")
    ax.set_title("Hard Hazard Overlay + Path",fontweight="bold"); ax.axis("off")

    fig.suptitle(f"DFC2018 Episode {episode_id}  |  path={len(path_rc)} cells  |  "
                 f"stages={len(stage_centers_rc)}",fontsize=13,fontweight="bold")
    plt.tight_layout(); fig.savefig(out_path,dpi=120,bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dfc2018_dataset(
    labels_full: np.ndarray,
    out_root: str,
    scene_id: str="dfc2018_houston",
    crop: Optional[Dict]=None,
    num_episodes: int=300,
    path_stride: int=6,
    max_stages: int=256,
    patch_size_geom: int=64,
    patch_size_risk: int=PATCH_SIZE,
    sigma_risk: float=2.0,
    robot_radius: float=1.5,
    geom_inflate: int=2,
    base_dt: float=0.04,
    gamma_o: float=4.0,
    min_dist: int=80,
    max_dist: int=600,
    seed: int=42,
    n_snap_viz: int=20,
    gsd: float=GT_GSD,
):
    out_root=Path(out_root)
    ep_root=out_root/"episodes"; snap_dir=out_root/"snapshots"
    ep_root.mkdir(parents=True,exist_ok=True); snap_dir.mkdir(parents=True,exist_ok=True)
    if crop is None:
        crop={"r0":0,"c0":0,"h":labels_full.shape[0],"w":labels_full.shape[1]}

    # 1. Build + save scene maps once
    print("Building scene maps …")
    maps=build_scene_maps(labels_full,sigma=sigma_risk,geom_inflate=geom_inflate,gsd=gsd)
    print(f"  Shape: {labels_full.shape}  hard={maps['hard_mask'].sum():,}  "
          f"geom={maps['geom_occ'].sum():,}")
    build_scene(out_root,scene_id,labels_full,crop,maps,gsd)

    # 2. Sample pairs with train/val/test split
    print(f"Sampling {num_episodes} pairs …")
    pairs=sample_episode_pairs(maps,labels_full,num_episodes,
                                min_dist=min_dist,max_dist=max_dist,seed=seed)
    n_tr=sum(1 for p in pairs if p["split"]=="train")
    n_va=sum(1 for p in pairs if p["split"]=="val")
    n_te=sum(1 for p in pairs if p["split"]=="test")
    print(f"  Found {len(pairs)}  (train={n_tr}  val={n_va}  test={n_te})")

    # 3. Build episodes
    records=[]; n_failed=0
    for ep_i,pair in enumerate(pairs):
        ep_id=f"{ep_i:04d}"
        start_rc=tuple(pair["start_rc"]); goal_rc=tuple(pair["goal_rc"])
        split=pair["split"]

        path=astar_aware(maps,start_rc,goal_rc)
        if path is None or len(path)<3:
            print(f"  [SKIP] {ep_id}: no path"); n_failed+=1; continue

        path_rc=np.array(path,dtype=np.float32)
        try:
            rec=build_episode(
                ep_dir         = ep_root/f"ep_{ep_id}",
                episode_id     = ep_id, scene_id=scene_id, split=split,
                maps=maps, path_rc=path_rc,
                start_rc=start_rc, goal_rc=goal_rc,
                path_stride=path_stride, max_stages=max_stages,
                patch_size_geom=patch_size_geom, patch_size_risk=patch_size_risk,
                base_dt=base_dt, robot_radius=robot_radius,
                gamma_o=gamma_o, gsd=gsd)
            records.append(rec)
        except Exception as e:
            print(f"  [SKIP] {ep_id}: {e}"); n_failed+=1; continue

        if ep_i<n_snap_viz:
            ck_jsonl=ep_root/f"ep_{ep_id}"/"logs"/"stagewise_checkpoints.jsonl"
            centers_rc=[]; exits_rc=[]
            with ck_jsonl.open() as fj:
                for line in fj:
                    ck=json.loads(line)
                    cx,cy=ck["center"]; ex,ey=ck["stage_exit"]
                    centers_rc.append((int(cy),int(cx))); exits_rc.append((int(ey),int(ex)))
            render_episode_snapshot(
                labels_full,maps,[(int(r),int(c)) for r,c in path_rc],
                centers_rc,exits_rc,start_rc,goal_rc,
                out_path=str(snap_dir/f"ep_{ep_id}.png"),episode_id=ep_id)

        if(ep_i+1)%25==0:
            print(f"  [{ep_i+1}/{len(pairs)}]  ok={len(records)}  fail={n_failed}")

    # 4. Manifest
    man_path=out_root/"manifest.json"
    with man_path.open("w") as f: json.dump(records,f,indent=2)
    print(f"\nDone.  {len(records)} episodes  ({n_failed} failed)")
    print(f"  Manifest  → {man_path}")
    print(f"  Snapshots → {snap_dir}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",   default="real",choices=["real","demo"])
    ap.add_argument("--gt",     default=None)
    ap.add_argument("--crop_r0",type=int,default=None)
    ap.add_argument("--crop_c0",type=int,default=None)
    ap.add_argument("--crop_h", type=int,default=None)
    ap.add_argument("--crop_w", type=int,default=None)
    ap.add_argument("--out",    default="data/dfc2018_stagewise")
    ap.add_argument("--num_episodes", type=int,   default=300)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--sigma",        type=float, default=2.0)
    ap.add_argument("--robot_radius", type=float, default=1.5)
    ap.add_argument("--path_stride",  type=int,   default=6)
    ap.add_argument("--max_stages",   type=int,   default=256)
    ap.add_argument("--geom_inflate", type=int,   default=2)
    ap.add_argument("--n_snap_viz",   type=int,   default=20)
    ap.add_argument("--min_dist",     type=int,   default=80)
    ap.add_argument("--max_dist",     type=int,   default=600)
    args=ap.parse_args()

    if args.mode=="demo":
        print("Mode: demo"); labels_full=make_demo_scene(300,600)
        crop={"r0":0,"c0":0,"h":300,"w":600}
    else:
        if not args.gt or not os.path.exists(args.gt):
            raise FileNotFoundError(f"--gt not found: {args.gt}")
        print(f"Loading GT: {args.gt}")
        labels_full=load_gt_labels(args.gt)
        print(f"  Shape: {labels_full.shape}")
        crop={"r0":0,"c0":0,"h":labels_full.shape[0],"w":labels_full.shape[1]}
        if args.crop_r0 is not None:
            r0=args.crop_r0; c0=args.crop_c0 or 0
            r1=r0+(args.crop_h or labels_full.shape[0])
            c1=c0+(args.crop_w or labels_full.shape[1])
            labels_full=labels_full[r0:r1,c0:c1].copy()
            crop={"r0":r0,"c0":c0,"h":r1-r0,"w":c1-c0}
            print(f"  Cropped → {labels_full.shape}")

    build_dfc2018_dataset(
        labels_full=labels_full,out_root=args.out,crop=crop,
        num_episodes=args.num_episodes,sigma_risk=args.sigma,
        robot_radius=args.robot_radius,path_stride=args.path_stride,
        max_stages=args.max_stages,geom_inflate=args.geom_inflate,
        seed=args.seed,n_snap_viz=args.n_snap_viz,
        min_dist=args.min_dist,max_dist=args.max_dist)


if __name__=="__main__":
    main()
