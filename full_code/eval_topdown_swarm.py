#!/usr/bin/env python3
"""
eval_topdown_swarm_sdfstable.py

Stagewise swarm (default N=16, shared goal) navigation demo on topdown PNG maps.

Key revisions:
  1) SDF gradients computed from +distance-to-obstacle (points away from obstacles).
  2) Stage waypoints: corner-aware + clearance-adaptive densification.
  3) Waypoints inflated toward corridor center via SDF gradient ascent.
  4) Local obstacle features use ROI connected-components (disc approximation), not ring sampling.
  5) Optional debug panels for per-stage extracted obstacles in ROI.

Outputs:
  - Per task: GIF (and optional MP4), per-task metrics JSON
  - Aggregate metrics JSON
  - If enabled: taskXX_debug_stage_obstacles/stage_YYY.png + stage_obstacles.json

Examples:
  python eval_topdown_swarm_sdfstable.py --input topdown_map.png --out results --agents 16 --tasks_per_map 6 --debug_stage_obstacles
  python eval_topdown_swarm_sdfstable.py --input maps_dir/ --out results --agents 16 --tasks_per_map 4 --write_mp4

  with learned coefficient model:
  python eval_topdown_swarm.py \
  --input data/ut_campus_mid_share/map/ut_campus_mid_topdown.png \
  --out outputs/results_swarm_demo_GRLSNAM \
  --agents 1 \
  --tasks_per_map 2 \
  --ckpt checkpoints/coef_energy_dungeon_v3/best.pt
"""

import os, glob, json, argparse, time, re, math
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import imageio.v3 as iio
from scipy import ndimage
from heapq import heappush, heappop


# -------------------------
# small helpers
# -------------------------

def mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def safe_name(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)

def clamp_xy(xy: np.ndarray, W: int, H: int) -> np.ndarray:
    x = float(np.clip(xy[0], 0, W - 1))
    y = float(np.clip(xy[1], 0, H - 1))
    return np.array([x, y], dtype=np.float32)

def path_length(path_xy: np.ndarray) -> float:
    if len(path_xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)))

def mean_speed(trajs: np.ndarray, dt: float) -> float:
    if trajs.shape[0] < 2:
        return 0.0
    v = (trajs[1:] - trajs[:-1]) / max(dt, 1e-8)
    return float(np.mean(np.linalg.norm(v, axis=-1)))

def min_pairwise_distance(xyN: np.ndarray) -> float:
    N = xyN.shape[0]
    if N < 2:
        return np.inf
    D2 = ((xyN[:, None, :] - xyN[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(D2, np.inf)
    return float(np.sqrt(np.min(D2)))

def los_clear(a: np.ndarray, b: np.ndarray, occ: np.ndarray, n: int = 96) -> bool:
    H, W = occ.shape
    ts = np.linspace(0.0, 1.0, n)
    pts = (1 - ts)[:, None] * a[None, :] + ts[:, None] * b[None, :]
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    return bool(np.all(occ[ys, xs] == 0))


# -------------------------
# map + SDF (FIXED SIGN)
# -------------------------

class SignedDistanceField:
    """
    occ: uint8 array where 1=obstacle, 0=free
    distance: distance-to-obstacle in free space (0 on obstacles), computed via EDT on free mask.
    gradient: computed from +distance, pointing toward *increasing* clearance (away from obstacles).
    """
    def __init__(self, occ: np.ndarray, grad_smooth_sigma: float = 1.0):
        self.occ = occ.astype(np.uint8)
        self.H, self.W = occ.shape
        free = 1 - self.occ
        self.distance = ndimage.distance_transform_edt(free).astype(np.float32)

        # Smooth distance before differentiating for corner stability.
        if grad_smooth_sigma and grad_smooth_sigma > 0:
            d_sm = ndimage.gaussian_filter(self.distance, sigma=float(grad_smooth_sigma)).astype(np.float32)
        else:
            d_sm = self.distance

        gy, gx = np.gradient(d_sm)  # +distance => away from obstacles
        self.gx = gx.astype(np.float32)
        self.gy = gy.astype(np.float32)

    def get_distance(self, x: float, y: float) -> float:
        ix = int(np.clip(np.round(x), 0, self.W - 1))
        iy = int(np.clip(np.round(y), 0, self.H - 1))
        return float(self.distance[iy, ix])

    def get_gradient(self, x: float, y: float) -> Tuple[float, float]:
        ix = int(np.clip(np.round(x), 0, self.W - 1))
        iy = int(np.clip(np.round(y), 0, self.H - 1))
        gx, gy = float(self.gx[iy, ix]), float(self.gy[iy, ix])
        n = (gx * gx + gy * gy) ** 0.5 + 1e-8
        return gx / n, gy / n


def load_topdown_occ(png_path: str, thresh: int = 128) -> np.ndarray:
    """
    Dark shapes are obstacles, light background is free.
    """
    img = Image.open(png_path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    occ = (arr < thresh).astype(np.uint8)  # 1=wall, 0=free
    return occ


def sample_free_cell_far_from_walls(
    sdf: SignedDistanceField,
    min_clear: float,
    rng: np.random.RandomState,
    max_tries: int = 30000,
) -> np.ndarray:
    H, W = sdf.H, sdf.W
    for _ in range(max_tries):
        x = rng.uniform(0, W - 1)
        y = rng.uniform(0, H - 1)
        if sdf.get_distance(x, y) >= min_clear:
            return np.array([x, y], dtype=np.float32)
    iy, ix = np.unravel_index(np.argmax(sdf.distance), sdf.distance.shape)
    return np.array([float(ix), float(iy)], dtype=np.float32)


def spawn_swarm_starts(
    sdf: SignedDistanceField,
    center: np.ndarray,
    N: int,
    radius: float,
    min_clear: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    starts = []
    H, W = sdf.H, sdf.W
    for _ in range(80000):
        if len(starts) >= N:
            break
        ang = rng.uniform(0, 2 * np.pi)
        r = radius * (0.2 + 0.8 * rng.rand())
        xy = center + np.array([r * np.cos(ang), r * np.sin(ang)], dtype=np.float32)
        xy = clamp_xy(xy, W, H)
        if sdf.get_distance(xy[0], xy[1]) >= min_clear:
            starts.append(xy)
    if len(starts) < N:
        while len(starts) < N:
            starts.append(sample_free_cell_far_from_walls(sdf, min_clear, rng))
    return np.stack(starts, axis=0).astype(np.float32)


# -------------------------
# A* and robust stagewise waypoints
# -------------------------

def astar_path_grid(occ: np.ndarray, start_yx: Tuple[int, int], goal_yx: Tuple[int, int]) -> List[Tuple[int, int]]:
    H, W = occ.shape
    sy, sx = start_yx
    gy, gx = goal_yx
    sy, sx = int(np.clip(sy, 0, H - 1)), int(np.clip(sx, 0, W - 1))
    gy, gx = int(np.clip(gy, 0, H - 1)), int(np.clip(gx, 0, W - 1))
    if occ[sy, sx] or occ[gy, gx]:
        return []

    def heur(y, x):
        return ((y - gy) ** 2 + (x - gx) ** 2) ** 0.5

    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]

    open_set = [(heur(sy, sx), 0.0, sy, sx)]
    parent = {}
    gscore = {(sy, sx): 0.0}
    closed = set()

    while open_set:
        _, g, y, x = heappop(open_set)
        if (y, x) in closed:
            continue
        closed.add((y, x))
        if (y, x) == (gy, gx):
            path = []
            cur = (gy, gx)
            while cur in parent:
                path.append(cur)
                cur = parent[cur]
            path.append((sy, sx))
            return path[::-1]
        for dy, dx, c in nbrs:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            if occ[ny, nx] != 0:
                continue
            ng = g + c
            if ng < gscore.get((ny, nx), 1e18):
                gscore[(ny, nx)] = ng
                parent[(ny, nx)] = (y, x)
                heappush(open_set, (ng + heur(ny, nx), ng, ny, nx))
    return []


def rdp_simplify(points: np.ndarray, eps: float) -> np.ndarray:
    if len(points) <= 2:
        return points

    def perp_dist(p, a, b):
        ab = b - a
        ap = p - a
        denom = float(np.dot(ab, ab)) + 1e-8
        t = float(np.dot(ap, ab)) / denom
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    a = points[0]
    b = points[-1]
    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        d = perp_dist(points[i], a, b)
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        left = rdp_simplify(points[: idx + 1], eps)
        right = rdp_simplify(points[idx:], eps)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([a, b])


def insert_corner_waypoints(points: np.ndarray, theta_deg: float = 30.0) -> np.ndarray:
    if len(points) < 3:
        return points
    th = math.radians(theta_deg)
    out = [points[0]]
    for i in range(1, len(points) - 1):
        p0, p1, p2 = points[i - 1], points[i], points[i + 1]
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = np.linalg.norm(v1) + 1e-8
        n2 = np.linalg.norm(v2) + 1e-8
        c = float(np.dot(v1, v2) / (n1 * n2))
        c = max(-1.0, min(1.0, c))
        ang = math.acos(c)
        if ang > th:
            out.append(p0 + 0.67 * (p1 - p0))
            out.append(p1)
            out.append(p1 + 0.33 * (p2 - p1))
        else:
            out.append(p1)
    out.append(points[-1])
    clean = [out[0]]
    for p in out[1:]:
        if np.linalg.norm(p - clean[-1]) > 1.0:
            clean.append(p)
    return np.stack(clean, axis=0).astype(np.float32)


def densify_by_clearance(points: np.ndarray, sdf: SignedDistanceField, alpha: float = 1.5, seg_min: float = 8.0) -> np.ndarray:
    if len(points) < 2:
        return points
    out = [points[0]]
    for i in range(len(points) - 1):
        a = out[-1]
        b = points[i + 1]
        mid = 0.5 * (a + b)
        d = sdf.get_distance(mid[0], mid[1])
        L = float(np.linalg.norm(b - a))
        Lmax = max(seg_min, alpha * max(d, 1.0))
        if L <= Lmax:
            out.append(b)
            continue
        m = int(math.ceil(L / max(Lmax, 1e-6)))
        for k in range(1, m + 1):
            out.append(a + (k / m) * (b - a))
    clean = [out[0]]
    for p in out[1:]:
        if np.linalg.norm(p - clean[-1]) > 0.75:
            clean.append(p)
    return np.stack(clean, axis=0).astype(np.float32)


def inflate_waypoints(points: np.ndarray, sdf: SignedDistanceField, steps: int = 12, eta: float = 2.0, min_clear: float = 3.0) -> np.ndarray:
    H, W = sdf.H, sdf.W
    out = points.copy().astype(np.float32)
    for i in range(len(out)):
        x = out[i].copy()
        for _ in range(steps):
            d = sdf.get_distance(x[0], x[1])
            gx, gy = sdf.get_gradient(x[0], x[1])  # away from obstacles
            step = eta * (1.5 if d < min_clear else 1.0)
            x = x + step * np.array([gx, gy], dtype=np.float32)
            x = clamp_xy(x, W, H)
        out[i] = x
    return out


@dataclass
class Stage:
    id: int
    center: Tuple[float, float]
    bounds: Tuple[int, int, int, int]  # (x_min, x_max, y_min, y_max)
    entry_point: Tuple[float, float]
    exit_point: Tuple[float, float]
    width: int


class StageManager:
    def __init__(self, stages: List[Stage]):
        self.stages = stages
        self.current_stage_id = 0

    def get_current_goal(self) -> np.ndarray:
        st = self.stages[min(self.current_stage_id, len(self.stages) - 1)]
        return np.array(st.exit_point, dtype=np.float32)

    def try_advance(self, pos: np.ndarray, threshold: float = 6.0) -> bool:
        if self.current_stage_id >= len(self.stages) - 1:
            return False
        g = self.get_current_goal()
        if np.linalg.norm(pos - g) < threshold:
            self.current_stage_id += 1
            return True
        return False


def build_stages_from_astar_robust(
    occ: np.ndarray,
    occ_plan: np.ndarray,
    sdf: SignedDistanceField,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    stage_width: int = 120,
    eps_rdp: float = 3.5,
    theta_deg: float = 32.0,
    densify_alpha: float = 1.5,
    densify_min: float = 10.0,
    inflate_steps: int = 12,
    inflate_eta: float = 2.0,
    inflate_min_clear: float = 3.0,
    max_waypoints: int = 40,
) -> List[Stage]:
    H, W = occ.shape
    start_grid = (int(np.round(start_xy[1])), int(np.round(start_xy[0])))
    goal_grid = (int(np.round(goal_xy[1])), int(np.round(goal_xy[0])))

    path = astar_path_grid(occ_plan, start_grid, goal_grid)
    if not path:
        return [Stage(0, (float(start_xy[0]), float(start_xy[1])), (0, W, 0, H),
                      (float(start_xy[0]), float(start_xy[1])),
                      (float(goal_xy[0]), float(goal_xy[1])), stage_width)]

    path_xy = np.array([[x + 0.5, y + 0.5] for (y, x) in path], dtype=np.float32)

    simp = rdp_simplify(path_xy, eps=eps_rdp).astype(np.float32)
    simp = insert_corner_waypoints(simp, theta_deg=theta_deg)
    dense = densify_by_clearance(simp, sdf, alpha=densify_alpha, seg_min=densify_min)

    if len(dense) > max_waypoints:
        idx = np.linspace(0, len(dense) - 1, max_waypoints).astype(int)
        dense = dense[idx]

    inflated = inflate_waypoints(dense, sdf, steps=inflate_steps, eta=inflate_eta, min_clear=inflate_min_clear)

    half = stage_width // 2
    stages: List[Stage] = []
    for i in range(len(inflated) - 1):
        cx, cy = inflated[i]
        nx, ny = inflated[i + 1]
        x_min = int(np.clip(cx - half, 0, W))
        x_max = int(np.clip(cx + half, 0, W))
        y_min = int(np.clip(cy - half, 0, H))
        y_max = int(np.clip(cy + half, 0, H))
        stages.append(Stage(
            id=i,
            center=(float(cx), float(cy)),
            bounds=(x_min, x_max, y_min, y_max),
            entry_point=(float(cx), float(cy)),
            exit_point=(float(nx), float(ny)),
            width=stage_width,
        ))
    cx, cy = inflated[-1]
    stages.append(Stage(
        id=len(inflated) - 1,
        center=(float(cx), float(cy)),
        bounds=(0, W, 0, H),
        entry_point=(float(cx), float(cy)),
        exit_point=(float(goal_xy[0]), float(goal_xy[1])),
        width=stage_width,
    ))
    return stages


# -------------------------
# Local obstacle extraction: ROI connected-components -> discs
# -------------------------

def extract_obstacles_roi_discs(
    occ: np.ndarray,
    x0: int, x1: int, y0: int, y1: int,
    *,
    max_discs: int = 60,
    min_area: int = 10,
    radius_floor: float = 2.0,
) -> Dict[str, np.ndarray]:
    H, W = occ.shape
    x0 = int(np.clip(x0, 0, W)); x1 = int(np.clip(x1, 0, W))
    y0 = int(np.clip(y0, 0, H)); y1 = int(np.clip(y1, 0, H))
    sub = occ[y0:y1, x0:x1].astype(np.uint8)
    if sub.size == 0:
        return {"centers": np.zeros((0, 2), np.float32),
                "radii": np.zeros((0,), np.float32),
                "weights": np.zeros((0,), np.float32)}

    lab, n = ndimage.label(sub > 0)
    if n == 0:
        return {"centers": np.zeros((0, 2), np.float32),
                "radii": np.zeros((0,), np.float32),
                "weights": np.zeros((0,), np.float32)}

    centers, radii, weights = [], [], []
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        area = len(xs)
        if area < min_area:
            continue
        pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)  # ROI coords
        mu = pts.mean(axis=0)
        X = pts - mu[None, :]
        C = (X.T @ X) / max(area, 1)
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        major = float(max(vals[0], 1e-6)) ** 0.5
        minor = float(max(vals[1], 1e-6)) ** 0.5
        eqr = float(max(radius_floor, math.sqrt(area / math.pi)))
        L = 4.0 * major
        n_disc = int(np.clip(math.ceil(L / max(2.0 * eqr, 1e-6)), 1, 6))
        axis = vecs[:, 0]
        for j in range(n_disc):
            t = 0.0 if n_disc == 1 else (-0.5 + j / (n_disc - 1))
            c = mu + (t * L) * axis
            cx = float(c[0] + x0 + 0.5)
            cy = float(c[1] + y0 + 0.5)
            r = float(max(radius_floor, 1.25 * max(minor, eqr * 0.6)))
            centers.append([cx, cy])
            radii.append(r)
            weights.append(1.0)

    if len(centers) == 0:
        return {"centers": np.zeros((0, 2), np.float32),
                "radii": np.zeros((0,), np.float32),
                "weights": np.zeros((0,), np.float32)}

    centers = np.array(centers, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)

    if len(centers) > max_discs:
        idx = np.argsort(-radii)[:max_discs]
        centers, radii, weights = centers[idx], radii[idx], weights[idx]

    return {"centers": centers, "radii": radii, "weights": weights}


def extract_local_obstacles_from_occ(
    occ: np.ndarray,
    pos: np.ndarray,
    *,
    radius: float = 36.0,
    n_samples: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = float(pos[0]), float(pos[1])
    x0, x1 = int(round(x - radius)), int(round(x + radius))
    y0, y1 = int(round(y - radius)), int(round(y + radius))
    pack = extract_obstacles_roi_discs(occ, x0, x1, y0, y1, max_discs=60)
    C, R, Ww = pack["centers"], pack["radii"], pack["weights"]
    if C.shape[0] == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    d = np.linalg.norm(C - pos[None, :], axis=1)
    idx = np.argsort(d)[:min(n_samples, len(d))]
    return C[idx].astype(np.float32), R[idx].astype(np.float32), Ww[idx].astype(np.float32)


# -------------------------
# Optional coefficient model (kept for compatibility)
# -------------------------

class ObstacleEncoder(torch.nn.Module):
    def __init__(self, d_in=6, d_tok=64):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_in, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, d_tok)
        )
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B, N = feats.shape[0], feats.shape[1]
        if N == 0:
            return feats.new_zeros(B, 0, 64)
        x = feats.reshape(B * N, feats.shape[-1])
        z = self.mlp(x).reshape(B, N, -1)
        return z

class CoefEnergyNet(torch.nn.Module):
    """
    Returns (alphas, beta, gamma) -> used as scales for barrier/goal/damping.
    """
    def __init__(self, d_obs=6, d_goal=4, d_tok=64):
        super().__init__()
        self.obs_enc = ObstacleEncoder(d_in=d_obs, d_tok=d_tok)
        self.goal_enc = torch.nn.Sequential(
            torch.nn.Linear(d_goal, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, d_tok)
        )
        enc = torch.nn.TransformerEncoderLayer(
            d_model=d_tok, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.fuser = torch.nn.TransformerEncoder(enc, num_layers=2)
        self.alpha_head = torch.nn.Sequential(torch.nn.Linear(d_tok, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        self.beta_head  = torch.nn.Sequential(torch.nn.Linear(d_tok, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        self.gamma_head = torch.nn.Sequential(torch.nn.Linear(d_tok, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

    def forward(self, obs_feats: torch.Tensor, obs_mask: torch.Tensor, goal_feats: torch.Tensor):
        B, N = obs_feats.shape[0], obs_feats.shape[1]
        z_goal = self.goal_enc(goal_feats).unsqueeze(1)
        if N == 0:
            tokens = z_goal
            pad = torch.zeros(B, 1, dtype=torch.bool, device=obs_feats.device)
            z_all = self.fuser(tokens, src_key_padding_mask=pad)
            ctx = z_all[:, 0]
            alphas = obs_feats.new_zeros(B, 0)
        else:
            z_obs = self.obs_enc(obs_feats)
            tokens = torch.cat([z_goal, z_obs], dim=1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=obs_mask.device), ~obs_mask], dim=1)
            z_all = self.fuser(tokens, src_key_padding_mask=pad)
            ctx = z_all[:, 0]
            a = F.softplus(self.alpha_head(z_all[:, 1:]).squeeze(-1))
            alphas = torch.where(obs_mask, a, torch.zeros_like(a))
        beta  = F.softplus(self.beta_head(ctx)).squeeze(-1)
        gamma = F.softplus(self.gamma_head(ctx)).squeeze(-1)
        return alphas, beta, gamma


def build_local_feats(pos: np.ndarray, goal: np.ndarray, C: np.ndarray, R: np.ndarray, Ww: np.ndarray):
    o = torch.as_tensor(pos, dtype=torch.float32)
    g = torch.as_tensor(goal, dtype=torch.float32)

    if C.size == 0:
        obs_feats = torch.zeros(1, 0, 6)
    else:
        C_t = torch.as_tensor(C, dtype=torch.float32)
        R_t = torch.as_tensor(R, dtype=torch.float32)
        W_t = torch.as_tensor(Ww, dtype=torch.float32)
        dxdy = g.unsqueeze(0) - C_t
        obs_feats = torch.cat([C_t, R_t.unsqueeze(-1), W_t.unsqueeze(-1), dxdy], dim=-1).unsqueeze(0)

    dg = g - o
    gdist = torch.linalg.norm(dg).unsqueeze(0)
    goal_feats = torch.stack([dg[0], dg[1], gdist[0], torch.tensor(1.0)], dim=0).unsqueeze(0)
    return obs_feats, goal_feats


# -------------------------
# swarm rollout
# -------------------------

def step_force_policy(
    pos: np.ndarray,
    vel: np.ndarray,
    goal: np.ndarray,
    sdf: SignedDistanceField,
    *,
    k_goal: float,
    k_bar: float,
    gamma: float,
    d_hat: float,
) -> np.ndarray:
    # goal attraction
    to_goal = goal - pos
    d = np.linalg.norm(to_goal) + 1e-8
    F_goal = k_goal * (to_goal / d)

    # wall repulsion: SDF gradient points away from obstacles
    d_wall = sdf.get_distance(pos[0], pos[1])
    F_bar = np.zeros(2, np.float32)
    if d_wall < d_hat:
        gx, gy = sdf.get_gradient(pos[0], pos[1])
        z = max(d_wall / max(d_hat, 1e-6), 1e-6)
        barrier = (1.0 - z) ** 2 * (-math.log(z))
        F_bar = (k_bar * float(barrier)) * np.array([gx, gy], dtype=np.float32)

    # damping
    F_damp = -gamma * vel
    return (F_goal + F_bar + F_damp).astype(np.float32)


def rollout_swarm_stagewise(
    occ: np.ndarray,
    sdf: SignedDistanceField,
    starts_xy: np.ndarray,      # (N,2)
    goal_xy: np.ndarray,        # (2,)
    stages: List,
    *,
    model: Optional[CoefEnergyNet],
    device: str,
    T: int = 900,
    dt: float = 0.35,
    v_max: float = 3.0,
    goal_tol: float = 8.0,
    stage_tol: float = 10.0,
    # base gains
    k_goal_base: float = 20.0,
    k_bar_base: float = 8.0,
    d_hat: float = 5.0,
    # swarm coupling
    w_sep: float = 1.2,
    w_coh: float = 0.15,
    w_align: float = 0.10,
    sep_dist: float = 10.0,
    coh_dist: float = 35.0,
    # local obstacle extraction
    local_obs_radius: float = 36.0,
    local_obs_n: int = 16,
) -> Dict[str, np.ndarray]:
    N = starts_xy.shape[0]
    pos = starts_xy.copy().astype(np.float32)
    vel = np.zeros((N, 2), dtype=np.float32)

    # Allow either shared stages (List[Stage]) or per-agent stages (List[List[Stage]]).
    if len(stages) > 0 and isinstance(stages[0], Stage):
        stages_list = [stages for _ in range(N)]
    else:
        stages_list = stages
        if len(stages_list) != N:
            raise ValueError(f"Expected per-agent stages_list of length N={N}, got {len(stages_list)}")

    stage_mgrs = [StageManager(stages_list[i]) for i in range(N)]
    traj = np.zeros((T + 1, N, 2), dtype=np.float32)
    traj[0] = pos

    cache = [{"t": -999, "p": None, "sid": -1, "k_goal": 1.0, "k_bar": 1.0, "gamma": 0.0} for _ in range(N)]
    obs_update_interval = 12
    cache_dist_thresh = 9.0

    t0 = time.time()
    for t in range(T):
        v_mean = np.mean(vel, axis=0)
        F_all = np.zeros((N, 2), dtype=np.float32)

        # per-agent base forces (stagewise)
        for i in range(N):
            stage_goal = stage_mgrs[i].get_current_goal()

            # learned scaling (optional)
            k_goal_scale, k_bar_scale, gamma = cache[i]["k_goal"], cache[i]["k_bar"], cache[i]["gamma"]
            need_update = (
                (model is not None) and (
                    (t - cache[i]["t"]) >= obs_update_interval or
                    cache[i]["p"] is None or
                    stage_mgrs[i].current_stage_id != cache[i]["sid"] or
                    np.linalg.norm(pos[i] - cache[i]["p"]) > cache_dist_thresh
                )
            )
            if need_update:
                C, R, Ww = extract_local_obstacles_from_occ(
                    occ, pos[i],
                    radius=local_obs_radius,
                    n_samples=local_obs_n,
                )
                obs_feats, goal_feats = build_local_feats(pos[i], stage_goal, C, R, Ww)
                obs_mask = (torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)
                            if obs_feats.shape[1] else torch.zeros(1, 0, dtype=torch.bool, device=device))
                with torch.no_grad():
                    #print("Here", i, "stage", stage_mgrs[i].current_stage_id, "obs", obs_feats.shape[1], "goal", goal_feats.shape[1])
                    alphas, beta, gam = model(obs_feats.to(device), obs_mask, goal_feats.to(device))
                k_bar_scale = float(alphas.mean().item()) if alphas.numel() else 1.0
                k_goal_scale = float(beta.item())
                gamma = float(gam.item())
                cache[i].update({"t": t, "p": pos[i].copy(), "sid": stage_mgrs[i].current_stage_id,
                                 "k_goal": k_goal_scale, "k_bar": k_bar_scale, "gamma": gamma})

            F_all[i] = step_force_policy(
                pos[i], vel[i], stage_goal, sdf,
                k_goal=k_goal_base * k_goal_scale,
                k_bar=k_bar_base * k_bar_scale,
                gamma=gamma,
                d_hat=d_hat,
            )

        # swarm coupling (separation/cohesion/alignment)
        for i in range(N):
            rij = pos[i][None, :] - pos
            dij = np.linalg.norm(rij, axis=1) + 1e-6

            mask_sep = (dij < sep_dist) & (dij > 1e-3)
            if np.any(mask_sep):
                dir = (rij[mask_sep] / dij[mask_sep][:, None])
                F_sep = np.sum(dir * (sep_dist - dij[mask_sep])[:, None], axis=0)
            else:
                F_sep = np.zeros(2, np.float32)

            mask_coh = (dij < coh_dist) & (dij > 1e-3)
            if np.any(mask_coh):
                centroid = np.mean(pos[mask_coh], axis=0)
                F_coh = (centroid - pos[i])
            else:
                F_coh = np.zeros(2, np.float32)

            F_align = (v_mean - vel[i])
            F_all[i] += (w_sep * F_sep + w_coh * F_coh + w_align * F_align).astype(np.float32)

        # integrate
        vel = vel + dt * F_all
        sp = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-8
        vel = np.where(sp > v_max, vel * (v_max / sp), vel)
        new_pos = pos + dt * vel

        # collision response
        for i in range(N):
            if sdf.get_distance(new_pos[i, 0], new_pos[i, 1]) > 0.75:
                pos[i] = new_pos[i]
            else:
                vel[i] *= -0.35

        traj[t + 1] = pos

        # advance stages and stop condition
        for i in range(N):
            stage_mgrs[i].try_advance(pos[i], threshold=stage_tol)

        dgoal = np.linalg.norm(pos - goal_xy[None, :], axis=1)
        if float(np.mean(dgoal < goal_tol)) >= 0.85:
            traj = traj[: t + 2]
            break

    t1 = time.time()
    return {"traj": traj, "runtime_s": np.array([t1 - t0], dtype=np.float32)}


# -------------------------
# metrics + rendering
# -------------------------

def compute_swarm_metrics(
    occ: np.ndarray,
    sdf: SignedDistanceField,
    traj: np.ndarray,   # (T,N,2)
    goal_xy: np.ndarray,
    *,
    goal_tol: float,
    dt: float,
) -> Dict:
    T, N = traj.shape[0], traj.shape[1]
    final = traj[-1]
    dgoal = np.linalg.norm(final - goal_xy[None, :], axis=1)
    success_mask = (dgoal <= goal_tol)
    L = np.array([path_length(traj[:, i, :]) for i in range(N)], dtype=np.float32)

    clears = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        for i in range(N):
            clears[t, i] = sdf.get_distance(traj[t, i, 0], traj[t, i, 1])

    los_to_goal = np.array([1.0 if los_clear(final[i], goal_xy, occ) else 0.0 for i in range(N)], dtype=np.float32)
    mpd = np.array([min_pairwise_distance(traj[t]) for t in range(T)], dtype=np.float32)

    return {
        "N": int(N),
        "T": int(T),
        "dt": float(dt),
        "goal_tol": float(goal_tol),

        "success_rate": float(np.mean(success_mask)),
        "success_count": int(np.sum(success_mask)),
        "mean_final_dist": float(np.mean(dgoal)),
        "p90_final_dist": float(np.percentile(dgoal, 90)),

        "mean_path_length": float(np.mean(L)),
        "p90_path_length": float(np.percentile(L, 90)),

        "min_clearance": float(np.min(clears)),
        "p10_clearance": float(np.percentile(clears, 10)),
        "barrier_viol_rate(d<2.0)": float(np.mean(clears < 2.0)),
        "tube_viol_rate(d<1.0)": float(np.mean(clears < 1.0)),

        "mean_speed": mean_speed(traj, dt),
        "min_pairwise_dist_min": float(np.min(mpd)),
        "min_pairwise_dist_p10": float(np.percentile(mpd, 10)),

        "los_to_goal_rate": float(np.mean(los_to_goal)),
    }


def _fig_to_rgb(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]  # drop A
    return img


def render_swarm_gif(
    png_path: str,
    occ: np.ndarray,
    traj: np.ndarray,
    stages: List[Stage],
    starts: np.ndarray,
    goal: np.ndarray,
    out_gif: str,
    *,
    out_mp4: Optional[str] = None,
    every: int = 2,
    dpi: int = 140,
):
    H, W = occ.shape
    bg = np.array(Image.open(png_path).convert("RGB"))
    frames = []
    T, N = traj.shape[0], traj.shape[1]
    stride = max(1, len(stages) // 9)

    for t in range(0, T, every):
        fig = plt.figure(figsize=(8, 8), dpi=dpi)
        ax = plt.gca()
        ax.imshow(bg)
        ax.set_xlim([0, W])
        ax.set_ylim([H, 0])
        ax.set_title(f"swarm stagewise   t={t}/{T-1}")

        # for st in stages[::stride]:
        #     x0, x1, y0, y1 = st.bounds
        #     rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=1.0)
        #     ax.add_patch(rect)

        for i in range(N):
            ax.plot(traj[:t+1, i, 0], traj[:t+1, i, 1], linewidth=1.0)

        ax.scatter(starts[:, 0], starts[:, 1], s=20)
        ax.scatter([goal[0]], [goal[1]], s=90, marker="*", edgecolors="k")
        ax.axis("off")

        frames.append(_fig_to_rgb(fig))
        plt.close(fig)

    iio.imwrite(out_gif, frames, duration=0.06)

    if out_mp4 is not None:
        try:
            iio.imwrite(out_mp4, frames, fps=16)
        except Exception as e:
            print(f"[WARN] MP4 write failed ({e}); GIF is still saved.")


def save_stage_obstacle_debug(
    png_path: str,
    occ: np.ndarray,
    stages: List[Stage],
    out_dir: str,
    *,
    pad: int = 80,
    max_stages: int = 40,
    max_discs: int = 60,
):
    mkdir(out_dir)
    bg = np.array(Image.open(png_path).convert("RGB"))
    H, W = occ.shape
    meta = []

    K = min(len(stages), max_stages)
    for i in range(K):
        st = stages[i]
        x0, x1, y0, y1 = st.bounds
        rx0 = max(0, x0 - pad); rx1 = min(W, x1 + pad)
        ry0 = max(0, y0 - pad); ry1 = min(H, y1 + pad)

        pack = extract_obstacles_roi_discs(occ, rx0, rx1, ry0, ry1, max_discs=max_discs)
        C, R = pack["centers"], pack["radii"]

        meta.append({
            "stage_id": int(st.id),
            "stage_bounds": [int(x0), int(x1), int(y0), int(y1)],
            "roi_bounds": [int(rx0), int(rx1), int(ry0), int(ry1)],
            "n_discs": int(C.shape[0]),
            "centers": C.tolist(),
            "radii": R.tolist(),
        })

        fig = plt.figure(figsize=(7.2, 7.2), dpi=150)
        ax = plt.gca()
        ax.imshow(bg[ry0:ry1, rx0:rx1])
        ax.set_title(f"stage {i}/{len(stages)-1}  discs={C.shape[0]}")

        rect = patches.Rectangle((x0 - rx0, y0 - ry0), x1 - x0, y1 - y0, fill=False, linewidth=2.0)
        ax.add_patch(rect)

        for k in range(C.shape[0]):
            cx, cy = C[k]
            circ = patches.Circle((cx - rx0, cy - ry0), radius=float(R[k]), fill=False, linewidth=1.0)
            ax.add_patch(circ)

        ex, ey = st.exit_point
        ax.scatter([ex - rx0], [ey - ry0], s=60, marker="*", edgecolors="k")
        ax.axis("off")

        img = _fig_to_rgb(fig)
        plt.close(fig)

        out_png = os.path.join(out_dir, f"stage_{i:03d}.png")
        iio.imwrite(out_png, img)

    with open(os.path.join(out_dir, "stage_obstacles.json"), "w") as f:
        json.dump(meta, f, indent=2)


# -------------------------
# task generation
# -------------------------

def make_task_pairs_stagewise(
    sdf: SignedDistanceField,
    *,
    num_tasks: int,
    N: int,
    rng: np.random.RandomState,
    start_clear: float = 6.0,
    goal_clear: float = 8.0,
    start_swarm_radius: float = 30.0,
    min_start_goal_sep_frac: float = 0.45,
) -> List[Dict]:
    tasks = []
    diag = float((sdf.W ** 2 + sdf.H ** 2) ** 0.5)
    for _ in range(num_tasks):
        goal = sample_free_cell_far_from_walls(sdf, goal_clear, rng)
        start_center = sample_free_cell_far_from_walls(sdf, start_clear, rng)

        tries = 0
        while np.linalg.norm(goal - start_center) < min_start_goal_sep_frac * diag and tries < 50:
            start_center = sample_free_cell_far_from_walls(sdf, start_clear, rng)
            tries += 1

        starts = spawn_swarm_starts(sdf, start_center, N=N, radius=start_swarm_radius,
                                    min_clear=start_clear, rng=rng)
        tasks.append({
            "start_center": start_center.tolist(),
            "starts": starts.tolist(),
            "goal": goal.tolist(),
        })
    return tasks

def sample_starts_poisson(
    sdf: SignedDistanceField,
    *,
    N: int,
    rng: np.random.RandomState,
    min_clear: float = 6.0,
    min_sep: float = 10.0,
    max_tries: int = 200000,
) -> np.ndarray:
    """
    Poisson-like rejection sampling in free space:
      - each start has clearance >= min_clear
      - pairwise distance between starts >= min_sep (best-effort)
    """
    starts = []
    W, H = sdf.W, sdf.H
    tries = 0
    while len(starts) < N and tries < max_tries:
        tries += 1
        x = rng.uniform(0, W - 1)
        y = rng.uniform(0, H - 1)
        if sdf.get_distance(x, y) < min_clear:
            continue
        p = np.array([x, y], dtype=np.float32)
        if starts:
            dmin = float(np.min([np.linalg.norm(p - q) for q in starts]))
            if dmin < min_sep:
                continue
        starts.append(p)

    # if the map is dense, finish without sep constraint
    while len(starts) < N:
        starts.append(sample_free_cell_far_from_walls(sdf, min_clear, rng))

    return np.stack(starts, axis=0).astype(np.float32)



def make_shared_goal_agents_task(
    sdf: SignedDistanceField,
    *,
    N: int,
    rng: np.random.RandomState,
    start_clear: float = 6.0,
    goal_clear: float = 8.0,
    min_start_goal_sep_frac: float = 0.40,
    min_sep: float = 10.0,
) -> Dict:
    """
    One task: N independent random starts, one shared goal.
    """
    diag = float((sdf.W ** 2 + sdf.H ** 2) ** 0.5)
    goal = sample_free_cell_far_from_walls(sdf, goal_clear, rng)

    starts = sample_starts_poisson(
        sdf, N=N, rng=rng,
        min_clear=start_clear,
        min_sep=min_sep,
    )

    # Avoid trivial starts too close to goal (best effort)
    tries = 0
    while np.mean(np.linalg.norm(starts - goal[None, :], axis=1)) < min_start_goal_sep_frac * diag and tries < 20:
        starts = sample_starts_poisson(
            sdf, N=N, rng=rng,
            min_clear=start_clear,
            min_sep=min_sep,
        )
        tries += 1

    return {"starts": starts.tolist(), "goal": goal.tolist()}

# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="PNG path or directory of PNGs")
    ap.add_argument("--out", type=str, default="results_topdown_swarm")

    ap.add_argument("--ckpt", type=str, default="", help="Optional CoefEnergyNet checkpoint (.pt)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--agents", type=int, default=16)
    ap.add_argument("--tasks_per_map", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--goal_tol", type=float, default=8.0)
    ap.add_argument("--stage_tol", type=float, default=10.0)
    ap.add_argument("--stage_width", type=int, default=160)

    ap.add_argument("--thresh", type=int, default=128, help="occupancy threshold: <thresh => obstacle")
    ap.add_argument("--inflate_cells", type=int, default=2, help="obstacle inflation for planning safety")

    ap.add_argument("--rdp_eps", type=float, default=3.5)
    ap.add_argument("--corner_theta_deg", type=float, default=32.0)
    ap.add_argument("--densify_alpha", type=float, default=1.5)
    ap.add_argument("--densify_min", type=float, default=10.0)
    ap.add_argument("--inflate_steps", type=int, default=12)
    ap.add_argument("--inflate_eta", type=float, default=2.0)
    ap.add_argument("--inflate_min_clear", type=float, default=3.0)
    ap.add_argument("--max_waypoints", type=int, default=120)

    ap.add_argument("--local_obs_radius", type=float, default=36.0)
    ap.add_argument("--local_obs_n", type=int, default=16)

    ap.add_argument("--write_mp4", action="store_true")

    ap.add_argument("--debug_stage_obstacles", action="store_true")
    ap.add_argument("--debug_stage_obstacles_max", type=int, default=40)
    ap.add_argument("--debug_stage_obstacles_pad", type=int, default=80)
    ap.add_argument("--debug_stage_obstacles_max_discs", type=int, default=60)
    ap.add_argument("--tasks_as_agents", action="store_true",
                help="Interpret tasks_per_map as #agents to simulate simultaneously (single shared-goal run per map).")
    ap.add_argument("--start_min_sep", type=float, default=1000.0,
                help="Minimum separation (pixels) between random starts when --tasks_as_agents is set.")

    args = ap.parse_args()

    out_dir = mkdir(args.out)
    rng = np.random.RandomState(args.seed)

    # load model if provided
    model = None
    if args.ckpt:
        model = CoefEnergyNet()
        ck = torch.load(args.ckpt, map_location="cpu")
        sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        print(sd.keys() if isinstance(sd, dict) else "state_dict is not a dict")
        model.load_state_dict(sd, strict=False)
        model.to(args.device).eval()
        print(f"[OK] loaded model from {args.ckpt}")

    # enumerate maps
    if os.path.isdir(args.input):
        maps = sorted(glob.glob(os.path.join(args.input, "*.png")))
    else:
        maps = [args.input]

    agg = []
    for mp in maps:
        name = safe_name(os.path.splitext(os.path.basename(mp))[0])
        occ = load_topdown_occ(mp, thresh=args.thresh)
        occ_plan = ndimage.binary_dilation(occ > 0, iterations=max(0, args.inflate_cells)).astype(np.uint8)
        sdf = SignedDistanceField(occ, grad_smooth_sigma=1.0)

        if args.tasks_as_agents:
            N_sim = int(args.tasks_per_map)
            if args.agents != 16:
                print(f"[INFO] --tasks_as_agents enabled: ignoring --agents={args.agents}; using N={N_sim} (tasks_per_map)")
            tasks = [make_shared_goal_agents_task(
                sdf, N=N_sim, rng=rng,
                start_clear=6.0, goal_clear=8.0,
                min_sep=float(args.start_min_sep),
            )]
        else:
            tasks = make_task_pairs_stagewise(
                sdf, occ,
                num_tasks=args.tasks_per_map,
                N=args.agents,
                rng=rng,
                start_clear=6.0,
                goal_clear=8.0,
                start_swarm_radius=30.0,
            )

        map_dir = mkdir(os.path.join(out_dir, name))
        with open(os.path.join(map_dir, "tasks.json"), "w") as f:
            json.dump(tasks, f, indent=2)

        for ti, task in enumerate(tasks):
            starts = np.array(task["starts"], dtype=np.float32)
            goal = np.array(task["goal"], dtype=np.float32)
            start_center = np.mean(starts, axis=0).astype(np.float32)
            #start_center = np.array(task["start_center"], dtype=np.float32)

            if args.tasks_as_agents:
                stages = []
                for ai in range(starts.shape[0]):
                    stages_ai = build_stages_from_astar_robust(
                        occ=occ, occ_plan=occ_plan, sdf=sdf,
                        start_xy=starts[ai], goal_xy=goal,
                        stage_width=args.stage_width,
                        eps_rdp=args.rdp_eps,
                        theta_deg=args.corner_theta_deg,
                        densify_alpha=args.densify_alpha,
                        densify_min=args.densify_min,
                        inflate_steps=args.inflate_steps,
                        inflate_eta=args.inflate_eta,
                        inflate_min_clear=args.inflate_min_clear,
                        max_waypoints=args.max_waypoints,
                    )
                    stages.append(stages_ai)
            else:
                stages = build_stages_from_astar_robust(
                    occ=occ, occ_plan=occ_plan, sdf=sdf,
                    start_xy=start_center, goal_xy=goal,
                    stage_width=args.stage_width,
                    eps_rdp=args.rdp_eps,
                    theta_deg=args.corner_theta_deg,
                    densify_alpha=args.densify_alpha,
                    densify_min=args.densify_min,
                    inflate_steps=args.inflate_steps,
                    inflate_eta=args.inflate_eta,
                    inflate_min_clear=args.inflate_min_clear,
                    max_waypoints=args.max_waypoints,
                )
            if args.debug_stage_obstacles:
                dbg_dir = mkdir(os.path.join(map_dir, f"task{ti:02d}_debug_stage_obstacles"))
                save_stage_obstacle_debug(
                    png_path=mp,
                    occ=occ,
                    stages=stages,
                    out_dir=dbg_dir,
                    pad=args.debug_stage_obstacles_pad,
                    max_stages=args.debug_stage_obstacles_max,
                    max_discs=args.debug_stage_obstacles_max_discs,
                )

            out_gif = os.path.join(map_dir, f"task{ti:02d}_swarm.gif")
            out_mp4 = os.path.join(map_dir, f"task{ti:02d}_swarm.mp4") if args.write_mp4 else None
            out_json = os.path.join(map_dir, f"task{ti:02d}_metrics.json")

            sim = rollout_swarm_stagewise(
                occ=occ,
                sdf=sdf,
                starts_xy=starts,
                goal_xy=goal,
                stages=stages,
                model=model,
                device=args.device,
                T=args.T,
                dt=args.dt,
                goal_tol=args.goal_tol,
                stage_tol=args.stage_tol,
                local_obs_radius=args.local_obs_radius,
                local_obs_n=args.local_obs_n,
            )
            traj = sim["traj"]

            metrics = compute_swarm_metrics(
                occ, sdf, traj,
                goal_xy=goal,
                goal_tol=args.goal_tol,
                dt=args.dt,
            )
            metrics["map"] = mp
            metrics["task_id"] = int(ti)
            metrics["runtime_s"] = float(sim["runtime_s"][0])
            metrics["gif"] = out_gif
            metrics["mp4"] = out_mp4
            metrics["n_stages"] = int(len(stages))

            with open(out_json, "w") as f:
                json.dump(metrics, f, indent=2)

            stages_for_render = stages[0] if (args.tasks_as_agents) else stages
            render_swarm_gif(
                mp, occ, traj, stages_for_render, starts, goal,
                out_gif=out_gif,
                out_mp4=out_mp4,
                every=2,
            )

            agg.append(metrics)
            print(f"[DONE] {name} task={ti} success_rate={metrics['success_rate']:.3f} n_stages={len(stages)}")

    with open(os.path.join(out_dir, "aggregate_metrics.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[OK] wrote aggregate_metrics.json with {len(agg)} runs")


if __name__ == "__main__":
    main()