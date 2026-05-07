from __future__ import annotations

import json
import heapq
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from train_material import (
        CoefEnergyNetMaterial,
        integrate_surrogate_material,
    )
except ImportError:
    CoefEnergyNetMaterial = None
    integrate_surrogate_material = None
from scripts.build_dfc2018_stagewise import (
    extract_local_geom_obstacles,
    extract_risk_patch,
    extract_rollout_patch,
)


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
STEP = {(dr, dc): 1.4142 if abs(dr) + abs(dc) == 2 else 1.0 for dr, dc in DIRS}


def astar_geom_only(
    maps: Dict[str, np.ndarray],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    h_rows, h_cols = maps["z2_labels"].shape
    geom = maps["geom_occ"].astype(bool)
    gr, gc = goal

    def h(r: int, c: int) -> float:
        dr = abs(r - gr)
        dc = abs(c - gc)
        return max(dr, dc) + (1.4142 - 1.0) * min(dr, dc)

    gscore = {start: 0.0}
    parent = {start: None}
    heap = [(h(*start), 0.0, start[0], start[1])]
    while heap:
        _, g, r, c = heapq.heappop(heap)
        node = (r, c)
        if node == goal:
            path: List[Tuple[int, int]] = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        if g > gscore.get(node, 1e18) + 1e-9:
            continue
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h_rows and 0 <= nc < h_cols):
                continue
            if geom[nr, nc]:
                continue
            ng = g + STEP[(dr, dc)]
            nxt = (nr, nc)
            if ng < gscore.get(nxt, 1e18):
                gscore[nxt] = ng
                parent[nxt] = node
                heapq.heappush(heap, (ng + h(nr, nc), ng, nr, nc))
    return None


def build_geom_waypoints(
    path_rc: List[Tuple[int, int]],
    *,
    stride: int = 6,
    max_stages: int = 256,
    patch_size: int = 64,
) -> List[Tuple[float, float]]:
    if not path_rc:
        return []
    total = len(path_rc)
    idxs = list(range(0, total, stride))
    if idxs[-1] != total - 1:
        idxs.append(total - 1)
    if len(idxs) > max_stages:
        sel = np.linspace(0, len(idxs) - 1, max_stages, dtype=int)
        idxs = [idxs[i] for i in sel]

    path_np = np.asarray(path_rc, dtype=np.float32)
    waypoints: List[Tuple[float, float]] = []
    max_r = patch_size / 2.0 - 2.0
    for ci in idxs:
        last_good = ci
        for j in range(ci + 3, min(ci + 64, total)):
            delta = path_np[j] - path_np[ci]
            if float(np.linalg.norm(delta)) <= max_r:
                last_good = j
            else:
                break
        if last_good == ci:
            last_good = min(ci + 3, total - 1)
        er, ec = int(path_np[last_good, 0]), int(path_np[last_good, 1])
        waypoints.append((float(ec), float(er)))
    return waypoints


def load_model(
    ckpt_path: str | Path,
    *,
    device: str,
    patch_size: int = 32,
) -> CoefEnergyNetMaterial:
    if CoefEnergyNetMaterial is None:
        raise ImportError(
            "Model checkpoint evaluation requires train_material.py and its "
            "training dependencies. Planner-only RELLIS-Dyn runs do not need it."
        )
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {})
    model = CoefEnergyNetMaterial(
        patch_size=cfg.get("patch_size", patch_size),
        lam_soft_max=cfg.get("lam_soft_max", 5.0),
        lam_hard_max=cfg.get("lam_hard_max", 10.0),
        mu_lat_max=cfg.get("mu_lat_max", 5.0),
    )
    state_dict = ck.get("model_state_dict", ck.get("model", ck))
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model


def load_episode_checkpoints(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    ck_path = Path(episode["logs"]["checkpoints_jsonl"])
    with ck_path.open() as f:
        return [json.loads(line) for line in f]


def build_episode_waypoints(
    episode: Dict[str, Any],
    maps: Dict[str, np.ndarray],
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    checkpoints: List[Dict[str, Any]],
    *,
    eval_mode: str = "endtoend",
    patch_size: int = 64,
) -> Tuple[List[Tuple[float, float]], List[float], List[float]]:
    ck_dts = [float(ck["dt"]) for ck in checkpoints]
    ck_d_hats = [float(ck["barrier"]["barrier_d_hat"]) for ck in checkpoints]
    if eval_mode == "stagewise":
        waypoints_xy = [tuple(ck["stage_exit"]) for ck in checkpoints]
        return waypoints_xy, ck_d_hats, ck_dts

    geom_path = astar_geom_only(maps, start_rc, goal_rc)
    if geom_path is None:
        geom_path = [start_rc, goal_rc]
    waypoints_xy = build_geom_waypoints(
        geom_path,
        stride=int(episode["meta"].get("path_stride", 6)),
        patch_size=patch_size,
    )
    while len(ck_dts) < len(waypoints_xy):
        ck_dts.append(ck_dts[-1])
    while len(ck_d_hats) < len(waypoints_xy):
        ck_d_hats.append(ck_d_hats[-1])
    return waypoints_xy, ck_d_hats, ck_dts


def _build_obs_feats(pos_xy, goal_xy, centers, radii, widths, device: str) -> torch.Tensor:
    n_obs = centers.shape[0]
    if n_obs == 0:
        return torch.zeros(1, 0, 6, device=device)
    c_t = torch.as_tensor(centers, dtype=torch.float32, device=device)
    r_t = torch.as_tensor(radii, dtype=torch.float32, device=device)
    w_t = torch.as_tensor(widths, dtype=torch.float32, device=device)
    g_t = torch.as_tensor(goal_xy, dtype=torch.float32, device=device)
    dxdy = g_t.unsqueeze(0) - c_t
    return torch.cat([c_t, r_t.unsqueeze(-1), w_t.unsqueeze(-1), dxdy], dim=-1).unsqueeze(0)


def _build_goal_feats(pos_xy, goal_xy, device: str) -> torch.Tensor:
    o = torch.as_tensor(pos_xy, dtype=torch.float32, device=device)
    g = torch.as_tensor(goal_xy, dtype=torch.float32, device=device)
    dg = g - o
    dist = torch.linalg.norm(dg).unsqueeze(0)
    return torch.cat([dg, dist, torch.ones(1, device=device)]).unsqueeze(0)


@torch.no_grad()
def run_model_episode(
    model: CoefEnergyNetMaterial,
    maps: Dict[str, np.ndarray],
    waypoints_xy: List[Tuple[float, float]],
    ck_d_hats: List[float],
    ck_dts: List[float],
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    *,
    device: str,
    stage: int,
    steps_per_stage: int = 80,
    robot_radius: float = 1.5,
    margin_factor: float = 0.5,
    patch_size: int = 32,
    d_hat_sdf: float = 3.0,
    goal_tol: float = 3.0,
) -> np.ndarray:
    if integrate_surrogate_material is None:
        raise ImportError(
            "Model rollout evaluation requires train_material.py and its "
            "training dependencies. Planner-only RELLIS-Dyn runs do not need it."
        )
    h_rows, h_cols = maps["z2_labels"].shape
    pos = np.array([float(start_rc[1]), float(start_rc[0])], dtype=np.float32)
    vel = np.zeros(2, dtype=np.float32)
    traj_xy = [pos.copy()]
    geom_occ = maps["geom_occ"]

    for si, waypoint_xy in enumerate(waypoints_xy):
        stage_goal_xy = np.asarray(waypoint_xy, dtype=np.float32)
        dt = float(ck_dts[si]) if si < len(ck_dts) else 0.04
        d_hat_v = float(ck_d_hats[si]) if si < len(ck_d_hats) else 3.0

        cr = int(np.clip(pos[1], 0, h_rows - 1))
        cc = int(np.clip(pos[0], 0, h_cols - 1))
        centers, radii, widths, _ = extract_local_geom_obstacles(
            geom_occ,
            (cr, cc),
            patch_size=64,
            robot_radius=robot_radius,
            margin_factor=margin_factor,
        )
        risk_patch_np, _ = extract_risk_patch(maps, (cr, cc), patch_size)
        obs_feats = _build_obs_feats(pos, stage_goal_xy, centers, radii, widths, device)
        obs_mask = (
            torch.ones(1, obs_feats.shape[1], dtype=torch.bool, device=device)
            if obs_feats.shape[1] > 0
            else torch.zeros(1, 0, dtype=torch.bool, device=device)
        )
        goal_feats = _build_goal_feats(pos, stage_goal_xy, device)
        risk_patch = torch.as_tensor(risk_patch_np, dtype=torch.float32, device=device).unsqueeze(0)

        alphas, beta, gamma, lam_soft, lam_hard, mu_lat = model(
            obs_feats,
            obs_mask,
            goal_feats,
            risk_patch,
        )
        del mu_lat
        if stage == 1:
            lam_soft = torch.zeros_like(lam_soft)
            lam_hard = torch.zeros_like(lam_hard)

        c_t = torch.as_tensor(centers, dtype=torch.float32, device=device).unsqueeze(0)
        r_t = torch.as_tensor(radii, dtype=torch.float32, device=device).unsqueeze(0)
        mask = (
            torch.ones(1, centers.shape[0], dtype=torch.bool, device=device)
            if centers.shape[0] > 0
            else torch.zeros(1, 0, dtype=torch.bool, device=device)
        )
        goal_t = torch.as_tensor(stage_goal_xy, dtype=torch.float32, device=device).unsqueeze(0)
        dt_t = torch.tensor([dt], dtype=torch.float32, device=device)
        d_hat_t = torch.tensor([d_hat_v], dtype=torch.float32, device=device)
        rr_t = torch.tensor([robot_radius], dtype=torch.float32, device=device)

        o = torch.as_tensor(pos, dtype=torch.float32, device=device).unsqueeze(0)
        v = torch.as_tensor(vel, dtype=torch.float32, device=device).unsqueeze(0)
        stage_traj_xy: List[np.ndarray] = []
        for _ in range(steps_per_stage):
            cr_s = int(np.clip(o[0, 1].item(), 0, h_rows - 1))
            cc_s = int(np.clip(o[0, 0].item(), 0, h_cols - 1))
            rollout_patch_np = np.asarray(
                extract_rollout_patch(maps, (cr_s, cc_s), patch_size),
                dtype=np.float32,
            )
            rollout_patch = torch.as_tensor(rollout_patch_np, dtype=torch.float32, device=device).unsqueeze(0)
            o_n, v_n, _, _, _, _ = integrate_surrogate_material(
                o0=o.clone(),
                v0=v.clone(),
                goal=goal_t,
                C=c_t,
                R=r_t,
                mask=mask,
                alphas=alphas,
                beta=beta,
                gamma=gamma,
                lam_soft=lam_soft,
                lam_hard=lam_hard,
                rollout_patch=rollout_patch,
                d_hat=d_hat_t,
                dt=dt_t,
                H=torch.tensor([1], dtype=torch.long, device=device),
                robot_radius=rr_t,
                margin_factor=margin_factor,
                d_hat_sdf=d_hat_sdf,
            )
            o = o_n
            v = v_n
            pos_s = o[0].cpu().numpy()
            stage_traj_xy.append(pos_s.copy())
            if np.linalg.norm(pos_s - stage_goal_xy) < goal_tol:
                break
        traj_xy.extend(stage_traj_xy)
        pos = o[0].cpu().numpy()
        vel = v[0].cpu().numpy()
        goal_xy = np.array([float(goal_rc[1]), float(goal_rc[0])], dtype=np.float32)
        if np.linalg.norm(pos - goal_xy) < goal_tol * 2:
            break

    traj_xy = np.asarray(traj_xy, dtype=np.float32)
    traj_rc = np.stack([traj_xy[:, 1], traj_xy[:, 0]], axis=-1)
    return traj_rc
