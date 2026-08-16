from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIRS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]
STEP = {(dr, dc): 1.4142 if abs(dr) + abs(dc) == 2 else 1.0 for dr, dc in DIRS}


def _clip_crop_bounds(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    shape: Tuple[int, int],
    *,
    margin: int,
) -> Tuple[int, int, int, int]:
    h, w = shape
    r0 = max(0, min(start[0], goal[0]) - margin)
    r1 = min(h, max(start[0], goal[0]) + margin + 1)
    c0 = max(0, min(start[1], goal[1]) - margin)
    c1 = min(w, max(start[1], goal[1]) + margin + 1)
    return r0, r1, c0, c1


def _block_reduce_mean(arr: np.ndarray, stride: int) -> np.ndarray:
    h, w = arr.shape
    hp = int(math.ceil(h / stride) * stride)
    wp = int(math.ceil(w / stride) * stride)
    pad = np.pad(arr, ((0, hp - h), (0, wp - w)), mode="edge")
    return pad.reshape(hp // stride, stride, wp // stride, stride).mean(axis=(1, 3))


def _block_reduce_max(arr: np.ndarray, stride: int) -> np.ndarray:
    h, w = arr.shape
    hp = int(math.ceil(h / stride) * stride)
    wp = int(math.ceil(w / stride) * stride)
    pad = np.pad(arr, ((0, hp - h), (0, wp - w)), mode="edge")
    return pad.reshape(hp // stride, stride, wp // stride, stride).max(axis=(1, 3))


def _line_cells(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    cells = [(r0, c0)]
    while (r0, c0) != (r1, c1):
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr
        cells.append((r0, c0))
    return cells


def _coarse_astar_teacher(
    risk: np.ndarray,
    geom: np.ndarray,
    hard: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    def octile(node: Tuple[int, int]) -> float:
        dr = abs(node[0] - goal[0])
        dc = abs(node[1] - goal[1])
        return max(dr, dc) + (1.4142 - 1.0) * min(dr, dc)

    open_heap = [(octile(start), 0.0, start)]
    parent = {start: None}
    gscore = {start: 0.0}
    h, w = risk.shape
    while open_heap:
        _, g, node = open_heap.pop(0)
        if node == goal:
            path = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        r, c = node
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if geom[nr, nc] > 0.5:
                continue
            step = STEP[(dr, dc)]
            cost = step * (1.0 + 4.0 * float(risk[nr, nc]) + 8.0 * float(hard[nr, nc] > 0.5))
            ng = g + cost
            nxt = (nr, nc)
            if ng < gscore.get(nxt, 1e18):
                gscore[nxt] = ng
                parent[nxt] = node
                prio = ng + octile(nxt)
                i = 0
                while i < len(open_heap) and open_heap[i][0] <= prio:
                    i += 1
                open_heap.insert(i, (prio, ng, nxt))
    return None


def _teacher_dataset(grid: CoarseGrid) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    path = _coarse_astar_teacher(grid.risk, grid.geom, grid.hard, grid.start, grid.goal)
    if not path or len(path) < 2:
        return None
    obs: List[np.ndarray] = []
    acts: List[int] = []
    prev_dir_idx = 0
    for cur, nxt in zip(path[:-1], path[1:]):
        obs.append(_state_features(grid, cur, prev_dir_idx))
        delta = (nxt[0] - cur[0], nxt[1] - cur[1])
        act_idx = DIRS.index(delta)
        acts.append(act_idx)
        prev_dir_idx = act_idx
    return np.asarray(obs, dtype=np.float32), np.asarray(acts, dtype=np.int64)


@dataclass(frozen=True)
class CoarseGrid:
    risk: np.ndarray
    hard: np.ndarray
    geom: np.ndarray
    row0: int
    col0: int
    stride: int
    start: Tuple[int, int]
    goal: Tuple[int, int]


def build_coarse_grid(
    maps: Dict[str, np.ndarray],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    crop_margin: int = 48,
    max_side: int = 96,
) -> CoarseGrid:
    risk = maps["risk_map"]
    hard = maps["hard_mask"].astype(np.float32)
    geom = maps["geom_occ"].astype(np.float32)

    r0, r1, c0, c1 = _clip_crop_bounds(start, goal, risk.shape, margin=crop_margin)
    risk_crop = risk[r0:r1, c0:c1]
    hard_crop = hard[r0:r1, c0:c1]
    geom_crop = geom[r0:r1, c0:c1]

    stride = max(1, int(math.ceil(max(risk_crop.shape) / max_side)))
    risk_small = _block_reduce_mean(risk_crop, stride)
    hard_small = _block_reduce_max(hard_crop, stride)
    geom_small = _block_reduce_max(geom_crop, stride)

    sr = min((start[0] - r0) // stride, risk_small.shape[0] - 1)
    sc = min((start[1] - c0) // stride, risk_small.shape[1] - 1)
    gr = min((goal[0] - r0) // stride, risk_small.shape[0] - 1)
    gc = min((goal[1] - c0) // stride, risk_small.shape[1] - 1)

    geom_small[sr, sc] = 0.0
    geom_small[gr, gc] = 0.0
    return CoarseGrid(
        risk=risk_small.astype(np.float32),
        hard=hard_small.astype(np.float32),
        geom=geom_small.astype(np.float32),
        row0=r0,
        col0=c0,
        stride=stride,
        start=(int(sr), int(sc)),
        goal=(int(gr), int(gc)),
    )


class PolicyNet(nn.Module):
    def __init__(self, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden_dim, len(DIRS))
        self.vr = nn.Linear(hidden_dim, 1)
        self.vc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.net(x)
        return self.pi(z), self.vr(z).squeeze(-1), self.vc(z).squeeze(-1)


def _state_features(
    grid: CoarseGrid,
    pos: Tuple[int, int],
    prev_dir_idx: int,
) -> np.ndarray:
    h, w = grid.risk.shape
    r, c = pos
    gr, gc = grid.goal
    risk = float(grid.risk[r, c])
    hard = float(grid.hard[r, c])
    geom = float(grid.geom[r, c])
    return np.asarray(
        [
            r / max(h - 1, 1),
            c / max(w - 1, 1),
            (gr - r) / max(h - 1, 1),
            (gc - c) / max(w - 1, 1),
            risk,
            hard,
            geom,
            prev_dir_idx / max(len(DIRS) - 1, 1),
        ],
        dtype=np.float32,
    )


def _transition(
    grid: CoarseGrid,
    pos: Tuple[int, int],
    action_idx: int,
    prev_dir_idx: int,
    *,
    risk_weight: float,
) -> Tuple[Tuple[int, int], float, float, bool]:
    h, w = grid.risk.shape
    dr, dc = DIRS[action_idx]
    nr = min(max(pos[0] + dr, 0), h - 1)
    nc = min(max(pos[1] + dc, 0), w - 1)
    blocked = bool(grid.geom[nr, nc] > 0.5)
    if blocked:
        nr, nc = pos
    step = STEP[(dr, dc)]
    risk = float(grid.risk[nr, nc])
    hard = float(grid.hard[nr, nc] > 0.5)
    turn_pen = 0.15 if prev_dir_idx >= 0 and prev_dir_idx != action_idx else 0.0
    reward = -0.12 * step - 0.08 * risk - turn_pen - (0.35 if blocked else 0.0)
    cost = hard + 0.35 * max(0.0, risk - 0.35)
    done = (nr, nc) == grid.goal
    if done:
        reward += 8.0
    return (nr, nc), reward, cost, done


def _compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    *,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    adv = np.zeros(len(rewards), dtype=np.float32)
    last = 0.0
    next_value = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
        next_value = values[t]
    ret = adv + np.asarray(values, dtype=np.float32)
    return adv, ret


def _rollout_policy(
    model: PolicyNet,
    grid: CoarseGrid,
    *,
    max_steps: int,
    risk_weight: float,
    deterministic: bool,
) -> Dict[str, object]:
    pos = grid.start
    prev_dir_idx = -1
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    logp_list: List[float] = []
    rew_list: List[float] = []
    cost_list: List[float] = []
    done_list: List[bool] = []
    val_r_list: List[float] = []
    val_c_list: List[float] = []
    path = [pos]

    for _ in range(max_steps):
        obs = _state_features(grid, pos, max(prev_dir_idx, 0))
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits, vr, vc = model(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = int(torch.argmax(logits, dim=-1).item()) if deterministic else int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor(action)).item())
        nxt, reward, cost, done = _transition(
            grid,
            pos,
            action,
            prev_dir_idx,
            risk_weight=risk_weight,
        )
        obs_list.append(obs)
        act_list.append(action)
        logp_list.append(logp)
        rew_list.append(reward)
        cost_list.append(cost)
        done_list.append(done)
        val_r_list.append(float(vr.item()))
        val_c_list.append(float(vc.item()))
        pos = nxt
        prev_dir_idx = action
        path.append(pos)
        if done:
            break

    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(act_list, dtype=np.int64),
        "logp": np.asarray(logp_list, dtype=np.float32),
        "rewards": rew_list,
        "costs": cost_list,
        "dones": done_list,
        "values_r": val_r_list,
        "values_c": val_c_list,
        "path": path,
    }


def _coarse_path_to_fine(grid: CoarseGrid, coarse_path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    fine_path: List[Tuple[int, int]] = []
    prev: Optional[Tuple[int, int]] = None
    for rr, cc in coarse_path:
        r = int(grid.row0 + rr * grid.stride + grid.stride // 2)
        c = int(grid.col0 + cc * grid.stride + grid.stride // 2)
        cell = (r, c)
        if prev is None:
            fine_path.append(cell)
        else:
            segment = _line_cells(prev, cell)
            fine_path.extend(segment[1:])
        prev = cell
    return fine_path


def ppo_lagrangian_plan(
    maps: Dict[str, np.ndarray],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    *,
    risk_weight: float = 10.0,
    crop_margin: int = 48,
    max_side: int = 96,
    total_updates: int = 18,
    episodes_per_update: int = 8,
    gamma: float = 0.98,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 3e-3,
    entropy_coef: float = 0.01,
    cost_limit: float = 0.12,
    lambda_lr: float = 0.05,
) -> Optional[List[Tuple[int, int]]]:
    grid = build_coarse_grid(
        maps,
        start,
        goal,
        crop_margin=crop_margin,
        max_side=max_side,
    )
    model = PolicyNet()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    lagrange = 0.0
    base_h = abs(grid.goal[0] - grid.start[0])
    base_w = abs(grid.goal[1] - grid.start[1])
    max_steps = max(24, 4 * (base_h + base_w + 1))
    teacher = _teacher_dataset(grid)

    if teacher is not None:
        obs_bc, act_bc = teacher
        obs_bc_t = torch.from_numpy(obs_bc)
        act_bc_t = torch.from_numpy(act_bc)
        for _ in range(80):
            logits, vr, vc = model(obs_bc_t)
            del vr, vc
            loss = F.cross_entropy(logits, act_bc_t)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

    for _ in range(total_updates):
        batch_obs: List[np.ndarray] = []
        batch_act: List[np.ndarray] = []
        batch_logp: List[np.ndarray] = []
        batch_adv: List[np.ndarray] = []
        batch_ret_r: List[np.ndarray] = []
        batch_ret_c: List[np.ndarray] = []
        ep_costs: List[float] = []

        for _ep in range(episodes_per_update):
            rollout = _rollout_policy(
                model,
                grid,
                max_steps=max_steps,
                risk_weight=risk_weight,
                deterministic=False,
            )
            adv_r, ret_r = _compute_gae(
                rollout["rewards"],
                rollout["values_r"],
                rollout["dones"],
                gamma=gamma,
                lam=gae_lambda,
            )
            adv_c, ret_c = _compute_gae(
                rollout["costs"],
                rollout["values_c"],
                rollout["dones"],
                gamma=gamma,
                lam=gae_lambda,
            )
            mixed_adv = adv_r - lagrange * adv_c
            mixed_adv = (mixed_adv - mixed_adv.mean()) / max(mixed_adv.std(), 1e-6)
            batch_obs.append(rollout["obs"])
            batch_act.append(rollout["actions"])
            batch_logp.append(rollout["logp"])
            batch_adv.append(mixed_adv.astype(np.float32))
            batch_ret_r.append(ret_r.astype(np.float32))
            batch_ret_c.append(ret_c.astype(np.float32))
            ep_costs.append(float(np.mean(rollout["costs"]) if rollout["costs"] else 0.0))

        obs_t = torch.from_numpy(np.concatenate(batch_obs, axis=0))
        act_t = torch.from_numpy(np.concatenate(batch_act, axis=0))
        logp_old_t = torch.from_numpy(np.concatenate(batch_logp, axis=0))
        adv_t = torch.from_numpy(np.concatenate(batch_adv, axis=0))
        ret_r_t = torch.from_numpy(np.concatenate(batch_ret_r, axis=0))
        ret_c_t = torch.from_numpy(np.concatenate(batch_ret_c, axis=0))

        for _epoch in range(5):
            logits, vr, vc = model(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act_t)
            ratio = torch.exp(logp - logp_old_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (F.mse_loss(vr, ret_r_t) + F.mse_loss(vc, ret_c_t))
            entropy = dist.entropy().mean()
            loss = policy_loss + value_loss - entropy_coef * entropy
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        lagrange = max(0.0, lagrange + lambda_lr * (float(np.mean(ep_costs)) - cost_limit))

    rollout = _rollout_policy(
        model,
        grid,
        max_steps=max_steps,
        risk_weight=risk_weight,
        deterministic=True,
    )
    coarse_path = rollout["path"]
    if not coarse_path or coarse_path[-1] != grid.goal:
        if teacher is None:
            return None
        teacher_path = _coarse_astar_teacher(grid.risk, grid.geom, grid.hard, grid.start, grid.goal)
        if not teacher_path:
            return None
        return _coarse_path_to_fine(grid, teacher_path)
    return _coarse_path_to_fine(grid, coarse_path)
