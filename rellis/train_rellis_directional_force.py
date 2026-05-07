#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grl_rellis import BevConfig


DIRS_16 = np.asarray(
    [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (2.0, 1.0),
        (2.0, -1.0),
        (-2.0, 1.0),
        (-2.0, -1.0),
        (1.0, 2.0),
        (-1.0, 2.0),
        (1.0, -2.0),
        (-1.0, -2.0),
    ],
    dtype=np.float32,
)
DIRS_16 = DIRS_16 / np.linalg.norm(DIRS_16, axis=1, keepdims=True)
GRID_DIRS_8 = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]
GRID_STEP = {(dr, dc): 1.4142 if abs(dr) + abs(dc) == 2 else 1.0 for dr, dc in GRID_DIRS_8}


def _as_path(raw: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in raw]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def _sample_grid(arr: np.ndarray, r: float, c: float) -> float:
    rr = int(np.clip(round(float(r)), 0, arr.shape[0] - 1))
    cc = int(np.clip(round(float(c)), 0, arr.shape[1] - 1))
    return float(arr[rr, cc])


def _direction_stats(
    maps: Dict[str, np.ndarray],
    pos: np.ndarray,
    direction: np.ndarray,
    *,
    horizon_cells: int,
    hard_margin_m: float,
) -> Tuple[float, bool, float, float]:
    risk = maps["risk_map"]
    hard = maps["hard_mask"].astype(bool)
    sdf = maps["sdf_hard"]
    total_risk = 0.0
    feasible = True
    min_sdf = float("inf")
    blocked_frac = 0.0
    for step in range(1, horizon_cells + 1):
        q = pos + float(step) * direction
        r = int(round(float(q[0])))
        c = int(round(float(q[1])))
        if not (0 <= r < risk.shape[0] and 0 <= c < risk.shape[1]):
            feasible = False
            blocked_frac += 1.0
            break
        rho = float(risk[r, c])
        phi = float(sdf[r, c])
        total_risk += rho
        min_sdf = min(min_sdf, phi)
        if hard[r, c] or phi < hard_margin_m:
            feasible = False
            blocked_frac += 1.0
            break
    return total_risk, feasible, min_sdf if math.isfinite(min_sdf) else 0.0, blocked_frac / horizon_cells


def _direction_endpoint(pos: np.ndarray, direction: np.ndarray, *, horizon_cells: int, shape: Tuple[int, int]) -> Tuple[int, int]:
    q = pos + float(horizon_cells) * direction
    r = int(np.clip(round(float(q[0])), 0, shape[0] - 1))
    c = int(np.clip(round(float(q[1])), 0, shape[1] - 1))
    return r, c


def _reverse_cost_to_go(maps: Dict[str, np.ndarray], goal: Tuple[int, int], *, risk_weight: float) -> np.ndarray:
    blocked = maps["geom_occ"].astype(bool)
    risk = maps["risk_map"]
    rows, cols = risk.shape
    gr, gc = goal
    if not (0 <= gr < rows and 0 <= gc < cols) or blocked[gr, gc]:
        return np.full((rows, cols), np.inf, dtype=np.float32)

    src: List[int] = []
    dst: List[int] = []
    data: List[float] = []
    for r in range(rows):
        for c in range(cols):
            if blocked[r, c]:
                continue
            cur = r * cols + c
            # Reverse graph edge: current -> neighbor has the cost of neighbor -> current
            # in the forward planning graph, which pays the current cell's risk.
            cell_cost = 1.0 + risk_weight * float(risk[r, c])
            for dr, dc in GRID_DIRS_8:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc]:
                    continue
                src.append(cur)
                dst.append(nr * cols + nc)
                data.append(GRID_STEP[(dr, dc)] * cell_cost)
    graph = coo_matrix((data, (src, dst)), shape=(rows * cols, rows * cols)).tocsr()
    flat = sparse_dijkstra(graph, directed=True, indices=gr * cols + gc)
    return np.asarray(flat, dtype=np.float32).reshape(rows, cols)


def _route_context(maps: Dict[str, np.ndarray], goal: Tuple[int, int], *, risk_weight: float) -> Dict[str, np.ndarray]:
    return {
        "geom_to_go": _reverse_cost_to_go(maps, goal, risk_weight=0.0),
        "risk_to_go": _reverse_cost_to_go(maps, goal, risk_weight=risk_weight),
    }


def _build_point(
    maps: Dict[str, np.ndarray],
    path: List[Tuple[int, int]],
    idx: int,
    *,
    regime: str,
    episode_id: str,
    horizon_cells: int,
    long_horizon_cells: int,
    hard_margin_m: float,
    improvement_margin: float,
    route: Optional[Dict[str, np.ndarray]] = None,
    route_max_ratio: float = 2.2,
) -> Optional[Dict[str, object]]:
    if len(path) < 3 or idx >= len(path) - 1:
        return None
    p = np.asarray(path[idx], dtype=np.float32)
    nxt = np.asarray(path[min(idx + 1, len(path) - 1)], dtype=np.float32)
    scaffold_dir = _unit(nxt - p)
    if float(np.linalg.norm(scaffold_dir)) < 1e-8:
        return None

    scaffold_risk, scaffold_feasible, scaffold_min_sdf, scaffold_blocked = _direction_stats(
        maps, p, scaffold_dir, horizon_cells=horizon_cells, hard_margin_m=hard_margin_m
    )
    scaffold_risk_long, scaffold_feasible_long, scaffold_min_sdf_long, scaffold_blocked_long = _direction_stats(
        maps, p, scaffold_dir, horizon_cells=long_horizon_cells, hard_margin_m=hard_margin_m
    )
    shape = maps["risk_map"].shape
    scaffold_end = _direction_endpoint(p, scaffold_dir, horizon_cells=horizon_cells, shape=shape)
    if route is not None:
        scaffold_route_geom = float(route["geom_to_go"][scaffold_end])
        scaffold_route_cost = float(route["risk_to_go"][scaffold_end])
    else:
        scaffold_route_geom = float("inf")
        scaffold_route_cost = float("inf")
    scaffold_route_total = scaffold_risk + scaffold_route_cost
    cand_features: List[List[float]] = []
    best_idx = -1
    best_risk = float("inf")
    best_route_total = float("inf")
    feasible_count = 0
    for k, d in enumerate(DIRS_16):
        cand_risk, feasible, min_sdf, blocked_frac = _direction_stats(
            maps, p, d, horizon_cells=horizon_cells, hard_margin_m=hard_margin_m
        )
        cand_risk_long, feasible_long, min_sdf_long, blocked_frac_long = _direction_stats(
            maps, p, d, horizon_cells=long_horizon_cells, hard_margin_m=hard_margin_m
        )
        cand_end = _direction_endpoint(p, d, horizon_cells=horizon_cells, shape=shape)
        if route is not None:
            cand_route_geom = float(route["geom_to_go"][cand_end])
            cand_route_cost = float(route["risk_to_go"][cand_end])
            route_feasible = bool(feasible and math.isfinite(cand_route_geom) and math.isfinite(scaffold_route_geom))
            route_ratio = cand_route_geom / max(scaffold_route_geom, 1e-6) if route_feasible else float("inf")
            route_total = cand_risk + cand_route_cost if route_feasible else float("inf")
            route_gain = scaffold_route_total - route_total if math.isfinite(route_total) and math.isfinite(scaffold_route_total) else 0.0
        else:
            cand_route_geom = 0.0
            cand_route_cost = 0.0
            route_feasible = False
            route_ratio = 0.0
            route_total = float("inf")
            route_gain = 0.0
        feasible_f = 1.0 if feasible else 0.0
        feasible_count += int(feasible)
        if route is not None:
            if route_feasible and route_ratio <= route_max_ratio and route_total < best_route_total:
                best_route_total = route_total
                best_risk = cand_risk
                best_idx = k
        else:
            if feasible_long and cand_risk_long < best_risk:
                best_risk = cand_risk_long
                best_idx = k
        cand_features.append(
            [
                cand_risk / max(1.0, float(horizon_cells)),
                feasible_f,
                (scaffold_risk - cand_risk) / max(1.0, float(horizon_cells)),
                min_sdf / 10.0,
                blocked_frac,
                cand_risk_long / max(1.0, float(long_horizon_cells)),
                1.0 if feasible_long else 0.0,
                (scaffold_risk_long - cand_risk_long) / max(1.0, float(long_horizon_cells)),
                min_sdf_long / 10.0,
                blocked_frac_long,
                1.0 if route_feasible and route_ratio <= route_max_ratio else 0.0,
                min(cand_route_geom, 999.0) / 100.0,
                min(cand_route_cost, 999.0) / 100.0,
                min(route_ratio, 9.99) / route_max_ratio,
                route_gain / 100.0,
                float(np.dot(scaffold_dir, d)),
            ]
        )
    if best_idx < 0:
        best_idx = 0
        best_risk = scaffold_risk_long
        best_route_total = scaffold_route_total

    if route is not None:
        has_safe_alt = bool(
            math.isfinite(best_route_total)
            and math.isfinite(scaffold_route_total)
            and (scaffold_route_total - best_route_total) >= improvement_margin
        )
    else:
        has_safe_alt = bool(scaffold_feasible_long and (scaffold_risk_long - best_risk) >= improvement_margin)
    label = 1 + best_idx if (regime == "R1" and has_safe_alt) else 0
    risk_grad = np.asarray(
        [
            _sample_grid(maps["grad_row"], p[0], p[1]),
            _sample_grid(maps["grad_col"], p[0], p[1]),
        ],
        dtype=np.float32,
    )
    local = [
        scaffold_risk / max(1.0, float(horizon_cells)),
        1.0 if scaffold_feasible else 0.0,
        scaffold_min_sdf / 10.0,
        scaffold_blocked,
        scaffold_risk_long / max(1.0, float(long_horizon_cells)),
        1.0 if scaffold_feasible_long else 0.0,
        scaffold_min_sdf_long / 10.0,
        scaffold_blocked_long,
        1.0 if math.isfinite(scaffold_route_geom) else 0.0,
        min(scaffold_route_geom, 999.0) / 100.0,
        min(scaffold_route_cost, 999.0) / 100.0,
        float(feasible_count) / float(len(DIRS_16)),
        _sample_grid(maps["risk_map"], p[0], p[1]),
        _sample_grid(maps["sdf_hard"], p[0], p[1]) / 10.0,
        float(risk_grad[0]),
        float(risk_grad[1]),
        float(idx) / max(1.0, float(len(path) - 1)),
    ]
    x = np.asarray(local + np.asarray(cand_features, dtype=np.float32).reshape(-1).tolist(), dtype=np.float32)
    return {
        "x": x,
        "label": int(label),
        "regime": regime,
        "episode_id": episode_id,
        "has_safe_alt": float(has_safe_alt),
        "best_idx": int(best_idx),
        "scaffold_risk": float(scaffold_risk),
        "safe_risk": float(best_risk),
        "scaffold_route_cost": float(scaffold_route_total) if math.isfinite(scaffold_route_total) else 999.0,
        "safe_route_cost": float(best_route_total) if math.isfinite(best_route_total) else 999.0,
    }


def _load_scene(bev_root: Path, rel_path: str) -> Dict:
    return torch.load(bev_root / rel_path, map_location="cpu", weights_only=False)


def build_dataset(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    bev_manifest = json.loads((args.bev_root / "manifest.json").read_text())
    pair_manifest = json.loads((args.pairs_root / "manifest.json").read_text())
    cfg = BevConfig(**bev_manifest["config"]["bev"])
    episodes = pair_manifest["episodes"][: args.max_episodes]
    rows: List[Dict[str, object]] = []
    scene_cache: Dict[str, Dict] = {}
    route_cache: Dict[Tuple[str, Tuple[int, int], float], Dict[str, np.ndarray]] = {}
    for ep in episodes:
        scene_path = str(ep["scene_path"])
        if scene_path not in scene_cache:
            scene_cache[scene_path] = _load_scene(args.bev_root, scene_path)
        maps = scene_cache[scene_path]["maps"]
        path = _as_path(ep["stage1_path"])
        route = None
        if args.route_aware:
            goal = tuple(int(x) for x in ep["goal_rc"])
            cache_key = (scene_path, goal, float(args.route_risk_weight))
            if cache_key not in route_cache:
                route_cache[cache_key] = _route_context(maps, goal, risk_weight=args.route_risk_weight)
            route = route_cache[cache_key]
        for idx in range(0, max(0, len(path) - 1), max(1, args.stride)):
            row = _build_point(
                maps,
                path,
                idx,
                regime=str(ep["regime"]),
                episode_id=str(ep["episode_id"]),
                horizon_cells=args.horizon_cells,
                long_horizon_cells=args.long_horizon_cells,
                hard_margin_m=args.hard_margin_m,
                improvement_margin=args.improvement_margin,
                route=route,
                route_max_ratio=args.route_max_ratio,
            )
            if row is not None:
                row["sequence"] = str(ep.get("sequence", ""))
                rows.append(row)
    meta = {
        "bev_root": str(args.bev_root),
        "pairs_root": str(args.pairs_root),
        "num_rows": len(rows),
        "num_episodes": len(episodes),
        "resolution": float(cfg.resolution),
        "stride": args.stride,
        "horizon_cells": args.horizon_cells,
        "long_horizon_cells": args.long_horizon_cells,
        "hard_margin_m": args.hard_margin_m,
        "improvement_margin": args.improvement_margin,
        "route_aware": args.route_aware,
        "route_risk_weight": args.route_risk_weight,
        "route_max_ratio": args.route_max_ratio,
    }
    return rows, meta


def split_rows(
    rows: List[Dict[str, object]],
    val_frac: float,
    *,
    seed: int = 0,
    split_mode: str = "episode",
    holdout_sequence: str = "",
) -> Tuple[List[int], List[int]]:
    if split_mode == "sequence":
        if not holdout_sequence:
            raise ValueError("--holdout-sequence is required when --split-mode=sequence")
        train_idx = [i for i, row in enumerate(rows) if str(row.get("sequence", "")) != holdout_sequence]
        val_idx = [i for i, row in enumerate(rows) if str(row.get("sequence", "")) == holdout_sequence]
        if not train_idx or not val_idx:
            raise RuntimeError(f"Invalid sequence split for holdout_sequence={holdout_sequence!r}")
        return train_idx, val_idx

    rng = np.random.default_rng(seed)
    by_regime: Dict[str, List[str]] = {}
    for row in rows:
        by_regime.setdefault(str(row["regime"]), [])
        eid = str(row["episode_id"])
        if eid not in by_regime[str(row["regime"])]:
            by_regime[str(row["regime"])].append(eid)
    val_eps = set()
    for eps in by_regime.values():
        eps = list(eps)
        if split_mode == "episode_seeded":
            rng.shuffle(eps)
        n_val = max(1, int(round(len(eps) * val_frac))) if len(eps) > 1 else 0
        val_eps.update(eps[-n_val:])
    train_idx, val_idx = [], []
    for i, row in enumerate(rows):
        (val_idx if str(row["episode_id"]) in val_eps else train_idx).append(i)
    return train_idx, val_idx


class DirectionalForceHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_loader(rows: List[Dict[str, object]], idxs: List[int], batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.as_tensor(np.stack([rows[i]["x"] for i in idxs], axis=0), dtype=torch.float32)
    y = torch.as_tensor([int(rows[i]["label"]) for i in idxs], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def train_model(rows: List[Dict[str, object]], train_idx: List[int], val_idx: List[int], args: argparse.Namespace) -> Dict:
    device = torch.device(args.device)
    in_dim = int(np.asarray(rows[0]["x"]).shape[0])
    model = DirectionalForceHead(in_dim, args.hidden, 1 + len(DIRS_16)).to(device)
    counts = np.bincount([int(rows[i]["label"]) for i in train_idx], minlength=1 + len(DIRS_16)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights[0] *= args.no_activation_weight
    weights[1:] *= args.activation_weight
    weights_t = torch.as_tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _make_loader(rows, train_idx, args.bs, True)
    val_loader = _make_loader(rows, val_idx, args.bs, False)
    best = {"val_loss": float("inf"), "epoch": -1, "state_dict": None}
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = F.cross_entropy(model(xb), yb, weight=weights_t)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += float(loss.item()) * xb.shape[0]
            tr_n += xb.shape[0]
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                loss = F.cross_entropy(model(xb), yb, weight=weights_t)
                val_loss += float(loss.item()) * xb.shape[0]
                val_n += xb.shape[0]
        val_loss /= max(1, val_n)
        tr_loss /= max(1, tr_n)
        print(f"Epoch {epoch:03d} train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best["val_loss"]:
            best = {
                "val_loss": val_loss,
                "epoch": epoch,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
    assert best["state_dict"] is not None
    model.load_state_dict(best["state_dict"])
    return {"model": model, "best": best, "class_counts": counts.tolist(), "class_weights": weights.tolist()}


@torch.no_grad()
def _predict_classes(
    rows: List[Dict[str, object]],
    idxs: List[int],
    model: DirectionalForceHead,
    args: argparse.Namespace,
    *,
    activation_threshold: Optional[float],
) -> np.ndarray:
    device = torch.device(args.device)
    x = torch.as_tensor(np.stack([rows[i]["x"] for i in idxs], axis=0), dtype=torch.float32, device=device)
    probs = torch.softmax(model(x), dim=-1)
    p0 = probs[:, 0]
    active_probs, active_idx = torch.max(probs[:, 1:], dim=-1)
    if activation_threshold is None:
        return torch.argmax(probs, dim=-1).cpu().numpy()
    scores = active_probs - p0
    pred = torch.where(
        scores >= float(activation_threshold),
        active_idx + 1,
        torch.zeros_like(active_idx),
    )
    return pred.cpu().numpy()


def _metrics_from_pred(
    rows: List[Dict[str, object]],
    idxs: List[int],
    pred: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, float]:
    car_den = 0
    car_num = 0
    far_den = 0
    far_num = 0
    perp_r1: List[float] = []
    perp_r2: List[float] = []
    correct_cls = 0
    for local_i, row_i in enumerate(idxs):
        row = rows[row_i]
        cls = int(pred[local_i])
        true_cls = int(row["label"])
        correct_cls += int(cls == true_cls)
        regime = str(row["regime"])
        if cls == 0:
            force = np.zeros(2, dtype=np.float32)
        else:
            force = DIRS_16[cls - 1]
        if true_cls > 0:
            car_den += 1
            safe = DIRS_16[int(row["best_idx"])]
            car_num += int(float(np.dot(force, safe)) > args.force_eps)
        if regime in ("R2", "R3"):
            far_den += 1
            far_num += int(float(np.linalg.norm(force)) > args.force_eps)
        if regime == "R1":
            perp_r1.append(float(np.linalg.norm(force)))
        elif regime == "R2":
            perp_r2.append(float(np.linalg.norm(force)))
    return {
        "n": float(len(idxs)),
        "accuracy": float(correct_cls / max(1, len(idxs))),
        "correct_activation_rate": float(car_num / max(1, car_den)),
        "false_activation_rate": float(far_num / max(1, far_den)),
        "selectivity_ratio": float(np.mean(perp_r1) / max(float(np.mean(perp_r2)), 1e-8)) if perp_r1 and perp_r2 else float("nan"),
    }


@torch.no_grad()
def calibrate_activation_threshold(
    rows: List[Dict[str, object]],
    idxs: List[int],
    model: DirectionalForceHead,
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, float]]:
    device = torch.device(args.device)
    x = torch.as_tensor(np.stack([rows[i]["x"] for i in idxs], axis=0), dtype=torch.float32, device=device)
    probs = torch.softmax(model(x), dim=-1)
    active_probs, active_idx_t = torch.max(probs[:, 1:], dim=-1)
    scores = (active_probs - probs[:, 0]).cpu().numpy()
    active_cls = (active_idx_t.cpu().numpy() + 1).astype(np.int64)

    true_cls = np.asarray([int(rows[i]["label"]) for i in idxs], dtype=np.int64)
    true_active = true_cls > 0
    far_mask = np.asarray([str(rows[i]["regime"]) in ("R2", "R3") for i in idxs], dtype=bool)
    best_idx = np.asarray([int(rows[i]["best_idx"]) for i in idxs], dtype=np.int64)
    active_dir = DIRS_16[active_cls - 1]
    safe_dir = DIRS_16[best_idx]
    correct_activation = true_active & (np.sum(active_dir * safe_dir, axis=1) > args.force_eps)
    correct_if_active = active_cls == true_cls
    true_inactive = ~true_active

    order = np.argsort(-scores)
    scores_s = scores[order]
    correct_activation_s = correct_activation[order].astype(np.float64)
    far_s = far_mask[order].astype(np.float64)
    correct_if_active_s = correct_if_active[order].astype(np.float64)
    true_inactive_s = true_inactive[order].astype(np.float64)

    cum_correct_activation = np.cumsum(correct_activation_s)
    cum_far = np.cumsum(far_s)
    cum_correct_if_active = np.cumsum(correct_if_active_s)
    cum_true_inactive_active = np.cumsum(true_inactive_s)
    car_den = float(np.maximum(true_active.sum(), 1))
    far_den = float(np.maximum(far_mask.sum(), 1))
    inactive_total = float(true_inactive.sum())
    n = float(len(idxs))

    # Candidate zero-active operating point.
    best_threshold = float(scores.max() + 1e-6)
    best_metrics = {
        "n": n,
        "accuracy": float(inactive_total / max(n, 1.0)),
        "correct_activation_rate": 0.0,
        "false_activation_rate": 0.0,
        "selectivity_ratio": 0.0,
    }
    best_key = (
        best_metrics["correct_activation_rate"],
        -best_metrics["false_activation_rate"],
        best_metrics["accuracy"],
    )

    unique_scores, counts = np.unique(scores_s, return_counts=True)
    # np.unique sorts ascending; convert counts into descending-score end indices.
    unique_scores_desc = unique_scores[::-1]
    counts_desc = counts[::-1]
    end_indices_desc = np.cumsum(counts_desc) - 1
    for threshold, end in zip(unique_scores_desc, end_indices_desc):
        end_i = int(end)
        far = float(cum_far[end_i] / far_den)
        if far > float(args.calibrate_target_far):
            continue
        car = float(cum_correct_activation[end_i] / car_den)
        correct = float(cum_correct_if_active[end_i] + inactive_total - cum_true_inactive_active[end_i])
        metrics = {
            "n": n,
            "accuracy": float(correct / max(n, 1.0)),
            "correct_activation_rate": car,
            "false_activation_rate": far,
            "selectivity_ratio": float("nan"),
        }
        key = (car, -far, metrics["accuracy"])
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


@torch.no_grad()
def evaluate(rows: List[Dict[str, object]], idxs: List[int], model: DirectionalForceHead, args: argparse.Namespace) -> Dict[str, float]:
    pred = _predict_classes(
        rows,
        idxs,
        model,
        args,
        activation_threshold=args.activation_threshold,
    )
    return _metrics_from_pred(rows, idxs, pred, args)


def write_rows_csv(rows: List[Dict[str, object]], idxs: List[int], path: Path) -> None:
    fields = ["episode_id", "sequence", "regime", "label", "has_safe_alt", "best_idx", "scaffold_risk", "safe_risk"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in idxs:
            writer.writerow({k: rows[i][k] for k in fields})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a RELLIS candidate-direction Stage-2 selectivity head.")
    ap.add_argument("--bev-root", type=Path, default=ROOT / "cache" / "rellis_bev_val_main_1500")
    ap.add_argument("--pairs-root", type=Path, default=ROOT / "cache" / "rellis_pairs_val_main_1500_balanced")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rellis_directional_force")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--horizon-cells", type=int, default=8)
    ap.add_argument("--long-horizon-cells", type=int, default=24)
    ap.add_argument("--hard-margin-m", type=float, default=1.0)
    ap.add_argument("--improvement-margin", type=float, default=0.1)
    ap.add_argument("--route-aware", action="store_true",
                    help="Add route-to-go candidate features and use route-aware R1 activation labels.")
    ap.add_argument("--route-risk-weight", type=float, default=12.0)
    ap.add_argument("--route-max-ratio", type=float, default=2.2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-mode", choices=["episode", "episode_seeded", "sequence"], default="episode")
    ap.add_argument("--holdout-sequence", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--activation-weight", type=float, default=2.0)
    ap.add_argument("--no-activation-weight", type=float, default=1.0)
    ap.add_argument("--activation-threshold", type=float, default=None,
                    help="If set, activate only when max active probability minus no-activation probability exceeds this threshold.")
    ap.add_argument("--calibrate-target-far", type=float, default=None,
                    help="If set, choose activation threshold on the training split to keep FAR at or below this value.")
    ap.add_argument("--force-eps", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, meta = build_dataset(args)
    if not rows:
        raise RuntimeError("No directional decision rows were built.")
    train_idx, val_idx = split_rows(
        rows,
        args.val_frac,
        seed=args.seed,
        split_mode=args.split_mode,
        holdout_sequence=args.holdout_sequence,
    )
    fit = train_model(rows, train_idx, val_idx, args)
    model = fit["model"]
    calibration_metrics = None
    if args.calibrate_target_far is not None:
        threshold, calibration_metrics = calibrate_activation_threshold(rows, train_idx, model, args)
        args.activation_threshold = threshold
        print(f"Calibrated activation_threshold={threshold:.6f} for train FAR<={args.calibrate_target_far:.3f}")
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "data": meta,
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "class_counts": fit["class_counts"],
        "best_epoch": fit["best"]["epoch"],
        "best_val_loss": fit["best"]["val_loss"],
        "activation_threshold": args.activation_threshold,
        "calibration_metrics": calibration_metrics,
        "train_metrics": evaluate(rows, train_idx, model, args),
        "val_metrics": evaluate(rows, val_idx, model, args),
        "all_metrics": evaluate(rows, list(range(len(rows))), model, args),
    }
    torch.save(
        {
            "model_state_dict": fit["best"]["state_dict"],
            "summary": summary,
            "in_dim": int(np.asarray(rows[0]["x"]).shape[0]),
            "dirs": DIRS_16,
        },
        args.out / "best.pt",
    )
    write_rows_csv(rows, train_idx, args.out / "train_rows.csv")
    write_rows_csv(rows, val_idx, args.out / "val_rows.csv")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
