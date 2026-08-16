#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_dfc2018_stagewise import (
    extract_local_geom_obstacles,
    extract_risk_patch,
    extract_rollout_patch,
)

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


def _as_path(raw: List[List[int]]) -> List[Tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in raw]


def _xy(path_rc: List[Tuple[int, int]], idx: int) -> List[float]:
    r, c = path_rc[int(np.clip(idx, 0, len(path_rc) - 1))]
    return [float(c), float(r)]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def _nearest_path_index(path_rc: List[Tuple[int, int]], point_rc: Tuple[int, int]) -> int:
    if not path_rc:
        return 0
    arr = np.asarray(path_rc, dtype=np.float32)
    p = np.asarray(point_rc, dtype=np.float32)
    return int(np.argmin(((arr - p[None, :]) ** 2).sum(axis=1)))


def _direction_integral(
    maps: Dict[str, np.ndarray],
    pos_rc: np.ndarray,
    direction_rc: np.ndarray,
    *,
    horizon_cells: int,
    hard_margin_m: float,
) -> Tuple[float, bool]:
    risk = maps["risk_map"]
    hard = maps["hard_mask"].astype(bool)
    sdf = maps["sdf_hard"]
    total = 0.0
    feasible = True
    for step in range(1, horizon_cells + 1):
        q = pos_rc + float(step) * direction_rc
        r = int(round(float(q[0])))
        c = int(round(float(q[1])))
        if not (0 <= r < risk.shape[0] and 0 <= c < risk.shape[1]):
            feasible = False
            break
        total += float(risk[r, c])
        if hard[r, c] or float(sdf[r, c]) < hard_margin_m:
            feasible = False
            break
    return total, feasible


def _selectivity_label(
    maps: Dict[str, np.ndarray],
    source_path_rc: List[Tuple[int, int]],
    scaffold_path_rc: List[Tuple[int, int]],
    idx: int,
    *,
    regime: str,
    horizon_cells: int = 8,
    hard_margin_m: float = 1.0,
    improvement_margin: float = 0.1,
) -> Dict[str, float | List[float]]:
    cr, cc = source_path_rc[idx]
    p = np.asarray([float(cr), float(cc)], dtype=np.float32)
    si = _nearest_path_index(scaffold_path_rc, (cr, cc))
    sn = min(si + 1, len(scaffold_path_rc) - 1)
    scaffold_rc = _unit(
        np.asarray(scaffold_path_rc[sn], dtype=np.float32)
        - np.asarray(scaffold_path_rc[si], dtype=np.float32)
    )
    scaffold_risk, scaffold_feasible = _direction_integral(
        maps, p, scaffold_rc, horizon_cells=horizon_cells, hard_margin_m=hard_margin_m
    )
    best_dir: Optional[np.ndarray] = None
    best_risk = float("inf")
    feasible_count = 0
    for d in DIRS_16:
        cand_risk, feasible = _direction_integral(
            maps, p, d, horizon_cells=horizon_cells, hard_margin_m=hard_margin_m
        )
        feasible_count += int(feasible)
        if feasible and cand_risk < best_risk:
            best_risk = cand_risk
            best_dir = d
    if best_dir is None:
        best_dir = scaffold_rc
        best_risk = scaffold_risk
    has_safe_alt = bool(scaffold_feasible and (scaffold_risk - best_risk) >= improvement_margin)
    active = bool(regime == "R1" and has_safe_alt)
    # Training forces use xy=(col,row), while path utilities use rc=(row,col).
    safe_xy = [float(best_dir[1]), float(best_dir[0])]
    scaffold_xy = [float(scaffold_rc[1]), float(scaffold_rc[0])]
    return {
        "selectivity_active": float(active),
        "selectivity_mask": 1.0,
        "safe_dir": safe_xy,
        "scaffold_dir": scaffold_xy,
        "has_safe_alt": float(has_safe_alt),
        "scaffold_risk": float(scaffold_risk),
        "safe_risk": float(best_risk),
        "feasible_direction_count": float(feasible_count),
    }


def _checkpoint(
    maps: Dict[str, np.ndarray],
    path_rc: List[Tuple[int, int]],
    idx: int,
    *,
    regime: str,
    scaffold_path_rc: List[Tuple[int, int]],
    target_path_rc: Optional[List[Tuple[int, int]]],
    dt: float,
    path_stride: int,
    patch_size: int,
    robot_radius: float,
    margin_factor: float,
    selectivity_override: Optional[Dict[str, float | List[float]]] = None,
) -> Dict:
    center = _xy(path_rc, idx)
    next_idx = min(idx + 1, len(path_rc) - 1)
    target_idx = min(idx + path_stride, len(path_rc) - 1)
    goal_idx = min(idx + max(path_stride * 3, 6), len(path_rc) - 1)
    if target_path_rc:
        nearest_target = _nearest_path_index(target_path_rc, path_rc[idx])
        target_idx = min(nearest_target + path_stride, len(target_path_rc) - 1)
        goal_idx = min(nearest_target + max(path_stride * 3, 6), len(target_path_rc) - 1)
    cr, cc = path_rc[idx]
    centers, radii, weights, d_hat = extract_local_geom_obstacles(
        maps["geom_occ"],
        (cr, cc),
        patch_size=64,
        robot_radius=robot_radius,
        margin_factor=margin_factor,
    )
    risk_patch, _ = extract_risk_patch(maps, (cr, cc), patch_size)
    rollout_patch = extract_rollout_patch(maps, (cr, cc), patch_size)
    target_path = target_path_rc or path_rc
    o_tgt = np.asarray(_xy(target_path, target_idx), dtype=np.float32)
    o_next = np.asarray(_xy(path_rc, next_idx), dtype=np.float32)
    o_cur = np.asarray(center, dtype=np.float32)
    v_tgt = (o_next - o_cur) / max(float(dt), 1e-6)
    selectivity = selectivity_override or _selectivity_label(
        maps, path_rc, scaffold_path_rc, idx, regime=regime
    )
    return {
        "t": int(idx),
        "dt": float(dt),
        "stage_idx": int(idx // max(1, path_stride)),
        "stage_bounds": [int(idx), int(goal_idx)],
        "stage_entry": center,
        "stage_exit": _xy(target_path, goal_idx),
        "center": center,
        "theta": 0.0,
        "min_d": float(maps["sdf_hard"][cr, cc]),
        "barrier": {"barrier_d_hat": float(d_hat)},
        "obstacles_effective": {
            "C": centers.astype(float).tolist(),
            "R_eff": radii.astype(float).tolist(),
            "W": weights.astype(float).tolist(),
        },
        "risk_patch": np.asarray(risk_patch, dtype=np.float32).tolist(),
        "rollout_patch": np.asarray(rollout_patch, dtype=np.float32).tolist(),
        "risk_grad": [
            float(maps["grad_col"][cr, cc]),
            float(maps["grad_row"][cr, cc]),
        ],
        "sdf_at_center": float(maps["sdf_hard"][cr, cc]),
        "o_tgt": o_tgt.astype(float).tolist(),
        "v_tgt": v_tgt.astype(float).tolist(),
        "selectivity": selectivity,
    }


def _decision_indices(
    maps: Dict[str, np.ndarray],
    path_rc: List[Tuple[int, int]],
    scaffold_path_rc: List[Tuple[int, int]],
    *,
    regime: str,
    stride: int,
    neg_per_active: int,
    min_negatives: int,
    max_per_episode: int,
    rng: np.random.Generator,
) -> Tuple[List[int], Dict[int, Dict[str, float | List[float]]]]:
    candidates = list(range(0, len(path_rc), max(1, stride)))
    if candidates[-1] != len(path_rc) - 1:
        candidates.append(len(path_rc) - 1)

    labels = {
        idx: _selectivity_label(maps, path_rc, scaffold_path_rc, idx, regime=regime)
        for idx in candidates
    }
    active = [idx for idx in candidates if float(labels[idx]["selectivity_active"]) > 0.5]
    inactive = [idx for idx in candidates if idx not in set(active)]

    def neg_score(idx: int) -> float:
        lab = labels[idx]
        risk_gap = float(lab["scaffold_risk"]) - float(lab["safe_risk"])
        return risk_gap + 0.05 * float(lab["scaffold_risk"])

    keep = set(active)
    n_neg = max(min_negatives, neg_per_active * max(1, len(active)))
    inactive_ranked = sorted(inactive, key=neg_score, reverse=True)
    keep.update(inactive_ranked[:n_neg])

    # Preserve a few ordinary trajectory points so rollout targets remain sane.
    keep.update(candidates[:: max(1, len(candidates) // 3)])
    keep.add(candidates[0])
    keep.add(candidates[-1])

    if len(keep) > max_per_episode:
        active_keep = [idx for idx in active if idx in keep]
        remaining = [idx for idx in sorted(keep) if idx not in set(active_keep)]
        budget = max(0, max_per_episode - len(active_keep))
        if len(remaining) > budget:
            ranked_remaining = sorted(remaining, key=lambda idx: (idx in inactive_ranked[:n_neg], neg_score(idx)), reverse=True)
            remaining = ranked_remaining[:budget]
        keep = set(active_keep + remaining)

    if len(keep) < 3:
        keep.update(candidates[: min(3, len(candidates))])
        keep.update(candidates[-min(3, len(candidates)):])

    selected = sorted(keep)
    return selected, {idx: labels[idx] for idx in selected}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build train_material-compatible RELLIS stagewise episodes.")
    ap.add_argument("--bev-root", type=Path, default=ROOT / "cache" / "rellis_bev_val_main_100")
    ap.add_argument("--pairs-root", type=Path, default=ROOT / "cache" / "rellis_pairs_val_main_100")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "rellis_stagewise_val100")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--path-stride", type=int, default=6)
    ap.add_argument("--checkpoint-stride", type=int, default=3)
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--robot-radius", type=float, default=1.5)
    ap.add_argument("--margin-factor", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--checkpoint-source", choices=["risk", "stage1"], default="risk",
                    help="Path used for checkpoint centers. stage1 trains forces from the geometry scaffold.")
    ap.add_argument("--checkpoint-policy", choices=["uniform", "decision"], default="uniform",
                    help="uniform stores full path checkpoints; decision stores active R1 points plus hard negatives.")
    ap.add_argument("--decision-neg-per-active", type=int, default=6)
    ap.add_argument("--decision-min-negatives", type=int, default=8)
    ap.add_argument("--decision-max-per-episode", type=int, default=32)
    ap.add_argument("--seed", type=int, default=11)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "episodes").mkdir(exist_ok=True)
    manifest_pairs = json.loads((args.pairs_root / "manifest.json").read_text())
    episodes = manifest_pairs["episodes"][: args.max_episodes]
    by_regime: Dict[str, List[int]] = {}
    for i, ep in enumerate(episodes):
        by_regime.setdefault(str(ep["regime"]), []).append(i)
    val_indices = set()
    for idxs in by_regime.values():
        n_val = max(1, int(round(len(idxs) * args.val_frac))) if len(idxs) > 1 else 0
        val_indices.update(idxs[-n_val:])

    seen_scenes = set()
    manifest = []
    rng = np.random.default_rng(args.seed)
    for ep_idx, ep in enumerate(tqdm(episodes, desc="building RELLIS stagewise")):
        scene = torch.load(args.bev_root / ep["scene_path"], map_location="cpu", weights_only=False)
        scene_id = str(ep["scene_id"])
        if scene_id not in seen_scenes:
            torch.save(scene, args.out / f"scene_{scene_id}.pt")
            seen_scenes.add(scene_id)
        maps = scene["maps"]
        risk_path_rc = _as_path(ep["risk_path"])
        stage1_path_rc = _as_path(ep["stage1_path"])
        path_rc = stage1_path_rc if args.checkpoint_source == "stage1" else risk_path_rc
        if len(path_rc) < 3:
            continue
        label_overrides: Dict[int, Dict[str, float | List[float]]] = {}
        if args.checkpoint_policy == "decision":
            indices, label_overrides = _decision_indices(
                maps,
                path_rc,
                stage1_path_rc,
                regime=ep["regime"],
                stride=args.checkpoint_stride,
                neg_per_active=args.decision_neg_per_active,
                min_negatives=args.decision_min_negatives,
                max_per_episode=args.decision_max_per_episode,
                rng=rng,
            )
        else:
            indices = list(range(0, len(path_rc), max(1, args.checkpoint_stride)))
            if indices[-1] != len(path_rc) - 1:
                indices.append(len(path_rc) - 1)
        checkpoints = [
            _checkpoint(
                maps,
                path_rc,
                idx,
                regime=ep["regime"],
                scaffold_path_rc=stage1_path_rc,
                target_path_rc=risk_path_rc if args.checkpoint_source == "stage1" else None,
                dt=args.dt,
                path_stride=args.path_stride,
                patch_size=args.patch_size,
                robot_radius=args.robot_radius,
                margin_factor=args.margin_factor,
                selectivity_override=label_overrides.get(idx),
            )
            for idx in indices
        ]
        ep_dir = args.out / "episodes" / f"ep_{ep_idx:05d}"
        logs_dir = ep_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ck_path = logs_dir / "stagewise_checkpoints.jsonl"
        with ck_path.open("w") as f:
            for ck in checkpoints:
                f.write(json.dumps(ck) + "\n")
        payload = {
            "meta": {
                "episode_id": f"{ep_idx:05d}",
                "source_episode_id": ep["episode_id"],
                "scene_id": scene_id,
                "split": "val" if ep_idx in val_indices else "train",
                "regime": ep["regime"],
                "start_rc": ep["start_rc"],
                "goal_rc": ep["goal_rc"],
                "dt": float(args.dt),
                "path_stride": int(args.path_stride),
                "checkpoint_source": args.checkpoint_source,
                "checkpoint_policy": args.checkpoint_policy,
            },
            "params": {
                "gamma_o": 4.0,
                "robot_radius": float(args.robot_radius),
                "margin_factor": float(args.margin_factor),
            },
            "success": True,
            "final_center": _xy(risk_path_rc, len(risk_path_rc) - 1),
            "logs": {"checkpoints_jsonl": str(ck_path)},
        }
        ep_path = ep_dir / "episode.pt"
        torch.save(payload, ep_path)
        manifest.append(
            {
                "episode_id": f"{ep_idx:05d}",
                "source_episode_id": ep["episode_id"],
                "scene_id": scene_id,
                "split": payload["meta"]["split"],
                "regime": ep["regime"],
                "path": str(ep_path),
                "start_rc": ep["start_rc"],
                "goal_rc": ep["goal_rc"],
                "success": True,
            }
        )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(args.pairs_root / "manifest.json", args.out / "source_pairs_manifest.json")
    print(f"Wrote {len(manifest)} RELLIS stagewise episodes to {args.out}")


if __name__ == "__main__":
    main()
