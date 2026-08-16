#!/usr/bin/env python3
"""
analyze_stage1.py

Diagnostic pass for Stage 1 highway-env checkpoints.

This script is intentionally measurement-first. It writes CSV/JSON artifacts
and optional matplotlib plots for:
  1. IDM dataset target statistics.
  2. Open-loop model rollout quality on the validation split.
  3. Closed-loop model traces on fixed seeds.
  4. IDM closed-loop baseline traces on the same seeds.

The goal is to decide whether failures come from open-loop imitation,
force/action conversion, receding-goal closed-loop instability, or a baseline
environment difficulty mismatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
LOCAL_HIGHWAY_ENV = HERE / "HighwayEnv"
if LOCAL_HIGHWAY_ENV.exists():
    sys.path.insert(0, str(LOCAL_HIGHWAY_ENV))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from bicycle_surrogate import ACCEL_RANGE, STEER_RANGE, force_to_action  # noqa: E402
from data_collect_idm import (  # noqa: E402
    H_TARGET, DT_TARGET, POLICY_FREQ, SIM_FREQ,
    _idle_action, _import_highway_env, _reset as idm_reset,
    _step as idm_step, make_env as make_idm_env,
    replace_ego_with_idm,
)
from env_wrapper import HighwayMaterialObservation, WrapperConfig, _ego_lane_center_y  # noqa: E402
from eval_stage1 import (  # noqa: E402
    _apply_alpha_floor, _force_component_diagnostics,
    _import_gym, _physical_to_normalized_action, _reset, _step,
    disable_transformer_nested_tensors, make_env as make_eval_env,
)
from surrogate_integrator import compute_surrogate_highway_force, integrate_surrogate_highway  # noqa: E402
from train_stage1 import Stage1IDMDataset, collate, _model_forward  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _summarize_rows(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Dict[str, float]]:
    return {k: _summarize(float(r[k]) for r in rows if k in r and r[k] != "") for k in keys}


def _to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=device.startswith("cuda"))
                if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def _speed(v: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(v, dim=-1)


def _nearest_geometry(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    o0 = batch["o0"]
    C = batch["C"]
    R = batch["R"]
    mask = batch["mask"]
    rel = C - o0.unsqueeze(1)
    dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
    clear = (dist - R).masked_fill(~mask, float("inf"))
    idx = clear.argmin(dim=1)
    row = torch.arange(o0.shape[0], device=o0.device)
    rel_i = rel[row, idx]
    vhat = torch.nn.functional.normalize(batch["v0"], dim=-1, eps=1e-6)
    lat = torch.stack([-vhat[:, 1], vhat[:, 0]], dim=-1)
    return {
        "clear": clear[row, idx],
        "rel_lon": (rel_i * vhat).sum(dim=-1),
        "rel_lat": (rel_i * lat).sum(dim=-1),
        "valid_count": mask.sum(dim=1).float(),
    }


def _model_coeffs(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    return _model_forward(model, batch)


def _load_checkpoint(ckpt: Path, device: str, dfc_root: str = ""):
    if dfc_root:
        sys.path.insert(0, dfc_root)
    from train_material import CoefEnergyNetMaterial  # noqa: E402

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {})
    model = CoefEnergyNetMaterial(
        lam_soft_max=float(cfg.get("lam_soft_max", 50.0)),
        lam_hard_max=float(cfg.get("lam_hard_max", 10.0)),
    ).to(device)
    disable_transformer_nested_tensors(model)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing:
        print(f"Missing keys (using init): {missing}")
    if unexpected:
        print(f"Unexpected keys (ignored): {unexpected}")
    model.eval()
    return ck, cfg, model


def _resolve_runtime_knobs(args, cfg: Dict[str, Any]) -> Dict[str, Any]:
    d_hat = args.d_hat if args.d_hat > 0 else float(cfg.get("d_hat", 0.0))
    alpha_floor = (args.alpha_floor if args.alpha_floor >= 0
                   else float(cfg.get("alpha_floor", 0.0)))
    if args.alpha_floor_ahead_only is not None:
        ahead_only = bool(args.alpha_floor_ahead_only)
    else:
        # Backward-compatible: old checkpoints with alpha_floor but no scope
        # were trained with all-obstacle floor.
        ahead_only = bool(cfg.get("alpha_floor_ahead_only", False))
    return {
        "d_hat": d_hat,
        "alpha_floor": alpha_floor,
        "alpha_floor_ahead_only": ahead_only,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset stats
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dataset(data_dir: Path, out_dir: Path, *, split: str,
                    max_samples: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ds = Stage1IDMDataset(data_dir / "manifest.json", split=split)
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    if max_samples > 0 and max_samples < len(idxs):
        idxs = rng.sample(idxs, max_samples)
    rows: List[Dict[str, Any]] = []
    for idx in idxs:
        s = ds[idx]
        batch = {k: v.unsqueeze(0) for k, v in s.items()}
        near = _nearest_geometry(batch)
        v0 = float(_speed(batch["v0"]).item())
        vt = float(_speed(batch["v_tgt"]).item())
        rows.append({
            "idx": idx,
            "speed0": v0,
            "speed_tgt": vt,
            "delta_speed_tgt": vt - v0,
            "target_disp": float(torch.linalg.norm(batch["o_tgt"] - batch["o0"], dim=-1).item()),
            "target_lon_disp": float(((batch["o_tgt"] - batch["o0"]) * torch.nn.functional.normalize(batch["v0"], dim=-1, eps=1e-6)).sum().item()),
            "dmin": float(near["clear"].item()),
            "nearest_lon": float(near["rel_lon"].item()),
            "nearest_lat": float(near["rel_lat"].item()),
            "valid_count": float(near["valid_count"].item()),
            "goal_dist": float(torch.linalg.norm(batch["goal"] - batch["o0"], dim=-1).item()),
            "lane_goal_offset": float(abs((batch["goal"] - batch["o0"])[0, 1].item())),
        })
    _write_csv(out_dir / "dataset_stats.csv", rows)
    summary = _summarize_rows(rows, [
        "speed0", "speed_tgt", "delta_speed_tgt", "target_disp",
        "target_lon_disp", "dmin", "nearest_lon", "nearest_lat",
        "valid_count", "goal_dist", "lane_goal_offset",
    ])
    return rows, summary


# ─────────────────────────────────────────────────────────────────────────────
# Open-loop model diagnostics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_open_loop(
    data_dir: Path,
    out_dir: Path,
    model,
    *,
    device: str,
    split: str,
    max_samples: int,
    batch_size: int,
    seed: int,
    d_hat: float,
    alpha_floor: float,
    alpha_floor_ahead_only: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ds = Stage1IDMDataset(data_dir / "manifest.json", split=split)
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    if max_samples > 0 and max_samples < len(idxs):
        idxs = rng.sample(idxs, max_samples)
    subset = Subset(ds, idxs)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)
    rows: List[Dict[str, Any]] = []
    global_i = 0
    for batch in loader:
        batch = _to_device(batch, device)
        if d_hat > 0:
            batch["d_hat"] = torch.full_like(batch["d_hat"], float(d_hat))
        alphas, beta, gamma, lam_soft, lam_hard, _mu_lat = _model_coeffs(model, batch)
        alphas = _apply_alpha_floor(
            batch, alphas, alpha_floor, ahead_only=alpha_floor_ahead_only
        )
        lam_soft = torch.zeros_like(lam_soft)
        lam_hard = torch.zeros_like(lam_hard)
        H = batch["H"].long() if batch["H"].dtype != torch.long else batch["H"]
        oT, vT, min_clear, cum_risk, hard_count, arc_length = integrate_surrogate_highway(
            o0=batch["o0"], v0=batch["v0"], goal=batch["goal"],
            C=batch["C"], R=batch["R"], mask=batch["mask"],
            alphas=alphas, beta=beta, gamma=gamma,
            lam_soft=lam_soft, lam_hard=lam_hard,
            rollout_patch=batch["rollout_patch"], d_hat=batch["d_hat"],
            dt=batch["dt"], H=H,
        )

        v0 = batch["v0"]
        speed0 = _speed(v0).clamp_min(1e-3)
        heading0 = torch.atan2(v0[:, 1], v0[:, 0])
        F, dmin_now, risk_val, sdf_val = compute_surrogate_highway_force(
            o=batch["o0"], heading=heading0, speed=speed0,
            o0=batch["o0"], heading_0=heading0, goal=batch["goal"],
            C=batch["C"], R_eff=batch["R"], mask=batch["mask"],
            alphas=alphas, beta=beta, gamma=gamma,
            lam_soft=lam_soft, lam_hard=lam_hard,
            rollout_patch=batch["rollout_patch"], d_hat=batch["d_hat"],
        )
        accel, steer = force_to_action(F, heading0, speed0)
        near = _nearest_geometry(batch)

        traj_err = torch.linalg.norm(oT - batch["o_tgt"], dim=-1)
        vel_err = torch.linalg.norm(vT - batch["v_tgt"], dim=-1)
        speed_pred = _speed(vT)
        speed_tgt = _speed(batch["v_tgt"])
        valid = batch["mask"]
        alpha_mean = torch.where(
            valid.any(dim=1),
            (alphas * valid.to(alphas.dtype)).sum(dim=1) / valid.sum(dim=1).clamp_min(1),
            torch.zeros_like(beta),
        )

        for j in range(oT.shape[0]):
            row = {
                "row": global_i,
                "speed0": float(speed0[j].item()),
                "speed_pred": float(speed_pred[j].item()),
                "speed_tgt": float(speed_tgt[j].item()),
                "speed_err": float((speed_pred[j] - speed_tgt[j]).item()),
                "traj_err": float(traj_err[j].item()),
                "vel_err": float(vel_err[j].item()),
                "min_clear_rollout": float(min_clear[j].item()),
                "dmin_now": float(dmin_now[j].item()),
                "nearest_lon": float(near["rel_lon"][j].item()),
                "nearest_lat": float(near["rel_lat"][j].item()),
                "accel_raw": float(accel[j].item()),
                "steer_raw": float(steer[j].item()),
                "accel_clip": float(accel[j].clamp(*ACCEL_RANGE).item()),
                "steer_clip": float(steer[j].clamp(*STEER_RANGE).item()),
                "alpha_max": float(alphas[j][valid[j]].max().item()) if bool(valid[j].any().item()) else 0.0,
                "alpha_mean": float(alpha_mean[j].item()),
                "beta": float(beta[j].item()),
                "gamma": float(gamma[j].item()),
                "cum_risk": float(cum_risk[j].item()),
                "hard_count": float(hard_count[j].item()),
                "arc_length": float(arc_length[j].item()),
                "risk_now": float(risk_val[j].item()),
                "sdf_now": float(sdf_val[j].item()),
            }
            if "accel_tgt" in batch and "steer_tgt" in batch:
                accel_clip = accel[j].clamp(*ACCEL_RANGE)
                steer_clip = steer[j].clamp(*STEER_RANGE)
                row.update({
                    "accel_tgt": float(batch["accel_tgt"][j].item()),
                    "steer_tgt": float(batch["steer_tgt"][j].item()),
                    "accel_err": float((accel_clip - batch["accel_tgt"][j]).item()),
                    "steer_err": float((steer_clip - batch["steer_tgt"][j]).item()),
                    "has_action_tgt": float(batch["has_action_tgt"][j].item()),
                })
            rows.append(row)
            global_i += 1
    _write_csv(out_dir / "open_loop.csv", rows)
    summary = _summarize_rows(rows, [
        "speed0", "speed_pred", "speed_tgt", "speed_err", "traj_err",
        "vel_err", "min_clear_rollout", "dmin_now", "nearest_lon",
        "accel_raw", "steer_raw", "accel_tgt", "steer_tgt",
        "accel_err", "steer_err", "alpha_max", "beta", "gamma",
    ])
    return rows, summary


# ─────────────────────────────────────────────────────────────────────────────
# Closed-loop traces
# ─────────────────────────────────────────────────────────────────────────────

def _lane_error(env) -> float:
    ego_pos = np.asarray(env.unwrapped.vehicle.position, dtype=np.float64)
    try:
        lane_y = _ego_lane_center_y(env, ego_pos)
        return abs(float(ego_pos[1]) - float(lane_y))
    except Exception:
        return float("nan")


def _lane_change(prev_idx, cur_idx) -> bool:
    if prev_idx is None or cur_idx is None:
        return False
    try:
        return prev_idx[2] != cur_idx[2]
    except (IndexError, TypeError):
        return prev_idx != cur_idx


@torch.no_grad()
def closed_loop_model_traces(
    out_dir: Path,
    model,
    *,
    device: str,
    env_id: str,
    seed0: int,
    episodes: int,
    max_steps: int,
    vehicles_count: int,
    lanes_count: int,
    n_max_vehicles: int,
    d_hat: float,
    alpha_floor: float,
    alpha_floor_ahead_only: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    gym = _import_gym()
    env = make_eval_env(gym, env_id, vehicles_count=vehicles_count,
                        lanes_count=lanes_count)
    observer = HighwayMaterialObservation(WrapperConfig(
        n_max_vehicles=n_max_vehicles,
        dt_surrogate=DT_TARGET,
        horizon_surrogate=H_TARGET,
    ))
    rows: List[Dict[str, Any]] = []
    eps: List[Dict[str, Any]] = []
    try:
        for ep in range(episodes):
            seed = seed0 + ep
            print(f"    model ep {ep + 1:03d}/{episodes:03d} seed={seed}", flush=True)
            _reset(env, seed)
            uenv = env.unwrapped
            o_start = np.asarray(uenv.vehicle.position, dtype=np.float64).copy()
            prev_lane = getattr(uenv.vehicle, "lane_index", None)
            lane_changes = 0
            collided = False
            truncated = False
            for step in range(max_steps):
                obs_np = observer.build(env)
                batch = {k: torch.as_tensor(v).unsqueeze(0).to(device)
                         for k, v in obs_np.items()}
                for k, v in list(batch.items()):
                    if v.dtype == torch.float64:
                        batch[k] = v.float()
                if d_hat > 0:
                    batch["d_hat"] = torch.full_like(batch["d_hat"], float(d_hat))

                alphas, beta, gamma, lam_soft, lam_hard, _mu_lat = _model_coeffs(model, batch)
                alphas = _apply_alpha_floor(
                    batch, alphas, alpha_floor, ahead_only=alpha_floor_ahead_only
                )
                lam_soft = torch.zeros_like(lam_soft)
                lam_hard = torch.zeros_like(lam_hard)
                v0 = batch["v0"]
                speed = _speed(v0).clamp_min(1e-3)
                heading = torch.atan2(v0[:, 1], v0[:, 0])
                F, dmin, risk_val, sdf_val = compute_surrogate_highway_force(
                    o=batch["o0"], heading=heading, speed=speed,
                    o0=batch["o0"], heading_0=heading,
                    goal=batch["goal"], C=batch["C"], R_eff=batch["R"],
                    mask=batch["mask"], alphas=alphas,
                    beta=beta, gamma=gamma,
                    lam_soft=lam_soft, lam_hard=lam_hard,
                    rollout_patch=batch["rollout_patch"], d_hat=batch["d_hat"],
                )
                accel, steer = force_to_action(F, heading, speed)
                accel_raw = float(accel.item())
                steer_raw = float(steer.item())
                accel_c = float(accel.clamp(*ACCEL_RANGE).item())
                steer_c = float(steer.clamp(*STEER_RANGE).item())
                action = _physical_to_normalized_action(accel_c, steer_c)
                diag = _force_component_diagnostics(
                    batch=batch, alphas=alphas, beta=beta, gamma=gamma,
                    heading=heading, speed=speed,
                )
                ego_pos = np.asarray(uenv.vehicle.position, dtype=np.float64)
                rows.append({
                    "episode": ep,
                    "seed": seed,
                    "step": step,
                    "x": float(ego_pos[0]),
                    "y": float(ego_pos[1]),
                    "speed": float(speed.item()),
                    "lane_error": _lane_error(env),
                    "dmin": float(dmin.item()),
                    "risk": float(risk_val.item()),
                    "sdf": float(sdf_val.item()),
                    "accel_raw": accel_raw,
                    "steer_raw": steer_raw,
                    "accel": accel_c,
                    "steer": steer_c,
                    "alpha_max": float(alphas[batch["mask"]].max().item()) if bool(batch["mask"].any().item()) else 0.0,
                    "beta": float(beta.item()),
                    "gamma": float(gamma.item()),
                    **diag,
                })
                _, _, term, trunc, info = _step(env, action)
                cur_lane = getattr(uenv.vehicle, "lane_index", None)
                if _lane_change(prev_lane, cur_lane):
                    lane_changes += 1
                prev_lane = cur_lane
                if term:
                    collided = bool(info.get("crashed", True))
                    break
                if trunc:
                    truncated = True
                    break
            o_end = np.asarray(uenv.vehicle.position, dtype=np.float64).copy()
            ep_rows = [r for r in rows if r["episode"] == ep]
            eps.append({
                "episode": ep,
                "seed": seed,
                "steps": len(ep_rows),
                "collided": collided,
                "truncated": truncated,
                "distance_m": float(np.linalg.norm(o_end - o_start)),
                "mean_speed": float(np.mean([r["speed"] for r in ep_rows])) if ep_rows else 0.0,
                "lane_keep_err": float(np.nanmean([r["lane_error"] for r in ep_rows])) if ep_rows else float("nan"),
                "lane_changes": lane_changes,
            })
    finally:
        env.close()

    _write_csv(out_dir / "closed_loop_model_trace.csv", rows)
    _write_csv(out_dir / "closed_loop_model_episodes.csv", eps)
    summary = {
        "episodes": len(eps),
        "collision_rate": float(np.mean([e["collided"] for e in eps])) if eps else float("nan"),
        "mean_speed": float(np.mean([e["mean_speed"] for e in eps])) if eps else float("nan"),
        "lane_keep_err": float(np.nanmean([e["lane_keep_err"] for e in eps])) if eps else float("nan"),
        "lane_changes_per_ep": float(np.mean([e["lane_changes"] for e in eps])) if eps else float("nan"),
        "mean_distance_m": float(np.mean([e["distance_m"] for e in eps])) if eps else float("nan"),
    }
    return rows, eps, summary


def closed_loop_idm_baseline(
    out_dir: Path,
    *,
    env_id: str,
    seed0: int,
    episodes: int,
    max_steps: int,
    vehicles_count: int,
    lanes_count: int,
    n_max_vehicles: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    gym, IDMVehicle = _import_highway_env()
    env = make_idm_env(gym, env_id, vehicles_count=vehicles_count,
                       lanes_count=lanes_count)
    observer = HighwayMaterialObservation(WrapperConfig(
        n_max_vehicles=n_max_vehicles,
        dt_surrogate=DT_TARGET,
        horizon_surrogate=H_TARGET,
    ))
    idle = _idle_action(env)
    rows: List[Dict[str, Any]] = []
    eps: List[Dict[str, Any]] = []
    try:
        for ep in range(episodes):
            seed = seed0 + ep
            print(f"    IDM   ep {ep + 1:03d}/{episodes:03d} seed={seed}", flush=True)
            idm_reset(env, seed)
            replace_ego_with_idm(env, IDMVehicle)
            uenv = env.unwrapped
            o_start = np.asarray(uenv.vehicle.position, dtype=np.float64).copy()
            prev_lane = getattr(uenv.vehicle, "lane_index", None)
            lane_changes = 0
            collided = False
            truncated = False
            for step in range(max_steps):
                obs = observer.build(env)
                batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
                near = _nearest_geometry(batch)
                ego_pos = np.asarray(uenv.vehicle.position, dtype=np.float64)
                speed = float(getattr(uenv.vehicle, "speed", np.linalg.norm(uenv.vehicle.velocity)))
                rows.append({
                    "episode": ep,
                    "seed": seed,
                    "step": step,
                    "x": float(ego_pos[0]),
                    "y": float(ego_pos[1]),
                    "speed": speed,
                    "lane_error": _lane_error(env),
                    "dmin": float(near["clear"].item()),
                    "nearest_lon": float(near["rel_lon"].item()),
                    "nearest_lat": float(near["rel_lat"].item()),
                })
                _, _, term, trunc, info = idm_step(env, idle)
                cur_lane = getattr(uenv.vehicle, "lane_index", None)
                if _lane_change(prev_lane, cur_lane):
                    lane_changes += 1
                prev_lane = cur_lane
                if term:
                    collided = bool(info.get("crashed", True))
                    break
                if trunc:
                    truncated = True
                    break
            o_end = np.asarray(uenv.vehicle.position, dtype=np.float64).copy()
            ep_rows = [r for r in rows if r["episode"] == ep]
            eps.append({
                "episode": ep,
                "seed": seed,
                "steps": len(ep_rows),
                "collided": collided,
                "truncated": truncated,
                "distance_m": float(np.linalg.norm(o_end - o_start)),
                "mean_speed": float(np.mean([r["speed"] for r in ep_rows])) if ep_rows else 0.0,
                "lane_keep_err": float(np.nanmean([r["lane_error"] for r in ep_rows])) if ep_rows else float("nan"),
                "lane_changes": lane_changes,
            })
    finally:
        env.close()

    _write_csv(out_dir / "closed_loop_idm_trace.csv", rows)
    _write_csv(out_dir / "closed_loop_idm_episodes.csv", eps)
    summary = {
        "episodes": len(eps),
        "collision_rate": float(np.mean([e["collided"] for e in eps])) if eps else float("nan"),
        "mean_speed": float(np.mean([e["mean_speed"] for e in eps])) if eps else float("nan"),
        "lane_keep_err": float(np.nanmean([e["lane_keep_err"] for e in eps])) if eps else float("nan"),
        "lane_changes_per_ep": float(np.mean([e["lane_changes"] for e in eps])) if eps else float("nan"),
        "mean_distance_m": float(np.mean([e["distance_m"] for e in eps])) if eps else float("nan"),
    }
    return rows, eps, summary


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _col(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows if key in r and r[key] != "" and np.isfinite(float(r[key]))])


def make_plots(out_dir: Path,
               dataset_rows: List[Dict[str, Any]],
               open_rows: List[Dict[str, Any]],
               model_trace: List[Dict[str, Any]],
               idm_trace: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plotting skipped: matplotlib unavailable ({exc})")
        return

    plot_dir = out_dir / "figs"
    _ensure_dir(plot_dir)

    def savefig(name: str):
        plt.tight_layout()
        plt.savefig(plot_dir / name, dpi=160)
        plt.close()

    if dataset_rows:
        plt.figure(figsize=(10, 6))
        plt.hist(_col(dataset_rows, "speed0"), bins=40, alpha=0.6, label="v0")
        plt.hist(_col(dataset_rows, "speed_tgt"), bins=40, alpha=0.6, label="v_tgt")
        plt.xlabel("speed (m/s)")
        plt.ylabel("count")
        plt.legend()
        plt.title("Dataset IDM Speed Distribution")
        savefig("dataset_speed_hist.png")

        plt.figure(figsize=(10, 6))
        plt.hist(_col(dataset_rows, "dmin"), bins=60)
        plt.xlabel("nearest clearance at observation (m)")
        plt.ylabel("count")
        plt.title("Dataset Nearest-Clearance Distribution")
        savefig("dataset_dmin_hist.png")

    if open_rows:
        plt.figure(figsize=(8, 8))
        plt.scatter(_col(open_rows, "speed_tgt"), _col(open_rows, "speed_pred"), s=5, alpha=0.35)
        lim = [min(0, _col(open_rows, "speed_tgt").min(), _col(open_rows, "speed_pred").min()),
               max(_col(open_rows, "speed_tgt").max(), _col(open_rows, "speed_pred").max())]
        plt.plot(lim, lim, "k--", linewidth=1)
        plt.xlabel("target speed at H (m/s)")
        plt.ylabel("predicted speed at H (m/s)")
        plt.title("Open-Loop Speed Prediction")
        savefig("open_loop_speed_pred_vs_target.png")

        plt.figure(figsize=(10, 6))
        plt.scatter(_col(open_rows, "dmin_now"), _col(open_rows, "accel_raw"), s=5, alpha=0.35)
        plt.axhline(0, color="k", linewidth=1)
        plt.axvline(10, color="r", linewidth=1, linestyle="--")
        plt.xlabel("nearest clearance now (m)")
        plt.ylabel("first-step accel_raw (m/s^2)")
        plt.title("Open-Loop First Action vs Clearance")
        savefig("open_loop_accel_vs_dmin.png")

    if model_trace:
        plt.figure(figsize=(10, 6))
        plt.scatter(_col(model_trace, "dmin"), _col(model_trace, "accel_raw"), s=8, alpha=0.45)
        plt.axhline(0, color="k", linewidth=1)
        plt.axvline(10, color="r", linewidth=1, linestyle="--")
        plt.xlabel("closed-loop nearest clearance (m)")
        plt.ylabel("accel_raw (m/s^2)")
        plt.title("Closed-Loop Model Accel vs Clearance")
        savefig("closed_loop_model_accel_vs_dmin.png")

        for key, ylabel, name in [
            ("speed", "speed (m/s)", "closed_loop_speed_by_episode.png"),
            ("lane_error", "lane error (m)", "closed_loop_lane_error_by_episode.png"),
            ("dmin", "nearest clearance (m)", "closed_loop_dmin_by_episode.png"),
        ]:
            plt.figure(figsize=(11, 6))
            for ep in sorted({int(r["episode"]) for r in model_trace})[:10]:
                rows = [r for r in model_trace if int(r["episode"]) == ep]
                plt.plot([r["step"] for r in rows], [r[key] for r in rows], alpha=0.8, label=f"ep{ep}")
            plt.xlabel("step")
            plt.ylabel(ylabel)
            plt.title(f"Model {ylabel} by Episode")
            plt.legend(ncol=2, fontsize=8)
            savefig(name)

    if model_trace and idm_trace:
        plt.figure(figsize=(10, 6))
        plt.hist(_col(idm_trace, "speed"), bins=40, alpha=0.6, label="IDM")
        plt.hist(_col(model_trace, "speed"), bins=40, alpha=0.6, label="model")
        plt.xlabel("closed-loop speed (m/s)")
        plt.ylabel("count")
        plt.title("Closed-Loop Speed: IDM vs Model")
        plt.legend()
        savefig("closed_loop_speed_model_vs_idm.png")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="runs/stage1_data")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs/stage1_analysis")
    ap.add_argument("--dfc-root", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--env", type=str, default="highway-v0")
    ap.add_argument("--vehicles-count", type=int, default=50)
    ap.add_argument("--lanes-count", type=int, default=4)
    ap.add_argument("--n-max-vehicles", type=int, default=15)
    ap.add_argument("--d-hat", type=float, default=0.0)
    ap.add_argument("--alpha-floor", type=float, default=-1.0)
    ap.add_argument("--alpha-floor-ahead-only", dest="alpha_floor_ahead_only",
                    action="store_true", default=None)
    ap.add_argument("--alpha-floor-all-obstacles", dest="alpha_floor_ahead_only",
                    action="store_false")
    ap.add_argument("--skip-idm", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    _ensure_dir(out_dir)
    ck, cfg, model = _load_checkpoint(Path(args.ckpt), args.device, args.dfc_root)
    knobs = _resolve_runtime_knobs(args, cfg)

    config = {
        "cmd_args": vars(args),
        "checkpoint_epoch": ck.get("epoch", None),
        "checkpoint_cfg": cfg,
        "runtime": knobs,
        "constants": {
            "H_TARGET": H_TARGET,
            "DT_TARGET": DT_TARGET,
            "POLICY_FREQ": POLICY_FREQ,
            "SIM_FREQ": SIM_FREQ,
        },
    }
    with open(out_dir / "analysis_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Stage 1 analysis")
    print(f"  checkpoint: {args.ckpt} epoch={ck.get('epoch', '?')}")
    print(f"  data:       {args.data} split={args.split}")
    print(f"  output:     {out_dir}")
    print(f"  runtime:    d_hat={knobs['d_hat']} alpha_floor={knobs['alpha_floor']} "
          f"ahead_only={knobs['alpha_floor_ahead_only']}")

    t0 = time.time()
    dataset_rows, dataset_summary = analyze_dataset(
        Path(args.data), out_dir, split=args.split,
        max_samples=args.max_samples, seed=args.seed,
    )
    print(f"  dataset stats: {len(dataset_rows)} samples")

    open_rows, open_summary = analyze_open_loop(
        Path(args.data), out_dir, model,
        device=args.device, split=args.split,
        max_samples=args.max_samples, batch_size=args.bs,
        seed=args.seed,
        d_hat=knobs["d_hat"],
        alpha_floor=knobs["alpha_floor"],
        alpha_floor_ahead_only=knobs["alpha_floor_ahead_only"],
    )
    print(f"  open-loop: {len(open_rows)} samples")

    model_trace, model_eps, model_summary = closed_loop_model_traces(
        out_dir, model,
        device=args.device, env_id=args.env,
        seed0=args.seed, episodes=args.episodes, max_steps=args.max_steps,
        vehicles_count=args.vehicles_count, lanes_count=args.lanes_count,
        n_max_vehicles=args.n_max_vehicles,
        d_hat=knobs["d_hat"],
        alpha_floor=knobs["alpha_floor"],
        alpha_floor_ahead_only=knobs["alpha_floor_ahead_only"],
    )
    print(f"  model closed-loop: {len(model_eps)} episodes")

    idm_trace: List[Dict[str, Any]] = []
    idm_eps: List[Dict[str, Any]] = []
    idm_summary: Dict[str, Any] = {}
    if not args.skip_idm:
        print("  running IDM baseline...", flush=True)
        idm_trace, idm_eps, idm_summary = closed_loop_idm_baseline(
            out_dir, env_id=args.env,
            seed0=args.seed, episodes=args.episodes, max_steps=args.max_steps,
            vehicles_count=args.vehicles_count, lanes_count=args.lanes_count,
            n_max_vehicles=args.n_max_vehicles,
        )
        print(f"  IDM baseline: {len(idm_eps)} episodes")

    summary = {
        "dataset": dataset_summary,
        "open_loop": open_summary,
        "closed_loop_model": model_summary,
        "closed_loop_idm": idm_summary,
        "wall_clock_s": time.time() - t0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not args.skip_plots:
        print("  writing plots...", flush=True)
        make_plots(out_dir, dataset_rows, open_rows, model_trace, idm_trace)

    print("\nSummary")
    print(json.dumps({
        "open_loop_traj_err": open_summary.get("traj_err", {}),
        "open_loop_speed_err": open_summary.get("speed_err", {}),
        "model_closed_loop": model_summary,
        "idm_closed_loop": idm_summary,
    }, indent=2))
    print(f"\nWrote analysis artifacts to {out_dir}")


if __name__ == "__main__":
    main()
