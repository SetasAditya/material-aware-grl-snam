#!/usr/bin/env python3
"""
eval_force_diagnostic.py — per-step force/μ_lat capture for paper figures.

Existing eval_stage2.py captures per-EPISODE metrics. Figures 2-4 of the
paper need per-STEP data: force decomposition, side-probe risks, μ_lat,
sampled at the moment of leader contact (the "trigger window").

This script:
    1. Loads Stage 1 and Stage 2 checkpoints.
    2. For each scenario, runs N paired episodes on identical seeds.
    3. For each step, captures:
         - F_goal, F_geom, F_soft, F_hard, F_lat (world frame, per stage)
         - μ_lat (raw)
         - side_score (right_probe_risk - left_probe_risk + bias)
         - probe risks (left, right)
         - dmin to nearest neighbor
         - speed, ego_pos, ego_heading
         - lane_index
    4. Identifies trigger window per episode: window around first leader-
       contact moment (default: t_contact ± [5, 10] steps where contact
       is dmin < contact_threshold). For scenarios without contact (default
       highway-v0 with sparse traffic), uses full episode.
    5. Dumps everything to JSON for downstream plotting.

Usage
-----
    python eval_force_diagnostic.py \\
        --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \\
        --stage2-ckpt checkpoints/highway_stage2_mu_lat/best.pt \\
        --scenarios default authored_slow_leader authored_slow_leader_boxed \\
        --episodes 20 --max-steps 120 \\
        --out runs/paper_data/force_diagnostic.json

Smoke (3 episodes, fewer steps):
    python eval_force_diagnostic.py \\
        --stage1-ckpt <s1> --stage2-ckpt <s2> \\
        --scenarios authored_slow_leader \\
        --episodes 3 --max-steps 60 \\
        --out runs/paper_data/force_diagnostic_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LOCAL_HIGHWAY_ENV = HERE / "HighwayEnv"
if LOCAL_HIGHWAY_ENV.exists():
    sys.path.insert(0, str(LOCAL_HIGHWAY_ENV))
sys.path.insert(0, str(HERE))

DEFAULT_DFC_ROOT = HERE.parent
sys.path.insert(0, str(DEFAULT_DFC_ROOT))

from env_wrapper import HighwayMaterialObservation, WrapperConfig  # noqa: E402
from eval_stage1 import _apply_alpha_floor, disable_transformer_nested_tensors  # noqa: E402
from bicycle_surrogate import (  # noqa: E402
    ACCEL_RANGE, STEER_RANGE, force_to_action,
)
from surrogate_integrator import (  # noqa: E402
    compute_surrogate_highway_force,
    _bilinear_sample_ego_patch,
    _lateral_probe_stats,
    _ttc_longitudinal_force,
    ipc_piecewise,
    sdf_barrier_grad,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

H_TARGET    = 20
DT_TARGET   = 0.1
POLICY_FREQ = 10
SIM_FREQ    = 30
LATERAL_LANE_WIDTH = 4.0
LATERAL_LOOKAHEAD  = 10.0
LATERAL_PREFERENCE_BIAS = 0.05

# Trigger window definition
CONTACT_THRESHOLD     = 8.0    # dmin < 8m = "leader contact"
WINDOW_PRE_STEPS      = 5      # ±N steps around contact onset
WINDOW_POST_STEPS     = 10


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios (mirror render_paired_gif's set, but no GIF-specific fields)
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "default": {
        "env_id": "highway-v0",
        "config": {
            "policy_frequency": POLICY_FREQ, "simulation_frequency": SIM_FREQ,
            "vehicles_count": 50, "lanes_count": 4, "duration": 40,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True,
                       "acceleration_range": list(ACCEL_RANGE),
                       "steering_range": list(STEER_RANGE)},
        },
    },
    "authored_slow_leader": {
        "env_id": "highway-slow-leader-v0",
        "config": {
            "policy_frequency": POLICY_FREQ, "simulation_frequency": SIM_FREQ,
            "duration": 40,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True,
                       "acceleration_range": list(ACCEL_RANGE),
                       "steering_range": list(STEER_RANGE)},
        },
    },
    "authored_slow_leader_x2": {
        "env_id": "highway-slow-leader-x2-v0",
        "config": {
            "policy_frequency": POLICY_FREQ, "simulation_frequency": SIM_FREQ,
            "duration": 50,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True,
                       "acceleration_range": list(ACCEL_RANGE),
                       "steering_range": list(STEER_RANGE)},
        },
    },
    "authored_slow_leader_x3": {
        "env_id": "highway-slow-leader-x3-v0",
        "config": {
            "policy_frequency": POLICY_FREQ, "simulation_frequency": SIM_FREQ,
            "duration": 50,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True,
                       "acceleration_range": list(ACCEL_RANGE),
                       "steering_range": list(STEER_RANGE)},
        },
    },
    "authored_slow_leader_x4": {
        "env_id": "highway-slow-leader-x4-v0",
        "config": {
            "policy_frequency": POLICY_FREQ, "simulation_frequency": SIM_FREQ,
            "duration": 50,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True,
                       "acceleration_range": list(ACCEL_RANGE),
                       "steering_range": list(STEER_RANGE)},
        },
    },
    "authored_slow_leader_boxed": {
        "env_id": "highway-slow-leader-boxed-v0",
        "config": {
            "policy_frequency": POLICY_FREQ, "simulation_frequency": SIM_FREQ,
            "duration": 40,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True,
                       "acceleration_range": list(ACCEL_RANGE),
                       "steering_range": list(STEER_RANGE)},
        },
    },
    "authored_slow_leader_boxed_x2": {
        "env_id": "highway-slow-leader-boxed-x2-v0",
        "config": {
            "policy_frequency": POLICY_FREQ, "simulation_frequency": SIM_FREQ,
            "duration": 50,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True,
                       "acceleration_range": list(ACCEL_RANGE),
                       "steering_range": list(STEER_RANGE)},
        },
    },
    "authored_slow_leader_boxed_x3": {
        "env_id": "highway-slow-leader-boxed-x3-v0",
        "config": {
            "policy_frequency": POLICY_FREQ, "simulation_frequency": SIM_FREQ,
            "duration": 50,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True,
                       "acceleration_range": list(ACCEL_RANGE),
                       "steering_range": list(STEER_RANGE)},
        },
    },
    "authored_slow_leader_boxed_x4": {
        "env_id": "highway-slow-leader-boxed-x4-v0",
        "config": {
            "policy_frequency": POLICY_FREQ, "simulation_frequency": SIM_FREQ,
            "duration": 50,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True,
                       "acceleration_range": list(ACCEL_RANGE),
                       "steering_range": list(STEER_RANGE)},
        },
    },
    "authored_slow_convoy": {
        "env_id": "highway-slow-convoy-v0",
        "config": {
            "policy_frequency": POLICY_FREQ, "simulation_frequency": SIM_FREQ,
            "duration": 50,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True,
                       "acceleration_range": list(ACCEL_RANGE),
                       "steering_range": list(STEER_RANGE)},
        },
    },
    "authored_slow_convoy_boxed": {
        "env_id": "highway-slow-convoy-boxed-v0",
        "config": {
            "policy_frequency": POLICY_FREQ, "simulation_frequency": SIM_FREQ,
            "duration": 50,
            "action": {"type": "ContinuousAction", "longitudinal": True,
                       "lateral": True,
                       "acceleration_range": list(ACCEL_RANGE),
                       "steering_range": list(STEER_RANGE)},
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Env / model utilities
# ─────────────────────────────────────────────────────────────────────────────

def _import_gym():
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401
        return gym
    except ImportError as exc:
        raise SystemExit(f"highway_env not importable: {exc}") from exc


def make_env(gym, scenario_name: str):
    spec = SCENARIOS[scenario_name]
    return gym.make(spec["env_id"], config=spec["config"])


def _reset(env, seed):
    out = env.reset(seed=seed)
    return out if (isinstance(out, tuple) and len(out) == 2) else (out, {})


def _step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, _, term, trunc, info = out
        return obs, bool(term), bool(trunc), info
    obs, _, done, info = out
    return obs, bool(done), False, info


def _normalize_action(accel_phys: float, steer_phys: float) -> np.ndarray:
    a_half = (ACCEL_RANGE[1] - ACCEL_RANGE[0]) / 2.0
    s_half = (STEER_RANGE[1] - STEER_RANGE[0]) / 2.0
    return np.array([
        float(np.clip(accel_phys / a_half, -1.0, 1.0)),
        float(np.clip(steer_phys / s_half, -1.0, 1.0)),
    ], dtype=np.float32)


def _to_batch(obs_np, device):
    out = {}
    for k, v in obs_np.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                t = torch.from_numpy(v.copy()).bool()
            elif v.dtype in (np.int32, np.int64):
                t = torch.from_numpy(v.copy()).long()
            else:
                t = torch.from_numpy(v.astype(np.float32, copy=False))
        elif isinstance(v, (np.floating, float)):
            t = torch.tensor(float(v), dtype=torch.float32)
        elif isinstance(v, (np.integer, int)):
            t = torch.tensor(int(v), dtype=torch.long)
        else:
            t = torch.as_tensor(v)
        out[k] = t.unsqueeze(0).to(device)
    return out


def _model_coeffs(model, batch):
    B = batch["o0"].shape[0]
    N = batch["C"].shape[1]
    if N > 0:
        obs_feats = torch.cat([
            batch["C"], batch["R"].unsqueeze(-1), batch["W"].unsqueeze(-1),
            batch["goal"].unsqueeze(1) - batch["C"],
        ], dim=-1)
    else:
        obs_feats = batch["o0"].new_zeros(B, 0, 6)
    goal_delta = batch["goal"] - batch["o0"]
    goal_feats = torch.cat([
        goal_delta,
        torch.linalg.norm(goal_delta, dim=-1, keepdim=True),
        batch["o0"].new_ones(B, 1),
    ], dim=-1)
    out = model(obs_feats=obs_feats, obs_mask=batch["mask"],
                goal_feats=goal_feats, risk_patch=batch["risk_patch"])
    if len(out) == 5:
        alphas, beta, gamma, lam_soft, lam_hard = out
        mu_lat = torch.zeros_like(beta)
    else:
        alphas, beta, gamma, lam_soft, lam_hard, mu_lat = out
    return alphas, beta, gamma, lam_soft, lam_hard, mu_lat


# ─────────────────────────────────────────────────────────────────────────────
# Force decomposition (matches surrogate_integrator's decomposition exactly)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decompose_forces(
    *, o, heading, speed, o0, heading_0, goal, C, R_eff, mask,
    alphas, beta, gamma, lam_soft, lam_hard, mu_lat,
    V_neighbors, rollout_patch, d_hat,
    ttc_gain: float = 0.0,
    ttc_threshold_s: float = 3.0,
    ttc_softness_s: float = 0.5,
    ttc_min_closing_speed: float = 0.5,
    ttc_lane_halfwidth: float = 2.0,
    ttc_boxed_risk_thresh: float = 0.25,
    ttc_boxed_gate_sharpness: float = 20.0,
) -> Dict[str, torch.Tensor]:
    """Return each force component separately, plus probe info.

    Mirrors the math inside compute_surrogate_highway_force; doing it
    explicitly here so we can pull each piece out for the figure.
    """
    # F_goal
    F_goal = -beta.unsqueeze(-1) * (o - goal)

    # F_damp
    vel_world = speed.unsqueeze(-1) * torch.stack(
        [torch.cos(heading), torch.sin(heading)], dim=-1)
    F_damp = -gamma.unsqueeze(-1) * vel_world

    # F_geom (per-vehicle barrier — sum of weighted gradients)
    B, N = C.shape[:2]
    if N > 0:
        diff = o.unsqueeze(1) - C
        r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)
        n_hat = diff / r.unsqueeze(-1)
        d = r - R_eff
        d = torch.where(mask, d, torch.full_like(d, 1e6))
        _, dbdd = ipc_piecewise(d, d_hat.view(-1, 1))
        dbdd = dbdd.clamp(-100.0, 100.0)
        F_geom = (alphas * dbdd).unsqueeze(-1) * n_hat
        F_geom = F_geom.sum(dim=1)
    else:
        F_geom = torch.zeros_like(o)

    # F_soft, F_hard from semantic patch sample
    sem = _bilinear_sample_ego_patch(
        rollout_patch, o, o0, heading_0,
        cell_size_lon=1.0, cell_size_lat=1.0, patch_lon_offset_frac=0.05,
    )
    risk_grad = torch.stack([sem[:, 2], sem[:, 3]], dim=-1)
    sdf_val = sem[:, 1].clamp(0.0, 50.0)
    sdf_grad = torch.stack([sem[:, 4], sem[:, 5]], dim=-1)
    F_soft = -lam_soft.unsqueeze(-1) * risk_grad
    _, db_dphi = sdf_barrier_grad(sdf_val, d_hat_sdf=3.0)
    F_hard = -lam_hard.unsqueeze(-1) * db_dphi.unsqueeze(-1) * sdf_grad

    # F_lat + F_ttc
    probe_stats = _lateral_probe_stats(
        o=o,
        heading=heading,
        o0=o0,
        heading_0=heading_0,
        rollout_patch=rollout_patch,
        cell_size_lon=1.0,
        cell_size_lat=1.0,
        patch_lon_offset_frac=0.05,
        lateral_lookahead=LATERAL_LOOKAHEAD,
        lateral_lane_width=LATERAL_LANE_WIDTH,
        lateral_preference_bias=LATERAL_PREFERENCE_BIAS,
    )
    risk_l = probe_stats["risk_left"]
    risk_r = probe_stats["risk_right"]
    side_score = probe_stats["side_score"]
    F_lat = -(mu_lat * side_score).unsqueeze(-1) * probe_stats["n_lat_world"]
    F_ttc, ttc_dbg = _ttc_longitudinal_force(
        o=o,
        heading=heading,
        speed=speed,
        C=C,
        V_neighbors=V_neighbors,
        R_eff=R_eff,
        mask=mask,
        forward_world=probe_stats["forward_world"],
        risk_left=risk_l,
        risk_right=risk_r,
        ttc_gain=ttc_gain,
        ttc_threshold_s=ttc_threshold_s,
        ttc_softness_s=ttc_softness_s,
        ttc_min_closing_speed=ttc_min_closing_speed,
        ttc_lane_halfwidth=ttc_lane_halfwidth,
        ttc_boxed_risk_thresh=ttc_boxed_risk_thresh,
        ttc_boxed_gate_sharpness=ttc_boxed_gate_sharpness,
    )

    return {
        "F_goal": F_goal, "F_damp": F_damp, "F_geom": F_geom,
        "F_soft": F_soft, "F_hard": F_hard, "F_lat": F_lat, "F_ttc": F_ttc,
        "risk_l": risk_l, "risk_r": risk_r, "side_score": side_score,
        "ttc": ttc_dbg["ttc"],
        "closing_speed": ttc_dbg["closing_speed"],
        "leader_gap_lon": ttc_dbg["leader_gap_lon"],
        "boxed_gate": ttc_dbg["boxed_gate"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-step capture
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuntimeKnobs:
    """Deployment knobs loaded from each checkpoint's training config."""
    d_hat: float = 0.0
    alpha_floor: float = 0.0
    alpha_floor_ahead_only: bool = False
    ttc_gain: float = 0.0
    ttc_threshold_s: float = 3.0
    ttc_softness_s: float = 0.5
    ttc_min_closing_speed: float = 0.5
    ttc_lane_halfwidth: float = 2.0
    ttc_boxed_risk_thresh: float = 0.25
    ttc_boxed_gate_sharpness: float = 20.0


@dataclass
class StepRecord:
    """Per-step diagnostic data for one stage."""
    step:     int
    ego_x:    float
    ego_y:    float
    ego_heading: float
    speed:    float
    # Forces — store as 2-tuples for JSON friendliness
    F_goal:   Tuple[float, float]
    F_damp:   Tuple[float, float]
    F_geom:   Tuple[float, float]
    F_soft:   Tuple[float, float]
    F_hard:   Tuple[float, float]
    F_lat:    Tuple[float, float]
    F_ttc:    Tuple[float, float]
    F_tot:    Tuple[float, float]
    # Lateral channel diagnostics
    mu_lat_raw:   float
    risk_left:    float
    risk_right:   float
    side_score:   float
    ttc:          float
    closing_speed: float
    leader_gap_lon: float
    boxed_gate:   float
    # Lambda
    lam_soft: float
    lam_hard: float
    # Geometry
    dmin:     float
    risk_at_ego: float
    # Lane
    lane_idx: Optional[List[int]]


def _resolve_runtime_knobs(
    cfg_train: Dict[str, Any],
    *,
    d_hat_override: Optional[float],
    alpha_floor_override: Optional[float],
    alpha_floor_ahead_only_override: Optional[bool],
    ttc_gain_override: Optional[float],
    ttc_threshold_s_override: Optional[float],
    ttc_softness_s_override: Optional[float],
    ttc_min_closing_speed_override: Optional[float],
    ttc_lane_halfwidth_override: Optional[float],
    ttc_boxed_risk_thresh_override: Optional[float],
    ttc_boxed_gate_sharpness_override: Optional[float],
) -> RuntimeKnobs:
    return RuntimeKnobs(
        d_hat=(
            float(d_hat_override)
            if d_hat_override is not None and d_hat_override > 0
            else float(cfg_train.get("d_hat", 0.0))
        ),
        alpha_floor=(
            float(alpha_floor_override)
            if alpha_floor_override is not None and alpha_floor_override >= 0
            else float(cfg_train.get("alpha_floor", 0.0))
        ),
        alpha_floor_ahead_only=(
            bool(alpha_floor_ahead_only_override)
            if alpha_floor_ahead_only_override is not None
            else bool(cfg_train.get("alpha_floor_ahead_only", False))
        ),
        ttc_gain=(
            float(ttc_gain_override)
            if ttc_gain_override is not None
            else float(cfg_train.get("ttc_gain", 0.0))
        ),
        ttc_threshold_s=(
            float(ttc_threshold_s_override)
            if ttc_threshold_s_override is not None
            else float(cfg_train.get("ttc_threshold_s", 3.0))
        ),
        ttc_softness_s=(
            float(ttc_softness_s_override)
            if ttc_softness_s_override is not None
            else float(cfg_train.get("ttc_softness_s", 0.5))
        ),
        ttc_min_closing_speed=(
            float(ttc_min_closing_speed_override)
            if ttc_min_closing_speed_override is not None
            else float(cfg_train.get("ttc_min_closing_speed", 0.5))
        ),
        ttc_lane_halfwidth=(
            float(ttc_lane_halfwidth_override)
            if ttc_lane_halfwidth_override is not None
            else float(cfg_train.get("ttc_lane_halfwidth", 2.0))
        ),
        ttc_boxed_risk_thresh=(
            float(ttc_boxed_risk_thresh_override)
            if ttc_boxed_risk_thresh_override is not None
            else float(cfg_train.get("ttc_boxed_risk_thresh", 0.25))
        ),
        ttc_boxed_gate_sharpness=(
            float(ttc_boxed_gate_sharpness_override)
            if ttc_boxed_gate_sharpness_override is not None
            else float(cfg_train.get("ttc_boxed_gate_sharpness", 20.0))
        ),
    )


@torch.no_grad()
def capture_step(model, observer, env, *, stage: int, device: str,
                 runtime: RuntimeKnobs) -> Tuple[np.ndarray, StepRecord]:
    """Build obs, compute forces, return (action_norm, step record)."""
    obs_np = observer.build(env)
    batch = _to_batch(obs_np, device)
    if runtime.d_hat > 0:
        batch["d_hat"] = torch.full_like(batch["d_hat"], float(runtime.d_hat))

    alphas, beta, gamma, lam_soft, lam_hard, mu_lat = _model_coeffs(model, batch)
    alphas = _apply_alpha_floor(
        batch, alphas, runtime.alpha_floor,
        ahead_only=runtime.alpha_floor_ahead_only,
    )
    if stage == 1:
        lam_soft = torch.zeros_like(lam_soft)
        lam_hard = torch.zeros_like(lam_hard)
        mu_lat = torch.zeros_like(mu_lat)

    v0 = batch["v0"]
    speed_0   = torch.linalg.norm(v0, dim=-1).clamp_min(1e-3)
    heading_0 = torch.atan2(v0[:, 1], v0[:, 0])

    decomp = decompose_forces(
        o=batch["o0"], heading=heading_0, speed=speed_0,
        o0=batch["o0"], heading_0=heading_0, goal=batch["goal"],
        C=batch["C"], V_neighbors=batch.get("V_neighbors"),
        R_eff=batch["R"], mask=batch["mask"],
        alphas=alphas, beta=beta, gamma=gamma,
        lam_soft=lam_soft, lam_hard=lam_hard, mu_lat=mu_lat,
        rollout_patch=batch["rollout_patch"], d_hat=batch["d_hat"],
        ttc_gain=runtime.ttc_gain,
        ttc_threshold_s=runtime.ttc_threshold_s,
        ttc_softness_s=runtime.ttc_softness_s,
        ttc_min_closing_speed=runtime.ttc_min_closing_speed,
        ttc_lane_halfwidth=runtime.ttc_lane_halfwidth,
        ttc_boxed_risk_thresh=runtime.ttc_boxed_risk_thresh,
        ttc_boxed_gate_sharpness=runtime.ttc_boxed_gate_sharpness,
    )

    # Deployed F_tot via the single-source-of-truth path
    F_tot, dmin, risk_val, _ = compute_surrogate_highway_force(
        o=batch["o0"], heading=heading_0, speed=speed_0,
        o0=batch["o0"], heading_0=heading_0, goal=batch["goal"],
        C=batch["C"], V_neighbors=batch.get("V_neighbors"),
        R_eff=batch["R"], mask=batch["mask"],
        alphas=alphas, beta=beta, gamma=gamma,
        lam_soft=lam_soft, lam_hard=lam_hard,
        mu_lat=(None if stage == 1 else mu_lat),
        rollout_patch=batch["rollout_patch"], d_hat=batch["d_hat"],
        ttc_gain=runtime.ttc_gain,
        ttc_threshold_s=runtime.ttc_threshold_s,
        ttc_softness_s=runtime.ttc_softness_s,
        ttc_min_closing_speed=runtime.ttc_min_closing_speed,
        ttc_lane_halfwidth=runtime.ttc_lane_halfwidth,
        ttc_boxed_risk_thresh=runtime.ttc_boxed_risk_thresh,
        ttc_boxed_gate_sharpness=runtime.ttc_boxed_gate_sharpness,
    )
    accel, steer = force_to_action(F_tot, heading_0, speed_0)
    accel_phys = float(accel.clamp(*ACCEL_RANGE).item())
    steer_phys = float(steer.clamp(*STEER_RANGE).item())
    action = _normalize_action(accel_phys, steer_phys)

    def _vec(t: torch.Tensor) -> Tuple[float, float]:
        v = t.squeeze(0).cpu().numpy()
        return (float(v[0]), float(v[1]))

    uenv = env.unwrapped
    rec = StepRecord(
        step=0,
        ego_x=float(uenv.vehicle.position[0]),
        ego_y=float(uenv.vehicle.position[1]),
        ego_heading=float(uenv.vehicle.heading),
        speed=float(uenv.vehicle.speed),
        F_goal=_vec(decomp["F_goal"]),
        F_damp=_vec(decomp["F_damp"]),
        F_geom=_vec(decomp["F_geom"]),
        F_soft=_vec(decomp["F_soft"]),
        F_hard=_vec(decomp["F_hard"]),
        F_lat =_vec(decomp["F_lat"]),
        F_ttc =_vec(decomp["F_ttc"]),
        F_tot =_vec(F_tot),
        mu_lat_raw=float(mu_lat.item()),
        risk_left=float(decomp["risk_l"].item()),
        risk_right=float(decomp["risk_r"].item()),
        side_score=float(decomp["side_score"].item()),
        ttc=float(decomp["ttc"].item()),
        closing_speed=float(decomp["closing_speed"].item()),
        leader_gap_lon=float(decomp["leader_gap_lon"].item()),
        boxed_gate=float(decomp["boxed_gate"].item()),
        lam_soft=float(lam_soft.item()),
        lam_hard=float(lam_hard.item()),
        dmin=float(dmin.item()),
        risk_at_ego=float(risk_val.item()),
        lane_idx=(list(uenv.vehicle.lane_index)
                   if uenv.vehicle.lane_index else None),
    )
    return action, rec


def capture_episode(model, observer, env, *, stage: int, max_steps: int,
                     seed: int, device: str, runtime: RuntimeKnobs
                     ) -> Tuple[List[StepRecord], bool, bool]:
    """Run one episode, return (records, crashed, truncated)."""
    _reset(env, seed)
    records: List[StepRecord] = []
    crashed = False
    truncated = False
    for t in range(max_steps):
        action, rec = capture_step(model, observer, env,
                                     stage=stage, device=device,
                                     runtime=runtime)
        rec.step = t
        records.append(rec)
        _, term, trunc, info = _step(env, action)
        if term:
            crashed = bool(info.get("crashed", True))
            break
        if trunc:
            truncated = True
            break
    return records, crashed, truncated


# ─────────────────────────────────────────────────────────────────────────────
# Trigger window
# ─────────────────────────────────────────────────────────────────────────────

def find_trigger_window(records: List[StepRecord],
                          *, contact_threshold: float = CONTACT_THRESHOLD,
                          pre: int = WINDOW_PRE_STEPS,
                          post: int = WINDOW_POST_STEPS,
                          ) -> Optional[Tuple[int, int]]:
    """Find first step where dmin < contact_threshold; return [t-pre, t+post]
    clipped to [0, len(records)). Returns None if no contact ever (use full
    episode in that case)."""
    for i, r in enumerate(records):
        if r.dmin < contact_threshold:
            return (max(0, i - pre), min(len(records), i + post + 1))
    return None


def filter_to_window(records: List[StepRecord],
                       window: Optional[Tuple[int, int]]) -> List[StepRecord]:
    if window is None:
        return list(records)
    lo, hi = window
    return list(records[lo:hi])


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(
    ckpt_path: Path,
    device: str,
    dfc_root: str = "",
    *,
    d_hat_override: Optional[float] = None,
    alpha_floor_override: Optional[float] = None,
    alpha_floor_ahead_only_override: Optional[bool] = None,
    ttc_gain_override: Optional[float] = None,
    ttc_threshold_s_override: Optional[float] = None,
    ttc_softness_s_override: Optional[float] = None,
    ttc_min_closing_speed_override: Optional[float] = None,
    ttc_lane_halfwidth_override: Optional[float] = None,
    ttc_boxed_risk_thresh_override: Optional[float] = None,
    ttc_boxed_gate_sharpness_override: Optional[float] = None,
):
    if dfc_root:
        sys.path.insert(0, dfc_root)
    from train_material import CoefEnergyNetMaterial  # noqa: E402

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_train = ck.get("cfg", {})
    lam_soft_max = cfg_train.get("lam_soft_max", 50.0)
    lam_hard_max = cfg_train.get("lam_hard_max", 10.0)
    mu_lat_max   = cfg_train.get("mu_lat_max", 5.0)
    try:
        model = CoefEnergyNetMaterial(
            lam_soft_max=lam_soft_max, lam_hard_max=lam_hard_max,
            mu_lat_max=mu_lat_max,
        ).to(device)
    except TypeError:
        model = CoefEnergyNetMaterial(
            lam_soft_max=lam_soft_max, lam_hard_max=lam_hard_max,
        ).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    disable_transformer_nested_tensors(model)
    runtime = _resolve_runtime_knobs(
        cfg_train,
        d_hat_override=d_hat_override,
        alpha_floor_override=alpha_floor_override,
        alpha_floor_ahead_only_override=alpha_floor_ahead_only_override,
        ttc_gain_override=ttc_gain_override,
        ttc_threshold_s_override=ttc_threshold_s_override,
        ttc_softness_s_override=ttc_softness_s_override,
        ttc_min_closing_speed_override=ttc_min_closing_speed_override,
        ttc_lane_halfwidth_override=ttc_lane_halfwidth_override,
        ttc_boxed_risk_thresh_override=ttc_boxed_risk_thresh_override,
        ttc_boxed_gate_sharpness_override=ttc_boxed_gate_sharpness_override,
    )
    return model, runtime


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--stage1-ckpt", type=str, required=True)
    ap.add_argument("--stage2-ckpt", type=str, required=True)
    ap.add_argument("--scenarios",   type=str, nargs="+",
                    default=["default", "authored_slow_leader",
                             "authored_slow_leader_boxed"])
    ap.add_argument("--episodes",    type=int, default=20)
    ap.add_argument("--max-steps",   type=int, default=120)
    ap.add_argument("--seed",        type=int, default=1000)
    ap.add_argument("--n-max-vehicles", type=int, default=15)
    ap.add_argument("--contact-threshold", type=float, default=CONTACT_THRESHOLD)
    ap.add_argument("--pre-steps",   type=int, default=WINDOW_PRE_STEPS)
    ap.add_argument("--post-steps",  type=int, default=WINDOW_POST_STEPS)
    ap.add_argument("--device",      type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dfc-root",    type=str, default="")
    ap.add_argument("--d-hat",       type=float, default=None)
    ap.add_argument("--alpha-floor", type=float, default=None)
    ap.add_argument("--alpha-floor-all-obstacles",
                    dest="alpha_floor_ahead_only", action="store_false")
    ap.set_defaults(alpha_floor_ahead_only=None)
    ap.add_argument("--ttc-gain", type=float, default=None)
    ap.add_argument("--ttc-threshold-s", type=float, default=None)
    ap.add_argument("--ttc-softness-s", type=float, default=None)
    ap.add_argument("--ttc-min-closing-speed", type=float, default=None)
    ap.add_argument("--ttc-lane-halfwidth", type=float, default=None)
    ap.add_argument("--ttc-boxed-risk-thresh", type=float, default=None)
    ap.add_argument("--ttc-boxed-gate-sharpness", type=float, default=None)
    ap.add_argument("--out",         type=str,
                    default="runs/paper_data/force_diagnostic.json")
    args = ap.parse_args()

    print("Loading models...")
    s1_model, s1_runtime = _load_model(
        Path(args.stage1_ckpt),
        args.device,
        args.dfc_root,
        d_hat_override=args.d_hat,
        alpha_floor_override=args.alpha_floor,
        alpha_floor_ahead_only_override=args.alpha_floor_ahead_only,
        ttc_gain_override=args.ttc_gain,
        ttc_threshold_s_override=args.ttc_threshold_s,
        ttc_softness_s_override=args.ttc_softness_s,
        ttc_min_closing_speed_override=args.ttc_min_closing_speed,
        ttc_lane_halfwidth_override=args.ttc_lane_halfwidth,
        ttc_boxed_risk_thresh_override=args.ttc_boxed_risk_thresh,
        ttc_boxed_gate_sharpness_override=args.ttc_boxed_gate_sharpness,
    )
    s2_model, s2_runtime = _load_model(
        Path(args.stage2_ckpt),
        args.device,
        args.dfc_root,
        d_hat_override=args.d_hat,
        alpha_floor_override=args.alpha_floor,
        alpha_floor_ahead_only_override=args.alpha_floor_ahead_only,
        ttc_gain_override=args.ttc_gain,
        ttc_threshold_s_override=args.ttc_threshold_s,
        ttc_softness_s_override=args.ttc_softness_s,
        ttc_min_closing_speed_override=args.ttc_min_closing_speed,
        ttc_lane_halfwidth_override=args.ttc_lane_halfwidth,
        ttc_boxed_risk_thresh_override=args.ttc_boxed_risk_thresh,
        ttc_boxed_gate_sharpness_override=args.ttc_boxed_gate_sharpness,
    )
    print(f"  Stage 1: {args.stage1_ckpt}")
    print(f"    d_hat={s1_runtime.d_hat:.1f} "
          f"alpha_floor={s1_runtime.alpha_floor:.4f} "
          f"ahead_only={s1_runtime.alpha_floor_ahead_only}")
    print(f"  Stage 2: {args.stage2_ckpt}")
    print(f"    d_hat={s2_runtime.d_hat:.1f} "
          f"alpha_floor={s2_runtime.alpha_floor:.4f} "
          f"ahead_only={s2_runtime.alpha_floor_ahead_only}")
    if s2_runtime.ttc_gain > 0:
        print(f"    ttc_gain={s2_runtime.ttc_gain:.2f} "
              f"ttc_threshold={s2_runtime.ttc_threshold_s:.2f}s "
              f"boxed_thresh={s2_runtime.ttc_boxed_risk_thresh:.2f}")
    print(f"  Trigger window: dmin < {args.contact_threshold}m, "
           f"±[{args.pre_steps}, {args.post_steps}] steps")

    gym = _import_gym()
    observer = HighwayMaterialObservation(WrapperConfig(
        n_max_vehicles=args.n_max_vehicles,
        dt_surrogate=DT_TARGET,
        horizon_surrogate=H_TARGET,
    ))

    out_data: Dict[str, Any] = {
        "config": vars(args),
        "trigger_params": {
            "contact_threshold": args.contact_threshold,
            "pre_steps": args.pre_steps,
            "post_steps": args.post_steps,
        },
        "scenarios": {},
    }

    t0 = time.time()
    for scn_name in args.scenarios:
        if scn_name not in SCENARIOS:
            print(f"Unknown scenario {scn_name!r}, skipping.")
            continue
        print(f"\n━━━ {scn_name} ━━━")

        scn_data: Dict[str, List[Dict[str, Any]]] = {"stage1": [], "stage2": []}

        for stage_label, model, runtime in [
            ("stage1", s1_model, s1_runtime),
            ("stage2", s2_model, s2_runtime),
        ]:
            stage_int = 1 if stage_label == "stage1" else 2
            print(f"  Capturing {stage_label}...")
            for i in range(args.episodes):
                seed = args.seed + i
                env = make_env(gym, scn_name)
                try:
                    records, crashed, truncated = capture_episode(
                        model, observer, env,
                        stage=stage_int, max_steps=args.max_steps,
                        seed=seed, device=args.device, runtime=runtime,
                    )
                except Exception as exc:
                    print(f"    ep {i:3d} seed={seed} FAILED: {exc}")
                    env.close()
                    continue
                env.close()

                window = find_trigger_window(
                    records,
                    contact_threshold=args.contact_threshold,
                    pre=args.pre_steps, post=args.post_steps,
                )
                window_records = filter_to_window(records, window)

                ep_data = {
                    "seed": seed,
                    "n_steps": len(records),
                    "crashed": crashed,
                    "truncated": truncated,
                    "trigger_window": list(window) if window else None,
                    "all_steps":     [asdict(r) for r in records],
                    "window_steps":  [asdict(r) for r in window_records],
                }
                scn_data[stage_label].append(ep_data)

                tag = "CRASH" if crashed else "OK"
                wstr = (f"window={window[0]}-{window[1]}"
                         if window else "no-contact")
                print(f"    ep {i:3d} seed={seed} {tag:5s} "
                      f"n={len(records):3d} {wstr}")

        out_data["scenarios"][scn_name] = scn_data

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=1)
    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Wall clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
