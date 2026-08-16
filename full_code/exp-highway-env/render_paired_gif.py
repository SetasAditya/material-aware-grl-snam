#!/usr/bin/env python3
"""
render_paired_gif.py — paired Stage 1 vs Stage 2 closed-loop visualization.

Produces side-by-side GIFs showing both stages running on the same seed of
the same scenario. The visual story:

  Stage 1 (left)                           Stage 2 (right)
  ┌─────────────────────────┐              ┌─────────────────────────┐
  │ risk-field heatmap       │              │ risk-field heatmap       │
  │   + lane lines           │              │   + lane lines           │
  │   + ego (arrow)          │              │   + ego (arrow)          │
  │   + neighbors (rects)    │              │   + neighbors (rects)    │
  │   + force vectors        │              │   + force vectors        │
  │   + side probes (S2)     │              │   + side probes          │
  └─────────────────────────┘              └─────────────────────────┘
  ─────────────── timeseries strip (last 4s rolling) ───────────────
  speed | |F_y| | mu_lat (S2 only) | crash flash if applicable

Two output configs:
  --config paper  → 150 DPI, clean look, axes minimal, palette muted
  --config debug  → 100 DPI, all info visible, palette saturated, FPS=15

Three scenarios → three GIFs, each ~12s at 10 FPS = 120 frames.

Usage
-----
    python render_paired_gif.py \\
        --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \\
        --stage2-ckpt checkpoints/highway_stage2_mu_lat/best.pt \\
        --scenarios default authored_slow_leader authored_slow_leader_boxed \\
        --seed 1000 --max-steps 120 \\
        --config paper \\
        --out runs/figures/

Smoke (one scenario, fewer frames):
    python render_paired_gif.py \\
        --stage1-ckpt <s1> --stage2-ckpt <s2> \\
        --scenarios authored_slow_leader \\
        --max-steps 40 --config debug
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.collections import PatchCollection

HERE = Path(__file__).resolve().parent
LOCAL_HIGHWAY_ENV = HERE / "HighwayEnv"
if LOCAL_HIGHWAY_ENV.exists():
    sys.path.insert(0, str(LOCAL_HIGHWAY_ENV))
sys.path.insert(0, str(HERE))

DEFAULT_DFC_ROOT = HERE.parent
sys.path.insert(0, str(DEFAULT_DFC_ROOT))

from env_wrapper import HighwayMaterialObservation, WrapperConfig  # noqa: E402
from bicycle_surrogate import (  # noqa: E402
    ACCEL_RANGE, STEER_RANGE, force_to_action,
)
from surrogate_integrator import (  # noqa: E402
    _lateral_probe_stats,
    _ttc_longitudinal_force,
    compute_surrogate_highway_force,
)
from eval_stage1 import _apply_alpha_floor, disable_transformer_nested_tensors  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

H_TARGET    = 20
DT_TARGET   = 0.1
POLICY_FREQ = 10
SIM_FREQ    = 30

# Lateral channel constants — must match surrogate_integrator's
LATERAL_LANE_WIDTH = 4.0
LATERAL_LOOKAHEAD  = 10.0

# Bicycle vehicle dimensions for visualization
VEHICLE_LENGTH = 5.0
VEHICLE_WIDTH  = 2.0

STAGE1_PANEL_TITLE = "Geometry Baseline\nleader-following scaffold"
STAGE2_PANEL_TITLE = "Risk-Shaped Policy\nlateral risk channel active"


# ─────────────────────────────────────────────────────────────────────────────
# Configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RenderCfg:
    """Visual styling. Two presets: 'paper' and 'debug'."""
    name: str
    dpi: int
    fps: int
    palette: Dict[str, str]
    show_force_legend: bool
    show_probe_dots: bool
    show_force_text: bool         # numeric force values in corner
    risk_alpha: float             # heatmap transparency
    figsize: Tuple[float, float]


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


PAPER_CFG = RenderCfg(
    name="paper",
    dpi=150,
    fps=10,
    palette={
        "ego_s1":    "#1f77b4",   # blue
        "ego_s2":    "#d62728",   # red
        "neighbor":  "#888888",
        "lane_line": "#444444",
        "F_goal":    "#2ca02c",   # green
        "F_geom":    "#ff7f0e",   # orange
        "F_lat":     "#9467bd",   # purple — the spotlight star
        "F_ttc":     "#8c564b",   # brown — TTC braking
        "F_soft":    "#17becf",   # teal
        "F_hard":    "#bcbd22",   # olive
        "probe_lo":  "#2ca02c",   # green = low risk
        "probe_hi":  "#d62728",   # red = high risk
        "crash":     "#e41a1c",
        "speed":     "#1f77b4",
        "mu_lat":    "#9467bd",
    },
    show_force_legend=True,
    show_probe_dots=True,
    show_force_text=False,
    risk_alpha=0.4,
    figsize=(14, 7),
)

DEBUG_CFG = RenderCfg(
    name="debug",
    dpi=100,
    fps=15,
    palette={
        "ego_s1":    "#1f77b4",
        "ego_s2":    "#d62728",
        "neighbor":  "#666666",
        "lane_line": "#222222",
        "F_goal":    "#00ff00",
        "F_geom":    "#ff8800",
        "F_lat":     "#ff00ff",
        "F_ttc":     "#a65628",
        "F_soft":    "#00ffff",
        "F_hard":    "#ffff00",
        "probe_lo":  "#00ff00",
        "probe_hi":  "#ff0000",
        "crash":     "#ff0000",
        "speed":     "#1f77b4",
        "mu_lat":    "#ff00ff",
    },
    show_force_legend=True,
    show_probe_dots=True,
    show_force_text=True,
    risk_alpha=0.6,
    figsize=(16, 9),
)


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
        "title": "default highway-v0",
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
        "title": "authored slow leader (passing lane open)",
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
        "title": "authored slow leader x2 (passing lane open)",
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
        "title": "authored slow leader x3 (passing lane open)",
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
        "title": "authored slow leader x4 (passing lane open)",
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
        "title": "authored slow leader BOXED (no escape)",
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
        "title": "authored slow leader BOXED x2 (no escape)",
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
        "title": "authored slow leader BOXED x3 (no escape)",
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
        "title": "authored slow leader BOXED x4 (no escape)",
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
        "title": "authored slow convoy (4 leaders, passing lane open)",
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
        "title": "authored slow convoy BOXED (4 leaders, no escape)",
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
    # Backward-compat: handle both 5-tuple and 6-tuple model returns
    if len(out) == 5:
        alphas, beta, gamma, lam_soft, lam_hard = out
        mu_lat = torch.zeros_like(beta)
    else:
        alphas, beta, gamma, lam_soft, lam_hard, mu_lat = out
    return alphas, beta, gamma, lam_soft, lam_hard, mu_lat


# ─────────────────────────────────────────────────────────────────────────────
# Per-step capture: state needed for one frame of GIF
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameState:
    """Everything we need to render one frame for one stage."""
    step:      int
    ego_pos:   np.ndarray         # (2,)
    ego_heading: float
    ego_speed: float
    neighbor_pos: np.ndarray      # (N, 2) — only valid neighbors
    neighbor_heading: np.ndarray  # (N,)
    # Force decomposition (world frame)
    F_goal: np.ndarray            # (2,)
    F_geom: np.ndarray
    F_soft: np.ndarray
    F_hard: np.ndarray
    F_lat:  np.ndarray
    F_ttc:  np.ndarray
    F_tot:  np.ndarray
    # Risk-field patch in ego frame, plus geometry to map to world
    risk_patch: np.ndarray        # (Hp, Wp)
    patch_lon_m: float
    patch_lat_m: float
    patch_lon_offset_frac: float
    # Side probes (for Stage 2 visualization)
    probe_left:  Optional[np.ndarray]   # (2,) world frame, or None for Stage 1
    probe_right: Optional[np.ndarray]
    probe_left_risk:  Optional[float]
    probe_right_risk: Optional[float]
    # Diagnostics
    mu_lat:   float
    lam_soft: float
    lam_hard: float
    crashed:  bool
    truncated: bool
    # Lane geometry (for road bounds)
    road_y_min: float
    road_y_max: float


def _force_decomposition_from_components(
    o, heading, speed, o0, heading_0, goal, C, R_eff, mask,
    alphas, beta, gamma, lam_soft, lam_hard, mu_lat, V_neighbors,
    rollout_patch, d_hat,
    ttc_gain=0.0,
    ttc_threshold_s=3.0,
    ttc_softness_s=0.5,
    ttc_min_closing_speed=0.5,
    ttc_lane_halfwidth=2.0,
    ttc_boxed_risk_thresh=0.25,
    ttc_boxed_gate_sharpness=20.0,
) -> Dict[str, torch.Tensor]:
    """Re-compute each force component separately (uses surrogate internals)."""
    # Goal attraction
    F_goal = -beta.unsqueeze(-1) * (o - goal)

    # Damping
    vel_world = speed.unsqueeze(-1) * torch.stack(
        [torch.cos(heading), torch.sin(heading)], dim=-1)
    F_damp = -gamma.unsqueeze(-1) * vel_world

    # Geometric (per-vehicle barrier)
    from surrogate_integrator import ipc_piecewise, sdf_barrier_grad
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

    # Soft + hard from risk patch
    from surrogate_integrator import _bilinear_sample_ego_patch
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
        lateral_preference_bias=0.05,
    )
    probe_left = o + LATERAL_LOOKAHEAD * probe_stats["forward_world"] - LATERAL_LANE_WIDTH * probe_stats["n_lat_world"]
    probe_right = o + LATERAL_LOOKAHEAD * probe_stats["forward_world"] + LATERAL_LANE_WIDTH * probe_stats["n_lat_world"]
    risk_l = probe_stats["risk_left"]
    risk_r = probe_stats["risk_right"]
    F_lat = -(mu_lat * probe_stats["side_score"]).unsqueeze(-1) * probe_stats["n_lat_world"]
    F_ttc, _ = _ttc_longitudinal_force(
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
        "F_goal": F_goal, "F_geom": F_geom, "F_soft": F_soft,
        "F_hard": F_hard, "F_lat": F_lat, "F_ttc": F_ttc,
        "probe_left": probe_left, "probe_right": probe_right,
        "risk_l": risk_l, "risk_r": risk_r,
    }


@torch.no_grad()
def capture_step(model, observer, env, *, stage: int, device: str,
                  runtime: RuntimeKnobs
                  ) -> Tuple[np.ndarray, FrameState]:
    """One step: read obs, compute model output + force decomposition, return
    action and FrameState (before stepping the env)."""
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

    decomp = _force_decomposition_from_components(
        o=batch["o0"], heading=heading_0, speed=speed_0,
        o0=batch["o0"], heading_0=heading_0, goal=batch["goal"],
        C=batch["C"], R_eff=batch["R"], mask=batch["mask"],
        alphas=alphas, beta=beta, gamma=gamma,
        lam_soft=lam_soft, lam_hard=lam_hard, mu_lat=mu_lat,
        V_neighbors=batch.get("V_neighbors"),
        rollout_patch=batch["rollout_patch"], d_hat=batch["d_hat"],
        ttc_gain=runtime.ttc_gain,
        ttc_threshold_s=runtime.ttc_threshold_s,
        ttc_softness_s=runtime.ttc_softness_s,
        ttc_min_closing_speed=runtime.ttc_min_closing_speed,
        ttc_lane_halfwidth=runtime.ttc_lane_halfwidth,
        ttc_boxed_risk_thresh=runtime.ttc_boxed_risk_thresh,
        ttc_boxed_gate_sharpness=runtime.ttc_boxed_gate_sharpness,
    )

    # Use compute_surrogate_highway_force for the actual deployed force
    # (matches what eval/train does). Pass mu_lat=None for stage 1.
    F_tot, _, _, _ = compute_surrogate_highway_force(
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

    # Build FrameState. Read neighbors from env directly (more reliable than
    # parsing from obs_np, which has fixed-size arrays with masks).
    uenv = env.unwrapped
    ego = uenv.vehicle
    neighbors = [v for v in uenv.road.vehicles if v is not ego]
    neighbor_pos = np.array([v.position for v in neighbors], dtype=np.float32)
    neighbor_heading = np.array([v.heading for v in neighbors], dtype=np.float32)

    # Risk patch — first channel of risk_patch (normalized risk)
    risk_patch_2d = obs_np["risk_patch"][0]   # (Hp, Wp)

    # Lane bounds: highway-env has lanes 0..lanes_count-1 with width 4m,
    # so road_y spans [-2, lanes_count*4 - 2]
    lanes_count = uenv.config.get("lanes_count", 4)
    road_y_min = -2.0
    road_y_max = lanes_count * LATERAL_LANE_WIDTH - 2.0

    state = FrameState(
        step=0,
        ego_pos=np.array(ego.position, dtype=np.float32),
        ego_heading=float(ego.heading),
        ego_speed=float(ego.speed),
        neighbor_pos=neighbor_pos,
        neighbor_heading=neighbor_heading,
        F_goal=decomp["F_goal"].squeeze(0).cpu().numpy(),
        F_geom=decomp["F_geom"].squeeze(0).cpu().numpy(),
        F_soft=decomp["F_soft"].squeeze(0).cpu().numpy(),
        F_hard=decomp["F_hard"].squeeze(0).cpu().numpy(),
        F_lat=decomp["F_lat"].squeeze(0).cpu().numpy(),
        F_ttc=decomp["F_ttc"].squeeze(0).cpu().numpy(),
        F_tot=F_tot.squeeze(0).cpu().numpy(),
        risk_patch=risk_patch_2d,
        patch_lon_m=64.0, patch_lat_m=32.0, patch_lon_offset_frac=0.05,
        probe_left=(decomp["probe_left"].squeeze(0).cpu().numpy() if stage == 2 else None),
        probe_right=(decomp["probe_right"].squeeze(0).cpu().numpy() if stage == 2 else None),
        probe_left_risk=(float(decomp["risk_l"].item()) if stage == 2 else None),
        probe_right_risk=(float(decomp["risk_r"].item()) if stage == 2 else None),
        mu_lat=float(mu_lat.item()),
        lam_soft=float(lam_soft.item()),
        lam_hard=float(lam_hard.item()),
        crashed=False, truncated=False,
        road_y_min=road_y_min, road_y_max=road_y_max,
    )
    return action, state


# ─────────────────────────────────────────────────────────────────────────────
# Capture full episode
# ─────────────────────────────────────────────────────────────────────────────

def capture_episode(model, observer, env, *, stage: int, max_steps: int,
                     seed: int, device: str,
                     runtime: RuntimeKnobs) -> List[FrameState]:
    """Run model closed-loop, capture FrameState at every step."""
    _reset(env, seed)
    frames: List[FrameState] = []
    for t in range(max_steps):
        action, state = capture_step(model, observer, env,
                                       stage=stage, device=device,
                                       runtime=runtime)
        state.step = t
        frames.append(state)
        _, term, trunc, info = _step(env, action)
        if term:
            frames[-1].crashed = bool(info.get("crashed", True))
            break
        if trunc:
            frames[-1].truncated = True
            break
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — one panel per stage, plus timeseries strip
# ─────────────────────────────────────────────────────────────────────────────

def _draw_road(ax, x_center: float, x_window: float, frame: FrameState,
                cfg: RenderCfg):
    """Draw lane lines as a road background."""
    lanes_count = int(round((frame.road_y_max - frame.road_y_min) / LATERAL_LANE_WIDTH))
    x_min = x_center - x_window / 2
    x_max = x_center + x_window / 2

    # Road surface (light grey)
    ax.add_patch(Rectangle(
        (x_min, frame.road_y_min), x_max - x_min,
        frame.road_y_max - frame.road_y_min,
        facecolor="#e8e8e8", zorder=0,
    ))
    # Lane separators
    for li in range(lanes_count + 1):
        y = frame.road_y_min + li * LATERAL_LANE_WIDTH
        if li == 0 or li == lanes_count:
            ax.axhline(y, color=cfg.palette["lane_line"], linewidth=2,
                        xmin=0, xmax=1, zorder=1)
        else:
            ax.plot([x_min, x_max], [y, y],
                     color=cfg.palette["lane_line"], linewidth=0.6,
                     linestyle=(0, (8, 8)), zorder=1, alpha=0.5)


def _draw_risk_underlay(ax, frame: FrameState, cfg: RenderCfg):
    """Render the ego-frame risk patch warped to world frame as a heatmap.

    Patch is in ego frame. We rotate it back to world frame using
    frame.ego_heading and place it relative to ego position. The ego frame
    has +x along heading (so the patch extends ahead of ego), and the patch's
    longitudinal range is [-α·L, (1-α)·L] where L=patch_lon_m, α=offset_frac.
    """
    patch = frame.risk_patch  # (Hp, Wp) — Hp=lateral, Wp=longitudinal
    Hp, Wp = patch.shape
    L_lon = frame.patch_lon_m
    L_lat = frame.patch_lat_m
    alpha = frame.patch_lon_offset_frac

    # Patch corners in ego frame: (lon, lat)
    lon_min = -alpha * L_lon
    lon_max = (1 - alpha) * L_lon
    lat_min = -L_lat / 2
    lat_max = +L_lat / 2

    # Rotate ego-frame corners to world frame, then translate to ego position
    cos_h = math.cos(frame.ego_heading)
    sin_h = math.sin(frame.ego_heading)
    corners_ego = np.array([
        [lon_min, lat_min], [lon_max, lat_min],
        [lon_max, lat_max], [lon_min, lat_max],
    ], dtype=np.float32)
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners_world = corners_ego @ R.T + frame.ego_pos[None, :]

    # imshow with extent assumes axis-aligned; we use it then transform
    # via a simple bbox extent approach — for ego_heading ≈ 0 (highway),
    # the rotation is identity and we just use extent. Highway-env keeps
    # ego ~aligned with +x, so this is good enough visually.
    extent_x = (corners_world[:, 0].min(), corners_world[:, 0].max())
    extent_y = (corners_world[:, 1].min(), corners_world[:, 1].max())

    # Patch is laid out: rows = lateral (row 0 = left, row Hp-1 = right);
    # cols = longitudinal (col 0 = behind ego, col Wp-1 = ahead).
    # imshow displays row 0 at top by default, so we flip vertically to
    # put left-of-ego at top of image (matplotlib default y-up convention).
    img = np.flipud(patch)
    ax.imshow(
        img, extent=(*extent_x, *extent_y),
        cmap="Reds", alpha=cfg.risk_alpha, vmin=0, vmax=1,
        aspect="auto", origin="upper", zorder=2,
    )


def _draw_vehicle(ax, pos: np.ndarray, heading: float, color: str,
                   *, edge="black", zorder=5):
    """Draw a vehicle as an oriented rectangle with a small forward arrow."""
    # Rectangle centered on pos, rotated by heading
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    half_l = VEHICLE_LENGTH / 2
    half_w = VEHICLE_WIDTH / 2
    corners = np.array([
        [-half_l, -half_w], [half_l, -half_w],
        [half_l, half_w], [-half_l, half_w],
    ])
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    world_corners = corners @ R.T + pos
    poly = plt.Polygon(world_corners, facecolor=color, edgecolor=edge,
                        linewidth=0.8, zorder=zorder)
    ax.add_patch(poly)
    # Forward arrow
    arrow_len = VEHICLE_LENGTH * 0.4
    arrow_end = pos + arrow_len * np.array([cos_h, sin_h])
    ax.annotate("", xy=arrow_end, xytext=pos,
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
                 zorder=zorder + 1)


def _draw_force_arrows(ax, frame: FrameState, cfg: RenderCfg):
    """Force vectors radiating from ego, scaled for visibility."""
    forces = [
        ("F_goal", frame.F_goal),
        ("F_geom", frame.F_geom),
        ("F_soft", frame.F_soft),
        ("F_hard", frame.F_hard),
        ("F_lat",  frame.F_lat),
        ("F_ttc",  frame.F_ttc),
    ]
    SCALE = 1.5  # m per unit force, tuned for visibility
    for name, F in forces:
        norm = np.linalg.norm(F)
        if norm < 0.05:
            continue
        end = frame.ego_pos + F * SCALE
        ax.annotate(
            "", xy=end, xytext=frame.ego_pos,
            arrowprops=dict(arrowstyle="->", color=cfg.palette[name],
                             lw=2.0, alpha=0.85),
            zorder=10,
        )


def _draw_probes(ax, frame: FrameState, cfg: RenderCfg):
    if frame.probe_left is None:
        return
    for pos, risk_val in [(frame.probe_left, frame.probe_left_risk),
                            (frame.probe_right, frame.probe_right_risk)]:
        color = cfg.palette["probe_hi"] if risk_val > 0.3 else cfg.palette["probe_lo"]
        ax.scatter(pos[0], pos[1], s=120, c=color, marker="o",
                    edgecolors="black", linewidths=1.5, zorder=8, alpha=0.9)
        ax.text(pos[0], pos[1] + 1.5, f"{risk_val:.2f}",
                 ha="center", fontsize=9, zorder=9)


def _draw_panel(ax, frame: FrameState, *, stage_id: int, stage_label: str,
                 cfg: RenderCfg, x_window: float):
    """Top-down view of one stage at one step."""
    ax.clear()
    x_center = frame.ego_pos[0]
    _draw_road(ax, x_center, x_window, frame, cfg)
    _draw_risk_underlay(ax, frame, cfg)

    # Neighbors
    for i in range(len(frame.neighbor_pos)):
        if abs(frame.neighbor_pos[i, 0] - x_center) > x_window / 2 + 5:
            continue
        _draw_vehicle(ax, frame.neighbor_pos[i], frame.neighbor_heading[i],
                       color=cfg.palette["neighbor"], zorder=5)

    # Force arrows + probes (probes only for stage 2, draw before ego)
    _draw_force_arrows(ax, frame, cfg)
    if cfg.show_probe_dots:
        _draw_probes(ax, frame, cfg)

    # Ego (drawn last, on top)
    ego_color = cfg.palette["ego_s1" if stage_id == 1 else "ego_s2"]
    _draw_vehicle(ax, frame.ego_pos, frame.ego_heading,
                   color=ego_color, edge="black", zorder=10)

    # Crash flash
    if frame.crashed:
        ax.add_patch(Rectangle(
            (x_center - x_window/2, frame.road_y_min),
            x_window, frame.road_y_max - frame.road_y_min,
            facecolor=cfg.palette["crash"], alpha=0.25, zorder=11,
        ))
        ax.text(x_center, (frame.road_y_min + frame.road_y_max)/2,
                 "CRASH", color=cfg.palette["crash"], fontsize=32,
                 ha="center", va="center", weight="bold", zorder=12)

    ax.set_xlim(x_center - x_window/2, x_center + x_window/2)
    ax.set_ylim(frame.road_y_min - 1, frame.road_y_max + 1)
    ax.set_aspect("equal")
    ax.set_title(f"{stage_label}\n"
                  f"t={frame.step * DT_TARGET:.1f}s   "
                  f"v={frame.ego_speed:.1f} m/s   "
                  f"mu_lat={frame.mu_lat:.2f}",
                  fontsize=14, weight="bold", pad=10)
    if cfg.name == "paper":
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.2)

    # Force-text overlay (debug only)
    if cfg.show_force_text:
        force_str = (
            f"|F_goal|={np.linalg.norm(frame.F_goal):4.2f}  "
            f"|F_geom|={np.linalg.norm(frame.F_geom):4.2f}  "
            f"|F_soft|={np.linalg.norm(frame.F_soft):4.2f}  "
            f"|F_hard|={np.linalg.norm(frame.F_hard):4.2f}  "
            f"|F_lat|={np.linalg.norm(frame.F_lat):4.2f}  "
            f"|F_ttc|={np.linalg.norm(frame.F_ttc):4.2f}"
        )
        ax.text(0.01, 0.97, force_str, transform=ax.transAxes,
                 fontsize=9, verticalalignment="top",
                 family="monospace",
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))


def _draw_force_legend(ax, cfg: RenderCfg):
    """Legend for force colors. One axes shared across both panels."""
    ax.clear()
    ax.axis("off")
    items = [
        ("F_goal (attraction)", "F_goal"),
        ("F_geom (barrier)",    "F_geom"),
        ("F_soft (risk-grad)",  "F_soft"),
        ("F_hard (SDF barrier)", "F_hard"),
        ("F_lat (lateral channel)", "F_lat"),
        ("F_ttc (longitudinal TTC)", "F_ttc"),
    ]
    for i, (label, key) in enumerate(items):
        ax.annotate("", xy=(0.10, 0.85 - i*0.18), xytext=(0.02, 0.85 - i*0.18),
                     arrowprops=dict(arrowstyle="->", color=cfg.palette[key], lw=2))
        ax.text(0.13, 0.85 - i*0.18, label, fontsize=10, va="center")


def _draw_timeseries(ax_speed, ax_force, ax_mu, frames: List[FrameState],
                      step_now: int, cfg: RenderCfg, *, stage_label: str):
    """Three-row strip: speed, |F_y|, mu_lat. Rolling 4-second window."""
    window_steps = int(4.0 / DT_TARGET)
    start = max(0, step_now - window_steps)
    times = np.array([f.step * DT_TARGET for f in frames[start:step_now+1]])
    if len(times) < 2:
        for ax in [ax_speed, ax_force, ax_mu]:
            ax.clear()
        return

    # Speed
    speeds = np.array([f.ego_speed for f in frames[start:step_now+1]])
    ax_speed.clear()
    ax_speed.plot(times, speeds, color=cfg.palette["speed"], lw=2)
    ax_speed.set_ylabel("speed (m/s)", fontsize=10)
    ax_speed.set_ylim(0, 35)
    ax_speed.grid(alpha=0.2)
    ax_speed.tick_params(labelsize=9)
    if cfg.name == "paper":
        ax_speed.set_xticklabels([])

    # |F_y|
    abs_fy = np.array([abs(f.F_tot[1]) for f in frames[start:step_now+1]])
    ax_force.clear()
    ax_force.plot(times, abs_fy, color=cfg.palette["F_lat"], lw=2)
    ax_force.set_ylabel("|F_y|", fontsize=10)
    ax_force.set_ylim(0, max(5, abs_fy.max() * 1.1))
    ax_force.grid(alpha=0.2)
    ax_force.tick_params(labelsize=9)

    # mu_lat
    mu_vals = np.array([f.mu_lat for f in frames[start:step_now+1]])
    ax_mu.clear()
    ax_mu.plot(times, mu_vals, color=cfg.palette["mu_lat"], lw=2)
    ax_mu.set_ylabel("mu_lat", fontsize=10)
    ax_mu.set_xlabel("time (s)", fontsize=10)
    ax_mu.set_ylim(-0.2, max(5, mu_vals.max() * 1.2))
    ax_mu.grid(alpha=0.2)
    ax_mu.tick_params(labelsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# Animation builder
# ─────────────────────────────────────────────────────────────────────────────

def render_paired_gif(
    s1_frames: List[FrameState], s2_frames: List[FrameState],
    *, scenario_title: str, out_path: Path, cfg: RenderCfg,
    x_window: float, visualization_only: bool = False,
    hide_scenario_title: bool = False,
):
    """Build a side-by-side GIF from captured frames of both stages.

    If episodes have different lengths (e.g., one crashed), we pad the shorter
    one by repeating the last frame so both panels stay synchronized in time.
    """
    n_frames = max(len(s1_frames), len(s2_frames))
    # Pad with last frame
    while len(s1_frames) < n_frames:
        s1_frames.append(s1_frames[-1])
    while len(s2_frames) < n_frames:
        s2_frames.append(s2_frames[-1])

    fig = plt.figure(
        figsize=((16, 3.8) if visualization_only else cfg.figsize),
        dpi=cfg.dpi,
    )
    if visualization_only:
        # Large bird's-eye comparison without the diagnostic time-series.
        gs = fig.add_gridspec(
            nrows=1, ncols=3,
            width_ratios=[6, 6, 1.35],
            left=0.025, right=0.985, bottom=0.08,
            top=(0.90 if hide_scenario_title else 0.72),
            wspace=0.12,
        )
        ax_s1 = fig.add_subplot(gs[0, 0])
        ax_s2 = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[0, 2])
        timeseries_axes = None
    else:
        # Layout: 2 main panels on top, then 3 timeseries rows (speed, |F_y|, mu)
        # × 2 columns (one per stage). Plus a legend strip on the right.
        gs = fig.add_gridspec(
            nrows=4, ncols=3,
            height_ratios=[3, 0.6, 0.6, 0.6],
            width_ratios=[5, 5, 1.2],
            hspace=0.35, wspace=0.15,
        )
        ax_s1 = fig.add_subplot(gs[0, 0])
        ax_s2 = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[0, 2])

        ax_s1_speed = fig.add_subplot(gs[1, 0])
        ax_s1_force = fig.add_subplot(gs[2, 0])
        ax_s1_mu    = fig.add_subplot(gs[3, 0])

        ax_s2_speed = fig.add_subplot(gs[1, 1])
        ax_s2_force = fig.add_subplot(gs[2, 1])
        ax_s2_mu    = fig.add_subplot(gs[3, 1])
        timeseries_axes = (
            ax_s1_speed, ax_s1_force, ax_s1_mu,
            ax_s2_speed, ax_s2_force, ax_s2_mu,
        )

    if not hide_scenario_title:
        fig.suptitle(scenario_title, fontsize=17, weight="bold")
    _draw_force_legend(ax_legend, cfg)

    def update(i):
        _draw_panel(ax_s1, s1_frames[i], stage_id=1,
                    stage_label=STAGE1_PANEL_TITLE, cfg=cfg,
                    x_window=x_window)
        _draw_panel(ax_s2, s2_frames[i], stage_id=2,
                    stage_label=STAGE2_PANEL_TITLE, cfg=cfg,
                    x_window=x_window)
        if timeseries_axes is not None:
            _draw_timeseries(*timeseries_axes[:3],
                             s1_frames, i, cfg, stage_label="Stage 1")
            _draw_timeseries(*timeseries_axes[3:],
                             s2_frames, i, cfg, stage_label="Stage 2")
        return []

    anim = FuncAnimation(fig, update, frames=n_frames,
                          interval=int(1000 / cfg.fps), blit=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=cfg.fps)
    anim.save(str(out_path), writer=writer, dpi=cfg.dpi)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(ckpt_path: Path, device: str, dfc_root: str = ""):
    if dfc_root:
        sys.path.insert(0, dfc_root)
    from train_material import CoefEnergyNetMaterial  # noqa: E402

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_train = ck.get("cfg", {})
    lam_soft_max = cfg_train.get("lam_soft_max", 50.0)
    lam_hard_max = cfg_train.get("lam_hard_max", 10.0)
    mu_lat_max   = cfg_train.get("mu_lat_max", 5.0)
    runtime = RuntimeKnobs(
        d_hat=float(cfg_train.get("d_hat", 0.0)),
        alpha_floor=float(cfg_train.get("alpha_floor", 0.0)),
        alpha_floor_ahead_only=bool(cfg_train.get("alpha_floor_ahead_only", False)),
    )

    # Try with mu_lat_max; if model class is older, fall back without it
    try:
        model = CoefEnergyNetMaterial(
            lam_soft_max=lam_soft_max, lam_hard_max=lam_hard_max,
            mu_lat_max=mu_lat_max,
        ).to(device)
    except TypeError:
        model = CoefEnergyNetMaterial(
            lam_soft_max=lam_soft_max, lam_hard_max=lam_hard_max,
        ).to(device)
    disable_transformer_nested_tensors(model)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model, runtime


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--stage1-ckpt", type=str, required=True)
    ap.add_argument("--stage2-ckpt", type=str, required=True)
    ap.add_argument("--scenarios",   type=str, nargs="+",
                    default=["default", "authored_slow_leader",
                             "authored_slow_leader_boxed"])
    ap.add_argument("--seed",        type=int, default=1000)
    ap.add_argument("--max-steps",   type=int, default=120)
    ap.add_argument("--config",      type=str, default="paper",
                    choices=["paper", "debug", "both"])
    ap.add_argument("--out",         type=str, default="runs/figures/")
    ap.add_argument("--n-max-vehicles", type=int, default=15)
    ap.add_argument("--x-window",    type=float, default=120.0,
                    help="Longitudinal meters shown in each bird's-eye panel. "
                         "Use 80 for close action, 120+ to include more "
                         "background traffic.")
    ap.add_argument("--visualization-only", action="store_true",
                    help="Render enlarged bird's-eye panels without the lower time-series plots.")
    ap.add_argument("--hide-scenario-title", action="store_true",
                    help="Omit the overall scenario heading above the paired panels.")
    ap.add_argument("--device",      type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dfc-root",    type=str, default="")
    ap.add_argument("--ttc-gain",    type=float, default=0.0,
                    help="Eval-time TTC braking gain for Stage 2 only.")
    ap.add_argument("--ttc-threshold-s", type=float, default=3.0,
                    help="TTC threshold for Stage 2 GIF capture.")
    ap.add_argument("--ttc-softness-s", type=float, default=0.5)
    ap.add_argument("--ttc-min-closing-speed", type=float, default=0.5)
    ap.add_argument("--ttc-lane-halfwidth", type=float, default=2.0)
    ap.add_argument("--ttc-boxed-risk-thresh", type=float, default=0.25)
    ap.add_argument("--ttc-boxed-gate-sharpness", type=float, default=20.0)
    args = ap.parse_args()

    print(f"Loading models...")
    s1_model, s1_runtime = _load_model(
        Path(args.stage1_ckpt), args.device, args.dfc_root)
    s2_model, s2_runtime = _load_model(
        Path(args.stage2_ckpt), args.device, args.dfc_root)
    s2_runtime.ttc_gain = float(args.ttc_gain)
    s2_runtime.ttc_threshold_s = float(args.ttc_threshold_s)
    s2_runtime.ttc_softness_s = float(args.ttc_softness_s)
    s2_runtime.ttc_min_closing_speed = float(args.ttc_min_closing_speed)
    s2_runtime.ttc_lane_halfwidth = float(args.ttc_lane_halfwidth)
    s2_runtime.ttc_boxed_risk_thresh = float(args.ttc_boxed_risk_thresh)
    s2_runtime.ttc_boxed_gate_sharpness = float(args.ttc_boxed_gate_sharpness)
    print(f"  Stage 1: {args.stage1_ckpt}")
    print(f"    d_hat={s1_runtime.d_hat:.1f}  "
          f"alpha_floor={s1_runtime.alpha_floor:.4f}  "
          f"ahead_only={s1_runtime.alpha_floor_ahead_only}")
    print(f"  Stage 2: {args.stage2_ckpt}")
    print(f"    d_hat={s2_runtime.d_hat:.1f}  "
          f"alpha_floor={s2_runtime.alpha_floor:.4f}  "
          f"ahead_only={s2_runtime.alpha_floor_ahead_only}")
    if s2_runtime.ttc_gain > 0:
        print(f"    ttc_gain={s2_runtime.ttc_gain:.2f}  "
              f"ttc_threshold={s2_runtime.ttc_threshold_s:.2f}s  "
              f"boxed_thresh={s2_runtime.ttc_boxed_risk_thresh:.2f}")

    gym = _import_gym()
    observer = HighwayMaterialObservation(WrapperConfig(
        n_max_vehicles=args.n_max_vehicles,
        dt_surrogate=DT_TARGET,
        horizon_surrogate=H_TARGET,
    ))

    out_dir = Path(args.out)
    configs = []
    if args.config in ("paper", "both"):
        configs.append(PAPER_CFG)
    if args.config in ("debug", "both"):
        configs.append(DEBUG_CFG)

    for scn_name in args.scenarios:
        if scn_name not in SCENARIOS:
            print(f"Unknown scenario {scn_name!r}, skipping.")
            continue
        print(f"\n━━━ {scn_name} ━━━")
        print(f"  Capturing Stage 1...")
        env_s1 = make_env(gym, scn_name)
        try:
            s1_frames = capture_episode(s1_model, observer, env_s1,
                                          stage=1, max_steps=args.max_steps,
                                          seed=args.seed, device=args.device,
                                          runtime=s1_runtime)
        finally:
            env_s1.close()
        n_s1 = len(s1_frames)
        crashed_s1 = s1_frames[-1].crashed if s1_frames else False
        print(f"    {n_s1} frames {'(CRASH)' if crashed_s1 else '(no crash)'}")

        print(f"  Capturing Stage 2...")
        env_s2 = make_env(gym, scn_name)
        try:
            s2_frames = capture_episode(s2_model, observer, env_s2,
                                          stage=2, max_steps=args.max_steps,
                                          seed=args.seed, device=args.device,
                                          runtime=s2_runtime)
        finally:
            env_s2.close()
        n_s2 = len(s2_frames)
        crashed_s2 = s2_frames[-1].crashed if s2_frames else False
        print(f"    {n_s2} frames {'(CRASH)' if crashed_s2 else '(no crash)'}")

        title = SCENARIOS[scn_name]["title"]
        for cfg in configs:
            tag = "" if len(configs) == 1 else f"_{cfg.name}"
            out_path = out_dir / f"{scn_name}{tag}.gif"
            print(f"  Rendering {cfg.name} → {out_path}")
            t0 = time.time()
            render_paired_gif(
                # Pass shallow copies so each render doesn't mutate the canonical lists
                list(s1_frames), list(s2_frames),
                scenario_title=title, out_path=out_path, cfg=cfg,
                x_window=args.x_window,
                visualization_only=args.visualization_only,
                hide_scenario_title=args.hide_scenario_title,
            )
            print(f"    done in {time.time()-t0:.1f}s")

    print(f"\nGIFs saved to {out_dir}/")


if __name__ == "__main__":
    main()
