#!/usr/bin/env python3
"""
eval_stage2.py — Step 6 component 4.

Comprehensive Stage 2 evaluation. Goes beyond eval_stage1.py with:

  1. Lateral activity diagnostics — mean and p95 |lateral accel|, lateral
     position variance. Distinguishes "no lane change because env is empty"
     from "no lane change because model never tries."
     Also reports on-road fraction and LKE tail/max so offroad lateral
     escapes cannot masquerade as successful lane changes.

  2. Scenario sweep — runs the same model across multiple highway-env
     configurations (slow-leader, dense-traffic, sparse). Reports per-scenario
     metrics. Context-sensitivity is the load-bearing claim for the spotlight
     pitch — if Stage 2 behaves the SAME across scenarios, the claim fails.

  3. Paired Stage 1 vs Stage 2 — runs both checkpoints on the same seed set
     and prints a side-by-side comparison table. Reviewer-ready numbers.

  4. CVaR-of-cum-risk over a held-out distribution. Stage 2's training-time
     CVaR is from on-policy collection; this is the eval-time analog over
     fixed eval seeds.

Usage
-----
    # Stage 2 alone, full sweep
    python eval_stage2.py \\
        --ckpt checkpoints/highway_stage2_navscale/best_closed_loop.pt \\
        --episodes 20 --scenarios slow_leader dense sparse \\
        --out runs/eval_stage2.json

    # Paired Stage 1 vs Stage 2 (the headline comparison)
    python eval_stage2.py \\
        --ckpt checkpoints/highway_stage2_navscale/best_closed_loop.pt \\
        --stage1-ckpt checkpoints/highway_stage1_action_dhat10_afloor0015/best.pt \\
        --episodes 20 \\
        --out runs/eval_paired.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

from env_wrapper import HighwayMaterialObservation, WrapperConfig, _ego_lane_center_y  # noqa: E402
from bicycle_surrogate import (  # noqa: E402
    ACCEL_RANGE, STEER_RANGE, force_to_action,
)
from surrogate_integrator import compute_surrogate_highway_force  # noqa: E402
from eval_stage1 import _apply_alpha_floor, disable_transformer_nested_tensors  # noqa: E402


H_TARGET    = 20
DT_TARGET   = 0.1
POLICY_FREQ = 10
SIM_FREQ    = 30
LANE_WIDTH  = 4.0
EPS         = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    """Highway-env config knobs that define a scenario.

    `expected_behavior` documents what Stage 2 SHOULD do in this scenario,
    for downstream interpretation. Doesn't affect simulation."""
    name: str
    vehicles_count: int
    lanes_count: int
    env_id: str = "highway-v0"
    initial_lane_id: Optional[int] = None
    ego_spacing: float = 2.0
    expected_behavior: str = ""


SCENARIOS: Dict[str, ScenarioConfig] = {
    # Default highway-v0 conditions, matches Stage 1 / training distribution
    "default": ScenarioConfig(
        name="default", vehicles_count=50, lanes_count=4,
        expected_behavior="follow leaders, occasional lane change",
    ),
    # Sparse traffic — ego should run at near-target speed unimpeded
    "sparse": ScenarioConfig(
        name="sparse", vehicles_count=20, lanes_count=4,
        expected_behavior="run at desired speed, no lateral activity",
    ),
    # Dense traffic — ego should slow to follow leaders
    "dense": ScenarioConfig(
        name="dense", vehicles_count=80, lanes_count=4,
        expected_behavior="slow to leader speed, low lateral activity",
    ),
    # Slow leader — ego should change lanes around blocking vehicle
    # NOTE: highway-v0 doesn't have a 'slow_leader' preset. We approximate
    # via mid-density traffic where lane changes are advantageous. True
    # slow-leader scenarios would need scenario authoring; flagged for later.
    "slow_leader": ScenarioConfig(
        name="slow_leader", vehicles_count=40, lanes_count=4,
        ego_spacing=1.4,           # tighter, biases toward leaders ahead
        expected_behavior="lane change around slow blocker",
    ),
    "authored_slow_leader": ScenarioConfig(
        name="authored_slow_leader",
        env_id="highway-slow-leader-v0",
        vehicles_count=0,
        lanes_count=4,
        expected_behavior="pass the fixed slow leader using the open adjacent lane",
    ),
    "authored_slow_leader_x2": ScenarioConfig(
        name="authored_slow_leader_x2",
        env_id="highway-slow-leader-x2-v0",
        vehicles_count=0,
        lanes_count=4,
        expected_behavior="attempt repeated overtakes across a 2-leader open convoy",
    ),
    "authored_slow_leader_x3": ScenarioConfig(
        name="authored_slow_leader_x3",
        env_id="highway-slow-leader-x3-v0",
        vehicles_count=0,
        lanes_count=4,
        expected_behavior="attempt repeated overtakes across a 3-leader open convoy",
    ),
    "authored_slow_leader_x4": ScenarioConfig(
        name="authored_slow_leader_x4",
        env_id="highway-slow-leader-x4-v0",
        vehicles_count=0,
        lanes_count=4,
        expected_behavior="attempt repeated overtakes across a 4-leader open convoy",
    ),
    "authored_slow_leader_boxed": ScenarioConfig(
        name="authored_slow_leader_boxed",
        env_id="highway-slow-leader-boxed-v0",
        vehicles_count=0,
        lanes_count=4,
        expected_behavior="slow behind the fixed leader because adjacent lanes are blocked",
    ),
    "authored_slow_leader_boxed_x2": ScenarioConfig(
        name="authored_slow_leader_boxed_x2",
        env_id="highway-slow-leader-boxed-x2-v0",
        vehicles_count=0,
        lanes_count=4,
        expected_behavior="follow safely behind a boxed 2-leader convoy",
    ),
    "authored_slow_leader_boxed_x3": ScenarioConfig(
        name="authored_slow_leader_boxed_x3",
        env_id="highway-slow-leader-boxed-x3-v0",
        vehicles_count=0,
        lanes_count=4,
        expected_behavior="follow safely behind a boxed 3-leader convoy",
    ),
    "authored_slow_leader_boxed_x4": ScenarioConfig(
        name="authored_slow_leader_boxed_x4",
        env_id="highway-slow-leader-boxed-x4-v0",
        vehicles_count=0,
        lanes_count=4,
        expected_behavior="follow safely behind a boxed 4-leader convoy",
    ),
    "authored_slow_convoy_boxed": ScenarioConfig(
        name="authored_slow_convoy_boxed",
        env_id="highway-slow-convoy-boxed-v0",
        vehicles_count=0,
        lanes_count=4,
        expected_behavior="sustain collision-free following behind a boxed convoy of slow leaders",
    ),
    "authored_slow_convoy": ScenarioConfig(
        name="authored_slow_convoy",
        env_id="highway-slow-convoy-v0",
        vehicles_count=0,
        lanes_count=4,
        expected_behavior="attempt repeated overtakes across a 4-leader open convoy",
    ),
}


@dataclass
class RuntimeKnobs:
    """Deployment knobs loaded from checkpoint config unless CLI overrides."""
    d_hat: float = 0.0
    alpha_floor: float = 0.0
    alpha_floor_ahead_only: bool = False
    disable_mu_lat: bool = False
    ttc_gain: float = 0.0
    ttc_threshold_s: float = 3.0
    ttc_softness_s: float = 0.5
    ttc_min_closing_speed: float = 0.5
    ttc_lane_halfwidth: float = 2.0
    ttc_boxed_risk_thresh: float = 0.25
    ttc_boxed_gate_sharpness: float = 20.0


@dataclass
class MetricConfig:
    """Thresholds/weights for highway paper metrics."""
    d_safe: float = 8.0
    d_near: float = 4.0
    ttc_safe_s: float = 3.0
    ttc_crit_s: float = 1.5
    v_ref: float = 25.0
    v_min: float = 1.0
    hard_brake_accel: float = -4.0
    w_collision: float = 1000.0
    w_offroad: float = 500.0
    w_lane: float = 50.0
    w_clearance: float = 20.0
    w_ttc: float = 25.0
    w_progress: float = 0.5
    w_control: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Env construction
# ─────────────────────────────────────────────────────────────────────────────

def _import_gym():
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401
        return gym
    except ImportError as exc:
        raise SystemExit(f"highway_env not importable: {exc}") from exc


def make_scenario_env(gym, scn: ScenarioConfig, *, offroad_terminal: bool = False):
    config = {
        "policy_frequency":     POLICY_FREQ,
        "simulation_frequency": SIM_FREQ,
        "vehicles_count":       scn.vehicles_count,
        "lanes_count":          scn.lanes_count,
        "ego_spacing":          scn.ego_spacing,
        "duration":             40,
        "offroad_terminal":     bool(offroad_terminal),
        "action": {
            "type":               "ContinuousAction",
            "longitudinal":       True,
            "lateral":            True,
            "acceleration_range": list(ACCEL_RANGE),
            "steering_range":     list(STEER_RANGE),
        },
    }
    if scn.initial_lane_id is not None:
        config["initial_lane_id"] = scn.initial_lane_id
    return gym.make(scn.env_id, config=config)


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


# ─────────────────────────────────────────────────────────────────────────────
# Action normalization (mirrors collect_onpolicy)
# ─────────────────────────────────────────────────────────────────────────────

_ACCEL_HALFRANGE = (ACCEL_RANGE[1] - ACCEL_RANGE[0]) / 2.0
_STEER_HALFRANGE = (STEER_RANGE[1] - STEER_RANGE[0]) / 2.0


def _normalize_action(accel_phys: float, steer_phys: float) -> np.ndarray:
    a_lo, a_hi = ACCEL_RANGE
    s_lo, s_hi = STEER_RANGE
    a = float(np.clip(2.0 * (accel_phys - a_lo) / (a_hi - a_lo) - 1.0,
                      -1.0, 1.0))
    s = float(np.clip(2.0 * (steer_phys - s_lo) / (s_hi - s_lo) - 1.0,
                      -1.0, 1.0))
    return np.array([a, s], dtype=np.float32)


def _vehicle_velocity(vehicle) -> np.ndarray:
    vel = getattr(vehicle, "velocity", None)
    if vel is not None:
        arr = np.asarray(vel, dtype=np.float64)
        if arr.shape == (2,):
            return arr
    speed = float(getattr(vehicle, "speed", 0.0))
    heading = float(getattr(vehicle, "heading", 0.0))
    return np.array([speed * math.cos(heading), speed * math.sin(heading)], dtype=np.float64)


def _min_ttc_to_neighbors(env, *, min_closing_speed: float = 0.1) -> float:
    """Centerline TTC proxy over all non-ego vehicles.

    This intentionally mirrors a metric, not a controller: it estimates the
    earliest temporal near-miss from relative center distance and closing speed.
    """
    uenv = env.unwrapped
    ego = uenv.vehicle
    road = getattr(uenv, "road", None)
    vehicles = getattr(road, "vehicles", []) if road is not None else []
    p_ego = np.asarray(ego.position, dtype=np.float64)
    v_ego = _vehicle_velocity(ego)
    ego_len = float(getattr(ego, "LENGTH", 5.0))
    best = float("inf")
    for other in vehicles:
        if other is ego:
            continue
        p_other = np.asarray(other.position, dtype=np.float64)
        rel_p = p_other - p_ego
        center_dist = float(np.linalg.norm(rel_p))
        if center_dist <= EPS:
            return 0.0
        v_other = _vehicle_velocity(other)
        rel_v = v_other - v_ego
        closing_speed = -float(np.dot(rel_p, rel_v)) / center_dist
        if closing_speed <= min_closing_speed:
            continue
        other_len = float(getattr(other, "LENGTH", 5.0))
        clearance = max(0.0, center_dist - 0.5 * (ego_len + other_len))
        best = min(best, clearance / closing_speed)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Model interface
# ─────────────────────────────────────────────────────────────────────────────

def _to_batch(obs_np: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
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


def _resolve_runtime_knobs(
    ck: Dict[str, Any],
    *,
    d_hat_override: float,
    alpha_floor_override: float,
    alpha_floor_ahead_only_override: Optional[bool],
    ttc_gain_override: Optional[float],
    ttc_threshold_s_override: Optional[float],
    ttc_softness_s_override: Optional[float],
    ttc_min_closing_speed_override: Optional[float],
    ttc_lane_halfwidth_override: Optional[float],
    ttc_boxed_risk_thresh_override: Optional[float],
    ttc_boxed_gate_sharpness_override: Optional[float],
) -> RuntimeKnobs:
    cfg = ck.get("cfg", {})
    return RuntimeKnobs(
        d_hat=(
            float(d_hat_override) if d_hat_override > 0
            else float(cfg.get("d_hat", 0.0))
        ),
        alpha_floor=(
            float(alpha_floor_override) if alpha_floor_override >= 0
            else float(cfg.get("alpha_floor", 0.0))
        ),
        alpha_floor_ahead_only=(
            bool(alpha_floor_ahead_only_override)
            if alpha_floor_ahead_only_override is not None
            else bool(cfg.get("alpha_floor_ahead_only", False))
        ),
        ttc_gain=(
            float(ttc_gain_override)
            if ttc_gain_override is not None
            else float(cfg.get("ttc_gain", 0.0))
        ),
        ttc_threshold_s=(
            float(ttc_threshold_s_override)
            if ttc_threshold_s_override is not None
            else float(cfg.get("ttc_threshold_s", 3.0))
        ),
        ttc_softness_s=(
            float(ttc_softness_s_override)
            if ttc_softness_s_override is not None
            else float(cfg.get("ttc_softness_s", 0.5))
        ),
        ttc_min_closing_speed=(
            float(ttc_min_closing_speed_override)
            if ttc_min_closing_speed_override is not None
            else float(cfg.get("ttc_min_closing_speed", 0.5))
        ),
        ttc_lane_halfwidth=(
            float(ttc_lane_halfwidth_override)
            if ttc_lane_halfwidth_override is not None
            else float(cfg.get("ttc_lane_halfwidth", 2.0))
        ),
        ttc_boxed_risk_thresh=(
            float(ttc_boxed_risk_thresh_override)
            if ttc_boxed_risk_thresh_override is not None
            else float(cfg.get("ttc_boxed_risk_thresh", 0.25))
        ),
        ttc_boxed_gate_sharpness=(
            float(ttc_boxed_gate_sharpness_override)
            if ttc_boxed_gate_sharpness_override is not None
            else float(cfg.get("ttc_boxed_gate_sharpness", 20.0))
        ),
    )


def _model_coeffs(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
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
    return model(obs_feats=obs_feats, obs_mask=batch["mask"],
                  goal_feats=goal_feats, risk_patch=batch["risk_patch"])


# ─────────────────────────────────────────────────────────────────────────────
# Per-step diagnostics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepDiag:
    """Per-step diagnostics. Aggregated to per-episode metrics."""
    speed:     float
    accel_phys: float
    steer_phys: float
    dmin:      float           # min clearance to nearest neighbour
    risk_val:  float           # risk-field value at ego
    lat_y:     float           # lateral position
    lane_y:    float           # lane center lateral
    lane_idx:  Any             # env's lane_index tuple
    on_road:   bool            # highway-env ego on-road flag
    lane_violation_m: float    # excess distance outside current lane band
    min_ttc_s: float           # min positive time-to-collision to neighbours
    progress_x: float          # longitudinal ego coordinate


@dataclass
class EpisodeMetrics:
    seed: int
    scenario: str
    steps: int
    collided: bool
    truncated: bool
    safe_success: bool
    failure_time_s: float
    distance_m: float
    progress_m: float
    progress_rate_mps: float
    mean_speed: float
    speed_std: float
    speed_deficit: float
    stagnation_rate: float
    # Lateral activity — the new diagnostics
    lateral_accel_mean_abs: float    # mean |steer * v² / L| approximation
    lateral_accel_p95_abs: float
    lateral_pos_std: float           # std of (y_pos - lane_center_y)
    lane_keep_err_mean: float
    lane_keep_err_p95: float
    lane_keep_err_max: float
    lane_changes: int
    on_road_fraction: float
    offroad_steps: int
    went_offroad: bool
    ended_offroad: bool
    lane_violation_duration_s: float
    lane_violation_cum_m_s: float
    lane_violation_max_m: float
    # Risk-field diagnostics
    cum_risk_eval: float             # sum of risk_val over rollout
    cvar_step_risk: float            # CVaR of per-step risk_val
    min_clearance: float
    mean_clearance: float
    clearance_violation_duration_s: float
    clearance_violation_cum_m_s: float
    near_miss: bool
    min_ttc_s: float
    ttc_violation_duration_s: float
    ttc_violation_cum_s: float
    critical_ttc_event: bool
    dynamic_risk_event: bool
    response_delay_s: float
    intervention_window_proxy_s: float
    post_risk_min_ttc_s: float
    post_risk_clearance_violation_cum_m_s: float
    evasive_success: bool
    # Force-side diagnostics
    lam_soft_mean: float
    lam_hard_mean: float
    F_norm_mean: float
    # Control quality
    accel_energy: float
    steering_energy: float
    control_jerk_energy: float
    hard_brake_count: int
    failure_score: float


# ─────────────────────────────────────────────────────────────────────────────
# Per-step action computation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_step(
    model, observer, env, *, device: str, runtime: RuntimeKnobs, stage: int = 2,
) -> Tuple[np.ndarray, StepDiag, Dict[str, float]]:
    """One step of model output: returns (normalized action, diagnostics)."""
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
        mu_lat = None
    elif runtime.disable_mu_lat:
        mu_lat = None

    v0 = batch["v0"]
    speed_0   = torch.linalg.norm(v0, dim=-1).clamp_min(1e-3)
    heading_0 = torch.atan2(v0[:, 1], v0[:, 0])

    F_tot, dmin, risk_val, _ = compute_surrogate_highway_force(
        o             = batch["o0"],
        heading       = heading_0,
        speed         = speed_0,
        o0            = batch["o0"],
        heading_0     = heading_0,
        goal          = batch["goal"],
        C             = batch["C"],
        V_neighbors   = batch.get("V_neighbors"),
        R_eff         = batch["R"],
        mask          = batch["mask"],
        alphas        = alphas,
        beta          = beta,
        gamma         = gamma,
        lam_soft      = lam_soft,
        lam_hard      = lam_hard,
        mu_lat        = mu_lat,
        rollout_patch = batch["rollout_patch"],
        d_hat         = batch["d_hat"],
        ttc_gain      = runtime.ttc_gain,
        ttc_threshold_s = runtime.ttc_threshold_s,
        ttc_softness_s = runtime.ttc_softness_s,
        ttc_min_closing_speed = runtime.ttc_min_closing_speed,
        ttc_lane_halfwidth = runtime.ttc_lane_halfwidth,
        ttc_boxed_risk_thresh = runtime.ttc_boxed_risk_thresh,
        ttc_boxed_gate_sharpness = runtime.ttc_boxed_gate_sharpness,
    )

    accel, steer = force_to_action(F_tot, heading_0, speed_0)
    accel_phys = float(accel.clamp(*ACCEL_RANGE).item())
    steer_phys = float(steer.clamp(*STEER_RANGE).item())
    action_norm = _normalize_action(accel_phys, steer_phys)

    # Lane diagnostics
    ego_pos = np.array(env.unwrapped.vehicle.position, dtype=np.float64)
    try:
        lane_y = _ego_lane_center_y(env, ego_pos)
    except Exception:
        lane_y = float("nan")
    lane_idx = getattr(env.unwrapped.vehicle, "lane_index", None)
    lane_violation = 0.0
    if not math.isnan(lane_y):
        lane_violation = max(0.0, abs(float(ego_pos[1]) - float(lane_y)) - LANE_WIDTH / 2.0)
    min_ttc = _min_ttc_to_neighbors(
        env,
        min_closing_speed=max(0.1, float(runtime.ttc_min_closing_speed)),
    )

    diag = StepDiag(
        speed      = float(speed_0.item()),
        accel_phys = accel_phys,
        steer_phys = steer_phys,
        dmin       = float(dmin.item()),
        risk_val   = float(risk_val.item()),
        lat_y      = float(ego_pos[1]),
        lane_y     = float(lane_y),
        lane_idx   = lane_idx,
        on_road    = bool(getattr(env.unwrapped.vehicle, "on_road", True)),
        lane_violation_m = float(lane_violation),
        min_ttc_s = float(min_ttc),
        progress_x = float(ego_pos[0]),
    )
    coeff_diag = {
        "F_norm":   float(F_tot.norm(dim=-1).item()),
        "lam_soft": float(lam_soft.item()),
        "lam_hard": float(lam_hard.item()),
    }
    return action_norm, diag, coeff_diag


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode rollout
# ─────────────────────────────────────────────────────────────────────────────

def _detect_lane_change(prev_idx, cur_idx) -> bool:
    if prev_idx is None or cur_idx is None:
        return False
    try:
        return prev_idx[2] != cur_idx[2]
    except (IndexError, TypeError):
        return prev_idx != cur_idx


# Approximate lateral acceleration from steering angle.
# Bicycle: a_lat ≈ v² · tan(δ) / L. With δ small, a_lat ≈ v² · δ / L.
# Wheelbase L ≈ 5m for highway-env's default vehicle.
_VEHICLE_WHEELBASE = 5.0


def run_episode(
    model, observer, env, *,
    seed: int, scenario_name: str, max_steps: int, device: str, stage: int,
    runtime: RuntimeKnobs,
    metric_cfg: MetricConfig,
    verbose: bool = False,
) -> EpisodeMetrics:
    _reset(env, seed)
    uenv = env.unwrapped
    o_start = np.array(uenv.vehicle.position, dtype=np.float64).copy()

    diags: List[StepDiag] = []
    coeff_diags: List[Dict[str, float]] = []
    lane_changes = 0
    prev_lane = getattr(uenv.vehicle, "lane_index", None)
    collided = False
    truncated = False

    for k in range(max_steps):
        action, diag, coeff_d = _compute_step(model, observer, env,
                                                device=device,
                                                runtime=runtime, stage=stage)
        diags.append(diag)
        coeff_diags.append(coeff_d)

        _, term, trunc, info = _step(env, action)
        cur_lane = getattr(uenv.vehicle, "lane_index", None)
        if _detect_lane_change(prev_lane, cur_lane):
            lane_changes += 1
        prev_lane = cur_lane

        if term:
            collided = bool(info.get("crashed", True))
            break
        if trunc:
            truncated = True
            break

    o_end = np.array(uenv.vehicle.position, dtype=np.float64).copy()
    final_on_road = bool(getattr(uenv.vehicle, "on_road", True))

    if not diags:
        # episode died on the very first step
        return EpisodeMetrics(
            seed=seed, scenario=scenario_name, steps=0, collided=collided,
            truncated=truncated, safe_success=not collided and final_on_road,
            failure_time_s=0.0 if (collided or not final_on_road) else 0.0,
            distance_m=0.0, progress_m=0.0, progress_rate_mps=0.0,
            mean_speed=0.0, speed_deficit=0.0, stagnation_rate=1.0,
            speed_std=0.0, lateral_accel_mean_abs=0.0,
            lateral_accel_p95_abs=0.0, lateral_pos_std=0.0,
            lane_keep_err_mean=float("nan"), lane_keep_err_p95=float("nan"),
            lane_keep_err_max=float("nan"), lane_changes=0,
            on_road_fraction=1.0 if final_on_road else 0.0,
            offroad_steps=0 if final_on_road else 1,
            went_offroad=not final_on_road,
            ended_offroad=not final_on_road,
            lane_violation_duration_s=0.0 if final_on_road else DT_TARGET,
            lane_violation_cum_m_s=0.0 if final_on_road else LANE_WIDTH * DT_TARGET,
            lane_violation_max_m=0.0 if final_on_road else LANE_WIDTH,
            cum_risk_eval=0.0, cvar_step_risk=0.0,
            min_clearance=float("inf"), mean_clearance=float("inf"),
            clearance_violation_duration_s=0.0,
            clearance_violation_cum_m_s=0.0,
            near_miss=False,
            min_ttc_s=float("inf"),
            ttc_violation_duration_s=0.0,
            ttc_violation_cum_s=0.0,
            critical_ttc_event=False,
            dynamic_risk_event=False,
            response_delay_s=float("nan"),
            intervention_window_proxy_s=float("nan"),
            post_risk_min_ttc_s=float("nan"),
            post_risk_clearance_violation_cum_m_s=float("nan"),
            evasive_success=False,
            lam_soft_mean=0.0, lam_hard_mean=0.0, F_norm_mean=0.0,
            accel_energy=0.0, steering_energy=0.0,
            control_jerk_energy=0.0, hard_brake_count=0,
            failure_score=(
                metric_cfg.w_collision * float(collided)
                + metric_cfg.w_offroad * float(not final_on_road)
            ),
        )

    speeds = np.array([d.speed for d in diags], dtype=np.float64)
    accels = np.array([d.accel_phys for d in diags], dtype=np.float64)
    steers = np.array([d.steer_phys for d in diags], dtype=np.float64)
    risks  = np.array([d.risk_val for d in diags], dtype=np.float64)
    dmins  = np.array([d.dmin for d in diags], dtype=np.float64)
    ttcs   = np.array([d.min_ttc_s for d in diags], dtype=np.float64)

    lat_pos = np.array([d.lat_y for d in diags], dtype=np.float64)
    lane_y  = np.array([d.lane_y for d in diags], dtype=np.float64)
    lke     = np.abs(lat_pos - lane_y)
    lke     = lke[~np.isnan(lke)]
    lane_violation = np.array([d.lane_violation_m for d in diags], dtype=np.float64)
    on_road = np.array([d.on_road for d in diags], dtype=np.bool_)
    lane_violation = np.where(on_road, lane_violation, np.maximum(lane_violation, LANE_WIDTH))
    on_road_with_final = np.concatenate(
        [on_road, np.array([final_on_road], dtype=np.bool_)]
    )
    offroad_steps = int((~on_road_with_final).sum())
    went_offroad = bool(offroad_steps > 0)
    elapsed_s = len(diags) / POLICY_FREQ
    first_offroad = np.flatnonzero(~on_road)
    offroad_fail_time = (
        float(first_offroad[0] / POLICY_FREQ)
        if first_offroad.size
        else (elapsed_s if not final_on_road else float("inf"))
    )
    collision_fail_time = elapsed_s if collided else float("inf")
    failure_time_s = min(offroad_fail_time, collision_fail_time)
    if not np.isfinite(failure_time_s):
        failure_time_s = elapsed_s

    # Lateral accel proxy: v² · steer / L  (small-angle bicycle)
    lat_accel = (speeds ** 2) * steers / _VEHICLE_WHEELBASE   # signed, m/s²
    abs_lat_accel = np.abs(lat_accel)

    # CVaR (worst 5%) of per-step risk_val. Same definition as training-time CVaR.
    if risks.size >= 20:
        risks_sorted = np.sort(risks)[::-1]
        n_tail = max(1, int(0.05 * risks.size))
        cvar_step = float(np.mean(risks_sorted[:n_tail]))
    else:
        cvar_step = float(risks.max()) if risks.size else 0.0

    F_norms   = np.array([c["F_norm"]   for c in coeff_diags], dtype=np.float64)
    lam_softs = np.array([c["lam_soft"] for c in coeff_diags], dtype=np.float64)
    lam_hards = np.array([c["lam_hard"] for c in coeff_diags], dtype=np.float64)
    clearance_violation = np.maximum(0.0, metric_cfg.d_safe - dmins)
    finite_ttc = ttcs[np.isfinite(ttcs)]
    ttc_violation = np.maximum(0.0, metric_cfg.ttc_safe_s - finite_ttc)
    control = np.stack([accels, steers], axis=-1) if accels.size else np.zeros((0, 2))
    jerk_energy = 0.0
    if control.shape[0] >= 2:
        d_control = np.diff(control, axis=0)
        jerk_energy = float(np.sum(np.sum(d_control ** 2, axis=-1)) / POLICY_FREQ)
    progress_m = float(o_end[0] - o_start[0])
    progress_rate = progress_m / max(elapsed_s, EPS)
    speed_deficit = float(np.sum(np.maximum(0.0, metric_cfg.v_ref - speeds)) / POLICY_FREQ)
    lane_violation_cum = float(np.sum(lane_violation) / POLICY_FREQ)
    clearance_violation_cum = float(np.sum(clearance_violation) / POLICY_FREQ)
    ttc_violation_cum = float(np.sum(ttc_violation) / POLICY_FREQ)
    accel_energy = float(np.sum(accels ** 2) / POLICY_FREQ)
    steering_energy = float(np.sum(steers ** 2) / POLICY_FREQ)
    min_clearance = float(dmins.min()) if dmins.size else float("inf")
    min_ttc = float(finite_ttc.min()) if finite_ttc.size else float("inf")
    risky_ttc_mask = np.isfinite(ttcs) & (ttcs < metric_cfg.ttc_safe_s)
    risky_clearance_mask = dmins < metric_cfg.d_safe
    dynamic_mask = risky_ttc_mask | risky_clearance_mask
    dynamic_indices = np.flatnonzero(dynamic_mask)
    dynamic_risk_event = bool(dynamic_indices.size > 0)
    response_delay_s = float("nan")
    intervention_window_proxy_s = float("nan")
    post_risk_min_ttc_s = float("nan")
    post_risk_clearance_violation_cum = float("nan")
    if dynamic_risk_event:
        onset_idx = int(dynamic_indices[0])
        response_mask = (
            (accels[onset_idx:] < -0.5)
            | (np.abs(steers[onset_idx:]) > 0.02)
        )
        response_indices = np.flatnonzero(response_mask)
        response_idx = onset_idx + int(response_indices[0]) if response_indices.size else None
        if response_idx is not None:
            response_delay_s = float((response_idx - onset_idx) / POLICY_FREQ)
            if np.isfinite(ttcs[response_idx]):
                intervention_window_proxy_s = float(ttcs[response_idx])
        post_ttc = ttcs[onset_idx:]
        post_ttc = post_ttc[np.isfinite(post_ttc)]
        if post_ttc.size:
            post_risk_min_ttc_s = float(post_ttc.min())
        post_risk_clearance_violation_cum = float(
            np.sum(clearance_violation[onset_idx:]) / POLICY_FREQ
        )
    failure_score = (
        metric_cfg.w_collision * float(collided)
        + metric_cfg.w_offroad * float(went_offroad)
        + metric_cfg.w_lane * lane_violation_cum
        + metric_cfg.w_clearance * clearance_violation_cum
        + metric_cfg.w_ttc * ttc_violation_cum
        + metric_cfg.w_progress * speed_deficit
        + metric_cfg.w_control * jerk_energy
    )

    return EpisodeMetrics(
        seed         = seed,
        scenario     = scenario_name,
        steps        = len(diags),
        collided     = collided,
        truncated    = truncated,
        safe_success  = (not collided and not went_offroad),
        failure_time_s = float(failure_time_s),
        distance_m   = float(np.linalg.norm(o_end - o_start)),
        progress_m    = progress_m,
        progress_rate_mps = float(progress_rate),
        mean_speed   = float(speeds.mean()),
        speed_deficit = speed_deficit,
        stagnation_rate = float(np.mean(speeds < metric_cfg.v_min)),
        speed_std    = float(speeds.std()),
        lateral_accel_mean_abs = float(abs_lat_accel.mean()),
        lateral_accel_p95_abs  = float(np.percentile(abs_lat_accel, 95)) if abs_lat_accel.size else 0.0,
        lateral_pos_std        = float(lat_pos.std()),
        lane_keep_err_mean     = float(lke.mean()) if lke.size else float("nan"),
        lane_keep_err_p95      = float(np.percentile(lke, 95)) if lke.size else float("nan"),
        lane_keep_err_max      = float(lke.max()) if lke.size else float("nan"),
        lane_changes           = lane_changes,
        on_road_fraction       = float(on_road_with_final.mean()) if on_road_with_final.size else 1.0,
        offroad_steps          = offroad_steps,
        went_offroad           = went_offroad,
        ended_offroad          = not final_on_road,
        lane_violation_duration_s = float(np.sum(lane_violation > 0.0) / POLICY_FREQ),
        lane_violation_cum_m_s = lane_violation_cum,
        lane_violation_max_m = float(lane_violation.max()) if lane_violation.size else 0.0,
        cum_risk_eval          = float(risks.sum()),
        cvar_step_risk         = cvar_step,
        min_clearance          = min_clearance,
        mean_clearance         = float(dmins.mean()) if dmins.size else float("inf"),
        clearance_violation_duration_s = float(np.sum(clearance_violation > 0.0) / POLICY_FREQ),
        clearance_violation_cum_m_s = clearance_violation_cum,
        near_miss              = bool((not collided) and min_clearance < metric_cfg.d_near),
        min_ttc_s              = min_ttc,
        ttc_violation_duration_s = float(np.sum(ttc_violation > 0.0) / POLICY_FREQ),
        ttc_violation_cum_s    = ttc_violation_cum,
        critical_ttc_event     = bool(min_ttc < metric_cfg.ttc_crit_s),
        dynamic_risk_event     = dynamic_risk_event,
        response_delay_s       = response_delay_s,
        intervention_window_proxy_s = intervention_window_proxy_s,
        post_risk_min_ttc_s    = post_risk_min_ttc_s,
        post_risk_clearance_violation_cum_m_s = post_risk_clearance_violation_cum,
        evasive_success        = bool(dynamic_risk_event and not collided and not went_offroad),
        lam_soft_mean          = float(lam_softs.mean()),
        lam_hard_mean          = float(lam_hards.mean()),
        F_norm_mean            = float(F_norms.mean()),
        accel_energy           = accel_energy,
        steering_energy        = steering_energy,
        control_jerk_energy    = jerk_energy,
        hard_brake_count       = int(np.sum(accels < metric_cfg.hard_brake_accel)),
        failure_score          = float(failure_score),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation + table reporting
# ─────────────────────────────────────────────────────────────────────────────

def _agg(metrics: List[EpisodeMetrics]) -> Dict[str, Any]:
    if not metrics:
        return {"n": 0}
    n = len(metrics)
    n_crashed = sum(1 for m in metrics if m.collided)
    fields = (
        "safe_success", "failure_time_s",
        "progress_m", "progress_rate_mps", "speed_deficit",
        "stagnation_rate",
        "mean_speed", "speed_std",
        "lateral_accel_mean_abs", "lateral_accel_p95_abs",
        "lateral_pos_std", "lane_keep_err_mean", "lane_keep_err_p95",
        "lane_keep_err_max", "lane_changes", "on_road_fraction",
        "offroad_steps", "lane_violation_duration_s",
        "lane_violation_cum_m_s", "lane_violation_max_m",
        "cum_risk_eval", "cvar_step_risk",
        "min_clearance", "mean_clearance",
        "clearance_violation_duration_s", "clearance_violation_cum_m_s",
        "near_miss", "min_ttc_s", "ttc_violation_duration_s",
        "ttc_violation_cum_s", "critical_ttc_event",
        "dynamic_risk_event", "response_delay_s",
        "intervention_window_proxy_s", "post_risk_min_ttc_s",
        "post_risk_clearance_violation_cum_m_s", "evasive_success",
        "lam_soft_mean", "lam_hard_mean", "F_norm_mean",
        "accel_energy", "steering_energy", "control_jerk_energy",
        "hard_brake_count", "failure_score",
        "distance_m", "steps",
    )
    out = {"n": n, "n_crashed": n_crashed,
           "collision_rate": n_crashed / n,
           "success_rate": 1.0 - n_crashed / n,
           "n_safe_success": sum(1 for m in metrics if m.safe_success),
           "safe_success_rate": sum(1 for m in metrics if m.safe_success) / n,
           "n_went_offroad": sum(1 for m in metrics if m.went_offroad),
           "offroad_rate": sum(1 for m in metrics if m.went_offroad) / n,
           "n_ended_offroad": sum(1 for m in metrics if m.ended_offroad),
           "ended_offroad_rate": sum(1 for m in metrics if m.ended_offroad) / n,
           "near_miss_rate": sum(1 for m in metrics if m.near_miss) / n,
           "critical_ttc_event_rate": sum(1 for m in metrics if m.critical_ttc_event) / n,
           "n_dynamic_risk_event": sum(1 for m in metrics if m.dynamic_risk_event),
           "dynamic_risk_event_rate": sum(1 for m in metrics if m.dynamic_risk_event) / n}
    n_dynamic = out["n_dynamic_risk_event"]
    out["evasive_success_rate"] = (
        sum(1 for m in metrics if m.evasive_success) / n_dynamic
        if n_dynamic else float("nan")
    )
    for f in fields:
        vals = np.array([getattr(m, f) for m in metrics], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        out[f"{f}_mean"]   = float(vals.mean()) if vals.size else float("nan")
        out[f"{f}_std"]    = float(vals.std())  if vals.size else float("nan")
    return out


def _print_summary(label: str, agg: Dict[str, Any]):
    print(f"\n  ── {label} ──")
    print(f"    episodes:        {agg['n']} ({agg['n_crashed']} crashed)")
    print(f"    collision rate:  {agg['collision_rate']:.1%}")
    print(f"    safe success:    {agg['safe_success_rate']:.1%}")
    print(f"    failure time:    {agg['failure_time_s_mean']:.2f}s")
    print(f"    mean speed:      {agg['mean_speed_mean']:5.2f} ± {agg['mean_speed_std']:.2f} m/s")
    print(f"    progress:        {agg['progress_m_mean']:6.1f} m")
    print(f"    speed deficit:   {agg['speed_deficit_mean']:6.1f}")
    print(f"    distance:        {agg['distance_m_mean']:6.1f} m")
    print(f"    lane changes/ep: {agg['lane_changes_mean']:.2f}")
    print(f"    on-road:")
    print(f"      went offroad:  {agg['offroad_rate']:.1%}")
    print(f"      ended offroad: {agg['ended_offroad_rate']:.1%}")
    print(f"      on-road frac:  {agg['on_road_fraction_mean']:.3f}")
    print(f"    lateral activity:")
    print(f"      |a_lat| mean:  {agg['lateral_accel_mean_abs_mean']:.3f} m/s²")
    print(f"      |a_lat| p95:   {agg['lateral_accel_p95_abs_mean']:.3f} m/s²")
    print(f"      lat-pos std:   {agg['lateral_pos_std_mean']:.3f} m")
    print(f"      LKE mean:      {agg['lane_keep_err_mean_mean']:.3f} m")
    print(f"      LKE p95:       {agg['lane_keep_err_p95_mean']:.3f} m")
    print(f"      LKE max:       {agg['lane_keep_err_max_mean']:.3f} m")
    print(f"      lane viol.:    {agg['lane_violation_cum_m_s_mean']:.3f} m*s")
    print(f"    risk:")
    print(f"      cum_risk:      {agg['cum_risk_eval_mean']:7.2f} ± {agg['cum_risk_eval_std']:.2f}")
    print(f"      step CVaR:     {agg['cvar_step_risk_mean']:.4f}")
    print(f"      min clearance: {agg['min_clearance_mean']:.2f} m")
    print(f"      clear viol.:   {agg['clearance_violation_cum_m_s_mean']:.3f} m*s")
    print(f"      min TTC:       {agg['min_ttc_s_mean']:.2f}s")
    print(f"      TTC viol.:     {agg['ttc_violation_cum_s_mean']:.3f}s")
    print(f"      near-miss:     {agg['near_miss_rate']:.1%}")
    print(f"    dynamic risk:")
    print(f"      event rate:    {agg['dynamic_risk_event_rate']:.1%}")
    if not math.isnan(agg.get("evasive_success_rate", float("nan"))):
        print(f"      evasive succ.: {agg['evasive_success_rate']:.1%}")
    print(f"      resp. delay:   {agg['response_delay_s_mean']:.2f}s")
    print(f"      TTC@response:  {agg['intervention_window_proxy_s_mean']:.2f}s")
    print(f"      post TTC min:  {agg['post_risk_min_ttc_s_mean']:.2f}s")
    print(f"    forces:")
    print(f"      |F| mean:      {agg['F_norm_mean_mean']:.2f}")
    print(f"      λ_s mean:      {agg['lam_soft_mean_mean']:5.2f}")
    print(f"      λ_h mean:      {agg['lam_hard_mean_mean']:5.2f}")
    print(f"    score:")
    print(f"      failure score: {agg['failure_score_mean']:.2f}")


def _print_paired_table(agg_s1: Dict[str, Any], agg_s2: Dict[str, Any]):
    """Side-by-side comparison table for paired Stage 1 / Stage 2 eval."""
    rows = [
        ("safe success",          "safe_success_rate",         ".1%",   None),
        ("collision rate",        "collision_rate",            ".1%",   None),
        ("offroad rate",          "offroad_rate",              ".1%",   None),
        ("failure time (s)",      "failure_time_s_mean",       ".2f",  None),
        ("mean speed (m/s)",      "mean_speed_mean",           "5.2f", "mean_speed_std"),
        ("progress (m)",          "progress_m_mean",           "6.1f", None),
        ("speed deficit",         "speed_deficit_mean",        "6.1f", None),
        ("distance (m)",          "distance_m_mean",           "6.1f", None),
        ("lane changes/ep",       "lane_changes_mean",         ".2f",  None),
        ("|a_lat| mean (m/s²)",   "lateral_accel_mean_abs_mean", ".3f", None),
        ("|a_lat| p95 (m/s²)",    "lateral_accel_p95_abs_mean", ".3f", None),
        ("lat-pos std (m)",       "lateral_pos_std_mean",      ".3f",  None),
        ("LKE mean (m)",          "lane_keep_err_mean_mean",   ".3f",  None),
        ("LKE p95 (m)",           "lane_keep_err_p95_mean",    ".3f",  None),
        ("LKE max (m)",           "lane_keep_err_max_mean",    ".3f",  None),
        ("on-road fraction",      "on_road_fraction_mean",     ".3f",  None),
        ("lane viol. (m*s)",      "lane_violation_cum_m_s_mean", ".3f", None),
        ("cum_risk (eval)",       "cum_risk_eval_mean",        "7.2f", "cum_risk_eval_std"),
        ("step CVaR",             "cvar_step_risk_mean",       ".4f",  None),
        ("min clearance (m)",     "min_clearance_mean",        ".2f",  None),
        ("clear viol. (m*s)",     "clearance_violation_cum_m_s_mean", ".3f", None),
        ("min TTC (s)",           "min_ttc_s_mean",            ".2f",  None),
        ("TTC viol. (s)",         "ttc_violation_cum_s_mean",  ".3f", None),
        ("risk event rate",       "dynamic_risk_event_rate",   ".1%", None),
        ("response delay (s)",    "response_delay_s_mean",     ".2f", None),
        ("TTC@response (s)",      "intervention_window_proxy_s_mean", ".2f", None),
        ("post-risk min TTC (s)", "post_risk_min_ttc_s_mean",  ".2f", None),
        ("jerk energy",           "control_jerk_energy_mean",  ".3f", None),
        ("hard brakes/ep",        "hard_brake_count_mean",     ".2f", None),
        ("failure score",         "failure_score_mean",        "7.2f", None),
    ]
    print(f"\n  ── Paired comparison ──")
    print(f"    {'metric':<24} {'Stage 1':>14}   {'Stage 2':>14}   {'Δ':>10}")
    print(f"    {'-'*24} {'-'*14}   {'-'*14}   {'-'*10}")
    for label, key, fmt, std_key in rows:
        v1 = agg_s1.get(key)
        v2 = agg_s2.get(key)
        if v1 is None or v2 is None or (
            isinstance(v1, float) and math.isnan(v1)) or (
            isinstance(v2, float) and math.isnan(v2)):
            continue
        if std_key:
            s1_str = f"{v1:{fmt}} ± {agg_s1.get(std_key, 0):.2f}"
            s2_str = f"{v2:{fmt}} ± {agg_s2.get(std_key, 0):.2f}"
        else:
            s1_str = f"{v1:{fmt}}"
            s2_str = f"{v2:{fmt}}"
        diff = v2 - v1
        diff_str = f"{diff:+.2f}" if abs(diff) >= 0.01 else f"{diff:+.4f}"
        print(f"    {label:<24} {s1_str:>14}   {s2_str:>14}   {diff_str:>10}")


PAPER_METRICS = [
    ("safe_success_rate", "Safe success", "high"),
    ("collision_rate", "Crash", "low"),
    ("offroad_rate", "Off-road", "low"),
    ("lane_violation_cum_m_s_mean", "Lane viol.", "low"),
    ("min_clearance_mean", "Min clear.", "high"),
    ("clearance_violation_cum_m_s_mean", "Clear viol.", "low"),
    ("min_ttc_s_mean", "Min TTC", "high"),
    ("ttc_violation_cum_s_mean", "TTC viol.", "low"),
    ("progress_rate_mps_mean", "Progress rate", "high"),
    ("hard_brake_count_mean", "Hard brakes", "low"),
    ("control_jerk_energy_mean", "Jerk energy", "low"),
    ("failure_score_mean", "Fail score", "low"),
]


DYNAMIC_METRICS = [
    ("dynamic_risk_event_rate", "Risk event", "high"),
    ("evasive_success_rate", "Evasive success", "high"),
    ("response_delay_s_mean", "Response delay", "low"),
    ("intervention_window_proxy_s_mean", "TTC@response", "high"),
    ("post_risk_min_ttc_s_mean", "Post-risk min TTC", "high"),
    ("post_risk_clearance_violation_cum_m_s_mean", "Post-risk clear viol.", "low"),
]


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "---"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(x):
        return "---"
    if math.isinf(x):
        return r"$\infty$" if x > 0 else r"$-\infty$"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _finite_or_nan(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def _write_highway_paper_artifacts(out_dir: Path, out_data: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregates = out_data["aggregates"]
    rows: List[Dict[str, Any]] = []
    for scenario, stages in aggregates.items():
        for stage, agg in stages.items():
            row = {"scenario": scenario, "stage": stage}
            for key, _, _ in PAPER_METRICS:
                row[key] = agg.get(key, float("nan"))
            rows.append(row)

    csv_path = out_dir / "highway_main_table.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["scenario", "stage", *[key for key, _, _ in PAPER_METRICS]]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tex_path = out_dir / "highway_main_table.tex"
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{ll" + "r" * len(PAPER_METRICS) + "}",
        r"\toprule",
        "Scenario & Stage & " + " & ".join(_tex_escape(label) for _, label, _ in PAPER_METRICS) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [_tex_escape(str(row["scenario"])), _tex_escape(str(row["stage"]))]
        cells.extend(_fmt_cell(row[key]) for key, _, _ in PAPER_METRICS)
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Highway-env safety, progress, and control-quality metrics. The aggregate failure score is used only as a ranking summary; components are reported separately.}",
        r"\label{tab:highway_main_metrics}",
        r"\end{table}",
        "",
    ])
    tex_path.write_text("\n".join(tex_lines))

    dynamic_csv_path = out_dir / "highway_dynamic_table.csv"
    with dynamic_csv_path.open("w", newline="") as f:
        fieldnames = ["scenario", "stage", *[key for key, _, _ in DYNAMIC_METRICS]]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario, stages in aggregates.items():
            for stage, agg in stages.items():
                row = {"scenario": scenario, "stage": stage}
                for key, _, _ in DYNAMIC_METRICS:
                    row[key] = agg.get(key, float("nan"))
                writer.writerow(row)

    dynamic_tex_path = out_dir / "highway_dynamic_table.tex"
    dynamic_tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll" + "r" * len(DYNAMIC_METRICS) + "}",
        r"\toprule",
        "Scenario & Stage & " + " & ".join(_tex_escape(label) for _, label, _ in DYNAMIC_METRICS) + r" \\",
        r"\midrule",
    ]
    for scenario, stages in aggregates.items():
        for stage, agg in stages.items():
            cells = [_tex_escape(str(scenario)), _tex_escape(str(stage))]
            cells.extend(_fmt_cell(agg.get(key, float("nan"))) for key, _, _ in DYNAMIC_METRICS)
            dynamic_tex_lines.append(" & ".join(cells) + r" \\")
    dynamic_tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Highway-env dynamic-risk proxy metrics. Risk onset is the first timestep where clearance or TTC enters the unsafe band; the intervention window proxy is TTC at first evasive control response.}",
        r"\label{tab:highway_dynamic_metrics}",
        r"\end{table}",
        "",
    ])
    dynamic_tex_path.write_text("\n".join(dynamic_tex_lines))

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [warn] could not import matplotlib for paper figures: {exc}")
        return

    stage_color = {"stage1": "#1f77b4", "stage2": "#d62728"}
    stage_marker = {"stage1": "o", "stage2": "s"}

    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    for row in rows:
        x = _finite_or_nan(row.get("progress_rate_mps_mean", float("nan")))
        y = _finite_or_nan(row.get("failure_score_mean", float("nan")))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        stage = str(row["stage"])
        label = f"{row['scenario']} / {stage}"
        ax.scatter(x, y, s=55, color=stage_color.get(stage, "#666666"),
                   marker=stage_marker.get(stage, "o"), label=label)
        ax.annotate(str(row["scenario"]), (x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel("Progress rate (m/s)")
    ax.set_ylabel("Highway failure score")
    ax.set_title("Safety-progress Pareto summary")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, loc="best")
    fig.savefig(out_dir / "highway_pareto_progress_failure.png", dpi=220)
    plt.close(fig)

    scenarios = list(aggregates.keys())
    x = np.arange(len(scenarios))
    width = 0.36
    for metric, ylabel, fname in [
        ("ttc_violation_cum_s_mean", "Cumulative TTC violation (s)", "highway_ttc_violation.png"),
        ("clearance_violation_cum_m_s_mean", "Cumulative clearance violation (m*s)", "highway_clearance_violation.png"),
        ("failure_score_mean", "Failure score", "highway_failure_score.png"),
        ("response_delay_s_mean", "Risk response delay (s)", "highway_dynamic_response_delay.png"),
        ("post_risk_min_ttc_s_mean", "Post-risk minimum TTC (s)", "highway_post_risk_ttc.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
        for offset, stage in [(-width / 2, "stage1"), (width / 2, "stage2")]:
            vals = [
                _finite_or_nan(aggregates[scn].get(stage, {}).get(metric, float("nan")))
                for scn in scenarios
            ]
            ax.bar(x + offset, vals, width=width, label=stage,
                   color=stage_color.get(stage, "#666666"), alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by scenario")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(out_dir / fname, dpi=220)
        plt.close(fig)

    print(f"  wrote highway paper artifacts to {out_dir}")


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

    model = CoefEnergyNetMaterial(
        lam_soft_max=lam_soft_max, lam_hard_max=lam_hard_max,
    ).to(device)
    disable_transformer_nested_tensors(model)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing:
        print(f"Missing keys (using init): {missing}")
    if unexpected:
        print(f"Unexpected keys (ignored): {unexpected}")
    model.eval()
    return model, ck


def evaluate_one(
    ckpt_path: Path, *, scenario: ScenarioConfig, episodes: int,
    max_steps: int, base_seed: int, device: str, stage: int,
    metric_cfg: MetricConfig,
    n_max_vehicles: int, dfc_root: str = "",
    d_hat_override: float = 0.0,
    alpha_floor_override: float = -1.0,
    alpha_floor_ahead_only_override: Optional[bool] = None,
    ttc_gain_override: Optional[float] = None,
    ttc_threshold_s_override: Optional[float] = None,
    ttc_softness_s_override: Optional[float] = None,
    ttc_min_closing_speed_override: Optional[float] = None,
    ttc_lane_halfwidth_override: Optional[float] = None,
    ttc_boxed_risk_thresh_override: Optional[float] = None,
    ttc_boxed_gate_sharpness_override: Optional[float] = None,
    disable_mu_lat: bool = False,
    offroad_terminal: bool = False,
) -> List[EpisodeMetrics]:
    model, ck = _load_model(ckpt_path, device, dfc_root)
    runtime = _resolve_runtime_knobs(
        ck,
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
    runtime.disable_mu_lat = bool(disable_mu_lat)
    print(f"    runtime: d_hat={runtime.d_hat:.1f} "
          f"alpha_floor={runtime.alpha_floor:.4f} "
          f"ahead_only={runtime.alpha_floor_ahead_only}")
    if runtime.ttc_gain > 0:
        print(f"    runtime: ttc_gain={runtime.ttc_gain:.2f} "
              f"ttc_threshold={runtime.ttc_threshold_s:.2f}s "
              f"boxed_thresh={runtime.ttc_boxed_risk_thresh:.2f}")
    if runtime.disable_mu_lat:
        print("    runtime: lateral channel disabled (mu_lat ablation)")
    gym = _import_gym()
    env = make_scenario_env(gym, scenario, offroad_terminal=offroad_terminal)
    observer = HighwayMaterialObservation(WrapperConfig(
        n_max_vehicles=n_max_vehicles,
        dt_surrogate=DT_TARGET,
        horizon_surrogate=H_TARGET,
    ))

    results: List[EpisodeMetrics] = []
    try:
        for i in range(episodes):
            seed = base_seed + i
            try:
                m = run_episode(model, observer, env,
                                 seed=seed, scenario_name=scenario.name,
                                 max_steps=max_steps, device=device, stage=stage,
                                 runtime=runtime, metric_cfg=metric_cfg)
            except Exception as exc:
                print(f"    ep {i:3d} seed={seed} FAILED: {exc}")
                continue
            results.append(m)
            tag = "CRASH" if m.collided else ("OFFRD" if m.ended_offroad else "DONE")
            offroad_tag = " offroad" if m.went_offroad else ""
            print(f"    ep {i:3d} seed={seed} {tag:5s} steps={m.steps:3d} "
                  f"v={m.mean_speed:5.2f} lc={m.lane_changes} "
                  f"onroad={m.on_road_fraction:.2f}{offroad_tag} "
                  f"|a_lat|p95={m.lateral_accel_p95_abs:.2f} "
                  f"cum_risk={m.cum_risk_eval:6.1f}")
    finally:
        env.close()
    return results


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--ckpt",          type=str, required=True,
                    help="Stage 2 checkpoint")
    ap.add_argument("--stage1-ckpt",   type=str, default="",
                    help="If set, run paired Stage 1 vs Stage 2 comparison")
    ap.add_argument("--scenarios",     type=str, nargs="+",
                    default=["default"],
                    help=f"One or more from: {list(SCENARIOS.keys())}")
    ap.add_argument("--episodes",      type=int, default=20)
    ap.add_argument("--max-steps",     type=int, default=120)
    ap.add_argument("--seed",          type=int, default=1000)
    ap.add_argument("--n-max-vehicles", type=int, default=15)
    ap.add_argument("--device",        type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dfc-root",      type=str, default="")
    ap.add_argument("--d-hat",         type=float, default=0.0,
                    help="Override IPC activation distance. Default 0 uses "
                         "each checkpoint cfg.")
    ap.add_argument("--alpha-floor",   type=float, default=-1.0,
                    help="Override alpha floor. Default -1 uses each "
                         "checkpoint cfg.")
    ap.add_argument("--alpha-floor-all-obstacles",
                    dest="alpha_floor_ahead_only", action="store_false",
                    help="Apply alpha floor to rear/side vehicles too. "
                         "Default uses each checkpoint cfg.")
    ap.set_defaults(alpha_floor_ahead_only=None)
    ap.add_argument("--disable-mu-lat", action="store_true",
                    help="Disable the Stage 2 lateral channel at eval time. "
                         "This yields a same-checkpoint inference-time "
                         "ablation of mu_lat.")
    ap.add_argument("--ttc-gain", type=float, default=None,
                    help="Enable TTC braking with the given gain. Default "
                         "uses checkpoint cfg or leaves TTC disabled.")
    ap.add_argument("--ttc-threshold-s", type=float, default=None,
                    help="TTC threshold in seconds for braking activation.")
    ap.add_argument("--ttc-softness-s", type=float, default=None,
                    help="Softness of the TTC gate in seconds.")
    ap.add_argument("--ttc-min-closing-speed", type=float, default=None,
                    help="Minimum closing speed before TTC braking activates.")
    ap.add_argument("--ttc-lane-halfwidth", type=float, default=None,
                    help="Half-width of the forward same-lane corridor in metres.")
    ap.add_argument("--ttc-boxed-risk-thresh", type=float, default=None,
                    help="Probe-risk threshold for the boxed-context gate.")
    ap.add_argument("--ttc-boxed-gate-sharpness", type=float, default=None,
                    help="Sharpness of the boxed-context sigmoid gate.")
    ap.add_argument("--offroad-terminal", action="store_true",
                    help="End episodes when ego leaves the road. Useful for "
                         "authored stress tests where offroad escapes must "
                         "count as failures.")
    ap.add_argument("--d-safe", type=float, default=8.0,
                    help="Clearance threshold in metres for proximity violation.")
    ap.add_argument("--d-near", type=float, default=4.0,
                    help="Near-miss clearance threshold in metres.")
    ap.add_argument("--ttc-safe-s", type=float, default=3.0,
                    help="TTC threshold in seconds for cumulative TTC violation.")
    ap.add_argument("--ttc-crit-s", type=float, default=1.5,
                    help="Critical TTC event threshold in seconds.")
    ap.add_argument("--v-ref", type=float, default=25.0,
                    help="Reference speed for progress/conservatism deficit.")
    ap.add_argument("--v-min", type=float, default=1.0,
                    help="Stagnation threshold in m/s.")
    ap.add_argument("--hard-brake-accel", type=float, default=-4.0,
                    help="Acceleration threshold for hard-brake count.")
    ap.add_argument("--w-collision", type=float, default=1000.0)
    ap.add_argument("--w-offroad", type=float, default=500.0)
    ap.add_argument("--w-lane", type=float, default=50.0)
    ap.add_argument("--w-clearance", type=float, default=20.0)
    ap.add_argument("--w-ttc", type=float, default=25.0)
    ap.add_argument("--w-progress", type=float, default=0.5)
    ap.add_argument("--w-control", type=float, default=0.5)
    ap.add_argument("--paper-out-dir", type=str, default="",
                    help="Optional directory for CSV/TeX tables and highway paper figures.")
    ap.add_argument("--out",           type=str, default="")
    args = ap.parse_args()
    metric_cfg = MetricConfig(
        d_safe=args.d_safe,
        d_near=args.d_near,
        ttc_safe_s=args.ttc_safe_s,
        ttc_crit_s=args.ttc_crit_s,
        v_ref=args.v_ref,
        v_min=args.v_min,
        hard_brake_accel=args.hard_brake_accel,
        w_collision=args.w_collision,
        w_offroad=args.w_offroad,
        w_lane=args.w_lane,
        w_clearance=args.w_clearance,
        w_ttc=args.w_ttc,
        w_progress=args.w_progress,
        w_control=args.w_control,
    )

    # Validate scenarios
    for s in args.scenarios:
        if s not in SCENARIOS:
            raise ValueError(f"Unknown scenario {s!r}. "
                              f"Available: {list(SCENARIOS.keys())}")

    paired = bool(args.stage1_ckpt)
    print(f"Stage 2 checkpoint: {args.ckpt}")
    if paired:
        print(f"Stage 1 checkpoint: {args.stage1_ckpt} (paired comparison)")
    print(f"Scenarios: {args.scenarios}")
    print(f"Episodes per scenario: {args.episodes} "
           f"(seeds {args.seed}..{args.seed + args.episodes - 1})\n")

    all_results: Dict[str, Dict[str, List[EpisodeMetrics]]] = {}
    t0 = time.time()

    for scn_name in args.scenarios:
        scn = SCENARIOS[scn_name]
        print(f"━━━ Scenario: {scn.name} ━━━")
        print(f"    {scn.expected_behavior}")
        print(f"    vehicles_count={scn.vehicles_count} lanes={scn.lanes_count} "
               f"ego_spacing={scn.ego_spacing}")

        scn_bucket: Dict[str, List[EpisodeMetrics]] = {}

        if paired:
            print(f"\n  Stage 1...")
            scn_bucket["stage1"] = evaluate_one(
                Path(args.stage1_ckpt), scenario=scn, episodes=args.episodes,
                max_steps=args.max_steps, base_seed=args.seed,
                device=args.device, stage=1,
                metric_cfg=metric_cfg,
                n_max_vehicles=args.n_max_vehicles, dfc_root=args.dfc_root,
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
                disable_mu_lat=False,
                offroad_terminal=args.offroad_terminal,
            )

        print(f"\n  Stage 2...")
        scn_bucket["stage2"] = evaluate_one(
            Path(args.ckpt), scenario=scn, episodes=args.episodes,
            max_steps=args.max_steps, base_seed=args.seed,
            device=args.device, stage=2,
            metric_cfg=metric_cfg,
            n_max_vehicles=args.n_max_vehicles, dfc_root=args.dfc_root,
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
            disable_mu_lat=args.disable_mu_lat,
            offroad_terminal=args.offroad_terminal,
        )

        all_results[scn_name] = scn_bucket

        # ── Per-scenario summary ────────────────────────────────────────────
        if paired:
            agg_s1 = _agg(scn_bucket["stage1"])
            agg_s2 = _agg(scn_bucket["stage2"])
            _print_summary(f"Stage 1 / {scn.name}", agg_s1)
            _print_summary(f"Stage 2 / {scn.name}", agg_s2)
            _print_paired_table(agg_s1, agg_s2)
        else:
            agg = _agg(scn_bucket["stage2"])
            _print_summary(f"Stage 2 / {scn.name}", agg)

    # ── Across-scenario summary (the context-sensitivity check) ──────────────
    if len(args.scenarios) > 1:
        print(f"\n━━━ Cross-scenario behavior (Stage 2) ━━━")
        print(f"    {'scenario':<15} {'crash%':>8} {'v_mean':>8} {'lc/ep':>7} "
              f"{'offroad%':>9} {'onroad':>7} {'|a_lat|p95':>11} {'cum_risk':>10}")
        print(f"    {'-'*15} {'-'*8} {'-'*8} {'-'*7} {'-'*9} {'-'*7} {'-'*11} {'-'*10}")
        for scn_name in args.scenarios:
            agg = _agg(all_results[scn_name]["stage2"])
            print(f"    {scn_name:<15} "
                  f"{agg['collision_rate']:>7.1%} "
                  f"{agg['mean_speed_mean']:>8.2f} "
                  f"{agg['lane_changes_mean']:>7.2f} "
                  f"{agg['offroad_rate']:>8.1%} "
                  f"{agg['on_road_fraction_mean']:>7.3f} "
                  f"{agg['lateral_accel_p95_abs_mean']:>11.3f} "
                  f"{agg['cum_risk_eval_mean']:>10.2f}")

        # Context-sensitivity verdict: if Stage 2 metrics are nearly identical
        # across scenarios, the spotlight pitch's load-bearing claim fails.
        v_means = [_agg(all_results[s]["stage2"])["mean_speed_mean"]
                    for s in args.scenarios]
        lc_means = [_agg(all_results[s]["stage2"])["lane_changes_mean"]
                     for s in args.scenarios]
        v_range = max(v_means) - min(v_means)
        lc_range = max(lc_means) - min(lc_means)
        print(f"\n    Cross-scenario speed range:        {v_range:.2f} m/s")
        print(f"    Cross-scenario lane-changes range: {lc_range:.2f}")
        if v_range > 3.0 or lc_range > 0.5:
            print(f"    → Stage 2 shows context-sensitive behavior across scenarios.")
        else:
            print(f"    → Stage 2 behavior is similar across scenarios; "
                   f"context-sensitivity claim weak.")

    print(f"\n  wall clock: {time.time() - t0:.1f}s")

    # ── Save + paper artifacts ───────────────────────────────────────────────
    out_data = {
        "config": vars(args),
        "metric_config": asdict(metric_cfg),
        "results_by_scenario": {
            scn: {
                stage_label: [asdict(m) for m in episodes_list]
                for stage_label, episodes_list in stages.items()
            }
            for scn, stages in all_results.items()
        },
        "aggregates": {
            scn: {
                stage_label: _agg(episodes_list)
                for stage_label, episodes_list in stages.items()
            }
            for scn, stages in all_results.items()
        },
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out_data, f, indent=2, default=str)
        print(f"  wrote {args.out}")

    paper_out_dir = None
    if args.paper_out_dir:
        paper_out_dir = Path(args.paper_out_dir)
    elif args.out:
        out_path = Path(args.out)
        paper_out_dir = out_path.parent / f"{out_path.stem}_paper"
    if paper_out_dir is not None:
        _write_highway_paper_artifacts(paper_out_dir, out_data)


if __name__ == "__main__":
    main()
