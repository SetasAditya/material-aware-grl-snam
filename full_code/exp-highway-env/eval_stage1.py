#!/usr/bin/env python3
"""
eval_stage1.py — Step 5, component 3.

Closed-loop highway-v0 evaluation of a Stage 1 checkpoint.

At every env step:
    1. Build observation dict via HighwayMaterialObservation.
    2. Forward CoefEnergyNetMaterial → (alphas, beta, gamma, _, _).
       Stage 1: lam_soft and lam_hard are forced to zero.
    3. Call compute_surrogate_highway_force at the current ego state to get
       the world-frame force F_tot. (At step 0 of each "plan" we use o=o0
       and heading=heading_0=atan2(v0_y,v0_x) — the helper matches H=1
       rollout exactly per the regression test in test_surrogate_integrator.)
    4. force_to_action(F_tot, heading, speed) → (accel, steer).
    5. Clip to ACCEL_RANGE / STEER_RANGE and pass to env.step.

This is MPC-with-H=1: re-plan every step. Cheap enough to run in real time
and gives Stage 1 the simplest possible deployment story before we turn on
Stage 2's CVaR-driven adaptation.

Metrics reported (per handoff §6):
    • Mean speed (m/s)
    • Collision rate
    • Lane-keep error (mean |y - y_lane_center|)
    • Success rate (1 - collision rate, no off-road for highway-v0)
    • Lane changes / episode (sanity: Stage 1 should be low; Stage 2 high)

Usage
-----
    python eval_stage1.py \\
        --ckpt checkpoints/highway_stage1/best.pt \\
        --episodes 20 --max-steps 200

For a quick smoke run:
    python eval_stage1.py --ckpt checkpoints/highway_stage1/last.pt \\
        --episodes 2 --max-steps 50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LOCAL_HIGHWAY_ENV = HERE / "HighwayEnv"
if LOCAL_HIGHWAY_ENV.exists():
    sys.path.insert(0, str(LOCAL_HIGHWAY_ENV))
sys.path.insert(0, str(HERE))

# DFC tree (parent of exp-highway-env by default; override via --dfc-root)
DEFAULT_DFC_ROOT = HERE.parent
sys.path.insert(0, str(DEFAULT_DFC_ROOT))

from env_wrapper import HighwayMaterialObservation, WrapperConfig, _ego_lane_center_y  # noqa: E402
from bicycle_surrogate import (  # noqa: E402
    ACCEL_RANGE, STEER_RANGE, force_to_action,
)
from surrogate_integrator import compute_surrogate_highway_force  # noqa: E402


def disable_transformer_nested_tensors(model) -> None:
    """Use the dense Transformer path for eval parity with training."""
    fuser = getattr(model, "fuser", None)
    changed = False
    for attr in ("enable_nested_tensor", "use_nested_tensor"):
        if hasattr(fuser, attr):
            setattr(fuser, attr, False)
            changed = True
    if changed:
        print("  Transformer nested tensors: disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Constants — must match data collection so the model's input distribution
# matches what it was trained on.
# ─────────────────────────────────────────────────────────────────────────────

H_TARGET = 20            # surrogate horizon (informational at eval; we use H=1 closed-loop)
DT_TARGET = 0.1
POLICY_FREQ = 10
SIM_FREQ = 30


def _import_gym():
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401
        return gym
    except ImportError as exc:
        raise SystemExit(f"highway_env not importable: {exc}") from exc


def make_env(gym, env_id: str, *, vehicles_count: int, lanes_count: int,
              render_mode: Optional[str] = None):
    """Highway-v0 with ContinuousAction so we can deploy (accel, steer)."""
    config = {
        "policy_frequency":     POLICY_FREQ,
        "simulation_frequency": SIM_FREQ,
        "vehicles_count":       vehicles_count,
        "lanes_count":          lanes_count,
        "duration":             40,
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "acceleration_range": list(ACCEL_RANGE),
            "steering_range":     list(STEER_RANGE),
        },
    }
    return gym.make(env_id, config=config, render_mode=render_mode)


def _reset(env, seed):
    out = env.reset(seed=seed)
    return out if (isinstance(out, tuple) and len(out) == 2) else (out, {})


def _step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, term, trunc, info = out
        return obs, reward, bool(term), bool(trunc), info
    obs, reward, done, info = out
    return obs, reward, bool(done), False, info


def _physical_to_normalized_action(accel: float, steer: float) -> np.ndarray:
    """Convert physical (m/s², rad) command to ContinuousAction's [-1, 1]."""
    a_lo, a_hi = ACCEL_RANGE
    s_lo, s_hi = STEER_RANGE
    accel_n = 2.0 * (accel - a_lo) / (a_hi - a_lo) - 1.0
    steer_n = 2.0 * (steer - s_lo) / (s_hi - s_lo) - 1.0
    return np.array([accel_n, steer_n], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model forward (mirrors train_stage1._model_forward)
# ─────────────────────────────────────────────────────────────────────────────

def _to_batch(obs_np: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    """Numpy obs dict → batched (B=1) torch tensors on device."""
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


def _model_coeffs(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    """Same construction as train_stage1._model_forward."""
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


def _apply_alpha_floor(
    batch: Dict[str, torch.Tensor],
    alphas: torch.Tensor,
    alpha_floor: float,
    ahead_only: bool = True,
) -> torch.Tensor:
    """Keep obstacle geometry active without waking padded/rear obstacle slots."""
    if alpha_floor <= 0 or alphas.numel() == 0:
        return alphas
    floor_mask = batch["mask"]
    if ahead_only:
        vhat = torch.nn.functional.normalize(batch["v0"], dim=-1, eps=1e-6)
        ahead = ((batch["C"] - batch["o0"].unsqueeze(1)) * vhat.unsqueeze(1)).sum(dim=-1) > 0.0
        floor_mask = floor_mask & ahead
    return alphas + float(alpha_floor) * floor_mask.to(alphas.dtype)


def _force_component_diagnostics(
    batch: Dict[str, torch.Tensor],
    alphas: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    heading: torch.Tensor,
    speed: torch.Tensor,
    ipc_grad_clip: float = 100.0,
) -> Dict[str, float]:
    """Small eval-only decomposition for interpreting closed-loop failures."""
    o = batch["o0"]
    rel = batch["C"] - o.unsqueeze(1)
    dist = torch.linalg.norm(rel, dim=-1).clamp_min(1e-6)
    clear = dist - batch["R"]
    clear = clear.masked_fill(~batch["mask"], float("inf"))
    idx = clear.argmin(dim=1)

    B = o.shape[0]
    row = torch.arange(B, device=o.device)
    rel_closest = rel[row, idx]
    clear_closest = clear[row, idx]

    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    fwd = torch.stack([cos_h, sin_h], dim=-1)
    lat = torch.stack([-sin_h, cos_h], dim=-1)

    rel_lon = (rel_closest * fwd).sum(dim=-1)
    rel_lat = (rel_closest * lat).sum(dim=-1)

    F_goal = -beta.unsqueeze(-1) * (o - batch["goal"])
    v_world = speed.unsqueeze(-1) * fwd
    F_damp = -gamma.unsqueeze(-1) * v_world

    # Match compute_surrogate_highway_force exactly: n_hat points from
    # obstacle centre to ego, so an obstacle ahead contributes negative
    # longitudinal force.
    diff = o.unsqueeze(1) - batch["C"]
    n_hat = diff / dist.unsqueeze(-1)
    from surrogate_integrator import ipc_piecewise  # local import: diagnostics only
    d = dist - batch["R"]
    _, dbdd = ipc_piecewise(d, batch["d_hat"].unsqueeze(-1))
    dbdd = dbdd.clamp(-ipc_grad_clip, ipc_grad_clip)
    F_geom_each = (alphas * dbdd).unsqueeze(-1) * n_hat
    F_geom_each = F_geom_each * batch["mask"].unsqueeze(-1).to(F_geom_each.dtype)
    F_geom = F_geom_each.sum(dim=1)
    F_total = F_goal + F_damp + F_geom

    def lon(F: torch.Tensor) -> torch.Tensor:
        return (F * fwd).sum(dim=-1)

    def lat_comp(F: torch.Tensor) -> torch.Tensor:
        return (F * lat).sum(dim=-1)

    return {
        "closest_clear": float(clear_closest.item()),
        "closest_lon": float(rel_lon.item()),
        "closest_lat": float(rel_lat.item()),
        "F_goal_lon": float(lon(F_goal).item()),
        "F_damp_lon": float(lon(F_damp).item()),
        "F_geom_lon": float(lon(F_geom).item()),
        "F_total_lon": float(lon(F_total).item()),
        "F_geom_lat": float(lat_comp(F_geom).item()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-step action computation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_action(
    model, observer, env, *, device: str, stage: int = 1,
    d_hat_override: float = 0.0,
    alpha_floor: float = 0.0,
    alpha_floor_ahead_only: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build obs, run model, return (continuous action, diagnostics)."""
    obs_np = observer.build(env)
    batch = _to_batch(obs_np, device)
    if d_hat_override > 0:
        batch["d_hat"] = torch.full_like(batch["d_hat"], float(d_hat_override))

    alphas, beta, gamma, lam_soft, lam_hard, _mu_lat = _model_coeffs(model, batch)
    alphas = _apply_alpha_floor(
        batch, alphas, alpha_floor, ahead_only=alpha_floor_ahead_only
    )
    if stage == 1:
        lam_soft = torch.zeros_like(lam_soft)
        lam_hard = torch.zeros_like(lam_hard)

    # Decompose v0 into (heading_0, speed_0). At closed-loop step 0 of every
    # plan, current state == observation-time state, so o = o0.
    v0 = batch["v0"]                                                # (1, 2)
    speed_0   = torch.linalg.norm(v0, dim=-1).clamp_min(1e-3)       # (1,)
    heading_0 = torch.atan2(v0[:, 1], v0[:, 0])                     # (1,)

    # ─── First-step force via the helper that the integrator uses internally.
    # Matches H=1 rollout per test_surrogate_integrator.py.
    F_tot, dmin, risk_val, sdf_val = compute_surrogate_highway_force(
        o             = batch["o0"],
        heading       = heading_0,
        speed         = speed_0,
        o0            = batch["o0"],
        heading_0     = heading_0,
        goal          = batch["goal"],
        C             = batch["C"],
        R_eff         = batch["R"],
        mask          = batch["mask"],
        alphas        = alphas,
        beta          = beta,
        gamma         = gamma,
        lam_soft      = lam_soft,
        lam_hard      = lam_hard,
        rollout_patch = batch["rollout_patch"],
        d_hat         = batch["d_hat"],
    )

    accel, steer = force_to_action(F_tot, heading_0, speed_0)
    accel_raw = float(accel.item())
    steer_raw = float(steer.item())
    accel = accel.clamp(*ACCEL_RANGE).item()
    steer = steer.clamp(*STEER_RANGE).item()
    action = _physical_to_normalized_action(accel, steer)

    valid = batch["mask"]
    if alphas.numel() > 0 and bool(valid.any().item()):
        alpha_valid = alphas[valid]
        alpha_max = float(alpha_valid.max().item())
        alpha_mean = float(alpha_valid.mean().item())
    else:
        alpha_max = 0.0
        alpha_mean = 0.0

    diag = {
        "speed":   float(speed_0.item()),
        "heading": float(heading_0.item()),
        "alpha_max": alpha_max,
        "alpha_mean": alpha_mean,
        "alpha_floor": float(alpha_floor),
        "alpha_floor_ahead_only": float(alpha_floor_ahead_only),
        "beta":    float(beta.item()),
        "gamma":   float(gamma.item()),
        "F_norm":  float(F_tot.norm(dim=-1).item()),
        "dmin":    float(dmin.item()),
        "d_hat":   float(batch["d_hat"].item()),
        "risk":    float(risk_val.item()),
        "sdf":     float(sdf_val.item()),
        "accel_raw": accel_raw,
        "steer_raw": steer_raw,
        "accel":   accel,
        "steer":   steer,
        "accel_norm": float(action[0]),
        "steer_norm": float(action[1]),
    }
    diag.update(_force_component_diagnostics(
        batch=batch,
        alphas=alphas,
        beta=beta,
        gamma=gamma,
        heading=heading_0,
        speed=speed_0,
    ))
    return action, diag


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode rollout
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeMetrics:
    seed:           int
    steps:          int
    collided:       bool
    truncated:      bool
    distance_m:     float
    mean_speed:     float
    lane_keep_err:  float        # mean |y - lane_center_y|
    lane_changes:   int
    diag_means:     Dict[str, float]


def _detect_lane_change(prev_idx, cur_idx) -> bool:
    if prev_idx is None or cur_idx is None:
        return False
    # lane_index is a tuple (from, to, lane_id); the third element is what
    # changes on a lane change within the same road segment.
    try:
        return prev_idx[2] != cur_idx[2]
    except (IndexError, TypeError):
        return prev_idx != cur_idx


def run_episode(
    env, observer, model, *, seed: int, max_steps: int,
    device: str, stage: int = 1, verbose: bool = False,
    d_hat_override: float = 0.0,
    alpha_floor: float = 0.0,
    alpha_floor_ahead_only: bool = True,
) -> EpisodeMetrics:
    _reset(env, seed)
    uenv = env.unwrapped

    o_start = np.array(uenv.vehicle.position, dtype=np.float64).copy()
    speeds, lke, diag_accum = [], [], []
    lane_changes = 0
    prev_lane = getattr(uenv.vehicle, "lane_index", None)

    collided = False
    truncated = False
    step_idx = 0
    for step_idx in range(max_steps):
        action, diag = compute_action(
            model, observer, env, device=device, stage=stage,
            d_hat_override=d_hat_override,
            alpha_floor=alpha_floor,
            alpha_floor_ahead_only=alpha_floor_ahead_only,
        )

        # Lane-keep error before stepping (so y_pos is at the obs we acted on)
        ego_pos = np.array(uenv.vehicle.position, dtype=np.float64)
        try:
            lane_y = _ego_lane_center_y(env, ego_pos)
            lke.append(abs(float(ego_pos[1]) - float(lane_y)))
        except Exception:
            pass
        speeds.append(float(uenv.vehicle.speed))
        diag_accum.append(diag)

        _, _, term, trunc, info = _step(env, action)

        # Lane change detection
        cur_lane = getattr(uenv.vehicle, "lane_index", None)
        if _detect_lane_change(prev_lane, cur_lane):
            lane_changes += 1
        prev_lane = cur_lane

        if verbose and step_idx % 20 == 0:
            print(f"    step {step_idx:03d}  "
                  f"v={diag['speed']:5.2f}  F={diag['F_norm']:6.2f}  "
                  f"|F|→accel={diag['accel']:+5.2f} steer={diag['steer']:+5.3f}  "
                  f"dmin={diag['dmin']:6.2f}  "
                  f"a={diag['alpha_max']:.2f} β={diag['beta']:.3f} γ={diag['gamma']:.3f}")
        if verbose and diag["dmin"] < 8.0 and step_idx % 5 == 0:
            print(f"      close  dmin={diag['dmin']:5.2f}/{diag['d_hat']:.1f}  "
                  f"accel_raw={diag['accel_raw']:+6.2f} steer_raw={diag['steer_raw']:+6.3f}  "
                  f"u=({diag['accel_norm']:+5.2f},{diag['steer_norm']:+5.2f})  "
                  f"rel=({diag['closest_lon']:+5.1f},{diag['closest_lat']:+4.1f})  "
                  f"Flon[g/d/geom]=({diag['F_goal_lon']:+5.2f},"
                  f"{diag['F_damp_lon']:+5.2f},{diag['F_geom_lon']:+5.2f})  "
                  f"sum={diag['F_total_lon']:+5.2f}")

        if term:
            collided = bool(info.get("crashed", True))  # highway-env sets info["crashed"]
            break
        if trunc:
            truncated = True
            break

    o_end = np.array(uenv.vehicle.position, dtype=np.float64).copy()
    distance = float(np.linalg.norm(o_end - o_start))

    diag_means = {}
    if diag_accum:
        keys = diag_accum[0].keys()
        diag_means = {k: float(np.mean([d[k] for d in diag_accum])) for k in keys}

    return EpisodeMetrics(
        seed          = seed,
        steps         = step_idx + 1,
        collided      = collided,
        truncated     = truncated,
        distance_m    = distance,
        mean_speed    = float(np.mean(speeds)) if speeds else 0.0,
        lane_keep_err = float(np.mean(lke)) if lke else float("nan"),
        lane_changes  = lane_changes,
        diag_means    = diag_means,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",          type=str, required=True)
    ap.add_argument("--env",           type=str, default="highway-v0")
    ap.add_argument("--episodes",      type=int, default=20)
    ap.add_argument("--max-steps",     type=int, default=200)
    ap.add_argument("--seed",          type=int, default=1000,
                    help="Eval seeds offset from training (default 1000+).")
    ap.add_argument("--vehicles-count", type=int, default=50)
    ap.add_argument("--lanes-count",   type=int, default=4)
    ap.add_argument("--n-max-vehicles", type=int, default=15)
    ap.add_argument("--device",        type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dfc-root",      type=str, default="")
    ap.add_argument("--out",           type=str, default="",
                    help="Optional path to dump metrics JSON.")
    ap.add_argument("--d-hat",         type=float, default=0.0,
                    help="Override IPC barrier activation distance. Default "
                         "0 uses checkpoint cfg d_hat if present, else obs default.")
    ap.add_argument("--alpha-floor",   type=float, default=-1.0,
                    help="Override valid-obstacle alpha floor. Default -1 uses "
                         "checkpoint cfg alpha_floor if present.")
    ap.add_argument("--alpha-floor-all-obstacles",
                    dest="alpha_floor_ahead_only", action="store_false",
                    help="Apply alpha floor to rear/side vehicles too. Default "
                         "uses checkpoint cfg if present, else ahead-only.")
    ap.set_defaults(alpha_floor_ahead_only=None)
    ap.add_argument("--verbose",       action="store_true")
    args = ap.parse_args()

    if args.dfc_root:
        sys.path.insert(0, args.dfc_root)
    from train_material import CoefEnergyNetMaterial  # noqa: E402

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ck_path = Path(args.ckpt)
    if not ck_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ck_path}")
    ck = torch.load(ck_path, map_location=args.device, weights_only=False)
    cfg_train = ck.get("cfg", {})
    lam_soft_max = cfg_train.get("lam_soft_max", 50.0)
    lam_hard_max = cfg_train.get("lam_hard_max", 10.0)
    d_hat_eval = args.d_hat if args.d_hat > 0 else float(cfg_train.get("d_hat", 0.0))
    alpha_floor_eval = (args.alpha_floor if args.alpha_floor >= 0
                        else float(cfg_train.get("alpha_floor", 0.0)))
    alpha_floor_ahead_only = (
        bool(args.alpha_floor_ahead_only)
        if args.alpha_floor_ahead_only is not None
        else bool(cfg_train.get("alpha_floor_ahead_only", False))
    )

    model = CoefEnergyNetMaterial(
        lam_soft_max=lam_soft_max, lam_hard_max=lam_hard_max,
    ).to(args.device)
    disable_transformer_nested_tensors(model)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing:
        print(f"Missing keys (using init): {missing}")
    if unexpected:
        print(f"Unexpected keys (ignored): {unexpected}")
    model.eval()
    print(f"Loaded {ck_path}  (epoch={ck.get('epoch', '?')} "
          f"lam_soft_max={lam_soft_max})")
    if d_hat_eval > 0:
        print(f"IPC d_hat override: {d_hat_eval:.1f} m")
    if alpha_floor_eval > 0:
        scope = "vehicles ahead of ego" if alpha_floor_ahead_only else "all valid obstacles"
        print(f"Alpha floor: {alpha_floor_eval:.4f} on {scope}")

    # ── Env + observer ───────────────────────────────────────────────────────
    gym = _import_gym()
    env = make_env(gym, args.env,
                    vehicles_count=args.vehicles_count,
                    lanes_count=args.lanes_count)
    cfg_obs = WrapperConfig(
        n_max_vehicles=args.n_max_vehicles,
        dt_surrogate=DT_TARGET,
        horizon_surrogate=H_TARGET,
    )
    observer = HighwayMaterialObservation(cfg_obs)

    # ── Roll episodes ────────────────────────────────────────────────────────
    print(f"\nEvaluating {args.episodes} episodes on {args.env} "
          f"(max_steps={args.max_steps}, device={args.device})")
    print(f"Seeds: {args.seed} … {args.seed + args.episodes - 1}\n")

    results: List[EpisodeMetrics] = []
    t0 = time.time()
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            try:
                m = run_episode(env, observer, model,
                                 seed=seed, max_steps=args.max_steps,
                                 device=args.device, stage=1,
                                 verbose=args.verbose,
                                 d_hat_override=d_hat_eval,
                                 alpha_floor=alpha_floor_eval,
                                 alpha_floor_ahead_only=alpha_floor_ahead_only)
            except Exception as exc:
                print(f"  ep {i:3d} seed={seed} FAILED: {exc}")
                continue
            results.append(m)
            outcome = ("CRASH" if m.collided
                        else "TRUNC" if m.truncated
                        else "DONE")
            print(f"  ep {i:3d} seed={seed} {outcome:5s}  "
                  f"steps={m.steps:3d}  dist={m.distance_m:6.1f}m  "
                  f"v_mean={m.mean_speed:5.2f}  lke={m.lane_keep_err:.2f}  "
                  f"lc={m.lane_changes}")
    finally:
        env.close()

    # ── Aggregate ────────────────────────────────────────────────────────────
    if not results:
        print("No successful episodes — aborting summary.")
        return

    n = len(results)
    n_crashed = sum(1 for r in results if r.collided)
    summary = {
        "n_episodes":      n,
        "n_crashed":       n_crashed,
        "collision_rate":  n_crashed / n,
        "success_rate":    1.0 - n_crashed / n,
        "mean_distance_m": float(np.mean([r.distance_m for r in results])),
        "mean_speed":      float(np.mean([r.mean_speed for r in results])),
        "lane_keep_err":   float(np.mean([r.lane_keep_err for r in results
                                            if not math.isnan(r.lane_keep_err)])),
        "lane_changes_per_ep": float(np.mean([r.lane_changes for r in results])),
        "mean_steps":      float(np.mean([r.steps for r in results])),
        "wall_clock_s":    time.time() - t0,
    }

    print("\n── Summary ──")
    print(f"  episodes:        {n} ({n_crashed} crashed)")
    print(f"  collision rate:  {summary['collision_rate']:.1%}")
    print(f"  success rate:    {summary['success_rate']:.1%}")
    print(f"  mean distance:   {summary['mean_distance_m']:.1f} m")
    print(f"  mean speed:      {summary['mean_speed']:.2f} m/s")
    print(f"  lane-keep err:   {summary['lane_keep_err']:.3f} m")
    print(f"  lane changes/ep: {summary['lane_changes_per_ep']:.2f}")
    print(f"  wall clock:      {summary['wall_clock_s']:.1f} s "
          f"({summary['wall_clock_s']/n:.2f} s/ep)")

    if args.out:
        out_data = {
            "summary": summary,
            "config": vars(args),
            "checkpoint": {"path": str(ck_path),
                            "epoch": ck.get("epoch")},
            "episodes": [
                {**{k: v for k, v in r.__dict__.items() if k != "diag_means"},
                  "diag_means": r.diag_means}
                for r in results
            ],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
