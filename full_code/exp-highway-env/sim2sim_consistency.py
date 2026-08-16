#!/usr/bin/env python3
"""
sim2sim_consistency.py — Step 5/6 boundary check (handoff §9).

Question: does the H-step training-time integrator output match the live
highway-env trajectory under the same initial conditions and Stage 1 model?

If yes (small drift over H=20 = 2s): Stage 2 training backprops through an
honest force surface, gradients are calibrated to deployment.
If no (large drift): Stage 2 will optimize against a fiction. Investigate
before spending GPU days on Step 6.

Three sources of drift, in order of fixability:
  1. Bicycle dynamics mismatch  (bicycle_step_deploy vs highway-env Vehicle.step)
  2. Coefficient staleness      (model output at t=k vs t=0)
  3. Frozen neighbors           (surrogate has C, R, mask fixed at t=0)

The "H=1 diagnostic" at the end isolates source 1: if step-1 drift is non-trivial
even with same initial state and same first-step force, the bicycle-vs-real-env
match is broken regardless of the integrator's other assumptions.

Decision rule (default H_check=20):
  median terminal drift < 1m AND 95th percentile < 3m   → ✅ honest, proceed
  median terminal drift > 5m                            → ❌ investigate
  in between                                            → marginal, decide

Usage
-----
    python sim2sim_consistency.py \\
        --ckpt checkpoints/highway_stage1_action_dhat10_afloor0015/best.pt \\
        --episodes 20 --h-check 20

    # Stage 2 checkpoint, including the lateral channel
    python sim2sim_consistency.py \\
        --ckpt checkpoints/highway_stage2_mu_lat/best.pt \\
        --stage 2 --episodes 20 --h-check 20

Quick smoke (3 episodes, H=10):
    python sim2sim_consistency.py --ckpt <path> --episodes 3 --h-check 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
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
from bicycle_surrogate import (  # noqa: E402
    ACCEL_RANGE, STEER_RANGE, force_to_action, bicycle_step_deploy,
)
from surrogate_integrator import compute_surrogate_highway_force  # noqa: E402
from eval_stage1 import (  # noqa: E402
    _apply_alpha_floor,
    _physical_to_normalized_action,
    disable_transformer_nested_tensors,
)


# Constants — match data collection / training exactly.
H_TARGET = 20
DT_TARGET = 0.1
POLICY_FREQ = 10
SIM_FREQ = 30


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


def make_env(gym, env_id: str, *, vehicles_count: int, lanes_count: int):
    """ContinuousAction env so we can deploy (accel, steer) for live rollouts."""
    config = {
        "policy_frequency":     POLICY_FREQ,
        "simulation_frequency": SIM_FREQ,
        "vehicles_count":       vehicles_count,
        "lanes_count":          lanes_count,
        "duration":             40,
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral":      True,
            "acceleration_range": list(ACCEL_RANGE),
            "steering_range":     list(STEER_RANGE),
        },
    }
    return gym.make(env_id, config=config)


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
# Model interface (same shape as eval_stage1)
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


def _override_d_hat(batch: Dict[str, torch.Tensor], d_hat: float) -> None:
    if d_hat > 0:
        batch["d_hat"] = torch.full_like(batch["d_hat"], float(d_hat))


# ─────────────────────────────────────────────────────────────────────────────
# The two trajectories
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def surrogate_rollout(
    model, batch: Dict[str, torch.Tensor],
    *,
    H_check: int,
    stage: int = 1,
    disable_mu_lat: bool = False,
    d_hat_override: float = 0.0,
    alpha_floor: float = 0.0,
    alpha_floor_ahead_only: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run the training-time integrator manually for H_check steps.

    Mirrors what `integrate_surrogate_highway` does internally:
        - coefficients computed once from t=0 obs, held fixed
        - C, R_eff, mask, rollout_patch held fixed at t=0
        - per-step force via compute_surrogate_highway_force
        - per-step ego dynamics via bicycle_step_deploy
                  (== bicycle_step_train post-unification)

    Returns (positions: (H_check+1, 2) world-frame, info_dict).
    """
    _override_d_hat(batch, d_hat_override)
    alphas, beta, gamma, lam_soft, lam_hard, mu_lat = _model_coeffs(model, batch)
    alphas = _apply_alpha_floor(
        batch, alphas, alpha_floor, ahead_only=alpha_floor_ahead_only
    )
    if stage == 1:
        lam_soft = torch.zeros_like(lam_soft)
        lam_hard = torch.zeros_like(lam_hard)
        mu_lat = None
    elif disable_mu_lat:
        mu_lat = None

    # Match integrate_surrogate_highway defaults: robot_radius=0, so no extra
    # margin is added beyond the observed vehicle radius.
    R_eff = batch["R"]                                        # (B, N)

    o = batch["o0"].clone()                                    # (1, 2)
    v = batch["v0"].clone()                                    # (1, 2)
    speed   = torch.linalg.norm(v, dim=-1).clamp_min(1e-3)     # (1,)
    heading = torch.atan2(v[:, 1], v[:, 0])                    # (1,)

    o0_fixed       = batch["o0"].clone()
    heading_0_fixed = heading.clone()

    positions = [o.cpu().numpy().squeeze(0).copy()]
    headings  = [float(heading.item())]
    speeds    = [float(speed.item())]
    forces    = []

    for k in range(H_check):
        F_tot, _, _, _ = compute_surrogate_highway_force(
            o             = o,
            heading       = heading,
            speed         = speed,
            o0            = o0_fixed,
            heading_0     = heading_0_fixed,
            goal          = batch["goal"],
            C             = batch["C"],
            R_eff         = R_eff,
            mask          = batch["mask"],
            alphas        = alphas,
            beta          = beta,
            gamma         = gamma,
            lam_soft      = lam_soft,
            lam_hard      = lam_hard,
            mu_lat        = mu_lat,
            rollout_patch = batch["rollout_patch"],
            d_hat         = batch["d_hat"],
        )
        forces.append(F_tot.cpu().numpy().squeeze(0).copy())

        accel, steer = force_to_action(F_tot, heading, speed)
        accel = accel.clamp(*ACCEL_RANGE)
        steer = steer.clamp(*STEER_RANGE)

        o, heading, speed = bicycle_step_deploy(
            o, heading, speed, accel, steer, dt=DT_TARGET,
        )
        positions.append(o.cpu().numpy().squeeze(0).copy())
        headings.append(float(heading.item()))
        speeds.append(float(speed.item()))

    return np.stack(positions, axis=0), {
        "headings": np.asarray(headings),
        "speeds":   np.asarray(speeds),
        "forces":   np.stack(forces, axis=0) if forces else np.zeros((0, 2)),
    }


@torch.no_grad()
def live_rollout(
    model, observer, env, *,
    H_check: int,
    device: str,
    stage: int = 1,
    disable_mu_lat: bool = False,
    d_hat_override: float = 0.0,
    alpha_floor: float = 0.0,
    alpha_floor_ahead_only: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Closed-loop with the actual env. Returns (positions: (H_check+1, 2), info).

    Stops early on collision; positions array still has length H_check+1, with
    NaN-padded tail after the collision step. The info dict reports it.
    """
    uenv = env.unwrapped
    pos0 = np.array(uenv.vehicle.position, dtype=np.float32).copy()
    positions = [pos0]
    crashed = False
    crash_step = -1

    for k in range(H_check):
        obs_np = observer.build(env)
        batch = _to_batch(obs_np, device)
        _override_d_hat(batch, d_hat_override)
        alphas, beta, gamma, lam_soft, lam_hard, mu_lat = _model_coeffs(model, batch)
        alphas = _apply_alpha_floor(
            batch, alphas, alpha_floor, ahead_only=alpha_floor_ahead_only
        )
        if stage == 1:
            lam_soft = torch.zeros_like(lam_soft)
            lam_hard = torch.zeros_like(lam_hard)
            mu_lat = None
        elif disable_mu_lat:
            mu_lat = None

        v0 = batch["v0"]
        speed_0   = torch.linalg.norm(v0, dim=-1).clamp_min(1e-3)
        heading_0 = torch.atan2(v0[:, 1], v0[:, 0])

        F_tot, _, _, _ = compute_surrogate_highway_force(
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
            mu_lat        = mu_lat,
            rollout_patch = batch["rollout_patch"],
            d_hat         = batch["d_hat"],
        )

        accel, steer = force_to_action(F_tot, heading_0, speed_0)
        accel_phys = float(accel.clamp(*ACCEL_RANGE).item())
        steer_phys = float(steer.clamp(*STEER_RANGE).item())
        action = _physical_to_normalized_action(accel_phys, steer_phys)

        _, term, trunc, info = _step(env, action)
        new_pos = np.array(uenv.vehicle.position, dtype=np.float32).copy()
        positions.append(new_pos)

        if term:
            crashed = bool(info.get("crashed", True))
            crash_step = k + 1
            break
        if trunc:
            crash_step = k + 1
            break

    # NaN-pad if we exited early
    while len(positions) < H_check + 1:
        positions.append(np.array([np.nan, np.nan], dtype=np.float32))

    return np.stack(positions, axis=0), {
        "crashed":    crashed,
        "crash_step": crash_step,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode comparison
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    seed:          int
    valid_steps:   int                  # how many steps both trajectories ran
    drift:         np.ndarray           # (H_check+1,) per-step drift, NaN past valid_steps
    crashed_live:  bool
    surrogate_speed_mean: float
    live_speed_mean:      float


def compare_episode(
    model, observer, env, *,
    seed: int,
    warmup: int,
    H_check: int,
    device: str,
    stage: int = 1,
    disable_mu_lat: bool = False,
    d_hat_override: float = 0.0,
    alpha_floor: float = 0.0,
    alpha_floor_ahead_only: bool = True,
) -> Optional[EpisodeResult]:
    """Reset env, advance through warmup, snapshot obs, run both trajectories
    from the snapshot, return per-step drift."""
    _reset(env, seed)

    # Warmup with IDLE-equivalent (zeros). For ContinuousAction, [0, 0] = no
    # accel, no steer, so the ego coasts. Simple and deterministic.
    idle = np.zeros(2, dtype=np.float32)
    for _ in range(warmup):
        _, term, trunc, _ = _step(env, idle)
        if term or trunc:
            return None  # episode died during warmup, skip

    # Snapshot the observation. This is what both rollouts start from.
    obs_np = observer.build(env)
    batch = _to_batch(obs_np, device)

    # Surrogate rollout: pure tensor math, doesn't touch env.
    sur_pos, sur_info = surrogate_rollout(
        model,
        batch,
        H_check=H_check,
        stage=stage,
        disable_mu_lat=disable_mu_lat,
        d_hat_override=d_hat_override,
        alpha_floor=alpha_floor,
        alpha_floor_ahead_only=alpha_floor_ahead_only,
    )

    # Live rollout: uses the env, mutates it. Must run after surrogate.
    live_pos, live_info = live_rollout(
        model,
        observer,
        env,
        H_check=H_check,
        device=device,
        stage=stage,
        disable_mu_lat=disable_mu_lat,
        d_hat_override=d_hat_override,
        alpha_floor=alpha_floor,
        alpha_floor_ahead_only=alpha_floor_ahead_only,
    )

    # Per-step drift, NaN where live trajectory ended early.
    diff = sur_pos - live_pos
    drift = np.linalg.norm(diff, axis=-1)
    valid_steps = (
        live_info["crash_step"] if live_info["crash_step"] > 0
        else H_check
    )

    return EpisodeResult(
        seed                 = seed,
        valid_steps          = valid_steps,
        drift                = drift,
        crashed_live         = live_info["crashed"],
        surrogate_speed_mean = float(np.nanmean(sur_info["speeds"])),
        live_speed_mean      = float(np.nanmean(np.linalg.norm(np.diff(live_pos, axis=0), axis=-1) / DT_TARGET)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate + report
# ─────────────────────────────────────────────────────────────────────────────

def _percentile(arr: np.ndarray, q: float) -> float:
    valid = arr[~np.isnan(arr)]
    return float(np.percentile(valid, q)) if valid.size else float("nan")


def report(results: List[EpisodeResult], H_check: int, *, verbose: bool = False):
    if not results:
        print("No valid episodes — aborting report.")
        return None

    n = len(results)
    n_crash = sum(r.crashed_live for r in results)

    # Drift per step, stacked. Mask invalid (post-crash NaN).
    drift_mat = np.stack([r.drift for r in results], axis=0)  # (N, H_check+1)
    drift_at_step = []
    for k in range(H_check + 1):
        col = drift_mat[:, k]
        col = col[~np.isnan(col)]
        drift_at_step.append({
            "step":   k,
            "n":      int(col.size),
            "median": float(np.median(col)) if col.size else float("nan"),
            "p95":    float(np.percentile(col, 95)) if col.size else float("nan"),
            "max":    float(col.max()) if col.size else float("nan"),
        })

    print(f"\n── sim2sim drift report ──")
    print(f"  episodes: {n} ({n_crash} live crashed during the comparison)")
    print(f"  step | median |    p95 |    max | n")
    print(f"  -----+--------+--------+--------+-----")
    # Print every step for short H, every Nth step for longer H
    stride = max(1, (H_check + 1) // 12)
    for k, d in enumerate(drift_at_step):
        if k % stride == 0 or k == H_check:
            print(f"  {d['step']:>4} | {d['median']:>6.3f} | {d['p95']:>6.3f} | "
                  f"{d['max']:>6.3f} | {d['n']:>3}")

    # Decision verdict at terminal
    term = drift_at_step[-1]
    print(f"\n  Terminal (step {H_check}, t={H_check*DT_TARGET:.1f}s):")
    print(f"    median = {term['median']:.3f} m")
    print(f"    p95    = {term['p95']:.3f} m")
    print(f"    max    = {term['max']:.3f} m")

    if term["median"] < 1.0 and term["p95"] < 3.0:
        verdict = "✅ HONEST  — Stage 2 backprop is calibrated to deployment."
    elif term["median"] > 5.0:
        verdict = "❌ INVESTIGATE — surrogate misrepresents deployment significantly."
    else:
        verdict = "⚠ MARGINAL — Stage 2 may train against an inaccurate surface."
    print(f"\n  Verdict: {verdict}")

    # H=1 diagnostic — drift after a single step. Isolates source 1 (bicycle
    # vs Vehicle.step mismatch) from sources 2 (coeff staleness, doesn't
    # apply at step 1) and 3 (neighbor motion, minimal in 0.1s).
    step1 = drift_at_step[1] if H_check >= 1 else None
    if step1:
        print(f"\n  H=1 diagnostic (bicycle vs highway-env Vehicle.step):")
        print(f"    median = {step1['median']:.3f} m, p95 = {step1['p95']:.3f} m")
        if step1["median"] > 0.05:
            print(f"    ⚠ unexpected step-1 drift suggests bicycle_step_deploy "
                   f"doesn't quite match Vehicle.step; investigate.")
        else:
            print(f"    ✓ bicycle dynamics match the live env at step 1.")

    return {
        "n_episodes":      n,
        "n_crashed_live":  n_crash,
        "drift_per_step":  drift_at_step,
        "terminal":        term,
        "step1":           step1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",          type=str, required=True)
    ap.add_argument("--env",           type=str, default="highway-v0")
    ap.add_argument("--episodes",      type=int, default=20)
    ap.add_argument("--h-check",       type=int, default=H_TARGET,
                    help="Steps to compare. Default = H_TARGET (matches training).")
    ap.add_argument("--warmup",        type=int, default=0,
                    help="IDLE env steps before snapshotting state. Default 0.")
    ap.add_argument("--seed",          type=int, default=1000)
    ap.add_argument("--vehicles-count", type=int, default=50)
    ap.add_argument("--lanes-count",   type=int, default=4)
    ap.add_argument("--n-max-vehicles", type=int, default=15)
    ap.add_argument("--device",        type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dfc-root",      type=str, default="")
    ap.add_argument("--stage",         type=int, choices=[1, 2], default=1,
                    help="Whether to run sim2sim for Stage 1 or Stage 2 "
                         "behavior. Default: 1.")
    ap.add_argument("--disable-mu-lat", action="store_true",
                    help="For Stage 2 checkpoints, disable the lateral "
                         "channel during sim2sim rollout.")
    ap.add_argument("--d-hat",         type=float, default=0.0,
                    help="Override IPC barrier activation distance. Default "
                         "0 uses checkpoint cfg d_hat if present.")
    ap.add_argument("--alpha-floor",   type=float, default=-1.0,
                    help="Override valid-obstacle alpha floor. Default -1 uses "
                         "checkpoint cfg alpha_floor if present.")
    ap.add_argument("--alpha-floor-all-obstacles",
                    dest="alpha_floor_ahead_only", action="store_false",
                    help="Apply alpha floor to rear/side vehicles too. Default "
                         "uses checkpoint cfg if present, else ahead-only.")
    ap.set_defaults(alpha_floor_ahead_only=None)
    ap.add_argument("--out",           type=str, default="",
                    help="Write detailed JSON to this path.")
    ap.add_argument("--plot",          type=str, default="",
                    help="If set, save drift-vs-time plot to this path "
                         "(requires matplotlib).")
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
    print(f"Loaded {ck_path}  (epoch={ck.get('epoch', '?')})")
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

    print(f"\nComparing surrogate vs live for {args.episodes} episodes")
    print(f"  stage   = {args.stage}")
    print(f"  H_check = {args.h_check} steps × dt={DT_TARGET}s = "
          f"{args.h_check*DT_TARGET:.1f}s")
    print(f"  warmup  = {args.warmup} env steps")
    print(f"  seeds   = {args.seed}..{args.seed + args.episodes - 1}\n")

    results: List[EpisodeResult] = []
    t0 = time.time()
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            try:
                r = compare_episode(
                    model, observer, env,
                    seed=seed, warmup=args.warmup, H_check=args.h_check,
                    device=args.device, stage=args.stage,
                    disable_mu_lat=args.disable_mu_lat,
                    d_hat_override=d_hat_eval,
                    alpha_floor=alpha_floor_eval,
                    alpha_floor_ahead_only=alpha_floor_ahead_only,
                )
            except Exception as exc:
                print(f"  ep {i:3d} seed={seed} FAILED: {exc}")
                continue
            if r is None:
                print(f"  ep {i:3d} seed={seed} skipped (died during warmup)")
                continue
            results.append(r)
            terminal_drift = r.drift[-1] if not math.isnan(r.drift[-1]) \
                              else r.drift[r.valid_steps]
            tag = "CRASH" if r.crashed_live else "OK"
            print(f"  ep {i:3d} seed={seed} {tag:5s}  valid={r.valid_steps:3d}  "
                  f"terminal_drift={terminal_drift:6.3f}m  "
                  f"v_sur={r.surrogate_speed_mean:5.2f} v_live={r.live_speed_mean:5.2f}")
    finally:
        env.close()

    summary = report(results, args.h_check, verbose=args.verbose)
    print(f"\n  wall clock: {time.time()-t0:.1f}s")

    # ── Optional outputs ────────────────────────────────────────────────────
    if args.out and summary is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({
                "summary": summary,
                "config":  vars(args),
                "checkpoint": {"path": str(ck_path), "epoch": ck.get("epoch")},
                "episodes": [
                    {"seed": r.seed, "valid_steps": r.valid_steps,
                      "crashed_live": r.crashed_live,
                      "drift": r.drift.tolist(),
                      "v_sur": r.surrogate_speed_mean,
                      "v_live": r.live_speed_mean}
                    for r in results
                ],
            }, f, indent=2)
        print(f"  wrote {args.out}")

    if args.plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  matplotlib not available, skipping plot")
        else:
            drift_mat = np.stack([r.drift for r in results], axis=0)
            steps = np.arange(args.h_check + 1) * DT_TARGET
            median = np.nanmedian(drift_mat, axis=0)
            p95    = np.nanpercentile(drift_mat, 95, axis=0)

            fig, ax = plt.subplots(figsize=(8, 5))
            for r in results:
                ax.plot(steps, r.drift, alpha=0.2, color="steelblue", linewidth=0.8)
            ax.plot(steps, median, color="black", linewidth=2, label="median")
            ax.plot(steps, p95,    color="firebrick", linewidth=2,
                     linestyle="--", label="95th percentile")
            ax.axhline(1.0, color="green", linestyle=":", label="1m target")
            ax.axhline(3.0, color="orange", linestyle=":", label="3m p95 target")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("position drift |sur - live| (m)")
            ax.set_title(f"sim2sim drift, H={args.h_check}, n={len(results)} eps")
            ax.legend()
            ax.grid(alpha=0.3)
            Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.plot, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {args.plot}")


if __name__ == "__main__":
    main()
