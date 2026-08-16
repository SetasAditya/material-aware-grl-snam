#!/usr/bin/env python3
"""
collect_onpolicy.py — Step 6, component 2.

Run a checkpointed model closed-loop in highway-env, packaging each step
as an OnPolicySample and appending complete episodes to an OnPolicyBuffer.

Two entry points:

    main()             — CLI driver for standalone collection / inspection.
                          Loads a checkpoint, builds env+observer, collects
                          N episodes, saves the buffer to disk.

    collect_episodes() — function the trainer imports for inline collection
                          at epoch boundaries. Mutates the buffer in place.

Critical convention
-------------------
`action_taken` is stored in the buffer as PHYSICAL units: accel in m/s²,
steer in radians. The env.step call receives a NORMALIZED action in [-1, 1]
because highway-env's ContinuousAction expects that. The two are different;
we record physical because that's the quantity the bicycle dynamics see and
the quantity any future action-loss term will compare against.

Design
------
We do MPC-with-H=1 closed-loop, identical to eval_stage1 — at every step
we re-evaluate the model, compute first-step force via the helper that the
integrator uses internally (`compute_surrogate_highway_force`), decompose
to physical (accel, steer) via `force_to_action`, normalize, and step.

The OnPolicySample we produce represents (S_t, A_t, S_{t+1}):
    obs       = S_t (model input that step)
    action    = A_t (physical, what we recorded)
    o_next    = ego pos at S_{t+1}
    v_next    = ego vel at S_{t+1}

After the episode ends (collision or max_steps), we backfill the
episode-level outcome fields onto every sample and call buf.append_episode.

Usage
-----
    python collect_onpolicy.py \\
        --ckpt checkpoints/highway_stage1_action_dhat10_afloor0015/best.pt \\
        --episodes 50 --stage 1 --out runs/onpolicy_seed_buffer.pt

Smoke (3 short episodes, fast):
    python collect_onpolicy.py --ckpt <path> --episodes 3 --max-steps 30 \\
        --out /tmp/onpolicy_smoke.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
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
    ACCEL_RANGE, STEER_RANGE, force_to_action,
)
from surrogate_integrator import compute_surrogate_highway_force  # noqa: E402
from onpolicy_buffer import OnPolicyBuffer, OnPolicySample  # noqa: E402
from eval_stage1 import _apply_alpha_floor, disable_transformer_nested_tensors  # noqa: E402


# Match data collection / training / eval exactly.
H_TARGET    = 20
DT_TARGET   = 0.1
POLICY_FREQ = 10
SIM_FREQ    = 30


# ─────────────────────────────────────────────────────────────────────────────
# Env construction (mirror eval_stage1 / sim2sim_consistency)
# ─────────────────────────────────────────────────────────────────────────────

def _import_gym():
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401
        return gym
    except ImportError as exc:
        raise SystemExit(f"highway_env not importable: {exc}") from exc


def make_env(gym, env_id: str, *, vehicles_count: int, lanes_count: int,
             offroad_terminal: bool = False):
    """ContinuousAction env. Action input range [-1, 1] mapped to physical."""
    config = {
        "policy_frequency":     POLICY_FREQ,
        "simulation_frequency": SIM_FREQ,
        "vehicles_count":       vehicles_count,
        "lanes_count":          lanes_count,
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
# Action normalization — the convention the user fixed in sim2sim
# ─────────────────────────────────────────────────────────────────────────────

# highway-env ContinuousAction with acceleration_range=(a_min, a_max) maps
# the input action a_norm ∈ [-1, 1] to physical via:
#    a_phys = a_min + (a_norm + 1) / 2 * (a_max - a_min)
# For symmetric ranges, a_phys = a_norm * a_max. So the inverse is:
#    a_norm = a_phys / a_max
# Same for steering: s_norm = s_phys / (π/4).

_ACCEL_HALFRANGE = (ACCEL_RANGE[1] - ACCEL_RANGE[0]) / 2.0   # = 5.0
_STEER_HALFRANGE = (STEER_RANGE[1] - STEER_RANGE[0]) / 2.0   # = π/4


def _normalize_action(accel_phys: float, steer_phys: float) -> np.ndarray:
    """Convert physical (m/s², rad) to highway-env normalized [-1, 1]."""
    accel_norm = float(np.clip(accel_phys / _ACCEL_HALFRANGE, -1.0, 1.0))
    steer_norm = float(np.clip(steer_phys / _STEER_HALFRANGE, -1.0, 1.0))
    return np.array([accel_norm, steer_norm], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Obs / model helpers
# ─────────────────────────────────────────────────────────────────────────────

def _obs_np_to_torch_unbatched(obs_np: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """numpy obs dict → CPU torch tensors, unbatched. Preserves dtypes."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in obs_np.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                out[k] = torch.from_numpy(v.copy()).bool()
            elif v.dtype in (np.int32, np.int64):
                out[k] = torch.from_numpy(v.copy()).long()
            else:
                out[k] = torch.from_numpy(v.astype(np.float32, copy=False))
        elif isinstance(v, (np.floating, float)):
            out[k] = torch.tensor(float(v), dtype=torch.float32)
        elif isinstance(v, (np.integer, int)):
            out[k] = torch.tensor(int(v), dtype=torch.long)
        else:
            out[k] = torch.as_tensor(v)
    return out


def _model_coeffs(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    """Mirrors train_stage1._model_forward / eval_stage1._model_coeffs."""
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


def _override_d_hat_unbatched(obs: Dict[str, torch.Tensor], d_hat: float) -> None:
    if d_hat > 0:
        obs["d_hat"] = torch.tensor(float(d_hat), dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-episode collection
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_one_episode(
    model, observer, env, *,
    max_steps: int,
    device: str,
    stage: int,
    lam_scale: float = 1.0,
    disable_mu_lat: bool = False,
    d_hat_override: float = 0.0,
    alpha_floor: float = 0.0,
    alpha_floor_ahead_only: bool = True,
    ttc_gain: float = 0.0,
    ttc_threshold_s: float = 3.0,
    ttc_softness_s: float = 0.5,
    ttc_min_closing_speed: float = 0.5,
    ttc_lane_halfwidth: float = 2.0,
    ttc_boxed_risk_thresh: float = 0.25,
    ttc_boxed_gate_sharpness: float = 20.0,
) -> Tuple[List[OnPolicySample], Dict[str, Any]]:
    """Run model closed-loop for one episode. Returns (samples, episode_stats).

    `stage`: 1 zeros lam_soft/lam_hard (Stage 1 deployment), 2 uses model
    output (Stage 2 deployment). `lam_scale` mirrors train_stage2's
    curriculum so the on-policy buffer is collected from the same policy
    being optimized in that epoch. Standalone collection defaults to full
    scale (`lam_scale=1.0`).
    """
    samples_pending: List[OnPolicySample] = []
    dmins: List[float] = []
    collided = False

    for t in range(max_steps):
        # ── Observe ─────────────────────────────────────────────────────────
        obs_np = observer.build(env)
        obs_unbatched = _obs_np_to_torch_unbatched(obs_np)
        _override_d_hat_unbatched(obs_unbatched, d_hat_override)
        # Batch (B=1) for model forward
        batch = {k: v.unsqueeze(0).to(device) for k, v in obs_unbatched.items()}

        # ── Model coefficients ──────────────────────────────────────────────
        alphas, beta, gamma, lam_soft, lam_hard, mu_lat = _model_coeffs(model, batch)
        alphas = _apply_alpha_floor(
            batch, alphas, alpha_floor, ahead_only=alpha_floor_ahead_only
        )
        if stage == 1:
            lam_soft = torch.zeros_like(lam_soft)
            lam_hard = torch.zeros_like(lam_hard)
            mu_lat = None
        else:
            lam_soft = lam_soft * float(lam_scale)
            lam_hard = lam_hard * float(lam_scale)
            mu_lat = None if disable_mu_lat else (mu_lat * float(lam_scale))

        # ── Force at current state (= o0 at observation moment) ─────────────
        v0 = batch["v0"]
        speed_0   = torch.linalg.norm(v0, dim=-1).clamp_min(1e-3)
        heading_0 = torch.atan2(v0[:, 1], v0[:, 0])

        F_tot, dmin, _, _ = compute_surrogate_highway_force(
            o             = batch["o0"],
            heading       = heading_0,
            speed         = speed_0,
            o0            = batch["o0"],
            heading_0     = heading_0,
            goal          = batch["goal"],
            C             = batch["C"],
            V_neighbors   = batch.get("V_neighbors"),
            R_eff         = batch["R"],          # match user's sim2sim convention
            mask          = batch["mask"],
            alphas        = alphas,
            beta          = beta,
            gamma         = gamma,
            lam_soft      = lam_soft,
            lam_hard      = lam_hard,
            mu_lat        = mu_lat,
            rollout_patch = batch["rollout_patch"],
            d_hat         = batch["d_hat"],
            ttc_gain      = ttc_gain,
            ttc_threshold_s = ttc_threshold_s,
            ttc_softness_s = ttc_softness_s,
            ttc_min_closing_speed = ttc_min_closing_speed,
            ttc_lane_halfwidth = ttc_lane_halfwidth,
            ttc_boxed_risk_thresh = ttc_boxed_risk_thresh,
            ttc_boxed_gate_sharpness = ttc_boxed_gate_sharpness,
        )
        dmins.append(float(dmin.item()))

        # ── Decompose to PHYSICAL action ────────────────────────────────────
        # action_taken is recorded in physical units (accel m/s², steer rad).
        # env.step receives the normalized version.
        accel, steer = force_to_action(F_tot, heading_0, speed_0)
        accel_phys = float(accel.clamp(*ACCEL_RANGE).item())
        steer_phys = float(steer.clamp(*STEER_RANGE).item())

        # ── Step the env ────────────────────────────────────────────────────
        action_norm = _normalize_action(accel_phys, steer_phys)
        _, term, trunc, info = _step(env, action_norm)

        # Capture next-step ego state
        new_pos = np.array(env.unwrapped.vehicle.position, dtype=np.float32).copy()
        new_vel = np.array(env.unwrapped.vehicle.velocity, dtype=np.float32).copy()

        # ── Build the sample (episode-level fields backfilled later) ────────
        # obs is stored unbatched on CPU (matching IDM dataset convention).
        sample = OnPolicySample(
            obs                   = {k: v.cpu() for k, v in obs_unbatched.items()},
            action                = torch.tensor(
                                        [accel_phys, steer_phys], dtype=torch.float32),
            o_next                = torch.from_numpy(new_pos),
            v_next                = torch.from_numpy(new_vel),
            step_in_episode       = t,
        )
        samples_pending.append(sample)

        if term:
            collided = bool(info.get("crashed", True))
            break
        if trunc:
            break

    # ── Backfill episode-level outcome onto every sample ────────────────────
    deploy_min_clear = min(dmins) if dmins else float("inf")
    deploy_length    = len(samples_pending)
    for s in samples_pending:
        s.deploy_collided       = collided
        s.deploy_min_clearance  = deploy_min_clear
        s.deploy_episode_length = deploy_length

    return samples_pending, {
        "collided":          collided,
        "min_clearance":     deploy_min_clear,
        "length":            deploy_length,
        "mean_dmin":         float(np.mean(dmins)) if dmins else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-episode driver — what train_stage2 imports
# ─────────────────────────────────────────────────────────────────────────────

def collect_episodes(
    model, observer, env, buffer: OnPolicyBuffer, *,
    n_episodes: int, max_steps: int, base_seed: int,
    device: str,
    stage: int,
    lam_scale: float = 1.0,
    disable_mu_lat: bool = False,
    d_hat_override: float = 0.0,
    alpha_floor: float = 0.0,
    alpha_floor_ahead_only: bool = True,
    ttc_gain: float = 0.0,
    ttc_threshold_s: float = 3.0,
    ttc_softness_s: float = 0.5,
    ttc_min_closing_speed: float = 0.5,
    ttc_lane_halfwidth: float = 2.0,
    ttc_boxed_risk_thresh: float = 0.25,
    ttc_boxed_gate_sharpness: float = 20.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Inline collection. Mutates `buffer` in place. Returns aggregate stats.

    For training-time use: call this at epoch boundaries, then
    `buffer.snapshot_dataset()` to start a fresh DataLoader for the new epoch.
    """
    n_attempted = n_episodes
    n_collected = 0
    n_crashed   = 0
    lengths: List[int] = []
    clearances: List[float] = []

    for i in range(n_episodes):
        seed = base_seed + i
        try:
            _reset(env, seed)
            samples, ep_stats = collect_one_episode(
                model, observer, env,
                max_steps=max_steps,
                device=device,
                stage=stage,
                lam_scale=lam_scale,
                disable_mu_lat=disable_mu_lat,
                d_hat_override=d_hat_override,
                alpha_floor=alpha_floor,
                alpha_floor_ahead_only=alpha_floor_ahead_only,
                ttc_gain=ttc_gain,
                ttc_threshold_s=ttc_threshold_s,
                ttc_softness_s=ttc_softness_s,
                ttc_min_closing_speed=ttc_min_closing_speed,
                ttc_lane_halfwidth=ttc_lane_halfwidth,
                ttc_boxed_risk_thresh=ttc_boxed_risk_thresh,
                ttc_boxed_gate_sharpness=ttc_boxed_gate_sharpness,
            )
        except Exception as exc:
            if verbose:
                print(f"  ep {i:3d} seed={seed} FAILED: {exc}")
            continue

        if not samples:
            if verbose:
                print(f"  ep {i:3d} seed={seed} produced 0 samples")
            continue

        ep_id = buffer.append_episode(samples)
        n_collected += 1
        lengths.append(ep_stats["length"])
        if ep_stats["collided"]:
            n_crashed += 1
        if ep_stats["min_clearance"] != float("inf"):
            clearances.append(ep_stats["min_clearance"])

        if verbose and (i % 5 == 0 or i == n_episodes - 1):
            tag = "CRASH" if ep_stats["collided"] else "OK   "
            print(f"  ep {i:3d} seed={seed} ep_id={ep_id:4d} "
                  f"len={ep_stats['length']:3d} {tag}  "
                  f"min_clear={ep_stats['min_clearance']:5.2f}  "
                  f"mean_dmin={ep_stats['mean_dmin']:5.2f}")

    return {
        "n_episodes_attempted":   n_attempted,
        "n_episodes_collected":   n_collected,
        "n_episodes_crashed":     n_crashed,
        "collision_rate":         n_crashed / n_collected if n_collected else 0.0,
        "mean_episode_length":    float(np.mean(lengths)) if lengths else 0.0,
        "mean_min_clearance":     float(np.mean(clearances)) if clearances else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--ckpt",          type=str, required=True)
    ap.add_argument("--out",           type=str, required=True,
                    help="Where to save the buffer (.pt).")
    ap.add_argument("--append-to",     type=str, default="",
                    help="Existing buffer .pt to load and append to "
                         "(otherwise create a new buffer).")
    ap.add_argument("--episodes",      type=int, default=20)
    ap.add_argument("--max-steps",     type=int, default=120)
    ap.add_argument("--seed",          type=int, default=2000,
                    help="Eval seeds start at 1000; use 2000+ for collection "
                         "to avoid distribution overlap with held-out eval.")
    ap.add_argument("--stage",         type=int, default=2, choices=[1, 2],
                    help="1: zero lam_soft/lam_hard. 2: use model outputs.")
    ap.add_argument("--lam-scale",     type=float, default=1.0,
                    help="Scale lam_soft/lam_hard during stage=2 collection. "
                         "Default 1.0 deploys the full Stage 2 policy.")
    ap.add_argument("--capacity",      type=int, default=5000,
                    help="Buffer capacity (only used if --append-to absent).")
    ap.add_argument("--env",           type=str, default="highway-v0")
    ap.add_argument("--vehicles-count", type=int, default=50)
    ap.add_argument("--lanes-count",   type=int, default=4)
    ap.add_argument("--n-max-vehicles", type=int, default=15)
    ap.add_argument("--device",        type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dfc-root",      type=str, default="")
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
    ap.add_argument("--ttc-gain", type=float, default=0.0,
                    help="Enable analytic TTC braking term with this gain. "
                         "Default 0 disables TTC.")
    ap.add_argument("--ttc-threshold-s", type=float, default=3.0,
                    help="TTC horizon in seconds below which the TTC term activates.")
    ap.add_argument("--ttc-softness-s", type=float, default=0.5,
                    help="Smoothness of the TTC activation around the threshold.")
    ap.add_argument("--ttc-min-closing-speed", type=float, default=0.5,
                    help="Minimum positive closing speed before TTC activates.")
    ap.add_argument("--ttc-lane-halfwidth", type=float, default=2.0,
                    help="Half-width of the ego lane corridor considered for TTC.")
    ap.add_argument("--ttc-boxed-risk-thresh", type=float, default=0.25,
                    help="Side-risk threshold for the boxed gate used by TTC.")
    ap.add_argument("--ttc-boxed-gate-sharpness", type=float, default=20.0,
                    help="Sharpness of the boxed gate sigmoid used by TTC.")
    args = ap.parse_args()

    if args.dfc_root:
        sys.path.insert(0, args.dfc_root)
    from train_material import CoefEnergyNetMaterial  # noqa: E402

    # ── Checkpoint ───────────────────────────────────────────────────────────
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
    print(f"Loaded {ck_path} (epoch={ck.get('epoch', '?')}, stage={args.stage})")
    if d_hat_eval > 0:
        print(f"IPC d_hat override: {d_hat_eval:.1f} m")
    if alpha_floor_eval > 0:
        scope = "vehicles ahead of ego" if alpha_floor_ahead_only else "all valid obstacles"
        print(f"Alpha floor: {alpha_floor_eval:.4f} on {scope}")
    if args.ttc_gain > 0:
        print(f"TTC: gain={args.ttc_gain:.3f}  "
              f"threshold={args.ttc_threshold_s:.2f}s  "
              f"boxed_thresh={args.ttc_boxed_risk_thresh:.2f}")

    # ── Buffer (load or create) ──────────────────────────────────────────────
    if args.append_to:
        buffer = OnPolicyBuffer.load(args.append_to)
        print(f"Loaded existing buffer: {buffer}")
    else:
        buffer = OnPolicyBuffer(capacity=args.capacity)
        print(f"Created fresh buffer: capacity={args.capacity}")

    # ── Env + observer ───────────────────────────────────────────────────────
    gym = _import_gym()
    env = make_env(gym, args.env,
                    vehicles_count=args.vehicles_count,
                    lanes_count=args.lanes_count)
    observer = HighwayMaterialObservation(WrapperConfig(
        n_max_vehicles=args.n_max_vehicles,
        dt_surrogate=DT_TARGET,
        horizon_surrogate=H_TARGET,
    ))

    print(f"\nCollecting {args.episodes} episodes from seed {args.seed}...")
    print(f"  env={args.env}  max_steps={args.max_steps}  "
          f"stage={args.stage}  lam_scale={args.lam_scale:.3f}\n")
    t0 = time.time()
    try:
        stats = collect_episodes(
            model, observer, env, buffer,
            n_episodes=args.episodes, max_steps=args.max_steps,
            base_seed=args.seed,
            device=args.device,
            stage=args.stage,
            lam_scale=args.lam_scale,
            d_hat_override=d_hat_eval,
            alpha_floor=alpha_floor_eval,
            alpha_floor_ahead_only=alpha_floor_ahead_only,
            ttc_gain=args.ttc_gain,
            ttc_threshold_s=args.ttc_threshold_s,
            ttc_softness_s=args.ttc_softness_s,
            ttc_min_closing_speed=args.ttc_min_closing_speed,
            ttc_lane_halfwidth=args.ttc_lane_halfwidth,
            ttc_boxed_risk_thresh=args.ttc_boxed_risk_thresh,
            ttc_boxed_gate_sharpness=args.ttc_boxed_gate_sharpness,
            verbose=True,
        )
    finally:
        env.close()

    print(f"\n── Collection summary ──")
    print(f"  attempted:        {stats['n_episodes_attempted']}")
    print(f"  collected:        {stats['n_episodes_collected']}")
    print(f"  crashed:          {stats['n_episodes_crashed']}")
    print(f"  collision rate:   {stats['collision_rate']:.1%}")
    print(f"  mean ep length:   {stats['mean_episode_length']:.1f}")
    print(f"  mean min clear:   {stats['mean_min_clearance']:.2f} m")
    print(f"  wall clock:       {time.time() - t0:.1f}s")
    print(f"  buffer:           {buffer}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    buffer.save(out_path)
    print(f"\nSaved buffer to {out_path}")


if __name__ == "__main__":
    main()
