#!/usr/bin/env python3
"""
train_stage2.py — Step 6, component 3.

Stage 2 risk-aware training. Loads a Stage 1 checkpoint as warm start,
reinitializes and unfreezes the risk heads, and trains with:

    • Mixed batches: IDM imitation stream + on-policy stream
    • Per-sample imitation mask (Stage 2 imit only on IDM samples)
    • λ-curriculum: zero during IDM-only warmup, then linear ramp
    • Inline on-policy collection at start of each epoch (post-warmup)
    • Surrogate val every epoch, closed-loop eval every 10 epochs

Curriculum is applied at the force interface, not at the loss:
    lam_soft_scaled = lam_soft_raw * curriculum_scale
    lam_hard_scaled = lam_hard_raw * curriculum_scale
This means the integrator and the deployed policy gradually become
risk-aware — structurally the right story for the enrichment principle.

Warmup mixing:
    epoch < warmup_epochs:  100% IDM batches (essentially a refinement of Stage 1)
    epoch >= warmup_epochs: 50/50 IDM / on-policy round-robin batches

Usage
-----
    python train_stage2.py \\
        --stage1-ckpt checkpoints/highway_stage1_action_dhat10_afloor0015/best.pt \\
        --idm-data runs/stage1_data \\
        --out checkpoints/highway_stage2 \\
        --epochs 30 --bs 64 --lr 3e-4

Smoke (small everything, fast):
    python train_stage2.py \\
        --stage1-ckpt <path> --idm-data runs/stage1_data_smoke \\
        --epochs 2 --bs 8 --collect-episodes 2 --closed-loop-episodes 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
LOCAL_HIGHWAY_ENV = HERE / "HighwayEnv"
if LOCAL_HIGHWAY_ENV.exists():
    sys.path.insert(0, str(LOCAL_HIGHWAY_ENV))
sys.path.insert(0, str(HERE))

DEFAULT_DFC_ROOT = HERE.parent
sys.path.insert(0, str(DEFAULT_DFC_ROOT))

from surrogate_integrator import (  # noqa: E402
    compute_surrogate_highway_force,
    integrate_surrogate_highway,
)
from bicycle_surrogate import (  # noqa: E402
    ACCEL_RANGE,
    STEER_RANGE,
    VEHICLE_LENGTH,
    bicycle_step_train,
    force_to_action,
)
from episode_costs import HighwayLossWeights, combine_losses  # noqa: E402
from train_stage1 import Stage1IDMDataset  # noqa: E402
from onpolicy_buffer import (  # noqa: E402
    OnPolicyBuffer,
)
from collect_onpolicy import collect_episodes, make_env, _import_gym  # noqa: E402
from env_wrapper import HighwayMaterialObservation, WrapperConfig  # noqa: E402
from eval_stage1 import _apply_alpha_floor, disable_transformer_nested_tensors  # noqa: E402

# Constants — must match all upstream files.
H_TARGET    = 20
DT_TARGET   = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 IDM dataset wrapper — emits matching keys as the on-policy snapshot
# ─────────────────────────────────────────────────────────────────────────────

class Stage2IDMDataset(Dataset):
    """Wraps Stage1IDMDataset to emit the keys collate_mixed expects.

    Adds:
      is_onpolicy=0, has_imit_target=1, has_action_tgt=1, action_taken=zeros,
      o_next/v_next placeholders, episode_id=-1, step_in_episode=-1,
      deploy_collided=0, deploy_min_clearance=inf

    The placeholder action_taken / o_next / v_next are not used by Stage 2's
    loss path (action loss is OFF in Stage 2 by default — handled in the
    trainer if it's ever turned back on, but for the spotlight pitch we want
    CVaR to drive behavior, not a per-step action match).
    """

    def __init__(self, manifest_path: Path, split: str):
        self._inner = Stage1IDMDataset(manifest_path, split=split)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._inner[idx]    # already has 12 obs keys + o_tgt + v_tgt
        if "V_neighbors" not in sample:
            sample["V_neighbors"] = torch.zeros_like(sample["C"])

        # Stream-identifying tags
        sample["is_onpolicy"]     = torch.tensor(0.0)
        sample["has_imit_target"] = torch.tensor(1.0)
        sample["has_action_tgt"]  = torch.tensor(1.0)

        # Action / next-state placeholders. Real values would matter only if
        # a per-step action loss were active; not used in Stage 2.
        sample["action_taken"] = torch.zeros(2, dtype=torch.float32)
        sample["o_next"]       = torch.zeros(2, dtype=torch.float32)
        sample["v_next"]       = torch.zeros(2, dtype=torch.float32)

        # Episode metadata placeholders. -1 distinguishes IDM origin from
        # any real on-policy episode_id (which start at 0).
        sample["episode_id"]           = torch.tensor(-1, dtype=torch.long)
        sample["step_in_episode"]      = torch.tensor(-1, dtype=torch.long)
        sample["deploy_collided"]      = torch.tensor(0.0)
        sample["deploy_min_clearance"] = torch.tensor(float("inf"))
        return sample


def collate_mixed(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack each key. Both Stage2IDMDataset and OnPolicySnapshot emit the
    same key set, so this is just a uniform stack across all keys."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Mixed loader: round-robin between two single-stream loaders
# ─────────────────────────────────────────────────────────────────────────────

class MixedLoader:
    """Round-robin between IDM and on-policy DataLoaders.

    `idm_per_step` and `onpolicy_per_step` are the number of samples drawn
    from each stream per batch. Their sum is the effective batch size.

    Iteration ends when the IDM loader is exhausted (the long stream — we
    pace by it). The on-policy loader cycles indefinitely via repeated
    re-iteration if it's smaller.

    During warmup, set `onpolicy_per_step = 0`: the loader degenerates to
    IDM-only batches.
    """

    def __init__(self, idm_loader: DataLoader,
                  onpolicy_loader: Optional[DataLoader],
                  idm_per_step: int, onpolicy_per_step: int):
        if idm_per_step <= 0:
            raise ValueError(f"idm_per_step must be > 0, got {idm_per_step}")
        if onpolicy_per_step > 0 and onpolicy_loader is None:
            raise ValueError("onpolicy_per_step > 0 but no onpolicy_loader")
        self.idm_loader      = idm_loader
        self.onpolicy_loader = onpolicy_loader
        self.idm_per_step    = idm_per_step
        self.onpolicy_per_step = onpolicy_per_step

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Configure DataLoaders with batch_size set to per-stream draw size.
        # We reach into them and assume their batch_size matches; the trainer
        # constructs them that way.
        idm_iter = iter(self.idm_loader)
        op_iter  = iter(self.onpolicy_loader) if self.onpolicy_per_step > 0 else None

        for idm_batch in idm_iter:
            if self.onpolicy_per_step == 0:
                yield idm_batch
                continue

            try:
                op_batch = next(op_iter)
            except StopIteration:
                op_iter = iter(self.onpolicy_loader)
                op_batch = next(op_iter)

            # Concatenate per-key. Both batches have the same key set (the
            # Stage2IDMDataset wrapper guarantees this).
            mixed = {}
            for k in idm_batch.keys():
                mixed[k] = torch.cat([idm_batch[k], op_batch[k]], dim=0)
            yield mixed

    def __len__(self) -> int:
        return len(self.idm_loader)


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum
# ─────────────────────────────────────────────────────────────────────────────

def curriculum_scale(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    """Risk-force curriculum scale for Stage 2.

    While the data mix is IDM-only (`epoch < warmup_epochs`), λ forces are
    held at zero so those epochs behave like Stage 1 refinement. Once
    on-policy collection starts, λ ramps linearly from a small positive value
    to 1.0 over the remaining epochs.
    """
    if total_epochs <= 0 or warmup_epochs <= 0:
        return 1.0
    if epoch < warmup_epochs:
        return 0.0
    post_epochs = max(1, total_epochs - warmup_epochs)
    post_idx = epoch - warmup_epochs + 1
    return min(1.0, float(post_idx) / float(post_epochs))


def _stage2_dataset_or_fallback(manifest_path: Path, split: str,
                                fallback: Optional[Dataset] = None) -> Dataset:
    try:
        return Stage2IDMDataset(manifest_path, split=split)
    except RuntimeError as exc:
        if fallback is not None and f"No records for split='{split}'" in str(exc):
            print(f"WARNING: no {split} split in manifest; using fallback dataset.")
            return fallback
        raise


def _warmup_epochs(total_epochs: int, warmup_frac: float) -> int:
    if total_epochs <= 1 or warmup_frac <= 0:
        return 0
    if warmup_frac >= 1:
        return max(0, total_epochs - 1)
    return int(warmup_frac * total_epochs)


# ─────────────────────────────────────────────────────────────────────────────
# Model setup: warm start + risk-head reinit + unfreeze
# ─────────────────────────────────────────────────────────────────────────────

def _kaiming_reinit(module: nn.Module) -> None:
    """Reset Linear/Conv weights to fresh random init."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def warm_start_from_stage1(model, stage1_ckpt_path: Path, device: str,
                            *, freeze_geometry: bool = False,
                            disable_mu_lat: bool = False,
                            ) -> Dict[str, Any]:
    """Load Stage 1 weights, then reinit + unfreeze risk-side heads.

    Older Stage 1 checkpoints do not contain ``mu_lat_head`` parameters, so
    loading is intentionally non-strict.
    """
    ck = torch.load(stage1_ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"  Warm-start note: missing={missing}, unexpected={unexpected}")

    # Reinit risk-side modules. Stage 1 trained them at zero-output (frozen),
    # so their weights are uninformative. Fresh random init lets Stage 2
    # discover risk features rather than starting from frozen-zero state.
    _kaiming_reinit(model.risk_enc)
    _kaiming_reinit(model.lam_soft_head)
    _kaiming_reinit(model.lam_hard_head)
    _kaiming_reinit(model.mu_lat_head)
    with torch.no_grad():
        if isinstance(model.mu_lat_head[-1], nn.Linear) and model.mu_lat_head[-1].bias is not None:
            model.mu_lat_head[-1].bias.fill_(-5.0)

    if freeze_geometry:
        # Stage 2 as true energy enrichment: preserve the Stage 1 geometry
        # scaffold, train only the material/risk branch that adds forces.
        for p in model.parameters():
            p.requires_grad = False
        _set_module_trainable(model.risk_enc, True)
        _set_module_trainable(model.lam_soft_head, True)
        _set_module_trainable(model.lam_hard_head, True)
        _set_module_trainable(model.mu_lat_head, not disable_mu_lat)
    else:
        # Original Stage 2 behavior: train all parameters.
        for p in model.parameters():
            p.requires_grad = True
        _set_module_trainable(model.mu_lat_head, not disable_mu_lat)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    trainable_names = [
        name for name, p in model.named_parameters() if p.requires_grad
    ]
    print(f"  Warm-started from {stage1_ckpt_path}")
    print(f"    Stage 1 epoch: {ck.get('epoch', '?')}")
    print(f"    Risk heads reinitialized + unfrozen.")
    if freeze_geometry:
        print("    Geometry scaffold frozen; training risk_enc + λ + mu_lat heads only.")
    if disable_mu_lat:
        print("    mu_lat disabled; training a no-lateral-channel ablation.")
    print(f"    Trainable params: {n_train:,} / {n_total:,}")
    return {
        "stage1_epoch": ck.get("epoch"),
        "n_trainable": n_train,
        "freeze_geometry": bool(freeze_geometry),
        "disable_mu_lat": bool(disable_mu_lat),
        "trainable_param_names": trainable_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Forward + loss for one mixed batch
# ─────────────────────────────────────────────────────────────────────────────

def _model_forward_features(batch: Dict[str, torch.Tensor]
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (obs_feats, goal_feats) for the model. Same as train_stage1."""
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
    return obs_feats, goal_feats


def _straight_through_clip(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    clipped = x.clamp(lo, hi)
    return x + (clipped - x).detach()


def _rollout_with_regularizer_stats(
    *,
    batch: Dict[str, torch.Tensor],
    alphas: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    lam_soft: torch.Tensor,
    lam_hard: torch.Tensor,
    mu_lat: Optional[torch.Tensor],
    lanes_count: int,
    lane_width: float,
    road_margin: float,
    road_tau: float,
    latacc_free: float,
    accel_rate_free: float,
    steer_rate_free: float,
    ttc_gain: float,
    ttc_threshold_s: float,
    ttc_softness_s: float,
    ttc_min_closing_speed: float,
    ttc_lane_halfwidth: float,
    ttc_boxed_risk_thresh: float,
    ttc_boxed_gate_sharpness: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Surrogate rollout plus legality/comfort diagnostics.

    This mirrors ``integrate_surrogate_highway`` but keeps the intermediate
    lateral positions and steering-derived lateral accelerations. It is used
    only when Stage 2 regularizers are enabled; the default trainer path still
    uses the standard integrator.
    """
    o0 = batch["o0"]
    v0 = batch["v0"]
    goal = batch["goal"]
    C = batch["C"]
    V_neighbors = batch.get("V_neighbors")
    R = batch["R"]
    mask = batch["mask"]
    rollout_patch = batch["rollout_patch"]
    d_hat = batch["d_hat"]
    dt = batch["dt"]
    H = batch["H"].long() if batch["H"].dtype != torch.long else batch["H"]

    B = o0.shape[0]
    o = o0.clone()
    speed = torch.linalg.norm(v0, dim=-1).clamp_min(1e-3)
    heading_0 = torch.atan2(v0[:, 1], v0[:, 0])
    heading = heading_0.clone()

    R_eff = R
    min_clear = torch.full((B,), float("inf"), dtype=o.dtype, device=o.device)
    cum_risk = torch.zeros(B, dtype=o.dtype, device=o.device)
    hard_count = torch.zeros(B, dtype=o.dtype, device=o.device)
    arc_length = torch.zeros(B, dtype=o.dtype, device=o.device)
    L_road_acc = torch.zeros((), dtype=o.dtype, device=o.device)
    L_latacc_acc = torch.zeros((), dtype=o.dtype, device=o.device)
    L_control_rate_acc = torch.zeros((), dtype=o.dtype, device=o.device)
    active_count = torch.zeros((), dtype=o.dtype, device=o.device)
    prev_accel: Optional[torch.Tensor] = None
    prev_steer: Optional[torch.Tensor] = None

    if dt.dim() == 0:
        dt_b = dt.expand(B)
    else:
        dt_b = dt
    dt_scalar = float(dt_b[0].item()) if dt_b.numel() > 0 else 0.1
    H_max = int(H.max().item())

    road_y_min = -0.5 * lane_width + road_margin
    road_y_max = (float(lanes_count) - 0.5) * lane_width - road_margin
    tau = max(float(road_tau), 1e-6)

    for s in range(H_max):
        active = (s < H).to(o.dtype)
        active2 = active.unsqueeze(-1)

        F_tot, dmin, risk_val, sdf_val = compute_surrogate_highway_force(
            o=o, heading=heading, speed=speed,
            o0=o0, heading_0=heading_0, goal=goal,
            C=C, V_neighbors=V_neighbors, R_eff=R_eff, mask=mask,
            alphas=alphas, beta=beta, gamma=gamma,
            lam_soft=lam_soft, lam_hard=lam_hard,
            mu_lat=mu_lat,
            rollout_patch=rollout_patch, d_hat=d_hat,
            ttc_gain=ttc_gain,
            ttc_threshold_s=ttc_threshold_s,
            ttc_softness_s=ttc_softness_s,
            ttc_min_closing_speed=ttc_min_closing_speed,
            ttc_lane_halfwidth=ttc_lane_halfwidth,
            ttc_boxed_risk_thresh=ttc_boxed_risk_thresh,
            ttc_boxed_gate_sharpness=ttc_boxed_gate_sharpness,
        )
        min_clear = torch.minimum(min_clear, dmin)

        accel_raw, steer_raw = force_to_action(
            F_tot, heading, speed, length=VEHICLE_LENGTH
        )
        accel = _straight_through_clip(accel_raw, *ACCEL_RANGE)
        steer = _straight_through_clip(steer_raw, *STEER_RANGE)

        # Comfort penalty: allow normal lane-change accelerations for free,
        # penalize only the tail that produces unrealistic lateral escape.
        latacc = speed.pow(2) * torch.tan(steer).abs() / VEHICLE_LENGTH
        latacc_excess = F.relu(latacc - float(latacc_free))
        L_latacc_acc = L_latacc_acc + (active * latacc_excess.pow(2)).sum()

        if prev_accel is not None and prev_steer is not None:
            accel_delta = F.relu((accel - prev_accel).abs() - float(accel_rate_free))
            steer_delta = F.relu((steer - prev_steer).abs() - float(steer_rate_free))
            L_control_rate_acc = L_control_rate_acc + (
                active * (accel_delta.pow(2) + 25.0 * steer_delta.pow(2))
            ).sum()
        prev_accel = accel.detach()
        prev_steer = steer.detach()

        # Road-boundary penalty over the current surrogate state. On a
        # straight highway with lane centers 0, 4, ..., this keeps ego within
        # the outer lane boundaries while still allowing legal lane changes.
        y = o[:, 1]
        below = F.softplus((road_y_min - y) / tau) * tau
        above = F.softplus((y - road_y_max) / tau) * tau
        L_road_acc = L_road_acc + (active * (below.pow(2) + above.pow(2))).sum()

        new_o, new_heading, new_speed = bicycle_step_train(
            o, heading, speed, accel, steer,
            dt=dt_scalar, length=VEHICLE_LENGTH,
        )
        new_o = o + active2 * (new_o - o)
        new_heading = heading + active * (new_heading - heading)
        new_speed = speed + active * (new_speed - speed)

        step_disp = torch.linalg.norm(new_o - o, dim=-1)
        arc_length = arc_length + active * step_disp
        cum_risk = cum_risk + active * risk_val * step_disp
        hard_count = hard_count + active * (sdf_val < 1.0).to(o.dtype)
        active_count = active_count + active.sum()

        o, heading, speed = new_o, new_heading, new_speed

    vT = speed.unsqueeze(-1) * torch.stack(
        [torch.cos(heading), torch.sin(heading)], dim=-1
    )
    denom = active_count.clamp_min(1.0)
    L_road = L_road_acc / denom
    L_latacc = L_latacc_acc / denom
    L_control_rate = L_control_rate_acc / denom
    return (
        o, vT, min_clear, cum_risk, hard_count, arc_length,
        L_road, L_latacc, L_control_rate,
    )


def step_batch_stage2(
    model, batch: Dict[str, torch.Tensor],
    weights: HighwayLossWeights,
    *, train: bool, optimizer=None, grad_clip: float = 5.0,
    curr_scale: float = 1.0,
    d_hat_override: float = 0.0,
    alpha_floor: float = 0.0,
    alpha_floor_ahead_only: bool = True,
    w_road: float = 0.0,
    road_margin: float = 0.25,
    road_tau: float = 0.25,
    road_lane_width: float = 4.0,
    lanes_count: int = 4,
    w_latacc: float = 0.0,
    latacc_free: float = 4.0,
    w_clear_target: float = 0.0,
    clear_target: float = 6.0,
    clear_target_tau: float = 1.0,
    w_control_rate: float = 0.0,
    accel_rate_free: float = 0.5,
    steer_rate_free: float = 0.02,
    disable_mu_lat: bool = False,
    ttc_gain: float = 0.0,
    ttc_threshold_s: float = 3.0,
    ttc_softness_s: float = 0.5,
    ttc_min_closing_speed: float = 0.5,
    ttc_lane_halfwidth: float = 2.0,
    ttc_boxed_risk_thresh: float = 0.25,
    ttc_boxed_gate_sharpness: float = 20.0,
) -> Dict[str, float]:
    """Forward + (optional) backward for a mixed batch.

    The curriculum scale multiplies lam_soft / lam_hard between model output
    and integrator input. This is the structural enrichment story: forces
    appear gradually, not loss weights.
    """
    if d_hat_override > 0:
        batch["d_hat"] = torch.full_like(batch["d_hat"], float(d_hat_override))

    obs_feats, goal_feats = _model_forward_features(batch)
    alphas, beta, gamma, lam_soft_raw, lam_hard_raw, mu_lat_raw = model(
        obs_feats=obs_feats, obs_mask=batch["mask"],
        goal_feats=goal_feats, risk_patch=batch["risk_patch"],
    )
    alphas = _apply_alpha_floor(
        batch, alphas, alpha_floor, ahead_only=alpha_floor_ahead_only
    )

    # Curriculum applied here, at the force interface.
    lam_soft = lam_soft_raw * curr_scale
    lam_hard = lam_hard_raw * curr_scale
    if disable_mu_lat:
        mu_lat_raw = torch.zeros_like(lam_soft_raw)
        mu_lat = None
    else:
        mu_lat = mu_lat_raw * curr_scale

    H = batch["H"].long() if batch["H"].dtype != torch.long else batch["H"]

    use_rollout_regularizers = (
        w_road > 0.0
        or w_latacc > 0.0
        or w_clear_target > 0.0
        or w_control_rate > 0.0
    )
    if use_rollout_regularizers:
        (oT, vT, min_clear, cum_risk, hard_count, arc_length,
         L_road, L_latacc, L_control_rate) = _rollout_with_regularizer_stats(
            batch=batch,
            alphas=alphas, beta=beta, gamma=gamma,
            lam_soft=lam_soft, lam_hard=lam_hard, mu_lat=mu_lat,
            lanes_count=lanes_count,
            lane_width=road_lane_width,
            road_margin=road_margin,
            road_tau=road_tau,
            latacc_free=latacc_free,
            accel_rate_free=accel_rate_free,
            steer_rate_free=steer_rate_free,
            ttc_gain=ttc_gain,
            ttc_threshold_s=ttc_threshold_s,
            ttc_softness_s=ttc_softness_s,
            ttc_min_closing_speed=ttc_min_closing_speed,
            ttc_lane_halfwidth=ttc_lane_halfwidth,
            ttc_boxed_risk_thresh=ttc_boxed_risk_thresh,
            ttc_boxed_gate_sharpness=ttc_boxed_gate_sharpness,
        )
    else:
        oT, vT, min_clear, cum_risk, hard_count, arc_length = integrate_surrogate_highway(
            o0            = batch["o0"],
            v0            = batch["v0"],
            goal          = batch["goal"],
            C             = batch["C"],
            V_neighbors   = batch.get("V_neighbors"),
            R             = batch["R"],
            mask          = batch["mask"],
            alphas        = alphas,
            beta          = beta,
            gamma         = gamma,
            lam_soft      = lam_soft,
            lam_hard      = lam_hard,
            mu_lat        = mu_lat,
            rollout_patch = batch["rollout_patch"],
            d_hat         = batch["d_hat"],
            dt            = batch["dt"],
            H             = H,
            ttc_gain      = ttc_gain,
            ttc_threshold_s = ttc_threshold_s,
            ttc_softness_s = ttc_softness_s,
            ttc_min_closing_speed = ttc_min_closing_speed,
            ttc_lane_halfwidth = ttc_lane_halfwidth,
            ttc_boxed_risk_thresh = ttc_boxed_risk_thresh,
            ttc_boxed_gate_sharpness = ttc_boxed_gate_sharpness,
        )
        L_road = batch["o0"].new_zeros(())
        L_latacc = batch["o0"].new_zeros(())
        L_control_rate = batch["o0"].new_zeros(())

    # Per-sample imitation mask: 1 for IDM, 0 for on-policy.
    imit_mask = batch["has_imit_target"].to(oT.device, oT.dtype)

    L, metrics = combine_losses(
        oT=oT, vT=vT,
        min_clear=min_clear, cum_risk=cum_risk,
        hard_count=hard_count, arc_length=arc_length,
        goal=batch["goal"],
        o_tgt=batch["o_tgt"], v_tgt=batch["v_tgt"],
        lam_soft=lam_soft, lam_hard=lam_hard,
        imit_mask=imit_mask,
        nav_scale=curr_scale,
        L_multi=None,
        cfg=weights,
    )
    tau_clear = max(float(clear_target_tau), 1e-6)
    L_clear_target = (
        F.softplus((float(clear_target) - min_clear) / tau_clear)
        * tau_clear
    ).pow(2).mean()
    L_extra = (
        float(w_road) * L_road
        + float(w_latacc) * L_latacc
        + float(w_clear_target) * L_clear_target
        + float(w_control_rate) * L_control_rate
    )
    if disable_mu_lat:
        L_mu_anti_collapse = batch["o0"].new_zeros(())
    else:
        L_mu_anti_collapse = 0.001 / (mu_lat_raw.mean() + 0.1)
    L_extra = L_extra + L_mu_anti_collapse
    if use_rollout_regularizers:
        L = L + L_extra
        metrics["loss"] = float(L.detach())
    else:
        L = L + L_mu_anti_collapse
        metrics["loss"] = float(L.detach())
    metrics["L_road"] = float(L_road.detach())
    metrics["L_latacc"] = float(L_latacc.detach())
    metrics["L_clear_target"] = float(L_clear_target.detach())
    metrics["L_control_rate"] = float(L_control_rate.detach())
    metrics["L_extra"] = float(L_extra.detach())
    metrics["L_mu_anti_collapse"] = float(L_mu_anti_collapse.detach())
    metrics["w_road"] = float(w_road)
    metrics["w_latacc"] = float(w_latacc)
    metrics["w_clear_target"] = float(w_clear_target)
    metrics["w_control_rate"] = float(w_control_rate)

    if train:
        assert optimizer is not None
        optimizer.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), grad_clip
        )
        optimizer.step()

    # Augment metrics with stream + curriculum diagnostics
    is_op = batch["is_onpolicy"].to(oT.device)
    metrics["n_idm"]         = float((1.0 - is_op).sum().item())
    metrics["n_onpolicy"]    = float(is_op.sum().item())
    metrics["curr_scale"]    = float(curr_scale)
    metrics["lam_soft_raw_mean"]    = float(lam_soft_raw.mean().detach())
    metrics["lam_soft_scaled_mean"] = float(lam_soft.mean().detach())
    metrics["lam_hard_raw_mean"]    = float(lam_hard_raw.mean().detach())
    metrics["lam_hard_scaled_mean"] = float(lam_hard.mean().detach())
    metrics["mu_lat_raw_mean"]      = float(mu_lat_raw.mean().detach())
    metrics["mu_lat_scaled_mean"]   = (
        0.0 if mu_lat is None else float(mu_lat.mean().detach())
    )
    metrics["d_hat"] = float(batch["d_hat"].detach().float().mean())
    metrics["alpha_floor"] = float(alpha_floor)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage2Cfg:
    stage1_ckpt:  str = ""
    idm_data:     str = "runs/stage1_data"
    out:          str = "checkpoints/highway_stage2"
    dfc_root:     str = ""
    epochs:       int = 30
    warmup_frac:  float = 0.30   # first 30% of epochs are λ-warmup
    bs:           int = 64       # total per-step (idm + onpolicy combined)
    lr:           float = 3e-4
    workers:      int = 0
    grad_clip:    float = 5.0
    save_every:   int = 5
    closed_loop_every: int = 10
    closed_loop_episodes: int = 10
    best_val_ltraj_guard: float = 10.0
    collect_episodes: int = 5
    collect_max_steps: int = 120
    onpolicy_capacity: int = 5000
    device:       str = "cuda" if torch.cuda.is_available() else "cpu"
    lam_soft_max: float = 50.0
    lam_hard_max: float = 10.0
    d_hat:       float = 10.0
    alpha_floor: float = 0.015
    alpha_floor_ahead_only: bool = True
    seed:         int = 0
    # Env config for collection
    vehicles_count: int = 50
    lanes_count:  int = 4
    n_max_vehicles: int = 15
    collect_envs: str = "highway-v0"
    best_eval_envs: str = "highway-v0"
    stress_offroad_terminal: bool = False
    # Optional Stage 2 legality/comfort regularizers. Defaults preserve the
    # original trainer exactly.
    w_road: float = 0.0
    road_margin: float = 0.25
    road_tau: float = 0.25
    road_lane_width: float = 4.0
    w_latacc: float = 0.0
    latacc_free: float = 4.0
    w_clear_target: float = 0.0
    clear_target: float = 6.0
    clear_target_tau: float = 1.0
    w_control_rate: float = 0.0
    accel_rate_free: float = 0.5
    steer_rate_free: float = 0.02
    freeze_geometry: bool = False
    disable_mu_lat: bool = False
    ttc_gain: float = 0.0
    ttc_threshold_s: float = 3.0
    ttc_softness_s: float = 0.5
    ttc_min_closing_speed: float = 0.5
    ttc_lane_halfwidth: float = 2.0
    ttc_boxed_risk_thresh: float = 0.25
    ttc_boxed_gate_sharpness: float = 20.0


def _to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _split_env_ids(envs: str) -> List[str]:
    return [e.strip() for e in str(envs).split(",") if e.strip()]


def _allocate_counts(total: int, n: int) -> List[int]:
    if n <= 0:
        return []
    base = total // n
    rem = total % n
    return [base + (1 if i < rem else 0) for i in range(n)]


def _aggregate_collect_stats(stats_by_env: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    attempted = sum(int(s.get("n_episodes_attempted", 0)) for s in stats_by_env.values())
    collected = sum(int(s.get("n_episodes_collected", 0)) for s in stats_by_env.values())
    crashed = sum(int(s.get("n_episodes_crashed", 0)) for s in stats_by_env.values())
    lengths = [
        float(s.get("mean_episode_length", 0.0)) * int(s.get("n_episodes_collected", 0))
        for s in stats_by_env.values()
    ]
    clear_num = 0.0
    clear_den = 0
    for s in stats_by_env.values():
        n = int(s.get("n_episodes_collected", 0))
        val = float(s.get("mean_min_clearance", float("nan")))
        if n > 0 and math.isfinite(val):
            clear_num += val * n
            clear_den += n
    return {
        "n_episodes_attempted": attempted,
        "n_episodes_collected": collected,
        "n_episodes_crashed": crashed,
        "collision_rate": crashed / collected if collected else 0.0,
        "mean_episode_length": sum(lengths) / collected if collected else 0.0,
        "mean_min_clearance": clear_num / clear_den if clear_den else 0.0,
        "by_env": stats_by_env,
    }


def _avg(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}


def _sum(metrics_list: List[Dict[str, float]], key: str) -> float:
    return sum(m.get(key, 0.0) for m in metrics_list)


def _closed_loop_best_score(
    cl_summary: Optional[Dict[str, float]],
    val_avg: Dict[str, float],
    *,
    max_val_ltraj: float,
) -> Optional[Tuple[float, float, float]]:
    """Score a checkpoint from deployment behavior.

    Lower tuple is better:
      1. lower closed-loop collision rate,
      2. higher closed-loop mean speed,
      3. lower surrogate validation trajectory error.

    The validation trajectory guard prevents a fast policy with a badly
    damaged geometric scaffold from becoming `best.pt`.
    """
    if cl_summary is None or cl_summary.get("n_episodes", 0) <= 0:
        return None
    val_ltraj = float(val_avg.get("L_traj", float("inf")))
    if not math.isfinite(val_ltraj):
        return None
    if max_val_ltraj > 0 and val_ltraj > max_val_ltraj:
        return None
    collision_rate = float(cl_summary.get("collision_rate", 1.0))
    mean_speed = float(cl_summary.get("mean_speed", 0.0))
    return (collision_rate, -mean_speed, val_ltraj)


def _stress_best_score(
    cl_summaries: Optional[Dict[str, Dict[str, float]]],
    val_avg: Dict[str, float],
    *,
    max_val_ltraj: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Best score for multi-env stress selection.

    Lower tuple is better:
      1. max collision rate across eval envs,
      2. mean collision rate across eval envs,
      3. negative mean speed across eval envs,
      4. validation L_traj guard/tiebreak.
    """
    if not cl_summaries:
        return None
    val_ltraj = float(val_avg.get("L_traj", float("inf")))
    if not math.isfinite(val_ltraj):
        return None
    if max_val_ltraj > 0 and val_ltraj > max_val_ltraj:
        return None
    vals = [
        s for s in cl_summaries.values()
        if s is not None and s.get("n_episodes", 0) > 0
    ]
    if not vals:
        return None
    crashes = [float(s.get("collision_rate", 1.0)) for s in vals]
    speeds = [float(s.get("mean_speed", 0.0)) for s in vals]
    return (max(crashes), sum(crashes) / len(crashes),
            -sum(speeds) / len(speeds), val_ltraj)


def _score_str(score: Tuple[float, float, float]) -> str:
    return (f"crash={score[0]:.1%}, "
            f"speed={-score[1]:.2f}m/s, "
            f"val_L_traj={score[2]:.4f}")


def _stress_score_str(score: Tuple[float, float, float, float]) -> str:
    return (f"max_crash={score[0]:.1%}, "
            f"mean_crash={score[1]:.1%}, "
            f"mean_speed={-score[2]:.2f}m/s, "
            f"val_L_traj={score[3]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-ckpt",  type=str, required=True)
    ap.add_argument("--idm-data",     type=str, default="runs/stage1_data")
    ap.add_argument("--out",          type=str, default="checkpoints/highway_stage2")
    ap.add_argument("--dfc-root",     type=str, default="")
    ap.add_argument("--epochs",       type=int, default=30)
    ap.add_argument("--warmup-frac",  type=float, default=0.30)
    ap.add_argument("--bs",           type=int, default=64)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--workers",      type=int, default=0)
    ap.add_argument("--grad-clip",    type=float, default=5.0)
    ap.add_argument("--save-every",   type=int, default=5)
    ap.add_argument("--closed-loop-every",    type=int, default=10)
    ap.add_argument("--closed-loop-episodes", type=int, default=10)
    ap.add_argument("--best-val-ltraj-guard", type=float, default=10.0,
                    help="Do not let a closed-loop checkpoint become best.pt "
                         "if surrogate val L_traj exceeds this value. Set <=0 "
                         "to disable the guard.")
    ap.add_argument("--collect-episodes",     type=int, default=5)
    ap.add_argument("--collect-max-steps",    type=int, default=120)
    ap.add_argument("--onpolicy-capacity",    type=int, default=5000)
    ap.add_argument("--device",       type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lam-soft-max", type=float, default=50.0)
    ap.add_argument("--lam-hard-max", type=float, default=10.0)
    ap.add_argument("--d-hat",        type=float, default=0.0,
                    help="Override IPC activation distance. Default 0 uses "
                         "stage1 checkpoint cfg, else 10.0 fallback.")
    ap.add_argument("--alpha-floor",  type=float, default=-1.0,
                    help="Override alpha floor. Default -1 uses stage1 "
                         "checkpoint cfg, else 0.015 fallback.")
    ap.add_argument("--alpha-floor-all-obstacles",
                    dest="alpha_floor_ahead_only", action="store_false",
                    help="Apply alpha floor to all valid obstacles. Default "
                         "uses stage1 checkpoint cfg, else ahead-only.")
    ap.set_defaults(alpha_floor_ahead_only=None)
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--vehicles-count", type=int, default=50)
    ap.add_argument("--lanes-count",  type=int, default=4)
    ap.add_argument("--n-max-vehicles", type=int, default=15)
    ap.add_argument("--collect-envs", type=str, default="highway-v0",
                    help="Comma-separated env ids for post-warm on-policy "
                         "collection. Episodes are split across envs.")
    ap.add_argument("--best-eval-envs", type=str, default="highway-v0",
                    help="Comma-separated env ids for closed-loop best.pt "
                         "selection. Use stress envs here to avoid selecting "
                         "a checkpoint that only works on default highway.")
    ap.add_argument("--stress-offroad-terminal", action="store_true",
                    help="Use offroad_terminal=True for collection/eval envs. "
                         "Recommended for authored stress scenarios.")
    ap.add_argument("--w-road", type=float, default=0.0,
                    help="Stage 2 road-boundary rollout penalty weight. "
                         "Default 0 preserves the original trainer.")
    ap.add_argument("--road-margin", type=float, default=0.25,
                    help="Meters inside the outer road boundary before the "
                         "soft road penalty starts.")
    ap.add_argument("--road-tau", type=float, default=0.25,
                    help="Softplus width in meters for road-boundary penalty.")
    ap.add_argument("--road-lane-width", type=float, default=4.0,
                    help="Lane width in meters for straight highway road "
                         "bounds used by the surrogate penalty.")
    ap.add_argument("--w-latacc", type=float, default=0.0,
                    help="Stage 2 lateral-acceleration excess penalty weight.")
    ap.add_argument("--latacc-free", type=float, default=4.0,
                    help="Free lateral acceleration band in m/s^2 before "
                         "comfort penalty activates.")
    ap.add_argument("--w-clear-target", type=float, default=0.0,
                    help="Penalty weight for keeping surrogate min clearance "
                         "above --clear-target.")
    ap.add_argument("--clear-target", type=float, default=6.0,
                    help="Target surrogate clearance in metres.")
    ap.add_argument("--clear-target-tau", type=float, default=1.0,
                    help="Softplus width for target-clearance penalty.")
    ap.add_argument("--w-control-rate", type=float, default=0.0,
                    help="Penalty weight for action-rate / jerk proxy.")
    ap.add_argument("--accel-rate-free", type=float, default=0.5,
                    help="Free per-step acceleration change before rate penalty.")
    ap.add_argument("--steer-rate-free", type=float, default=0.02,
                    help="Free per-step steering change before rate penalty.")
    ap.add_argument("--w-risk", type=float, default=None,
                    help="Override Stage 2 navigation cumulative-risk weight.")
    ap.add_argument("--w-hard", type=float, default=None,
                    help="Override Stage 2 navigation hard-count weight.")
    ap.add_argument("--w-len", type=float, default=None,
                    help="Override Stage 2 navigation length weight.")
    ap.add_argument("--w-clear", type=float, default=None,
                    help="Override base clearance-barrier loss weight.")
    ap.add_argument("--w-lreg", type=float, default=None,
                    help="Override lambda-head regularization weight.")
    ap.add_argument("--cvar-alpha", type=float, default=None,
                    help="Override Stage 2 CVaR alpha.")
    ap.add_argument("--stage2-imit-scale", type=float, default=None,
                    help="Override Stage 2 imitation anchor scale.")
    ap.add_argument("--freeze-geometry", action="store_true",
                    help="Freeze the Stage 1 geometry scaffold and train "
                         "only risk_enc, lam_soft_head, lam_hard_head, and "
                         "mu_lat_head. "
                         "This makes Stage 2 a true risk-force enrichment "
                         "instead of a full policy rewrite.")
    ap.add_argument("--disable-mu-lat", action="store_true",
                    help="Disable the lateral channel during Stage 2 "
                         "training, collection, and checkpoint selection. "
                         "Use this for a clean no-mu_lat ablation.")
    ap.add_argument("--ttc-gain", type=float, default=0.0,
                    help="Enable analytic TTC braking term during Stage 2 "
                         "rollout, collection, and checkpoint selection.")
    ap.add_argument("--ttc-threshold-s", type=float, default=3.0,
                    help="TTC horizon below which the TTC term activates.")
    ap.add_argument("--ttc-softness-s", type=float, default=0.5,
                    help="Smoothness of TTC activation around the threshold.")
    ap.add_argument("--ttc-min-closing-speed", type=float, default=0.5,
                    help="Minimum positive closing speed before TTC activates.")
    ap.add_argument("--ttc-lane-halfwidth", type=float, default=2.0,
                    help="Half-width of the ego-lane corridor considered by TTC.")
    ap.add_argument("--ttc-boxed-risk-thresh", type=float, default=0.25,
                    help="Side-risk threshold for the TTC boxed gate.")
    ap.add_argument("--ttc-boxed-gate-sharpness", type=float, default=20.0,
                    help="Sigmoid sharpness for the TTC boxed gate.")
    args = ap.parse_args()

    # Pull Stage 1 deployment/runtime knobs forward unless explicitly
    # overridden. This keeps Stage 2's collection/training path aligned with
    # the protected Stage 1 baseline.
    stage1_cfg = {}
    if args.stage1_ckpt:
        ck_probe = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
        stage1_cfg = ck_probe.get("cfg", {})
    cfg_kwargs = {k: getattr(args, k) for k in Stage2Cfg.__dataclass_fields__
                  if hasattr(args, k)}
    cfg_kwargs["d_hat"] = (
        float(args.d_hat) if args.d_hat > 0
        else float(stage1_cfg.get("d_hat", Stage2Cfg.d_hat))
    )
    cfg_kwargs["alpha_floor"] = (
        float(args.alpha_floor) if args.alpha_floor >= 0
        else float(stage1_cfg.get("alpha_floor", Stage2Cfg.alpha_floor))
    )
    cfg_kwargs["alpha_floor_ahead_only"] = (
        bool(args.alpha_floor_ahead_only)
        if args.alpha_floor_ahead_only is not None
        else bool(stage1_cfg.get("alpha_floor_ahead_only",
                                 Stage2Cfg.alpha_floor_ahead_only))
    )
    cfg = Stage2Cfg(**cfg_kwargs)
    loss_kwargs = {}
    for arg_name, field_name in [
        ("w_risk", "w_risk"),
        ("w_hard", "w_hard"),
        ("w_len", "w_len"),
        ("w_clear", "w_clear"),
        ("w_lreg", "w_lreg"),
        ("cvar_alpha", "cvar_alpha"),
        ("stage2_imit_scale", "stage2_imit_scale"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            loss_kwargs[field_name] = float(value)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.dfc_root:
        sys.path.insert(0, cfg.dfc_root)
    from train_material import CoefEnergyNetMaterial  # noqa: E402

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage2_cfg.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # ── IDM Dataset ──────────────────────────────────────────────────────────
    manifest_path = Path(cfg.idm_data) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    train_idm = Stage2IDMDataset(manifest_path, split="train")
    val_idm   = _stage2_dataset_or_fallback(manifest_path, "val", fallback=train_idm)
    print(f"IDM dataset: train={len(train_idm)}  val={len(val_idm)}")

    # Per-stream batch sizes. Pre-warmup: 100% IDM. Post-warmup: 50/50.
    bs_idm_warmup    = cfg.bs
    bs_idm_postwarm  = cfg.bs // 2
    bs_op_postwarm   = cfg.bs - bs_idm_postwarm        # handles odd cfg.bs cleanly

    # ── Model + warm start ──────────────────────────────────────────────────
    model = CoefEnergyNetMaterial(
        lam_soft_max=cfg.lam_soft_max, lam_hard_max=cfg.lam_hard_max,
    ).to(cfg.device)
    disable_transformer_nested_tensors(model)
    warm_meta = warm_start_from_stage1(
        model, Path(cfg.stage1_ckpt), cfg.device,
        freeze_geometry=cfg.freeze_geometry,
        disable_mu_lat=cfg.disable_mu_lat,
    )

    weights = HighwayLossWeights.stage2()
    weights.lam_soft_max = cfg.lam_soft_max
    weights.lam_hard_max = cfg.lam_hard_max
    for field_name, value in loss_kwargs.items():
        setattr(weights, field_name, value)
    if loss_kwargs:
        print("Loss overrides:")
        for field_name in sorted(loss_kwargs):
            print(f"  {field_name}: {getattr(weights, field_name)}")

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.1
    )

    # ── On-policy buffer + collection env ───────────────────────────────────
    buffer = OnPolicyBuffer(capacity=cfg.onpolicy_capacity)

    gym = _import_gym()
    collect_env_ids = _split_env_ids(cfg.collect_envs)
    best_eval_env_ids = _split_env_ids(cfg.best_eval_envs)
    collect_envs = {
        env_id: make_env(
            gym, env_id,
            vehicles_count=cfg.vehicles_count,
            lanes_count=cfg.lanes_count,
            offroad_terminal=cfg.stress_offroad_terminal,
        )
        for env_id in collect_env_ids
    }
    eval_envs = {
        env_id: (
            collect_envs[env_id] if env_id in collect_envs else make_env(
                gym, env_id,
                vehicles_count=cfg.vehicles_count,
                lanes_count=cfg.lanes_count,
                offroad_terminal=cfg.stress_offroad_terminal,
            )
        )
        for env_id in best_eval_env_ids
    }
    primary_eval_env = eval_envs[best_eval_env_ids[0]]
    observer = HighwayMaterialObservation(WrapperConfig(
        n_max_vehicles=cfg.n_max_vehicles,
        dt_surrogate=DT_TARGET,
        horizon_surrogate=H_TARGET,
    ))

    # ── Training loop ────────────────────────────────────────────────────────
    warmup_epochs = _warmup_epochs(cfg.epochs, cfg.warmup_frac)
    print(f"\nDevice: {cfg.device}  Epochs: {cfg.epochs}  "
           f"(warmup: {warmup_epochs})  Batch: {cfg.bs}")
    print(f"Mixing: warmup→100% IDM, post-warmup→{bs_idm_postwarm}/{bs_op_postwarm} "
           f"(IDM/on-policy per batch)")
    print(f"Runtime knobs: d_hat={cfg.d_hat:.1f}  "
          f"alpha_floor={cfg.alpha_floor:.4f}  "
          f"ahead_only={cfg.alpha_floor_ahead_only}")
    if cfg.ttc_gain > 0:
        print(f"TTC knobs: gain={cfg.ttc_gain:.2f}  "
              f"threshold={cfg.ttc_threshold_s:.2f}s  "
              f"softness={cfg.ttc_softness_s:.2f}s  "
              f"boxed_thresh={cfg.ttc_boxed_risk_thresh:.2f}")
    print(f"Freeze geometry: {cfg.freeze_geometry}")
    print(f"Disable mu_lat: {cfg.disable_mu_lat}")
    print(f"Collection: {cfg.collect_episodes} episodes/epoch starting epoch "
           f"{warmup_epochs}\n")
    print(f"Collect envs: {collect_env_ids}")
    print(f"Best-eval envs: {best_eval_env_ids}  "
          f"offroad_terminal={cfg.stress_offroad_terminal}\n")
    if (
        cfg.w_road > 0
        or cfg.w_latacc > 0
        or cfg.w_clear_target > 0
        or cfg.w_control_rate > 0
    ):
        print("Regularizers:")
        print(f"  road:   w={cfg.w_road:.4g}, margin={cfg.road_margin:.2f}m, "
              f"tau={cfg.road_tau:.2f}m, lane_width={cfg.road_lane_width:.2f}m")
        print(f"  latacc: w={cfg.w_latacc:.4g}, free={cfg.latacc_free:.2f}m/s^2")
        print(f"  clear-target: w={cfg.w_clear_target:.4g}, "
              f"target={cfg.clear_target:.2f}m, tau={cfg.clear_target_tau:.2f}m")
        print(f"  control-rate: w={cfg.w_control_rate:.4g}, "
              f"accel_free={cfg.accel_rate_free:.2f}, "
              f"steer_free={cfg.steer_rate_free:.3f}\n")

    history: List[Dict[str, Any]] = []
    best_closed_loop_score: Optional[Tuple[float, float, float, float]] = None

    for epoch in range(cfg.epochs):
        in_warmup = (epoch < warmup_epochs)
        if cfg.freeze_geometry and warmup_epochs > 0:
            # With geometry frozen, warmup should train the risk heads on
            # IDM/Stage-1 states while withholding on-policy collection. The
            # old curr=0 warmup would produce zero useful gradient for the
            # only trainable modules.
            curr = ((epoch + 1) / float(warmup_epochs)) if in_warmup else 1.0
        else:
            curr = curriculum_scale(epoch, cfg.epochs, warmup_epochs)
        seed_offset = 2000 + epoch * 100  # disjoint from eval (1000) + buffer-collection seeds

        # ── On-policy collection (post-warmup only) ─────────────────────────
        if not in_warmup:
            print(f"[epoch {epoch:03d}] collecting {cfg.collect_episodes} on-policy episodes "
                  f"(seeds {seed_offset}+)")
            model.eval()
            t_collect = time.time()
            stats_by_env: Dict[str, Dict[str, float]] = {}
            counts = _allocate_counts(cfg.collect_episodes, len(collect_env_ids))
            for env_i, (env_id, n_eps) in enumerate(zip(collect_env_ids, counts)):
                if n_eps <= 0:
                    continue
                try:
                    stats_by_env[env_id] = collect_episodes(
                        model, observer, collect_envs[env_id], buffer,
                        n_episodes=n_eps,
                        max_steps=cfg.collect_max_steps,
                        base_seed=seed_offset + env_i * 10000,
                        device=cfg.device,
                        stage=2,
                        lam_scale=curr,
                        disable_mu_lat=cfg.disable_mu_lat,
                        d_hat_override=cfg.d_hat,
                        alpha_floor=cfg.alpha_floor,
                        alpha_floor_ahead_only=cfg.alpha_floor_ahead_only,
                        ttc_gain=cfg.ttc_gain,
                        ttc_threshold_s=cfg.ttc_threshold_s,
                        ttc_softness_s=cfg.ttc_softness_s,
                        ttc_min_closing_speed=cfg.ttc_min_closing_speed,
                        ttc_lane_halfwidth=cfg.ttc_lane_halfwidth,
                        ttc_boxed_risk_thresh=cfg.ttc_boxed_risk_thresh,
                        ttc_boxed_gate_sharpness=cfg.ttc_boxed_gate_sharpness,
                        verbose=False,
                    )
                except Exception as exc:
                    print(f"  collection failed for {env_id}: {exc}")
                    stats_by_env[env_id] = {
                        "n_episodes_attempted": n_eps,
                        "n_episodes_collected": 0,
                        "n_episodes_crashed": 0,
                        "collision_rate": 0.0,
                        "mean_episode_length": 0.0,
                        "mean_min_clearance": 0.0,
                    }
            collect_stats = _aggregate_collect_stats(stats_by_env)
            print(f"  collected {collect_stats.get('n_episodes_collected', 0)} eps "
                  f"({collect_stats.get('n_episodes_crashed', 0)} crashes, "
                  f"crash_rate={collect_stats.get('collision_rate', 0):.1%})  "
                  f"{time.time()-t_collect:.1f}s")
            for env_id, st in collect_stats.get("by_env", {}).items():
                print(f"    {env_id}: n={int(st.get('n_episodes_collected', 0))} "
                      f"crash={st.get('collision_rate', 0.0):.1%} "
                      f"len={st.get('mean_episode_length', 0.0):.1f}")
            print(f"  buffer: {buffer}")
        else:
            collect_stats = {}

        # ── Build mixed loader for this epoch ───────────────────────────────
        train_idm_loader = DataLoader(
            train_idm,
            batch_size = bs_idm_warmup if in_warmup else bs_idm_postwarm,
            shuffle=True, num_workers=cfg.workers,
            collate_fn=collate_mixed, drop_last=True,
        )
        if in_warmup or len(buffer) == 0:
            mixed_loader = MixedLoader(
                idm_loader=train_idm_loader, onpolicy_loader=None,
                idm_per_step=bs_idm_warmup, onpolicy_per_step=0,
            )
        else:
            op_snapshot = buffer.snapshot_dataset()
            op_loader = DataLoader(
                op_snapshot, batch_size=bs_op_postwarm,
                shuffle=True, num_workers=cfg.workers,
                collate_fn=collate_mixed, drop_last=True,
            )
            mixed_loader = MixedLoader(
                idm_loader=train_idm_loader, onpolicy_loader=op_loader,
                idm_per_step=bs_idm_postwarm, onpolicy_per_step=bs_op_postwarm,
            )

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        t_train = time.time()
        train_metrics_list: List[Dict[str, float]] = []
        for i, batch in enumerate(mixed_loader):
            batch = _to_device(batch, cfg.device)
            m = step_batch_stage2(
                model, batch, weights,
                train=True, optimizer=optimizer,
                grad_clip=cfg.grad_clip, curr_scale=curr,
                d_hat_override=cfg.d_hat,
                alpha_floor=cfg.alpha_floor,
                alpha_floor_ahead_only=cfg.alpha_floor_ahead_only,
                w_road=cfg.w_road,
                road_margin=cfg.road_margin,
                road_tau=cfg.road_tau,
                road_lane_width=cfg.road_lane_width,
                lanes_count=cfg.lanes_count,
                w_latacc=cfg.w_latacc,
                latacc_free=cfg.latacc_free,
                w_clear_target=cfg.w_clear_target,
                clear_target=cfg.clear_target,
                clear_target_tau=cfg.clear_target_tau,
                w_control_rate=cfg.w_control_rate,
                accel_rate_free=cfg.accel_rate_free,
                steer_rate_free=cfg.steer_rate_free,
                disable_mu_lat=cfg.disable_mu_lat,
                ttc_gain=cfg.ttc_gain,
                ttc_threshold_s=cfg.ttc_threshold_s,
                ttc_softness_s=cfg.ttc_softness_s,
                ttc_min_closing_speed=cfg.ttc_min_closing_speed,
                ttc_lane_halfwidth=cfg.ttc_lane_halfwidth,
                ttc_boxed_risk_thresh=cfg.ttc_boxed_risk_thresh,
                ttc_boxed_gate_sharpness=cfg.ttc_boxed_gate_sharpness,
            )
            train_metrics_list.append(m)
            if i % 50 == 0:
                print(f"  ep{epoch:03d} it{i:04d}  "
                      f"L={m['loss']:.4f}  L_nav={m['L_nav']:.4f}  "
                      f"L_traj={m['L_traj']:.4f}  cum_risk={m['cum_risk_mean']:.3f}  "
                      f"L_road={m['L_road']:.4f} L_lat={m['L_latacc']:.4f}  "
                      f"L_clrt={m['L_clear_target']:.4f} L_rate={m['L_control_rate']:.4f}  "
                      f"λ_s_raw={m['lam_soft_raw_mean']:.2f} λ_s_sc={m['lam_soft_scaled_mean']:.2f}  "
                      f"μ_lat_raw={m['mu_lat_raw_mean']:.3f} μ_lat_sc={m['mu_lat_scaled_mean']:.3f}  "
                      f"n_idm={int(m['n_idm'])} n_op={int(m['n_onpolicy'])}")
        train_dt = time.time() - t_train
        train_avg = _avg(train_metrics_list)

        # ── Validate (surrogate, IDM val set) ────────────────────────────────
        val_idm_loader = DataLoader(
            val_idm, batch_size=cfg.bs, shuffle=False,
            num_workers=cfg.workers, collate_fn=collate_mixed, drop_last=False,
        )
        model.eval()
        val_metrics_list: List[Dict[str, float]] = []
        with torch.no_grad():
            for batch in val_idm_loader:
                batch = _to_device(batch, cfg.device)
                m = step_batch_stage2(model, batch, weights,
                                       train=False, optimizer=None,
                                       curr_scale=curr,
                                       d_hat_override=cfg.d_hat,
                                       alpha_floor=cfg.alpha_floor,
                                       alpha_floor_ahead_only=cfg.alpha_floor_ahead_only,
                                       w_road=cfg.w_road,
                                       road_margin=cfg.road_margin,
                                       road_tau=cfg.road_tau,
                                       road_lane_width=cfg.road_lane_width,
                                       lanes_count=cfg.lanes_count,
                                       w_latacc=cfg.w_latacc,
                                       latacc_free=cfg.latacc_free,
                                       w_clear_target=cfg.w_clear_target,
                                       clear_target=cfg.clear_target,
                                       clear_target_tau=cfg.clear_target_tau,
                                       w_control_rate=cfg.w_control_rate,
                                       accel_rate_free=cfg.accel_rate_free,
                                       steer_rate_free=cfg.steer_rate_free,
                                       disable_mu_lat=cfg.disable_mu_lat,
                                       ttc_gain=cfg.ttc_gain,
                                       ttc_threshold_s=cfg.ttc_threshold_s,
                                       ttc_softness_s=cfg.ttc_softness_s,
                                       ttc_min_closing_speed=cfg.ttc_min_closing_speed,
                                       ttc_lane_halfwidth=cfg.ttc_lane_halfwidth,
                                       ttc_boxed_risk_thresh=cfg.ttc_boxed_risk_thresh,
                                       ttc_boxed_gate_sharpness=cfg.ttc_boxed_gate_sharpness)
                val_metrics_list.append(m)
        val_avg = _avg(val_metrics_list)

        scheduler.step()

        # ── Closed-loop eval (every closed_loop_every epochs) ────────────────
        cl_summary = None
        cl_summaries = None
        if (cfg.closed_loop_every > 0
                and (epoch % cfg.closed_loop_every == 0
                       or epoch == cfg.epochs - 1)):
            print(f"  closed-loop eval ({cfg.closed_loop_episodes} eps)...")
            cl_summaries = {}
            for env_i, env_id in enumerate(best_eval_env_ids):
                cl_summaries[env_id] = run_closed_loop_eval(
                    model, observer, eval_envs[env_id],
                    episodes=cfg.closed_loop_episodes,
                    max_steps=cfg.collect_max_steps,
                    device=cfg.device, base_seed=1000 + env_i * 10000,
                    d_hat_override=cfg.d_hat,
                    alpha_floor=cfg.alpha_floor,
                    alpha_floor_ahead_only=cfg.alpha_floor_ahead_only,
                    disable_mu_lat=cfg.disable_mu_lat,
                    ttc_gain=cfg.ttc_gain,
                    ttc_threshold_s=cfg.ttc_threshold_s,
                    ttc_softness_s=cfg.ttc_softness_s,
                    ttc_min_closing_speed=cfg.ttc_min_closing_speed,
                    ttc_lane_halfwidth=cfg.ttc_lane_halfwidth,
                    ttc_boxed_risk_thresh=cfg.ttc_boxed_risk_thresh,
                    ttc_boxed_gate_sharpness=cfg.ttc_boxed_gate_sharpness,
                )
                s = cl_summaries[env_id]
                print(f"    {env_id}: crash_rate={s['collision_rate']:.1%}  "
                      f"v_mean={s['mean_speed']:.2f}m/s  "
                      f"lc/ep={s['lane_changes_per_ep']:.2f}")
            cl_summary = cl_summaries.get(best_eval_env_ids[0])

        # ── Log ──────────────────────────────────────────────────────────────
        print(f"\n[epoch {epoch:03d}] {train_dt:.1f}s  curr={curr:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"{'WARMUP' if in_warmup else 'POST-WARMUP'}")
        print(f"  train: L={train_avg['loss']:.4f}  L_nav={train_avg['L_nav']:.4f}  "
              f"L_traj={train_avg['L_traj']:.4f}  L_clear={train_avg['L_clear']:.4f}  "
              f"L_lreg={train_avg['L_lreg']:.4f}")
        if (
            cfg.w_road > 0
            or cfg.w_latacc > 0
            or cfg.w_clear_target > 0
            or cfg.w_control_rate > 0
        ):
            print(f"         L_road={train_avg['L_road']:.4f}  "
                  f"L_latacc={train_avg['L_latacc']:.4f}  "
                  f"L_clrt={train_avg['L_clear_target']:.4f}  "
                  f"L_rate={train_avg['L_control_rate']:.4f}  "
                  f"L_extra={train_avg['L_extra']:.4f}")
        print(f"         cum_risk={train_avg['cum_risk_mean']:.3f}  "
              f"hard_count={train_avg['hard_count_mean']:.3f}")
        print(f"         λ_s raw={train_avg['lam_soft_raw_mean']:.2f}/scaled="
               f"{train_avg['lam_soft_scaled_mean']:.2f}  "
               f"λ_h raw={train_avg['lam_hard_raw_mean']:.2f}/scaled="
               f"{train_avg['lam_hard_scaled_mean']:.2f}  "
               f"μ_lat raw={train_avg['mu_lat_raw_mean']:.3f}/scaled="
               f"{train_avg['mu_lat_scaled_mean']:.3f}")
        print(f"         L_mu_anti={train_avg['L_mu_anti_collapse']:.4f}")
        print(f"         n_idm/batch={train_avg['n_idm']:.1f}  "
              f"n_op/batch={train_avg['n_onpolicy']:.1f}  "
              f"n_imit_total={int(_sum(train_metrics_list, 'n_imit_samples'))}")
        print(f"  val  : L={val_avg['loss']:.4f}  L_nav={val_avg['L_nav']:.4f}  "
              f"L_traj={val_avg['L_traj']:.4f}  "
              f"L_road={val_avg.get('L_road', 0.0):.4f}  "
              f"L_latacc={val_avg.get('L_latacc', 0.0):.4f}\n")

        history.append({
            "epoch": epoch,
            "curr_scale": curr,
            "in_warmup": in_warmup,
            "lr": scheduler.get_last_lr()[0],
            "train": train_avg,
            "val": val_avg,
            "collect": collect_stats,
            "buffer": buffer.stats(),
            "closed_loop": cl_summary,
            "closed_loop_by_env": cl_summaries,
            "train_dt_s": train_dt,
        })
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

        # ── Checkpoint ───────────────────────────────────────────────────────
        ck = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "cfg": asdict(cfg),
            "weights": asdict(weights),
            "warm_meta": warm_meta,
        }
        torch.save(ck, out_dir / "last.pt")
        if cfg.save_every > 0 and (epoch % cfg.save_every == 0
                                     or epoch == cfg.epochs - 1):
            torch.save(ck, out_dir / f"epoch_{epoch:03d}.pt")

        # Best-by-closed-loop behavior. Scaled validation L_nav is not a
        # comparable quantity across curriculum epochs, so checkpoint
        # selection is tied to deployment metrics instead.
        if not in_warmup and cl_summaries is not None:
            score = _stress_best_score(
                cl_summaries, val_avg, max_val_ltraj=cfg.best_val_ltraj_guard
            )
            if score is None:
                print("  best.pt unchanged "
                      f"(closed-loop score failed val_L_traj guard "
                      f"{cfg.best_val_ltraj_guard:.2f})")
            elif best_closed_loop_score is None or score < best_closed_loop_score:
                best_closed_loop_score = score
                torch.save(ck, out_dir / "best.pt")
                print(f"  ✓ new best closed-loop {_stress_score_str(score)} → best.pt")

    # ── Final save ───────────────────────────────────────────────────────────
    buffer.save(out_dir / "final_buffer.pt")
    closed = set()
    for env in list(collect_envs.values()) + list(eval_envs.values()):
        if id(env) not in closed:
            env.close()
            closed.add(id(env))
    if best_closed_loop_score is None:
        print("\nDone. No post-warm closed-loop checkpoint satisfied the best.pt rule.")
    else:
        print(f"\nDone. Best closed-loop {_stress_score_str(best_closed_loop_score)}")
    print(f"Checkpoints: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Closed-loop eval (lightweight version of eval_stage1, inlined for cadence)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_closed_loop_eval(
    model, observer, env, *,
    episodes: int, max_steps: int, device: str, base_seed: int = 1000,
    d_hat_override: float = 0.0,
    alpha_floor: float = 0.0,
    alpha_floor_ahead_only: bool = True,
    disable_mu_lat: bool = False,
    ttc_gain: float = 0.0,
    ttc_threshold_s: float = 3.0,
    ttc_softness_s: float = 0.5,
    ttc_min_closing_speed: float = 0.5,
    ttc_lane_halfwidth: float = 2.0,
    ttc_boxed_risk_thresh: float = 0.25,
    ttc_boxed_gate_sharpness: float = 20.0,
) -> Dict[str, float]:
    """Closed-loop rollouts in highway-v0. Computes the same key metrics as
    eval_stage1 (collision rate, mean speed, lane changes/ep). curr_scale=1
    here — closed-loop eval uses the model's full risk-head outputs."""
    from collect_onpolicy import collect_one_episode

    model.eval()
    n_crashed = 0
    speeds = []
    lengths = []
    lane_changes = []

    for i in range(episodes):
        seed = base_seed + i
        try:
            from collect_onpolicy import _reset
            _reset(env, seed)
            samples, ep_stats = collect_one_episode(
                model, observer, env,
                max_steps=max_steps, device=device, stage=2,
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
            print(f"    eval ep {i} seed={seed} FAILED: {exc}")
            continue
        if not samples:
            continue
        if ep_stats["collided"]:
            n_crashed += 1
        lengths.append(ep_stats["length"])

        # Mean speed from action_taken's o_next field is ergonomic.
        ep_speeds = [float(torch.linalg.norm(s.v_next).item()) for s in samples]
        if ep_speeds:
            speeds.append(float(np.mean(ep_speeds)))

        # Lane changes via env's lane_index — we don't have it from samples,
        # so we just report 0 here. eval_stage2.py (next file) gets this right.
        lane_changes.append(0)

    n_done = len([s for s in lengths])
    return {
        "n_episodes":     n_done,
        "n_crashed":      n_crashed,
        "collision_rate": n_crashed / n_done if n_done else 0.0,
        "mean_speed":     float(np.mean(speeds)) if speeds else 0.0,
        "mean_length":    float(np.mean(lengths)) if lengths else 0.0,
        "lane_changes_per_ep": float(np.mean(lane_changes)) if lane_changes else 0.0,
    }


if __name__ == "__main__":
    main()
