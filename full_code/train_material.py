#!/usr/bin/env python3
"""
train_material.py

Setting 2 training: GRL-SNAM geometry backbone + oracle material risk.

Architecture
------------
CoefEnergyNetMaterial extends CoefEnergyNet with:
  - risk_enc: small CNN over (2,P,P) risk patch [r̃, hard_mask]
  - λ_soft head: weight on F_mat_soft = -λ_soft * ∇r̃(q)
  - λ_hard head: weight on F_mat_hard = -λ_hard * ∇b(φ(q))

Surrogate dynamics
------------------
integrate_surrogate_material adds two material force terms to the
existing geometry forces (IPC barriers + goal attraction + damping):
  F_mat_soft = -λ_soft * risk_grad          (soft risk field gradient)
  F_mat_hard = -λ_hard * sdf_barrier_grad   (hard hazard SDF barrier)

Loss
----
Episode cost J accumulates per-step:
  - path length (step cost)
  - soft risk exposure: λ_risk * r̃(q_t) * Δs
  - hard hazard penalty: λ_hard_pen * 1[q_t ∈ H]
CVaR_α(J) is the primary navigation loss (α=0.95, worst 5% of rollouts).
Additional terms: L_goal, L_clear, L_multi (unchanged from geometry trainer).

Three-stage curriculum
----------------------
Stage 1: geometry only (freeze risk heads) → Setting 1 checkpoint
Stage 2: unfreeze risk heads, add material losses
Stage 3: (Setting 3) replace oracle patches with belief patches + perception loss

Usage
-----
# Stage 1 (geometry only, produces Setting 1 checkpoint)
python train_material.py \\
    --root data/dfc2018_stagewise \\
    --stage 1 --epochs 30 --out checkpoints/s1

# Stage 2 (material-aware, produces Setting 2 checkpoint)
python train_material.py \\
    --root data/dfc2018_stagewise \\
    --stage 2 --epochs 50 --out checkpoints/s2 \\
    --ckpt_s1 checkpoints/s1/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Re-use IPC barrier from existing codebase
try:
    from train_coef_energy import ipc_piecewise, CoefEnergyNet
except ImportError:
    raise ImportError("train_coef_energy.py must be on the Python path.")

try:
    from surrogate_robust import multi_start_penalty
except ImportError:
    raise ImportError("surrogate_robust.py must be on the Python path.")

try:
    from scripts.build_dfc2018_stagewise import (
        extract_local_geom_obstacles,
        extract_risk_patch,
        extract_rollout_patch,
        dijkstra_geom,
        _pick_local_goal_index,
    )
except ImportError:
    raise ImportError("scripts/build_dfc2018_stagewise.py must be on the Python path.")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class RiskPatchEncoder(nn.Module):
    """
    Small CNN over (2, P, P) risk patch:
      ch0 = smoothed r̃(x)
      ch1 = hard hazard binary mask

    Output: (B, d_out) risk context vector.
    """
    def __init__(self, patch_size: int = 32, d_out: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 2, P, P)
            nn.Conv2d(2, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            # (B, 32, P/2, P/2)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            # (B, 64, P/4, P/4)
            nn.AdaptiveAvgPool2d(4),          # (B, 64, 4, 4)
            nn.Flatten(),                      # (B, 1024)
            nn.Linear(64 * 4 * 4, d_out), nn.ReLU(),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.net(patch)   # (B, d_out)


class CoefEnergyNetMaterial(nn.Module):
    """
    Extends CoefEnergyNet with material-risk outputs.

    New outputs (all ≥ 0 via softplus):
      λ_soft: weight on F_mat_soft = -λ_soft * ∇r̃(q)
      λ_hard: weight on F_mat_hard = -λ_hard * ∇b(φ(q))

    Setting 1 mode: set λ_soft = λ_hard = 0 (freeze risk heads or zero output).
    Setting 2 mode: full model with oracle risk patches.
    Setting 3 mode: replace oracle patch with inferred belief patch (same interface).

    Input
    -----
    obs_feats  : (B, N, 6)  geometric obstacle features
    obs_mask   : (B, N)     bool, True = valid obstacle
    goal_feats : (B, 4)     [Δx, Δy, dist, 1.0]
    risk_patch : (B, 2, P, P)  [r̃, hard_mask]

    Output
    ------
    alphas  : (B, N)   per-obstacle barrier weights
    beta    : (B,)     goal attraction
    gamma   : (B,)     damping
    lam_soft: (B,)     soft risk weight
    lam_hard: (B,)     hard hazard weight
    mu_lat  : (B,)     lateral risk-opportunity weight
    """
    def __init__(
        self,
        d_obs: int = 6,
        d_goal: int = 4,
        d_tok: int = 64,
        patch_size: int = 32,
        d_risk: int = 64,
        lam_soft_max: float = 5.0,
        lam_hard_max: float = 10.0,
        mu_lat_max: float = 5.0,
    ):
        super().__init__()
        self.lam_soft_max = lam_soft_max
        self.lam_hard_max = lam_hard_max
        self.mu_lat_max = mu_lat_max

        # ── Geometry backbone (identical to CoefEnergyNet) ──────────────
        from torch import nn as _nn
        self.obs_enc  = _nn.Sequential(
            _nn.Linear(d_obs, 128), _nn.ReLU(), _nn.Linear(128, d_tok))
        self.goal_enc = _nn.Sequential(
            _nn.Linear(d_goal, 64), _nn.ReLU(), _nn.Linear(64, d_tok))
        enc = _nn.TransformerEncoderLayer(
            d_model=d_tok, nhead=4, dim_feedforward=128, batch_first=True)
        self.fuser = _nn.TransformerEncoder(enc, num_layers=2)
        self.alpha_head = _nn.Sequential(
            _nn.Linear(d_tok, 64), _nn.ReLU(), _nn.Linear(64, 1))
        self.beta_head  = _nn.Sequential(
            _nn.Linear(d_tok, 64), _nn.ReLU(), _nn.Linear(64, 1))
        self.gamma_head = _nn.Sequential(
            _nn.Linear(d_tok, 64), _nn.ReLU(), _nn.Linear(64, 1))

        # ── Material risk branch ────────────────────────────────────────
        self.risk_enc = RiskPatchEncoder(patch_size=patch_size, d_out=d_risk)

        # λ_soft and λ_hard condition on both risk context and goal context
        self.lam_soft_head = _nn.Sequential(
            _nn.Linear(d_risk + d_tok, 64), _nn.ReLU(),
            _nn.Linear(64, 1))
        self.lam_hard_head = _nn.Sequential(
            _nn.Linear(d_risk + d_tok, 64), _nn.ReLU(),
            _nn.Linear(64, 1))
        self.mu_lat_head = _nn.Sequential(
            _nn.Linear(d_risk + d_tok, 64), _nn.ReLU(),
            _nn.Linear(64, 1))
        with torch.no_grad():
            self.mu_lat_head[-1].bias.fill_(-5.0)

    def forward(
        self,
        obs_feats:  torch.Tensor,   # (B, N, 6)
        obs_mask:   torch.Tensor,   # (B, N) bool
        goal_feats: torch.Tensor,   # (B, 4)
        risk_patch: torch.Tensor,   # (B, 2, P, P)
    ) -> Tuple[torch.Tensor, ...]:

        B, N = obs_feats.shape[:2]
        z_goal = self.goal_enc(goal_feats).unsqueeze(1)   # (B,1,d)

        if N == 0:
            tokens = z_goal
            pad    = torch.zeros(B, 1, dtype=torch.bool, device=obs_feats.device)
            z_all  = self.fuser(tokens, src_key_padding_mask=pad)
            ctx    = z_all[:, 0]
            alphas = obs_feats.new_zeros(B, 0)
        else:
            z_obs  = self.obs_enc(obs_feats.reshape(B*N, -1)).reshape(B, N, -1)
            tokens = torch.cat([z_goal, z_obs], dim=1)    # (B, 1+N, d)
            pad    = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=obs_mask.device),
                ~obs_mask], dim=1)
            z_all  = self.fuser(tokens, src_key_padding_mask=pad)
            ctx    = z_all[:, 0]                           # goal context token
            a      = F.softplus(self.alpha_head(z_all[:, 1:]).squeeze(-1))
            alphas = torch.where(obs_mask, a, torch.zeros_like(a))

        beta  = F.softplus(self.beta_head(ctx)).squeeze(-1)    # (B,)
        gamma = F.softplus(self.gamma_head(ctx)).squeeze(-1)   # (B,)

        # Material heads — condition on risk context + goal context
        risk_ctx  = self.risk_enc(risk_patch)                  # (B, d_risk)
        mat_feats = torch.cat([risk_ctx, ctx], dim=-1)         # (B, d_risk+d_tok)

        lam_soft = self.lam_soft_max * torch.sigmoid(
            self.lam_soft_head(mat_feats).squeeze(-1))          # (B,) ∈ (0, lam_soft_max)
        lam_hard = self.lam_hard_max * torch.sigmoid(
            self.lam_hard_head(mat_feats).squeeze(-1))          # (B,) ∈ (0, lam_hard_max)
        mu_lat = self.mu_lat_max * torch.sigmoid(
            self.mu_lat_head(mat_feats).squeeze(-1))            # (B,) ∈ (0, mu_lat_max)

        return alphas, beta, gamma, lam_soft, lam_hard, mu_lat


def load_geometry_weights(
    material_model: CoefEnergyNetMaterial,
    geom_ckpt_path: str,
    device: str = "cpu",
):
    """
    Copy geometry backbone weights from a Stage 1 CoefEnergyNet checkpoint
    into CoefEnergyNetMaterial.  Risk heads are left randomly initialised.
    """
    ck = torch.load(geom_ckpt_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        sd = ck["model"]
    elif isinstance(ck, dict) and "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    else:
        sd = ck
    # Only copy keys that exist in both models (geometry backbone only)
    own_sd  = material_model.state_dict()
    matched = {k: v for k, v in sd.items() if k in own_sd and
               own_sd[k].shape == v.shape}
    own_sd.update(matched)
    material_model.load_state_dict(own_sd)
    print(f"  Loaded {len(matched)}/{len(sd)} geometry weights from {geom_ckpt_path}")
    if not matched:
        raise RuntimeError(
            f"No compatible geometry weights found in {geom_ckpt_path}. "
            "Check that the Stage 1 checkpoint format matches the model."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate dynamics with material forces
# ─────────────────────────────────────────────────────────────────────────────

def _sdf_barrier_grad(
    sdf_val: torch.Tensor,    # (B,)  φ(q), metres to nearest hard hazard
    d_hat_sdf: float = 3.0,   # activation distance in metres
    k_sharp: float = 5.0,     # sharpness
) -> torch.Tensor:
    """
    Softplus SDF barrier value (scalar, for loss) and its gradient magnitude.
    The gradient direction comes from grad_sdf passed in separately.

    b(φ) = (1/k) * log(1 + exp(k*(d_hat - φ)))   active when φ < d_hat
    db/dφ = -sigmoid(k*(d_hat - φ))               negative → points toward open space
    """
    inner   = k_sharp * (d_hat_sdf - sdf_val)
    b_val   = F.softplus(inner) / k_sharp               # (B,)
    db_dphi = -torch.sigmoid(inner)                     # (B,), negative
    return b_val, db_dphi


def bilinear_sample_patch(
    patch:  torch.Tensor,   # (B, C, Hp, Wp)  local raster
    o:      torch.Tensor,   # (B, 2)  current position in global pixel coords (x=col, y=row)
    o0:     torch.Tensor,   # (B, 2)  patch centre in global pixel coords
) -> torch.Tensor:
    """
    Bilinearly sample `patch` at position `o` given that the patch is
    centred at `o0` in global pixel coordinates.

    Returns (B, C) values at o.  Out-of-bounds positions clamp to border.
    """
    B, C, Hp, Wp = patch.shape
    # Offset from patch centre in pixel units (x=col direction, y=row direction)
    offset = o - o0                                          # (B, 2)
    half_w = (Wp - 1) / 2.0
    half_h = (Hp - 1) / 2.0
    # Normalise to [-1, 1] for F.grid_sample (x maps cols, y maps rows)
    gx = offset[:, 0] / (half_w + 1e-8)                     # (B,)
    gy = offset[:, 1] / (half_h + 1e-8)
    grid = torch.stack([gx, gy], dim=-1).view(B, 1, 1, 2)   # (B,1,1,2)
    sampled = F.grid_sample(
        patch, grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    )                                                        # (B, C, 1, 1)
    return sampled.view(B, C)                                # (B, C)


def integrate_surrogate_material(
    o0:         torch.Tensor,   # (B,2)  initial position
    v0:         torch.Tensor,   # (B,2)  initial velocity
    goal:       torch.Tensor,   # (B,2)  stage goal
    C:          torch.Tensor,   # (B,N,2) obstacle centres
    R:          torch.Tensor,   # (B,N)   effective radii
    mask:       torch.Tensor,   # (B,N)   bool, True=valid
    alphas:     torch.Tensor,   # (B,N)
    beta:       torch.Tensor,   # (B,)
    gamma:      torch.Tensor,   # (B,)
    lam_soft:   torch.Tensor,   # (B,)
    lam_hard:   torch.Tensor,   # (B,)
    rollout_patch: torch.Tensor, # (B,4,Hp,Wp)  local raster centred at o0
                                 # ch0=risk_map r̃  ch1=sdf_hard φ
                                 # ch2=∂r̃/∂x(col)  ch3=∂r̃/∂y(row)
    d_hat:      torch.Tensor,   # (B,)
    dt:         torch.Tensor,   # (B,)
    H:          torch.Tensor,   # (B,)  int, horizon steps
    robot_radius: float = 0.0,
    margin_factor: float = 0.5,
    mass: float = 1.0,
    d_hat_sdf: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Surrogate integrator with material forces.

    Semantic fields are resampled at the current position o at every step
    via bilinear interpolation from `rollout_patch`.  This means forces and
    cost accumulation reflect the actual path taken, not a frozen initial value.

    Force model (per step):
        sampled = bilinear_sample(rollout_patch, o)
        r̃(o)   = sampled[:,0]   ∈ [0,1]
        φ(o)   = sampled[:,1]   metres to hard hazard
        ∂r̃/∂x = sampled[:,2]
        ∂r̃/∂y = sampled[:,3]

        F_mat_soft = -lam_soft * [∂r̃/∂x, ∂r̃/∂y]
        db/dφ      = -sigmoid(k*(d_hat_sdf - φ))
        F_mat_hard = -lam_hard * db/dφ * ∇φ  (approximated via ∂φ/∂x≈-∂r̃/∂x proxy)
        F_tot      = F_geom + F_goal + F_mat_soft + F_mat_hard - gamma*v

    Returns: (oT, vT, min_clear_geom, cum_risk, hard_count, arc_length)
      arc_length: Σ |o_{t+1} - o_t|  — true path length for episode cost
    """
    B, N = C.shape[:2]

    if not torch.is_tensor(robot_radius):
        rr = o0.new_tensor(float(robot_radius))
    else:
        rr = robot_radius.to(o0.device, o0.dtype)
    R_eff = R + margin_factor * rr[:, None] if rr.ndim >= 1 else R + margin_factor * rr

    o   = o0.clone()
    v   = v0.clone()
    min_clear  = torch.full((B,), float("inf"), dtype=o.dtype, device=o.device)
    cum_risk   = torch.zeros(B, dtype=o.dtype, device=o.device)
    hard_count = torch.zeros(B, dtype=o.dtype, device=o.device)
    arc_length = torch.zeros(B, dtype=o.dtype, device=o.device)

    for s in range(int(H.max().item())):
        active = (s < H).to(o.dtype).unsqueeze(-1)           # (B,1)

        # ── Resample semantic fields at current position ───────────────
        sem = bilinear_sample_patch(rollout_patch, o, o0)     # (B,6)
        risk_val  = sem[:, 0].clamp(0.0, 1.0)                 # r̃(o)  ∈[0,1]
        sdf_val   = sem[:, 1].clamp(0.0, 50.0)                # φ(o)  metres
        grad_rx   = sem[:, 2]                                  # ∂r̃/∂x
        grad_ry   = sem[:, 3]                                  # ∂r̃/∂y
        sdf_gx    = sem[:, 4]                                  # ∂φ/∂x  (true oracle)
        sdf_gy    = sem[:, 5]                                  # ∂φ/∂y  (true oracle)
        risk_grad = torch.stack([grad_rx, grad_ry], dim=-1)   # (B,2)
        sdf_grad  = torch.stack([sdf_gx,  sdf_gy],  dim=-1)   # (B,2)  ∇φ → open space

        # ── Geometry forces ────────────────────────────────────────────
        F_goal = -beta.unsqueeze(-1) * (o - goal)

        if N == 0:
            F_geom = torch.zeros_like(o)
            dmin   = torch.full((B,), float("inf"), device=o.device)
        else:
            diff   = o.unsqueeze(1) - C
            r      = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)
            n_hat  = diff / r.unsqueeze(-1)
            d      = r - R_eff
            d      = torch.where(mask, d, torch.full_like(d, 1e6))
            _, dbdd = ipc_piecewise(d, d_hat.view(-1, 1))
            F_geom = -(alphas * dbdd).unsqueeze(-1) * n_hat
            F_geom = F_geom.sum(dim=1)
            dmin   = torch.where(mask, d,
                                  torch.full_like(d, float("inf"))).min(dim=1).values

        min_clear = torch.minimum(min_clear, dmin)

        # ── Material forces (resampled at o each step) ─────────────────
        # Soft: gradient descent on oracle risk field
        F_mat_soft = -lam_soft.unsqueeze(-1) * risk_grad       # (B,2)

        # Hard: SDF barrier — db/dφ < 0, negated → pushes away from hazard.
        # Uses true oracle ∇φ (ch4, ch5 of rollout_patch), not the -∇r̃ proxy.
        _, db_dphi = _sdf_barrier_grad(sdf_val, d_hat_sdf=d_hat_sdf)
        F_mat_hard = (-lam_hard.unsqueeze(-1) *
                       db_dphi.unsqueeze(-1) *
                       sdf_grad)                                # (B,2), true ∇φ

        # ── Total force + integration ──────────────────────────────────
        F_tot = F_goal + F_geom + F_mat_soft + F_mat_hard - gamma.unsqueeze(-1) * v
        a     = F_tot / mass
        v_new = v + active * dt.unsqueeze(-1) * a
        o_new = o + active * dt.unsqueeze(-1) * v_new

        # ── Accumulate true path length and semantic cost ──────────────
        step_disp  = torch.linalg.norm(o_new - o, dim=-1)     # (B,)
        act_1d     = active.squeeze(-1)
        arc_length = arc_length + act_1d * step_disp

        # cum_risk: semantic r̃(o), NOT sdf proxy
        cum_risk   = cum_risk   + act_1d * risk_val * step_disp
        # hard_count: binary — agent is near a hard hazard cell
        hard_count = hard_count + act_1d * (sdf_val < 1.0).to(o.dtype)

        v = v_new
        o = o_new

    return o, v, min_clear, cum_risk, hard_count, arc_length


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DFC2018RolloutCfg:
    min_h: int   = 2
    max_h: int   = 6
    dt_mult_range: Tuple[float, float] = (1.0, 3.0)
    split: str   = "train"   # "train" | "val" | "test"
    waypoint_mode: str = "oracle"  # "oracle" | "geom"
    selectivity_active_prob: float = 0.0


def _nearest_path_index(path_rc: np.ndarray, point_xy: List[float]) -> int:
    if path_rc.size == 0:
        return 0
    point_rc = np.asarray([float(point_xy[1]), float(point_xy[0])], dtype=np.float32)
    d2 = ((path_rc - point_rc[None, :]) ** 2).sum(axis=1)
    return int(np.argmin(d2))


def _geom_targets_for_checkpoint(
    geom_path_rc: np.ndarray,
    center_xy: List[float],
    *,
    dt: float,
    path_stride: int,
    patch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if geom_path_rc.shape[0] < 3:
        center = np.asarray(center_xy, dtype=np.float32)
        return center, center, np.zeros(2, dtype=np.float32)
    ci = _nearest_path_index(geom_path_rc, center_xy)
    gi = _pick_local_goal_index(geom_path_rc, ci, patch_size=patch_size)
    ti = min(ci + max(1, path_stride), geom_path_rc.shape[0] - 2)
    goal_xy = np.asarray([geom_path_rc[gi, 1], geom_path_rc[gi, 0]], dtype=np.float32)
    o_tgt_xy = np.asarray([geom_path_rc[ti, 1], geom_path_rc[ti, 0]], dtype=np.float32)
    v_tgt_xy = np.asarray(
        [
            (geom_path_rc[ti + 1, 1] - geom_path_rc[ti, 1]) / float(dt),
            (geom_path_rc[ti + 1, 0] - geom_path_rc[ti, 0]) / float(dt),
        ],
        dtype=np.float32,
    )
    return goal_xy, o_tgt_xy, v_tgt_xy


class DFC2018ShortRollouts(Dataset):
    """
    Short rollout dataset from DFC2018 stagewise episodes.

    Each __getitem__ returns a dict with all fields needed by
    integrate_surrogate_material:
      o0, v0, goal, C, R, W, d_hat, dt_prime, H  (geometry, as before)
      risk_patch  (2,P,P)  oracle risk + hazard mask
      risk_grad   (2,)     ∇r̃ at o0 in (x,y)
      sdf_val     ()       φ(q0) metres to hard hazard
      o_tgt, v_tgt         surrogate targets

    Loads from manifest.json → episode.pt → stagewise_checkpoints.jsonl.
    Filters by split field.
    """
    def __init__(self, root: str, cfg: Optional[DFC2018RolloutCfg] = None):
        super().__init__()
        self.cfg = cfg or DFC2018RolloutCfg()
        root     = Path(root)

        man_path = root / "manifest.json"
        with man_path.open() as f:
            records = json.load(f)

        # Filter by split
        records = [r for r in records
                   if r.get("split", "train") == self.cfg.split]
        if not records:
            raise RuntimeError(f"No records found for split='{self.cfg.split}'")

        self.items: List[Dict] = []
        scene_cache: Dict[str, Dict[str, np.ndarray]] = {}
        geom_cache: Dict[str, Dict[str, object]] = {}
        cache_path = root / f"geom_waypoint_cache_v1_{self.cfg.split}.pt"
        cache_dirty = False
        if self.cfg.waypoint_mode == "geom" and cache_path.exists():
            try:
                loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
                if isinstance(loaded, dict):
                    geom_cache = loaded
                print(f"  Loaded geom waypoint cache: {cache_path} ({len(geom_cache)} episodes)")
            except Exception as e:
                print(f"  [WARN] Failed to load geom waypoint cache {cache_path}: {e}")
                geom_cache = {}

        total_records = len(records)
        for rec_idx, rec in enumerate(records, start=1):
            if rec_idx == 1 or rec_idx % 50 == 0 or rec_idx == total_records:
                print(
                    f"  Building {self.cfg.split} dataset [{self.cfg.waypoint_mode}] "
                    f"{rec_idx}/{total_records}",
                    flush=True,
                )
            ep = torch.load(rec["path"], map_location="cpu",
                            weights_only=False)
            ck_path = ep["logs"]["checkpoints_jsonl"]
            with open(ck_path) as f:
                cks = [json.loads(line) for line in f]
            if len(cks) < 3:
                continue
            dt_base  = float(ep["meta"]["dt"])
            gamma_o  = float(ep["params"].get("gamma_o", 4.0))
            item = {"cks": cks, "dt": dt_base, "gamma_o": gamma_o}
            item["active_idxs"] = [
                i for i, ck in enumerate(cks[:-2])
                if float(ck.get("selectivity", {}).get("selectivity_active", 0.0)) > 0.5
            ]
            if self.cfg.waypoint_mode == "geom":
                episode_id = str(ep["meta"].get("episode_id", rec.get("episode_id", rec_idx)))
                cached = geom_cache.get(episode_id)
                if cached is not None:
                    geom_path_rc = cached.get("geom_path_rc")
                    if geom_path_rc is not None:
                        item["geom_path_rc"] = np.asarray(geom_path_rc, dtype=np.float32)
                        item["path_stride"] = int(cached.get("path_stride", ep["meta"].get("path_stride", 6)))
                else:
                    scene_id = ep["meta"]["scene_id"]
                    if scene_id not in scene_cache:
                        scene = torch.load(root / f"scene_{scene_id}.pt",
                                           map_location="cpu", weights_only=False)
                        scene_cache[scene_id] = scene["maps"]
                    maps = scene_cache[scene_id]
                    start_rc = tuple(int(x) for x in ep["meta"]["start_rc"])
                    goal_rc = tuple(int(x) for x in ep["meta"]["goal_rc"])
                    geom_path = dijkstra_geom(maps, start_rc, goal_rc)
                    if geom_path is not None and len(geom_path) >= 3:
                        geom_path_rc = np.asarray(geom_path, dtype=np.float32)
                        path_stride = int(ep["meta"].get("path_stride", 6))
                        item["geom_path_rc"] = geom_path_rc
                        item["path_stride"] = path_stride
                        geom_cache[episode_id] = {
                            "geom_path_rc": geom_path_rc,
                            "path_stride": path_stride,
                        }
                    else:
                        geom_cache[episode_id] = {"geom_path_rc": None}
                    cache_dirty = True
            self.items.append(item)

        if not self.items:
            raise RuntimeError(f"No valid episodes for split='{self.cfg.split}'")
        if self.cfg.waypoint_mode == "geom" and cache_dirty:
            torch.save(geom_cache, cache_path)
            print(f"  Saved geom waypoint cache: {cache_path} ({len(geom_cache)} episodes)")

    def __len__(self) -> int:
        return max(1, sum(len(it["cks"]) for it in self.items) // 6)

    @staticmethod
    def _vel_fd(c0, c1, dt):
        return (torch.tensor(c1, dtype=torch.float32) -
                torch.tensor(c0, dtype=torch.float32)) / float(dt)

    def __getitem__(self, _idx: int) -> Dict[str, torch.Tensor]:
        it   = random.choice(self.items)
        cks  = it["cks"]; dt = it["dt"]
        T    = len(cks)
        H    = random.randint(self.cfg.min_h, self.cfg.max_h)
        mult = random.uniform(*self.cfg.dt_mult_range)
        K    = max(1, int(round(H * mult)))
        max_t0 = max(1, T - K - 2)
        active_idxs = [i for i in it.get("active_idxs", []) if i <= max_t0]
        if active_idxs and random.random() < self.cfg.selectivity_active_prob:
            t0 = random.choice(active_idxs)
        else:
            t0 = random.randint(0, max_t0)
        t1   = min(T - 2, t0 + K)

        c0   = cks[t0]; c1 = cks[t1]; c_next = cks[min(T-1, t1+1)]

        # ── Geometry fields (unchanged from ShortRollouts unless waypoint_mode=geom) ─
        o0_xy = np.asarray(c0["center"], dtype=np.float32)
        goal_xy = np.asarray(c0["stage_exit"], dtype=np.float32)
        o_tgt_xy = np.asarray(c0["o_tgt"], dtype=np.float32)
        v_tgt_xy = np.asarray(c0["v_tgt"], dtype=np.float32)
        if self.cfg.waypoint_mode == "geom" and "geom_path_rc" in it:
            goal_xy, o_tgt_xy, v_tgt_xy = _geom_targets_for_checkpoint(
                it["geom_path_rc"],
                c0["center"],
                dt=dt,
                path_stride=int(it.get("path_stride", 6)),
                patch_size=64,
            )
        o0   = torch.tensor(o0_xy,    dtype=torch.float32)
        goal = torch.tensor(goal_xy,  dtype=torch.float32)
        o_tgt= torch.tensor(o_tgt_xy, dtype=torch.float32)
        v_tgt= torch.tensor(v_tgt_xy, dtype=torch.float32)
        v0   = self._vel_fd(c0["center"], cks[t0+1]["center"], dt)

        obs  = c0["obstacles_effective"]
        C_np = np.array(obs["C"],     dtype=np.float32) if obs["C"] else np.zeros((0,2),np.float32)
        R_np = np.array(obs["R_eff"], dtype=np.float32) if obs["R_eff"] else np.zeros((0,),np.float32)
        W_np = np.array(obs["W"],     dtype=np.float32) if obs["W"] else np.zeros((0,),np.float32)
        d_hat= torch.tensor(c0["barrier"]["barrier_d_hat"], dtype=torch.float32)

        gamma_o = torch.tensor(it["gamma_o"], dtype=torch.float32)
        dt_prime= torch.tensor(dt * mult,     dtype=torch.float32)
        H_t     = torch.tensor(H,             dtype=torch.long)

        # ── Material fields ─────────────────────────────────────────────
        # risk_patch (2,P,P): [r̃, hard_mask]  — model encoder input (always present)
        raw_patch  = np.array(c0["risk_patch"], dtype=np.float32)  # (2,P,P)

        # rollout_patch (6,P,P): [r̃, φ, ∂r̃/∂x, ∂r̃/∂y, ∂φ/∂x, ∂φ/∂y]
        # Prefer the true oracle version stored by build_dfc2018_stagewise.py.
        # Fall back to an approximation if the dataset predates this patch.
        if "rollout_patch" in c0:
            rp_np = np.array(c0["rollout_patch"], dtype=np.float32)   # (6,P,P)
        else:
            # Approximate fallback: derive from 2-channel risk_patch
            r_tilde = raw_patch[0]
            hard_ch = raw_patch[1]
            grad_x  = np.zeros_like(r_tilde)
            grad_y  = np.zeros_like(r_tilde)
            grad_x[:, 1:-1] = (r_tilde[:, 2:] - r_tilde[:, :-2]) / 2.0
            grad_y[1:-1, :] = (r_tilde[2:, :] - r_tilde[:-2, :]) / 2.0
            # φ proxy — no true SDF; use distance to hard mask (rough, in pixels)
            sdf_approx = (1.0 - hard_ch) * 10.0
            # ∇φ proxy ≈ -∇r̃  (risk increases toward hazards)
            rp_np = np.stack([r_tilde, sdf_approx,
                               grad_x, grad_y,
                               -grad_x, -grad_y], axis=0)   # (6,P,P)

        rollout_patch = torch.from_numpy(rp_np)              # (6,P,P)

        return {
            "o0":           o0,
            "v0":           v0,
            "goal":         goal,
            "C":            torch.from_numpy(C_np),
            "R":            torch.from_numpy(R_np),
            "W":            torch.from_numpy(W_np),
            "d_hat":        d_hat,
            "dt_prime":     dt_prime,
            "H":            H_t,
            "gamma_o":      gamma_o,
            "o_tgt":        o_tgt,
            "v_tgt":        v_tgt,
            "risk_patch":   torch.from_numpy(raw_patch),   # (2,P,P) — model encoder
            "rollout_patch":rollout_patch,                  # (6,P,P) — integrator
            "selectivity_active": torch.tensor(
                float(c0.get("selectivity", {}).get("selectivity_active", 0.0)),
                dtype=torch.float32,
            ),
            "selectivity_mask": torch.tensor(
                float(c0.get("selectivity", {}).get("selectivity_mask", 0.0)),
                dtype=torch.float32,
            ),
            "safe_dir": torch.tensor(
                c0.get("selectivity", {}).get("safe_dir", [0.0, 0.0]),
                dtype=torch.float32,
            ),
            "scaffold_dir": torch.tensor(
                c0.get("selectivity", {}).get("scaffold_dir", [0.0, 0.0]),
                dtype=torch.float32,
            ),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad variable-length obstacle lists to max N in batch."""
    max_N = max(b["C"].shape[0] for b in batch)
    B     = len(batch)

    def pad_obs(key, shape_suffix):
        out = []
        for b in batch:
            t = b[key]
            n = t.shape[0]
            if n < max_N:
                pad_shape = (max_N - n,) + shape_suffix
                t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)
            out.append(t)
        return torch.stack(out, dim=0)

    C    = pad_obs("C", (2,))
    R    = pad_obs("R", ())
    W    = pad_obs("W", ())
    mask = torch.zeros(B, max_N, dtype=torch.bool)
    for i, b in enumerate(batch):
        mask[i, :b["C"].shape[0]] = True

    scalar_keys = [
        "o0", "v0", "goal", "d_hat", "dt_prime", "gamma_o", "o_tgt", "v_tgt",
        "selectivity_active", "selectivity_mask", "safe_dir", "scaffold_dir",
    ]
    out = {k: torch.stack([b[k] for b in batch]) for k in scalar_keys}
    out["C"]            = C
    out["R"]            = R
    out["W"]            = W
    out["mask"]         = mask
    out["H"]            = torch.stack([b["H"] for b in batch])
    out["risk_patch"]   = torch.stack([b["risk_patch"]    for b in batch])  # (B,2,P,P)
    out["rollout_patch"]= torch.stack([b["rollout_patch"] for b in batch])  # (B,6,P,P)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def cvar_loss(
    costs: torch.Tensor,   # (B,)
    alpha: float = 0.95,
) -> torch.Tensor:
    """
    CVaR_α(J) = min_η { η + 1/(1-α) * E[(J-η)_+] }
    Concentrates on worst (1-α) fraction of rollouts.
    α=0.95 → worst 5% → emphasises rare catastrophic hazard encounters.
    """
    eta = torch.quantile(costs.detach(), alpha)
    excess = F.relu(costs - eta)
    return eta + excess.mean() / (1.0 - alpha)


def episode_cost(
    oT:         torch.Tensor,   # (B,2)
    goal:       torch.Tensor,   # (B,2)
    min_clear:  torch.Tensor,   # (B,)
    cum_risk:   torch.Tensor,   # (B,)
    hard_count: torch.Tensor,   # (B,)
    path_len:   torch.Tensor,   # (B,)  step count * dt
    cfg: "TrainCfgMaterial",
) -> torch.Tensor:
    """
    J = w_goal * ||oT - goal||²
      + w_len  * path_len
      + w_risk * cum_risk
      + w_hard * hard_count
    """
    goal_err = ((oT - goal) ** 2).sum(dim=-1)                 # (B,)
    J = (cfg.w_goal * goal_err
         + cfg.w_len  * path_len
         + cfg.w_risk * cum_risk
         + cfg.w_hard * hard_count)
    return J   # (B,)


def initial_material_force(
    o0: torch.Tensor,
    rollout_patch: torch.Tensor,
    lam_soft: torch.Tensor,
    lam_hard: torch.Tensor,
    *,
    d_hat_sdf: float,
) -> torch.Tensor:
    """Material force at the current state, matching integrate_surrogate_material."""
    sem = bilinear_sample_patch(rollout_patch, o0, o0)
    sdf_val = sem[:, 1].clamp(0.0, 50.0)
    risk_grad = torch.stack([sem[:, 2], sem[:, 3]], dim=-1)
    sdf_grad = torch.stack([sem[:, 4], sem[:, 5]], dim=-1)
    _, db_dphi = _sdf_barrier_grad(sdf_val, d_hat_sdf=d_hat_sdf)
    f_soft = -lam_soft.unsqueeze(-1) * risk_grad
    f_hard = -lam_hard.unsqueeze(-1) * db_dphi.unsqueeze(-1) * sdf_grad
    return f_soft + f_hard


def selectivity_loss(
    batch: Dict[str, torch.Tensor],
    lam_soft: torch.Tensor,
    lam_hard: torch.Tensor,
    cfg: "TrainCfgMaterial",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Supervise Stage-2 material-force activation on RELLIS selectivity labels."""
    mask = batch.get("selectivity_mask")
    if mask is None or float(mask.sum().detach().cpu()) <= 0.0:
        zero = lam_soft.sum() * 0.0
        return zero, {"L_select_active": 0.0, "L_select_inactive": 0.0}

    force = initial_material_force(
        batch["o0"],
        batch["rollout_patch"],
        lam_soft,
        lam_hard,
        d_hat_sdf=cfg.d_hat_sdf,
    )
    safe_dir = F.normalize(batch["safe_dir"], dim=-1, eps=1e-6)
    scaffold_dir = F.normalize(batch["scaffold_dir"], dim=-1, eps=1e-6)
    active = (batch["selectivity_active"] > 0.5).to(force.dtype) * mask
    inactive = (1.0 - (batch["selectivity_active"] > 0.5).to(force.dtype)) * mask

    dot_safe = (force * safe_dir).sum(dim=-1)
    parallel = (force * scaffold_dir).sum(dim=-1, keepdim=True) * scaffold_dir
    perp_norm_sq = ((force - parallel) ** 2).sum(dim=-1)
    lam_penalty = (
        (lam_soft / max(cfg.lam_soft_max, 1e-6)) ** 2
        + (lam_hard / max(cfg.lam_hard_max, 1e-6)) ** 2
    )

    active_loss = F.softplus(cfg.selectivity_margin - dot_safe)
    inactive_loss = perp_norm_sq + cfg.w_select_lambda * lam_penalty

    active_den = active.sum().clamp_min(1.0)
    inactive_den = inactive.sum().clamp_min(1.0)
    L_active = (active_loss * active).sum() / active_den
    L_inactive = (inactive_loss * inactive).sum() / inactive_den
    L_select = L_active + cfg.w_select_inactive * L_inactive
    return L_select, {
        "L_select_active": float(L_active.detach().cpu()),
        "L_select_inactive": float(L_inactive.detach().cpu()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainCfgMaterial:
    # ── General ──────────────────────────────────────────────────────────
    stage:   int   = 2          # 1=geometry only, 2=material
    epochs:  int   = 50
    bs:      int   = 64
    lr:      float = 1e-4
    workers: int   = 4
    device:  str   = "cuda" if torch.cuda.is_available() else "cpu"
    out:     str   = "checkpoints/material"
    log_every: int = 50

    # ── Episode cost weights ──────────────────────────────────────────────
    w_goal:  float = 2.0    # goal error
    w_len:   float = 0.01   # path length
    w_risk:  float = 1.0    # cumulative soft risk
    w_hard:  float = 5.0    # hard hazard step count
    cvar_alpha: float = 0.95

    # ── Auxiliary loss weights ────────────────────────────────────────────
    w_traj:    float = 1.0    # trajectory imitation (to risk-aware o_tgt)
    w_vel:     float = 0.5    # velocity imitation
    w_friction:float = 0.1    # damping regularization
    w_clear:   float = 5e-3   # geometric clearance penalty
    w_multi:   float = 0.5    # multi-start robustness
    w_lreg:    float = 0.01   # λ entropy regularization
    w_selectivity: float = 0.0  # RELLIS CAR/FAR force-selectivity supervision
    w_select_inactive: float = 1.0
    w_select_lambda: float = 0.05
    selectivity_margin: float = 0.005

    # ── Model / dynamics ─────────────────────────────────────────────────
    robot_radius:  float = 1.5
    margin_factor: float = 0.5
    d_hat_sdf:     float = 3.0    # SDF barrier activation distance
    lam_soft_max:  float = 5.0
    lam_hard_max:  float = 10.0
    patch_size:    int   = 32
    waypoint_mode: str   = "oracle"
    train_risk_only: bool = False
    selectivity_active_prob: float = 0.0
    selectivity_only: bool = False

    # NOTE — hazard ontology alignment with paper:
    # The dataset builder uses HARD_CLASSES={7,8,9,14,15,19} (expanded,
    # includes buildings/trains) for planning barriers and force repulsion.
    # The paper's stated rare catastrophic set is H={7,14,15} (water,
    # highways, railways only).  Evaluation metrics (hard_hits, CVaR
    # reported in the paper) should be computed against H={7,14,15}.
    # The expanded set provides a safety margin during training and should
    # NOT be used when reporting "rare catastrophic hazard entries."
    ms_count:    int   = 10
    ms_h:        int   = 3
    ms_dt_mult:  float = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class MaterialTrainer:
    def __init__(self, model: CoefEnergyNetMaterial, cfg: TrainCfgMaterial):
        self.model = model.to(cfg.device)
        self.cfg   = cfg

        # Stage 1: freeze risk heads → Setting 1 behaviour
        self._set_risk_heads_frozen(cfg.stage == 1)
        if cfg.train_risk_only and cfg.stage == 2:
            self._set_train_risk_only()

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters remain after freeze configuration.")
        self.opt   = torch.optim.Adam(trainable, lr=cfg.lr)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.1)
        os.makedirs(cfg.out, exist_ok=True)

    def _set_risk_heads_frozen(self, freeze: bool):
        for p in self.model.risk_enc.parameters():
            p.requires_grad = not freeze
        for p in self.model.lam_soft_head.parameters():
            p.requires_grad = not freeze
        for p in self.model.lam_hard_head.parameters():
            p.requires_grad = not freeze
        status = "frozen" if freeze else "trainable"
        print(f"  Risk heads: {status}")

    def _set_train_risk_only(self):
        for p in self.model.parameters():
            p.requires_grad = False
        for module in (self.model.risk_enc, self.model.lam_soft_head, self.model.lam_hard_head):
            for p in module.parameters():
                p.requires_grad = True
        print("  Fine-tune mode: risk encoder + λ heads only")

    def _to(self, batch: Dict) -> Dict:
        dev = self.cfg.device
        return {k: v.to(dev) if torch.is_tensor(v) else v
                for k, v in batch.items()}

    def step_batch(self, batch: Dict) -> Dict[str, float]:
        cfg = self.cfg
        batch = self._to(batch)

        B    = batch["o0"].shape[0]
        N    = batch["C"].shape[1]
        rr   = batch["o0"].new_tensor(cfg.robot_radius).expand(B)

        # Model forward
        alphas, beta, gamma, lam_soft, lam_hard, _mu_lat = self.model(
            obs_feats  = torch.cat([
                batch["C"],
                batch["R"].unsqueeze(-1),
                batch["W"].unsqueeze(-1),
                (batch["goal"].unsqueeze(1) - batch["C"]),
            ], dim=-1) if N > 0 else batch["o0"].new_zeros(B, 0, 6),
            obs_mask   = batch["mask"],
            goal_feats = torch.cat([
                batch["goal"] - batch["o0"],
                torch.linalg.norm(batch["goal"] - batch["o0"],
                                   dim=-1, keepdim=True),
                batch["o0"].new_ones(B, 1),
            ], dim=-1),
            risk_patch = batch["risk_patch"],
        )

        # Stage 1: zero out material outputs (frozen, but explicit for clarity)
        if cfg.stage == 1:
            lam_soft = torch.zeros_like(lam_soft)
            lam_hard = torch.zeros_like(lam_hard)

        # Surrogate rollout — now returns arc_length as 6th element
        oT, vT, min_clear, cum_risk, hard_count, arc_length = integrate_surrogate_material(
            o0             = batch["o0"],
            v0             = batch["v0"],
            goal           = batch["goal"],
            C              = batch["C"],
            R              = batch["R"],
            mask           = batch["mask"],
            alphas         = alphas,
            beta           = beta,
            gamma          = gamma,
            lam_soft       = lam_soft,
            lam_hard       = lam_hard,
            rollout_patch  = batch["rollout_patch"],   # (B,6,P,P) — resampled per step
            d_hat          = batch["d_hat"],
            dt             = batch["dt_prime"],
            H              = batch["H"],
            robot_radius   = rr,
            margin_factor  = cfg.margin_factor,
            d_hat_sdf      = cfg.d_hat_sdf,
        )

        # ── Auxiliary losses (geometry, unchanged from train_coef_energy) ─
        L_traj = F.mse_loss(oT, batch["o_tgt"])
        L_vel  = F.mse_loss(vT, batch["v_tgt"])
        L_fric = F.mse_loss(gamma, batch["gamma_o"].clamp(0, 20))
        L_clear = F.softplus(-min_clear / 0.05).mean()

        # Multi-start geometric robustness
        L_multi = multi_start_penalty(
            batch["o0"], batch["v0"], batch["goal"],
            batch["C"], batch["R"], batch["mask"],
            alphas, beta, gamma,
            batch["d_hat"], batch["dt_prime"], batch["H"],
            robot_radius  = rr,
            margin_factor = cfg.margin_factor,
            ms_count      = cfg.ms_count,
            ms_h          = cfg.ms_h,
            ms_dt_mult    = cfg.ms_dt_mult,
        )

        # ── Primary material navigation loss (CVaR) ───────────────────────
        # arc_length is true surrogate path length — distinguishes long detours
        # from short direct paths unlike the old dt_prime * H proxy.
        J    = episode_cost(oT, batch["goal"], min_clear,
                             cum_risk, hard_count, arc_length, cfg)
        L_nav = cvar_loss(J, alpha=cfg.cvar_alpha)

        # ── λ regularization (prevent collapse) ──────────────────────────
        # Encourage non-trivial λ distribution (not all-zero or all-max)
        lam_norm_soft = lam_soft / cfg.lam_soft_max   # ∈ (0,1)
        lam_norm_hard = lam_hard / cfg.lam_hard_max
        eps = 1e-6
        L_lreg = -(lam_norm_soft.clamp(eps, 1-eps).log().mean() +
                   (1-lam_norm_soft).clamp(eps,1-eps).log().mean() +
                   lam_norm_hard.clamp(eps,1-eps).log().mean() +
                   (1-lam_norm_hard).clamp(eps,1-eps).log().mean()) * 0.25

        L_select, select_metrics = selectivity_loss(batch, lam_soft, lam_hard, cfg)

        # ── Total loss ────────────────────────────────────────────────────
        if cfg.stage == 1:
            # Geometry only: imitation + safety
            L = (cfg.w_traj    * L_traj
               + cfg.w_vel     * L_vel
               + cfg.w_friction* L_fric
               + cfg.w_clear   * L_clear
               + cfg.w_multi   * L_multi)
        else:
            # Stage 2: add CVaR navigation loss, drop pure imitation dominance
            if cfg.selectivity_only:
                L = cfg.w_selectivity * L_select + cfg.w_lreg * L_lreg
            else:
                L = (cfg.w_traj    * L_traj * 0.3    # reduced: o_tgt is risk-aware, so
                   + cfg.w_vel     * L_vel  * 0.3    # imitation is consistent but secondary
                   + cfg.w_friction* L_fric
                   + cfg.w_clear   * L_clear
                   + cfg.w_multi   * L_multi
                   + L_nav                           # primary: CVaR episode cost
                   + cfg.w_lreg   * L_lreg
                   + cfg.w_selectivity * L_select)

        self.opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.opt.step()

        return {
            "loss":    L.item(),
            "L_nav":   L_nav.item(),
            "L_traj":  L_traj.item(),
            "L_vel":   L_vel.item(),
            "L_fric":  L_fric.item(),
            "L_clear": L_clear.item(),
            "L_multi": L_multi.item() if torch.is_tensor(L_multi) else float(L_multi),
            "L_lreg":  L_lreg.item() if cfg.stage > 1 else 0.0,
            "L_select": L_select.item() if cfg.stage > 1 else 0.0,
            **select_metrics,
            "lam_soft_mean": lam_soft.mean().item(),
            "lam_hard_mean": lam_hard.mean().item(),
            "cum_risk_mean": cum_risk.mean().item(),
            "hard_cnt_mean": hard_count.mean().item(),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   Optional[DataLoader] = None,
    ):
        cfg   = self.cfg
        best_val = float("inf")

        for epoch in range(cfg.epochs):
            self.model.train()
            running: Dict[str, float] = {}
            n_batches = 0

            for batch in train_loader:
                metrics = self.step_batch(batch)
                for k, v in metrics.items():
                    running[k] = running.get(k, 0.0) + v
                n_batches += 1

                if n_batches % cfg.log_every == 0:
                    avg = {k: v/n_batches for k, v in running.items()}
                    print(f"  ep{epoch} [{n_batches}]  "
                          f"loss={avg['loss']:.4f}  "
                          f"L_nav={avg['L_nav']:.4f}  "
                          f"L_sel={avg.get('L_select', 0.0):.4f}  "
                          f"risk={avg['cum_risk_mean']:.3f}  "
                          f"hard={avg['hard_cnt_mean']:.2f}  "
                          f"λs={avg['lam_soft_mean']:.3f}  "
                          f"λh={avg['lam_hard_mean']:.3f}")

            self.sched.step()
            avg = {k: v/max(n_batches,1) for k, v in running.items()}

            # Validation
            val_loss = self._validate(val_loader) if val_loader else avg["loss"]

            print(f"Epoch {epoch:3d}  "
                  f"train_loss={avg['loss']:.4f}  val_loss={val_loss:.4f}  "
                  f"L_nav={avg['L_nav']:.4f}  "
                  f"L_sel={avg.get('L_select', 0.0):.4f}  "
                  f"λ_soft={avg['lam_soft_mean']:.3f}  "
                  f"λ_hard={avg['lam_hard_mean']:.3f}")

            # Save checkpoints
            ck = {
                "epoch":            epoch,
                "model_state_dict": self.model.state_dict(),
                "opt_state_dict":   self.opt.state_dict(),
                "train_metrics":    avg,
                "val_loss":         val_loss,
                "cfg":              cfg.__dict__,
            }
            torch.save(ck, os.path.join(cfg.out, f"epoch_{epoch:03d}.pt"))
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ck, os.path.join(cfg.out, "best.pt"))
                print(f"  → best checkpoint  (val={best_val:.4f})")

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0; n = 0
        for batch in loader:
            batch = self._to(batch)
            B     = batch["o0"].shape[0]
            N     = batch["C"].shape[1]
            rr    = batch["o0"].new_tensor(self.cfg.robot_radius).expand(B)

            alphas, beta, gamma, lam_soft, lam_hard, _mu_lat = self.model(
                obs_feats = torch.cat([
                    batch["C"], batch["R"].unsqueeze(-1),
                    batch["W"].unsqueeze(-1),
                    (batch["goal"].unsqueeze(1) - batch["C"]),
                ], dim=-1) if N > 0 else batch["o0"].new_zeros(B, 0, 6),
                obs_mask   = batch["mask"],
                goal_feats = torch.cat([
                    batch["goal"] - batch["o0"],
                    torch.linalg.norm(batch["goal"] - batch["o0"],
                                       dim=-1, keepdim=True),
                    batch["o0"].new_ones(B, 1),
                ], dim=-1),
                risk_patch = batch["risk_patch"],
            )
            if self.cfg.stage == 1:
                lam_soft = torch.zeros_like(lam_soft)
                lam_hard = torch.zeros_like(lam_hard)

            oT, _, min_clear, cum_risk, hard_count, arc_length = integrate_surrogate_material(
                batch["o0"], batch["v0"], batch["goal"],
                batch["C"], batch["R"], batch["mask"],
                alphas, beta, gamma, lam_soft, lam_hard,
                rollout_patch  = batch["rollout_patch"],
                d_hat          = batch["d_hat"],
                dt             = batch["dt_prime"],
                H              = batch["H"],
                robot_radius=rr, margin_factor=self.cfg.margin_factor,
                d_hat_sdf=self.cfg.d_hat_sdf,
            )
            # Use actual min_clear and arc_length — consistent with training objective
            J = episode_cost(oT, batch["goal"], min_clear,
                              cum_risk, hard_count, arc_length, self.cfg)
            val_obj = cvar_loss(J, self.cfg.cvar_alpha)
            if self.cfg.stage > 1 and self.cfg.w_selectivity > 0.0:
                L_select, _ = selectivity_loss(batch, lam_soft, lam_hard, self.cfg)
                if self.cfg.selectivity_only:
                    val_obj = self.cfg.w_selectivity * L_select
                else:
                    val_obj = val_obj + self.cfg.w_selectivity * L_select
            total += val_obj.item()
            n += 1
        return total / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    required=True,
                    help="Path to data/dfc2018_stagewise (manifest.json)")
    ap.add_argument("--stage",   type=int, default=2, choices=[1, 2],
                    help="1=geometry only (Setting 1), 2=material-aware (Setting 2)")
    ap.add_argument("--epochs",  type=int,   default=50)
    ap.add_argument("--bs",      type=int,   default=64)
    ap.add_argument("--lr",      type=float, default=1e-4)
    ap.add_argument("--workers", type=int,   default=4)
    ap.add_argument("--out",     default="checkpoints/material")
    ap.add_argument("--ckpt_s1", default=None,
                    help="Stage 1 checkpoint to warm-start Stage 2 geometry weights")
    ap.add_argument("--ckpt_init", default=None,
                    help="Optional full checkpoint initialization before training/fine-tuning")
    ap.add_argument("--w_goal",  type=float, default=2.0)
    ap.add_argument("--w_risk",  type=float, default=1.0)
    ap.add_argument("--w_hard",  type=float, default=5.0)
    ap.add_argument("--w_multi", type=float, default=0.5)
    ap.add_argument("--w_selectivity", type=float, default=0.0,
                    help="Weight for RELLIS force-selectivity supervision.")
    ap.add_argument("--w_select_inactive", type=float, default=1.0,
                    help="Weight on inactive R2/R3 force suppression inside selectivity loss.")
    ap.add_argument("--w_select_lambda", type=float, default=0.05,
                    help="Inactive-state lambda shrinkage inside selectivity loss.")
    ap.add_argument("--selectivity_margin", type=float, default=0.005,
                    help="Required positive dot(F_ctx, d_safe) margin for active R1 labels.")
    ap.add_argument("--cvar_alpha", type=float, default=0.95)
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--waypoint_mode", type=str, default="oracle",
                    choices=["oracle", "geom"],
                    help="Local-goal source during training: oracle stage exits or geometry-derived waypoints.")
    ap.add_argument("--train_risk_only", action="store_true",
                    help="Freeze geometry backbone and fine-tune only risk encoder / lambda heads.")
    ap.add_argument("--selectivity_active_prob", type=float, default=0.0,
                    help="Probability of sampling an active R1 selectivity checkpoint when available.")
    ap.add_argument("--selectivity_only", action="store_true",
                    help="Train only the explicit RELLIS selectivity objective plus light lambda regularization.")
    args = ap.parse_args()

    cfg = TrainCfgMaterial(
        stage      = args.stage,
        epochs     = args.epochs,
        bs         = args.bs,
        lr         = args.lr,
        workers    = args.workers,
        out        = args.out,
        w_goal     = args.w_goal,
        w_risk     = args.w_risk,
        w_hard     = args.w_hard,
        w_multi    = args.w_multi,
        w_selectivity = args.w_selectivity,
        w_select_inactive = args.w_select_inactive,
        w_select_lambda = args.w_select_lambda,
        selectivity_margin = args.selectivity_margin,
        cvar_alpha = args.cvar_alpha,
        patch_size = args.patch_size,
        waypoint_mode = args.waypoint_mode,
        train_risk_only = args.train_risk_only,
        selectivity_active_prob = args.selectivity_active_prob,
        selectivity_only = args.selectivity_only,
    )

    print(f"Stage {cfg.stage}  device={cfg.device}")

    model = CoefEnergyNetMaterial(
        patch_size   = cfg.patch_size,
        lam_soft_max = cfg.lam_soft_max,
        lam_hard_max = cfg.lam_hard_max,
    )

    if args.ckpt_init:
        print(f"Loading full checkpoint initialization from {args.ckpt_init} …")
        ck = torch.load(args.ckpt_init, map_location=cfg.device, weights_only=False)
        state_dict = ck.get("model_state_dict", ck.get("model", ck))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded init checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys")
    elif args.ckpt_s1 and cfg.stage == 2:
        print("Loading Stage 1 geometry weights …")
        load_geometry_weights(model, args.ckpt_s1, cfg.device)

    train_ds = DFC2018ShortRollouts(args.root,
                                     DFC2018RolloutCfg(
                                         split="train",
                                         waypoint_mode=cfg.waypoint_mode,
                                         selectivity_active_prob=cfg.selectivity_active_prob,
                                     ))
    val_ds   = DFC2018ShortRollouts(args.root,
                                     DFC2018RolloutCfg(split="val", waypoint_mode=cfg.waypoint_mode))

    train_loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True,
                               num_workers=cfg.workers, collate_fn=collate_fn,
                               pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.bs, shuffle=False,
                               num_workers=cfg.workers, collate_fn=collate_fn,
                               pin_memory=True)

    print(f"Train: {len(train_ds)} samples  Val: {len(val_ds)} samples")

    trainer = MaterialTrainer(model, cfg)
    trainer.train(train_loader, val_loader)
    print("Done.")


if __name__ == "__main__":
    main()
