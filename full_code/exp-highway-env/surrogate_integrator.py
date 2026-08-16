"""
surrogate_integrator.py

Stage 2 surrogate integrator for highway-env. Replaces DFC2018's
`integrate_surrogate_material` while keeping its input/output contract
identical so downstream code (the model, the CVaR loss, the dataset
collate) can be reused unchanged.

Differences from `train_material.py:integrate_surrogate_material`:

  1. Dynamics: bicycle (matches highway-env Vehicle.step) instead of a
     point-mass. Position/velocity are replaced by (position, heading, speed);
     velocity is reconstructed from (heading, speed) for backward compatibility
     with the cost function. The learned `gamma` coefficient still contributes
     explicit velocity damping inside the shared force law.

  2. Patch frame: the rollout patch is laid out in EGO frame (longitudinal
     axis = ego heading at observation time), so to bilinear-sample at a
     world-frame position `o`, we first rotate `o - o0` into ego frame.
     DFC's patch was world-aligned and didn't need this rotation.

  3. F_geom: per-vehicle IPC barrier same as DFC, but vehicles are also
     dynamic (they'll move during the surrogate rollout). At Stage 2 we
     freeze them at observation time and treat them as static-for-this-rollout
     — a deliberate simplification that keeps the gradient clean at the cost
     of slight inaccuracy. The risk patch's integrated lookahead already
     anticipates other-vehicle motion, so this isn't double-counting.

Returns the same six tensors as the DFC integrator:
    (oT, vT, min_clear_geom, cum_risk, hard_count, arc_length)

The orientation handling is internal: input v0 is decomposed into
(heading_0, speed_0); output vT is recomposed from (heading_T, speed_T).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from bicycle_surrogate import (
    ACCEL_RANGE,
    STEER_RANGE,
    VEHICLE_LENGTH,
    bicycle_step_train,
    force_to_action,
)


LATERAL_PREFERENCE_BIAS = 0.05
LATERAL_LANE_WIDTH = 4.0
LATERAL_LOOKAHEAD = 10.0


# ──────────────────────────────────────────────────────────────────────────
# Helpers — IPC barrier and SDF-barrier gradient
# ──────────────────────────────────────────────────────────────────────────

def ipc_piecewise(
    d: torch.Tensor,
    d_hat: torch.Tensor,
    vp: float = -5e2,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """IPC barrier potential and its derivative.

    Mirrors `train_coef_energy.py:ipc_piecewise` exactly. The barrier
    activates only when d < d_hat:
        b(d) = -vp * (d - d_hat)² * log(d / d_hat)        d ∈ (0, d_hat)
        b'(d) = -vp * [2 (d - d_hat) log(d/d_hat) + (d - d_hat)² / d]
    For d >= d_hat, both are zero.
    """
    active = (d < d_hat).to(d.dtype)
    d_safe = d.clamp(min=eps)
    diff = d_safe - d_hat
    log_term = torch.log(d_safe / d_hat.clamp(min=eps))
    b_val = -vp * diff * diff * log_term
    db_dd = -vp * (2 * diff * log_term + diff * diff / d_safe)
    return active * b_val, active * db_dd


def sdf_barrier_grad(
    sdf_val: torch.Tensor,
    d_hat_sdf: float = 3.0,
    k_sharp: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softplus SDF barrier value and ∂b/∂φ. Mirrors DFC's _sdf_barrier_grad."""
    inner = k_sharp * (d_hat_sdf - sdf_val)
    b_val = F.softplus(inner) / k_sharp
    db_dphi = -torch.sigmoid(inner)        # negative; multiplied by ∇φ to repel
    return b_val, db_dphi


# ──────────────────────────────────────────────────────────────────────────
# Bilinear patch sampling — corrected for ego-frame patches
# ──────────────────────────────────────────────────────────────────────────

def _bilinear_sample_ego_patch(
    patch: torch.Tensor,         # (B, C, Hp, Wp)  in EGO frame
    o: torch.Tensor,             # (B, 2)  current world-frame position
    o0: torch.Tensor,            # (B, 2)  patch centre (ego at observation time)
    heading_0: torch.Tensor,     # (B,)    ego heading at observation time
    cell_size_lon: float,
    cell_size_lat: float,
    patch_lon_offset_frac: float = 0.05,
) -> torch.Tensor:
    """Bilinear-sample an ego-frame patch at a world-frame position.

    The patch is laid out in ego frame: column 0 corresponds to
    longitudinal coordinate `-patch_lon_offset_frac * patch_lon_m`
    (slightly behind the ego), increasing along the ego heading.
    Row 0 is at the lateral coordinate `-patch_lat_m / 2`, increasing
    perpendicular (left-handed: cross-product of heading and lateral
    is +z).

    To sample at world position `o`, we:
      1. Compute the world-frame offset: `o - o0`.
      2. Rotate into ego frame using `heading_0` (the heading AT THE TIME
         the patch was built, NOT the current heading — the patch is fixed).
      3. Convert to normalized [-1, 1] pixel coordinates for grid_sample.

    Returns (B, C) sampled values.
    """
    B, C, Hp, Wp = patch.shape

    # Step 1: world-frame offset
    delta = o - o0                                      # (B, 2)

    # Step 2: rotate into ego frame
    cos_h = torch.cos(heading_0)
    sin_h = torch.sin(heading_0)
    # ego-x (longitudinal) = world-x cos + world-y sin
    # ego-y (lateral)      = -world-x sin + world-y cos
    delta_lon = delta[:, 0] * cos_h + delta[:, 1] * sin_h
    delta_lat = -delta[:, 0] * sin_h + delta[:, 1] * cos_h

    # Step 3: convert to pixel coordinates within the patch
    # Patch ego-x range: [-α·L, (1-α)·L] where α = patch_lon_offset_frac, L = patch_lon_m
    #   col 0 corresponds to ego-x = -α·L + 0.5·cell_size_lon
    #   col Wp-1 corresponds to ego-x = (1-α)·L - 0.5·cell_size_lon
    # Patch ego-y range: [-W/2, W/2] where W = patch_lat_m
    patch_lon_m = Wp * cell_size_lon
    patch_lat_m = Hp * cell_size_lat
    ego_offset_m = patch_lon_offset_frac * patch_lon_m

    # Continuous pixel coords (col_idx, row_idx) in [0, Wp-1] × [0, Hp-1]
    col = (delta_lon + ego_offset_m) / cell_size_lon - 0.5
    row = (delta_lat + 0.5 * patch_lat_m) / cell_size_lat - 0.5

    # Normalize to [-1, 1] for grid_sample
    half_w = (Wp - 1) / 2.0
    half_h = (Hp - 1) / 2.0
    gx = (col - half_w) / (half_w + 1e-8)
    gy = (row - half_h) / (half_h + 1e-8)
    grid = torch.stack([gx, gy], dim=-1).view(B, 1, 1, 2)
    sampled = F.grid_sample(
        patch, grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    return sampled.view(B, C)


def _straight_through_clip(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Hard clip in the forward pass, identity gradient in the backward pass."""
    clipped = x.clamp(lo, hi)
    return x + (clipped - x).detach()


def _lateral_probe_stats(
    *,
    o: torch.Tensor,
    heading: torch.Tensor,
    o0: torch.Tensor,
    heading_0: torch.Tensor,
    rollout_patch: torch.Tensor,
    cell_size_lon: float = 1.0,
    cell_size_lat: float = 1.0,
    patch_lon_offset_frac: float = 0.05,
    lateral_lookahead: float = LATERAL_LOOKAHEAD,
    lateral_lane_width: float = LATERAL_LANE_WIDTH,
    lateral_preference_bias: float = LATERAL_PREFERENCE_BIAS,
) -> Dict[str, torch.Tensor]:
    """Sample the left/right probe risks that drive the lateral channel."""
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    forward_world = torch.stack([cos_h, sin_h], dim=-1)
    n_lat_world = torch.stack([-sin_h, cos_h], dim=-1)

    probe_center = o + lateral_lookahead * forward_world
    probe_left = probe_center - lateral_lane_width * n_lat_world
    probe_right = probe_center + lateral_lane_width * n_lat_world

    sem_left = _bilinear_sample_ego_patch(
        rollout_patch, probe_left, o0, heading_0,
        cell_size_lon=cell_size_lon,
        cell_size_lat=cell_size_lat,
        patch_lon_offset_frac=patch_lon_offset_frac,
    )
    sem_right = _bilinear_sample_ego_patch(
        rollout_patch, probe_right, o0, heading_0,
        cell_size_lon=cell_size_lon,
        cell_size_lat=cell_size_lat,
        patch_lon_offset_frac=patch_lon_offset_frac,
    )

    risk_left = sem_left[:, 0].clamp(0.0, 1.0)
    risk_right = sem_right[:, 0].clamp(0.0, 1.0)
    side_score = (risk_right - risk_left) + lateral_preference_bias
    return {
        "forward_world": forward_world,
        "n_lat_world": n_lat_world,
        "risk_left": risk_left,
        "risk_right": risk_right,
        "side_score": side_score,
    }


def _ttc_longitudinal_force(
    *,
    o: torch.Tensor,
    heading: torch.Tensor,
    speed: torch.Tensor,
    C: torch.Tensor,
    V_neighbors: Optional[torch.Tensor],
    R_eff: torch.Tensor,
    mask: torch.Tensor,
    forward_world: torch.Tensor,
    risk_left: torch.Tensor,
    risk_right: torch.Tensor,
    ttc_gain: float = 0.0,
    ttc_threshold_s: float = 3.0,
    ttc_softness_s: float = 0.5,
    ttc_min_closing_speed: float = 0.5,
    ttc_lane_halfwidth: float = 2.0,
    ttc_boxed_risk_thresh: float = 0.25,
    ttc_boxed_gate_sharpness: float = 20.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Analytic TTC-triggered braking force for same-lane leaders ahead."""
    B = o.shape[0]
    zeros = torch.zeros_like(speed)
    infs = torch.full_like(speed, float("inf"))

    debug = {
        "leader_gap_lon": infs,
        "leader_lat_abs": infs,
        "leader_speed_lon": zeros,
        "closing_speed": zeros,
        "ttc": infs,
        "boxed_gate": zeros,
        "ttc_gate": zeros,
        "leader_found": zeros,
    }
    if (
        ttc_gain <= 0.0
        or V_neighbors is None
        or C.numel() == 0
    ):
        return torch.zeros_like(o), debug

    n_lat_world = torch.stack([-forward_world[:, 1], forward_world[:, 0]], dim=-1)
    rel = C - o.unsqueeze(1)
    gap_lon_center = (rel * forward_world.unsqueeze(1)).sum(dim=-1)
    gap_lat = (rel * n_lat_world.unsqueeze(1)).sum(dim=-1)
    gap_lon = gap_lon_center - R_eff

    leader_mask = (
        mask
        & (gap_lon > 0.0)
        & (gap_lat.abs() <= float(ttc_lane_halfwidth))
    )
    rank_metric = torch.where(leader_mask, gap_lon, torch.full_like(gap_lon, float("inf")))
    leader_idx = rank_metric.argmin(dim=1)
    leader_found = leader_mask.any(dim=1)

    gather_idx = leader_idx.unsqueeze(-1)
    leader_gap_lon = gap_lon.gather(1, gather_idx).squeeze(1)
    leader_lat_abs = gap_lat.abs().gather(1, gather_idx).squeeze(1)
    leader_speed_lon = (V_neighbors * forward_world.unsqueeze(1)).sum(dim=-1)
    leader_speed_lon = leader_speed_lon.gather(1, gather_idx).squeeze(1)

    closing_speed = (speed - leader_speed_lon).clamp_min(0.0)
    leader_gap_lon = torch.where(leader_found, leader_gap_lon.clamp_min(eps), infs)
    ttc = torch.where(
        leader_found & (closing_speed > float(ttc_min_closing_speed)),
        leader_gap_lon / closing_speed.clamp_min(eps),
        infs,
    )

    boxed_signal = torch.minimum(risk_left, risk_right)
    boxed_gate = torch.sigmoid(
        float(ttc_boxed_gate_sharpness) * (boxed_signal - float(ttc_boxed_risk_thresh))
    )
    ttc_gate = torch.sigmoid(
        (float(ttc_threshold_s) - ttc) / max(float(ttc_softness_s), eps)
    )
    ttc_gate = torch.where(leader_found, ttc_gate, zeros)

    brake_mag = float(ttc_gain) * boxed_gate * ttc_gate
    F_ttc = -brake_mag.unsqueeze(-1) * forward_world

    debug.update({
        "leader_gap_lon": leader_gap_lon,
        "leader_lat_abs": leader_lat_abs,
        "leader_speed_lon": leader_speed_lon,
        "closing_speed": closing_speed,
        "ttc": ttc,
        "boxed_gate": boxed_gate,
        "ttc_gate": ttc_gate,
        "leader_found": leader_found.to(speed.dtype),
    })
    return F_ttc, debug


# ──────────────────────────────────────────────────────────────────────────
# Force law — shared by rollout and closed-loop deployment
# ──────────────────────────────────────────────────────────────────────────

def compute_surrogate_highway_force(
    *,
    o:          torch.Tensor,    # (B, 2) current world-frame position
    heading:    torch.Tensor,    # (B,) current heading
    speed:      torch.Tensor,    # (B,) current speed; accepted for API symmetry
    o0:         torch.Tensor,    # (B, 2) patch centre / observation position
    heading_0:  torch.Tensor,    # (B,) patch heading at observation time
    goal:       torch.Tensor,    # (B, 2)
    C:          torch.Tensor,    # (B, N, 2)
    R_eff:      torch.Tensor,    # (B, N) radii after robot/margin inflation
    mask:       torch.Tensor,    # (B, N) bool
    alphas:     torch.Tensor,    # (B, N)
    beta:       torch.Tensor,    # (B,)
    gamma:      torch.Tensor,    # (B,) velocity damping
    lam_soft:   torch.Tensor,    # (B,)
    lam_hard:   torch.Tensor,    # (B,)
    rollout_patch: torch.Tensor, # (B, 6, Hp, Wp)
    d_hat:      torch.Tensor,    # (B,)
    V_neighbors: Optional[torch.Tensor] = None,  # (B, N, 2) optional obstacle velocities
    mu_lat:     Optional[torch.Tensor] = None,  # (B,) optional lateral channel scale
    cell_size_lon: float = 1.0,
    cell_size_lat: float = 1.0,
    patch_lon_offset_frac: float = 0.05,
    d_hat_sdf: float = 3.0,
    ipc_grad_clip: float = 100.0,
    lateral_lookahead: float = LATERAL_LOOKAHEAD,
    lateral_lane_width: float = LATERAL_LANE_WIDTH,
    lateral_preference_bias: float = LATERAL_PREFERENCE_BIAS,
    ttc_gain: float = 0.0,
    ttc_threshold_s: float = 3.0,
    ttc_softness_s: float = 0.5,
    ttc_min_closing_speed: float = 0.5,
    ttc_lane_halfwidth: float = 2.0,
    ttc_boxed_risk_thresh: float = 0.25,
    ttc_boxed_gate_sharpness: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the instantaneous world-frame surrogate force.

    This is the force law used by :func:`integrate_surrogate_highway` at each
    rollout step. Closed-loop evaluation can call it once at observation time
    to obtain the first-step force, convert that force with ``force_to_action``,
    and send the resulting continuous action to highway-env.

    Returns ``(F_tot, dmin, risk_val, sdf_val)``.
    """
    B, N = C.shape[:2]

    F_goal = -beta.unsqueeze(-1) * (o - goal)                      # (B, 2)
    vel_world = speed.unsqueeze(-1) * torch.stack(
        [torch.cos(heading), torch.sin(heading)], dim=-1
    )
    F_damp = -gamma.unsqueeze(-1) * vel_world                      # (B, 2)

    if N == 0:
        F_geom = torch.zeros_like(o)
        dmin = torch.full((B,), float("inf"), device=o.device, dtype=o.dtype)
    else:
        diff = o.unsqueeze(1) - C                                  # (B, N, 2)
        r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)
        n_hat = diff / r.unsqueeze(-1)
        d = r - R_eff
        d = torch.where(mask, d, torch.full_like(d, 1e6))
        _, dbdd = ipc_piecewise(d, d_hat.view(-1, 1))
        dbdd = dbdd.clamp(-ipc_grad_clip, ipc_grad_clip)
        # db/dd is positive in the active IPC region, and n_hat points from
        # obstacle centre to ego. Positive sign is therefore repulsive.
        F_geom = (alphas * dbdd).unsqueeze(-1) * n_hat             # (B, N, 2)
        F_geom = F_geom.sum(dim=1)                                 # (B, 2)
        dmin = torch.where(mask, d, torch.full_like(d, float("inf"))).min(dim=1).values

    sem = _bilinear_sample_ego_patch(
        rollout_patch, o, o0, heading_0,
        cell_size_lon=cell_size_lon,
        cell_size_lat=cell_size_lat,
        patch_lon_offset_frac=patch_lon_offset_frac,
    )                                                              # (B, 6)
    risk_val = sem[:, 0].clamp(0.0, 1.0)
    sdf_val = sem[:, 1].clamp(0.0, 50.0)
    risk_grad = torch.stack([sem[:, 2], sem[:, 3]], dim=-1)        # (B, 2)
    sdf_grad = torch.stack([sem[:, 4], sem[:, 5]], dim=-1)         # (B, 2)

    F_mat_soft = -lam_soft.unsqueeze(-1) * risk_grad
    _, db_dphi = sdf_barrier_grad(sdf_val, d_hat_sdf=d_hat_sdf)
    F_mat_hard = -lam_hard.unsqueeze(-1) * db_dphi.unsqueeze(-1) * sdf_grad

    probe_stats = _lateral_probe_stats(
        o=o,
        heading=heading,
        o0=o0,
        heading_0=heading_0,
        rollout_patch=rollout_patch,
        cell_size_lon=cell_size_lon,
        cell_size_lat=cell_size_lat,
        patch_lon_offset_frac=patch_lon_offset_frac,
        lateral_lookahead=lateral_lookahead,
        lateral_lane_width=lateral_lane_width,
        lateral_preference_bias=lateral_preference_bias,
    )
    if mu_lat is None:
        F_lat = torch.zeros_like(o)
    else:
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
        risk_left=probe_stats["risk_left"],
        risk_right=probe_stats["risk_right"],
        ttc_gain=ttc_gain,
        ttc_threshold_s=ttc_threshold_s,
        ttc_softness_s=ttc_softness_s,
        ttc_min_closing_speed=ttc_min_closing_speed,
        ttc_lane_halfwidth=ttc_lane_halfwidth,
        ttc_boxed_risk_thresh=ttc_boxed_risk_thresh,
        ttc_boxed_gate_sharpness=ttc_boxed_gate_sharpness,
    )
    # Keep the analytic TTC branch forward-only. The leader-selection logic
    # is intentionally piecewise and non-smooth; detaching it preserves the
    # deployed behavior while avoiding brittle autograd through argmin/gather
    # style routing in training.
    F_ttc = F_ttc.detach()

    F_tot = F_goal + F_damp + F_geom + F_mat_soft + F_mat_hard + F_lat + F_ttc
    return F_tot, dmin, risk_val, sdf_val


# ──────────────────────────────────────────────────────────────────────────
# The integrator — drop-in replacement for integrate_surrogate_material
# ──────────────────────────────────────────────────────────────────────────

def integrate_surrogate_highway(
    o0:         torch.Tensor,    # (B, 2)
    v0:         torch.Tensor,    # (B, 2)  world-frame velocity
    goal:       torch.Tensor,    # (B, 2)
    C:          torch.Tensor,    # (B, N, 2)
    R:          torch.Tensor,    # (B, N)
    mask:       torch.Tensor,    # (B, N) bool
    alphas:     torch.Tensor,    # (B, N)
    beta:       torch.Tensor,    # (B,)
    gamma:      torch.Tensor,    # (B,) learned velocity damping
    lam_soft:   torch.Tensor,    # (B,)
    lam_hard:   torch.Tensor,    # (B,)
    rollout_patch: torch.Tensor, # (B, 6, Hp, Wp) ego-frame patch
    d_hat:      torch.Tensor,    # (B,)
    dt:         torch.Tensor,    # (B,) or scalar
    H:          torch.Tensor,    # (B,) int rollout horizon
    V_neighbors: Optional[torch.Tensor] = None,  # (B, N, 2)
    mu_lat:     Optional[torch.Tensor] = None,
    *,
    cell_size_lon: float = 1.0,
    cell_size_lat: float = 1.0,
    patch_lon_offset_frac: float = 0.05,
    robot_radius: float = 0.0,
    margin_factor: float = 0.5,
    d_hat_sdf: float = 3.0,
    vehicle_length: float = VEHICLE_LENGTH,
    ipc_grad_clip: float = 100.0,
    ttc_gain: float = 0.0,
    ttc_threshold_s: float = 3.0,
    ttc_softness_s: float = 0.5,
    ttc_min_closing_speed: float = 0.5,
    ttc_lane_halfwidth: float = 2.0,
    ttc_boxed_risk_thresh: float = 0.25,
    ttc_boxed_gate_sharpness: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage 2 surrogate integrator with bicycle dynamics for highway-env.

    Same return signature as DFC's `integrate_surrogate_material`:
        (oT, vT, min_clear_geom, cum_risk, hard_count, arc_length)

    Internally we maintain (position, heading, speed). On output, we
    reconstruct vT = speed_T * (cos heading_T, sin heading_T) so the
    cost function and downstream loss code don't need to change.

    NOTE on `v0` decomposition: highway-env vehicles' velocity is
        v_world = speed * (cos(heading + beta_slip), sin(heading + beta_slip))
    Without knowing the slip angle (which depends on past steering), we
    approximate `heading_0 = atan2(v0_y, v0_x)`, `speed_0 = ||v0||`. This
    is exact at zero steering angle, which is the case at observation
    time on a straight highway.
    """
    B, N = C.shape[:2]

    # Ensure robot_radius is per-batch broadcasts
    if not torch.is_tensor(robot_radius):
        rr = o0.new_tensor(float(robot_radius))
    else:
        rr = robot_radius.to(o0.device, o0.dtype)
    R_eff = R + margin_factor * (rr if rr.ndim == 0 else rr[:, None])

    # Decompose v0 into (heading_0, speed_0).
    speed_0 = torch.linalg.norm(v0, dim=-1).clamp_min(1e-3)         # (B,)
    heading_0 = torch.atan2(v0[:, 1], v0[:, 0])                     # (B,)

    # State, accumulators
    o = o0.clone()
    heading = heading_0.clone()
    speed = speed_0.clone()
    min_clear  = torch.full((B,), float("inf"), dtype=o.dtype, device=o.device)
    cum_risk   = torch.zeros(B, dtype=o.dtype, device=o.device)
    hard_count = torch.zeros(B, dtype=o.dtype, device=o.device)
    arc_length = torch.zeros(B, dtype=o.dtype, device=o.device)

    # `dt` may be scalar or (B,)
    if dt.dim() == 0:
        dt_b = dt.expand(B)
    else:
        dt_b = dt
    # bicycle_step_train takes scalar dt; we'll multiply by dt_b on per-batch
    # quantities (accel·dt, etc.) and use dt=1.0 in the bicycle call. To keep
    # the surrogate honest with possible per-batch dt, we apply per-batch
    # scaling explicitly.
    # For simplicity in this first version, assume all dt are equal:
    dt_scalar = float(dt_b[0].item()) if dt_b.numel() > 0 else 0.1
    # (We could generalize to per-batch dt if your dataset varies it.)

    H_max = int(H.max().item())

    for s in range(H_max):
        active = (s < H).to(o.dtype)                              # (B,)
        active2 = active.unsqueeze(-1)                            # (B, 1)

        # ── Force law shared with closed-loop deployment ──────────────────
        F_tot, dmin, risk_val, sdf_val = compute_surrogate_highway_force(
            o=o, heading=heading, speed=speed, o0=o0, heading_0=heading_0,
            goal=goal, C=C, V_neighbors=V_neighbors, R_eff=R_eff, mask=mask,
            alphas=alphas, beta=beta,
            gamma=gamma,
            lam_soft=lam_soft, lam_hard=lam_hard, mu_lat=mu_lat,
            rollout_patch=rollout_patch, d_hat=d_hat,
            cell_size_lon=cell_size_lon,
            cell_size_lat=cell_size_lat,
            patch_lon_offset_frac=patch_lon_offset_frac,
            d_hat_sdf=d_hat_sdf,
            ipc_grad_clip=ipc_grad_clip,
            ttc_gain=ttc_gain,
            ttc_threshold_s=ttc_threshold_s,
            ttc_softness_s=ttc_softness_s,
            ttc_min_closing_speed=ttc_min_closing_speed,
            ttc_lane_halfwidth=ttc_lane_halfwidth,
            ttc_boxed_risk_thresh=ttc_boxed_risk_thresh,
            ttc_boxed_gate_sharpness=ttc_boxed_gate_sharpness,
        )
        min_clear = torch.minimum(min_clear, dmin)

        # ── Force → action → bicycle step ────────────────────────────────
        accel_raw, steer_raw = force_to_action(F_tot, heading, speed,
                                                length=vehicle_length)
        # Match deployment action bounds in the forward pass. The
        # straight-through gradient keeps saturated Stage-1 examples trainable
        # instead of marooning beta/gamma behind a zero-gradient hard clamp.
        accel = _straight_through_clip(accel_raw, *ACCEL_RANGE)
        steer = _straight_through_clip(steer_raw, *STEER_RANGE)

        new_o, new_heading, new_speed = bicycle_step_train(
            o, heading, speed, accel, steer, dt=dt_scalar,
            length=vehicle_length,
        )

        # Apply active mask: inactive batch elements should not move
        new_o       = o       + active2 * (new_o - o)
        new_heading = heading + active  * (new_heading - heading)
        new_speed   = speed   + active  * (new_speed - speed)

        # ── Accumulate ────────────────────────────────────────────────────
        step_disp = torch.linalg.norm(new_o - o, dim=-1)           # (B,)
        arc_length = arc_length + active * step_disp
        cum_risk   = cum_risk   + active * risk_val * step_disp
        hard_count = hard_count + active * (sdf_val < 1.0).to(o.dtype)

        o, heading, speed = new_o, new_heading, new_speed

    # Reconstruct final velocity
    vT = speed.unsqueeze(-1) * torch.stack(
        [torch.cos(heading), torch.sin(heading)], dim=-1
    )                                                                # (B, 2)

    return o, vT, min_clear, cum_risk, hard_count, arc_length
