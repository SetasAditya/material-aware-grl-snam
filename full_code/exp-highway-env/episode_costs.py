"""
episode_costs.py

Episode cost J(rollout) and auxiliary loss terms for Stage 2 training in
highway-env. Mirrors the structure of `train_material.py` (DFC) so the
total loss assembly in Step 6 reuses the same pattern.

What's here:

  episode_cost_highway   — Eq. (4) of the paper, adapted for highway-env:
      J = w_g  ||q_T - q_g||²
        + w_l  · arc_length
        + w_r  · cum_risk
        + w_h  · hard_count       (proximity-event by default)
      The hard_count source is configurable: 'proximity' (dense, training)
      or 'collision' (sparse, evaluation).

  cvar_loss              — Rockafellar–Uryasev with detached quantile.
                            Identical to train_material.py:cvar_loss.

  loss_imitation         — L_traj + L_vel against an externally-supplied
                            target trajectory. Used at Stage 1 (IDM target)
                            and at Stage 2 with reduced weight.

  loss_clearance         — L_clear: softplus penalty on negative clearance.
                            Same as DFC.

  loss_lambda_entropy    — L_λ: anti-saturation regularizer for sigmoid-
                            bounded λ heads. Same as DFC.

  loss_multi_start       — L_multi: short rollouts from feasible starts
                            near nearest vehicle, penalising penetrations.
                            Geometry-only; reuses the highway-env integrator
                            with lam_soft = lam_hard = 0.

  HighwayLossWeights     — dataclass holding all weight + threshold knobs.
                            Stage 1 / Stage 2 are pre-configured factories.

  combine_losses         — Reference total-loss assembly, with the
                            stage-aware schedule from the paper:
                                Stage 1: imitation + safety
                                Stage 2: 0.3·imitation + safety + nav_scale·L_nav + L_λ
                            Returns the scalar loss plus a metrics dict.

What's deliberately NOT here:

  - hard-coded reference trajectories. `loss_imitation` takes them as
    arguments. Step 5 (Stage 1 IDM training) provides them.
  - on-policy buffer logic. Step 6 owns that.
  - a training loop. This module just gives you the loss building blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from surrogate_integrator import integrate_surrogate_highway


# ──────────────────────────────────────────────────────────────────────────
# Episode cost
# ──────────────────────────────────────────────────────────────────────────

def episode_cost_highway(
    oT:          torch.Tensor,    # (B, 2)  final ego position
    goal:        torch.Tensor,    # (B, 2)  episode goal
    cum_risk:    torch.Tensor,    # (B,)    cumulative soft-risk exposure
    hard_count:  torch.Tensor,    # (B,)    hard-hazard event count (see below)
    arc_length:  torch.Tensor,    # (B,)    true rollout arc length
    *,
    w_goal:  float = 2.0,
    w_len:   float = 0.01,
    w_risk:  float = 1.0,
    w_hard:  float = 5.0,
) -> torch.Tensor:
    """Per-rollout episode cost. Returns (B,).

    Same shape as Eq. (4):
        J = w_g ||q_T - q_g||² + w_l · arc + w_r · cum_risk + w_h · hard_count

    The interpretation of `hard_count` depends on how the integrator was
    invoked:
      * Default proximity construction in surrogate_integrator.py:
          hard_count_t = sum_t 1[phi(q_t) < 1.0]
        Dense per-step accumulator, useful for training gradients.
      * For evaluation against highway-env's terminal collision flag, set
        hard_count externally to {0, 1} per rollout. The cost still works.

    The arc-length term penalises long detours but not lane changes — see
    method-section paragraph "Goal term reinterpretation".
    """
    goal_err = ((oT - goal) ** 2).sum(dim=-1)                         # (B,)
    return (w_goal * goal_err
            + w_len  * arc_length
            + w_risk * cum_risk
            + w_hard * hard_count)


# ──────────────────────────────────────────────────────────────────────────
# CVaR (re-exported for clarity; identical to DFC)
# ──────────────────────────────────────────────────────────────────────────

def cvar_loss(
    costs: torch.Tensor,
    alpha: float = 0.95,
) -> torch.Tensor:
    """Rockafellar–Uryasev CVaR with detached empirical quantile.

    Mirrors `train_material.py:cvar_loss`. Gradient flows only through the
    upper-α tail of the batch.
    """
    eta = torch.quantile(costs.detach(), alpha)
    excess = F.relu(costs - eta)
    return eta + excess.mean() / (1.0 - alpha)


# ──────────────────────────────────────────────────────────────────────────
# Auxiliary losses
# ──────────────────────────────────────────────────────────────────────────

def loss_imitation(
    oT:    torch.Tensor,           # (B, 2)
    vT:    torch.Tensor,           # (B, 2)
    o_tgt: Optional[torch.Tensor], # (B, 2)
    v_tgt: Optional[torch.Tensor], # (B, 2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """L_traj, L_vel: MSE imitation against an externally-supplied target.

    The target comes from:
      - Stage 1: IDM rollout (provided by Step 5)
      - Stage 2: a risk-aware target — typically the Stage 1 rollout
        on the same scene, kept as a stability anchor.

    If either target is None, the corresponding loss is a zero tensor —
    the caller is then responsible for omitting it from the weighted sum
    via cfg.w_traj=0 (or by relying on the zero contribution).
    """
    if o_tgt is None:
        L_traj = oT.new_zeros(())
    else:
        L_traj = F.mse_loss(oT, o_tgt)
    if v_tgt is None:
        L_vel = vT.new_zeros(())
    else:
        L_vel = F.mse_loss(vT, v_tgt)
    return L_traj, L_vel


def loss_imitation_masked(
    oT:    torch.Tensor,            # (B, 2)
    vT:    torch.Tensor,            # (B, 2)
    o_tgt: Optional[torch.Tensor],  # (B, 2)
    v_tgt: Optional[torch.Tensor],  # (B, 2)
    mask:  torch.Tensor,            # (B,) float, 1 = valid imitation target
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-sample-masked imitation MSE for mixed IDM + on-policy batches.

    Scale matches ``loss_imitation``: an all-ones mask gives the same value
    as ``F.mse_loss`` over the full batch. A zero mask gives a differentiable
    zero contribution through the predicted tensors.
    """
    mask = mask.to(device=oT.device, dtype=oT.dtype).view(-1)
    denom = mask.sum().clamp_min(1.0)

    if o_tgt is None:
        L_traj = oT.new_zeros(())
    else:
        sq = (oT - o_tgt.to(oT.device, oT.dtype)).pow(2).sum(dim=-1)
        L_traj = (sq * mask).sum() / (denom * float(oT.shape[-1]))

    if v_tgt is None:
        L_vel = vT.new_zeros(())
    else:
        sq = (vT - v_tgt.to(vT.device, vT.dtype)).pow(2).sum(dim=-1)
        L_vel = (sq * mask).sum() / (denom * float(vT.shape[-1]))

    return L_traj, L_vel


def loss_clearance(
    min_clear: torch.Tensor,    # (B,) per-rollout minimum clearance (m)
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """L_clear: softplus penalty for low minimum clearance.

    Same form as DFC: smooth surrogate for a barrier on min_clearance > 0.
    Penalises rollouts whose closest approach to a vehicle bounding circle
    is small or negative. `margin` is the softness scale (smaller = sharper).
    """
    return F.softplus(-min_clear / margin).mean()


def loss_lambda_entropy(
    lam_soft:     torch.Tensor,   # (B,) values in (0, lam_soft_max)
    lam_hard:     torch.Tensor,   # (B,) values in (0, lam_hard_max)
    lam_soft_max: float,
    lam_hard_max: float,
    eps:          float = 1e-6,
) -> torch.Tensor:
    """L_λ: prevents the sigmoid-bounded λ heads from saturating at 0 or max.

    Lifted directly from `train_material.py:790-800`. Penalises the head
    distribution being concentrated at the extremes of the sigmoid range.
    Without this term, Stage 2 frequently collapses to either λ ≡ 0 (no
    risk avoidance) or λ ≡ λ_max (oscillating overcorrection).

    Form: -[mean(log s) + mean(log(1-s))] for s ∈ {soft_norm, hard_norm}.
    Result is non-negative; minimum is achieved at uniform-over-(0,1).
    """
    s = (lam_soft / lam_soft_max).clamp(eps, 1 - eps)
    h = (lam_hard / lam_hard_max).clamp(eps, 1 - eps)
    term = (
        -(s.log().mean() + (1 - s).log().mean()
          + h.log().mean() + (1 - h).log().mean())
    )
    return 0.25 * term


def loss_multi_start(
    o0: torch.Tensor,             # (B, 2)
    v0: torch.Tensor,             # (B, 2)
    goal: torch.Tensor,           # (B, 2)
    C: torch.Tensor,              # (B, N, 2)
    R: torch.Tensor,              # (B, N)
    mask: torch.Tensor,           # (B, N)
    alphas: torch.Tensor,         # (B, N)
    beta: torch.Tensor,           # (B,)
    gamma: torch.Tensor,          # (B,)
    rollout_patch: torch.Tensor,  # (B, 6, Hp, Wp)
    d_hat: torch.Tensor,          # (B,)
    dt: torch.Tensor,             # scalar or (B,)
    H: torch.Tensor,              # (B,) int
    *,
    cell_size_lon: float = 1.0,
    cell_size_lat: float = 1.0,
    ms_count:    int   = 3,
    ms_h:        int   = 2,
    ms_dt_mult:  float = 1.5,
    margin_factor: float = 0.5,
    tau:         float = 0.05,
) -> torch.Tensor:
    """L_multi: short rollouts from feasible starts near nearest vehicle.

    Mirrors `surrogate_robust.py:multi_start_penalty` but uses the
    highway-env integrator with risk coefficients zeroed (geometry-only).
    The highway-env integrator's signature requires lam_soft/lam_hard, so
    we pass zeros — the multi-start check is geometric-only by design.

    Behavior: pick a near-vehicle starting point, rollout for ms_h steps,
    penalise any negative clearance encountered.
    """
    B, N = C.shape[:2]
    if ms_count <= 0 or N == 0:
        return o0.new_tensor(0.0)

    # Compute nearest-vehicle direction from o0 for each batch element
    diff = o0.unsqueeze(1) - C                               # (B, N, 2)
    r = torch.linalg.norm(diff, dim=-1).clamp_min(1e-9)
    d = r - R                                                # (B, N) signed clearance
    d = torch.where(mask, d, torch.full_like(d, 1e6))
    dmin, idx = d.min(dim=1)                                 # (B,) closest
    n_hat = diff[torch.arange(B), idx] / r[torch.arange(B), idx].unsqueeze(-1)  # (B, 2)

    L_acc = o0.new_tensor(0.0)
    H_ms = torch.full_like(H, ms_h)
    if dt.dim() == 0:
        dt_ms = ms_dt_mult * dt
    else:
        dt_ms = ms_dt_mult * dt
    lam_zero = torch.zeros(B, device=o0.device, dtype=o0.dtype)

    for _ in range(ms_count):
        frac = 0.9
        step = (frac * dmin).unsqueeze(-1) * n_hat            # (B, 2) toward closest
        o_ms = o0 - step                                       # move toward closest vehicle

        # Feasibility check: don't start INSIDE a vehicle
        diff_ms = o_ms.unsqueeze(1) - C
        r_ms = torch.linalg.norm(diff_ms, dim=-1).clamp_min(1e-9)
        d_ms = torch.where(mask, r_ms - R, torch.full_like(r_ms, 1e6))
        ok = (d_ms.min(dim=1).values >= 0)
        o_ms = torch.where(ok.unsqueeze(-1), o_ms, o0 + 0.5 * step)   # fallback

        # Short geometry-only rollout
        _, _, clr_ms, _, _, _ = integrate_surrogate_highway(
            o0=o_ms, v0=v0, goal=goal,
            C=C, R=R, mask=mask,
            alphas=alphas, beta=beta, gamma=gamma,
            lam_soft=lam_zero, lam_hard=lam_zero,
            rollout_patch=rollout_patch,
            d_hat=d_hat, dt=dt_ms, H=H_ms,
            cell_size_lon=cell_size_lon,
            cell_size_lat=cell_size_lat,
            margin_factor=margin_factor,
        )
        L_acc = L_acc + F.softplus((-clr_ms) / tau).mean()

    return L_acc / float(ms_count)


# ──────────────────────────────────────────────────────────────────────────
# Loss-weight bundle and stage-aware combiner
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class HighwayLossWeights:
    """All weights and thresholds in one place. Stage 1/2 factories below."""

    # Episode cost
    w_goal:  float = 2.0
    w_len:   float = 0.01
    w_risk:  float = 1.0
    w_hard:  float = 5.0
    cvar_alpha: float = 0.95

    # Auxiliary
    w_traj:    float = 1.0
    w_vel:     float = 0.5
    w_clear:   float = 5e-3
    w_multi:   float = 0.5
    w_lreg:    float = 0.01

    # λ-head ranges (must match the model's sigmoid bounds)
    lam_soft_max: float = 50.0    # highway scale, vs DFC's 5.0
    lam_hard_max: float = 10.0

    # Multi-start params
    ms_count:    int   = 3
    ms_h:        int   = 2
    ms_dt_mult:  float = 1.5

    # Clearance softness
    clear_margin: float = 0.05

    # Stage selector ('1' or '2')
    stage: int = 2

    # Stage 2 imitation attenuation: in DFC they multiply L_traj/L_vel by
    # 0.3 to let the CVaR signal reshape the field. Same idea here.
    stage2_imit_scale: float = 0.3

    @classmethod
    def stage1(cls) -> "HighwayLossWeights":
        """Stage 1 weights: geometry-only, full-strength imitation, no L_nav."""
        return cls(
            stage=1,
            w_traj=1.0, w_vel=0.5,
            w_clear=5e-3, w_multi=0.5,
            w_lreg=0.0,    # λ heads are frozen at 0 in Stage 1
        )

    @classmethod
    def stage2(cls) -> "HighwayLossWeights":
        """Stage 2 weights tuned for highway-env scale.

        Positions are in meters and the goal is far ahead, unlike DFC's
        normalized units. Imitation already anchors the terminal state, so
        Stage 2's navigation cost focuses on risk/clearance statistics
        instead of another large squared-distance term.
        """
        return cls(
            stage=2,
            w_goal=0.0,
            w_len=0.01,
            w_risk=1.0,
            w_hard=1.0,
            stage2_imit_scale=0.3,
        )


def combine_losses(
    *,
    # Rollout outputs (from the integrator)
    oT:        torch.Tensor,
    vT:        torch.Tensor,
    min_clear: torch.Tensor,
    cum_risk:  torch.Tensor,
    hard_count:torch.Tensor,
    arc_length:torch.Tensor,
    # Episode goal
    goal:      torch.Tensor,
    # Imitation targets (None at first Stage 1 epoch before targets are built)
    o_tgt:     Optional[torch.Tensor],
    v_tgt:     Optional[torch.Tensor],
    # Predicted material coefficients (for L_λ)
    lam_soft:  torch.Tensor,
    lam_hard:  torch.Tensor,
    # Optional pre-computed L_multi (None to skip)
    L_multi:   Optional[torch.Tensor] = None,
    # Optional per-sample imitation mask for mixed IDM + on-policy batches.
    # None preserves the original Stage 1 / pure-imitation behavior.
    imit_mask: Optional[torch.Tensor] = None,
    # Stage 2 navigation-loss scale. The trainer passes the same curriculum
    # value used for lam_soft/lam_hard so nav gradients come online only when
    # risk forces are actually active.
    nav_scale: float = 1.0,
    # Weights
    cfg:       Optional[HighwayLossWeights] = None,
) -> Tuple[torch.Tensor, dict]:
    """Stage-aware total-loss assembly. Returns (scalar_loss, metrics_dict).

    Stage 1: L = w_traj·L_traj + w_vel·L_vel + w_clear·L_clear + w_multi·L_multi
    Stage 2: L = α·(w_traj·L_traj + w_vel·L_vel) + w_clear·L_clear + w_multi·L_multi
              + nav_scale·L_nav  + w_lreg·L_λ
             where α = stage2_imit_scale and L_nav = CVaR_α(J).

    The metrics dict contains all individual loss components for logging.
    """
    cfg = cfg or HighwayLossWeights.stage2()

    # Episode cost J + CVaR
    J = episode_cost_highway(
        oT, goal, cum_risk, hard_count, arc_length,
        w_goal=cfg.w_goal, w_len=cfg.w_len,
        w_risk=cfg.w_risk, w_hard=cfg.w_hard,
    )
    L_nav_raw = cvar_loss(J, alpha=cfg.cvar_alpha)
    L_nav = L_nav_raw * float(nav_scale)

    # Auxiliary
    if imit_mask is None:
        L_traj, L_vel = loss_imitation(oT, vT, o_tgt, v_tgt)
    else:
        L_traj, L_vel = loss_imitation_masked(oT, vT, o_tgt, v_tgt, imit_mask)
    L_clear = loss_clearance(min_clear, margin=cfg.clear_margin)
    L_lreg  = loss_lambda_entropy(
        lam_soft, lam_hard,
        lam_soft_max=cfg.lam_soft_max,
        lam_hard_max=cfg.lam_hard_max,
    )
    L_ms = oT.new_zeros(()) if L_multi is None else L_multi

    # Stage assembly
    if cfg.stage == 1:
        L = (cfg.w_traj  * L_traj
             + cfg.w_vel   * L_vel
             + cfg.w_clear * L_clear
             + cfg.w_multi * L_ms)
    else:
        L = (cfg.stage2_imit_scale * (cfg.w_traj * L_traj + cfg.w_vel * L_vel)
             + cfg.w_clear * L_clear
             + cfg.w_multi * L_ms
             + L_nav
             + cfg.w_lreg * L_lreg)

    metrics = {
        "loss":   float(L.detach()),
        "L_nav":  float(L_nav.detach()),
        "L_nav_raw":  float(L_nav_raw.detach()),
        "nav_scale":  float(nav_scale),
        "L_traj": float(L_traj.detach()),
        "L_vel":  float(L_vel.detach()),
        "L_clear":float(L_clear.detach()),
        "L_multi":float(L_ms.detach()),
        "L_lreg": float(L_lreg.detach()),
        "J_mean": float(J.mean().detach()),
        "J_max":  float(J.max().detach()),
        "cum_risk_mean": float(cum_risk.mean().detach()),
        "hard_count_mean": float(hard_count.mean().detach()),
        "n_imit_samples": (float(imit_mask.detach().sum())
                           if imit_mask is not None else float(oT.shape[0])),
    }
    return L, metrics
