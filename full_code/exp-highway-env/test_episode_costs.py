"""
test_episode_costs.py

Tests for episode_costs.py. Five tests:

  T9  — Episode cost composition: each weight independently scales its term.
        Set three of four weights to zero, verify J equals the surviving
        weighted term.

  T10 — Stage 1 vs Stage 2 loss schedule: combine_losses produces different
        scalar losses with the right structure.
          Stage 1: imitation dominates, no L_nav.
          Stage 2: L_nav present, imitation attenuated.

  T11 — Imitation targets None: when o_tgt or v_tgt is None, L_traj or L_vel
        is exactly zero (no NaN, no spurious gradient).

  T12 — Lambda-entropy regularizer at extremes: penalty is high at λ=0 and
        λ=max, low at λ=mid. The entropy term should not be the cheapest to
        minimize by going to 0 or max.

  T13 — Full-pipeline gradient: integrator → episode_cost → CVaR → backward.
        With a risk-only navigation config, d(L_nav)/d(λ_soft) is finite
        and on average negative on the slow-leader scene (matches T8's
        invariant but through the full loss-combining pipeline).

  T19 — Masked imitation calibration: an all-ones mask matches the original
        unmasked imitation loss, and a half-mask matches slicing the batch.

These build on Steps 1-3. T13 specifically validates that combine_losses
plumbed correctly into the integrator backprop path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from env_wrapper import (  # noqa: E402
    HighwayMaterialObservation,
    WrapperConfig,
    _MockEnv,
    _MockVehicle,
)
from surrogate_integrator import integrate_surrogate_highway  # noqa: E402
from episode_costs import (  # noqa: E402
    HighwayLossWeights,
    combine_losses,
    cvar_loss,
    episode_cost_highway,
    loss_clearance,
    loss_imitation,
    loss_imitation_masked,
    loss_lambda_entropy,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers (shared with test_surrogate_integrator)
# ──────────────────────────────────────────────────────────────────────────

def _build_slow_leader_obs() -> dict:
    cfg = WrapperConfig()
    ego = _MockVehicle(x=0.0, y=4.0, heading=0.0, speed=25.0)
    others = [
        _MockVehicle(x=30.0, y=4.0, heading=0.0, speed=12.0),
        _MockVehicle(x=-15.0, y=8.0, heading=0.0, speed=28.0),
    ]
    env = _MockEnv(ego, others)
    return HighwayMaterialObservation(cfg).build(env)


def _obs_to_tensors(obs: dict, B: int = 1) -> dict:
    out = {}
    out["o0"]            = torch.from_numpy(obs["o0"]).unsqueeze(0).repeat(B, 1)
    out["v0"]            = torch.from_numpy(obs["v0"]).unsqueeze(0).repeat(B, 1)
    out["goal"]          = torch.from_numpy(obs["goal"]).unsqueeze(0).repeat(B, 1)
    out["C"]             = torch.from_numpy(obs["C"]).unsqueeze(0).repeat(B, 1, 1)
    out["R"]             = torch.from_numpy(obs["R"]).unsqueeze(0).repeat(B, 1)
    out["mask"]          = torch.from_numpy(obs["mask"]).unsqueeze(0).repeat(B, 1)
    out["rollout_patch"] = torch.from_numpy(obs["rollout_patch"]).unsqueeze(0).repeat(
        B, 1, 1, 1
    )
    out["d_hat"]         = torch.tensor([float(obs["d_hat"])] * B)
    out["dt"]            = torch.tensor(float(obs["dt"]))
    out["H"]             = torch.tensor([int(obs["H"])] * B)
    return out


# ──────────────────────────────────────────────────────────────────────────
# T9 — Episode cost composition
# ──────────────────────────────────────────────────────────────────────────

def test_episode_cost_composition() -> None:
    """T9: Each weight independently scales its term."""
    print("\n[T9] Episode cost composition test")
    B = 4
    oT         = torch.tensor([[1.0, 0.0]] * B)
    goal       = torch.tensor([[0.0, 0.0]] * B)
    cum_risk   = torch.full((B,), 2.0)
    hard_count = torch.full((B,), 3.0)
    arc_length = torch.full((B,), 10.0)

    # Goal-only
    J_goal = episode_cost_highway(
        oT, goal, cum_risk, hard_count, arc_length,
        w_goal=1.0, w_len=0.0, w_risk=0.0, w_hard=0.0,
    )
    assert torch.allclose(J_goal, torch.full((B,), 1.0)), \
        f"Goal-only cost should be 1.0 ((1²+0²)·1), got {J_goal}"

    # Length-only
    J_len = episode_cost_highway(
        oT, goal, cum_risk, hard_count, arc_length,
        w_goal=0.0, w_len=1.0, w_risk=0.0, w_hard=0.0,
    )
    assert torch.allclose(J_len, torch.full((B,), 10.0))

    # Risk-only
    J_risk = episode_cost_highway(
        oT, goal, cum_risk, hard_count, arc_length,
        w_goal=0.0, w_len=0.0, w_risk=1.0, w_hard=0.0,
    )
    assert torch.allclose(J_risk, torch.full((B,), 2.0))

    # Hard-only
    J_hard = episode_cost_highway(
        oT, goal, cum_risk, hard_count, arc_length,
        w_goal=0.0, w_len=0.0, w_risk=0.0, w_hard=1.0,
    )
    assert torch.allclose(J_hard, torch.full((B,), 3.0))

    # Mixed: weights linearly combine
    J_mixed = episode_cost_highway(
        oT, goal, cum_risk, hard_count, arc_length,
        w_goal=2.0, w_len=0.5, w_risk=1.0, w_hard=3.0,
    )
    expected = 2.0 * 1.0 + 0.5 * 10.0 + 1.0 * 2.0 + 3.0 * 3.0   # = 18.0
    assert torch.allclose(J_mixed, torch.full((B,), expected))

    print(f"   J_goal={J_goal[0].item():.2f}  J_len={J_len[0].item():.2f}  "
          f"J_risk={J_risk[0].item():.2f}  J_hard={J_hard[0].item():.2f}  "
          f"J_mixed={J_mixed[0].item():.2f} (expected {expected:.1f})")
    print("   ✓ Each weight scales its term linearly and independently.")


# ──────────────────────────────────────────────────────────────────────────
# T10 — Stage 1 vs Stage 2 loss schedule
# ──────────────────────────────────────────────────────────────────────────

def test_stage_loss_schedule() -> None:
    """T10: Stage 1 has no L_nav, Stage 2 includes it."""
    print("\n[T10] Stage 1 vs Stage 2 loss schedule")
    B = 4
    oT  = torch.tensor([[1.0, 0.0]] * B)
    vT  = torch.tensor([[20.0, 0.0]] * B)
    goal = torch.tensor([[0.0, 0.0]] * B)
    o_tgt = torch.tensor([[0.5, 0.0]] * B)
    v_tgt = torch.tensor([[18.0, 0.0]] * B)
    min_clear = torch.full((B,), 1.0)
    cum_risk = torch.full((B,), 2.0)
    hard_count = torch.full((B,), 0.0)
    arc_length = torch.full((B,), 10.0)
    lam_soft = torch.full((B,), 25.0)   # mid-range
    lam_hard = torch.full((B,), 5.0)

    # Stage 1
    cfg1 = HighwayLossWeights.stage1()
    L1, m1 = combine_losses(
        oT=oT, vT=vT, min_clear=min_clear,
        cum_risk=cum_risk, hard_count=hard_count, arc_length=arc_length,
        goal=goal, o_tgt=o_tgt, v_tgt=v_tgt,
        lam_soft=lam_soft, lam_hard=lam_hard, cfg=cfg1,
    )

    # Stage 2 with same inputs
    cfg2 = HighwayLossWeights.stage2()
    L2, m2 = combine_losses(
        oT=oT, vT=vT, min_clear=min_clear,
        cum_risk=cum_risk, hard_count=hard_count, arc_length=arc_length,
        goal=goal, o_tgt=o_tgt, v_tgt=v_tgt,
        lam_soft=lam_soft, lam_hard=lam_hard, cfg=cfg2,
    )

    # In Stage 1, L_nav doesn't enter the total even though it's logged
    print(f"   Stage 1: total={L1.item():.4f}  L_nav={m1['L_nav']:.4f}  "
          f"L_traj={m1['L_traj']:.4f}")
    print(f"   Stage 2: total={L2.item():.4f}  L_nav={m2['L_nav']:.4f}  "
          f"L_traj={m2['L_traj']:.4f}")

    # Stage 2 includes L_nav (positive on this rollout); Stage 1 does not.
    # Since L_nav is purely additive in stage 2, Stage 2 ≥ Stage 1·alpha
    # under most configurations. We verify the right structural difference:
    # Stage 1 has zero L_nav contribution, Stage 2 has nonzero.
    L_nav_implied_stage1 = L1 - cfg1.w_traj * m1["L_traj"] - cfg1.w_vel * m1["L_vel"] \
                              - cfg1.w_clear * m1["L_clear"]
    L_nav_implied_stage2 = L2 - cfg2.stage2_imit_scale * (
                                  cfg2.w_traj * m2["L_traj"] + cfg2.w_vel * m2["L_vel"]
                              ) - cfg2.w_clear * m2["L_clear"] \
                              - cfg2.w_lreg * m2["L_lreg"]

    print(f"   Implied L_nav in stage1 total: {L_nav_implied_stage1.item():.6f} (≈0)")
    print(f"   Implied L_nav in stage2 total: {L_nav_implied_stage2.item():.4f} (=L_nav)")

    assert abs(L_nav_implied_stage1.item()) < 1e-4, \
        "Stage 1 should NOT include L_nav in its total"
    assert abs(L_nav_implied_stage2.item() - m2["L_nav"]) < 1e-3, \
        f"Stage 2's total should include exactly L_nav={m2['L_nav']:.4f}; " \
        f"residual is {L_nav_implied_stage2.item() - m2['L_nav']:.4f}"
    print("   ✓ Stage 1 omits L_nav; Stage 2 includes it with correct weight.")


# ──────────────────────────────────────────────────────────────────────────
# T11 — Imitation targets None
# ──────────────────────────────────────────────────────────────────────────

def test_imitation_targets_none() -> None:
    """T11: When targets are None, L_traj and L_vel are exactly zero."""
    print("\n[T11] Imitation targets-None handling")
    B = 4
    oT = torch.tensor([[1.0, 2.0]] * B)
    vT = torch.tensor([[15.0, 0.0]] * B)

    L_traj_none, L_vel_none = loss_imitation(oT, vT, None, None)
    assert L_traj_none.item() == 0.0 and L_vel_none.item() == 0.0
    assert torch.isfinite(L_traj_none) and torch.isfinite(L_vel_none)

    # Mixed: provide o_tgt only
    o_tgt = torch.tensor([[1.0, 2.0]] * B)
    L_traj_zero, L_vel_none2 = loss_imitation(oT, vT, o_tgt, None)
    assert L_traj_zero.item() == 0.0    # exact match
    assert L_vel_none2.item() == 0.0    # None → zero

    # Both provided, nontrivial
    v_tgt = torch.tensor([[10.0, 0.0]] * B)
    L_traj_real, L_vel_real = loss_imitation(oT, vT, o_tgt, v_tgt)
    assert L_traj_real.item() == 0.0       # oT == o_tgt
    assert abs(L_vel_real.item() - 12.5) < 1e-4   # F.mse_loss averages over x/y: (25+0)/2

    print(f"   None,None       → L_traj={L_traj_none.item():.4f}, L_vel={L_vel_none.item():.4f}")
    print(f"   o_tgt match     → L_traj={L_traj_zero.item():.4f}")
    print(f"   v_tgt mismatch  → L_vel={L_vel_real.item():.4f} (expected 12.50)")
    print("   ✓ Targets=None gives exactly zero, non-None gives correct MSE.")


# ──────────────────────────────────────────────────────────────────────────
# T12 — Lambda-entropy regularizer
# ──────────────────────────────────────────────────────────────────────────

def test_lambda_entropy_regularizer() -> None:
    """T12: L_λ is high at extremes, low at middle."""
    print("\n[T12] λ-entropy regularizer behavior")
    lam_soft_max = 50.0
    lam_hard_max = 10.0
    B = 16

    # All at middle
    lam_s_mid = torch.full((B,), lam_soft_max / 2)
    lam_h_mid = torch.full((B,), lam_hard_max / 2)
    L_mid = loss_lambda_entropy(lam_s_mid, lam_h_mid, lam_soft_max, lam_hard_max)

    # All at extremes (very small + very large mix)
    lam_s_ext = torch.cat([torch.full((B//2,), 1e-3),
                            torch.full((B//2,), lam_soft_max - 1e-3)])
    lam_h_ext = torch.cat([torch.full((B//2,), 1e-3),
                            torch.full((B//2,), lam_hard_max - 1e-3)])
    L_ext = loss_lambda_entropy(lam_s_ext, lam_h_ext, lam_soft_max, lam_hard_max)

    # All at zero (worst case for log)
    lam_s_zero = torch.full((B,), 0.0)
    lam_h_zero = torch.full((B,), 0.0)
    L_zero = loss_lambda_entropy(lam_s_zero, lam_h_zero, lam_soft_max, lam_hard_max)

    print(f"   λ at middle:    L_lreg = {L_mid.item():.4f}")
    print(f"   λ at extremes:  L_lreg = {L_ext.item():.4f}")
    print(f"   λ at zero:      L_lreg = {L_zero.item():.4f}")

    # The regularizer should penalise extremes more than middle
    assert L_ext.item() > L_mid.item(), \
        f"L_lreg at extremes ({L_ext.item():.4f}) should exceed middle ({L_mid.item():.4f})"
    assert L_zero.item() > L_mid.item()
    # Middle is finite and bounded
    assert torch.isfinite(L_mid) and L_mid.item() > 0

    print("   ✓ L_λ is minimum at the middle of the sigmoid range, "
          "as the form prescribes.")


# ──────────────────────────────────────────────────────────────────────────
# T13 — Full-pipeline gradient through combine_losses
# ──────────────────────────────────────────────────────────────────────────

def test_full_pipeline_gradient() -> None:
    """T13: integrator → episode_cost → CVaR → backward through combine_losses.

    Verifies that calling combine_losses on the integrator output and
    backpropagating produces a useful (non-NaN, non-zero) gradient on
    λ_soft. We use a risk-only navigation config so the expected sign is
    about the material-risk path specifically; default Step 2 weights also
    include goal/proximity terms whose CVaR tail can legitimately oppose
    the local risk gradient on a tiny batch.
    """
    print("\n[T13] Full-pipeline gradient through combine_losses")
    obs = _build_slow_leader_obs()
    cfg = WrapperConfig()
    B = 8
    t = _obs_to_tensors(obs, B=B)
    N = t["C"].shape[1]

    alphas = torch.full((B, N), 0.5)
    beta_t = torch.full((B,), 0.05)
    gamma_t = torch.full((B,), 0.5)
    lam_soft = torch.linspace(30.0, 65.0, B, requires_grad=True)
    lam_hard = torch.full((B,), 5.0)

    # Run integrator
    oT, vT, min_clear, cum_risk, hard_count, arc_length = integrate_surrogate_highway(
        o0=t["o0"], v0=t["v0"], goal=t["goal"],
        C=t["C"], R=t["R"], mask=t["mask"],
        alphas=alphas, beta=beta_t, gamma=gamma_t,
        lam_soft=lam_soft, lam_hard=lam_hard,
        rollout_patch=t["rollout_patch"],
        d_hat=t["d_hat"], dt=t["dt"], H=t["H"],
        cell_size_lon=cfg.risk_field.cell_size_lon,
        cell_size_lat=cfg.risk_field.cell_size_lat,
    )

    # Combine losses with a risk-only navigation term. This isolates the
    # load-bearing gradient path λ_soft → rollout → cum_risk → CVaR.
    weights = HighwayLossWeights.stage2()
    weights.w_goal = 0.0
    weights.w_len = 0.0
    weights.w_hard = 0.0
    weights.w_risk = 1.0
    weights.w_clear = 0.0
    weights.w_lreg = 0.0
    L, metrics = combine_losses(
        oT=oT, vT=vT, min_clear=min_clear,
        cum_risk=cum_risk, hard_count=hard_count, arc_length=arc_length,
        goal=t["goal"],
        o_tgt=None, v_tgt=None,    # no targets in this test
        lam_soft=lam_soft, lam_hard=lam_hard,
        cfg=weights,
    )

    L.backward()
    grad = lam_soft.grad

    print(f"   total loss = {L.item():.4f}")
    print(f"   metrics: L_nav={metrics['L_nav']:.4f}, L_clear={metrics['L_clear']:.4f}, "
          f"L_lreg={metrics['L_lreg']:.4f}")
    print(f"   J_mean={metrics['J_mean']:.4f}, J_max={metrics['J_max']:.4f}")
    print(f"   d(L)/d(λ_soft) per element: {grad.cpu().numpy()}")
    print(f"   mean grad: {grad.mean().item():+.6f}")

    assert torch.isfinite(grad).all(), "gradient contains NaN/inf"
    assert grad.abs().max() > 1e-6, "gradient is identically zero — pipeline disconnected"
    # In the risk-only slow-leader scene, increasing λ_soft reduces cum_risk
    # → reduces L_nav.
    # Mean gradient should be negative.
    assert grad.mean().item() < 0, (
        f"Expected mean d(L)/d(λ_soft) < 0 (raising λ should lower loss); "
        f"got {grad.mean().item():+.6f}"
    )
    print("   ✓ Full pipeline backprops correctly: integrator → cost → CVaR → loss.")


# ──────────────────────────────────────────────────────────────────────────
# T19 — Masked imitation calibration
# ──────────────────────────────────────────────────────────────────────────

def test_masked_imitation_matches_unmasked() -> None:
    """T19: all-ones mask preserves Stage 1 imitation-loss scale."""
    print("\n[T19] Masked imitation calibration")
    torch.manual_seed(0)
    B = 16
    oT = torch.randn(B, 2, requires_grad=True)
    vT = torch.randn(B, 2, requires_grad=True)
    o_tgt = torch.randn(B, 2)
    v_tgt = torch.randn(B, 2)

    L_traj_ref, L_vel_ref = loss_imitation(oT, vT, o_tgt, v_tgt)

    ones = torch.ones(B)
    L_traj_m, L_vel_m = loss_imitation_masked(oT, vT, o_tgt, v_tgt, ones)

    assert torch.allclose(L_traj_ref, L_traj_m, rtol=1e-5, atol=1e-7), \
        f"all-ones masked L_traj {L_traj_m} != unmasked {L_traj_ref}"
    assert torch.allclose(L_vel_ref, L_vel_m, rtol=1e-5, atol=1e-7), \
        f"all-ones masked L_vel {L_vel_m} != unmasked {L_vel_ref}"

    half = torch.zeros(B)
    half[:B // 2] = 1.0
    L_traj_half, L_vel_half = loss_imitation_masked(oT, vT, o_tgt, v_tgt, half)
    L_traj_ref_half, L_vel_ref_half = loss_imitation(
        oT[:B // 2], vT[:B // 2], o_tgt[:B // 2], v_tgt[:B // 2]
    )

    assert torch.allclose(L_traj_half, L_traj_ref_half, rtol=1e-5, atol=1e-7)
    assert torch.allclose(L_vel_half, L_vel_ref_half, rtol=1e-5, atol=1e-7)

    zeros = torch.zeros(B)
    L_traj_zero, L_vel_zero = loss_imitation_masked(oT, vT, o_tgt, v_tgt, zeros)
    assert L_traj_zero.item() == 0.0
    assert L_vel_zero.item() == 0.0
    (L_traj_zero + L_vel_zero).backward()
    assert oT.grad is not None and vT.grad is not None

    print(f"   unmasked: L_traj={L_traj_ref.item():.4f}, L_vel={L_vel_ref.item():.4f}")
    print(f"   half-mask: L_traj={L_traj_half.item():.4f}, L_vel={L_vel_half.item():.4f}")
    print("   ✓ Masked imitation preserves scale and zero-mask gradients are clean.")


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Episode-cost / loss-assembly tests (Step 4)")
    print("=" * 60)
    test_episode_cost_composition()
    test_stage_loss_schedule()
    test_imitation_targets_none()
    test_lambda_entropy_regularizer()
    test_full_pipeline_gradient()
    test_masked_imitation_matches_unmasked()
    print("\n" + "=" * 60)
    print("All Step 4 tests passed.")


if __name__ == "__main__":
    main()
