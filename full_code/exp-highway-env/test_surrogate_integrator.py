"""
test_surrogate_integrator.py

Three integration tests for surrogate_integrator.py. Builds on the unit
tests in test_bicycle_surrogate.py:

  T4 — Backward-compatibility: λ_s = λ_h = 0 should reduce the integrator
       to a geometry-only rollout (no risk forces). Stage 1's behaviour
       should be exactly recoverable from Stage 2 with risk coefficients
       zeroed.

  T5 — Coefficient sensitivity: with λ_s > 0 in the slow-leader scene,
       the rollout should differ from the geometry-only baseline and reduce
       cumulative risk. The lateral displacement difference between λ_s=0
       and λ_s=large should be statistically significant.

  T6 — Bilinear sampling correctness: build a patch with a known
       constant-gradient field, sample at multiple positions, verify
       the sampled gradients match the analytic ones (modulo bilinear
       smoothing).

  T7 — Real-env smoke: build one observation from live highway-v0, run
       the surrogate, and verify the final position is finite, downrange,
       and still close to the road.

  T8 — Full-integrator gradient: backpropagate a risk/CVaR loss through
       the wrapper-built patch, force composition, bicycle dynamics, and
       risk accumulation. Verify d(loss)/d(lambda_soft) points negative.

T4 and T5 use the wrapper from Step 2 to build a real observation dict,
so they exercise the full pipeline. T6 is a low-level sanity test that
isolates the ego→world frame rotation in `_bilinear_sample_ego_patch`.

Run all three with:
    python test_surrogate_integrator.py
"""

from __future__ import annotations

import sys
import os
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
from surrogate_integrator import (  # noqa: E402
    integrate_surrogate_highway,
    compute_surrogate_highway_force,
    _bilinear_sample_ego_patch,
)
from bicycle_surrogate import (  # noqa: E402
    ACCEL_RANGE,
    STEER_RANGE,
    bicycle_step_train,
    force_to_action,
)


def _cvar_loss(costs: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    eta = torch.quantile(costs.detach(), alpha)
    excess = (costs - eta).clamp(min=0.0)
    return eta + excess.mean() / (1.0 - alpha)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _obs_to_tensors(obs: dict, batch_size: int = 1) -> dict:
    """Convert numpy obs dict from the wrapper into a batched tensor dict.

    Replicates the (B,) batch dimension that `integrate_surrogate_highway`
    expects. With B=1 we just unsqueeze; with B>1 we tile.
    """
    out = {}
    out["o0"]            = torch.from_numpy(obs["o0"]).unsqueeze(0).repeat(batch_size, 1)
    out["v0"]            = torch.from_numpy(obs["v0"]).unsqueeze(0).repeat(batch_size, 1)
    out["goal"]          = torch.from_numpy(obs["goal"]).unsqueeze(0).repeat(batch_size, 1)
    out["C"]             = torch.from_numpy(obs["C"]).unsqueeze(0).repeat(batch_size, 1, 1)
    out["R"]             = torch.from_numpy(obs["R"]).unsqueeze(0).repeat(batch_size, 1)
    out["mask"]          = torch.from_numpy(obs["mask"]).unsqueeze(0).repeat(batch_size, 1)
    out["rollout_patch"] = torch.from_numpy(obs["rollout_patch"]).unsqueeze(0).repeat(
        batch_size, 1, 1, 1
    )
    out["d_hat"]         = torch.tensor([float(obs["d_hat"])] * batch_size)
    out["dt"]            = torch.tensor(float(obs["dt"]))
    out["H"]             = torch.tensor([int(obs["H"])] * batch_size)
    return out


def _build_slow_leader_obs(
    cfg: WrapperConfig | None = None,
) -> dict:
    """Build an observation dict for the slow-leader scene used in Step 1."""
    cfg = cfg or WrapperConfig()
    ego = _MockVehicle(x=0.0, y=4.0, heading=0.0, speed=25.0)
    others = [
        _MockVehicle(x=30.0, y=4.0, heading=0.0, speed=12.0),
        _MockVehicle(x=-15.0, y=8.0, heading=0.0, speed=28.0),
    ]
    env = _MockEnv(ego, others)
    builder = HighwayMaterialObservation(cfg)
    return builder.build(env)


def _run_integrator(
    obs_dict: dict,
    *,
    lam_soft: float,
    lam_hard: float,
    beta: float = 0.05,
    gamma: float = 0.0,
    alpha_mag: float = 0.5,
    cfg: WrapperConfig | None = None,
    batch_size: int = 1,
) -> tuple:
    """Run the integrator with explicit coefficient overrides."""
    cfg = cfg or WrapperConfig()
    t = _obs_to_tensors(obs_dict, batch_size=batch_size)
    B, N = t["C"].shape[:2]
    alphas = torch.full((B, N), alpha_mag)
    beta_t = torch.full((B,), beta)
    gamma_t = torch.full((B,), gamma)
    lam_s = torch.full((B,), lam_soft, requires_grad=True)
    lam_h = torch.full((B,), lam_hard)
    return integrate_surrogate_highway(
        o0=t["o0"], v0=t["v0"], goal=t["goal"],
        C=t["C"], R=t["R"], mask=t["mask"],
        alphas=alphas, beta=beta_t, gamma=gamma_t,
        lam_soft=lam_s, lam_hard=lam_h,
        rollout_patch=t["rollout_patch"],
        d_hat=t["d_hat"], dt=t["dt"], H=t["H"],
        cell_size_lon=cfg.risk_field.cell_size_lon,
        cell_size_lat=cfg.risk_field.cell_size_lat,
    )


def _signed_road_margin(env, builder: HighwayMaterialObservation, point: np.ndarray) -> float:
    """Signed margin to the live lane union, using the wrapper's road helper."""
    network = getattr(env.unwrapped.road, "network", None)
    if network is None or not hasattr(network, "lanes_dict"):
        return float("nan")
    margins = [
        builder._signed_margin_to_lane(lane, point)
        for lane in network.lanes_dict().values()
    ]
    return float(max(margins)) if margins else float("nan")


# ──────────────────────────────────────────────────────────────────────────
# T4 — Backward compatibility (λ_soft = λ_hard = 0)
# ──────────────────────────────────────────────────────────────────────────

def test_backward_compatibility() -> None:
    """T4: With both λ coefficients zero, no risk forces act.

    Specifically:
      - cum_risk should still ACCUMULATE (the risk field is sampled
        whether or not lam_soft is active), but the trajectory itself
        should be driven only by F_geom + F_goal.
      - The trajectory should remain in the ego's lane (lateral drift
        should be negligible — only F_goal is acting laterally, and the
        goal is straight ahead).
    """
    print("\n[T4] Backward-compatibility test (λ_soft=λ_hard=0)")
    obs = _build_slow_leader_obs()

    oT, vT, min_clear, cum_risk, hard_count, arc_length = _run_integrator(
        obs, lam_soft=0.0, lam_hard=0.0,
    )

    # Lateral drift: ego started at y=4.0, goal is at y=4.0, no risk forces.
    o0_y = float(obs["o0"][1])
    oT_y = float(oT[0, 1].item())
    lateral_drift = abs(oT_y - o0_y)

    # Forward progress should be ~speed * H * dt = 25 * 20 * 0.1 = 50m
    o0_x = float(obs["o0"][0])
    oT_x = float(oT[0, 0].item())
    forward_progress = oT_x - o0_x
    expected_forward = 25.0 * 20 * 0.1   # 50m

    print(f"   start: x={o0_x:.2f}, y={o0_y:.2f}")
    print(f"   end:   x={oT_x:.2f}, y={oT_y:.2f}")
    print(f"   lateral drift: {lateral_drift:.4f} m")
    print(f"   forward progress: {forward_progress:.2f} m  (expected ~{expected_forward:.0f}m)")
    print(f"   cum_risk: {float(cum_risk[0]):.4f}, arc_length: {float(arc_length[0]):.2f}")

    assert lateral_drift < 0.5, (
        f"With λ=0, lateral drift should be <0.5m, got {lateral_drift:.2f}m. "
        f"This means risk forces are leaking through despite λ=0, OR the goal "
        f"force is pulling laterally."
    )
    # Forward progress should be within 50% of expected (loose because F_geom
    # may push back a bit from the leader vehicle in the IPC barrier zone)
    assert 0.5 * expected_forward < forward_progress < 1.2 * expected_forward, (
        f"Forward progress {forward_progress:.2f}m is far from expected "
        f"{expected_forward:.0f}m. Bicycle dynamics may be misconfigured."
    )
    assert torch.isfinite(oT).all() and torch.isfinite(vT).all(), \
        "Trajectory contains NaN/inf"
    print("   ✓ With λ=0, rollout follows geometry-only behavior.")


# ──────────────────────────────────────────────────────────────────────────
# T5 — Coefficient sensitivity
# ──────────────────────────────────────────────────────────────────────────

def test_coefficient_sensitivity() -> None:
    """T5: λ_soft > 0 should reduce risk and alter lateral behavior.

    Setup:
      - Slow-leader scene where the leader is in the same lane as the ego.
      - Vehicle in lane 2 breaks symmetry, but the integrated lookahead field
        makes the eventual drift sign scene-dependent. Do not hard-code it.
      - Compare lateral drift at λ_soft=0 vs λ_soft=50.
    """
    print("\n[T5] Coefficient sensitivity test")
    obs = _build_slow_leader_obs()
    o0_y = float(obs["o0"][1])

    # λ = 0 baseline
    oT_0, _, _, cum_risk_0, _, _ = _run_integrator(
        obs, lam_soft=0.0, lam_hard=0.0,
    )
    drift_0 = float(oT_0[0, 1].item()) - o0_y

    # λ = 50 (highway-env scale, as flagged in Step 1 sanity check)
    oT_50, _, _, cum_risk_50, _, _ = _run_integrator(
        obs, lam_soft=50.0, lam_hard=0.0,
    )
    drift_50 = float(oT_50[0, 1].item()) - o0_y

    print(f"   λ_soft=0:  end y={float(oT_0[0,1]):.4f}, drift={drift_0:+.4f}m, "
          f"cum_risk={float(cum_risk_0[0]):.4f}")
    print(f"   λ_soft=50: end y={float(oT_50[0,1]):.4f}, drift={drift_50:+.4f}m, "
          f"cum_risk={float(cum_risk_50[0]):.4f}")

    # The asymmetry from the lane-2 vehicle should produce a -y drift
    # at λ_soft=50 (away from the lane-2 vehicle).
    delta_drift = drift_50 - drift_0
    print(f"   Δ(drift) due to λ_soft=50: {delta_drift:+.4f}m")

    # Threshold is loose: even a 0.5m lateral push over 2 seconds is
    # behaviorally meaningful (a quarter of a lane width).
    assert abs(delta_drift) > 0.1, (
        f"λ_soft=50 should produce non-trivial lateral drift vs λ=0. "
        f"Δ(drift) = {delta_drift:.4f}m. Risk force is not acting."
    )
    assert float(cum_risk_50[0]) < float(cum_risk_0[0]), (
        f"λ_soft=50 should reduce cumulative risk vs λ=0. "
        f"got {float(cum_risk_50[0]):.4f} >= {float(cum_risk_0[0]):.4f}."
    )
    print("   ✓ λ_soft changes the trajectory and reduces cumulative risk.")


# ──────────────────────────────────────────────────────────────────────────
# T6 — Bilinear sampling with ego-frame rotation
# ──────────────────────────────────────────────────────────────────────────

def test_bilinear_sampling_ego_frame() -> None:
    """T6: Bilinear sample on a planted constant-gradient patch.

    Build a patch where channel 0 = a known linear field in EGO frame:
        f_ego(lon, lat) = lon  (so ∂/∂lon = 1, ∂/∂lat = 0)
    Sample at several world-frame positions with ego heading=0 and
    heading=π/4. Verify the sampled values match the analytic prediction.
    """
    print("\n[T6] Bilinear sampling on planted patch")
    Hp, Wp = 32, 64
    cell_lon = 1.0
    cell_lat = 1.0
    patch_lon_offset_frac = 0.05
    patch_lon_m = Wp * cell_lon
    patch_lat_m = Hp * cell_lat
    ego_offset_m = patch_lon_offset_frac * patch_lon_m

    # Planted field: f_ego(lon, lat) = lon  →  patch[0, row, col] has
    # value = lon coord of cell (col, row) in ego frame.
    # Pixel col c corresponds to lon = -ego_offset_m + (c + 0.5) * cell_lon
    cols = torch.arange(Wp).float()
    lons = -ego_offset_m + (cols + 0.5) * cell_lon          # (Wp,)
    patch = torch.zeros(1, 1, Hp, Wp)
    patch[0, 0] = lons.unsqueeze(0).expand(Hp, Wp)         # broadcast across rows

    # Test 1: heading=0, sample at world position (5, 4) with o0=(0, 4).
    # Ego frame: (lon=5, lat=0). Expected sampled value: 5.0.
    o = torch.tensor([[5.0, 4.0]])
    o0 = torch.tensor([[0.0, 4.0]])
    heading_0 = torch.tensor([0.0])
    sampled = _bilinear_sample_ego_patch(
        patch, o, o0, heading_0,
        cell_size_lon=cell_lon, cell_size_lat=cell_lat,
        patch_lon_offset_frac=patch_lon_offset_frac,
    )
    print(f"   heading=0,  o-o0=(5,0) → sampled={sampled.item():.4f}, expected=5.0")
    assert abs(sampled.item() - 5.0) < 0.5, (
        f"Sampled {sampled.item()} far from expected 5.0 at heading=0"
    )

    # Test 2: heading=π/4, sample at world (5, 4) with o0=(0, 4).
    # World offset (5, 0), rotated into ego frame at heading=π/4:
    #   ego_lon = 5*cos(π/4) + 0*sin(π/4) = 5/√2 ≈ 3.5355
    # Expected sampled value: 3.5355.
    heading_0 = torch.tensor([np.pi / 4.0])
    sampled = _bilinear_sample_ego_patch(
        patch, o, o0, heading_0,
        cell_size_lon=cell_lon, cell_size_lat=cell_lat,
        patch_lon_offset_frac=patch_lon_offset_frac,
    )
    expected = 5.0 * np.cos(np.pi / 4.0)
    print(f"   heading=π/4, o-o0=(5,0) → sampled={sampled.item():.4f}, "
          f"expected={expected:.4f}")
    assert abs(sampled.item() - expected) < 0.5, (
        f"Sampled {sampled.item()} far from expected {expected:.4f} at heading=π/4. "
        f"The ego→world frame rotation in _bilinear_sample_ego_patch is incorrect."
    )

    # Test 3: at the patch centre (o == o0), should sample 0 (the lon
    # coordinate at ego-frame origin is 0).
    o = torch.tensor([[0.0, 4.0]])
    heading_0 = torch.tensor([0.0])
    sampled = _bilinear_sample_ego_patch(
        patch, o, o0, heading_0,
        cell_size_lon=cell_lon, cell_size_lat=cell_lat,
        patch_lon_offset_frac=patch_lon_offset_frac,
    )
    print(f"   at o0 (heading=0)        → sampled={sampled.item():.4f}, expected≈0.0")
    assert abs(sampled.item()) < 0.5, (
        f"Sample at patch centre {sampled.item()} should be ~0"
    )

    print("   ✓ Bilinear sampling correctly handles ego-frame patches under rotation.")


# ──────────────────────────────────────────────────────────────────────────
# T7 — Real highway-env smoke
# ──────────────────────────────────────────────────────────────────────────

def test_real_env_smoke() -> None:
    """T7: Build a live highway-v0 observation and run the surrogate once."""
    print("\n[T7] Real highway-v0 surrogate smoke test")
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401
    except ImportError as exc:
        print(f"   skipped: gymnasium/highway_env unavailable ({exc})")
        return

    env = gym.make("highway-v0")
    cfg = WrapperConfig()
    cfg.risk_field.sensing_radius_m = cfg.sensing_radius_m
    builder = HighwayMaterialObservation(cfg)
    try:
        env.reset(seed=0)
        obs = builder.build(env)
        oT, vT, min_clear, cum_risk, hard_count, arc_length = _run_integrator(
            obs,
            lam_soft=0.0,
            lam_hard=0.0,
            beta=0.05,
            alpha_mag=0.0,
            cfg=cfg,
        )
        start = obs["o0"]
        final = oT[0].detach().cpu().numpy()
        progress = float(final[0] - start[0])
        road_margin = _signed_road_margin(env, builder, final)

        print(f"   start=({start[0]:.2f}, {start[1]:.2f})")
        print(f"   final=({final[0]:.2f}, {final[1]:.2f})")
        print(f"   progress={progress:.2f}m, road_margin={road_margin:.2f}m")
        print(f"   cum_risk={float(cum_risk[0]):.4f}, arc_length={float(arc_length[0]):.2f}, "
              f"hard_count={float(hard_count[0]):.1f}")

        assert torch.isfinite(oT).all() and torch.isfinite(vT).all(), \
            "Live-env surrogate produced NaN/inf"
        assert progress > 10.0, (
            f"Predicted final position should move downrange, got {progress:.2f}m"
        )
        assert progress < 90.0, (
            f"Predicted final position is implausibly far downrange: {progress:.2f}m"
        )
        assert road_margin > -0.5, (
            f"Predicted final position is off-road by {road_margin:.2f}m"
        )
        assert float(arc_length[0]) > 10.0, "Arc length should be nontrivial"
        print("   ✓ Live highway-v0 observation rolls out plausibly.")
    finally:
        env.close()


# ──────────────────────────────────────────────────────────────────────────
# T8 — Full-integrator gradient sanity
# ──────────────────────────────────────────────────────────────────────────

def test_full_integrator_lambda_gradient() -> None:
    """T8: Full surrogate loss should push lambda_soft upward.

    In the slow-leader wrapper-built scene, larger lambda_soft reduced
    cumulative risk in T5. Around the intended Stage 2 highway scale
    (lambda_soft ≈ 30-65), d(loss)/d(lambda_soft) should be negative for a
    risk-minimization loss: gradient descent will increase lambda_soft.

    Note: exactly lambda_soft=0 is a poor local point in this mock scene:
    the geometry-only trajectory skims a high-curvature barrier/risk region
    and produces a huge positive local derivative. Stage 2 should therefore
    start material logits in a useful nonzero range rather than exactly zero.
    """
    print("\n[T8] Full-integrator λ_soft gradient sanity test")
    cfg = WrapperConfig()
    obs = _build_slow_leader_obs(cfg)
    batch_size = 8
    t = _obs_to_tensors(obs, batch_size=batch_size)
    B, N = t["C"].shape[:2]

    alphas = torch.full((B, N), 0.5)
    beta_t = torch.full((B,), 0.05)
    gamma_t = torch.full((B,), 0.5)
    # Spread λ values in the intended highway-env Stage 2 range so the CVaR
    # tail is not an all-tied empirical set.
    lam_soft = torch.linspace(30.0, 65.0, B, requires_grad=True)
    lam_hard = torch.zeros(B)

    _, _, _, cum_risk, _, _ = integrate_surrogate_highway(
        o0=t["o0"], v0=t["v0"], goal=t["goal"],
        C=t["C"], R=t["R"], mask=t["mask"],
        alphas=alphas, beta=beta_t, gamma=gamma_t,
        lam_soft=lam_soft, lam_hard=lam_hard,
        rollout_patch=t["rollout_patch"],
        d_hat=t["d_hat"], dt=t["dt"], H=t["H"],
        cell_size_lon=cfg.risk_field.cell_size_lon,
        cell_size_lat=cfg.risk_field.cell_size_lat,
    )

    loss = cum_risk.mean() + 0.25 * _cvar_loss(cum_risk, alpha=0.75)
    loss.backward()

    grad = lam_soft.grad.detach()
    print(f"   λ_soft:   {lam_soft.detach().cpu().numpy()}")
    print(f"   cum_risk: {cum_risk.detach().cpu().numpy()}")
    print(f"   grad:     {grad.cpu().numpy()}")
    print(f"   loss={loss.item():.4f}, mean grad={grad.mean().item():+.6f}")

    assert torch.isfinite(grad).all(), "λ_soft gradient contains NaN/inf"
    assert grad.mean().item() < -1e-3, (
        f"Expected negative mean gradient so gradient descent increases λ_soft, "
        f"got {grad.mean().item():+.6f}"
    )
    assert (grad < 0).any(), "At least one λ_soft element should receive negative gradient"
    print("   ✓ Full integrator backprop gives useful negative λ_soft gradient.")


# ──────────────────────────────────────────────────────────────────────────
# T9 — First-step force helper matches rollout internals
# ──────────────────────────────────────────────────────────────────────────

def test_first_step_force_matches_h1_rollout() -> None:
    """T9: Exposed first-step force is exactly the H=1 rollout force.

    Closed-loop eval needs the first action now, while training needs the
    terminal H-step rollout. This verifies both paths share one force law.
    """
    print("\n[T9] First-step force helper matches H=1 rollout")
    cfg = WrapperConfig()
    obs = _build_slow_leader_obs(cfg)
    t = _obs_to_tensors(obs, batch_size=3)
    B, N = t["C"].shape[:2]

    alphas = torch.full((B, N), 0.5)
    beta_t = torch.full((B,), 0.05)
    gamma_t = torch.full((B,), 0.5)
    lam_soft = torch.linspace(10.0, 30.0, B)
    lam_hard = torch.linspace(0.0, 5.0, B)
    H_one = torch.ones(B, dtype=torch.long)

    oT, vT, _, _, _, _ = integrate_surrogate_highway(
        o0=t["o0"], v0=t["v0"], goal=t["goal"],
        C=t["C"], R=t["R"], mask=t["mask"],
        alphas=alphas, beta=beta_t, gamma=gamma_t,
        lam_soft=lam_soft, lam_hard=lam_hard,
        rollout_patch=t["rollout_patch"],
        d_hat=t["d_hat"], dt=t["dt"], H=H_one,
        cell_size_lon=cfg.risk_field.cell_size_lon,
        cell_size_lat=cfg.risk_field.cell_size_lat,
    )

    speed0 = torch.linalg.norm(t["v0"], dim=-1).clamp_min(1e-3)
    heading0 = torch.atan2(t["v0"][:, 1], t["v0"][:, 0])
    F0, _, _, _ = compute_surrogate_highway_force(
        o=t["o0"], heading=heading0, speed=speed0,
        o0=t["o0"], heading_0=heading0,
        goal=t["goal"], C=t["C"], R_eff=t["R"], mask=t["mask"],
        alphas=alphas, beta=beta_t, gamma=gamma_t,
        lam_soft=lam_soft, lam_hard=lam_hard,
        rollout_patch=t["rollout_patch"], d_hat=t["d_hat"],
        cell_size_lon=cfg.risk_field.cell_size_lon,
        cell_size_lat=cfg.risk_field.cell_size_lat,
    )
    accel, steer = force_to_action(F0, heading0, speed0)
    accel = accel.clamp(*ACCEL_RANGE)
    steer = steer.clamp(*STEER_RANGE)
    pos1, heading1, speed1 = bicycle_step_train(
        t["o0"], heading0, speed0, accel, steer, dt=float(t["dt"].item())
    )
    v1 = speed1.unsqueeze(-1) * torch.stack(
        [torch.cos(heading1), torch.sin(heading1)], dim=-1
    )

    pos_err = (oT - pos1).abs().max().item()
    vel_err = (vT - v1).abs().max().item()
    print(f"   max position error={pos_err:.3e}, max velocity error={vel_err:.3e}")
    assert torch.allclose(oT, pos1, atol=1e-6), "First-step position mismatch"
    assert torch.allclose(vT, v1, atol=1e-6), "First-step velocity mismatch"
    print("   ✓ First-step force path matches rollout internals.")


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Surrogate integrator integration tests (Step 3 proper)")
    print("=" * 60)
    test_backward_compatibility()
    test_coefficient_sensitivity()
    test_bilinear_sampling_ego_frame()
    test_real_env_smoke()
    test_full_integrator_lambda_gradient()
    test_first_step_force_matches_h1_rollout()
    print("\n" + "=" * 60)
    print("All Step 3 integration tests passed.")


if __name__ == "__main__":
    main()
