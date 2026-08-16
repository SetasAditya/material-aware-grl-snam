import math

import numpy as np
import pytest
import torch

from rebuttal_experiments.exp1_gate_ablation import (
    primitive_feasibility_gate,
)
from repair_experiments.v1_controller import (
    DirectWaypointLatch,
    HysteresisConfig,
    StatefulFeasibilityGate,
    TemporalObstacleReleaseGate,
    TemporalReleaseConfig,
    WaypointLatchConfig,
    align_rollout_soft_force,
    direct_activation_admitted,
    enumerate_stagewise_primitive_candidates,
    primitive_direction_xy,
    primitive_ray_is_hard_feasible,
    project_vectors_to_cone,
)


def test_rc_to_xy_direction_conversion():
    np.testing.assert_allclose(primitive_direction_xy((1.0, 0.0)), [0.0, 1.0])
    np.testing.assert_allclose(primitive_direction_xy((0.0, -2.0)), [-1.0, 0.0])


def test_cone_projection_preserves_magnitude_and_constrains_angle():
    vectors = torch.tensor([[[[-1.0]], [[0.0]]]])
    projected, changed = project_vectors_to_cone(
        vectors, torch.tensor([[1.0, 0.0]]), half_angle_degrees=30.0
    )
    assert bool(changed.item())
    assert torch.linalg.vector_norm(projected).item() == pytest.approx(1.0)
    cosine = projected[0, 0, 0, 0].item()
    assert cosine == pytest.approx(math.cos(math.radians(30.0)), abs=1e-6)


def test_cone_projection_leaves_inside_and_zero_vectors_unchanged():
    vectors = torch.tensor([[[[1.0, 0.0]], [[0.0, 0.0]]]])
    projected, changed = project_vectors_to_cone(
        vectors, torch.tensor([[1.0, 0.0]]), half_angle_degrees=20.0
    )
    torch.testing.assert_close(projected, vectors)
    assert not bool(changed.any())


def test_patch_alignment_changes_only_risk_gradient_channels():
    patch = torch.zeros(1, 6, 5, 5)
    patch[:, 0] = 0.7
    patch[:, 1] = 4.0
    patch[:, 2] = 1.0  # soft force is left, while selected ray is right
    patch[:, 4] = 0.3
    aligned, diagnostics = align_rollout_soft_force(
        patch, (0.0, 1.0), half_angle_degrees=25.0
    )
    torch.testing.assert_close(aligned[:, 0:2], patch[:, 0:2])
    torch.testing.assert_close(aligned[:, 4:6], patch[:, 4:6])
    assert diagnostics.center_alignment_after >= math.cos(math.radians(25.0)) - 1e-6
    assert diagnostics.projected_pixel_fraction == pytest.approx(1.0)
    assert torch.linalg.vector_norm(aligned[:, 2:4], dim=1).mean().item() == pytest.approx(1.0)


def test_patch_alignment_makes_lambda_the_force_magnitude():
    patch = torch.zeros(1, 6, 3, 3)
    patch[:, 2] = -0.003
    patch[:, 3] = -0.004
    aligned, diagnostics = align_rollout_soft_force(
        patch,
        (0.0, 1.0),
        half_angle_degrees=45.0,
        gradient_confidence_threshold=1e-4,
    )
    # The canonical integrator computes F_soft=-lambda*grad.  A unit gradient
    # therefore makes ||F_soft|| exactly lambda.
    torch.testing.assert_close(
        torch.linalg.vector_norm(aligned[:, 2:4], dim=1),
        torch.ones(1, 3, 3),
    )
    assert not diagnostics.center_low_confidence_fallback


def test_flat_gradient_falls_back_to_selected_axis_and_logs_it():
    patch = torch.zeros(1, 6, 3, 3)
    patch[:, 0] = 0.8
    patch[:, 1] = 2.0
    patch[:, 4] = 0.7
    patch[:, 5] = -0.2
    aligned, diagnostics = align_rollout_soft_force(
        patch,
        (1.0, 0.0),
        half_angle_degrees=30.0,
        gradient_confidence_threshold=0.01,
        low_confidence_fallback_policy="selected_axis",
    )
    expected_gradient = torch.zeros(1, 2, 3, 3)
    expected_gradient[:, 1] = -1.0  # selected row direction -> +y force
    torch.testing.assert_close(aligned[:, 2:4], expected_gradient)
    torch.testing.assert_close(aligned[:, 0:2], patch[:, 0:2])
    torch.testing.assert_close(aligned[:, 4:6], patch[:, 4:6])
    assert diagnostics.center_low_confidence_fallback
    assert diagnostics.low_confidence_pixel_fraction == pytest.approx(1.0)


def test_flat_gradient_zero_policy_disables_force():
    patch = torch.zeros(1, 6, 3, 3)
    aligned, diagnostics = align_rollout_soft_force(
        patch,
        (0.0, 1.0),
        half_angle_degrees=30.0,
        gradient_confidence_threshold=0.01,
        low_confidence_fallback_policy="zero",
    )
    torch.testing.assert_close(aligned, patch)
    assert diagnostics.center_low_confidence_fallback


def test_gate_requires_persistence_holds_dwell_then_uses_off_persistence():
    gate = StatefulFeasibilityGate(
        HysteresisConfig(
            on_persistence_steps=2,
            off_persistence_steps=2,
            minimum_dwell_steps=4,
        )
    )
    high = dict(nominal_risk=0.8, best_risk=0.2, feasible_count=1)
    low = dict(nominal_risk=0.1, best_risk=0.1, feasible_count=1)
    assert not gate.update(**high, hard_hazard_override=False).active
    update = gate.update(**high, hard_hazard_override=False)
    assert update.active and update.reason == "activate_after_persistence"
    # Dwell prevents early deactivation even with repeated off evidence.
    assert gate.update(**low, hard_hazard_override=False).active
    assert gate.update(**low, hard_hazard_override=False).active
    assert not gate.update(**low, hard_hazard_override=False).active


def test_hard_override_is_immediate_and_resets_state():
    gate = StatefulFeasibilityGate(
        HysteresisConfig(on_persistence_steps=1, minimum_dwell_steps=10)
    )
    high = dict(nominal_risk=0.8, best_risk=0.2, feasible_count=1)
    assert gate.update(**high, hard_hazard_override=False).active
    update = gate.update(**high, hard_hazard_override=True)
    assert not update.active
    assert update.transitioned
    assert update.reason == "hard_hazard_override"
    assert update.dwell_steps == 0


def test_stateful_gate_absorbs_flickering_learned_magnitude():
    gate = StatefulFeasibilityGate(
        HysteresisConfig(
            on_persistence_steps=2,
            off_persistence_steps=2,
            minimum_dwell_steps=4,
        )
    )
    high = dict(nominal_risk=0.8, best_risk=0.2, feasible_count=1)
    eligibility = [True, False] * 6
    stateless = [int(item) for item in eligibility]
    stateful = [
        int(
            gate.update(
                **high,
                magnitude_eligible=item,
                hard_hazard_override=False,
            ).active
        )
        for item in eligibility
    ]
    stateless_transitions = sum(
        stateless[index] != stateless[index - 1]
        for index in range(1, len(stateless))
    )
    stateful_transitions = sum(
        stateful[index] != stateful[index - 1]
        for index in range(1, len(stateful))
    )
    assert stateful_transitions < stateless_transitions
    assert stateful_transitions == 0


def test_magnitude_drop_deactivates_only_after_dwell_and_persistence():
    gate = StatefulFeasibilityGate(
        HysteresisConfig(
            on_persistence_steps=1,
            off_persistence_steps=2,
            minimum_dwell_steps=3,
        )
    )
    high = dict(nominal_risk=0.8, best_risk=0.2, feasible_count=1)
    assert gate.update(
        **high, magnitude_eligible=True, hard_hazard_override=False
    ).active
    first_drop = gate.update(
        **high, magnitude_eligible=False, hard_hazard_override=False
    )
    assert first_drop.active
    assert first_drop.reason == "minimum_dwell"
    second_drop = gate.update(
        **high, magnitude_eligible=False, hard_hazard_override=False
    )
    assert not second_drop.active
    assert second_drop.reason == "deactivate_after_hysteresis"


def test_direct_waypoint_rolls_target_but_keeps_activation_direction():
    latch = DirectWaypointLatch(
        WaypointLatchConfig(distance_m=1.0),
        gsd=0.5,
    )
    position = np.asarray([10.0, 10.0], dtype=np.float32)
    update = latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert update.active
    assert update.transitioned
    assert update.reason == "activate_from_feasibility_gate"
    np.testing.assert_allclose(update.waypoint_xy, [12.0, 10.0])

    moved = position + [0.4, 0.3]
    update = latch.update(
        position_xy=moved,
        gate_active=False,
        # A changing gate selection cannot rotate the active direction.
        selected_direction_rc=(1.0, 0.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert update.active
    assert update.reason == "hold_rolling_target"
    np.testing.assert_allclose(update.direction_rc, [0.0, 1.0])
    np.testing.assert_allclose(update.waypoint_xy, moved + [2.0, 0.0])
    assert update.forward_progress_delta_m == pytest.approx(0.2)
    assert update.cumulative_forward_progress_m == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("trigger", "reason"),
    [
        ("hard", "hard_hazard_override"),
        ("invalid", "latched_primitive_invalid"),
    ],
)
def test_direct_waypoint_immediate_safety_release(trigger, reason):
    latch = DirectWaypointLatch(WaypointLatchConfig(), gsd=0.5)
    position = np.asarray([10.0, 10.0], dtype=np.float32)
    latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    update = latch.update(
        position_xy=position,
        gate_active=False,
        selected_direction_rc=(0.0, 0.0),
        latched_direction_valid=trigger != "invalid",
        hard_hazard_override=trigger == "hard",
    )
    assert not update.active
    assert update.transitioned
    assert update.reason == reason


def test_direct_waypoint_active_step_limit_is_bounded():
    latch = DirectWaypointLatch(
        WaypointLatchConfig(active_step_limit=3),
        gsd=0.5,
    )
    position = np.asarray([10.0, 10.0], dtype=np.float32)
    latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    for expected_active_steps in (2, 3):
        update = latch.update(
            position_xy=position,
            gate_active=False,
            selected_direction_rc=(0.0, 0.0),
            latched_direction_valid=True,
            hard_hazard_override=False,
        )
        assert update.active
        assert update.active_steps == expected_active_steps
    update = latch.update(
        position_xy=position,
        gate_active=False,
        selected_direction_rc=(0.0, 0.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert not update.active
    assert update.reason == "active_step_limit_reached"
    assert update.active_steps == 3


def test_latched_ray_validity_checks_only_requested_waypoint_horizon():
    maps = {
        "hard_mask": np.zeros((9, 9), dtype=bool),
        "sdf_hard": np.full((9, 9), 5.0, dtype=np.float32),
    }
    # The waypoint two cells to the right is clear; a hazard six cells away
    # must not invalidate that short target.
    maps["hard_mask"][4, 7] = True
    maps["sdf_hard"][4, 7] = 0.0
    position_xy = np.asarray([1.0, 4.0], dtype=np.float32)
    assert primitive_ray_is_hard_feasible(
        maps,
        position_xy,
        (0.0, 1.0),
        horizon_cells=2,
        hard_margin_m=1.0,
    )
    assert not primitive_ray_is_hard_feasible(
        maps,
        position_xy,
        (0.0, 1.0),
        horizon_cells=6,
        hard_margin_m=1.0,
    )


def _candidate_enumeration_fixture():
    rows, cols = np.indices((31, 31))
    risk = (
        0.15
        + 0.65 * (cols / 30.0)
        + 0.1 * np.sin(rows / 3.0)
    ).astype(np.float32)
    hard = np.zeros((31, 31), dtype=bool)
    sdf = np.full((31, 31), 4.0, dtype=np.float32)
    # Block some, but not all, progress-making rays.
    hard[15, 18] = True
    hard[13, 18] = True
    sdf[hard] = 0.0
    return {
        "risk_map": risk,
        "hard_mask": hard,
        "sdf_hard": sdf,
    }


def test_candidate_enumeration_is_exactly_equivalent_to_current_gate():
    maps = _candidate_enumeration_fixture()
    position_xy = np.asarray([15.0, 15.0], dtype=np.float32)
    goal_xy = np.asarray([27.0, 15.0], dtype=np.float32)
    kwargs = dict(
        primitive_count=16,
        horizon_cells=6,
        hard_margin_m=1.0,
    )
    enumeration = enumerate_stagewise_primitive_candidates(
        maps, position_xy, goal_xy, **kwargs
    )
    gate = primitive_feasibility_gate(
        maps,
        position_xy,
        goal_xy,
        **kwargs,
        improvement_margin=0.05,
        material_trigger=0.45,
    )

    assert len(enumeration.candidates) == gate.feasible_count
    assert enumeration.nominal_mean_risk == pytest.approx(
        gate.nominal_risk, abs=0.0
    )
    best = min(enumeration.candidates, key=lambda item: item.mean_risk)
    assert best.mean_risk == pytest.approx(gate.best_risk, abs=0.0)
    np.testing.assert_allclose(
        best.direction_rc, gate.selected_direction_rc, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        best.endpoint_rc, gate.selected_endpoint_rc, rtol=0.0, atol=0.0
    )
    assert best.min_clearance_m == pytest.approx(
        gate.selected_min_clearance_m, abs=0.0
    )


def test_candidate_enumeration_is_deterministic_and_index_ordered():
    maps = _candidate_enumeration_fixture()
    arguments = dict(
        maps=maps,
        position_xy=np.asarray([15.0, 15.0], dtype=np.float32),
        goal_xy=np.asarray([27.0, 15.0], dtype=np.float32),
        primitive_count=32,
        horizon_cells=7,
        hard_margin_m=1.0,
    )
    first = enumerate_stagewise_primitive_candidates(**arguments)
    second = enumerate_stagewise_primitive_candidates(**arguments)
    assert first == second
    indices = [item.primitive_index for item in first.candidates]
    assert indices == sorted(indices)

    position_rc = arguments["position_xy"][::-1]
    goal_rc = arguments["goal_xy"][::-1]
    initial_distance = float(np.linalg.norm(goal_rc - position_rc))
    for candidate in first.candidates:
        endpoint_distance = float(
            np.linalg.norm(goal_rc - np.asarray(candidate.endpoint_rc))
        )
        assert endpoint_distance < initial_distance - 0.5
        assert candidate.min_clearance_m >= arguments["hard_margin_m"]


def test_receding_enumeration_includes_backward_hard_feasible_rays():
    maps = {
        "risk_map": np.zeros((31, 31), dtype=np.float32),
        "hard_mask": np.zeros((31, 31), dtype=bool),
        "sdf_hard": np.full((31, 31), 5.0, dtype=np.float32),
    }
    kwargs = dict(
        maps=maps,
        position_xy=np.asarray([15.0, 15.0], dtype=np.float32),
        goal_xy=np.asarray([25.0, 15.0], dtype=np.float32),
        primitive_count=16,
        horizon_cells=5,
        hard_margin_m=1.0,
    )
    legacy = enumerate_stagewise_primitive_candidates(**kwargs)
    receding = enumerate_stagewise_primitive_candidates(
        **kwargs, require_endpoint_progress=False
    )
    assert len(receding.candidates) == 16
    assert len(legacy.candidates) < len(receding.candidates)
    assert {item.primitive_index for item in receding.candidates} == set(
        range(16)
    )


def test_direct_waypoint_is_one_shot_until_feasibility_state_rearms():
    latch = DirectWaypointLatch(
        WaypointLatchConfig(
            active_step_limit=1,
            rearm_inactive_steps=5,
        ),
        gsd=0.5,
    )
    position = np.asarray([10.0, 10.0], dtype=np.float32)
    activated = latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert activated.active
    assert not activated.armed

    released = latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert not released.active
    assert not released.armed
    assert released.reason == "active_step_limit_reached"

    blocked = latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(1.0, 0.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert not blocked.active
    assert blocked.reason == "activation_blocked_until_rearm"
    assert blocked.activation_block_reason == "feasibility_state_still_active"

    for expected_streak in range(1, 5):
        collecting = latch.update(
            position_xy=position,
            gate_active=False,
            selected_direction_rc=(0.0, 0.0),
            latched_direction_valid=True,
            hard_hazard_override=False,
        )
        assert not collecting.armed
        assert collecting.rearm_inactive_streak == expected_streak
        assert collecting.reason == "collect_rearm_inactive"
    rearmed = latch.update(
        position_xy=position,
        gate_active=False,
        selected_direction_rc=(0.0, 0.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert rearmed.armed
    assert rearmed.rearmed
    assert rearmed.rearm_inactive_streak == 5
    assert rearmed.reason == "rearm_after_sustained_feasibility_inactive"

    second = latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(1.0, 0.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert second.active
    assert second.reason == "activate_from_feasibility_gate"


def test_direct_waypoint_requires_stateful_feasibility_persistence():
    gate = StatefulFeasibilityGate(
        HysteresisConfig(
            on_persistence_steps=3,
            off_persistence_steps=1,
            minimum_dwell_steps=1,
        )
    )
    latch = DirectWaypointLatch(WaypointLatchConfig(), gsd=0.5)
    position = np.asarray([10.0, 10.0], dtype=np.float32)
    high = dict(nominal_risk=0.8, best_risk=0.2, feasible_count=1)
    low = dict(nominal_risk=0.1, best_risk=0.1, feasible_count=1)

    # One raw spike cannot launch a waypoint.
    state = gate.update(
        **high, magnitude_eligible=True, hard_hazard_override=False
    )
    update = latch.update(
        position_xy=position,
        gate_active=state.active,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert not update.active
    gate.update(**low, magnitude_eligible=True, hard_hazard_override=False)

    # Three consecutive feasibility observations satisfy persistence.
    for _ in range(3):
        state = gate.update(
            **high, magnitude_eligible=True, hard_hazard_override=False
        )
        update = latch.update(
            position_xy=position,
            gate_active=state.active,
            selected_direction_rc=(0.0, 1.0),
            latched_direction_valid=True,
            hard_hazard_override=False,
        )
    assert state.active
    assert update.active


def test_hard_override_releases_immediately_but_requires_sustained_rearm():
    latch = DirectWaypointLatch(WaypointLatchConfig(), gsd=0.5)
    position = np.asarray([10.0, 10.0], dtype=np.float32)
    latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    released = latch.update(
        position_xy=position,
        gate_active=False,
        selected_direction_rc=(0.0, 0.0),
        latched_direction_valid=True,
        hard_hazard_override=True,
    )
    assert not released.active
    assert not released.armed
    assert released.reason == "hard_hazard_override"


def test_direct_waypoint_releases_at_cumulative_forward_progress_limit():
    latch = DirectWaypointLatch(
        WaypointLatchConfig(
            distance_m=1.0,
            active_step_limit=10,
            cumulative_forward_limit_m=3.0,
        ),
        gsd=1.0,
    )
    origin = np.asarray([10.0, 10.0], dtype=np.float32)
    latch.update(
        position_xy=origin,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    for x in (11.0, 10.5, 12.0):
        update = latch.update(
            position_xy=np.asarray([x, 10.0], dtype=np.float32),
            gate_active=True,
            selected_direction_rc=(0.0, 1.0),
            latched_direction_valid=True,
            hard_hazard_override=False,
        )
    # Progress is max(1,0,1.5) = 2.5m: backward motion never subtracts.
    assert update.active
    assert update.cumulative_forward_progress_m == pytest.approx(2.5)
    released = latch.update(
        position_xy=np.asarray([12.5, 10.0], dtype=np.float32),
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert not released.active
    assert released.transitioned
    assert released.reason == "cumulative_forward_limit_reached"
    assert released.forward_progress_delta_m == pytest.approx(0.5)
    assert released.cumulative_forward_progress_m == pytest.approx(3.0)


def test_direct_waypoint_lateral_motion_adds_no_forward_progress():
    latch = DirectWaypointLatch(
        WaypointLatchConfig(
            distance_m=1.0,
            cumulative_forward_limit_m=3.0,
        ),
        gsd=0.5,
    )
    origin = np.asarray([10.0, 10.0], dtype=np.float32)
    latch.update(
        position_xy=origin,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    lateral = latch.update(
        position_xy=np.asarray([10.0, 14.0], dtype=np.float32),
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert lateral.active
    assert lateral.reason == "hold_rolling_target"
    assert lateral.forward_progress_delta_m == pytest.approx(0.0)
    assert lateral.cumulative_forward_progress_m == pytest.approx(0.0)


def test_progress_release_preserves_sustained_one_shot_rearm_semantics():
    latch = DirectWaypointLatch(
        WaypointLatchConfig(
            cumulative_forward_limit_m=1.0,
            rearm_inactive_steps=5,
        ),
        gsd=0.5,
    )
    origin = np.asarray([10.0, 10.0], dtype=np.float32)
    latch.update(
        position_xy=origin,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    released = latch.update(
        position_xy=np.asarray([12.0, 11.0], dtype=np.float32),
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert released.reason == "cumulative_forward_limit_reached"
    assert not released.armed

    blocked = latch.update(
        position_xy=origin,
        gate_active=True,
        selected_direction_rc=(1.0, 0.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert not blocked.active
    assert blocked.reason == "activation_blocked_until_rearm"

    for _ in range(5):
        rearmed = latch.update(
            position_xy=origin,
            gate_active=False,
            selected_direction_rc=(0.0, 0.0),
            latched_direction_valid=True,
            hard_hazard_override=False,
        )
    assert rearmed.rearmed
    second = latch.update(
        position_xy=origin,
        gate_active=True,
        selected_direction_rc=(1.0, 0.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert second.active


def test_temporal_rule_static_or_no_added_hard_is_unchanged():
    gate = TemporalObstacleReleaseGate(TemporalReleaseConfig())
    static = np.zeros((4, 4), dtype=bool)
    update = gate.update(
        raw_activation_requested=True,
        current_hard_mask=static,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert update.state == gate.NORMAL
    assert not update.suppress_activation
    assert not update.immediate_activation_pulse
    assert update.added_hard_count == 0


def test_temporal_rule_counts_only_release_of_snapshotted_cells():
    gate = TemporalObstacleReleaseGate(TemporalReleaseConfig())
    static = np.zeros((4, 4), dtype=bool)
    current = static.copy()
    current[1, 1] = True
    current[2, 2] = True
    waiting = gate.update(
        raw_activation_requested=True,
        current_hard_mask=current,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert waiting.state == gate.WAIT_RELEASE
    assert waiting.suppress_activation
    assert waiting.snapshotted_hard_count == 2

    # A cell that was never in the snapshot can appear and become free without
    # satisfying release.
    unrelated_previous = current.copy()
    unrelated_previous[0, 0] = True
    gate.update(
        raw_activation_requested=False,
        current_hard_mask=unrelated_previous,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    unrelated_previous[0, 0] = False
    still_waiting = gate.update(
        raw_activation_requested=False,
        current_hard_mask=unrelated_previous,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert still_waiting.state == gate.WAIT_RELEASE
    assert still_waiting.released_snapshot_cell_count == 0

    released_mask = current.copy()
    released_mask[1, 1] = False
    credit = gate.update(
        raw_activation_requested=False,
        current_hard_mask=released_mask,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert credit.state == gate.RELEASE_CREDIT
    assert credit.suppress_activation
    assert credit.released_snapshot_cell_count == 1
    assert credit.release_credit_remaining == 12


def test_temporal_release_credit_supplies_one_immediate_pulse():
    gate = TemporalObstacleReleaseGate(
        TemporalReleaseConfig(release_credit_steps=12)
    )
    static = np.zeros((3, 3), dtype=bool)
    blocked = static.copy()
    blocked[1, 1] = True
    gate.update(
        raw_activation_requested=True,
        current_hard_mask=blocked,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    gate.update(
        raw_activation_requested=False,
        current_hard_mask=static,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    pulse = gate.update(
        raw_activation_requested=True,
        current_hard_mask=static,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert pulse.immediate_activation_pulse
    assert pulse.state == gate.NORMAL
    assert pulse.release_credit_remaining == 0
    following = gate.update(
        raw_activation_requested=True,
        current_hard_mask=static,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert not following.immediate_activation_pulse


def test_temporal_release_credit_expires_after_exactly_twelve_ticks():
    gate = TemporalObstacleReleaseGate(TemporalReleaseConfig())
    static = np.zeros((3, 3), dtype=bool)
    blocked = static.copy()
    blocked[1, 1] = True
    gate.update(
        raw_activation_requested=True,
        current_hard_mask=blocked,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    credit = gate.update(
        raw_activation_requested=False,
        current_hard_mask=static,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert credit.release_credit_remaining == 12
    for expected in range(11, 0, -1):
        credit = gate.update(
            raw_activation_requested=False,
            current_hard_mask=static,
            static_hard_mask=static,
            hard_hazard_override=False,
        )
        assert credit.state == gate.RELEASE_CREDIT
        assert credit.release_credit_remaining == expected
    expired = gate.update(
        raw_activation_requested=False,
        current_hard_mask=static,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert expired.state == gate.NORMAL
    assert expired.release_credit_remaining == 0
    assert expired.reason == "release_credit_expired"


def test_temporal_wait_timeout_enters_bypass_until_raw_request_clears():
    gate = TemporalObstacleReleaseGate(
        TemporalReleaseConfig(wait_timeout_steps=12)
    )
    static = np.zeros((3, 3), dtype=bool)
    blocked = static.copy()
    blocked[1, 1] = True
    gate.update(
        raw_activation_requested=True,
        current_hard_mask=blocked,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    for _ in range(11):
        update = gate.update(
            raw_activation_requested=True,
            current_hard_mask=blocked,
            static_hard_mask=static,
            hard_hazard_override=False,
        )
        assert update.state == gate.WAIT_RELEASE
        assert update.suppress_activation
    bypass = gate.update(
        raw_activation_requested=True,
        current_hard_mask=blocked,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert bypass.state == gate.BYPASS_UNTIL_CLEAR
    assert not bypass.suppress_activation
    cleared = gate.update(
        raw_activation_requested=False,
        current_hard_mask=blocked,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert cleared.state == gate.NORMAL


def test_temporal_pulse_bypasses_persistence_but_hard_override_dominates():
    temporal = TemporalObstacleReleaseGate(TemporalReleaseConfig())
    feasibility = StatefulFeasibilityGate(
        HysteresisConfig(on_persistence_steps=3)
    )
    latch = DirectWaypointLatch(WaypointLatchConfig(), gsd=0.5)
    static = np.zeros((3, 3), dtype=bool)
    blocked = static.copy()
    blocked[1, 1] = True
    high = dict(nominal_risk=0.8, best_risk=0.2, feasible_count=1)
    position = np.asarray([5.0, 5.0], dtype=np.float32)

    state = feasibility.update(
        **high, magnitude_eligible=True, hard_hazard_override=False
    )
    waiting = temporal.update(
        raw_activation_requested=True,
        current_hard_mask=blocked,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert not state.active
    assert waiting.suppress_activation

    # Release plus one raw request produces a pulse even though persistence
    # has not accumulated three observations.
    pulse = temporal.update(
        raw_activation_requested=True,
        current_hard_mask=static,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    activated = latch.update(
        position_xy=position,
        gate_active=state.active or pulse.immediate_activation_pulse,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert pulse.immediate_activation_pulse
    assert activated.active

    hard = latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=True,
    )
    assert not hard.active
    assert hard.reason == "hard_hazard_override"


def test_release_credit_suppresses_lingering_stateful_activation():
    temporal = TemporalObstacleReleaseGate(TemporalReleaseConfig())
    feasibility = StatefulFeasibilityGate(
        HysteresisConfig(
            on_persistence_steps=1,
            off_persistence_steps=2,
            minimum_dwell_steps=5,
        )
    )
    latch = DirectWaypointLatch(WaypointLatchConfig(), gsd=0.5)
    static = np.zeros((3, 3), dtype=bool)
    blocked = static.copy()
    blocked[1, 1] = True
    position = np.asarray([5.0, 5.0], dtype=np.float32)
    high = dict(nominal_risk=0.8, best_risk=0.2, feasible_count=1)
    low = dict(nominal_risk=0.1, best_risk=0.1, feasible_count=1)

    active_state = feasibility.update(
        **high, magnitude_eligible=True, hard_hazard_override=False
    )
    waiting = temporal.update(
        raw_activation_requested=True,
        current_hard_mask=blocked,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert active_state.active
    assert waiting.suppress_activation

    lingering_state = feasibility.update(
        **low, magnitude_eligible=True, hard_hazard_override=False
    )
    credit = temporal.update(
        raw_activation_requested=False,
        current_hard_mask=static,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    assert lingering_state.active
    assert credit.state == temporal.RELEASE_CREDIT
    assert credit.suppress_activation
    assert not credit.immediate_activation_pulse
    admitted = direct_activation_admitted(
        stateful_feasibility_active=lingering_state.active,
        temporal_update=credit,
        hard_hazard_override=False,
    )
    update = latch.update(
        position_xy=position,
        gate_active=admitted,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert not update.active


def test_temporal_hard_override_never_generates_immediate_pulse():
    gate = TemporalObstacleReleaseGate(TemporalReleaseConfig())
    static = np.zeros((3, 3), dtype=bool)
    blocked = static.copy()
    blocked[1, 1] = True
    gate.update(
        raw_activation_requested=True,
        current_hard_mask=blocked,
        static_hard_mask=static,
        hard_hazard_override=False,
    )
    no_pulse = gate.update(
        raw_activation_requested=True,
        current_hard_mask=static,
        static_hard_mask=static,
        hard_hazard_override=True,
    )
    assert no_pulse.state == gate.RELEASE_CREDIT
    assert not no_pulse.immediate_activation_pulse
    assert no_pulse.release_credit_remaining == 12
    assert not direct_activation_admitted(
        stateful_feasibility_active=True,
        temporal_update=no_pulse,
        hard_hazard_override=True,
    )


def test_inactive_armed_hard_observation_preserves_unused_arm():
    latch = DirectWaypointLatch(WaypointLatchConfig(), gsd=0.5)
    position = np.asarray([5.0, 5.0], dtype=np.float32)
    held = latch.update(
        position_xy=position,
        gate_active=False,
        selected_direction_rc=(0.0, 0.0),
        latched_direction_valid=True,
        hard_hazard_override=True,
    )
    assert not held.active
    assert held.armed
    assert held.rearm_inactive_streak == 0
    assert held.reason == "inactive_hard_hazard_hold"

    # A later temporal pulse is represented by the admitted gate signal.
    activated = latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert activated.active
    assert activated.reason == "activate_from_feasibility_gate"


def test_inactive_disarmed_hard_observation_preserves_rearm_progress():
    latch = DirectWaypointLatch(
        WaypointLatchConfig(active_step_limit=1, rearm_inactive_steps=5),
        gsd=0.5,
    )
    position = np.asarray([5.0, 5.0], dtype=np.float32)
    latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    released = latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    assert not released.active
    assert not released.armed
    for expected in (1, 2):
        collecting = latch.update(
            position_xy=position,
            gate_active=False,
            selected_direction_rc=(0.0, 0.0),
            latched_direction_valid=True,
            hard_hazard_override=False,
        )
        assert collecting.rearm_inactive_streak == expected

    held = latch.update(
        position_xy=position,
        gate_active=False,
        selected_direction_rc=(0.0, 0.0),
        latched_direction_valid=True,
        hard_hazard_override=True,
    )
    assert not held.armed
    assert held.rearm_inactive_streak == 2
    assert held.reason == "inactive_hard_hazard_hold"


def test_active_hard_override_still_cancels_and_disarms():
    latch = DirectWaypointLatch(WaypointLatchConfig(), gsd=0.5)
    position = np.asarray([5.0, 5.0], dtype=np.float32)
    latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=False,
    )
    cancelled = latch.update(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        latched_direction_valid=True,
        hard_hazard_override=True,
    )
    assert not cancelled.active
    assert not cancelled.armed
    assert cancelled.rearm_inactive_streak == 0
    assert cancelled.reason == "hard_hazard_override"


def test_receding_safe_replans_are_continuous_but_no_safe_gap_deactivates():
    controller = DirectWaypointLatch(WaypointLatchConfig(), gsd=0.5)
    position = np.asarray([5.0, 5.0], dtype=np.float32)
    first = controller.update_receding(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        selection_attempted=True,
        hard_hazard_override=False,
    )
    assert first.active
    assert first.transitioned
    assert first.reason == "activate_safe_direction"
    np.testing.assert_allclose(first.waypoint_xy, [7.0, 5.0])

    second = controller.update_receding(
        position_xy=position + [0.1, 0.0],
        gate_active=True,
        selected_direction_rc=(1.0, 0.0),
        selection_attempted=True,
        hard_hazard_override=False,
    )
    assert second.active
    assert not second.transitioned
    assert second.replan_due
    assert second.reason == "replan_safe_direction"
    np.testing.assert_allclose(second.waypoint_xy, [5.1, 7.0])

    gap = controller.update_receding(
        position_xy=position + [0.1, 0.0],
        gate_active=False,
        selected_direction_rc=(0.0, 0.0),
        selection_attempted=True,
        hard_hazard_override=False,
        latched_direction_valid=False,
    )
    assert not gap.active
    assert gap.transitioned
    assert gap.reason == "no_safe_improving_candidate"


def test_receding_controller_holds_last_safe_direction_for_bounded_gap():
    controller = DirectWaypointLatch(
        WaypointLatchConfig(active_step_limit=3),
        gsd=0.5,
    )
    position = np.asarray([5.0, 5.0], dtype=np.float32)
    controller.update_receding(
        position_xy=position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        selection_attempted=True,
        hard_hazard_override=False,
    )
    held = controller.update_receding(
        position_xy=position + [0.1, 0.0],
        gate_active=False,
        selected_direction_rc=(0.0, 0.0),
        selection_attempted=True,
        hard_hazard_override=False,
        latched_direction_valid=True,
    )
    assert held.active
    assert not held.transitioned
    assert held.reason == "hold_last_safe_direction"
    assert held.active_steps == 2


def test_receding_controller_replans_from_each_actual_position():
    controller = DirectWaypointLatch(WaypointLatchConfig(), gsd=0.5)
    first_position = np.asarray([5.0, 5.0], dtype=np.float32)
    first = controller.update_receding(
        position_xy=first_position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        selection_attempted=True,
        hard_hazard_override=False,
    )
    actual_next_position = np.asarray([5.4, 5.2], dtype=np.float32)
    second = controller.update_receding(
        position_xy=actual_next_position,
        gate_active=True,
        selected_direction_rc=(0.0, 1.0),
        selection_attempted=True,
        hard_hazard_override=False,
    )
    np.testing.assert_allclose(first.waypoint_xy, [7.0, 5.0])
    np.testing.assert_allclose(
        second.waypoint_xy, actual_next_position + [2.0, 0.0]
    )
    assert second.active_steps == 2
