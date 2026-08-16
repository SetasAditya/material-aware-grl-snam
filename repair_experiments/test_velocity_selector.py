import numpy as np
import pytest

from repair_experiments.v1_controller import (
    StagewisePrimitiveCandidate,
    StagewisePrimitiveEnumeration,
    VelocityAwareCandidateDiagnostics,
    VelocityAwareSelectorConfig,
    primitive_direction_xy,
    receding_candidate_pool,
    velocity_tracking_direction_rc,
)
from repair_experiments.velocity_selector import (
    preserved_numpy_rng,
    rank_velocity_candidates,
    require_nominal_forecast_improvement,
    simulate_velocity_candidate,
)


def _candidate(index, direction_rc):
    return StagewisePrimitiveCandidate(
        primitive_index=index,
        direction_rc=direction_rc,
        endpoint_rc=(0.0, 0.0),
        mean_risk=0.1,
        min_clearance_m=5.0,
    )


def _record(
    index,
    *,
    risk=0.2,
    progress=1.0,
    clearance=3.0,
    cosine=0.0,
    accepted=True,
    rejection="",
):
    return VelocityAwareCandidateDiagnostics(
        primitive_index=index,
        direction_rc=(0.0, 1.0),
        predicted_endpoint_xy=(1.0, 1.0),
        predicted_terminal_velocity_xy=(0.0, 0.0),
        accepted=accepted,
        rejection_reason=rejection,
        simulated_stages=10,
        swept_sample_count=10,
        path_weighted_mean_risk=risk,
        stage_goal_progress_m=progress,
        minimum_sdf_clearance_m=clearance,
        initial_velocity_cosine=cosine,
    )


def test_lexicographic_ranking_uses_declared_order():
    records = [
        _record(5, risk=0.3, progress=9.0, clearance=9.0, cosine=1.0),
        _record(4, risk=0.2, progress=1.0, clearance=3.0, cosine=0.0),
        _record(3, risk=0.2, progress=2.0, clearance=2.0, cosine=0.0),
        _record(2, risk=0.2, progress=2.0, clearance=4.0, cosine=-1.0),
        _record(1, risk=0.2, progress=2.0, clearance=4.0, cosine=0.5),
        _record(0, risk=0.2, progress=2.0, clearance=4.0, cosine=0.5),
    ]
    result = rank_velocity_candidates(records)
    assert result.selected_primitive_index == 5
    assert result.accepted_count == 6


def test_initial_velocity_changes_selection_when_prior_terms_tie():
    maps = {
        "risk_map": np.full((31, 31), 0.2, dtype=np.float32),
        "hard_mask": np.zeros((31, 31), dtype=bool),
        "sdf_hard": np.full((31, 31), 5.0, dtype=np.float32),
    }
    candidates = [_candidate(0, (0.0, 1.0)), _candidate(1, (1.0, 0.0))]
    config = VelocityAwareSelectorConfig(
        prediction_steps=2,
        progress_min_m=0.0,
    )

    def stage_step(position, velocity, goal, direction_rc, stage):
        del velocity, goal, stage
        direction_xy = primitive_direction_xy(direction_rc)
        return position + 0.2 * direction_xy, direction_xy

    selections = []
    for initial_velocity in (
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    ):
        records = [
            simulate_velocity_candidate(
                candidate=candidate,
                initial_position_xy=np.asarray([10.0, 10.0]),
                initial_velocity_xy=initial_velocity,
                stage_goal_xy=np.asarray([20.0, 20.0]),
                maps=maps,
                gsd=1.0,
                rolling_distance_m=1.0,
                config=config,
                stage_step=stage_step,
            )
            for candidate in candidates
        ]
        selections.append(
            rank_velocity_candidates(records).selected_primitive_index
        )
    assert selections == [0, 1]


def test_swept_sampling_detects_collision_between_segment_endpoints():
    maps = {
        "risk_map": np.zeros((12, 12), dtype=np.float32),
        "hard_mask": np.zeros((12, 12), dtype=bool),
        "sdf_hard": np.full((12, 12), 5.0, dtype=np.float32),
    }
    maps["hard_mask"][5, 5] = True
    maps["sdf_hard"][5, 5] = 0.0
    config = VelocityAwareSelectorConfig(
        prediction_steps=1,
        progress_min_m=0.0,
        swept_sample_spacing_m=0.25,
    )

    def jump(position, velocity, goal, direction_rc, stage):
        del velocity, goal, direction_rc, stage
        return position + np.asarray([4.0, 0.0]), np.asarray([4.0, 0.0])

    record = simulate_velocity_candidate(
        candidate=_candidate(0, (0.0, 1.0)),
        initial_position_xy=np.asarray([3.0, 5.0]),
        initial_velocity_xy=np.zeros(2),
        stage_goal_xy=np.asarray([10.0, 5.0]),
        maps=maps,
        gsd=1.0,
        rolling_distance_m=1.0,
        config=config,
        stage_step=jump,
    )
    assert not record.accepted
    assert record.rejection_reason == "hard_collision"
    assert record.swept_sample_count > 1


def test_preserved_numpy_rng_restores_exact_state():
    np.random.seed(1234)
    expected = np.random.random(8)
    np.random.seed(1234)
    with preserved_numpy_rng(999):
        np.random.random(100)
    actual = np.random.random(8)
    np.testing.assert_array_equal(actual, expected)


def test_no_safe_candidate_returns_explicit_fallback():
    records = [
        _record(0, accepted=False, rejection="hard_collision"),
        _record(1, accepted=False, rejection="clearance"),
        _record(2, accepted=False, rejection="hard_collision"),
    ]
    result = rank_velocity_candidates(records)
    assert result.selected_primitive_index is None
    assert result.selected_direction_rc == (0.0, 0.0)
    assert result.accepted_count == 0
    assert dict(result.rejection_counts) == {
        "clearance": 1,
        "hard_collision": 2,
    }


def test_rejection_priority_places_nonfinite_before_other_checks():
    maps = {
        "risk_map": np.zeros((5, 5), dtype=np.float32),
        "hard_mask": np.ones((5, 5), dtype=bool),
        "sdf_hard": np.zeros((5, 5), dtype=np.float32),
    }
    maps["hard_mask"][2, 2] = False
    maps["sdf_hard"][2, 2] = 5.0

    def invalid(*args):
        del args
        return np.asarray([np.nan, 100.0]), np.asarray([np.nan, 0.0])

    record = simulate_velocity_candidate(
        candidate=_candidate(0, (0.0, 1.0)),
        initial_position_xy=np.asarray([2.0, 2.0]),
        initial_velocity_xy=np.zeros(2),
        stage_goal_xy=np.asarray([4.0, 2.0]),
        maps=maps,
        gsd=1.0,
        rolling_distance_m=1.0,
        config=VelocityAwareSelectorConfig(prediction_steps=1),
        stage_step=invalid,
    )
    assert record.rejection_reason == "invalid_nonfinite"


def test_v7_default_prediction_horizon_is_six():
    assert VelocityAwareSelectorConfig().prediction_steps == 6


def test_forward_direction_guard_rejects_reverse_candidate_before_simulation():
    maps = {
        "risk_map": np.zeros((10, 10), dtype=np.float32),
        "hard_mask": np.zeros((10, 10), dtype=bool),
        "sdf_hard": np.full((10, 10), 5.0, dtype=np.float32),
    }

    def must_not_run(*args):
        raise AssertionError(f"stage callback unexpectedly called: {args}")

    record = simulate_velocity_candidate(
        candidate=_candidate(8, (0.0, -1.0)),
        initial_position_xy=np.asarray([5.0, 5.0]),
        initial_velocity_xy=np.asarray([1.0, 0.0]),
        stage_goal_xy=np.asarray([9.0, 5.0]),
        maps=maps,
        gsd=1.0,
        rolling_distance_m=1.0,
        config=VelocityAwareSelectorConfig(prediction_steps=1),
        stage_step=must_not_run,
    )
    assert not record.accepted
    assert record.rejection_reason == "nonforward_goal_direction"


def test_velocity_tracking_direction_rotates_toward_selected_primitive():
    # Current velocity points +x and the primitive points +y.  The velocity
    # error therefore points northwest: (-x,+y), returned in (row,col).
    direction_rc = velocity_tracking_direction_rc(
        np.asarray([2.0, 0.0]),
        (1.0, 0.0),
    )
    np.testing.assert_allclose(
        direction_rc,
        [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
        atol=1e-7,
    )


def test_current_state_clearance_rejected_before_stage_integration():
    maps = {
        "risk_map": np.zeros((5, 5), dtype=np.float32),
        "hard_mask": np.zeros((5, 5), dtype=bool),
        "sdf_hard": np.full((5, 5), 5.0, dtype=np.float32),
    }
    maps["sdf_hard"][2, 2] = 0.5

    def must_not_run(*args):
        raise AssertionError(f"stage callback unexpectedly called: {args}")

    record = simulate_velocity_candidate(
        candidate=_candidate(0, (0.0, 1.0)),
        initial_position_xy=np.asarray([2.0, 2.0]),
        initial_velocity_xy=np.zeros(2),
        stage_goal_xy=np.asarray([4.0, 2.0]),
        maps=maps,
        gsd=1.0,
        rolling_distance_m=1.0,
        config=VelocityAwareSelectorConfig(prediction_steps=12),
        stage_step=must_not_run,
    )
    assert not record.accepted
    assert record.rejection_reason == "clearance"
    assert record.simulated_stages == 0


def test_nominal_and_candidate_use_equivalent_closed_loop_forecast():
    maps = {
        "risk_map": np.full((20, 20), 0.3, dtype=np.float32),
        "hard_mask": np.zeros((20, 20), dtype=bool),
        "sdf_hard": np.full((20, 20), 5.0, dtype=np.float32),
    }
    config = VelocityAwareSelectorConfig(
        prediction_steps=12,
        progress_min_m=0.0,
    )

    def step(position, velocity, goal, direction_rc, stage):
        del velocity, goal, stage
        direction = primitive_direction_xy(direction_rc)
        return position + 0.1 * direction, direction

    kwargs = dict(
        candidate=_candidate(0, (0.0, 1.0)),
        initial_position_xy=np.asarray([5.0, 5.0]),
        initial_velocity_xy=np.asarray([0.2, 0.0]),
        stage_goal_xy=np.asarray([15.0, 5.0]),
        maps=maps,
        gsd=1.0,
        rolling_distance_m=1.0,
        config=config,
        stage_step=step,
    )
    candidate = simulate_velocity_candidate(**kwargs, enforce_safety=True)
    nominal = simulate_velocity_candidate(**kwargs, enforce_safety=False)
    assert candidate.accepted and nominal.accepted
    assert candidate.simulated_stages == nominal.simulated_stages == 12
    assert candidate.predicted_endpoint_xy == nominal.predicted_endpoint_xy
    assert candidate.path_weighted_mean_risk == pytest.approx(
        nominal.path_weighted_mean_risk
    )


def test_nominal_forecast_does_not_truncate_at_hard_collision():
    maps = {
        "risk_map": np.full((20, 20), 0.4, dtype=np.float32),
        "hard_mask": np.zeros((20, 20), dtype=bool),
        "sdf_hard": np.full((20, 20), 5.0, dtype=np.float32),
    }
    maps["hard_mask"][5, 7] = True
    maps["sdf_hard"][5, 7] = 0.0

    def step(position, velocity, goal, direction_rc, stage):
        del velocity, goal, direction_rc, stage
        return position + np.asarray([0.5, 0.0]), np.asarray([0.5, 0.0])

    common = dict(
        candidate=_candidate(0, (0.0, 1.0)),
        initial_position_xy=np.asarray([5.0, 5.0]),
        initial_velocity_xy=np.zeros(2),
        stage_goal_xy=np.asarray([15.0, 5.0]),
        maps=maps,
        gsd=1.0,
        rolling_distance_m=1.0,
        config=VelocityAwareSelectorConfig(
            prediction_steps=12, progress_min_m=0.0
        ),
        stage_step=step,
    )
    candidate = simulate_velocity_candidate(**common, enforce_safety=True)
    nominal = simulate_velocity_candidate(**common, enforce_safety=False)
    assert candidate.rejection_reason == "hard_collision"
    assert nominal.simulated_stages == 12
    assert np.isfinite(nominal.path_weighted_mean_risk)


def test_dynamic_improvement_threshold_is_inclusive_at_point_zero_five():
    records = [
        _record(0, risk=0.45),
        _record(1, risk=0.450001),
    ]
    filtered = require_nominal_forecast_improvement(
        records,
        nominal_path_weighted_mean_risk=0.5,
        improvement_margin=0.05,
    )
    assert filtered[0].accepted
    assert not filtered[1].accepted
    assert (
        filtered[1].rejection_reason
        == "insufficient_dynamic_risk_improvement"
    )


def test_receding_pool_considers_all_geometric_rays_without_static_filter():
    candidates = (
        _candidate(0, (0.0, 1.0)),
        _candidate(1, (1.0, 0.0)),
    )
    enumeration = StagewisePrimitiveEnumeration(
        nominal_direction_rc=(0.0, 1.0),
        nominal_endpoint_rc=(2.0, 2.0),
        nominal_mean_risk=0.2,
        # Candidate 1 is statically worse than nominal; it must still be
        # dynamically forecast in v6.
        candidates=(
            candidates[0],
            StagewisePrimitiveCandidate(
                primitive_index=1,
                direction_rc=(1.0, 0.0),
                endpoint_rc=(2.0, 2.0),
                mean_risk=0.9,
                min_clearance_m=5.0,
            ),
        ),
    )
    assert receding_candidate_pool(enumeration) == enumeration.candidates
    assert len(receding_candidate_pool(enumeration)) == 2
