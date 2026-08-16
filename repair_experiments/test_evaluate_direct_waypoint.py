from __future__ import annotations

import copy

import pytest

from repair_experiments.evaluate_direct_waypoint import (
    DIRECT_WAYPOINT_MODES,
    _install_direct_waypoint_study,
    add_direct_waypoint_separation_fields,
    compute_direct_waypoint_metrics,
    main,
)
from repair_experiments import evaluate_v1 as base_evaluator


def _step(mode: str, *, x: float, active: int) -> dict[str, object]:
    return {
        "dynamic_episode_uid": "event:0",
        "episode_uid": "episode:0",
        "scene_id": "scene:0",
        "regime": "R1",
        "event_type": "mud_onset",
        "mode": mode,
        "step": 0,
        "position_x": 0.0,
        "position_y": 0.0,
        "next_x": x,
        "next_y": 0.0,
        "effective_gate_active": active,
        "mechanism_window_complete": 1,
        "horizon_endpoint_alignment": 1.0,
        "horizon_clearance_retained": 1,
        "horizon_predicted_risk_realized": 1,
    }


def test_separation_uses_direct_waypoint_as_primary() -> None:
    rows = [
        _step("direct_waypoint", x=3.0, active=1),
        _step("stateless_projected", x=7.0, active=0),
        _step("gate_off", x=1.0, active=0),
        _step("geometry_only", x=9.0, active=0),
    ]
    original = copy.deepcopy(rows)
    result = add_direct_waypoint_separation_fields(
        rows, horizon_steps=1, gsd=0.5
    )
    assert rows == original
    primary = next(row for row in result if row["mode"] == "direct_waypoint")
    previous = next(
        row for row in result if row["mode"] == "stateless_projected"
    )
    assert primary["paired_gate_state_differs"] == 1
    assert primary["paired_endpoint_separation_m"] == pytest.approx(1.0)
    assert previous["paired_gate_state_differs"] == 0


def test_metrics_label_primary_and_keep_test_sealed() -> None:
    episodes = []
    traces = []
    for mode in DIRECT_WAYPOINT_MODES:
        episodes.append(
            {
                "dynamic_episode_uid": "event:0",
                "episode_uid": "episode:0",
                "scene_id": "scene:0",
                "regime": "R1",
                "event_type": "mud_onset",
                "mode": mode,
                "hard_contact_episode": 0,
                "cvar20_violation": 0.0,
                "static_activation": int(mode == "direct_waypoint"),
                "activation_transitions": 1,
                "post_open_reaction_delay_steps": 0,
            }
        )
        traces.append(_step(mode, x=1.0, active=int(mode == "direct_waypoint")))
    metrics = compute_direct_waypoint_metrics(
        episodes, traces, n_boot=0
    )
    assert metrics["mode_semantics"]["primary"] == "direct_waypoint"
    assert (
        metrics["mode_semantics"]["transition_diagnostic_comparator"]
        == "stateless_projected"
    )
    assert metrics["decision"]["exploratory_development_validation"] is True
    assert metrics["decision"]["go_for_one_shot_test"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--allow-sealed-test"],
        ["--frozen-config", "anything.json"],
        ["--split", "test"],
        ["--split=test"],
    ],
)
def test_cli_rejects_test_authorization_arguments(arguments: list[str]) -> None:
    with pytest.raises(SystemExit, match="validation-only"):
        main(arguments)


def test_cli_exposes_matched_54_item_waypoint_configuration() -> None:
    _install_direct_waypoint_study()
    args = base_evaluator.parse_args(["--max-dynamic-items", "54"])
    assert args.split == "validation"
    assert args.max_dynamic_items == 54
    assert tuple(args.modes) == DIRECT_WAYPOINT_MODES
    assert args.waypoint_distance_m == pytest.approx(1.0)
    assert args.waypoint_active_step_limit == 10
    assert args.waypoint_cumulative_forward_limit_m == pytest.approx(3.0)
    assert args.waypoint_rearm_inactive_steps == 5
    assert args.selector_prediction_steps == 6
    assert args.selector_progress_min_m == pytest.approx(0.1)
    assert args.selector_swept_sample_spacing_m == pytest.approx(0.25)
    assert args.selector_goal_direction_cosine_min == pytest.approx(0.25)
    assert args.selector_velocity_direction_cosine_min == pytest.approx(0.0)
    assert args.temporal_wait_timeout_steps == 12
    assert args.temporal_release_credit_steps == 12
    # Legacy fixed-waypoint flags remain parse-compatible but do not drive
    # rolling-target behavior.
    assert args.waypoint_replan_interval_steps == 10
    assert args.waypoint_reach_tolerance_m == pytest.approx(0.25)
    assert args.waypoint_minimum_hold_steps == 50
    assert args.waypoint_maximum_hold_steps == 100
    assert args.direct_lambda_floor == pytest.approx(1.5)
