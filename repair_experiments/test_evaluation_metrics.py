from __future__ import annotations

import math
from pathlib import Path

import pytest

from repair_experiments.evaluate_v1 import (
    ROLLOUT_FREEZE_KEYS,
    SealedTestError,
    _sha256,
    _validate_frozen_config,
    behavior_effect_horizon_steps,
    build_frozen_configuration,
    resolve_split_paths,
    stratified_dynamic_limit,
)
from repair_experiments.evaluation_metrics import (
    add_horizon_mechanism_fields,
    add_paired_separation_fields,
    build_episode_fields,
    compute_preregistered_metrics,
)


def _step(
    *,
    mode: str = "repaired",
    step: int,
    x: float,
    active: int = 1,
) -> dict:
    return {
        "dynamic_episode_uid": "episode:event:mud_onset",
        "episode_uid": "episode",
        "scene_id": "scene",
        "sequence": "00003",
        "regime": "R1",
        "event_type": "mud_onset",
        "mode": mode,
        "step": step,
        "event_step": 4,
        "opening_step": 14,
        "position_x": x,
        "position_y": 0.0,
        "next_x": x + 1.0,
        "next_y": 0.0,
        "effective_gate_active": active,
        "current_sdf_m": 2.0,
        "current_hard": 0,
        "hard_contact": 0,
        "risk": 0.1,
        "nominal_primitive_risk": 0.8,
        "best_primitive_risk": 0.2,
        "selected_direction_row": 0.0,
        "selected_direction_col": 1.0,
    }


def test_sealed_test_is_rejected_before_files_are_opened(tmp_path: Path) -> None:
    with pytest.raises(SealedTestError):
        resolve_split_paths(
            tmp_path,
            "test",
            allow_sealed_test=False,
            frozen_config=None,
        )
    with pytest.raises(SealedTestError):
        resolve_split_paths(
            tmp_path,
            "test",
            allow_sealed_test=True,
            frozen_config=None,
        )


def test_validation_is_default_safe_split(tmp_path: Path) -> None:
    static, dynamic, lock = resolve_split_paths(
        tmp_path,
        "validation",
        allow_sealed_test=False,
        frozen_config=None,
    )
    assert static.name == "validation_static.json"
    assert dynamic.name == "validation_dynamic.json"
    assert lock.name == "SPLIT_LOCK.json"


def test_single_frozen_artifact_completes_test_handshake(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixed checkpoint")
    arguments = {key: 1 for key in ROLLOUT_FREEZE_KEYS}
    config = {
        "controller_version": "v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "split_manifest_hashes": {"validation_static.json": "hash"},
        "bev_manifest_sha256": "bev-hash",
        "modes": [
            "repaired",
            "stateless_projected",
            "gate_off",
            "geometry_only",
        ],
    }
    frozen = build_frozen_configuration(
        config=config,
        validation_decision={"go_for_one_shot_test": True},
        rollout_arguments=arguments,
    )
    path = tmp_path / "frozen_configuration.json"
    path.write_text(__import__("json").dumps(frozen))
    loaded = _validate_frozen_config(path, checkpoint=checkpoint)
    assert loaded is not None
    assert loaded["arguments"] == arguments
    assert loaded["validation_decision"]["go_for_one_shot_test"] is True


def test_dynamic_smoke_limit_round_robins_event_and_regime() -> None:
    items = [
        {
            "dynamic_episode_uid": f"{event}:{regime}:{index}",
            "event_type": event,
            "regime": regime,
        }
        for event in ("event_b", "event_a")
        for regime in ("R2", "R1")
        for index in range(4)
    ]
    selected = stratified_dynamic_limit(items, 8)
    counts: dict[tuple[str, str], int] = {}
    for item in selected:
        key = (item["event_type"], item["regime"])
        counts[key] = counts.get(key, 0) + 1
    assert selected == stratified_dynamic_limit(items, 8)
    assert set(counts.values()) == {2}
    assert stratified_dynamic_limit(items, 0) == items


def test_behavior_effect_horizon_is_physical_time_not_primitive_window() -> None:
    assert behavior_effect_horizon_steps(1.0, 0.01) == 100
    assert behavior_effect_horizon_steps(0.75, 0.05) == 15
    with pytest.raises(ValueError):
        behavior_effect_horizon_steps(0.0, 0.01)
    with pytest.raises(ValueError):
        behavior_effect_horizon_steps(1.0, 0.0)


def test_horizon_metrics_use_executed_trajectory() -> None:
    rows = [_step(step=step, x=float(step)) for step in range(3)]
    decorated = add_horizon_mechanism_fields(
        rows, horizon_steps=3, hard_margin_m=1.0
    )
    first = decorated[0]
    assert first["mechanism_window_complete"] == 1
    assert first["horizon_endpoint_alignment"] == pytest.approx(1.0)
    assert first["horizon_clearance_retained"] == 1
    assert first["horizon_actual_risk_improvement"] == pytest.approx(0.7)
    assert first["horizon_predicted_risk_realized"] == 1
    assert decorated[-1]["mechanism_window_complete"] == 0
    assert math.isnan(decorated[-1]["horizon_endpoint_alignment"])


def test_paired_endpoint_separation_uses_same_horizon_and_gate_state() -> None:
    repaired = [
        _step(mode="repaired", step=step, x=0.0, active=0)
        for step in range(3)
    ]
    gate_off = [
        _step(mode="gate_off", step=step, x=float(step), active=1)
        for step in range(3)
    ]
    rows = add_paired_separation_fields(
        repaired + gate_off, horizon_steps=3, gsd=0.5
    )
    first = next(
        row for row in rows if row["mode"] == "repaired" and row["step"] == 0
    )
    assert first["paired_gate_state_differs"] == 1
    assert first["paired_endpoint_separation_m"] == pytest.approx(1.0)


def test_behavior_horizon_uses_absorbing_successful_terminal_state() -> None:
    repaired = [
        {**_step(mode="repaired", step=step, x=0.0, active=0), "rollout_success": 1}
        for step in range(3)
    ]
    gate_off = [
        {
            **_step(mode="gate_off", step=step, x=float(step), active=1),
            "rollout_success": 1,
        }
        for step in range(3)
    ]
    rows = add_paired_separation_fields(
        repaired + gate_off, horizon_steps=100, gsd=0.5
    )
    first = next(
        row for row in rows if row["mode"] == "repaired" and row["step"] == 0
    )
    assert first["paired_behavior_horizon_complete"] == 1
    assert first["paired_endpoint_used_absorbing_terminal"] == 1
    assert first["paired_incomplete_failure_imputed"] == 0
    assert first["paired_endpoint_separation_m"] == pytest.approx(1.0)


def test_incomplete_failed_behavior_window_is_counted_as_zero_separation() -> None:
    repaired = [
        {**_step(mode="repaired", step=step, x=0.0, active=0), "rollout_success": 0}
        for step in range(3)
    ]
    gate_off = [
        {
            **_step(mode="gate_off", step=step, x=float(step), active=1),
            "rollout_success": 0,
        }
        for step in range(3)
    ]
    rows = add_paired_separation_fields(
        repaired + gate_off, horizon_steps=100, gsd=0.5
    )
    first = next(
        row for row in rows if row["mode"] == "repaired" and row["step"] == 0
    )
    assert first["paired_behavior_horizon_complete"] == 0
    assert first["paired_incomplete_failure_imputed"] == 1
    assert first["paired_endpoint_separation_m"] == 0.0


def test_episode_behavior_keeps_nonreaction_in_denominator() -> None:
    trace = [_step(step=step, x=float(step), active=0) for step in range(5)]
    for row in trace:
        row["event_type"] = "delayed_required_escape"
        row["opening_step"] = 3
        row["event_step"] = 1
        row["mechanism_window_complete"] = 0
    episode = {
        "dynamic_episode_uid": "episode:event:mud_onset",
        "episode_uid": "episode",
        "scene_id": "scene",
        "sequence": "00003",
        "regime": "R1",
        "event_type": "delayed_required_escape",
        "mode": "repaired",
        "steps": 5,
        "success": 0,
        "hard_contacts": 0,
        "cvar20_violation": 0.1,
        "activation_rate": 0.0,
        "activation_transitions": 0,
        "event_step": 1,
        "opening_step": 3,
    }
    result = build_episode_fields([episode], trace, max_steps=10)[0]
    assert result["post_open_success"] == 0
    assert result["post_open_reaction_delay_steps"] == 8
    assert result["false_pre_activation"] == 0


def test_subset_cannot_accidentally_pass_preregistered_decision() -> None:
    result = compute_preregistered_metrics([], [], n_boot=5)
    assert result["decision"]["go_for_one_shot_test"] is False
    assert result["decision"]["behavior_pass_count"] == 0
