from pathlib import Path

import pytest
import torch

from repair_experiments.train_behavioral_soft_force import (
    RepairConfig,
    _forward_material_outputs,
    _load_manifest,
    _stratified_limit,
    behavioral_loss,
    calibrate_lambda_threshold,
    checkpoint_selection_key,
    load_initial_model,
)


def _episode(sequence: str, regime: str, suffix: int):
    return {
        "sequence": sequence,
        "regime": regime,
        "episode_uid": f"{sequence}:{regime}:{suffix}",
    }


def test_sealed_manifest_is_rejected_before_read(tmp_path: Path):
    sealed = tmp_path / "test_static.json"
    sealed.write_text("this is deliberately not JSON")
    with pytest.raises(ValueError, match="sealed test"):
        _load_manifest(
            sealed,
            expected_split="test",
            allowed_sequences=frozenset({"00004"}),
        )


def test_stratified_limit_round_robins_sequence_and_regime():
    episodes = [
        _episode(sequence, regime, suffix)
        for suffix in range(3)
        for sequence in ("00000", "00001")
        for regime in ("R1", "R2", "R3")
    ]
    selected = _stratified_limit(episodes, 6)
    assert len(selected) == 6
    assert {(row["sequence"], row["regime"]) for row in selected} == {
        (sequence, regime)
        for sequence in ("00000", "00001")
        for regime in ("R1", "R2", "R3")
    }


def test_behavioral_loss_prefers_high_active_and_low_inactive_lambda():
    cfg = RepairConfig()
    active = torch.tensor([1.0])
    inactive = torch.tensor([0.0])
    grad = torch.tensor([0.2])
    active_low, _, _ = behavioral_loss(
        torch.tensor([0.1]), active, grad, cfg
    )
    active_high, _, _ = behavioral_loss(
        torch.tensor([2.0]), active, grad, cfg
    )
    inactive_low, _, _ = behavioral_loss(
        torch.tensor([0.1]), inactive, grad, cfg
    )
    inactive_high, _, _ = behavioral_loss(
        torch.tensor([2.0]), inactive, grad, cfg
    )
    assert active_high < active_low
    assert inactive_low < inactive_high


def test_calibration_respects_far_constraint():
    records = [
        {"lam_soft": 4.0, "active": 1.0, "regime": "R1"},
        {"lam_soft": 3.0, "active": 1.0, "regime": "R1"},
        {"lam_soft": 2.0, "active": 0.0, "regime": "R2"},
        {"lam_soft": 1.0, "active": 0.0, "regime": "R3"},
    ]
    threshold, metrics = calibrate_lambda_threshold(
        records,
        target_far=0.25,
        min_threshold=2.0,
        target_r3=0.20,
    )
    assert threshold >= 2.0
    assert metrics["false_activation_rate_R2_R3"] <= 0.25
    assert metrics["R2_activation_rate"] <= 0.25
    assert metrics["R3_activation_rate"] <= 0.20
    assert metrics["correct_activation_rate"] == pytest.approx(1.0)


def test_checkpoint_selection_prioritizes_car_then_far_then_loss():
    baseline = {
        "correct_activation_rate": 0.50,
        "false_activation_rate_R2_R3": 0.10,
    }
    higher_car = {
        "correct_activation_rate": 0.51,
        "false_activation_rate_R2_R3": 0.20,
    }
    lower_far = {
        "correct_activation_rate": 0.50,
        "false_activation_rate_R2_R3": 0.05,
    }
    assert checkpoint_selection_key(higher_car, 10.0) > (
        checkpoint_selection_key(baseline, 1.0)
    )
    assert checkpoint_selection_key(lower_far, 10.0) > (
        checkpoint_selection_key(baseline, 1.0)
    )
    assert checkpoint_selection_key(baseline, 0.5) > (
        checkpoint_selection_key(baseline, 1.0)
    )


def test_historical_checkpoint_loads_exact_model_interface():
    checkpoint = Path(
        "/mnt/data/adityas/GRL-SNAM/exp-rellis/checkpoints/"
        "rellis_stage2_decision_mid_ep12/best.pt"
    )
    if not checkpoint.exists():
        pytest.skip("Historical checkpoint is unavailable")
    model, metadata = load_initial_model(checkpoint, RepairConfig())
    outputs = model(
        torch.zeros(1, 0, 6),
        torch.zeros(1, 0, dtype=torch.bool),
        torch.tensor([[1.0, 0.0, 1.0, 1.0]]),
        torch.zeros(1, 2, 32, 32),
    )
    assert len(outputs) == 6
    assert outputs[3].shape == (1,)
    assert metadata["unexpected_parameters"] == []
    # Capacity repair trains only the risk representation and magnitude head.
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith(("risk_enc.", "lam_soft_head."))
        for name in trainable
    )


def test_head_only_mode_keeps_risk_encoder_frozen():
    checkpoint = Path(
        "/mnt/data/adityas/GRL-SNAM/exp-rellis/checkpoints/"
        "rellis_stage2_decision_mid_ep12/best.pt"
    )
    if not checkpoint.exists():
        pytest.skip("Historical checkpoint is unavailable")
    model, _ = load_initial_model(
        checkpoint,
        RepairConfig(train_risk_encoder=False),
    )
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("lam_soft_head.") for name in trainable)


def test_teacher_distillation_is_zero_at_initialization_and_detects_drift():
    checkpoint = Path(
        "/mnt/data/adityas/GRL-SNAM/exp-rellis/checkpoints/"
        "rellis_stage2_decision_mid_ep12/best.pt"
    )
    if not checkpoint.exists():
        pytest.skip("Historical checkpoint is unavailable")
    model, _ = load_initial_model(checkpoint, RepairConfig())
    model.eval()
    risk_patch = torch.rand(3, 2, 32, 32)
    geometry_context = torch.rand(3, 64)
    with torch.no_grad():
        risk_context = model.risk_enc(risk_patch)
        features = torch.cat([risk_context, geometry_context], dim=-1)
        teacher_hard = model.lam_hard_max * torch.sigmoid(
            model.lam_hard_head(features).squeeze(-1)
        )
        teacher_mu = model.mu_lat_max * torch.sigmoid(
            model.mu_lat_head(features).squeeze(-1)
        )
    batch = {
        "risk_patch": risk_patch,
        "geometry_context": geometry_context,
        "teacher_lam_hard": teacher_hard,
        "teacher_mu_lat": teacher_mu,
    }
    _, preservation, hard_mean, mu_mean, hard_max, mu_max = (
        _forward_material_outputs(model, batch, torch.device("cpu"))
    )
    assert float(preservation) == pytest.approx(0.0, abs=1e-10)
    assert float(hard_mean + mu_mean + hard_max + mu_max) == pytest.approx(
        0.0, abs=1e-10
    )
    with torch.no_grad():
        next(model.risk_enc.parameters()).add_(0.1)
    _, preservation, *_ = _forward_material_outputs(
        model, batch, torch.device("cpu")
    )
    assert float(preservation) > 0.0
