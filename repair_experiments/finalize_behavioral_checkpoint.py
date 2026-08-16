#!/usr/bin/env python3
"""Materialize the preregistration-selected immutable epoch as ``best.pt``.

This performs no optimization and loads only the validation manifest.  It
exists because CPU ``detach().cpu()`` can share storage with live parameters;
the immutable per-epoch files are authoritative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from repair_experiments.train_behavioral_soft_force import (
    ALLOWED_VALIDATION_SEQUENCES,
    DecisionDataset,
    RepairConfig,
    _load_manifest,
    build_decision_specs,
    calibrate_lambda_threshold,
    checkpoint_selection_key,
    collect_predictions,
    collate_decisions,
    encode_frozen_dataset,
    load_initial_model,
    summarize_predictions,
    _write_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path("repair_experiments/splits/validation_static.json"),
    )
    args = parser.parse_args()

    summary_path = args.run / "summary.json"
    config_path = args.run / "config.json"
    summary = json.loads(summary_path.read_text())
    config_payload = json.loads(config_path.read_text())
    cfg = RepairConfig(**config_payload["repair_training_config"])
    curves = summary["trained"]["curves"]
    if all("checkpoint_selection_key" in curve for curve in curves):
        selected = max(
            curves,
            key=lambda curve: tuple(curve["checkpoint_selection_key"]),
        )
    else:
        # Historical default control selected minimum aggregate validation loss.
        selected = min(
            curves,
            key=lambda curve: float(curve["validation"]["loss"]),
        )
    selected_epoch = int(selected["epoch"])
    epoch_path = args.run / f"epoch_{selected_epoch:03d}.pt"
    epoch_payload = torch.load(
        epoch_path, map_location="cpu", weights_only=False
    )

    validation_manifest, bev_root = _load_manifest(
        args.validation_manifest,
        expected_split="validation",
        allowed_sequences=ALLOWED_VALIDATION_SEQUENCES,
    )
    specs, data_summary = build_decision_specs(
        validation_manifest["episodes"], bev_root, cfg
    )
    raw_dataset = DecisionDataset(specs, bev_root, cfg)
    raw_loader = DataLoader(
        raw_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_decisions,
    )
    initial_checkpoint = Path(
        config_payload["provenance"]["initialization"]["checkpoint"]
    )
    teacher_model, _ = load_initial_model(initial_checkpoint, cfg)
    encoded_dataset, encode_seconds = encode_frozen_dataset(
        teacher_model, raw_loader, cfg
    )
    loader = DataLoader(
        encoded_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    selected_model, _ = load_initial_model(initial_checkpoint, cfg)
    selected_model.load_state_dict(
        epoch_payload["model_state_dict"], strict=True
    )
    validation = collect_predictions(selected_model, loader, cfg)
    required_lambda = (
        2.0 * cfg.min_effect_m
        / (
            cfg.horizon_seconds ** 2
            * cfg.resolution_m_per_cell
        )
    )
    threshold, calibration = calibrate_lambda_threshold(
        validation["records"],
        cfg.target_far,
        min_threshold=required_lambda,
        target_r3=0.20,
    )
    if (
        "validation_lambda_threshold" in selected
        and abs(
            threshold - float(selected["validation_lambda_threshold"])
        ) > 1e-6
    ):
        raise RuntimeError("Recomputed selected-epoch threshold mismatch")
    selection_key = list(
        checkpoint_selection_key(
            calibration, float(selected["validation"]["loss"])
        )
    )

    trained = summary["trained"]
    trained.update(
        {
            "best_epoch": selected_epoch,
            "best_validation_loss": float(selected["validation"]["loss"]),
            "best_checkpoint_selection_key": selection_key,
            "validation_lambda_threshold": threshold,
            "validation_calibration": calibration,
            "validation_summary": summarize_predictions(
                validation["records"], threshold, cfg
            ),
        }
    )
    final_payload = {
        **epoch_payload,
        "repair_calibration": {
            "lambda_active_threshold": threshold,
            "target_far": cfg.target_far,
            "horizon_seconds": cfg.horizon_seconds,
            "resolution_m_per_cell": cfg.resolution_m_per_cell,
            "min_effect_m": cfg.min_effect_m,
            "minimum_effect_lambda": required_lambda,
            "checkpoint_selection": (
                "maximize validation CAR subject to lambda>=2, R2<=0.25, "
                "R3<=0.20; tie-break pooled FAR then validation loss"
            ),
            "validation_metrics": calibration,
        },
        "validation_metrics": selected["validation"],
        "finalization": {
            "source_epoch_checkpoint": str(epoch_path.resolve()),
            "optimization_performed": False,
            "validation_manifest": str(args.validation_manifest.resolve()),
            "validation_data": data_summary,
            "encoding_seconds": encode_seconds,
        },
    }
    torch.save(final_payload, args.run / "best.pt")
    _write_predictions(
        args.run / "validation_predictions.csv",
        validation["records"],
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    (args.run / "finalization.json").write_text(
        json.dumps(final_payload["finalization"], indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "selected_epoch": selected_epoch,
                "threshold": threshold,
                "calibration": calibration,
                "validation_summary": trained["validation_summary"],
                "validation_drift": selected["validation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
