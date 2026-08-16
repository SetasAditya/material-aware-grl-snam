#!/usr/bin/env python3
"""Calibrate an existing RELLIS directional-head checkpoint on its train split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_rellis_directional_force import (
    DIRS_16,
    DirectionalForceHead,
    build_dataset,
    calibrate_activation_threshold,
    evaluate,
    split_rows,
    write_rows_csv,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Calibrate a trained RELLIS directional-force checkpoint.")
    ap.add_argument("--run", type=Path, required=True, help="Existing training run containing best.pt and summary.json.")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for calibrated summary/checkpoint.")
    ap.add_argument("--target-far", type=float, default=0.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    source_summary = json.loads((args.run / "summary.json").read_text())
    cfg = dict(source_summary["config"])
    cfg.setdefault("long_horizon_cells", 24)
    cfg.setdefault("route_aware", False)
    cfg.setdefault("route_risk_weight", 12.0)
    cfg.setdefault("route_max_ratio", 2.2)
    cfg["device"] = args.device
    cfg["activation_threshold"] = None
    cfg["calibrate_target_far"] = args.target_far
    cfg["bev_root"] = Path(cfg["bev_root"])
    cfg["pairs_root"] = Path(cfg["pairs_root"])
    cfg["out"] = args.out
    ns = SimpleNamespace(**cfg)

    rows, meta = build_dataset(ns)
    train_idx, val_idx = split_rows(
        rows,
        ns.val_frac,
        seed=ns.seed,
        split_mode=ns.split_mode,
        holdout_sequence=ns.holdout_sequence,
    )

    ckpt = torch.load(args.run / "best.pt", map_location=args.device, weights_only=False)
    model = DirectionalForceHead(int(ckpt["in_dim"]), int(ns.hidden), 1 + len(DIRS_16)).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    threshold, calibration_metrics = calibrate_activation_threshold(rows, train_idx, model, ns)
    ns.activation_threshold = threshold

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(ns).items()},
        "source_run": str(args.run),
        "data": meta,
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "class_counts": source_summary.get("class_counts"),
        "best_epoch": source_summary.get("best_epoch"),
        "best_val_loss": source_summary.get("best_val_loss"),
        "activation_threshold": threshold,
        "calibration_metrics": calibration_metrics,
        "train_metrics": evaluate(rows, train_idx, model, ns),
        "val_metrics": evaluate(rows, val_idx, model, ns),
        "all_metrics": evaluate(rows, list(range(len(rows))), model, ns),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": ckpt["model_state_dict"],
            "summary": summary,
            "in_dim": int(ckpt["in_dim"]),
            "dirs": np.asarray(ckpt["dirs"]),
            "activation_threshold": threshold,
            "source_run": str(args.run),
        },
        args.out / "best.pt",
    )
    write_rows_csv(rows, train_idx, args.out / "train_rows.csv")
    write_rows_csv(rows, val_idx, args.out / "val_rows.csv")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
