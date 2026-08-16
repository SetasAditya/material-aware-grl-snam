#!/usr/bin/env python3
"""Experiment 5: within-sequence layout-diversity scaling for Stage 2.

This script intentionally trains the paper's ``CoefEnergyNetMaterial`` through
``MaterialTrainer`` and ``selectivity_loss``.  It does not use the separate
candidate-direction classifier.

The source stagewise bundle contains only RELLIS sequence 00000.  Consequently,
this experiment measures generalization to held-out scenes/layouts within that
sequence; it is not a cross-sequence generalization experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-exp5")

DEFAULT_SOURCE_REPO = Path("/mnt/data/adityas/GRL-SNAM")
DEFAULT_STAGEWISE_ROOT = (
    DEFAULT_SOURCE_REPO / "exp-rellis/data/rellis_stagewise_val1500_decision"
)
DEFAULT_STAGE1_CKPT = DEFAULT_SOURCE_REPO / "checkpoints/s1/best.pt"
FRACTIONS = (0.10, 0.25, 0.50, 1.00)
REGIMES = ("R1", "R2", "R3")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_STAGEWISE_ROOT)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CKPT)
    parser.add_argument("--out", type=Path, default=here / "outputs")
    parser.add_argument("--seed", type=int, default=27370)
    parser.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force-eps", type=float, default=0.02)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--horizon-cells", type=int, default=8)
    parser.add_argument("--hard-margin-m", type=float, default=1.0)
    parser.add_argument("--improvement-margin", type=float, default=0.1)
    parser.add_argument(
        "--phase",
        choices=("manifests", "smoke", "full", "evaluate", "all"),
        default="all",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun completed smoke/full training runs.",
    )
    return parser.parse_args()


def _add_source_imports(source_repo: Path) -> None:
    for path in (source_repo, source_repo / "exp-rellis"):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "episodes": len(rows),
        "unique_scenes": len({str(row["scene_id"]) for row in rows}),
        "episodes_by_regime": dict(sorted(Counter(str(row["regime"]) for row in rows).items())),
        "scenes_by_regime": {
            regime: len({str(row["scene_id"]) for row in rows if row["regime"] == regime})
            for regime in REGIMES
        },
    }


def _stratified_scene_order(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
) -> List[str]:
    """Return a deterministic nested scene order with regime-balanced coverage.

    Each step selects the remaining scene that most improves the least-covered
    regime, with a seeded hash used only to break ties.  A scene can cover more
    than one regime.
    """

    scene_regime_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        scene_regime_counts[str(row["scene_id"])][str(row["regime"])] += 1
    remaining = set(scene_regime_counts)
    selected: List[str] = []
    selected_episode_counts: Counter[str] = Counter()

    def tie_value(scene_id: str) -> str:
        return hashlib.sha256(f"{seed}:{scene_id}".encode()).hexdigest()

    while remaining:
        total = max(sum(selected_episode_counts.values()), 1)
        shares = {
            regime: selected_episode_counts[regime] / total for regime in REGIMES
        }
        target = 1.0 / len(REGIMES)

        def score(scene_id: str) -> tuple[float, float, str]:
            counts = scene_regime_counts[scene_id]
            deficit_gain = sum(
                max(target - shares[regime], 0.0) * counts[regime]
                for regime in REGIMES
            )
            coverage_gain = sum(1.0 for regime in REGIMES if counts[regime] > 0)
            return (-deficit_gain, -coverage_gain, tie_value(scene_id))

        chosen = min(remaining, key=score)
        selected.append(chosen)
        selected_episode_counts.update(scene_regime_counts[chosen])
        remaining.remove(chosen)
    return selected


def build_manifests(
    source_root: Path,
    manifests_root: Path,
    fractions: Sequence[float],
    seed: int,
) -> Dict[str, Any]:
    source_manifest_path = source_root / "manifest.json"
    source_rows: List[Dict[str, Any]] = json.loads(source_manifest_path.read_text())

    sequences = sorted({str(row["scene_id"]).split("_", 1)[0] for row in source_rows})
    if sequences != ["00000"]:
        raise RuntimeError(
            f"Expected the supplied stagewise bundle to contain only sequence 00000; got {sequences}"
        )

    # Preserve the original balanced validation episodes.  To prevent layout
    # leakage, every scene represented in that validation set is excluded from
    # every training subset (including train-labelled records from 9 overlapping
    # scenes in the supplied manifest).
    fixed_val = [dict(row) for row in source_rows if row.get("split") == "val"]
    validation_scenes = {str(row["scene_id"]) for row in fixed_val}
    train_pool = [
        dict(row)
        for row in source_rows
        if row.get("split") == "train" and str(row["scene_id"]) not in validation_scenes
    ]
    if not train_pool or not fixed_val:
        raise RuntimeError("Source manifest must contain nonempty train and validation records")

    source_repo = source_root.parents[2]
    for row in train_pool + fixed_val:
        raw = Path(str(row["path"]))
        row["path"] = str(raw if raw.is_absolute() else (source_repo / raw).resolve())

    scene_order = _stratified_scene_order(train_pool, seed)
    manifests_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "source_manifest": str(source_manifest_path.resolve()),
        "source_sequence_ids": sequences,
        "claim_scope": "held-out layout generalization within RELLIS sequence 00000",
        "seed": seed,
        "validation_policy": (
            "Original 180 balanced validation episodes; all 28 scenes appearing "
            "in validation are excluded from every training subset."
        ),
        "source_train_validation_scene_overlap": len(
            {
                str(row["scene_id"])
                for row in source_rows
                if row.get("split") == "train"
            }
            & validation_scenes
        ),
        "eligible_train_pool": _counts(train_pool),
        "fixed_validation": _counts(fixed_val),
        "scene_order": scene_order,
        "subsets": {},
    }

    previous_scenes: set[str] = set()
    for fraction in sorted(set(float(value) for value in fractions)):
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Fractions must lie in (0,1], got {fraction}")
        n_scenes = min(len(scene_order), max(1, int(round(fraction * len(scene_order)))))
        selected_scenes = set(scene_order[:n_scenes])
        if not previous_scenes.issubset(selected_scenes):
            raise AssertionError("Scene subsets are not nested")
        previous_scenes = selected_scenes
        train_rows = [
            {**row, "split": "train"}
            for row in train_pool
            if str(row["scene_id"]) in selected_scenes
        ]
        val_rows = [{**row, "split": "val"} for row in fixed_val]
        if selected_scenes & validation_scenes:
            raise AssertionError("Training and validation scenes overlap")

        label = f"p{int(round(100 * fraction)):03d}"
        subset_root = manifests_root / label
        subset_root.mkdir(parents=True, exist_ok=True)
        manifest_rows = train_rows + val_rows
        (subset_root / "manifest.json").write_text(json.dumps(manifest_rows, indent=2) + "\n")
        subset_info = {
            "label": label,
            "requested_fraction": fraction,
            "selected_scene_fraction": n_scenes / len(scene_order),
            "selected_scene_ids": [scene for scene in scene_order if scene in selected_scenes],
            "train": _counts(train_rows),
            "validation": _counts(val_rows),
            "tensor_policy": "No tensors copied; manifest episode paths are absolute source references.",
        }
        (subset_root / "subset_metadata.json").write_text(
            json.dumps(subset_info, indent=2) + "\n"
        )
        summary["subsets"][label] = subset_info

    (manifests_root / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def _state_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_shared_initialization(
    source_repo: Path,
    stage1_checkpoint: Path,
    out_path: Path,
    seed: int,
    threads: int,
) -> Dict[str, Any]:
    _add_source_imports(source_repo)
    from train_material import CoefEnergyNetMaterial, load_geometry_weights

    seed_everything(seed, threads)
    model = CoefEnergyNetMaterial(patch_size=32, lam_soft_max=5.0, lam_hard_max=10.0)
    load_geometry_weights(model, str(stage1_checkpoint), "cpu")
    payload = {
        "seed": seed,
        "stage1_checkpoint": str(stage1_checkpoint.resolve()),
        "model_state_dict": model.state_dict(),
        "state_sha256": _state_hash(model.state_dict()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {key: value for key, value in payload.items() if key != "model_state_dict"}


def _load_dataset(
    source_repo: Path,
    subset_root: Path,
    *,
    split: str,
    active_probability: float,
):
    from train_material import DFC2018RolloutCfg, DFC2018ShortRollouts

    # Episode checkpoint paths embedded in the source tensors are relative to
    # the original repository.  The dataset itself remains unmodified.
    with _working_directory(source_repo):
        return DFC2018ShortRollouts(
            str(subset_root),
            DFC2018RolloutCfg(
                split=split,
                waypoint_mode="oracle",
                selectivity_active_prob=active_probability,
            ),
        )


def train_one(
    *,
    source_repo: Path,
    subset_root: Path,
    init_path: Path,
    run_root: Path,
    seed: int,
    threads: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Dict[str, Any]:
    from train_material import (
        CoefEnergyNetMaterial,
        MaterialTrainer,
        TrainCfgMaterial,
        collate_fn,
    )

    seed_everything(seed, threads)
    initialization = torch.load(init_path, map_location="cpu", weights_only=False)
    model = CoefEnergyNetMaterial(patch_size=32, lam_soft_max=5.0, lam_hard_max=10.0)
    model.load_state_dict(initialization["model_state_dict"], strict=True)
    if _state_hash(model.state_dict()) != initialization["state_sha256"]:
        raise RuntimeError("Shared model initialization hash mismatch")

    train_ds = _load_dataset(
        source_repo, subset_root, split="train", active_probability=0.50
    )
    val_ds = _load_dataset(
        source_repo, subset_root, split="val", active_probability=0.0
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    cfg = TrainCfgMaterial(
        stage=2,
        epochs=epochs,
        bs=batch_size,
        lr=lr,
        workers=0,
        device="cpu",
        out=str(run_root),
        log_every=10_000,
        w_selectivity=1.0,
        w_select_inactive=1.0,
        w_select_lambda=0.05,
        selectivity_margin=0.005,
        patch_size=32,
        waypoint_mode="oracle",
        train_risk_only=True,
        selectivity_active_prob=0.50,
        selectivity_only=True,
    )
    run_root.mkdir(parents=True, exist_ok=True)
    trainer = MaterialTrainer(model, cfg)
    started = time.perf_counter()
    trainer.train(train_loader, val_loader)
    duration = time.perf_counter() - started
    final_checkpoint = run_root / f"epoch_{epochs - 1:03d}.pt"
    if not final_checkpoint.exists():
        raise RuntimeError(f"Missing final checkpoint {final_checkpoint}")
    metadata = {
        "subset": subset_root.name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "device": "cpu",
        "train_dataset_draws_per_epoch": len(train_ds),
        "validation_dataset_draws": len(val_ds),
        "training_seconds": duration,
        "seconds_per_epoch": duration / max(epochs, 1),
        "timing_scope": (
            "Optimizer epochs including the trainer's validation pass; excludes "
            "dataset construction and post-training force evaluation."
        ),
        "initialization_sha256": initialization["state_sha256"],
        "parameter_count_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_count_trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "training_path": (
            "CoefEnergyNetMaterial + MaterialTrainer + selectivity_loss; "
            "risk encoder and lambda heads trainable; geometry frozen."
        ),
        "final_checkpoint": str(final_checkpoint.resolve()),
    }
    (run_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _load_eval_episodes(source_root: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stage_rows = json.loads((source_root / "manifest.json").read_text())
    val_source_ids = {
        str(row["source_episode_id"])
        for row in stage_rows
        if row.get("split") == "val"
    }
    pairs = json.loads((source_root / "source_pairs_manifest.json").read_text())
    lookup = {str(row["episode_id"]): row for row in pairs["episodes"]}
    episodes = [lookup[source_id] for source_id in sorted(val_source_ids)]
    return episodes, pairs


def evaluate_checkpoint(
    *,
    source_repo: Path,
    source_root: Path,
    checkpoint: Path,
    force_eps: float,
    stride: int,
    horizon_cells: int,
    hard_margin_m: float,
    improvement_margin: float,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from eval_rellis_selectivity import _selectivity_rows, _summarize_selectivity
    from scripts.baselines.dfc.models import load_model

    model = load_model(checkpoint, device="cpu")
    episodes, pairs = _load_eval_episodes(source_root)
    bev_root_raw = pairs.get("config", {}).get("bev_root")
    if not bev_root_raw:
        raise RuntimeError("Source pairs manifest does not identify its BEV root")
    bev_root = Path(str(bev_root_raw))
    if not bev_root.is_absolute():
        bev_root = source_repo / bev_root
    bev_manifest = json.loads((bev_root / "manifest.json").read_text())
    resolution = float(bev_manifest["config"]["bev"]["resolution"])
    force_rows: List[Dict[str, Any]] = []
    scene_cache: Dict[str, Dict[str, Any]] = {}
    for episode in episodes:
        scene_id = str(episode["scene_id"])
        if scene_id not in scene_cache:
            scene = torch.load(
                source_root / f"scene_{scene_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            scene_cache[scene_id] = scene["maps"]
        path = [(int(point[0]), int(point[1])) for point in episode["stage1_path"]]
        force_rows.extend(
            _selectivity_rows(
                scene_cache[scene_id],
                path,
                regime=str(episode["regime"]),
                episode_id=str(episode["episode_id"]),
                lam_soft=0.0,
                lam_hard=0.0,
                gsd=resolution,
                horizon_cells=horizon_cells,
                hard_margin_m=hard_margin_m,
                improvement_margin=improvement_margin,
                stride=stride,
                force_source="coef_energy_material_checkpoint_lambdas",
                model=model,
                device="cpu",
                model_patch_size=32,
                model_waypoint_stride=6,
            )
        )
    summary = _summarize_selectivity(force_rows, eps=force_eps)
    summary.update(
        {
            "evaluation_episodes": len(episodes),
            "evaluation_scenes": len({str(row["scene_id"]) for row in episodes}),
            "evaluation_regime_counts": dict(
                sorted(Counter(str(row["regime"]) for row in episodes).items())
            ),
            "checkpoint": str(checkpoint.resolve()),
            "bev_resolution_m": resolution,
            "force_eps": force_eps,
            "stride": stride,
            "claim_scope": "held-out layouts within RELLIS sequence 00000",
        }
    )
    return summary, force_rows


def _write_force_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def consolidate_results(
    out: Path,
    manifest_summary: Mapping[str, Any],
    fractions: Sequence[float],
    epochs: int,
    eval_config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fraction in sorted(set(float(value) for value in fractions)):
        label = f"p{int(round(100 * fraction)):03d}"
        subset = manifest_summary["subsets"][label]
        full_root = out / "training" / label / f"full_{epochs}ep"
        run_metadata = json.loads((full_root / "run_metadata.json").read_text())
        eval_summary = json.loads((full_root / "evaluation_summary.json").read_text())
        rows.append(
            {
                "subset": label,
                "requested_scene_fraction": fraction,
                "train_scenes": subset["train"]["unique_scenes"],
                "train_episodes": subset["train"]["episodes"],
                "train_R1_episodes": subset["train"]["episodes_by_regime"].get("R1", 0),
                "train_R2_episodes": subset["train"]["episodes_by_regime"].get("R2", 0),
                "train_R3_episodes": subset["train"]["episodes_by_regime"].get("R3", 0),
                "eval_scenes": eval_summary["evaluation_scenes"],
                "eval_episodes": eval_summary["evaluation_episodes"],
                "CAR": eval_summary["correct_activation_rate"],
                "FAR": eval_summary["false_activation_rate"],
                "SR": eval_summary["selectivity_ratio"],
                "force_risk_alignment": eval_summary["force_risk_alignment"],
                "training_seconds": run_metadata["training_seconds"],
                "seconds_per_epoch": run_metadata["seconds_per_epoch"],
                "parameter_count_total": run_metadata["parameter_count_total"],
                "parameter_count_trainable": run_metadata["parameter_count_trainable"],
                "initialization_sha256": run_metadata["initialization_sha256"],
                "checkpoint": eval_summary["checkpoint"],
            }
        )

    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "experiment": 5,
        "title": "Training-layout diversity (within sequence 00000)",
        "claim_scope": "held-out layout generalization within RELLIS sequence 00000",
        "not_supported_claim": "cross-sequence generalization",
        "evaluation_config": dict(eval_config),
        "rows": rows,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Experiment 5 — Training-layout diversity",
        "",
        "> **Scope limitation:** the available `rellis_stagewise_val1500_decision` "
        "bundle contains only RELLIS sequence `00000`. These results test held-out "
        "scene/layout generalization *within one sequence*, not cross-sequence generalization.",
        "",
        "All variants use `CoefEnergyNetMaterial`, the same byte-identical initialization, "
        "the same seed/configuration, frozen geometry, and train only the risk encoder and "
        "lambda heads with the model's selectivity objective. The final equal-budget "
        f"checkpoint after {epochs} epochs is evaluated on the same 180 validation episodes "
        "(60 per regime). Validation scenes are excluded from every training subset.",
        "",
        "| Train scenes | Train episodes (R1/R2/R3) | CAR ↑ | FAR ↓ | SR ↑ | Train time (s) | Total/trainable params |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['train_scenes']} | {row['train_episodes']} "
            f"({row['train_R1_episodes']}/{row['train_R2_episodes']}/{row['train_R3_episodes']}) "
            f"| {row['CAR']:.4f} | {row['FAR']:.4f} | {row['SR']:.4f} "
            f"| {row['training_seconds']:.1f} | "
            f"{row['parameter_count_total']:,}/{row['parameter_count_trainable']:,} |"
        )
    lines.extend(
        [
            "",
            "CAR/FAR use force threshold "
            f"`{eval_config['force_eps']}` and the checkpoint-predicted "
            r"$\lambda_{\mathrm{soft}},\lambda_{\mathrm{hard}}$ values.",
            "",
            "## Interpretation",
            "",
            "Increasing training layouts from 10 to 100 reduced held-out FAR from "
            f"{rows[0]['FAR']:.3f} to {rows[-1]['FAR']:.3f} and changed SR from "
            f"{rows[0]['SR']:.3f} to {rows[-1]['SR']:.3f}. CAR did not improve "
            f"({rows[0]['CAR']:.3f} to {rows[-1]['CAR']:.3f}). Thus this short, "
            "single-seed study supports only a modest improvement in suppression "
            "with greater within-sequence layout coverage; it does not establish "
            "improved activation or cross-sequence generalization.",
            "",
            "Each checkpoint was evaluated on 2,671 force samples: 850 R1, 903 R2, "
            "and 918 R3. The CAR denominator contains the 54 R1 samples that satisfy "
            "the evaluator's safe-alternative condition; the FAR denominator contains "
            "all 1,821 R2/R3 samples.",
            "",
            "Reported training time covers optimizer epochs and the trainer's internal "
            "validation passes, but excludes dataset construction and post-training "
            "force evaluation. Parameter count is fixed across subsets.",
            "",
            "The 1-epoch smoke checkpoints are retained separately under `training/*/smoke_1ep/`; "
            "they are validation artifacts and are not mixed into the reported equal-budget results.",
        ]
    )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return rows


def main() -> None:
    args = parse_args()
    if args.smoke_epochs < 1 or args.epochs < 1:
        raise ValueError("Epoch budgets must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    _add_source_imports(args.source_repo)
    seed_everything(args.seed, args.threads)

    manifests_root = args.out / "manifests"
    manifest_summary = build_manifests(
        args.source_root, manifests_root, args.fractions, args.seed
    )
    if args.phase == "manifests":
        print(json.dumps(manifest_summary, indent=2))
        return

    init_path = args.out / "shared_initialization.pt"
    init_metadata = make_shared_initialization(
        args.source_repo,
        args.stage1_checkpoint,
        init_path,
        args.seed,
        args.threads,
    )
    (args.out / "shared_initialization.json").write_text(
        json.dumps(init_metadata, indent=2) + "\n"
    )

    fractions = sorted(set(float(value) for value in args.fractions))
    phases: List[tuple[str, int]] = []
    if args.phase in ("smoke", "all"):
        phases.append((f"smoke_{args.smoke_epochs}ep", args.smoke_epochs))
    if args.phase in ("full", "all"):
        phases.append((f"full_{args.epochs}ep", args.epochs))
    for phase_label, epoch_budget in phases:
        for fraction in fractions:
            label = f"p{int(round(100 * fraction)):03d}"
            run_root = args.out / "training" / label / phase_label
            if (run_root / "run_metadata.json").exists() and not args.overwrite:
                print(f"Skipping completed run {run_root}")
                continue
            print(f"\n=== {phase_label}: {label} ===", flush=True)
            train_one(
                source_repo=args.source_repo,
                subset_root=manifests_root / label,
                init_path=init_path,
                run_root=run_root,
                seed=args.seed,
                threads=args.threads,
                epochs=epoch_budget,
                batch_size=args.batch_size,
                lr=args.lr,
            )

    if args.phase in ("evaluate", "all"):
        eval_config = {
            "force_eps": args.force_eps,
            "stride": args.stride,
            "horizon_cells": args.horizon_cells,
            "hard_margin_m": args.hard_margin_m,
            "improvement_margin": args.improvement_margin,
        }
        for fraction in fractions:
            label = f"p{int(round(100 * fraction)):03d}"
            run_root = args.out / "training" / label / f"full_{args.epochs}ep"
            checkpoint = run_root / f"epoch_{args.epochs - 1:03d}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Full checkpoint missing for {label}; run --phase full first: {checkpoint}"
                )
            print(f"\n=== evaluate: {label} ===", flush=True)
            summary, force_rows = evaluate_checkpoint(
                source_repo=args.source_repo,
                source_root=args.source_root,
                checkpoint=checkpoint,
                **eval_config,
            )
            (run_root / "evaluation_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            _write_force_rows(run_root / "force_samples.csv", force_rows)
        rows = consolidate_results(
            args.out, manifest_summary, fractions, args.epochs, eval_config
        )
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
