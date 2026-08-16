#!/usr/bin/env python3
"""Leakage-free ground-truth versus predicted-semantics comparison.

The auxiliary predictor is deliberately lightweight: it maps geometric and
intensity statistics in each observed LiDAR BEV cell to one of six semantic
risk groups.  It is trained on RELLIS sequences 00000--00002 and evaluated on
the frozen validation sequence 00003.  Sequence 00004 is never loaded.

The frozen sequence-00003 route-aware directional head is then evaluated on
the same 450 balanced episodes with either oracle labels or predicted labels.
Eligibility and direction correctness are always defined by the oracle map.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rellis.grl_rellis.bev import BevConfig
from rellis.grl_rellis.ontology import RellisOntology, load_ontology
from rellis.train_rellis_directional_force import DIRS_16, _as_path, _build_point
from rebuttal_experiments.exp7_semantic_corruption import (
    cluster_bootstrap,
    load_head,
    maps_from_label_grid,
    metric_sufficient,
    metrics_from_sufficient,
    predict_rows,
    route_contexts_for_goals,
)


SOURCE = Path("/mnt/data/adityas/GRL-SNAM/exp-rellis")
FEATURE_NAMES = (
    "log_point_count",
    "mean_z",
    "std_z",
    "min_z",
    "max_z",
    "z_range",
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "x_center",
    "y_center",
    "radial_distance",
    "sin_azimuth",
    "cos_azimuth",
)
GROUP_NAMES = ("low", "grass", "bush", "high_soft", "hard", "other")
REPRESENTATIVE_LABELS = np.asarray([23, 3, 19, 33, 4, 0], dtype=np.uint16)


class CellSemanticMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_lookup(ontology: RellisOntology) -> np.ndarray:
    lookup = np.full(65536, 5, dtype=np.int64)
    lookup[list(ontology.low_ids)] = 0
    lookup[3] = 1
    lookup[19] = 2
    lookup[[31, 33, 34]] = 3
    lookup[list(ontology.hard_ids)] = 4
    return lookup


def cell_features(points: np.ndarray, cfg: BevConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return one feature row per BEV cell and the corresponding point counts."""
    # Preserve the cache builder's float32 coordinate arithmetic exactly.
    x, y, z, intensity = (points[:, i] for i in range(4))
    valid = (
        (x >= cfg.x_min) & (x < cfg.x_max)
        & (y >= cfg.y_min) & (y < cfg.y_max)
    )
    x, y, z, intensity = x[valid], y[valid], z[valid], intensity[valid]
    c = np.floor((x - cfg.x_min) / cfg.resolution).astype(np.int64)
    r = np.floor((cfg.y_max - y) / cfg.resolution).astype(np.int64)
    in_grid = (r >= 0) & (r < cfg.rows) & (c >= 0) & (c < cfg.cols)
    r, c, z, intensity = r[in_grid], c[in_grid], z[in_grid], intensity[in_grid]
    linear = r * cfg.cols + c
    size = cfg.rows * cfg.cols
    count = np.bincount(linear, minlength=size).astype(np.int64)
    denom = np.maximum(count, 1)

    def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        total = np.bincount(linear, weights=values, minlength=size)
        total2 = np.bincount(linear, weights=values * values, minlength=size)
        mean = total / denom
        std = np.sqrt(np.maximum(total2 / denom - mean * mean, 0.0))
        low = np.full(size, np.inf)
        high = np.full(size, -np.inf)
        np.minimum.at(low, linear, values)
        np.maximum.at(high, linear, values)
        low[count == 0] = 0.0
        high[count == 0] = 0.0
        return mean, std, low, high

    z_mean, z_std, z_min, z_max = moments(z)
    i_mean, i_std, i_min, i_max = moments(intensity)
    rr, cc = np.indices((cfg.rows, cfg.cols))
    x_center = cfg.x_min + (cc.ravel() + 0.5) * cfg.resolution
    y_center = cfg.y_max - (rr.ravel() + 0.5) * cfg.resolution
    radius = np.hypot(x_center, y_center)
    azimuth = np.arctan2(y_center, x_center)
    features = np.column_stack(
        (
            np.log1p(count),
            z_mean,
            z_std,
            z_min,
            z_max,
            z_max - z_min,
            i_mean,
            i_std,
            i_min,
            i_max,
            x_center,
            y_center,
            radius,
            np.sin(azimuth),
            np.cos(azimuth),
        )
    ).astype(np.float32)
    return features.reshape(cfg.rows, cfg.cols, -1), count.reshape(cfg.rows, cfg.cols)


def load_split(path: Path, expected_sequences: set[str]) -> list[dict]:
    data = json.loads(path.read_text())
    episodes = data["episodes"]
    actual = {str(row["sequence"]) for row in episodes}
    if actual != expected_sequences:
        raise ValueError(f"{path} has sequences {actual}, expected {expected_sequences}")
    return episodes


def unique_scenes(episodes: Sequence[dict]) -> list[dict]:
    by_path: dict[str, dict] = {}
    for episode in episodes:
        by_path[str(episode["scene_path"])] = episode
    return [by_path[path] for path in sorted(by_path)]


def load_scene_samples(
    scenes: Sequence[dict],
    *,
    bev_root: Path,
    cfg: BevConfig,
    label_groups: np.ndarray,
    include_targets: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    cache: list[dict] = []
    for index, record in enumerate(scenes, start=1):
        payload = torch.load(
            bev_root / record["scene_path"], map_location="cpu", weights_only=False
        )
        if str(payload["meta"]["sequence"]) == "00004":
            raise RuntimeError("Sealed sequence 00004 must not be loaded")
        points = np.fromfile(payload["meta"]["scan_path"], dtype=np.float32).reshape(-1, 4)
        features, count = cell_features(points, cfg)
        maps = payload["maps"]
        observed = np.asarray(maps["observed_mask"], dtype=bool)
        if not np.array_equal(count.astype(np.int32), np.asarray(maps["point_count"])):
            raise AssertionError(f"Point-count reconstruction mismatch: {record['scene_path']}")
        if include_targets:
            labels = np.asarray(maps["z2_labels"], dtype=np.uint16)
            feature_parts.append(features[observed])
            target_parts.append(label_groups[labels[observed]])
        cache.append(
            {
                "record": record,
                "payload": payload,
                "features": features,
            }
        )
        if index % 25 == 0 or index == len(scenes):
            print(f"Loaded semantic features {index}/{len(scenes)} scenes", flush=True)
    x = np.concatenate(feature_parts) if feature_parts else np.empty((0, len(FEATURE_NAMES)))
    y = np.concatenate(target_parts) if target_parts else np.empty(0, dtype=np.int64)
    return x, y, cache


def train_predictor(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[CellSemanticMLP, np.ndarray, np.ndarray, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    x_norm = ((x - mean) / std).astype(np.float32)
    counts = np.bincount(y, minlength=len(GROUP_NAMES)).astype(np.float64)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights /= weights.mean()
    model = CellSemanticMLP(x.shape[1], hidden, len(GROUP_NAMES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32))
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        total_loss = 0.0
        correct = 0
        seen = 0
        start_time = time.perf_counter()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = torch.from_numpy(x_norm[idx])
            yb = torch.from_numpy(y[idx])
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(idx)
            correct += int((logits.argmax(dim=1) == yb).sum())
            seen += len(idx)
        row = {
            "epoch": epoch,
            "loss": total_loss / seen,
            "unweighted_accuracy": correct / seen,
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        history.append(row)
        print(
            f"Semantic epoch {epoch + 1}/{epochs}: "
            f"loss={row['loss']:.4f}, acc={row['unweighted_accuracy']:.3f}",
            flush=True,
        )
    model.eval()
    return model, mean, std, history


@torch.no_grad()
def predict_groups(
    model: CellSemanticMLP,
    features: np.ndarray,
    observed: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    flat = features[np.asarray(observed, dtype=bool)]
    flat = ((flat - mean) / std).astype(np.float32)
    pred_parts = []
    for start in range(0, len(flat), batch_size):
        pred_parts.append(model(torch.from_numpy(flat[start : start + batch_size])).argmax(1).numpy())
    out = np.full(np.asarray(observed).shape, 5, dtype=np.int64)
    out[np.asarray(observed, dtype=bool)] = np.concatenate(pred_parts)
    return out


def confusion_metrics(confusion: np.ndarray) -> tuple[list[dict], dict]:
    rows = []
    ious = []
    for idx, name in enumerate(GROUP_NAMES):
        tp = float(confusion[idx, idx])
        support = float(confusion[idx].sum())
        predicted = float(confusion[:, idx].sum())
        union = support + predicted - tp
        iou = tp / union if union else math.nan
        recall = tp / support if support else math.nan
        precision = tp / predicted if predicted else math.nan
        if np.isfinite(iou):
            ious.append(iou)
        rows.append(
            {
                "group": name,
                "support": int(support),
                "precision": precision,
                "recall": recall,
                "iou": iou,
            }
        )
    return rows, {
        "accuracy": float(np.trace(confusion) / max(confusion.sum(), 1)),
        "mean_iou": float(np.mean(ious)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument(
        "--train-split", type=Path, default=ROOT / "repair_experiments/splits/train_static.json"
    )
    parser.add_argument(
        "--validation-split",
        type=Path,
        default=ROOT / "repair_experiments/splits/validation_static.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "rebuttal_experiments/results/gap3_predicted_semantics",
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=27_370)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    bev_root = args.source_root / "cache/rellis_bev_all_seqbalanced_2500"
    ontology_path = args.source_root / "grl_rellis/risk_ontology.yaml"
    checkpoint_path = (
        args.source_root / "runs/rellis_directional_routeaware_aw050_far020_00003/best.pt"
    )
    bev_manifest_path = bev_root / "manifest.json"
    bev_manifest = json.loads(bev_manifest_path.read_text())
    cfg = BevConfig(**bev_manifest["config"]["bev"])
    ontology = load_ontology(ontology_path, "main")
    lookup = group_lookup(ontology)

    train_eps = load_split(args.train_split, {"00000", "00001", "00002"})
    validation_eps = load_split(args.validation_split, {"00003"})
    train_scenes = unique_scenes(train_eps)
    validation_scenes = unique_scenes(validation_eps)
    x_train, y_train, _ = load_scene_samples(
        train_scenes,
        bev_root=bev_root,
        cfg=cfg,
        label_groups=lookup,
        include_targets=True,
    )
    model, mean, std, history = train_predictor(
        x_train,
        y_train,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    write_csv(args.out / "training_history.csv", history)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "feature_mean": mean,
            "feature_std": std,
            "group_names": GROUP_NAMES,
            "representative_labels": REPRESENTATIVE_LABELS,
            "train_sequences": ["00000", "00001", "00002"],
            "validation_sequence": "00003",
        },
        args.out / "semantic_group_predictor.pt",
    )

    _, _, validation_cache = load_scene_samples(
        validation_scenes,
        bev_root=bev_root,
        cfg=cfg,
        label_groups=lookup,
        include_targets=False,
    )
    predicted_maps: dict[str, dict[str, np.ndarray]] = {}
    confusion = np.zeros((len(GROUP_NAMES), len(GROUP_NAMES)), dtype=np.int64)
    scene_audit = []
    for item in validation_cache:
        payload = item["payload"]
        clean_maps = payload["maps"]
        observed = np.asarray(clean_maps["observed_mask"], dtype=bool)
        clean_labels = np.asarray(clean_maps["z2_labels"], dtype=np.uint16)
        truth = lookup[clean_labels]
        predicted = predict_groups(
            model, item["features"], observed, mean, std, args.batch_size
        )
        np.add.at(confusion, (truth[observed], predicted[observed]), 1)
        predicted_labels = np.zeros_like(clean_labels)
        predicted_labels[observed] = REPRESENTATIVE_LABELS[predicted[observed]]
        scene_path = str(item["record"]["scene_path"])
        predicted_maps[scene_path] = maps_from_label_grid(
            predicted_labels,
            observed,
            clean_maps["point_count"],
            ontology,
            cfg,
        )
        scene_audit.append(
            {
                "scene_id": payload["meta"]["scene_id"],
                "sequence": payload["meta"]["sequence"],
                "observed_cells": int(observed.sum()),
                "group_accuracy": float(np.mean(predicted[observed] == truth[observed])),
                "changed_fraction": float(np.mean(predicted[observed] != truth[observed])),
            }
        )
    semantic_rows, semantic_summary = confusion_metrics(confusion)
    write_csv(args.out / "semantic_group_metrics.csv", semantic_rows)
    write_csv(args.out / "scene_semantic_audit.csv", scene_audit)
    np.savetxt(args.out / "semantic_group_confusion.csv", confusion, fmt="%d", delimiter=",")

    head, threshold, head_config = load_head(checkpoint_path, "cpu")
    if (
        not head_config.get("route_aware")
        or str(head_config.get("holdout_sequence")) != "00003"
    ):
        raise ValueError("Directional checkpoint is not the frozen sequence-00003 LOSO head")

    episodes_by_scene: dict[str, list[dict]] = defaultdict(list)
    for episode in validation_eps:
        episodes_by_scene[str(episode["scene_path"])].append(episode)
    raw_rows: list[dict] = []
    for scene_number, scene_path in enumerate(sorted(episodes_by_scene), start=1):
        payload = torch.load(bev_root / scene_path, map_location="cpu", weights_only=False)
        clean_maps = payload["maps"]
        maps_by_source = {
            "ground_truth": clean_maps,
            "predicted": predicted_maps[scene_path],
        }
        goals = [tuple(int(x) for x in ep["goal_rc"]) for ep in episodes_by_scene[scene_path]]
        routes = {
            source: route_contexts_for_goals(
                maps,
                goals,
                risk_weight=float(head_config["route_risk_weight"]),
            )
            for source, maps in maps_by_source.items()
        }
        for episode in episodes_by_scene[scene_path]:
            path = _as_path(episode["stage1_path"])
            goal = tuple(int(x) for x in episode["goal_rc"])
            episode_uid = str(episode["episode_uid"])
            built_by_source: dict[str, list[dict]] = {}
            for source, maps in maps_by_source.items():
                built = []
                for idx in range(
                    0,
                    max(0, len(path) - 1),
                    max(1, int(head_config["stride"])),
                ):
                    row = _build_point(
                        maps,
                        path,
                        idx,
                        regime=str(episode["regime"]),
                        episode_id=episode_uid,
                        horizon_cells=int(head_config["horizon_cells"]),
                        long_horizon_cells=int(head_config["long_horizon_cells"]),
                        hard_margin_m=float(head_config["hard_margin_m"]),
                        improvement_margin=float(head_config["improvement_margin"]),
                        route=routes[source][goal],
                        route_max_ratio=float(head_config["route_max_ratio"]),
                    )
                    if row is not None:
                        row["path_index"] = idx
                        built.append(row)
                built_by_source[source] = built
            clean_by_idx = {
                int(row["path_index"]): row for row in built_by_source["ground_truth"]
            }
            paired_indices = sorted(
                set(clean_by_idx)
                & {int(row["path_index"]) for row in built_by_source["predicted"]}
            )
            for source in ("ground_truth", "predicted"):
                observed = {
                    int(row["path_index"]): row for row in built_by_source[source]
                }
                ordered = [observed[idx] for idx in paired_indices]
                pred, scores, p_noop = predict_rows(
                    head, ordered, threshold, "cpu", args.batch_size
                )
                for local_index, path_index in enumerate(paired_indices):
                    clean = clean_by_idx[path_index]
                    pred_class = int(pred[local_index])
                    correct = bool(
                        pred_class > 0
                        and int(clean["label"]) > 0
                        and np.dot(
                            DIRS_16[pred_class - 1],
                            DIRS_16[int(clean["best_idx"])],
                        )
                        > 1e-3
                    )
                    raw_rows.append(
                        {
                            "episode_uid": episode_uid,
                            "episode_id": episode["episode_id"],
                            "scene_id": episode["scene_id"],
                            "sequence": "00003",
                            "regime": episode["regime"],
                            "path_index": path_index,
                            "map_source": source,
                            "corruption_probability": 0.0 if source == "ground_truth" else 1.0,
                            "pred_class": pred_class,
                            "active": int(pred_class > 0),
                            "p_noop": float(p_noop[local_index]),
                            "activation_score": float(scores[local_index]),
                            "activation_threshold": threshold,
                            "clean_label": int(clean["label"]),
                            "clean_best_idx": int(clean["best_idx"]),
                            "clean_has_safe_alt": float(clean["has_safe_alt"]),
                            "correct_activation": int(correct),
                            "observed_label": int(ordered[local_index]["label"]),
                            "observed_has_safe_alt": float(ordered[local_index]["has_safe_alt"]),
                        }
                    )
        if scene_number % 10 == 0 or scene_number == len(episodes_by_scene):
            print(
                f"Evaluated predicted semantics {scene_number}/{len(episodes_by_scene)} scenes",
                flush=True,
            )

    ci_rows, delta_rows = cluster_bootstrap(
        raw_rows,
        [0.0, 1.0],
        reps=args.bootstrap_reps,
        seed=args.seed + 1,
    )
    for row in ci_rows:
        row["map_source"] = (
            "ground_truth" if float(row["corruption_probability"]) == 0.0 else "predicted"
        )
    for row in delta_rows:
        row["map_source"] = (
            "ground_truth" if float(row["corruption_probability"]) == 0.0 else "predicted"
        )
    summary_rows = []
    for source, level in (("ground_truth", 0.0), ("predicted", 1.0)):
        selected = [
            row for row in raw_rows
            if float(row["corruption_probability"]) == level
        ]
        summary_rows.append(
            {
                "map_source": source,
                **metrics_from_sufficient(metric_sufficient(selected)),
            }
        )
    write_csv(args.out / "raw_control_predictions.csv", raw_rows)
    write_csv(args.out / "control_summary.csv", summary_rows)
    write_csv(args.out / "control_cluster_bootstrap_ci.csv", ci_rows)
    write_csv(args.out / "control_paired_deltas.csv", delta_rows)

    ci_lookup = {
        (row["map_source"], row["metric"]): row for row in ci_rows
    }
    delta_lookup = {
        row["metric"]: row for row in delta_rows if row["map_source"] == "predicted"
    }
    lines = [
        "# Gap 3 — ground-truth versus predicted semantics",
        "",
        "A lightweight LiDAR-cell semantic-risk predictor was trained on sequences "
        "`00000–00002` and evaluated on validation sequence `00003`. The sealed "
        "sequence `00004` was not loaded. The comparison uses the same frozen "
        "route-aware directional head and the same 450 balanced validation episodes.",
        "",
        "This auxiliary predictor is a controlled perception-stress test, not a claim "
        "of state-of-the-art RELLIS semantic segmentation.",
        "",
        "## Perception quality",
        "",
        f"- Observed-cell semantic-group accuracy: **{semantic_summary['accuracy']:.3f}**.",
        f"- Mean group IoU: **{semantic_summary['mean_iou']:.3f}**.",
        f"- Train/validation scenes: **{len(train_scenes)}/{len(validation_scenes)}**.",
        "",
        "## Same-episode control metrics",
        "",
        "| Map input | CAR (95% CI) | FAR (95% CI) | SR (95% CI) | Activation |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        source = str(row["map_source"])
        cells = []
        for metric in ("CAR", "FAR", "SR"):
            ci = ci_lookup[(source, metric)]
            cells.append(
                f"{row[metric]:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
            )
        lines.append(
            f"| {source.replace('_', ' ').title()} | {cells[0]} | {cells[1]} | "
            f"{cells[2]} | {row['activation_rate']:.3f} |"
        )
    lines += [
        "",
        "## Paired predicted-minus-ground-truth change",
        "",
        "| Metric | Δ (95% CI) |",
        "|---|---:|",
    ]
    for metric in ("CAR", "FAR", "SR", "activation_rate"):
        row = delta_lookup[metric]
        lines.append(
            f"| {metric} | {row['paired_delta']:+.3f} "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] |"
        )
    lines += [
        "",
        "Eligibility and correct-direction labels always come from the ground-truth "
        "map; only the frozen head's semantic inputs and route context change. CIs use "
        "a paired episode-cluster bootstrap.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python rebuttal_experiments/exp_predicted_semantics.py",
        "```",
    ]
    (args.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    provenance = {
        "experiment": "gap3_predicted_semantics",
        "scope": "validation-only; sequence 00004 sealed",
        "train_sequences": ["00000", "00001", "00002"],
        "validation_sequence": "00003",
        "num_train_scenes": len(train_scenes),
        "num_validation_scenes": len(validation_scenes),
        "num_validation_episodes": len(validation_eps),
        "feature_names": FEATURE_NAMES,
        "group_names": GROUP_NAMES,
        "representative_labels": REPRESENTATIVE_LABELS.tolist(),
        "semantic_summary": semantic_summary,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "hashes": {
            "train_split_sha256": sha256(args.train_split),
            "validation_split_sha256": sha256(args.validation_split),
            "bev_manifest_sha256": sha256(bev_manifest_path),
            "ontology_sha256": sha256(ontology_path),
            "directional_checkpoint_sha256": sha256(checkpoint_path),
            "semantic_predictor_sha256": sha256(args.out / "semantic_group_predictor.pt"),
        },
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(summary_rows, indent=2))
    print(f"Wrote predicted-semantics artifacts to {args.out}")


if __name__ == "__main__":
    main()
