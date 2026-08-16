#!/usr/bin/env python3
"""Experiment 7: deterministic semantic-map corruption on RELLIS LOSO.

The clean BEV remains the evaluation ground truth.  Only the semantic map
observed by the frozen directional head is corrupted.  This is a pointwise
selectivity study; it intentionally does not report navigation rollout metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, sobel
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rellis.grl_rellis.bev import BevConfig
from rellis.grl_rellis.ontology import RellisOntology, load_ontology
from rellis.train_rellis_directional_force import (
    DIRS_16,
    GRID_DIRS_8,
    GRID_STEP,
    DirectionalForceHead,
    _as_path,
    _build_point,
)


DEFAULT_SOURCE = Path("/mnt/data/adityas/GRL-SNAM/exp-rellis")
DEFAULT_LEVELS = (0.0, 0.1, 0.2, 0.3)


def stable_scene_seed(base_seed: int, scene_id: str) -> int:
    payload = f"exp7-semantic-corruption-v1:{int(base_seed)}:{scene_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def maps_from_label_grid(
    labels: np.ndarray,
    observed: np.ndarray,
    point_count: np.ndarray,
    ontology: RellisOntology,
    cfg: BevConfig,
) -> Dict[str, np.ndarray]:
    """Rebuild every label-dependent canonical BEV field."""
    labels = np.asarray(labels, dtype=np.uint16)
    observed_b = np.asarray(observed, dtype=bool)
    rho_lookup = np.full(65536, ontology.void_risk, dtype=np.float32)
    for idx, rho in ontology.rho_by_id.items():
        rho_lookup[int(idx)] = float(rho)
    risk_raw = np.where(observed_b, rho_lookup[labels], ontology.void_risk).astype(np.float32)
    risk_map = gaussian_filter(risk_raw, sigma=float(cfg.risk_sigma_cells)).astype(np.float32)
    risk_map = np.clip(risk_map, 0.0, 1.0)

    hard_lookup = np.zeros(65536, dtype=bool)
    hard_lookup[list(ontology.hard_ids)] = True
    hard_mask = hard_lookup[labels] & observed_b
    unknown_mask = ~observed_b if ontology.unknown_is_obstacle else np.zeros_like(observed_b)
    if cfg.unknown_inflate_cells > 0:
        unknown_mask = binary_dilation(unknown_mask, iterations=int(cfg.unknown_inflate_cells))
    hard_mask = hard_mask | unknown_mask
    if cfg.hard_inflate_cells > 0:
        hard_mask = binary_dilation(hard_mask, iterations=int(cfg.hard_inflate_cells))
    hard_mask = hard_mask.astype(np.uint8)

    soft_lookup = np.zeros(65536, dtype=bool)
    soft_lookup[list(ontology.soft_ids)] = True
    low_lookup = np.zeros(65536, dtype=bool)
    low_lookup[list(ontology.low_ids)] = True
    soft_mask = (soft_lookup[labels] & observed_b).astype(np.uint8)
    low_mask = (low_lookup[labels] & observed_b).astype(np.uint8)
    sdf_hard = (distance_transform_edt(~hard_mask.astype(bool)) * cfg.resolution).astype(np.float32)
    grad_row = (sobel(risk_map, axis=0) / (2.0 * cfg.resolution)).astype(np.float32)
    grad_col = (sobel(risk_map, axis=1) / (2.0 * cfg.resolution)).astype(np.float32)
    sdf_grad_row = (sobel(sdf_hard, axis=0) / (2.0 * cfg.resolution)).astype(np.float32)
    sdf_grad_col = (sobel(sdf_hard, axis=1) / (2.0 * cfg.resolution)).astype(np.float32)
    return {
        "z2_labels": labels,
        "risk_map": risk_map,
        "hard_mask": hard_mask,
        "soft_mask": soft_mask,
        "low_mask": low_mask,
        "geom_occ": hard_mask.copy(),
        "observed_mask": observed_b.astype(np.uint8),
        "point_count": np.asarray(point_count, dtype=np.int32),
        "sdf_hard": sdf_hard,
        "grad_row": grad_row,
        "grad_col": grad_col,
        "sdf_grad_row": sdf_grad_row,
        "sdf_grad_col": sdf_grad_col,
    }


def corrupt_label_grid(
    labels: np.ndarray,
    observed: np.ndarray,
    probability: float,
    empirical_counts: Mapping[int, int],
    *,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float | int]]:
    """Corrupt exactly round(p*N) observed cells, with nested masks across p.

    A changed cell is sampled from the empirical observed-cell label
    distribution conditional on the replacement differing from its clean
    label.  The same scene seed is used for every p, so lower-p masks are
    strict subsets of higher-p masks.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    clean = np.asarray(labels, dtype=np.uint16)
    obs_flat = np.flatnonzero(np.asarray(observed, dtype=bool).ravel())
    n_change = int(round(float(probability) * len(obs_flat)))
    out = clean.copy()
    if n_change == 0:
        return out, {
            "observed_cells": int(len(obs_flat)),
            "changed_cells": 0,
            "changed_fraction": 0.0,
        }
    label_ids = np.asarray(sorted(int(k) for k, v in empirical_counts.items() if int(v) > 0), dtype=np.uint16)
    weights = np.asarray([float(empirical_counts[int(k)]) for k in label_ids], dtype=np.float64)
    if label_ids.size < 2:
        raise ValueError("At least two empirical labels are required for different-label corruption")
    rng = np.random.default_rng(seed)
    selected = obs_flat[rng.permutation(len(obs_flat))[:n_change]]
    clean_selected = clean.ravel()[selected]
    replacements = np.empty(n_change, dtype=np.uint16)
    for current in np.unique(clean_selected):
        idx = np.flatnonzero(clean_selected == current)
        allowed = label_ids != current
        probs = weights[allowed]
        probs /= probs.sum()
        replacements[idx] = rng.choice(label_ids[allowed], size=len(idx), replace=True, p=probs)
    out.ravel()[selected] = replacements
    changed = int(np.count_nonzero(out.ravel()[obs_flat] != clean.ravel()[obs_flat]))
    if changed != n_change:
        raise AssertionError(f"Expected {n_change} changed cells, got {changed}")
    return out, {
        "observed_cells": int(len(obs_flat)),
        "changed_cells": changed,
        "changed_fraction": float(changed / max(1, len(obs_flat))),
    }


def assert_maps_exact(actual: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray]) -> None:
    keys = (
        "z2_labels", "risk_map", "hard_mask", "soft_mask", "low_mask", "geom_occ",
        "observed_mask", "point_count", "sdf_hard", "grad_row", "grad_col",
        "sdf_grad_row", "sdf_grad_col",
    )
    for key in keys:
        if not np.array_equal(np.asarray(actual[key]), np.asarray(expected[key]), equal_nan=True):
            delta = np.max(np.abs(np.asarray(actual[key], dtype=float) - np.asarray(expected[key], dtype=float)))
            raise AssertionError(f"p=0 reconstruction mismatch for {key}; max_abs_delta={delta}")


def select_balanced_episodes(
    episodes: Sequence[dict], episodes_per_sequence_regime: int | None
) -> List[dict]:
    if episodes_per_sequence_regime is None:
        return list(episodes)
    counts: Counter[Tuple[str, str]] = Counter()
    selected: List[dict] = []
    for episode in episodes:
        key = (str(episode["sequence"]), str(episode["regime"]))
        if counts[key] < episodes_per_sequence_regime:
            selected.append(episode)
            counts[key] += 1
    return selected


def empirical_cell_counts(bev_root: Path, episodes: Sequence[dict]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for scene_path in sorted({str(ep["scene_path"]) for ep in episodes}):
        scene = torch.load(bev_root / scene_path, map_location="cpu", weights_only=False)
        maps = scene["maps"]
        values, nums = np.unique(
            np.asarray(maps["z2_labels"])[np.asarray(maps["observed_mask"], dtype=bool)],
            return_counts=True,
        )
        counts.update({int(v): int(n) for v, n in zip(values, nums)})
    return counts


def route_contexts_for_goals(
    maps: Mapping[str, np.ndarray],
    goals: Sequence[Tuple[int, int]],
    *,
    risk_weight: float,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """Compute all per-scene route contexts with two shared sparse graphs.

    This is algebraically identical to ``_route_context`` but avoids rebuilding
    the same graph once per episode.
    """
    blocked = np.asarray(maps["geom_occ"], dtype=bool)
    risk = np.asarray(maps["risk_map"])
    rows, cols = risk.shape
    rr, cc = np.indices((rows, cols))
    src_parts: List[np.ndarray] = []
    dst_parts: List[np.ndarray] = []
    step_parts: List[np.ndarray] = []
    for dr, dc in GRID_DIRS_8:
        nr = rr + dr
        nc = cc + dc
        valid = (
            (~blocked)
            & (nr >= 0) & (nr < rows)
            & (nc >= 0) & (nc < cols)
        )
        valid &= ~blocked[np.clip(nr, 0, rows - 1), np.clip(nc, 0, cols - 1)]
        src_parts.append((rr[valid] * cols + cc[valid]).astype(np.int64))
        dst_parts.append((nr[valid] * cols + nc[valid]).astype(np.int64))
        step_parts.append(np.full(int(valid.sum()), GRID_STEP[(dr, dc)], dtype=np.float64))
    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    steps = np.concatenate(step_parts)
    risk_at_src = risk.ravel()[src].astype(np.float64)
    shape = (rows * cols, rows * cols)
    geom_graph = coo_matrix((steps, (src, dst)), shape=shape).tocsr()
    risk_graph = coo_matrix(
        (steps * (1.0 + float(risk_weight) * risk_at_src), (src, dst)),
        shape=shape,
    ).tocsr()
    unique_goals = list(dict.fromkeys(goals))
    valid_goals = [
        goal for goal in unique_goals
        if 0 <= goal[0] < rows and 0 <= goal[1] < cols and not blocked[goal]
    ]
    out = {
        goal: {
            "geom_to_go": np.full((rows, cols), np.inf, dtype=np.float32),
            "risk_to_go": np.full((rows, cols), np.inf, dtype=np.float32),
        }
        for goal in unique_goals
    }
    if valid_goals:
        indices = np.asarray([r * cols + c for r, c in valid_goals], dtype=np.int64)
        geom_dist = np.atleast_2d(sparse_dijkstra(geom_graph, directed=True, indices=indices))
        risk_dist = np.atleast_2d(sparse_dijkstra(risk_graph, directed=True, indices=indices))
        for index, goal in enumerate(valid_goals):
            out[goal] = {
                "geom_to_go": np.asarray(geom_dist[index], dtype=np.float32).reshape(rows, cols),
                "risk_to_go": np.asarray(risk_dist[index], dtype=np.float32).reshape(rows, cols),
            }
    return out


def load_head(path: Path, device: str) -> Tuple[DirectionalForceHead, float, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    summary = checkpoint["summary"]
    config = summary["config"]
    model = DirectionalForceHead(
        int(checkpoint["in_dim"]),
        int(config["hidden"]),
        1 + len(np.asarray(checkpoint["dirs"])),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, float(config["activation_threshold"]), config


@torch.no_grad()
def predict_rows(
    model: DirectionalForceHead, rows: Sequence[dict], threshold: float, device: str, batch_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []
    p_noop_all: List[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        x = torch.as_tensor(
            np.stack([row["x"] for row in rows[start : start + batch_size]]),
            dtype=torch.float32,
            device=device,
        )
        probs = torch.softmax(model(x), dim=-1)
        active_prob, active_idx = torch.max(probs[:, 1:], dim=-1)
        score = active_prob - probs[:, 0]
        pred = torch.where(score >= threshold, active_idx + 1, torch.zeros_like(active_idx))
        predictions.append(pred.cpu().numpy())
        scores_all.append(score.cpu().numpy())
        p_noop_all.append(probs[:, 0].cpu().numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(scores_all),
        np.concatenate(p_noop_all),
    )


def metric_sufficient(rows: Iterable[dict]) -> Dict[str, float]:
    sums = defaultdict(float)
    for row in rows:
        regime = str(row["regime"])
        active = float(row["pred_class"]) > 0
        if float(row["clean_label"]) > 0:
            sums["car_den"] += 1
            sums["car_num"] += float(bool(row["correct_activation"]))
        if regime in ("R2", "R3"):
            sums["far_den"] += 1
            sums["far_num"] += float(active)
        sums["active_num"] += float(active)
        sums["active_den"] += 1
        if regime == "R1":
            sums["r1_num"] += float(active)
            sums["r1_den"] += 1
        if regime == "R2":
            sums["r2_num"] += float(active)
            sums["r2_den"] += 1
        for reg in ("R1", "R2", "R3"):
            if regime == reg:
                sums[f"{reg}_num"] += float(active)
                sums[f"{reg}_den"] += 1
    return dict(sums)


def metrics_from_sufficient(sums: Mapping[str, float]) -> Dict[str, float]:
    def rate(num: str, den: str) -> float:
        return float(sums.get(num, 0.0) / max(sums.get(den, 0.0), 1.0))

    car = rate("car_num", "car_den")
    far = rate("far_num", "far_den")
    r1 = rate("r1_num", "r1_den")
    r2 = rate("r2_num", "r2_den")
    return {
        "CAR": car,
        "FAR": far,
        "SR": float(r1 / max(r2, 1e-8)),
        "activation_rate": rate("active_num", "active_den"),
        "noop_rate": 1.0 - rate("active_num", "active_den"),
        "R1_activation_rate": r1,
        "R2_activation_rate": r2,
        "R3_activation_rate": rate("R3_num", "R3_den"),
        "n_samples": float(sums.get("active_den", 0.0)),
        "n_CAR_eligible": float(sums.get("car_den", 0.0)),
        "n_FAR_eligible": float(sums.get("far_den", 0.0)),
    }


def add_sufficient(parts: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    total = defaultdict(float)
    for part in parts:
        for key, value in part.items():
            total[key] += float(value)
    return dict(total)


def cluster_bootstrap(
    rows: Sequence[dict],
    levels: Sequence[float],
    *,
    reps: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """Paired episode-cluster bootstrap CIs for levels and deltas from p=0."""
    by_level_episode: Dict[float, Dict[str, dict]] = {}
    for level in levels:
        grouped: Dict[str, List[dict]] = defaultdict(list)
        for row in rows:
            if math.isclose(float(row["corruption_probability"]), float(level)):
                grouped[str(row["episode_uid"])].append(row)
        by_level_episode[float(level)] = {
            episode: metric_sufficient(ep_rows) for episode, ep_rows in grouped.items()
        }
    episode_ids = sorted(set.intersection(*(set(v) for v in by_level_episode.values())))
    if not episode_ids:
        raise RuntimeError("No paired episode clusters found")
    rng = np.random.default_rng(seed)
    metric_names = ("CAR", "FAR", "SR", "activation_rate", "noop_rate")
    boot: Dict[Tuple[float, str], List[float]] = defaultdict(list)
    delta_boot: Dict[Tuple[float, str], List[float]] = defaultdict(list)
    for _ in range(reps):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        metrics = {}
        for level in levels:
            total = add_sufficient(by_level_episode[float(level)][eid] for eid in sampled)
            metrics[float(level)] = metrics_from_sufficient(total)
            for name in metric_names:
                boot[(float(level), name)].append(metrics[float(level)][name])
        base = metrics[0.0]
        for level in levels:
            for name in metric_names:
                delta_boot[(float(level), name)].append(metrics[float(level)][name] - base[name])

    ci_rows: List[dict] = []
    delta_rows: List[dict] = []
    for level in levels:
        point = metrics_from_sufficient(add_sufficient(by_level_episode[float(level)].values()))
        for name in metric_names:
            values = np.asarray(boot[(float(level), name)])
            ci_rows.append({
                "corruption_probability": float(level),
                "metric": name,
                "estimate": point[name],
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_unit": "episode",
                "bootstrap_reps": reps,
            })
            d_values = np.asarray(delta_boot[(float(level), name)])
            delta_rows.append({
                "corruption_probability": float(level),
                "reference_probability": 0.0,
                "metric": name,
                "paired_delta": point[name] - metrics_from_sufficient(
                    add_sufficient(by_level_episode[0.0].values())
                )[name],
                "ci_low": float(np.quantile(d_values, 0.025)),
                "ci_high": float(np.quantile(d_values, 0.975)),
                "bootstrap_unit": "paired_episode",
                "bootstrap_reps": reps,
            })
    return ci_rows, delta_rows


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bev-root", type=Path, default=None)
    parser.add_argument("--pairs-root", type=Path, default=None)
    parser.add_argument("--checkpoint-pattern", default="runs/rellis_directional_routeaware_aw050_far020_{sequence}/best.pt")
    parser.add_argument("--ontology", type=Path, default=None)
    parser.add_argument("--mapping", default="main")
    parser.add_argument("--out", type=Path, default=ROOT / "rebuttal_experiments/results/exp7_semantic_corruption")
    parser.add_argument("--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS))
    parser.add_argument("--seed", type=int, default=27370)
    parser.add_argument("--bootstrap-seed", type=int, default=27371)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--episodes-per-sequence-regime", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.bev_root = args.bev_root or args.source_root / "cache/rellis_bev_all_seqbalanced_2500"
    args.pairs_root = args.pairs_root or args.source_root / "cache/rellis_pairs_all_seqbalanced_2500_loso"
    args.ontology = args.ontology or args.source_root / "grl_rellis/risk_ontology.yaml"
    levels = sorted(set(float(x) for x in args.levels))
    if 0.0 not in levels:
        raise ValueError("--levels must include 0.0 for paired robustness deltas")
    args.out.mkdir(parents=True, exist_ok=True)

    bev_manifest_path = args.bev_root / "manifest.json"
    pair_manifest_path = args.pairs_root / "manifest.json"
    bev_manifest = json.loads(bev_manifest_path.read_text())
    pair_manifest = json.loads(pair_manifest_path.read_text())
    episodes = select_balanced_episodes(
        pair_manifest["episodes"], args.episodes_per_sequence_regime
    )
    cfg = BevConfig(**bev_manifest["config"]["bev"])
    ontology = load_ontology(args.ontology, args.mapping)
    empirical_counts = empirical_cell_counts(args.bev_root, episodes)
    sequences = sorted({str(ep["sequence"]) for ep in episodes})

    checkpoint_paths = {
        sequence: args.source_root / args.checkpoint_pattern.format(sequence=sequence)
        for sequence in sequences
    }
    heads = {
        sequence: load_head(checkpoint_paths[sequence], args.device)
        for sequence in sequences
    }
    for sequence, (_, _, config) in heads.items():
        if not config.get("route_aware") or str(config.get("holdout_sequence")) != sequence:
            raise ValueError(f"Checkpoint for {sequence} is not its clean route-aware LOSO head")

    raw_rows: List[dict] = []
    corruption_rows: List[dict] = []
    episodes_by_scene: Dict[str, List[dict]] = defaultdict(list)
    for episode in episodes:
        episodes_by_scene[str(episode["scene_path"])].append(episode)

    for scene_number, scene_path in enumerate(sorted(episodes_by_scene), start=1):
        scene = torch.load(args.bev_root / scene_path, map_location="cpu", weights_only=False)
        clean_maps = scene["maps"]
        scene_id = str(scene["meta"]["scene_id"])
        observed = np.asarray(clean_maps["observed_mask"], dtype=bool)
        labels = np.asarray(clean_maps["z2_labels"], dtype=np.uint16)
        seed = stable_scene_seed(args.seed, scene_id)
        maps_by_level: Dict[float, Dict[str, np.ndarray]] = {}
        for level in levels:
            corrupted, stats = corrupt_label_grid(
                labels, observed, level, empirical_counts, seed=seed
            )
            rebuilt = maps_from_label_grid(
                corrupted, observed, clean_maps["point_count"], ontology, cfg
            )
            if level == 0.0:
                assert_maps_exact(rebuilt, clean_maps)
            maps_by_level[level] = rebuilt
            corruption_rows.append({
                "scene_id": scene_id,
                "sequence": str(scene["meta"]["sequence"]),
                "corruption_probability": level,
                "scene_seed": seed,
                **stats,
            })
        goals = [
            tuple(int(x) for x in episode["goal_rc"])
            for episode in episodes_by_scene[scene_path]
        ]
        risk_weight = float(
            heads[str(episodes_by_scene[scene_path][0]["sequence"])][2]["route_risk_weight"]
        )
        routes_by_level = {
            level: route_contexts_for_goals(
                maps_by_level[level], goals, risk_weight=risk_weight
            )
            for level in levels
        }

        for episode in episodes_by_scene[scene_path]:
            sequence = str(episode["sequence"])
            model, threshold, config = heads[sequence]
            path = _as_path(episode["stage1_path"])
            goal = tuple(int(x) for x in episode["goal_rc"])
            episode_uid = f"{sequence}:{episode['episode_id']}"
            rows_by_level: Dict[float, List[dict]] = {}
            for level in levels:
                maps = maps_by_level[level]
                route = routes_by_level[level][goal]
                built: List[dict] = []
                for idx in range(0, max(0, len(path) - 1), max(1, int(config["stride"]))):
                    row = _build_point(
                        maps,
                        path,
                        idx,
                        regime=str(episode["regime"]),
                        episode_id=episode_uid,
                        horizon_cells=int(config["horizon_cells"]),
                        long_horizon_cells=int(config["long_horizon_cells"]),
                        hard_margin_m=float(config["hard_margin_m"]),
                        improvement_margin=float(config["improvement_margin"]),
                        route=route,
                        route_max_ratio=float(config["route_max_ratio"]),
                    )
                    if row is not None:
                        row["path_index"] = idx
                        built.append(row)
                rows_by_level[level] = built

            clean_by_index = {int(row["path_index"]): row for row in rows_by_level[0.0]}
            paired_indices = sorted(set.intersection(*(
                {int(row["path_index"]) for row in rows_by_level[level]} for level in levels
            )))
            for level in levels:
                observed_rows = {
                    int(row["path_index"]): row for row in rows_by_level[level]
                }
                ordered = [observed_rows[idx] for idx in paired_indices]
                pred, scores, p_noop = predict_rows(
                    model, ordered, threshold, args.device, args.batch_size
                )
                for local_idx, path_index in enumerate(paired_indices):
                    clean = clean_by_index[path_index]
                    pred_class = int(pred[local_idx])
                    correct_activation = False
                    if pred_class > 0 and int(clean["label"]) > 0:
                        correct_activation = bool(
                            np.dot(DIRS_16[pred_class - 1], DIRS_16[int(clean["best_idx"])]) > 1e-3
                        )
                    raw_rows.append({
                        "episode_uid": episode_uid,
                        "episode_id": str(episode["episode_id"]),
                        "scene_id": scene_id,
                        "sequence": sequence,
                        "regime": str(episode["regime"]),
                        "path_index": path_index,
                        "corruption_probability": level,
                        "pred_class": pred_class,
                        "active": int(pred_class > 0),
                        "p_noop": float(p_noop[local_idx]),
                        "activation_score": float(scores[local_idx]),
                        "activation_threshold": threshold,
                        "clean_label": int(clean["label"]),
                        "clean_best_idx": int(clean["best_idx"]),
                        "clean_has_safe_alt": float(clean["has_safe_alt"]),
                        "correct_activation": int(correct_activation),
                        "observed_label": int(ordered[local_idx]["label"]),
                        "observed_has_safe_alt": float(ordered[local_idx]["has_safe_alt"]),
                    })
        if scene_number % 10 == 0 or scene_number == len(episodes_by_scene):
            print(f"Processed scenes {scene_number}/{len(episodes_by_scene)}", flush=True)

    ci_rows, delta_rows = cluster_bootstrap(
        raw_rows, levels, reps=args.bootstrap_reps, seed=args.bootstrap_seed
    )
    summary_rows: List[dict] = []
    for level in levels:
        level_rows = [r for r in raw_rows if float(r["corruption_probability"]) == level]
        summary_rows.append({
            "corruption_probability": level,
            **metrics_from_sufficient(metric_sufficient(level_rows)),
        })
    sequence_rows: List[dict] = []
    for sequence in sequences:
        for level in levels:
            selected_rows = [
                row for row in raw_rows
                if str(row["sequence"]) == sequence
                and float(row["corruption_probability"]) == level
            ]
            sequence_rows.append({
                "sequence": sequence,
                "corruption_probability": level,
                **metrics_from_sufficient(metric_sufficient(selected_rows)),
            })

    # The p=0 fold metrics must recover the validation metrics saved with each
    # frozen LOSO checkpoint. This detects feature/protocol drift.
    p0_fold_validation: Dict[str, dict] = {}
    for sequence in sequences:
        actual = next(
            row for row in sequence_rows
            if row["sequence"] == sequence and row["corruption_probability"] == 0.0
        )
        checkpoint = torch.load(checkpoint_paths[sequence], map_location="cpu", weights_only=False)
        expected = checkpoint["summary"]["val_metrics"]
        comparisons = {
            "CAR": (actual["CAR"], float(expected["correct_activation_rate"])),
            "FAR": (actual["FAR"], float(expected["false_activation_rate"])),
            "SR": (actual["SR"], float(expected["selectivity_ratio"])),
        }
        max_error = max(abs(a - b) for a, b in comparisons.values())
        if max_error > 1e-7:
            raise AssertionError(
                f"p=0 fold {sequence} does not recover checkpoint metrics; max_error={max_error}"
            )
        p0_fold_validation[sequence] = {
            "max_abs_metric_error": max_error,
            "passed": True,
        }

    write_csv(args.out / "raw_predictions.csv", raw_rows)
    write_csv(args.out / "corruption_audit.csv", corruption_rows)
    write_csv(args.out / "summary_metrics.csv", summary_rows)
    write_csv(args.out / "summary_by_sequence.csv", sequence_rows)
    write_csv(args.out / "cluster_bootstrap_ci.csv", ci_rows)
    write_csv(args.out / "paired_deltas_vs_clean.csv", delta_rows)
    label_rows = [
        {
            "label_id": label,
            "class_name": ontology.class_name(label),
            "observed_cell_count": count,
            "empirical_probability": count / sum(empirical_counts.values()),
        }
        for label, count in sorted(empirical_counts.items())
    ]
    write_csv(args.out / "empirical_label_distribution.csv", label_rows)

    provenance = {
        "experiment": "exp7_semantic_corruption",
        "corruption_definition": (
            "For each scene, exactly round(p*N) observed BEV cells are selected by a "
            "SHA-256-derived fixed scene seed. Their modal semantic ID is independently "
            "sampled from the evaluated scenes' empirical observed-cell distribution, "
            "conditioned on differing from the clean ID. Masks are nested across p."
        ),
        "ground_truth_definition": (
            "CAR/FAR eligibility and best direction come from the clean ontology map; "
            "only model inputs are recomputed from corrupted semantics."
        ),
        "metrics_scope": (
            "Pointwise directional-head selectivity only. No navigation rollout metrics "
            "are available or claimed."
        ),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "num_episodes": len(episodes),
        "num_scenes": len(episodes_by_scene),
        "counts_by_sequence_regime": dict(Counter(
            f"{ep['sequence']}:{ep['regime']}" for ep in episodes
        )),
        "hashes": {
            "bev_manifest_sha256": sha256_file(bev_manifest_path),
            "pair_manifest_sha256": sha256_file(pair_manifest_path),
            "ontology_sha256": sha256_file(args.ontology),
            **{
                f"checkpoint_{sequence}_sha256": sha256_file(path)
                for sequence, path in checkpoint_paths.items()
            },
        },
        "checkpoint_paths": {k: str(v) for k, v in checkpoint_paths.items()},
        "tests_run_inline": [
            "p=0 exact equality for all canonical map arrays on every evaluated scene",
            "changed-cell count equals round(p * observed cells) on every scene/level",
            "p=0 CAR/FAR/SR recovers every frozen checkpoint's saved holdout metrics",
        ],
        "p0_checkpoint_metric_validation": p0_fold_validation,
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))

    ci_lookup = {(float(r["corruption_probability"]), str(r["metric"])): r for r in ci_rows}
    delta_lookup = {(float(r["corruption_probability"]), str(r["metric"])): r for r in delta_rows}
    lines = [
        "# Experiment 7 — Semantic-label corruption robustness",
        "",
        f"Frozen clean LOSO route-aware heads were evaluated on **{len(episodes)} episodes** "
        f"from **{len(episodes_by_scene)} scenes**. This is a pointwise selectivity study; "
        "navigation rollout outcomes are not available and are not reported.",
        "",
        "## Results",
        "",
        "| Corruption | CAR (95% CI) | FAR (95% CI) | SR (95% CI) | Active | No-op |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        level = float(row["corruption_probability"])
        values = []
        for metric in ("CAR", "FAR", "SR"):
            ci = ci_lookup[(level, metric)]
            values.append(f"{row[metric]:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
        lines.append(
            f"| {100*level:.0f}% | {values[0]} | {values[1]} | {values[2]} | "
            f"{row['activation_rate']:.3f} | {row['noop_rate']:.3f} |"
        )
    lines += [
        "",
        "## Paired change from the clean map",
        "",
        "| Corruption | ΔCAR (95% CI) | ΔFAR (95% CI) | ΔSR (95% CI) |",
        "|---:|---:|---:|---:|",
    ]
    for level in levels:
        values = []
        for metric in ("CAR", "FAR", "SR"):
            row = delta_lookup[(level, metric)]
            values.append(
                f"{row['paired_delta']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
            )
        lines.append(f"| {100*level:.0f}% | {values[0]} | {values[1]} | {values[2]} |")
    lines += [
        "",
        "CIs are percentile cluster-bootstrap intervals (episode is the resampling unit); "
        "delta intervals use the same resampled episodes at every corruption level.",
        "",
        "## Finding",
        "",
        "Corruption makes the frozen gate progressively more conservative: activation "
        f"falls from {summary_rows[0]['activation_rate']:.3f} to "
        f"{summary_rows[-1]['activation_rate']:.3f}, while CAR falls from "
        f"{summary_rows[0]['CAR']:.3f} to {summary_rows[-1]['CAR']:.3f}. The lower FAR "
        "under corruption is therefore not evidence of improved robustness; it accompanies "
        "a large loss of required activations. SR also degrades. These pointwise results do "
        "not establish navigation success or safety under corruption.",
        "",
        "## Protocol",
        "",
        provenance["corruption_definition"],
        "",
        provenance["ground_truth_definition"],
        "",
        "All risk, hard-hazard, SDF, risk-gradient, and SDF-gradient fields are recomputed "
        "with the canonical `main` ontology after corruption. The controller weights and "
        "fold-specific clean calibration threshold remain frozen.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python rebuttal_experiments/exp7_semantic_corruption.py",
        "```",
        "",
        "See `provenance.json` for fixed seeds, input hashes, checkpoints, and full configuration.",
    ]
    (args.out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary_rows, indent=2))
    print(f"Wrote Experiment 7 artifacts to {args.out}")


if __name__ == "__main__":
    main()
