#!/usr/bin/env python3
"""Leakage-safe validation harness for repaired-controller fixes 1--4.

The default is a one-item validation smoke run.  A complete validation run is
requested explicitly with ``--max-dynamic-items 0``.  Test manifests are
sealed unless both ``--allow-sealed-test`` and a frozen validation
configuration are supplied.
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
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

# ``python repair_experiments/evaluate_v1.py`` sets sys.path to the script
# directory, not the repository root.  Keep the documented direct CLI usable.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repair_experiments.evaluation_metrics import (
    REQUIRED_MODES,
    add_horizon_mechanism_fields,
    add_paired_separation_fields,
    build_episode_fields,
    compute_preregistered_metrics,
    stratified_summaries,
)
from repair_experiments.run_v1_dynamic import (
    DEFAULT_BEV_ROOT,
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_ROOT,
    HysteresisConfig,
    TemporalReleaseConfig,
    VelocityAwareSelectorConfig,
    WaypointLatchConfig,
    _load_model,
    rollout,
)
from repair_experiments.v1_controller import CONTROLLER_VERSION


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-repair-eval")
SPLIT_NAMES = ("validation", "test")
ROLLOUT_FREEZE_KEYS = (
    "dt",
    "max_steps",
    "event_fraction",
    "event_duration",
    "stage_lookahead_cells",
    "patch_size",
    "obstacle_patch_size",
    "robot_radius",
    "margin_factor",
    "d_hat_sdf",
    "primitive_count",
    "primitive_horizon_cells",
    "behavior_effect_horizon_seconds",
    "hard_margin_m",
    "hard_override_clearance_m",
    "cone_half_angle_degrees",
    "gradient_confidence_threshold",
    "low_confidence_fallback_policy",
    "lambda_active_threshold",
    "waypoint_distance_m",
    "waypoint_active_step_limit",
    "waypoint_cumulative_forward_limit_m",
    "waypoint_rearm_inactive_steps",
    "selector_prediction_steps",
    "selector_progress_min_m",
    "selector_swept_sample_spacing_m",
    "selector_goal_direction_cosine_min",
    "selector_velocity_direction_cosine_min",
    "temporal_wait_timeout_steps",
    "temporal_release_credit_steps",
    "waypoint_replan_interval_steps",
    "waypoint_reach_tolerance_m",
    "waypoint_minimum_hold_steps",
    "waypoint_maximum_hold_steps",
    "direct_lambda_floor",
    "on_improvement",
    "off_improvement",
    "on_material_trigger",
    "off_material_trigger",
    "on_persistence_steps",
    "off_persistence_steps",
    "minimum_dwell_steps",
    "hard_violation_penalty",
    "bootstrap_samples",
    "static_anchor_event",
    "seed",
    "device",
)


class SealedTestError(RuntimeError):
    """Raised before opening a sealed manifest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_split_paths(
    split_root: Path,
    split: str,
    *,
    allow_sealed_test: bool,
    frozen_config: Path | None,
) -> tuple[Path, Path, Path]:
    """Resolve locked manifests, rejecting sealed paths before reading them."""

    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    static_path = split_root / f"{split}_static.json"
    dynamic_path = split_root / f"{split}_dynamic.json"
    lock_path = split_root / "SPLIT_LOCK.json"
    if split == "test":
        if not allow_sealed_test:
            raise SealedTestError(
                "The test split is sealed. Pass --allow-sealed-test only after "
                "validation passes and the configuration is frozen."
            )
        if frozen_config is None:
            raise SealedTestError(
                "A test run also requires --frozen-config from the accepted "
                "validation run."
            )
    return static_path, dynamic_path, lock_path


def _load_locked_manifests(
    *,
    split_root: Path,
    split: str,
    allow_sealed_test: bool,
    frozen_config: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    static_path, dynamic_path, lock_path = resolve_split_paths(
        split_root,
        split,
        allow_sealed_test=allow_sealed_test,
        frozen_config=frozen_config,
    )
    lock = json.loads(lock_path.read_text())
    expected = lock["output_manifest_hashes"]
    hashes = {
        static_path.name: _sha256(static_path),
        dynamic_path.name: _sha256(dynamic_path),
        lock_path.name: _sha256(lock_path),
    }
    for path in (static_path, dynamic_path):
        if hashes[path.name] != expected[path.name]:
            raise RuntimeError(
                f"{path} does not match SPLIT_LOCK.json; refusing evaluation"
            )
    static = json.loads(static_path.read_text())
    dynamic = json.loads(dynamic_path.read_text())
    expected_sequence = "00003" if split == "validation" else "00004"
    if set(map(str, static["sequences"])) != {expected_sequence}:
        raise RuntimeError(
            f"{split} manifest has unexpected sequences {static['sequences']}"
        )
    if str(dynamic["split_name"]) != split:
        raise RuntimeError("static/dynamic split mismatch")
    return static, dynamic, lock, hashes


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(
                json.dumps(_json_safe(dict(row)), sort_keys=True) + "\n"
            )


def _write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("x", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json"
    }


def _validate_frozen_config(
    frozen_path: Path | None,
    *,
    checkpoint: Path,
) -> dict[str, Any] | None:
    if frozen_path is None:
        return None
    frozen = json.loads(frozen_path.read_text())
    expected = frozen.get("checkpoint_sha256")
    actual = _sha256(checkpoint)
    if expected is None or expected != actual:
        raise RuntimeError(
            "The requested checkpoint does not match the frozen validation "
            "configuration."
        )
    decision = frozen.get("validation_decision", {})
    if not bool(decision.get("go_for_one_shot_test", False)):
        raise RuntimeError(
            "The frozen configuration does not record a validation go decision."
        )
    return frozen


def build_frozen_configuration(
    *,
    config: Mapping[str, Any],
    validation_decision: Mapping[str, Any],
    rollout_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the single handoff artifact required by a one-shot test run."""

    return {
        "schema_version": "repaired-frozen-configuration-v1",
        "controller_version": config["controller_version"],
        "checkpoint": config["checkpoint"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "split_manifest_hashes": config["split_manifest_hashes"],
        "bev_manifest_sha256": config["bev_manifest_sha256"],
        "modes": config["modes"],
        "arguments": {
            key: rollout_arguments[key] for key in ROLLOUT_FREEZE_KEYS
        },
        "validation_decision": dict(validation_decision),
    }


def stratified_dynamic_limit(
    items: Sequence[Mapping[str, Any]], limit: int
) -> list[Mapping[str, Any]]:
    """Take a deterministic round-robin sample over event × regime strata."""

    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if limit == 0 or limit >= len(items):
        return list(items)
    strata: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        strata[(str(item["event_type"]), str(item["regime"]))].append(item)
    keys = sorted(strata)
    offsets = {key: 0 for key in keys}
    selected: list[Mapping[str, Any]] = []
    while len(selected) < limit:
        made_progress = False
        for key in keys:
            index = offsets[key]
            if index >= len(strata[key]):
                continue
            selected.append(strata[key][index])
            offsets[key] += 1
            made_progress = True
            if len(selected) == limit:
                break
        if not made_progress:  # Defensive; limit is bounded by len(items).
            break
    return selected


def behavior_effect_horizon_steps(seconds: float, dt: float) -> int:
    """Convert the fixed physical-time effect horizon to control steps."""

    if seconds <= 0.0:
        raise ValueError("behavior-effect horizon must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return max(1, int(round(seconds / dt)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bev-root", type=Path, default=DEFAULT_BEV_ROOT)
    parser.add_argument(
        "--split-root", type=Path, default=Path("repair_experiments/splits")
    )
    parser.add_argument("--split", choices=SPLIT_NAMES, default="validation")
    parser.add_argument("--allow-sealed-test", action="store_true")
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("repair_experiments/results/v1_validation_smoke"),
    )
    parser.add_argument(
        "--max-dynamic-items",
        type=int,
        default=1,
        help="Default 1 is a smoke run; 0 means the entire selected split.",
    )
    parser.add_argument(
        "--event-types",
        nargs="+",
        help="Optional event subset for smoke/debugging; full validation uses all.",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=REQUIRED_MODES, default=list(REQUIRED_MODES)
    )
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--event-fraction", type=float, default=0.38)
    parser.add_argument("--event-duration", type=int, default=80)
    parser.add_argument("--stage-lookahead-cells", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--obstacle-patch-size", type=int, default=64)
    parser.add_argument("--robot-radius", type=float, default=1.5)
    parser.add_argument("--margin-factor", type=float, default=0.5)
    parser.add_argument("--d-hat-sdf", type=float, default=3.0)
    parser.add_argument("--primitive-count", type=int, default=16)
    parser.add_argument("--primitive-horizon-cells", type=int, default=12)
    parser.add_argument(
        "--behavior-effect-horizon-seconds",
        type=float,
        default=1.0,
        help=(
            "Physical-time horizon for repaired-vs-gate-off endpoint "
            "separation; independent of the primitive mechanism horizon."
        ),
    )
    parser.add_argument("--hard-margin-m", type=float, default=1.0)
    parser.add_argument("--hard-override-clearance-m", type=float, default=0.5)
    parser.add_argument("--cone-half-angle-degrees", type=float, default=35.0)
    parser.add_argument(
        "--gradient-confidence-threshold", type=float, default=1e-3
    )
    parser.add_argument(
        "--low-confidence-fallback-policy",
        choices=("selected_axis", "zero"),
        default="selected_axis",
    )
    parser.add_argument(
        "--lambda-active-threshold",
        type=float,
        default=None,
        help=(
            "Override checkpoint repair_calibration.lambda_active_threshold; "
            "historical checkpoints default to zero."
        ),
    )
    parser.add_argument("--waypoint-distance-m", type=float, default=1.0)
    parser.add_argument("--waypoint-active-step-limit", type=int, default=10)
    parser.add_argument(
        "--waypoint-cumulative-forward-limit-m", type=float, default=3.0
    )
    parser.add_argument(
        "--waypoint-rearm-inactive-steps", type=int, default=5
    )
    parser.add_argument("--selector-prediction-steps", type=int, default=6)
    parser.add_argument("--selector-progress-min-m", type=float, default=0.1)
    parser.add_argument(
        "--selector-swept-sample-spacing-m", type=float, default=0.25
    )
    parser.add_argument(
        "--selector-goal-direction-cosine-min", type=float, default=0.25
    )
    parser.add_argument(
        "--selector-velocity-direction-cosine-min",
        type=float,
        default=0.0,
    )
    parser.add_argument("--temporal-wait-timeout-steps", type=int, default=12)
    parser.add_argument(
        "--temporal-release-credit-steps", type=int, default=12
    )
    parser.add_argument(
        "--waypoint-replan-interval-steps", type=int, default=10
    )
    parser.add_argument(
        "--waypoint-reach-tolerance-m", type=float, default=0.25
    )
    parser.add_argument(
        "--waypoint-minimum-hold-steps", type=int, default=50
    )
    parser.add_argument(
        "--waypoint-maximum-hold-steps", type=int, default=100
    )
    parser.add_argument("--direct-lambda-floor", type=float, default=1.5)
    parser.add_argument("--on-improvement", type=float, default=0.05)
    parser.add_argument("--off-improvement", type=float, default=0.025)
    parser.add_argument("--on-material-trigger", type=float, default=0.45)
    parser.add_argument("--off-material-trigger", type=float, default=0.35)
    parser.add_argument("--on-persistence-steps", type=int, default=3)
    parser.add_argument("--off-persistence-steps", type=int, default=2)
    parser.add_argument("--minimum-dwell-steps", type=int, default=5)
    parser.add_argument("--hard-violation-penalty", type=float, default=2.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--static-anchor-event", default="mud_onset")
    parser.add_argument("--seed", type=int, default=27370)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.max_dynamic_items < 0:
        parser.error("--max-dynamic-items must be nonnegative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.split == "test" and (
        args.max_dynamic_items != 0
        or args.event_types
        or set(args.modes) != set(REQUIRED_MODES)
    ):
        raise SealedTestError(
            "The one-shot test must evaluate the complete event manifest with "
            "all four preregistered modes (--max-dynamic-items 0)."
        )
    static, dynamic, split_lock, split_hashes = _load_locked_manifests(
        split_root=args.split_root,
        split=args.split,
        allow_sealed_test=args.allow_sealed_test,
        frozen_config=args.frozen_config,
    )
    frozen = _validate_frozen_config(
        args.frozen_config, checkpoint=args.checkpoint
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.source_root))
    sys.path.insert(0, str(args.source_root / "exp-rellis"))
    from grl_rellis import BevConfig
    from grl_rellis.dyn_events import MAIN_EVENT_TYPES, make_event_spec

    requested_events = (
        set(args.event_types) if args.event_types else set(MAIN_EVENT_TYPES)
    )
    unknown_events = requested_events - set(MAIN_EVENT_TYPES)
    if unknown_events:
        raise ValueError(f"unknown event types: {sorted(unknown_events)}")
    eligible_items = [
        item
        for item in dynamic["items"]
        if str(item["event_type"]) in requested_events
    ]
    items = stratified_dynamic_limit(
        eligible_items, args.max_dynamic_items
    )
    selected_strata: dict[str, int] = defaultdict(int)
    for item in items:
        selected_strata[
            f"{item['event_type']}|{item['regime']}"
        ] += 1

    episode_index = {
        str(episode["episode_uid"]): episode
        for episode in static["episodes"]
    }
    model, checkpoint_cfg = _load_model(
        args.checkpoint, args.source_root, args.device
    )
    checkpoint_payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    repair_calibration = dict(
        checkpoint_payload.get("repair_calibration", {})
    )
    checkpoint_lambda_threshold = float(
        repair_calibration.get("lambda_active_threshold", 0.0)
    )
    lambda_active_threshold = (
        float(args.lambda_active_threshold)
        if args.lambda_active_threshold is not None
        else checkpoint_lambda_threshold
    )
    if lambda_active_threshold < 0.0:
        raise ValueError("lambda_active_threshold must be nonnegative")
    effect_horizon_steps = behavior_effect_horizon_steps(
        args.behavior_effect_horizon_seconds, args.dt
    )
    effective_arguments = {
        key: (
            lambda_active_threshold
            if key == "lambda_active_threshold"
            else getattr(args, key)
        )
        for key in ROLLOUT_FREEZE_KEYS
    }
    if frozen is not None:
        frozen_args = frozen.get("arguments", {})
        mismatches = {
            key: (frozen_args.get(key), effective_arguments[key])
            for key in ROLLOUT_FREEZE_KEYS
            if frozen_args.get(key) != effective_arguments[key]
        }
        if mismatches:
            raise RuntimeError(
                f"test configuration differs from frozen validation: {mismatches}"
            )
        if list(frozen.get("modes", [])) != list(args.modes):
            raise RuntimeError("test mode order differs from frozen validation")
    bev_manifest_path = args.bev_root / "manifest.json"
    bev_manifest = json.loads(bev_manifest_path.read_text())
    gsd = float(BevConfig(**bev_manifest["config"]["bev"]).resolution)
    gate_config = HysteresisConfig(
        on_improvement=args.on_improvement,
        off_improvement=args.off_improvement,
        on_material_trigger=args.on_material_trigger,
        off_material_trigger=args.off_material_trigger,
        on_persistence_steps=args.on_persistence_steps,
        off_persistence_steps=args.off_persistence_steps,
        minimum_dwell_steps=args.minimum_dwell_steps,
    )
    config = {
        "schema_version": "repaired-evaluation-v1",
        "controller_version": CONTROLLER_VERSION,
        "split": args.split,
        "sealed_test_authorized": bool(args.allow_sealed_test),
        "frozen_config_path": str(args.frozen_config) if args.frozen_config else None,
        "frozen_config_sha256": (
            _sha256(args.frozen_config) if args.frozen_config else None
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_cfg": checkpoint_cfg,
        "repair_calibration": repair_calibration,
        "lambda_active_threshold": {
            "effective": lambda_active_threshold,
            "source": (
                "cli_override"
                if args.lambda_active_threshold is not None
                else (
                    "checkpoint_repair_calibration"
                    if "lambda_active_threshold" in repair_calibration
                    else "historical_zero_fallback"
                )
            ),
        },
        "same_checkpoint_all_modes": True,
        "same_event_seed_all_modes": True,
        "projected_force_semantics": (
            "unit direction; learned lambda_soft is force magnitude"
        ),
        "modes": list(args.modes),
        "item_selection": {
            "method": "deterministic_round_robin_event_type_x_regime",
            "eligible_items": len(eligible_items),
            "selected_items": len(items),
            "counts_by_event_type_and_regime": dict(
                sorted(selected_strata.items())
            ),
        },
        "evaluation_horizons": {
            "primitive_mechanism_steps": args.primitive_horizon_cells,
            "behavioral_effect_seconds": args.behavior_effect_horizon_seconds,
            "behavioral_effect_steps": effect_horizon_steps,
            "control_dt_seconds": args.dt,
            "successful_terminal_policy": "absorbing_terminal_state",
            "failed_incomplete_window_policy": "zero_separation_failure",
        },
        "split_manifest_hashes": split_hashes,
        "split_lock_schema": split_lock["schema_version"],
        "bev_manifest": str(bev_manifest_path),
        "bev_manifest_sha256": _sha256(bev_manifest_path),
        "hysteresis": asdict(gate_config),
        "direct_waypoint": {
            "distance_m": args.waypoint_distance_m,
            "active_step_limit": args.waypoint_active_step_limit,
            "cumulative_forward_limit_m": (
                args.waypoint_cumulative_forward_limit_m
            ),
            "rearm_inactive_steps": args.waypoint_rearm_inactive_steps,
            "velocity_selector": {
                "prediction_steps": args.selector_prediction_steps,
                "progress_min_m": args.selector_progress_min_m,
                "hard_margin_m": args.hard_margin_m,
                "swept_sample_spacing_m": (
                    args.selector_swept_sample_spacing_m
                ),
                "goal_direction_cosine_min": (
                    args.selector_goal_direction_cosine_min
                ),
                "velocity_direction_cosine_min": (
                    args.selector_velocity_direction_cosine_min
                ),
                "ranking": (
                    "progress_then_clearance_then_risk_then_velocity"
                ),
                "map_policy": "current_frozen_map_only",
            },
            "temporal_release_rule": {
                "wait_timeout_steps": args.temporal_wait_timeout_steps,
                "release_credit_steps": args.temporal_release_credit_steps,
            },
            "rolling_target": True,
            "fixed_direction_per_activation": False,
            "execution": "one real step then replan from actual state",
            "replanning_transition_policy": (
                "safe active-to-active direction changes are not transitions; "
                "no-safe or hard-override steps deactivate actual soft control"
            ),
            "legacy_non_driving_fields": [
                "replan_interval_steps",
                "reach_tolerance_m",
                "minimum_hold_steps",
                "maximum_hold_steps",
            ],
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    if args.out.exists():
        raise FileExistsError(
            f"{args.out} already exists; immutable runs are never overwritten"
        )
    partial = args.out.with_name(args.out.name + ".partial")
    if partial.exists():
        raise FileExistsError(
            f"{partial} already exists from another/incomplete run"
        )
    partial.mkdir(parents=True)
    _write_json_new(partial / "config.json", config)

    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    scene_cache: dict[str, Mapping[str, Any]] = {}
    for item_index, item in enumerate(items):
        episode = episode_index[str(item["base_episode_uid"])]
        scene_path = str(episode["scene_path"])
        if scene_path not in scene_cache:
            scene_cache[scene_path] = torch.load(
                args.bev_root / scene_path,
                map_location="cpu",
                weights_only=False,
            )
        base_maps = scene_cache[scene_path]["maps"]
        spec = make_event_spec(
            str(item["event_type"]),
            episode["stage1_path"],
            episode["risk_path"],
            episode["goal_rc"],
            event_fraction=args.event_fraction,
            duration=args.event_duration,
        )
        for mode in args.modes:
            metric, trace = rollout(
                mode=mode,
                model=model,
                base_maps=base_maps,
                spec=spec,
                episode=episode,
                source_root=args.source_root,
                device=args.device,
                gsd=gsd,
                max_steps=args.max_steps,
                dt=args.dt,
                stage_lookahead_cells=args.stage_lookahead_cells,
                patch_size=args.patch_size,
                obstacle_patch_size=args.obstacle_patch_size,
                robot_radius=args.robot_radius,
                margin_factor=args.margin_factor,
                d_hat_sdf=args.d_hat_sdf,
                primitive_count=args.primitive_count,
                primitive_horizon_cells=args.primitive_horizon_cells,
                hard_margin_m=args.hard_margin_m,
                hard_override_clearance_m=args.hard_override_clearance_m,
                cone_half_angle_degrees=args.cone_half_angle_degrees,
                gradient_confidence_threshold=args.gradient_confidence_threshold,
                low_confidence_fallback_policy=(
                    args.low_confidence_fallback_policy
                ),
                lambda_active_threshold=lambda_active_threshold,
                waypoint_config=WaypointLatchConfig(
                    distance_m=args.waypoint_distance_m,
                    active_step_limit=args.waypoint_active_step_limit,
                    cumulative_forward_limit_m=(
                        args.waypoint_cumulative_forward_limit_m
                    ),
                    rearm_inactive_steps=args.waypoint_rearm_inactive_steps,
                    replan_interval_steps=(
                        args.waypoint_replan_interval_steps
                    ),
                    reach_tolerance_m=args.waypoint_reach_tolerance_m,
                    minimum_hold_steps=args.waypoint_minimum_hold_steps,
                    maximum_hold_steps=args.waypoint_maximum_hold_steps,
                ),
                temporal_config=TemporalReleaseConfig(
                    wait_timeout_steps=args.temporal_wait_timeout_steps,
                    release_credit_steps=args.temporal_release_credit_steps,
                ),
                selector_config=VelocityAwareSelectorConfig(
                    prediction_steps=args.selector_prediction_steps,
                    progress_min_m=args.selector_progress_min_m,
                    hard_margin_m=args.hard_margin_m,
                    swept_sample_spacing_m=(
                        args.selector_swept_sample_spacing_m
                    ),
                    goal_direction_cosine_min=(
                        args.selector_goal_direction_cosine_min
                    ),
                    velocity_direction_cosine_min=(
                        args.selector_velocity_direction_cosine_min
                    ),
                ),
                direct_lambda_floor=args.direct_lambda_floor,
                gate_config=gate_config,
                hard_violation_penalty=args.hard_violation_penalty,
                seed=int(item["event_seed"]),
            )
            common = {
                "dynamic_episode_uid": str(item["dynamic_episode_uid"]),
                "event_seed": int(item["event_seed"]),
                "event_step": int(spec.event_step),
                "opening_step": int(spec.event_step + spec.open_delay),
                "evaluation_item_index": item_index,
            }
            episode_rows.append({**metric, **common})
            step_rows.extend(
                {
                    **row,
                    **common,
                    "rollout_success": int(metric["success"]),
                    "rollout_total_steps": int(metric["steps"]),
                }
                for row in trace
            )
            print(
                f"{item['dynamic_episode_uid']} {mode}: "
                f"success={metric['success']} steps={metric['steps']}",
                flush=True,
            )

    for row in step_rows:
        after_x = float(row["center_soft_x_after"])
        after_y = float(row["center_soft_y_after"])
        center_norm = (
            float(math.hypot(after_x, after_y))
            if math.isfinite(after_x) and math.isfinite(after_y)
            else float("nan")
        )
        row["center_soft_direction_norm_after"] = center_norm
        row["unit_soft_direction_when_projected"] = int(
            int(row["projection_applied"])
            and math.isfinite(center_norm)
            and abs(center_norm - 1.0) <= 1e-5
        )
    step_rows = add_horizon_mechanism_fields(
        step_rows,
        horizon_steps=args.primitive_horizon_cells,
        hard_margin_m=args.hard_margin_m,
    )
    step_rows = add_paired_separation_fields(
        step_rows,
        horizon_steps=effect_horizon_steps,
        gsd=gsd,
    )
    episode_rows = build_episode_fields(
        episode_rows, step_rows, max_steps=args.max_steps
    )
    metrics = compute_preregistered_metrics(
        episode_rows,
        step_rows,
        static_anchor_event=args.static_anchor_event,
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )
    summaries = stratified_summaries(episode_rows)
    complete_selected_split_run = bool(
        args.max_dynamic_items == 0
        and not args.event_types
        and set(args.modes) == set(REQUIRED_MODES)
    )
    complete_primary_run = bool(
        args.split == "validation" and complete_selected_split_run
    )
    if complete_primary_run:
        reported_decision = metrics["decision"]
    elif args.split == "test" and complete_selected_split_run:
        reported_decision = {
            **metrics["decision"],
            "go_for_one_shot_test": False,
            "reason": (
                "This is the one-shot held-out result; it cannot authorize a "
                "new tuning/test cycle."
            ),
        }
    else:
        reported_decision = {
            **metrics["decision"],
            "go_for_one_shot_test": False,
            "reason": "Smoke/subset runs cannot authorize held-out test access.",
        }
    run_summary = {
        "schema_version": "repaired-evaluation-v1",
        "split": args.split,
        "complete_primary_run": complete_primary_run,
        "num_dynamic_items": len(items),
        "num_episode_rows": len(episode_rows),
        "num_step_rows": len(step_rows),
        "evaluation_horizons": config["evaluation_horizons"],
        "preregistered_metrics": metrics,
        "stratified": summaries,
        "validation_decision": reported_decision,
    }
    _write_csv_new(partial / "per_episode.csv", episode_rows)
    _write_jsonl_new(partial / "per_episode.jsonl", episode_rows)
    _write_csv_new(partial / "per_step.csv", step_rows)
    _write_jsonl_new(partial / "per_step.jsonl", step_rows)
    _write_json_new(partial / "metrics.json", run_summary)

    if complete_primary_run:
        frozen_output = build_frozen_configuration(
            config=config,
            validation_decision=run_summary["validation_decision"],
            rollout_arguments=effective_arguments,
        )
        _write_json_new(
            partial / "frozen_configuration.json", frozen_output
        )

    manifest = {
        "schema_version": "immutable-artifacts-v1",
        "files": _artifact_hashes(partial),
    }
    _write_json_new(partial / "ARTIFACT_MANIFEST.json", manifest)
    partial.rename(args.out)
    print(json.dumps(_json_safe(run_summary["validation_decision"]), indent=2))
    print(f"Immutable outputs: {args.out.resolve()}")


if __name__ == "__main__":
    main()
