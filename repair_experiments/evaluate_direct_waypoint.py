#!/usr/bin/env python3
"""Validation-only evaluator for the direct-waypoint controller repair.

This CLI deliberately reuses the locked-manifest loading, rollout pairing,
artifact immutability, and metric definitions from :mod:`evaluate_v1`.  The
only semantic change is the study-arm mapping:

``direct_waypoint``
    Primary repaired arm.
``stateless_projected``
    Raw frame-wise feasibility gate, used only as the temporal transition
    diagnostic comparator.
``gate_off``
    Behavioral-effect comparator.
``geometry_only``
    Safety comparator.

The held-out test split is not exposed by this development evaluator.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Preserve the documented ``python repair_experiments/...`` invocation.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repair_experiments import evaluate_v1 as base_evaluator
from repair_experiments.evaluation_metrics import (
    add_paired_separation_fields as _base_add_paired_separation_fields,
    compute_preregistered_metrics as _base_compute_preregistered_metrics,
)
from repair_experiments.v1_controller import (
    DIRECT_WAYPOINT_CONTROLLER_VERSION,
)


DIRECT_WAYPOINT_MODES = (
    "direct_waypoint",
    "stateless_projected",
    "gate_off",
    "geometry_only",
)

# The v1 metric implementation is well tested.  Alias the new study arms onto
# its abstract primary/temporal-control names instead of duplicating formulas.
_TO_V1_MODE = {
    "direct_waypoint": "repaired",
    "stateless_projected": "stateless_projected",
    "gate_off": "gate_off",
    "geometry_only": "geometry_only",
}
_FROM_V1_MODE = {value: key for key, value in _TO_V1_MODE.items()}


def _alias_modes(
    rows: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        mode = str(row["mode"])
        if mode not in mapping:
            raise ValueError(
                f"unexpected study mode {mode!r}; expected "
                f"{tuple(mapping)}"
            )
        row["mode"] = mapping[mode]
        output.append(row)
    return output


def add_direct_waypoint_separation_fields(
    traces: Sequence[Mapping[str, Any]],
    *,
    horizon_steps: int,
    gsd: float,
) -> list[dict[str, Any]]:
    """Pair direct-waypoint trajectories with the gate-off control."""

    aliased = _alias_modes(traces, _TO_V1_MODE)
    paired = _base_add_paired_separation_fields(
        aliased,
        horizon_steps=horizon_steps,
        gsd=gsd,
    )
    return _alias_modes(paired, _FROM_V1_MODE)


def compute_direct_waypoint_metrics(
    episodes: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    *,
    static_anchor_event: str = "mud_onset",
    n_boot: int = 1000,
    seed: int = 27370,
) -> dict[str, Any]:
    """Compute frozen formulas with ``direct_waypoint`` as the primary arm."""

    metrics = _base_compute_preregistered_metrics(
        _alias_modes(episodes, _TO_V1_MODE),
        _alias_modes(traces, _TO_V1_MODE),
        static_anchor_event=static_anchor_event,
        n_boot=n_boot,
        seed=seed,
    )
    metrics["mode_semantics"] = {
        "primary": "direct_waypoint",
        "behavioral_effect_comparator": "gate_off",
        "safety_comparator": "geometry_only",
        "transition_diagnostic_comparator": "stateless_projected",
        "transition_diagnostic_interpretation": (
            "Direct-waypoint latch transitions relative to the raw "
            "frame-wise feasibility gate with projection; this is a "
            "diagnostic, not a new preregistered efficacy claim."
        ),
    }
    metrics["decision"].update(
        {
            "go_for_one_shot_test": False,
            "exploratory_development_validation": True,
            "reason": (
                "The direct-waypoint repair is a post-hoc development "
                "experiment; this validation-only CLI cannot authorize "
                "held-out test access."
            ),
        }
    )
    return metrics


def _install_direct_waypoint_study() -> None:
    """Configure the shared harness without changing its source module."""

    base_evaluator.REQUIRED_MODES = DIRECT_WAYPOINT_MODES
    base_evaluator.CONTROLLER_VERSION = DIRECT_WAYPOINT_CONTROLLER_VERSION
    base_evaluator.add_paired_separation_fields = (
        add_direct_waypoint_separation_fields
    )
    base_evaluator.compute_preregistered_metrics = (
        compute_direct_waypoint_metrics
    )


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    forbidden = {
        "--allow-sealed-test",
        "--frozen-config",
    }
    present = sorted(forbidden.intersection(arguments))
    if present:
        raise SystemExit(
            "Direct-waypoint development is validation-only; forbidden "
            f"argument(s): {', '.join(present)}"
        )
    requested_split = "validation"
    for index, argument in enumerate(arguments):
        if argument == "--split" and index + 1 < len(arguments):
            requested_split = arguments[index + 1]
        elif argument.startswith("--split="):
            requested_split = argument.split("=", 1)[1]
    if requested_split != "validation":
        raise SystemExit(
            "Direct-waypoint development is validation-only; "
            f"requested split: {requested_split!r}"
        )
    _install_direct_waypoint_study()
    original_splits = base_evaluator.SPLIT_NAMES
    base_evaluator.SPLIT_NAMES = ("validation",)
    try:
        base_evaluator.main(arguments)
    finally:
        base_evaluator.SPLIT_NAMES = original_splits


if __name__ == "__main__":
    main()
