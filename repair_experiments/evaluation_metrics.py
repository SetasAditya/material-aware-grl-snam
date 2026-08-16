"""Preregistered metrics for the repaired-controller validation study.

This module is deliberately independent of model loading and RELLIS files so
that the metric definitions can be unit tested before validation is run.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


PRIMARY_MODE = "repaired"
STATELESS_MODE = "stateless_projected"
GATE_OFF_MODE = "gate_off"
GEOMETRY_MODE = "geometry_only"
REQUIRED_MODES = (
    PRIMARY_MODE,
    STATELESS_MODE,
    GATE_OFF_MODE,
    GEOMETRY_MODE,
)
DELAYED_EVENT = "delayed_required_escape"


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _mean(values: Iterable[Any]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _median(values: Iterable[Any]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return float(np.median(finite)) if finite else float("nan")


def add_horizon_mechanism_fields(
    traces: Sequence[Mapping[str, Any]],
    *,
    horizon_steps: int,
    hard_margin_m: float,
    risk_tolerance: float = 1e-6,
) -> list[dict[str, Any]]:
    """Recompute alignment, clearance, and realized risk from execution.

    A decision at step ``t`` is evaluated using the endpoint and samples from
    ``t`` through ``t + horizon_steps - 1``.  Incomplete windows are retained
    in the output but marked unevaluable, so they cannot silently enter a
    successful numerator.
    """

    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in traces:
        row = dict(raw)
        identity = str(
            row.get("dynamic_episode_uid")
            or f"{row.get('episode_uid')}:{row.get('event_type')}"
        )
        grouped[(identity, str(row["mode"]))].append(row)

    output: list[dict[str, Any]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
        for index, row in enumerate(rows):
            row["mechanism_window_complete"] = 0
            row["horizon_endpoint_alignment"] = float("nan")
            row["horizon_clearance_retained"] = float("nan")
            row["horizon_actual_risk_improvement"] = float("nan")
            row["horizon_predicted_risk_realized"] = float("nan")
            end_index = index + horizon_steps - 1
            if end_index >= len(rows):
                output.append(row)
                continue
            window = rows[index : end_index + 1]
            expected_steps = list(
                range(int(row["step"]), int(row["step"]) + horizon_steps)
            )
            if [int(item["step"]) for item in window] != expected_steps:
                output.append(row)
                continue

            start = np.asarray(
                [float(row["position_x"]), float(row["position_y"])],
                dtype=np.float64,
            )
            endpoint = np.asarray(
                [
                    float(window[-1]["next_x"]),
                    float(window[-1]["next_y"]),
                ],
                dtype=np.float64,
            )
            direction = np.asarray(
                [
                    float(row["selected_direction_col"]),
                    float(row["selected_direction_row"]),
                ],
                dtype=np.float64,
            )
            displacement = endpoint - start
            denominator = float(
                np.linalg.norm(displacement) * np.linalg.norm(direction)
            )
            alignment = (
                float(np.dot(displacement, direction) / denominator)
                if denominator > 1e-12
                else 0.0
            )
            clearance_ok = all(
                float(item["current_sdf_m"]) >= hard_margin_m
                and not bool(int(item["current_hard"]))
                and not bool(int(item["hard_contact"]))
                for item in window
            )
            actual_risk = _mean(item["risk"] for item in window)
            nominal_risk = float(row["nominal_primitive_risk"])
            best_risk = float(row["best_primitive_risk"])
            actual_improvement = nominal_risk - actual_risk
            # The primitive predicts the best-ray mean risk.  A realization
            # succeeds only when the executed window is no riskier than that
            # prediction (up to fixed numerical tolerance).
            realized = bool(
                _finite(actual_risk)
                and _finite(best_risk)
                and actual_risk <= best_risk + risk_tolerance
            )
            row["mechanism_window_complete"] = 1
            row["horizon_endpoint_alignment"] = alignment
            row["horizon_clearance_retained"] = int(clearance_ok)
            row["horizon_actual_risk_improvement"] = actual_improvement
            row["horizon_predicted_risk_realized"] = int(realized)
            output.append(row)
    return output


def add_paired_separation_fields(
    traces: Sequence[Mapping[str, Any]],
    *,
    horizon_steps: int,
    gsd: float,
) -> list[dict[str, Any]]:
    """Attach repaired-vs-gate-off endpoint separation to repaired rows.

    Successful rollouts are absorbing after their recorded terminal state.
    If either arm ends unsuccessfully before the requested horizon, the paired
    window remains in the denominator with zero separation.
    """

    output = [dict(row) for row in traces]
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in output:
        identity = str(
            row.get("dynamic_episode_uid")
            or f"{row.get('episode_uid')}:{row.get('event_type')}"
        )
        grouped[(identity, str(row["mode"]))][int(row["step"])] = row
        row["paired_gate_state_differs"] = 0
        row["paired_endpoint_separation_m"] = float("nan")
        row["paired_behavior_horizon_complete"] = 0
        row["paired_endpoint_used_absorbing_terminal"] = 0
        row["paired_incomplete_failure_imputed"] = 0

    identities = {identity for identity, _ in grouped}
    for identity in identities:
        repaired = grouped.get((identity, PRIMARY_MODE), {})
        gate_off = grouped.get((identity, GATE_OFF_MODE), {})

        def horizon_endpoint(
            rows: Mapping[int, Mapping[str, Any]], target: int
        ) -> tuple[np.ndarray | None, bool]:
            exact = rows.get(target)
            if exact is not None:
                return np.asarray(
                    [float(exact["next_x"]), float(exact["next_y"])]
                ), False
            if not rows:
                return None, False
            final_step = max(rows)
            final = rows[final_step]
            if target > final_step and bool(int(final.get("rollout_success", 0))):
                return np.asarray(
                    [float(final["next_x"]), float(final["next_y"])]
                ), True
            return None, False

        for step, row in repaired.items():
            other = gate_off.get(step)
            if other is None:
                continue
            differs = int(row["effective_gate_active"]) != int(
                other["effective_gate_active"]
            )
            row["paired_gate_state_differs"] = int(differs)
            if differs:
                target = step + horizon_steps - 1
                rep_xy, rep_absorbing = horizon_endpoint(repaired, target)
                off_xy, off_absorbing = horizon_endpoint(gate_off, target)
                if rep_xy is None or off_xy is None:
                    # Failed/timed-out incomplete windows remain explicit
                    # failures in the preregistered denominator.
                    row["paired_endpoint_separation_m"] = 0.0
                    row["paired_incomplete_failure_imputed"] = 1
                else:
                    row["paired_behavior_horizon_complete"] = 1
                    row["paired_endpoint_used_absorbing_terminal"] = int(
                        rep_absorbing or off_absorbing
                    )
                    row["paired_endpoint_separation_m"] = float(
                        gsd * np.linalg.norm(rep_xy - off_xy)
                    )
    return output


def build_episode_fields(
    episodes: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    *,
    max_steps: int,
) -> list[dict[str, Any]]:
    """Add episode-level behavioral fields while retaining every rollout."""

    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in traces:
        identity = str(
            row.get("dynamic_episode_uid")
            or f"{row.get('episode_uid')}:{row.get('event_type')}"
        )
        by_key[(identity, str(row["mode"]))].append(row)

    output: list[dict[str, Any]] = []
    for raw in episodes:
        row = dict(raw)
        identity = str(
            row.get("dynamic_episode_uid")
            or f"{row.get('episode_uid')}:{row.get('event_type')}"
        )
        trace = sorted(
            by_key.get((identity, str(row["mode"])), []),
            key=lambda item: int(item["step"]),
        )
        event_step = int(row.get("event_step", trace[0]["event_step"] if trace else 0))
        opening_step = int(
            row.get("opening_step", trace[0]["opening_step"] if trace else event_step)
        )
        static_prefix = [item for item in trace if int(item["step"]) < event_step]
        pre_open = [
            item
            for item in trace
            if event_step <= int(item["step"]) < opening_step
        ]
        post_open = [item for item in trace if int(item["step"]) >= opening_step]
        static_active = (
            int(static_prefix[-1]["effective_gate_active"])
            if static_prefix
            else 0
        )
        false_pre = int(
            any(int(item["effective_gate_active"]) for item in pre_open)
        )
        first_active = next(
            (
                int(item["step"])
                for item in post_open
                if int(item["effective_gate_active"])
            ),
            None,
        )
        # Non-reactions remain in the denominator as a conservative censored
        # delay one step beyond the configured rollout.
        reaction_delay = (
            first_active - opening_step
            if first_active is not None
            else max(0, max_steps - opening_step + 1)
        )
        successful_windows = [
            item
            for item in post_open
            if int(item["effective_gate_active"])
            and int(item.get("mechanism_window_complete", 0))
            and int(item.get("horizon_clearance_retained", 0))
            and int(item.get("horizon_predicted_risk_realized", 0))
        ]
        row.update(
            {
                "hard_contact_episode": int(int(row.get("hard_contacts", 0)) > 0),
                "static_activation": static_active,
                "false_pre_activation": false_pre,
                "post_open_reaction_delay_steps": reaction_delay,
                "post_open_success": int(bool(successful_windows)),
                "trace_complete": int(len(trace) == int(row.get("steps", len(trace)))),
            }
        )
        output.append(row)
    return output


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[list[Mapping[str, Any]]], float],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    if not rows:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(list(rows)))
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["scene_id"])].append(row)
    names = sorted(clusters)
    if not names or n_boot < 1:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(names, size=len(names), replace=True)
        sample = [row for name in chosen for row in clusters[str(name)]]
        value = float(statistic(sample))
        if np.isfinite(value):
            samples.append(value)
    if not samples:
        return point, float("nan"), float("nan")
    low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
    return point, float(low), float(high)


def _paired_rows(
    episodes: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
) -> list[dict[str, Any]]:
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in episodes:
        identity = str(
            row.get("dynamic_episode_uid")
            or f"{row.get('episode_uid')}:{row.get('event_type')}"
        )
        index[(identity, str(row["mode"]))] = row
    pairs: list[dict[str, Any]] = []
    for identity in sorted({key[0] for key in index}):
        if (identity, left) in index and (identity, right) in index:
            pairs.append(
                {
                    "scene_id": index[(identity, left)]["scene_id"],
                    "left": index[(identity, left)],
                    "right": index[(identity, right)],
                }
            )
    return pairs


def _criterion(
    name: str,
    group: str,
    rows: Sequence[Mapping[str, Any]],
    statistic: Callable[[list[Mapping[str, Any]]], float],
    *,
    comparator: str,
    threshold: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    estimate, low, high = _cluster_bootstrap(
        rows, statistic, n_boot=n_boot, seed=seed
    )
    passed = bool(
        np.isfinite(estimate)
        and (
            (comparator == ">=" and estimate >= threshold)
            or (comparator == "<=" and estimate <= threshold)
        )
    )
    return {
        "name": name,
        "group": group,
        "n": len(rows),
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "comparator": comparator,
        "threshold": threshold,
        "passed": passed,
        "evaluable": bool(rows) and np.isfinite(estimate),
    }


def compute_preregistered_metrics(
    episodes: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    *,
    static_anchor_event: str = "mud_onset",
    n_boot: int = 1000,
    seed: int = 27370,
) -> dict[str, Any]:
    """Compute every frozen criterion and the validation go/no-go decision."""

    primary_windows = [
        row
        for row in traces
        if row["mode"] == PRIMARY_MODE
        and int(row["effective_gate_active"])
        and int(row.get("mechanism_window_complete", 0))
    ]
    paired_geom = _paired_rows(episodes, PRIMARY_MODE, GEOMETRY_MODE)
    safety = [
        _criterion(
            "hard_contact_rate_difference_vs_geometry",
            "safety",
            paired_geom,
            lambda rows: _mean(
                int(row["left"]["hard_contact_episode"])
                - int(row["right"]["hard_contact_episode"])
                for row in rows
            ),
            comparator="<=",
            threshold=0.02,
            n_boot=n_boot,
            seed=seed,
        ),
        _criterion(
            "violation_cvar_difference_vs_geometry",
            "safety",
            paired_geom,
            lambda rows: _mean(
                float(row["left"]["cvar20_violation"])
                - float(row["right"]["cvar20_violation"])
                for row in rows
            ),
            comparator="<=",
            threshold=0.05,
            n_boot=n_boot,
            seed=seed + 1,
        ),
        _criterion(
            "median_primitive_execution_cosine",
            "mechanism",
            primary_windows,
            lambda rows: _median(row["horizon_endpoint_alignment"] for row in rows),
            comparator=">=",
            threshold=0.70,
            n_boot=n_boot,
            seed=seed + 2,
        ),
        _criterion(
            "clearance_retention_rate",
            "mechanism",
            primary_windows,
            lambda rows: _mean(row["horizon_clearance_retained"] for row in rows),
            comparator=">=",
            threshold=0.90,
            n_boot=n_boot,
            seed=seed + 3,
        ),
        _criterion(
            "predicted_risk_reduction_realization_rate",
            "mechanism",
            primary_windows,
            lambda rows: _mean(
                row["horizon_predicted_risk_realized"] for row in rows
            ),
            comparator=">=",
            threshold=0.70,
            n_boot=n_boot,
            seed=seed + 4,
        ),
    ]

    anchor = [
        row
        for row in episodes
        if row["mode"] == PRIMARY_MODE
        and row["event_type"] == static_anchor_event
    ]
    behavior_specs = (
        ("static_R1_CAR", "R1", ">=", 0.65),
        ("static_R2_FAR", "R2", "<=", 0.25),
        ("static_R3_activation_rate", "R3", "<=", 0.20),
    )
    behavior = [
        _criterion(
            name,
            "behavior",
            [row for row in anchor if row["regime"] == regime],
            lambda rows: _mean(row["static_activation"] for row in rows),
            comparator=comparator,
            threshold=threshold,
            n_boot=n_boot,
            seed=seed + 10 + index,
        )
        for index, (name, regime, comparator, threshold) in enumerate(
            behavior_specs
        )
    ]
    delayed = [
        row
        for row in episodes
        if row["mode"] == PRIMARY_MODE and row["event_type"] == DELAYED_EVENT
    ]
    behavior.extend(
        [
            _criterion(
                "delayed_required_escape_post_open_success",
                "behavior",
                delayed,
                lambda rows: _mean(row["post_open_success"] for row in rows),
                comparator=">=",
                threshold=0.70,
                n_boot=n_boot,
                seed=seed + 13,
            ),
            _criterion(
                "delayed_required_escape_false_pre_activation",
                "behavior",
                delayed,
                lambda rows: _mean(row["false_pre_activation"] for row in rows),
                comparator="<=",
                threshold=0.25,
                n_boot=n_boot,
                seed=seed + 14,
            ),
        ]
    )

    separation_rows = [
        row
        for row in traces
        if row["mode"] == PRIMARY_MODE
        and int(row.get("paired_gate_state_differs", 0))
        and _finite(row.get("paired_endpoint_separation_m"))
    ]
    evidence = _criterion(
        "differing_gate_windows_separating_at_least_0.5m",
        "behavioral_effect",
        separation_rows,
        lambda rows: _mean(
            float(row["paired_endpoint_separation_m"]) >= 0.5 for row in rows
        ),
        comparator=">=",
        threshold=0.25,
        n_boot=n_boot,
        seed=seed + 20,
    )

    paired_temporal = _paired_rows(
        episodes, PRIMARY_MODE, STATELESS_MODE
    )

    def transition_reduction(rows: list[Mapping[str, Any]]) -> float:
        repaired = _mean(row["left"]["activation_transitions"] for row in rows)
        stateless = _mean(row["right"]["activation_transitions"] for row in rows)
        return 1.0 - repaired / stateless if stateless > 0 else float("nan")

    temporal_transitions = _criterion(
        "activation_transition_relative_reduction",
        "temporal",
        paired_temporal,
        transition_reduction,
        comparator=">=",
        threshold=0.30,
        n_boot=n_boot,
        seed=seed + 21,
    )
    delayed_pairs = [
        row
        for row in paired_temporal
        if row["left"]["event_type"] == DELAYED_EVENT
    ]
    temporal_delay = _criterion(
        "median_reaction_delay_increase_steps",
        "temporal",
        delayed_pairs,
        lambda rows: _median(
            float(row["left"]["post_open_reaction_delay_steps"])
            - float(row["right"]["post_open_reaction_delay_steps"])
            for row in rows
        ),
        comparator="<=",
        threshold=1.0,
        n_boot=n_boot,
        seed=seed + 22,
    )

    safety_pass = all(item["passed"] for item in safety)
    behavior_pass_count = sum(item["passed"] for item in behavior)
    temporal_pass = temporal_transitions["passed"] and temporal_delay["passed"]
    go = bool(
        safety_pass
        and behavior_pass_count >= 4
        and evidence["passed"]
        and temporal_pass
    )
    return {
        "safety_and_mechanism": safety,
        "behavior": behavior,
        "behavioral_effect": evidence,
        "temporal": [temporal_transitions, temporal_delay],
        "decision": {
            "all_safety_and_mechanism_pass": safety_pass,
            "behavior_pass_count": behavior_pass_count,
            "behavior_required_pass_count": 4,
            "behavioral_effect_pass": evidence["passed"],
            "temporal_pass": temporal_pass,
            "go_for_one_shot_test": go,
        },
    }


def stratified_summaries(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Report denominators and headline values without pooling regimes/events."""

    def summarize(keys: tuple[str, ...]) -> list[dict[str, Any]]:
        groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in episodes:
            groups[tuple(str(row[key]) for key in keys)].append(row)
        output = []
        for identity, rows in sorted(groups.items()):
            record = {key: value for key, value in zip(keys, identity)}
            record.update(
                {
                    "n": len(rows),
                    "success_rate": _mean(row["success"] for row in rows),
                    "hard_contact_rate": _mean(
                        row["hard_contact_episode"] for row in rows
                    ),
                    "mean_cvar20_violation": _mean(
                        row["cvar20_violation"] for row in rows
                    ),
                    "activation_rate": _mean(row["activation_rate"] for row in rows),
                    "post_open_success_rate": _mean(
                        row["post_open_success"] for row in rows
                    ),
                }
            )
            output.append(record)
        return output

    return {
        "by_regime": summarize(("mode", "regime")),
        "by_event_type": summarize(("mode", "event_type")),
        "by_event_type_and_regime": summarize(("mode", "event_type", "regime")),
    }
