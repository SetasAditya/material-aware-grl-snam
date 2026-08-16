from __future__ import annotations

import numpy as np
import pandas as pd

from rebuttal_experiments.paired_cluster_bootstrap import (
    analyze,
    prepare_dynamic,
    prepare_static,
)


def test_multi_event_resamples_episode_clusters() -> None:
    rows = []
    for episode_id in ("0", "1", "2"):
        for event in ("a", "b"):
            base = float(episode_id)
            rows.extend(
                [
                    {
                        "episode_id": episode_id,
                        "event_type": event,
                        "method": "a",
                        "success": base + 1.0,
                    },
                    {
                        "episode_id": episode_id,
                        "event_type": event,
                        "method": "b",
                        "success": base,
                    },
                ]
            )
    result, validation = analyze(
        pd.DataFrame(rows),
        mode="dynamic",
        comparisons=[("a", "b")],
        metrics=["success"],
        scopes=["aggregate"],
        n_bootstrap=200,
        seed=7,
    )
    assert validation.loc[0, "paired_observations"] == 6
    assert validation.loc[0, "min_observations_per_cluster"] == 2
    assert validation.loc[0, "max_observations_per_cluster"] == 2
    assert result.loc[0, "n_clusters"] == 3
    assert result.loc[0, "difference_a_minus_b"] == 1.0
    assert np.isclose(result.loc[0, "ci95_low"], 1.0)
    assert np.isclose(result.loc[0, "ci95_high"], 1.0)


def test_false_pre_activation_uses_per_episode_open_delay(tmp_path) -> None:
    rollouts = pd.DataFrame(
        [
            {
                "episode_id": 0,
                "event_type": "delayed_required_escape",
                "method": "a",
                "route_deviation_delay": 4,
            },
            {
                "episode_id": 0,
                "event_type": "delayed_required_escape",
                "method": "b",
                "route_deviation_delay": 5,
            },
            {
                "episode_id": 1,
                "event_type": "delayed_required_escape",
                "method": "a",
                "route_deviation_delay": 9,
            },
            {
                "episode_id": 1,
                "event_type": "delayed_required_escape",
                "method": "b",
                "route_deviation_delay": 10,
            },
        ]
    )
    specs = pd.DataFrame(
        [
            {"episode_id": 0, "event_type": "delayed_required_escape", "open_delay": 5},
            {"episode_id": 1, "event_type": "delayed_required_escape", "open_delay": 10},
        ]
    )
    rollout_path = tmp_path / "rollouts.csv"
    specs_path = tmp_path / "specs.csv"
    rollouts.to_csv(rollout_path, index=False)
    specs.to_csv(specs_path, index=False)
    prepared = prepare_dynamic(rollout_path, specs_path)
    a = prepared.loc[prepared["method"].eq("a"), "false_pre_activation"]
    b = prepared.loc[prepared["method"].eq("b"), "false_pre_activation"]
    assert a.tolist() == [1.0, 1.0]
    assert b.tolist() == [0.0, 0.0]
    assert np.allclose(prepared["suppression"], 1.0 - prepared["false_pre_activation"])


def test_static_source_names_are_preserved(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "episode_id": "ep0",
                "regime": "R1",
                "path_index": 0,
                "force_source": "analytic_fixed_lambda",
                "has_safe_alt": 1,
                "dot_safe": 0.1,
                "force_perp_norm": 0.2,
            },
            {
                "episode_id": "ep0",
                "regime": "R1",
                "path_index": 0,
                "force_source": "stage2_directional_head",
                "has_safe_alt": 1,
                "dot_safe": 0.0,
                "force_perp_norm": 0.0,
            },
        ]
    )
    path = tmp_path / "force_samples.csv"
    frame.to_csv(path, index=False)
    prepared = prepare_static(path, force_eps=0.02)
    assert set(prepared["method"]) == {
        "analytic_fixed_lambda",
        "stage2_directional_head",
    }
    assert "gate_off" not in set(prepared["method"])
