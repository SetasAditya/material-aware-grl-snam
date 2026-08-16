from __future__ import annotations

import numpy as np

from rebuttal_experiments.exp7_semantic_corruption import (
    corrupt_label_grid,
    route_contexts_for_goals,
    stable_scene_seed,
)
from rellis.train_rellis_directional_force import _route_context


def fixture():
    labels = np.asarray([[0, 1, 3, 4], [4, 3, 1, 0], [1, 3, 4, 0]], dtype=np.uint16)
    observed = np.asarray([[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1]], dtype=bool)
    counts = {0: 10, 1: 30, 3: 40, 4: 20}
    return labels, observed, counts


def test_p0_is_exact_and_unobserved_never_changes():
    labels, observed, counts = fixture()
    out, audit = corrupt_label_grid(labels, observed, 0.0, counts, seed=9)
    assert np.array_equal(out, labels)
    assert audit["changed_cells"] == 0
    noisy, _ = corrupt_label_grid(labels, observed, 0.3, counts, seed=9)
    assert np.array_equal(noisy[~observed], labels[~observed])


def test_corruption_is_deterministic_and_changes_exact_fraction():
    labels, observed, counts = fixture()
    a, audit_a = corrupt_label_grid(labels, observed, 0.3, counts, seed=123)
    b, audit_b = corrupt_label_grid(labels, observed, 0.3, counts, seed=123)
    assert np.array_equal(a, b)
    assert audit_a == audit_b
    expected = round(0.3 * int(observed.sum()))
    assert audit_a["changed_cells"] == expected
    assert np.all(a[observed][a[observed] != labels[observed]] != labels[observed][a[observed] != labels[observed]])


def test_masks_are_nested_across_levels():
    labels, observed, counts = fixture()
    low, _ = corrupt_label_grid(labels, observed, 0.2, counts, seed=77)
    high, _ = corrupt_label_grid(labels, observed, 0.4, counts, seed=77)
    low_changed = low != labels
    high_changed = high != labels
    assert np.all(~low_changed | high_changed)


def test_scene_seed_is_stable_and_scene_specific():
    assert stable_scene_seed(27370, "scene_a") == stable_scene_seed(27370, "scene_a")
    assert stable_scene_seed(27370, "scene_a") != stable_scene_seed(27370, "scene_b")


def test_batched_route_context_matches_canonical():
    rng = np.random.default_rng(4)
    risk = rng.random((9, 8), dtype=np.float32)
    blocked = np.zeros((9, 8), dtype=np.uint8)
    blocked[2:5, 3] = 1
    maps = {"risk_map": risk, "geom_occ": blocked}
    goals = [(0, 0), (8, 7)]
    batched = route_contexts_for_goals(maps, goals, risk_weight=12.0)
    for goal in goals:
        expected = _route_context(maps, goal, risk_weight=12.0)
        assert np.array_equal(batched[goal]["geom_to_go"], expected["geom_to_go"])
        assert np.array_equal(batched[goal]["risk_to_go"], expected["risk_to_go"])
