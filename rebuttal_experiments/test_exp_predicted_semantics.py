import json

import numpy as np
import pytest

from rellis.grl_rellis.bev import BevConfig
from rebuttal_experiments.exp_predicted_semantics import (
    cell_features,
    confusion_metrics,
    load_split,
)


def test_cell_features_preserve_bev_point_counts():
    cfg = BevConfig(
        x_min=0.0,
        x_max=2.0,
        y_min=0.0,
        y_max=2.0,
        resolution=1.0,
    )
    points = np.asarray(
        [
            [0.25, 1.75, 1.0, 0.1],
            [0.75, 1.25, 3.0, 0.3],
            [1.25, 0.75, -1.0, 0.5],
            [3.00, 3.00, 9.0, 0.9],
        ],
        dtype=np.float32,
    )
    features, count = cell_features(points, cfg)
    assert features.shape == (2, 2, 15)
    assert count.tolist() == [[2, 0], [0, 1]]
    assert features[0, 0, 0] == pytest.approx(np.log1p(2))
    assert features[0, 0, 1] == pytest.approx(2.0)
    assert features[0, 0, 5] == pytest.approx(2.0)


def test_confusion_metrics_use_rows_as_ground_truth():
    confusion = np.eye(6, dtype=np.int64)
    confusion[:2, :2] = np.asarray([[8, 2], [1, 9]], dtype=np.int64)
    rows, summary = confusion_metrics(confusion)
    assert summary["accuracy"] == pytest.approx(21 / 24)
    assert rows[0]["recall"] == pytest.approx(0.8)
    assert rows[0]["precision"] == pytest.approx(8 / 9)


def test_load_split_rejects_sealed_or_unexpected_sequence(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"episodes": [{"sequence": "00004"}]}))
    with pytest.raises(ValueError, match="expected"):
        load_split(path, {"00003"})
