import math

import torch

from rebuttal_experiments.exp9_soft_coefficient_isolation import (
    arm_lambda,
    summarize_metrics,
)


def test_arm_lambda_changes_only_soft_value() -> None:
    learned = torch.tensor([0.25])
    assert arm_lambda("zero", learned, 1.5, True).item() == 0.0
    assert arm_lambda("learned", learned, 1.5, True).item() == 0.25
    assert arm_lambda("fixed", learned, 1.5, True).item() == 1.5
    assert arm_lambda("fixed", learned, 1.5, False).item() == 0.0


def test_metric_summary_keeps_three_arms_separate() -> None:
    rows = []
    for index, arm in enumerate(("zero", "learned", "fixed")):
        rows.append(
            {
                "dataset": "toy",
                "arm": arm,
                "success": 1.0,
                "path_length_ratio": 1.0 + index,
                "risk_exposure": 3.0 - index,
                "mean_rho": 0.1,
            }
        )
    summary = summarize_metrics(rows)
    assert set(summary["toy"]) == {"zero", "learned", "fixed"}
    assert math.isclose(summary["toy"]["learned"]["risk_exposure"]["mean"], 2.0)
