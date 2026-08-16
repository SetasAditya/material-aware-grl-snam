"""Regression checks for the accurately named fixed semantic APF baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "full_code" / "exp-rellis" / "eval_rellis_dyn.py"
SPEC = importlib.util.spec_from_file_location("eval_rellis_dyn_exp8", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
dyn = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dyn)


def _maps() -> dict[str, np.ndarray]:
    risk = np.zeros((7, 7), dtype=np.float32)
    risk[3, 4] = 1.0
    return {
        "risk_map": risk,
        "hard_mask": np.zeros_like(risk, dtype=bool),
        "sdf_hard": np.full_like(risk, 10.0),
    }


def test_legacy_name_canonicalizes_to_semantic_apf() -> None:
    assert dyn._canonical_method("neural_potential_field") == "semantic_apf"
    assert "semantic_apf" in dyn.METHODS
    assert "neural_potential_field" not in dyn.METHODS


def test_legacy_and_canonical_rollouts_are_identical() -> None:
    maps = _maps()
    kwargs = dict(
        base_maps=maps,
        spec=dyn.DynamicEventSpec(
            event_type="mud_onset",
            event_step=100,
            duration=1,
            center_rc=(0, 0),
            detour_rc=(0, 0),
            goal_rc=(3, 5),
            axis_rc=(0, 1),
            radius_cells=0,
        ),
        start=(3, 3),
        goal=(3, 5),
        stage1_path=[(3, 3), (3, 4), (3, 5)],
        risk_path=[(3, 3), (2, 4), (3, 5)],
        gsd=1.0,
        max_steps=2,
        replan_period=8,
        risk_weight=18.0,
        hard_margin_m=1.0,
        route_horizon=2,
        improvement_margin=0.25,
    )
    canonical = dyn._rollout("semantic_apf", **kwargs)[0]
    legacy = dyn._rollout("neural_potential_field", **kwargs)[0]
    assert canonical == legacy


def test_semantic_apf_is_fixed_and_risk_aware() -> None:
    maps = _maps()
    step = dyn._semantic_apf_step(maps, (3, 3), (3, 5), hard_margin_m=1.0)
    assert step != (3, 4), "the high-risk nominal neighbor should be avoided"
    assert not bool(maps["hard_mask"][step])
