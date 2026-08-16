import numpy as np

from repair_experiments.evaluate_gate_conditioned_soft import _synthetic_path


def test_synthetic_path_starts_at_actual_position_and_keeps_reference_tail() -> None:
    reference = np.asarray([[10, 10], [9, 11], [8, 12], [7, 13]], dtype=np.float32)
    path, nearest = _synthetic_path(
        reference,
        np.asarray([11.2, 9.1], dtype=np.float32),
        previous_index=0,
    )
    assert nearest == 1
    assert path[0] == (9, 11)
    assert path[-1] == (7, 13)
