from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("generate_rellis_splits.py")
SPEC = importlib.util.spec_from_file_location("generate_rellis_splits", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
splits = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(splits)


def _episode(sequence: str, index: int, regime: str) -> dict:
    frame = f"{index:06d}"
    return {
        "episode_id": f"{sequence}_{frame}_00",
        "scene_id": f"{sequence}_{frame}",
        "sequence": sequence,
        "frame_id": frame,
        "regime": regime,
        "start_rc": [50, 10],
        "goal_rc": [50, 90],
    }


def test_dynamic_index_is_balanced_and_deterministic() -> None:
    episodes = [
        {
            **_episode("00000", index, regime),
            "episode_uid": f"rellis:00000:{index}",
        }
        for index, regime in enumerate(("R1", "R2", "R3"))
    ]
    first = splits._make_dynamic_manifest("train", "train_static.json", episodes)
    second = splits._make_dynamic_manifest("train", "train_static.json", episodes)
    assert first == second
    assert first["num_dynamic_episodes"] == 3 * len(splits.DYNAMIC_EVENT_TYPES)
    assert set(first["counts_by_event_type"].values()) == {3}
    for counts in first["counts_by_event_type_regime"].values():
        assert counts == {"R1": 1, "R2": 1, "R3": 1}


def test_overlap_audit_rejects_shared_scene() -> None:
    base = {
        "episode_uid": "rellis:00000:ep",
        "scene_id": "00000_000001",
        "sequence": "00000",
    }
    manifests = {
        "train": {"episodes": [base]},
        "validation": {
            "episodes": [
                {
                    **base,
                    "episode_uid": "rellis:00003:ep",
                    "sequence": "00003",
                }
            ]
        },
        "test": {
            "episodes": [
                {
                    **base,
                    "episode_uid": "rellis:00004:ep",
                    "scene_id": "00004_000001",
                    "sequence": "00004",
                }
            ]
        },
    }
    try:
        splits._assert_no_overlap(manifests)
    except ValueError as error:
        assert "Leakage detected" in str(error)
    else:
        raise AssertionError("shared scenes must fail the leakage audit")


def test_overlap_audit_accepts_sequence_disjoint_manifests() -> None:
    manifests = {}
    for split_name, sequence in (
        ("train", "00000"),
        ("validation", "00003"),
        ("test", "00004"),
    ):
        manifests[split_name] = {
            "episodes": [
                {
                    "episode_uid": f"rellis:{sequence}:ep",
                    "scene_id": f"{sequence}_000001",
                    "sequence": sequence,
                }
            ]
        }
    audit = splits._assert_no_overlap(manifests)
    assert audit["status"] == "PASS"
    assert audit["num_unique_episode_uids"] == 3
