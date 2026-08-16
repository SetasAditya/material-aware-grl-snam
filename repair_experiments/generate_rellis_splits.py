#!/usr/bin/env python3
"""Generate the frozen, leakage-free manifests for the repaired RELLIS study.

This command only indexes already-sampled episodes.  It does not load BEV
tensors, run a controller, or compute an outcome metric.  In particular, it
must be run before tuning and is safe to use while the held-out test sequence
is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "rellis-repair-split-v1"
TRAIN_SEQUENCES = ("00000", "00001", "00002")
VALIDATION_SEQUENCES = ("00003",)
TEST_SEQUENCES = ("00004",)
REGIMES = ("R1", "R2", "R3")

# Frozen from grl_rellis.dyn_events.MAIN_EVENT_TYPES.  Keeping the values in
# the split generator prevents a later code change from silently changing the
# preregistered benchmark.
DYNAMIC_EVENT_TYPES = (
    "mud_onset",
    "puddle_expansion",
    "corridor_closes",
    "corridor_opens",
    "crossing_obstacle",
    "moving_obstacle_blocks_detour",
    "mud_onset_detour_blocked",
    "delayed_escape_opens",
    "delayed_required_escape",
)

SPLIT_SEQUENCES = {
    "train": TRAIN_SEQUENCES,
    "validation": VALIDATION_SEQUENCES,
    "test": TEST_SEQUENCES,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(identifier: str) -> int:
    # 31-bit seeds are accepted by NumPy, PyTorch, and Python's random module.
    return int.from_bytes(hashlib.sha256(identifier.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_manifest_path(source_root: Path, sequence: str) -> Path:
    return source_root / f"rellis_pairs_all_seqbalanced_2500_seq{sequence}" / "manifest.json"


def _resolve_bev_root(source_root: Path, configured_root: str) -> Path:
    configured = Path(configured_root)
    if configured.is_absolute():
        return configured.resolve()
    # source_root is normally <repo>/exp-rellis/cache, while the canonical
    # manifest records exp-rellis/cache/<bev-name> relative to <repo>.
    repo_root = source_root.resolve().parents[1]
    return (repo_root / configured).resolve()


def _identity_row(episode: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_uid": episode["episode_uid"],
        "sequence": episode["sequence"],
        "scene_id": episode["scene_id"],
        "frame_id": episode["frame_id"],
        "regime": episode["regime"],
        "start_rc": episode["start_rc"],
        "goal_rc": episode["goal_rc"],
    }


def _load_sequence(source_root: Path, sequence: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _source_manifest_path(source_root, sequence)
    raw_bytes = path.read_bytes()
    source = json.loads(raw_bytes)
    episodes = source.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{path}: missing episodes list")

    prepared: list[dict[str, Any]] = []
    seen_episode_uids: set[str] = set()
    for index, original in enumerate(episodes):
        episode = dict(original)
        found_sequence = str(episode.get("sequence", ""))
        if found_sequence != sequence:
            raise ValueError(
                f"{path}: episode {index} claims sequence {found_sequence!r}, expected {sequence!r}"
            )
        regime = str(episode.get("regime", ""))
        if regime not in REGIMES:
            raise ValueError(f"{path}: episode {index} has unknown regime {regime!r}")
        scene_id = str(episode.get("scene_id", ""))
        if not scene_id.startswith(f"{sequence}_"):
            raise ValueError(f"{path}: scene {scene_id!r} is not namespaced by sequence {sequence}")
        source_episode_id = str(episode.get("episode_id", ""))
        episode_uid = f"rellis:{sequence}:{source_episode_id}"
        if episode_uid in seen_episode_uids:
            raise ValueError(f"{path}: duplicate episode UID {episode_uid}")
        seen_episode_uids.add(episode_uid)
        episode["source_episode_id"] = source_episode_id
        episode["episode_uid"] = episode_uid
        # The repaired scripts use episode_uid.  Preserve episode_id for legacy
        # readers, but make it globally unique and stable across concatenation.
        episode["episode_id"] = episode_uid
        prepared.append(episode)

    counts = Counter(str(ep["regime"]) for ep in prepared)
    expected_counts = {regime: 150 for regime in REGIMES}
    if len(prepared) != 450 or dict(counts) != expected_counts:
        raise ValueError(
            f"{path}: expected 450 episodes and {expected_counts}, "
            f"found {len(prepared)} and {dict(counts)}"
        )

    configured_bev_root = str(source.get("config", {}).get("bev_root", ""))
    if not configured_bev_root:
        raise ValueError(f"{path}: missing config.bev_root")
    bev_root = _resolve_bev_root(source_root, configured_bev_root)
    source_record = {
        "sequence": sequence,
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(raw_bytes),
        "bev_root": str(bev_root),
        "num_episodes": len(prepared),
        "counts_by_regime": expected_counts,
    }
    return prepared, source_record


def _audit_bev_manifest(
    sequence_episodes: Mapping[str, list[dict[str, Any]]],
    sequence_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    bev_roots = {str(source["bev_root"]) for source in sequence_sources.values()}
    if len(bev_roots) != 1:
        raise ValueError(f"Pair manifests reference different BEV roots: {sorted(bev_roots)}")
    bev_root = Path(next(iter(bev_roots)))
    bev_manifest_path = bev_root / "manifest.json"
    raw_bytes = bev_manifest_path.read_bytes()
    bev_manifest = json.loads(raw_bytes)
    records = bev_manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{bev_manifest_path}: missing records list")

    records_by_scene: dict[str, Mapping[str, Any]] = {}
    for record in records:
        scene_id = str(record.get("scene_id", ""))
        if not scene_id or scene_id in records_by_scene:
            raise ValueError(f"{bev_manifest_path}: missing or duplicate scene ID {scene_id!r}")
        records_by_scene[scene_id] = record
    counts_by_sequence = Counter(str(record.get("sequence", "")) for record in records)
    expected_counts = {sequence: 500 for sequence in sorted(sequence_sources)}
    if len(records) != 2500 or dict(sorted(counts_by_sequence.items())) != expected_counts:
        raise ValueError(
            f"{bev_manifest_path}: expected 2,500 scenes and {expected_counts}, "
            f"found {len(records)} and {dict(sorted(counts_by_sequence.items()))}"
        )

    for sequence, episodes in sequence_episodes.items():
        for episode in episodes:
            scene_id = str(episode["scene_id"])
            if scene_id not in records_by_scene:
                raise ValueError(f"Pair episode references unknown BEV scene {scene_id}")
            record = records_by_scene[scene_id]
            if str(record.get("sequence")) != sequence:
                raise ValueError(f"BEV scene {scene_id} has inconsistent sequence metadata")
            if str(record.get("path")) != str(episode.get("scene_path")):
                raise ValueError(f"Pair episode and BEV manifest disagree on path for {scene_id}")

    return {
        "path": str(bev_manifest_path.resolve()),
        "sha256": _sha256_bytes(raw_bytes),
        "num_scenes": len(records),
        "counts_by_sequence": expected_counts,
        "pair_scene_reference_audit": "PASS",
    }


def _count_nested(
    rows: Iterable[Mapping[str, Any]], first_key: str, second_key: str
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for row in rows:
        first = str(row[first_key])
        counts.setdefault(first, Counter())[str(row[second_key])] += 1
    return {
        key: dict(sorted(value.items()))
        for key, value in sorted(counts.items())
    }


def _make_static_manifest(
    split_name: str,
    sequences: tuple[str, ...],
    episodes: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    bev_source: Mapping[str, Any],
) -> dict[str, Any]:
    identities = [_identity_row(ep) for ep in episodes]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "static_episode_manifest",
        "split_name": split_name,
        "role": {
            "train": "optimization",
            "validation": "model_selection_and_threshold_calibration",
            "test": "sealed_one_shot_final_evaluation",
        }[split_name],
        "evaluation_status": (
            "SEALED_NOT_EVALUATED" if split_name == "test" else "AVAILABLE_FOR_DEVELOPMENT"
        ),
        "sequences": list(sequences),
        "bev_source": dict(bev_source),
        "source_manifests": sources,
        "num_episodes": len(episodes),
        "counts_by_sequence": dict(sorted(Counter(ep["sequence"] for ep in episodes).items())),
        "counts_by_regime": dict(sorted(Counter(ep["regime"] for ep in episodes).items())),
        "counts_by_sequence_regime": _count_nested(episodes, "sequence", "regime"),
        "episode_identity_sha256": _sha256_bytes(_canonical_bytes(identities)),
        "episode_payload_sha256": _sha256_bytes(_canonical_bytes(episodes)),
        "episodes": episodes,
    }


def _make_dynamic_manifest(
    split_name: str,
    static_manifest_name: str,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for episode in episodes:
        for event_type in DYNAMIC_EVENT_TYPES:
            dynamic_uid = f"{episode['episode_uid']}:event:{event_type}"
            items.append(
                {
                    "dynamic_episode_uid": dynamic_uid,
                    "base_episode_uid": episode["episode_uid"],
                    "sequence": episode["sequence"],
                    "scene_id": episode["scene_id"],
                    "regime": episode["regime"],
                    "event_type": event_type,
                    "event_seed": _stable_seed(dynamic_uid),
                }
            )
    identities = [
        {
            "dynamic_episode_uid": item["dynamic_episode_uid"],
            "base_episode_uid": item["base_episode_uid"],
            "event_type": item["event_type"],
            "event_seed": item["event_seed"],
        }
        for item in items
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "dynamic_episode_index",
        "split_name": split_name,
        "evaluation_status": (
            "SEALED_NOT_EVALUATED" if split_name == "test" else "AVAILABLE_FOR_DEVELOPMENT"
        ),
        "static_manifest": static_manifest_name,
        "construction": "full Cartesian product of base episodes and frozen event types",
        "event_types": list(DYNAMIC_EVENT_TYPES),
        "num_dynamic_episodes": len(items),
        "counts_by_event_type": dict(sorted(Counter(row["event_type"] for row in items).items())),
        "counts_by_regime": dict(sorted(Counter(row["regime"] for row in items).items())),
        "counts_by_event_type_regime": _count_nested(items, "event_type", "regime"),
        "dynamic_identity_sha256": _sha256_bytes(_canonical_bytes(identities)),
        "items": items,
    }


def _assert_no_overlap(static_manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    episode_sets: dict[str, set[str]] = {}
    scene_sets: dict[str, set[str]] = {}
    sequence_sets: dict[str, set[str]] = {}
    for split_name, manifest in static_manifests.items():
        episodes = manifest["episodes"]
        episode_sets[split_name] = {str(ep["episode_uid"]) for ep in episodes}
        scene_sets[split_name] = {str(ep["scene_id"]) for ep in episodes}
        sequence_sets[split_name] = {str(ep["sequence"]) for ep in episodes}

    pair_names = (("train", "validation"), ("train", "test"), ("validation", "test"))
    intersections: dict[str, Any] = {}
    for left, right in pair_names:
        key = f"{left}__{right}"
        intersections[key] = {
            "sequence_overlap": sorted(sequence_sets[left] & sequence_sets[right]),
            "scene_overlap": sorted(scene_sets[left] & scene_sets[right]),
            "episode_overlap": sorted(episode_sets[left] & episode_sets[right]),
        }
        if any(intersections[key].values()):
            raise ValueError(f"Leakage detected between {left} and {right}: {intersections[key]}")

    all_episode_uids = set().union(*episode_sets.values())
    expected_total = sum(len(values) for values in episode_sets.values())
    if len(all_episode_uids) != expected_total:
        raise ValueError("An episode UID is duplicated across split manifests")
    return {
        "status": "PASS",
        "boundary": "whole RELLIS sequence",
        "pairwise_intersections": intersections,
        "num_unique_episode_uids": len(all_episode_uids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/data/adityas/GRL-SNAM/exp-rellis/cache"),
        help="Directory containing the five canonical per-sequence pair-manifest folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "splits",
        help="Output directory for the frozen split manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sequence_episodes: dict[str, list[dict[str, Any]]] = {}
    sequence_sources: dict[str, dict[str, Any]] = {}
    for sequence in (*TRAIN_SEQUENCES, *VALIDATION_SEQUENCES, *TEST_SEQUENCES):
        episodes, source = _load_sequence(args.source_root, sequence)
        sequence_episodes[sequence] = episodes
        sequence_sources[sequence] = source
    bev_source = _audit_bev_manifest(sequence_episodes, sequence_sources)

    static_manifests: dict[str, dict[str, Any]] = {}
    dynamic_manifests: dict[str, dict[str, Any]] = {}
    for split_name, sequences in SPLIT_SEQUENCES.items():
        episodes = [ep for sequence in sequences for ep in sequence_episodes[sequence]]
        sources = [sequence_sources[sequence] for sequence in sequences]
        static_name = f"{split_name}_static.json"
        static = _make_static_manifest(split_name, sequences, episodes, sources, bev_source)
        dynamic = _make_dynamic_manifest(split_name, static_name, episodes)
        static_manifests[split_name] = static
        dynamic_manifests[split_name] = dynamic

    audit = _assert_no_overlap(static_manifests)

    output_hashes: dict[str, str] = {}
    for split_name in SPLIT_SEQUENCES:
        for kind, manifest in (
            ("static", static_manifests[split_name]),
            ("dynamic", dynamic_manifests[split_name]),
        ):
            filename = f"{split_name}_{kind}.json"
            _write_json(args.out / filename, manifest)
            output_hashes[filename] = _sha256_file(args.out / filename)

    lock = {
        "schema_version": SCHEMA_VERSION,
        "split_assignment": {
            split_name: list(sequences)
            for split_name, sequences in SPLIT_SEQUENCES.items()
        },
        "dynamic_event_types": list(DYNAMIC_EVENT_TYPES),
        "source_manifest_hashes": {
            sequence: source["sha256"]
            for sequence, source in sorted(sequence_sources.items())
        },
        "bev_manifest": bev_source,
        "output_manifest_hashes": dict(sorted(output_hashes.items())),
        "leakage_audit": audit,
        "test_policy": (
            "Do not load test_static.json, test_dynamic.json, sequence 00004 BEVs, "
            "or any outcome files derived from them until the model, controller, "
            "thresholds, seeds, and analysis code are frozen from train/validation."
        ),
    }
    _write_json(args.out / "SPLIT_LOCK.json", lock)

    print(f"Wrote frozen manifests to {args.out.resolve()}")
    print(
        "Static episodes:",
        {name: manifest["num_episodes"] for name, manifest in static_manifests.items()},
    )
    print(
        "Dynamic cases:",
        {
            name: manifest["num_dynamic_episodes"]
            for name, manifest in dynamic_manifests.items()
        },
    )
    print("Leakage audit:", audit["status"])
    print("Held-out test status: SEALED_NOT_EVALUATED")


if __name__ == "__main__":
    main()
