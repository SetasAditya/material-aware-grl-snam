#!/usr/bin/env python3
"""Create a lightweight union manifest over multiple IDM dataset directories.

No episode tensors are copied. The output directory contains only a
manifest.json whose paths point back to the source episode files. Source specs
can include a repeat factor, e.g.

    python make_idm_union_manifest.py \
        --out runs/stage1_data_default_slow_x4 \
        runs/stage1_data:1 runs/stage1_data_slow_leader:4

The repeat factor is useful when a small but important scenario collection
should not be washed out by a larger default highway-v0 collection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_source_spec(spec: str) -> Tuple[Path, int]:
    path_s, sep, repeat_s = spec.rpartition(":")
    if sep and repeat_s.isdigit():
        path = Path(path_s)
        repeat = int(repeat_s)
    else:
        path = Path(spec)
        repeat = 1
    if repeat <= 0:
        raise ValueError(f"Repeat must be positive in source spec {spec!r}")
    return path, repeat


def _load_manifest(src_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = src_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    with open(manifest_path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise RuntimeError(f"Manifest is not a list: {manifest_path}")
    return records


def _counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for rec in records:
        split = rec.get("split", "train")
        out[split] = out.get(split, 0) + int(rec.get("n_samples", 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, required=True,
                    help="Output directory for union manifest.json.")
    ap.add_argument("sources", nargs="+",
                    help="Dataset dirs, optionally suffixed with :repeat.")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    union: List[Dict[str, Any]] = []
    source_summary: List[Dict[str, Any]] = []
    next_episode_idx = 0

    for spec in args.sources:
        src_dir, repeat = _parse_source_spec(spec)
        src_dir = src_dir.resolve()
        records = _load_manifest(src_dir)
        source_counts = _counts(records)
        source_envs = sorted({str(r.get("env_id", "?")) for r in records})
        source_summary.append({
            "source": str(src_dir),
            "repeat": repeat,
            "episodes": len(records),
            "samples": sum(int(r.get("n_samples", 0)) for r in records),
            "weighted_samples": repeat * sum(int(r.get("n_samples", 0)) for r in records),
            "splits": source_counts,
            "env_ids": source_envs,
        })

        for rep in range(repeat):
            for rec in records:
                src_episode = src_dir / rec["path"]
                if not src_episode.exists():
                    raise FileNotFoundError(f"Episode listed in manifest is missing: {src_episode}")
                new_rec = dict(rec)
                new_rec["path"] = os.path.relpath(src_episode, out_dir)
                new_rec["episode_idx"] = next_episode_idx
                new_rec["source_episode_idx"] = rec.get("episode_idx")
                new_rec["source_dir"] = str(src_dir)
                new_rec["source_repeat_idx"] = rep
                union.append(new_rec)
                next_episode_idx += 1

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(union, f, indent=2)
    with open(out_dir / "union_info.json", "w") as f:
        json.dump({
            "sources": source_summary,
            "total_episodes": len(union),
            "total_samples": sum(int(r.get("n_samples", 0)) for r in union),
            "splits": _counts(union),
        }, f, indent=2)

    print(f"Wrote {out_dir / 'manifest.json'}")
    print(f"  episodes: {len(union)}")
    print(f"  samples:  {sum(int(r.get('n_samples', 0)) for r in union)}")
    print(f"  splits:   {_counts(union)}")
    for src in source_summary:
        print(
            f"  source: {src['source']} x{src['repeat']} "
            f"episodes={src['episodes']} samples={src['samples']} "
            f"weighted={src['weighted_samples']} envs={src['env_ids']}"
        )


if __name__ == "__main__":
    main()
