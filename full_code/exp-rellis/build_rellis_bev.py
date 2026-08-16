#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grl_rellis import BevConfig, build_bev_maps, iter_split_frames, load_frame, load_ontology


def _class_hist(labels: np.ndarray) -> Dict[str, int]:
    vals, counts = np.unique(labels, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build RELLIS-3D local BEV risk/SDF cache.")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "Rellis-3D")
    ap.add_argument("--split-file", type=Path, default=ROOT / "data" / "pt_val.lst")
    ap.add_argument("--out", type=Path, default=ROOT / "cache" / "rellis_bev_val_main")
    ap.add_argument("--ontology", type=Path, default=ROOT / "grl_rellis" / "risk_ontology.yaml")
    ap.add_argument("--mapping", default="main")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--x-min", type=float, default=-5.0)
    ap.add_argument("--x-max", type=float, default=45.0)
    ap.add_argument("--y-min", type=float, default=-25.0)
    ap.add_argument("--y-max", type=float, default=25.0)
    ap.add_argument("--resolution", type=float, default=0.5)
    ap.add_argument("--risk-sigma-cells", type=float, default=1.0)
    ap.add_argument("--hard-inflate-cells", type=int, default=1)
    ap.add_argument("--unknown-inflate-cells", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BevConfig(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        resolution=args.resolution,
        risk_sigma_cells=args.risk_sigma_cells,
        hard_inflate_cells=args.hard_inflate_cells,
        unknown_inflate_cells=args.unknown_inflate_cells,
    )
    ontology = load_ontology(args.ontology, args.mapping)
    args.out.mkdir(parents=True, exist_ok=True)
    scene_dir = args.out / "scenes"
    scene_dir.mkdir(exist_ok=True)

    records = []
    aggregate_hist: Dict[str, int] = {}
    frames = list(iter_split_frames(args.data_root, args.split_file, max_frames=args.max_frames))
    for frame in tqdm(frames, desc="building RELLIS BEV"):
        points, labels = load_frame(frame)
        maps = build_bev_maps(points, labels, ontology, cfg)
        scene_id = f"{frame.sequence}_{frame.frame_id}"
        rel_path = Path("scenes") / f"scene_{scene_id}.pt"
        hist = _class_hist(labels)
        for k, v in hist.items():
            aggregate_hist[k] = aggregate_hist.get(k, 0) + v
        payload: Dict[str, Any] = {
            "meta": {
                "scene_id": scene_id,
                "sequence": frame.sequence,
                "frame_id": frame.frame_id,
                "split": frame.split,
                "scan_path": str(frame.scan_path),
                "label_path": str(frame.label_path),
                "bev": cfg.to_dict(),
            },
            "maps": maps,
            "risk_ontology": {
                "name": ontology.name,
                "description": ontology.description,
                "rho_by_id": {str(k): float(v) for k, v in ontology.rho_by_id.items()},
                "hard_ids": sorted(int(x) for x in ontology.hard_ids),
                "soft_ids": sorted(int(x) for x in ontology.soft_ids),
                "low_ids": sorted(int(x) for x in ontology.low_ids),
            },
            "label_hist": hist,
        }
        torch.save(payload, args.out / rel_path)
        records.append(
            {
                "scene_id": scene_id,
                "split": frame.split,
                "sequence": frame.sequence,
                "frame_id": frame.frame_id,
                "path": str(rel_path),
                "scan_path": str(frame.scan_path),
                "label_path": str(frame.label_path),
                "observed_fraction": float(maps["observed_mask"].mean()),
                "hard_fraction": float(maps["hard_mask"].mean()),
                "risk_mean": float(maps["risk_map"].mean()),
                "risk_std": float(maps["risk_map"].std()),
            }
        )

    manifest = {
        "config": {
            "data_root": str(args.data_root),
            "split_file": str(args.split_file),
            "ontology": str(args.ontology),
            "mapping": args.mapping,
            "bev": cfg.to_dict(),
        },
        "num_scenes": len(records),
        "label_hist": aggregate_hist,
        "records": records,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(records)} BEV scenes to {args.out}")


if __name__ == "__main__":
    main()
