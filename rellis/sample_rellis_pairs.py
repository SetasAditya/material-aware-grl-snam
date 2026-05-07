#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grl_rellis import BevConfig, xy_to_rc
from scripts.baselines.dfc.metrics import FailureWeights, compute_path_metrics
from scripts.baselines.dfc.planners import geometry_astar, risk_weighted_astar


def _load_scene(bev_root: Path, rel_path: str) -> Dict:
    return torch.load(bev_root / rel_path, map_location="cpu", weights_only=False)


def _path_ok(path: Optional[List[Tuple[int, int]]]) -> bool:
    return path is not None and len(path) >= 3


def _nearest_free(maps: Dict[str, np.ndarray], rc: Tuple[int, int], radius: int = 5) -> Optional[Tuple[int, int]]:
    hard = maps["geom_occ"].astype(bool)
    rows, cols = hard.shape
    r0, c0 = rc
    if 0 <= r0 < rows and 0 <= c0 < cols and not hard[r0, c0]:
        return rc
    best = None
    best_d = 1e18
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < rows and 0 <= c < cols and not hard[r, c]:
                d = dr * dr + dc * dc
                if d < best_d:
                    best = (r, c)
                    best_d = d
    return best


def _classify(
    maps: Dict[str, np.ndarray],
    stage1_path: List[Tuple[int, int]],
    risk_path: List[Tuple[int, int]],
    *,
    gsd: float,
    risk_margin: float,
    homogeneous_std: float,
) -> Optional[str]:
    weights = FailureWeights()
    m1 = compute_path_metrics(stage1_path, maps, reference_length_m=None, gsd=gsd, weights=weights)
    m2 = compute_path_metrics(
        risk_path,
        maps,
        reference_length_m=float(m1["path_length_m"]),
        gsd=gsd,
        weights=weights,
    )
    risk_gain = float(m1["risk_exposure"]) - float(m2["risk_exposure"])
    ratio = float(m2["path_length_ratio"])
    path_risks = np.asarray([maps["risk_map"][r, c] for r, c in stage1_path], dtype=np.float32)
    if path_risks.std() <= homogeneous_std and risk_gain <= risk_margin:
        return "R3"
    if risk_gain >= risk_margin and ratio <= 1.8:
        return "R1"
    if path_risks.max() >= 0.55 and risk_gain < risk_margin:
        return "R2"
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sample R1/R2/R3 RELLIS local traversal episodes.")
    ap.add_argument("--bev-root", type=Path, default=ROOT / "cache" / "rellis_bev_val_main")
    ap.add_argument("--out", type=Path, default=ROOT / "cache" / "rellis_pairs_val_main")
    ap.add_argument("--pairs-per-scene", type=int, default=4)
    ap.add_argument("--target-per-regime", type=int, default=None,
                    help="If set, stop once each R1/R2/R3 bucket reaches this count.")
    ap.add_argument("--candidate-mult", type=int, default=16,
                    help="Candidate attempts per requested pair per scene.")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--sequence", default=None,
                    help="Optional RELLIS sequence id filter, e.g. 00001.")
    ap.add_argument("--shuffle-scenes", action="store_true",
                    help="Shuffle scene order before sampling; useful for multi-sequence subsets.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--risk-margin", type=float, default=2.5)
    ap.add_argument("--homogeneous-std", type=float, default=0.08)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    manifest = json.loads((args.bev_root / "manifest.json").read_text())
    records = manifest["records"]
    if args.sequence is not None:
        records = [r for r in records if str(r["sequence"]) == str(args.sequence)]
    if args.shuffle_scenes:
        rng.shuffle(records)
    records = records[: args.max_scenes]
    args.out.mkdir(parents=True, exist_ok=True)

    cfg_raw = manifest["config"]["bev"]
    cfg = BevConfig(**cfg_raw)
    gsd = float(cfg.resolution)
    y_candidates = [-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0]
    start_xs = [0.0, 2.0, 4.0]
    goal_xs = [32.0, 36.0, 40.0]

    episodes = []
    counts = {"R1": 0, "R2": 0, "R3": 0}
    for rec in tqdm(records, desc="sampling RELLIS pairs"):
        if args.target_per_regime is not None and all(
            counts[k] >= args.target_per_regime for k in counts
        ):
            break
        scene = _load_scene(args.bev_root, rec["path"])
        maps = scene["maps"]
        tried = set()
        scene_eps = 0
        candidates = [
            (rng.choice(start_xs), rng.choice(goal_xs), rng.choice(y_candidates), rng.choice(y_candidates))
            for _ in range(args.pairs_per_scene * max(1, args.candidate_mult))
        ]
        for sx, gx, sy, gy in candidates:
            key = (sx, gx, sy, gy)
            if key in tried:
                continue
            tried.add(key)
            start = _nearest_free(maps, xy_to_rc(sx, sy, cfg))
            goal = _nearest_free(maps, xy_to_rc(gx, gy, cfg))
            if start is None or goal is None or start == goal:
                continue
            stage1_path = geometry_astar(maps, start, goal)
            risk_path = risk_weighted_astar(maps, start, goal, risk_weight=12.0)
            if not _path_ok(stage1_path) or not _path_ok(risk_path):
                continue
            regime = _classify(
                maps,
                stage1_path,
                risk_path,
                gsd=gsd,
                risk_margin=args.risk_margin,
                homogeneous_std=args.homogeneous_std,
            )
            if regime is None:
                continue
            if args.target_per_regime is not None and counts[regime] >= args.target_per_regime:
                continue
            ep_id = f"{rec['scene_id']}_{scene_eps:02d}"
            episodes.append(
                {
                    "episode_id": ep_id,
                    "scene_id": rec["scene_id"],
                    "split": rec["split"],
                    "sequence": rec["sequence"],
                    "frame_id": rec["frame_id"],
                    "scene_path": rec["path"],
                    "start_rc": list(start),
                    "goal_rc": list(goal),
                    "regime": regime,
                    "stage1_path": [list(p) for p in stage1_path],
                    "risk_path": [list(p) for p in risk_path],
                }
            )
            counts[regime] += 1
            scene_eps += 1
            if scene_eps >= args.pairs_per_scene:
                break

    out_manifest = {
        "config": {
            "bev_root": str(args.bev_root),
            "pairs_per_scene": args.pairs_per_scene,
            "target_per_regime": args.target_per_regime,
            "candidate_mult": args.candidate_mult,
            "sequence": args.sequence,
            "shuffle_scenes": args.shuffle_scenes,
            "seed": args.seed,
            "risk_margin": args.risk_margin,
            "homogeneous_std": args.homogeneous_std,
        },
        "num_episodes": len(episodes),
        "counts_by_regime": counts,
        "episodes": episodes,
    }
    (args.out / "manifest.json").write_text(json.dumps(out_manifest, indent=2))
    print(f"Wrote {len(episodes)} sampled episodes to {args.out}")
    print("Counts by regime:", counts)


if __name__ == "__main__":
    main()
