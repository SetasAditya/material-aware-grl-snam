#!/usr/bin/env python3
"""Inspect what an IDM ego actually does in a highway-env scenario.

This is a cheap diagnostic before augmenting Stage 1 data: if IDM itself
changes lanes or brakes in an authored slow-leader scenario, that capability can
be copied into the Stage 1 geometric scaffold. If IDM also crashes, Stage 2
cannot inherit a maneuver that is absent from the imitation source.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List

import numpy as np

from data_collect_idm import (
    POLICY_FREQ,
    _idle_action,
    _import_highway_env,
    _reset,
    _step,
    make_env,
    replace_ego_with_idm,
)


def _lane_id(vehicle: Any) -> int | None:
    lane_index = getattr(vehicle, "lane_index", None)
    if isinstance(lane_index, tuple) and len(lane_index) >= 3:
        return int(lane_index[2])
    return None


def _nearest_distance(env: Any, ego: Any) -> float:
    ego_pos = np.asarray(ego.position, dtype=np.float32)
    best = math.inf
    for vehicle in getattr(env.unwrapped.road, "vehicles", []):
        if vehicle is ego:
            continue
        pos = np.asarray(vehicle.position, dtype=np.float32)
        dist = float(np.linalg.norm(pos - ego_pos))
        if dist < best:
            best = dist
    return best


def _episode_summary(
    env: Any,
    IDMVehicle: Any,
    *,
    seed: int,
    max_steps: int,
) -> Dict[str, Any]:
    _reset(env, seed)
    ego = replace_ego_with_idm(env, IDMVehicle)
    idle = _idle_action(env)

    xs: List[float] = []
    ys: List[float] = []
    speeds: List[float] = []
    lanes: List[int | None] = []
    nearest: List[float] = []
    crashed = False
    terminated = False
    truncated = False

    for _ in range(max_steps):
        ego = env.unwrapped.vehicle
        pos = np.asarray(ego.position, dtype=np.float32)
        vel = np.asarray(ego.velocity, dtype=np.float32)
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        speeds.append(float(np.linalg.norm(vel)))
        lanes.append(_lane_id(ego))
        nearest.append(_nearest_distance(env, ego))

        _, _, term, trunc, info = _step(env, idle)
        crashed = bool(info.get("crashed", False)) or bool(getattr(ego, "crashed", False))
        terminated = bool(term)
        truncated = bool(trunc)
        if term or trunc:
            break

    lane_changes = 0
    last_lane = lanes[0] if lanes else None
    for lane in lanes[1:]:
        if lane is not None and last_lane is not None and lane != last_lane:
            lane_changes += 1
        if lane is not None:
            last_lane = lane

    return {
        "seed": seed,
        "steps": len(xs),
        "seconds": len(xs) / float(POLICY_FREQ),
        "crashed": crashed,
        "terminated": terminated,
        "truncated": truncated,
        "lane_start": lanes[0] if lanes else None,
        "lane_end": lanes[-1] if lanes else None,
        "lane_changes": lane_changes,
        "x0": xs[0] if xs else float("nan"),
        "xT": xs[-1] if xs else float("nan"),
        "y0": ys[0] if ys else float("nan"),
        "yT": ys[-1] if ys else float("nan"),
        "y_min": min(ys) if ys else float("nan"),
        "y_max": max(ys) if ys else float("nan"),
        "speed_min": min(speeds) if speeds else float("nan"),
        "speed_mean": float(np.mean(speeds)) if speeds else float("nan"),
        "speed_max": max(speeds) if speeds else float("nan"),
        "nearest_min": min(nearest) if nearest else float("nan"),
        "lane_trace": lanes[:30],
        "y_trace": ys[:30],
        "speed_trace": speeds[:30],
    }


def _print_episode(summary: Dict[str, Any]) -> None:
    status = "CRASH" if summary["crashed"] else "OK"
    print(
        f"  seed={summary['seed']} {status} "
        f"steps={summary['steps']:3d} ({summary['seconds']:.1f}s) "
        f"lane={summary['lane_start']}->{summary['lane_end']} "
        f"lc={summary['lane_changes']} "
        f"x={summary['x0']:.1f}->{summary['xT']:.1f} "
        f"y={summary['y0']:.2f}->{summary['yT']:.2f} "
        f"v_mean={summary['speed_mean']:.2f}m/s "
        f"v_min={summary['speed_min']:.2f} "
        f"nearest_min={summary['nearest_min']:.2f}m"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", type=str, default="highway-slow-leader-v0")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--vehicles-count", type=int, default=50)
    ap.add_argument("--lanes-count", type=int, default=4)
    ap.add_argument("--trace", action="store_true",
                    help="Print first 30 lane/y/speed values for each episode.")
    args = ap.parse_args()

    gym, IDMVehicle = _import_highway_env()
    env = make_env(
        gym,
        args.env,
        vehicles_count=args.vehicles_count,
        lanes_count=args.lanes_count,
    )

    print(f"Inspecting IDM ego in {args.env}")
    print(f"  episodes={args.episodes} max_steps={args.max_steps} seed0={args.seed}")

    summaries = []
    try:
        for ep in range(args.episodes):
            summary = _episode_summary(
                env,
                IDMVehicle,
                seed=args.seed + ep,
                max_steps=args.max_steps,
            )
            summaries.append(summary)
            _print_episode(summary)
            if args.trace:
                print(f"    lane trace: {summary['lane_trace']}")
                y_trace = ", ".join(f"{y:.2f}" for y in summary["y_trace"])
                v_trace = ", ".join(f"{v:.1f}" for v in summary["speed_trace"])
                print(f"    y trace:    [{y_trace}]")
                print(f"    speed:      [{v_trace}]")
    finally:
        env.close()

    if not summaries:
        return

    crash_rate = sum(1 for s in summaries if s["crashed"]) / len(summaries)
    lane_changes = [s["lane_changes"] for s in summaries]
    speed_means = [s["speed_mean"] for s in summaries]
    nearest_mins = [s["nearest_min"] for s in summaries]
    print("\nSummary")
    print(f"  crash_rate:       {100.0 * crash_rate:.1f}%")
    print(f"  lane_changes/ep:  {float(np.mean(lane_changes)):.2f}")
    print(f"  mean speed:       {float(np.mean(speed_means)):.2f} m/s")
    print(f"  min clearance:    {float(np.mean(nearest_mins)):.2f} m")

    if crash_rate < 1.0 and float(np.mean(lane_changes)) > 0.0:
        print("  verdict: IDM has an avoidance maneuver worth adding to Stage 1 data.")
    elif crash_rate < 1.0:
        print("  verdict: IDM avoids without lane changes; Stage 1 can learn braking here.")
    else:
        print("  verdict: IDM also fails here; augmenting Stage 1 with this exact scenario will not teach avoidance.")


if __name__ == "__main__":
    main()
