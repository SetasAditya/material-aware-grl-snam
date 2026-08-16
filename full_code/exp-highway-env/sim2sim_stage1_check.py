#!/usr/bin/env python3
"""
sim2sim_stage1_check.py

One-step consistency check for the protected Stage 1 checkpoint.

For fixed highway-v0 seeds, this script:
  1. Builds the same material observation used by eval_stage1.py.
  2. Computes the Stage 1 action through the checkpoint and deployment path.
  3. Steps a local bicycle surrogate with the same physical action.
  4. Steps live highway-env with the normalized action.
  5. Compares the resulting ego position, heading, and speed.

This is intentionally a narrow pre-Stage-2 check. It does not judge policy
quality; closed-loop eval already does that. It verifies that checkpoint
loading, action normalization, and the simulator/surrogate dynamics agree
well enough for Stage 2 risk-force training to reuse the same path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LOCAL_HIGHWAY_ENV = HERE / "HighwayEnv"
if LOCAL_HIGHWAY_ENV.exists():
    sys.path.insert(0, str(LOCAL_HIGHWAY_ENV))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from bicycle_surrogate import bicycle_step_deploy  # noqa: E402
from env_wrapper import HighwayMaterialObservation, WrapperConfig  # noqa: E402
from eval_stage1 import (  # noqa: E402
    DT_TARGET,
    POLICY_FREQ,
    SIM_FREQ,
    _import_gym,
    _reset,
    _step,
    compute_action,
    disable_transformer_nested_tensors,
    make_env,
)


def _angle_diff(a: float, b: float) -> float:
    return float(math.atan2(math.sin(a - b), math.cos(a - b)))


def _surrogate_step_from_state(
    pos: np.ndarray,
    heading: float,
    speed: float,
    accel: float,
    steer: float,
    *,
    sim_frequency: int,
    policy_frequency: int,
) -> Tuple[np.ndarray, float, float]:
    """Mirror highway-env's repeated substep integration for the ego."""
    pos_t = torch.as_tensor(pos, dtype=torch.float32).view(1, 2)
    heading_t = torch.tensor([heading], dtype=torch.float32)
    speed_t = torch.tensor([speed], dtype=torch.float32)
    accel_t = torch.tensor([accel], dtype=torch.float32)
    steer_t = torch.tensor([steer], dtype=torch.float32)
    frames = int(sim_frequency // policy_frequency)
    dt = 1.0 / float(sim_frequency)
    for _ in range(frames):
        pos_t, heading_t, speed_t = bicycle_step_deploy(
            pos_t, heading_t, speed_t, accel_t, steer_t, dt=dt
        )
    return (
        pos_t.squeeze(0).detach().cpu().numpy().astype(np.float64),
        float(heading_t.item()),
        float(speed_t.item()),
    )


@dataclass
class CheckCfg:
    ckpt: str
    env: str = "highway-v0"
    episodes: int = 5
    steps: int = 40
    seed: int = 1000
    vehicles_count: int = 50
    lanes_count: int = 4
    n_max_vehicles: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dfc_root: str = ""
    d_hat: float = 0.0
    alpha_floor: float = -1.0
    alpha_floor_ahead_only: bool | None = None
    pos_tol: float = 0.35
    speed_tol: float = 0.08
    heading_tol: float = 0.01
    out: str = ""


def _load_model(cfg: CheckCfg):
    if cfg.dfc_root:
        sys.path.insert(0, cfg.dfc_root)
    from train_material import CoefEnergyNetMaterial  # noqa: E402

    ck_path = Path(cfg.ckpt)
    if not ck_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ck_path}")
    ck = torch.load(ck_path, map_location=cfg.device, weights_only=False)
    train_cfg = ck.get("cfg", {})
    model = CoefEnergyNetMaterial(
        lam_soft_max=float(train_cfg.get("lam_soft_max", 50.0)),
        lam_hard_max=float(train_cfg.get("lam_hard_max", 10.0)),
    ).to(cfg.device)
    disable_transformer_nested_tensors(model)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing:
        print(f"Missing keys (using init): {missing}")
    if unexpected:
        print(f"Unexpected keys (ignored): {unexpected}")
    model.eval()
    return ck, train_cfg, model


def _runtime_knobs(cfg: CheckCfg, train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg.alpha_floor_ahead_only is None:
        ahead_only = bool(train_cfg.get("alpha_floor_ahead_only", False))
    else:
        ahead_only = bool(cfg.alpha_floor_ahead_only)
    return {
        "d_hat": cfg.d_hat if cfg.d_hat > 0 else float(train_cfg.get("d_hat", 0.0)),
        "alpha_floor": (
            cfg.alpha_floor if cfg.alpha_floor >= 0
            else float(train_cfg.get("alpha_floor", 0.0))
        ),
        "alpha_floor_ahead_only": ahead_only,
    }


def run_check(cfg: CheckCfg) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ck, train_cfg, model = _load_model(cfg)
    knobs = _runtime_knobs(cfg, train_cfg)

    gym = _import_gym()
    env = make_env(
        gym,
        cfg.env,
        vehicles_count=cfg.vehicles_count,
        lanes_count=cfg.lanes_count,
    )
    observer = HighwayMaterialObservation(WrapperConfig(
        n_max_vehicles=cfg.n_max_vehicles,
        dt_surrogate=DT_TARGET,
        horizon_surrogate=20,
    ))

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    try:
        for ep in range(cfg.episodes):
            seed = cfg.seed + ep
            _reset(env, seed)
            for step in range(cfg.steps):
                ego = env.unwrapped.vehicle
                pos0 = np.asarray(ego.position, dtype=np.float64).copy()
                heading0 = float(ego.heading)
                speed0 = float(ego.speed)

                action, diag = compute_action(
                    model,
                    observer,
                    env,
                    device=cfg.device,
                    stage=1,
                    d_hat_override=knobs["d_hat"],
                    alpha_floor=knobs["alpha_floor"],
                    alpha_floor_ahead_only=knobs["alpha_floor_ahead_only"],
                )
                pred_pos, pred_heading, pred_speed = _surrogate_step_from_state(
                    pos0,
                    heading0,
                    speed0,
                    diag["accel"],
                    diag["steer"],
                    sim_frequency=SIM_FREQ,
                    policy_frequency=POLICY_FREQ,
                )

                _, _, term, trunc, info = _step(env, action)
                pos1 = np.asarray(env.unwrapped.vehicle.position, dtype=np.float64).copy()
                heading1 = float(env.unwrapped.vehicle.heading)
                speed1 = float(env.unwrapped.vehicle.speed)

                pos_err = float(np.linalg.norm(pos1 - pred_pos))
                speed_err = abs(speed1 - pred_speed)
                heading_err = abs(_angle_diff(heading1, pred_heading))
                rows.append({
                    "episode": ep,
                    "seed": seed,
                    "step": step,
                    "pos_err_m": pos_err,
                    "speed_err_mps": speed_err,
                    "heading_err_rad": heading_err,
                    "env_x": float(pos1[0]),
                    "env_y": float(pos1[1]),
                    "sur_x": float(pred_pos[0]),
                    "sur_y": float(pred_pos[1]),
                    "env_speed": speed1,
                    "sur_speed": pred_speed,
                    "accel": float(diag["accel"]),
                    "steer": float(diag["steer"]),
                    "dmin": float(diag["dmin"]),
                    "collided": bool(info.get("crashed", False)),
                })
                if term or trunc:
                    break
    finally:
        env.close()

    if not rows:
        raise RuntimeError("No sim2sim rows collected.")

    pos = np.asarray([r["pos_err_m"] for r in rows], dtype=np.float64)
    speed = np.asarray([r["speed_err_mps"] for r in rows], dtype=np.float64)
    heading = np.asarray([r["heading_err_rad"] for r in rows], dtype=np.float64)
    passed = (
        float(pos.max()) <= cfg.pos_tol
        and float(speed.max()) <= cfg.speed_tol
        and float(heading.max()) <= cfg.heading_tol
    )
    summary = {
        "passed": passed,
        "n_steps": len(rows),
        "checkpoint": str(Path(cfg.ckpt)),
        "checkpoint_epoch": ck.get("epoch"),
        "runtime": knobs,
        "thresholds": {
            "pos_tol": cfg.pos_tol,
            "speed_tol": cfg.speed_tol,
            "heading_tol": cfg.heading_tol,
        },
        "pos_err_m": {
            "mean": float(pos.mean()),
            "p95": float(np.percentile(pos, 95)),
            "max": float(pos.max()),
        },
        "speed_err_mps": {
            "mean": float(speed.mean()),
            "p95": float(np.percentile(speed, 95)),
            "max": float(speed.max()),
        },
        "heading_err_rad": {
            "mean": float(heading.mean()),
            "p95": float(np.percentile(heading, 95)),
            "max": float(heading.max()),
        },
        "wall_clock_s": time.time() - t0,
    }
    return summary, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env", default="highway-v0")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--vehicles-count", type=int, default=50)
    ap.add_argument("--lanes-count", type=int, default=4)
    ap.add_argument("--n-max-vehicles", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dfc-root", default="")
    ap.add_argument("--d-hat", type=float, default=0.0)
    ap.add_argument("--alpha-floor", type=float, default=-1.0)
    ap.add_argument("--alpha-floor-all-obstacles",
                    dest="alpha_floor_ahead_only", action="store_false")
    ap.set_defaults(alpha_floor_ahead_only=None)
    ap.add_argument("--pos-tol", type=float, default=0.35)
    ap.add_argument("--speed-tol", type=float, default=0.08)
    ap.add_argument("--heading-tol", type=float, default=0.01)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = CheckCfg(**vars(args))
    summary, rows = run_check(cfg)

    print("Loaded Stage 1 checkpoint:", cfg.ckpt)
    print(f"Checked {summary['n_steps']} one-step transitions.")
    print("Runtime knobs:", summary["runtime"])
    print("\n── Sim2Sim Summary ──")
    print(f"  position error: mean={summary['pos_err_m']['mean']:.4f} m  "
          f"p95={summary['pos_err_m']['p95']:.4f} m  "
          f"max={summary['pos_err_m']['max']:.4f} m")
    print(f"  speed error:    mean={summary['speed_err_mps']['mean']:.4f} m/s  "
          f"p95={summary['speed_err_mps']['p95']:.4f} m/s  "
          f"max={summary['speed_err_mps']['max']:.4f} m/s")
    print(f"  heading error:  mean={summary['heading_err_rad']['mean']:.5f} rad  "
          f"p95={summary['heading_err_rad']['p95']:.5f} rad  "
          f"max={summary['heading_err_rad']['max']:.5f} rad")
    print(f"  result:         {'PASS' if summary['passed'] else 'FAIL'}")

    if cfg.out:
        out_path = Path(cfg.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "summary": summary,
                "config": asdict(cfg),
                "rows": rows,
            }, f, indent=2)
        print(f"\nWrote {out_path}")

    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
