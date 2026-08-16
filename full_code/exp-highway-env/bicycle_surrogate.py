"""
bicycle_surrogate.py

Autograd-friendly bicycle dynamics that mirror highway-env's Vehicle.step().
Two variants are provided:

  bicycle_step_train  : hard speed clip, matching deployment dynamics. Use
                        with the integrator's straight-through action clip.
  bicycle_step_deploy : hard clip on speed (matches highway-env exactly).
                        Use at evaluation time, when we plug Stage 2 into the
                        live env. Gradients zero at the boundary, but we don't
                        need gradients there.

Plus the force → action decomposition that maps a 2-D world-frame force to
(acceleration, steering) commands compatible with ContinuousAction in
highway-env.

Constants match highway-env exactly. See:
  HighwayEnv-master/highway_env/vehicle/kinematics.py:130-152
  HighwayEnv-master/highway_env/envs/common/action.py:80-86
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import torch


# ──────────────────────────────────────────────────────────────────────────
# Constants — match highway-env defaults
# ──────────────────────────────────────────────────────────────────────────

VEHICLE_LENGTH = 5.0       # highway_env Vehicle.LENGTH
ACCEL_RANGE = (-5.0, 5.0)  # ContinuousAction.ACCELERATION_RANGE  m/s²
STEER_RANGE = (-math.pi / 4, math.pi / 4)  # ContinuousAction.STEERING_RANGE  rad
MIN_SPEED = -40.0          # Vehicle.MIN_SPEED  m/s
MAX_SPEED = 40.0           # Vehicle.MAX_SPEED  m/s


# ──────────────────────────────────────────────────────────────────────────
# Bicycle update — train and deploy variants
# ──────────────────────────────────────────────────────────────────────────

def _bicycle_core(
    pos: torch.Tensor,
    heading: torch.Tensor,
    speed: torch.Tensor,
    accel: torch.Tensor,
    steer: torch.Tensor,
    dt: float,
    length: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The pre-saturation bicycle update. Identical for train and deploy."""
    beta = torch.atan(0.5 * torch.tan(steer))                # slip angle
    cos_h = torch.cos(heading + beta)
    sin_h = torch.sin(heading + beta)
    v_world = speed.unsqueeze(-1) * torch.stack([cos_h, sin_h], dim=-1)  # (..., 2)
    new_pos = pos + v_world * dt
    new_heading = heading + speed * torch.sin(beta) / (0.5 * length) * dt
    raw_speed = speed + accel * dt
    return new_pos, new_heading, raw_speed


def bicycle_step_train(
    pos: torch.Tensor,
    heading: torch.Tensor,
    speed: torch.Tensor,
    accel: torch.Tensor,
    steer: torch.Tensor,
    dt: float,
    length: float = VEHICLE_LENGTH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bicycle update with the same hard speed clip used by highway-env.

    Earlier versions used ``MAX_SPEED * tanh(raw_speed / MAX_SPEED)`` as a
    smooth training envelope. At highway speeds (25-30 m/s), that envelope is
    already strongly damping, which creates a train/deploy mismatch: full
    throttle looks harmless in training and becomes a crash in the live env.
    """
    new_pos, new_heading, raw_speed = _bicycle_core(
        pos, heading, speed, accel, steer, dt, length
    )
    new_speed = raw_speed.clamp(MIN_SPEED, MAX_SPEED)
    return new_pos, new_heading, new_speed


def bicycle_step_deploy(
    pos: torch.Tensor,
    heading: torch.Tensor,
    speed: torch.Tensor,
    accel: torch.Tensor,
    steer: torch.Tensor,
    dt: float,
    length: float = VEHICLE_LENGTH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bicycle update with hard speed clip. Matches highway-env exactly."""
    new_pos, new_heading, raw_speed = _bicycle_core(
        pos, heading, speed, accel, steer, dt, length
    )
    new_speed = raw_speed.clamp(MIN_SPEED, MAX_SPEED)
    return new_pos, new_heading, new_speed


# ──────────────────────────────────────────────────────────────────────────
# Force → action decomposition
# ──────────────────────────────────────────────────────────────────────────

def force_to_action(
    F: torch.Tensor,         # (B, 2)  force in world frame  m/s²  (unit-mass)
    heading: torch.Tensor,   # (B,)    radians
    speed: torch.Tensor,     # (B,)    m/s
    length: float = VEHICLE_LENGTH,
    eps_speed: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map a 2-D unit-mass force to (acceleration, steering).

    Decomposition (consistent with the unit-mass surrogate convention used
    by `train_material.py:integrate_surrogate_material`):

        F_lon  =  F · (cos θ, sin θ)             longitudinal in ego frame
        F_lat  =  F · (-sin θ, cos θ)            lateral
        accel  =  F_lon                           directly the acceleration
        steer  =  arcsin(F_lat * L / (v² + ε²))   small-angle bicycle inverse

    The eps_speed term is essential: without it, sin(δ) blows up at v=0,
    making lane-changes from a stop physically impossible to express. With
    eps_speed=1.0 m/s, the inverse remains finite but is correctly weak at
    low speed (you can't aggressively steer a stopped car).

    Outputs are NOT clipped to action ranges here. Caller decides whether
    to clip (deploy) or pass through (training, where clip → zero gradient).
    """
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    F_lon = F[:, 0] * cos_h + F[:, 1] * sin_h            # (B,)
    F_lat = -F[:, 0] * sin_h + F[:, 1] * cos_h           # (B,)

    accel = F_lon
    v_eff_sq = speed * speed + eps_speed * eps_speed
    sin_delta = (F_lat * length / v_eff_sq).clamp(-0.999, 0.999)
    steer = torch.asin(sin_delta)
    return accel, steer


def surrogate_rollout(
    pos0: torch.Tensor,       # (B, 2)
    heading0: torch.Tensor,   # (B,)
    speed0: torch.Tensor,     # (B,)
    forces_fn: Callable[[dict, int], torch.Tensor],
    H: int,
    dt: float,
    *,
    train: bool = True,
    clip_actions: bool = False,
    length: float = VEHICLE_LENGTH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll out bicycle dynamics under a differentiable force callback.

    `forces_fn(state, t)` receives the current state dict and timestep, and
    returns a world-frame force tensor `(B, 2)`. Keeping this as a callback is
    important for Stage 2: material forces depend on the current position
    because risk/SDF gradients are resampled along the rollout.

    Returns
    -------
    pos_traj:
        `(B, H+1, 2)` positions, including the initial state.
    heading_traj:
        `(B, H+1)` headings.
    speed_traj:
        `(B, H+1)` speeds.
    """
    pos = pos0
    heading = heading0
    speed = speed0
    pos_hist = [pos]
    heading_hist = [heading]
    speed_hist = [speed]
    step_fn = bicycle_step_train if train else bicycle_step_deploy

    for t in range(int(H)):
        state = {"pos": pos, "heading": heading, "speed": speed}
        force = forces_fn(state, t)
        accel, steer = force_to_action(force, heading, speed, length=length)
        if clip_actions:
            accel = accel.clamp(*ACCEL_RANGE)
            steer = steer.clamp(*STEER_RANGE)
        pos, heading, speed = step_fn(
            pos, heading, speed, accel, steer, dt=dt, length=length
        )
        pos_hist.append(pos)
        heading_hist.append(heading)
        speed_hist.append(speed)

    return (
        torch.stack(pos_hist, dim=1),
        torch.stack(heading_hist, dim=1),
        torch.stack(speed_hist, dim=1),
    )
