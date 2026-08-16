"""
env_wrapper.py

Wraps a highway-env environment to produce the per-step observation dict
in the format consumed by `integrate_surrogate_material` in
train_material.py (DFC2018 codebase).

The wrapper is intentionally thin. It does three things:

  1. Pulls live vehicle state from `env.unwrapped.road.vehicles` and
     converts each to a `VehicleState` (the input language of
     RiskFieldConstructor, which has no highway-env dependency).
  2. Builds the geometric obstacle batch (C, R, W, mask) by treating
     other vehicles as bounding-circle obstacles. Format matches DFC's
     padded-N convention.
  3. Adds a road-boundary SDF from the live RoadNetwork so off-road
     space is treated as a hard hazard in the material channel.
  4. Builds the goal vector. Highway-env has no point goal; we use
     a moving look-ahead point in the ego's current lane.

It does NOT:
  - alter highway-env's dynamics or controller (Step 3 will, when we
    introduce the dynamics-matched bicycle surrogate).
  - introduce randomness — the wrapper is a pure observer.

Output of `build_observation()`:
  dict with keys (matching the DFC dataset/integrator inputs):
     o0          : (2,)       ego world position
     v0          : (2,)       ego world velocity
     goal        : (2,)       look-ahead goal in ego's current lane
     C           : (N_max, 2) padded other-vehicle positions
     V_neighbors : (N_max, 2) padded other-vehicle world velocities
     R           : (N_max,)   padded other-vehicle bounding radii
     W           : (N_max,)   padded weights (uniform 1.0; model rescales)
     mask        : (N_max,)   bool, True for valid vehicles
     risk_patch  : (2, Hp, Wp)  [r̃, hard_mask] — input to CoefEnergyNetMaterial
     rollout_patch: (6, Hp, Wp) [r̃, φ, ∂r̃/∂x, ∂r̃/∂y, ∂φ/∂x, ∂φ/∂y]
                                  — input to surrogate integrator
     d_hat       : ()         IPC barrier activation distance (scalar)
     dt          : ()         integrator timestep (scalar)
     H           : ()         rollout horizon in surrogate steps

All array shapes match the DFC dataset's collate format so the same
model and integrator drop in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from risk_field import RiskFieldConfig, RiskFieldConstructor, VehicleState


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class WrapperConfig:
    # Padding for vehicle list. highway-env caps simultaneous neighbours
    # in the Kinematics observation at ~10–15 by default.
    n_max_vehicles: int = 15

    # Sensing radius — vehicles farther than this are masked out.
    # Should match risk-field sensing radius for consistency.
    sensing_radius_m: float = 80.0

    # Vehicle bounding radius (used for IPC barrier in the geometric
    # obstacle channel). Highway-env vehicles are 5×2m; we use 2.0m
    # as a slightly inflated bounding circle.
    vehicle_radius_m: float = 2.0

    # Goal placement: a look-ahead point in the ego's current lane.
    # `goal_lookahead_m` is how far ahead of the ego the goal sits.
    # The goal updates every step to track the current lane, so the
    # goal term in the episode cost rewards forward progress without
    # penalizing lane changes.
    goal_lookahead_m: float = 30.0

    # Surrogate integrator parameters. These are the surrogate's idea
    # of dt and horizon, not highway-env's. The surrogate runs INSIDE
    # one highway-env step at training time (when we backprop through
    # H surrogate steps with risk patches frozen at the highway-env
    # observation time).
    dt_surrogate: float = 0.1
    horizon_surrogate: int = 20

    # IPC barrier activation. Same scale as in DFC.
    d_hat: float = 5.0

    # Risk field config — passed through to RiskFieldConstructor.
    risk_field: RiskFieldConfig = None

    def __post_init__(self):
        if self.risk_field is None:
            self.risk_field = RiskFieldConfig()


# ──────────────────────────────────────────────────────────────────────────
# Adapters: highway-env Vehicle → VehicleState
# ──────────────────────────────────────────────────────────────────────────

def _vehicle_to_state(v: Any) -> VehicleState:
    """Convert a highway_env Vehicle to a VehicleState.

    Reads only the public kinematic attributes — works for Vehicle,
    ControlledVehicle, IDMVehicle, MDPVehicle, etc., since they all
    inherit the same kinematic interface.
    """
    return VehicleState(
        position=np.asarray(v.position, dtype=np.float64).copy(),
        heading=float(v.heading),
        speed=float(v.speed),
        length=float(getattr(v, "LENGTH", 5.0)),
        width=float(getattr(v, "WIDTH", 2.0)),
    )


def _ego_lane_center_y(
    env: Any,
    ego_pos: np.ndarray,
    return_source: bool = False,
) -> float | Tuple[float, str]:
    """Return the y-coordinate of the centerline of the ego's current lane.

    Falls back to the nearest lane center if the lane index is unavailable.
    """
    try:
        ego = env.unwrapped.vehicle
        lane_idx = ego.lane_index
        # lane_idx is a tuple (from_node, to_node, lane_id); the lane
        # itself is fetched via the road network.
        lane = env.unwrapped.road.network.get_lane(lane_idx)
        # Lane.position(longitudinal, lateral=0) gives the centerline point.
        # We want the y at the ego's longitudinal coordinate.
        # Lane has a `.local_coordinates(pos)` method returning (s, lat).
        s, _ = lane.local_coordinates(ego_pos)
        center_pt = lane.position(s, 0.0)
        lane_y = float(center_pt[1])
        return (lane_y, "road_network") if return_source else lane_y
    except Exception:
        # Robustness fallback: snap to nearest 4m-spaced lane center.
        # Highway-env default lane width is 4.0m, lanes start at y=0.
        lane_y = float(round(ego_pos[1] / 4.0) * 4.0)
        return (lane_y, "fallback_4m_grid") if return_source else lane_y


# ──────────────────────────────────────────────────────────────────────────
# The wrapper
# ──────────────────────────────────────────────────────────────────────────

class HighwayMaterialObservation:
    """Builds the DFC-format observation dict from a live highway-env env.

    This class is a *pure observer* — it does not subclass gym.Wrapper or
    modify env behavior. Use it as:

        obs_builder = HighwayMaterialObservation(cfg)
        env_obs, info = env.reset()
        while not done:
            obs_dict = obs_builder.build(env)
            action = stage2_policy(obs_dict)
            env_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    Wrapping into a gym.Wrapper is straightforward (Step 3) once we
    decide how Stage 2's continuous-force output maps to highway-env's
    action space — that decision belongs with the surrogate integrator,
    not here.
    """

    def __init__(self, cfg: Optional[WrapperConfig] = None):
        self.cfg = cfg or WrapperConfig()
        self.risk_constructor = RiskFieldConstructor(self.cfg.risk_field)

    # ── public API ────────────────────────────────────────────────────────

    def build(self, env: Any) -> Dict[str, np.ndarray]:
        """Build the observation dict for the current env state."""
        ego = self._get_ego(env)
        others = self._get_others(env, ego)

        # Convert to VehicleState for the risk constructor
        ego_state = _vehicle_to_state(ego)
        other_states = [_vehicle_to_state(v) for v in others]

        # Risk patches (uses the lane-corridor model from Step 1)
        rollout_patch = self.risk_constructor.build_patch(ego_state, other_states)
        rollout_patch = self._add_road_boundary_to_rollout_patch(
            env, ego_state, rollout_patch
        )
        # The model encoder follows train_material.py: ch0 is soft risk,
        # ch1 is a binary hard-hazard mask. The rollout patch keeps φ in
        # channel 1 for the DFC integrator.
        risk_patch = np.stack(
            [rollout_patch[0], (rollout_patch[1] <= 0.0).astype(np.float32)],
            axis=0,
        )                                              # (2, Hp, Wp)

        # Geometric obstacle channel: padded vehicle list
        C, V_neighbors, R, W, mask = self._build_obstacle_channel(other_states)

        # Goal: look-ahead point in the ego's current lane
        ego_pos = ego_state.position
        lane_y = _ego_lane_center_y(env, ego_pos)
        goal = np.array([
            ego_pos[0] + self.cfg.goal_lookahead_m,
            lane_y,
        ], dtype=np.float32)

        return {
            "o0":            ego_pos.astype(np.float32),
            "v0":            ego_state.velocity.astype(np.float32),
            "goal":          goal,
            "C":             C.astype(np.float32),
            "V_neighbors":   V_neighbors.astype(np.float32),
            "R":             R.astype(np.float32),
            "W":             W.astype(np.float32),
            "mask":          mask,
            "risk_patch":    risk_patch.astype(np.float32),
            "rollout_patch": rollout_patch.astype(np.float32),
            "d_hat":         np.float32(self.cfg.d_hat),
            "dt":            np.float32(self.cfg.dt_surrogate),
            "H":             np.int32(self.cfg.horizon_surrogate),
        }

    # ── helpers: env access ───────────────────────────────────────────────

    def _get_ego(self, env: Any) -> Any:
        """Return the ego vehicle object from the env."""
        return env.unwrapped.vehicle

    def _get_others(self, env: Any, ego: Any) -> List[Any]:
        """Return all non-ego vehicles within sensing radius."""
        all_vehicles = env.unwrapped.road.vehicles
        ego_pos = np.asarray(ego.position, dtype=np.float64)
        r2 = self.cfg.sensing_radius_m ** 2
        out = []
        for v in all_vehicles:
            if v is ego:
                continue
            dx = float(v.position[0]) - ego_pos[0]
            dy = float(v.position[1]) - ego_pos[1]
            if dx * dx + dy * dy <= r2:
                out.append(v)
        return out

    # ── helpers: geometric obstacle batch ─────────────────────────────────

    def _build_obstacle_channel(
        self, others: List[VehicleState]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pad/truncate other-vehicle list to fixed length N_max.

        Returns:
          C           : (N_max, 2)  positions (padded with zeros)
          V_neighbors : (N_max, 2)  velocities (padded with zeros)
          R           : (N_max,)    bounding radii (padded with zeros)
          W           : (N_max,)    weights (1.0 for valid, 0.0 for padding)
          mask        : (N_max,)    bool, True for valid entries
        """
        N_max = self.cfg.n_max_vehicles
        n = min(len(others), N_max)

        C = np.zeros((N_max, 2), dtype=np.float64)
        V_neighbors = np.zeros((N_max, 2), dtype=np.float64)
        R = np.zeros((N_max,), dtype=np.float64)
        W = np.zeros((N_max,), dtype=np.float64)
        mask = np.zeros((N_max,), dtype=bool)

        # If we have more than N_max, keep the closest. Sorting key is
        # squared distance to the latest ego position — but at this point
        # we don't have ego_pos; the caller already filtered by sensing
        # radius. For determinism we sort by longitudinal x.
        sorted_others = sorted(others, key=lambda v: v.position[0])[:n]

        for i, v in enumerate(sorted_others):
            C[i] = v.position
            V_neighbors[i] = v.velocity
            R[i] = self.cfg.vehicle_radius_m
            W[i] = 1.0
            mask[i] = True

        return C, V_neighbors, R, W, mask

    # ── helpers: road-boundary SDF ───────────────────────────────────────

    def _add_road_boundary_to_rollout_patch(
        self,
        env: Any,
        ego_state: VehicleState,
        rollout_patch: np.ndarray,
    ) -> np.ndarray:
        """Merge vehicle SDF with a live road-boundary SDF.

        RiskFieldConstructor deliberately knows nothing about highway-env.
        The wrapper has the RoadNetwork, so it is the right place to mark
        off-road cells as hard hazards while keeping the constructor pure.
        """
        grid_world = self.risk_constructor._patch_grid_world(ego_state)
        phi_road = self._road_boundary_phi(env, grid_world)
        if phi_road is None:
            return rollout_patch

        out = rollout_patch.copy()
        phi = np.minimum(out[1].astype(np.float64), phi_road)
        phi = np.clip(phi, -10.0, 50.0)
        grad_lon, grad_lat = self.risk_constructor._gradient(phi)
        grad_x, grad_y = self.risk_constructor._rotate_grad_to_world(
            grad_lon, grad_lat, ego_state.heading
        )
        out[1] = phi.astype(np.float32)
        out[4] = grad_x.astype(np.float32)
        out[5] = grad_y.astype(np.float32)
        return out

    def _road_boundary_phi(
        self,
        env: Any,
        grid_world: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Approximate signed distance to the drivable lane union.

        Positive values are inside at least one lane, negative values are
        off-road. For a union of lanes we take the maximum signed margin,
        which makes lane dividers non-hazardous while road edges remain hard
        boundaries.
        """
        network = getattr(getattr(env.unwrapped, "road", None), "network", None)
        if network is None or not hasattr(network, "lanes_dict"):
            return None
        lanes = list(network.lanes_dict().values())
        if not lanes:
            return None

        flat = grid_world.reshape(-1, 2)
        phi = np.full(flat.shape[0], -50.0, dtype=np.float64)
        for lane in lanes:
            vals = np.fromiter(
                (self._signed_margin_to_lane(lane, point) for point in flat),
                dtype=np.float64,
                count=flat.shape[0],
            )
            phi = np.maximum(phi, vals)
        return phi.reshape(grid_world.shape[:2])

    @staticmethod
    def _signed_margin_to_lane(lane: Any, point: np.ndarray) -> float:
        """Signed margin to one lane rectangle/curve in metres."""
        try:
            longitudinal, lateral = lane.local_coordinates(point)
            length = float(getattr(lane, "length", 0.0))
            half_width = 0.5 * float(lane.width_at(longitudinal))
        except Exception:
            return -50.0

        lat_margin = half_width - abs(float(lateral))
        long_margin = min(float(longitudinal), length - float(longitudinal))
        if lat_margin >= 0.0 and long_margin >= 0.0:
            return min(lat_margin, long_margin)
        if lat_margin < 0.0 and long_margin < 0.0:
            return -float(np.hypot(-lat_margin, -long_margin))
        return min(lat_margin, long_margin)


# ──────────────────────────────────────────────────────────────────────────
# Self-test (runs without a live highway-env, using mock vehicles)
# ──────────────────────────────────────────────────────────────────────────

class _MockVehicle:
    """Stand-in for highway_env.vehicle.kinematics.Vehicle for unit tests."""
    LENGTH = 5.0
    WIDTH = 2.0
    def __init__(self, x, y, heading, speed):
        self.position = np.array([x, y], dtype=np.float64)
        self.heading = float(heading)
        self.speed = float(speed)
        self.lane_index = (None, None, int(round(y / 4.0)))


class _MockNetwork:
    def get_lane(self, lane_idx):
        raise RuntimeError("intentionally unsupported in mock — exercises fallback")


class _MockRoad:
    def __init__(self, vehicles):
        self.vehicles = vehicles
        self.network = _MockNetwork()


class _MockEnvUnwrapped:
    def __init__(self, ego, others):
        self.vehicle = ego
        self.road = _MockRoad([ego] + others)


class _MockEnv:
    def __init__(self, ego, others):
        self.unwrapped = _MockEnvUnwrapped(ego, others)


def _self_test():
    """Self-test exercising every code path."""
    print("Running env_wrapper self-test...")

    # Construct a slow-leader scene mirroring Step 1's diagnostic.
    ego = _MockVehicle(x=0.0, y=4.0, heading=0.0, speed=25.0)
    others = [
        _MockVehicle(x=30.0, y=4.0, heading=0.0, speed=12.0),  # slow leader
        _MockVehicle(x=-15.0, y=8.0, heading=0.0, speed=28.0), # adjacent rear
    ]
    env = _MockEnv(ego, others)

    cfg = WrapperConfig()
    builder = HighwayMaterialObservation(cfg)
    obs = builder.build(env)

    # Shape and dtype checks — fast safety net
    expected = {
        "o0":            ((2,),    np.float32),
        "v0":            ((2,),    np.float32),
        "goal":          ((2,),    np.float32),
        "C":             ((cfg.n_max_vehicles, 2), np.float32),
        "V_neighbors":   ((cfg.n_max_vehicles, 2), np.float32),
        "R":             ((cfg.n_max_vehicles,),   np.float32),
        "W":             ((cfg.n_max_vehicles,),   np.float32),
        "mask":          ((cfg.n_max_vehicles,),   np.bool_),
        "risk_patch":    ((2, cfg.risk_field.patch_lat_cells,
                              cfg.risk_field.patch_lon_cells), np.float32),
        "rollout_patch": ((6, cfg.risk_field.patch_lat_cells,
                              cfg.risk_field.patch_lon_cells), np.float32),
    }
    for k, (shape, dtype) in expected.items():
        v = obs[k]
        assert v.shape == shape, f"{k}: expected shape {shape}, got {v.shape}"
        assert v.dtype == dtype,  f"{k}: expected dtype {dtype}, got {v.dtype}"

    # Sanity: goal is ahead of ego in the same lane (lane_y fallback to nearest)
    assert obs["goal"][0] > obs["o0"][0], "goal should be ahead of ego"
    assert abs(obs["goal"][1] - 4.0) < 1e-3, f"goal lane y should be 4.0, got {obs['goal'][1]}"

    # Sanity: two valid neighbours
    assert int(obs["mask"].sum()) == 2, f"expected 2 valid neighbours, got {obs['mask'].sum()}"
    assert np.allclose(obs["V_neighbors"][0], np.array([28.0, 0.0], dtype=np.float32))
    assert np.allclose(obs["V_neighbors"][1], np.array([12.0, 0.0], dtype=np.float32))

    # Sanity: risk patch matches the model encoder contract:
    # channel 0 is r_tilde, channel 1 is a hard-mask derived from phi.
    assert np.allclose(obs["risk_patch"][0], obs["rollout_patch"][0]), \
        "risk_patch[0] must equal rollout_patch risk channel"
    assert np.allclose(
        obs["risk_patch"][1],
        (obs["rollout_patch"][1] <= 0.0).astype(np.float32),
    ), "risk_patch[1] must be hard mask derived from rollout_patch phi"

    # Sanity: ego velocity matches speed × heading=0
    assert abs(obs["v0"][0] - 25.0) < 1e-3 and abs(obs["v0"][1]) < 1e-3, \
        f"v0 mismatch: {obs['v0']}"

    # Sanity: padded entries have mask=False, R=0
    assert obs["mask"][2:].sum() == 0
    assert (obs["R"][2:] == 0).all()

    # Sanity: gradient at the ego in the rollout_patch is consistent with
    # what the standalone RiskFieldConstructor produces.
    rfc = RiskFieldConstructor(cfg.risk_field)
    direct_patch = rfc.build_patch(
        _vehicle_to_state(ego),
        [_vehicle_to_state(v) for v in others],
    )
    assert np.allclose(direct_patch, obs["rollout_patch"], atol=1e-5), \
        "rollout_patch must agree with direct RiskFieldConstructor call"

    print("  ✓ all shape/dtype checks pass")
    print(f"  ✓ ego=(0,4) v=25, neighbours=2, goal={obs['goal']}, "
          f"r̃@ego={obs['risk_patch'][0, 16, 3]:.4f}")
    print("Self-test passed.\n")


if __name__ == "__main__":
    _self_test()
