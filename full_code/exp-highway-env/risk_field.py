"""
risk_field.py

Risk field construction for highway-env-style scenarios.

Builds (r̃, φ) and their gradients over a BEV patch centered on the ego
vehicle, given the kinematic state of nearby vehicles. Designed to mirror
the 6-channel `rollout_patch` format consumed by `integrate_surrogate_material`
in the DFC2018 codebase, so the same surrogate integrator can be reused.

Patch channel layout (matches DFC code):
    ch 0 : r̃     soft material risk in [0, 1]
    ch 1 : φ     signed distance to hard hazards (metres, ≥0 outside)
    ch 2 : ∂r̃/∂x  (along world-x = longitudinal)
    ch 3 : ∂r̃/∂y  (along world-y = lateral)
    ch 4 : ∂φ/∂x
    ch 5 : ∂φ/∂y

Design choices (justified inline):

1. **Anisotropic patch.** Highway scenarios are dominated by longitudinal
   structure (slow leader 30m ahead). A square patch wastes resolution.
   We default to (longitudinal, lateral) = (64m, 32m), patch resolution
   (Hp, Wp) = (32, 64) so each cell is roughly (1m lateral × 1m longitudinal).

2. **Integrated lookahead.** Risk from each other vehicle is integrated
   over a short prediction horizon with exponential decay. Single-lookahead
   constructions are brittle — a 2s lookahead misses leaders that are 30m
   ahead at 5 m/s closing speed (TTC = 6s).

3. **Anisotropic Gaussian per vehicle.** Variance along the predicted
   velocity is large (uncertainty in their future speed); variance
   perpendicular is small (lanes constrain lateral motion). This gives
   the right gradient direction at the ego — pointing out of the
   predicted-trajectory tube.

4. **Hard hazard SDF from same predictions.** Using the same predicted
   positions as the soft field keeps the two channels consistent.
   Otherwise F_soft and F_hard can fight each other.

The class is deliberately pure-numpy so it can be unit-tested without
spinning up highway-env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Vehicle state dataclass — the input "language" of the constructor
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class VehicleState:
    """Minimal per-vehicle state needed for risk field construction.

    Mirrors the fields exposed by highway_env.vehicle.kinematics.Vehicle:
      - position:  np.ndarray, shape (2,)  world (x, y)
      - heading:   float, radians (0 = +x axis)
      - speed:     float, m/s
    """
    position: np.ndarray
    heading: float
    speed: float
    length: float = 5.0   # highway-env Vehicle.LENGTH
    width: float  = 2.0   # highway-env Vehicle.WIDTH

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class RiskFieldConfig:
    # Patch geometry — anisotropic by default (see design note 1)
    patch_lon_m: float = 64.0      # length along ego heading (m)
    patch_lat_m: float = 32.0      # width perpendicular to heading (m)
    patch_lon_cells: int = 64
    patch_lat_cells: int = 32

    # Integrated lookahead (see design note 2)
    lookahead_max_s: float = 5.0     # T in the integrated-lookahead formula
    lookahead_steps: int = 6          # number of discrete τ samples in [0, T]
    lookahead_decay: float = 0.4      # β in exp(-β τ) — risk attenuation per second

    # Anisotropic Gaussian widths (design note 3)
    # σ_par scaled to highway leader-spacing (~30m typical) so the ego
    # sees nonzero gradient from a leader 1-2 vehicle-spacings ahead.
    sigma_perp_m: float = 1.8         # ~lane half-width — lateral uncertainty
    sigma_par_base_m: float = 6.0     # base longitudinal uncertainty (highway-scale)
    sigma_par_speed_coef: float = 0.5 # extra σ_par per (m/s) of relative speed

    # Hard hazard (design note 4)
    hazard_inflation_m: float = 0.5   # extra margin beyond vehicle bounding ellipse

    # Neighbour selection
    sensing_radius_m: float = 80.0    # only consider vehicles within this distance

    @property
    def cell_size_lon(self) -> float:
        return self.patch_lon_m / self.patch_lon_cells

    @property
    def cell_size_lat(self) -> float:
        return self.patch_lat_m / self.patch_lat_cells


# ──────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────

class RiskFieldConstructor:
    """Builds the 6-channel rollout patch for an ego vehicle in highway-env.

    The patch is in *ego-frame*: the longitudinal axis (length `patch_lon_m`)
    aligns with the ego's current heading, the lateral axis aligns with
    the perpendicular. The ego sits at the patch's "left-center" — i.e.,
    most of the longitudinal extent is *ahead* of the ego — because the
    decision-relevant risk lies forward.

    Public entry point: `build_patch(ego, others) -> np.ndarray of shape (6,Hp,Wp)`.
    """

    def __init__(self, cfg: Optional[RiskFieldConfig] = None):
        self.cfg = cfg or RiskFieldConfig()
        # Pre-compute the lookahead time samples and decay weights once
        if self.cfg.lookahead_steps == 1:
            self._tau_samples = np.array([self.cfg.lookahead_max_s])
        else:
            self._tau_samples = np.linspace(
                0.0, self.cfg.lookahead_max_s, self.cfg.lookahead_steps
            )
        self._tau_weights = np.exp(-self.cfg.lookahead_decay * self._tau_samples)
        # Normalize so that a static vehicle (no lookahead-induced drift)
        # contributes a Gaussian of unit peak — keeps r̃ roughly in [0,1].
        self._tau_weights = self._tau_weights / self._tau_weights.sum()

    # ── public API ────────────────────────────────────────────────────────

    def build_patch(
        self,
        ego: VehicleState,
        others: list[VehicleState],
    ) -> np.ndarray:
        """Construct the 6-channel patch for the current scene.

        Returns
        -------
        patch : np.ndarray, shape (6, Hp, Wp)
            ch0=r̃, ch1=φ, ch2=∂r̃/∂x_world, ch3=∂r̃/∂y_world,
            ch4=∂φ/∂x_world, ch5=∂φ/∂y_world.
            Gradients are returned in WORLD frame (the surrogate integrator
            consumes world-frame forces) even though the patch grid is in
            ego frame.
        """
        # Step 1: get patch-grid world coordinates
        grid_world = self._patch_grid_world(ego)              # (Hp, Wp, 2)

        # Step 2: filter neighbours by sensing radius
        nearby = self._filter_nearby(ego, others)

        # Step 3: build soft risk r̃ over the grid via integrated lookahead
        r_tilde = self._soft_risk_field(grid_world, nearby)   # (Hp, Wp)

        # Step 4: build hard SDF φ over the grid (using same predictions)
        phi = self._hard_sdf_field(grid_world, nearby)        # (Hp, Wp)

        # Step 5: numerical gradients in WORLD frame
        # The patch grid is in ego frame, but we want ∂r̃/∂x_world etc.
        # Compute gradients along patch axes, then rotate to world.
        gr_lon, gr_lat = self._gradient(r_tilde)              # along patch axes
        gp_lon, gp_lat = self._gradient(phi)
        gr_x, gr_y = self._rotate_grad_to_world(gr_lon, gr_lat, ego.heading)
        gp_x, gp_y = self._rotate_grad_to_world(gp_lon, gp_lat, ego.heading)

        patch = np.stack([r_tilde, phi, gr_x, gr_y, gp_x, gp_y], axis=0)
        return patch.astype(np.float32)

    # ── pieces, in order called above ─────────────────────────────────────

    def _patch_grid_world(self, ego: VehicleState) -> np.ndarray:
        """Return (Hp, Wp, 2) array of WORLD coordinates of each patch cell.

        Patch layout in ego frame:
          - longitudinal axis (length patch_lon_m, Wp cells) along ego heading
          - lateral axis (width patch_lat_m, Hp cells) perpendicular to heading
          - ego sits at longitudinal index ~5% of Wp (most patch is ahead)
          - lateral index Hp/2 is the ego's lane
        """
        cfg = self.cfg
        Hp, Wp = cfg.patch_lat_cells, cfg.patch_lon_cells

        # Ego frame coordinates of cell centers
        # Longitudinal: from -ego_offset to patch_lon_m - ego_offset
        ego_offset = 0.05 * cfg.patch_lon_m   # 5% of patch behind ego
        lon = np.linspace(-ego_offset, cfg.patch_lon_m - ego_offset,
                          Wp, endpoint=False) + 0.5 * cfg.cell_size_lon
        lat = np.linspace(-cfg.patch_lat_m/2, cfg.patch_lat_m/2,
                          Hp, endpoint=False) + 0.5 * cfg.cell_size_lat

        # Meshgrid in ego frame: lon along columns, lat along rows
        # (so the patch image displays "longitudinal = horizontal")
        Lon, Lat = np.meshgrid(lon, lat, indexing="xy")  # both (Hp, Wp)

        # Rotate ego frame → world frame
        c, s = np.cos(ego.heading), np.sin(ego.heading)
        World_x = ego.position[0] + c * Lon - s * Lat
        World_y = ego.position[1] + s * Lon + c * Lat

        return np.stack([World_x, World_y], axis=-1)        # (Hp, Wp, 2)

    def _filter_nearby(
        self, ego: VehicleState, others: list[VehicleState]
    ) -> list[VehicleState]:
        """Drop vehicles outside the sensing radius."""
        r2 = self.cfg.sensing_radius_m ** 2
        return [
            v for v in others
            if np.sum((v.position - ego.position) ** 2) <= r2
        ]

    def _soft_risk_field(
        self, grid_world: np.ndarray, others: list[VehicleState]
    ) -> np.ndarray:
        """Sum-of-anisotropic-Gaussians integrated over lookahead horizon."""
        Hp, Wp, _ = grid_world.shape
        if not others:
            return np.zeros((Hp, Wp), dtype=np.float64)

        # Peak contribution per vehicle is normalized to 1 (before sum).
        # We rely on ego seeing few enough neighbours that the sum stays
        # bounded; we also clip at the end.
        risk = np.zeros((Hp, Wp), dtype=np.float64)
        for v in others:
            risk += self._vehicle_risk_contribution(grid_world, v)

        # Squash to [0, 1] for compatibility with the DFC integrator's
        # `risk_val.clamp(0,1)`. We use a soft saturation (1 - exp(-x))
        # rather than hard clip so gradients remain informative beyond 1.
        return 1.0 - np.exp(-risk)

    def _vehicle_risk_contribution(
        self, grid_world: np.ndarray, v: VehicleState
    ) -> np.ndarray:
        """Lane-corridor risk model for one vehicle.

        Models the vehicle as occupying a *forward-extending corridor* in
        its current lane, swept over the lookahead horizon. Risk is high
        inside the corridor and falls off both longitudinally (away from
        the swept positions) and laterally (across the lane boundary).

        This is a better model than a radially-symmetric Gaussian because:
          - Cars stay in their lanes; risk is naturally a strip, not a blob.
          - An ego upstream in the same lane sees a clean lateral gradient
            pointing OUT of the lane at the corridor's lateral edges.
          - The longitudinal falloff still attenuates with TTC.

        Form:
            r̃_v(c) = sum_τ w(τ) · gauss_lon(d_par(τ)) · gauss_lat(d_perp)
        where:
            d_par(τ)  = signed distance from the τ-step predicted position,
                        projected onto the velocity direction
            d_perp    = signed distance perpendicular to velocity (≈ lateral
                        offset from the lane center)
            gauss_lon(d) = exp(-0.5 (d/σ_par)^2) but only on the upstream
                           side (d ≤ 0 means the ego is BEHIND the predicted
                           position — that's the dangerous side); we use a
                           half-Gaussian on the downstream side (d > 0 means
                           the ego is AHEAD of the prediction, which is safe)
                           with much faster falloff.
            gauss_lat(d) = exp(-0.5 (d/σ_perp)^2)
        """
        Hp, Wp, _ = grid_world.shape
        Gx = grid_world[..., 0]
        Gy = grid_world[..., 1]

        # Predicted velocity direction unit vector (constant-velocity model)
        vx, vy = v.velocity
        v_mag = np.hypot(vx, vy)
        if v_mag < 1e-3:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = vx / v_mag, vy / v_mag
        # Perpendicular unit vector (right-hand rule)
        px, py = -uy, ux

        sigma_par = self.cfg.sigma_par_base_m + self.cfg.sigma_par_speed_coef * v_mag
        sigma_perp = self.cfg.sigma_perp_m

        contrib = np.zeros((Hp, Wp), dtype=np.float64)
        for tau, w_tau in zip(self._tau_samples, self._tau_weights):
            cx = v.position[0] + tau * vx
            cy = v.position[1] + tau * vy
            dx = Gx - cx
            dy = Gy - cy
            d_par  = dx * ux + dy * uy   # signed: <0 means upstream of pred. pos
            d_perp = dx * px + dy * py

            # Asymmetric longitudinal falloff:
            #  - Upstream of predicted position (d_par <= 0): broad Gaussian
            #    on σ_par. This is where the *ego* sits relative to a leader.
            #  - Downstream (d_par > 0): much sharper falloff (factor 3×).
            #    Beyond a leader's predicted position, risk is low.
            sharp_factor = np.where(d_par > 0, 3.0, 1.0)
            lon_term = np.exp(-0.5 * (d_par / sigma_par * sharp_factor) ** 2)

            # Lateral falloff: standard Gaussian on σ_perp
            lat_term = np.exp(-0.5 * (d_perp / sigma_perp) ** 2)

            contrib += w_tau * lon_term * lat_term

        return contrib

    def _hard_sdf_field(
        self, grid_world: np.ndarray, others: list[VehicleState]
    ) -> np.ndarray:
        """SDF (in metres) to the nearest predicted vehicle bounding ellipse.

        We use the τ=0 prediction (current position) for the hazard SDF
        because collision is defined against the *current* vehicle bounding
        box, not its future position. The lookahead is reflected in the
        soft field; the hard field is geometric.
        """
        Hp, Wp, _ = grid_world.shape
        if not others:
            return np.full((Hp, Wp), 50.0, dtype=np.float64)  # cap matches DFC

        # Initialize to large sentinel
        phi = np.full((Hp, Wp), 50.0, dtype=np.float64)
        Gx = grid_world[..., 0]
        Gy = grid_world[..., 1]

        for v in others:
            # Vehicle bounding ellipse — semi-axes
            a = 0.5 * v.length + self.cfg.hazard_inflation_m   # along heading
            b = 0.5 * v.width  + self.cfg.hazard_inflation_m   # perpendicular
            ux, uy = np.cos(v.heading), np.sin(v.heading)
            px, py = -uy, ux

            dx = Gx - v.position[0]
            dy = Gy - v.position[1]
            # Project to vehicle frame
            d_par  = dx * ux + dy * uy
            d_perp = dx * px + dy * py
            # Approximate SDF to ellipse: r * (1 - 1/sqrt(...)) approximation
            # For our purposes a Mahalanobis-style distance is enough — this
            # is *not* a true SDF but it is monotone in distance to the
            # ellipse boundary, which is what the barrier needs.
            scaled = np.sqrt((d_par / a) ** 2 + (d_perp / b) ** 2)
            # Approximate metric SDF: (scaled - 1) * effective_radius
            eff_r = np.sqrt(a * b)
            this_phi = (scaled - 1.0) * eff_r
            phi = np.minimum(phi, this_phi)

        # Clamp to [-eff_r, 50] — DFC integrator clamps φ to [0, 50] but we
        # keep the negative tail visible for diagnostic plots, then clip
        # for the integrator separately at consume time.
        phi = np.clip(phi, -10.0, 50.0)
        return phi

    # ── numerical-helpers ─────────────────────────────────────────────────

    def _gradient(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Central differences on the patch.

        Returns (∂field/∂lon_ego, ∂field/∂lat_ego), where
        - lon_ego corresponds to columns (axis=1)
        - lat_ego corresponds to rows (axis=0)
        Both have units of [field] / metre.
        """
        d_lon = np.gradient(field, self.cfg.cell_size_lon, axis=1)
        d_lat = np.gradient(field, self.cfg.cell_size_lat, axis=0)
        return d_lon, d_lat

    def _rotate_grad_to_world(
        self, g_lon: np.ndarray, g_lat: np.ndarray, heading: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate (∂/∂lon_ego, ∂/∂lat_ego) → (∂/∂x_world, ∂/∂y_world)."""
        c, s = np.cos(heading), np.sin(heading)
        gx = c * g_lon - s * g_lat
        gy = s * g_lon + c * g_lat
        return gx, gy
