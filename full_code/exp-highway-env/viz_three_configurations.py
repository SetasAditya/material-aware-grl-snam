#!/usr/bin/env python3
"""
Visual diagnostics for Step 1 + Step 2 on a real highway-env install.

This used to render hand-authored VehicleState scenes. It now creates real
highway-env environments, lets them step forward, builds the DFC-format
observation through HighwayMaterialObservation, and saves figures that expose
the actual bridge we care about:

  - live env lanes and vehicles from env.unwrapped.road
  - moving lane-center goal used by the wrapper
  - ego-frame patch projected back into world coordinates
  - soft risk r_tilde, model hard-mask, SDF phi
  - world-frame force directions -grad r_tilde and +grad phi

The most useful reviewer check is visual: the patch should sit in front of the
ego, lane_y should come from road_network on highway/merge envs, and gradients
should line up with actual nearby vehicles rather than with a synthetic scene.

Examples
--------
python exp-highway-env/viz_three_configurations.py
python exp-highway-env/viz_three_configurations.py --env highway-v0 --steps 8 --save-every 2
python exp-highway-env/viz_three_configurations.py --env merge-v0 --action random
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle


HERE = Path(__file__).resolve().parent
LOCAL_HIGHWAY_ENV = HERE / "HighwayEnv"
if LOCAL_HIGHWAY_ENV.exists():
    sys.path.insert(0, str(LOCAL_HIGHWAY_ENV))
sys.path.insert(0, str(HERE))

from env_wrapper import (  # noqa: E402
    HighwayMaterialObservation,
    WrapperConfig,
    _ego_lane_center_y,
    _vehicle_to_state,
)


DEFAULT_OUT = HERE / "live_env_visuals"


def _import_gymnasium():
    try:
        import gymnasium as gym
        import highway_env  # noqa: F401  Registers highway-env ids.

        return gym
    except ImportError as exc:
        raise SystemExit(
            "Could not import gymnasium/highway_env. Install the local checkout first:\n"
            "  python -m pip install -e exp-highway-env/HighwayEnv\n"
            f"Original import error: {exc}"
        ) from exc


def _reset(env: Any, seed: int):
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def _step(env: Any, action: Any):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated), bool(truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), False, info


def _idle_action(env: Any) -> int:
    action_type = getattr(env.unwrapped, "action_type", None)
    indexes = getattr(action_type, "actions_indexes", None)
    if indexes and "IDLE" in indexes:
        return int(indexes["IDLE"])
    return 0


def _choose_action(env: Any, mode: str, step_idx: int) -> Any:
    if mode == "random":
        return env.action_space.sample()
    if mode == "cycle":
        n = getattr(env.action_space, "n", 1)
        return int(step_idx % max(int(n), 1))
    return _idle_action(env)


def _vehicle_polygon(vehicle: Any) -> np.ndarray:
    state = _vehicle_to_state(vehicle)
    cx, cy = state.position
    length, width = state.length, state.width
    c, s = np.cos(state.heading), np.sin(state.heading)
    long_axis = np.array([c, s])
    lat_axis = np.array([-s, c])
    return np.array(
        [
            [cx, cy] + 0.5 * length * long_axis + 0.5 * width * lat_axis,
            [cx, cy] + 0.5 * length * long_axis - 0.5 * width * lat_axis,
            [cx, cy] - 0.5 * length * long_axis - 0.5 * width * lat_axis,
            [cx, cy] - 0.5 * length * long_axis + 0.5 * width * lat_axis,
        ]
    )


def _draw_vehicle(ax, vehicle: Any, *, is_ego: bool = False) -> None:
    state = _vehicle_to_state(vehicle)
    color = "#28a745" if is_ego else "#1f77b4"
    edge = "white" if is_ego else "#0b2545"
    poly = Polygon(
        _vehicle_polygon(vehicle),
        closed=True,
        facecolor=color,
        edgecolor=edge,
        linewidth=1.4 if is_ego else 0.9,
        alpha=0.90 if is_ego else 0.65,
        zorder=6 if is_ego else 5,
    )
    ax.add_patch(poly)
    if state.speed > 0.1:
        dx = np.cos(state.heading) * min(state.speed * 0.45, 12.0)
        dy = np.sin(state.heading) * min(state.speed * 0.45, 12.0)
        ax.arrow(
            state.position[0],
            state.position[1],
            dx,
            dy,
            head_width=0.55,
            head_length=0.85,
            fc=color,
            ec=color,
            linewidth=0.9,
            alpha=0.8,
            length_includes_head=True,
            zorder=7,
        )


def _lane_samples(env: Any, x_bounds: tuple[float, float]) -> list[tuple[Any, np.ndarray]]:
    lanes = []
    network = env.unwrapped.road.network
    lanes_dict = network.lanes_dict() if hasattr(network, "lanes_dict") else {}
    for lane_index, lane in lanes_dict.items():
        length = float(getattr(lane, "length", 0.0))
        if length <= 0:
            continue
        ss = np.linspace(0.0, length, 100)
        pts = np.array([lane.position(float(s), 0.0) for s in ss], dtype=float)
        visible = (pts[:, 0] >= x_bounds[0] - 20.0) & (pts[:, 0] <= x_bounds[1] + 20.0)
        if visible.any():
            lanes.append((lane_index, pts[visible]))
    return lanes


def _patch_grid_world(observer: HighwayMaterialObservation, env: Any) -> np.ndarray:
    ego_state = _vehicle_to_state(env.unwrapped.vehicle)
    return observer.risk_constructor._patch_grid_world(ego_state)


def _patch_outline(grid_world: np.ndarray) -> np.ndarray:
    return np.array(
        [
            grid_world[0, 0],
            grid_world[0, -1],
            grid_world[-1, -1],
            grid_world[-1, 0],
        ]
    )


def _ego_patch_index(obs: dict[str, np.ndarray]) -> tuple[int, int]:
    hp, wp = obs["rollout_patch"].shape[1:]
    return hp // 2, max(0, int(round(0.05 * wp)))


def _window_from_patch(grid_world: np.ndarray, pad: float = 8.0) -> tuple[float, float, float, float]:
    xs = grid_world[..., 0]
    ys = grid_world[..., 1]
    return (
        float(xs.min() - pad),
        float(xs.max() + pad),
        float(ys.min() - pad),
        float(ys.max() + pad),
    )


def _scatter_patch(ax, grid_world: np.ndarray, values: np.ndarray, *, cmap: str,
                   vmin: float | None = None, vmax: float | None = None,
                   title: str = "", alpha: float = 0.88):
    flat_xy = grid_world.reshape(-1, 2)
    sc = ax.scatter(
        flat_xy[:, 0],
        flat_xy[:, 1],
        c=values.reshape(-1),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        marker="s",
        s=22,
        linewidths=0,
        alpha=alpha,
        zorder=2,
    )
    ax.set_title(title)
    return sc


def _draw_lanes_vehicles_goal(
    ax,
    env: Any,
    obs: dict[str, np.ndarray],
    grid_world: np.ndarray,
    *,
    show_patch: bool = True,
) -> None:
    x0, x1, y0, y1 = _window_from_patch(grid_world)
    for lane_index, pts in _lane_samples(env, (x0, x1)):
        ax.plot(pts[:, 0], pts[:, 1], color="0.15", linewidth=1.0, alpha=0.55, zorder=1)
        mid = pts[len(pts) // 2]
        ax.text(
            mid[0],
            mid[1] + 0.25,
            str(lane_index),
            fontsize=6,
            color="0.25",
            alpha=0.75,
            clip_on=True,
            zorder=8,
        )

    if show_patch:
        outline = _patch_outline(grid_world)
        ax.add_patch(
            Polygon(
                outline,
                closed=True,
                fill=False,
                edgecolor="#ff9f1c",
                linewidth=1.6,
                linestyle="-",
                zorder=4,
            )
        )

    ego = env.unwrapped.vehicle
    for vehicle in env.unwrapped.road.vehicles:
        _draw_vehicle(ax, vehicle, is_ego=(vehicle is ego))

    ax.plot(obs["goal"][0], obs["goal"][1], marker="*", markersize=12,
            color="#ffd60a", markeredgecolor="black", zorder=9)
    ax.plot(obs["o0"][0], obs["o0"][1], marker="o", markersize=4,
            color="white", markeredgecolor="black", zorder=10)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="0.9", linewidth=0.5)


def _quiver_force(
    ax,
    grid_world: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    step_lat: int,
    step_lon: int,
    sign: float = -1.0,
    color: str = "black",
) -> None:
    sub = (slice(None, None, step_lat), slice(None, None, step_lon))
    ax.quiver(
        grid_world[..., 0][sub],
        grid_world[..., 1][sub],
        sign * gx[sub],
        sign * gy[sub],
        color=color,
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.0025,
        alpha=0.85,
        zorder=6,
    )


def _add_ego_force_arrow(
    ax,
    obs: dict[str, np.ndarray],
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    scale: float,
    color: str,
    sign: float = -1.0,
) -> np.ndarray:
    er, ec = _ego_patch_index(obs)
    force = np.array([sign * gx[er, ec], sign * gy[er, ec]], dtype=float)
    ax.arrow(
        obs["o0"][0],
        obs["o0"][1],
        scale * force[0],
        scale * force[1],
        head_width=0.75,
        head_length=0.9,
        fc=color,
        ec=color,
        linewidth=2.0,
        length_includes_head=True,
        zorder=11,
    )
    return force


def _validate_observation(obs: dict[str, np.ndarray], cfg: WrapperConfig) -> None:
    expected = {
        "o0": (2,),
        "v0": (2,),
        "goal": (2,),
        "C": (cfg.n_max_vehicles, 2),
        "R": (cfg.n_max_vehicles,),
        "W": (cfg.n_max_vehicles,),
        "mask": (cfg.n_max_vehicles,),
        "risk_patch": (2, cfg.risk_field.patch_lat_cells, cfg.risk_field.patch_lon_cells),
        "rollout_patch": (6, cfg.risk_field.patch_lat_cells, cfg.risk_field.patch_lon_cells),
    }
    for key, shape in expected.items():
        if obs[key].shape != shape:
            raise AssertionError(f"{key}: expected {shape}, got {obs[key].shape}")
    if obs["risk_patch"].dtype != np.float32 or obs["rollout_patch"].dtype != np.float32:
        raise AssertionError("risk_patch and rollout_patch must be float32")
    if obs["mask"].dtype != np.bool_:
        raise AssertionError("mask must be bool")
    if not np.allclose(obs["risk_patch"][0], obs["rollout_patch"][0]):
        raise AssertionError("risk_patch[0] must match rollout_patch risk channel")
    hard_from_phi = (obs["rollout_patch"][1] <= 0.0).astype(np.float32)
    if not np.allclose(obs["risk_patch"][1], hard_from_phi):
        raise AssertionError("risk_patch[1] must be hard-mask derived from phi")


def render_frame(
    env: Any,
    observer: HighwayMaterialObservation,
    obs: dict[str, np.ndarray],
    out_path: Path,
    *,
    env_id: str,
    seed: int,
    step_idx: int,
    action: Any,
) -> dict[str, Any]:
    grid_world = _patch_grid_world(observer, env)
    rollout = obs["rollout_patch"]
    risk = obs["risk_patch"][0]
    hard = obs["risk_patch"][1]
    phi = rollout[1]
    grx, gry = rollout[2], rollout[3]
    gpx, gpy = rollout[4], rollout[5]
    hp, wp = risk.shape
    step_lat = max(1, hp // 12)
    step_lon = max(1, wp // 20)
    lane_y, lane_source = _ego_lane_center_y(env, obs["o0"], return_source=True)
    er, ec = _ego_patch_index(obs)

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    ego = env.unwrapped.vehicle
    fig.suptitle(
        f"{env_id} seed={seed} step={step_idx} action={action} | "
        f"ego=({obs['o0'][0]:.1f},{obs['o0'][1]:.1f}) "
        f"heading={float(ego.heading):.3f} lane={getattr(ego, 'lane_index', None)} | "
        f"goal=({obs['goal'][0]:.1f},{obs['goal'][1]:.1f}) "
        f"lane_y={lane_y:.2f} via {lane_source}",
        fontsize=12,
        fontweight="bold",
    )

    ax = axs[0, 0]
    _draw_lanes_vehicles_goal(ax, env, obs, grid_world)
    ax.set_title("Live highway-env scene + observer patch")

    ax = axs[0, 1]
    sc = _scatter_patch(ax, grid_world, risk, cmap="Reds", vmin=0, vmax=1,
                        title="risk_patch[0] = soft risk r_tilde")
    _draw_lanes_vehicles_goal(ax, env, obs, grid_world)
    fig.colorbar(sc, ax=ax, fraction=0.035)

    ax = axs[0, 2]
    sc = _scatter_patch(ax, grid_world, hard, cmap="gray_r", vmin=0, vmax=1,
                        title="risk_patch[1] = hard mask from phi <= 0")
    _draw_lanes_vehicles_goal(ax, env, obs, grid_world)
    fig.colorbar(sc, ax=ax, fraction=0.035)

    ax = axs[1, 0]
    vmax = max(1.0, float(np.percentile(phi, 95)))
    sc = _scatter_patch(ax, grid_world, phi, cmap="viridis", vmin=float(phi.min()), vmax=vmax,
                        title="rollout_patch[1] = hard SDF phi")
    _draw_lanes_vehicles_goal(ax, env, obs, grid_world)
    fig.colorbar(sc, ax=ax, fraction=0.035)

    ax = axs[1, 1]
    _scatter_patch(ax, grid_world, risk, cmap="Reds", vmin=0, vmax=1,
                   title="world-frame F_soft direction = -grad r_tilde", alpha=0.38)
    _draw_lanes_vehicles_goal(ax, env, obs, grid_world)
    _quiver_force(ax, grid_world, grx, gry, step_lat=step_lat, step_lon=step_lon)
    force_r = _add_ego_force_arrow(ax, obs, grx, gry, scale=8.0, color="#d00000")

    ax = axs[1, 2]
    _scatter_patch(ax, grid_world, phi, cmap="viridis", vmin=float(phi.min()), vmax=vmax,
                   title="world-frame F_hard direction = +grad phi", alpha=0.38)
    _draw_lanes_vehicles_goal(ax, env, obs, grid_world)
    _quiver_force(ax, grid_world, gpx, gpy, step_lat=step_lat, step_lon=step_lon, sign=1.0)
    force_phi = _add_ego_force_arrow(
        ax, obs, gpx, gpy, scale=1.5, color="#d00000", sign=1.0
    )

    for ax in axs.flat:
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world y [m]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    return {
        "env": env_id,
        "seed": seed,
        "step": step_idx,
        "action": str(action),
        "x": float(obs["o0"][0]),
        "y": float(obs["o0"][1]),
        "heading": float(ego.heading),
        "lane_index": str(getattr(ego, "lane_index", None)),
        "goal_x": float(obs["goal"][0]),
        "goal_y": float(obs["goal"][1]),
        "lane_y": float(lane_y),
        "lane_source": lane_source,
        "vehicles_total": len(env.unwrapped.road.vehicles),
        "valid_obstacles": int(obs["mask"].sum()),
        "risk_min": float(risk.min()),
        "risk_max": float(risk.max()),
        "hard_pixels": int(hard.sum()),
        "phi_min": float(phi.min()),
        "phi_at_ego": float(phi[er, ec]),
        "force_soft_x": float(force_r[0]),
        "force_soft_y": float(force_r[1]),
        "force_hard_x": float(force_phi[0]),
        "force_hard_y": float(force_phi[1]),
        "png": str(out_path),
    }


def run_env(args: argparse.Namespace, env_id: str, out_root: Path) -> list[dict[str, Any]]:
    gym = _import_gymnasium()
    env_config: dict[str, Any] = {}
    if args.vehicles_count is not None:
        env_config["vehicles_count"] = args.vehicles_count
    if args.lanes_count is not None:
        env_config["lanes_count"] = args.lanes_count
    if args.policy_frequency is not None:
        env_config["policy_frequency"] = args.policy_frequency
    if args.simulation_frequency is not None:
        env_config["simulation_frequency"] = args.simulation_frequency

    env = gym.make(env_id, config=env_config or None)
    cfg = WrapperConfig(
        n_max_vehicles=args.n_max_vehicles,
        sensing_radius_m=args.sensing_radius_m,
        goal_lookahead_m=args.goal_lookahead_m,
    )
    cfg.risk_field.sensing_radius_m = args.sensing_radius_m
    observer = HighwayMaterialObservation(cfg)

    rows: list[dict[str, Any]] = []
    try:
        _reset(env, args.seed)
        for step_idx in range(args.steps):
            obs = observer.build(env)
            _validate_observation(obs, cfg)
            action = _choose_action(env, args.action, step_idx)
            should_save = step_idx % args.save_every == 0 or step_idx == args.steps - 1
            if should_save:
                out_path = out_root / env_id / f"seed_{args.seed:04d}_step_{step_idx:03d}.png"
                row = render_frame(
                    env,
                    observer,
                    obs,
                    out_path,
                    env_id=env_id,
                    seed=args.seed,
                    step_idx=step_idx,
                    action=action,
                )
                rows.append(row)
                print(
                    f"saved {out_path} | lane_y={row['lane_y']:.2f}({row['lane_source']}) "
                    f"risk_max={row['risk_max']:.3f} hard_px={row['hard_pixels']} "
                    f"valid_obs={row['valid_obstacles']}"
                )
            _, _, terminated, truncated, _ = _step(env, action)
            if terminated or truncated:
                print(f"{env_id}: episode ended at step {step_idx} "
                      f"terminated={terminated} truncated={truncated}")
                break
    finally:
        env.close()

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        action="append",
        dest="envs",
        default=None,
        help="Environment id. Can be provided multiple times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--action", choices=["idle", "random", "cycle"], default="idle")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vehicles-count", type=int, default=None)
    parser.add_argument("--lanes-count", type=int, default=None)
    parser.add_argument("--policy-frequency", type=int, default=None)
    parser.add_argument("--simulation-frequency", type=int, default=None)
    parser.add_argument("--goal-lookahead-m", type=float, default=30.0)
    parser.add_argument("--n-max-vehicles", type=int, default=15)
    parser.add_argument("--sensing-radius-m", type=float, default=80.0)
    return parser.parse_args()


def write_summary(rows: list[dict[str, Any]], out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.csv"
    if not rows:
        return summary_path
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main() -> None:
    args = parse_args()
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    envs = args.envs or ["highway-v0", "merge-v0"]
    all_rows: list[dict[str, Any]] = []
    print(f"Rendering live highway-env diagnostics to {args.out}")
    for env_id in envs:
        all_rows.extend(run_env(args, env_id, args.out))
    summary_path = write_summary(all_rows, args.out)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
