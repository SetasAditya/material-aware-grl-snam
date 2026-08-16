from __future__ import annotations

from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle


class HighwaySlowLeaderEnv(HighwayEnv):
    """A deterministic slow-leader passing scenario.

    Ego starts behind a slow vehicle in the same lane. The adjacent lane is
    intentionally left open near the ego/leader pair, so a policy with a
    lateral passing response can use it.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "lanes_count": 4,
                "vehicles_count": 0,
                "controlled_vehicles": 1,
                "initial_lane_id": 1,
                "ego_x": 100.0,
                "ego_speed": 25.0,
                "leader_offset": 30.0,
                "leader_count": 1,
                "leader_spacing": 28.0,
                "leader_speed": 10.0,
                "leader_target_speed": 10.0,
                "background": True,
                "vehicles_density": 1,
            }
        )
        return config

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _lane_index(self, lane_id: int):
        return ("0", "1", lane_id)

    def _spawn_vehicle(self, lane_id: int, x: float, speed: float,
                       target_speed: float | None = None,
                       enable_lane_change: bool = False) -> IDMVehicle:
        lane_index = self._lane_index(lane_id)
        lane = self.road.network.get_lane(lane_index)
        vehicle = IDMVehicle(
            self.road,
            lane.position(x, 0.0),
            heading=lane.heading_at(x),
            speed=speed,
            target_lane_index=lane_index,
            target_speed=target_speed if target_speed is not None else speed,
            enable_lane_change=enable_lane_change,
        )
        self.road.vehicles.append(vehicle)
        return vehicle

    def _spawn_ego(self) -> tuple[int, float]:
        ego_lane = int(self.config["initial_lane_id"])
        ego_x = float(self.config["ego_x"])
        ego_speed = float(self.config["ego_speed"])
        lane_index = self._lane_index(ego_lane)
        lane = self.road.network.get_lane(lane_index)

        ego = Vehicle(
            self.road,
            lane.position(ego_x, 0.0),
            lane.heading_at(ego_x),
            ego_speed,
        )
        ego = self.action_type.vehicle_class(
            self.road, ego.position, ego.heading, ego.speed
        )
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)
        return ego_lane, ego_x

    def _spawn_leader_convoy(self, ego_lane: int, ego_x: float) -> None:
        leader_count = int(self.config.get("leader_count", 1))
        leader_offset = float(self.config["leader_offset"])
        leader_spacing = float(self.config.get("leader_spacing", 28.0))
        leader_speed = float(self.config["leader_speed"])
        leader_target_speed = float(self.config["leader_target_speed"])

        for idx in range(leader_count):
            self._spawn_vehicle(
                ego_lane,
                ego_x + leader_offset + idx * leader_spacing,
                leader_speed,
                target_speed=leader_target_speed,
                enable_lane_change=False,
            )

    def _spawn_background(self, ego_lane: int, ego_x: float) -> None:
        if not self.config.get("background", True):
            return

        # Far vehicles keep the scene highway-like without blocking the
        # immediate adjacent-lane passing opportunity.
        self._spawn_vehicle(ego_lane, ego_x + 95.0, 22.0, 22.0)
        self._spawn_vehicle(ego_lane, ego_x - 45.0, 24.0, 24.0)
        self._spawn_vehicle(max(0, ego_lane - 1), ego_x + 85.0, 24.0, 24.0)
        self._spawn_vehicle(
            min(self.config["lanes_count"] - 1, ego_lane + 1),
            ego_x - 55.0,
            24.0,
            24.0,
        )

    def _create_vehicles(self) -> None:
        ego_lane, ego_x = self._spawn_ego()
        self._spawn_leader_convoy(ego_lane, ego_x)
        self._spawn_background(ego_lane, ego_x)


class HighwaySlowLeaderBoxedEnv(HighwaySlowLeaderEnv):
    """Slow leader with adjacent-lane blockers.

    This variant should reward controlled slowing more than a lane-change pass.
    It is useful as a contrast to HighwaySlowLeaderEnv.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "leader_offset": 28.0,
                "leader_count": 1,
                "leader_spacing": 28.0,
                "leader_speed": 10.0,
                "leader_target_speed": 10.0,
                "boxed_blocker_offsets": [18.0, 52.0],
                "boxed_blocker_speed": 18.0,
                "background": False,
            }
        )
        return config

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        ego_lane = int(self.config["initial_lane_id"])
        ego_x = float(self.config["ego_x"])
        left_lane = max(0, ego_lane - 1)
        right_lane = min(self.config["lanes_count"] - 1, ego_lane + 1)

        # Vehicles bracketing the ego/leader pair in adjacent lanes. They make
        # an immediate pass unattractive while keeping the scenario collision-free.
        blocker_speed = float(self.config.get("boxed_blocker_speed", 18.0))
        for offset in self.config.get("boxed_blocker_offsets", [18.0, 52.0]):
            self._spawn_vehicle(left_lane, ego_x + float(offset), blocker_speed, blocker_speed)
            self._spawn_vehicle(right_lane, ego_x + float(offset), blocker_speed, blocker_speed)


class HighwaySlowLeaderX2Env(HighwaySlowLeaderEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({"leader_count": 2})
        return config


class HighwaySlowLeaderX3Env(HighwaySlowLeaderEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({"leader_count": 3})
        return config


class HighwaySlowLeaderX4Env(HighwaySlowLeaderEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({"leader_count": 4})
        return config


class HighwaySlowLeaderBoxedX2Env(HighwaySlowLeaderBoxedEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "leader_count": 2,
            "boxed_blocker_offsets": [18.0, 52.0, 86.0],
        })
        return config


class HighwaySlowLeaderBoxedX3Env(HighwaySlowLeaderBoxedEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "leader_count": 3,
            "boxed_blocker_offsets": [18.0, 52.0, 86.0, 120.0],
        })
        return config


class HighwaySlowLeaderBoxedX4Env(HighwaySlowLeaderBoxedEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "leader_count": 4,
            "boxed_blocker_offsets": [18.0, 52.0, 86.0, 120.0, 154.0],
        })
        return config


class HighwaySlowConvoyEnv(HighwaySlowLeaderX4Env):
    """Alias for the open 4-leader convoy scenario."""


class HighwaySlowConvoyBoxedEnv(HighwaySlowLeaderBoxedX4Env):
    """Alias for the boxed 4-leader convoy scenario."""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({"boxed_blocker_offsets": [18.0, 46.0, 74.0, 102.0, 130.0]})
        return config
