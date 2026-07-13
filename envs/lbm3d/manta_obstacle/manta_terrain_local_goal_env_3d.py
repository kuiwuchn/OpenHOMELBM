"""Manta terrain local-goal environment for low-level waypoint tracking training.

This env keeps the current direction+distance reward formulation from
`MantaMultiTaskEnv`, but replaces random goal sampling with a moving local goal
that advances along a reference figure-eight trajectory around two static
pillars. The low-level policy therefore trains as a local waypoint follower in a
terrain scene, while collision still terminates the episode.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, List

import numpy as np
import warp as wp
import mujoco_warp as mjw

from ..manta.manta_multitask_env_3d import MantaMultiTaskEnv, TASK_FORWARD
from .manta_obstacle_lbm_env_3d import _TERRAIN_XMLS


class MantaTerrainLocalGoalEnv3D(MantaMultiTaskEnv):
    """Terrain-aware local-goal env for figure-eight waypoint tracking."""

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        root_link: str = "body",
        root_position: Optional[Tuple[float, float, float]] = None,
        nx: int = 200,
        ny: int = 200,
        nz: int = 80,
        lbm_scale: float = 1.0,
        nworld: int = 1,
        max_episode_steps: int = 2000,
        per_frame_steps: int = 10,
        fluid_density: float = 1000.0,
        device: Optional[str] = None,
        task_switch_interval: int = 0,
        enabled_tasks: Optional[List[str]] = None,
        reward_w_task: float = 1.0,
        reward_w_roll: float = 0.10,
        reward_w_smooth: float = 0.015,
        reward_w_offaxis: float = 0.02,
        target_forward_vel: float = 0.20,
        target_yaw_rate: float = 0.15,
        target_vertical_vel: float = 0.12,
        disable_speed_targets: bool = True,
        use_direction_dist_tasks: bool = True,
        direction_dist_min: float = 0.08,
        direction_dist_max: float = 0.16,
        direction_goal_threshold: float = 0.05,
        direction_dist_w_dist: float = 100.0,
        direction_dist_w_roll: float = 0.5,
        direction_dist_w_heading: float = 0.2,
        direction_dist_w_forward: float = 0.1,
        direction_dist_goal_bonus: float = 10.0,
        direction_dist_terminate_on_goal: bool = False,
        alive_cost: float = 0.0,
        termination_penalty: float = 1.0,
        temporal_stack_obs: bool = False,
        k_harmonics: int = 2,
        b_bar: float = 1.0,
        use_reduced_order: bool = False,
        control_mode: str = "direct",
        obstacle_collision_radius: float = 0.04,
        terrain_variant: str = "figure8",
        trajectory_points: int = 120,
        trajectory_lookahead: int = 6,
        trajectory_search_window: int = 18,
        trajectory_waypoint_threshold: float = 0.05,
        trajectory_radius_scale: float = 0.84,
        start_on_trajectory: bool = True,
        trajectory_start_idx: int = 75,
        align_start_to_goal: bool = True,
    ):
        if mjcf_path is None:
            xml_name = _TERRAIN_XMLS.get(terrain_variant, "manta_terrain_figure8_3d.xml")
            mjcf_path = os.path.join(os.path.dirname(__file__), xml_name)

        if root_position is None:
            root_position = (nx / 2, ny * 0.36, nz / 2)

        if enabled_tasks is None:
            enabled_tasks = ["forward"]

        super().__init__(
            mjcf_path=mjcf_path,
            root_link=root_link,
            root_position=root_position,
            nx=nx,
            ny=ny,
            nz=nz,
            lbm_scale=lbm_scale,
            nworld=nworld,
            max_episode_steps=max_episode_steps,
            per_frame_steps=per_frame_steps,
            fluid_density=fluid_density,
            device=device,
            task_switch_interval=task_switch_interval,
            enabled_tasks=enabled_tasks,
            reward_w_task=reward_w_task,
            reward_w_roll=reward_w_roll,
            reward_w_smooth=reward_w_smooth,
            reward_w_offaxis=reward_w_offaxis,
            target_forward_vel=target_forward_vel,
            target_yaw_rate=target_yaw_rate,
            target_vertical_vel=target_vertical_vel,
            disable_speed_targets=disable_speed_targets,
            use_direction_dist_tasks=use_direction_dist_tasks,
            direction_dist_min=direction_dist_min,
            direction_dist_max=direction_dist_max,
            direction_goal_threshold=direction_goal_threshold,
            direction_dist_w_dist=direction_dist_w_dist,
            direction_dist_w_roll=direction_dist_w_roll,
            direction_dist_w_heading=direction_dist_w_heading,
            direction_dist_w_forward=direction_dist_w_forward,
            direction_dist_goal_bonus=direction_dist_goal_bonus,
            direction_dist_terminate_on_goal=direction_dist_terminate_on_goal,
            alive_cost=alive_cost,
            termination_penalty=termination_penalty,
            temporal_stack_obs=temporal_stack_obs,
            k_harmonics=k_harmonics,
            b_bar=b_bar,
            use_reduced_order=use_reduced_order,
            control_mode=control_mode,
        )

        if self.n_static < 2:
            raise ValueError(
                "MantaTerrainLocalGoalEnv3D expects at least two static terrain bodies "
                f"for figure-eight training, got n_static={self.n_static}."
            )

        self.terrain_variant = terrain_variant
        self.obstacle_collision_radius = float(obstacle_collision_radius)
        self.terrain_body_names = [cfg["link_name"] for cfg in self.static_link_config]
        root_cfg = next(
            (cfg for cfg in self.dynamic_link_config if cfg["link_name"] == root_link),
            self.dynamic_link_config[0],
        )
        self._root_lbm_origin = np.array(root_cfg["lbm_position"], dtype=np.float32)
        self._root_coord_scale = float(self.lbm_scale * self.nx)
        self._terrain_positions_normalized = np.array(
            [
                [
                    cfg["lbm_position"][0] / self.nx,
                    cfg["lbm_position"][1] / self.ny,
                    cfg["lbm_position"][2] / self.nz,
                ]
                for cfg in self.static_link_config
            ],
            dtype=np.float32,
        )
        self._terrain_positions_normalized = self._terrain_positions_normalized[
            np.argsort(self._terrain_positions_normalized[:, 0])
        ]

        self.trajectory_points = max(40, int(trajectory_points))
        self.trajectory_lookahead = max(1, int(trajectory_lookahead))
        self.trajectory_search_window = max(self.trajectory_lookahead + 2, int(trajectory_search_window))
        self.trajectory_waypoint_threshold = float(trajectory_waypoint_threshold)
        self.trajectory_radius_scale = float(trajectory_radius_scale)
        self.start_on_trajectory = bool(start_on_trajectory)
        self.trajectory_start_idx = int(trajectory_start_idx)
        self.align_start_to_goal = bool(align_start_to_goal)
        self._trajectory = self._build_figure8_trajectory(self.trajectory_points)
        self._trajectory_len = self._trajectory.shape[0]
        self._trajectory_progress = np.zeros(self.nworld, dtype=np.int32)
        self.trajectory_start_idx %= self._trajectory_len

        self.single_obs_dim = self.single_obs_dim + 3
        self.obs_dim = self.single_obs_dim * 2 if self.temporal_stack_obs else self.single_obs_dim
        self.observation_space = self._create_observation_space()

        self._task_ids[:] = TASK_FORWARD
        self._update_task_ids_wp()

        print("MantaTerrainLocalGoalEnv3D initialized:")
        print(f"  Terrain variant: {self.terrain_variant}")
        print(f"  Terrain bodies: {self.terrain_body_names}")
        print(f"  Obs dim: {self.obs_dim}")
        print(
            f"  Figure-8 local goals: points={self.trajectory_points}, "
            f"lookahead={self.trajectory_lookahead}, search_window={self.trajectory_search_window}"
        )
        print(
            f"  Start: on_trajectory={self.start_on_trajectory}, "
            f"start_idx={self.trajectory_start_idx}, align_to_goal={self.align_start_to_goal}"
        )

    def _create_observation_space(self):
        from gym import spaces

        single_obs_dim = getattr(self, "single_obs_dim", None)
        if single_obs_dim is None:
            n_joints = self.mj_model.njnt - 1
            base_single_obs_dim = 22 + 3 * n_joints + 3 + 5 + 2
            extra_task_obs_dim = 4 if getattr(self, "use_direction_dist_tasks", False) else 0
            single_obs_dim = base_single_obs_dim + extra_task_obs_dim + 3
        obs_dim = single_obs_dim * 2 if getattr(self, "temporal_stack_obs", False) else single_obs_dim

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.nworld, obs_dim),
            dtype=np.float32,
        )

    def _sanitize_observation(
        self,
        observation: np.ndarray,
        reward: Optional[np.ndarray] = None,
        done: Optional[np.ndarray] = None,
        info: Optional[dict] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clamp non-finite observations before they enter replay/training."""
        nonfinite_mask = np.any(~np.isfinite(observation), axis=1)
        if not np.any(nonfinite_mask):
            return observation, nonfinite_mask

        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

        if reward is not None:
            reward[nonfinite_mask] = getattr(self, "anomaly_penalty", -1.0)
        if done is not None:
            done[nonfinite_mask] = True

        if info is not None:
            terminated = np.asarray(info.get("terminated", np.zeros(self.nworld, dtype=bool)), dtype=bool)
            terminated[nonfinite_mask] = True
            info["terminated"] = terminated

            anomaly = np.asarray(info.get("anomaly", np.zeros(self.nworld, dtype=bool)), dtype=bool)
            anomaly[nonfinite_mask] = True
            info["anomaly"] = anomaly
            info["obs_nan"] = nonfinite_mask.copy()

            term_reason = info.get("term_reason", ["running"] * self.nworld)
            if isinstance(term_reason, str):
                term_reason = [term_reason] * self.nworld
            patched = []
            for w in range(self.nworld):
                raw = term_reason[w]
                parts = [] if raw in (None, "", "running") else [p for p in str(raw).split("|") if p and p != "running"]
                if nonfinite_mask[w] and "obs_nan" not in parts:
                    parts.append("obs_nan")
                patched.append("|".join(parts) if parts else "running")
            info["term_reason"] = patched

        return observation, nonfinite_mask

    def _build_figure8_trajectory(self, num_points: int) -> np.ndarray:
        left = self._terrain_positions_normalized[0]
        right = self._terrain_positions_normalized[-1]
        spacing = float(abs(right[0] - left[0]))
        base_radius = 0.5 * spacing
        radius = max(base_radius * self.trajectory_radius_scale, self.obstacle_collision_radius + 0.035)
        z_val = float(0.5 * (left[2] + right[2]))

        n_left = num_points // 2
        n_right = num_points - n_left

        left_theta = np.linspace(0.0, -2.0 * np.pi, n_left, endpoint=False, dtype=np.float32)
        right_theta = np.linspace(np.pi, 3.0 * np.pi, n_right, endpoint=False, dtype=np.float32)

        left_loop = np.stack(
            [
                left[0] + radius * np.cos(left_theta),
                left[1] + radius * np.sin(left_theta),
                np.full(n_left, z_val, dtype=np.float32),
            ],
            axis=1,
        )
        right_loop = np.stack(
            [
                right[0] + radius * np.cos(right_theta),
                right[1] + radius * np.sin(right_theta),
                np.full(n_right, z_val, dtype=np.float32),
            ],
            axis=1,
        )

        traj = np.concatenate([left_loop, right_loop], axis=0).astype(np.float32)
        margin = np.array(
            [
                self.boundary_margin / float(self.nx),
                self.boundary_margin / float(self.ny),
                self.boundary_margin / float(self.nz),
            ],
            dtype=np.float32,
        )
        return np.clip(traj, margin, 1.0 - margin)

    def _compute_nearest_terrain_obs_features(self) -> np.ndarray:
        body_positions = self._body_positions_normalized()
        qpos_np = self.mjw_data.qpos.numpy()
        features = np.zeros((self.nworld, 3), dtype=np.float32)

        for w in range(self.nworld):
            rel = self._terrain_positions_normalized - body_positions[w][None, :]
            dist_sq = np.sum(rel * rel, axis=1)
            nearest_world = rel[int(np.argmin(dist_sq))]
            quat = qpos_np[w, 3:7].astype(np.float32, copy=False)
            nearest_body = self._quat_rotate_vec_inv_np(quat, nearest_world.astype(np.float32, copy=False))
            features[w] = nearest_body.astype(np.float32)

        return features

    def _get_obs(self) -> np.ndarray:
        base_obs = super()._get_obs()
        terrain_obs = self._compute_nearest_terrain_obs_features()
        return np.concatenate((base_obs, terrain_obs), axis=1)

    def _check_terrain_collision(self) -> np.ndarray:
        collision = np.zeros(self.nworld, dtype=bool)
        radius_sq = self.obstacle_collision_radius * self.obstacle_collision_radius

        for w in range(self.nworld):
            flow = self.lbm_solver.flows[w]
            positions = flow.solid_position.numpy()
            for dyn_idx in range(self.n_dynamic):
                dyn = np.array(
                    [
                        positions[dyn_idx][0] / self.nx,
                        positions[dyn_idx][1] / self.ny,
                        positions[dyn_idx][2] / self.nz,
                    ],
                    dtype=np.float32,
                )
                for static_pos in self._terrain_positions_normalized:
                    delta = dyn - static_pos
                    if float(np.dot(delta, delta)) < radius_sq:
                        collision[w] = True
                        break
                if collision[w]:
                    break

        return collision

    def _find_global_progress(self, body_pos: np.ndarray) -> int:
        dist_sq = np.sum((self._trajectory - body_pos[None, :]) ** 2, axis=1)
        return int(np.argmin(dist_sq))

    def _advance_progress(self, world_idx: int, body_pos: np.ndarray) -> int:
        start = int(self._trajectory_progress[world_idx])
        best_idx = start
        best_dist = float("inf")

        for offset in range(self.trajectory_search_window):
            idx = (start + offset) % self._trajectory_len
            dist = float(np.linalg.norm(self._trajectory[idx] - body_pos))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        while True:
            next_idx = (best_idx + 1) % self._trajectory_len
            next_dist = float(np.linalg.norm(self._trajectory[next_idx] - body_pos))
            if next_dist > self.trajectory_waypoint_threshold:
                break
            best_idx = next_idx
            if next_idx == start:
                break

        return best_idx

    @staticmethod
    def _yaw_quat_from_direction(direction: np.ndarray) -> np.ndarray:
        yaw = float(np.arctan2(direction[0], direction[1]))
        half = 0.5 * yaw
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)

    def _apply_trajectory_start_pose(self, mask: Optional[np.ndarray] = None):
        if not self.start_on_trajectory:
            return
        if mask is None:
            mask = np.ones(self.nworld, dtype=bool)
        if not np.any(mask):
            return

        start_idx = self.trajectory_start_idx % self._trajectory_len
        target_idx = (start_idx + self.trajectory_lookahead) % self._trajectory_len
        start_norm = self._trajectory[start_idx].astype(np.float32)
        target_norm = self._trajectory[target_idx].astype(np.float32)
        start_grid = start_norm * np.array([self.nx, self.ny, self.nz], dtype=np.float32)
        start_mujoco = (start_grid - self._root_lbm_origin) / self._root_coord_scale

        direction = target_norm - start_norm
        if self.align_start_to_goal and float(np.linalg.norm(direction[:2])) > 1.0e-6:
            quat = self._yaw_quat_from_direction(direction)
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        qpos = self.mjw_data.qpos.numpy().copy()
        qvel = self.mjw_data.qvel.numpy().copy()
        for w in range(self.nworld):
            if mask[w]:
                qpos[w, 0:3] = start_mujoco
                qpos[w, 3:7] = quat
                qvel[w, :] = 0.0
                self._trajectory_progress[w] = start_idx

        wp.copy(self.mjw_data.qpos, wp.array(qpos.astype(np.float32), dtype=wp.float32, device=self.device))
        wp.copy(self.mjw_data.qvel, wp.array(qvel.astype(np.float32), dtype=wp.float32, device=self.device))
        mjw.forward(self.mjw_model, self.mjw_data)

        # Keep the LBM solid centers consistent with the adjusted MuJoCo root.
        for w in range(self.nworld):
            if not mask[w]:
                continue
            flow = self.lbm_solver.flows[w]
            positions = flow.solid_position.numpy()
            quats = flow.solid_quaternion.numpy()
            root_delta = start_grid - positions[0]
            positions[: self.n_dynamic] += root_delta
            quats[: self.n_dynamic] = quat
            wp.copy(flow.solid_position, wp.array(positions.astype(np.float32), dtype=wp.vec3, device=self.device))
            wp.copy(flow.solid_quaternion, wp.array(quats.astype(np.float32), dtype=wp.vec4, device=self.device))

    def _set_local_goals_from_trajectory(self, mask: Optional[np.ndarray] = None, initialize: bool = False):
        if mask is None:
            mask = np.ones(self.nworld, dtype=bool)
        if not np.any(mask):
            return

        body_positions = self._body_positions_normalized()
        prev_dist = self._prev_dist_wp.numpy().copy()

        for w in range(self.nworld):
            if not mask[w]:
                continue
            if initialize and self.start_on_trajectory:
                progress_idx = int(self._trajectory_progress[w])
            elif initialize:
                progress_idx = self._find_global_progress(body_positions[w])
            else:
                progress_idx = self._advance_progress(w, body_positions[w])
            self._trajectory_progress[w] = progress_idx
            target_idx = (progress_idx + self.trajectory_lookahead) % self._trajectory_len
            goal = self._trajectory[target_idx]
            self._direction_goal_positions_np[w] = goal.astype(np.float32)
            prev_dist[w] = float(np.linalg.norm(goal - body_positions[w]))

        wp.copy(
            self._goal_positions_wp,
            wp.array(self._direction_goal_positions_np.astype(np.float32), dtype=wp.float32, device=self.device),
        )
        wp.copy(self._prev_dist_wp, wp.array(prev_dist, dtype=wp.float32, device=self.device))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        obs = super().reset(seed=seed, options=options)
        del obs
        self._task_ids[:] = TASK_FORWARD
        self._update_task_ids_wp()
        self._trajectory_progress[:] = 0
        self._apply_trajectory_start_pose()
        self._set_local_goals_from_trajectory(initialize=True)
        observation = self._get_obs()
        if self.temporal_stack_obs:
            observation = np.concatenate((observation, observation), axis=1)
        observation, _ = self._sanitize_observation(observation)
        return observation

    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        obs = super().partial_reset(reset_mask)
        del obs
        if not np.any(reset_mask):
            observation = self._get_obs()
            if self.temporal_stack_obs:
                observation = np.concatenate((observation, observation), axis=1)
            observation, _ = self._sanitize_observation(observation)
            return observation
        self._task_ids[reset_mask] = TASK_FORWARD
        self._update_task_ids_wp()
        self._trajectory_progress[reset_mask] = 0
        self._apply_trajectory_start_pose(reset_mask)
        self._set_local_goals_from_trajectory(reset_mask, initialize=True)
        observation = self._get_obs()
        if self.temporal_stack_obs:
            observation = np.concatenate((observation, observation), axis=1)
        observation, _ = self._sanitize_observation(observation)
        return observation

    def step(self, action: np.ndarray):
        prev_goals = self._direction_goal_positions_np.copy()
        observation, reward, done, info = super().step(action)

        terrain_collision = self._check_terrain_collision()
        if np.any(terrain_collision):
            done = np.asarray(done, dtype=bool)
            reward = np.asarray(reward, dtype=np.float32)
            done[terrain_collision] = True
            reward[terrain_collision] = -max(self.termination_penalty, 1.0)

            info_terminated = np.asarray(info.get("terminated", np.zeros(self.nworld, dtype=bool)), dtype=bool)
            info_terminated[terrain_collision] = True
            info["terminated"] = info_terminated

            term_reason = info.get("term_reason", ["running"] * self.nworld)
            if isinstance(term_reason, str):
                term_reason = [term_reason] * self.nworld
            patched = []
            for w in range(self.nworld):
                raw = term_reason[w]
                parts = [] if raw in (None, "", "running") else [p for p in str(raw).split("|") if p and p != "running"]
                if terrain_collision[w] and "terrain_collision" not in parts:
                    parts.append("terrain_collision")
                patched.append("|".join(parts) if parts else "running")
            info["term_reason"] = patched

        active_mask = ~np.asarray(done, dtype=bool)
        if np.any(active_mask):
            self._set_local_goals_from_trajectory(active_mask, initialize=False)
            next_single_obs = self._get_obs()
            if self.temporal_stack_obs:
                observation[:, -self.single_obs_dim:] = next_single_obs
            else:
                observation = next_single_obs

        observation, obs_nonfinite = self._sanitize_observation(observation, reward, done, info)
        if np.any(obs_nonfinite):
            info["terminated"] = np.asarray(info.get("terminated", done), dtype=bool)

        info["goal_pos_normalized"] = prev_goals
        info["next_goal_pos_normalized"] = self._direction_goal_positions_np.copy()
        info["terrain_collision"] = terrain_collision.copy()
        info["terrain_pos_normalized"] = self._terrain_positions_normalized.copy()
        info["terrain_body_names"] = list(self.terrain_body_names)
        info["trajectory_progress_idx"] = self._trajectory_progress.copy()
        return observation, reward, done, info
