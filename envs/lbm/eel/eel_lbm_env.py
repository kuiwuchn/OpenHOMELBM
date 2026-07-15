"""2D eel LBM environment built on the existing 2D rigid-fluid solver."""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warp as wp
from gym import spaces

from ..fish.fish_lbm_env import FishLBMEnv


class Eel2DLBMEnv(FishLBMEnv):
    """Long-chain 2D eel swimmer using the same HOME LBM cut-cell solver as FishLBMEnv."""

    def __init__(
        self,
        xml_path: Optional[str] = None,
        solid_config: Optional[List[Dict[str, Any]]] = None,
        nx: int = 400,
        ny: int = 600,
        lbm_scale: float = 0.2,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        per_frame_steps: int = 10,
        nworld: int = 1,
        include_image: bool = False,
        image_size: Tuple[int, int] = (64, 64),
    ):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "eel_2d.xml")

        if solid_config is None:
            # Use the compact six-segment fallback model.
            segment_lengths = [0.0, 0.09, 0.18, 0.27, 0.36, 0.45]
            solid_config = [
                {
                    "solid_id": i,
                    "body_id": i + 1,
                    "body_or_geom_name": f"seg{i + 1}_geom",
                    "lbm_position": (nx * 0.5, 250.0 - segment_lengths[i] * nx * lbm_scale),
                    "is_body": False,
                    "n_samples": 24,
                }
                for i in range(6)
            ]

        super().__init__(
            xml_path=xml_path,
            solid_config=solid_config,
            nx=nx,
            ny=ny,
            lbm_scale=lbm_scale,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            per_frame_steps=per_frame_steps,
            nworld=nworld,
            include_image=include_image,
            image_size=image_size,
        )

        # Configure the goal-conditioned task.
        self.target_ahead_fraction = 0.20
        self.target_distance_range_fraction = (0.12, 0.25)
        self.target_angle_range_deg = (-70.0, 70.0)
        self.randomize_target = True
        self.target_radius_fraction = 0.02
        self.target_progress_weight = 100.0
        self.target_reached_bonus = 5.0
        self.target_positions_lbm: Optional[np.ndarray] = None
        self.prev_target_distances: Optional[np.ndarray] = None
        self.target_rng = np.random.default_rng()
        self.last_success = np.zeros(self.nworld, dtype=bool)

    def _single_obs_dim(self) -> int:
        # Omit fluid forces and fixed roll joints from the planar state.
        paired_projected_eel = (
            self.model.nu % 2 == 0
            and self.model.nq - 7 == self.model.nu
            and self.model.nv - 6 == self.model.nu
        )
        if paired_projected_eel:
            yaw_count = self.model.nu // 2
            return 7 + yaw_count + 6 + yaw_count + 4
        return self.model.nq + self.model.nv + 4

    def _create_observation_space(self) -> spaces.Space:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.nworld, self._single_obs_dim() * 2),
            dtype=np.float32,
        )

    def _get_obs(self):
        qpos = self.data.qpos.numpy().astype(np.float32)
        qvel = self.data.qvel.numpy().astype(np.float32)
        obs = np.zeros((self.nworld, self._single_obs_dim()), dtype=np.float32)

        for world_idx in range(self.nworld):
            head_pos = self.solver.flows[world_idx].solid_position.numpy()[0]
            lbm_pos = np.array([head_pos[0] / self.nx, head_pos[1] / self.ny], dtype=np.float32)
            if self.target_positions_lbm is None:
                # Use the normalized fallback before target initialization.
                target_world = np.array(
                    [self.target_point_x * self.nx, self.target_point_y * self.ny],
                    dtype=np.float32,
                )
            else:
                target_world = self.target_positions_lbm[world_idx]
            delta_world = target_world - head_pos
            qw, qx, qy, qz = qpos[world_idx, 3:7]
            yaw = np.arctan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            # Express the goal in the eel's local frame.
            target_delta = np.array(
                [
                    cos_yaw * delta_world[0] + sin_yaw * delta_world[1],
                    -sin_yaw * delta_world[0] + cos_yaw * delta_world[1],
                ],
                dtype=np.float32,
            ) / float(self.ny)
            paired_projected_eel = (
                self.model.nu % 2 == 0
                and self.model.nq - 7 == self.model.nu
                and self.model.nv - 6 == self.model.nu
            )
            if paired_projected_eel:
                # Keep root and yaw states; roll is fixed in 2D.
                planar_qpos = np.concatenate(
                    [qpos[world_idx, :7], qpos[world_idx, 7::2]]
                )
                planar_qvel = np.concatenate(
                    [qvel[world_idx, :6], qvel[world_idx, 6::2]]
                )
            else:
                planar_qpos = qpos[world_idx]
                planar_qvel = qvel[world_idx]
            obs[world_idx] = np.concatenate(
                [planar_qpos, planar_qvel, lbm_pos, target_delta]
            )

        return obs

    def _place_targets_in_front(self, reset_mask: Optional[np.ndarray] = None) -> None:
        if reset_mask is None:
            reset_mask = np.ones(self.nworld, dtype=bool)
        else:
            reset_mask = np.asarray(reset_mask, dtype=bool)

        if self.target_positions_lbm is None:
            self.target_positions_lbm = np.zeros((self.nworld, 2), dtype=np.float32)
        if self.prev_target_distances is None:
            self.prev_target_distances = np.zeros(self.nworld, dtype=np.float32)

        margin = float((self.target_radius_fraction + 0.01) * self.ny)
        for world_idx in range(self.nworld):
            if not reset_mask[world_idx]:
                continue
            head = np.asarray(
                self.solver.flows[world_idx].solid_position.numpy()[0],
                dtype=np.float32,
            )
            if self.randomize_target:
                # Sample range and bearing in the forward sector.
                distance_fraction = self.target_rng.uniform(
                    *self.target_distance_range_fraction
                )
                angle_deg = self.target_rng.uniform(*self.target_angle_range_deg)
            else:
                distance_fraction = self.target_ahead_fraction
                angle_deg = 0.0
            distance = float(distance_fraction * self.ny)
            angle = np.deg2rad(angle_deg)
            local_delta = np.array(
                [np.sin(angle) * distance, np.cos(angle) * distance],
                dtype=np.float32,
            )
            qpos = self.data.qpos.numpy().astype(np.float32)[world_idx]
            qw, qx, qy, qz = qpos[3:7]
            yaw = np.arctan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            world_delta = np.array(
                [
                    cos_yaw * local_delta[0] - sin_yaw * local_delta[1],
                    sin_yaw * local_delta[0] + cos_yaw * local_delta[1],
                ],
                dtype=np.float32,
            )
            target = head + world_delta
            # Keep the target inside the safe fluid domain.
            target[0] = np.clip(target[0], margin, self.nx - margin)
            target[1] = np.clip(target[1], margin, self.ny - margin)
            self.target_positions_lbm[world_idx] = target
            self.prev_target_distances[world_idx] = float(np.linalg.norm(target - head))
            self.last_success[world_idx] = False

        # Share world zero's target with the base termination kernel.
        self.target_point_x = float(self.target_positions_lbm[0, 0] / self.nx)
        self.target_point_y = float(self.target_positions_lbm[0, 1] / self.ny)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        if seed is not None:
            # Re-seed goal sampling with the environment seed.
            self.target_rng = np.random.default_rng(seed)
        super().reset(seed=seed, options=options)
        self._place_targets_in_front()
        observation = self._get_obs()
        return np.concatenate([observation, observation], axis=1)

    def set_target_lbm(self, x: float, y: float, world_idx: int = 0) -> None:
        """Set a target manually without introducing a one-step reward jump."""
        if self.target_positions_lbm is None or self.prev_target_distances is None:
            self._place_targets_in_front()
        margin = float((self.target_radius_fraction + 0.01) * self.ny)
        target = np.array(
            [
                np.clip(float(x), margin, self.nx - margin),
                np.clip(float(y), margin, self.ny - margin),
            ],
            dtype=np.float32,
        )
        self.target_positions_lbm[world_idx] = target
        head = np.asarray(
            self.solver.flows[world_idx].solid_position.numpy()[0],
            dtype=np.float32,
        )
        # Reset distance tracking to avoid a synthetic reward spike.
        self.prev_target_distances[world_idx] = float(np.linalg.norm(target - head))
        self.last_success[world_idx] = False
        if world_idx == 0:
            self.target_point_x = float(target[0] / self.nx)
            self.target_point_y = float(target[1] / self.ny)

    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        super().partial_reset(reset_mask)
        self._place_targets_in_front(reset_mask)
        observation = self._get_obs()
        return np.concatenate([observation, observation], axis=1)

    def _compute_reward(self, instability_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if self.target_positions_lbm is None or self.prev_target_distances is None:
            self._place_targets_in_front()

        current_distances = np.zeros(self.nworld, dtype=np.float32)
        for world_idx in range(self.nworld):
            head = np.asarray(
                self.solver.flows[world_idx].solid_position.numpy()[0],
                dtype=np.float32,
            )
            current_distances[world_idx] = float(
                np.linalg.norm(self.target_positions_lbm[world_idx] - head)
            )

        # Reward normalized reduction in head-to-goal distance.
        reward = self.target_progress_weight * (
            self.prev_target_distances - current_distances
        ) / float(self.ny)
        # Add the arrival bonus inside the success radius.
        reached = current_distances <= self.target_radius_fraction * float(self.ny)
        reward += self.target_reached_bonus * reached.astype(np.float32)
        self.prev_target_distances = current_distances

        if instability_mask is not None:
            reward = np.where(instability_mask, self.anomaly_penalty, reward)

        if np.any(~np.isfinite(reward)):
            reward = np.nan_to_num(reward, nan=self.anomaly_penalty, posinf=self.anomaly_penalty, neginf=self.anomaly_penalty)

        return reward.astype(np.float32)

    def _is_terminated(
        self,
        instability_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Preserve base failures and add goal completion.
        terminated = super()._is_terminated(instability_mask)
        if self.target_positions_lbm is not None:
            distances = np.zeros(self.nworld, dtype=np.float32)
            for world_idx in range(self.nworld):
                head = np.asarray(
                    self.solver.flows[world_idx].solid_position.numpy()[0],
                    dtype=np.float32,
                )
                distances[world_idx] = float(
                    np.linalg.norm(self.target_positions_lbm[world_idx] - head)
                )
            terminated |= distances <= self.target_radius_fraction * float(self.ny)
            self.last_success = distances <= self.target_radius_fraction * float(self.ny)
            wp.copy(
                self._terminated_buffer,
                wp.array(
                    terminated.astype(np.int32),
                    dtype=wp.int32,
                    device=self.solver.device,
                ),
            )
        return terminated

    def _get_info(self) -> Dict[str, Any]:
        # Expose goal metrics to Monitor and TensorBoard.
        if self.target_positions_lbm is None:
            return {"is_success": False, "target_distance_lbm": float("inf")}
        head = np.asarray(
            self.solver.flows[0].solid_position.numpy()[0],
            dtype=np.float32,
        )
        distance = float(np.linalg.norm(self.target_positions_lbm[0] - head))
        return {
            "is_success": bool(self.last_success[0]),
            "target_distance_lbm": distance,
        }
