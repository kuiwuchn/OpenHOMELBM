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

        self.action_smoothness_weight = 0.15
        self.rotation_penalty_weight = 0.1
        self.phase_reward_weight = 0.03

    def _single_obs_dim(self) -> int:
        return self.model.nv + self.model.nq + self.model.nv + 2

    def _create_observation_space(self) -> spaces.Space:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.nworld, self._single_obs_dim() * 2),
            dtype=np.float32,
        )

    def _get_obs(self):
        qfrc = self.data.qfrc_applied.numpy().astype(np.float32)
        qpos = self.data.qpos.numpy().astype(np.float32)
        qvel = self.data.qvel.numpy().astype(np.float32)
        obs = np.zeros((self.nworld, self._single_obs_dim()), dtype=np.float32)

        for world_idx in range(self.nworld):
            head_pos = self.solver.flows[world_idx].solid_position.numpy()[0]
            lbm_pos = np.array([head_pos[0] / self.nx, head_pos[1] / self.ny], dtype=np.float32)
            obs[world_idx] = np.concatenate([qfrc[world_idx], qpos[world_idx], qvel[world_idx], lbm_pos])

        return obs

    def _compute_reward(self, instability_mask: Optional[np.ndarray] = None) -> np.ndarray:
        current_y = np.zeros(self.nworld, dtype=np.float32)
        for world_idx in range(self.nworld):
            current_y[world_idx] = self.solver.flows[world_idx].solid_position.numpy()[0][1]

        prev_y = self.prev_positions_y_wp.numpy().astype(np.float32)
        reward = 100.0 * (current_y - prev_y) / float(self.ny)
        self.prev_positions_y_wp = wp.array(current_y, dtype=wp.float32, device=self.solver.device)


        if self._prev_actions_wp is not None and hasattr(self, "current_actions"):
            current_actions = np.asarray(self.current_actions, dtype=np.float32)
            prev_actions = self._prev_actions_wp.numpy().astype(np.float32)
            action_diff = current_actions - prev_actions
            reward -= self.action_smoothness_weight * np.mean(action_diff * action_diff, axis=1)

            qvel = self.data.qvel.numpy().astype(np.float32)
            reward -= self.rotation_penalty_weight * qvel[:, 5] * qvel[:, 5]

            joint_vel = qvel[:, 6 : 6 + current_actions.shape[1]]
            if current_actions.shape[1] > 1 and joint_vel.shape[1] >= current_actions.shape[1]:
                phase = joint_vel[:, :-1] * current_actions[:, 1:] - joint_vel[:, 1:] * current_actions[:, :-1]
                reward += self.phase_reward_weight * np.mean(np.abs(phase), axis=1)

        if instability_mask is not None:
            reward = np.where(instability_mask, self.anomaly_penalty, reward)

        if np.any(~np.isfinite(reward)):
            reward = np.nan_to_num(reward, nan=self.anomaly_penalty, posinf=self.anomaly_penalty, neginf=self.anomaly_penalty)

        return reward.astype(np.float32)
