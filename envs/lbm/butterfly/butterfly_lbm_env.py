"""
Butterfly LBM Environment for MuJoCo Warp with nworld support
Adapted for DreamerV3 training
"""

from ..lbm_fluid_env import LBMFluidEnv
from gym import spaces
import numpy as np
import warp as wp
import os
from typing import Optional, Tuple, Dict, Any, List
from ..lbm_fluid_env_func import (
    extract_solid_position_kernel,
    extract_solid_position_2d_kernel,
    compute_butterfly_obs,
    compute_butterfly_reward,
    check_butterfly_termination,
)


class ButterflyLBMEnv(LBMFluidEnv):
    """
    Butterfly swimming environment with LBM fluid simulation
    Supports parallel training with nworld environments
    """

    def __init__(
        self,
        xml_path: str = None,
        solid_config: Optional[List[Dict[str, Any]]] = None,
        nx: int = 400,
        ny: int = 600,
        lbm_scale: float = 0.2,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        per_frame_steps: int = 30,
        nworld: int = 1,
    ):
        """
        Initialize Butterfly LBM Environment

        Args:
            xml_path: Path to MuJoCo XML model
            solid_config: Configuration for solid bodies (optional, uses default if None)
            nx, ny: LBM grid dimensions
            lbm_scale: MuJoCo to LBM scaling ratio
            render_mode: Rendering mode ('human', 'rgb_array', None)
            max_episode_steps: Maximum steps per episode
            per_frame_steps: Sub-simulation steps per environment step
            nworld: Number of parallel environments
        """
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "butterfly_2d_v1.xml")
        
        if solid_config is None:
            # Default butterfly configuration: center body + two wings
            solid_config = [
                {
                    "solid_id": 0,
                    "body_id": 1,
                    "body_or_geom_name": "center",
                    "lbm_position": (200, 200),
                },
                {
                    "solid_id": 1,
                    "body_id": 2,
                    "body_or_geom_name": "left_wing",
                    "lbm_position": (200 - 0.344 * nx * lbm_scale, 200),
                },
                {
                    "solid_id": 2,
                    "body_id": 3,
                    "body_or_geom_name": "right_wing",
                    "lbm_position": (200 + 0.344 * nx * lbm_scale, 200),
                },
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
        )

        # Action scaling
        self.action_scale = 0.03

        # Video parameters
        self.video_path = "results/butterfly_lbm_episode.mp4"
        self.video_vmax = 0.1  # Velocity field maximum value

        # Target position for reward calculation (normalized)
        self.target_point_x = 0.5
        self.target_point_y = 0.7
        self.reward_weight = 1.0

        # Boundary parameters (LBM coordinate system)
        self.boundary_margin = 1.0  # Boundary safety margin (LBM grid units)

        # Numerical stability parameters
        self.enable_stability_check = True
        self.instability_penalty = -10.0
        self.max_force = 1000.0
        self.max_velocity = 50.0
        self.max_angular_velocity = 50.0

        # Store maximum radius for each rigid body
        self.solid_max_radii = None
        self.solid_max_radii_wp = None

        # Pre-allocate Warp buffers for kernels
        self._obs_buffer = wp.zeros((nworld, 19), dtype=wp.float32)
        self._reward_buffer = wp.zeros(nworld, dtype=wp.float32)
        self._terminated_buffer = wp.zeros(nworld, dtype=wp.uint8)
        self._solid_positions_buffer = wp.zeros((nworld, 2), dtype=wp.float32)
        self._all_solid_positions_buffer = wp.zeros(
            (nworld, len(solid_config), 2), dtype=wp.float32
        )

    def _create_observation_space(self) -> spaces.Space:
        """
        Create observation space - 19-dimensional vector per world

        Observation structure:
        [fx, fy, tau_z, tau_left, tau_right,  # 5D: generalized forces
         x, y, theta_z,                        # 3D: position and angle
         vx, vy, omega_z,                      # 3D: velocity
         joint_left, joint_right,              # 2D: joint angles
         joint_left_vel, joint_right_vel,      # 2D: joint velocities
         current_x, current_y,                 # 2D: current position (normalized)
         target_x, target_y]                   # 2D: target position

        Returns:
            gym.spaces.Space: Observation space definition
        """
        obs_dim = 19 * 2  # temporal stacking: [obs_before, obs_after]

        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.nworld, obs_dim), dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        """
        Get current observation for all worlds using Warp kernel

        Returns:
            np.ndarray: (nworld, 19) observation array
        """
        # Get center body positions from all solvers
        center_solid_id = self.solid_config[0]["solid_id"]
        wp.launch(
            extract_solid_position_kernel,
            dim=self.nworld,
            inputs=[
                self.solver.flows_wp,
                center_solid_id,
                self._solid_positions_buffer,
            ],
        )

        # Launch kernel to compute observations
        wp.launch(
            compute_butterfly_obs,
            dim=self.nworld,
            inputs=[
                self.data.qfrc_applied,
                self.data.qpos,
                self.data.qvel,
                self._solid_positions_buffer,
                self._obs_buffer,
                self.nx,
                self.ny,
                self.target_point_x,
                self.target_point_y,
            ],
        )

        return self._obs_buffer.numpy()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Reset environment and record initial position

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation: Initial observation array
        """
        observation = super().reset(seed=seed, options=options)

        # Record initial center position
        center_body_id = self.solid_config[0]["body_id"]
        xpos_np = self.data.xpos.numpy()
        self.prev_center_pos = xpos_np[0, center_body_id].copy()

        # Store maximum radius for each rigid body (only on first reset)
        if self.solid_max_radii is None:
            self.solid_max_radii = self.solver.flows[0].solid_max_radius.numpy().copy()
            self.solid_max_radii_wp = wp.array(self.solid_max_radii, dtype=wp.float32)

        return observation

    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        """
        Reset only specific worlds indicated by reset_mask.
        Extends parent's partial_reset to handle Butterfly-specific state.

        Args:
            reset_mask: Boolean array of shape (nworld,) where True indicates world needs reset

        Returns:
            np.ndarray: New observations for all worlds
        """
        obs = super().partial_reset(reset_mask)

        # No additional Butterfly-specific state to reset currently
        # (prev_center_pos is not used in reward calculation for this env)

        return obs

    def _compute_reward(
        self, instability_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute reward function using Warp kernel

        Reward is based on distance to target point (exponential decay).

        Args:
            instability_mask: Pre-computed instability mask (optional)

        Returns:
            np.ndarray: Reward array of shape (nworld,)
        """
        # Get center body positions
        center_solid_id = self.solid_config[0]["solid_id"]
        wp.launch(
            extract_solid_position_kernel,
            dim=self.nworld,
            inputs=[
                self.solver.flows_wp,
                center_solid_id,
                self._solid_positions_buffer,
            ],
        )

        # Launch kernel to compute rewards
        wp.launch(
            compute_butterfly_reward,
            dim=self.nworld,
            inputs=[
                self._solid_positions_buffer,
                self._reward_buffer,
                self.nx,
                self.ny,
                self.target_point_x,
                self.target_point_y,
                self.reward_weight,
            ],
        )

        rewards = self._reward_buffer.numpy()

        # Apply instability penalty if mask provided
        if instability_mask is not None and np.any(instability_mask):
            rewards[instability_mask] = self.instability_penalty

        return rewards

    def _check_numerical_stability(self) -> np.ndarray:
        """
        Check for numerical instability in the current state

        Checks for:
        - NaN or Inf in observations
        - Extreme forces
        - Extreme velocities
        - Extreme angular velocities

        Returns:
            np.ndarray: Boolean mask (nworld,) indicating which worlds are unstable
        """
        unstable = np.zeros(self.nworld, dtype=bool)

        # Get current observation
        obs = self._obs_buffer.numpy()

        # Check for NaN or Inf
        has_nan = np.any(np.isnan(obs), axis=1)
        has_inf = np.any(np.isinf(obs), axis=1)
        unstable |= has_nan | has_inf

        # Check for extreme forces (indices 0-4)
        max_forces = np.max(np.abs(obs[:, 0:5]), axis=1)
        unstable |= max_forces > self.max_force

        # Check for extreme velocities (indices 8-10)
        max_linear_vel = np.max(np.abs(obs[:, 8:10]), axis=1)
        unstable |= max_linear_vel > self.max_velocity

        max_angular_vel = np.abs(obs[:, 10])
        unstable |= max_angular_vel > self.max_angular_velocity

        # Check for extreme joint velocities (indices 13-14)
        max_joint_vel = np.max(np.abs(obs[:, 13:15]), axis=1)
        unstable |= max_joint_vel > self.max_angular_velocity

        # Log when instability is detected
        if np.any(unstable):
            unstable_count = np.sum(unstable)
            print(
                f"⚠️  Numerical instability detected in {unstable_count}/{self.nworld} worlds at step {self.current_steps[0]}"
            )

            first_unstable = np.where(unstable)[0][0]
            print(f"   World {first_unstable} state:")
            print(f"     Forces: {obs[first_unstable, 0:5]}")
            print(f"     Velocities: {obs[first_unstable, 8:11]}")
            print(f"     Joint velocities: {obs[first_unstable, 13:15]}")
            print(
                f"     Has NaN: {has_nan[first_unstable]}, Has Inf: {has_inf[first_unstable]}"
            )

        return unstable

    def _is_terminated(
        self, instability_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Check termination condition for all worlds using Warp kernel

        Termination conditions:
        - Reached target point
        - Any part of the robot reaches simulation boundary
        - Numerical instability detected

        Args:
            instability_mask: Pre-computed instability mask (optional)

        Returns:
            np.ndarray: Boolean array of shape (nworld,)
        """
        n_solids = len(self.solid_config)

        # Extract all solid positions
        for solid_idx in range(n_solids):
            wp.launch(
                extract_solid_position_2d_kernel,
                dim=self.nworld,
                inputs=[
                    self.solver.flows_wp,
                    solid_idx,
                    self._all_solid_positions_buffer,
                ],
            )

        # Launch kernel to check termination
        wp.launch(
            check_butterfly_termination,
            dim=self.nworld,
            inputs=[
                self._all_solid_positions_buffer,
                self.solid_max_radii_wp,
                self._terminated_buffer,
                self.nx,
                self.ny,
                self.target_point_x,
                self.target_point_y,
                self.boundary_margin,
                n_solids,
            ],
        )

        terminated = self._terminated_buffer.numpy().astype(bool)

        # Add numerical instability to termination
        if instability_mask is not None and np.any(instability_mask):
            terminated |= instability_mask

        return terminated
