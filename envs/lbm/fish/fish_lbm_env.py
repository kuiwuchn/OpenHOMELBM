"""
Fish LBM Environment for MuJoCo Warp with nworld support
Based on rl_fish_v2 branch implementation, adapted for parallel training
"""

from ..lbm_fluid_env import LBMFluidEnv
from gym import spaces
import numpy as np
import warp as wp
from typing import Optional, Tuple, Dict, Any, List
from ..lbm_core import HomeFlow


# ============== Warp Kernels for Fish Environment ==============


@wp.kernel
def compute_fish_obs_kernel(
    qfrc_applied: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    qpos: wp.array2d(dtype=wp.float32),  # (nworld, nq)
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    flows: wp.array(dtype=HomeFlow),  # flows array
    obs_out: wp.array2d(dtype=wp.float32),  # (nworld, 17)
    nx: float,
    ny: float,
):
    """
    Compute observation for all worlds in parallel
    Observation: [fx, fy, tau_z, tau_joint1, tau_joint2,
                  x, y, theta_z, vx, vy, omega_z,
                  joint_angles(2), joint_velocities(2),
                  current_x, current_y (normalized)]
    """
    world_idx = wp.tid()
    flow = flows[world_idx]

    # Generalized forces (indices 0,1,5,6,7)
    obs_out[world_idx, 0] = qfrc_applied[world_idx, 0]  # fx
    obs_out[world_idx, 1] = qfrc_applied[world_idx, 1]  # fy
    obs_out[world_idx, 2] = qfrc_applied[world_idx, 5]  # tau_z
    obs_out[world_idx, 3] = qfrc_applied[world_idx, 6]  # tau_joint1
    obs_out[world_idx, 4] = qfrc_applied[world_idx, 7]  # tau_joint2

    # Position (x, y from qpos[0:2])
    obs_out[world_idx, 5] = qpos[world_idx, 0]  # x
    obs_out[world_idx, 6] = qpos[world_idx, 1]  # y

    # Extract z-axis rotation angle from quaternion (qpos[3:7] = w,x,y,z)
    qw = qpos[world_idx, 3]
    qx = qpos[world_idx, 4]
    qy = qpos[world_idx, 5]
    qz = qpos[world_idx, 6]
    theta_z = wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    obs_out[world_idx, 7] = theta_z

    # Velocity (vx, vy, omega_z from qvel[0,1,5])
    obs_out[world_idx, 8] = qvel[world_idx, 0]  # vx
    obs_out[world_idx, 9] = qvel[world_idx, 1]  # vy
    obs_out[world_idx, 10] = qvel[world_idx, 5]  # omega_z

    # Joint angles (qpos[7:9])
    obs_out[world_idx, 11] = qpos[world_idx, 7]
    obs_out[world_idx, 12] = qpos[world_idx, 8]

    # Joint velocities (qvel[6:8])
    obs_out[world_idx, 13] = qvel[world_idx, 6]
    obs_out[world_idx, 14] = qvel[world_idx, 7]

    # Current LBM position (normalized) - head is solid 0
    head_pos = flow.solid_position[0]
    obs_out[world_idx, 15] = head_pos[0] / nx
    obs_out[world_idx, 16] = head_pos[1] / ny


@wp.kernel
def check_termination_kernel(
    flows: wp.array(dtype=HomeFlow),
    solid_max_radii: wp.array(dtype=wp.float32),  # (n_solids,)
    terminated_out: wp.array(dtype=wp.int32),  # (nworld,) output: 1=terminated, 0=not
    nx: float,
    ny: float,
    boundary_margin: float,
    target_x: float,  # normalized target x
    target_y: float,  # normalized target y
    n_solids: int,
):
    """
    Check termination condition for all worlds in parallel

    Termination conditions:
    - Reached target point (distance < 0.0001)
    - Any solid exceeds boundary
    """
    world_idx = wp.tid()
    flow = flows[world_idx]

    # Get head position (solid 0)
    head_pos = flow.solid_position[0]

    # Check if reached target point
    dx = head_pos[0] / nx - target_x
    dy = head_pos[1] / ny - target_y
    dist_sq = dx * dx + dy * dy
    if dist_sq < 0.0001 * 0.0001:
        terminated_out[world_idx] = 1
        return

    # Check boundary for each solid
    x_min = 0.0
    x_max = nx
    y_min = 0.0
    y_max = ny

    for solid_idx in range(n_solids):
        pos = flow.solid_position[solid_idx]
        max_radius = solid_max_radii[solid_idx]

        x = pos[0]
        y = pos[1]

        # Check if solid exceeds boundary
        if (
            x - max_radius < x_min + boundary_margin
            or x + max_radius > x_max - boundary_margin
            or y - max_radius < y_min + boundary_margin
            or y + max_radius > y_max - boundary_margin
        ):
            terminated_out[world_idx] = 1
            return

    terminated_out[world_idx] = 0


@wp.kernel
def check_stability_kernel(
    qpos: wp.array2d(dtype=wp.float32),  # (nworld, nq)
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    instability_out: wp.array(dtype=wp.int32),  # (nworld,) output: 1=unstable, 0=stable
    nq: int,
    nv: int,
):
    """
    Check numerical stability for all worlds in parallel

    Checks for NaN/Inf in qpos and qvel
    """
    world_idx = wp.tid()

    # Check qpos
    for i in range(nq):
        val = qpos[world_idx, i]
        if wp.isnan(val) or wp.isinf(val):
            instability_out[world_idx] = 1
            return

    # Check qvel
    for i in range(nv):
        val = qvel[world_idx, i]
        if wp.isnan(val) or wp.isinf(val):
            instability_out[world_idx] = 1
            return

    instability_out[world_idx] = 0


@wp.kernel
def compute_reward_kernel(
    flows: wp.array(dtype=HomeFlow),
    prev_positions_y: wp.array(dtype=wp.float32),  # (nworld,) previous y positions
    rewards_out: wp.array(dtype=wp.float32),  # (nworld,)
    current_positions_y_out: wp.array(dtype=wp.float32),  # (nworld,) for updating prev
    ny: float,
):
    """
    Compute reward for all worlds in parallel

    Reward = 100.0 * dy / ny (forward movement reward)
    """
    world_idx = wp.tid()
    flow = flows[world_idx]

    # Get current head position (solid 0)
    head_pos = flow.solid_position[0]
    current_y = head_pos[1]

    # Compute y displacement
    dy = current_y - prev_positions_y[world_idx]

    # Compute reward
    rewards_out[world_idx] = 100.0 * dy / ny

    # Store current y for next step
    current_positions_y_out[world_idx] = current_y


@wp.kernel
def compute_smooth_reward_kernel(
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    current_actions: wp.array2d(dtype=wp.float32),  # (nworld, action_dim)
    prev_actions: wp.array2d(dtype=wp.float32),  # (nworld, action_dim)
    smooth_rewards_out: wp.array(dtype=wp.float32),  # (nworld,)
    action_smoothness_weight: float,  # Weight for action smoothness penalty
    rotation_penalty_weight: float,  # Weight for rotation penalty
    phase_reward_weight: float,  # Weight for phase difference reward
):
    """
    Compute smoothness reward for encouraging sinusoidal swimming patterns

    Components:
    1. Action smoothness: penalize large changes between consecutive actions
    2. Rotation penalty: penalize large angular velocity (omega_z)
    3. Phase difference reward: encourage phase lag between joints for traveling wave

    For fish swimming, we want:
    - Smooth, periodic motion (like sin wave)
    - Minimal rotation (swim straight)
    - Phase lag between joints (wave travels from tail to head)
    """
    world_idx = wp.tid()

    # ========== 1. Action Smoothness Penalty ==========
    # Penalize large changes in action (encourage smooth periodic motion)
    # action[0] = joint1 (front), action[1] = joint2 (back/tail)
    action_diff_0 = current_actions[world_idx, 0] - prev_actions[world_idx, 0]
    action_diff_1 = current_actions[world_idx, 1] - prev_actions[world_idx, 1]
    action_change_sq = action_diff_0 * action_diff_0 + action_diff_1 * action_diff_1

    # Smooth transition reward (negative of change magnitude)
    action_smooth_reward = -action_smoothness_weight * action_change_sq

    # ========== 2. Rotation Penalty ==========
    # Penalize large angular velocity omega_z (index 5 in qvel)
    omega_z = qvel[world_idx, 5]
    rotation_penalty = -rotation_penalty_weight * omega_z * omega_z

    # ========== 3. Phase Difference Reward ==========
    # For traveling wave: rear joint should lead front joint
    # If both actions have same sign but different magnitudes with time lag, it's good
    # We approximate this by rewarding when actions have opposite signs (phase difference ~pi/2)
    # or by checking velocity patterns

    # Simple heuristic: reward when joint velocities suggest wave propagation
    # Joint velocities are qvel[6] and qvel[7]
    joint1_vel = qvel[world_idx, 6]  # front joint velocity
    joint2_vel = qvel[world_idx, 7]  # rear joint velocity

    # Reward phase difference: when one joint is moving fast and the other is reversing
    # This encourages traveling wave pattern
    # cross product of velocities with actions indicates phase relationship
    phase_indicator = (
        joint1_vel * current_actions[world_idx, 1]
        - joint2_vel * current_actions[world_idx, 0]
    )
    phase_reward = phase_reward_weight * wp.abs(phase_indicator)


    # ========== Total Smooth Reward ==========
    smooth_rewards_out[world_idx] = (
        action_smooth_reward + rotation_penalty + phase_reward
    )


@wp.kernel
def add_smooth_rewards_kernel(
    rewards: wp.array(dtype=wp.float32),
    smooth_rewards: wp.array(dtype=wp.float32),
):
    """Add smooth rewards to main rewards buffer (in-place on GPU)."""
    world_idx = wp.tid()
    rewards[world_idx] = rewards[world_idx] + smooth_rewards[world_idx]


@wp.kernel
def apply_instability_penalty_kernel(
    rewards: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    instability_mask: wp.array(dtype=wp.int32),
    penalty: float,
):
    """
    Apply instability penalty to rewards and update terminated flags
    """
    world_idx = wp.tid()
    if instability_mask[world_idx] == 1:
        rewards[world_idx] = penalty
        terminated[world_idx] = 1


@wp.kernel
def check_reward_and_force_anomaly_kernel(
    rewards: wp.array(dtype=wp.float32),
    qfrc_applied: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    terminated: wp.array(dtype=wp.int32),
    anomaly_out: wp.array(dtype=wp.int32),  # (nworld,) output: 1=anomaly, 0=normal
    force_threshold: float,
    penalty: float,
):
    """
    Check for reward NaN/Inf and abnormally large forces.
    If anomaly detected, set terminated=1 and apply penalty to reward.

    Args:
        rewards: Reward array to check and modify
        qfrc_applied: Applied generalized forces (nworld, nv)
        terminated: Termination flags to update
        anomaly_out: Output anomaly detection results
        force_threshold: Maximum allowed force magnitude
        penalty: Penalty to apply when anomaly detected
    """
    world_idx = wp.tid()
    anomaly_out[world_idx] = 0

    reward_val = rewards[world_idx]

    # Check if reward is NaN or Inf
    if wp.isnan(reward_val) or wp.isinf(reward_val):
        anomaly_out[world_idx] = 1
        rewards[world_idx] = penalty
        terminated[world_idx] = 1
        return

    # Check for abnormally large forces (check first 8 components: fx, fy, fz, tau_x, tau_y, tau_z, joint1, joint2)
    for i in range(8):
        force_val = qfrc_applied[world_idx, i]
        if wp.isnan(force_val) or wp.isinf(force_val):
            anomaly_out[world_idx] = 1
            rewards[world_idx] = penalty
            terminated[world_idx] = 1
            return
        if wp.abs(force_val) > force_threshold:
            anomaly_out[world_idx] = 1
            rewards[world_idx] = penalty
            terminated[world_idx] = 1
            return


@wp.kernel
def reset_prev_y_kernel(
    flows: wp.array(dtype=HomeFlow),
    reset: wp.array(dtype=wp.int32),
    prev_y: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    if reset[w] != 0:
        prev_y[w] = flows[w].solid_position[0][1]


@wp.kernel
def reset_prev_actions_kernel(
    reset: wp.array(dtype=wp.int32),
    prev_actions: wp.array2d(dtype=wp.float32),
):
    w = wp.tid()
    if reset[w] != 0:
        for j in range(prev_actions.shape[1]):
            prev_actions[w, j] = 0.0


# ============== Fish LBM Environment Class ==============


class FishLBMEnv(LBMFluidEnv):
    """
    Fish swimming environment with LBM fluid simulation
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
        include_image: bool = False,
        image_size: Tuple[int, int] = (64, 64),
    ):
        # Use a module-relative path.
        if xml_path is None:
            import os

            xml_path = os.path.join(os.path.dirname(__file__), "fish_2d_v3.xml")

        # Store image settings before base initialization.
        self.include_image = include_image
        self.image_size = image_size

        if solid_config is None:
            # Head and body use geom names (avoid convex hull issues)
            # Tail uses body name (merge tail_stem + tail_fin into convex hull for cone shape)
            # Smaller size: head 0.1, body 0.1, tail 0.1+0.1
            # Spacing calculation: spacing * 80
            solid_config = [
                {
                    "solid_id": 0,
                    "body_id": 1,  # root body
                    "body_or_geom_name": "head_geom",
                    "lbm_position": (200, 250 - 0.05 * nx * lbm_scale),
                    "is_body": False,  # Use geom name
                },
                {
                    "solid_id": 1,
                    "body_id": 2,  # body
                    "body_or_geom_name": "body_geom",
                    "lbm_position": (200, 250 - (0.1 + 0.05) * nx * lbm_scale),
                    "is_body": False,
                },
                {
                    "solid_id": 2,
                    "body_id": 3,  # tail body - use convex hull to merge tail_stem and tail_fin
                    "body_or_geom_name": "tail",
                    "lbm_position": (200, 250 - (0.1 + 0.1 + 0.15) * nx * lbm_scale),
                    "is_body": True,  # Use body name, convex hull will form cone shape
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

        self.action_scale = 1.0
        self.video_path = "results/fish_lbm_episode.mp4"
        self.video_vmax = 0.1  # Velocity field maximum

        # Target position for reward calculation
        self.target_point_x = 0.5
        self.target_point_y = 0.8
        self.reward_weight = 1.0

        # Boundary parameters (LBM coordinate system)
        self.boundary_margin = 1.0  # Boundary safety margin (LBM grid units)

        # Store maximum radius for each rigid body (LBM coordinate system)
        self.solid_max_radii = None
        self.solid_max_radii_wp = None  # Warp array version

        # For displacement reward calculation (per world)
        self.prev_positions_y_wp = None  # Warp array: (nworld,) - only y component

        # Pre-allocate Warp buffers for kernels
        self._obs_buffer = wp.zeros((nworld, 17), dtype=wp.float32)
        self._terminated_buffer = wp.zeros(nworld, dtype=wp.int32)
        self._instability_buffer = wp.zeros(nworld, dtype=wp.int32)
        self._rewards_buffer = wp.zeros(nworld, dtype=wp.float32)
        self._current_y_buffer = wp.zeros(nworld, dtype=wp.float32)
        self._anomaly_buffer = wp.zeros(
            nworld, dtype=wp.int32
        )  # For reward/force anomaly detection

        # Anomaly detection parameters
        self.force_threshold = 1e5  # Maximum allowed force magnitude
        self.anomaly_penalty = -10.0  # Penalty when anomaly detected

        # ========== Smoothness Reward Parameters ==========
        # These encourage sinusoidal swimming patterns
        self.action_smoothness_weight = 0.5  # Penalize jerky movements
        self.rotation_penalty_weight = 0.1  # Penalize spinning in place
        self.forward_reward_weight = 100.0  # Weight for forward movement
        self.phase_reward_weight = 0.05  # Reward for phase difference between joints

        # Pre-allocate buffer for smooth rewards
        self._smooth_rewards_buffer = wp.zeros(nworld, dtype=wp.float32)
        self._current_actions_wp = None  # Will be set in step()
        self._prev_actions_wp = None  # Will be set in reset()

        # Configure image observations.
        if self.include_image:
            from .lbm_func import get_velocity_rgb

            self._get_velocity_rgb = get_velocity_rgb
            # Preallocate the image buffer.
            self._image_buffer = np.zeros(
                (nworld, self.image_size[0], self.image_size[1], 3), dtype=np.uint8
            )
            # Normalize with the maximum lattice velocity.
            self._max_velocity = 0.3  # Tune for the expected flow.

    def _create_observation_space(self) -> spaces.Space:
        """
        Create observation space - [obs_before, obs_after] = 2×17 = 34 dims per world + optional image

        Returns:
            gym.spaces.Space: Observation space definition
        """
        obs_dim = 17 * 2  # temporal stacking: before + after action

        if self.include_image:
            # Return vector and image observations.
            return spaces.Dict(
                {
                    "vector": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.nworld, obs_dim),
                        dtype=np.float32,
                    ),
                    "image": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.nworld, self.image_size[0], self.image_size[1], 3),
                        dtype=np.uint8,
                    ),
                }
            )
        else:
            return spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.nworld, obs_dim), dtype=np.float32
            )

    def _get_velocity_image(self) -> np.ndarray:
        """
        Return three-channel velocity images for all worlds.

        R: velocity magnitude
        G: x velocity
        B: y velocity

        Returns:
            np.ndarray: uint8 images shaped (nworld, H, W, 3)
        """
        import cv2

        for world_idx in range(self.nworld):
            flow = self.solver.flows[world_idx]

            # Compute the three-channel velocity field.
            wp.launch(
                self._get_velocity_rgb,
                dim=(flow.nx, flow.ny),
                inputs=[flow, self._max_velocity],
            )

        wp.synchronize()

        # Read and process each image.
        for world_idx in range(self.nworld):
            flow = self.solver.flows[world_idx]
            # (nx, ny, 3) -> (ny, nx, 3)
            img_rgb = flow.u_img_rgb.numpy().transpose(1, 0, 2)  # (ny, nx, 3)
            # Flip y so LBM +y points upward in the image.
            img_rgb = np.flipud(img_rgb)

            # Convert to uint8 [0, 255].
            img_uint8 = (img_rgb * 255).astype(np.uint8)

            # Resize the image.
            # Downscale with AREA to reduce blur; upscale with CUBIC for smoother edges
            interp = (
                cv2.INTER_AREA
                if img_uint8.shape[0] >= self.image_size[0]
                and img_uint8.shape[1] >= self.image_size[1]
                else cv2.INTER_CUBIC
            )
            img_resized = cv2.resize(img_uint8, self.image_size, interpolation=interp)

            self._image_buffer[world_idx] = img_resized

        return self._image_buffer.copy()

    def _get_obs(self):
        """
        Get current observation for all worlds using Warp kernel

        Returns:
            If include_image=False: np.ndarray (nworld, 17)
            If include_image=True: dict with 'vector' and 'image'
        """
        # Launch Warp kernel - all computation on GPU
        wp.launch(
            compute_fish_obs_kernel,
            dim=self.nworld,
            inputs=[
                self.data.qfrc_applied,
                self.data.qpos,
                self.data.qvel,
                self.solver.flows_wp,
                self._obs_buffer,
                float(self.nx),
                float(self.ny),
            ],
        )

        vector_obs = self._obs_buffer.numpy()

        if self.include_image:
            image_obs = self._get_velocity_image()
            return {
                "vector": vector_obs,
                "image": image_obs,
            }
        else:
            return vector_obs

    def set_new_y(self, y_ratio: float):
        """Set new Y position for all solids"""
        for i in range(self.solid_num):
            old_pos = self.solid_config[i]["lbm_position"]
            self.solid_config[i]["lbm_position"] = (old_pos[0], y_ratio * self.ny)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        """Reset environment and record initial positions for all worlds"""
        observation = super().reset(seed=seed, options=options)

        # Store maximum radius for each rigid body (only on first reset)
        if self.solid_max_radii is None:
            self.solid_max_radii = self.solver.flows[0].solid_max_radius.numpy().copy()
            self.solid_max_radii_wp = wp.array(self.solid_max_radii, dtype=wp.float32)

        # Initialize previous y positions for all worlds (Warp array)
        # Extract head (solid 0) y positions using a simple kernel
        init_y = np.zeros(self.nworld, dtype=np.float32)
        for world_idx in range(self.nworld):
            flow = self.solver.flows[world_idx]
            head_pos = flow.solid_position.numpy()[0]
            init_y[world_idx] = head_pos[1]
        self.prev_positions_y_wp = wp.array(init_y, dtype=wp.float32)

        # Initialize previous actions (GPU only)
        action_dim = self.action_space.shape[1]  # (nworld, action_dim)
        self._prev_actions_wp = wp.zeros((self.nworld, action_dim), dtype=wp.float32)

        return observation

    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        """
        Reset only specific worlds indicated by reset_mask.
        Extends parent's partial_reset to handle Fish-specific state.

        Args:
            reset_mask: Boolean array of shape (nworld,) where True indicates world needs reset

        Returns:
            np.ndarray: New observations for all worlds
        """
        # Call parent's partial_reset to handle MuJoCo and LBM reset
        obs = super().partial_reset(reset_mask)

        if not np.any(reset_mask):
            return obs

        reset_mask_wp = wp.array(reset_mask.astype(np.int32), dtype=wp.int32)

        wp.launch(
            reset_prev_y_kernel,
            dim=self.nworld,
            inputs=[self.solver.flows_wp, reset_mask_wp, self.prev_positions_y_wp],
        )

        if self._prev_actions_wp is None:
            self._prev_actions_wp = wp.zeros(
                (self.nworld, self.action_space.shape[1]), dtype=wp.float32
            )

        wp.launch(
            reset_prev_actions_kernel,
            dim=self.nworld,
            inputs=[reset_mask_wp, self._prev_actions_wp],
        )

        return obs

    def step(self, action: np.ndarray):
        """
        Override step to properly handle reward/force anomaly detection.

        The order is important:
        1. Execute simulation
        2. Get observation
        3. Check basic termination (boundary, target reached)
        4. Compute reward (which also detects NaN/Inf and large force anomalies)
        5. The anomaly check in _compute_reward updates terminated buffer
        6. Return final terminated state

        Args:
            action: (nworld, action_dim) action array

        Returns:
            observation, reward, done, info (gym API)
        """
        # Save current action
        self.current_actions = np.array(action).copy()
        self._current_actions_wp = wp.array(self.current_actions, dtype=wp.float32)

        observation_before = self._get_obs()

        # Handle action shape - expect (nworld, action_dim)
        action = (
            np.clip(action, self.action_space.low, self.action_space.high)
            * self.action_scale
        )
        wp.copy(self.data.ctrl, wp.array(action, dtype=wp.float32))

        # Execute physics simulation step (including LBM-MuJoCo coupling)
        self._simulation_step()

        # Update step counts
        self.current_steps += 1

        # Get new observation
        observation_after = self._get_obs()

        observation = np.concatenate((observation_before, observation_after), axis=1)

        # Check numerical stability ONCE before computing reward and termination
        instability_mask = self._check_numerical_stability()

        # Check for NaN/Inf in observations and mark those worlds as unstable
        # Handle both dict and array observations
        if isinstance(observation, dict):
            obs_for_check = observation["vector"]
        else:
            obs_for_check = observation
        obs_nan_mask = np.any(np.isnan(obs_for_check) | np.isinf(obs_for_check), axis=1)
        if np.any(obs_nan_mask):
            # Combine with instability mask
            if instability_mask is None:
                instability_mask = obs_nan_mask
            else:
                instability_mask = instability_mask | obs_nan_mask
            # Convert NaN/Inf for network input safety
            if isinstance(observation, dict):
                observation["vector"] = np.nan_to_num(
                    observation["vector"], nan=0.0, posinf=0.0, neginf=0.0
                )
            else:
                observation = np.nan_to_num(
                    observation, nan=0.0, posinf=0.0, neginf=0.0
                )
            print(
                f"[WARNING] NaN/Inf observation detected in {np.sum(obs_nan_mask)} worlds"
            )

        # Check basic termination condition first (boundary, target reached)
        # This sets _terminated_buffer for basic conditions
        self._is_terminated(instability_mask)

        # Compute reward - this also checks for NaN/Inf rewards and large forces
        # and updates _terminated_buffer if anomaly detected
        reward = self._compute_reward(instability_mask)

        # Get final terminated state (includes anomaly detection from _compute_reward)
        terminated = self._terminated_buffer.numpy().astype(bool)

        # Apply termination penalty
        reward[terminated] -= 1.0

        # Final safety check: ensure reward is never NaN/Inf
        # This is a safety net in case any edge case slips through
        if np.any(np.isnan(reward)) or np.any(np.isinf(reward)):
            bad_mask = np.isnan(reward) | np.isinf(reward)
            reward[bad_mask] = self.anomaly_penalty - 1.0  # Apply penalty
            terminated[bad_mask] = True  # Force termination
            print(
                f"[WARNING] NaN/Inf reward detected in {np.sum(bad_mask)} worlds, applying penalty and terminating"
            )

        truncated = np.array(self.current_steps >= self.max_episode_steps)

        # Combine terminated and truncated into done (gym API)
        done = terminated | truncated

        # Get additional information
        info = self._get_info()
        info["terminated"] = terminated
        info["truncated"] = truncated

        # Update previous actions (GPU buffer)
        self._prev_actions_wp = wp.array(self.current_actions, dtype=wp.float32)

        return observation, reward, done, info

    def _compute_reward(
        self, instability_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute reward function using Warp kernel

        Reward components:
        1. Forward movement reward (main objective)
        2. Action smoothness penalty (encourage sinusoidal patterns)
        3. Rotation penalty (discourage spinning)

        Also checks for reward NaN/Inf and abnormally large forces.
        If anomaly detected, terminates the episode and applies penalty.

        Args:
            instability_mask: Pre-computed instability mask (optional)

        Returns:
            np.ndarray: Reward array of shape (nworld,)
        """
        # ========== 1. Forward Movement Reward ==========
        wp.launch(
            compute_reward_kernel,
            dim=self.nworld,
            inputs=[
                self.solver.flows_wp,
                self.prev_positions_y_wp,
                self._rewards_buffer,
                self._current_y_buffer,
                float(self.ny),
            ],
        )

        # Update previous y positions (swap buffers)
        wp.copy(self.prev_positions_y_wp, self._current_y_buffer)

        # ========== 2. Smoothness Reward (action smoothness + rotation penalty + phase reward) ==========
        if self._prev_actions_wp is not None and self._current_actions_wp is not None:
            wp.launch(
                compute_smooth_reward_kernel,
                dim=self.nworld,
                inputs=[
                    self.data.qvel,
                    self._current_actions_wp,
                    self._prev_actions_wp,
                    self._smooth_rewards_buffer,
                    self.action_smoothness_weight,
                    self.rotation_penalty_weight,
                    self.phase_reward_weight,
                ],
            )

            # Add smooth rewards to main rewards (directly on GPU)
            wp.launch(
                add_smooth_rewards_kernel,
                dim=self.nworld,
                inputs=[
                    self._rewards_buffer,
                    self._smooth_rewards_buffer,
                ],
            )

        if instability_mask is not None:
            # Convert instability mask to Warp array if needed
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32)
            wp.launch(
                apply_instability_penalty_kernel,
                dim=self.nworld,
                inputs=[
                    self._rewards_buffer,
                    self._terminated_buffer,  # Not used here but needed for signature
                    instability_wp,
                    -10.0,
                ],
            )

        # Check for reward NaN/Inf and abnormally large forces
        # This will update both rewards and terminated buffers
        wp.launch(
            check_reward_and_force_anomaly_kernel,
            dim=self.nworld,
            inputs=[
                self._rewards_buffer,
                self.data.qfrc_applied,
                self._terminated_buffer,
                self._anomaly_buffer,
                self.force_threshold,
                self.anomaly_penalty,
            ],
        )

        # Only convert to numpy at the end
        return self._rewards_buffer.numpy()

    def _is_terminated(
        self, instability_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Check termination condition for all worlds using Warp kernel

        Termination conditions:
        - Any part of the robot reaches simulation boundary
        - Numerical instability detected

        Args:
            instability_mask: Pre-computed instability mask (optional)

        Returns:
            np.ndarray: Boolean array of shape (nworld,)
        """
        # Reset terminated buffer
        self._terminated_buffer.zero_()

        # Launch termination check kernel
        wp.launch(
            check_termination_kernel,
            dim=self.nworld,
            inputs=[
                self.solver.flows_wp,
                self.solid_max_radii_wp,
                self._terminated_buffer,
                float(self.nx),
                float(self.ny),
                self.boundary_margin,
                self.target_point_x,
                self.target_point_y,
                len(self.solid_config),
            ],
        )

        # Apply instability termination if provided (on GPU)
        if instability_mask is not None:
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32)
            wp.launch(
                apply_instability_penalty_kernel,
                dim=self.nworld,
                inputs=[
                    self._rewards_buffer,  # Not used here but needed for signature
                    self._terminated_buffer,
                    instability_wp,
                    0.0,  # penalty not used for termination
                ],
            )

        # Convert to numpy boolean array at the end
        return self._terminated_buffer.numpy().astype(bool)

    def _check_numerical_stability(self) -> np.ndarray:
        """
        Check numerical stability for all worlds using Warp kernel

        Returns:
            np.ndarray: Boolean mask where True indicates instability
        """
        # Reset instability buffer
        self._instability_buffer.zero_()

        # Launch stability check kernel
        wp.launch(
            check_stability_kernel,
            dim=self.nworld,
            inputs=[
                self.data.qpos,
                self.data.qvel,
                self._instability_buffer,
                self.model.nq,
                self.model.nv,
            ],
        )

        # Convert to numpy boolean array at the end
        return self._instability_buffer.numpy().astype(bool)
