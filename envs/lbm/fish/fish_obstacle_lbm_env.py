"""
Fish LBM Environment with Obstacle
Fish robot navigating around a static obstacle to swim upward
"""

from ..lbm_fluid_env import LBMFluidEnv
from gym import spaces
import numpy as np
import warp as wp
from typing import Optional, Tuple, Dict, Any, List
from ..lbm_core import HomeFlow


# ============== Warp Kernels for Fish Obstacle Environment ==============


@wp.kernel
def compute_fish_obstacle_obs_kernel(
    qfrc_applied: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    qpos: wp.array2d(dtype=wp.float32),  # (nworld, nq)
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    flows: wp.array(dtype=HomeFlow),  # flows array
    obs_out: wp.array2d(dtype=wp.float32),  # (nworld, 19)
    nx: float,
    ny: float,
    obstacle_x: float,  # obstacle position x (normalized)
    obstacle_y: float,  # obstacle position y (normalized)
):
    """
    Compute observation for all worlds in parallel
    
    Observation structure (19 dims):
    [0-4]: generalized forces (fx, fy, tau_z, tau_joint1, tau_joint2)
    [5-7]: position (x, y, theta_z)
    [8-10]: velocity (vx, vy, omega_z)
    [11-12]: joint angles
    [13-14]: joint velocities
    [15-16]: current position normalized (x, y)
    [17-18]: relative position to obstacle (dx, dy) normalized
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
    current_x = head_pos[0] / nx
    current_y = head_pos[1] / ny
    obs_out[world_idx, 15] = current_x
    obs_out[world_idx, 16] = current_y

    # Relative position to obstacle (normalized)
    obs_out[world_idx, 17] = obstacle_x - current_x
    obs_out[world_idx, 18] = obstacle_y - current_y


@wp.kernel
def compute_obstacle_reward_kernel(
    flows: wp.array(dtype=HomeFlow),
    prev_positions_y: wp.array(dtype=wp.float32),
    rewards_out: wp.array(dtype=wp.float32),
    current_positions_y_out: wp.array(dtype=wp.float32),
    ny: float,
    obstacle_x: float,  # obstacle position x (LBM coords)
    obstacle_y: float,  # obstacle position y (LBM coords)
    obstacle_radius: float,  # obstacle radius (LBM coords)
    avoidance_weight: float,  # weight for obstacle avoidance reward
):
    """
    Compute reward for all worlds in parallel
    Reward = forward movement reward + obstacle avoidance bonus
    """
    world_idx = wp.tid()
    flow = flows[world_idx]

    head_pos = flow.solid_position[0]
    current_y = head_pos[1]
    current_x = head_pos[0]

    # Forward movement reward
    dy = current_y - prev_positions_y[world_idx]
    forward_reward = 100.0 * dy / ny

    # Obstacle avoidance: reward for maintaining safe distance
    dx_obs = current_x - obstacle_x
    dy_obs = current_y - obstacle_y
    dist_to_obstacle = wp.sqrt(dx_obs * dx_obs + dy_obs * dy_obs)
    
    # Safe distance is 2.5x obstacle radius
    safe_distance = obstacle_radius * 2.5
    
    # Avoidance reward: penalty when too close
    avoidance_reward = 0.0
    if dist_to_obstacle < safe_distance:
        # Penalty for being too close (linear decay)
        avoidance_reward = -avoidance_weight * (1.0 - dist_to_obstacle / safe_distance)

    rewards_out[world_idx] = forward_reward + avoidance_reward
    current_positions_y_out[world_idx] = current_y


@wp.kernel
def check_obstacle_termination_kernel(
    flows: wp.array(dtype=HomeFlow),
    solid_max_radii: wp.array(dtype=wp.float32),  # (n_solids,)
    terminated_out: wp.array(dtype=wp.int32),  # (nworld,)
    nx: float,
    ny: float,
    boundary_margin: float,
    target_x: float,
    target_y: float,
    n_fish_solids: int,  # number of fish solids (exclude obstacle)
):
    """
    Check termination condition: boundary and target reached only.
    No obstacle collision check (obstacle is a box, circle approximation is inaccurate).
    """
    world_idx = wp.tid()
    flow = flows[world_idx]

    # Get head position (solid 0)
    head_pos = flow.solid_position[0]

    # Check if reached target point
    dx = head_pos[0] / nx - target_x
    dy = head_pos[1] / ny - target_y
    dist_sq = dx * dx + dy * dy
    if dist_sq < 0.01 * 0.01:
        terminated_out[world_idx] = 1
        return

    # Check boundary for each fish solid
    x_min = 0.0
    x_max = nx
    y_min = 0.0
    y_max = ny

    for solid_idx in range(n_fish_solids):
        pos = flow.solid_position[solid_idx]
        max_radius = solid_max_radii[solid_idx]

        x = pos[0]
        y = pos[1]

        # Boundary check
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
def check_fish_stability_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    instability_out: wp.array(dtype=wp.int32),
    nq: int,
    nv: int,
):
    """Check numerical stability for all worlds"""
    world_idx = wp.tid()

    for i in range(nq):
        val = qpos[world_idx, i]
        if wp.isnan(val) or wp.isinf(val):
            instability_out[world_idx] = 1
            return

    for i in range(nv):
        val = qvel[world_idx, i]
        if wp.isnan(val) or wp.isinf(val):
            instability_out[world_idx] = 1
            return

    instability_out[world_idx] = 0


@wp.kernel
def compute_smooth_reward_kernel(
    qvel: wp.array2d(dtype=wp.float32),
    current_actions: wp.array2d(dtype=wp.float32),
    prev_actions: wp.array2d(dtype=wp.float32),
    smooth_rewards_out: wp.array(dtype=wp.float32),
    action_smoothness_weight: float,
    rotation_penalty_weight: float,
    phase_reward_weight: float,
):
    """Compute smoothness reward for fish swimming"""
    world_idx = wp.tid()

    # Action smoothness penalty
    action_diff_0 = current_actions[world_idx, 0] - prev_actions[world_idx, 0]
    action_diff_1 = current_actions[world_idx, 1] - prev_actions[world_idx, 1]
    action_change_sq = action_diff_0 * action_diff_0 + action_diff_1 * action_diff_1
    action_smooth_reward = -action_smoothness_weight * action_change_sq

    # Rotation penalty
    omega_z = qvel[world_idx, 5]
    rotation_penalty = -rotation_penalty_weight * omega_z * omega_z

    # Phase difference reward
    joint1_vel = qvel[world_idx, 6]
    joint2_vel = qvel[world_idx, 7]
    phase_indicator = (
        joint1_vel * current_actions[world_idx, 1]
        - joint2_vel * current_actions[world_idx, 0]
    )
    phase_reward = phase_reward_weight * wp.abs(phase_indicator)


    smooth_rewards_out[world_idx] = action_smooth_reward + rotation_penalty + phase_reward


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
    """Apply instability penalty to rewards and update terminated flags"""
    world_idx = wp.tid()
    if instability_mask[world_idx] == 1:
        rewards[world_idx] = penalty
        terminated[world_idx] = 1


@wp.kernel
def check_reward_anomaly_kernel(
    rewards: wp.array(dtype=wp.float32),
    qfrc_applied: wp.array2d(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    anomaly_out: wp.array(dtype=wp.int32),
    force_threshold: float,
    penalty: float,
):
    """Check for reward NaN/Inf and abnormally large forces"""
    world_idx = wp.tid()
    anomaly_out[world_idx] = 0

    reward_val = rewards[world_idx]
    if wp.isnan(reward_val) or wp.isinf(reward_val):
        anomaly_out[world_idx] = 1
        rewards[world_idx] = penalty
        terminated[world_idx] = 1
        return

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


# ============== Fish Obstacle LBM Environment Class ==============


class FishObstacleLBMEnv(LBMFluidEnv):
    """
    Fish swimming environment with a static obstacle.
    The fish needs to navigate around the obstacle to swim upward.
    
    Initial configuration:
    - Fish starts at lower position with head pointing upward (+y)
    - Obstacle is placed in front of the fish
    - Goal: swim upward while avoiding the obstacle
    """

    def __init__(
        self,
        xml_path: str = None,
        solid_config: Optional[List[Dict[str, Any]]] = None,
        nx: int = 400,
        ny: int = 600,
        lbm_scale: float = 0.3,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        per_frame_steps: int = 30,
        nworld: int = 1,
        obstacle_position: Tuple[float, float] = None,  # LBM coordinates
    ):
        # Set default XML path to obstacle version
        if xml_path is None:
            import os
            xml_path = os.path.join(os.path.dirname(__file__), "fish_2d_obstacle.xml")

        # Obstacle parameters (in LBM coordinates)
        # Default: place obstacle at (200, 380) - above the fish tail (closer now)
        if obstacle_position is None:
            obstacle_position = (200, 250 + 1.625 * nx * lbm_scale)
        self.obstacle_lbm_x = float(obstacle_position[0])
        self.obstacle_lbm_y = float(obstacle_position[1])
        self.obstacle_radius = 0.50 * nx * lbm_scale

        # Fish initial position (fish center at y=250, head pointing down)
        fish_start_y = 250
        
        # Number of fish solids (excluding obstacle)
        self.n_fish_solids = 3

        if solid_config is None:
            # LBM scale factor
            scale = nx * lbm_scale
            
            center_x = 200
            center_y = fish_start_y
            
            # Head is at -0.05 relative to root (pointing -y, downward)
            head_y_offset = -0.05 * scale
            # Body center is at +0.05 relative to root
            body_y_offset = 0.05 * scale
            # Tail center is at +0.15 relative to root
            tail_y_offset = 0.15 * scale
            
            solid_config = [
                # Fish parts (solid_id 0-2)
                {
                    "solid_id": 0,
                    "body_id": 2,  # root body - body_id 2 because obstacle is body_id 1
                    "body_or_geom_name": "head_geom",
                    "lbm_position": (center_x, center_y + head_y_offset),
                    "is_body": False,
                },
                {
                    "solid_id": 1,
                    "body_id": 3,  # body
                    "body_or_geom_name": "body_geom",
                    "lbm_position": (center_x, center_y + body_y_offset),
                    "is_body": False,
                },
                {
                    "solid_id": 2,
                    "body_id": 4,  # tail body
                    "body_or_geom_name": "tail",
                    "lbm_position": (center_x, center_y + tail_y_offset),
                    "is_body": True,
                },
                # Static obstacle (solid_id 3)
                {
                    "solid_id": 3,
                    "body_id": 1,  # obstacle body
                    "body_or_geom_name": "obstacle_geom",
                    "lbm_position": (self.obstacle_lbm_x, self.obstacle_lbm_y),
                    "is_body": False,
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

        # No initial rotation needed - fish is already facing +y in XML

        self.action_scale = 1.0
        self.video_path = "results/fish_obstacle_lbm_episode.mp4"
        self.video_vmax = 0.1

        # Target position for reward calculation
        self.target_point_x = 0.5
        self.target_point_y = 0.85
        self.reward_weight = 1.0

        # Boundary parameters
        self.boundary_margin = 1.0

        # Store maximum radius for each rigid body
        self.solid_max_radii = None
        self.solid_max_radii_wp = None

        # For displacement reward calculation
        self.prev_positions_y_wp = None

        # Pre-allocate Warp buffers
        self._obs_buffer = wp.zeros((nworld, 19), dtype=wp.float32)
        self._terminated_buffer = wp.zeros(nworld, dtype=wp.int32)
        self._instability_buffer = wp.zeros(nworld, dtype=wp.int32)
        self._rewards_buffer = wp.zeros(nworld, dtype=wp.float32)
        self._current_y_buffer = wp.zeros(nworld, dtype=wp.float32)
        self._anomaly_buffer = wp.zeros(nworld, dtype=wp.int32)

        # Anomaly detection parameters
        self.force_threshold = 1e5
        self.anomaly_penalty = -10.0

        # Smoothness reward parameters
        self.action_smoothness_weight = 0.5
        self.rotation_penalty_weight = 0.1
        self.forward_reward_weight = 100.0
        self.phase_reward_weight = 0.05

        # Obstacle avoidance weight
        self.avoidance_weight = 5.0

        self._smooth_rewards_buffer = wp.zeros(nworld, dtype=wp.float32)
        self._current_actions_wp = None
        self._prev_actions_wp = None

    def _create_observation_space(self) -> spaces.Space:
        """Create observation space - [obs_before, obs_after] = 2×19 = 38 dims per world"""
        obs_dim = 19 * 2  # temporal stacking
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.nworld, obs_dim), dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        """Get current observation for all worlds using Warp kernel"""
        wp.launch(
            compute_fish_obstacle_obs_kernel,
            dim=self.nworld,
            inputs=[
                self.data.qfrc_applied,
                self.data.qpos,
                self.data.qvel,
                self.solver.flows_wp,
                self._obs_buffer,
                float(self.nx),
                float(self.ny),
                self.obstacle_lbm_x / self.nx,  # normalized
                self.obstacle_lbm_y / self.ny,  # normalized
            ],
        )
        return self._obs_buffer.numpy()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        """Reset environment and record initial positions"""
        observation = super().reset(seed=seed, options=options)

        if self.solid_max_radii is None:
            self.solid_max_radii = self.solver.flows[0].solid_max_radius.numpy().copy()
            self.solid_max_radii_wp = wp.array(self.solid_max_radii, dtype=wp.float32)

        init_y = np.zeros(self.nworld, dtype=np.float32)
        for world_idx in range(self.nworld):
            flow = self.solver.flows[world_idx]
            head_pos = flow.solid_position.numpy()[0]
            init_y[world_idx] = head_pos[1]
        self.prev_positions_y_wp = wp.array(init_y, dtype=wp.float32)

        action_dim = self.action_space.shape[1]
        self._prev_actions_wp = wp.zeros((self.nworld, action_dim), dtype=wp.float32)

        return observation

    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        """Reset only specific worlds indicated by reset_mask"""
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
        """Execute one environment step"""
        self.current_actions = np.array(action).copy()
        self._current_actions_wp = wp.array(self.current_actions, dtype=wp.float32)

        observation_before = self._get_obs()

        action = (
            np.clip(action, self.action_space.low, self.action_space.high)
            * self.action_scale
        )
        wp.copy(self.data.ctrl, wp.array(action, dtype=wp.float32))

        self._simulation_step()
        self.current_steps += 1

        observation_after = self._get_obs()
        observation = np.concatenate((observation_before, observation_after), axis=1)

        instability_mask = self._check_numerical_stability()

        obs_nan_mask = np.any(np.isnan(observation) | np.isinf(observation), axis=1)
        if np.any(obs_nan_mask):
            if instability_mask is None:
                instability_mask = obs_nan_mask
            else:
                instability_mask = instability_mask | obs_nan_mask
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

        self._is_terminated(instability_mask)
        reward = self._compute_reward(instability_mask)

        terminated = self._terminated_buffer.numpy().astype(bool)
        reward[terminated] -= 1.0

        if np.any(np.isnan(reward)) or np.any(np.isinf(reward)):
            bad_mask = np.isnan(reward) | np.isinf(reward)
            reward[bad_mask] = self.anomaly_penalty - 1.0
            terminated[bad_mask] = True

        truncated = np.array(self.current_steps >= self.max_episode_steps)
        done = terminated | truncated

        info = self._get_info()
        info["terminated"] = terminated
        info["truncated"] = truncated

        self._prev_actions_wp = wp.array(self.current_actions, dtype=wp.float32)

        return observation, reward, done, info

    def _compute_reward(
        self, instability_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute reward function with obstacle avoidance"""
        # Forward movement + obstacle avoidance reward
        wp.launch(
            compute_obstacle_reward_kernel,
            dim=self.nworld,
            inputs=[
                self.solver.flows_wp,
                self.prev_positions_y_wp,
                self._rewards_buffer,
                self._current_y_buffer,
                float(self.ny),
                self.obstacle_lbm_x,
                self.obstacle_lbm_y,
                self.obstacle_radius,
                self.avoidance_weight,
            ],
        )

        wp.copy(self.prev_positions_y_wp, self._current_y_buffer)

        # Smoothness reward
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
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32)
            wp.launch(
                apply_instability_penalty_kernel,
                dim=self.nworld,
                inputs=[
                    self._rewards_buffer,
                    self._terminated_buffer,
                    instability_wp,
                    -10.0,
                ],
            )

        wp.launch(
            check_reward_anomaly_kernel,
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

        return self._rewards_buffer.numpy()

    def _is_terminated(
        self, instability_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Check termination condition (boundary and target only, no obstacle collision)"""
        self._terminated_buffer.zero_()

        wp.launch(
            check_obstacle_termination_kernel,
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
                self.n_fish_solids,
            ],
        )

        if instability_mask is not None:
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32)
            wp.launch(
                apply_instability_penalty_kernel,
                dim=self.nworld,
                inputs=[
                    self._rewards_buffer,
                    self._terminated_buffer,
                    instability_wp,
                    0.0,
                ],
            )

        return self._terminated_buffer.numpy().astype(bool)

    def _check_numerical_stability(self) -> np.ndarray:
        """Check numerical stability for all worlds"""
        self._instability_buffer.zero_()

        wp.launch(
            check_fish_stability_kernel,
            dim=self.nworld,
            inputs=[
                self.data.qpos,
                self.data.qvel,
                self._instability_buffer,
                self.model.nq,
                self.model.nv,
            ],
        )

        return self._instability_buffer.numpy().astype(bool)
