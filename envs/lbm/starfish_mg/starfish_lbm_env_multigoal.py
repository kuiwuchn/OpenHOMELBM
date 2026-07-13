"""
Starfish LBM Environment for MuJoCo Warp with nworld support
Multi-goal version: robot starts from center, navigates to 9 goal points sequentially
"""

from ..lbm_fluid_env import LBMFluidEnv
from gym import spaces
import numpy as np
import warp as wp
from typing import Optional, Tuple, Dict, Any, List
from ..lbm_core import HomeFlow


# ============== Warp Kernels for Starfish Environment ==============


@wp.kernel
def compute_starfish_obs_multigoal_kernel(
    qfrc_applied: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    qpos: wp.array2d(dtype=wp.float32),  # (nworld, nq)
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    flows: wp.array(dtype=HomeFlow),  # flows array
    goal_positions: wp.array2d(dtype=wp.float32),  # (nworld, 2) - current goal for each world
    obs_out: wp.array2d(dtype=wp.float32),  # (nworld, 25)
    nx: float,
    ny: float,
):
    """
    Compute observation for all worlds in parallel
    
    Observation structure (25 dims):
    [0-6]: generalized forces (fx, fy, tau_z, tau_up, tau_down, tau_left, tau_right)
    [7-9]: position (x, y, theta_z)
    [10-12]: velocity (vx, vy, omega_z)
    [13-16]: joint angles (up, down, left, right)
    [17-20]: joint velocities (up, down, left, right)
    [21-22]: current position normalized (x, y)
    [23-24]: current goal position normalized (x, y)
    """
    world_idx = wp.tid()
    flow = flows[world_idx]

    # Generalized forces (indices 0,1,5,6,7,8,9 in qfrc_applied)
    obs_out[world_idx, 0] = qfrc_applied[world_idx, 0]  # fx
    obs_out[world_idx, 1] = qfrc_applied[world_idx, 1]  # fy
    obs_out[world_idx, 2] = qfrc_applied[world_idx, 5]  # tau_z
    obs_out[world_idx, 3] = qfrc_applied[world_idx, 6]  # tau_up
    obs_out[world_idx, 4] = qfrc_applied[world_idx, 7]  # tau_down
    obs_out[world_idx, 5] = qfrc_applied[world_idx, 8]  # tau_left
    obs_out[world_idx, 6] = qfrc_applied[world_idx, 9]  # tau_right

    # Position (x, y from qpos[0:2])
    obs_out[world_idx, 7] = qpos[world_idx, 0]  # x
    obs_out[world_idx, 8] = qpos[world_idx, 1]  # y

    # Extract z-axis rotation angle from quaternion (qpos[3:7] = w,x,y,z)
    qw = qpos[world_idx, 3]
    qx = qpos[world_idx, 4]
    qy = qpos[world_idx, 5]
    qz = qpos[world_idx, 6]
    theta_z = wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    obs_out[world_idx, 9] = theta_z

    # Velocity (vx, vy, omega_z from qvel[0,1,5])
    obs_out[world_idx, 10] = qvel[world_idx, 0]  # vx
    obs_out[world_idx, 11] = qvel[world_idx, 1]  # vy
    obs_out[world_idx, 12] = qvel[world_idx, 5]  # omega_z

    # Joint angles (qpos[7:11])
    obs_out[world_idx, 13] = qpos[world_idx, 7]   # joint_up
    obs_out[world_idx, 14] = qpos[world_idx, 8]   # joint_down
    obs_out[world_idx, 15] = qpos[world_idx, 9]   # joint_left
    obs_out[world_idx, 16] = qpos[world_idx, 10]  # joint_right

    # Joint velocities (qvel[6:10])
    obs_out[world_idx, 17] = qvel[world_idx, 6]   # joint_up_vel
    obs_out[world_idx, 18] = qvel[world_idx, 7]   # joint_down_vel
    obs_out[world_idx, 19] = qvel[world_idx, 8]   # joint_left_vel
    obs_out[world_idx, 20] = qvel[world_idx, 9]   # joint_right_vel

    # Current LBM position (normalized) - center is solid 0
    center_pos = flow.solid_position[0]
    obs_out[world_idx, 21] = center_pos[0] / nx
    obs_out[world_idx, 22] = center_pos[1] / ny

    # Current goal position (normalized)
    obs_out[world_idx, 23] = goal_positions[world_idx, 0]
    obs_out[world_idx, 24] = goal_positions[world_idx, 1]


@wp.kernel
def check_starfish_boundary_kernel(
    flows: wp.array(dtype=HomeFlow),
    solid_max_radii: wp.array(dtype=wp.float32),  # (n_solids,)
    terminated_out: wp.array(dtype=wp.int32),  # (nworld,)
    nx: float,
    ny: float,
    boundary_margin: float,
    n_solids: int,
):
    """
    Check boundary termination condition for all worlds in parallel
    Only checks if robot hits boundary (not goal reaching)
    """
    world_idx = wp.tid()
    flow = flows[world_idx]

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
def check_goal_reached_kernel(
    flows: wp.array(dtype=HomeFlow),
    goal_positions: wp.array2d(dtype=wp.float32),  # (nworld, 2)
    goal_reached_out: wp.array(dtype=wp.int32),  # (nworld,)
    nx: float,
    ny: float,
    goal_threshold: float,  # normalized distance threshold
):
    """
    Check if robot reached current goal for all worlds
    """
    world_idx = wp.tid()
    flow = flows[world_idx]

    # Get center position (solid 0)
    center_pos = flow.solid_position[0]
    current_x = center_pos[0] / nx
    current_y = center_pos[1] / ny

    goal_x = goal_positions[world_idx, 0]
    goal_y = goal_positions[world_idx, 1]

    dx = current_x - goal_x
    dy = current_y - goal_y
    dist_sq = dx * dx + dy * dy

    if dist_sq < goal_threshold * goal_threshold:
        goal_reached_out[world_idx] = 1
    else:
        goal_reached_out[world_idx] = 0


@wp.kernel
def check_starfish_stability_kernel(
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
def compute_goal_reward_kernel(
    flows: wp.array(dtype=HomeFlow),
    goal_positions: wp.array2d(dtype=wp.float32),  # (nworld, 2)
    prev_dist: wp.array(dtype=wp.float32),  # (nworld,)
    rewards_out: wp.array(dtype=wp.float32),  # (nworld,)
    current_dist_out: wp.array(dtype=wp.float32),  # (nworld,)
    nx: float,
    ny: float,
):
    """
    Compute reward based on distance to goal
    Reward = 100.0 * (prev_dist - current_dist) (positive when getting closer)
    """
    world_idx = wp.tid()
    flow = flows[world_idx]

    center_pos = flow.solid_position[0]
    current_x = center_pos[0] / nx
    current_y = center_pos[1] / ny

    goal_x = goal_positions[world_idx, 0]
    goal_y = goal_positions[world_idx, 1]

    dx = current_x - goal_x
    dy = current_y - goal_y
    current_dist = wp.sqrt(dx * dx + dy * dy)

    # Reward for getting closer to goal
    dist_improvement = prev_dist[world_idx] - current_dist
    rewards_out[world_idx] = 100.0 * dist_improvement

    current_dist_out[world_idx] = current_dist


@wp.kernel
def compute_starfish_smooth_reward_kernel(
    qvel: wp.array2d(dtype=wp.float32),
    current_actions: wp.array2d(dtype=wp.float32),
    prev_actions: wp.array2d(dtype=wp.float32),
    smooth_rewards_out: wp.array(dtype=wp.float32),
    action_smoothness_weight: float,
    rotation_penalty_weight: float,
):
    """
    Compute smoothness reward for starfish
    - Penalize jerky movements
    - Penalize spinning in place
    """
    world_idx = wp.tid()

    # Action smoothness penalty (4 joints)
    action_change_sq = 0.0
    for i in range(4):
        diff = current_actions[world_idx, i] - prev_actions[world_idx, i]
        action_change_sq = action_change_sq + diff * diff

    action_smooth_reward = -action_smoothness_weight * action_change_sq

    # Rotation penalty
    omega_z = qvel[world_idx, 5]
    rotation_penalty = -rotation_penalty_weight * omega_z * omega_z

    smooth_rewards_out[world_idx] = action_smooth_reward + rotation_penalty


@wp.kernel
def apply_starfish_instability_penalty_kernel(
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
def check_starfish_anomaly_kernel(
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

    # Check forces (10 components for starfish: 6 free joint + 4 hinge joints)
    for i in range(10):
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
def reset_starfish_prev_dist_kernel(
    flows: wp.array(dtype=HomeFlow),
    goal_positions: wp.array2d(dtype=wp.float32),
    reset: wp.array(dtype=wp.int32),
    prev_dist: wp.array(dtype=wp.float32),
    nx: float,
    ny: float,
):
    """Reset previous distance for specific worlds"""
    w = wp.tid()
    if reset[w] != 0:
        center_pos = flows[w].solid_position[0]
        current_x = center_pos[0] / nx
        current_y = center_pos[1] / ny
        goal_x = goal_positions[w, 0]
        goal_y = goal_positions[w, 1]
        dx = current_x - goal_x
        dy = current_y - goal_y
        prev_dist[w] = wp.sqrt(dx * dx + dy * dy)


@wp.kernel
def reset_starfish_prev_actions_kernel(
    reset: wp.array(dtype=wp.int32),
    prev_actions: wp.array2d(dtype=wp.float32),
):
    w = wp.tid()
    if reset[w] != 0:
        for j in range(prev_actions.shape[1]):
            prev_actions[w, j] = 0.0


# ============== Starfish LBM Environment Class ==============


class StarfishLBMEnvMultigoal(LBMFluidEnv):
    """
    Starfish (cross-shaped) swimming environment with LBM fluid simulation
    Multi-goal version: robot navigates to 9 goal points sequentially
    
    Structure:
    - Center body (cylinder)
    - 4 arms: up (+y), down (-y), left (-x), right (+x)
    - 4 hinge joints for arm control
    
    Goals:
    - 9 goal points in a 3x3 grid
    - Robot starts from center
    - When reaching a goal, next goal is selected (different from current)
    - Episode lasts 2000 steps
    """

    def __init__(
        self,
        xml_path: str = None,
        solid_config: Optional[List[Dict[str, Any]]] = None,
        nx: int = 600,
        ny: int = 600,
        lbm_scale: float = 0.2,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 2000,
        per_frame_steps: int = 30,
        nworld: int = 1,
        goal_threshold: float = 0.05,  # normalized distance to consider goal reached
    ):
        if xml_path is None:
            import os
            xml_path = os.path.join(os.path.dirname(__file__), "starfish_2d_v1.xml")

        if solid_config is None:
            # Starfish configuration: center + 4 arms
            arm_offset = 0.06 * nx * lbm_scale
            arm_length = 0.12 * nx * lbm_scale

            center_x = nx * 0.5
            center_y = ny * 0.5
            
            solid_config = [
                {
                    "solid_id": 0,
                    "body_id": 1,
                    "body_or_geom_name": "center_geom",
                    "lbm_position": (center_x, center_y),
                    "is_body": False,
                },
                {
                    "solid_id": 1,
                    "body_id": 2,
                    "body_or_geom_name": "arm_up_geom",
                    "lbm_position": (center_x, center_y + arm_offset + arm_length / 2),
                    "is_body": False,
                },
                {
                    "solid_id": 2,
                    "body_id": 3,
                    "body_or_geom_name": "arm_down_geom",
                    "lbm_position": (center_x, center_y - arm_offset - arm_length / 2),
                    "is_body": False,
                },
                {
                    "solid_id": 3,
                    "body_id": 4,
                    "body_or_geom_name": "arm_left_geom",
                    "lbm_position": (center_x - arm_offset - arm_length / 2, center_y),
                    "is_body": False,
                },
                {
                    "solid_id": 4,
                    "body_id": 5,
                    "body_or_geom_name": "arm_right_geom",
                    "lbm_position": (center_x + arm_offset + arm_length / 2, center_y),
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

        self.action_scale = 1.0
        self.video_path = "results/starfish_multigoal_episode.mp4"
        self.video_vmax = 0.1

        # Goal configuration - 9 goals in 3x3 grid (normalized coordinates)
        # Avoid center (0.5, 0.5) since robot starts there
        self.goal_positions_list = [
            (0.25, 0.25),  # bottom-left
            (0.50, 0.25),  # bottom-center
            (0.75, 0.25),  # bottom-right
            (0.25, 0.50),  # middle-left
            (0.75, 0.50),  # middle-right
            (0.25, 0.75),  # top-left
            (0.50, 0.75),  # top-center
            (0.75, 0.75),  # top-right
        ]
        self.num_goals = len(self.goal_positions_list)
        self.goal_threshold = goal_threshold

        # Boundary parameters
        self.boundary_margin = 1.0

        # Store maximum radius for each rigid body
        self.solid_max_radii = None
        self.solid_max_radii_wp = None

        # Current goal index for each world
        self.current_goal_idx = np.zeros(nworld, dtype=np.int32)
        
        # Goal history to avoid selecting same goal
        self.goal_history = [[] for _ in range(nworld)]
        
        # Goals reached counter
        self.goals_reached = np.zeros(nworld, dtype=np.int32)

        # Current goal positions (nworld, 2)
        self._goal_positions_wp = wp.zeros((nworld, 2), dtype=wp.float32)

        # For distance-based reward calculation
        self._prev_dist_wp = wp.zeros(nworld, dtype=wp.float32)
        self._current_dist_buffer = wp.zeros(nworld, dtype=wp.float32)

        # Pre-allocate Warp buffers
        self._obs_buffer = wp.zeros((nworld, 25), dtype=wp.float32)
        self._terminated_buffer = wp.zeros(nworld, dtype=wp.int32)
        self._instability_buffer = wp.zeros(nworld, dtype=wp.int32)
        self._rewards_buffer = wp.zeros(nworld, dtype=wp.float32)
        self._goal_reached_buffer = wp.zeros(nworld, dtype=wp.int32)
        self._anomaly_buffer = wp.zeros(nworld, dtype=wp.int32)

        # Anomaly detection parameters
        self.force_threshold = 1e5
        self.anomaly_penalty = -10.0

        # Smoothness reward parameters
        self.action_smoothness_weight = 0.5
        self.rotation_penalty_weight = 0.1

        self._smooth_rewards_buffer = wp.zeros(nworld, dtype=wp.float32)
        self._current_actions_wp = None
        self._prev_actions_wp = None

        # Goal reaching bonus
        self.goal_reached_bonus = 10.0

    def _create_observation_space(self) -> spaces.Space:
        """
        Create observation space - [obs_before, obs_after] = 2×25 = 50 dims per world
        (25 = 23 original + 2 for goal position)
        """
        obs_dim = 25 * 2  # temporal stacking
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.nworld, obs_dim), dtype=np.float32
        )

    def _select_next_goal(self, world_idx: int) -> int:
        """
        Select next goal for a world, avoiding recently visited goals
        """
        # Get available goals (not in recent history)
        history = self.goal_history[world_idx]
        available = [i for i in range(self.num_goals) if i not in history[-3:]]  # Avoid last 3 goals
        
        if not available:
            # If all goals recently visited, allow any except current
            current = self.current_goal_idx[world_idx]
            available = [i for i in range(self.num_goals) if i != current]
        
        # Randomly select from available
        next_goal = np.random.choice(available)
        return next_goal

    def _update_goal_positions_wp(self):
        """Update the Warp array with current goal positions"""
        goal_pos_np = np.zeros((self.nworld, 2), dtype=np.float32)
        for w in range(self.nworld):
            goal_idx = self.current_goal_idx[w]
            goal_pos_np[w, 0] = self.goal_positions_list[goal_idx][0]
            goal_pos_np[w, 1] = self.goal_positions_list[goal_idx][1]
        wp.copy(self._goal_positions_wp, wp.array(goal_pos_np, dtype=wp.float32))

    def _get_obs(self) -> np.ndarray:
        """Get current observation for all worlds using Warp kernel"""
        wp.launch(
            compute_starfish_obs_multigoal_kernel,
            dim=self.nworld,
            inputs=[
                self.data.qfrc_applied,
                self.data.qpos,
                self.data.qvel,
                self.solver.flows_wp,
                self._goal_positions_wp,
                self._obs_buffer,
                float(self.nx),
                float(self.ny),
            ],
        )
        return self._obs_buffer.numpy()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> np.ndarray:
        """Reset environment and initialize goals"""
        if seed is not None:
            np.random.seed(seed)

        observation = super().reset(seed=seed, options=options)

        if self.solid_max_radii is None:
            self.solid_max_radii = self.solver.flows[0].solid_max_radius.numpy().copy()
            self.solid_max_radii_wp = wp.array(self.solid_max_radii, dtype=wp.float32)

        # Reset goal tracking for all worlds
        self.goals_reached[:] = 0
        self.goal_history = [[] for _ in range(self.nworld)]
        
        # Select initial goal for each world
        for w in range(self.nworld):
            self.current_goal_idx[w] = self._select_next_goal(w)
            self.goal_history[w].append(self.current_goal_idx[w])

        # Update goal positions in Warp
        self._update_goal_positions_wp()

        # Initialize previous distance
        self._init_prev_dist()

        # Initialize previous actions
        action_dim = self.action_space.shape[1]
        self._prev_actions_wp = wp.zeros((self.nworld, action_dim), dtype=wp.float32)

        return self._get_obs()

    def _init_prev_dist(self):
        """Initialize previous distance for all worlds"""
        dist_np = np.zeros(self.nworld, dtype=np.float32)
        for w in range(self.nworld):
            flow = self.solver.flows[w]
            center_pos = flow.solid_position.numpy()[0]
            current_x = center_pos[0] / self.nx
            current_y = center_pos[1] / self.ny
            goal_x, goal_y = self.goal_positions_list[self.current_goal_idx[w]]
            dist_np[w] = np.sqrt((current_x - goal_x)**2 + (current_y - goal_y)**2)
        self._prev_dist_wp = wp.array(dist_np, dtype=wp.float32)

    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        """Reset only specific worlds indicated by reset_mask"""
        obs = super().partial_reset(reset_mask)

        if not np.any(reset_mask):
            return self._get_obs()

        # Reset goal tracking for reset worlds
        for w in range(self.nworld):
            if reset_mask[w]:
                self.goals_reached[w] = 0
                self.goal_history[w] = []
                self.current_goal_idx[w] = self._select_next_goal(w)
                self.goal_history[w].append(self.current_goal_idx[w])

        # Update goal positions
        self._update_goal_positions_wp()

        # Reset previous distance
        reset_mask_wp = wp.array(reset_mask.astype(np.int32), dtype=wp.int32)
        wp.launch(
            reset_starfish_prev_dist_kernel,
            dim=self.nworld,
            inputs=[
                self.solver.flows_wp,
                self._goal_positions_wp,
                reset_mask_wp,
                self._prev_dist_wp,
                float(self.nx),
                float(self.ny),
            ],
        )

        if self._prev_actions_wp is None:
            self._prev_actions_wp = wp.zeros(
                (self.nworld, self.action_space.shape[1]), dtype=wp.float32
            )

        wp.launch(
            reset_starfish_prev_actions_kernel,
            dim=self.nworld,
            inputs=[reset_mask_wp, self._prev_actions_wp],
        )

        return self._get_obs()

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

        # Check if goals are reached
        self._check_goals_reached()

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
        info["goals_reached"] = self.goals_reached.copy()

        self._prev_actions_wp = wp.array(self.current_actions, dtype=wp.float32)

        return observation, reward, done, info

    def _check_goals_reached(self):
        """Check if any world reached its goal and update accordingly"""
        wp.launch(
            check_goal_reached_kernel,
            dim=self.nworld,
            inputs=[
                self.solver.flows_wp,
                self._goal_positions_wp,
                self._goal_reached_buffer,
                float(self.nx),
                float(self.ny),
                self.goal_threshold,
            ],
        )

        goal_reached = self._goal_reached_buffer.numpy()
        
        for w in range(self.nworld):
            if goal_reached[w]:
                self.goals_reached[w] += 1
                # Select next goal
                self.current_goal_idx[w] = self._select_next_goal(w)
                self.goal_history[w].append(self.current_goal_idx[w])

        # Update goal positions if any changed
        if np.any(goal_reached):
            self._update_goal_positions_wp()
            # Reset previous distance for worlds that reached goal
            reset_mask_wp = wp.array(goal_reached.astype(np.int32), dtype=wp.int32)
            wp.launch(
                reset_starfish_prev_dist_kernel,
                dim=self.nworld,
                inputs=[
                    self.solver.flows_wp,
                    self._goal_positions_wp,
                    reset_mask_wp,
                    self._prev_dist_wp,
                    float(self.nx),
                    float(self.ny),
                ],
            )

    def _compute_reward(
        self, instability_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute reward function using Warp kernel"""
        # Distance-based reward
        wp.launch(
            compute_goal_reward_kernel,
            dim=self.nworld,
            inputs=[
                self.solver.flows_wp,
                self._goal_positions_wp,
                self._prev_dist_wp,
                self._rewards_buffer,
                self._current_dist_buffer,
                float(self.nx),
                float(self.ny),
            ],
        )

        wp.copy(self._prev_dist_wp, self._current_dist_buffer)

        # Add goal reached bonus
        goal_reached = self._goal_reached_buffer.numpy()
        rewards_np = self._rewards_buffer.numpy()
        rewards_np[goal_reached.astype(bool)] += self.goal_reached_bonus
        wp.copy(self._rewards_buffer, wp.array(rewards_np, dtype=wp.float32))

        # Smoothness reward
        if self._prev_actions_wp is not None and self._current_actions_wp is not None:
            wp.launch(
                compute_starfish_smooth_reward_kernel,
                dim=self.nworld,
                inputs=[
                    self.data.qvel,
                    self._current_actions_wp,
                    self._prev_actions_wp,
                    self._smooth_rewards_buffer,
                    self.action_smoothness_weight,
                    self.rotation_penalty_weight,
                ],
            )

            rewards_np = self._rewards_buffer.numpy()
            smooth_rewards_np = self._smooth_rewards_buffer.numpy()
            combined_rewards = rewards_np + smooth_rewards_np
            wp.copy(self._rewards_buffer, wp.array(combined_rewards, dtype=wp.float32))

        if instability_mask is not None:
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32)
            wp.launch(
                apply_starfish_instability_penalty_kernel,
                dim=self.nworld,
                inputs=[
                    self._rewards_buffer,
                    self._terminated_buffer,
                    instability_wp,
                    -10.0,
                ],
            )

        wp.launch(
            check_starfish_anomaly_kernel,
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
        """Check termination condition for all worlds"""
        self._terminated_buffer.zero_()

        wp.launch(
            check_starfish_boundary_kernel,
            dim=self.nworld,
            inputs=[
                self.solver.flows_wp,
                self.solid_max_radii_wp,
                self._terminated_buffer,
                float(self.nx),
                float(self.ny),
                self.boundary_margin,
                len(self.solid_config),
            ],
        )

        if instability_mask is not None:
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32)
            wp.launch(
                apply_starfish_instability_penalty_kernel,
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
            check_starfish_stability_kernel,
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

    def get_current_goal(self, world_idx: int = 0) -> Tuple[float, float]:
        """Get current goal position for a world (for visualization)"""
        goal_idx = self.current_goal_idx[world_idx]
        return self.goal_positions_list[goal_idx]
