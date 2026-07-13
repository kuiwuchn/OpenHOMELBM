"""
3D Fish LBM Environment for MuJoCo Warp with nworld support
Multi-goal version: robot navigates to multiple goal points sequentially in 3D space

Based on 2D starfish_multigoal and 3D fish environment.
Uses gym (not gymnasium) for compatibility with dreamer_vec_wrapper.
All data processing uses Warp kernels - numpy only at entry/exit points.
"""
import gym
from gym import spaces
import numpy as np
import warp as wp
import os
import mujoco
import mujoco_warp as mjw
from typing import Optional, Tuple, Dict, Any, List

from ..lbm_fluid_env_3d import LBMFluidEnv3D
from ..lbm_core_3d import HomeFlow3D


# ============== Warp Kernels for 3D Fish Multi-goal Environment ==============


@wp.kernel
def compute_fish_obs_3d_multigoal_kernel(
    qfrc_applied: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    qpos: wp.array2d(dtype=wp.float32),  # (nworld, nq)
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),  # (nworld, 3) - current goal for each world
    obs_out: wp.array2d(dtype=wp.float32),  # (nworld, obs_dim)
    nx: float,
    ny: float,
    nz: float,
    n_joints: int,
):
    """
    Compute observation for all worlds in parallel.
    
    Observation layout (for n_joints joints):
    - Forces (6): fx, fy, fz, tau_x, tau_y, tau_z (root body generalized forces)
    - Joint torques (n_joints): joint generalized forces
    - Position (3): x, y, z
    - Quaternion (4): w, x, y, z
    - Velocity (3): vx, vy, vz
    - Angular velocity (3): omega_x, omega_y, omega_z
    - Joint angles (n_joints): joint positions
    - Joint velocities (n_joints): joint velocities
    - LBM position (3): normalized x, y, z
    - Goal position (3): normalized goal x, y, z
    
    Total: 6 + n_joints + 3 + 4 + 3 + 3 + n_joints + n_joints + 3 + 3 = 28 + 3*n_joints
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    idx = 0
    
    # Generalized forces on root body (6): fx, fy, fz, tau_x, tau_y, tau_z
    for i in range(6):
        obs_out[world_idx, idx] = qfrc_applied[world_idx, i]
        idx = idx + 1
    
    # Joint torques (n_joints)
    for i in range(n_joints):
        obs_out[world_idx, idx] = qfrc_applied[world_idx, 6 + i]
        idx = idx + 1
    
    # Position (3): qpos[0:3]
    obs_out[world_idx, idx] = qpos[world_idx, 0]
    obs_out[world_idx, idx + 1] = qpos[world_idx, 1]
    obs_out[world_idx, idx + 2] = qpos[world_idx, 2]
    idx = idx + 3
    
    # Quaternion (4): qpos[3:7]
    obs_out[world_idx, idx] = qpos[world_idx, 3]
    obs_out[world_idx, idx + 1] = qpos[world_idx, 4]
    obs_out[world_idx, idx + 2] = qpos[world_idx, 5]
    obs_out[world_idx, idx + 3] = qpos[world_idx, 6]
    idx = idx + 4
    
    # Velocity (3): qvel[0:3]
    obs_out[world_idx, idx] = qvel[world_idx, 0]
    obs_out[world_idx, idx + 1] = qvel[world_idx, 1]
    obs_out[world_idx, idx + 2] = qvel[world_idx, 2]
    idx = idx + 3
    
    # Angular velocity (3): qvel[3:6]
    obs_out[world_idx, idx] = qvel[world_idx, 3]
    obs_out[world_idx, idx + 1] = qvel[world_idx, 4]
    obs_out[world_idx, idx + 2] = qvel[world_idx, 5]
    idx = idx + 3
    
    # Joint angles (n_joints): qpos[7:7+n_joints]
    for i in range(n_joints):
        obs_out[world_idx, idx] = qpos[world_idx, 7 + i]
        idx = idx + 1
    
    # Joint velocities (n_joints): qvel[6:6+n_joints]
    for i in range(n_joints):
        obs_out[world_idx, idx] = qvel[world_idx, 6 + i]
        idx = idx + 1
    
    # LBM position (normalized) (3)
    head_pos = flow.solid_position[0]
    obs_out[world_idx, idx] = head_pos[0] / nx
    obs_out[world_idx, idx + 1] = head_pos[1] / ny
    obs_out[world_idx, idx + 2] = head_pos[2] / nz
    idx = idx + 3
    
    # Goal position (normalized) (3)
    obs_out[world_idx, idx] = goal_positions[world_idx, 0]
    obs_out[world_idx, idx + 1] = goal_positions[world_idx, 1]
    obs_out[world_idx, idx + 2] = goal_positions[world_idx, 2]


@wp.kernel
def check_boundary_3d_multigoal_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    terminated_out: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    boundary_margin: float,
    n_solids: int,
):
    """
    Check boundary termination condition for all worlds in parallel.
    Only checks if robot hits boundary (not goal reaching).
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    # Check boundary for each solid (center position only)
    for solid_idx in range(n_solids):
        pos = flow.solid_position[solid_idx]
        x = pos[0]
        y = pos[1]
        z = pos[2]
        
        # Check if solid center exceeds boundary
        if (x < boundary_margin or x > nx - boundary_margin or
            y < boundary_margin or y > ny - boundary_margin or
            z < boundary_margin or z > nz - boundary_margin):
            terminated_out[world_idx] = 1
            return
    
    terminated_out[world_idx] = 0


@wp.kernel
def check_goal_reached_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),  # (nworld, 3)
    goal_reached_out: wp.array(dtype=wp.int32),  # (nworld,)
    nx: float,
    ny: float,
    nz: float,
    goal_threshold: float,  # normalized distance threshold
):
    """
    Check if robot reached current goal for all worlds.
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    # Get head position (solid 0)
    head_pos = flow.solid_position[0]
    current_x = head_pos[0] / nx
    current_y = head_pos[1] / ny
    current_z = head_pos[2] / nz
    
    goal_x = goal_positions[world_idx, 0]
    goal_y = goal_positions[world_idx, 1]
    goal_z = goal_positions[world_idx, 2]
    
    dx = current_x - goal_x
    dy = current_y - goal_y
    dz = current_z - goal_z
    dist_sq = dx * dx + dy * dy + dz * dz
    
    if dist_sq < goal_threshold * goal_threshold:
        goal_reached_out[world_idx] = 1
    else:
        goal_reached_out[world_idx] = 0


@wp.kernel
def check_stability_3d_multigoal_kernel(
    qpos: wp.array2d(dtype=wp.float32),  # (nworld, nq)
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    instability_out: wp.array(dtype=wp.int32),  # (nworld,)
    nq: int,
    nv: int,
):
    """
    Check numerical stability for all worlds in parallel.
    Checks for NaN/Inf in qpos and qvel.
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
def compute_goal_reward_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),  # (nworld, 3)
    prev_dist: wp.array(dtype=wp.float32),  # (nworld,)
    rewards_out: wp.array(dtype=wp.float32),  # (nworld,)
    current_dist_out: wp.array(dtype=wp.float32),  # (nworld,)
    nx: float,
    ny: float,
    nz: float,
):
    """
    Compute reward based on distance to goal.
    Reward = 100.0 * (prev_dist - current_dist) (positive when getting closer)
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    head_pos = flow.solid_position[0]
    current_x = head_pos[0] / nx
    current_y = head_pos[1] / ny
    current_z = head_pos[2] / nz
    
    goal_x = goal_positions[world_idx, 0]
    goal_y = goal_positions[world_idx, 1]
    goal_z = goal_positions[world_idx, 2]
    
    dx = current_x - goal_x
    dy = current_y - goal_y
    dz = current_z - goal_z
    current_dist = wp.sqrt(dx * dx + dy * dy + dz * dz)
    
    # Reward for getting closer to goal
    dist_improvement = prev_dist[world_idx] - current_dist
    rewards_out[world_idx] = 100.0 * dist_improvement
    
    current_dist_out[world_idx] = current_dist


@wp.kernel
def compute_smooth_reward_3d_multigoal_kernel(
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    current_actions: wp.array2d(dtype=wp.float32),  # (nworld, action_dim)
    prev_actions: wp.array2d(dtype=wp.float32),  # (nworld, action_dim)
    smooth_rewards_out: wp.array(dtype=wp.float32),  # (nworld,)
    action_smoothness_weight: float,
    phase_reward_weight: float,
    action_dim: int,
):
    """
    Compute smoothness reward for encouraging smooth swimming patterns.
    (Same as fish environment)
    
    Components:
    1. Action smoothness: penalize large changes between consecutive actions
    2. Phase difference reward: encourage traveling wave pattern (rear leads front)
    
    3D Fish joint layout (qvel indices):
    - qvel[0:6]: freejoint (vx, vy, vz, ωx, ωy, ωz)
    - qvel[6:9]: joint1 (pitch, roll, yaw) - head-body joint
    - qvel[9:12]: joint2 (pitch, roll, yaw) - body-tail joint
    
    Main swimming motion is yaw (Z-axis rotation):
    - joint1_yaw velocity: qvel[8]
    - joint2_yaw velocity: qvel[11]
    """
    world_idx = wp.tid()
    
    # ========== 1. Action Smoothness Penalty ==========
    action_change_sq = float(0.0)
    for i in range(action_dim):
        action_diff = current_actions[world_idx, i] - prev_actions[world_idx, i]
        action_change_sq = action_change_sq + action_diff * action_diff
    
    action_smooth_reward = -action_smoothness_weight * action_change_sq
    
    # ========== 2. Phase Difference Reward ==========
    # For traveling wave: rear joint should lead front joint
    # Joint1 yaw velocity (front): qvel[8]
    # Joint2 yaw velocity (rear): qvel[11]
    # Action mapping: action[2] = joint1_yaw, action[5] = joint2_yaw
    joint1_yaw_vel = qvel[world_idx, 8]   # front joint yaw velocity
    joint2_yaw_vel = qvel[world_idx, 11]  # rear joint yaw velocity
    
    phase_indicator = (
        joint1_yaw_vel * current_actions[world_idx, 5]   # front vel * rear action
        - joint2_yaw_vel * current_actions[world_idx, 2]  # rear vel * front action
    )
    phase_reward = phase_reward_weight * wp.abs(phase_indicator)
    
    # ========== Total Smooth Reward ==========
    smooth_rewards_out[world_idx] = action_smooth_reward + phase_reward


@wp.kernel
def add_smooth_rewards_3d_multigoal_kernel(
    rewards: wp.array(dtype=wp.float32),
    smooth_rewards: wp.array(dtype=wp.float32),
):
    """Add smooth rewards to main rewards buffer (in-place on GPU)."""
    world_idx = wp.tid()
    rewards[world_idx] = rewards[world_idx] + smooth_rewards[world_idx]


@wp.kernel
def apply_instability_penalty_3d_multigoal_kernel(
    rewards: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    instability_mask: wp.array(dtype=wp.int32),
    penalty: float,
):
    """Apply instability penalty to rewards and update terminated flags."""
    world_idx = wp.tid()
    if instability_mask[world_idx] == 1:
        rewards[world_idx] = penalty
        terminated[world_idx] = 1


@wp.kernel
def check_anomaly_3d_multigoal_kernel(
    rewards: wp.array(dtype=wp.float32),
    qfrc_applied: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    terminated: wp.array(dtype=wp.int32),
    anomaly_out: wp.array(dtype=wp.int32),
    force_threshold: float,
    penalty: float,
    nv: int,
):
    """
    Check for reward NaN/Inf and abnormally large forces.
    If anomaly detected, set terminated=1 and apply penalty.
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
    
    # Check for abnormally large forces
    for i in range(nv):
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
def reset_prev_dist_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    reset: wp.array(dtype=wp.int32),
    prev_dist: wp.array(dtype=wp.float32),
    nx: float,
    ny: float,
    nz: float,
):
    """Reset previous distance for specific worlds."""
    w = wp.tid()
    if reset[w] != 0:
        head_pos = flows[w].solid_position[0]
        current_x = head_pos[0] / nx
        current_y = head_pos[1] / ny
        current_z = head_pos[2] / nz
        goal_x = goal_positions[w, 0]
        goal_y = goal_positions[w, 1]
        goal_z = goal_positions[w, 2]
        dx = current_x - goal_x
        dy = current_y - goal_y
        dz = current_z - goal_z
        prev_dist[w] = wp.sqrt(dx * dx + dy * dy + dz * dz)


@wp.kernel
def reset_prev_actions_3d_multigoal_kernel(
    reset: wp.array(dtype=wp.int32),
    prev_actions: wp.array2d(dtype=wp.float32),
    action_dim: int,
):
    """Reset previous actions for specific worlds."""
    w = wp.tid()
    if reset[w] != 0:
        for j in range(action_dim):
            prev_actions[w, j] = 0.0


@wp.kernel
def init_prev_dist_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    prev_dist: wp.array(dtype=wp.float32),
    nx: float,
    ny: float,
    nz: float,
):
    """Initialize previous distance for all worlds."""
    w = wp.tid()
    head_pos = flows[w].solid_position[0]
    current_x = head_pos[0] / nx
    current_y = head_pos[1] / ny
    current_z = head_pos[2] / nz
    goal_x = goal_positions[w, 0]
    goal_y = goal_positions[w, 1]
    goal_z = goal_positions[w, 2]
    dx = current_x - goal_x
    dy = current_y - goal_y
    dz = current_z - goal_z
    prev_dist[w] = wp.sqrt(dx * dx + dy * dy + dz * dz)


# ============== 3D Fish Multi-goal LBM Environment Class ==============


class FishLBMEnv3DMultigoal(LBMFluidEnv3D):
    """
    3D Fish swimming environment with LBM fluid simulation.
    Multi-goal version: robot navigates to multiple goal points sequentially in 3D space.
    
    Supports parallel training with nworld environments.
    Uses gym API for compatibility with dreamer_vec_wrapper.
    
    Goals:
    - Multiple goal points in a 3x3x3 grid (excluding center where robot starts)
    - Robot starts from center
    - When reaching a goal, next goal is selected (different from current)
    - Episode lasts max_episode_steps
    """
    
    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        root_link: str = 'head',
        root_position: Optional[Tuple[float, float, float]] = None,
        nx: int = 256,
        ny: int = 256,
        nz: int = 256,
        lbm_scale: float = 0.5,
        nworld: int = 1,
        max_episode_steps: int = 2000,
        per_frame_steps: int = 10,
        device: Optional[str] = None,
        goal_threshold: float = 0.08,  # normalized distance to consider goal reached (larger for 3D)
    ):
        """
        Initialize 3D Fish Multi-goal LBM Environment.
        
        Args:
            mjcf_path: Path to fish MJCF XML file
            root_link: Name of root link (default: 'head')
            root_position: LBM grid position of root link (default: center)
            nx, ny, nz: LBM grid dimensions
            lbm_scale: Scale factor for geometry
            nworld: Number of parallel worlds
            max_episode_steps: Maximum steps per episode
            per_frame_steps: LBM-MuJoCo coupling iterations per environment step
            device: Warp device
            goal_threshold: Distance threshold to consider goal reached
        """
        # Use the fish_3d.xml from the fish folder
        if mjcf_path is None:
            mjcf_path = os.path.join(os.path.dirname(__file__), '..', 'fish', 'fish_3d.xml')
        
        # Default root position: center of the domain
        if root_position is None:
            root_position = (nx / 2, ny / 2, nz / 2)
        
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
            device=device,
        )
        
        # Get number of joints from MuJoCo model
        self.n_joints = self.mjw_model.nu
        
        # Action space: joint controls
        self.action_dim = self.n_joints
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.nworld, self.action_dim),
            dtype=np.float32
        )
        self.action_scale = 1.0
        
        # Observation space: 28 + 3 * n_joints dimensions
        # 6 (forces) + n_joints (joint torques) + 3 (pos) + 4 (quat) + 3 (vel) + 3 (omega)
        # + n_joints (joint angles) + n_joints (joint vels) + 3 (lbm pos) + 3 (goal pos)
        self.obs_dim = 28 + 3 * self.n_joints
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.nworld, self.obs_dim),
            dtype=np.float32
        )
        
        # Goal configuration - 3x3x3 grid (normalized coordinates), excluding center
        # Generate all combinations of [0.25, 0.5, 0.75] for x, y, z
        self.goal_positions_list = []
        for x in [0.25, 0.5, 0.75]:
            for y in [0.25, 0.5, 0.75]:
                for z in [0.25, 0.5, 0.75]:
                    # Exclude center position where robot starts
                    if not (x == 0.5 and y == 0.5 and z == 0.5):
                        self.goal_positions_list.append((x, y, z))
        
        self.num_goals = len(self.goal_positions_list)  # 26 goals
        self.goal_threshold = goal_threshold
        
        # Boundary parameters (LBM coordinate system)
        self.boundary_margin = 5.0  # Boundary safety margin (LBM grid units)
        
        # Current goal index for each world
        self.current_goal_idx = np.zeros(nworld, dtype=np.int32)
        
        # Goal history to avoid selecting same goal
        self.goal_history = [[] for _ in range(nworld)]
        
        # Goals reached counter
        self.goals_reached = np.zeros(nworld, dtype=np.int32)
        
        # Current goal positions (nworld, 3)
        self._goal_positions_wp = wp.zeros((nworld, 3), dtype=wp.float32, device=self.device)
        
        # For distance-based reward calculation
        self._prev_dist_wp = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._current_dist_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        
        # Smoothness reward parameters (same as fish environment)
        self.action_smoothness_weight = 0.5  # Penalize jerky movements
        self.phase_reward_weight = 0.0001  # Reward traveling wave pattern
        
        # Anomaly detection parameters
        self.force_threshold = 1e5
        self.anomaly_penalty = -10.0
        
        # Goal reaching bonus
        self.goal_reached_bonus = 10.0
        
        # Pre-allocate Warp buffers on device
        self._obs_buffer = wp.zeros((nworld, self.obs_dim), dtype=wp.float32, device=self.device)
        self._terminated_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._instability_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._rewards_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._smooth_rewards_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._goal_reached_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._anomaly_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        
        # Action buffers for smoothness reward
        self._current_actions_wp = None
        self._prev_actions_wp = None
    
    def _create_observation_space(self) -> spaces.Space:
        """Create observation space."""
        if not hasattr(self, 'obs_dim'):
            return spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.nworld, 1),
                dtype=np.float32
            )
        return spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.nworld, self.obs_dim),
            dtype=np.float32
        )
    
    def _select_next_goal(self, world_idx: int) -> int:
        """
        Select next goal for a world, avoiding recently visited goals.
        """
        # Get available goals (not in recent history)
        history = self.goal_history[world_idx]
        available = [i for i in range(self.num_goals) if i not in history[-5:]]  # Avoid last 5 goals
        
        if not available:
            # If all goals recently visited, allow any except current
            current = self.current_goal_idx[world_idx]
            available = [i for i in range(self.num_goals) if i != current]
        
        # Randomly select from available
        next_goal = np.random.choice(available)
        return next_goal
    
    def _update_goal_positions_wp(self):
        """Update the Warp array with current goal positions."""
        goal_pos_np = np.zeros((self.nworld, 3), dtype=np.float32)
        for w in range(self.nworld):
            goal_idx = self.current_goal_idx[w]
            goal_pos_np[w, 0] = self.goal_positions_list[goal_idx][0]
            goal_pos_np[w, 1] = self.goal_positions_list[goal_idx][1]
            goal_pos_np[w, 2] = self.goal_positions_list[goal_idx][2]
        wp.copy(self._goal_positions_wp, wp.array(goal_pos_np, dtype=wp.float32, device=self.device))
    
    def _get_obs(self) -> np.ndarray:
        """Get observations for all worlds using kernel."""
        wp.launch(
            compute_fish_obs_3d_multigoal_kernel,
            dim=self.nworld,
            inputs=[
                self.mjw_data.qfrc_applied,
                self.mjw_data.qpos,
                self.mjw_data.qvel,
                self.lbm_solver.flows_wp,
                self._goal_positions_wp,
                self._obs_buffer,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.n_joints,
            ],
            device=self.device,
        )
        return self._obs_buffer.numpy()
    
    def _check_goals_reached(self):
        """Check if any world reached its goal and update accordingly."""
        wp.launch(
            check_goal_reached_3d_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._goal_positions_wp,
                self._goal_reached_buffer,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.goal_threshold,
            ],
            device=self.device,
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
            reset_mask_wp = wp.array(goal_reached.astype(np.int32), dtype=wp.int32, device=self.device)
            wp.launch(
                reset_prev_dist_3d_kernel,
                dim=self.nworld,
                inputs=[
                    self.lbm_solver.flows_wp,
                    self._goal_positions_wp,
                    reset_mask_wp,
                    self._prev_dist_wp,
                    float(self.nx),
                    float(self.ny),
                    float(self.nz),
                ],
                device=self.device,
            )
    
    def _compute_reward(self, instability_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute reward function using Warp kernels.
        
        Reward components:
        1. Distance-based reward (main objective - getting closer to goal)
        2. Goal reached bonus
        3. Action smoothness penalty
        4. Rotation penalty
        """
        # Distance-based reward
        wp.launch(
            compute_goal_reward_3d_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._goal_positions_wp,
                self._prev_dist_wp,
                self._rewards_buffer,
                self._current_dist_buffer,
                float(self.nx),
                float(self.ny),
                float(self.nz),
            ],
            device=self.device,
        )
        
        # Update previous distance
        wp.copy(self._prev_dist_wp, self._current_dist_buffer)
        
        # Add goal reached bonus
        goal_reached = self._goal_reached_buffer.numpy()
        rewards_np = self._rewards_buffer.numpy()
        rewards_np[goal_reached.astype(bool)] += self.goal_reached_bonus
        wp.copy(self._rewards_buffer, wp.array(rewards_np, dtype=wp.float32, device=self.device))
        
        # Smoothness reward
        if self._prev_actions_wp is not None and self._current_actions_wp is not None:
            wp.launch(
                compute_smooth_reward_3d_multigoal_kernel,
                dim=self.nworld,
                inputs=[
                    self.mjw_data.qvel,
                    self._current_actions_wp,
                    self._prev_actions_wp,
                    self._smooth_rewards_buffer,
                    self.action_smoothness_weight,
                    self.phase_reward_weight,
                    self.action_dim,
                ],
                device=self.device,
            )
            
            # Add smooth rewards
            wp.launch(
                add_smooth_rewards_3d_multigoal_kernel,
                dim=self.nworld,
                inputs=[
                    self._rewards_buffer,
                    self._smooth_rewards_buffer,
                ],
                device=self.device,
            )
        
        # Apply instability penalty
        if instability_mask is not None:
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32, device=self.device)
            wp.launch(
                apply_instability_penalty_3d_multigoal_kernel,
                dim=self.nworld,
                inputs=[
                    self._rewards_buffer,
                    self._terminated_buffer,
                    instability_wp,
                    self.anomaly_penalty,
                ],
                device=self.device,
            )
        
        # Check for anomalies
        wp.launch(
            check_anomaly_3d_multigoal_kernel,
            dim=self.nworld,
            inputs=[
                self._rewards_buffer,
                self.mjw_data.qfrc_applied,
                self._terminated_buffer,
                self._anomaly_buffer,
                self.force_threshold,
                self.anomaly_penalty,
                self.mjw_model.nv,
            ],
            device=self.device,
        )
        
        return self._rewards_buffer.numpy()
    
    def _is_terminated(self, instability_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Check termination condition for all worlds.
        
        Termination conditions:
        - Any part of the robot reaches simulation boundary
        - Numerical instability detected
        """
        self._terminated_buffer.zero_()
        
        wp.launch(
            check_boundary_3d_multigoal_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._terminated_buffer,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.boundary_margin,
                self.solid_num,
            ],
            device=self.device,
        )
        
        # Apply instability termination if provided
        if instability_mask is not None:
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32, device=self.device)
            wp.launch(
                apply_instability_penalty_3d_multigoal_kernel,
                dim=self.nworld,
                inputs=[
                    self._rewards_buffer,
                    self._terminated_buffer,
                    instability_wp,
                    0.0,
                ],
                device=self.device,
            )
        
        return self._terminated_buffer.numpy().astype(bool)
    
    def _check_numerical_stability(self) -> np.ndarray:
        """Check numerical stability for all worlds using Warp kernel."""
        self._instability_buffer.zero_()
        
        wp.launch(
            check_stability_3d_multigoal_kernel,
            dim=self.nworld,
            inputs=[
                self.mjw_data.qpos,
                self.mjw_data.qvel,
                self._instability_buffer,
                self.mjw_model.nq,
                self.mjw_model.nv,
            ],
            device=self.device,
        )
        
        return self._instability_buffer.numpy().astype(bool)
    
    def step(self, action: np.ndarray):
        """
        Execute one environment step.
        """
        # Save current action
        self.current_actions = np.array(action).copy()
        self._current_actions_wp = wp.array(self.current_actions, dtype=wp.float32, device=self.device)
        
        # Handle action shape - expect (nworld, action_dim)
        action = np.clip(action, self.action_space.low, self.action_space.high) * self.action_scale
        wp.copy(self.mjw_data.ctrl, wp.array(action, dtype=wp.float32, device=self.device))
        
        # Execute physics simulation step
        self._simulation_step()
        
        # Update step counts
        self.step_counts += 1
        
        # Check if goals are reached
        self._check_goals_reached()
        
        # Get new observation
        observation = self._get_obs()
        
        # Check numerical stability
        instability_mask = self._check_numerical_stability()
        
        # Check for NaN/Inf in observations
        obs_nan_mask = np.any(np.isnan(observation) | np.isinf(observation), axis=1)
        if np.any(obs_nan_mask):
            instability_mask = instability_mask | obs_nan_mask
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check basic termination condition
        self._is_terminated(instability_mask)
        
        # Compute reward
        reward = self._compute_reward(instability_mask)
        
        # Get final terminated state
        terminated = self._terminated_buffer.numpy().astype(bool)
        
        # Apply termination penalty
        reward[terminated] -= 1.0
        
        # Final safety check
        if np.any(np.isnan(reward)) or np.any(np.isinf(reward)):
            bad_mask = np.isnan(reward) | np.isinf(reward)
            reward[bad_mask] = self.anomaly_penalty - 1.0
            terminated[bad_mask] = True
        
        truncated = np.array(self.step_counts >= self.max_episode_steps)
        done = terminated | truncated
        
        info = self._get_info()
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["goals_reached"] = self.goals_reached.copy()
        
        # Update previous actions
        self._prev_actions_wp = wp.array(self.current_actions, dtype=wp.float32, device=self.device)
        
        return observation, reward, done, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> np.ndarray:
        """Reset all worlds."""
        if seed is not None:
            np.random.seed(seed)
        
        obs = super().reset(seed=seed, options=options)
        
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
        wp.launch(
            init_prev_dist_3d_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._goal_positions_wp,
                self._prev_dist_wp,
                float(self.nx),
                float(self.ny),
                float(self.nz),
            ],
            device=self.device,
        )
        
        # Initialize previous actions
        self._prev_actions_wp = wp.zeros((self.nworld, self.action_dim), dtype=wp.float32, device=self.device)
        
        # Get observations using kernel
        obs = self._get_obs()
        
        return obs
    
    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        """Reset specific worlds indicated by reset_mask."""
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
        reset_mask_wp = wp.array(reset_mask.astype(np.int32), dtype=wp.int32, device=self.device)
        wp.launch(
            reset_prev_dist_3d_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._goal_positions_wp,
                reset_mask_wp,
                self._prev_dist_wp,
                float(self.nx),
                float(self.ny),
                float(self.nz),
            ],
            device=self.device,
        )
        
        # Reset previous actions for reset worlds
        if self._prev_actions_wp is None:
            self._prev_actions_wp = wp.zeros((self.nworld, self.action_dim), dtype=wp.float32, device=self.device)
        
        wp.launch(
            reset_prev_actions_3d_multigoal_kernel,
            dim=self.nworld,
            inputs=[reset_mask_wp, self._prev_actions_wp, self.action_dim],
            device=self.device,
        )
        
        return self._get_obs()
    
    def get_current_goal(self, world_idx: int = 0) -> Tuple[float, float, float]:
        """Get current goal position for a world (for visualization)."""
        goal_idx = self.current_goal_idx[world_idx]
        return self.goal_positions_list[goal_idx]
