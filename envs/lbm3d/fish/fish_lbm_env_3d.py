"""
3D Fish LBM Environment for MuJoCo Warp with nworld support

Based on 2D fish environment, adapted for 3D LBM simulation.
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


# ============== Warp Kernels for 3D Fish Environment ==============


@wp.kernel
def compute_fish_obs_3d_kernel(
    qfrc_applied: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    qpos: wp.array2d(dtype=wp.float32),  # (nworld, nq)
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    flows: wp.array(dtype=HomeFlow3D),
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
    
    Total: 6 + n_joints + 3 + 4 + 3 + 3 + n_joints + n_joints + 3 = 25 + 3*n_joints
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


@wp.kernel
def check_termination_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    terminated_out: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    boundary_margin: float,
    target_y: float,  # normalized target y (swim direction)
    n_solids: int,
):
    """
    Check termination condition for all worlds in parallel.
    
    Termination conditions:
    - Reached target point (y > target_y * ny, swimming in +Y direction)
    - Any solid center exceeds boundary margin
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    # Get head position (solid 0)
    head_pos = flow.solid_position[0]
    
    # Check if reached target (swimming in +Y direction)
    if head_pos[1] > target_y * ny:
        terminated_out[world_idx] = 1
        return
    
    # Check boundary for each solid (center position only, mesh has hierarchical BVH)
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
def check_stability_3d_kernel(
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
def compute_reward_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    prev_positions_y: wp.array(dtype=wp.float32),
    rewards_out: wp.array(dtype=wp.float32),
    current_positions_y_out: wp.array(dtype=wp.float32),
    ny: float,
):
    """
    Compute forward movement reward for all worlds.
    Reward = 100.0 * dy / ny (positive when moving in +Y direction)
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    head_pos = flow.solid_position[0]
    current_y = head_pos[1]
    
    # Forward movement reward (swimming in +Y direction)
    dy = current_y - prev_positions_y[world_idx]  # Positive when moving in +Y
    rewards_out[world_idx] = 10000.0 * dy / ny
    
    current_positions_y_out[world_idx] = current_y




@wp.kernel
def compute_smooth_reward_3d_kernel(
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
    # Reward when joint velocities suggest wave propagation
    # 
    # Joint1 yaw velocity (front): qvel[8]
    # Joint2 yaw velocity (rear): qvel[11]
    # Action mapping: action[2] = joint1_yaw, action[5] = joint2_yaw
    #
    # Reward phase difference: when one joint is moving fast and the other is reversing
    # This encourages traveling wave pattern
    joint1_yaw_vel = qvel[world_idx, 8]   # front joint yaw velocity
    joint2_yaw_vel = qvel[world_idx, 11]  # rear joint yaw velocity
    
    # Cross product of velocities with actions indicates phase relationship
    # action indices: 0=pitch1, 1=roll1, 2=yaw1, 3=pitch2, 4=roll2, 5=yaw2
    phase_indicator = (
        joint1_yaw_vel * current_actions[world_idx, 5]   # front vel * rear action
        - joint2_yaw_vel * current_actions[world_idx, 2]  # rear vel * front action
    )
    phase_reward = phase_reward_weight * wp.abs(phase_indicator)
    
    # ========== Total Smooth Reward ==========
    smooth_rewards_out[world_idx] = action_smooth_reward + phase_reward


@wp.kernel
def add_smooth_rewards_3d_kernel(
    rewards: wp.array(dtype=wp.float32),
    smooth_rewards: wp.array(dtype=wp.float32),
):
    """Add smooth rewards to main rewards buffer (in-place on GPU)."""
    world_idx = wp.tid()
    rewards[world_idx] = rewards[world_idx] + smooth_rewards[world_idx]


@wp.kernel
def apply_instability_penalty_3d_kernel(
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
def check_reward_and_force_anomaly_3d_kernel(
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
def reset_prev_y_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    reset: wp.array(dtype=wp.int32),
    prev_y: wp.array(dtype=wp.float32),
):
    """Reset previous Y position for specific worlds."""
    w = wp.tid()
    if reset[w] != 0:
        prev_y[w] = flows[w].solid_position[0][1]


@wp.kernel
def reset_prev_actions_3d_kernel(
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
def init_prev_y_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    prev_y: wp.array(dtype=wp.float32),
):
    """Initialize previous Y position for all worlds."""
    w = wp.tid()
    prev_y[w] = flows[w].solid_position[0][1]


# ============== 3D Fish LBM Environment Class ==============


class FishLBMEnv3D(LBMFluidEnv3D):
    """
    3D Fish swimming environment with LBM fluid simulation.
    
    Supports parallel training with nworld environments.
    Uses gym API for compatibility with dreamer_vec_wrapper.
    
    Features (adapted from 2D version):
    - Forward movement reward (main objective)
    - Action smoothness penalty (encourage smooth swimming)
    - Rotation penalty (discourage spinning)
    - Numerical stability checking
    - Force anomaly detection
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
    ):
        """
        Initialize 3D Fish LBM Environment.
        
        Args:
            mjcf_path: Path to fish MJCF XML file
            root_link: Name of root link (default: 'head')
            root_position: LBM grid position of root link (default: center, slightly forward)
            nx, ny, nz: LBM grid dimensions
            lbm_scale: Scale factor for geometry
            nworld: Number of parallel worlds
            max_episode_steps: Maximum steps per episode
            per_frame_steps: LBM-MuJoCo coupling iterations per environment step
            device: Warp device
        """
        if mjcf_path is None:
            mjcf_path = os.path.join(os.path.dirname(__file__), 'fish_3d.xml')
        
        # Default root position: start at low Y, swim toward high Y
        if root_position is None:
            root_position = (nx / 2, ny * 0.3, nz / 2)
        
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
        
        # Observation space: 25 + 3 * n_joints dimensions
        # 6 (forces) + n_joints (joint torques) + 3 (pos) + 4 (quat) + 3 (vel) + 3 (omega)
        # + n_joints (joint angles) + n_joints (joint vels) + 3 (lbm pos)
        self.obs_dim = 25 + 3 * self.n_joints
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.nworld, self.obs_dim),
            dtype=np.float32
        )
        
        # Target position for reward calculation (swim in +Y direction)
        self.target_y = 0.9  # normalized target y (terminate when y > target_y * ny)
        
        # Boundary parameters (LBM coordinate system)
        self.boundary_margin = 5.0  # Boundary safety margin (LBM grid units)
        
        # ========== Smoothness Reward Parameters ==========
        self.action_smoothness_weight = 0.5  # Penalize jerky movements
        self.phase_reward_weight = 0.0001  # Reward traveling wave pattern
        self.forward_reward_weight = 10000.0  # Weight for forward movement
        
        # Anomaly detection parameters
        self.force_threshold = 1e5  # Maximum allowed force magnitude
        self.anomaly_penalty = -10.0  # Penalty when anomaly detected
        
        # Pre-allocate Warp buffers on device
        self._obs_buffer = wp.zeros((nworld, self.obs_dim), dtype=wp.float32, device=self.device)
        self._terminated_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._instability_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._rewards_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._smooth_rewards_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._current_y_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._prev_positions_y_wp = wp.zeros(nworld, dtype=wp.float32, device=self.device)
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
    
    def _get_obs(self) -> np.ndarray:
        """Get observations for all worlds using kernel."""
        wp.launch(
            compute_fish_obs_3d_kernel,
            dim=self.nworld,
            inputs=[
                self.mjw_data.qfrc_applied,
                self.mjw_data.qpos,
                self.mjw_data.qvel,
                self.lbm_solver.flows_wp,
                self._obs_buffer,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.n_joints,
            ],
            device=self.device,
        )
        return self._obs_buffer.numpy()
    
    def _compute_reward(self, instability_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute reward function using Warp kernels.
        
        Reward components:
        1. Forward movement reward (main objective)
        2. Action smoothness penalty (encourage smooth swimming)
        3. Rotation penalty (discourage spinning)
        
        Also checks for reward NaN/Inf and abnormally large forces.
        """
        # ========== 1. Forward Movement Reward ==========
        wp.launch(
            compute_reward_3d_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._prev_positions_y_wp,
                self._rewards_buffer,
                self._current_y_buffer,
                float(self.ny),
            ],
            device=self.device,
        )
        
        # Update previous y positions
        wp.copy(self._prev_positions_y_wp, self._current_y_buffer)
        
        # ========== 2. Smoothness Reward ==========
        if self._prev_actions_wp is not None and self._current_actions_wp is not None:
            wp.launch(
                compute_smooth_reward_3d_kernel,
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
            
            # Add smooth rewards to main rewards (directly on GPU)
            wp.launch(
                add_smooth_rewards_3d_kernel,
                dim=self.nworld,
                inputs=[
                    self._rewards_buffer,
                    self._smooth_rewards_buffer,
                ],
                device=self.device,
            )
        
        # ========== 3. Apply Instability Penalty ==========
        if instability_mask is not None:
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32, device=self.device)
            wp.launch(
                apply_instability_penalty_3d_kernel,
                dim=self.nworld,
                inputs=[
                    self._rewards_buffer,
                    self._terminated_buffer,
                    instability_wp,
                    self.anomaly_penalty,
                ],
                device=self.device,
            )
        
        # ========== 4. Check for Anomalies ==========
        wp.launch(
            check_reward_and_force_anomaly_3d_kernel,
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
        Check termination condition for all worlds using Warp kernel.
        
        Termination conditions:
        - Any part of the robot reaches simulation boundary
        - Reached target position
        - Numerical instability detected
        """
        self._terminated_buffer.zero_()
        
        wp.launch(
            check_termination_3d_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._terminated_buffer,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.boundary_margin,
                self.target_y,
                self.solid_num,
            ],
            device=self.device,
        )
        
        # Apply instability termination if provided
        if instability_mask is not None:
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32, device=self.device)
            wp.launch(
                apply_instability_penalty_3d_kernel,
                dim=self.nworld,
                inputs=[
                    self._rewards_buffer,
                    self._terminated_buffer,
                    instability_wp,
                    0.0,  # penalty not used for termination
                ],
                device=self.device,
            )
        
        return self._terminated_buffer.numpy().astype(bool)
    
    def _check_numerical_stability(self) -> np.ndarray:
        """Check numerical stability for all worlds using Warp kernel."""
        self._instability_buffer.zero_()
        
        wp.launch(
            check_stability_3d_kernel,
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
        Override step to properly handle reward/force anomaly detection.
        
        The order is important:
        1. Save current action for smoothness reward
        2. Execute simulation
        3. Get observation
        4. Check numerical stability
        5. Check basic termination
        6. Compute reward (which also detects anomalies)
        7. Return final state
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
        
        # Get new observation
        observation = self._get_obs()
        
        # Check numerical stability
        instability_mask = self._check_numerical_stability()
        
        # Check for NaN/Inf in observations
        obs_nan_mask = np.any(np.isnan(observation) | np.isinf(observation), axis=1)
        if np.any(obs_nan_mask):
            instability_mask = instability_mask | obs_nan_mask
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"[WARNING] NaN/Inf observation detected in {np.sum(obs_nan_mask)} worlds")
        
        # Check basic termination condition
        self._is_terminated(instability_mask)
        
        # Compute reward (also checks for anomalies and updates terminated buffer)
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
            print(f"[WARNING] NaN/Inf reward detected in {np.sum(bad_mask)} worlds")
        
        truncated = np.array(self.step_counts >= self.max_episode_steps)
        done = terminated | truncated
        
        info = self._get_info()
        info["terminated"] = terminated
        info["truncated"] = truncated
        
        # Update previous actions
        self._prev_actions_wp = wp.array(self.current_actions, dtype=wp.float32, device=self.device)
        
        return observation, reward, done, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> np.ndarray:
        """Reset all worlds."""
        obs = super().reset(seed=seed, options=options)
        
        # Initialize previous y positions using kernel
        wp.launch(
            init_prev_y_kernel,
            dim=self.nworld,
            inputs=[self.lbm_solver.flows_wp, self._prev_positions_y_wp],
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
        
        if np.any(reset_mask):
            reset_mask_wp = wp.array(reset_mask.astype(np.int32), dtype=wp.int32, device=self.device)
            
            # Reset previous y positions for reset worlds
            wp.launch(
                reset_prev_y_3d_kernel,
                dim=self.nworld,
                inputs=[self.lbm_solver.flows_wp, reset_mask_wp, self._prev_positions_y_wp],
                device=self.device,
            )
            
            # Reset previous actions for reset worlds
            if self._prev_actions_wp is None:
                self._prev_actions_wp = wp.zeros((self.nworld, self.action_dim), dtype=wp.float32, device=self.device)
            
            wp.launch(
                reset_prev_actions_3d_kernel,
                dim=self.nworld,
                inputs=[reset_mask_wp, self._prev_actions_wp, self.action_dim],
                device=self.device,
            )
        
        return self._get_obs()
