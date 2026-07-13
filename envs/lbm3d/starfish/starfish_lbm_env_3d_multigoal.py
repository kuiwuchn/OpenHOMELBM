"""
3D Starfish LBM Environment for MuJoCo Warp with nworld support
Multi-goal version: robot starts from center, navigates to goal points sequentially in XY plane

Yaw-only planar swimming with thin paddle legs.
Based on successful eel3d implementation pattern.
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
from ..lbm_fluid_env_3d_func import (
    extract_body_states_3d,
    convert_and_update_solid_batch_3d,
    extract_forces_torques_physical_3d,
    fill_xfrc_3d_kernel,
)


# ============== Warp Kernels for 3D Starfish Multi-goal Environment ==============


@wp.kernel
def clamp_forces_3d_kernel(
    forces: wp.array3d(dtype=wp.float32),   # (nworld, n_bodies, 3)
    torques: wp.array3d(dtype=wp.float32),  # (nworld, n_bodies, 3)
    max_force: float,
    max_torque: float,
):
    """Clamp per-body forces and torques to prevent numerical divergence."""
    world_idx, body_idx = wp.tid()

    for c in range(3):
        f = forces[world_idx, body_idx, c]
        if f > max_force:
            f = max_force
        if f < -max_force:
            f = -max_force
        forces[world_idx, body_idx, c] = f

        t = torques[world_idx, body_idx, c]
        if t > max_torque:
            t = max_torque
        if t < -max_torque:
            t = -max_torque
        torques[world_idx, body_idx, c] = t


@wp.kernel
def compute_starfish_obs_3d_multigoal_kernel(
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
    
    Observation layout for 3D Starfish (4 yaw joints):
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
    
    Total: 6 + n_joints + 3 + 4 + 3 + 3 + n_joints + n_joints + 3 + 3 = 25 + 3*n_joints
    For 4 joints: 25 + 12 = 37 dims
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
    center_pos = flow.solid_position[0]  # center body is solid 0
    obs_out[world_idx, idx] = center_pos[0] / nx
    obs_out[world_idx, idx + 1] = center_pos[1] / ny
    obs_out[world_idx, idx + 2] = center_pos[2] / nz
    idx = idx + 3
    
    # Goal position (normalized) (3)
    obs_out[world_idx, idx] = goal_positions[world_idx, 0]
    obs_out[world_idx, idx + 1] = goal_positions[world_idx, 1]
    obs_out[world_idx, idx + 2] = goal_positions[world_idx, 2]


@wp.kernel
def check_boundary_3d_starfish_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    terminated_out: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    boundary_margin: float,
    n_solids: int,
):
    """Check boundary termination condition for all worlds in parallel."""
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    for solid_idx in range(n_solids):
        pos = flow.solid_position[solid_idx]
        x = pos[0]
        y = pos[1]
        z = pos[2]
        
        if (x < boundary_margin or x > nx - boundary_margin or
            y < boundary_margin or y > ny - boundary_margin or
            z < boundary_margin or z > nz - boundary_margin):
            terminated_out[world_idx] = 1
            return
    
    terminated_out[world_idx] = 0


@wp.kernel
def check_goal_reached_3d_starfish_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    goal_reached_out: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    goal_threshold: float,
):
    """Check if robot reached current goal for all worlds."""
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    center_pos = flow.solid_position[0]
    current_x = center_pos[0] / nx
    current_y = center_pos[1] / ny
    current_z = center_pos[2] / nz
    
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
def check_stability_3d_starfish_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    instability_out: wp.array(dtype=wp.int32),
    nq: int,
    nv: int,
):
    """Check numerical stability for all worlds in parallel."""
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
def compute_goal_reward_3d_starfish_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    prev_dist: wp.array(dtype=wp.float32),
    rewards_out: wp.array(dtype=wp.float32),
    current_dist_out: wp.array(dtype=wp.float32),
    nx: float,
    ny: float,
    nz: float,
):
    """Compute reward based on distance to goal."""
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    center_pos = flow.solid_position[0]
    current_x = center_pos[0] / nx
    current_y = center_pos[1] / ny
    current_z = center_pos[2] / nz
    
    goal_x = goal_positions[world_idx, 0]
    goal_y = goal_positions[world_idx, 1]
    goal_z = goal_positions[world_idx, 2]
    
    dx = current_x - goal_x
    dy = current_y - goal_y
    dz = current_z - goal_z
    current_dist = wp.sqrt(dx * dx + dy * dy + dz * dz)
    
    dist_improvement = prev_dist[world_idx] - current_dist
    rewards_out[world_idx] = 100.0 * dist_improvement
    current_dist_out[world_idx] = current_dist


@wp.kernel
def compute_smooth_reward_3d_starfish_kernel(
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    current_actions: wp.array2d(dtype=wp.float32),  # (nworld, n_actuators)
    prev_actions: wp.array2d(dtype=wp.float32),  # (nworld, n_actuators)
    smooth_rewards_out: wp.array(dtype=wp.float32),
    action_smoothness_weight: float,
    rotation_penalty_weight: float,
    wave_reward_weight: float,
    n_actuators: int,
):
    """
    Compute smoothness reward for 3D starfish with paddle wave.
    
    Components:
    1. Action smoothness: penalize large changes between consecutive actions
    2. Rotation penalty: penalize excessive roll (keep planar)
    3. Paddle wave reward: encourage opposing legs to have phase difference
    
    4 actuators (yaw only): leg_0(+Y), leg_1(+X), leg_2(-Y), leg_3(-X)
    Opposing pairs: (leg_0, leg_2) and (leg_1, leg_3)
    
    qvel layout:
    - qvel[0:3]: linear velocity (vx, vy, vz)
    - qvel[3:6]: angular velocity (ωx, ωy, ωz)
    - qvel[6:10]: joint velocities (4 yaw joints)
    """
    world_idx = wp.tid()
    
    # ========== 1. Action Smoothness Penalty ==========
    action_change_sq = float(0.0)
    for i in range(n_actuators):
        action_diff = current_actions[world_idx, i] - prev_actions[world_idx, i]
        action_change_sq = action_change_sq + action_diff * action_diff
    
    action_smooth_reward = -action_smoothness_weight * action_change_sq
    
    # ========== 2. Roll Penalty (keep planar, penalize roll only) ==========
    omega_x = qvel[world_idx, 3]  # roll angular velocity
    rotation_penalty = -rotation_penalty_weight * (omega_x * omega_x)
    
    # ========== 3. Paddle Wave Reward ==========
    # Encourage phase difference between opposing leg pairs
    # Pair 1: leg_0 (+Y) vs leg_2 (-Y) → indices 0 and 2
    # Pair 2: leg_1 (+X) vs leg_3 (-X) → indices 1 and 3
    # Phase diff via cross product of (velocity, action)
    wave_reward = float(0.0)
    
    # Pair 1: leg_0 (actuator 0, qvel 6) vs leg_2 (actuator 2, qvel 8)
    phase_diff_02 = qvel[world_idx, 6] * current_actions[world_idx, 2] - qvel[world_idx, 8] * current_actions[world_idx, 0]
    wave_reward = wave_reward + wp.abs(phase_diff_02)
    
    # Pair 2: leg_1 (actuator 1, qvel 7) vs leg_3 (actuator 3, qvel 9)
    phase_diff_13 = qvel[world_idx, 7] * current_actions[world_idx, 3] - qvel[world_idx, 9] * current_actions[world_idx, 1]
    wave_reward = wave_reward + wp.abs(phase_diff_13)
    
    wave_reward = wave_reward_weight * wave_reward
    
    # ========== Total Smooth Reward ==========
    smooth_rewards_out[world_idx] = action_smooth_reward + rotation_penalty + wave_reward


@wp.kernel
def add_smooth_rewards_3d_starfish_kernel(
    rewards: wp.array(dtype=wp.float32),
    smooth_rewards: wp.array(dtype=wp.float32),
):
    """Add smooth rewards to main rewards buffer (in-place on GPU)."""
    world_idx = wp.tid()
    rewards[world_idx] = rewards[world_idx] + smooth_rewards[world_idx]


@wp.kernel
def apply_instability_penalty_3d_starfish_kernel(
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
def check_anomaly_3d_starfish_kernel(
    rewards: wp.array(dtype=wp.float32),
    qfrc_applied: wp.array2d(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    anomaly_out: wp.array(dtype=wp.int32),
    force_threshold: float,
    penalty: float,
    nv: int,
):
    """Check for reward NaN/Inf and abnormally large forces."""
    world_idx = wp.tid()
    anomaly_out[world_idx] = 0
    
    reward_val = rewards[world_idx]
    
    if wp.isnan(reward_val) or wp.isinf(reward_val):
        anomaly_out[world_idx] = 1
        rewards[world_idx] = penalty
        terminated[world_idx] = 1
        return
    
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
def reset_prev_dist_3d_starfish_kernel(
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
        center_pos = flows[w].solid_position[0]
        current_x = center_pos[0] / nx
        current_y = center_pos[1] / ny
        current_z = center_pos[2] / nz
        goal_x = goal_positions[w, 0]
        goal_y = goal_positions[w, 1]
        goal_z = goal_positions[w, 2]
        dx = current_x - goal_x
        dy = current_y - goal_y
        dz = current_z - goal_z
        prev_dist[w] = wp.sqrt(dx * dx + dy * dy + dz * dz)


@wp.kernel
def reset_prev_actions_3d_starfish_kernel(
    reset: wp.array(dtype=wp.int32),
    prev_actions: wp.array2d(dtype=wp.float32),
    n_actuators: int,
):
    """Reset previous actions for specific worlds."""
    w = wp.tid()
    if reset[w] != 0:
        for j in range(n_actuators):
            prev_actions[w, j] = 0.0


@wp.kernel
def init_prev_dist_3d_starfish_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    prev_dist: wp.array(dtype=wp.float32),
    nx: float,
    ny: float,
    nz: float,
):
    """Initialize previous distance for all worlds."""
    w = wp.tid()
    center_pos = flows[w].solid_position[0]
    current_x = center_pos[0] / nx
    current_y = center_pos[1] / ny
    current_z = center_pos[2] / nz
    goal_x = goal_positions[w, 0]
    goal_y = goal_positions[w, 1]
    goal_z = goal_positions[w, 2]
    dx = current_x - goal_x
    dy = current_y - goal_y
    dz = current_z - goal_z
    prev_dist[w] = wp.sqrt(dx * dx + dy * dy + dz * dz)


# ============== 3D Starfish Multi-goal LBM Environment Class ==============


class Starfish3DLBMEnvMultigoal(LBMFluidEnv3D):
    """
    3D Starfish swimming environment with LBM fluid simulation.
    Planar swimmer with yaw-only control (no pitch).
    
    Structure:
    - Center body (flat cylinder)
    - 4 thin paddle legs arranged in cross pattern (+Y, +X, -Y, -X)
    - Each leg has 1 yaw joint (horizontal plane rotation)
    - Total: 4 actuated yaw joints
    
    Control modes:
    - 'direct': 4-dim action → 4 yaw joint angles
    - 'multi_sine': (4*K*2)-dim action → multi-frequency sine wave parameters
    
    Movement: asymmetric paddle strokes in XY plane, z fixed.
    """
    
    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        root_link: str = 'center',
        root_position: Optional[Tuple[float, float, float]] = None,
        nx: int = 100,
        ny: int = 100,
        nz: int = 100,
        lbm_scale: float = 0.5,
        nworld: int = 1,
        max_episode_steps: int = 2000,
        per_frame_steps: int = 10,
        fluid_density: float = 1000.0,
        device: Optional[str] = None,
        goal_threshold: float = 0.08,
        single_goal_mode: bool = True,
        goal_position: Optional[List[float]] = None,
        control_mode: str = 'direct',
        K: int = 3,
        B_bar: float = 1.0,
    ):
        # Store init params before super().__init__
        self._init_single_goal_mode = single_goal_mode
        self._init_goal_position = goal_position if goal_position is not None else [0.5, 0.75, 0.5]
        self.control_mode = control_mode
        self.K = K
        self.B_bar = B_bar
        
        if mjcf_path is None:
            mjcf_path = os.path.join(os.path.dirname(__file__), 'starfish_3d.xml')
        
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
            fluid_density=fluid_density,
            device=device,
        )
        
        # Force clamp: prevent LBM forces from causing NaN
        # Starfish has small mass bodies; large transient forces can cause divergence
        self.max_force_per_body = 2.0   # N (clamp individual body forces)
        self.max_torque_per_body = 0.5  # N·m
        
        # Number of actuators in XML (4 yaw joints)
        self.n_actuators = self.mjw_model.nu  # Should be 4
        self.n_yaw_joints = self.n_actuators  # All actuators are yaw
        self.n_joints = self.n_actuators  # For obs kernel compatibility
        
        # Yaw angle limit (50° = 0.873 rad)
        self.yaw_max = 0.873
        
        # ============== Control Mode Setup ==============
        if self.control_mode == 'multi_sine':
            self.action_dim = self.n_yaw_joints * self.K * 2  # 4 * K * 2
            self.B_freqs = np.array([(j + 1) * self.B_bar / self.K for j in range(self.K)])
            self.multi_sine_t = np.zeros(nworld, dtype=np.float32)
            self.multi_sine_dt = 1.0 / 30.0
        else:
            self.action_dim = self.n_yaw_joints  # 4
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.nworld, self.action_dim),
            dtype=np.float32
        )
        
        # Observation space: 25 + 3 * n_joints
        # 6 (forces) + n_joints (torques) + 3 (pos) + 4 (quat) + 3 (vel) + 3 (omega)
        # + n_joints (angles) + n_joints (vels) + 3 (lbm pos) + 3 (goal pos)
        self.obs_dim = 25 + 3 * self.n_joints  # 25 + 12 = 37 for 4 joints
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.nworld, self.obs_dim),
            dtype=np.float32
        )
        
        # Goal configuration
        self.single_goal_mode = self._init_single_goal_mode
        
        if self.single_goal_mode:
            gp = self._init_goal_position
            self.goal_positions_list = [(gp[0], gp[1], gp[2])]
        else:
            # Yaw-only control: fix z=0.5
            self.goal_positions_list = []
            for x in [0.25, 0.5, 0.75]:
                for y in [0.25, 0.5, 0.75]:
                    if not (x == 0.5 and y == 0.5):
                        self.goal_positions_list.append((x, y, 0.5))
        
        self.num_goals = len(self.goal_positions_list)
        self.goal_threshold = goal_threshold
        
        self.boundary_margin = 5.0
        
        self.current_goal_idx = np.zeros(nworld, dtype=np.int32)
        self.goal_history = [[] for _ in range(nworld)]
        self.goals_reached = np.zeros(nworld, dtype=np.int32)
        
        self._goal_positions_wp = wp.zeros((nworld, 3), dtype=wp.float32, device=self.device)
        self._prev_dist_wp = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._current_dist_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        
        # Reward parameters (following eel pattern)
        self.action_smoothness_weight = 0.3
        self.rotation_penalty_weight = 0.1
        self.wave_reward_weight = 0.005
        self.force_threshold = 1e5
        self.anomaly_penalty = -10.0
        self.goal_reached_bonus = 10.0
        
        # Pre-allocate Warp buffers
        self._obs_buffer = wp.zeros((nworld, self.obs_dim), dtype=wp.float32, device=self.device)
        self._terminated_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._instability_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._rewards_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._smooth_rewards_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._goal_reached_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._anomaly_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        
        # Action buffers (always n_actuators dim for kernel)
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
    
    def _simulation_step(self):
        """
        Override base class to add force clamping.
        Starfish has low-mass paddle legs that can diverge without force limits.
        """
        n_bodies = len(self.link_config)
        
        for _ in range(self.per_frame_steps):
            # 1. Extract rigid body states from MuJoCo
            wp.launch(
                extract_body_states_3d,
                dim=(self.nworld, n_bodies),
                inputs=[
                    self.mjw_data.xipos,
                    self.mjw_data.xquat,
                    self.body_ids_wp,
                    self.positions_buffer,
                    self.quaternions_buffer,
                ],
                device=self.device,
            )
            
            # 2. Update rigid body positions in LBM
            wp.launch(
                convert_and_update_solid_batch_3d,
                dim=(self.nworld, n_bodies),
                inputs=[
                    self.lbm_solver.flows_wp,
                    self.solid_ids_wp,
                    self.positions_buffer,
                    self.quaternions_buffer,
                    self._mujoco_origins_wp,
                    self._lbm_origins_wp,
                    self._scales_wp,
                ],
                device=self.device,
            )
            
            # 3. LBM fluid solver step
            self.lbm_solver.step()
            
            # 4. Get fluid forces with physical unit conversion
            wp.launch(
                extract_forces_torques_physical_3d,
                dim=(self.nworld, n_bodies),
                inputs=[
                    self.lbm_solver.flows_wp,
                    self.solid_ids_wp,
                    self.force_conversion,
                    self.torque_conversion,
                    self.forces_buffer,
                    self.torques_buffer,
                ],
                device=self.device,
            )
            
            # 4.5 CLAMP forces to prevent divergence
            wp.launch(
                clamp_forces_3d_kernel,
                dim=(self.nworld, n_bodies),
                inputs=[
                    self.forces_buffer,
                    self.torques_buffer,
                    self.max_force_per_body,
                    self.max_torque_per_body,
                ],
                device=self.device,
            )
            
            # 5. Apply clamped fluid forces to MuJoCo
            self.mjw_data.xfrc_applied.zero_()
            
            wp.launch(
                fill_xfrc_3d_kernel,
                dim=(self.nworld, n_bodies),
                inputs=[
                    self.mjw_data.xfrc_applied,
                    self.body_ids_wp,
                    self.forces_buffer,
                    self.torques_buffer,
                ],
                device=self.device,
            )
            
            self.mjw_data.qfrc_applied.zero_()
            mjw.xfrc_accumulate(self.mjw_model, self.mjw_data, self.mjw_data.qfrc_applied)
            
            # 6. MuJoCo step
            if not self.graph_initialized:
                with wp.ScopedCapture() as capture:
                    mjw.step(self.mjw_model, self.mjw_data)
                self.graph_initialized = True
                self.mujoco_single_step_graph = capture.graph
            else:
                wp.capture_launch(self.mujoco_single_step_graph)
            wp.synchronize()
    
    def _select_next_goal(self, world_idx: int) -> int:
        """Select next goal for a world, avoiding recently visited goals."""
        if self.single_goal_mode or self.num_goals == 1:
            return 0
        
        history = self.goal_history[world_idx]
        available = [i for i in range(self.num_goals) if i not in history[-5:]]
        
        if not available:
            current = self.current_goal_idx[world_idx]
            available = [i for i in range(self.num_goals) if i != current]
            if not available:
                available = list(range(self.num_goals))
        
        return np.random.choice(available)
    
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
            compute_starfish_obs_3d_multigoal_kernel,
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
            check_goal_reached_3d_starfish_kernel,
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
                
                if not self.single_goal_mode:
                    self.current_goal_idx[w] = self._select_next_goal(w)
                    self.goal_history[w].append(self.current_goal_idx[w])
        
        if not self.single_goal_mode and np.any(goal_reached):
            self._update_goal_positions_wp()
            reset_mask_wp = wp.array(goal_reached.astype(np.int32), dtype=wp.int32, device=self.device)
            wp.launch(
                reset_prev_dist_3d_starfish_kernel,
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
        
        return goal_reached
    
    def _compute_reward(self, instability_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute reward function using Warp kernels."""
        # Distance-based reward
        wp.launch(
            compute_goal_reward_3d_starfish_kernel,
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
        
        wp.copy(self._prev_dist_wp, self._current_dist_buffer)
        
        # Add goal reached bonus
        goal_reached = self._goal_reached_buffer.numpy()
        rewards_np = self._rewards_buffer.numpy()
        rewards_np[goal_reached.astype(bool)] += self.goal_reached_bonus
        wp.copy(self._rewards_buffer, wp.array(rewards_np, dtype=wp.float32, device=self.device))
        
        # Smoothness + wave reward
        if self._prev_actions_wp is not None and self._current_actions_wp is not None:
            wp.launch(
                compute_smooth_reward_3d_starfish_kernel,
                dim=self.nworld,
                inputs=[
                    self.mjw_data.qvel,
                    self._current_actions_wp,
                    self._prev_actions_wp,
                    self._smooth_rewards_buffer,
                    self.action_smoothness_weight,
                    self.rotation_penalty_weight,
                    self.wave_reward_weight,
                    self.n_actuators,
                ],
                device=self.device,
            )
            
            wp.launch(
                add_smooth_rewards_3d_starfish_kernel,
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
                apply_instability_penalty_3d_starfish_kernel,
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
            check_anomaly_3d_starfish_kernel,
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
        """Check termination condition for all worlds."""
        self._terminated_buffer.zero_()
        
        wp.launch(
            check_boundary_3d_starfish_kernel,
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
        
        if instability_mask is not None:
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32, device=self.device)
            wp.launch(
                apply_instability_penalty_3d_starfish_kernel,
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
            check_stability_3d_starfish_kernel,
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
        """Execute one environment step.
        
        Args:
            action: Depends on control_mode:
                - 'direct': (nworld, 4) normalized Yaw joint angles [-1, 1]
                - 'multi_sine': (nworld, 4*K*2) A and C parameters [-1, 1]
        """
        action = np.clip(action, -1.0, 1.0)
        
        # Convert action to n_actuators-dim control signal
        full_ctrl = np.zeros((self.nworld, self.n_actuators), dtype=np.float32)
        
        if self.control_mode == 'multi_sine':
            n_params = self.n_yaw_joints * self.K
            A_flat = action[:, :n_params]  # (nworld, 4*K)
            C_flat = action[:, n_params:]  # (nworld, 4*K)
            
            A = A_flat.reshape(self.nworld, self.n_yaw_joints, self.K)
            C = C_flat.reshape(self.nworld, self.n_yaw_joints, self.K) * np.pi
            
            for w in range(self.nworld):
                t = self.multi_sine_t[w]
                for i in range(self.n_yaw_joints):
                    theta_norm = 0.0
                    for j in range(self.K):
                        theta_norm += A[w, i, j] * np.sin(np.pi / 2 * self.B_freqs[j] * t + C[w, i, j])
                    
                    theta_norm = np.clip(theta_norm, -1.0, 1.0)
                    full_ctrl[w, i] = theta_norm * self.yaw_max
                
                self.multi_sine_t[w] += self.multi_sine_dt
        else:
            # Direct control: action → yaw angles
            for i in range(self.n_yaw_joints):
                full_ctrl[:, i] = action[:, i] * self.yaw_max
        
        # Save current action for smoothness reward
        self.current_actions = full_ctrl.copy()
        self._current_actions_wp = wp.array(self.current_actions, dtype=wp.float32, device=self.device)
        
        # Apply control
        wp.copy(self.mjw_data.ctrl, wp.array(full_ctrl, dtype=wp.float32, device=self.device))
        
        # Execute physics simulation step
        self._simulation_step()
        
        self.step_counts += 1
        
        goal_reached = self._check_goals_reached()
        
        observation = self._get_obs()
        
        instability_mask = self._check_numerical_stability()
        
        obs_nan_mask = np.any(np.isnan(observation) | np.isinf(observation), axis=1)
        if np.any(obs_nan_mask):
            instability_mask = instability_mask | obs_nan_mask
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check termination
        self._is_terminated(instability_mask)
        
        # Record boundary termination before reward modifies terminated buffer
        boundary_terminated = self._terminated_buffer.numpy().astype(bool).copy()
        
        # Compute reward
        reward = self._compute_reward(instability_mask)
        
        # Get final terminated state
        terminated = self._terminated_buffer.numpy().astype(bool)
        anomaly_terminated = self._anomaly_buffer.numpy().astype(bool)
        
        # Single goal mode: terminate when goal reached
        goal_reached_mask = goal_reached.astype(bool)
        if self.single_goal_mode and np.any(goal_reached_mask):
            reward[goal_reached_mask] += self.goal_reached_bonus
            terminated = terminated | goal_reached_mask
        
        # Termination penalty (non-goal)
        non_goal_terminated = terminated & ~goal_reached_mask if self.single_goal_mode else terminated
        reward[non_goal_terminated] -= 1.0
        
        # Final safety check
        reward_nan_mask = np.zeros(self.nworld, dtype=bool)
        if np.any(np.isnan(reward)) or np.any(np.isinf(reward)):
            reward_nan_mask = np.isnan(reward) | np.isinf(reward)
            reward[reward_nan_mask] = self.anomaly_penalty - 1.0
            terminated[reward_nan_mask] = True
        
        truncated = np.array(self.step_counts >= self.max_episode_steps)
        done = terminated | truncated
        
        # Build termination reason per world
        term_reasons = []
        for w in range(self.nworld):
            reasons = []
            if boundary_terminated[w]:
                reasons.append("boundary")
            if instability_mask[w]:
                reasons.append("instability(NaN/Inf in qpos/qvel)")
            if obs_nan_mask[w]:
                reasons.append("obs_nan")
            if anomaly_terminated[w]:
                reasons.append("anomaly(force/reward)")
            if goal_reached_mask[w]:
                reasons.append("goal_reached")
            if reward_nan_mask[w]:
                reasons.append("reward_nan")
            if truncated[w]:
                reasons.append("truncated(max_steps)")
            term_reasons.append("|".join(reasons) if reasons else "running")
        
        # Get center positions for debug
        head_positions = np.zeros((self.nworld, 3), dtype=np.float32)
        for w in range(self.nworld):
            flow = self.lbm_solver.flows[w]
            pos = flow.solid_position.numpy()[0]
            head_positions[w] = [pos[0] / self.nx, pos[1] / self.ny, pos[2] / self.nz]
        
        info = self._get_info()
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["goals_reached"] = self.goals_reached.copy()
        info["term_reason"] = term_reasons
        info["head_pos_normalized"] = head_positions
        info["boundary_terminated"] = boundary_terminated
        info["instability"] = instability_mask
        info["anomaly"] = anomaly_terminated
        
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
        
        self.goals_reached[:] = 0
        self.goal_history = [[] for _ in range(self.nworld)]
        
        for w in range(self.nworld):
            self.current_goal_idx[w] = self._select_next_goal(w)
            self.goal_history[w].append(self.current_goal_idx[w])
        
        self._update_goal_positions_wp()
        
        wp.launch(
            init_prev_dist_3d_starfish_kernel,
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
        
        self._prev_actions_wp = wp.zeros((self.nworld, self.n_actuators), dtype=wp.float32, device=self.device)
        
        # Reset multi-sine time counter
        if self.control_mode == 'multi_sine':
            self.multi_sine_t[:] = 0.0
        
        obs = self._get_obs()
        return obs
    
    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        """Reset specific worlds indicated by reset_mask."""
        obs = super().partial_reset(reset_mask)
        
        if not np.any(reset_mask):
            return self._get_obs()
        
        for w in range(self.nworld):
            if reset_mask[w]:
                self.goals_reached[w] = 0
                self.goal_history[w] = []
                self.current_goal_idx[w] = self._select_next_goal(w)
                self.goal_history[w].append(self.current_goal_idx[w])
                
                # Reset multi-sine time counter
                if self.control_mode == 'multi_sine':
                    self.multi_sine_t[w] = 0.0
        
        self._update_goal_positions_wp()
        
        reset_mask_wp = wp.array(reset_mask.astype(np.int32), dtype=wp.int32, device=self.device)
        wp.launch(
            reset_prev_dist_3d_starfish_kernel,
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
        
        if self._prev_actions_wp is None:
            self._prev_actions_wp = wp.zeros((self.nworld, self.n_actuators), dtype=wp.float32, device=self.device)
        
        wp.launch(
            reset_prev_actions_3d_starfish_kernel,
            dim=self.nworld,
            inputs=[reset_mask_wp, self._prev_actions_wp, self.n_actuators],
            device=self.device,
        )
        
        return self._get_obs()
    
    def get_current_goal(self, world_idx: int = 0) -> Tuple[float, float, float]:
        """Get current goal position for a world (for visualization)."""
        goal_idx = self.current_goal_idx[world_idx]
        return self.goal_positions_list[goal_idx]
