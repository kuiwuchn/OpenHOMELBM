"""
3D Eel/Ribbon Fish LBM Environment for MuJoCo Warp with nworld support
Multi-goal version: robot navigates to goal points in 3D space

带鱼/鳗鱼机器人环境 - 基于LBM流体仿真 (Yaw+Roll双关节版本)
结构:
- 12节锥形身体 (头部较窄，中部最宽，尾部最细)
- 22个关节 (11对 Yaw + Roll):
  - Yaw关节 (绕Z轴): 水平摆动，水平行波推进
  - Roll关节 (绕Y轴): 滚转

控制模式 (control_mode):
1. 'direct' (默认): 直接控制11个Yaw关节角度
   - Action: (nworld, 11) normalized [-1, 1] -> 目标角度

2. 'multi_sine': 多频率正弦波控制 (频域PD控制)
   - Action: (nworld, 11*K*2) 包含振幅A和相位C参数
   - 公式: θ_i* = Σ_{j=1}^{K} A_{ij} * sin(π/2 * B_j * t + C_{ij})
   - B_j = j * B_bar / K 为频率系数
   - Agent 学习 A 和 C 参数，实现更自然的波动控制

特点:
- 典型的鳗鱼式行波游动 (anguilliform locomotion)
- 支持3D空间任意方向游动
- 从头到尾振幅递增
- 高效的波动推进
- 头部朝向目标奖励 + 有效推力增强奖励
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


# ============== Warp Kernels for Eel Environment ==============


@wp.kernel
def compute_eel_obs_kernel(
    qfrc_applied: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    qpos: wp.array2d(dtype=wp.float32),  # (nworld, nq)
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),  # (nworld, 3)
    obs_out: wp.array2d(dtype=wp.float32),  # (nworld, obs_dim)
    nx: float,
    ny: float,
    nz: float,
    n_joints: int,
):
    """
    Compute observation for eel robot.

    Observation layout (n_joints=22, 11 pairs of Yaw+Roll):
    - Forces (6): fx, fy, fz, tau_x, tau_y, tau_z
    - Joint torques (n_joints): joint generalized forces
    - Position (3): x, y, z
    - Quaternion (4): w, x, y, z
    - Velocity (3): vx, vy, vz
    - Angular velocity (3): omega_x, omega_y, omega_z
    - Joint angles (n_joints): joint positions
    - Joint velocities (n_joints): joint velocities
    - LBM position (3): normalized x, y, z
    - Goal position (3): normalized goal x, y, z

    Total: 6 + 22 + 3 + 4 + 3 + 3 + 22 + 22 + 3 + 3 = 91 dims (for 22 joints)
    Formula: 25 + 3 * n_joints
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    idx = 0
    
    # Generalized forces on root body (6)
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
    
    # LBM position (normalized) (3) - use head (solid 0)
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
def check_boundary_eel_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    terminated_out: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    boundary_margin: float,
    n_solids: int,
):
    """Check boundary termination for all worlds."""
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
def check_goal_reached_eel_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    goal_reached_out: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    goal_threshold: float,
):
    """Check if robot reached goal."""
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
    dist_sq = dx * dx + dy * dy + dz * dz
    
    if dist_sq < goal_threshold * goal_threshold:
        goal_reached_out[world_idx] = 1
    else:
        goal_reached_out[world_idx] = 0


@wp.kernel
def check_stability_eel_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    instability_out: wp.array(dtype=wp.int32),
    nq: int,
    nv: int,
):
    """Check numerical stability."""
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
def compute_goal_reward_eel_kernel(
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
    
    dist_improvement = prev_dist[world_idx] - current_dist
    rewards_out[world_idx] = 100.0 * dist_improvement
    
    current_dist_out[world_idx] = current_dist


@wp.kernel
def compute_smooth_reward_eel_kernel(
    qvel: wp.array2d(dtype=wp.float32),  # (nworld, nv)
    qpos: wp.array2d(dtype=wp.float32),  # (nworld, nq)
    current_actions: wp.array2d(dtype=wp.float32),  # (nworld, action_dim)
    prev_actions: wp.array2d(dtype=wp.float32),  # (nworld, action_dim)
    smooth_rewards_out: wp.array(dtype=wp.float32),  # (nworld,)
    action_smoothness_weight: float,
    rotation_penalty_weight: float,
    wave_reward_weight: float,
    action_dim: int,
):
    """
    Compute smoothness reward for eel with traveling wave.

    Components:
    1. Action smoothness: penalize large changes between consecutive actions
    2. Rotation penalty: penalize excessive spinning (mainly roll)
    3. Traveling wave reward: encourage phase difference between adjacent YAW joints ONLY

    Joint layout (22 joints = 11 pairs of Yaw+Roll):
    - action[0]:  joint1_yaw  (seg1-seg2)   <- head, 参与行波
    - action[1]:  joint1_roll
    - action[2]:  joint2_yaw  (seg2-seg3)   <- 参与行波
    - action[3]:  joint2_roll
    - ...
    - action[20]: joint11_yaw (seg11-seg12) <- tail, 参与行波
    - action[21]: joint11_roll

    Yaw joints: even indices 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20  (11 yaw joints)
    Roll joints: odd indices 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21  (not in wave reward)

    qvel layout:
    - qvel[0:3]: linear velocity (vx, vy, vz)
    - qvel[3:6]: angular velocity (ωx, ωy, ωz)
    - qvel[6:28]: joint velocities (22 joints, same order as actions)
    """
    world_idx = wp.tid()

    # ========== 1. Action Smoothness Penalty ==========
    action_change_sq = float(0.0)
    for i in range(action_dim):
        action_diff = current_actions[world_idx, i] - prev_actions[world_idx, i]
        action_change_sq = action_change_sq + action_diff * action_diff

    action_smooth_reward = -action_smoothness_weight * action_change_sq

    # ========== 2. Roll Penalty (only penalize roll, allow yaw for navigation) ==========
    omega_x = qvel[world_idx, 3]  # roll angular velocity
    rotation_penalty = -rotation_penalty_weight * (omega_x * omega_x)

    # ========== 3. Traveling Wave Reward (YAW joints ONLY) ==========
    # Encourage phase difference between consecutive yaw joints
    # For efficient eel swimming: wave propagates from head to tail
    # Yaw action indices: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20  (11 joints)
    # Yaw qvel indices:   6+0, 6+2, 6+4, ... 6+20                 (11 joints)
    # 10 consecutive pairs: (joint1,joint2), ..., (joint10,joint11)

    wave_reward = float(0.0)

    # pair (joint1_yaw idx 0) - (joint2_yaw idx 2)
    pd = qvel[world_idx, 6] * current_actions[world_idx, 2] - qvel[world_idx, 8] * current_actions[world_idx, 0]
    wave_reward = wave_reward + wp.abs(pd)
    # pair (joint2_yaw idx 2) - (joint3_yaw idx 4)
    pd = qvel[world_idx, 8] * current_actions[world_idx, 4] - qvel[world_idx, 10] * current_actions[world_idx, 2]
    wave_reward = wave_reward + wp.abs(pd)
    # pair (joint3_yaw idx 4) - (joint4_yaw idx 6)
    pd = qvel[world_idx, 10] * current_actions[world_idx, 6] - qvel[world_idx, 12] * current_actions[world_idx, 4]
    wave_reward = wave_reward + wp.abs(pd)
    # pair (joint4_yaw idx 6) - (joint5_yaw idx 8)
    pd = qvel[world_idx, 12] * current_actions[world_idx, 8] - qvel[world_idx, 14] * current_actions[world_idx, 6]
    wave_reward = wave_reward + wp.abs(pd)
    # pair (joint5_yaw idx 8) - (joint6_yaw idx 10)
    pd = qvel[world_idx, 14] * current_actions[world_idx, 10] - qvel[world_idx, 16] * current_actions[world_idx, 8]
    wave_reward = wave_reward + wp.abs(pd)
    # pair (joint6_yaw idx 10) - (joint7_yaw idx 12)
    pd = qvel[world_idx, 16] * current_actions[world_idx, 12] - qvel[world_idx, 18] * current_actions[world_idx, 10]
    wave_reward = wave_reward + wp.abs(pd)
    # pair (joint7_yaw idx 12) - (joint8_yaw idx 14)
    pd = qvel[world_idx, 18] * current_actions[world_idx, 14] - qvel[world_idx, 20] * current_actions[world_idx, 12]
    wave_reward = wave_reward + wp.abs(pd)
    # pair (joint8_yaw idx 14) - (joint9_yaw idx 16)
    pd = qvel[world_idx, 20] * current_actions[world_idx, 16] - qvel[world_idx, 22] * current_actions[world_idx, 14]
    wave_reward = wave_reward + wp.abs(pd)
    # pair (joint9_yaw idx 16) - (joint10_yaw idx 18)
    pd = qvel[world_idx, 22] * current_actions[world_idx, 18] - qvel[world_idx, 24] * current_actions[world_idx, 16]
    wave_reward = wave_reward + wp.abs(pd)
    # pair (joint10_yaw idx 18) - (joint11_yaw idx 20)
    pd = qvel[world_idx, 24] * current_actions[world_idx, 20] - qvel[world_idx, 26] * current_actions[world_idx, 18]
    wave_reward = wave_reward + wp.abs(pd)

    wave_reward = wave_reward_weight * wave_reward

    # ========== Total Smooth Reward ==========
    smooth_rewards_out[world_idx] = action_smooth_reward + rotation_penalty + wave_reward


@wp.kernel
def compute_heading_thrust_reward_kernel(
    qpos: wp.array2d(dtype=wp.float32),   # (nworld, nq)
    qvel: wp.array2d(dtype=wp.float32),   # (nworld, nv)
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),  # (nworld, 3) normalized
    rewards_out: wp.array(dtype=wp.float32),        # (nworld,) add-on
    nx: float,
    ny: float,
    nz: float,
    heading_weight: float,
    thrust_weight: float,
):
    """
    Heading + thrust reward:
      1. heading_reward: cos similarity between body_forward and goal direction
         encourages head to face the target at all times.
      2. thrust_bonus: v_forward * max(cos, 0) — effective thrust toward goal.
         Only counts thrust when heading is already aligned.

    Body coordinate (MuJoCo XML convention):
      - body_forward = local Y-axis rotated by root quaternion -> world frame
        (seg1 is placed at origin, seg2 is at y=-0.028, so body points in -Y locally;
         but the root freejoint quaternion carries the global heading)
      - MuJoCo quaternion: qpos[3..6] = (w, x, y, z)
    """
    world_idx = wp.tid()

    qw = qpos[world_idx, 3]
    qx = qpos[world_idx, 4]
    qy = qpos[world_idx, 5]
    qz = qpos[world_idx, 6]

    # Rotate local -Y (eel swims tail-to-head in -Y direction in body frame)
    # body_forward_world = R * (0, -1, 0)
    # Using quaternion rotation formula: v' = q * v * q^{-1}
    # For v=(0,-1,0):
    #   x' = 2*(qx*(-qy) + (-1)*(qw*qz + qx*0)) + ... simplified:
    # R * (0,-1,0):
    #   x' = -2*qx*qy - 2*qw*qz    (but sign from -Y)
    # More directly: body_forward = -body_Y_axis
    # body_Y_world: R*(0,1,0) = (2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx))
    by_x = 2.0 * (qx * qy + qw * qz)
    by_y = 1.0 - 2.0 * (qx * qx + qz * qz)
    by_z = 2.0 * (qy * qz - qw * qx)
    # eel body runs in -Y local direction (head at origin, tail at -Y)
    # so the "forward" direction the head points = +Y_world direction
    fwd_x = by_x
    fwd_y = by_y
    fwd_z = by_z

    # Goal direction from head in normalized LBM coords
    head_pos = flows[world_idx].solid_position[0]
    hx = head_pos[0] / nx
    hy = head_pos[1] / ny
    hz = head_pos[2] / nz

    gx = goal_positions[world_idx, 0] - hx
    gy = goal_positions[world_idx, 1] - hy
    gz = goal_positions[world_idx, 2] - hz

    dist = wp.sqrt(gx * gx + gy * gy + gz * gz) + 1.0e-8
    gx = gx / dist
    gy = gy / dist
    gz = gz / dist

    cos_align = fwd_x * gx + fwd_y * gy + fwd_z * gz

    # 1. Heading reward: cosine similarity in [-1, 1]
    r_heading = heading_weight * cos_align

    # 2. Thrust bonus: forward speed along body axis, weighted by alignment
    vx = qvel[world_idx, 0]
    vy = qvel[world_idx, 1]
    vz = qvel[world_idx, 2]
    v_forward = fwd_x * vx + fwd_y * vy + fwd_z * vz

    cos_clamp = float(0.0)
    if cos_align > 0.0:
        cos_clamp = cos_align
    r_thrust = thrust_weight * v_forward * cos_clamp

    rewards_out[world_idx] = rewards_out[world_idx] + r_heading + r_thrust


@wp.kernel
def add_smooth_rewards_kernel(
    rewards: wp.array(dtype=wp.float32),
    smooth_rewards: wp.array(dtype=wp.float32),
):
    """Add smooth rewards to main rewards buffer."""
    world_idx = wp.tid()
    rewards[world_idx] = rewards[world_idx] + smooth_rewards[world_idx]


@wp.kernel
def apply_instability_penalty_kernel(
    rewards: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    instability_mask: wp.array(dtype=wp.int32),
    penalty: float,
):
    """Apply instability penalty."""
    world_idx = wp.tid()
    if instability_mask[world_idx] == 1:
        rewards[world_idx] = penalty
        terminated[world_idx] = 1


@wp.kernel
def check_anomaly_kernel(
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
def reset_prev_dist_kernel(
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
def reset_prev_actions_kernel(
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
def init_prev_dist_kernel(
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


# ============== Eel 3D LBM Environment ==============


class Eel3DLBMEnv(LBMFluidEnv3D):
    """
    3D Eel/Ribbon Fish swimming environment with LBM fluid simulation.
    
    带鱼/鳗鱼机器人，采用典型的行波推进 (anguilliform locomotion)。
    双关节版本，支持3D空间任意方向游动。
    
    Structure:
    - 8 body segments (seg1 ~ seg8): flat ribbon-like boxes
    - seg1 is root body with freejoint
    - 14 joints connecting segments (7 pairs of Yaw + Roll)
      - Yaw joints: rotate around Z-axis for horizontal wave motion
      - Roll joints: rotate around Y-axis for body roll control
    
    Control: 14 joints (7 pairs)
    - Yaw joints: produce traveling wave for forward swimming
    - Roll joints: control body roll
    
    Swimming Strategy:
    - Horizontal: Yaw traveling wave only
    - Roll: body roll via Roll joints
    
    Goals:
    - Multiple goal points in a 3x3x3 grid (excluding center where robot starts)
    - Robot starts from center
    - Navigate to goals in 3D space
    """
    
    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        root_link: str = 'seg1',
        root_position: Optional[Tuple[float, float, float]] = None,
        nx: int = 150,
        ny: int = 250,
        nz: int = 60,
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
        """
        Initialize Eel 3D LBM Environment.
        
        Args:
            mjcf_path: Path to eel MJCF XML file
            root_link: Name of root link (default: 'seg1')
            root_position: LBM grid position of root link
            nx, ny, nz: LBM grid dimensions
            lbm_scale: Scale factor for geometry
            nworld: Number of parallel worlds
            max_episode_steps: Maximum steps per episode
            per_frame_steps: LBM-MuJoCo coupling iterations per step
            fluid_density: Fluid density in kg/m³
            device: Warp device
            goal_threshold: Distance threshold to reach goal
            single_goal_mode: If True, terminate when goal reached
            goal_position: Goal position [x, y, z] for single goal mode (normalized 0-1)
            control_mode: 'direct' (7-dim Yaw angles) or 'multi_sine' (A & C params for K freqs)
            K: Number of frequency components for multi_sine mode (default 3)
            B_bar: Maximum frequency parameter for multi_sine mode (default 1.0)
        """
        # Store goal mode settings before super().__init__
        self._init_single_goal_mode = single_goal_mode
        self._init_goal_position = goal_position if goal_position is not None else [0.5, 0.75, 0.5]
        
        # Store control mode settings
        self.control_mode = control_mode
        self.K = K
        self.B_bar = B_bar
        
        # Use eel_3d.xml from the same folder
        if mjcf_path is None:
            mjcf_path = os.path.join(os.path.dirname(__file__), 'eel_3d.xml')
        
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
            fluid_density=fluid_density,
            device=device,
        )
        
        # Number of actuators in XML (22: 11 yaw + 11 roll)
        self.n_actuators = self.mjw_model.nu  # 22

        # Number of controlled joints (only Yaw joints)
        self.n_yaw_joints = 11  # 11 yaw joints for swimming

        # Position control: angle limits in radians
        # Yaw: ±50° = ±0.873 rad, Roll: ±30° = ±0.524 rad
        self.yaw_max = 0.873   # 50 degrees in radians
        self.roll_max = 0.524  # 30 degrees in radians

        # Actuator indices in XML (22 actuators: yaw1, roll1, yaw2, roll2, ...)
        self.yaw_actuator_indices = list(range(0, self.n_actuators, 2))    # 0, 2, 4, ..., 20  (11 yaw)
        self.roll_actuator_indices = list(range(1, self.n_actuators, 2))   # 1, 3, 5, ..., 21  (11 roll)

        # For backward compatibility with reward computation
        self.n_joints = self.n_actuators  # 22 for obs/reward kernels
        self.yaw_indices = self.yaw_actuator_indices
        self.roll_indices = self.roll_actuator_indices
        
        # ============== Control Mode Setup ==============
        if self.control_mode == 'multi_sine':
            # Multi-sine wave control mode
            # Action: A (amplitudes) + C (phases) for K frequency components
            # θ_i* = Σ_{j=1}^{K} A_{ij} * sin(π/2 * B_j * t + C_{ij})
            # A: (n_yaw_joints, K) = (11, K), range [-1, 1]
            # C: (n_yaw_joints, K) = (11, K), range [-π, π] (normalized to [-1, 1])
            self.action_dim = self.n_yaw_joints * self.K * 2  # 11 * K * 2 (A + C)
            
            # Frequency coefficients: B_j = j * B_bar / K
            self.B_freqs = np.array([(j + 1) * self.B_bar / self.K for j in range(self.K)])
            
            # Time counter for sine wave
            self.multi_sine_t = np.zeros(nworld, dtype=np.float32)
            self.multi_sine_dt = 1.0 / 30.0  # Assume ~30 Hz control
            

        else:
            # Direct control mode (default)
            # Action: 11 Yaw joint angles, Roll joints set to 0
            self.action_dim = self.n_yaw_joints

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.nworld, self.action_dim),
            dtype=np.float32
        )

        # Observation space: 25 + 3 * n_actuators = 25 + 66 = 91 dims
        # (still observe all 22 joint states for full information)
        self.obs_dim = 25 + 3 * self.n_actuators
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
            # Yaw-only control: robot cannot move in z, fix z=0.5
            self.goal_positions_list = []
            for x in [0.25, 0.5, 0.75]:
                for y in [0.25, 0.5, 0.75]:
                    if not (x == 0.5 and y == 0.5):
                        self.goal_positions_list.append((x, y, 0.5))
        
        self.num_goals = len(self.goal_positions_list)
        self.goal_threshold = goal_threshold
        
        # Boundary parameters
        self.boundary_margin = 5.0
        
        # Current goal index for each world
        self.current_goal_idx = np.zeros(nworld, dtype=np.int32)
        
        # Goal history
        self.goal_history = [[] for _ in range(nworld)]
        
        # Goals reached counter
        self.goals_reached = np.zeros(nworld, dtype=np.int32)
        
        # Current goal positions (nworld, 3)
        self._goal_positions_wp = wp.zeros((nworld, 3), dtype=wp.float32, device=self.device)
        
        # For distance-based reward calculation
        self._prev_dist_wp = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._current_dist_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        
        # Smoothness reward parameters
        self.action_smoothness_weight = 0.3
        self.rotation_penalty_weight = 0.1
        self.wave_reward_weight = 0.005  # Traveling wave reward (yaw joints only)

        # Heading + thrust reward parameters
        self.heading_reward_weight = 2.0   # cosine similarity: head faces goal
        self.thrust_reward_weight = 5.0    # effective forward speed toward goal
        
        # Anomaly detection parameters
        self.force_threshold = 1e5
        self.anomaly_penalty = -10.0
        
        # Goal reaching bonus
        self.goal_reached_bonus = 10.0
        
        # Pre-allocate Warp buffers
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
            return spaces.Box(low=-np.inf, high=np.inf, shape=(self.nworld, 1), dtype=np.float32)
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.nworld, self.obs_dim), dtype=np.float32)
    
    def _select_next_goal(self, world_idx: int) -> int:
        """Select next goal for a world."""
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
        """Get observations for all worlds."""
        wp.launch(
            compute_eel_obs_kernel,
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
        """Check if any world reached its goal."""
        wp.launch(
            check_goal_reached_eel_kernel,
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
                reset_prev_dist_kernel,
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
        """Compute reward function."""
        # Distance-based reward
        wp.launch(
            compute_goal_reward_eel_kernel,
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

        # Heading + thrust reward: head faces goal + effective propulsion
        wp.launch(
            compute_heading_thrust_reward_kernel,
            dim=self.nworld,
            inputs=[
                self.mjw_data.qpos,
                self.mjw_data.qvel,
                self.lbm_solver.flows_wp,
                self._goal_positions_wp,
                self._rewards_buffer,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.heading_reward_weight,
                self.thrust_reward_weight,
            ],
            device=self.device,
        )
        
        # Smoothness reward (includes traveling wave)
        if self._prev_actions_wp is not None and self._current_actions_wp is not None:
            wp.launch(
                compute_smooth_reward_eel_kernel,
                dim=self.nworld,
                inputs=[
                    self.mjw_data.qvel,
                    self.mjw_data.qpos,
                    self._current_actions_wp,
                    self._prev_actions_wp,
                    self._smooth_rewards_buffer,
                    self.action_smoothness_weight,
                    self.rotation_penalty_weight,
                    self.wave_reward_weight,
                    self.action_dim,
                ],
                device=self.device,
            )
            
            wp.launch(
                add_smooth_rewards_kernel,
                dim=self.nworld,
                inputs=[self._rewards_buffer, self._smooth_rewards_buffer],
                device=self.device,
            )
        
        # Apply instability penalty
        if instability_mask is not None:
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32, device=self.device)
            wp.launch(
                apply_instability_penalty_kernel,
                dim=self.nworld,
                inputs=[self._rewards_buffer, self._terminated_buffer, instability_wp, self.anomaly_penalty],
                device=self.device,
            )
        
        # Check for anomalies
        wp.launch(
            check_anomaly_kernel,
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
        """Check termination condition."""
        self._terminated_buffer.zero_()
        
        wp.launch(
            check_boundary_eel_kernel,
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
                apply_instability_penalty_kernel,
                dim=self.nworld,
                inputs=[self._rewards_buffer, self._terminated_buffer, instability_wp, 0.0],
                device=self.device,
            )
        
        return self._terminated_buffer.numpy().astype(bool)
    
    def _check_numerical_stability(self) -> np.ndarray:
        """Check numerical stability."""
        self._instability_buffer.zero_()
        
        wp.launch(
            check_stability_eel_kernel,
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
                - 'direct': (nworld, 7) normalized Yaw joint angles [-1, 1]
                - 'multi_sine': (nworld, 7*K*2) A and C parameters [-1, 1]
                  Layout: [A_00, A_01, ..., A_0K, A_10, ..., A_6K, C_00, C_01, ..., C_6K]
        """
        # Clip action to [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        # Convert action to 14-dim control signal (radians)
        full_ctrl = np.zeros((self.nworld, self.n_actuators), dtype=np.float32)
        
        if self.control_mode == 'multi_sine':
            # Multi-sine wave control mode
            # Parse action into A (amplitudes) and C (phases)
            n_params = self.n_yaw_joints * self.K
            A_flat = action[:, :n_params]  # (nworld, 7*K)
            C_flat = action[:, n_params:]  # (nworld, 7*K)
            
            # Reshape to (nworld, n_yaw_joints, K)
            A = A_flat.reshape(self.nworld, self.n_yaw_joints, self.K)
            # C is normalized [-1, 1], convert to [-π, π]
            C = C_flat.reshape(self.nworld, self.n_yaw_joints, self.K) * np.pi
            
            # Compute joint angles using multi-sine formula
            # θ_i* = Σ_{j=1}^{K} A_{ij} * sin(π/2 * B_j * t + C_{ij})
            for w in range(self.nworld):
                t = self.multi_sine_t[w]
                for i in range(self.n_yaw_joints):
                    theta_norm = 0.0
                    for j in range(self.K):
                        theta_norm += A[w, i, j] * np.sin(np.pi / 2 * self.B_freqs[j] * t + C[w, i, j])
                    
                    # Clip to [-1, 1] and convert to radians
                    theta_norm = np.clip(theta_norm, -1.0, 1.0)
                    yaw_idx = self.yaw_actuator_indices[i]
                    full_ctrl[w, yaw_idx] = theta_norm * self.yaw_max
                
                # Update time for this world
                self.multi_sine_t[w] += self.multi_sine_dt
        else:
            # Direct control mode (default)
            # Convert normalized Yaw action to target angles (radians)
            for i, yaw_idx in enumerate(self.yaw_actuator_indices):
                full_ctrl[:, yaw_idx] = action[:, i] * self.yaw_max
        
        # Roll joints remain 0 (horizontal swimming)
        
        # Save current action for smoothness reward (use full 14-dim for kernel compatibility)
        self.current_actions = full_ctrl.copy()
        self._current_actions_wp = wp.array(self.current_actions, dtype=wp.float32, device=self.device)
        
        # Apply control
        wp.copy(self.mjw_data.ctrl, wp.array(full_ctrl, dtype=wp.float32, device=self.device))
        
        # Physics simulation
        self._simulation_step()
        
        # Update step counts
        self.step_counts += 1
        
        # Check if goals are reached
        goal_reached = self._check_goals_reached()
        
        # Get observation
        observation = self._get_obs()
        
        # Check stability
        instability_mask = self._check_numerical_stability()
        
        # Handle NaN/Inf in observations
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
        
        # Get final terminated state (may include anomaly from reward computation)
        terminated = self._terminated_buffer.numpy().astype(bool)
        anomaly_terminated = self._anomaly_buffer.numpy().astype(bool)
        
        # Single goal mode: terminate when goal reached
        goal_reached_mask = goal_reached.astype(bool)
        if self.single_goal_mode and np.any(goal_reached_mask):
            reward[goal_reached_mask] += self.goal_reached_bonus
            terminated = terminated | goal_reached_mask
        
        # Termination penalty
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
        
        # Get head positions for debug
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
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        """Reset all worlds."""
        if seed is not None:
            np.random.seed(seed)
        
        obs = super().reset(seed=seed, options=options)
        
        # Reset goal tracking
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
            init_prev_dist_kernel,
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
        
        # Initialize previous actions (use n_actuators=22 for kernel compatibility)
        self._prev_actions_wp = wp.zeros((self.nworld, self.n_actuators), dtype=wp.float32, device=self.device)
        
        # Reset multi-sine time counter
        if self.control_mode == 'multi_sine':
            self.multi_sine_t = np.zeros(self.nworld, dtype=np.float32)
        
        return self._get_obs()
    
    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        """Reset specific worlds."""
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
                
                # Reset multi-sine time counter for this world
                if self.control_mode == 'multi_sine':
                    self.multi_sine_t[w] = 0.0
        
        # Update goal positions
        self._update_goal_positions_wp()
        
        # Reset previous distance
        reset_mask_wp = wp.array(reset_mask.astype(np.int32), dtype=wp.int32, device=self.device)
        wp.launch(
            reset_prev_dist_kernel,
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
        
        # Reset previous actions (use n_actuators=22 for kernel compatibility)
        if self._prev_actions_wp is None:
            self._prev_actions_wp = wp.zeros((self.nworld, self.n_actuators), dtype=wp.float32, device=self.device)

        wp.launch(
            reset_prev_actions_kernel,
            dim=self.nworld,
            inputs=[reset_mask_wp, self._prev_actions_wp, self.n_actuators],
            device=self.device,
        )

        return self._get_obs()
    
    def get_current_goal(self, world_idx: int = 0) -> Tuple[float, float, float]:
        """Get current goal position for a world (for visualization)."""
        goal_idx = self.current_goal_idx[world_idx]
        return self.goal_positions_list[goal_idx]
