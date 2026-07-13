"""
3D Torus (Ring) Robot LBM Environment for MuJoCo Warp with nworld support
Multi-goal version: robot navigates to goal points in 3D space

Torus ring robot environment v3 — based on LBM fluid simulation
Structure:
- 12 arc-segment full-tube meshes arranged along a torus centerline (closed ring via weld)
- 24 joints (12 bend + 12 wave):
  - bend:    hinge along radial direction (ring contraction/expansion)
  - wave:    hinge along Y axis (out-of-plane bending, ring normal direction)

Anti-collision design:
  - Geom shortened 15% for inter-segment gap
  - Tube radius reduced to 0.012m for radial clearance
  - Conservative joint limits: bend ±30°, wave ±20°

Control mode (control_mode):
1. 'direct' (default): directly control 24 joint angles
   - Action: (nworld, 24) normalized [-1, 1] -> target angles

Swimming modes:
  1. Elliptical squeeze (bend joints with cos(2θ) pattern): main propulsion
     - Opposite sides of the ring alternately compress and expand
     - Creates oscillating elliptical deformation that pushes fluid
     - Generates thrust along ring normal (Y axis)
  2. Out-of-plane wave (wave joints): ring warping for ascend/descend
  3. Asymmetric squeeze (modulated cos(2θ)): steering/turning
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


# ============== Warp Helper Functions for Ring Center ==============

N_SEGMENTS = 12  # Number of torus segments


@wp.func
def compute_ring_center_pos(flow: HomeFlow3D) -> wp.vec3:
    """Compute ring center position as average of 12 segment positions."""
    cx = float(0.0)
    cy = float(0.0)
    cz = float(0.0)
    for i in range(12):
        p = flow.solid_position[i]
        cx = cx + p[0]
        cy = cy + p[1]
        cz = cz + p[2]
    return wp.vec3(cx / 12.0, cy / 12.0, cz / 12.0)


@wp.func
def compute_ring_normal(flow: HomeFlow3D) -> wp.vec3:
    """
    Compute ring plane normal from 12 segment positions.
    Uses cross product of adjacent (p_i - center) vectors accumulated around the ring.
    Returns normalized normal vector.
    """
    # First compute center
    center = compute_ring_center_pos(flow)

    # Accumulate cross products of adjacent radial vectors
    nx_acc = float(0.0)
    ny_acc = float(0.0)
    nz_acc = float(0.0)
    for i in range(12):
        j = (i + 1) % 12
        ri = flow.solid_position[i] - center
        rj = flow.solid_position[j] - center
        # Cross product ri × rj
        nx_acc = nx_acc + (ri[1] * rj[2] - ri[2] * rj[1])
        ny_acc = ny_acc + (ri[2] * rj[0] - ri[0] * rj[2])
        nz_acc = nz_acc + (ri[0] * rj[1] - ri[1] * rj[0])

    length = wp.sqrt(nx_acc * nx_acc + ny_acc * ny_acc + nz_acc * nz_acc)
    if length < 1.0e-8:
        # Fallback: ring normal is Y-axis (rest pose)
        return wp.vec3(0.0, 1.0, 0.0)
    return wp.vec3(nx_acc / length, ny_acc / length, nz_acc / length)


@wp.func
def compute_ring_center_vel(flow: HomeFlow3D) -> wp.vec3:
    """Compute ring center velocity as average of 12 segment linear velocities."""
    vx = float(0.0)
    vy = float(0.0)
    vz = float(0.0)
    for i in range(12):
        v = flow.linear_v[i]
        vx = vx + v[0]
        vy = vy + v[1]
        vz = vz + v[2]
    return wp.vec3(vx / 12.0, vy / 12.0, vz / 12.0)


# ============== Warp Kernels for Torus Environment ==============


@wp.kernel
def compute_torus_obs_kernel(
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
    Compute observation for torus ring robot.

    Uses ring center position, normal, and velocity instead of seg1's qpos/qvel.

    Observation layout (n_joints=24, 12 bend + 12 wave):
    - Forces (6): fx, fy, fz, tau_x, tau_y, tau_z
    - Joint torques (n_joints): joint generalized forces
    - Ring center position (3): x, y, z (from qpos, MuJoCo frame)
    - Ring normal (3): nx, ny, nz (ring plane normal from segment positions)
    - Ring center velocity (3): vx, vy, vz (average of 12 segment velocities)
    - Angular velocity (3): omega_x, omega_y, omega_z (from root qvel)
    - Joint angles (n_joints): joint positions
    - Joint velocities (n_joints): joint velocities
    - LBM ring center position (3): normalized x, y, z
    - Goal position (3): normalized goal x, y, z

    Total: 6 + 24 + 3 + 3 + 3 + 3 + 24 + 24 + 3 + 3 = 96 dims (for 24 joints)
    Formula: 24 + 3 * n_joints
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

    # Ring center position (3): from qpos (MuJoCo frame, for state tracking)
    obs_out[world_idx, idx] = qpos[world_idx, 0]
    obs_out[world_idx, idx + 1] = qpos[world_idx, 1]
    obs_out[world_idx, idx + 2] = qpos[world_idx, 2]
    idx = idx + 3

    # Ring normal (3): computed from 12 segment positions
    ring_normal = compute_ring_normal(flow)
    obs_out[world_idx, idx] = ring_normal[0]
    obs_out[world_idx, idx + 1] = ring_normal[1]
    obs_out[world_idx, idx + 2] = ring_normal[2]
    idx = idx + 3

    # Ring center velocity (3): average of 12 segment linear velocities
    ring_vel = compute_ring_center_vel(flow)
    obs_out[world_idx, idx] = ring_vel[0]
    obs_out[world_idx, idx + 1] = ring_vel[1]
    obs_out[world_idx, idx + 2] = ring_vel[2]
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

    # LBM ring center position (normalized) (3) - average of 12 segments
    ring_center = compute_ring_center_pos(flow)
    obs_out[world_idx, idx] = ring_center[0] / nx
    obs_out[world_idx, idx + 1] = ring_center[1] / ny
    obs_out[world_idx, idx + 2] = ring_center[2] / nz
    idx = idx + 3

    # Goal position (normalized) (3)
    obs_out[world_idx, idx] = goal_positions[world_idx, 0]
    obs_out[world_idx, idx + 1] = goal_positions[world_idx, 1]
    obs_out[world_idx, idx + 2] = goal_positions[world_idx, 2]


@wp.kernel
def check_boundary_torus_kernel(
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
def check_goal_reached_torus_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    goal_reached_out: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    goal_threshold: float,
):
    """Check if robot reached goal (using ring center = average of 12 segments)."""
    world_idx = wp.tid()
    flow = flows[world_idx]

    ring_center = compute_ring_center_pos(flow)
    current_x = ring_center[0] / nx
    current_y = ring_center[1] / ny
    current_z = ring_center[2] / nz

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
def check_stability_torus_kernel(
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
def compute_goal_reward_torus_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    prev_dist: wp.array(dtype=wp.float32),
    rewards_out: wp.array(dtype=wp.float32),
    current_dist_out: wp.array(dtype=wp.float32),
    nx: float,
    ny: float,
    nz: float,
):
    """Compute reward based on distance to goal (using ring center)."""
    world_idx = wp.tid()
    flow = flows[world_idx]

    ring_center = compute_ring_center_pos(flow)
    current_x = ring_center[0] / nx
    current_y = ring_center[1] / ny
    current_z = ring_center[2] / nz

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
def compute_smooth_reward_torus_kernel(
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
    Compute smoothness reward for torus ring robot.

    Components:
    1. Action smoothness: penalize large changes between consecutive actions
    2. Rotation penalty: penalize excessive spinning
    3. Squeeze reward: encourage elliptical deformation pattern (cos(2θ) mode)
       by rewarding opposite bend joints having similar values (in-phase)
       and adjacent-quadrant joints having opposite values (anti-phase).

    Joint layout (24 joints = 12 bend + 12 wave):
    Actuator ordering: [bend1,wave1, bend2,wave2, ...]
    - action[0]:  joint1_bend     (seg1-seg2, at 30°)
    - action[1]:  joint1_wave     (seg1-seg2, at 30°)
    - action[2]:  joint2_bend     (seg2-seg3, at 60°)
    - action[3]:  joint2_wave     (seg2-seg3, at 60°)
    - ...
    - action[22]: joint12_bend    (seg12-seg1 closure, at 0°)
    - action[23]: joint12_wave    (seg12-seg1 closure, at 0°)

    Bend joints: actuator indices 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22
    Wave joints: actuator indices 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23

    For squeeze reward: opposite bend pairs should be in-phase (same sign),
    and 90°-apart pairs should be anti-phase (opposite sign).
    Bend pair mapping (opposite pairs, 180° apart):
      bend1(30°) <-> bend7(210°):  act 0 <-> act 12
      bend2(60°) <-> bend8(240°):  act 2 <-> act 14
      bend3(90°) <-> bend9(270°):  act 4 <-> act 16
      bend4(120°)<-> bend10(300°): act 6 <-> act 18
      bend5(150°)<-> bend11(330°): act 8 <-> act 20
      bend6(180°)<-> bend12(0°):   act 10 <-> act 22
    """
    world_idx = wp.tid()

    # ========== 1. Action Smoothness Penalty ==========
    action_change_sq = float(0.0)
    for i in range(action_dim):
        action_diff = current_actions[world_idx, i] - prev_actions[world_idx, i]
        action_change_sq = action_change_sq + action_diff * action_diff

    action_smooth_reward = -action_smoothness_weight * action_change_sq

    # ========== 2. Rotation Penalty ==========
    # Penalize excessive angular velocity in all axes
    omega_x = qvel[world_idx, 3]
    omega_y = qvel[world_idx, 4]
    omega_z = qvel[world_idx, 5]
    rotation_penalty = -rotation_penalty_weight * (
        omega_x * omega_x + omega_y * omega_y + omega_z * omega_z
    )

    # ========== 3. Squeeze Reward (BEND joints) ==========
    # Encourage elliptical squeeze pattern: opposite bend joints should be
    # in-phase (same value), rewarding cos(2θ) deformation mode.
    # Opposite pairs (180° apart): bend_i and bend_{i+6}
    # Reward = sum of |bend_i - bend_{i+6}| being SMALL (opposite joints in-phase)
    # We use negative of difference: reward similarity of opposite pairs

    squeeze_reward = float(0.0)

    # Opposite pairs (should be similar for cos(2θ) mode):
    # bend1(act 0) ~ bend7(act 12)
    squeeze_reward = squeeze_reward - wp.abs(current_actions[world_idx, 0] - current_actions[world_idx, 12])
    # bend2(act 2) ~ bend8(act 14)
    squeeze_reward = squeeze_reward - wp.abs(current_actions[world_idx, 2] - current_actions[world_idx, 14])
    # bend3(act 4) ~ bend9(act 16)
    squeeze_reward = squeeze_reward - wp.abs(current_actions[world_idx, 4] - current_actions[world_idx, 16])
    # bend4(act 6) ~ bend10(act 18)
    squeeze_reward = squeeze_reward - wp.abs(current_actions[world_idx, 6] - current_actions[world_idx, 18])
    # bend5(act 8) ~ bend11(act 20)
    squeeze_reward = squeeze_reward - wp.abs(current_actions[world_idx, 8] - current_actions[world_idx, 20])
    # bend6(act 10) ~ bend12(act 22)
    squeeze_reward = squeeze_reward - wp.abs(current_actions[world_idx, 10] - current_actions[world_idx, 22])

    # Also reward 90°-apart pairs being anti-phase (opposite sign):
    # bend1(30°, act 0) vs bend4(120°, act 6) should be opposite
    squeeze_reward = squeeze_reward - wp.abs(current_actions[world_idx, 0] + current_actions[world_idx, 6])
    # bend2(60°, act 2) vs bend5(150°, act 8)
    squeeze_reward = squeeze_reward - wp.abs(current_actions[world_idx, 2] + current_actions[world_idx, 8])
    # bend3(90°, act 4) vs bend6(180°, act 10)
    squeeze_reward = squeeze_reward - wp.abs(current_actions[world_idx, 4] + current_actions[world_idx, 10])
    # bend4(120°, act 6) vs bend7(210°, act 12)
    squeeze_reward = squeeze_reward - wp.abs(current_actions[world_idx, 6] + current_actions[world_idx, 12])
    # bend5(150°, act 8) vs bend8(240°, act 14)
    squeeze_reward = squeeze_reward - wp.abs(current_actions[world_idx, 8] + current_actions[world_idx, 14])
    # bend6(180°, act 10) vs bend9(270°, act 16)
    squeeze_reward = squeeze_reward - wp.abs(current_actions[world_idx, 10] + current_actions[world_idx, 16])

    squeeze_reward = wave_reward_weight * squeeze_reward

    # ========== Total Smooth Reward ==========
    smooth_rewards_out[world_idx] = action_smooth_reward + rotation_penalty + squeeze_reward


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
    """Reset previous distance for specific worlds (using ring center)."""
    w = wp.tid()
    if reset[w] != 0:
        ring_center = compute_ring_center_pos(flows[w])
        current_x = ring_center[0] / nx
        current_y = ring_center[1] / ny
        current_z = ring_center[2] / nz
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
    """Initialize previous distance for all worlds (using ring center)."""
    w = wp.tid()
    ring_center = compute_ring_center_pos(flows[w])
    current_x = ring_center[0] / nx
    current_y = ring_center[1] / ny
    current_z = ring_center[2] / nz
    goal_x = goal_positions[w, 0]
    goal_y = goal_positions[w, 1]
    goal_z = goal_positions[w, 2]
    dx = current_x - goal_x
    dy = current_y - goal_y
    dz = current_z - goal_z
    prev_dist[w] = wp.sqrt(dx * dx + dy * dy + dz * dz)


# ============== Torus 3D LBM Environment ==============


class Torus3DLBMEnv(LBMFluidEnv3D):
    """
    3D Torus (Ring) swimming environment v3 with LBM fluid simulation.

    Closed ring with weld constraint: 12 arc-segment full-tube meshes
    connected by 24 joints (12 bend + 12 wave).
    seg12→seg1 closed by equality weld constraint (via seg12_closure sub-body).

    Structure:
    - 12 body segments (seg1 ~ seg12): full-tube arc meshes along torus centerline
    - seg1 is root body with freejoint
    - 24 joints: 12 bend (radial) + 12 wave (Z-axis)
    - 1 weld constraint closing the ring (seg12_closure→seg1)

    Anti-collision design:
    - Geom shortened 15% for inter-segment gap (~3mm)
    - Tube radius reduced to 0.012m for radial clearance
    - Conservative joint limits: bend ±30°, wave ±20°

    Control: 24 joints (12 bend + 12 wave)
    - Bend joints: ring contraction/expansion (elliptical squeeze)
    - Wave joints: out-of-plane bending (ring warping for ascend/descend)

    Swimming Strategy:
    - Elliptical squeeze: bend joints with cos(2θ) pattern (main propulsion)
    - Asymmetric squeeze: modulated cos(2θ) for steering
    - Warping: wave joints for out-of-plane motion (ascend/descend)

    Goals:
    - Multiple goal points in a 3x3x3 grid (excluding center)
    - Robot starts from center
    - Navigate to goals in 3D space
    """

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        root_link: str = 'seg1',
        root_position: Optional[Tuple[float, float, float]] = None,
        nx: int = 100,
        ny: int = 100,
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
    ):
        """
        Initialize Torus 3D LBM Environment.

        Args:
            mjcf_path: Path to torus MJCF XML file
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
            control_mode: 'direct' (21-dim all joints)
        """
        # Store goal mode settings before super().__init__
        self._init_single_goal_mode = single_goal_mode
        self._init_goal_position = goal_position if goal_position is not None else [0.5, 0.75, 0.5]

        # Store control mode settings
        self.control_mode = control_mode

        # Use torus_3d.xml from the same folder
        if mjcf_path is None:
            mjcf_path = os.path.join(os.path.dirname(__file__), 'torus_3d.xml')

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

        # Number of actuators in XML (24: 12 bend + 12 wave)
        self.n_actuators = self.mjw_model.nu  # 24

        # Joint layout: 24 joints ordered [bend1,wave1, bend2,wave2, ...]
        # Actuator indices:
        #   bend:    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22
        #   wave:    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23
        self.n_bend_joints = 12
        self.n_wave_joints = 12
        self.bend_actuator_indices = list(range(0, self.n_actuators, 2))    # [0,2,4,...,22]
        self.wave_actuator_indices = list(range(1, self.n_actuators, 2))    # [1,3,5,...,23]

        # For backward compatibility with reward/obs computation
        self.n_joints = self.n_actuators  # 24 for obs/reward kernels

        # Joint angle limits (radians)
        self.bend_max = 0.524     # 30 degrees
        self.wave_max = 0.349     # 20 degrees

        # Action dimension: all 24 joints in direct mode
        self.action_dim = self.n_actuators

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.nworld, self.action_dim),
            dtype=np.float32
        )

        # Observation space:
        # 6(forces) + n_actuators(torques) + 3(pos) + 3(normal) + 3(vel) + 3(omega)
        # + n_actuators(angles) + n_actuators(velocities) + 3(lbm_pos) + 3(goal_pos)
        self.obs_dim = 6 + self.n_actuators + 3 + 3 + 3 + 3 + self.n_actuators + self.n_actuators + 3 + 3
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
            self.goal_positions_list = []
            for x in [0.25, 0.5, 0.75]:
                for y in [0.25, 0.5, 0.75]:
                    for z in [0.25, 0.5, 0.75]:
                        if not (x == 0.5 and y == 0.5 and z == 0.5):
                            self.goal_positions_list.append((x, y, z))

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
        self.wave_reward_weight = 0.005  # Ring wave reward (bend joints)

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
            compute_torus_obs_kernel,
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
            check_goal_reached_torus_kernel,
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
            compute_goal_reward_torus_kernel,
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

        # Smoothness reward (includes ring wave)
        if self._prev_actions_wp is not None and self._current_actions_wp is not None:
            wp.launch(
                compute_smooth_reward_torus_kernel,
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
            check_boundary_torus_kernel,
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
            check_stability_torus_kernel,
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
            action: (nworld, 24) normalized joint angles [-1, 1]
                    Layout: [bend1,wave1, bend2,wave2, ...]
        """
        # Clip action to [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        # Convert action to control signal (radians)
        full_ctrl = np.zeros((self.nworld, self.n_actuators), dtype=np.float32)

        # Map normalized actions to joint angle limits per type
        for bend_idx in self.bend_actuator_indices:
            full_ctrl[:, bend_idx] = action[:, bend_idx] * self.bend_max
        for wave_idx in self.wave_actuator_indices:
            full_ctrl[:, wave_idx] = action[:, wave_idx] * self.wave_max

        # Save current action for smoothness reward
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

        # Get final terminated state
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

        # Get ring center positions for debug (average of 12 segments)
        ring_center_positions = np.zeros((self.nworld, 3), dtype=np.float32)
        for w in range(self.nworld):
            flow = self.lbm_solver.flows[w]
            all_pos = flow.solid_position.numpy()  # (n_solids, 3)
            center = np.mean(all_pos[:12], axis=0)  # average of 12 segments
            ring_center_positions[w] = [center[0] / self.nx, center[1] / self.ny, center[2] / self.nz]

        info = self._get_info()
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["goals_reached"] = self.goals_reached.copy()
        info["term_reason"] = term_reasons
        info["head_pos_normalized"] = ring_center_positions
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

        # Initialize previous actions
        self._prev_actions_wp = wp.zeros((self.nworld, self.n_actuators), dtype=wp.float32, device=self.device)

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

        # Reset previous actions
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
