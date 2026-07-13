"""
3D Manta Ray LBM Environment for MuJoCo Warp with nworld support.

Manta ray swimming with articulated pectoral wings and tail.
Based on successful starfish/eel implementation pattern.
Uses gym (not gymnasium) for compatibility with dreamer_vec_wrapper.
All data processing uses Warp kernels - numpy only at entry/exit points.

Structure:
  body (freejoint, 6DOF root)
    wing_root_R/L (1DOF: flap around Y-axis)
      wing_mid_R/L (2DOF: flap + twist)
        wing_tip_R/L (2DOF: flap + twist)
    tail (1DOF: yaw around Z-axis)
    cephalic_fin_R/L (fixed, no joints)

Total actuators: 2*(1+2+2) + 1 = 11 DOF
Coordinate: X=lateral, Y=forward, Z=up
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




# ============== Warp Kernels for 3D Manta Environment ==============





@wp.kernel
def compute_manta_obs_3d_kernel(
    qfrc_applied: wp.array2d(dtype=wp.float32),
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    obs_out: wp.array2d(dtype=wp.float32),
    nx: float,
    ny: float,
    nz: float,
    n_joints: int,
):
    """
    Compute observation for all worlds in parallel.

    Observation layout for 3D Manta (11 actuated joints):
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
    For 11 joints: 25 + 33 = 58 dims
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
    center_pos = flow.solid_position[0]  # body is solid 0
    obs_out[world_idx, idx] = center_pos[0] / nx
    obs_out[world_idx, idx + 1] = center_pos[1] / ny
    obs_out[world_idx, idx + 2] = center_pos[2] / nz
    idx = idx + 3

    # Goal position (normalized) (3)
    obs_out[world_idx, idx] = goal_positions[world_idx, 0]
    obs_out[world_idx, idx + 1] = goal_positions[world_idx, 1]
    obs_out[world_idx, idx + 2] = goal_positions[world_idx, 2]


@wp.kernel
def check_boundary_3d_manta_kernel(
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
def check_stability_3d_manta_kernel(
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
def check_goal_reached_manta_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    goal_reached_out: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    goal_threshold: float,
):
    """Check if manta reached its goal."""
    world_idx = wp.tid()
    flow = flows[world_idx]

    body_pos = flow.solid_position[0]
    current_x = body_pos[0] / nx
    current_y = body_pos[1] / ny
    current_z = body_pos[2] / nz

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
def apply_instability_penalty_kernel(
    rewards: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    instability_mask: wp.array(dtype=wp.int32),
    penalty: float,
):
    """Apply instability penalty and mark terminated worlds."""
    world_idx = wp.tid()
    if instability_mask[world_idx] == 1:
        rewards[world_idx] = penalty
        terminated[world_idx] = 1


@wp.kernel
def check_anomaly_3d_manta_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    qvel: wp.array2d(dtype=wp.float32),
    qfrc_applied: wp.array2d(dtype=wp.float32),
    terminated_out: wp.array(dtype=wp.int32),
    anomaly_out: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    position_slack: float,
    velocity_threshold: float,
    force_threshold: float,
    n_solids: int,
    nv: int,
):
    """Check for finite-but-exploding manta states before reward computation."""
    world_idx = wp.tid()
    anomaly_out[world_idx] = 0

    for i in range(nv):
        vel_val = qvel[world_idx, i]
        if wp.isnan(vel_val) or wp.isinf(vel_val) or wp.abs(vel_val) > velocity_threshold:
            anomaly_out[world_idx] = 1
            terminated_out[world_idx] = 1
            return

        force_val = qfrc_applied[world_idx, i]
        if wp.isnan(force_val) or wp.isinf(force_val) or wp.abs(force_val) > force_threshold:
            anomaly_out[world_idx] = 1
            terminated_out[world_idx] = 1
            return

    for solid_idx in range(n_solids):
        pos = flows[world_idx].solid_position[solid_idx]
        x = pos[0]
        y = pos[1]
        z = pos[2]
        if (
            wp.isnan(x) or wp.isinf(x) or
            wp.isnan(y) or wp.isinf(y) or
            wp.isnan(z) or wp.isinf(z)
        ):
            anomaly_out[world_idx] = 1
            terminated_out[world_idx] = 1
            return

        if (
            x < -position_slack or x > nx + position_slack or
            y < -position_slack or y > ny + position_slack or
            z < -position_slack or z > nz + position_slack
        ):
            anomaly_out[world_idx] = 1
            terminated_out[world_idx] = 1
            return


@wp.kernel
def compute_goal_reward_manta_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    prev_dist: wp.array(dtype=wp.float32),
    rewards_out: wp.array(dtype=wp.float32),
    current_dist_out: wp.array(dtype=wp.float32),
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    nx: float,
    ny: float,
    nz: float,
    w_dist: float,
    w_roll: float,
    w_heading: float,
    w_forward: float,
):
    """
    Compute comprehensive reward for manta swimming toward a goal.

    Components:
      1. Distance improvement: reward for getting closer to goal
      2. Roll penalty: penalize deviation from upright pose (roll ≈ 0)
      3. Heading reward: reward body forward direction (local Y) aligning with goal direction
      4. Forward velocity reward: reward velocity component along body forward axis toward goal

    Quaternion convention (MuJoCo): qpos[3:7] = (w, x, y, z)
    Coordinate convention: X=lateral, Y=forward, Z=up
    Body forward direction in world frame: rotate local Y-axis (0,1,0) by quaternion.
    """
    world_idx = wp.tid()
    flow = flows[world_idx]

    # --- Current position (normalized) ---
    body_pos = flow.solid_position[0]
    current_x = body_pos[0] / nx
    current_y = body_pos[1] / ny
    current_z = body_pos[2] / nz

    # --- Goal position ---
    goal_x = goal_positions[world_idx, 0]
    goal_y = goal_positions[world_idx, 1]
    goal_z = goal_positions[world_idx, 2]

    # --- 1. Distance improvement reward ---
    dx = current_x - goal_x
    dy = current_y - goal_y
    dz = current_z - goal_z
    current_dist = wp.sqrt(dx * dx + dy * dy + dz * dz)
    dist_improvement_raw = prev_dist[world_idx] - current_dist
    r_dist = w_dist * dist_improvement_raw

    # --- Quaternion: qpos[3]=w, qpos[4]=x, qpos[5]=y, qpos[6]=z ---
    qw = qpos[world_idx, 3]
    qx = qpos[world_idx, 4]
    qy = qpos[world_idx, 5]
    qz = qpos[world_idx, 6]

    # --- 2. Roll penalty ---
    # Roll (rotation about Y-axis in body frame) from quaternion:
    # For Manta coordinate (X=lateral, Y=forward, Z=up), roll is rotation about
    # the forward (Y) axis. Using atan2 formulation:
    #   roll = atan2(2*(qw*qy + qx*qz), 1 - 2*(qy*qy + qx*qx))
    # But a simpler proxy for small-angle penalty: use the sin(roll) ≈ roll
    # approximation via the Z-component of the body's local Z-axis in world frame.
    #
    # Body local Z-axis (0,0,1) rotated by quaternion gives world-frame "up" direction
    # of the body. For an upright manta, this should point along world +Z.
    # body_up = quat_rotate( (0,0,1) )
    #   body_up_x = 2*(qx*qz + qw*qy)     -- should be ~0 if no roll
    #   body_up_y = 2*(qy*qz - qw*qx)     -- should be ~0 if no pitch
    #   body_up_z = 1 - 2*(qx*qx + qy*qy) -- should be ~1 if upright
    body_up_x = 2.0 * (qx * qz + qw * qy)
    body_up_y = 2.0 * (qy * qz - qw * qx)
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    # Upright penalty: penalize deviation from world up directly.
    # This is stricter than the old |x|^2 + |y|^2 proxy because it
    # distinguishes upright (body_up_z=+1) from upside-down (body_up_z=-1).
    upright_error = 1.0 - body_up_z
    r_roll = -w_roll * (upright_error * upright_error)

    # --- 3. Heading reward: body forward direction aligned with goal direction ---
    # Body forward = local Y-axis (0,1,0) rotated by quaternion
    #   fwd_x = 2*(qx*qy + qw*qz)
    #   fwd_y = 1 - 2*(qx*qx + qz*qz)
    #   fwd_z = 2*(qy*qz - qw*qx)
    fwd_x = 2.0 * (qx * qy + qw * qz)
    fwd_y = 1.0 - 2.0 * (qx * qx + qz * qz)
    fwd_z = 2.0 * (qy * qz - qw * qx)

    # Direction from body to goal (unnormalized, using position diff)
    # Note: dx,dy,dz = current - goal, so to_goal = -dx,-dy,-dz
    to_goal_x = -dx
    to_goal_y = -dy
    to_goal_z = -dz
    to_goal_len = wp.sqrt(to_goal_x * to_goal_x + to_goal_y * to_goal_y + to_goal_z * to_goal_z)

    # --- 4. Goal-directed velocity reward ---
    # World-frame velocity: qvel[0:3] = (vx, vy, vz)
    vx = qvel[world_idx, 0]
    vy = qvel[world_idx, 1]
    vz = qvel[world_idx, 2]
    # Project velocity onto body forward direction (fwd is already unit-length from quaternion)
    v_forward = vx * fwd_x + vy * fwd_y + vz * fwd_z
    v_goal = float(0.0)
    if to_goal_len > 1.0e-6:
        inv_goal_len = 1.0 / to_goal_len
        v_goal = (
            vx * to_goal_x + vy * to_goal_y + vz * to_goal_z
        ) * inv_goal_len

    # Only reward positive velocity toward the goal.
    r_forward = float(0.0)
    if v_goal > 0.0:
        r_forward = w_forward * v_goal

    # cos(angle) = dot(fwd, to_goal_normalized)
    # Range: [-1, 1], 1 means perfectly aligned.
    # Only grant heading reward when the manta is actually moving toward the
    # goal or reducing goal distance so pure in-place spinning is not
    # rewarded.
    r_heading = float(0.0)
    if to_goal_len > 1.0e-6 and (dist_improvement_raw > 0.0 or v_goal > 0.0):
        cos_angle = (fwd_x * to_goal_x + fwd_y * to_goal_y + fwd_z * to_goal_z) / to_goal_len
        r_heading = w_heading * cos_angle

    # --- Total reward ---
    rewards_out[world_idx] = wp.clamp(r_dist + r_roll + r_heading + r_forward, -100.0, 100.0)

    current_dist_out[world_idx] = current_dist


@wp.kernel
def reset_prev_dist_manta_kernel(
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
        body_pos = flows[w].solid_position[0]
        current_x = body_pos[0] / nx
        current_y = body_pos[1] / ny
        current_z = body_pos[2] / nz
        goal_x = goal_positions[w, 0]
        goal_y = goal_positions[w, 1]
        goal_z = goal_positions[w, 2]
        dx = current_x - goal_x
        dy = current_y - goal_y
        dz = current_z - goal_z
        prev_dist[w] = wp.sqrt(dx * dx + dy * dy + dz * dz)


@wp.kernel
def init_prev_dist_manta_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    prev_dist: wp.array(dtype=wp.float32),
    nx: float,
    ny: float,
    nz: float,
):
    """Initialize previous distance for all worlds."""
    w = wp.tid()
    body_pos = flows[w].solid_position[0]
    current_x = body_pos[0] / nx
    current_y = body_pos[1] / ny
    current_z = body_pos[2] / nz
    goal_x = goal_positions[w, 0]
    goal_y = goal_positions[w, 1]
    goal_z = goal_positions[w, 2]
    dx = current_x - goal_x
    dy = current_y - goal_y
    dz = current_z - goal_z
    prev_dist[w] = wp.sqrt(dx * dx + dy * dy + dz * dz)


# ============== 3D Manta Ray LBM Environment Class ==============


class Manta3DLBMEnv(LBMFluidEnv3D):
    """
    3D Manta Ray swimming environment with LBM fluid simulation.

    Structure:
    - Main body (flat manta-shaped)
    - Left/right pectoral wings, each with 3 segments (root, mid, tip)
    - Tail (single segment)
    - Cephalic fins (fixed, no actuation)

    Actuators (11 DOF):
      Right wing:  wr_R_flap, wm_R_flap, wm_R_twist, wt_R_flap, wt_R_twist  (5)
      Left wing:   wr_L_flap, wm_L_flap, wm_L_twist, wt_L_flap, wt_L_twist  (5)
      Tail:        tail_yaw  (1)

    Control: position actuators with ctrlrange normalized to [-1, 1]
    Coordinate: X=lateral, Y=forward, Z=up
    """

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        root_link: str = 'body',
        root_position: Optional[Tuple[float, float, float]] = None,
        nx: int = 200,
        ny: int = 200,
        nz: int = 80,
        lbm_scale: float = 1.0,
        nworld: int = 1,
        max_episode_steps: int = 2000,
        per_frame_steps: int = 10,
        fluid_density: float = 1000.0,
        device: Optional[str] = None,
        goal_threshold: float = 0.08,
        single_goal_mode: bool = True,
        goal_position: Optional[List[float]] = None,
        # Reward shaping weights (inspired by "Task Specification" in swimming creature papers)
        reward_w_dist: float = 100.0,       # distance improvement weight (original)
        reward_w_roll: float = 0.5,         # roll penalty weight (keep upright)
        reward_w_heading: float = 0.2,      # heading alignment reward weight
        reward_w_forward: float = 0.1,      # forward velocity reward weight
    ):
        # Store goal mode settings before super().__init__
        self._init_single_goal_mode = single_goal_mode
        self._init_goal_position = goal_position if goal_position is not None else [0.5, 0.75, 0.5]

        if mjcf_path is None:
            mjcf_path = os.path.join(os.path.dirname(__file__), 'manta_3d.xml')

        if root_position is None:
            root_position = (nx / 2, ny * 0.4, nz / 2)

        super().__init__(
            mjcf_path=mjcf_path,
            root_link=root_link,
            root_position=root_position,
            nx=nx, ny=ny, nz=nz,
            lbm_scale=lbm_scale,
            nworld=nworld,
            max_episode_steps=max_episode_steps,
            per_frame_steps=per_frame_steps,
            fluid_density=fluid_density,
            device=device,
        )

        # Number of actuated joints
        self.n_joints = self.mj_model.njnt - 1  # exclude freejoint
        self.n_actuators = self.mjw_model.nu
        self.enable_stability_check = True

        # Reward shaping weights
        self.reward_w_dist = reward_w_dist
        self.reward_w_roll = reward_w_roll
        self.reward_w_heading = reward_w_heading
        self.reward_w_forward = reward_w_forward

        # Observation dimension: 25 + 3*n_joints (includes goal position)
        self.obs_dim = 25 + 3 * self.n_joints

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

        # Goal reaching bonus
        self.goal_reached_bonus = 10.0

        # Guardrails against finite-but-exploding coupled fluid/rigid states.
        # Keep the penalty aligned with the existing NaN/Inf handling so the
        # anomaly detector prevents reward spikes without overwhelming learning.
        self.force_threshold = 1e5
        self.velocity_threshold = 1e3
        self.position_slack = 10.0
        self.anomaly_penalty = -5.0

        # Observation buffer
        self._obs_buffer = wp.zeros((nworld, self.obs_dim), dtype=wp.float32, device=self.device)

        # Terminated buffer
        self._boundary_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._terminated_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._instability_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._anomaly_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._goal_reached_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._rewards_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)

        # Print configuration info
        print(f"Manta3DLBMEnv initialized:")
        print(f"  Grid: {nx}x{ny}x{nz}, scale: {lbm_scale}")
        print(f"  Bodies: {self.n_bodies}, Joints: {self.n_joints}, Actuators: {self.n_actuators}")
        print(f"  Solids (total): {self.solid_num}, Dynamic: {self.n_dynamic}, Static: {self.n_static}")
        print(f"  Obs dim: {self.obs_dim}, Action dim: {self.n_actuators}")
        print(f"  Force conversion: {self.force_conversion:.6f}")
        print(f"  Torque conversion: {self.torque_conversion:.8f}")
        print(f"  Reward weights: dist={self.reward_w_dist}, roll={self.reward_w_roll}, "
              f"heading={self.reward_w_heading}, forward={self.reward_w_forward}")

    def _create_observation_space(self) -> spaces.Space:
        """Create observation space."""
        n_joints = self.mj_model.njnt - 1
        obs_dim = 25 + 3 * n_joints
        return spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.nworld, obs_dim),
            dtype=np.float32
        )

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
        """Get observation using Warp kernel."""
        wp.launch(
            compute_manta_obs_3d_kernel,
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
        obs = self._obs_buffer.numpy().copy()
        return obs

    def _compute_reward(self, instability_mask=None) -> np.ndarray:
        """Comprehensive reward: distance + roll penalty + heading + forward velocity."""
        wp.launch(
            compute_goal_reward_manta_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._goal_positions_wp,
                self._prev_dist_wp,
                self._rewards_buffer,
                self._current_dist_buffer,
                self.mjw_data.qpos,
                self.mjw_data.qvel,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.reward_w_dist,
                self.reward_w_roll,
                self.reward_w_heading,
                self.reward_w_forward,
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

        return self._rewards_buffer.numpy()

    def _check_goals_reached(self) -> np.ndarray:
        """Check if any world reached its goal."""
        wp.launch(
            check_goal_reached_manta_kernel,
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
                reset_prev_dist_manta_kernel,
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



    def _is_terminated(self, instability_mask=None) -> np.ndarray:
        """Check termination: boundary violation or numerical instability."""
        self._boundary_buffer.zero_()
        self._terminated_buffer.zero_()
        self._anomaly_buffer.zero_()

        # Check boundary
        wp.launch(
            check_boundary_3d_manta_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._boundary_buffer,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.boundary_margin,
                self.solid_num,
            ],
            device=self.device,
        )
        wp.copy(self._terminated_buffer, self._boundary_buffer)

        # Check stability (same handling style as eel3d / serial_fish)
        if instability_mask is not None:
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32, device=self.device)
            wp.launch(
                apply_instability_penalty_kernel,
                dim=self.nworld,
                inputs=[self._rewards_buffer, self._terminated_buffer, instability_wp, 0.0],
                device=self.device,
            )

        wp.launch(
            check_anomaly_3d_manta_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self.mjw_data.qvel,
                self.mjw_data.qfrc_applied,
                self._terminated_buffer,
                self._anomaly_buffer,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.position_slack,
                self.velocity_threshold,
                self.force_threshold,
                self.solid_num,
                self.mjw_model.nv,
            ],
            device=self.device,
        )

        return self._terminated_buffer.numpy().astype(bool)

    def _check_numerical_stability(self) -> Optional[np.ndarray]:
        """Check for NaN/Inf in state."""
        wp.launch(
            check_stability_3d_manta_kernel,
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
        """Execute one environment step with goal-based reward."""
        action = np.clip(action, self.action_space.low, self.action_space.high) * self.action_scale
        wp.copy(self.mjw_data.ctrl, wp.array(action, dtype=wp.float32, device=self.device))

        # Physics simulation
        self._simulation_step()

        # Update step counts
        self.step_counts += 1

        # Check if goals are reached
        goal_reached = self._check_goals_reached()

        # Get observation
        observation = self._get_obs()

        # Check stability
        instability_mask = (
            self._check_numerical_stability()
            if hasattr(self, "enable_stability_check") and self.enable_stability_check
            else np.zeros(self.nworld, dtype=bool)
        )

        # Handle NaN/Inf in observations (same style as eel3d)
        obs_nan_mask = np.any(np.isnan(observation) | np.isinf(observation), axis=1)
        if np.any(obs_nan_mask):
            instability_mask = instability_mask | obs_nan_mask
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

        # Check termination and keep a clean copy of boundary terminations
        self._is_terminated(instability_mask)
        boundary_terminated = self._boundary_buffer.numpy().astype(bool).copy()

        # Compute reward
        reward = self._compute_reward(instability_mask)

        # Get final terminated state
        terminated = self._terminated_buffer.numpy().astype(bool)

        anomaly_terminated = self._anomaly_buffer.numpy().astype(bool).copy()
        instability_mask = np.asarray(instability_mask, dtype=bool)

        # Single goal mode: terminate when goal reached
        goal_reached_mask = goal_reached.astype(bool)
        if self.single_goal_mode and np.any(goal_reached_mask):
            terminated = terminated | goal_reached_mask

        non_goal_terminated = terminated & ~goal_reached_mask if self.single_goal_mode else terminated
        reward[non_goal_terminated] -= 1.0
        reward[anomaly_terminated | instability_mask] = self.anomaly_penalty

        # Final safety check
        reward_nan_mask = np.zeros(self.nworld, dtype=bool)
        if np.any(np.isnan(reward)) or np.any(np.isinf(reward)):
            reward_nan_mask = np.isnan(reward) | np.isinf(reward)
            reward[reward_nan_mask] = self.anomaly_penalty
            terminated[reward_nan_mask] = True

        truncated = np.array(self.step_counts >= self.max_episode_steps)
        done = terminated | truncated

        # Build termination reason per world (aligned with eel3d-style diagnostics)
        term_reasons = []
        body_positions = np.zeros((self.nworld, 3), dtype=np.float32)
        goal_positions = np.zeros((self.nworld, 3), dtype=np.float32)
        for w in range(self.nworld):
            reasons = []
            if boundary_terminated[w]:
                reasons.append("boundary")
            if instability_mask[w]:
                reasons.append("instability(NaN/Inf in qpos/qvel)")
            if obs_nan_mask[w]:
                reasons.append("obs_nan")
            if anomaly_terminated[w]:
                reasons.append("anomaly")
            if goal_reached_mask[w]:
                reasons.append("goal_reached")
            if reward_nan_mask[w]:
                reasons.append("reward_nan")
            if truncated[w]:
                reasons.append("truncated(max_steps)")
            term_reasons.append("|".join(reasons) if reasons else "running")

            pos = self.lbm_solver.flows[w].solid_position.numpy()[0]
            body_positions[w] = [pos[0] / self.nx, pos[1] / self.ny, pos[2] / self.nz]
            goal_positions[w] = np.array(self.get_current_goal(w), dtype=np.float32)

        info = self._get_info()
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["goals_reached"] = self.goals_reached.copy()
        info["term_reason"] = term_reasons
        info["head_pos_normalized"] = body_positions
        info["goal_pos_normalized"] = goal_positions
        info["boundary_terminated"] = boundary_terminated
        info["instability"] = instability_mask
        info["anomaly"] = anomaly_terminated

        return observation, reward, done, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        """Reset all worlds."""
        if seed is not None:
            np.random.seed(seed)

        super().reset(seed=seed, options=options)

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
            init_prev_dist_manta_kernel,
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

        return self._get_obs()

    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        """Reset specific worlds."""
        super().partial_reset(reset_mask)

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
            reset_prev_dist_manta_kernel,
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

        return self._get_obs()

    def get_current_goal(self, world_idx: int = 0) -> Tuple[float, float, float]:
        """Get current goal position for a world (for visualization)."""
        goal_idx = self.current_goal_idx[world_idx]
        return self.goal_positions_list[goal_idx]
