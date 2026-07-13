"""
3D Manta Ray Multi-Task LBM Environment with optional PD control modes.

Multi-task training for manta ray locomotion skills.
Inherits from Manta3DLBMEnv but replaces goal-reaching with velocity-based
task-conditioned rewards.

5 Tasks:
  0: FORWARD    — swim along +Y (body forward)
  1: TURN_LEFT  — yaw left while maintaining gentle forward cruise
  2: TURN_RIGHT — yaw right while maintaining gentle forward cruise
  3: ASCEND     — swim upward (+Z) with slight forward cruise
  4: DESCEND    — swim downward (-Z) with slight forward cruise

=== Frequency-Domain PD Control (Paper Section 3 & 5) ===

Instead of directly outputting joint targets, the neural network outputs
frequency-domain parameters (amplitude A and phase C) for each joint group:

  θ_i* = Σ_{j=1}^{K} A_{ij} * sin(π/2 * B_j * t + C_{ij})   (Eq. 7)

where B_j = j * B̄ / K, B̄ = max frequency (default 1.0).

MuJoCo's built-in position actuators with PD gains then track θ_i*:
  τ_i = -[kp * (θ_i - θ_i*) + kd * θ̇_i]                     (Eq. 6)

=== Reduced-Order Model (Paper Section 5) ===

For smooth motion, joints within a wing share control signals:
  Group 0: Right flap  (wr_R_flap, wm_R_flap, wt_R_flap)    — 3 joints → 1 signal
  Group 1: Right twist  (wm_R_twist, wt_R_twist)              — 2 joints → 1 signal
  Group 2: Left flap   (wr_L_flap, wm_L_flap, wt_L_flap)    — 3 joints → 1 signal
  Group 3: Left twist  (wm_L_twist, wt_L_twist)               — 2 joints → 1 signal
  Group 4: Tail        (tail_yaw)                              — 1 joint → 1 signal

Total: N_groups=5, K=4 harmonics → network outputs 5 * 4 * 2 = 40 values (A + C)

Observation: base obs (22 + 3*n_joints) + task one-hot (5) + phase (2)
The agent learns a single policy conditioned on task ID.

At inference time, a command sequence (e.g. ["forward","turn_left","ascend","forward"])
can be fed to produce complex maneuvers.

Coordinate: X=lateral, Y=forward, Z=up
Quaternion convention (MuJoCo): qpos[3:7] = (w, x, y, z)
"""
import gym
from gym import spaces
import numpy as np
import math
import warp as wp
import os
from typing import Optional, Tuple, Dict, Any, List

from .manta_lbm_env_3d import (
    Manta3DLBMEnv,
    compute_manta_obs_3d_kernel,
    check_boundary_3d_manta_kernel,
    check_stability_3d_manta_kernel,
    apply_instability_penalty_kernel,
    compute_goal_reward_manta_kernel,
)
from ..lbm_core_3d import HomeFlow3D


# ============== Frequency-Domain PD Control Constants ==============

# Number of harmonics (frequency components)
DEFAULT_K_HARMONICS = 2

# Max frequency (B̄ in the paper). B_j = j * B_BAR / K
DEFAULT_B_BAR = 1.0

# Reduced-order joint groups for Manta (11 joints → 5 groups)
# Each group: (group_name, [actuator_indices])
# Actuator order from XML:
#   0: pos_wr_R_flap    (right wing root flap)
#   1: pos_wm_R_flap    (right wing mid flap)
#   2: pos_wm_R_twist   (right wing mid twist)
#   3: pos_wt_R_flap    (right wing tip flap)
#   4: pos_wt_R_twist   (right wing tip twist)
#   5: pos_wr_L_flap    (left wing root flap)
#   6: pos_wm_L_flap    (left wing mid flap)
#   7: pos_wm_L_twist   (left wing mid twist)
#   8: pos_wt_L_flap    (left wing tip flap)
#   9: pos_wt_L_twist   (left wing tip twist)
#  10: pos_tail_yaw     (tail yaw)
REDUCED_ORDER_GROUPS = [
    ("right_flap",  [0, 1, 3]),     # wr_R_flap, wm_R_flap, wt_R_flap
    ("right_twist", [2, 4]),         # wm_R_twist, wt_R_twist
    ("left_flap",   [5, 6, 8]),     # wr_L_flap, wm_L_flap, wt_L_flap
    ("left_twist",  [7, 9]),         # wm_L_twist, wt_L_twist
    ("tail",        [10]),           # tail_yaw
]
N_GROUPS = len(REDUCED_ORDER_GROUPS)  # 5


# ============== Task Definitions ==============

TASK_FORWARD    = 0
TASK_TURN_LEFT  = 1
TASK_TURN_RIGHT = 2
TASK_ASCEND     = 3
TASK_DESCEND    = 4

NUM_TASKS = 5

TASK_NAMES = ["forward", "turn_left", "turn_right", "ascend", "descend"]


# ============== Warp Kernels for Multi-Task Reward ==============


@wp.func
def cross_vec3(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@wp.func
def quat_rotate_vec(qw: wp.float32, qx: wp.float32, qy: wp.float32, qz: wp.float32, v: wp.vec3) -> wp.vec3:
    """Rotate a body-frame vector into world frame using a MuJoCo quaternion."""
    qv = wp.vec3(qx, qy, qz)
    t = cross_vec3(qv, v)
    t = wp.vec3(2.0 * t[0], 2.0 * t[1], 2.0 * t[2])
    c = cross_vec3(qv, t)
    return wp.vec3(
        v[0] + qw * t[0] + c[0],
        v[1] + qw * t[1] + c[1],
        v[2] + qw * t[2] + c[2],
    )


@wp.func
def dot_vec3(a: wp.vec3, b: wp.vec3) -> wp.float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@wp.func
def safe_normalize_vec3(v: wp.vec3, fallback: wp.vec3) -> wp.vec3:
    norm = wp.sqrt(dot_vec3(v, v))
    if norm > 1.0e-6:
        inv = 1.0 / norm
        return wp.vec3(v[0] * inv, v[1] * inv, v[2] * inv)
    return fallback


@wp.func
def clamp_scalar(x: wp.float32, lo: wp.float32, hi: wp.float32) -> wp.float32:
    return wp.max(lo, wp.min(hi, x))


@wp.func
def wrap_angle(angle: wp.float32) -> wp.float32:
    pi = wp.float32(3.14159265358979323846)
    two_pi = wp.float32(6.28318530717958647692)
    if angle > pi:
        return angle - two_pi
    if angle < -pi:
        return angle + two_pi
    return angle


@wp.func
def quat_rotate_vec_inv(qw: wp.float32, qx: wp.float32, qy: wp.float32, qz: wp.float32, v: wp.vec3) -> wp.vec3:
    return quat_rotate_vec(qw, -qx, -qy, -qz, v)


@wp.func
def body_yaw_from_forward(body_forward: wp.vec3) -> wp.float32:
    return wp.atan2(body_forward[0], body_forward[1])


@wp.func
def body_pitch_from_forward(body_forward: wp.vec3) -> wp.float32:
    horiz = wp.sqrt(body_forward[0] * body_forward[0] + body_forward[1] * body_forward[1])
    return wp.atan2(body_forward[2], horiz + 1.0e-6)


@wp.func
def body_roll_from_right(body_right: wp.vec3) -> wp.float32:
    return wp.asin(clamp_scalar(body_right[2], -1.0, 1.0))


@wp.func
def soft_penalty(value: wp.float32, scale: wp.float32) -> wp.float32:
    v2 = value * value
    s2 = scale * scale + 1.0e-6
    return v2 / (v2 + s2)


@wp.func
def directional_reward(value: wp.float32, scale: wp.float32) -> wp.float32:
    return value / (wp.abs(value) + scale + 1.0e-6)


@wp.func
def positive_reward(value: wp.float32, scale: wp.float32) -> wp.float32:
    pos = wp.max(value, wp.float32(0.0))
    return pos / (pos + scale + 1.0e-6)


@wp.func
def free_directional_reward(value: wp.float32) -> wp.float32:
    return value / (wp.float32(1.0) + wp.abs(value))


@wp.func
def free_positive_reward(value: wp.float32) -> wp.float32:
    pos = wp.max(value, wp.float32(0.0))
    return pos / (wp.float32(1.0) + pos)


@wp.func
def free_soft_penalty(value: wp.float32) -> wp.float32:
    v2 = value * value
    return v2 / (wp.float32(1.0) + v2)


@wp.kernel
def compute_multitask_reward_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    task_ids: wp.array(dtype=wp.int32),
    rewards_out: wp.array(dtype=wp.float32),
    # Reward weights
    w_task: float,
    w_roll: float,
    w_smooth: float,
    w_offaxis: float,
    # Target velocities
    target_forward_vel: float,
    target_yaw_rate: float,
    target_vertical_vel: float,
    disable_speed_targets: int,
):
    """
    Compute task-conditioned reward for multi-task manta training.

    Reward design:
      1. Reward motion in the commanded direction with bounded, monotonic shaping.
      2. Encourage natural cruise-like turning / climbing instead of pure axis-isolated motion.
      3. Penalize off-axis drift and posture errors gently so learning stays easy.
      4. Penalize unnecessary world-frame angular velocity with small weights.

    If disable_speed_targets != 0, the main task term becomes target-free: it rewards
    signed progress in the commanded direction without preferring a specific speed.
    """
    world_idx = wp.tid()
    task = task_ids[world_idx]

    qw = qpos[world_idx, 3]
    qx = qpos[world_idx, 4]
    qy = qpos[world_idx, 5]
    qz = qpos[world_idx, 6]

    body_right = quat_rotate_vec(qw, qx, qy, qz, wp.vec3(1.0, 0.0, 0.0))
    body_forward = quat_rotate_vec(qw, qx, qy, qz, wp.vec3(0.0, 1.0, 0.0))
    body_up = quat_rotate_vec(qw, qx, qy, qz, wp.vec3(0.0, 0.0, 1.0))

    vel_world = wp.vec3(
        qvel[world_idx, 0],
        qvel[world_idx, 1],
        qvel[world_idx, 2],
    )
    omega_body = wp.vec3(
        qvel[world_idx, 3],
        qvel[world_idx, 4],
        qvel[world_idx, 5],
    )
    omega_world = quat_rotate_vec(qw, qx, qy, qz, omega_body)

    v_forward = dot_vec3(vel_world, body_forward)
    v_lateral = dot_vec3(vel_world, body_right)
    v_vertical = vel_world[2]
    yaw_rate = omega_world[2]

    upright_roll = body_up[0] * body_up[0]
    upright_pitch = body_up[1] * body_up[1]

    r_task = float(0.0)
    r_upright = float(0.0)
    r_offaxis = float(0.0)

    turn_forward_scale = 0.5 * target_forward_vel
    vertical_forward_scale = 0.5 * target_forward_vel

    if task == TASK_FORWARD:
        if disable_speed_targets != 0:
            r_task = w_task * free_directional_reward(v_forward)
            r_offaxis = -w_offaxis * (
                0.60 * free_soft_penalty(v_lateral)
                + 0.15 * free_soft_penalty(v_vertical)
                + 0.15 * free_soft_penalty(yaw_rate)
            )
        else:
            r_task = w_task * directional_reward(v_forward, target_forward_vel)
            r_offaxis = -w_offaxis * (
                0.60 * soft_penalty(v_lateral, target_forward_vel)
                + 0.15 * soft_penalty(v_vertical, target_vertical_vel)
                + 0.15 * soft_penalty(yaw_rate, target_yaw_rate)
            )
        r_upright = -w_roll * (0.50 * upright_roll + 0.50 * upright_pitch)

    elif task == TASK_TURN_LEFT:
        if disable_speed_targets != 0:
            r_task = w_task * (
                0.85 * free_directional_reward(yaw_rate)
                + 0.15 * free_positive_reward(v_forward)
            )
            r_offaxis = -w_offaxis * (
                0.45 * free_soft_penalty(v_lateral)
                + 0.15 * free_soft_penalty(v_vertical)
            )
        else:
            r_task = w_task * (
                0.85 * directional_reward(yaw_rate, target_yaw_rate)
                + 0.15 * positive_reward(v_forward, turn_forward_scale)
            )
            r_offaxis = -w_offaxis * (
                0.45 * soft_penalty(v_lateral, target_forward_vel)
                + 0.15 * soft_penalty(v_vertical, target_vertical_vel)
            )
        r_upright = -w_roll * (0.10 * upright_roll + 0.40 * upright_pitch)

    elif task == TASK_TURN_RIGHT:
        if disable_speed_targets != 0:
            r_task = w_task * (
                0.85 * free_directional_reward(-yaw_rate)
                + 0.15 * free_positive_reward(v_forward)
            )
            r_offaxis = -w_offaxis * (
                0.45 * free_soft_penalty(v_lateral)
                + 0.15 * free_soft_penalty(v_vertical)
            )
        else:
            r_task = w_task * (
                0.85 * directional_reward(-yaw_rate, target_yaw_rate)
                + 0.15 * positive_reward(v_forward, turn_forward_scale)
            )
            r_offaxis = -w_offaxis * (
                0.45 * soft_penalty(v_lateral, target_forward_vel)
                + 0.15 * soft_penalty(v_vertical, target_vertical_vel)
            )
        r_upright = -w_roll * (0.10 * upright_roll + 0.40 * upright_pitch)

    elif task == TASK_ASCEND:
        if disable_speed_targets != 0:
            r_task = w_task * (
                0.85 * free_directional_reward(v_vertical)
                + 0.15 * free_positive_reward(v_forward)
            )
            r_offaxis = -w_offaxis * (
                0.45 * free_soft_penalty(v_lateral)
                + 0.15 * free_soft_penalty(yaw_rate)
            )
        else:
            r_task = w_task * (
                0.85 * directional_reward(v_vertical, target_vertical_vel)
                + 0.15 * positive_reward(v_forward, vertical_forward_scale)
            )
            r_offaxis = -w_offaxis * (
                0.45 * soft_penalty(v_lateral, target_forward_vel)
                + 0.15 * soft_penalty(yaw_rate, target_yaw_rate)
            )
        r_upright = -w_roll * (0.15 * upright_roll + 0.25 * upright_pitch)

    elif task == TASK_DESCEND:
        if disable_speed_targets != 0:
            r_task = w_task * (
                0.85 * free_directional_reward(-v_vertical)
                + 0.15 * free_positive_reward(v_forward)
            )
            r_offaxis = -w_offaxis * (
                0.45 * free_soft_penalty(v_lateral)
                + 0.15 * free_soft_penalty(yaw_rate)
            )
        else:
            r_task = w_task * (
                0.85 * directional_reward(-v_vertical, target_vertical_vel)
                + 0.15 * positive_reward(v_forward, vertical_forward_scale)
            )
            r_offaxis = -w_offaxis * (
                0.45 * soft_penalty(v_lateral, target_forward_vel)
                + 0.15 * soft_penalty(yaw_rate, target_yaw_rate)
            )
        r_upright = -w_roll * (0.15 * upright_roll + 0.25 * upright_pitch)

    if task == TASK_TURN_LEFT or task == TASK_TURN_RIGHT:
        r_smooth = -w_smooth * (
            0.35 * omega_world[0] * omega_world[0]
            + 0.35 * omega_world[1] * omega_world[1]
        )
    else:
        r_smooth = -w_smooth * (
            0.50 * omega_world[0] * omega_world[0]
            + 0.50 * omega_world[1] * omega_world[1]
            + 0.15 * omega_world[2] * omega_world[2]
        )

    rewards_out[world_idx] = r_task + r_upright + r_offaxis + r_smooth


@wp.kernel
def compute_multitask_obs_kernel(
    qfrc_applied: wp.array2d(dtype=wp.float32),
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    flows: wp.array(dtype=HomeFlow3D),
    task_ids: wp.array(dtype=wp.int32),
    time_val: wp.array(dtype=wp.float32),
    obs_out: wp.array2d(dtype=wp.float32),
    nx: float,
    ny: float,
    nz: float,
    n_joints: int,
    n_tasks: int,
):
    """
    Compute observation for multi-task manta environment with phase info.

    Layout:
    - Forces (6): fx, fy, fz, tau_x, tau_y, tau_z
    - Joint torques (n_joints)
    - Position (3): x, y, z
    - Quaternion (4): w, x, y, z
    - Velocity (3): vx, vy, vz
    - Angular velocity (3): omega_x, omega_y, omega_z
    - Joint angles (n_joints)
    - Joint velocities (n_joints)
    - LBM position (3): normalized x, y, z
    - Task one-hot (n_tasks): 5-dim one-hot vector
    - Phase (2): sin(t), cos(t) — for frequency-domain control

    Total: 22 + 3*n_joints + 3 + n_tasks + 2
    For 11 joints, n_tasks=5: 22 + 33 + 3 + 5 + 2 = 65 dims
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

    # LBM position (normalized) (3)
    center_pos = flow.solid_position[0]
    obs_out[world_idx, idx] = center_pos[0] / nx
    obs_out[world_idx, idx + 1] = center_pos[1] / ny
    obs_out[world_idx, idx + 2] = center_pos[2] / nz
    idx = idx + 3

    # Task one-hot (n_tasks)
    task = task_ids[world_idx]
    for i in range(n_tasks):
        if i == task:
            obs_out[world_idx, idx] = 1.0
        else:
            obs_out[world_idx, idx] = 0.0
        idx = idx + 1

    # Phase info (2): sin(t), cos(t) for frequency-domain control
    t = time_val[world_idx]
    obs_out[world_idx, idx] = wp.sin(t)
    obs_out[world_idx, idx + 1] = wp.cos(t)
    idx = idx + 2


# ============== Multi-Task Environment Class ==============


class MantaMultiTaskEnv(Manta3DLBMEnv):
    """
    Multi-task manta locomotion environment with frequency-domain PD control.

    Instead of goal-reaching, the agent is given a task ID (one-hot in obs)
    and rewarded for performing the corresponding motion primitive:
      forward, turn_left, turn_right, ascend, descend.

    Control: The neural network outputs frequency-domain parameters (A, C)
    which are converted to joint target angles via Fourier synthesis,
    then tracked by MuJoCo's built-in PD controllers.

    Reduced-order: Multiple joints in a wing share one control signal
    for smooth, coordinated motion.

    Training: tasks are randomly sampled at reset (and optionally mid-episode).
    Inference: a command sequence can be provided to execute complex maneuvers.
    """

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        root_link: str = 'body',
        root_position: Optional[Tuple[float, float, float]] = None,
        nx: int = 250,
        ny: int = 250,
        nz: int = 100,
        lbm_scale: float = 0.8,    # 0.8 × 250 = 200: same robot grid size as (1.0, 200)
        nworld: int = 1,
        max_episode_steps: int = 2000,
        per_frame_steps: int = 10,
        fluid_density: float = 1000.0,
        device: Optional[str] = None,
        # Multi-task specific
        task_switch_interval: int = 0,      # steps between task switches (0 = only at reset)
        enabled_tasks: Optional[List[str]] = None,  # subset of tasks to train on
        # Task reward weights
        reward_w_task: float = 1.0,
        reward_w_roll: float = 0.10,
        reward_w_smooth: float = 0.015,
        reward_w_offaxis: float = 0.02,
        # Reference targets / scales for paper-style task rewards
        target_forward_vel: float = 0.20,
        target_yaw_rate: float = 0.15,
        target_vertical_vel: float = 0.12,
        disable_speed_targets: bool = False,
        use_direction_dist_tasks: bool = False,
        direction_dist_min: float = 0.12,
        direction_dist_max: float = 0.25,
        direction_goal_threshold: float = 0.06,
        direction_dist_w_dist: float = 100.0,
        direction_dist_w_roll: float = 0.5,
        direction_dist_w_heading: float = 0.2,
        direction_dist_w_forward: float = 0.1,
        direction_dist_goal_bonus: float = 10.0,
        direction_dist_terminate_on_goal: bool = True,
        alive_cost: float = 0.0,
        termination_penalty: float = 1.0,
        temporal_stack_obs: bool = False,
        # Frequency-domain PD control parameters
        k_harmonics: int = DEFAULT_K_HARMONICS,
        b_bar: float = DEFAULT_B_BAR,
        use_reduced_order: bool = True,
        control_mode: str = "direct",
    ):
        self.temporal_stack_obs = bool(temporal_stack_obs)
        # Initialize parent with dummy goal settings (we won't use goals)
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
            goal_threshold=1.0,       # effectively disabled
            single_goal_mode=True,
            goal_position=[0.5, 0.5, 0.5],  # dummy
            reward_w_dist=0.0,       # disable distance reward from parent
            reward_w_roll=0.0,       # we handle roll ourselves
            reward_w_heading=0.0,
            reward_w_forward=0.0,
        )

        # --- Multi-task config ---
        self.task_switch_interval = task_switch_interval

        # Parse enabled tasks
        if enabled_tasks is not None:
            self.enabled_task_ids = [TASK_NAMES.index(t) for t in enabled_tasks]
        else:
            self.enabled_task_ids = list(range(NUM_TASKS))

        # Task reward weights
        self.mt_reward_w_task = reward_w_task
        self.mt_reward_w_roll = reward_w_roll
        self.mt_reward_w_smooth = reward_w_smooth
        self.mt_reward_w_offaxis = reward_w_offaxis
        self.target_forward_vel = target_forward_vel
        self.target_yaw_rate = target_yaw_rate
        self.target_vertical_vel = target_vertical_vel
        self.disable_speed_targets = bool(disable_speed_targets)

        # Optional direction+distance task mode (PPO-friendly)
        self.use_direction_dist_tasks = bool(use_direction_dist_tasks)
        self.direction_dist_min = float(min(direction_dist_min, direction_dist_max))
        self.direction_dist_max = float(max(direction_dist_min, direction_dist_max))
        self.direction_goal_threshold = float(direction_goal_threshold)
        self.direction_dist_w_dist = float(direction_dist_w_dist)
        self.direction_dist_w_roll = float(direction_dist_w_roll)
        self.direction_dist_w_heading = float(direction_dist_w_heading)
        self.direction_dist_w_forward = float(direction_dist_w_forward)
        self.direction_dist_terminate_on_goal = bool(direction_dist_terminate_on_goal)
        self.alive_cost = float(alive_cost)
        self.termination_penalty = float(termination_penalty)
        self.goal_threshold = self.direction_goal_threshold if self.use_direction_dist_tasks else self.goal_threshold
        self.goal_reached_bonus = float(direction_dist_goal_bonus)
        self._direction_goal_positions_np = np.full((nworld, 3), 0.5, dtype=np.float32)

        # --- Control mode ---
        self.control_mode = control_mode.lower()
        if self.control_mode not in {"frequency", "direct"}:
            raise ValueError(f"Unknown control_mode '{control_mode}'. Expected 'frequency' or 'direct'.")

        # --- Frequency-domain / direct PD control ---
        self.k_harmonics = k_harmonics
        self.b_bar = b_bar
        self.use_reduced_order = use_reduced_order

        if use_reduced_order:
            self.n_ctrl_groups = N_GROUPS  # 5
            self.joint_groups = REDUCED_ORDER_GROUPS
        else:
            # Full-order control uses one policy output per actuator.
            self.n_ctrl_groups = self.n_actuators  # 11
            self.joint_groups = [
                (f"joint_{i}", [i]) for i in range(self.n_actuators)
            ]

        # Action dimensions for the two control parameterizations.
        self.direct_action_dim = self.n_ctrl_groups
        self.freq_action_dim = self.n_ctrl_groups * k_harmonics * 2

        # Build group-to-actuator mapping (numpy, used in CPU freq→ctrl conversion)
        self._group_actuator_indices = [g[1] for g in self.joint_groups]  # list of lists

        # Extract ctrl_range from MuJoCo model for scaling θ* → ctrl
        self._ctrl_lo = self.mj_model.actuator_ctrlrange[:, 0].copy()  # (n_actuators,)
        self._ctrl_hi = self.mj_model.actuator_ctrlrange[:, 1].copy()  # (n_actuators,)

        # Precompute frequency bases: B_j = (j+1) * b_bar / K for j=0..K-1
        self._freq_bases = np.array(
            [(j + 1) * b_bar / k_harmonics for j in range(k_harmonics)],
            dtype=np.float32
        )  # shape (K,)

        # Time tracking for frequency-domain control
        self._time_val = np.zeros(nworld, dtype=np.float32)
        self._time_val_wp = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._dt = self.mj_model.opt.timestep * per_frame_steps  # time per env step
        self._prev_qpos_buffer = wp.zeros((nworld, self.mj_model.nq), dtype=wp.float32, device=self.device)

        # --- Override obs dim: optionally append direction+distance task features ---
        self.base_single_obs_dim = 22 + 3 * self.n_joints + 3 + NUM_TASKS + 2
        self.extra_task_obs_dim = 4 if self.use_direction_dist_tasks else 0
        self.single_obs_dim = self.base_single_obs_dim + self.extra_task_obs_dim
        self.obs_dim = self.single_obs_dim * 2 if self.temporal_stack_obs else self.single_obs_dim

        # Current task for each world
        self._task_ids = np.zeros(nworld, dtype=np.int32)
        self._task_ids_wp = wp.zeros(nworld, dtype=wp.int32, device=self.device)

        # Steps since last task switch
        self._steps_since_switch = np.zeros(nworld, dtype=np.int32)

        # --- Per-episode statistics for logging ---
        self._ep_reward_sum = np.zeros(nworld, dtype=np.float32)
        self._ep_step_count = np.zeros(nworld, dtype=np.int32)
        self._ep_v_forward_sum = np.zeros(nworld, dtype=np.float32)
        self._ep_v_lateral_sum = np.zeros(nworld, dtype=np.float32)
        self._ep_v_vertical_sum = np.zeros(nworld, dtype=np.float32)
        self._ep_yaw_rate_sum = np.zeros(nworld, dtype=np.float32)
        self._ep_boundary_count = np.zeros(nworld, dtype=np.int32)

        # Internal Warp buffer stores the base single-frame observation; optional
        # direction+distance features are appended on CPU after the kernel launch.
        self._obs_buffer = wp.zeros((nworld, self.base_single_obs_dim), dtype=wp.float32, device=self.device)
        self.observation_space = self._create_observation_space()

        if self.control_mode == "frequency":
            self.action_dim = self.freq_action_dim
        else:
            self.action_dim = self.direct_action_dim

        # Override action space for the selected control mode.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(nworld, self.action_dim),
            dtype=np.float32
        )

        # Command sequence for inference mode
        self._command_sequence: Optional[List[dict]] = None
        self._command_index = 0
        self._command_step_counter = 0
        self._prev_ctrl = np.zeros((nworld, self.n_actuators), dtype=np.float32)
        self._prev_ctrl_valid = np.zeros(nworld, dtype=bool)

        print(f"MantaMultiTaskEnv initialized:")
        print(f"  Tasks: {[TASK_NAMES[i] for i in self.enabled_task_ids]}")
        print(f"  Task switch interval: {task_switch_interval} (0=only at reset)")
        print(f"  Control mode: {self.control_mode}")
        if self.control_mode == "frequency":
            print(f"  Control: K={k_harmonics} harmonics, B̄={b_bar}, reduced_order={use_reduced_order}")
            print(f"  Groups ({self.n_ctrl_groups}): {[g[0] for g in self.joint_groups]}")
            print(f"  Action dim: {self.freq_action_dim} (frequency params)")
        else:
            if self.use_reduced_order:
                print(f"  Control: direct grouped target angles, reduced_order=True")
            else:
                print(f"  Control: direct actuator target angles, reduced_order=False")
            print(f"  Groups ({self.n_ctrl_groups}): {[g[0] for g in self.joint_groups]}")
            print(f"  Action dim: {self.action_dim} (direct targets)")
        print(
            f"  Obs dim: {self.obs_dim} "
            f"({'temporal 2x stack' if self.temporal_stack_obs else 'single frame'})"
        )
        print(
            f"  Reward scales: fwd={target_forward_vel}, yaw={target_yaw_rate}, vert={target_vertical_vel}"
        )
        print(f"  disable_speed_targets={self.disable_speed_targets}")
        print(
            "  direction+distance tasks: "
            f"enabled={self.use_direction_dist_tasks}, "
            f"dist_range=({self.direction_dist_min:.3f}, {self.direction_dist_max:.3f}), "
            f"goal_threshold={self.goal_threshold:.3f}"
        )

    def _create_observation_space(self) -> spaces.Space:
        """Create observation space with task/goal conditioning and phase info appended."""
        single_obs_dim = getattr(self, "single_obs_dim", None)
        if single_obs_dim is None:
            n_joints = self.mj_model.njnt - 1
            base_single_obs_dim = 22 + 3 * n_joints + 3 + NUM_TASKS + 2
            extra_task_obs_dim = 4 if getattr(self, "use_direction_dist_tasks", False) else 0
            single_obs_dim = base_single_obs_dim + extra_task_obs_dim
        obs_dim = single_obs_dim * 2 if getattr(self, "temporal_stack_obs", False) else single_obs_dim
        return spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.nworld, obs_dim),
            dtype=np.float32
        )

    # --- Task management ---

    def _sample_tasks(self, mask: Optional[np.ndarray] = None):
        """Randomly assign tasks. If mask is given, only re-sample those worlds."""
        if mask is None:
            mask = np.ones(self.nworld, dtype=bool)
        for w in range(self.nworld):
            if mask[w]:
                self._task_ids[w] = np.random.choice(self.enabled_task_ids)
        self._update_task_ids_wp()
        if self.use_direction_dist_tasks:
            self._sample_direction_distance_goals(mask)

    def _update_task_ids_wp(self):
        """Sync task IDs to Warp."""
        wp.copy(
            self._task_ids_wp,
            wp.array(self._task_ids.astype(np.int32), dtype=wp.int32, device=self.device)
        )

    def _body_positions_normalized(self) -> np.ndarray:
        """Return current body positions in normalized LBM coordinates."""
        body_positions = np.zeros((self.nworld, 3), dtype=np.float32)
        for w in range(self.nworld):
            pos = self.lbm_solver.flows[w].solid_position.numpy()[0]
            body_positions[w] = [pos[0] / self.nx, pos[1] / self.ny, pos[2] / self.nz]
        return body_positions

    def _direction_template_for_task(self, task_id: int) -> np.ndarray:
        templates = {
            TASK_FORWARD: np.array([0.0, 1.0, 0.0], dtype=np.float32),
            TASK_TURN_LEFT: np.array([-0.55, 1.0, 0.0], dtype=np.float32),
            TASK_TURN_RIGHT: np.array([0.55, 1.0, 0.0], dtype=np.float32),
            TASK_ASCEND: np.array([0.0, 1.0, 0.75], dtype=np.float32),
            TASK_DESCEND: np.array([0.0, 1.0, -0.75], dtype=np.float32),
        }
        direction = templates.get(task_id, templates[TASK_FORWARD]).astype(np.float32, copy=True)
        norm = np.linalg.norm(direction)
        if norm > 1.0e-6:
            direction /= norm
        return direction

    def _sample_direction_distance_goals(self, mask: Optional[np.ndarray] = None):
        """Sample per-world goals using task directions expressed in the current body frame."""
        if mask is None:
            mask = np.ones(self.nworld, dtype=bool)
        if not np.any(mask):
            return

        body_positions = self._body_positions_normalized()
        qpos_np = self.mjw_data.qpos.numpy()
        margin = np.array([
            self.boundary_margin / float(self.nx),
            self.boundary_margin / float(self.ny),
            self.boundary_margin / float(self.nz),
        ], dtype=np.float32)

        for w in range(self.nworld):
            if not mask[w]:
                continue
            direction_body = self._direction_template_for_task(int(self._task_ids[w]))
            quat = qpos_np[w, 3:7].astype(np.float32, copy=False)
            direction = self._quat_rotate_vec_np(quat, direction_body.astype(np.float32, copy=False))
            norm = np.linalg.norm(direction)
            if norm > 1.0e-6:
                direction = direction / norm
            else:
                direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            travel_dist = np.random.uniform(self.direction_dist_min, self.direction_dist_max)
            goal = body_positions[w] + direction * np.float32(travel_dist)
            self._direction_goal_positions_np[w] = np.clip(goal, margin, 1.0 - margin)

        wp.copy(
            self._goal_positions_wp,
            wp.array(self._direction_goal_positions_np.astype(np.float32), dtype=wp.float32, device=self.device)
        )

        prev_dist = self._prev_dist_wp.numpy().copy()
        prev_dist[mask] = np.linalg.norm(
            self._direction_goal_positions_np[mask] - body_positions[mask], axis=1
        ).astype(np.float32)
        wp.copy(self._prev_dist_wp, wp.array(prev_dist, dtype=wp.float32, device=self.device))

    @staticmethod
    def _quat_rotate_vec_np(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quat
        qv = np.array([qx, qy, qz], dtype=np.float32)
        t = 2.0 * np.cross(qv, vec)
        return vec + qw * t + np.cross(qv, t)

    @staticmethod
    def _quat_rotate_vec_inv_np(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quat
        qv = np.array([-qx, -qy, -qz], dtype=np.float32)
        t = 2.0 * np.cross(qv, vec)
        return vec + qw * t + np.cross(qv, t)

    def _compute_direction_distance_obs_features(self) -> np.ndarray:
        """Append body-frame command direction and remaining distance."""
        body_positions = self._body_positions_normalized()
        qpos_np = self.mjw_data.qpos.numpy()
        features = np.zeros((self.nworld, 4), dtype=np.float32)

        goal_vec_world = self._direction_goal_positions_np - body_positions
        dist = np.linalg.norm(goal_vec_world, axis=1)
        for w in range(self.nworld):
            if dist[w] > 1.0e-6:
                dir_world = goal_vec_world[w] / dist[w]
            else:
                dir_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            quat = qpos_np[w, 3:7].astype(np.float32, copy=False)
            dir_body = self._quat_rotate_vec_inv_np(quat, dir_world.astype(np.float32, copy=False))
            features[w, :3] = dir_body
            features[w, 3] = dist[w]
        return features

    def set_command_sequence(self, commands: List[dict]):
        """
        Set a command sequence for inference.

        Args:
            commands: list of dicts, each with:
                - "task": str, one of TASK_NAMES
                - "steps": int, number of steps to hold this command
        Example:
            [
                {"task": "forward",    "steps": 200},
                {"task": "turn_left",  "steps": 100},
                {"task": "ascend",     "steps": 150},
                {"task": "forward",    "steps": 200},
            ]
        """
        self._command_sequence = commands
        self._command_index = 0
        self._command_step_counter = 0
        # Set initial task
        if commands:
            task_name = commands[0]["task"]
            task_id = TASK_NAMES.index(task_name)
            self._task_ids[:] = task_id
            self._update_task_ids_wp()
            if self.use_direction_dist_tasks:
                self._sample_direction_distance_goals()

    def _advance_command_sequence(self):
        """Advance through command sequence if in inference mode."""
        if self._command_sequence is None:
            return

        self._command_step_counter += 1
        cmd = self._command_sequence[self._command_index]

        if self._command_step_counter >= cmd["steps"]:
            self._command_index += 1
            self._command_step_counter = 0

            if self._command_index >= len(self._command_sequence):
                # Loop back or stay at last
                self._command_index = len(self._command_sequence) - 1
                self._command_step_counter = cmd["steps"]  # freeze at last
                return

            # Switch to next command
            next_cmd = self._command_sequence[self._command_index]
            task_name = next_cmd["task"]
            task_id = TASK_NAMES.index(task_name)
            self._task_ids[:] = task_id
            self._update_task_ids_wp()
            if self.use_direction_dist_tasks:
                self._sample_direction_distance_goals()

    # --- Override core methods ---

    def _get_obs(self) -> np.ndarray:
        """Get observation with task one-hot and phase info."""
        # Sync time to Warp
        wp.copy(
            self._time_val_wp,
            wp.array(self._time_val.astype(np.float32), dtype=wp.float32, device=self.device)
        )
        wp.launch(
            compute_multitask_obs_kernel,
            dim=self.nworld,
            inputs=[
                self.mjw_data.qfrc_applied,
                self.mjw_data.qpos,
                self.mjw_data.qvel,
                self.lbm_solver.flows_wp,
                self._task_ids_wp,
                self._time_val_wp,
                self._obs_buffer,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.n_joints,
                NUM_TASKS,
            ],
            device=self.device,
        )
        obs = self._obs_buffer.numpy().copy()
        if self.use_direction_dist_tasks:
            obs = np.concatenate((obs, self._compute_direction_distance_obs_features()), axis=1)
        return obs

    def _compute_reward(self, instability_mask=None) -> np.ndarray:
        """Task-conditioned reward computation."""
        if self.use_direction_dist_tasks:
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
                    self.direction_dist_w_dist,
                    self.direction_dist_w_roll,
                    self.direction_dist_w_heading,
                    self.direction_dist_w_forward,
                ],
                device=self.device,
            )
            wp.copy(self._prev_dist_wp, self._current_dist_buffer)

            goal_reached = self._goal_reached_buffer.numpy()
            rewards_np = self._rewards_buffer.numpy()
            rewards_np[goal_reached.astype(bool)] += self.goal_reached_bonus
            wp.copy(self._rewards_buffer, wp.array(rewards_np, dtype=wp.float32, device=self.device))
            return self._rewards_buffer.numpy()

        wp.launch(
            compute_multitask_reward_kernel,
            dim=self.nworld,
            inputs=[
                self.mjw_data.qpos,
                self.mjw_data.qvel,
                self._task_ids_wp,
                self._rewards_buffer,
                self.mt_reward_w_task,
                self.mt_reward_w_roll,
                self.mt_reward_w_smooth,
                self.mt_reward_w_offaxis,
                self.target_forward_vel,
                self.target_yaw_rate,
                self.target_vertical_vel,
                int(self.disable_speed_targets),
            ],
            device=self.device,
        )
        return self._rewards_buffer.numpy()

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Map policy output to MuJoCo actuator targets for the active control mode."""
        action = np.clip(action, -1.0, 1.0)
        if self.control_mode == "direct":
            return self._direct_to_ctrl(action)
        return self._freq_to_ctrl(action)

    def _direct_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Map normalized direct actions to MuJoCo targets via the active joint grouping."""
        nw = action.shape[0]
        ctrl = np.zeros((nw, self.n_actuators), dtype=np.float32)

        for g_idx, (_, act_indices) in enumerate(self.joint_groups):
            val = action[:, g_idx]
            for act_idx in act_indices:
                lo, hi = self._ctrl_lo[act_idx], self._ctrl_hi[act_idx]
                ctrl[:, act_idx] = lo + (val + 1.0) * 0.5 * (hi - lo)

        return ctrl

    def _freq_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """
        Convert frequency-domain action parameters to MuJoCo ctrl (target angles).

        The neural network outputs (A, C) pairs for each group and harmonic.
        Action layout per group g (K harmonics):
            [A_g0, C_g0, A_g1, C_g1, ..., A_g(K-1), C_g(K-1)]

        We compute:  θ*_g = Σ_{j=0}^{K-1} A_{gj} * sin(π/2 * B_j * t + C_{gj})
        Then scale θ*_g from [-1,1] to each actuator's ctrl_range.

        MuJoCo's position actuators (with built-in kp/kv PD gains) automatically
        compute: τ = kp * (θ* - θ) - kv * θ̇

        Args:
            action: (nworld, freq_action_dim) — raw NN output, clipped to [-1, 1]

        Returns:
            ctrl: (nworld, n_actuators) — target angles for MuJoCo position actuators
        """
        nw = action.shape[0]
        K = self.k_harmonics
        ctrl = np.zeros((nw, self.n_actuators), dtype=np.float32)

        for g_idx in range(self.n_ctrl_groups):
            base = g_idx * K * 2
            # Extract interleaved A and C: [A0, C0, A1, C1, ...] → separate arrays
            # A_gj: direct NN output (Eq.7) — action=0 → A=0 → stationary
            A_gj = action[:, base + 0: base + K * 2: 2]  # (nworld, K) — amplitudes
            c_raw = action[:, base + 1: base + K * 2: 2]  # (nworld, K) — phases raw
            # C_gj ∈ [-π, π]: linear scaling from action space [-1, 1]
            C_gj = c_raw * np.pi  # (nworld, K)


            # θ*_g = Σ_j A_{gj} * sin(π/2 * B_j * t + C_{gj})
            t = self._time_val[:nw, None]  # (nworld, 1)
            phase = (np.pi / 2) * self._freq_bases[None, :] * t + C_gj  # (nworld, K)
            theta_star = np.sum(A_gj * np.sin(phase), axis=1)  # (nworld,)

            # Clamp to [-1, 1]
            theta_star = np.clip(theta_star, -1.0, 1.0)

            # Apply to all actuators in this group, scaled to ctrl_range
            for act_idx in self._group_actuator_indices[g_idx]:
                lo, hi = self._ctrl_lo[act_idx], self._ctrl_hi[act_idx]
                ctrl[:, act_idx] = lo + (theta_star + 1.0) * 0.5 * (hi - lo)

        return ctrl

    def step(self, action: np.ndarray):
        """
        Execute one step with the selected control mode.

        Frequency mode converts Fourier coefficients to target angles.
        Direct mode sends actuator target angles to MuJoCo directly.
        """
        observation_before = self._get_obs() if self.temporal_stack_obs else None
        ctrl = self._action_to_ctrl(action)
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        # Write target angles to MuJoCo ctrl — position actuators do PD tracking
        wp.copy(self.mjw_data.ctrl, wp.array(ctrl, dtype=wp.float32, device=self.device))

        # Only advance the phase clock when the policy controls Fourier coefficients.
        if self.control_mode == "frequency":
            self._time_val += self._dt

        # Physics simulation
        self._simulation_step()

        # Update step counts
        self.step_counts += 1

        goal_reached = self._check_goals_reached() if self.use_direction_dist_tasks else np.zeros(self.nworld, dtype=np.int32)

        # Check stability
        instability_mask = (
            self._check_numerical_stability()
            if hasattr(self, "enable_stability_check") and self.enable_stability_check
            else np.zeros(self.nworld, dtype=bool)
        )

        # Check termination (boundary + instability)
        self._is_terminated(instability_mask)
        boundary_terminated = self._boundary_buffer.numpy().astype(bool).copy()
        anomaly_terminated = self._anomaly_buffer.numpy().astype(bool).copy()
        instability_mask = np.asarray(instability_mask, dtype=bool)

        # Compute reward before task switching so each action is credited against
        # the task that generated it.
        reward = self._compute_reward(instability_mask)

        # Small control-smoothness penalty on target-angle changes.
        if np.any(self._prev_ctrl_valid):
            ctrl_range = np.maximum(self._ctrl_hi - self._ctrl_lo, 1.0e-6)[None, :]
            ctrl_delta = (ctrl - self._prev_ctrl) / ctrl_range
            valid_mask = self._prev_ctrl_valid.astype(np.float32)
            reward += (
                -0.15 * self.mt_reward_w_smooth
                * np.mean(ctrl_delta * ctrl_delta, axis=1)
                * valid_mask
            )

        # Very small control-effort regularization to avoid chasing unnecessary
        # extreme commands now that task reward no longer has a hard speed target.
        ctrl_center = (0.5 * (self._ctrl_lo + self._ctrl_hi))[None, :]
        ctrl_half_range = np.maximum(0.5 * (self._ctrl_hi - self._ctrl_lo), 1.0e-6)[None, :]
        ctrl_effort = (ctrl - ctrl_center) / ctrl_half_range
        reward += -0.05 * self.mt_reward_w_smooth * np.mean(ctrl_effort * ctrl_effort, axis=1)

        self._prev_ctrl[:] = ctrl
        self._prev_ctrl_valid[:] = True

        # Per-step alive cost: penalise idling so "do nothing" is no longer safe.
        if self.alive_cost > 0.0:
            reward -= self.alive_cost

        # Get final terminated state
        terminated = self._terminated_buffer.numpy().astype(bool)
        goal_reached_mask = goal_reached.astype(bool)
        if self.use_direction_dist_tasks and self.direction_dist_terminate_on_goal and np.any(goal_reached_mask):
            terminated = terminated | goal_reached_mask

        # Termination penalty (configurable, default 1.0; set to 0 to rely only on
        # the implicit penalty of losing future returns).
        non_goal_terminated = (
            terminated & ~goal_reached_mask
            if self.use_direction_dist_tasks and self.direction_dist_terminate_on_goal
            else terminated
        )
        if self.termination_penalty > 0.0:
            reward[non_goal_terminated] -= self.termination_penalty
        reward[anomaly_terminated | instability_mask] = self.anomaly_penalty

        # Final safety check
        reward_nan_mask = np.zeros(self.nworld, dtype=bool)
        if np.any(np.isnan(reward)) or np.any(np.isinf(reward)):
            reward_nan_mask = np.isnan(reward) | np.isinf(reward)
            reward[reward_nan_mask] = self.anomaly_penalty
            terminated[reward_nan_mask] = True

        truncated = np.array(self.step_counts >= self.max_episode_steps)
        done = terminated | truncated

        # Mid-episode task switching only affects the next observation.
        if self.task_switch_interval > 0 and self._command_sequence is None:
            self._steps_since_switch += 1
            switch_mask = (self._steps_since_switch >= self.task_switch_interval) & ~done
            if np.any(switch_mask):
                self._sample_tasks(switch_mask)
                self._steps_since_switch[switch_mask] = 0

        # Advance command sequence for the next observation, not the current reward.
        if self._command_sequence is not None and not np.any(done):
            self._advance_command_sequence()

        # Returned observation should expose the next task after any switch.
        observation_after = self._get_obs()
        observation = (
            np.concatenate((observation_before, observation_after), axis=1)
            if self.temporal_stack_obs
            else observation_after
        )

        # Handle NaN/Inf in observations
        obs_nan_mask = np.any(np.isnan(observation) | np.isinf(observation), axis=1)
        if np.any(obs_nan_mask):
            instability_mask = instability_mask | obs_nan_mask
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
            reward[obs_nan_mask] = -1.0
            terminated[obs_nan_mask] = True
            done = terminated | truncated

        # --- Per-step statistics accumulation ---
        qvel_np = self.mjw_data.qvel.numpy()  # (nworld, nv)
        qpos_np = self.mjw_data.qpos.numpy()  # (nworld, nq)
        for w in range(self.nworld):
            # Extract body-frame velocities (same decomposition as reward kernel)
            qw, qx, qy, qz = qpos_np[w, 3], qpos_np[w, 4], qpos_np[w, 5], qpos_np[w, 6]
            vx, vy, vz = qvel_np[w, 0], qvel_np[w, 1], qvel_np[w, 2]
            # Body forward = local Y rotated by quaternion
            fwd_x = 2.0 * (qx * qy + qw * qz)
            fwd_y = 1.0 - 2.0 * (qx * qx + qz * qz)
            fwd_z = 2.0 * (qy * qz - qw * qx)
            # Body right = local X rotated by quaternion
            rt_x = 1.0 - 2.0 * (qy * qy + qz * qz)
            rt_y = 2.0 * (qx * qy - qw * qz)
            rt_z = 2.0 * (qx * qz + qw * qy)
            v_fwd = vx * fwd_x + vy * fwd_y + vz * fwd_z
            v_lat = vx * rt_x + vy * rt_y + vz * rt_z
            self._ep_v_forward_sum[w] += v_fwd
            self._ep_v_lateral_sum[w] += abs(v_lat)
            self._ep_v_vertical_sum[w] += vz
            omega_z = qvel_np[w, 5]
            self._ep_yaw_rate_sum[w] += abs(omega_z)
        self._ep_reward_sum += reward
        self._ep_step_count += 1
        self._ep_boundary_count += boundary_terminated.astype(np.int32)

        # --- Episode summary: attach to info when done ---
        # SB3 Monitor picks up info["episode"] automatically on done.
        for w in range(self.nworld):
            if done[w] and self._ep_step_count[w] > 0:
                n = float(self._ep_step_count[w])
                # These will be available as custom info keys
                # (logged by our EnvMetricCallback in train_sb3.py)
                pass  # per-world metrics added below in info dict

        # Build info
        term_reasons = []
        body_positions = np.zeros((self.nworld, 3), dtype=np.float32)
        for w in range(self.nworld):
            reasons = []
            if boundary_terminated[w]:
                reasons.append("boundary")
            if anomaly_terminated[w]:
                reasons.append("anomaly")
            if instability_mask[w]:
                reasons.append("instability")
            if obs_nan_mask[w]:
                reasons.append("obs_nan")
            if reward_nan_mask[w]:
                reasons.append("reward_nan")
            if goal_reached_mask[w]:
                reasons.append("goal_reached")
            if truncated[w]:
                reasons.append("truncated(max_steps)")
            term_reasons.append("|".join(reasons) if reasons else "running")

            pos = self.lbm_solver.flows[w].solid_position.numpy()[0]
            body_positions[w] = [pos[0] / self.nx, pos[1] / self.ny, pos[2] / self.nz]

        info = self._get_info()
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["term_reason"] = term_reasons
        info["head_pos_normalized"] = body_positions
        if self.use_direction_dist_tasks:
            info["goal_pos_normalized"] = self._direction_goal_positions_np.copy()
            info["goal_reached"] = goal_reached_mask.copy()
        info["current_tasks"] = [TASK_NAMES[t] for t in self._task_ids]
        info["task_ids"] = self._task_ids.copy()
        info["boundary_terminated"] = boundary_terminated
        info["anomaly"] = anomaly_terminated
        info["instability"] = instability_mask

        # Per-world episode metrics (nworld arrays, consumed by SB3 wrapper/callback)
        n_steps = np.maximum(self._ep_step_count.astype(np.float32), 1.0)
        info["ep_reward_mean"] = self._ep_reward_sum / n_steps
        info["ep_v_forward_mean"] = self._ep_v_forward_sum / n_steps
        info["ep_v_lateral_mean"] = self._ep_v_lateral_sum / n_steps
        info["ep_v_vertical_mean"] = self._ep_v_vertical_sum / n_steps
        info["ep_yaw_rate_mean"] = self._ep_yaw_rate_sum / n_steps
        info["ep_boundary_count"] = self._ep_boundary_count.copy()
        info["ep_length"] = self._ep_step_count.copy()

        # Reset accumulators for done worlds
        for w in range(self.nworld):
            if done[w]:
                self._ep_reward_sum[w] = 0.0
                self._ep_step_count[w] = 0
                self._ep_v_forward_sum[w] = 0.0
                self._ep_v_lateral_sum[w] = 0.0
                self._ep_v_vertical_sum[w] = 0.0
                self._ep_yaw_rate_sum[w] = 0.0
                self._ep_boundary_count[w] = 0

        return observation, reward, done, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        """Reset all worlds and sample new tasks."""
        if seed is not None:
            np.random.seed(seed)

        # Call grandparent reset (LBMFluidEnv3D.reset), skip Manta3DLBMEnv's goal logic
        from ..lbm_fluid_env_3d import LBMFluidEnv3D
        LBMFluidEnv3D.reset(self, seed=seed, options=options)

        # Sample new tasks for all worlds
        if self._command_sequence is None:
            self._sample_tasks()
        else:
            # In command sequence mode, reset to first command
            self._command_index = 0
            self._command_step_counter = 0
            if self._command_sequence:
                task_id = TASK_NAMES.index(self._command_sequence[0]["task"])
                self._task_ids[:] = task_id
                self._update_task_ids_wp()
                if self.use_direction_dist_tasks:
                    self._sample_direction_distance_goals()

        self.goals_reached[:] = 0
        self._goal_reached_buffer.zero_()
        self._steps_since_switch[:] = 0
        self._time_val[:] = 0.0  # Reset frequency-domain time
        self._prev_ctrl.fill(0.0)
        self._prev_ctrl_valid[:] = False
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        # Reset episode accumulators
        self._ep_reward_sum[:] = 0.0
        self._ep_step_count[:] = 0
        self._ep_v_forward_sum[:] = 0.0
        self._ep_v_lateral_sum[:] = 0.0
        self._ep_v_vertical_sum[:] = 0.0
        self._ep_yaw_rate_sum[:] = 0.0
        self._ep_boundary_count[:] = 0

        observation = self._get_obs()
        if self.temporal_stack_obs:
            observation = np.concatenate((observation, observation), axis=1)
        return observation

    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        """Reset specific worlds and re-sample their tasks."""
        from ..lbm_fluid_env_3d import LBMFluidEnv3D
        LBMFluidEnv3D.partial_reset(self, reset_mask)

        if not np.any(reset_mask):
            return self._get_obs()

        # Re-sample tasks for reset worlds
        if self._command_sequence is None:
            self._sample_tasks(reset_mask)
        elif self.use_direction_dist_tasks:
            self._sample_direction_distance_goals(reset_mask)

        self.goals_reached[reset_mask] = 0
        self._steps_since_switch[reset_mask] = 0
        self._time_val[reset_mask] = 0.0  # Reset frequency-domain time for reset worlds
        self._prev_ctrl[reset_mask] = 0.0
        self._prev_ctrl_valid[reset_mask] = False
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        # Reset episode accumulators for reset worlds
        self._ep_reward_sum[reset_mask] = 0.0
        self._ep_step_count[reset_mask] = 0
        self._ep_v_forward_sum[reset_mask] = 0.0
        self._ep_v_lateral_sum[reset_mask] = 0.0
        self._ep_v_vertical_sum[reset_mask] = 0.0
        self._ep_yaw_rate_sum[reset_mask] = 0.0
        self._ep_boundary_count[reset_mask] = 0

        observation = self._get_obs()
        if self.temporal_stack_obs:
            observation = np.concatenate((observation, observation), axis=1)
        return observation

    def get_current_task(self, world_idx: int = 0) -> str:
        """Get current task name for a world."""
        return TASK_NAMES[self._task_ids[world_idx]]

    def get_current_goal(self, world_idx: int = 0) -> Tuple[float, float, float]:
        """Return the active sampled goal when direction+distance mode is enabled."""
        if self.use_direction_dist_tasks:
            goal = self._direction_goal_positions_np[world_idx]
            return (float(goal[0]), float(goal[1]), float(goal[2]))
        return (0.5, 0.5, 0.5)
