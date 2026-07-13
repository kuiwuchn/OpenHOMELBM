
# -*- coding: utf-8 -*-
"""
3D Sea Turtle Multi-Task LBM Environment with optional PD control modes.

Multi-task training for sea turtle locomotion skills.
Inherits from Turtle3DLBMEnv but replaces goal-reaching with velocity-based
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

For smooth motion, joints are grouped by function:
  Group 0: front_flap    (f_FR_flap, f_FL_flap)               — 2 joints → 1 signal
  Group 1: front_rotate  (f_FR_rotate, f_FL_rotate)            — 2 joints → 1 signal
  Group 2: front_sweep   (f_FR_sweep, f_FL_sweep)              — 2 joints → 1 signal
  Group 3: rear_flap     (f_RR_flap, f_RL_flap)               — 2 joints → 1 signal
  Group 4: rear_rotate   (f_RR_rotate, f_RL_rotate)            — 2 joints → 1 signal
  Group 5: tail          (tail_yaw)                             — 1 joint → 1 signal

Total: N_groups=6, K=2 harmonics → network outputs 6 * 2 * 2 = 24 values (A + C)

Observation: base obs (22 + 3*n_joints) + lbm_pos(3) + task one-hot (5) + phase (2)
  n_joints = 11 (all joints observed for full state info)
  Total: 22 + 33 + 3 + 5 + 2 = 65 dims

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

from .turtle_lbm_env_3d import (
    Turtle3DLBMEnv,
    compute_turtle_obs_3d_kernel,
    check_boundary_3d_turtle_kernel,
    check_stability_3d_turtle_kernel,
    apply_instability_penalty_turtle_kernel,
)
from ..lbm_core_3d import HomeFlow3D


# ============== Frequency-Domain PD Control Constants ==============

# Number of harmonics (frequency components)
DEFAULT_K_HARMONICS = 2

# Max frequency (B̄ in the paper). B_j = j * B_BAR / K
DEFAULT_B_BAR = 1.0

# Turtle XML actuator layout (11 actuators — v5 ball-like shoulder):
#   0: pos_f_FR_rotate   (front-right flipper rotation / twist around Y)
#   1: pos_f_FR_flap     (front-right flipper up-down flap around X)
#   2: pos_f_FR_sweep    (front-right flipper fore-aft sweep around Z)
#   3: pos_f_FL_rotate   (front-left flipper rotation / twist around Y)
#   4: pos_f_FL_flap     (front-left flipper up-down flap around X)
#   5: pos_f_FL_sweep    (front-left flipper fore-aft sweep around Z)
#   6: pos_f_RR_rotate   (rear-right flipper rotation / twist)
#   7: pos_f_RR_flap     (rear-right flipper up-down flap)
#   8: pos_f_RL_rotate   (rear-left flipper rotation / twist)
#   9: pos_f_RL_flap     (rear-left flipper up-down flap)
#   10: pos_tail_yaw      (tail yaw)

# Reduced-order joint groups for Turtle (11 joints → 6 groups)
# Grouping rationale:
# - Front flap (main propulsion): left/right flap together for synchronized power stroke
# - Front rotate (attack angle): left/right rotate together for pitch/roll control
# - Front sweep (fore-aft): left/right sweep together for stroke trajectory
# - Rear flap: left/right flap together for auxiliary thrust
# - Rear rotate: left/right rotate together for steering assist
# - Tail: independent yaw for fine directional control
REDUCED_ORDER_GROUPS = [
    ("front_flap",   [1, 4]),     # f_FR_flap + f_FL_flap — primary propulsion
    ("front_rotate", [0, 3]),     # f_FR_rotate + f_FL_rotate — attack angle
    ("front_sweep",  [2, 5]),     # f_FR_sweep + f_FL_sweep — fore-aft stroke
    ("rear_flap",    [7, 9]),     # f_RR_flap + f_RL_flap — auxiliary thrust
    ("rear_rotate",  [6, 8]),     # f_RR_rotate + f_RL_rotate — steering
    ("tail",         [10]),       # tail_yaw — fine direction control
]
N_GROUPS = len(REDUCED_ORDER_GROUPS)  # 6


# ============== Wave Control Constants ==============
# Physics-parameterized wave control for cheloniiform locomotion.
# 5-dim action: [A_flap, omega_flap, A_rot, stroke_asym, flap_asym]
#
# Stroke asymmetry (stroke_asym ∈ [-1, 1]):
#   Controls asymmetric flapping for vertical thrust generation.
#   When stroke_asym != 0, the flap waveform becomes asymmetric:
#     - The "fast" half-cycle is shorter in time but same amplitude → higher velocity
#     - The "slow" half-cycle is longer in time but same amplitude → lower velocity
#   This creates net vertical force because impulse = force × time,
#   and the fast stroke generates more instantaneous force.
#
#   Additionally, the attack angle (rotation) is modulated per half-cycle:
#     - Fast half-cycle: attack angle reduced → pure vertical force
#     - Slow half-cycle: attack angle maintained → minimize opposing force
#
#   stroke_asym > 0  → fast downstroke (ascend: net upward force)
#   stroke_asym < 0  → fast upstroke (descend: net downward force)
#   stroke_asym = 0  → symmetric flapping (no vertical bias)
#
# Fixed defaults for removed parameters:
#   A_sweep    = 0.5   (moderate fore-aft sweep, 90° phase lead)
#   rear_scale = 0.75  (rear flippers follow front at 75%)
#   pitch_bias = 0.0   (no ascend/descend bias)
#
# Actuator index mapping (9 DOFs — sweep commented out in XML):
#   0: f_FR_rotate   1: f_FR_flap
#   2: f_FL_rotate   3: f_FL_flap
#   4: f_RR_rotate   5: f_RR_flap
#   6: f_RL_rotate   7: f_RL_flap
#   8: tail_yaw
# NOTE: sweep actuators are commented out in turtle_3d.xml → 9 DOFs total
WAVE_FR_ROT   = 0     # front-right rotation
WAVE_FR_FLAP  = 1     # front-right flap
WAVE_FL_ROT   = 2     # front-left rotation
WAVE_FL_FLAP  = 3     # front-left flap
WAVE_RR_ROT   = 4     # rear-right rotation
WAVE_RR_FLAP  = 5     # rear-right flap
WAVE_RL_ROT   = 6     # rear-left rotation
WAVE_RL_FLAP  = 7     # rear-left flap
WAVE_TAIL     = 8     # tail yaw

WAVE_OMEGA_MAX      = 3.2    # rad/s (~0.51 Hz) — capped for LBM stability on 200^3 grid
WAVE_FIN_AMP_SCALE  = 0.7    # front flipper amplitude scale (safety margin)
WAVE_ROT_AMP_SCALE  = 0.8    # rotation (attack angle) amplitude scale
WAVE_ROT_SHARPNESS  = 5.0    # tanh sharpness for soft-trapezoidal rotation waveform
WAVE_ASYM_SMOOTH_K  = 20.0   # UNUSED: sigmoid removed, flap is now pure sine
N_WAVE_ACTIONS      = 5      # 5-dim physics-parameterized action

# Fixed defaults for parameters removed from action space
WAVE_DEFAULT_A_SWEEP    = 0.5    # moderate fore-aft sweep
WAVE_DEFAULT_REAR_SCALE = 0.6    # rear flippers at 60% of front
WAVE_DEFAULT_PITCH_BIAS = 0.0    # no ascend/descend bias
WAVE_DEFAULT_ROT_PHASE  = -0.5   # fixed rotation phase (-90° lag, optimal for thrust)
WAVE_DEFAULT_TAIL_BIAS  = 0.0    # fixed tail yaw (no bias)


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


@wp.kernel
def compute_turtle_multitask_reward_kernel(
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
):
    """
    Compute task-conditioned reward for multi-task turtle training.

    Reward design (v2 — improved signal-to-noise):
      1. Task reward: directional reward with higher scale for better gradient.
      2. Upright penalty: gentle, only penalize large deviations.
      3. Off-axis penalty: reduced weight to avoid suppressing exploration.
      4. Smoothness penalty: reduced weight with tighter angular velocity scale.
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

    # Smooth penalty scale for angular velocity (tighter than 1.0 for LBM)
    omega_penalty_scale = float(0.3)

    if task == TASK_FORWARD:
        r_task = w_task * directional_reward(v_forward, target_forward_vel)
        r_upright = -w_roll * (0.50 * upright_roll + 0.50 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.50 * soft_penalty(v_lateral, target_forward_vel)
            + 0.20 * soft_penalty(v_vertical, target_vertical_vel)
            + 0.10 * soft_penalty(yaw_rate, target_yaw_rate)
        )

    elif task == TASK_TURN_LEFT:
        r_task = w_task * (
            0.80 * directional_reward(yaw_rate, target_yaw_rate)
            + 0.20 * positive_reward(v_forward, turn_forward_scale)
        )
        r_upright = -w_roll * (0.10 * upright_roll + 0.40 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.40 * soft_penalty(v_lateral, target_forward_vel)
            + 0.10 * soft_penalty(v_vertical, target_vertical_vel)
        )

    elif task == TASK_TURN_RIGHT:
        r_task = w_task * (
            0.80 * directional_reward(-yaw_rate, target_yaw_rate)
            + 0.20 * positive_reward(v_forward, turn_forward_scale)
        )
        r_upright = -w_roll * (0.10 * upright_roll + 0.40 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.40 * soft_penalty(v_lateral, target_forward_vel)
            + 0.10 * soft_penalty(v_vertical, target_vertical_vel)
        )

    elif task == TASK_ASCEND:
        r_task = w_task * (
            0.80 * directional_reward(v_vertical, target_vertical_vel)
            + 0.20 * positive_reward(v_forward, vertical_forward_scale)
        )
        r_upright = -w_roll * (0.15 * upright_roll + 0.25 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.40 * soft_penalty(v_lateral, target_forward_vel)
            + 0.10 * soft_penalty(yaw_rate, target_yaw_rate)
        )

    elif task == TASK_DESCEND:
        r_task = w_task * (
            0.80 * directional_reward(-v_vertical, target_vertical_vel)
            + 0.20 * positive_reward(v_forward, vertical_forward_scale)
        )
        r_upright = -w_roll * (0.15 * upright_roll + 0.25 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.40 * soft_penalty(v_lateral, target_forward_vel)
            + 0.10 * soft_penalty(yaw_rate, target_yaw_rate)
        )

    if task == TASK_TURN_LEFT or task == TASK_TURN_RIGHT:
        r_smooth = -w_smooth * (
            0.35 * soft_penalty(omega_world[0], omega_penalty_scale)
            + 0.35 * soft_penalty(omega_world[1], omega_penalty_scale)
        )
    else:
        r_smooth = -w_smooth * (
            0.40 * soft_penalty(omega_world[0], omega_penalty_scale)
            + 0.40 * soft_penalty(omega_world[1], omega_penalty_scale)
            + 0.10 * soft_penalty(omega_world[2], omega_penalty_scale)
        )

    rewards_out[world_idx] = r_task + r_upright + r_offaxis + r_smooth


@wp.kernel
def compute_turtle_multitask_obs_kernel(
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
    Compute observation for multi-task turtle environment with phase info.

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
    For 9 joints, n_tasks=5: 22 + 27 + 3 + 5 + 2 = 59 dims
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


class TurtleMultiTaskEnv(Turtle3DLBMEnv):
    """
    Multi-task sea turtle locomotion environment with frequency-domain PD control.

    Instead of goal-reaching, the agent is given a task ID (one-hot in obs)
    and rewarded for performing the corresponding motion primitive:
      forward, turn_left, turn_right, ascend, descend.

    Control: The neural network outputs frequency-domain parameters (A, C)
    which are converted to joint target angles via Fourier synthesis,
    then tracked by MuJoCo's built-in PD controllers.

    Reduced-order: Flipper joints are grouped by function (flap/rotate)
    for smooth, coordinated motion.

    Training: tasks are randomly sampled at reset (and optionally mid-episode).
    Inference: a command sequence can be provided to execute complex maneuvers.
    """

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        root_link: str = 'shell',
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
        # Multi-task specific
        task_switch_interval: int = 0,      # steps between task switches (0 = only at reset)
        enabled_tasks: Optional[List[str]] = None,  # subset of tasks to train on
        # Task reward weights
        reward_w_task: float = 1.0,
        reward_w_roll: float = 0.05,
        reward_w_smooth: float = 0.008,
        reward_w_offaxis: float = 0.01,
        # Reference targets / scales for paper-style task rewards
        target_forward_vel: float = 0.05,
        target_yaw_rate: float = 0.03,
        target_vertical_vel: float = 0.03,
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

        # --- Control mode ---
        self.control_mode = control_mode.lower()
        if self.control_mode not in {"frequency", "direct", "wave"}:
            raise ValueError(f"Unknown control_mode '{control_mode}'. Expected 'frequency', 'direct', or 'wave'.")

        # --- Frequency-domain / direct PD control ---
        self.k_harmonics = k_harmonics
        self.b_bar = b_bar
        self.use_reduced_order = use_reduced_order

        if self.control_mode == "wave":
            # Wave mode uses built-in physics structure, no group abstraction needed
            self.n_ctrl_groups = N_WAVE_ACTIONS  # 5
            self.joint_groups = REDUCED_ORDER_GROUPS  # kept for compatibility
        elif self.control_mode == "frequency" and use_reduced_order:
            self.n_ctrl_groups = N_GROUPS  # 5
            self.joint_groups = REDUCED_ORDER_GROUPS
        else:
            # Direct control always uses per-actuator targets; frequency mode can also disable grouping.
            self.n_ctrl_groups = self.n_actuators  # 9
            self.joint_groups = [
                (f"joint_{i}", [i]) for i in range(self.n_actuators)
            ]

        # Action dimension: N_groups * K * 2 (A + C for each harmonic)
        self.freq_action_dim = self.n_ctrl_groups * k_harmonics * 2

        # Build group-to-actuator mapping (numpy, used in CPU freq→ctrl conversion)
        self._group_actuator_indices = [g[1] for g in self.joint_groups]  # list of lists

        # Extract ctrl_range from MuJoCo model for scaling θ* → ctrl
        self._ctrl_lo = self.mj_model.actuator_ctrlrange[:, 0].copy()  # (9,)
        self._ctrl_hi = self.mj_model.actuator_ctrlrange[:, 1].copy()  # (9,)

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

        # --- Override obs dim: optionally stack [obs_before, obs_after] like fish_2d ---
        self.single_obs_dim = 22 + 3 * self.n_joints + 3 + NUM_TASKS + 2
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

        # Internal buffer always stores a single-frame observation.
        self._obs_buffer = wp.zeros((nworld, self.single_obs_dim), dtype=wp.float32, device=self.device)
        self.observation_space = self._create_observation_space()

        if self.control_mode == "wave":
            self.action_dim = N_WAVE_ACTIONS  # 5
        elif self.control_mode == "frequency":
            self.action_dim = self.freq_action_dim
        else:
            self.action_dim = self.n_actuators

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

        print(f"TurtleMultiTaskEnv initialized:")
        print(f"  Tasks: {[TASK_NAMES[i] for i in self.enabled_task_ids]}")
        print(f"  Task switch interval: {task_switch_interval} (0=only at reset)")
        print(f"  Control mode: {self.control_mode}")
        if self.control_mode == "wave":
            print(f"  Control: wave mode — {N_WAVE_ACTIONS}-dim physics-parameterized action")
            print(f"  Action: [A_flap, omega_flap, A_rot, stroke_asym, flap_asym]")
            print(f"  Fixed: A_sweep={WAVE_DEFAULT_A_SWEEP}, rot_phase={WAVE_DEFAULT_ROT_PHASE}, ")
            print(f"         tail_bias={WAVE_DEFAULT_TAIL_BIAS}, rear_scale={WAVE_DEFAULT_REAR_SCALE}, pitch_bias={WAVE_DEFAULT_PITCH_BIAS}")
        elif self.control_mode == "frequency":
            print(f"  Control: K={k_harmonics} harmonics, B̄={b_bar}, reduced_order={use_reduced_order}")
            print(f"  Groups ({self.n_ctrl_groups}): {[g[0] for g in self.joint_groups]}")
            print(f"  Action dim: {self.freq_action_dim} (frequency params)")
        else:
            print(f"  Control: direct actuator target angles")
            print(f"  Action dim: {self.n_actuators} (direct targets)")
        print(
            f"  Obs dim: {self.obs_dim} "
            f"({'temporal 2x stack' if self.temporal_stack_obs else 'single frame'})"
        )
        print(
            f"  Reward scales: fwd={target_forward_vel}, yaw={target_yaw_rate}, vert={target_vertical_vel}"
        )

    def _create_observation_space(self) -> spaces.Space:
        """Create observation space with task one-hot and phase info appended."""
        n_joints = self.mj_model.njnt - 1  # 9
        single_obs_dim = 22 + 3 * n_joints + 3 + NUM_TASKS + 2  # +3 lbm_pos, +NUM_TASKS task, +2 phase
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

    def _update_task_ids_wp(self):
        """Sync task IDs to Warp."""
        wp.copy(
            self._task_ids_wp,
            wp.array(self._task_ids.astype(np.int32), dtype=wp.int32, device=self.device)
        )

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

    # --- Override core methods ---

    def _get_obs(self) -> np.ndarray:
        """Get observation with task one-hot and phase info."""
        # Sync time to Warp
        wp.copy(
            self._time_val_wp,
            wp.array(self._time_val.astype(np.float32), dtype=wp.float32, device=self.device)
        )
        wp.launch(
            compute_turtle_multitask_obs_kernel,
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
        return obs

    def _compute_reward(self, instability_mask=None) -> np.ndarray:
        """Task-conditioned reward computation."""
        wp.launch(
            compute_turtle_multitask_reward_kernel,
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
            ],
            device=self.device,
        )
        return self._rewards_buffer.numpy()

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Map policy output to MuJoCo actuator targets for the active control mode."""
        action = np.clip(action, -1.0, 1.0)
        if self.control_mode == "direct":
            return action.astype(np.float32, copy=False)
        elif self.control_mode == "wave":
            return self._wave_to_ctrl(action)
        return self._freq_to_ctrl(action)

    def _wave_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """
        Convert 6-dim physics-parameterized wave action to 11-dim MuJoCo ctrl.

        Mirrors turtle_wave_viewer.py params_to_ctrl exactly.

        action layout (5-dim):
        [0] A_flap      : front flipper flap amplitude       [-1, 1]
        [1] omega_flap  : front flipper flap frequency       [-1, 1] → [0, OMEGA_MAX]
        [2] A_rot        : rotation (attack angle) amplitude  [-1, 1]
        [3] stroke_asym : up/down stroke timing asymmetry     [-1, 1]
                            >0 → fast downstroke (ascend: net upward force)
                            <0 → fast upstroke (descend: net downward force)
                            =0 → symmetric flapping (no vertical bias)
        [4] flap_asym   : left/right flap amplitude bias      [-1, 1]
                            >0 → right flipper stronger (turn left)
                            <0 → left flipper stronger (turn right)
                            =0 → symmetric L/R

        Flap waveform (asymmetric sine via phase distortion):
          When stroke_asym=0: flap_signal = A_flap · sin(ω·t)  (pure sine)
          When stroke_asym≠0: phase θ is distorted so that one half-cycle
          of sin is traversed quickly (power stroke) and the other slowly
          (recovery stroke), creating net vertical force.
          Attack angle is also reduced during the power stroke to produce
          pure vertical force instead of forward thrust.

        Fixed parameters (not in action space):
          A_sweep    = 0.5   (moderate fore-aft sweep)
          rear_scale = 0.75  (rear at 75% of front)
          pitch_bias = 0.0   (no ascend/descend bias)
          rot_phase  = -0.5  (fixed -90° lag, optimal for thrust)
          tail_bias  = 0.0   (fixed, no tail yaw bias)

        Rotation waveform (soft trapezoidal):
          rot_signal = tanh(k·sin(ω·t + rot_phase·π)) / tanh(k)
          where k = WAVE_ROT_SHARPNESS (default 8.0).
          This creates wide plateau regions near ±1, so the attack angle
          holds steady during the power/recovery strokes instead of
          continuously varying like a pure sine wave.  k=8 gives ~80%+
          plateau duty cycle, ensuring the attack angle is fully
          established when flap velocity peaks.

        Control mapping to 9 actuators (sweep commented out):
          FR_rot   [0]  = +A_rot·rot_signal
          FR_flap  [1]  = +A_flap·asym_sin(ω·t)
          FL_rot   [2]  = -A_rot·rot_signal              (mirror)
          FL_flap  [3]  = -A_flap·asym_sin(ω·t)            (negate: mirror euler)
          RR_rot   [4]  = +rear_s·A_rot·rot_signal
          RR_flap  [5]  = +rear_s·A_flap·asym_sin(ω·t)
          RL_rot   [6]  = -rear_s·A_rot·rot_signal        (mirror)
          RL_flap  [7]  = +rear_s·A_flap·asym_sin(ω·t)
          tail     [8]  = tail_bias (fixed)                      (DC only)
        """
        nw = action.shape[0]
        ctrl = np.zeros((nw, self.n_actuators), dtype=np.float32)

        # Unpack 5-dim action
        A_flap       = action[:, 0]
        omega_flap_n = action[:, 1]
        A_rot        = action[:, 2]
        stroke_asym  = action[:, 3]
        flap_asym    = action[:, 4]

        # Fixed defaults (removed from action space)
        rot_phase    = WAVE_DEFAULT_ROT_PHASE
        tail_bias    = WAVE_DEFAULT_TAIL_BIAS

        # Fixed defaults
        A_sweep    = WAVE_DEFAULT_A_SWEEP
        rear_s     = WAVE_DEFAULT_REAR_SCALE
        pitch_bias = WAVE_DEFAULT_PITCH_BIAS

        # Physical frequency
        omega = (omega_flap_n + 1.0) * 0.5 * WAVE_OMEGA_MAX  # [0, OMEGA_MAX] rad/s
        t = self._time_val[:nw]

        # Phase offsets
        rot_phase_rad = rot_phase * np.pi  # [-π, π]

        def _write(act_idx, theta_norm):
            """Write normalized [-1,1] value to ctrl, scaling to actuator range."""
            theta_c = np.clip(theta_norm, -1.0, 1.0)
            lo, hi = self._ctrl_lo[act_idx], self._ctrl_hi[act_idx]
            ctrl[:, act_idx] = lo + (theta_c + 1.0) * 0.5 * (hi - lo)

        # ── Front flipper flap (asymmetric waveform for vertical thrust) ──
        # When stroke_asym != 0, we use a phase-distortion technique:
        #   Map linear phase θ ∈ [0, 2π) to a distorted phase φ ∈ [0, 2π)
        #   such that one half-cycle of sin(φ) is traversed quickly (short
        #   time = fast stroke) and the other slowly (long time = slow stroke).
        #
        # The key is to change the TIME each half-cycle occupies, not the
        # amplitude.  We split the period at θ_split instead of at π:
        #   θ ∈ [0, θ_split)        → φ ∈ [0, π)    (first half of sin)
        #   θ ∈ [θ_split, 2π)       → φ ∈ [π, 2π)   (second half of sin)
        # where θ_split = π·(1 + asym_k).
        #
        # When asym_k > 0 (ascend):
        #   θ_split > π → first half (sin>0, upstroke) takes MORE time (slow)
        #                → second half (sin<0, downstroke) takes LESS time (fast)
        #   Fast downstroke → strong upward reaction force
        #   Slow upstroke  → weak downward reaction force
        #   Net result: upward lift
        #
        # When asym_k < 0 (descend): mirror — fast upstroke, slow downstroke
        phase_flap = omega * t
        asym_k = np.clip(stroke_asym * 0.8, -0.95, 0.95)  # safety clamp

        # Wrap phase to [0, 2π)
        theta = np.mod(phase_flap, 2.0 * np.pi)

        # Split point: where the first half-cycle ends in theta-space
        # asym_k=0 → θ_split=π (symmetric); asym_k=0.8 → θ_split=1.8π
        theta_split = np.pi * (1.0 + asym_k)

        # Piecewise linear mapping: θ → φ
        # First half:  θ ∈ [0, θ_split)  →  φ ∈ [0, π)
        # Second half: θ ∈ [θ_split, 2π) →  φ ∈ [π, 2π)
        in_first_half = (theta < theta_split)
        phi = np.where(
            in_first_half,
            theta * np.pi / np.maximum(theta_split, 1e-6),
            np.pi + (theta - theta_split) * np.pi / np.maximum(2.0 * np.pi - theta_split, 1e-6),
        )
        sin_flap = np.sin(phi)
        cos_flap = np.cos(phi)

        # Detect which half-cycle we're in (for attack angle modulation)
        # is_power_stroke: True during the fast half-cycle
        # When asym_k > 0: second half (downstroke, sin<0) is fast = power stroke
        # When asym_k < 0: first half (upstroke, sin>0) is fast = power stroke
        is_power_stroke = np.where(
            asym_k >= 0,
            ~in_first_half,  # asym>0: downstroke (2nd half) is power stroke
            in_first_half,   # asym<0: upstroke (1st half) is power stroke
        )

        right_amp = np.clip((A_flap + flap_asym) * WAVE_FIN_AMP_SCALE, -1.0, 1.0)
        left_amp  = np.clip((A_flap - flap_asym) * WAVE_FIN_AMP_SCALE, -1.0, 1.0)

        _write(WAVE_FR_FLAP,  right_amp * sin_flap + pitch_bias)   # FR flap
        _write(WAVE_FL_FLAP, -(left_amp  * sin_flap + pitch_bias))  # FL flap (negate: mirror euler)

        # ── Front flipper rotation (attack angle) — asymmetric modulation ──
        # Use tanh(k·sin(x))/tanh(k) to create plateau regions near ±1.
        #
        # CRITICAL: rotation phase must use the DISTORTED phase (phi) so that
        # the attack angle stays synchronized with the actual flap motion.
        # Using the undistorted phase (omega*t) would cause AoA to drift out
        # of sync with the flap position during asymmetric strokes.
        #
        # When stroke_asym != 0, modulate attack angle per half-cycle:
        #   Power stroke (fast): ZERO AoA → flipper is flat, pure vertical force
        #   Recovery stroke (slow): FULL AoA → flipper feathers through fluid,
        #     blade-like profile minimizes opposing vertical force
        phase_rot = phi + rot_phase_rad  # Use distorted phase for synchronization
        _k = WAVE_ROT_SHARPNESS
        sin_rot = np.tanh(_k * np.sin(phase_rot)) / np.tanh(_k)

        A_rot_scaled = A_rot * WAVE_ROT_AMP_SCALE
        # Modulate attack angle: ZERO during power stroke, FULL during recovery
        # power_stroke_rot_scale: 0.0 = zero AoA during power stroke
        #                         1.0 = full AoA (symmetric, no modulation)
        abs_asym = np.abs(asym_k)
        # Use (1-abs_asym)^2 for aggressive reduction: at asym_k=0.64 → 0.13
        power_rot_scale = (1.0 - abs_asym) ** 2
        rot_scale = np.where(is_power_stroke, power_rot_scale, 1.0)
        _write(WAVE_FR_ROT,  A_rot_scaled * sin_rot * rot_scale)    # FR rotation
        _write(WAVE_FL_ROT, -A_rot_scaled * sin_rot * rot_scale)    # FL rotation (mirror)

        # ── Front flipper sweep disabled (actuators commented out in XML) ──

        # ── Rear flipper flap (follows front, scaled) ──
        rear_right_amp = rear_s * right_amp
        rear_left_amp  = rear_s * left_amp

        _write(WAVE_RR_FLAP,  rear_right_amp * sin_flap)   # RR flap
        _write(WAVE_RL_FLAP,  rear_left_amp  * sin_flap)   # RL flap (same axis)

        # ── Rear flipper rotation (follows front, scaled, same asymmetric modulation) ──
        rear_rot_amp = rear_s * A_rot_scaled

        _write(WAVE_RR_ROT,  rear_rot_amp * sin_rot * rot_scale)    # RR rotation
        _write(WAVE_RL_ROT, -rear_rot_amp * sin_rot * rot_scale)    # RL rotation (mirror)

        # ── Tail yaw (DC bias only) ──
        _write(WAVE_TAIL, tail_bias)

        return ctrl

    def _freq_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """
        Convert frequency-domain action parameters to MuJoCo ctrl (target angles).

        The neural network outputs (A, C) pairs for each group and harmonic.
        Action layout per group g (K harmonics):
            [A_g0, C_g0, A_g1, C_g1, ..., A_g(K-1), C_g(K-1)]

        We compute:  θ*_g = Σ_{j=0}^{K-1} A_{gj} * sin(π/2 * B_j * t + C_{gj})
        Then scale θ*_g from [-1,1] to each actuator's ctrl_range.

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

        # Advance the phase clock for frequency-domain and wave control modes.
        if self.control_mode in ("frequency", "wave"):
            self._time_val += self._dt

        # Physics simulation
        self._simulation_step()

        # Update step counts
        self.step_counts += 1

        # Check stability
        instability_mask = (
            self._check_numerical_stability()
            if hasattr(self, "enable_stability_check") and self.enable_stability_check
            else np.zeros(self.nworld, dtype=bool)
        )

        # Check termination (boundary + instability)
        self._is_terminated(instability_mask)
        boundary_terminated = self._terminated_buffer.numpy().astype(bool).copy()

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

        # Very small control-effort regularization
        ctrl_center = (0.5 * (self._ctrl_lo + self._ctrl_hi))[None, :]
        ctrl_half_range = np.maximum(0.5 * (self._ctrl_hi - self._ctrl_lo), 1.0e-6)[None, :]
        ctrl_effort = (ctrl - ctrl_center) / ctrl_half_range
        reward += -0.05 * self.mt_reward_w_smooth * np.mean(ctrl_effort * ctrl_effort, axis=1)

        self._prev_ctrl[:] = ctrl
        self._prev_ctrl_valid[:] = True

        # Get final terminated state
        terminated = self._terminated_buffer.numpy().astype(bool)

        # Termination penalty (no goal-reaching termination in multi-task)
        reward[terminated] -= 1.0

        # Final safety check
        reward_nan_mask = np.zeros(self.nworld, dtype=bool)
        if np.any(np.isnan(reward)) or np.any(np.isinf(reward)):
            reward_nan_mask = np.isnan(reward) | np.isinf(reward)
            reward[reward_nan_mask] = -1.0
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

        # Build info
        term_reasons = []
        body_positions = np.zeros((self.nworld, 3), dtype=np.float32)
        for w in range(self.nworld):
            reasons = []
            if boundary_terminated[w]:
                reasons.append("boundary")
            if instability_mask[w]:
                reasons.append("instability")
            if obs_nan_mask[w]:
                reasons.append("obs_nan")
            if reward_nan_mask[w]:
                reasons.append("reward_nan")
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
        info["current_tasks"] = [TASK_NAMES[t] for t in self._task_ids]
        info["task_ids"] = self._task_ids.copy()
        info["boundary_terminated"] = boundary_terminated
        info["instability"] = instability_mask

        # Per-world episode metrics
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

        # Call grandparent reset (LBMFluidEnv3D.reset), skip Turtle3DLBMEnv's goal logic
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
        """Override for compatibility: return center position as dummy goal."""
        return (0.5, 0.5, 0.5)
