"""
3D Eel Multi-Task LBM Environment with optional PD control modes.

Multi-task training for eel/ribbon fish locomotion skills.
Inherits from Eel3DLBMEnv but replaces goal-reaching with velocity-based
task-conditioned rewards.

5 Tasks:
  0: FORWARD    — swim along +Y (body forward)
  1: TURN_LEFT  — yaw left while maintaining gentle forward cruise
  2: TURN_RIGHT — yaw right while maintaining gentle forward cruise
  3: ASCEND     — swim upward (+Z) with slight forward cruise
  4: DESCEND    — swim downward (-Z) with slight forward cruise

=== Control Modes ===

'wave' (recommended, default):
  Physics-parameterized traveling wave — 5-dim action with built-in biomimetic structure:
    action[0]: A         — global wave amplitude       [-1, 1]
    action[1]: omega     — oscillation frequency       [-1, 1]  (negative = reverse)
    action[2]: k_wave    — spatial wave number         [-1, 1]  (how many wavelengths fit body)
    action[3]: head_bias — head DC offset for turning  [-1, 1]  (tapers to zero at tail)
    action[4]: roll      — unified roll of all joints  [-1, 1]

  Joint i target angle (i = 0..10, head→tail):
    s_i       = i / 10                                  (normalized position)
    envelope_i = 0.05 + 0.95 * s_i                      (head ~0, tail large)
    theta_i   = A * envelope_i * sin(omega*pi*t + k_wave*pi*s_i)
              + head_bias * (1 - s_i)                   (bias tapers to 0 at tail)

  This encodes the traveling wave as an inductive bias — the network only needs to
  learn 5 physically meaningful scalars. No group-phase tuning needed.

'direct':
  5-dimensional grouped action (same groups as before, for ablation/comparison):
    Group 0: head_yaw   (joint1_yaw)        — head direction
    Group 1: front_yaw  (joint2~4_yaw)      — front body
    Group 2: mid_yaw    (joint5~7_yaw)      — mid body
    Group 3: tail_yaw   (joint8~11_yaw)     — tail
    Group 4: all_roll   (joint1~11_roll)    — unified roll

'frequency':
  20-dimensional Fourier coefficients (5 groups × 2 harmonics × 2 params).

Observation: base obs (22 + 3*n_joints) + lbm_pos(3) + task one-hot (5) + phase (2)
  n_joints = 22 (all Yaw+Roll joints observed for full state info)
  Total: 22 + 66 + 3 + 5 + 2 = 98 dims

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

from .eel_lbm_env_3d import (
    Eel3DLBMEnv,
    compute_eel_obs_kernel,
    check_boundary_eel_kernel,
    check_stability_eel_kernel,
)
from ..lbm_core_3d import HomeFlow3D

# Shared task definitions and observation kernels are independent of animal geometry.
from ..multitask import (
    compute_multitask_obs_kernel,
    quat_rotate_vec,
    dot_vec3,
    TASK_FORWARD,
    TASK_TURN_LEFT,
    TASK_TURN_RIGHT,
    TASK_ASCEND,
    TASK_DESCEND,
    NUM_TASKS,
    TASK_NAMES,
)
# Eel has its own goal-reaching reward kernel (compute_goal_reward_eel_kernel)
# that uses displacement-based velocity, chord-angle heading, and adaptive roll penalty.


# ============== Eel-specific Exponential Reward Functions ==============
# Exponential saturation versions of reward helpers.
# Original linear forms: value / (|value| + scale) are replaced with
# exponential forms: sign(v)*(1 - exp(-|v|/scale)) for sharper gradient
# near zero and smoother saturation at large values.
# All outputs are normalized to the same range as the originals.

@wp.func
def directional_reward_exp(value: wp.float32, scale: wp.float32) -> wp.float32:
    """Exponential directional reward: sign(v) * (1 - exp(-|v|/scale)).
    Output range: [-1, 1].  Replaces linear: v / (|v| + scale)."""
    abs_v = wp.abs(value)
    safe_scale = scale + 1.0e-6
    sat = 1.0 - wp.exp(-abs_v / safe_scale)
    sign = float(0.0)
    if value > 0.0:
        sign = 1.0
    elif value < 0.0:
        sign = -1.0
    return sign * sat


@wp.func
def positive_reward_exp(value: wp.float32, scale: wp.float32) -> wp.float32:
    """Exponential positive reward: 1 - exp(-max(v,0)/scale).
    Output range: [0, 1].  Replaces linear: max(v,0) / (max(v,0) + scale)."""
    pos = wp.max(value, wp.float32(0.0))
    safe_scale = scale + 1.0e-6
    return 1.0 - wp.exp(-pos / safe_scale)


@wp.func
def soft_penalty_exp(value: wp.float32, scale: wp.float32) -> wp.float32:
    """Exponential soft penalty: 1 - exp(-v^2/s^2).
    Output range: [0, 1].  Replaces quadratic: v^2 / (v^2 + s^2)."""
    v2 = value * value
    s2 = scale * scale + 1.0e-6
    return 1.0 - wp.exp(-v2 / s2)


@wp.func
def cos_heading_exp(cos_val: wp.float32) -> wp.float32:
    """Exponential heading reward: exp(cos_val - 1).
    Output range: [exp(-2), 1] ≈ [0.135, 1].  cos=1 → 1, cos=-1 → exp(-2).
    Normalized to [0, 1]: (exp(cos-1) - exp(-2)) / (1 - exp(-2))."""
    e_neg2 = wp.exp(-2.0)
    return (wp.exp(cos_val - 1.0) - e_neg2) / (1.0 - e_neg2 + 1.0e-6)


@wp.func
def bell_reward_exp(value: wp.float32, target: wp.float32, scale: wp.float32) -> wp.float32:
    """Bell-shaped reward: exp(-((value - target)^2) / scale^2).
    Output range: (0, 1]. Peak = 1 at value == target, decays symmetrically.
    Used when we want the agent to HIT a specific target rather than maximize.
    Prevents the "saturate to extreme" cheat of directional_reward_exp."""
    diff = value - target
    s2 = scale * scale + 1.0e-6
    return wp.exp(-(diff * diff) / s2)


@wp.func
def upright_penalty_exp(body_up_component: wp.float32, scale: wp.float32) -> wp.float32:
    """Exponential upright penalty: 1 - exp(-body_up_component^2 / scale^2).
    Output range: [0, 1].  Replaces quadratic: body_up_component^2."""
    v2 = body_up_component * body_up_component
    s2 = scale * scale + 1.0e-6
    return 1.0 - wp.exp(-v2 / s2)


@wp.func
def quat_mul(aw: wp.float32, ax: wp.float32, ay: wp.float32, az: wp.float32,
            bw: wp.float32, bx: wp.float32, by: wp.float32, bz: wp.float32) -> wp.vec4:
    """Hamilton product of two quaternions (w,x,y,z)."""
    return wp.vec4(
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


@wp.func
def segment_up_penalty(
    qpos: wp.array2d(dtype=wp.float32),
    world_idx: int,
    head_qw: wp.float32, head_qx: wp.float32, head_qy: wp.float32, head_qz: wp.float32,
    mode: int,
) -> wp.float32:
    """Compute whole-body upright penalty by forward-chaining quaternions.

    Walks the 12-segment chain (head + 11 joint pairs) and accumulates
    the orientation of each segment.  For each segment we extract the
    local up vector (body-frame Z rotated to world) and penalize:

      mode == 0  (FORWARD / TURN):
          penalty += up_x^2 + up_y^2   (up should be ≈ [0,0,±1], perpendicular to xy)
      mode == 1  (ASCEND / DESCEND):
          penalty += up_x^2            (up should lie in zy plane, up_x ≈ 0)

    Returns the MEAN penalty over 12 segments, range [0, 2] for mode 0
    or [0, 1] for mode 1.

    Joint layout in qpos (after freejoint [0:7]):
      joint i (i=1..11):  qpos[7 + 2*(i-1)] = yaw_i,  qpos[8 + 2*(i-1)] = roll_i
      yaw axis = (0,0,1) = Z,  roll axis = (0,1,0) = Y
    """
    penalty = float(0.0)

    # Current accumulated quaternion = head orientation
    cw = head_qw
    cx = head_qx
    cy = head_qy
    cz = head_qz

    # --- Segment 1 (head): no joint, orientation = head quat ---
    up1 = quat_rotate_vec(cw, cx, cy, cz, wp.vec3(0.0, 0.0, 1.0))
    if mode == 0:
        penalty += up1[0] * up1[0] + up1[1] * up1[1]
    else:
        penalty += up1[0] * up1[0]

    # --- Segments 2..12: each has yaw + roll joint ---
    # Joint 1: qpos indices 7 (yaw), 8 (roll)
    yaw_1 = qpos[world_idx, 7]
    roll_1 = qpos[world_idx, 8]
    half_y1 = yaw_1 * 0.5
    half_r1 = roll_1 * 0.5
    # quat from yaw (axis Z): (cos(h), 0, 0, sin(h))
    # quat from roll (axis Y): (cos(h), 0, sin(h), 0)
    # Combined: q_yaw * q_roll
    jq1 = quat_mul(wp.cos(half_y1), 0.0, 0.0, wp.sin(half_y1),
                   wp.cos(half_r1), 0.0, wp.sin(half_r1), 0.0)
    q2 = quat_mul(cw, cx, cy, cz, jq1[0], jq1[1], jq1[2], jq1[3])
    cw = q2[0]; cx = q2[1]; cy = q2[2]; cz = q2[3]
    up2 = quat_rotate_vec(cw, cx, cy, cz, wp.vec3(0.0, 0.0, 1.0))
    if mode == 0:
        penalty += up2[0] * up2[0] + up2[1] * up2[1]
    else:
        penalty += up2[0] * up2[0]

    # Joint 2: qpos indices 9 (yaw), 10 (roll)
    yaw_2 = qpos[world_idx, 9]
    roll_2 = qpos[world_idx, 10]
    half_y2 = yaw_2 * 0.5
    half_r2 = roll_2 * 0.5
    jq2 = quat_mul(wp.cos(half_y2), 0.0, 0.0, wp.sin(half_y2),
                   wp.cos(half_r2), 0.0, wp.sin(half_r2), 0.0)
    q3 = quat_mul(cw, cx, cy, cz, jq2[0], jq2[1], jq2[2], jq2[3])
    cw = q3[0]; cx = q3[1]; cy = q3[2]; cz = q3[3]
    up3 = quat_rotate_vec(cw, cx, cy, cz, wp.vec3(0.0, 0.0, 1.0))
    if mode == 0:
        penalty += up3[0] * up3[0] + up3[1] * up3[1]
    else:
        penalty += up3[0] * up3[0]

    # Joint 3: qpos indices 11 (yaw), 12 (roll)
    yaw_3 = qpos[world_idx, 11]
    roll_3 = qpos[world_idx, 12]
    half_y3 = yaw_3 * 0.5
    half_r3 = roll_3 * 0.5
    jq3 = quat_mul(wp.cos(half_y3), 0.0, 0.0, wp.sin(half_y3),
                   wp.cos(half_r3), 0.0, wp.sin(half_r3), 0.0)
    q4 = quat_mul(cw, cx, cy, cz, jq3[0], jq3[1], jq3[2], jq3[3])
    cw = q4[0]; cx = q4[1]; cy = q4[2]; cz = q4[3]
    up4 = quat_rotate_vec(cw, cx, cy, cz, wp.vec3(0.0, 0.0, 1.0))
    if mode == 0:
        penalty += up4[0] * up4[0] + up4[1] * up4[1]
    else:
        penalty += up4[0] * up4[0]

    # Joint 4: qpos indices 13 (yaw), 14 (roll)
    yaw_4 = qpos[world_idx, 13]
    roll_4 = qpos[world_idx, 14]
    half_y4 = yaw_4 * 0.5
    half_r4 = roll_4 * 0.5
    jq4 = quat_mul(wp.cos(half_y4), 0.0, 0.0, wp.sin(half_y4),
                   wp.cos(half_r4), 0.0, wp.sin(half_r4), 0.0)
    q5 = quat_mul(cw, cx, cy, cz, jq4[0], jq4[1], jq4[2], jq4[3])
    cw = q5[0]; cx = q5[1]; cy = q5[2]; cz = q5[3]
    up5 = quat_rotate_vec(cw, cx, cy, cz, wp.vec3(0.0, 0.0, 1.0))
    if mode == 0:
        penalty += up5[0] * up5[0] + up5[1] * up5[1]
    else:
        penalty += up5[0] * up5[0]

    # Joint 5: qpos indices 15 (yaw), 16 (roll)
    yaw_5 = qpos[world_idx, 15]
    roll_5 = qpos[world_idx, 16]
    half_y5 = yaw_5 * 0.5
    half_r5 = roll_5 * 0.5
    jq5 = quat_mul(wp.cos(half_y5), 0.0, 0.0, wp.sin(half_y5),
                   wp.cos(half_r5), 0.0, wp.sin(half_r5), 0.0)
    q6 = quat_mul(cw, cx, cy, cz, jq5[0], jq5[1], jq5[2], jq5[3])
    cw = q6[0]; cx = q6[1]; cy = q6[2]; cz = q6[3]
    up6 = quat_rotate_vec(cw, cx, cy, cz, wp.vec3(0.0, 0.0, 1.0))
    if mode == 0:
        penalty += up6[0] * up6[0] + up6[1] * up6[1]
    else:
        penalty += up6[0] * up6[0]

    # Joint 6: qpos indices 17 (yaw), 18 (roll)
    yaw_6 = qpos[world_idx, 17]
    roll_6 = qpos[world_idx, 18]
    half_y6 = yaw_6 * 0.5
    half_r6 = roll_6 * 0.5
    jq6 = quat_mul(wp.cos(half_y6), 0.0, 0.0, wp.sin(half_y6),
                   wp.cos(half_r6), 0.0, wp.sin(half_r6), 0.0)
    q7 = quat_mul(cw, cx, cy, cz, jq6[0], jq6[1], jq6[2], jq6[3])
    cw = q7[0]; cx = q7[1]; cy = q7[2]; cz = q7[3]
    up7 = quat_rotate_vec(cw, cx, cy, cz, wp.vec3(0.0, 0.0, 1.0))
    if mode == 0:
        penalty += up7[0] * up7[0] + up7[1] * up7[1]
    else:
        penalty += up7[0] * up7[0]

    # Joint 7: qpos indices 19 (yaw), 20 (roll)
    yaw_7 = qpos[world_idx, 19]
    roll_7 = qpos[world_idx, 20]
    half_y7 = yaw_7 * 0.5
    half_r7 = roll_7 * 0.5
    jq7 = quat_mul(wp.cos(half_y7), 0.0, 0.0, wp.sin(half_y7),
                   wp.cos(half_r7), 0.0, wp.sin(half_r7), 0.0)
    q8 = quat_mul(cw, cx, cy, cz, jq7[0], jq7[1], jq7[2], jq7[3])
    cw = q8[0]; cx = q8[1]; cy = q8[2]; cz = q8[3]
    up8 = quat_rotate_vec(cw, cx, cy, cz, wp.vec3(0.0, 0.0, 1.0))
    if mode == 0:
        penalty += up8[0] * up8[0] + up8[1] * up8[1]
    else:
        penalty += up8[0] * up8[0]

    # Joint 8: qpos indices 21 (yaw), 22 (roll)
    yaw_8 = qpos[world_idx, 21]
    roll_8 = qpos[world_idx, 22]
    half_y8 = yaw_8 * 0.5
    half_r8 = roll_8 * 0.5
    jq8 = quat_mul(wp.cos(half_y8), 0.0, 0.0, wp.sin(half_y8),
                   wp.cos(half_r8), 0.0, wp.sin(half_r8), 0.0)
    q9 = quat_mul(cw, cx, cy, cz, jq8[0], jq8[1], jq8[2], jq8[3])
    cw = q9[0]; cx = q9[1]; cy = q9[2]; cz = q9[3]
    up9 = quat_rotate_vec(cw, cx, cy, cz, wp.vec3(0.0, 0.0, 1.0))
    if mode == 0:
        penalty += up9[0] * up9[0] + up9[1] * up9[1]
    else:
        penalty += up9[0] * up9[0]

    # Joint 9: qpos indices 23 (yaw), 24 (roll)
    yaw_9 = qpos[world_idx, 23]
    roll_9 = qpos[world_idx, 24]
    half_y9 = yaw_9 * 0.5
    half_r9 = roll_9 * 0.5
    jq9 = quat_mul(wp.cos(half_y9), 0.0, 0.0, wp.sin(half_y9),
                   wp.cos(half_r9), 0.0, wp.sin(half_r9), 0.0)
    q10 = quat_mul(cw, cx, cy, cz, jq9[0], jq9[1], jq9[2], jq9[3])
    cw = q10[0]; cx = q10[1]; cy = q10[2]; cz = q10[3]
    up10 = quat_rotate_vec(cw, cx, cy, cz, wp.vec3(0.0, 0.0, 1.0))
    if mode == 0:
        penalty += up10[0] * up10[0] + up10[1] * up10[1]
    else:
        penalty += up10[0] * up10[0]

    # Joint 10: qpos indices 25 (yaw), 26 (roll)
    yaw_10 = qpos[world_idx, 25]
    roll_10 = qpos[world_idx, 26]
    half_y10 = yaw_10 * 0.5
    half_r10 = roll_10 * 0.5
    jq10 = quat_mul(wp.cos(half_y10), 0.0, 0.0, wp.sin(half_y10),
                    wp.cos(half_r10), 0.0, wp.sin(half_r10), 0.0)
    q11 = quat_mul(cw, cx, cy, cz, jq10[0], jq10[1], jq10[2], jq10[3])
    cw = q11[0]; cx = q11[1]; cy = q11[2]; cz = q11[3]
    up11 = quat_rotate_vec(cw, cx, cy, cz, wp.vec3(0.0, 0.0, 1.0))
    if mode == 0:
        penalty += up11[0] * up11[0] + up11[1] * up11[1]
    else:
        penalty += up11[0] * up11[0]

    # Joint 11: qpos indices 27 (yaw), 28 (roll)
    yaw_11 = qpos[world_idx, 27]
    roll_11 = qpos[world_idx, 28]
    half_y11 = yaw_11 * 0.5
    half_r11 = roll_11 * 0.5
    jq11 = quat_mul(wp.cos(half_y11), 0.0, 0.0, wp.sin(half_y11),
                    wp.cos(half_r11), 0.0, wp.sin(half_r11), 0.0)
    q12 = quat_mul(cw, cx, cy, cz, jq11[0], jq11[1], jq11[2], jq11[3])
    cw = q12[0]; cx = q12[1]; cy = q12[2]; cz = q12[3]
    up12 = quat_rotate_vec(cw, cx, cy, cz, wp.vec3(0.0, 0.0, 1.0))
    if mode == 0:
        penalty += up12[0] * up12[0] + up12[1] * up12[1]
    else:
        penalty += up12[0] * up12[0]

    return penalty / 12.0


@wp.func
def smooth_penalty_exp(omega: wp.float32, scale: wp.float32) -> wp.float32:
    """Exponential smoothness penalty: 1 - exp(-omega^2 / scale^2).
    Output range: [0, 1].  Replaces quadratic: omega^2."""
    w2 = omega * omega
    s2 = scale * scale + 1.0e-6
    return 1.0 - wp.exp(-w2 / s2)


# ============== Eel-specific Reward Kernel ==============
# Eel uses a barrel-roll strategy for ascend/descend:
#   Roll ~90° to redirect Yaw traveling-wave thrust vertically.
# Key differences from other fish:
#   - ASCEND/DESCEND: Roll penalty = 0 (must allow barrel-roll)
#   - FORWARD: Roll penalty = 0.20 (highest: 8-segment chain is very roll-prone)
#   - 50/50 vertical/forward split for ASCEND/DESCEND (barrel-roll produces mixed thrust)
#   - Very low pitch penalty (0.05): after 90° roll, pitch axis is effectively roll axis
#   - TURN uses head yaw_rate for instantaneous turning signal (unaffected by
#     tail oscillation) PLUS a "chord angle" alignment bonus that rewards the
#     final body orientation without penalizing traveling-wave tail swings.

@wp.kernel
def compute_eel_multitask_reward_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    disp_vel: wp.array2d(dtype=wp.float32),  # (nworld, 3) displacement-based velocity from LBM position diff
    task_ids: wp.array(dtype=wp.int32),
    task_forward_dir: wp.array2d(dtype=wp.float32),  # (nworld, 3) body_forward snapshot at task switch
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
    Eel-specific task-conditioned reward with barrel-roll vertical strategy.

    All directional rewards use body-frame relative signals that are heading-invariant.
    v_forward / v_lateral are projected onto task_forward_dir — an EMA-smoothed
    snapshot of the body forward direction captured at task-switch time (smoothing
    prevents the agent from gaming snapshot timing with a momentary head flick).

    FORWARD: reward body-frame forward velocity + yaw_rate penalty (anti-circling)
             + chord-angle-straightness (prevents slowly-arching-body cheat).
    TURN: reward chord_angle_offset (DC body curvature, immune to asymmetric
          head-swings) + yaw_rate (turn initiation) + low-saturation v_forward
          (anti-spin, but no gradient for cruising faster in old direction,
          preventing the "small-angle drift" cheat).
    ASCEND/DESCEND: reward world-Z velocity (gravity is absolute) + forward cruise.
    ASCEND/DESCEND: Roll penalty is ZERO to allow the eel to barrel-roll ~90°.
    FORWARD: Roll penalty is highest (0.20) to keep the long chain upright.
    """
    world_idx = wp.tid()
    task = task_ids[world_idx]

    qw = qpos[world_idx, 3]
    qx = qpos[world_idx, 4]
    qy = qpos[world_idx, 5]
    qz = qpos[world_idx, 6]

    # Instantaneous body axes (used for roll/upright penalty and alignment)
    body_up = quat_rotate_vec(qw, qx, qy, qz, wp.vec3(0.0, 0.0, 1.0))
    body_forward = quat_rotate_vec(qw, qx, qy, qz, wp.vec3(0.0, 1.0, 0.0))

    # Task-frame axes: snapshot of body_forward/body_right at task-switch time.
    # These are stable reference directions, immune to head oscillation noise.
    snap_fwd = wp.vec3(
        task_forward_dir[world_idx, 0],
        task_forward_dir[world_idx, 1],
        task_forward_dir[world_idx, 2],
    )
    # Derive snap_right as cross(snap_fwd, world_up), then normalize.
    # world_up = (0, 0, 1); cross(fwd, up) = (fwd.y, -fwd.x, 0)
    snap_right_raw = wp.vec3(snap_fwd[1], -snap_fwd[0], 0.0)
    snap_right_len = wp.sqrt(snap_right_raw[0] * snap_right_raw[0] + snap_right_raw[1] * snap_right_raw[1] + 1.0e-6)
    snap_right = wp.vec3(snap_right_raw[0] / snap_right_len, snap_right_raw[1] / snap_right_len, 0.0)

    # Displacement-based velocity from LBM position difference (whole-body, smooth)
    dv = wp.vec3(
        disp_vel[world_idx, 0],
        disp_vel[world_idx, 1],
        disp_vel[world_idx, 2],
    )

    # Head (root freejoint) instantaneous angular velocity (still useful for yaw_rate)
    omega_body = wp.vec3(
        qvel[world_idx, 3],
        qvel[world_idx, 4],
        qvel[world_idx, 5],
    )
    omega_world = quat_rotate_vec(qw, qx, qy, qz, omega_body)

    # Project displacement velocity onto task-frame snapshot axes
    v_forward = dot_vec3(dv, snap_fwd)
    v_lateral = dot_vec3(dv, snap_right)
    v_vertical = dv[2]
    yaw_rate = omega_world[2]

    # --- Head chord angle: HEAD-SEGMENT body curvature (for turn alignment) ---
    # We measure the DC turning bias from the first 4 yaw joints (head segment,
    # s=0.0..0.3) using near-uniform weights so that NO single joint can
    # dominate the signal. Rationale:
    #
    # 1. Physical semantics of "turning": in anguilliform swimming, the
    #    SWIMMING DIRECTION is set by where the HEAD points. Tail-side body
    #    curvature only redirects thrust; it does NOT rotate the heading.
    #    So we restrict to head joints only.
    #
    # 2. Only 2 head joints (joint1, joint2): using fewer joints minimizes
    #    interference from the traveling wave. Joint1 envelope=0.05, joint2
    #    envelope=0.145 — both tiny, so AC leakage is negligible.
    #    This prevents the tail-curl problem: with 4 joints, the wave
    #    oscillation destabilized the bell reward, making the policy prefer
    #    to stop tail-beating to keep chord stable. With only 2 joints,
    #    tail-beating barely affects chord, so the policy can freely swing
    #    the tail while maintaining chord reward.
    #
    # qpos layout: [0:3]=pos, [3:7]=quat, [7::2]=yaw_joint_angles
    # Head yaw joints at qpos indices: 7, 9 (joints 1..2)
    # Weights w_i = 1/(1-s_i) for s_i ∈ {0.0, 0.1}:
    #   w0=1.000, w1=1.111
    # Sum = 2.111. DC part = head_bias * (1.0*1.0 + 1.111*0.9) / 2.111
    #            = head_bias * 2.0 / 2.111 ≈ 0.948 * head_bias.
    # At head_bias=0.25, chord_angle_offset ≈ 0.237 rad → target = 0.24.
    weighted_sum = float(0.0)
    weighted_sum += 1.000 * qpos[world_idx, 7]    # joint1_yaw (s=0.0)
    weighted_sum += 1.111 * qpos[world_idx, 9]    # joint2_yaw (s=0.1)
    chord_angle_offset = weighted_sum / 2.111

    # All tasks use body-frame relative signals — no fixed world-axis heading needed.

    r_task = float(0.0)
    r_upright = float(0.0)
    r_offaxis = float(0.0)

    # Chord-angle target: HEAD-segment DC curvature magnitude for a "committed
    # turn". With head_bias=0.25 rad and 2-joint weights (joints 1..2),
    # the steady-state chord_angle_offset ≈ 0.948 * 0.25 ≈ 0.24 rad.
    target_chord_angle = wp.float32(0.24)

    # --- Whole-body upright penalty ---
    # Forward-chain quaternions through all 12 segments and penalize
    # each segment's up-vector deviation from the desired plane:
    #   FORWARD/TURN (mode=0): up should be ⊥ xy-plane → penalize up_x² + up_y²
    #   ASCEND/DESCEND (mode=1): up should be ⊥ zy-plane → penalize up_x²
    upright_mode = int(0)
    if task == TASK_ASCEND or task == TASK_DESCEND:
        upright_mode = 1
    whole_body_upright_pen = segment_up_penalty(
        qpos, world_idx, qw, qx, qy, qz, upright_mode)

    if task == TASK_FORWARD:

        cos_fwd = dot_vec3(body_forward, snap_fwd)
        r_task = w_task * (
            0.50 * directional_reward_exp(v_forward, target_forward_vel)
            + 0.50 * cos_heading_exp(cos_fwd)
        )
        # Whole-body upright: all segments should have up ≈ [0,0,±1]
        r_upright = -w_roll * 1.00 * soft_penalty_exp(
            wp.sqrt(whole_body_upright_pen + 1.0e-6), wp.float32(0.15))
        r_offaxis = -w_offaxis * (
            0.60 * soft_penalty_exp(v_lateral, target_forward_vel)
            + 0.30 * soft_penalty_exp(v_vertical, target_vertical_vel)
        )

    elif task == TASK_TURN_LEFT:
        snap_left = wp.vec3(-snap_right[0], -snap_right[1], 0.0)
        heading_turned = dot_vec3(body_forward, snap_left)

        r_task = w_task * (
            0.15 * bell_reward_exp(chord_angle_offset, target_chord_angle, wp.float32(0.15))
            + 0.35 * positive_reward_exp(heading_turned, wp.float32(0.7))
        )
        r_upright = -w_roll * 1.00 * soft_penalty_exp(
            wp.sqrt(whole_body_upright_pen + 1.0e-6), wp.float32(0.15))
        r_offaxis = -w_offaxis * (
            0.40 * soft_penalty_exp(v_vertical, target_vertical_vel)
        )

    elif task == TASK_TURN_RIGHT:
        heading_turned = dot_vec3(body_forward, snap_right)

        r_task = w_task * (
            0.15 * bell_reward_exp(chord_angle_offset, -target_chord_angle, wp.float32(0.15))
            + 0.35 * positive_reward_exp(heading_turned, wp.float32(0.7))
        )
        r_upright = -w_roll * 1.00 * soft_penalty_exp(
            wp.sqrt(whole_body_upright_pen + 1.0e-6), wp.float32(0.15))
        r_offaxis = -w_offaxis * (
            0.40 * soft_penalty_exp(v_vertical, target_vertical_vel)
        )

    elif task == TASK_ASCEND:

        r_task = w_task * (
            0.50 * directional_reward_exp(v_vertical, target_vertical_vel)
            + 0.50 * positive_reward_exp(v_forward, target_forward_vel)
        )
        # Whole-body upright: all segments should have up_x ≈ 0 (in zy plane)
        r_upright = -w_roll * 0.30 * soft_penalty_exp(
            wp.sqrt(whole_body_upright_pen + 1.0e-6), wp.float32(0.30))
        r_offaxis = -w_offaxis * (
            0.45 * soft_penalty_exp(v_lateral, target_forward_vel)
        )

    elif task == TASK_DESCEND:

        r_task = w_task * (
            0.50 * directional_reward_exp(-v_vertical, target_vertical_vel)
            + 0.50 * positive_reward_exp(v_forward, target_forward_vel)
        )
        r_upright = -w_roll * 0.30 * soft_penalty_exp(
            wp.sqrt(whole_body_upright_pen + 1.0e-6), wp.float32(0.30))
        r_offaxis = -w_offaxis * (
            0.45 * soft_penalty_exp(v_lateral, target_forward_vel)
        )

    # Angular velocity smoothness penalty (exponential form)
    # scale = 1.0 rad/s: penalty saturates around |omega| ~ 1 rad/s
    if task == TASK_TURN_LEFT or task == TASK_TURN_RIGHT:
        # Allow yaw angular velocity (omega_world[2]) during turns
        r_smooth = -w_smooth * (
            0.35 * smooth_penalty_exp(omega_world[0], 1.0)
            + 0.35 * smooth_penalty_exp(omega_world[1], 1.0)
        )
    elif task == TASK_ASCEND or task == TASK_DESCEND:
        # During barrel-roll, allow roll angular velocity (omega_world[1])
        r_smooth = -w_smooth * (
            0.30 * smooth_penalty_exp(omega_world[0], 1.0)
            + 0.10 * smooth_penalty_exp(omega_world[1], 1.0)
            + 0.10 * smooth_penalty_exp(omega_world[2], 1.0)
        )
    else:
        r_smooth = -w_smooth * (
            0.50 * smooth_penalty_exp(omega_world[0], 1.0)
            + 0.50 * smooth_penalty_exp(omega_world[1], 1.0)
            + 0.15 * smooth_penalty_exp(omega_world[2], 1.0)
        )

    # --- Roll joint jitter penalty ---
    # Penalize high-frequency small-amplitude roll joint angular velocities.
    # The agent exploits rapid roll oscillations across all joints to "cheat"
    # rewards without producing meaningful locomotion. We sum |qvel| of all
    # 11 roll joints and apply soft_penalty_exp with a tight scale (0.5 rad/s)
    # so even small jitter is penalized.
    # Roll joint qvel indices: 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27
    roll_jitter = (wp.abs(qvel[world_idx, 7])
                   + wp.abs(qvel[world_idx, 9])
                   + wp.abs(qvel[world_idx, 11])
                   + wp.abs(qvel[world_idx, 13])
                   + wp.abs(qvel[world_idx, 15])
                   + wp.abs(qvel[world_idx, 17])
                   + wp.abs(qvel[world_idx, 19])
                   + wp.abs(qvel[world_idx, 21])
                   + wp.abs(qvel[world_idx, 23])
                   + wp.abs(qvel[world_idx, 25])
                   + wp.abs(qvel[world_idx, 27]))
    if task == TASK_ASCEND or task == TASK_DESCEND:
        # Barrel-roll is a valid strategy: lower penalty weight
        r_roll_jitter = -w_smooth * 0.15 * soft_penalty_exp(roll_jitter, wp.float32(3.0))
    else:
        # FORWARD / TURN: roll jitter is never useful
        r_roll_jitter = -w_smooth * 0.30 * soft_penalty_exp(roll_jitter, wp.float32(2.0))

    rewards_out[world_idx] = r_task + r_upright + r_offaxis + r_smooth + r_roll_jitter


# ============== Eel-specific Goal Reward Kernel (direction-distance) ==============
# This eel-specific goal reward kernel:
#   1. Uses displacement-based velocity (disp_vel) instead of qvel for forward reward
#      (immune to head oscillation noise in the long eel chain)
#   2. Uses chord-angle heading (inverse-envelope weighted yaw joints) instead of
#      simple body_forward for heading alignment (captures true body orientation)
#   3. Applies adaptive roll penalty that relaxes when the goal has a large vertical
#      component (allowing barrel-roll for ascend/descend)
#   4. Includes angular velocity smoothness penalty (r_smooth)

@wp.kernel
def compute_goal_reward_eel_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    prev_dist: wp.array(dtype=wp.float32),
    rewards_out: wp.array(dtype=wp.float32),
    current_dist_out: wp.array(dtype=wp.float32),
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    disp_vel: wp.array2d(dtype=wp.float32),  # (nworld, 3) displacement-based velocity
    nx: float,
    ny: float,
    nz: float,
    w_dist: float,
    w_roll: float,
    w_heading: float,
    w_forward: float,
    w_smooth: float,
    w_offaxis: float,
    target_forward_vel: float,
    target_yaw_rate: float,
    target_vertical_vel: float,
):
    """
    Eel-specific goal reward for direction-distance navigation.
    Aligned with compute_eel_multitask_reward_kernel design:
      - Uses all 11 yaw joints for chord angle (sum=44.632)
      - Uses exponential reward/penalty helpers (directional_reward_exp, etc.)
      - Infers implicit task from goal direction to select per-task weights
      - Decomposes velocity into body-frame components for off-axis penalty

    Components:
      1. r_dist:    distance improvement (closer to goal)
      2. r_heading: chord-angle alignment with goal direction (cos_heading_exp)
      3. r_forward: displacement velocity projected onto goal (directional_reward_exp)
      4. r_upright: adaptive roll penalty (relaxed for vertical goals, per multitask)
      5. r_smooth:  angular velocity smoothness penalty (per-task weights)
      6. r_offaxis: off-axis velocity penalty (body-frame decomposition)

    Coordinate convention: X=lateral, Y=forward, Z=up
    Quaternion convention (MuJoCo): qpos[3:7] = (w, x, y, z)
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
    dist_improvement = wp.clamp(dist_improvement_raw, -0.05, 0.05)
    r_dist = w_dist * dist_improvement
    current_dist_out[world_idx] = current_dist

    # --- Quaternion ---
    qw = qpos[world_idx, 3]
    qx = qpos[world_idx, 4]
    qy = qpos[world_idx, 5]
    qz_q = qpos[world_idx, 6]

    # Body axes from quaternion
    body_right = quat_rotate_vec(qw, qx, qy, qz_q, wp.vec3(1.0, 0.0, 0.0))
    body_forward = quat_rotate_vec(qw, qx, qy, qz_q, wp.vec3(0.0, 1.0, 0.0))
    body_up = quat_rotate_vec(qw, qx, qy, qz_q, wp.vec3(0.0, 0.0, 1.0))

    # --- Direction from body to goal ---
    to_goal_x = -dx
    to_goal_y = -dy
    to_goal_z = -dz
    to_goal_len = wp.sqrt(to_goal_x * to_goal_x + to_goal_y * to_goal_y + to_goal_z * to_goal_z)

    # --- Infer implicit task from goal direction ---
    # vert_frac: fraction of goal direction that is vertical
    # horiz_frac: fraction that is horizontal
    # Used to blend between forward-like and vertical-like reward weights
    vert_frac = float(0.0)
    if to_goal_len > 1.0e-6:
        vert_frac = wp.abs(to_goal_z) / to_goal_len
    horiz_frac = 1.0 - vert_frac
    # is_vertical: true when goal is mostly vertical (vert_frac > 0.5)
    # alpha: 1.0 for horizontal goals, ~0.05 for pure vertical goals
    alpha = wp.max(horiz_frac, 0.05)

    # --- Displacement-based velocity (immune to head oscillation) ---
    dv = wp.vec3(
        disp_vel[world_idx, 0],
        disp_vel[world_idx, 1],
        disp_vel[world_idx, 2],
    )

    # Body-frame velocity decomposition (same as multitask kernel)
    v_forward = dot_vec3(dv, body_forward)
    v_lateral = dot_vec3(dv, body_right)
    v_vertical = dv[2]

    # Head angular velocity in body frame -> world frame
    omega_body = wp.vec3(
        qvel[world_idx, 3],
        qvel[world_idx, 4],
        qvel[world_idx, 5],
    )
    omega_world = quat_rotate_vec(qw, qx, qy, qz_q, omega_body)
    yaw_rate = omega_world[2]

    # --- 2. Heading reward: chord-angle alignment with goal direction ---
    # Chord angle: inverse-envelope weighted sum of ALL 11 yaw joint angles
    # (consistent with compute_eel_multitask_reward_kernel).
    # qpos layout: [0:3]=pos, [3:7]=quat, [7::2]=yaw_joint_angles
    # 11 yaw joints at qpos indices: 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27
    # Precomputed weights (1/envelope): sum = 44.632
    weighted_sum = float(0.0)
    weighted_sum += 20.000 * qpos[world_idx, 7]   # joint1_yaw  (s=0.0, env=0.05)
    weighted_sum += 6.897 * qpos[world_idx, 9]    # joint2_yaw  (s=0.1, env=0.145)
    weighted_sum += 4.167 * qpos[world_idx, 11]   # joint3_yaw  (s=0.2, env=0.24)
    weighted_sum += 2.985 * qpos[world_idx, 13]   # joint4_yaw  (s=0.3, env=0.335)
    weighted_sum += 2.326 * qpos[world_idx, 15]   # joint5_yaw  (s=0.4, env=0.43)
    weighted_sum += 1.905 * qpos[world_idx, 17]   # joint6_yaw  (s=0.5, env=0.525)
    weighted_sum += 1.613 * qpos[world_idx, 19]   # joint7_yaw  (s=0.6, env=0.62)
    weighted_sum += 1.399 * qpos[world_idx, 21]   # joint8_yaw  (s=0.7, env=0.715)
    weighted_sum += 1.235 * qpos[world_idx, 23]   # joint9_yaw  (s=0.8, env=0.81)
    weighted_sum += 1.105 * qpos[world_idx, 25]   # joint10_yaw (s=0.9, env=0.905)
    weighted_sum += 1.000 * qpos[world_idx, 27]   # joint11_yaw (s=1.0, env=1.0)
    # Normalize by sum of weights (44.632)
    chord_angle = weighted_sum / 44.632

    # Build chord heading vector: rotate body_forward by chord_angle around world Z.
    cos_ca = wp.cos(chord_angle)
    sin_ca = wp.sin(chord_angle)
    chord_fwd_x = body_forward[0] * cos_ca - body_forward[1] * sin_ca
    chord_fwd_y = body_forward[0] * sin_ca + body_forward[1] * cos_ca
    chord_fwd_z = body_forward[2]
    # Normalize chord_fwd
    chord_len = wp.sqrt(chord_fwd_x * chord_fwd_x + chord_fwd_y * chord_fwd_y + chord_fwd_z * chord_fwd_z)
    if chord_len > 1.0e-6:
        inv_cl = 1.0 / chord_len
        chord_fwd_x = chord_fwd_x * inv_cl
        chord_fwd_y = chord_fwd_y * inv_cl
        chord_fwd_z = chord_fwd_z * inv_cl

    # cos(angle) between chord heading and to_goal direction
    # Use cos_heading_exp (exponential form, consistent with multitask kernel)
    r_heading = float(0.0)
    if to_goal_len > 1.0e-6:
        inv_gl = 1.0 / to_goal_len
        cos_angle = (chord_fwd_x * to_goal_x + chord_fwd_y * to_goal_y + chord_fwd_z * to_goal_z) * inv_gl
        cos_30deg = 0.8660
        if cos_angle >= cos_30deg:
            # Within ±30°: exponential heading reward (same as multitask FORWARD)
            heading_base = w_heading * cos_heading_exp(cos_angle)
            if dist_improvement > 0.0:
                r_heading = heading_base * (1.0 + 0.5 * dist_improvement)
            else:
                r_heading = heading_base
        else:
            # Beyond ±30°: soft linear penalty (reduced to 30% of w_heading)
            r_heading = -0.01 * w_heading * (cos_30deg - cos_angle) / (1.0 + cos_30deg)

    # --- 3. Forward velocity reward: disp_vel projected onto goal direction ---
    # Use directional_reward_exp (exponential form, consistent with multitask kernel)
    r_forward = float(0.0)
    if to_goal_len > 1.0e-6:
        inv_gl2 = 1.0 / to_goal_len
        v_goal = (dv[0] * to_goal_x + dv[1] * to_goal_y + dv[2] * to_goal_z) * inv_gl2
        # Blend forward scale: horizontal goals use full target, vertical goals use half
        fwd_scale = horiz_frac * target_forward_vel + vert_frac * 0.5 * target_forward_vel
        r_forward = w_forward * directional_reward_exp(v_goal, fwd_scale)

    # --- 4. Whole-body upright penalty ---
    # Forward-chain quaternions through all 12 segments.
    # Horizontal goals (alpha~1): mode=0, penalize up_x²+up_y² (up ⊥ xy)
    # Vertical goals (alpha~0):  mode=1, penalize up_x² (up ⊥ zy)
    # We blend the two modes via alpha for smooth transition.
    pen_horiz = segment_up_penalty(qpos, world_idx, qw, qx, qy, qz_q, 0)  # mode 0
    pen_vert  = segment_up_penalty(qpos, world_idx, qw, qx, qy, qz_q, 1)  # mode 1
    blended_pen = horiz_frac * pen_horiz + vert_frac * pen_vert
    # Weight: horizontal goals get strong penalty, vertical goals get lighter
    upright_coeff = alpha * 1.00 + (1.0 - alpha) * 0.30
    upright_scale = alpha * 0.15 + (1.0 - alpha) * 0.30
    r_upright = -w_roll * upright_coeff * soft_penalty_exp(
        wp.sqrt(blended_pen + 1.0e-6), wp.float32(upright_scale))

    # --- 5. Angular velocity smoothness penalty ---
    # Consistent with multitask kernel per-task smooth weights:
    #   FORWARD: 0.50*wx + 0.50*wy + 0.15*wz
    #   TURN:    0.35*wx + 0.35*wy + 0.0*wz
    #   ASCEND/DESCEND: 0.30*wx + 0.10*wy + 0.10*wz
    # Blend using alpha:
    smooth_wx = alpha * 0.50 + (1.0 - alpha) * 0.30
    smooth_wy = alpha * 0.50 + (1.0 - alpha) * 0.10
    smooth_wz = alpha * 0.15 + (1.0 - alpha) * 0.10
    r_smooth = -w_smooth * (
        smooth_wx * smooth_penalty_exp(omega_world[0], 1.0)
        + smooth_wy * smooth_penalty_exp(omega_world[1], 1.0)
        + smooth_wz * smooth_penalty_exp(omega_world[2], 1.0)
    )

    # --- 5b. Roll joint jitter penalty ---
    # Same as multitask kernel: penalize sum of |qvel| for all 11 roll joints.
    roll_jitter = (wp.abs(qvel[world_idx, 7])
                   + wp.abs(qvel[world_idx, 9])
                   + wp.abs(qvel[world_idx, 11])
                   + wp.abs(qvel[world_idx, 13])
                   + wp.abs(qvel[world_idx, 15])
                   + wp.abs(qvel[world_idx, 17])
                   + wp.abs(qvel[world_idx, 19])
                   + wp.abs(qvel[world_idx, 21])
                   + wp.abs(qvel[world_idx, 23])
                   + wp.abs(qvel[world_idx, 25])
                   + wp.abs(qvel[world_idx, 27]))
    # Blend: horizontal goals (alpha~1) use full penalty, vertical goals use reduced
    roll_jitter_coeff = alpha * 0.30 + (1.0 - alpha) * 0.15
    roll_jitter_scale = alpha * 2.0 + (1.0 - alpha) * 3.0
    r_roll_jitter = -w_smooth * roll_jitter_coeff * soft_penalty_exp(roll_jitter, wp.float32(roll_jitter_scale))

    # --- 6. Off-axis penalty ---
    # Consistent with multitask kernel: decompose into lateral + vertical/yaw_rate
    # FORWARD: 0.60*lateral + 0.30*vertical
    # TURN:    0.45*lateral + 0.15*vertical
    # ASCEND/DESCEND: 0.45*lateral + 0.15*yaw_rate
    # Blend using alpha:
    offaxis_lat_coeff = alpha * 0.60 + (1.0 - alpha) * 0.45
    offaxis_sec_coeff = alpha * 0.30 + (1.0 - alpha) * 0.15
    r_offaxis = float(0.0)
    if w_offaxis > 0.0:
        # Lateral velocity penalty (always penalized)
        lat_pen = soft_penalty_exp(v_lateral, target_forward_vel)
        # Secondary penalty: vertical for horizontal goals, yaw_rate for vertical goals
        sec_pen = horiz_frac * soft_penalty_exp(v_vertical, target_vertical_vel) \
                + vert_frac * soft_penalty_exp(yaw_rate, target_yaw_rate)
        r_offaxis = -w_offaxis * (
            offaxis_lat_coeff * lat_pen
            + offaxis_sec_coeff * sec_pen
        )

    # --- Total reward ---
    rewards_out[world_idx] = r_dist + r_heading + r_forward + r_upright + r_smooth + r_roll_jitter + r_offaxis


@wp.func
def traveling_wave_bonus(k_abs: wp.float32) -> wp.float32:
    """
    Reward for having a non-zero spatial wave number (k_wave).
    Encourages the network to learn proper traveling wave propagation
    instead of collapsing to k_wave=0 (all joints in phase).

    Returns a value in [0, 1]: saturates around |k| ~ 0.5.
    """
    return k_abs / (k_abs + 0.3 + 1.0e-6)


# ============== Frequency-Domain PD Control Constants ==============

# Number of harmonics (frequency components)
DEFAULT_K_HARMONICS = 2

# Max frequency (B̄). B_j = j * B_BAR / K
DEFAULT_B_BAR = 1.0

# Eel XML actuator layout (22 actuators, 11 pairs interleaved yaw/roll):
#   0: pos1_yaw   (joint1_yaw,  seg1-seg2)   HEAD
#   1: pos1_roll  (joint1_roll, seg1-seg2)
#   2: pos2_yaw   (joint2_yaw,  seg2-seg3)   FRONT
#   3: pos2_roll  (joint2_roll, seg2-seg3)
#   4: pos3_yaw   (joint3_yaw,  seg3-seg4)
#   5: pos3_roll  (joint3_roll, seg3-seg4)
#   6: pos4_yaw   (joint4_yaw,  seg4-seg5)
#   7: pos4_roll  (joint4_roll, seg4-seg5)
#   8: pos5_yaw   (joint5_yaw,  seg5-seg6)   MID
#   9: pos5_roll  (joint5_roll, seg5-seg6)
#  10: pos6_yaw   (joint6_yaw,  seg6-seg7)
#  11: pos6_roll  (joint6_roll, seg6-seg7)
#  12: pos7_yaw   (joint7_yaw,  seg7-seg8)
#  13: pos7_roll  (joint7_roll, seg7-seg8)
#  14: pos8_yaw   (joint8_yaw,  seg8-seg9)   TAIL
#  15: pos8_roll  (joint8_roll, seg8-seg9)
#  16: pos9_yaw   (joint9_yaw,  seg9-seg10)
#  17: pos9_roll  (joint9_roll, seg9-seg10)
#  18: pos10_yaw  (joint10_yaw, seg10-seg11)
#  19: pos10_roll (joint10_roll,seg10-seg11)
#  20: pos11_yaw  (joint11_yaw, seg11-seg12)
#  21: pos11_roll (joint11_roll,seg11-seg12)

# Yaw actuator indices (even): 0, 2, 4, ..., 20
YAW_ACTUATOR_INDICES = list(range(0, 22, 2))   # 11 yaw actuators

# Roll actuator indices (odd): 1, 3, 5, ..., 21
ROLL_ACTUATOR_INDICES = list(range(1, 22, 2))   # 11 roll actuators

# All Roll joints are active (unified group), no locked joints
ACTIVE_ROLL_ACTUATOR_INDICES = ROLL_ACTUATOR_INDICES  # all 11 roll actuators
LOCKED_ROLL_ACTUATOR_INDICES = []                     # none locked

# ============== Biomimetic Grouped Control ==============
# Grouping rationale:
# - Yaw split into 4 spatial groups for traveling-wave phase gradient
#   (head → front → mid → tail, each group receives one control signal)
# - All Roll joints unified into 1 group to prevent chaotic twisting
#   (the entire body rolls together as a coordinated unit)
#
# This yields 5 groups = 5-dim direct action, or 5*K*2 frequency params.

REDUCED_ORDER_GROUPS = [
    ("head_yaw",   [0]),                    # joint1_yaw — head direction control
    ("front_yaw",  [2, 4, 6]),              # joint2~4_yaw — front body wave
    ("mid_yaw",    [8, 10, 12]),            # joint5~7_yaw — mid body wave
    ("tail_yaw",   [14, 16, 18, 20]),       # joint8~11_yaw — tail body wave
    ("all_roll",   [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),  # all roll unified
]
N_GROUPS = len(REDUCED_ORDER_GROUPS)  # 5
N_DIRECT_ACTIONS = N_GROUPS           # 5 (one action per group)

# ============== Wave Control Constants ==============
# Physics-parameterized traveling wave for anguilliform locomotion.
#
# 11 yaw joints (index 0=head .. 10=tail), even actuator indices: 0,2,...,20
# 11 roll joints, odd actuator indices: 1,3,...,21
N_YAW_JOINTS = 11

# Normalized body position s_i ∈ [0, 1] for each yaw joint (head→tail)
WAVE_S = np.array([i / (N_YAW_JOINTS - 1) for i in range(N_YAW_JOINTS)], dtype=np.float32)

# Amplitude envelope: head ~5%, tail 100% (head nearly fixed, tail max swing)
# Biological anguilliform fish keep the head stable for sensing/navigation
# while the tail generates thrust via large-amplitude undulation.
WAVE_ENVELOPE = (0.05 + 0.95 * WAVE_S).astype(np.float32)

# Wave action: 5-dim [A, omega, k_wave, head_bias, roll]
N_WAVE_ACTIONS = 5


# ============== Multi-Task Environment Class ==============


class EelMultiTaskEnv(Eel3DLBMEnv):
    """
    Multi-task eel locomotion environment with optional frequency-domain PD control.

    Instead of goal-reaching, the agent is given a task ID (one-hot in obs)
    and rewarded for performing the corresponding motion primitive:
      forward, turn_left, turn_right, ascend, descend.

    Action space (Biomimetic Grouped Control):
      - 4 Yaw groups: traveling wave with spatial phase gradient
      - 1 Roll group: all roll joints unified (prevents chaotic twisting)
      Total: 5-dim action

    Control modes:
      - 'direct': 5-dim action → 22-dim MuJoCo ctrl (grouped)
      - 'frequency': 20-dim Fourier coefficients → 22-dim MuJoCo ctrl

    Training: tasks are randomly sampled at reset (and optionally mid-episode).
    Inference: a command sequence can be provided to execute complex maneuvers.
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
        # Multi-task specific
        task_switch_interval: int = 0,      # steps between task switches (0 = only at reset)
        enabled_tasks: Optional[List[str]] = None,  # subset of tasks to train on
        # Task reward weights (tuned for eel: barrel-roll vertical strategy)
        reward_w_task: float = 1.0,
        reward_w_roll: float = 0.20,        # highest during FORWARD: 8-segment chain is very roll-prone
        reward_w_smooth: float = 0.03,      # highest: serial chain end-effector amplification
        reward_w_offaxis: float = 0.04,     # highest: traveling wave produces lateral force
        # Reference targets / scales (eel-specific, barrel-roll strategy)
        target_forward_vel: float = 0.12,
        target_yaw_rate: float = 0.08,      # lowest: long body has large turning radius
        target_vertical_vel: float = 0.05,  # barrel-roll vertical: Yaw wave redirected upward
        # Speed-target / direction-distance task options
        disable_speed_targets: bool = False,
        use_direction_dist_tasks: bool = False,
        direction_dist_min: float = 0.12,
        direction_dist_max: float = 0.25,
        direction_goal_threshold: float = 0.06,
        direction_dist_w_dist: float = 100.0,
        direction_dist_w_roll: float = 0.5,
        direction_dist_w_heading: float = 0.3,
        direction_dist_w_forward: float = 0.1,
        direction_dist_w_smooth: float = 0.15,
        direction_dist_w_offaxis: float = 0.0,
        direction_dist_goal_bonus: float = 10.0,
        direction_dist_terminate_on_goal: bool = True,
        alive_cost: float = 0.0,
        termination_penalty: float = 1.0,
        # Frequency-domain PD control parameters
        k_harmonics: int = DEFAULT_K_HARMONICS,
        b_bar: float = DEFAULT_B_BAR,
        use_reduced_order: bool = True,
        control_mode: str = "direct",
    ):
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
            goal_position=[0.5, 0.5, 0.5],  # dummy center
            control_mode='direct',    # parent's control_mode (we override below)
            K=3,                      # parent's K (unused, we override)
            B_bar=1.0,               # parent's B_bar (unused, we override)
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
        self.direction_dist_w_smooth = float(direction_dist_w_smooth)
        self.direction_dist_w_offaxis = float(direction_dist_w_offaxis)
        self.direction_dist_terminate_on_goal = bool(direction_dist_terminate_on_goal)
        self.alive_cost = float(alive_cost)
        self.termination_penalty = float(termination_penalty)
        self.goal_threshold = self.direction_goal_threshold if self.use_direction_dist_tasks else self.goal_threshold
        self.goal_reached_bonus = float(direction_dist_goal_bonus)
        self._direction_goal_positions_np = np.full((nworld, 3), 0.5, dtype=np.float32)

        # --- Control mode (override parent's) ---
        self.control_mode = control_mode.lower()
        if self.control_mode not in {"frequency", "direct", "wave"}:
            raise ValueError(f"Unknown control_mode '{control_mode}'. Expected 'wave', 'direct', or 'frequency'.")

        # --- Frequency-domain / direct PD control ---
        self.k_harmonics = k_harmonics
        self.b_bar = b_bar
        self.use_reduced_order = use_reduced_order

        # Both direct and frequency modes use the same biomimetic grouping
        self.n_ctrl_groups = N_GROUPS  # 5
        self.joint_groups = REDUCED_ORDER_GROUPS

        # Action dimension
        self.freq_action_dim = self.n_ctrl_groups * k_harmonics * 2

        # Build group-to-actuator mapping
        self._group_actuator_indices = [g[1] for g in self.joint_groups]

        # Extract ctrl_range from MuJoCo model for scaling θ* → ctrl
        self._ctrl_lo = self.mj_model.actuator_ctrlrange[:, 0].copy()  # (22,)
        self._ctrl_hi = self.mj_model.actuator_ctrlrange[:, 1].copy()  # (22,)

        # Precompute frequency bases: B_j = (j+1) * b_bar / K for j=0..K-1
        self._freq_bases = np.array(
            [(j + 1) * b_bar / k_harmonics for j in range(k_harmonics)],
            dtype=np.float32
        )

        # Time tracking for frequency-domain control
        self._time_val = np.zeros(nworld, dtype=np.float32)
        self._time_val_wp = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._dt = self.mj_model.opt.timestep * per_frame_steps
        self._prev_qpos_buffer = wp.zeros((nworld, self.mj_model.nq), dtype=wp.float32, device=self.device)

        # --- Override obs_dim ---
        # Kernel writes: 22 + 3*n_joints + NUM_TASKS + 2 dims
        #   22 = 6(forces) + 3(pos) + 4(quat) + 3(vel) + 3(angvel) + 3(LBM pos)
        #   3*n_joints = joint_torques + joint_angles + joint_velocities
        # direction-distance mode appends 4 extra features in Python (body-frame goal dir + dist)
        self._dirdist_obs_extra = 4 if self.use_direction_dist_tasks else 0
        self._kernel_obs_dim = 22 + 3 * self.n_joints + NUM_TASKS + 2
        self.obs_dim = self._kernel_obs_dim + self._dirdist_obs_extra

        # Current task for each world
        self._task_ids = np.zeros(nworld, dtype=np.int32)
        self._task_ids_wp = wp.zeros(nworld, dtype=wp.int32, device=self.device)

        # Steps since last task switch
        self._steps_since_switch = np.zeros(nworld, dtype=np.int32)

        # Override observation buffer (kernel-written dims only; dirdist appended in Python)
        self._obs_buffer = wp.zeros((nworld, self._kernel_obs_dim), dtype=wp.float32, device=self.device)

        if self.control_mode == "frequency":
            self.action_dim = self.freq_action_dim
        elif self.control_mode == "wave":
            self.action_dim = N_WAVE_ACTIONS  # 5: [A, omega, k_wave, head_bias, roll]
        else:
            self.action_dim = N_DIRECT_ACTIONS  # 5

        # Override action space for the selected control mode
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(nworld, self.action_dim),
            dtype=np.float32
        )

        # Override observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(nworld, self.obs_dim),
            dtype=np.float32
        )

        # Command sequence for inference mode
        self._command_sequence: Optional[List[dict]] = None
        self._command_index = 0
        self._command_step_counter = 0
        self._prev_ctrl = np.zeros((nworld, self.n_actuators), dtype=np.float32)
        self._prev_ctrl_valid = np.zeros(nworld, dtype=bool)

        # Displacement-based velocity buffers (LBM position diff / dt)
        self._prev_lbm_pos = np.zeros((nworld, 3), dtype=np.float32)
        self._prev_lbm_pos_valid = np.zeros(nworld, dtype=bool)
        self._disp_vel_wp = wp.zeros((nworld, 3), dtype=wp.float32, device=self.device)

        # Task-frame forward direction snapshot (captured at task-switch time).
        # Used by reward kernel instead of instantaneous head direction to avoid
        # head-oscillation noise from the traveling wave.
        # Initialized to world +Y (default forward); overwritten at each task switch.
        self._task_forward_dir = np.tile([0.0, 1.0, 0.0], (nworld, 1)).astype(np.float32)
        self._task_forward_dir_wp = wp.array(self._task_forward_dir, dtype=wp.float32, device=self.device)

        # EMA (exponential moving average) of body_forward over recent steps.
        # The snapshot taken at task-switch uses THIS smoothed value, NOT the
        # instantaneous body_forward. This prevents a reward-hacking strategy
        # where the agent flicks its head to a favorable orientation in the
        # single frame before task switch to inflate subsequent reward.
        # alpha=0.05 gives a smoothing horizon of ~1/alpha = 20 steps, which is
        # longer than one tail-beat period but short enough to track real turns.
        self._body_forward_ema = np.tile([0.0, 1.0, 0.0], (nworld, 1)).astype(np.float32)
        self._body_forward_ema_valid = np.zeros(nworld, dtype=bool)
        self._body_forward_ema_alpha = 0.05

        # --- Per-episode statistics for logging ---
        self._ep_reward_sum = np.zeros(nworld, dtype=np.float32)
        self._ep_step_count = np.zeros(nworld, dtype=np.int32)
        self._ep_v_forward_sum = np.zeros(nworld, dtype=np.float32)
        self._ep_v_lateral_sum = np.zeros(nworld, dtype=np.float32)
        self._ep_v_vertical_sum = np.zeros(nworld, dtype=np.float32)
        self._ep_yaw_rate_sum = np.zeros(nworld, dtype=np.float32)
        self._ep_boundary_count = np.zeros(nworld, dtype=np.int32)

        print(f"EelMultiTaskEnv initialized:")
        print(f"  Tasks: {[TASK_NAMES[i] for i in self.enabled_task_ids]}")
        print(f"  Task switch interval: {task_switch_interval} (0=only at reset)")
        print(f"  Control mode: {self.control_mode}")
        if self.control_mode == "wave":
            print(f"  Control: physics-parameterized traveling wave (5-dim)")
            print(f"    action[0]=A, [1]=omega, [2]=k_wave, [3]=head_bias, [4]=roll")
            print(f"    Amplitude envelope: head={WAVE_ENVELOPE[0]:.2f} → tail={WAVE_ENVELOPE[-1]:.2f}")
        elif self.control_mode == "frequency":
            print(f"  Control: K={k_harmonics} harmonics, B̄={b_bar}")
            print(f"  Action dim: {self.freq_action_dim} (frequency params)")
            print(f"  Groups ({self.n_ctrl_groups}): {[g[0] for g in self.joint_groups]}")
        else:
            print(f"  Control: direct grouped (4 Yaw groups + 1 Roll group = {self.action_dim} dims)")
            print(f"  Groups ({self.n_ctrl_groups}): {[g[0] for g in self.joint_groups]}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Obs dim: {self.obs_dim} (base {22+3*self.n_joints} + lbm 3 + task {NUM_TASKS} + phase 2)")
        print(
            f"  Reward weights: task={reward_w_task}, roll={reward_w_roll}, "
            f"smooth={reward_w_smooth}, offaxis={reward_w_offaxis}"
        )
        print(
            f"  Target velocities: fwd={target_forward_vel}, yaw={target_yaw_rate}, vert={target_vertical_vel}"
        )

    def _create_observation_space(self) -> spaces.Space:
        """Create observation space with task one-hot and phase info appended."""
        n_joints = self.mj_model.njnt - 1  # 22
        dirdist_extra = getattr(self, "_dirdist_obs_extra", 0)
        # 22 = 6(forces)+3(pos)+4(quat)+3(vel)+3(angvel)+3(LBM pos)
        obs_dim = 22 + 3 * n_joints + NUM_TASKS + 2 + dirdist_extra
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
        self._snapshot_task_forward_dir(mask)
        if self.use_direction_dist_tasks:
            self._sample_direction_distance_goals(mask)

    def _update_body_forward_ema(self):
        """Maintain an EMA of body_forward (rotation of body +Y into world frame).

        Called every step. The EMA value is later used as the task-switch snapshot,
        making the reference direction robust to instantaneous head flicks that
        could otherwise be exploited by the policy to game the reward at the
        exact moment of task switching.
        """
        qpos_np = self.mjw_data.qpos.numpy()
        alpha = self._body_forward_ema_alpha
        for w in range(self.nworld):
            qw = qpos_np[w, 3]
            qx = qpos_np[w, 4]
            qy = qpos_np[w, 5]
            qz = qpos_np[w, 6]
            vx, vy, vz = 0.0, 1.0, 0.0
            tx = 2.0 * (qy * vz - qz * vy)
            ty = 2.0 * (qz * vx - qx * vz)
            tz = 2.0 * (qx * vy - qy * vx)
            fwd = np.array([
                vx + qw * tx + (qy * tz - qz * ty),
                vy + qw * ty + (qz * tx - qx * tz),
                vz + qw * tz + (qx * ty - qy * tx),
            ], dtype=np.float32)
            n = np.linalg.norm(fwd)
            if n > 1.0e-6:
                fwd /= n
            if not self._body_forward_ema_valid[w]:
                self._body_forward_ema[w] = fwd
                self._body_forward_ema_valid[w] = True
            else:
                ema = (1.0 - alpha) * self._body_forward_ema[w] + alpha * fwd
                en = np.linalg.norm(ema)
                if en > 1.0e-6:
                    ema /= en
                self._body_forward_ema[w] = ema

    def _snapshot_task_forward_dir(self, mask: Optional[np.ndarray] = None):
        """Capture the task-frame forward direction for worlds in mask.

        Policy:
          - FORWARD task: ALWAYS use world +Y as the reference direction.
            This defeats FW-1 (S-shape wriggling in place with body_forward
            drifting) and FW-2 (crab-walking at 20° offset). Because the
            reference is a fixed world axis, v_forward and cos_fwd both
            require the eel to TRULY move along / face world +Y — any head
            drift or sideways gait directly loses reward.
          - TURN / ASCEND / DESCEND tasks: use the EMA-smoothed body_forward
            as before, so the reference direction represents "where the eel
            was heading at the moment of task switch". EMA smoothing (α=0.05)
            defeats the "flick head before switch" cheat.

        The snapshot is used by the reward kernel as a stable task-frame
        reference, avoiding the heavy head oscillation caused by the
        traveling wave.
        """
        if mask is None:
            mask = np.ones(self.nworld, dtype=bool)
        # Make sure the EMA reflects the current state (handles the "reset then
        # immediately snapshot" path where EMA may be stale / uninitialized).
        self._update_body_forward_ema()
        world_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        for w in range(self.nworld):
            if not mask[w]:
                continue
            task_id = int(self._task_ids[w])
            if task_id == TASK_FORWARD:
                # FORWARD: lock reference to world +Y (absolute world axis).
                fwd = world_y.copy()
            else:
                fwd = self._body_forward_ema[w].copy()
                norm = np.linalg.norm(fwd)
                if norm > 1.0e-6:
                    fwd /= norm
                else:
                    # Fallback to default forward if EMA is degenerate.
                    fwd = world_y.copy()
            self._task_forward_dir[w] = fwd
        wp.copy(
            self._task_forward_dir_wp,
            wp.array(self._task_forward_dir, dtype=wp.float32, device=self.device)
        )

    def _update_task_ids_wp(self):
        """Sync task IDs to Warp."""
        wp.copy(
            self._task_ids_wp,
            wp.array(self._task_ids.astype(np.int32), dtype=wp.int32, device=self.device)
        )

    # --- Direction-distance task helpers ---

    def _body_positions_normalized(self) -> np.ndarray:
        """Return current body positions in normalized LBM coordinates."""
        body_positions = np.zeros((self.nworld, 3), dtype=np.float32)
        for w in range(self.nworld):
            pos = self.lbm_solver.flows[w].solid_position.numpy()[0]
            body_positions[w] = [pos[0] / self.nx, pos[1] / self.ny, pos[2] / self.nz]
        return body_positions

    def _direction_template_for_task(self, task_id: int) -> np.ndarray:
        """Return body-frame direction template for a given task."""
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
        """Rotate vec by quaternion (w, x, y, z)."""
        qw, qx, qy, qz = quat
        qv = np.array([qx, qy, qz], dtype=np.float32)
        t = 2.0 * np.cross(qv, vec)
        return vec + qw * t + np.cross(qv, t)

    @staticmethod
    def _quat_rotate_vec_inv_np(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Rotate vec by inverse quaternion (w, x, y, z)."""
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
                # Stay at last command
                self._command_index = len(self._command_sequence) - 1
                self._command_step_counter = cmd["steps"]
                return

            # Switch to next command
            next_cmd = self._command_sequence[self._command_index]
            task_name = next_cmd["task"]
            task_id = TASK_NAMES.index(task_name)
            self._task_ids[:] = task_id
            self._update_task_ids_wp()
            self._snapshot_task_forward_dir()

    # --- Override core methods ---

    def _get_obs(self) -> np.ndarray:
        """Get observation with task one-hot and phase info."""
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

    def _update_disp_vel(self):
        """Compute displacement-based velocity from LBM head position difference.

        Uses solid_position[0] (head) from each world's LBM flow.
        Result is stored in self._disp_vel_wp for the reward kernel.
        """
        dt = self._dt  # mj_model.opt.timestep * per_frame_steps
        cur_pos = np.zeros((self.nworld, 3), dtype=np.float32)
        for w in range(self.nworld):
            p = self.lbm_solver.flows[w].solid_position.numpy()[0]
            cur_pos[w] = [p[0], p[1], p[2]]

        disp_vel = np.zeros((self.nworld, 3), dtype=np.float32)
        valid = self._prev_lbm_pos_valid
        if np.any(valid) and dt > 0:
            disp_vel[valid] = (cur_pos[valid] - self._prev_lbm_pos[valid]) / dt

        self._prev_lbm_pos[:] = cur_pos
        self._prev_lbm_pos_valid[:] = True
        wp.copy(self._disp_vel_wp, wp.array(disp_vel, dtype=wp.float32, device=self.device))

    def _compute_reward(self, instability_mask=None) -> np.ndarray:
        """Task-conditioned reward computation (eel-specific barrel-roll kernel)."""
        if self.use_direction_dist_tasks:
            wp.launch(
                compute_goal_reward_eel_kernel,
                dim=self.nworld,
                inputs=[
                    self.lbm_solver.flows_wp,
                    self._goal_positions_wp,
                    self._prev_dist_wp,
                    self._rewards_buffer,
                    self._current_dist_buffer,
                    self.mjw_data.qpos,
                    self.mjw_data.qvel,
                    self._disp_vel_wp,
                    float(self.nx),
                    float(self.ny),
                    float(self.nz),
                    self.direction_dist_w_dist,
                    self.direction_dist_w_roll,
                    self.direction_dist_w_heading,
                    self.direction_dist_w_forward,
                    self.direction_dist_w_smooth,
                    self.direction_dist_w_offaxis,
                    self.target_forward_vel,
                    self.target_yaw_rate,
                    self.target_vertical_vel,
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
            compute_eel_multitask_reward_kernel,
            dim=self.nworld,
            inputs=[
                self.mjw_data.qpos,
                self.mjw_data.qvel,
                self._disp_vel_wp,
                self._task_ids_wp,
                self._task_forward_dir_wp,
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
        """
        Map policy output to 22-dim MuJoCo actuator targets.

        wave mode:      5-dim physical params  → 22-dim ctrl
        direct mode:    5-dim grouped targets  → 22-dim ctrl
        frequency mode: 20-dim Fourier params  → 22-dim ctrl
        """
        action = np.clip(action, -1.0, 1.0)
        if self.control_mode == "wave":
            return self._wave_to_ctrl(action)
        if self.control_mode == "direct":
            return self._direct_to_ctrl(action)
        return self._freq_to_ctrl(action)

    def _wave_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """
        Convert 5-dim physical wave parameters to 22-dim MuJoCo ctrl.

        action layout:
          [0] A         : global amplitude scale       in [-1, 1]
          [1] omega     : oscillation frequency scale  in [-1, 1]
                          maps to Omega = omega * OMEGA_MAX (rad/s)
                          negative omega -> wave travels tail-to-head (reverse swim)
          [2] k_wave    : spatial wave number scale    in [-1, 1]
          maps to k = k_wave * K_MAX = k_wave * 1.5 (wavelengths along body)
          [3] head_bias : DC heading offset            in [-1, 1]
                          tapers from full at head to zero at tail -> drives turning
          [4] roll      : unified roll for all roll joints in [-1, 1]

        Joint i target angle (yaw, i = 0..10, head->tail):
          s_i        = i / 10                            (spatial position in [0,1])
          envelope_i = 0.05 + 0.95 * s_i                (head ~0, tail large)
          phi_i      = Omega * t + k * pi * s_i         (traveling wave phase)
          theta_i    = A * envelope_i * sin(phi_i)
                     + head_bias * (1 - s_i)            (bias tapers to 0 at tail)
        """
        nw = action.shape[0]
        ctrl = np.zeros((nw, self.n_actuators), dtype=np.float32)

        # Unpack parameters
        A         = action[:, 0]   # (nw,) amplitude in [-1, 1]
        omega_n   = action[:, 1]   # (nw,) normalized frequency
        k_n       = action[:, 2]   # (nw,) normalized wave number
        head_bias = action[:, 3]   # (nw,) heading bias
        roll_cmd  = action[:, 4]   # (nw,) unified roll

        # Physical scales
        OMEGA_MAX = np.pi * 2.0   # max 2pi rad/s ~ 1 Hz flapping
        K_MAX     = 1.5           # up to 1.5 full wavelengths along the body

        omega = omega_n * OMEGA_MAX   # (nw,)
        k     = k_n * K_MAX           # (nw,)
        t     = self._time_val[:nw]   # (nw,)

        # Broadcast over joints: shapes (nw, 11)
        s   = WAVE_S[None, :]        # (1, 11)
        env = WAVE_ENVELOPE[None, :] # (1, 11)

        phase = (omega[:, None] * t[:, None]
                 + k[:, None] * np.pi * s)   # traveling wave phase (nw, 11)

        theta_norm = (A[:, None] * env * np.sin(phase)
                      + head_bias[:, None] * (1.0 - s))  # (nw, 11)
        theta_norm = np.clip(theta_norm, -1.0, 1.0)

        # Write yaw joints (even actuator indices 0,2,...,20)
        for i, act_idx in enumerate(YAW_ACTUATOR_INDICES):
            lo, hi = self._ctrl_lo[act_idx], self._ctrl_hi[act_idx]
            ctrl[:, act_idx] = lo + (theta_norm[:, i] + 1.0) * 0.5 * (hi - lo)

        # Write roll joints (odd actuator indices 1,3,...,21) — unified
        for act_idx in ROLL_ACTUATOR_INDICES:
            lo, hi = self._ctrl_lo[act_idx], self._ctrl_hi[act_idx]
            ctrl[:, act_idx] = lo + (roll_cmd + 1.0) * 0.5 * (hi - lo)

        return ctrl


    def _direct_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """
        Convert 5-dim grouped action to 22-dim MuJoCo ctrl.

        action[:, 0] → head_yaw group  (joint1_yaw)
        action[:, 1] → front_yaw group (joint2~4_yaw)
        action[:, 2] → mid_yaw group   (joint5~7_yaw)
        action[:, 3] → tail_yaw group  (joint8~11_yaw)
        action[:, 4] → all_roll group  (joint1~11_roll, unified)
        """
        nw = action.shape[0]
        ctrl = np.zeros((nw, self.n_actuators), dtype=np.float32)

        # Map each group's single action value to all actuators in that group
        for g_idx, (g_name, act_indices) in enumerate(self.joint_groups):
            val = action[:, g_idx]  # (nworld,) in [-1, 1]
            for act_idx in act_indices:
                lo, hi = self._ctrl_lo[act_idx], self._ctrl_hi[act_idx]
                ctrl[:, act_idx] = lo + (val + 1.0) * 0.5 * (hi - lo)

        return ctrl

    def _freq_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """
        Convert frequency-domain action parameters to MuJoCo ctrl (target angles).

        Action layout per group g (K harmonics):
            [A_g0, C_g0, A_g1, C_g1, ..., A_g(K-1), C_g(K-1)]

        θ*_g = Σ_{j=0}^{K-1} A_{gj} * sin(π/2 * B_j * t + C_{gj})
        Then scale θ*_g from [-1,1] to each actuator's ctrl_range.

        Locked Roll actuators are set to center of their ctrl_range.

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
            A_gj = action[:, base + 0: base + K * 2: 2]  # (nworld, K)
            c_raw = action[:, base + 1: base + K * 2: 2]  # (nworld, K)
            C_gj = c_raw * np.pi  # (nworld, K)

            t = self._time_val[:nw, None]  # (nworld, 1)
            phase = (np.pi / 2) * self._freq_bases[None, :] * t + C_gj  # (nworld, K)
            theta_star = np.sum(A_gj * np.sin(phase), axis=1)  # (nworld,)

            theta_star = np.clip(theta_star, -1.0, 1.0)

            for act_idx in self._group_actuator_indices[g_idx]:
                lo, hi = self._ctrl_lo[act_idx], self._ctrl_hi[act_idx]
                ctrl[:, act_idx] = lo + (theta_star + 1.0) * 0.5 * (hi - lo)

        return ctrl

    def step(self, action: np.ndarray):
        """
        Execute one step with the selected control mode.

        Frequency mode converts Fourier coefficients to target angles.
        Direct mode maps 5-dim grouped action to 22-dim MuJoCo ctrl.
        """
        ctrl = self._action_to_ctrl(action)
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        # Write target angles to MuJoCo ctrl — position actuators do PD tracking
        wp.copy(self.mjw_data.ctrl, wp.array(ctrl, dtype=wp.float32, device=self.device))

        # Advance phase clock for time-based control modes
        if self.control_mode in {"frequency", "wave"}:
            self._time_val += self._dt

        # Physics simulation
        self._simulation_step()

        # Update step counts
        self.step_counts += 1

        # Check stability
        instability_mask = self._check_numerical_stability()

        # Check goals reached (direction-distance mode)
        goal_reached = self._check_goals_reached() if self.use_direction_dist_tasks else np.zeros(self.nworld, dtype=np.int32)

        # Check termination (boundary + instability)
        self._is_terminated(instability_mask)
        boundary_terminated = self._terminated_buffer.numpy().astype(bool).copy()

        # Compute displacement-based velocity from LBM position difference
        self._update_disp_vel()

        # Update EMA of body_forward for stable task-switch snapshot
        # (must happen every step so the EMA is fresh when a switch occurs).
        self._update_body_forward_ema()

        # Compute reward before task switching so each action is credited against
        # the task that generated it.
        reward = self._compute_reward(instability_mask)

        # Small control-smoothness penalty on target-angle changes
        if np.any(self._prev_ctrl_valid):
            ctrl_range = np.maximum(self._ctrl_hi - self._ctrl_lo, 1.0e-6)[None, :]
            ctrl_delta = (ctrl - self._prev_ctrl) / ctrl_range
            valid_mask = self._prev_ctrl_valid.astype(np.float32)
            reward += (
                -0.15 * self.mt_reward_w_smooth
                * np.mean(ctrl_delta * ctrl_delta, axis=1)
                * valid_mask
            )

        # Very small control-effort regularization to avoid extreme commands
        ctrl_center = (0.5 * (self._ctrl_lo + self._ctrl_hi))[None, :]
        ctrl_half_range = np.maximum(0.5 * (self._ctrl_hi - self._ctrl_lo), 1.0e-6)[None, :]
        ctrl_effort = (ctrl - ctrl_center) / ctrl_half_range
        reward += -0.05 * self.mt_reward_w_smooth * np.mean(ctrl_effort * ctrl_effort, axis=1)

        # Traveling wave bonus: reward non-zero |k_wave| to prevent k_wave→0 collapse
        if self.control_mode == "wave":
            k_wave_abs = np.abs(action[:, 2])  # action[2] = k_wave in [-1, 1]
            # Bonus: 0.15 * k/(k+0.3) — encourages |k_wave| ≥ 0.3
            wave_bonus = 0.15 * k_wave_abs / (k_wave_abs + 0.3 + 1e-6)
            reward += wave_bonus

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

        # Termination penalty (configurable, default 1.0)
        non_goal_terminated = (
            terminated & ~goal_reached_mask
            if self.use_direction_dist_tasks and self.direction_dist_terminate_on_goal
            else terminated
        )
        if self.termination_penalty > 0.0:
            reward[non_goal_terminated] -= self.termination_penalty

        # Final safety check
        reward_nan_mask = np.zeros(self.nworld, dtype=bool)
        if np.any(np.isnan(reward)) or np.any(np.isinf(reward)):
            reward_nan_mask = np.isnan(reward) | np.isinf(reward)
            reward[reward_nan_mask] = -1.0
            terminated[reward_nan_mask] = True

        truncated = np.array(self.step_counts >= self.max_episode_steps)
        done = terminated | truncated

        # Mid-episode task switching only affects the next observation
        if self.task_switch_interval > 0 and self._command_sequence is None:
            self._steps_since_switch += 1
            switch_mask = (self._steps_since_switch >= self.task_switch_interval) & ~done
            if np.any(switch_mask):
                self._sample_tasks(switch_mask)
                self._steps_since_switch[switch_mask] = 0

        # Advance command sequence for the next observation
        if self._command_sequence is not None and not np.any(done):
            self._advance_command_sequence()

        # Returned observation exposes the next task after any switch
        observation = self._get_obs()

        # Handle NaN/Inf in observations
        obs_nan_mask = np.any(np.isnan(observation) | np.isinf(observation), axis=1)
        if np.any(obs_nan_mask):
            instability_mask = instability_mask | obs_nan_mask
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
            reward[obs_nan_mask] = -1.0
            terminated[obs_nan_mask] = True
            done = terminated | truncated

        # --- Per-step statistics accumulation ---
        qvel_np = self.mjw_data.qvel.numpy()
        qpos_np = self.mjw_data.qpos.numpy()
        for w in range(self.nworld):
            qw, qx, qy, qz = qpos_np[w, 3], qpos_np[w, 4], qpos_np[w, 5], qpos_np[w, 6]
            vx, vy, vz = qvel_np[w, 0], qvel_np[w, 1], qvel_np[w, 2]
            fwd_x = 2.0 * (qx * qy + qw * qz)
            fwd_y = 1.0 - 2.0 * (qx * qx + qz * qz)
            fwd_z = 2.0 * (qy * qz - qw * qx)
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

        # Per-world episode metrics (consumed by SB3 EnvMetricCallback)
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

        # Call grandparent reset (LBMFluidEnv3D.reset), skip Eel3DLBMEnv's goal logic
        from ..lbm_fluid_env_3d import LBMFluidEnv3D
        LBMFluidEnv3D.reset(self, seed=seed, options=options)

        # Invalidate EMA so the upcoming snapshot uses the fresh post-reset
        # body_forward as its initial value (not a stale trajectory from the
        # previous episode).
        self._body_forward_ema_valid[:] = False

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
            self._snapshot_task_forward_dir()

        self._steps_since_switch[:] = 0
        self._time_val[:] = 0.0
        self._prev_ctrl.fill(0.0)
        self._prev_ctrl_valid[:] = False
        self._prev_lbm_pos_valid[:] = False
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        # Reset episode accumulators
        self._ep_reward_sum[:] = 0.0
        self._ep_step_count[:] = 0
        self._ep_v_forward_sum[:] = 0.0
        self._ep_v_lateral_sum[:] = 0.0
        self._ep_v_vertical_sum[:] = 0.0
        self._ep_yaw_rate_sum[:] = 0.0
        self._ep_boundary_count[:] = 0

        return self._get_obs()

    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        """Reset specific worlds and re-sample their tasks."""
        from ..lbm_fluid_env_3d import LBMFluidEnv3D
        LBMFluidEnv3D.partial_reset(self, reset_mask)

        if not np.any(reset_mask):
            return self._get_obs()

        # Invalidate EMA for the reset worlds so the upcoming snapshot uses the
        # fresh post-reset body_forward as its initial value.
        self._body_forward_ema_valid[reset_mask] = False

        # Re-sample tasks for reset worlds
        if self._command_sequence is None:
            self._sample_tasks(reset_mask)

        self._steps_since_switch[reset_mask] = 0
        self._time_val[reset_mask] = 0.0
        self._prev_ctrl[reset_mask] = 0.0
        self._prev_ctrl_valid[reset_mask] = False
        self._prev_lbm_pos_valid[reset_mask] = False
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        # Reset episode accumulators for reset worlds
        self._ep_reward_sum[reset_mask] = 0.0
        self._ep_step_count[reset_mask] = 0
        self._ep_v_forward_sum[reset_mask] = 0.0
        self._ep_v_lateral_sum[reset_mask] = 0.0
        self._ep_v_vertical_sum[reset_mask] = 0.0
        self._ep_yaw_rate_sum[reset_mask] = 0.0
        self._ep_boundary_count[reset_mask] = 0

        return self._get_obs()

    def get_current_task(self, world_idx: int = 0) -> str:
        """Get current task name for a world."""
        return TASK_NAMES[self._task_ids[world_idx]]

    def get_current_goal(self, world_idx: int = 0) -> Tuple[float, float, float]:
        """Override for compatibility: return direction-distance goal or center position."""
        if self.use_direction_dist_tasks:
            g = self._direction_goal_positions_np[world_idx]
            return (float(g[0]), float(g[1]), float(g[2]))
        return (0.5, 0.5, 0.5)
