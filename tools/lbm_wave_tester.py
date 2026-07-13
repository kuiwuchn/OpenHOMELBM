"""
LBM Wave Tester — validate reward design with real fluid simulation.

Runs preset wave parameters (from *_wave_viewer.py) through the full
MultitaskEnv (with LBM solver), collects real qpos/qvel/reward, and
exports video + CSV metrics.

No trained agent is needed — the controller is purely open-loop using
the physics-parameterized wave presets.

Supported animals: tuna, eel, clownfish, turtle  (extensible via ANIMAL_REGISTRY)

Usage examples:
    # Tuna forward preset, 300 steps, MuJoCo-only video
    python tools/lbm_wave_tester.py --animal tuna --preset forward --steps 300

    # Tuna all presets, 200 steps each
    python tools/lbm_wave_tester.py --animal tuna --preset all --steps 200

    # Tuna with LBM flow visualization
    python tools/lbm_wave_tester.py --animal tuna --preset forward --steps 300 --with-lbm

    # Eel forward preset
    python tools/lbm_wave_tester.py --animal eel --preset forward --steps 300

    # Clownfish forward preset
    python tools/lbm_wave_tester.py --animal clownfish --preset forward --steps 300

    # Custom parameter override
    python tools/lbm_wave_tester.py --animal tuna --preset forward --steps 300 \
        --override "A_tail=0.9,omega=0.3"
"""

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lbm3d_runtime import (
    MuJoCoRenderer,
    combine_frames_left_right,
    get_mujoco_frame,
    get_raw_frame_3d,
    load_named_config,
    make_multitask_env,
    process_raw_to_frame,
    save_video,
)


# ── Task constants (shared across all animals) ───────────────────────────────
TASK_NAMES = ["forward", "turn_left", "turn_right", "ascend", "descend"]
NUM_TASKS = 5

# ── Task color mapping for video overlay (BGR) ──────────────────────────────
TASK_COLORS = {
    "forward":    (100, 255, 100),   # green
    "turn_left":  (100, 200, 255),   # light blue
    "turn_right": (255, 200, 100),   # orange
    "ascend":     (200, 100, 255),   # purple
    "descend":    (255, 255, 100),   # yellow
}


# =============================================================================
# Animal Registry — preset parameters + action mapping per animal
# =============================================================================

# ── Tuna presets ─────────────────────────────────────────────────────────────────
# 7-dim wave action (tail + 4 stabilizer fins, propeller mode):
#   [A_tail, omega_tail, yaw_bias, tail_pitch, fin_amp, fin_asym, fin_freq]
#
# Actuator layout (14 DOFs — 4 fins + 10 tail):
#   [0] fin_right_pitch  [1] fin_left_pitch
#   [2] fin_rear_right_pitch  [3] fin_rear_left_pitch
#   [4-5] t0 yaw/pitch (passive anchor, always 0)
#   [6-13] t1-t4 yaw/pitch#
# Tail propeller rotation + lateral fin stabilization:
#   - Yaw joints: ±90°~±150° range (wide for aggressive tail whipping)
#   - Pitch joints: ±80°~±135° range (wide for aggressive maneuvering)
#   - TAIL_PITCH_RATIO = 0.8 (pitch oscillation = 0.8× yaw amplitude)
#   - Fins: small lateral stabilizers that counter-flap to cancel roll torque
# === v4 presets (2026-04-10) — independent fin frequency ===
# fin_freq controls fin flap frequency independently from tail.
# fin_freq=-0.20 → ~4.8 rad/s (moderate anti-roll flap)
TUNA_PRESETS = {
    "forward":  dict(A_tail=0.95, omega_tail=-0.20, yaw_bias= 0.00, tail_pitch= 0.20, fin_amp=0.55, fin_asym= 0.00, fin_freq=-0.20),
    "turn_r":   dict(A_tail=0.80, omega_tail= 0.10, yaw_bias=-0.60, tail_pitch= 0.00, fin_amp=0.45, fin_asym=-0.10, fin_freq=-0.20),
    "turn_l":   dict(A_tail=0.80, omega_tail= 0.10, yaw_bias= 0.60, tail_pitch= 0.00, fin_amp=0.45, fin_asym= 0.10, fin_freq=-0.20),
    "ascend":   dict(A_tail=0.70, omega_tail= 0.10, yaw_bias= 0.00, tail_pitch= 0.60, fin_amp=0.40, fin_asym= 0.00, fin_freq=-0.20),
    "descend":  dict(A_tail=0.70, omega_tail= 0.10, yaw_bias= 0.00, tail_pitch=-0.60, fin_amp=0.40, fin_asym= 0.00, fin_freq=-0.20),
    "fast":     dict(A_tail=1.00, omega_tail= 0.50, yaw_bias= 0.00, tail_pitch= 0.25, fin_amp=0.65, fin_asym= 0.00, fin_freq= 0.00),
    "tail_only":dict(A_tail=1.00, omega_tail= 0.20, yaw_bias= 0.00, tail_pitch= 0.25, fin_amp=0.00, fin_asym= 0.00, fin_freq= 0.00),
    "glide":    dict(A_tail=0.40, omega_tail=-0.10, yaw_bias= 0.00, tail_pitch= 0.10, fin_amp=0.20, fin_asym= 0.00, fin_freq=-0.30),
    "cold_start": dict(A_tail=1.00, omega_tail= 0.05, yaw_bias= 0.00, tail_pitch= 0.30, fin_amp=0.60, fin_asym= 0.00, fin_freq=-0.10),
}
TUNA_ACTION_KEYS = ["A_tail", "omega_tail", "yaw_bias", "tail_pitch", "fin_amp", "fin_asym", "fin_freq"]
# ── Eel presets (mirror of eel_wave_viewer.py SliderPanel presets) ─────────────
# 5-dim wave action: [A, omega, k_wave, head_bias, roll]
EEL_PRESETS = {
    "forward":        dict(A=0.8,  omega=-0.5, k_wave=0.5,  head_bias= 0.0,  roll=0.0),
    "turn_l":         dict(A=0.7,  omega=-0.5, k_wave=0.5,  head_bias= 0.6,  roll=0.0),
    "turn_r":         dict(A=0.7,  omega=-0.5, k_wave=0.5,  head_bias=-0.6,  roll=0.0),
    "reverse":        dict(A=0.8,  omega=0.5,  k_wave=0.5,  head_bias= 0.0,  roll=0.0),
    "fast":           dict(A=1.0,  omega=-1.0, k_wave=0.8,  head_bias= 0.0,  roll=0.0),
    "freeze":         dict(A=0.0,  omega=0.0,  k_wave=0.5,  head_bias= 0.0,  roll=0.0),
    # Head-tail in-phase swing: k_wave=0 → whole body oscillates as one unit
    # (no traveling wave), head and tail swing together in the same direction.
    # Expected: poor forward thrust (no momentum transfer), high lateral drag.
    "head_tail_swing": dict(A=0.8,  omega=-0.5, k_wave=0.0,  head_bias= 0.0,  roll=0.0),
}
EEL_ACTION_KEYS = ["A", "omega", "k_wave", "head_bias", "roll"]

# ── Clownfish presets (mirror of clownfish_wave_viewer.py SliderPanel presets) ──
# 7-dim wave action: [A, omega, k_wave, head_bias, pitch, caudal_pitch, caudal_twist]
#
# Clownfish (carangiform) — 4-segment body, 7 DOF MOTOR actuators (torque control):
#   body_yaw(0, gear=3), body_pitch(1, gear=1.5), peduncle_yaw(2, gear=5),
#   peduncle_pitch(3, gear=1.5), caudal_rotate(4, gear=1.5),
#   caudal_yaw(5, gear=8), caudal_pitch(6, gear=2)
#
# Head (body_yaw) does NOT participate in the traveling wave — it only
# receives the DC head_bias for steering.
# Traveling wave torque (peduncle + caudal only):
#   τᵢ = A · env(sᵢ) · sin(ω·t + k·π·sᵢ)
#   2 wave joints; amplitude envelope [0.50, 1.0]
#   head_bias → body_yaw DC steering only
#   pitch controls body+peduncle pitch uniformly
#   caudal_pitch and caudal_twist are independent caudal fin controls
CLOWNFISH_PRESETS = {
    "forward":  dict(A=1.0,  omega=-0.85, k_wave=0.5,  head_bias= 0.0,
                     pitch= 0.0,  caudal_pitch= 0.0,  caudal_twist=0.0),
    "turn_l":   dict(A=1.0,  omega=-0.85, k_wave=0.5,  head_bias= 0.70,
                     pitch= 0.0,  caudal_pitch= 0.0,  caudal_twist= 0.20),
    "turn_r":   dict(A=1.0,  omega=-0.85, k_wave=0.5,  head_bias=-0.70,
                     pitch= 0.0,  caudal_pitch= 0.0,  caudal_twist=-0.20),
    "ascend":   dict(A=0.90, omega=-0.75, k_wave=0.6,  head_bias= 0.0,
                     pitch=-0.55, caudal_pitch=-0.40,  caudal_twist=0.0),
    "descend":  dict(A=0.90, omega=-0.75, k_wave=0.6,  head_bias= 0.0,
                     pitch= 0.55, caudal_pitch= 0.40,  caudal_twist=0.0),
    "fast":     dict(A=1.0,  omega=-1.0,  k_wave=0.8,  head_bias= 0.0,
                     pitch= 0.0,  caudal_pitch= 0.0,  caudal_twist=0.0),
    "freeze":   dict(A=0.0,  omega= 0.0,  k_wave=0.5,  head_bias= 0.0,
                     pitch= 0.0,  caudal_pitch= 0.0,  caudal_twist=0.0),
}
CLOWNFISH_ACTION_KEYS = ["A", "omega", "k_wave", "head_bias", "pitch", "caudal_pitch", "caudal_twist"]


def tuna_preset_to_action(params: dict) -> np.ndarray:
    """Convert tuna preset dict to 5-dim action array (tail only)."""
    return np.array([params[k] for k in TUNA_ACTION_KEYS], dtype=np.float32)


def eel_preset_to_action(params: dict) -> np.ndarray:
    """Convert eel preset dict to 5-dim action array."""
    return np.array([params[k] for k in EEL_ACTION_KEYS], dtype=np.float32)


def clownfish_preset_to_action(params: dict) -> np.ndarray:
    """Convert clownfish preset dict to 7-dim action array."""
    return np.array([params[k] for k in CLOWNFISH_ACTION_KEYS], dtype=np.float32)


# ── Turtle presets (v9 — 5-dim, synced with turtle_wave_viewer.py) ─────────────
# 5-dim wave action: [A_flap, omega_flap, A_rot, stroke_asym, flap_asym]
# Fixed: rot_phase=-0.50 (optimal -90° lag), tail_bias=0.0
#
# Sea turtle (cheloniiform) locomotion — 9 DOF actuators (sweep disabled):
#   - Front flipper flap is PRIMARY propulsion (synchronized up/down stroke)
#   - Front flipper rotation controls attack angle (phase offset vs flap)
#   - Stroke asymmetry models biological downstroke/upstroke difference
#   - L/R flap amplitude asymmetry (flap_asym) for differential turning
#   - Rear flippers follow front with fixed scaled-down amplitude (rear_scale=0.75)
#   - Tail yaw is DC bias only for fine directional control
#
# Key physics:
#   rot_phase = -0.50 (FIXED, -90° lag) → max horizontal thrust (cruise/sprint)
#     rotation lags flap by 90°: AoA positive during downstroke,
#     negative during upstroke → both half-cycles produce forward thrust.
#     Trapezoidal plateau aligns with peak flap velocity.
#   stroke_asym > 0 → fast downstroke + slow upstroke → net upward force (ascend)
#     Power stroke (fast, low AoA): pure vertical reaction force
#     Recovery stroke (slow, full AoA): feathers through fluid
#   stroke_asym < 0 → fast upstroke + slow downstroke → net downward force (descend)
#   flap_asym > 0 → right flipper stronger → turn left
#   flap_asym < 0 → left flipper stronger → turn right
#
# OMEGA_MAX = 3.2 rad/s (~0.51 Hz); omega_flap maps [-1,1] → [0, 3.2] rad/s
# Fixed defaults: A_sweep=0.5, rear_scale=0.60, pitch_bias=0.0,
#                  rot_phase=-0.50, tail_bias=0.0
# ROT_AMP_SCALE=0.8 applied inside env; presets use raw A_rot values
TURTLE_PRESETS = {
    # ── Forward (cruise): efficient long-distance swimming ──
    # rot_phase=-0.50 (-90° lag): rotation lags flap by quarter-cycle
    #   → AoA positive during downstroke, negative during upstroke
    #   → BOTH half-cycles produce forward thrust
    #   → trapezoidal plateau aligns with peak flap velocity
    # omega_flap=0.65 → 2.64 rad/s (~0.42 Hz), moderate frequency
    # stroke_asym=0: symmetric flap avoids sigmoid distortion & downward tilt
    "forward":  dict(A_flap=0.80, omega_flap= 0.65,
                     A_rot=0.70, stroke_asym= 0.00, flap_asym= 0.00),

    # ── Forward (fast): maximum effort sprint ──
    # Full amplitude + high frequency for maximum thrust
    # stroke_asym=0: unused (flap is pure sine; thrust via rot_phase only)
    "fast":     dict(A_flap=1.00, omega_flap= 1.00,
                     A_rot=0.70, stroke_asym= 0.00, flap_asym= 0.00),

    # ── Turn Left: right flipper full power + tail bias left ──
    # A_flap=0.80, flap_asym=+0.80 → right=(0.80+0.80)*0.5=0.80, left=0.00
    #   one-sided thrust: only right flipper active → maximum yaw torque
    # omega_flap=0.65 → 2.64 rad/s (~0.42 Hz), same as forward cruise
    "turn_l":   dict(A_flap=0.80, omega_flap= 0.65,
                     A_rot=0.70, stroke_asym= 0.00, flap_asym= 0.80),

    # ── Turn Right: mirror of Turn Left ──
    # flap_asym=-0.80 → only left flipper active → yaw right
    "turn_r":   dict(A_flap=0.80, omega_flap= 0.65,
                     A_rot=0.70, stroke_asym= 0.00, flap_asym=-0.80),

    # ── Ascend: asymmetric flapping for net upward force ──
    # stroke_asym=+0.80: strong downstroke asymmetry
    #   → downstroke (power stroke) is FAST with near-zero attack angle
    #     → pure upward reaction force (no forward component)
    #   → upstroke (recovery stroke) is SLOW with full attack angle
    #     → minimizes downward force, feathers through the fluid
    # rot_phase=-0.50: standard forward-thrust phase (attack angle active
    #   only during recovery stroke due to asymmetric modulation)
    # A_flap=1.00, omega_flap=0.80: high amplitude + fast frequency for max force
    # A_rot=0.80: moderate attack angle for recovery stroke feathering
    "ascend":   dict(A_flap=1.00, omega_flap= 0.80,
                     A_rot=0.80, stroke_asym= 0.80, flap_asym= 0.00),

    # ── Descend: asymmetric flapping for net downward force ──
    # stroke_asym=-0.80: strong upstroke asymmetry (mirror of ascend)
    #   → upstroke (power stroke) is FAST with near-zero attack angle
    #     → pure downward reaction force
    #   → downstroke (recovery stroke) is SLOW with full attack angle
    #     → minimizes upward force, feathers through the fluid
    # Same logic as ascend but inverted
    "descend":  dict(A_flap=1.00, omega_flap= 0.80,
                     A_rot=0.80, stroke_asym=-0.80, flap_asym= 0.00),

    # ── Glide: minimal effort coasting ──
    # Very low amplitude + low frequency → energy-saving coast
    # stroke_asym=0: unused (flap is pure sine)
    "glide":    dict(A_flap=0.25, omega_flap= 0.20,
                     A_rot=0.10, stroke_asym= 0.00, flap_asym= 0.00),

    # ── Freeze: all zero — no motion (debug) ──
    "freeze":   dict(A_flap=0.00, omega_flap= 0.00,
                     A_rot=0.00, stroke_asym= 0.00, flap_asym= 0.00),
}
TURTLE_ACTION_KEYS = ["A_flap", "omega_flap", "A_rot", "stroke_asym", "flap_asym"]

def turtle_preset_to_action(params: dict) -> np.ndarray:
    """Convert turtle preset dict to 5-dim action array."""
    return np.array([params[k] for k in TURTLE_ACTION_KEYS], dtype=np.float32)


# ── Animal registry ───────────────────────────────────────────────────────────
ANIMAL_REGISTRY = {
    "tuna": {
        "configs": ["lbm3d", "tuna_multitask"],
        "dirdist_configs": None,  # tuna does not support dirdist yet
        "control_mode": "wave",
        "presets": TUNA_PRESETS,
        "action_keys": TUNA_ACTION_KEYS,
        "preset_to_action": tuna_preset_to_action,
    },
    "eel": {
        "configs": ["lbm3d", "eel_multitask"],
        "dirdist_configs": ["lbm3d", "sb3_eel_multitask_nospeed_dirdist"],
        "control_mode": "wave",
        "presets": EEL_PRESETS,
        "action_keys": EEL_ACTION_KEYS,
        "preset_to_action": eel_preset_to_action,
    },
    "turtle": {
        "configs": ["lbm3d", "turtle_multitask"],
        "dirdist_configs": None,  # turtle does not support dirdist yet
        "control_mode": "wave",
        "presets": TURTLE_PRESETS,
        "action_keys": TURTLE_ACTION_KEYS,
        "preset_to_action": turtle_preset_to_action,
    },
    "clownfish": {
        "configs": ["lbm3d", "clownfish_multitask_wave"],
        "dirdist_configs": None,  # clownfish does not support dirdist yet
        "control_mode": "wave",
        "presets": CLOWNFISH_PRESETS,
        "action_keys": CLOWNFISH_ACTION_KEYS,
        "preset_to_action": clownfish_preset_to_action,
    },
}


# =============================================================================
# Eel goal-reward component decomposition (Python mirror of the warp kernel)
# =============================================================================
import math as _math

def _cos_heading_exp(cos_val: float) -> float:
    """(exp(cos-1) - exp(-2)) / (1 - exp(-2)) → [0, 1]"""
    e_neg2 = _math.exp(-2.0)
    return (_math.exp(cos_val - 1.0) - e_neg2) / (1.0 - e_neg2 + 1e-6)

def _upright_penalty_exp(component: float, scale: float = 0.5) -> float:
    """1 - exp(-component²/scale²) → [0, 1]"""
    return 1.0 - _math.exp(-(component ** 2) / (scale ** 2 + 1e-6))

def _smooth_penalty_exp(omega: float, scale: float = 1.0) -> float:
    """1 - exp(-omega²/scale²) → [0, 1]"""
    return 1.0 - _math.exp(-(omega ** 2) / (scale ** 2 + 1e-6))

def _soft_penalty_exp(v: float, scale: float) -> float:
    """1 - exp(-v²/scale²) → [0, 1]"""
    return 1.0 - _math.exp(-(v ** 2) / (scale ** 2 + 1e-6))


def _directional_reward_exp(value: float, scale: float) -> float:
    """sign(v) * (1 - exp(-|v|/scale)) → [-1, 1]"""
    abs_v = abs(value)
    safe_scale = scale + 1e-6
    sat = 1.0 - _math.exp(-abs_v / safe_scale)
    if value > 0.0:
        return sat
    elif value < 0.0:
        return -sat
    return 0.0


def compute_eel_goal_reward_components(
    qpos: np.ndarray,       # (nqpos,)  — single world
    qvel: np.ndarray,       # (nqvel,)  — single world
    disp_vel: np.ndarray,   # (3,)      — displacement-based velocity
    body_pos_norm: np.ndarray,  # (3,)  — normalized body position [0,1]
    goal_pos: np.ndarray,   # (3,)      — normalized goal position [0,1]
    prev_dist: float,
    w_dist: float,
    w_roll: float,
    w_heading: float,
    w_forward: float,
    w_smooth: float,
    w_offaxis: float,
    target_forward_vel: float = 0.1,
    target_yaw_rate: float = 0.5,
    target_vertical_vel: float = 0.05,
) -> dict:
    """
    Python mirror of compute_goal_reward_eel_kernel.
    Returns a dict with keys:
      r_dist, r_heading, r_forward, r_upright, r_smooth, r_offaxis,
      total, current_dist, dist_improvement
    """
    # --- 1. Distance improvement ---
    dx = body_pos_norm[0] - goal_pos[0]
    dy = body_pos_norm[1] - goal_pos[1]
    dz = body_pos_norm[2] - goal_pos[2]
    current_dist = _math.sqrt(dx*dx + dy*dy + dz*dz)
    dist_improvement = prev_dist - current_dist
    r_dist = w_dist * dist_improvement

    # --- Quaternion → body axes ---
    qw, qx, qy, qz_q = float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])
    body_forward = quat_rotate_vec([qw, qx, qy, qz_q], np.array([0.0, 1.0, 0.0]))
    body_up      = quat_rotate_vec([qw, qx, qy, qz_q], np.array([0.0, 0.0, 1.0]))

    # --- Direction to goal ---
    to_goal = np.array([-dx, -dy, -dz], dtype=np.float64)
    to_goal_len = float(np.linalg.norm(to_goal))

    # --- Infer implicit task from goal direction ---
    vert_frac = abs(float(to_goal[2])) / to_goal_len if to_goal_len > 1e-6 else 0.0
    horiz_frac = 1.0 - vert_frac
    alpha = max(horiz_frac, 0.05)

    # Body-frame velocity decomposition
    body_right = quat_rotate_vec([qw, qx, qy, qz_q], np.array([1.0, 0.0, 0.0]))
    v_forward_body = float(np.dot(disp_vel, body_forward))
    v_lateral = float(np.dot(disp_vel, body_right))
    v_vertical = float(disp_vel[2])
    omega_body = np.array([float(qvel[3]), float(qvel[4]), float(qvel[5])], dtype=np.float64)
    omega_world = quat_rotate_vec([qw, qx, qy, qz_q], omega_body)
    yaw_rate = float(omega_world[2])

    # --- 2. Chord-angle heading ---
    # Inverse-envelope weighted sum of ALL 11 yaw joints (consistent with multitask kernel)
    # qpos indices: 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27
    INV_ENV_WEIGHTS = [
        20.000, 6.897, 4.167, 2.985, 2.326,
        1.905,  1.613, 1.399, 1.235, 1.105, 1.000
    ]
    weighted_sum = sum(INV_ENV_WEIGHTS[i] * float(qpos[7 + 2*i]) for i in range(11))
    chord_angle = weighted_sum / 44.632

    cos_ca = _math.cos(chord_angle)
    sin_ca = _math.sin(chord_angle)
    cfx = body_forward[0] * cos_ca - body_forward[1] * sin_ca
    cfy = body_forward[0] * sin_ca + body_forward[1] * cos_ca
    cfz = body_forward[2]
    chord_len = _math.sqrt(cfx*cfx + cfy*cfy + cfz*cfz)
    if chord_len > 1e-6:
        cfx /= chord_len; cfy /= chord_len; cfz /= chord_len

    # Reward within ±30° (cos >= cos30 ≈ 0.8660); penalize beyond that.
    r_heading = 0.0
    if to_goal_len > 1e-6:
        cos_angle = float(cfx*to_goal[0] + cfy*to_goal[1] + cfz*to_goal[2]) / to_goal_len
        cos_30deg = 0.8660
        if cos_angle >= cos_30deg:
            # Within ±30°: exponential heading reward (same as multitask FORWARD)
            heading_base = w_heading * _cos_heading_exp(cos_angle)
            if dist_improvement > 0.0:
                r_heading = heading_base * (1.0 + 0.5 * dist_improvement)
            else:
                r_heading = heading_base
        else:
            # Beyond ±30°: soft linear penalty (reduced to 30% of w_heading)
            r_heading = -0.3 * w_heading * (cos_30deg - cos_angle) / (1.0 + cos_30deg)

    # --- 3. Forward velocity toward goal (directional_reward_exp) ---
    r_forward = 0.0
    if to_goal_len > 1e-6:
        v_goal = float(np.dot(disp_vel, to_goal)) / to_goal_len
        # Blend forward scale: horizontal goals use full target, vertical goals use half
        fwd_scale = horiz_frac * target_forward_vel + vert_frac * 0.5 * target_forward_vel
        r_forward = w_forward * _directional_reward_exp(v_goal, fwd_scale)

    # --- 4. Adaptive roll penalty ---
    # Blend per-task roll weights using alpha (horiz_frac)
    roll_x_coeff = alpha * 0.60
    roll_y_coeff = alpha * 0.40 + (1.0 - alpha) * 0.05
    r_upright = -w_roll * (
        roll_x_coeff * _upright_penalty_exp(float(body_up[0]), 0.5)
        + roll_y_coeff * _upright_penalty_exp(float(body_up[1]), 0.5)
    )

    # --- 5. Angular velocity smoothness (per-task blended weights) ---
    smooth_wx = alpha * 0.50 + (1.0 - alpha) * 0.30
    smooth_wy = alpha * 0.50 + (1.0 - alpha) * 0.10
    smooth_wz = alpha * 0.15 + (1.0 - alpha) * 0.10
    r_smooth = -w_smooth * (
        smooth_wx * _smooth_penalty_exp(float(omega_world[0]), 1.0)
        + smooth_wy * _smooth_penalty_exp(float(omega_world[1]), 1.0)
        + smooth_wz * _smooth_penalty_exp(float(omega_world[2]), 1.0)
    )

    # --- 6. Off-axis penalty (body-frame decomposition, per-task blended) ---
    offaxis_lat_coeff = alpha * 0.60 + (1.0 - alpha) * 0.45
    offaxis_sec_coeff = alpha * 0.30 + (1.0 - alpha) * 0.15
    r_offaxis = 0.0
    if w_offaxis > 0.0:
        lat_pen = _soft_penalty_exp(v_lateral, target_forward_vel)
        sec_pen = horiz_frac * _soft_penalty_exp(v_vertical, target_vertical_vel) \
                + vert_frac * _soft_penalty_exp(yaw_rate, target_yaw_rate)
        r_offaxis = -w_offaxis * (
            offaxis_lat_coeff * lat_pen
            + offaxis_sec_coeff * sec_pen
        )

    total = r_dist + r_heading + r_forward + r_upright + r_smooth + r_offaxis
    return {
        "r_dist":    r_dist,
        "r_heading": r_heading,
        "r_forward": r_forward,
        "r_upright": r_upright,
        "r_smooth":  r_smooth,
        "r_offaxis": r_offaxis,
        "total":     total,
        "current_dist":      current_dist,
        "dist_improvement":  dist_improvement,
    }


# =============================================================================
# Video overlay drawing
# =============================================================================

def draw_wave_overlay(frame, step, total_steps, preset_name, task_name,
                      reward, reward_components, vel_info):
    """Draw informative overlay on video frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Preset name + task
    task_color = TASK_COLORS.get(task_name, (255, 255, 255))
    cv2.putText(frame, f"PRESET: {preset_name.upper()}", (10, 24),
                font, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, f"TASK: {task_name.upper()}", (10, 50),
                font, 0.6, task_color, 1, cv2.LINE_AA)

    # Progress bar
    progress = min(1.0, step / max(1, total_steps))
    bar_x, bar_y, bar_w, bar_h = w - 220, 10, 200, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h),
                  task_color, -1)
    cv2.putText(frame, f"{step}/{total_steps}", (bar_x + bar_w + 5, bar_y + 11),
                font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    # Velocity info
    if vel_info is not None:
        cv2.putText(frame, f"lbm_vy={vel_info['lbm_vel_y']:+.4f}  "
                    f"qvel_y={vel_info['qvel_y']:+.4f}  "
                    f"v_fwd={vel_info['v_forward']:+.4f}  "
                    f"v_lat={vel_info['v_lateral']:+.4f}",
                    (10, 70), font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"yaw_rate={vel_info['yaw_rate']:+.4f}",
                    (w - 200, 70), font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    # Bottom bar: reward
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 50), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"Step: {step}  |  Reward: {reward:.4f}", (10, h - 30),
                font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    if reward_components:
        comp_str = "  ".join(f"{k}={v:+.3f}" for k, v in reward_components.items())
        cv2.putText(frame, comp_str, (10, h - 10),
                    font, 0.32, (160, 160, 160), 1, cv2.LINE_AA)

    return frame


# =============================================================================
# Velocity extraction helper
# =============================================================================

def quat_rotate_vec(q, v):
    """Rotate vector v by quaternion q=(w,x,y,z)."""
    qw, qx, qy, qz = q
    u = np.array([qx, qy, qz], dtype=np.float64)
    s = float(qw)
    return 2.0 * np.dot(u, v) * u + (s*s - np.dot(u, u)) * v + 2.0 * s * np.cross(u, v)


def extract_velocity_info(qpos, qvel, lbm_pos_now=None, lbm_pos_prev=None, dt=1.0):
    """Extract velocity decomposition from qpos/qvel + LBM displacement.

    Parameters
    ----------
    qpos, qvel : np.ndarray
        MuJoCo root-body state.  qvel[0:3] is the *head* (root freejoint)
        instantaneous world-frame velocity — NOT the whole-body velocity.
    lbm_pos_now, lbm_pos_prev : np.ndarray or None
        Head position in LBM grid coordinates at current and previous step.
        Used to compute displacement-based velocity that reflects real
        whole-body motion (smoothed over one step).
    dt : float
        Time between two consecutive steps (in simulation seconds).
    """
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
    body_right   = quat_rotate_vec([qw, qx, qy, qz], np.array([1.0, 0.0, 0.0]))
    body_forward = quat_rotate_vec([qw, qx, qy, qz], np.array([0.0, 1.0, 0.0]))
    body_up      = quat_rotate_vec([qw, qx, qy, qz], np.array([0.0, 0.0, 1.0]))

    vel_world = qvel[0:3]
    omega_body = qvel[3:6]
    omega_world = quat_rotate_vec([qw, qx, qy, qz], omega_body)

    # Displacement-based velocity from LBM positions (reflects real motion)
    if lbm_pos_now is not None and lbm_pos_prev is not None and dt > 0:
        lbm_vel = (lbm_pos_now - lbm_pos_prev) / dt
    else:
        lbm_vel = np.zeros(3)

    return {
        # LBM displacement-based velocity (real whole-body motion)
        "lbm_vel_x":  float(lbm_vel[0]),
        "lbm_vel_y":  float(lbm_vel[1]),
        "lbm_vel_z":  float(lbm_vel[2]),
        # Head (root freejoint) instantaneous velocity (oscillates with wave)
        "qvel_y":     float(vel_world[1]),
        "v_forward":  float(np.dot(vel_world, body_forward)),
        "v_lateral":  float(np.dot(vel_world, body_right)),
        "v_vertical": float(vel_world[2]),
        "yaw_rate":   float(omega_world[2]),
        "roll_rate":  float(omega_world[0]),
        "pitch_rate": float(omega_world[1]),
    }


# =============================================================================
# Main simulation loop
# =============================================================================

def run_wave_test(env, preset_name, action_array, task_name, total_steps,
                  mujoco_renderer, mujoco_only=True,
                  render_type="vorticity", view_mode="topdown",
                  with_fluid_force=False, warmup_steps=20,
                  reward_mode="multitask"):
    """
    Run open-loop wave control through LBM environment and collect data.

    Parameters
    ----------
    warmup_steps : int
        Number of initial steps over which the action amplitude is linearly
        ramped from 0 to the target value.  This prevents LBM numerical
        divergence caused by sudden large deformations at t=0.

    Returns:
        frames: list of video frames
        metrics: dict with per-step data
    """
    base_env = env._env

    # Set task for the environment
    from envs.lbm3d.manta.manta_multitask_env_3d import TASK_NAMES as MT_TASK_NAMES
    task_id = MT_TASK_NAMES.index(task_name)
    base_env._task_ids[:] = task_id
    base_env._update_task_ids_wp()

    # Reset environment
    obs = env.reset()

    # Re-set task after reset (reset may re-sample tasks)
    base_env._task_ids[:] = task_id
    base_env._update_task_ids_wp()

    # For dirdist mode, re-sample goal in the correct task direction after reset
    goal_info_str = ""
    if reward_mode == "dirdist" and hasattr(base_env, 'use_direction_dist_tasks') and base_env.use_direction_dist_tasks:
        base_env._sample_direction_distance_goals()
        goal_pos = base_env._direction_goal_positions_np[0]
        body_pos_norm = base_env._body_positions_normalized()[0]
        goal_dist = np.linalg.norm(goal_pos - body_pos_norm)
        goal_info_str = f"goal=({goal_pos[0]:.3f},{goal_pos[1]:.3f},{goal_pos[2]:.3f}) dist={goal_dist:.4f}"

    # Prepare action: repeat for nworld=1
    action_target = action_array.reshape(1, -1)  # (1, action_dim)

    # ── Flipper force tracking (turtle only) ──────────────────────────────
    flipper_fr_idx = None
    flipper_fl_idx = None
    if hasattr(base_env, 'dynamic_link_config'):
        for i, cfg in enumerate(base_env.dynamic_link_config):
            if cfg['link_name'] == 'flipper_FR':
                flipper_fr_idx = i
            elif cfg['link_name'] == 'flipper_FL':
                flipper_fl_idx = i
    has_flipper_forces = (flipper_fr_idx is not None and flipper_fl_idx is not None)
    if has_flipper_forces:
        print(f"  Flipper force tracking enabled: FR_idx={flipper_fr_idx}, FL_idx={flipper_fl_idx}")
        # Also print all dynamic body names for reference
        dyn_names = [c['link_name'] for c in base_env.dynamic_link_config]
        print(f"  Dynamic bodies: {dyn_names}")

    # Data collection
    step_data = []
    mujoco_frames = []
    lbm_frames_raw = []
    total_reward = 0.0
    prev_lbm_pos = None  # for displacement-based velocity

    pbar = tqdm(total=total_steps, desc=f"{preset_name} [{task_name}]")

    for step in range(1, total_steps + 1):
        # Warm-up ramp: linearly increase action amplitude over warmup_steps
        if warmup_steps > 0 and step <= warmup_steps:
            ramp = step / warmup_steps
            action_2d = action_target * ramp
        else:
            action_2d = action_target

        obs_list, rewards, dones, infos = env.step(action_2d)
        reward = float(rewards[0])
        total_reward += reward

        # Extract state
        qpos = base_env.mjw_data.qpos.numpy()[0].copy()
        qvel = base_env.mjw_data.qvel.numpy()[0].copy()

        # Get body position from LBM solver (unnormalized for velocity calc)
        pos_raw = base_env.lbm_solver.flows[0].solid_position.numpy()[0].copy()
        lbm_pos_now = np.array(pos_raw, dtype=np.float64)
        body_pos = [pos_raw[0] / base_env.nx, pos_raw[1] / base_env.ny, pos_raw[2] / base_env.nz]

        # Compute velocity using LBM position displacement
        sim_dt = base_env.mj_model.opt.timestep * getattr(base_env, 'per_frame_steps', 1)
        vel_info = extract_velocity_info(qpos, qvel, lbm_pos_now, prev_lbm_pos, dt=sim_dt)
        prev_lbm_pos = lbm_pos_now.copy()

        # ── Extract flipper forces (turtle only) ────────────────────────
        flipper_force_info = {}
        if has_flipper_forces:
            forces_np = base_env.forces_buffer.numpy()[0]  # (n_dynamic, 3)
            torques_np = base_env.torques_buffer.numpy()[0]  # (n_dynamic, 3)

            fr_force = forces_np[flipper_fr_idx]  # [fx, fy, fz] in N
            fl_force = forces_np[flipper_fl_idx]  # [fx, fy, fz] in N
            fr_torque = torques_np[flipper_fr_idx]
            fl_torque = torques_np[flipper_fl_idx]
            combined_force = fr_force + fl_force
            combined_torque = fr_torque + fl_torque
            combined_mag = float(np.linalg.norm(combined_force))

            flipper_force_info = {
                # Front-Right flipper
                "FR_fx": float(fr_force[0]),
                "FR_fy": float(fr_force[1]),
                "FR_fz": float(fr_force[2]),
                "FR_f_mag": float(np.linalg.norm(fr_force)),
                "FR_tx": float(fr_torque[0]),
                "FR_ty": float(fr_torque[1]),
                "FR_tz": float(fr_torque[2]),
                # Front-Left flipper
                "FL_fx": float(fl_force[0]),
                "FL_fy": float(fl_force[1]),
                "FL_fz": float(fl_force[2]),
                "FL_f_mag": float(np.linalg.norm(fl_force)),
                "FL_tx": float(fl_torque[0]),
                "FL_ty": float(fl_torque[1]),
                "FL_tz": float(fl_torque[2]),
                # Combined (both front flippers)
                "combined_fx": float(combined_force[0]),
                "combined_fy": float(combined_force[1]),
                "combined_fz": float(combined_force[2]),
                "combined_f_mag": combined_mag,
                # Semantic aliases
                "lift_force": float(combined_force[2]),    # Z-up
                "thrust_force": float(combined_force[1]),   # Y-forward
                "lateral_force": float(combined_force[0]),  # X-lateral
            }

        # ── Eel goal-reward component decomposition (dirdist mode only) ────
        eel_reward_comps = {}
        if (reward_mode == "dirdist"
                and hasattr(base_env, 'use_direction_dist_tasks')
                and base_env.use_direction_dist_tasks
                and hasattr(base_env, 'direction_dist_w_dist')):
            goal_pos_now = base_env._direction_goal_positions_np[0]
            body_pos_norm_now = np.array(body_pos, dtype=np.float64)
            # prev_dist: use the distance stored before this step's kernel ran
            # (base_env._prev_dist_wp was already updated by env.step, so we
            #  use current_dist from the last step, stored in step_data[-1] if available)
            if step_data:
                prev_dist_val = step_data[-1].get("goal_dist", float(np.linalg.norm(body_pos_norm_now - goal_pos_now)))
            else:
                prev_dist_val = float(np.linalg.norm(body_pos_norm_now - goal_pos_now))
            eel_reward_comps = compute_eel_goal_reward_components(
                qpos=qpos,
                qvel=qvel,
                disp_vel=np.array([vel_info["lbm_vel_x"], vel_info["lbm_vel_y"], vel_info["lbm_vel_z"]], dtype=np.float64),
                body_pos_norm=body_pos_norm_now,
                goal_pos=goal_pos_now,
                prev_dist=prev_dist_val,
                w_dist=base_env.direction_dist_w_dist,
                w_roll=base_env.direction_dist_w_roll,
                w_heading=base_env.direction_dist_w_heading,
                w_forward=base_env.direction_dist_w_forward,
                w_smooth=base_env.direction_dist_w_smooth,
                w_offaxis=base_env.direction_dist_w_offaxis,
                target_forward_vel=getattr(base_env, 'target_forward_vel', 0.1),
                target_yaw_rate=getattr(base_env, 'target_yaw_rate', 0.5),
                target_vertical_vel=getattr(base_env, 'target_vertical_vel', 0.05),
            )

        # Record step data
        # Extract actuator ctrl values for diagnostics (rotation = attack angle)
        ctrl_np = base_env.mjw_data.ctrl.numpy()[0].copy()
        ctrl_info = {
            "FR_rot_ctrl": float(ctrl_np[0]),   # front-right rotation (attack angle)
            "FR_flap_ctrl": float(ctrl_np[1]),  # front-right flap
            "FL_rot_ctrl": float(ctrl_np[2]),   # front-left rotation
            "FL_flap_ctrl": float(ctrl_np[3]),  # front-left flap
        }
        step_record = {
            "step": step,
            "reward": reward,
            "cumulative_reward": total_reward,
            "pos_x": body_pos[0],
            "pos_y": body_pos[1],
            "pos_z": body_pos[2],
            **vel_info,
            **flipper_force_info,
            **ctrl_info,
        }
        # Add goal distance info for dirdist mode
        if reward_mode == "dirdist" and hasattr(base_env, 'use_direction_dist_tasks') and base_env.use_direction_dist_tasks:
            goal_pos = base_env._direction_goal_positions_np[0]
            cur_dist = np.linalg.norm(np.array(body_pos) - goal_pos)
            step_record["goal_dist"] = cur_dist
            step_record["goal_x"] = goal_pos[0]
            step_record["goal_y"] = goal_pos[1]
            step_record["goal_z"] = goal_pos[2]
        # Add eel reward components
        if eel_reward_comps:
            for k, v in eel_reward_comps.items():
                step_record[f"eel_{k}"] = v
        step_data.append(step_record)

        # Render frames
        reward_components = {"r": reward}
        if reward_mode == "dirdist" and "goal_dist" in step_record:
            reward_components["d"] = step_record["goal_dist"]
        if eel_reward_comps:
            reward_components["r_dist"]    = eel_reward_comps["r_dist"]
            reward_components["r_hdg"]     = eel_reward_comps["r_heading"]
            reward_components["r_fwd"]     = eel_reward_comps["r_forward"]
            reward_components["r_roll"]    = eel_reward_comps["r_upright"]
            reward_components["r_smo"]     = eel_reward_comps["r_smooth"]
        try:
            if mujoco_only:
                frame = get_mujoco_frame(env, mujoco_renderer, world_idx=0,
                                         with_fluid_force=with_fluid_force)
                frame = draw_wave_overlay(
                    frame, step, total_steps, preset_name, task_name,
                    total_reward, reward_components, vel_info,
                )
                mujoco_frames.append(frame)
            else:
                lbm_raw = get_raw_frame_3d(env, world_idx=0,
                                           render_type=render_type, view_mode=view_mode)
                lbm_frames_raw.append(lbm_raw)

                mj_frame = get_mujoco_frame(env, mujoco_renderer, world_idx=0,
                                            with_fluid_force=with_fluid_force)
                mj_frame = draw_wave_overlay(
                    mj_frame, step, total_steps, preset_name, task_name,
                    total_reward, reward_components, vel_info,
                )
                mujoco_frames.append(mj_frame)
        except Exception as e:
            print(f"Warning: failed to render frame at step {step}: {e}")

        # Check termination
        if np.any(dones):
            info0 = infos[0] if isinstance(infos, list) else infos
            reason = info0.get("term_reason", "unknown")
            if isinstance(reason, list):
                reason = reason[0]
            print(f"\n[TERMINATED] step={step}, reason={reason}")
            break

        pbar.update(1)
        postfix_dict = dict(reward=f"{total_reward:.3f}",
                            lbm_vy=f"{vel_info['lbm_vel_y']:.4f}")
        if has_flipper_forces:
            postfix_dict["lift"] = f"{flipper_force_info['lift_force']:+.4f}"
            postfix_dict["thrust"] = f"{flipper_force_info['thrust_force']:+.4f}"
        pbar.set_postfix(**postfix_dict)

    pbar.close()

    # Process LBM frames if needed
    if not mujoco_only and lbm_frames_raw:
        all_lbm = np.stack(lbm_frames_raw)
        if render_type == "vorticity":
            fluid_mask = all_lbm < 999.0
            vmax = np.max(np.abs(all_lbm[fluid_mask])) if np.any(fluid_mask) else 1.0
            global_vmax = vmax * 0.2 + 1e-8
        else:
            global_vmax = np.max(all_lbm) * 0.6 + 1e-8
        lbm_frames = [process_raw_to_frame(r, global_vmax, render_type) for r in lbm_frames_raw]
        # Left: MuJoCo (60%), Right: LBM flow (40%)
        frames = combine_frames_left_right(mujoco_frames, lbm_frames, left_ratio=0.6)
    else:
        frames = mujoco_frames

    metrics = {
        "step_data": step_data,
        "total_reward": total_reward,
        "total_steps": len(step_data),
        "preset_name": preset_name,
        "task_name": task_name,
        "reward_mode": reward_mode,
        "goal_info": goal_info_str,
    }

    return frames, metrics


def save_metrics_csv(filepath, step_data, action_keys, action_values):
    """Save per-step metrics to CSV."""
    if not step_data:
        return

    fieldnames = list(step_data[0].keys())
    # Add action columns
    for k in action_keys:
        fieldnames.append(f"action_{k}")

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in step_data:
            # Add constant action values to each row
            for k, v in zip(action_keys, action_values):
                row[f"action_{k}"] = f"{v:.4f}"
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in row.items()})

    print(f"  CSV saved: {filepath}")


def print_summary(metrics, action_keys, action_values):
    """Print a summary table of the test run."""
    sd = metrics["step_data"]
    if not sd:
        print("  No data collected.")
        return

    n = len(sd)
    print(f"\n{'='*70}")
    print(f"  Preset: {metrics['preset_name']}  |  Task: {metrics['task_name']}  |  Steps: {n}")
    print(f"{'='*70}")

    # Action parameters
    print("  Action parameters:")
    for k, v in zip(action_keys, action_values):
        print(f"    {k:>14s} = {v:+.3f}")

    # Velocity statistics (last 50% of steps for steady-state)
    steady_start = max(1, n // 2)
    steady = sd[steady_start:]

    def stat(key):
        vals = [r[key] for r in steady]
        return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

    print(f"\n  Steady-state velocity (steps {steady_start+1}–{n}):")
    for key in ["lbm_vel_y", "lbm_vel_x", "lbm_vel_z", "qvel_y", "v_forward", "v_lateral", "v_vertical", "yaw_rate"]:
        mean, std, vmin, vmax = stat(key)
        print(f"    {key:>14s}: mean={mean:+.5f}  std={std:.5f}  range=[{vmin:+.5f}, {vmax:+.5f}]")

    # Position displacement
    p0 = sd[0]
    pf = sd[-1]
    dx = pf["pos_x"] - p0["pos_x"]
    dy = pf["pos_y"] - p0["pos_y"]
    dz = pf["pos_z"] - p0["pos_z"]
    print(f"\n  Position displacement (normalized):")
    print(f"    Δx={dx:+.4f}  Δy={dy:+.4f}  Δz={dz:+.4f}  |Δ|={np.sqrt(dx*dx+dy*dy+dz*dz):.4f}")

    # Goal distance statistics (dirdist mode only)
    if metrics.get("reward_mode") == "dirdist" and "goal_dist" in sd[0]:
        goal_info = metrics.get("goal_info", "")
        if goal_info:
            print(f"\n  Goal info: {goal_info}")
        goal_dists = [r["goal_dist"] for r in sd]
        print(f"  Goal distance:")
        print(f"    initial={goal_dists[0]:.4f}  final={goal_dists[-1]:.4f}  "
              f"min={np.min(goal_dists):.4f}  reduction={goal_dists[0]-goal_dists[-1]:+.4f}")

    # Flipper force statistics (turtle only)
    if "lift_force" in sd[0]:
        print(f"\n  Front-flipper forces (steady-state, steps {steady_start+1}–{n}):")
        for key in ["lift_force", "thrust_force", "lateral_force",
                    "combined_f_mag",
                    "FR_fz", "FL_fz", "FR_fy", "FL_fy",
                    "FR_f_mag", "FL_f_mag"]:
            mean, std, vmin, vmax = stat(key)
            print(f"    {key:>18s}: mean={mean:+.6f}  std={std:.6f}  "
                  f"range=[{vmin:+.6f}, {vmax:+.6f}]")

        # Per-cycle analysis: detect sign changes in combined_fz to find half-cycles
        fz_vals = [r["combined_fz"] for r in steady]
        positive_fz = [v for v in fz_vals if v > 0]
        negative_fz = [v for v in fz_vals if v < 0]
        if positive_fz and negative_fz:
            print(f"\n  Lift asymmetry analysis (steady-state):")
            print(f"    Upward  phases: {len(positive_fz):>4d} steps, "
                  f"mean_fz={np.mean(positive_fz):+.6f}, sum={np.sum(positive_fz):+.6f}")
            print(f"    Downward phases: {len(negative_fz):>4d} steps, "
                  f"mean_fz={np.mean(negative_fz):+.6f}, sum={np.sum(negative_fz):+.6f}")
            net_impulse = np.sum(fz_vals)
            print(f"    Net Z-impulse (sum of fz): {net_impulse:+.6f}")
            print(f"    → {'NET UPWARD' if net_impulse > 0 else 'NET DOWNWARD'} lift")

    # Actuator ctrl diagnostics (attack angle)
    if "FR_rot_ctrl" in sd[0]:
        print(f"\n  Actuator ctrl (steady-state, steps {steady_start+1}–{n}):")
        for key in ["FR_rot_ctrl", "FR_flap_ctrl", "FL_rot_ctrl", "FL_flap_ctrl"]:
            vals = [r[key] for r in steady]
            mean_v, std_v = np.mean(vals), np.std(vals)
            vmin, vmax = np.min(vals), np.max(vals)
            print(f"    {key:>18s}: mean={mean_v:+.6f}  std={std_v:.6f}  "
                  f"range=[{vmin:+.6f}, {vmax:+.6f}]")
        # Show attack angle asymmetry: compare AoA during upstroke vs downstroke
        fr_rot = [r["FR_rot_ctrl"] for r in steady]
        fr_flap = [r["FR_flap_ctrl"] for r in steady]
        # Detect flap direction: positive flap_ctrl derivative = upstroke
        flap_arr = np.array(fr_flap)
        rot_arr = np.array(fr_rot)
        flap_diff = np.diff(flap_arr)
        # upstroke: flap increasing; downstroke: flap decreasing
        up_mask = flap_diff > 0
        down_mask = flap_diff < 0
        if np.any(up_mask) and np.any(down_mask):
            rot_up = np.abs(rot_arr[:-1][up_mask])
            rot_down = np.abs(rot_arr[:-1][down_mask])
            print(f"\n  Attack angle asymmetry (|FR_rot| during strokes):")
            print(f"    Upstroke:   mean_|AoA|={np.mean(rot_up):+.6f}  "
                  f"({np.sum(up_mask)} steps)")
            print(f"    Downstroke: mean_|AoA|={np.mean(rot_down):+.6f}  "
                  f"({np.sum(down_mask)} steps)")
            ratio = np.mean(rot_up) / max(np.mean(rot_down), 1e-8)
            print(f"    AoA ratio (up/down): {ratio:.2f}x")

    # Reward statistics
    rewards = [r["reward"] for r in sd]
    print(f"\n  Reward ({metrics.get('reward_mode', 'multitask')}):\n"
          f"    total={metrics['total_reward']:+.4f}  mean={np.mean(rewards):+.5f}  "
          f"std={np.std(rewards):.5f}")
    print(f"    steady-state mean={np.mean([r['reward'] for r in steady]):+.5f}")

    # Eel goal-reward component statistics (dirdist mode)
    if "eel_r_dist" in sd[0]:
        print(f"\n  Eel goal-reward components (steady-state, steps {steady_start+1}–{n}):")
        comp_keys = [
            ("eel_r_dist",    "r_dist   (distance improvement)"),
            ("eel_r_heading", "r_heading (chord-angle alignment)"),
            ("eel_r_forward", "r_forward (vel toward goal)"),
            ("eel_r_upright", "r_upright (adaptive roll penalty)"),
            ("eel_r_smooth",  "r_smooth  (angular vel smoothness)"),
            ("eel_r_offaxis", "r_offaxis (off-axis penalty)"),
            ("eel_total",     "total     (sum of components)"),
        ]
        for key, label in comp_keys:
            if key in sd[0]:
                vals = [r[key] for r in steady]
                print(f"    {label:<40s}: mean={np.mean(vals):+.5f}  "
                      f"std={np.std(vals):.5f}  "
                      f"range=[{np.min(vals):+.5f}, {np.max(vals):+.5f}]")
        # dist_improvement distribution
        if "eel_dist_improvement" in sd[0]:
            di_vals = [r["eel_dist_improvement"] for r in steady]
            pos_frac = sum(1 for v in di_vals if v > 0) / max(len(di_vals), 1)
            print(f"    {'dist_improvement > 0':<40s}: {pos_frac*100:.1f}% of steady steps")

    print(f"{'='*70}")

# =============================================================================
# Main
# =============================================================================

def main():
    total_start = time.time()

    parser = argparse.ArgumentParser(
        description="LBM Wave Tester — validate reward design with real fluid simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--animal", type=str, default="tuna",
                        choices=list(ANIMAL_REGISTRY.keys()),
                        help="Animal type (default: tuna)")
    parser.add_argument("--preset", type=str, default="forward",
                        help="Preset name or 'all' to run all presets")
    parser.add_argument("--task", type=str, default=None,
                        help="Task to evaluate reward for (default: auto-match preset)")
    parser.add_argument("--steps", type=int, default=300,
                        help="Number of simulation steps per preset")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs/lbm_wave_test/<animal>)")
    parser.add_argument("--override", type=str, default=None,
                        help="Override preset params: 'key=val,key=val,...'")

    # Video options
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=1.9)
    parser.add_argument("--camera-azimuth", type=float, default=45)
    parser.add_argument("--camera-elevation", type=float, default=-45)
    parser.add_argument("--with-lbm", action="store_true",
                        help="Include LBM flow visualization in video")
    parser.add_argument("--render-type", type=str, default="vorticity",
                        choices=["velocity", "vorticity"])
    parser.add_argument("--view-mode", type=str, default="topdown",
                        choices=["topdown", "max_topdown", "side", "front"])
    parser.add_argument("--show-fluid-force", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=20,
                        help="Number of warm-up steps to ramp action from 0 to target (default: 20)")
    parser.add_argument("--reward-mode", type=str, default="multitask",
                        choices=["multitask", "dirdist"],
                        help="Reward mode: 'multitask' (default task-conditioned) or "
                             "'dirdist' (nospeed direction-distance navigation reward)")

    # Config overrides
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Override config sections (default: from registry)")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    # ── Resolve animal config ─────────────────────────────────────────────
    animal_info = ANIMAL_REGISTRY[args.animal]
    reward_mode = args.reward_mode

    # Select configs based on reward mode
    if reward_mode == "dirdist":
        dirdist_configs = animal_info.get("dirdist_configs")
        if dirdist_configs is None:
            print(f"Error: Animal '{args.animal}' does not support --reward-mode dirdist.")
            print(f"Only these animals support dirdist: "
                  f"{[k for k, v in ANIMAL_REGISTRY.items() if v.get('dirdist_configs')]}")
            return
        configs = args.configs or dirdist_configs
    else:
        configs = args.configs or animal_info["configs"]

    presets = animal_info["presets"]
    action_keys = animal_info["action_keys"]
    preset_to_action = animal_info["preset_to_action"]

    # ── Resolve preset list ───────────────────────────────────────────────
    if args.preset == "all":
        preset_names = list(presets.keys())
    else:
        if args.preset not in presets:
            print(f"Error: Unknown preset '{args.preset}' for {args.animal}.")
            print(f"Available: {list(presets.keys())}")
            return
        preset_names = [args.preset]

    # ── Auto-match task to preset ─────────────────────────────────────────
    PRESET_TASK_MAP = {
        "cruise": "forward", "fast": "forward", "powerstroke": "forward",
        "glide": "forward", "freeze": "forward",
        "forward": "forward", "tail_only": "forward",
        "cold_start": "forward", "reverse": "forward",
        "head_tail_swing": "forward",
        "turn_l": "turn_left", "turn_r": "turn_right",
        "ascend": "ascend", "descend": "descend",
    }

    # ── Output directory ──────────────────────────────────────────────────
    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    else:
        suffix = "_dirdist" if reward_mode == "dirdist" else ""
        output_dir = pathlib.Path("outputs") / "lbm_wave_test" / f"{args.animal}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ───────────────────────────────────────────────────────
    config_overrides = {
        "device": args.device,
        "compile": False,
        "video_pred_log": False,
        "time_limit": args.steps + 200,
        "task_switch_interval": 0,
        "control_mode": animal_info["control_mode"],
    }
    config = load_named_config(["defaults", *configs], overrides=config_overrides)
    config.time_limit = (args.steps + 200) // getattr(config, "action_repeat", 1)

    mujoco_only = not args.with_lbm

    # ── Print banner ──────────────────────────────────────────────────────
    print("=" * 70)
    print("  LBM Wave Tester — Real Fluid Simulation Reward Validation")
    print("=" * 70)
    print(f"  Animal:  {args.animal}")
    print(f"  Configs: {configs}")
    print(f"  Control: {animal_info['control_mode']}")
    print(f"  Presets: {preset_names}")
    print(f"  Steps:   {args.steps} per preset")
    print(f"  Output:  {output_dir}")
    print(f"  LBM viz: {'yes' if args.with_lbm else 'no (MuJoCo only)'}")
    print(f"  Warmup:  {args.warmup_steps} steps (action ramp 0→target)")
    print(f"  Reward:  {reward_mode}")
    print("=" * 70)

    # ── Create environment (once, reuse for all presets) ──────────────────
    print("\nCreating LBM environment...")
    env = make_multitask_env(config, nworld=1)
    base_env = env._env
    obs_space = env.observation_space
    act_space = env.action_space
    print(f"  Obs: {obs_space}, Act: {act_space}")

    # ── Create MuJoCo renderer ────────────────────────────────────────────
    mujoco_renderer = MuJoCoRenderer(
        base_env.mj_model,
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        show_position=True,
        show_fluid_force=args.show_fluid_force,
    )

    # ── Run each preset ───────────────────────────────────────────────────
    all_summaries = []

    for preset_name in preset_names:
        params = dict(presets[preset_name])  # copy

        # Apply overrides
        if args.override:
            for pair in args.override.split(","):
                k, v = pair.strip().split("=")
                if k in params:
                    params[k] = float(v)
                else:
                    print(f"  Warning: override key '{k}' not in preset params, ignoring.")

        # Resolve task
        task_name = args.task or PRESET_TASK_MAP.get(preset_name, "forward")

        # Convert to action array
        action_array = preset_to_action(params)
        action_values = [params[k] for k in action_keys]

        print(f"\n{'─'*70}")
        print(f"  Running preset: {preset_name}  →  task: {task_name}")
        print(f"  Action: {dict(zip(action_keys, action_values))}")
        print(f"{'─'*70}")

        # Run simulation
        frames, metrics = run_wave_test(
            env, preset_name, action_array, task_name, args.steps,
            mujoco_renderer, mujoco_only=mujoco_only,
            render_type=args.render_type, view_mode=args.view_mode,
            with_fluid_force=args.show_fluid_force,
            warmup_steps=args.warmup_steps,
            reward_mode=reward_mode,
        )

        # Print summary
        print_summary(metrics, action_keys, action_values)
        all_summaries.append(metrics)

        # Save video
        video_path = output_dir / f"{preset_name}_{task_name}.mp4"
        if frames:
            print(f"\n  Saving video ({len(frames)} frames)...")
            save_video(frames, video_path, fps=args.fps)

        # Save CSV
        csv_path = output_dir / f"{preset_name}_{task_name}.csv"
        save_metrics_csv(csv_path, metrics["step_data"], action_keys, action_values)

    # ── Final comparison table (if multiple presets) ──────────────────────
    if len(all_summaries) > 1:
        print(f"\n\n{'='*70}")
        print("  COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"  {'Preset':<14s} {'Task':<12s} {'Steps':>6s} {'Reward':>10s} "
              f"{'v_y_mean':>10s} {'v_vert':>10s} {'yaw_rate':>10s}")
        print(f"  {'-'*14} {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for m in all_summaries:
            sd = m["step_data"]
            n = len(sd)
            steady_start = max(1, n // 2)
            steady = sd[steady_start:]

            v_y_mean = np.mean([r["lbm_vel_y"] for r in steady])
            v_vert_mean = np.mean([r["v_vertical"] for r in steady])
            yaw_mean = np.mean([r["yaw_rate"] for r in steady])

            print(f"  {m['preset_name']:<14s} {m['task_name']:<12s} {n:>6d} "
                  f"{m['total_reward']:>+10.3f} {v_y_mean:>+10.5f} "
                  f"{v_vert_mean:>+10.5f} {yaw_mean:>+10.5f}")

        print(f"{'='*70}")

    # ── Cleanup ───────────────────────────────────────────────────────────
    mujoco_renderer.close()
    env.close()

    total_time = time.time() - total_start
    print(f"\nTotal runtime: {total_time:.1f}s")
    print(f"Output directory: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
