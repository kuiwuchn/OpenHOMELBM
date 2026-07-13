"""
3D Tuna Multi-Task LBM Environment with optional PD control modes.

Multi-task training for tuna locomotion skills.
Inherits from Tuna3DLBMEnv but replaces goal-reaching with velocity-based
task-conditioned rewards.

5 Tasks:
  0: FORWARD    — swim along +Y (body forward)
  1: TURN_LEFT  — yaw left while maintaining gentle forward cruise
  2: TURN_RIGHT — yaw right while maintaining gentle forward cruise
  3: ASCEND     — swim upward (+Z) with slight forward cruise
  4: DESCEND    — swim downward (-Z) with slight forward cruise

=== Frequency-Domain PD Control ===

Instead of directly outputting joint targets, the neural network outputs
frequency-domain parameters (amplitude A and phase C) for each joint group:

  θ_i* = Σ_{j=1}^{K} A_{ij} * sin(π/2 * B_j * t + C_{ij})

where B_j = j * B̄ / K, B̄ = max frequency (default 1.0).

MuJoCo's built-in position actuators with PD gains then track θ_i*:
  τ_i = -[kp * (θ_i - θ_i*) + kd * θ̇_i]

=== Reduced-Order Model ===

For smooth motion, joints within a functional group share control signals:
  Group 0: Tail yaw         (t1_yaw … t4_yaw, carangiform wave)    — stride=5: [ω, A1,C1, A2,C2]
  Group 1: Tail pitch       (t1_pitch … t4_pitch, vertical ctrl)   — stride=5: [ω, A1,C1, A2,C2]

Total action dim: 2 groups × 5 = 10 (N_groups=2, K=2 harmonics)

=== Wave Control (7-dim) ===

Single tail oscillator group + fin stabilizer (independent frequency):
  ── Tail (4 params) ──
  [0]  A_tail      tail yaw wave amplitude
  [1]  omega_tail  tail oscillation frequency
  [2]  yaw_bias    tail yaw DC offset (turning)
  [3]  tail_pitch  tail pitch bias (ascend/descend + propeller)
  ── Fins (3 params) ──
  [4]  fin_amp     fin flap amplitude (anti-roll stabilization)
  [5]  fin_asym    fin left/right asymmetry (roll correction)
  [6]  fin_freq    fin independent frequency (decoupled from tail)

Observation: base obs (22 + 3*n_joints) + lbm_pos(3) + task one-hot (5) + phase (2)
The agent learns a single policy conditioned on task ID.

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

from .tuna_lbm_env_3d import (
    Tuna3DLBMEnv,
    compute_tuna_obs_3d_kernel,
    check_boundary_3d_tuna_kernel,
    check_stability_3d_tuna_kernel,
    apply_instability_penalty_tuna_kernel,
)
from ..lbm_core_3d import HomeFlow3D

# Reuse generic multi-task Warp kernels from manta (they only depend on qpos/qvel)
from ..manta.manta_multitask_env_3d import (
    compute_multitask_obs_kernel,
    quat_rotate_vec,
    dot_vec3,
    directional_reward,
    positive_reward,
    soft_penalty,
    TASK_FORWARD,
    TASK_TURN_LEFT,
    TASK_TURN_RIGHT,
    TASK_ASCEND,
    TASK_DESCEND,
    NUM_TASKS,
    TASK_NAMES,
)


# ============== Tuna-specific Reward Kernel ==============
# Tuna has 4 pairs of fins that can directly generate vertical thrust,
# so we use the same 85/15 vertical/forward split as Manta.
# Key differences from Manta:
#   - Higher roll penalty (0.15 vs 0.10): slender body is more roll-prone
#   - Higher smooth penalty (0.025 vs 0.015): 16-DOF serial chain amplifies end-effector
#   - Higher offaxis penalty (0.03 vs 0.02): tail swing causes lateral drift

@wp.kernel
def compute_tuna_multitask_reward_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    disp_vel: wp.array2d(dtype=wp.float32),  # (nworld, 3) LBM displacement-based velocity
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
    Tuna-specific task-conditioned reward.

    Key design choices:
      - ASCEND/DESCEND use 85/15 vertical/forward split (same as Manta)
        because tuna has 4 pairs of fins that can directly generate vertical thrust.
      - Pitch posture penalty during ASCEND/DESCEND is standard (0.25)
        since fins provide direct vertical force without needing body tilt.
      - Higher roll penalty than Manta: slender fusiform body is more roll-prone.
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

    # Use displacement velocity for directional rewards (immune to oscillation noise)
    v_world_y  = dv[1]
    v_forward = dot_vec3(dv, body_forward)
    v_lateral = dot_vec3(dv, body_right)
    v_vertical = dv[2]
    yaw_rate = omega_world[2]

    upright_roll = body_up[0] * body_up[0]
    upright_pitch = body_up[1] * body_up[1]

    r_task = float(0.0)
    r_upright = float(0.0)
    r_offaxis = float(0.0)

    turn_forward_scale = 0.5 * target_forward_vel
    vertical_forward_scale = 0.5 * target_forward_vel

    if task == TASK_FORWARD:
        r_task = w_task * directional_reward(v_world_y, target_forward_vel)
        r_upright = -w_roll * (0.50 * upright_roll + 0.50 * upright_pitch)
        # Lateral penalty reduced (0.60→0.30): tail yaw oscillation inevitably produces
        # some lateral force; over-penalizing it suppresses the tail stroke entirely.
        r_offaxis = -w_offaxis * (
            0.30 * soft_penalty(v_lateral, target_forward_vel)
            + 0.15 * soft_penalty(v_vertical, target_vertical_vel)
            + 0.15 * soft_penalty(yaw_rate, target_yaw_rate)
        )

    elif task == TASK_TURN_LEFT:
        r_task = w_task * (
            0.85 * directional_reward(yaw_rate, target_yaw_rate)
            + 0.15 * positive_reward(v_forward, turn_forward_scale)
        )
        r_upright = -w_roll * (0.10 * upright_roll + 0.40 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.45 * soft_penalty(v_lateral, target_forward_vel)
            + 0.15 * soft_penalty(v_vertical, target_vertical_vel)
        )

    elif task == TASK_TURN_RIGHT:
        r_task = w_task * (
            0.85 * directional_reward(-yaw_rate, target_yaw_rate)
            + 0.15 * positive_reward(v_forward, turn_forward_scale)
        )
        r_upright = -w_roll * (0.10 * upright_roll + 0.40 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.45 * soft_penalty(v_lateral, target_forward_vel)
            + 0.15 * soft_penalty(v_vertical, target_vertical_vel)
        )

    elif task == TASK_ASCEND:
        # 85% vertical + 15% forward (same as Manta: fins provide direct vertical thrust)
        r_task = w_task * (
            0.85 * directional_reward(v_vertical, target_vertical_vel)
            + 0.15 * positive_reward(v_forward, vertical_forward_scale)
        )
        # Standard pitch penalty (fins provide direct vertical force)
        r_upright = -w_roll * (0.15 * upright_roll + 0.25 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.45 * soft_penalty(v_lateral, target_forward_vel)
            + 0.15 * soft_penalty(yaw_rate, target_yaw_rate)
        )

    elif task == TASK_DESCEND:
        # Same 85/15 split as ASCEND
        r_task = w_task * (
            0.85 * directional_reward(-v_vertical, target_vertical_vel)
            + 0.15 * positive_reward(v_forward, vertical_forward_scale)
        )
        # Same standard pitch penalty
        r_upright = -w_roll * (0.15 * upright_roll + 0.25 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.45 * soft_penalty(v_lateral, target_forward_vel)
            + 0.15 * soft_penalty(yaw_rate, target_yaw_rate)
        )

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


# ============== Frequency-Domain PD Control Constants ==============

# Number of harmonics (frequency components)
DEFAULT_K_HARMONICS = 2

# Max frequency (B̄). B_j = j * B_BAR / K
DEFAULT_B_BAR = 1.0

# Reduced-order joint groups for Tuna (16 joints → 6 groups)
# Each group: (group_name, [actuator_indices])
# Actuator layout (14 DOFs):
#   0: fin_right_pitch        (front-right pectoral fin pitch)
#   1: fin_left_pitch         (front-left pectoral fin pitch)
#   2: fin_rear_right_pitch   (rear-right pelvic fin pitch)
#   3: fin_rear_left_pitch    (rear-left pelvic fin pitch)
#   4: t0_yaw                 (tail segment 0 yaw — passive anchor)
#   5: t0_pitch               (tail segment 0 pitch — passive anchor)
#   6: t1_yaw                 (tail segment 1 yaw)
#   7: t1_pitch               (tail segment 1 pitch)
#   8: t2_yaw                 (tail segment 2 yaw)
#   9: t2_pitch               (tail segment 2 pitch)
#  10: t3_yaw                 (tail segment 3 yaw)
#  11: t3_pitch               (tail segment 3 pitch)
#  12: t4_yaw                 (tail segment 4 yaw)
#  13: t4_pitch               (tail segment 4 pitch)
#
# Grouping rationale:
# - right/left flap separated: L/R differential enables turning.
# - phi_fr (front→rear phase offset) embedded in each flap group:
#     front fin: θ*(t);  rear fin: θ*(t) with extra phase of phi_fr*π rad.
#     φ=0: synchronized;  φ≈0.5 (90°): biomimetic fin wave propagation.
# - fins_rotation_sym: all 4 fins share one attack-angle signal (feathering).
# - tail_yaw / tail_pitch: carangiform wave vs vertical control (independent).
#
# Action layout (K=2 harmonics):
#   Flap groups  (g=0,1): [omega, phi_fr, A1, C1, A2, C2]  stride=6
#   Other groups (g=2..4): [omega, A1, C1, A2, C2]          stride=5
REDUCED_ORDER_GROUPS = [
    ("right_fins_flap",   [0, 2]),          # front-R + rear-R pitch
    ("left_fins_flap",    [1, 3]),          # front-L + rear-L pitch
    ("tail_yaw",          [6, 8, 10, 12]),  # t1-t4 yaw (carangiform wave)
    ("tail_pitch",        [7, 9, 11, 13]),  # t1-t4 pitch (vertical control)
]
N_GROUPS = len(REDUCED_ORDER_GROUPS)  # 5

# Group names that carry a front→rear phase offset parameter (phi_fr)
FLAP_GROUP_NAMES = {"right_fins_flap", "left_fins_flap"}

# ============== Wave Control Constants ==============
# Physics-parameterized wave control for thunniform locomotion.
# 12-dim action with decoupled tail / fin-flap / fin-rotation control.
#
# Design principles:
#   1. Tail: independent frequency (omega_tail) for propeller effect + yaw/pitch bias
#   2. Fin flap: independent frequency (omega_fin), same-side front/rear phase offset
#      (fin_phase), L/R asymmetry (fin_asym) but no L/R phase difference
#   3. Fin rotation: frequency locked to fin flap via ratio (rot_ratio * omega_fin),
#      with adjustable phase offset (rot_phase), enabling different attack angle patterns
#
# Actuator layout (14 DOFs):
#   Fins (4 DOFs):
#     [0] fin_right_pitch  [1] fin_left_pitch
#     [2] fin_rear_right_pitch  [3] fin_rear_left_pitch
#   Tail (10 DOFs):
#     [4] t0_yaw  [5] t0_pitch   (passive anchor, always 0)
#     [6] t1_yaw  [7] t1_pitch
#     [8] t2_yaw  [9] t2_pitch
#     [10] t3_yaw  [11] t3_pitch
#     [12] t4_yaw  [13] t4_pitch

# Actuator index groups
WAVE_FIN_FRONT_RIGHT = 0   # fin_right_pitch
WAVE_FIN_FRONT_LEFT  = 1   # fin_left_pitch
WAVE_FIN_REAR_RIGHT  = 2   # fin_rear_right_pitch
WAVE_FIN_REAR_LEFT   = 3   # fin_rear_left_pitch
WAVE_TAIL0_YAW   = 4       # t0_yaw (passive anchor)
WAVE_TAIL0_PITCH = 5       # t0_pitch (passive anchor)
WAVE_TAIL_YAW   = [6, 8, 10, 12]   # t1-t4 yaw
WAVE_TAIL_PITCH  = [7, 9, 11, 13]   # t1-t4 pitch

# Tail traveling-wave constants
N_TAIL_SEGS = 4
TAIL_S = np.array([i / (N_TAIL_SEGS - 1) for i in range(N_TAIL_SEGS)], dtype=np.float32)
# Amplitude envelope: proximal 65% → distal 100% (thunniform: thrust concentrated at tail)
TAIL_ENVELOPE = (0.65 + 0.35 * TAIL_S).astype(np.float32)

# Frequency limits
WAVE_OMEGA_MAX = 12.0   # rad/s (~1.9 Hz)  — raised for faster tail propeller

# Tail pitch propeller ratio: pitch oscillation amplitude relative to yaw
# When > 0, tail tip traces an ellipse (yaw + pitch with 90° phase offset)
# producing propeller-like thrust.  0.0 = pure lateral wave, 1.0 = circular.
# NOTE: kept ≤ 1.0 so that pitch ctrl stays within [-1,1] without clipping;
# values > 1.0 cause the distal segment to saturate at t=0 (-cos(0)=-1),
# producing an instantaneous large deformation that crashes the LBM solver.
TAIL_PITCH_RATIO = 0.8

# Warm-up ramp: number of steps over which ctrl amplitude is linearly
# increased from 0 to the target value after each reset.  This prevents
# LBM numerical divergence caused by sudden large deformations.
WARM_UP_STEPS = 20

# Wave action dimension (tail + fins)
N_WAVE_ACTIONS = 7
# action layout (7-dim):
#   ── Tail (4 params) ──
#   [0]  A_tail     : tail yaw wave amplitude          [-1, 1]
#   [1]  omega_tail : tail oscillation frequency       [-1, 1] → [0, OMEGA_MAX] rad/s
#   [2]  yaw_bias   : tail yaw DC offset for turning   [-1, 1] (proximal large, distal small)
#   [3]  tail_pitch : tail pitch bias (ascend/descend) [-1, 1] (propeller osc + static bias)
#   ── Fins (3 params) ──
#   [4]  fin_amp    : fin flap amplitude (anti-roll)   [-1, 1]
#   [5]  fin_asym   : fin L/R asymmetry (roll correct) [-1, 1]
#   [6]  fin_freq   : fin independent frequency        [-1, 1] → [0, OMEGA_MAX] rad/s


# ============== Multi-Task Environment Class ==============


class TunaMultiTaskEnv(Tuna3DLBMEnv):
    """
    Multi-task tuna locomotion environment with frequency-domain PD control.

    Instead of goal-reaching, the agent is given a task ID (one-hot in obs)
    and rewarded for performing the corresponding motion primitive:
      forward, turn_left, turn_right, ascend, descend.

    Control: The neural network outputs frequency-domain parameters (omega, A, C)
    which are converted to joint target angles via Fourier synthesis,
    then tracked by MuJoCo's built-in PD controllers.

    Each control group outputs one learnable omega (shared across its K harmonics),
    plus K (A, C) pairs. The harmonic frequencies are omega * j for j=1..K.

    Action layout per group g (K harmonics):
        [omega_g, A_g0, C_g0, A_g1, C_g1, ..., A_g(K-1), C_g(K-1)]
    Total per group: 1 + 2*K dims. Total action dim: N_groups * (1 + 2*K).
    With N_groups=8, K=2: action dim = 8 * 5 = 40.

    omega_g ∈ [-1, 1] (raw) → mapped to [omega_min, omega_max] rad/s.
    θ*_g = Σ_{j=1}^{K} A_{gj} * sin(j * omega_g * t + C_{gj})

    Reduced-order: Symmetric fin pairs share control signals, and tail
    yaw/pitch are separated for independent horizontal/vertical control.

    Training: tasks are randomly sampled at reset (and optionally mid-episode).
    Inference: a command sequence can be provided to execute complex maneuvers.
    """

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        root_link: str = 'head',
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
        # Task reward weights (tuned for tuna: slender body, 16 DOF, tail-driven)
        reward_w_task: float = 1.0,
        reward_w_roll: float = 0.15,        # highest: slender fusiform body is very roll-prone
        reward_w_smooth: float = 0.025,     # highest: 16-DOF serial chain amplifies end-effector
        reward_w_offaxis: float = 0.03,     # higher than manta: tail swing causes lateral drift
        # Reference targets / scales (tuna-specific)
        target_forward_vel: float = 0.10,
        target_yaw_rate: float = 0.12,      # fins + tail cooperative turning
        target_vertical_vel: float = 0.08,  # 4 pairs of fins provide direct vertical thrust
        # Frequency-domain PD control parameters
        k_harmonics: int = DEFAULT_K_HARMONICS,
        b_bar: float = DEFAULT_B_BAR,
        use_reduced_order: bool = True,
        control_mode: str = "frequency",
        # Learnable omega range (rad/s). omega_raw ∈ [-1,1] → [omega_min, omega_max].
        # Typical fish: 1–4 Hz → 6–25 rad/s. Keep min>0 to avoid standing still.
        omega_min: float = 1.0,             # ~0.16 Hz: slow cruise
        omega_max: float = 8.0,             # ~1.3 Hz: fast sprint (reduced from 12 for stability)
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

        # Learnable omega range
        self.omega_min = omega_min
        self.omega_max = omega_max

        if self.control_mode == "wave":
            # Wave mode uses built-in physics structure, no group abstraction needed
            self.n_ctrl_groups = N_WAVE_ACTIONS  # 12
            self.joint_groups = REDUCED_ORDER_GROUPS  # kept for compatibility
        elif self.control_mode == "frequency" and use_reduced_order:
            self.n_ctrl_groups = N_GROUPS  # 5
            self.joint_groups = REDUCED_ORDER_GROUPS
        else:
            # Direct control uses per-actuator targets (no grouping)
            self.n_ctrl_groups = self.n_actuators  # 16
            self.joint_groups = [
                (f"joint_{i}", [i]) for i in range(self.n_actuators)
            ]

        # Action layout per group (frequency mode):
        #   Flap groups  : [omega, phi_fr, A1,C1, ..., AK,CK]  stride = 2 + 2*K
        #   Other groups : [omega, A1,C1, ..., AK,CK]           stride = 1 + 2*K
        _strides = []
        for g_name, _ in self.joint_groups:
            if g_name in FLAP_GROUP_NAMES:
                _strides.append(2 + 2 * k_harmonics)
            else:
                _strides.append(1 + 2 * k_harmonics)
        self._group_strides = _strides
        self._group_action_base = [int(sum(_strides[:i])) for i in range(len(_strides))]
        self.freq_action_dim = int(sum(_strides))

        # Build group-to-actuator mapping
        self._group_actuator_indices = [g[1] for g in self.joint_groups]

        # Extract ctrl_range from MuJoCo model for scaling θ* → ctrl
        self._ctrl_lo = self.mj_model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.mj_model.actuator_ctrlrange[:, 1].copy()

        # Time tracking for frequency-domain control
        self._time_val = np.zeros(nworld, dtype=np.float32)
        self._time_val_wp = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._dt = self.mj_model.opt.timestep * per_frame_steps
        self._prev_qpos_buffer = wp.zeros((nworld, self.mj_model.nq), dtype=wp.float32, device=self.device)

        # --- Override obs_dim: base(22+3*n_joints) + lbm(3) + task(NUM_TASKS) + phase(2) ---
        self.obs_dim = 22 + 3 * self.n_joints + 3 + NUM_TASKS + 2

        # Current task for each world
        self._task_ids = np.zeros(nworld, dtype=np.int32)
        self._task_ids_wp = wp.zeros(nworld, dtype=wp.int32, device=self.device)

        # Steps since last task switch
        self._steps_since_switch = np.zeros(nworld, dtype=np.int32)

        # Override observation buffer with new size
        self._obs_buffer = wp.zeros((nworld, self.obs_dim), dtype=wp.float32, device=self.device)

        if self.control_mode == "wave":
            self.action_dim = N_WAVE_ACTIONS  # 12
        elif self.control_mode == "frequency":
            self.action_dim = self.freq_action_dim
        else:
            self.action_dim = self.n_actuators

        # Override action space for the selected control mode
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

        # Warm-up ramp counter (per world): counts steps since last reset
        self._warmup_counter = np.zeros(nworld, dtype=np.int32)

        # Displacement-based velocity buffers (LBM position diff / dt)
        self._prev_lbm_pos = np.zeros((nworld, 3), dtype=np.float32)
        self._prev_lbm_pos_valid = np.zeros(nworld, dtype=bool)
        self._disp_vel_wp = wp.zeros((nworld, 3), dtype=wp.float32, device=self.device)

        print(f"TunaMultiTaskEnv initialized:")
        print(f"  Tasks: {[TASK_NAMES[i] for i in self.enabled_task_ids]}")
        print(f"  Task switch interval: {task_switch_interval} (0=only at reset)")
        print(f"  Control mode: {self.control_mode}")
        if self.control_mode == "wave":
            print(f"  Control: wave mode — {N_WAVE_ACTIONS}-dim physics-parameterized action")
            print(f"  Action: [A_tail, omega_tail, yaw_bias, tail_pitch, fin_amp, fin_asym, fin_freq]")
            print(f"  Tail envelope: {TAIL_ENVELOPE.tolist()}")
            print(f"  Omega max: {WAVE_OMEGA_MAX:.1f} rad/s ({WAVE_OMEGA_MAX/(2*np.pi):.2f} Hz)")
        elif self.control_mode == "frequency":
            print(f"  Control: K={k_harmonics} harmonics, omega=[{omega_min:.1f},{omega_max:.1f}] rad/s, reduced_order={use_reduced_order}")
            print(f"  Groups ({self.n_ctrl_groups}): {[g[0] for g in self.joint_groups]}")
            print(f"  Strides: {self._group_strides}  →  action dim = {self.freq_action_dim}")
            print(f"  (flap groups include phi_fr: front→rear phase offset)")
        else:
            print(f"  Control: direct actuator target angles")
            print(f"  Action dim: {self.n_actuators} (direct targets)")
        print(f"  Obs dim: {self.obs_dim} (base {22+3*self.n_joints} + lbm 3 + task {NUM_TASKS} + phase 2)")
        print(
            f"  Reward scales: fwd={target_forward_vel}, yaw={target_yaw_rate}, vert={target_vertical_vel}"
        )

    def _create_observation_space(self) -> spaces.Space:
        """Create observation space with task one-hot and phase info appended."""
        n_joints = self.mj_model.njnt - 1
        obs_dim = 22 + 3 * n_joints + 3 + NUM_TASKS + 2
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
                self._command_index = len(self._command_sequence) - 1
                self._command_step_counter = cmd["steps"]
                return

            next_cmd = self._command_sequence[self._command_index]
            task_name = next_cmd["task"]
            task_id = TASK_NAMES.index(task_name)
            self._task_ids[:] = task_id
            self._update_task_ids_wp()

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
        """Task-conditioned reward computation (tuna-specific kernel)."""
        wp.launch(
            compute_tuna_multitask_reward_kernel,
            dim=self.nworld,
            inputs=[
                self.mjw_data.qpos,
                self.mjw_data.qvel,
                self._disp_vel_wp,
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
        Convert 7-dim physical wave parameters to 14-dim MuJoCo ctrl (tail_0 anchor + tail + fins).

        action layout (7-dim):
          ── Tail (4 params) ──
          [0]  A_tail     : tail yaw wave amplitude          [-1, 1]
          [1]  omega_tail : tail oscillation frequency       [-1, 1] → [0, OMEGA_MAX]
          [2]  yaw_bias   : tail yaw DC offset (turning)     [-1, 1]
          [3]  tail_pitch : tail pitch bias (ascend/descend) [-1, 1]
          ── Fins (3 params) ──
          [4]  fin_amp    : fin flap amplitude (anti-roll)   [-1, 1]
          [5]  fin_asym   : fin L/R asymmetry (roll correct) [-1, 1]
          [6]  fin_freq   : fin independent frequency        [-1, 1] → [0, OMEGA_MAX]

        All tail joints share the same phase (propeller mode, no traveling wave):
          phi_tail = omega_tail * t

        Tail yaw joint i (i=0..3):
          theta_i = A_tail * envelope[i] * sin(phi_tail) + yaw_bias * (1 - s_i)

        Tail pitch joint i (propeller + static bias):
          theta_i = A_tail * PITCH_RATIO * envelope[i] * (-cos(phi_tail))
                  + tail_pitch * envelope[i]

        Fins (counter-flap for roll stabilization, independent frequency):
          phi_fin = omega_fin * t
          fin_right = (fin_amp + fin_asym) * sin(phi_fin)
          fin_left  = -(fin_amp - fin_asym) * sin(phi_fin)
          Opposite sign: when tail rotates CW, fins flap to cancel roll torque.
        """
        nw = action.shape[0]
        ctrl = np.zeros((nw, self.n_actuators), dtype=np.float32)

        # Unpack parameters
        A_tail     = action[:, 0]
        omega_tail_n = action[:, 1]
        yaw_bias   = action[:, 2]
        tail_pitch = action[:, 3]
        fin_amp    = action[:, 4]
        fin_asym   = action[:, 5]
        fin_freq_n = action[:, 6]

        # Physical scales
        omega_tail = (omega_tail_n + 1.0) * 0.5 * WAVE_OMEGA_MAX  # [0, OMEGA_MAX]
        t = self._time_val[:nw]

        def _write(act_idx, theta_norm):
            """Write normalized [-1,1] value to ctrl, scaling to actuator range."""
            theta_c = np.clip(theta_norm, -1.0, 1.0)
            lo, hi = self._ctrl_lo[act_idx], self._ctrl_hi[act_idx]
            ctrl[:, act_idx] = lo + (theta_c + 1.0) * 0.5 * (hi - lo)

        # ── Shared phase (propeller mode: all joints rotate in sync) ──
        phase = omega_tail * t

        # ── Tail_0: passive anchor (always held at center = 0) ──
        _write(WAVE_TAIL0_YAW,   np.zeros(nw, dtype=np.float32))
        _write(WAVE_TAIL0_PITCH, np.zeros(nw, dtype=np.float32))

        # ── Tail Yaw ──
        for i in range(N_TAIL_SEGS):
            theta_i = (A_tail * TAIL_ENVELOPE[i] * np.sin(phase)
                       + yaw_bias * (1.0 - TAIL_S[i]))
            _write(WAVE_TAIL_YAW[i], theta_i)

        # ── Tail Pitch: 90° lag behind yaw (sin→-cos) → propeller rotation ──
        for i in range(N_TAIL_SEGS):
            pitch_osc = A_tail * TAIL_PITCH_RATIO * TAIL_ENVELOPE[i] * (-np.cos(phase))
            pitch_bias = tail_pitch * TAIL_ENVELOPE[i]
            theta_i = pitch_osc + pitch_bias
            _write(WAVE_TAIL_PITCH[i], theta_i)

        # ── Fins: counter-flap for roll stabilization (independent frequency) ──
        # Fins have their own frequency to decouple from tail propeller rotation.
        # fin_amp controls overall flap strength; fin_asym allows L/R bias.
        # Rear fins use 70% of front fin amplitude (smaller, further back).
        omega_fin = (fin_freq_n + 1.0) * 0.5 * WAVE_OMEGA_MAX
        fin_phase = omega_fin * t
        fin_right_val = (fin_amp + fin_asym) * np.sin(fin_phase)
        fin_left_val  = -(fin_amp - fin_asym) * np.sin(fin_phase)
        _write(WAVE_FIN_FRONT_RIGHT, fin_right_val)
        _write(WAVE_FIN_FRONT_LEFT,  fin_left_val)
        _write(WAVE_FIN_REAR_RIGHT,  fin_right_val * 0.7)
        _write(WAVE_FIN_REAR_LEFT,   fin_left_val * 0.7)

        return ctrl

    def _freq_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """
        Convert frequency-domain action parameters to MuJoCo ctrl (target angles).

        Flap groups  (right/left_fins_flap):
            [omega, phi_fr, A1, C1, A2, C2, ...]  stride = 2 + 2*K
            omega  ∈ [-1,1] → [omega_min, omega_max] rad/s
            phi_fr ∈ [-1,1] → [-π, π] rad  (front→rear phase delay)
            front fin: θ* = Σ A_j * sin(j·ω·t + C_j)
            rear  fin: θ* = Σ A_j * sin(j·ω·t + C_j + phi_fr·π)

        Other groups (rotation, tail yaw/pitch):
            [omega, A1, C1, A2, C2, ...]  stride = 1 + 2*K
            All actuators in the group share the same θ*.
        """
        nw = action.shape[0]
        K = self.k_harmonics
        ctrl = np.zeros((nw, self.n_actuators), dtype=np.float32)
        t = self._time_val[:nw]

        for g_idx, (g_name, act_indices) in enumerate(self.joint_groups):
            base = self._group_action_base[g_idx]
            is_flap = g_name in FLAP_GROUP_NAMES

            omega = self.omega_min + (action[:, base] + 1.0) * 0.5 * (self.omega_max - self.omega_min)

            if is_flap:
                phi_fr = action[:, base + 1] * np.pi   # [-1,1] → [-π, π]
                ak = base + 2
            else:
                ak = base + 1

            for local_i, act_idx in enumerate(act_indices):
                # Rear actuator (local_i > 0) gets the phase offset
                extra_phase = phi_fr * float(local_i) if is_flap else np.zeros(nw, dtype=np.float32)

                theta_star = np.zeros(nw, dtype=np.float32)
                for j in range(1, K + 1):
                    A_gj = action[:, ak + (j - 1) * 2]
                    C_gj = action[:, ak + (j - 1) * 2 + 1] * np.pi
                    theta_star += A_gj * np.sin(j * omega * t + C_gj + extra_phase)

                theta_star = np.clip(theta_star, -1.0, 1.0)

                lo, hi = self._ctrl_lo[act_idx], self._ctrl_hi[act_idx]
                ctrl[:, act_idx] = lo + (theta_star + 1.0) * 0.5 * (hi - lo)

        return ctrl

    def step(self, action: np.ndarray):
        """
        Execute one step with the selected control mode.

        Frequency mode converts Fourier coefficients to target angles.
        Direct mode sends actuator target angles to MuJoCo directly.

        A warm-up ramp is applied for the first WARM_UP_STEPS after each
        reset: ctrl amplitude is linearly scaled from 0 → 1 to prevent
        LBM numerical divergence from sudden large deformations.
        """
        ctrl = self._action_to_ctrl(action)

        # Warm-up ramp: linearly scale ctrl amplitude for the first N steps
        self._warmup_counter += 1
        ramp = np.clip(self._warmup_counter / WARM_UP_STEPS, 0.0, 1.0).astype(np.float32)
        ctrl = ctrl * ramp[:, None]  # (nworld, n_actuators) * (nworld, 1)

        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        # Write target angles to MuJoCo ctrl
        wp.copy(self.mjw_data.ctrl, wp.array(ctrl, dtype=wp.float32, device=self.device))

        # Advance the phase clock for frequency-domain and wave control modes
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

        # Compute displacement-based velocity from LBM position difference
        self._update_disp_vel()

        # Compute reward before task switching
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

        # Mid-episode task switching
        if self.task_switch_interval > 0 and self._command_sequence is None:
            self._steps_since_switch += 1
            switch_mask = (self._steps_since_switch >= self.task_switch_interval) & ~done
            if np.any(switch_mask):
                self._sample_tasks(switch_mask)
                self._steps_since_switch[switch_mask] = 0

        # Advance command sequence
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

        return observation, reward, done, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        """Reset all worlds and sample new tasks."""
        if seed is not None:
            np.random.seed(seed)

        # Call grandparent reset (LBMFluidEnv3D.reset), skip Tuna3DLBMEnv's goal logic
        from ..lbm_fluid_env_3d import LBMFluidEnv3D
        LBMFluidEnv3D.reset(self, seed=seed, options=options)

        # Sample new tasks for all worlds
        if self._command_sequence is None:
            self._sample_tasks()
        else:
            self._command_index = 0
            self._command_step_counter = 0
            if self._command_sequence:
                task_id = TASK_NAMES.index(self._command_sequence[0]["task"])
                self._task_ids[:] = task_id
                self._update_task_ids_wp()

        self._steps_since_switch[:] = 0
        self._time_val[:] = 0.0
        self._prev_ctrl.fill(0.0)
        self._prev_ctrl_valid[:] = False
        self._prev_lbm_pos_valid[:] = False
        self._warmup_counter[:] = 0
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        return self._get_obs()

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
        self._time_val[reset_mask] = 0.0
        self._prev_ctrl[reset_mask] = 0.0
        self._prev_ctrl_valid[reset_mask] = False
        self._prev_lbm_pos_valid[reset_mask] = False
        self._warmup_counter[reset_mask] = 0
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        return self._get_obs()

    def get_current_task(self, world_idx: int = 0) -> str:
        """Get current task name for a world."""
        return TASK_NAMES[self._task_ids[world_idx]]

    def get_current_goal(self, world_idx: int = 0) -> Tuple[float, float, float]:
        """Override for compatibility: return center position as dummy goal."""
        return (0.5, 0.5, 0.5)
