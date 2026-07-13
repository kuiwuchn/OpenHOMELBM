"""
3D Torus Multi-Task LBM Environment v3 with wave control mode.

Multi-task training for torus ring robot locomotion skills.
Inherits from Torus3DLBMEnv but replaces goal-reaching with velocity-based
task-conditioned rewards.

5 Tasks:
  0: FORWARD    — swim along +Y (ring normal direction)
  1: TURN_LEFT  — yaw left while maintaining gentle forward cruise
  2: TURN_RIGHT — yaw right while maintaining gentle forward cruise
  3: ASCEND     — swim upward (+Z) with slight forward cruise
  4: DESCEND    — swim downward (-Z) with slight forward cruise

=== Control Modes ===

'wave' (recommended, default):
  Physics-parameterized elliptical squeeze + out-of-plane wave — 7-dim action:
    action[0]: A_squeeze — squeeze amplitude (elliptical deformation) [-1, 1]
    action[1]: A_wave    — wave amplitude (out-of-plane bending)     [-1, 1]
    action[2]: omega     — oscillation frequency                     [-1, 1]
    action[3]: squeeze_phase — squeeze phase offset (thrust direction)[-1, 1]
    action[4]: squeeze_asym  — left/right squeeze asymmetry (turning)[-1, 1]
    action[5]: wave_bias — wave DC offset (ascend/descend)           [-1, 1]
    action[6]: wave_asym — wave left/right asymmetry                 [-1, 1]

  Joint i at ring angle θ_i (i = 0..11):
    θ_i       = joint_angle[i]  (30°, 60°, ..., 330°, 0°)
    squeeze_i = A_squeeze * (1 + squeeze_asym * cos(θ_i)) * cos(2*θ_i + φ) * cos(ω*t)
    wave_i    = A_wave * (1 + wave_asym * cos(θ_i)) * sin(ω*t) + wave_bias

  Squeeze mode (cos(2θ+φ) standing wave × cos(ωt) oscillation):
    The ring deforms into a FIXED elliptical shape (determined by φ),
    and the deformation amplitude oscillates in time via cos(ωt).
    This periodic squeeze produces net thrust along the ring normal (Y axis)
    via fluid momentum exchange. The squeeze axis does NOT rotate.

'direct':
  4-dimensional grouped action:
    Group 0: all_bend    (12 bend joints)     — ring pulsation
    Group 1: front_bend  (bend 1-6)           — asymmetric pulsation for steering
    Group 2: back_bend   (bend 7-12)          — asymmetric pulsation for steering
    Group 3: all_wave    (12 wave joints)     — out-of-plane bending

Observation: base obs + lbm_pos(3) + task one-hot (5) + phase (2)
  Base obs uses ring center position (3), ring normal (3), ring center velocity (3),
  angular velocity (3), instead of quaternion (4) + velocity (3) from seg1.
  n_joints = 24 (all joints observed for full state info)
  Total: 6 + 24 + 3 + 3 + 3 + 3 + 24 + 24 + 3 + 5 + 2 = 100 dims

Note on squeeze asymmetry:
  squeeze_asym > 0 → segments near θ=0° (seg1 side) squeeze harder → asymmetric thrust → turn
  squeeze_asym < 0 → segments near θ=180° (seg7 side) squeeze harder → turn opposite
  wave_bias > 0 → all wave joints offset upward → ring tilts → ascend
  wave_bias < 0 → all wave joints offset downward → descend
  wave_asym modulates wave amplitude left/right for fine vertical control

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

from .torus_lbm_env_3d import (
    Torus3DLBMEnv,
    compute_torus_obs_kernel,
    check_boundary_torus_kernel,
    check_stability_torus_kernel,
    compute_ring_center_pos,
    compute_ring_normal,
    compute_ring_center_vel,
)
from ..lbm_core_3d import HomeFlow3D

# Reuse generic multi-task Warp kernels from manta
from ..manta.manta_multitask_env_3d import (
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


# ============== Torus-specific Multitask Obs Kernel ==============

@wp.kernel
def compute_torus_multitask_obs_kernel(
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
    Compute observation for torus multi-task environment.

    Uses ring center position, normal, and velocity instead of seg1's quaternion/velocity.

    Layout:
    - Forces (6): fx, fy, fz, tau_x, tau_y, tau_z
    - Joint torques (n_joints)
    - Ring center position (3): x, y, z (from qpos)
    - Ring normal (3): nx, ny, nz (from segment positions)
    - Ring center velocity (3): vx, vy, vz (average of 12 segment velocities)
    - Angular velocity (3): omega_x, omega_y, omega_z
    - Joint angles (n_joints)
    - Joint velocities (n_joints)
    - LBM ring center position (3): normalized x, y, z
    - Task one-hot (n_tasks): 5-dim one-hot vector
    - Phase (2): sin(t), cos(t)

    Total: 6 + n_joints + 3 + 3 + 3 + 3 + n_joints + n_joints + 3 + n_tasks + 2
    For 24 joints, n_tasks=5: 6 + 24 + 3 + 3 + 3 + 3 + 24 + 24 + 3 + 5 + 2 = 100 dims
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

    # Ring center position (3): from qpos (MuJoCo frame)
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

    # LBM ring center position (normalized) (3)
    ring_center = compute_ring_center_pos(flow)
    obs_out[world_idx, idx] = ring_center[0] / nx
    obs_out[world_idx, idx + 1] = ring_center[1] / ny
    obs_out[world_idx, idx + 2] = ring_center[2] / nz
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


# ============== Torus-specific Reward Kernel ==============

@wp.kernel
def compute_torus_multitask_reward_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    flows: wp.array(dtype=HomeFlow3D),
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
    Torus-specific task-conditioned reward using ring center quantities.

    Instead of using seg1's quaternion and velocity, we compute:
    - Ring normal: from cross products of adjacent segment position vectors
    - Ring center velocity: average of 12 segment linear velocities
    - Ring center angular velocity: from root body qvel (still valid for overall rotation)

    The ring normal serves as the "forward" direction (thrust direction).
    """
    world_idx = wp.tid()
    task = task_ids[world_idx]
    flow = flows[world_idx]

    # Ring normal = forward direction (replaces body_forward from quaternion)
    ring_normal = compute_ring_normal(flow)

    # Ring center velocity (replaces seg1 qvel[0:3])
    ring_vel = compute_ring_center_vel(flow)

    # Angular velocity from root body (still meaningful for overall rotation)
    # We use qpos quaternion to rotate body-frame omega to world frame
    qw = qpos[world_idx, 3]
    qx = qpos[world_idx, 4]
    qy = qpos[world_idx, 5]
    qz = qpos[world_idx, 6]
    omega_body = wp.vec3(
        qvel[world_idx, 3],
        qvel[world_idx, 4],
        qvel[world_idx, 5],
    )
    omega_world = quat_rotate_vec(qw, qx, qy, qz, omega_body)

    # Derive body axes from ring normal
    # ring_normal = forward direction (thrust)
    # We need a "right" and "up" vector orthogonal to ring_normal
    # Use world Z as reference to construct right vector
    body_forward = ring_normal

    # Construct right = normalize(forward × world_Z)
    # If forward is nearly parallel to Z, use world_X as fallback
    cross_x = body_forward[1] * 1.0 - body_forward[2] * 0.0  # forward × (0,0,1)
    cross_y = body_forward[2] * 0.0 - body_forward[0] * 1.0
    cross_z = body_forward[0] * 0.0 - body_forward[1] * 0.0
    cross_len = wp.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)

    if cross_len < 1.0e-4:
        # Forward is nearly parallel to Z, use X as reference
        cross_x = body_forward[1] * 0.0 - body_forward[2] * 0.0  # forward × (1,0,0)
        cross_y = body_forward[2] * 1.0 - body_forward[0] * 0.0
        cross_z = body_forward[0] * 0.0 - body_forward[1] * 1.0
        cross_len = wp.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)

    body_right = wp.vec3(cross_x / cross_len, cross_y / cross_len, cross_z / cross_len)

    # body_up = forward × right
    body_up = wp.vec3(
        body_forward[1] * body_right[2] - body_forward[2] * body_right[1],
        body_forward[2] * body_right[0] - body_forward[0] * body_right[2],
        body_forward[0] * body_right[1] - body_forward[1] * body_right[0],
    )

    # Velocity projections using ring center velocity
    v_forward = dot_vec3(ring_vel, body_forward)
    v_lateral = dot_vec3(ring_vel, body_right)
    v_vertical = ring_vel[2]
    yaw_rate = omega_world[2]

    # World +Y velocity component (for FORWARD task) — using ring center velocity
    v_world_y = ring_vel[1]

    # Heading alignment: cosine between ring normal and world +Y axis
    cos_heading = body_forward[1]

    # body_up projected onto world axes: roll = body_up.x, pitch = body_up.y
    upright_roll = body_up[0] * body_up[0]
    upright_pitch = body_up[1] * body_up[1]

    r_task = float(0.0)
    r_upright = float(0.0)
    r_offaxis = float(0.0)

    turn_forward_scale = 0.5 * target_forward_vel
    vertical_forward_scale = 0.5 * target_forward_vel

    if task == TASK_FORWARD:
        # Thrust: reward world +Y velocity
        # Heading: reward body_forward aligned with world +Y
        r_task = w_task * (
            0.50 * directional_reward(v_world_y, target_forward_vel)
            + 0.50 * cos_heading
        )
        # Moderate roll penalty: ring is naturally stable
        r_upright = -w_roll * (0.40 * upright_roll + 0.30 * upright_pitch)
        # Off-axis penalty
        r_offaxis = -w_offaxis * (
            0.50 * soft_penalty(v_lateral, target_forward_vel)
            + 0.30 * soft_penalty(v_vertical, target_vertical_vel)
        )

    elif task == TASK_TURN_LEFT:
        r_task = w_task * (
            0.50 * directional_reward(yaw_rate, target_yaw_rate)
            + 0.50 * directional_reward(v_forward, turn_forward_scale)
        )
        r_upright = -w_roll * (0.25 * upright_roll + 0.15 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.40 * soft_penalty(v_lateral, target_forward_vel)
            + 0.15 * soft_penalty(v_vertical, target_vertical_vel)
        )

    elif task == TASK_TURN_RIGHT:
        r_task = w_task * (
            0.50 * directional_reward(-yaw_rate, target_yaw_rate)
            + 0.50 * directional_reward(v_forward, turn_forward_scale)
        )
        r_upright = -w_roll * (0.25 * upright_roll + 0.15 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.40 * soft_penalty(v_lateral, target_forward_vel)
            + 0.15 * soft_penalty(v_vertical, target_vertical_vel)
        )

    elif task == TASK_ASCEND:
        # Torus can tilt ring plane to redirect pulsation thrust upward
        r_task = w_task * (
            0.55 * directional_reward(v_vertical, target_vertical_vel)
            + 0.45 * positive_reward(v_forward, vertical_forward_scale)
        )
        # Low roll penalty: allow ring tilting
        r_upright = -w_roll * (0.10 * upright_roll + 0.05 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.40 * soft_penalty(v_lateral, target_forward_vel)
            + 0.15 * soft_penalty(yaw_rate, target_yaw_rate)
        )

    elif task == TASK_DESCEND:
        r_task = w_task * (
            0.55 * directional_reward(-v_vertical, target_vertical_vel)
            + 0.45 * positive_reward(v_forward, vertical_forward_scale)
        )
        r_upright = -w_roll * (0.10 * upright_roll + 0.05 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.40 * soft_penalty(v_lateral, target_forward_vel)
            + 0.15 * soft_penalty(yaw_rate, target_yaw_rate)
        )

    # Angular velocity smoothness penalty
    if task == TASK_TURN_LEFT or task == TASK_TURN_RIGHT:
        r_smooth = -w_smooth * (
            0.35 * omega_world[0] * omega_world[0]
            + 0.35 * omega_world[1] * omega_world[1]
        )
    elif task == TASK_ASCEND or task == TASK_DESCEND:
        r_smooth = -w_smooth * (
            0.25 * omega_world[0] * omega_world[0]
            + 0.15 * omega_world[1] * omega_world[1]
            + 0.10 * omega_world[2] * omega_world[2]
        )
    else:
        r_smooth = -w_smooth * (
            0.40 * omega_world[0] * omega_world[0]
            + 0.40 * omega_world[1] * omega_world[1]
            + 0.15 * omega_world[2] * omega_world[2]
        )

    rewards_out[world_idx] = r_task + r_upright + r_offaxis + r_smooth


# ============== Torus Control Constants ==============

# Number of joints per type
N_BEND_JOINTS = 12
N_WAVE_JOINTS = 12
N_TOTAL_ACTUATORS = 24

# Actuator indices in the 24-dim ctrl vector
# Ordering: [bend1,wave1, bend2,wave2, ...]
BEND_ACTUATOR_INDICES = list(range(0, N_TOTAL_ACTUATORS, 2))     # [0,2,4,...,22]
WAVE_ACTUATOR_INDICES = list(range(1, N_TOTAL_ACTUATORS, 2))     # [1,3,5,...,23]

# Wave action: 7-dim [A_squeeze, A_wave, omega, squeeze_phase, squeeze_asym, wave_bias, wave_asym]
N_WAVE_ACTIONS = 7

# Joint angles (θ) for each of the 12 joints around the ring (in radians)
# Joint 1 at 30°, Joint 2 at 60°, ..., Joint 11 at 330°, Joint 12 at 0° (closure)
JOINT_THETA = np.array([
    30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 0
], dtype=np.float32) * np.pi / 180.0  # (12,)

# Grouped control: 4 groups
REDUCED_ORDER_GROUPS = [
    ("all_bend",    BEND_ACTUATOR_INDICES),                       # all 12 bend joints
    ("front_bend",  BEND_ACTUATOR_INDICES[:6]),                   # bend 1-6 (front half)
    ("back_bend",   BEND_ACTUATOR_INDICES[6:]),                   # bend 7-12 (back half)
    ("all_wave",    WAVE_ACTUATOR_INDICES),                       # all 12 wave joints
]
N_GROUPS = len(REDUCED_ORDER_GROUPS)
N_DIRECT_ACTIONS = N_GROUPS


# ============== Multi-Task Environment Class ==============


class TorusMultiTaskEnv(Torus3DLBMEnv):
    """
    Multi-task torus ring locomotion environment.

    Instead of goal-reaching, the agent is given a task ID (one-hot in obs)
    and rewarded for performing the corresponding motion primitive:
      forward, turn_left, turn_right, ascend, descend.

    Control modes:
      - 'wave': 7-dim physical params → 24-dim MuJoCo ctrl (ring wave)
      - 'direct': 4-dim grouped targets → 24-dim MuJoCo ctrl

    Training: tasks are randomly sampled at reset (and optionally mid-episode).
    Inference: a command sequence can be provided to execute complex maneuvers.
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
        # Multi-task specific
        task_switch_interval: int = 0,
        enabled_tasks: Optional[List[str]] = None,
        # Task reward weights
        reward_w_task: float = 1.0,
        reward_w_roll: float = 0.15,
        reward_w_smooth: float = 0.03,
        reward_w_offaxis: float = 0.04,
        # Reference targets
        target_forward_vel: float = 0.08,
        target_yaw_rate: float = 0.10,
        target_vertical_vel: float = 0.04,
        # Control mode
        control_mode: str = "wave",
    ):
        # Initialize parent with dummy goal settings
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
            goal_threshold=1.0,
            single_goal_mode=True,
            goal_position=[0.5, 0.5, 0.5],
            control_mode='direct',
        )

        # --- Multi-task config ---
        self.task_switch_interval = task_switch_interval

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

        # --- Control mode (override parent's) ---
        self.control_mode = control_mode.lower()
        if self.control_mode not in {"direct", "wave"}:
            raise ValueError(f"Unknown control_mode '{control_mode}'. Expected 'wave' or 'direct'.")

        # Extract ctrl_range from MuJoCo model
        self._ctrl_lo = self.mj_model.actuator_ctrlrange[:, 0].copy()  # (21,)
        self._ctrl_hi = self.mj_model.actuator_ctrlrange[:, 1].copy()  # (21,)

        # Time tracking for wave control
        self._time_val = np.zeros(nworld, dtype=np.float32)
        self._time_val_wp = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._dt = self.mj_model.opt.timestep * per_frame_steps
        self._prev_qpos_buffer = wp.zeros((nworld, self.mj_model.nq), dtype=wp.float32, device=self.device)

        # --- Override obs_dim ---
        # 6(forces) + n_joints(torques) + 3(pos) + 3(normal) + 3(vel) + 3(omega)
        # + n_joints(angles) + n_joints(velocities) + 3(lbm_pos) + NUM_TASKS + 2(phase)
        self.obs_dim = 6 + self.n_joints + 3 + 3 + 3 + 3 + self.n_joints + self.n_joints + 3 + NUM_TASKS + 2

        # Current task for each world
        self._task_ids = np.zeros(nworld, dtype=np.int32)
        self._task_ids_wp = wp.zeros(nworld, dtype=wp.int32, device=self.device)

        # Steps since last task switch
        self._steps_since_switch = np.zeros(nworld, dtype=np.int32)

        # Override observation buffer
        self._obs_buffer = wp.zeros((nworld, self.obs_dim), dtype=wp.float32, device=self.device)

        if self.control_mode == "wave":
            self.action_dim = N_WAVE_ACTIONS  # 4
        else:
            self.action_dim = N_DIRECT_ACTIONS  # 4

        # Override action space
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

        print(f"TorusMultiTaskEnv initialized:")
        print(f"  Tasks: {[TASK_NAMES[i] for i in self.enabled_task_ids]}")
        print(f"  Task switch interval: {task_switch_interval} (0=only at reset)")
        print(f"  Control mode: {self.control_mode}")
        if self.control_mode == "wave":
            print(f"  Control: elliptical squeeze + out-of-plane wave (7-dim)")
            print(f"    action[0]=A_squeeze, [1]=A_wave, [2]=omega, [3]=squeeze_phase")
            print(f"    action[4]=squeeze_asym, [5]=wave_bias, [6]=wave_asym")
        else:
            print(f"  Control: direct grouped ({self.action_dim} dims)")
            print(f"  Groups ({N_GROUPS}): {[g[0] for g in REDUCED_ORDER_GROUPS]}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Obs dim: {self.obs_dim}")
        print(
            f"  Reward weights: task={reward_w_task}, roll={reward_w_roll}, "
            f"smooth={reward_w_smooth}, offaxis={reward_w_offaxis}"
        )
        print(
            f"  Target velocities: fwd={target_forward_vel}, yaw={target_yaw_rate}, vert={target_vertical_vel}"
        )

    def _create_observation_space(self) -> spaces.Space:
        """Create observation space with task one-hot and phase info appended."""
        n_joints = self.mj_model.njnt - 1  # 24 (excluding freejoint)
        # 6 + n_joints + 3 + 3 + 3 + 3 + n_joints + n_joints + 3 + NUM_TASKS + 2
        obs_dim = 6 + n_joints + 3 + 3 + 3 + 3 + n_joints + n_joints + 3 + NUM_TASKS + 2
        return spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.nworld, obs_dim),
            dtype=np.float32
        )

    # --- Task management ---

    def _sample_tasks(self, mask: Optional[np.ndarray] = None):
        """Randomly assign tasks."""
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
        """Get observation with task one-hot, ring center info, and phase."""
        wp.copy(
            self._time_val_wp,
            wp.array(self._time_val.astype(np.float32), dtype=wp.float32, device=self.device)
        )
        wp.launch(
            compute_torus_multitask_obs_kernel,
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
        """Task-conditioned reward computation (torus-specific kernel with ring center)."""
        wp.launch(
            compute_torus_multitask_reward_kernel,
            dim=self.nworld,
            inputs=[
                self.mjw_data.qpos,
                self.mjw_data.qvel,
                self.lbm_solver.flows_wp,
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
        """Map policy output to 24-dim MuJoCo actuator targets."""
        action = np.clip(action, -1.0, 1.0)
        if self.control_mode == "wave":
            return self._wave_to_ctrl(action)
        return self._direct_to_ctrl(action)

    def _wave_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """
        Convert 7-dim elliptical squeeze parameters to 24-dim MuJoCo ctrl.

        The torus propels itself by periodic elliptical deformation (squeeze):
        opposite sides of the ring alternately compress and expand, pushing
        fluid and generating thrust along the ring normal (Y axis).

        action layout:
          [0] A_squeeze    : squeeze amplitude (elliptical deformation) in [-1, 1]
          [1] A_wave       : wave amplitude (out-of-plane bending)     in [-1, 1]
          [2] omega        : oscillation frequency                     in [-1, 1]
          [3] squeeze_phase: squeeze phase offset (thrust direction)   in [-1, 1]
          [4] squeeze_asym : left/right squeeze asymmetry (turning)    in [-1, 1]
          [5] wave_bias    : wave DC offset (ascend/descend)           in [-1, 1]
          [6] wave_asym    : wave left/right asymmetry                 in [-1, 1]

        Bend joint i at ring angle θ_i:
          squeeze_i = A_squeeze * (1 + squeeze_asym * cos(θ_i)) * cos(2*θ_i + φ) * cos(ω*t)

        The cos(2θ+φ) is a FIXED spatial mode (standing wave, not rotating):
          - At θ=0° and θ=180°: cos(2θ)=+1 → squeeze inward together
          - At θ=90° and θ=270°: cos(2θ)=-1 → expand outward together
          - cos(ωt) modulates the amplitude in time → ring oscillates between
            X-elongated and Z-elongated ellipses (left-right squeeze)
          - φ rotates the squeeze axis (which pair of sides squeezes)

        Wave joints (out-of-plane, for ascend/descend):
          wave_i = A_wave * (1 + wave_asym * cos(θ_i)) * sin(ω*t) + wave_bias
          All wave joints oscillate in-phase (uniform ring tilting)

        Asymmetry convention:
          squeeze_asym > 0 → segments near θ=0° squeeze harder → asymmetric thrust → turn
          wave_bias > 0 → all wave joints offset upward → ring tilts → ascend
        """
        nw = action.shape[0]
        ctrl = np.zeros((nw, self.n_actuators), dtype=np.float32)

        A_squeeze     = action[:, 0]
        A_wave        = action[:, 1]
        omega_n       = action[:, 2]
        squeeze_phase = action[:, 3]  # phase offset for squeeze direction
        squeeze_asym  = action[:, 4]  # left/right asymmetry for turning
        wave_bias     = action[:, 5]  # wave DC offset for ascend/descend
        wave_asym     = action[:, 6]  # wave left/right asymmetry

        # Physical scales
        OMEGA_MAX = np.pi * 4.0   # max 4pi rad/s ~ 2 Hz
        PHASE_MAX = np.pi         # max phase offset ±π
        BIAS_MAX  = 0.6           # max DC offset (in normalized [-1,1] range)

        omega = omega_n * OMEGA_MAX
        phi   = squeeze_phase * PHASE_MAX
        t     = self._time_val[:nw]

        # Joint angles around ring: (1, 12)
        theta = JOINT_THETA[None, :]  # (1, 12)

        # Elliptical squeeze (standing wave): cos(2θ + φ) * cos(ωt)
        # cos(2θ+φ) = fixed spatial mode (n=2 ellipse), cos(ωt) = time oscillation
        spatial_mode = np.cos(2.0 * theta + phi[:, None])  # (nw, 12) — fixed shape
        time_osc = np.cos(omega[:, None] * t[:, None])     # (nw, 1) → broadcast

        # Asymmetry modulation: cos(θ_i) = +1 at θ=0°, -1 at θ=180°
        asym_mod = np.cos(theta)  # (1, 12)

        # Bend joints: A_squeeze * (1 + squeeze_asym * cos(θ)) * cos(2θ+φ) * cos(ωt)
        squeeze_envelope = 1.0 + squeeze_asym[:, None] * asym_mod  # (nw, 12)
        bend_norm = A_squeeze[:, None] * squeeze_envelope * spatial_mode * time_osc
        bend_norm = np.clip(bend_norm, -1.0, 1.0)

        # Wave joints: A_wave * (1 + wave_asym * cos(θ)) * sin(ωt) + wave_bias
        # All wave joints oscillate in-phase (uniform tilting for ascend/descend)
        wave_envelope = 1.0 + wave_asym[:, None] * asym_mod  # (nw, 12)
        wave_osc = np.sin(omega[:, None] * t[:, None])  # (nw, 1) broadcast to (nw, 12)
        wave_norm = A_wave[:, None] * wave_envelope * wave_osc + wave_bias[:, None] * BIAS_MAX
        wave_norm = np.clip(wave_norm, -1.0, 1.0)

        # Write to ctrl: bend actuators
        for i, act_idx in enumerate(BEND_ACTUATOR_INDICES):
            lo, hi = self._ctrl_lo[act_idx], self._ctrl_hi[act_idx]
            ctrl[:, act_idx] = lo + (bend_norm[:, i] + 1.0) * 0.5 * (hi - lo)

        # Write to ctrl: wave actuators
        for i, act_idx in enumerate(WAVE_ACTUATOR_INDICES):
            lo, hi = self._ctrl_lo[act_idx], self._ctrl_hi[act_idx]
            ctrl[:, act_idx] = lo + (wave_norm[:, i] + 1.0) * 0.5 * (hi - lo)

        return ctrl

    def _direct_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """
        Convert 4-dim grouped action to 24-dim MuJoCo ctrl.

        action[:, 0] → all_bend group    (12 bend joints)
        action[:, 1] → front_bend group  (bend 1-6, override)
        action[:, 2] → back_bend group   (bend 7-12, override)
        action[:, 3] → all_wave group    (12 wave joints)
        """
        nw = action.shape[0]
        ctrl = np.zeros((nw, self.n_actuators), dtype=np.float32)

        for g_idx, (g_name, act_indices) in enumerate(REDUCED_ORDER_GROUPS):
            val = action[:, g_idx]
            for act_idx in act_indices:
                lo, hi = self._ctrl_lo[act_idx], self._ctrl_hi[act_idx]
                ctrl[:, act_idx] = lo + (val + 1.0) * 0.5 * (hi - lo)

        return ctrl

    def step(self, action: np.ndarray):
        """Execute one step with the selected control mode."""
        ctrl = self._action_to_ctrl(action)
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        # Write target angles to MuJoCo ctrl
        wp.copy(self.mjw_data.ctrl, wp.array(ctrl, dtype=wp.float32, device=self.device))

        # Advance phase clock
        if self.control_mode == "wave":
            self._time_val += self._dt

        # Physics simulation
        self._simulation_step()

        # Update step counts
        self.step_counts += 1

        # Check stability
        instability_mask = self._check_numerical_stability()

        # Check termination
        self._is_terminated(instability_mask)
        boundary_terminated = self._terminated_buffer.numpy().astype(bool).copy()

        # Compute reward
        reward = self._compute_reward(instability_mask)

        # Control smoothness penalty
        if np.any(self._prev_ctrl_valid):
            ctrl_range = np.maximum(self._ctrl_hi - self._ctrl_lo, 1.0e-6)[None, :]
            ctrl_delta = (ctrl - self._prev_ctrl) / ctrl_range
            valid_mask = self._prev_ctrl_valid.astype(np.float32)
            reward += (
                -0.15 * self.mt_reward_w_smooth
                * np.mean(ctrl_delta * ctrl_delta, axis=1)
                * valid_mask
            )

        # Control effort regularization
        ctrl_center = (0.5 * (self._ctrl_lo + self._ctrl_hi))[None, :]
        ctrl_half_range = np.maximum(0.5 * (self._ctrl_hi - self._ctrl_lo), 1.0e-6)[None, :]
        ctrl_effort = (ctrl - ctrl_center) / ctrl_half_range
        reward += -0.05 * self.mt_reward_w_smooth * np.mean(ctrl_effort * ctrl_effort, axis=1)

        # Squeeze amplitude bonus for wave mode: encourage active deformation
        if self.control_mode == "wave":
            squeeze_abs = np.abs(action[:, 0])
            squeeze_bonus = 0.15 * squeeze_abs / (squeeze_abs + 0.3 + 1e-6)
            reward += squeeze_bonus

            # Asymmetry usage bonus: encourage using squeeze_asym for turn tasks
            # and wave_bias for ascend/descend tasks
            for w in range(action.shape[0]):
                task = self._task_ids[w]
                if task == TASK_TURN_LEFT or task == TASK_TURN_RIGHT:
                    asym_abs = abs(action[w, 4])
                    reward[w] += 0.05 * asym_abs / (asym_abs + 0.3 + 1e-6)
                elif task == TASK_ASCEND or task == TASK_DESCEND:
                    bias_abs = abs(action[w, 5])
                    reward[w] += 0.05 * bias_abs / (bias_abs + 0.3 + 1e-6)

        self._prev_ctrl[:] = ctrl
        self._prev_ctrl_valid[:] = True

        # Get final terminated state
        terminated = self._terminated_buffer.numpy().astype(bool)

        # Termination penalty
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

        # Get observation (after any task switch)
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

            # Ring center position (average of 12 segments)
            all_pos = self.lbm_solver.flows[w].solid_position.numpy()  # (n_solids, 3)
            center = np.mean(all_pos[:12], axis=0)
            body_positions[w] = [center[0] / self.nx, center[1] / self.ny, center[2] / self.nz]

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

        # Call grandparent reset (LBMFluidEnv3D.reset), skip Torus3DLBMEnv's goal logic
        from ..lbm_fluid_env_3d import LBMFluidEnv3D
        LBMFluidEnv3D.reset(self, seed=seed, options=options)

        # Sample new tasks
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
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        return self._get_obs()

    def partial_reset(self, reset_mask: np.ndarray) -> np.ndarray:
        """Reset specific worlds and re-sample their tasks."""
        from ..lbm_fluid_env_3d import LBMFluidEnv3D
        LBMFluidEnv3D.partial_reset(self, reset_mask)

        if not np.any(reset_mask):
            return self._get_obs()

        if self._command_sequence is None:
            self._sample_tasks(reset_mask)

        self._steps_since_switch[reset_mask] = 0
        self._time_val[reset_mask] = 0.0
        self._prev_ctrl[reset_mask] = 0.0
        self._prev_ctrl_valid[reset_mask] = False
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        return self._get_obs()

    def get_current_task(self, world_idx: int = 0) -> str:
        """Get current task name for a world."""
        return TASK_NAMES[self._task_ids[world_idx]]

    def get_current_goal(self, world_idx: int = 0) -> Tuple[float, float, float]:
        """Override for compatibility: return center position as dummy goal."""
        return (0.5, 0.5, 0.5)
