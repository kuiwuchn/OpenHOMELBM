"""
3D Clownfish Multi-Task LBM Environment with optional PD control modes.

Multi-task training for clownfish locomotion skills.
Inherits from Clownfish3DLBMEnv but replaces goal-reaching with velocity-based
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

For smooth motion, joints within a segment share control signals:
  Group 0: Body lateral      (body_yaw)                        — 1 joint → 1 signal
  Group 1: Body vertical     (body_pitch)                      — 1 joint → 1 signal
  Group 2: Peduncle lateral  (peduncle_yaw)                    — 1 joint → 1 signal
  Group 3: Peduncle vertical (peduncle_pitch)                  — 1 joint → 1 signal
  Group 4: Caudal lateral    (caudal_rotate, caudal_yaw)       — 2 joints → 1 signal
  Group 5: Caudal vertical   (caudal_pitch)                    — 1 joint → 1 signal

Total: N_groups=6, K=2 harmonics → network outputs 6 * 2 * 2 = 24 values (A + C)

Note: Caudal pitch is separated from caudal lateral so the agent can
independently control vertical tail motion for ascend/descend tasks.

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

from .clownfish_lbm_env_3d import (
    Clownfish3DLBMEnv,
    compute_clownfish_obs_3d_kernel,
    check_boundary_3d_clownfish_kernel,
    check_stability_3d_clownfish_kernel,
    apply_instability_penalty_clownfish_kernel,
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


# ============== Clownfish-specific Reward Kernel ==============
# Adapted from manta's compute_multitask_reward_kernel with key changes:
#   - ASCEND/DESCEND: forward velocity weight raised from 15% to 35%
#     (clownfish must pitch body and swim forward to gain vertical speed)
#   - ASCEND/DESCEND: pitch posture penalty reduced from 0.25 to 0.08
#     (allow larger pitch angles needed for vertical thrust)
#   - target_vertical_vel should be set lower (~0.04) in constructor

@wp.kernel
def compute_clownfish_multitask_reward_kernel(
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
    Clownfish-specific task-conditioned reward.

    Key differences from manta kernel:
      - ASCEND/DESCEND use 65/35 vertical/forward split (manta: 85/15)
        because clownfish generates vertical motion by pitching + swimming forward.
      - Pitch posture penalty during ASCEND/DESCEND is greatly reduced (0.08 vs 0.25)
        to allow the body to tilt for vertical thrust.
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
        r_task = w_task * directional_reward(v_forward, target_forward_vel)
        r_upright = -w_roll * (0.50 * upright_roll + 0.50 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.60 * soft_penalty(v_lateral, target_forward_vel)
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
        # Clownfish-specific: 65% vertical + 35% forward (manta: 85/15)
        # Clownfish must pitch body and swim forward to gain vertical speed
        r_task = w_task * (
            0.65 * directional_reward(v_vertical, target_vertical_vel)
            + 0.35 * positive_reward(v_forward, vertical_forward_scale)
        )
        # Greatly reduced pitch penalty (0.08 vs manta's 0.25)
        # to allow body tilting for vertical thrust
        r_upright = -w_roll * (0.15 * upright_roll + 0.08 * upright_pitch)
        r_offaxis = -w_offaxis * (
            0.45 * soft_penalty(v_lateral, target_forward_vel)
            + 0.15 * soft_penalty(yaw_rate, target_yaw_rate)
        )

    elif task == TASK_DESCEND:
        # Same 65/35 split as ASCEND
        r_task = w_task * (
            0.65 * directional_reward(-v_vertical, target_vertical_vel)
            + 0.35 * positive_reward(v_forward, vertical_forward_scale)
        )
        # Same reduced pitch penalty
        r_upright = -w_roll * (0.15 * upright_roll + 0.08 * upright_pitch)
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

# Reduced-order joint groups for Clownfish (7 joints → 6 groups)
# Each group: (group_name, [actuator_indices])
# Actuator order from XML:
#   0: pos_body_yaw        Body: lateral swing (yaw around Z)
#   1: pos_body_pitch      Body: vertical pitch (around X)
#   2: pos_peduncle_yaw    Peduncle: lateral swing
#   3: pos_peduncle_pitch  Peduncle: vertical pitch
#   4: pos_caudal_rotate   Caudal fin: rotation / twist (around Y)
#   5: pos_caudal_yaw      Caudal fin: lateral swing (largest range)
#   6: pos_caudal_pitch    Caudal fin: vertical pitch
#
# Grouping rationale:
# - body and peduncle yaw/pitch are independent (lateral vs vertical motion)
# - caudal fin lateral (rotate+yaw) grouped together for horizontal thrust
# - caudal fin pitch is SEPARATE so vertical motion can be controlled
#   independently from horizontal tail swing (critical for ascend/descend)
REDUCED_ORDER_GROUPS = [
    ("body_lateral",     [0]),        # body_yaw
    ("body_vertical",    [1]),        # body_pitch
    ("peduncle_lateral", [2]),        # peduncle_yaw
    ("peduncle_vertical",[3]),        # peduncle_pitch
    ("caudal_lateral",   [4, 5]),    # caudal_rotate + caudal_yaw
    ("caudal_vertical",  [6]),        # caudal_pitch (independent for vertical control)
]
N_GROUPS = len(REDUCED_ORDER_GROUPS)  # 6


# ============== Multi-Task Environment Class ==============


class ClownfishMultiTaskEnv(Clownfish3DLBMEnv):
    """
    Multi-task clownfish locomotion environment with frequency-domain PD control.

    Instead of goal-reaching, the agent is given a task ID (one-hot in obs)
    and rewarded for performing the corresponding motion primitive:
      forward, turn_left, turn_right, ascend, descend.

    Control: The neural network outputs frequency-domain parameters (A, C)
    which are converted to joint target angles via Fourier synthesis,
    then tracked by MuJoCo's built-in PD controllers.

    Reduced-order: Caudal fin lateral joints share one control signal,
    while caudal pitch is independent for vertical motion control.

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
        # Task reward weights (tuned for clownfish: smaller body, tail-driven)
        reward_w_task: float = 1.0,
        reward_w_roll: float = 0.12,        # slightly higher than manta (0.10): slender body rolls easier
        reward_w_smooth: float = 0.02,      # higher than manta (0.015): serial chain amplifies end-effector
        reward_w_offaxis: float = 0.03,     # higher than manta (0.02): tail swing causes lateral drift
        # Reference targets / scales (lower than manta: smaller body, less thrust)
        target_forward_vel: float = 0.10,   # manta: 0.20
        target_yaw_rate: float = 0.10,      # manta: 0.15
        target_vertical_vel: float = 0.04,  # lowered from 0.08: clownfish vertical speed is indirect
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
        if self.control_mode not in {"frequency", "direct"}:
            raise ValueError(f"Unknown control_mode '{control_mode}'. Expected 'frequency' or 'direct'.")

        # --- Frequency-domain / direct PD control ---
        self.k_harmonics = k_harmonics
        self.b_bar = b_bar
        self.use_reduced_order = use_reduced_order

        if self.control_mode == "frequency" and use_reduced_order:
            self.n_ctrl_groups = N_GROUPS  # 5
            self.joint_groups = REDUCED_ORDER_GROUPS
        else:
            # Direct control always uses per-actuator targets
            self.n_ctrl_groups = self.n_actuators  # 7
            self.joint_groups = [
                (f"joint_{i}", [i]) for i in range(self.n_actuators)
            ]

        # Action dimension: N_groups * K * 2 (A + C for each harmonic)
        self.freq_action_dim = self.n_ctrl_groups * k_harmonics * 2

        # Build group-to-actuator mapping
        self._group_actuator_indices = [g[1] for g in self.joint_groups]

        # Extract ctrl_range from MuJoCo model for scaling θ* → ctrl
        self._ctrl_lo = self.mj_model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.mj_model.actuator_ctrlrange[:, 1].copy()

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

        # --- Override obs_dim: base(22+3*n_joints) + lbm(3) + task(NUM_TASKS) + phase(2) ---
        self.obs_dim = 22 + 3 * self.n_joints + 3 + NUM_TASKS + 2

        # Current task for each world
        self._task_ids = np.zeros(nworld, dtype=np.int32)
        self._task_ids_wp = wp.zeros(nworld, dtype=wp.int32, device=self.device)

        # Steps since last task switch
        self._steps_since_switch = np.zeros(nworld, dtype=np.int32)

        # Override observation buffer with new size
        self._obs_buffer = wp.zeros((nworld, self.obs_dim), dtype=wp.float32, device=self.device)

        if self.control_mode == "frequency":
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

        print(f"ClownfishMultiTaskEnv initialized:")
        print(f"  Tasks: {[TASK_NAMES[i] for i in self.enabled_task_ids]}")
        print(f"  Task switch interval: {task_switch_interval} (0=only at reset)")
        print(f"  Control mode: {self.control_mode}")
        if self.control_mode == "frequency":
            print(f"  Control: K={k_harmonics} harmonics, B̄={b_bar}, reduced_order={use_reduced_order}")
            print(f"  Groups ({self.n_ctrl_groups}): {[g[0] for g in self.joint_groups]}")
            print(f"  Action dim: {self.freq_action_dim} (frequency params)")
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

    def _compute_reward(self, instability_mask=None) -> np.ndarray:
        """Task-conditioned reward computation (clownfish-specific kernel)."""
        wp.launch(
            compute_clownfish_multitask_reward_kernel,
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
        return self._freq_to_ctrl(action)

    def _freq_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """
        Convert frequency-domain action parameters to MuJoCo ctrl (target angles).

        Action layout per group g (K harmonics):
            [A_g0, C_g0, A_g1, C_g1, ..., A_g(K-1), C_g(K-1)]

        θ*_g = Σ_{j=0}^{K-1} A_{gj} * sin(π/2 * B_j * t + C_{gj})
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
        Direct mode sends actuator target angles to MuJoCo directly.
        """
        ctrl = self._action_to_ctrl(action)
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        # Write target angles to MuJoCo ctrl
        wp.copy(self.mjw_data.ctrl, wp.array(ctrl, dtype=wp.float32, device=self.device))

        # Only advance the phase clock when the policy controls Fourier coefficients
        if self.control_mode == "frequency":
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

        # Call grandparent reset (LBMFluidEnv3D.reset), skip Clownfish3DLBMEnv's goal logic
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
        wp.copy(self._prev_qpos_buffer, self.mjw_data.qpos)

        return self._get_obs()

    def get_current_task(self, world_idx: int = 0) -> str:
        """Get current task name for a world."""
        return TASK_NAMES[self._task_ids[world_idx]]

    def get_current_goal(self, world_idx: int = 0) -> Tuple[float, float, float]:
        """Override for compatibility: return center position as dummy goal."""
        return (0.5, 0.5, 0.5)
