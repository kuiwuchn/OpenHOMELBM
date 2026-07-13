"""
3D Sea Turtle LBM Environment for MuJoCo Warp with nworld support.

Sea turtle swimming with articulated front/rear flippers and tail.
Based on the Manta Ray implementation pattern (Multi-Goal reward).
Uses gym (not gymnasium) for compatibility with dreamer_vec_wrapper.
All data processing uses Warp kernels - numpy only at entry/exit points.

Structure (v5 — ball-like shoulder on front flippers):
  shell (freejoint, 6DOF root)
    head (fixed, no joints)
    flipper_FR (3DOF: rotate Y + flap X + sweep Z)
    flipper_FL (3DOF: rotate Y + flap X + sweep Z)   [mirrored]
    flipper_RR (2DOF: rotate Z + flap X)
    flipper_RL (2DOF: rotate Z + flap X)   [mirrored]
    tail (1DOF: yaw)

Total actuators: 3*2 + 2*2 + 1 = 11 DOF
Coordinate: X=lateral (right+), Y=forward, Z=up
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


# ============== Warp Kernels for 3D Turtle Environment ==============


@wp.kernel
def compute_turtle_obs_3d_kernel(
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

    Observation layout for 3D Sea Turtle:
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
    center_pos = flow.solid_position[0]  # shell is solid 0
    obs_out[world_idx, idx] = center_pos[0] / nx
    obs_out[world_idx, idx + 1] = center_pos[1] / ny
    obs_out[world_idx, idx + 2] = center_pos[2] / nz
    idx = idx + 3

    # Goal position (normalized) (3)
    obs_out[world_idx, idx] = goal_positions[world_idx, 0]
    obs_out[world_idx, idx + 1] = goal_positions[world_idx, 1]
    obs_out[world_idx, idx + 2] = goal_positions[world_idx, 2]


@wp.kernel
def check_boundary_3d_turtle_kernel(
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
def check_stability_3d_turtle_kernel(
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
def check_goal_reached_turtle_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    goal_reached_out: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    goal_threshold: float,
):
    """Check if turtle reached its goal."""
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
def apply_instability_penalty_turtle_kernel(
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
def compute_goal_reward_turtle_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    prev_dist: wp.array(dtype=wp.float32),
    rewards_out: wp.array(dtype=wp.float32),
    current_dist_out: wp.array(dtype=wp.float32),
    nx: float,
    ny: float,
    nz: float,
):
    """Compute reward based on distance improvement to goal."""
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
    current_dist = wp.sqrt(dx * dx + dy * dy + dz * dz)

    dist_improvement = prev_dist[world_idx] - current_dist
    rewards_out[world_idx] = 100.0 * dist_improvement

    current_dist_out[world_idx] = current_dist


@wp.kernel
def reset_prev_dist_turtle_kernel(
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
def init_prev_dist_turtle_kernel(
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


# ============== 3D Sea Turtle LBM Environment Class ==============


class Turtle3DLBMEnv(LBMFluidEnv3D):
    """
    3D Sea Turtle swimming environment with LBM fluid simulation.
    Multi-Goal reward: distance improvement toward goal + goal reached bonus.

    Structure (v4 — rotate + flap on all flippers):
    - Shell (carapace, root body with freejoint)
    - Head (fixed to shell)
    - Front flippers (left/right), each single segment (2DOF: rotate + flap)
    - Rear flippers (left/right), each 2DOF (rotate + flap)
    - Tail (1DOF yaw)

    Actuators (11 DOF):
      Front-right:  f_FR_rotate, f_FR_flap, f_FR_sweep   (3)
      Front-left:   f_FL_rotate, f_FL_flap, f_FL_sweep   (3)
      Rear-right:   f_RR_rotate, f_RR_flap   (2)
      Rear-left:    f_RL_rotate, f_RL_flap   (2)
      Tail:         tail_yaw                  (1)

    Control: position actuators with ctrlrange normalized to [-1, 1]
    Coordinate: X=lateral (right+), Y=forward, Z=up
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
        goal_threshold: float = 0.08,
        single_goal_mode: bool = True,
        goal_position: Optional[List[float]] = None,
    ):
        # Store goal mode settings before super().__init__
        self._init_single_goal_mode = single_goal_mode
        self._init_goal_position = goal_position if goal_position is not None else [0.5, 0.75, 0.5]

        if mjcf_path is None:
            mjcf_path = os.path.join(os.path.dirname(__file__), 'turtle_3d.xml')

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

        # Observation buffer
        self._obs_buffer = wp.zeros((nworld, self.obs_dim), dtype=wp.float32, device=self.device)

        # Terminated buffer
        self._terminated_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._instability_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._goal_reached_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)
        self._rewards_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)

        # Print configuration info
        print(f"Turtle3DLBMEnv initialized:")
        print(f"  Grid: {nx}x{ny}x{nz}, scale: {lbm_scale}")
        print(f"  Bodies: {self.n_bodies}, Joints: {self.n_joints}, Actuators: {self.n_actuators}")
        print(f"  Solids (total): {self.solid_num}, Dynamic: {self.n_dynamic}, Static: {self.n_static}")
        print(f"  Obs dim: {self.obs_dim}, Action dim: {self.n_actuators}")
        print(f"  Force conversion: {self.force_conversion:.6f}")
        print(f"  Torque conversion: {self.torque_conversion:.8f}")
        print(f"  Goal mode: {'single' if self.single_goal_mode else 'multi'}, "
              f"num_goals: {self.num_goals}, threshold: {self.goal_threshold}")

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
            compute_turtle_obs_3d_kernel,
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
        """Distance-based reward toward goal."""
        wp.launch(
            compute_goal_reward_turtle_kernel,
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

        return self._rewards_buffer.numpy()

    def _check_goals_reached(self) -> np.ndarray:
        """Check if any world reached its goal."""
        wp.launch(
            check_goal_reached_turtle_kernel,
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
                reset_prev_dist_turtle_kernel,
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
        self._terminated_buffer.zero_()

        # Check boundary
        wp.launch(
            check_boundary_3d_turtle_kernel,
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

        # Check stability
        if instability_mask is not None:
            instability_wp = wp.array(instability_mask.astype(np.int32), dtype=wp.int32, device=self.device)
            wp.launch(
                apply_instability_penalty_turtle_kernel,
                dim=self.nworld,
                inputs=[self._rewards_buffer, self._terminated_buffer, instability_wp, 0.0],
                device=self.device,
            )

        return self._terminated_buffer.numpy().astype(bool)

    def _check_numerical_stability(self) -> Optional[np.ndarray]:
        """Check for NaN/Inf in state."""
        wp.launch(
            check_stability_3d_turtle_kernel,
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
        return self._instability_buffer.numpy().copy()

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

        # Handle NaN/Inf in observations
        obs_nan_mask = np.any(np.isnan(observation) | np.isinf(observation), axis=1)
        if np.any(obs_nan_mask):
            instability_mask = instability_mask | obs_nan_mask
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

        # Check termination and keep a clean copy of boundary terminations
        self._is_terminated(instability_mask)
        boundary_terminated = self._terminated_buffer.numpy().astype(bool).copy()

        # Compute reward
        reward = self._compute_reward(instability_mask)

        # Get final terminated state
        terminated = self._terminated_buffer.numpy().astype(bool)

        anomaly_terminated = np.zeros(self.nworld, dtype=bool)

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
            reward[reward_nan_mask] = -1.0
            terminated[reward_nan_mask] = True

        truncated = np.array(self.step_counts >= self.max_episode_steps)
        done = terminated | truncated

        # Build termination reason per world
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
            init_prev_dist_turtle_kernel,
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
            reset_prev_dist_turtle_kernel,
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
