"""
3D Eel/Ribbon Fish LBM Environment with Static Obstacle

The eel swims around a static obstacle toward a goal.
Extensions to Eel3DLBMEnv:
1. A jointless static obstacle
2. Obstacle-relative position in observations
3. An obstacle-avoidance reward penalty
4. An obstacle-distance termination check
"""

import gym
from gym import spaces
import numpy as np
import warp as wp
import os
import mujoco
from typing import Optional, Tuple, Dict, Any, List

from ..eel.eel_lbm_env_3d import (
    Eel3DLBMEnv,
    compute_eel_obs_kernel,
    compute_goal_reward_eel_kernel,
    compute_smooth_reward_eel_kernel,
    add_smooth_rewards_kernel,
    apply_instability_penalty_kernel,
    check_anomaly_kernel,
    check_stability_eel_kernel,
    check_boundary_eel_kernel,
    reset_prev_dist_kernel,
    reset_prev_actions_kernel,
    init_prev_dist_kernel,
    check_goal_reached_eel_kernel,
)
from ..lbm_core_3d import HomeFlow3D


# ============== Warp Kernels for Obstacle Environment ==============


@wp.kernel
def compute_obstacle_obs_kernel(
    qfrc_applied: wp.array2d(dtype=wp.float32),
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    flows: wp.array(dtype=HomeFlow3D),
    goal_positions: wp.array2d(dtype=wp.float32),
    obs_out: wp.array2d(dtype=wp.float32),
    dynamic_solid_ids: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    n_joints: int,
    obstacle_x: float,
    obstacle_y: float,
    obstacle_z: float,
):
    """
    Compute observation for eel with obstacle.
    
    Layout = base eel obs (67 dims) + obstacle relative position (3 dims)
    
    Base obs (25 + 3 * n_joints = 67 for 14 joints):
    - Forces (6): fx, fy, fz, tau_x, tau_y, tau_z
    - Joint torques (n_joints): joint generalized forces
    - Position (3): x, y, z
    - Quaternion (4): w, x, y, z
    - Velocity (3): vx, vy, vz
    - Angular velocity (3): omega_x, omega_y, omega_z
    - Joint angles (n_joints)
    - Joint velocities (n_joints)
    - LBM position (3): normalized x, y, z
    - Goal position (3): normalized goal x, y, z
    
    Additional (3):
    - Obstacle relative position (3): normalized (obs_x - head_x, obs_y - head_y, obs_z - head_z)
    
    Total: 67 + 3 = 70 dims
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
    
    # Position (3)
    obs_out[world_idx, idx] = qpos[world_idx, 0]
    obs_out[world_idx, idx + 1] = qpos[world_idx, 1]
    obs_out[world_idx, idx + 2] = qpos[world_idx, 2]
    idx = idx + 3
    
    # Quaternion (4)
    obs_out[world_idx, idx] = qpos[world_idx, 3]
    obs_out[world_idx, idx + 1] = qpos[world_idx, 4]
    obs_out[world_idx, idx + 2] = qpos[world_idx, 5]
    obs_out[world_idx, idx + 3] = qpos[world_idx, 6]
    idx = idx + 4
    
    # Velocity (3)
    obs_out[world_idx, idx] = qvel[world_idx, 0]
    obs_out[world_idx, idx + 1] = qvel[world_idx, 1]
    obs_out[world_idx, idx + 2] = qvel[world_idx, 2]
    idx = idx + 3
    
    # Angular velocity (3)
    obs_out[world_idx, idx] = qvel[world_idx, 3]
    obs_out[world_idx, idx + 1] = qvel[world_idx, 4]
    obs_out[world_idx, idx + 2] = qvel[world_idx, 5]
    idx = idx + 3
    
    # Joint angles (n_joints)
    for i in range(n_joints):
        obs_out[world_idx, idx] = qpos[world_idx, 7 + i]
        idx = idx + 1
    
    # Joint velocities (n_joints)
    for i in range(n_joints):
        obs_out[world_idx, idx] = qvel[world_idx, 6 + i]
        idx = idx + 1
    
    # LBM position (normalized) - use dynamic_solid_ids[0] for head (first dynamic solid)
    head_solid_id = dynamic_solid_ids[0]
    head_pos = flow.solid_position[head_solid_id]
    head_x = head_pos[0] / nx
    head_y = head_pos[1] / ny
    head_z = head_pos[2] / nz
    obs_out[world_idx, idx] = head_x
    obs_out[world_idx, idx + 1] = head_y
    obs_out[world_idx, idx + 2] = head_z
    idx = idx + 3
    
    # Goal position (normalized)
    obs_out[world_idx, idx] = goal_positions[world_idx, 0]
    obs_out[world_idx, idx + 1] = goal_positions[world_idx, 1]
    obs_out[world_idx, idx + 2] = goal_positions[world_idx, 2]
    idx = idx + 3
    
    # Obstacle relative position (normalized): obstacle - head
    obs_out[world_idx, idx] = obstacle_x - head_x
    obs_out[world_idx, idx + 1] = obstacle_y - head_y
    obs_out[world_idx, idx + 2] = obstacle_z - head_z


@wp.kernel
def compute_obstacle_avoidance_reward_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    avoidance_rewards_out: wp.array(dtype=wp.float32),
    dynamic_solid_ids: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    obstacle_x: float,
    obstacle_y: float,
    obstacle_z: float,
    safe_radius: float,
    avoidance_weight: float,
    n_dynamic_solids: int,
):
    """
    Compute obstacle avoidance penalty for all dynamic solids (eel segments).
    
    Penalizes each segment that enters the safe radius around the obstacle.
    Penalty is linear: max at obstacle center, 0 at safe_radius boundary.
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    total_penalty = float(0.0)
    
    for i in range(n_dynamic_solids):
        solid_idx = dynamic_solid_ids[i]
        pos = flow.solid_position[solid_idx]
        sx = pos[0] / nx
        sy = pos[1] / ny
        sz = pos[2] / nz
        
        dx = sx - obstacle_x
        dy = sy - obstacle_y
        dz = sz - obstacle_z
        dist = wp.sqrt(dx * dx + dy * dy + dz * dz)
        
        if dist < safe_radius:
            penalty = avoidance_weight * (1.0 - dist / safe_radius)
            total_penalty = total_penalty + penalty
    
    avoidance_rewards_out[world_idx] = -total_penalty


@wp.kernel
def check_obstacle_collision_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    terminated_out: wp.array(dtype=wp.int32),
    dynamic_solid_ids: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    obstacle_x: float,
    obstacle_y: float,
    obstacle_z: float,
    collision_radius: float,
    n_dynamic_solids: int,
):
    """
    Check if any eel segment collides with the obstacle.
    Collision = any segment center within collision_radius of obstacle center.
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    for i in range(n_dynamic_solids):
        solid_idx = dynamic_solid_ids[i]
        pos = flow.solid_position[solid_idx]
        sx = pos[0] / nx
        sy = pos[1] / ny
        sz = pos[2] / nz
        
        dx = sx - obstacle_x
        dy = sy - obstacle_y
        dz = sz - obstacle_z
        dist_sq = dx * dx + dy * dy + dz * dz
        
        if dist_sq < collision_radius * collision_radius:
            terminated_out[world_idx] = 1
            return


@wp.kernel
def add_avoidance_rewards_kernel(
    rewards: wp.array(dtype=wp.float32),
    avoidance_rewards: wp.array(dtype=wp.float32),
):
    """Add avoidance rewards to main rewards buffer."""
    world_idx = wp.tid()
    rewards[world_idx] = rewards[world_idx] + avoidance_rewards[world_idx]


# ============== Eel 3D Obstacle LBM Environment ==============


class Eel3DObstacleLBMEnv(Eel3DLBMEnv):
    """
    3D Eel swimming environment with a static obstacle.
    
    The eel must swim around a static obstacle toward the goal.
    The obstacle blocks LBM flow without receiving fluid-force feedback.
    
    Changes from base Eel3DLBMEnv:
    - Uses eel_obstacle_3d.xml (includes static obstacle body)
    - Observation extended with obstacle relative position (+3 dims)
    - Reward includes obstacle avoidance penalty
    - Termination includes obstacle collision check
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
        goal_threshold: float = 0.08,
        single_goal_mode: bool = True,
        goal_position: Optional[List[float]] = None,
        control_mode: str = 'direct',
        K: int = 3,
        B_bar: float = 1.0,
        # Obstacle parameters
        obstacle_avoidance_weight: float = 5.0,
        obstacle_safe_radius: float = 0.12,
        obstacle_collision_radius: float = 0.05,
    ):
        """
        Initialize Eel 3D Obstacle LBM Environment.
        
        Additional Args (beyond Eel3DLBMEnv):
            obstacle_avoidance_weight: Weight of avoidance penalty in reward
            obstacle_safe_radius: Normalized radius within which avoidance penalty activates
            obstacle_collision_radius: Normalized radius for collision termination
        """
        # Store obstacle params before super().__init__
        self._obstacle_avoidance_weight = obstacle_avoidance_weight
        self._obstacle_safe_radius = obstacle_safe_radius
        self._obstacle_collision_radius = obstacle_collision_radius
        
        # Use obstacle XML
        if mjcf_path is None:
            mjcf_path = os.path.join(os.path.dirname(__file__), 'eel_obstacle_3d.xml')
        
        # Call parent __init__
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
            goal_threshold=goal_threshold,
            single_goal_mode=single_goal_mode,
            goal_position=goal_position,
            control_mode=control_mode,
            K=K,
            B_bar=B_bar,
        )
        
        # Store obstacle parameters
        self.obstacle_avoidance_weight = obstacle_avoidance_weight
        self.obstacle_safe_radius = obstacle_safe_radius
        self.obstacle_collision_radius = obstacle_collision_radius
        
        # Find obstacle body and compute its LBM position (normalized)
        self._obstacle_body_name = 'obstacle'
        self._obstacle_pos_normalized = self._compute_obstacle_position_normalized()
        
        # Extended observation: base obs (67) + obstacle relative pos (3) = 70
        self.obs_dim = 25 + 3 * self.n_actuators + 3  # 67 + 3 = 70
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.nworld, self.obs_dim),
            dtype=np.float32
        )
        
        # Re-allocate obs buffer with new size
        self._obs_buffer = wp.zeros((nworld, self.obs_dim), dtype=wp.float32, device=self.device)
        
        # Avoidance reward buffer
        self._avoidance_rewards_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)
    
    def _compute_obstacle_position_normalized(self) -> Tuple[float, float, float]:
        """
        Compute obstacle position in normalized LBM coordinates [0, 1].
        First tries link_config, then falls back to MuJoCo body position.
        """
        # Try link_config first
        for cfg in self.link_config:
            if cfg['link_name'] == self._obstacle_body_name:
                lbm_pos = cfg.get('lbm_position', (self.nx * 0.5, self.ny * 0.5, self.nz * 0.5))
                return (
                    lbm_pos[0] / self.nx,
                    lbm_pos[1] / self.ny,
                    lbm_pos[2] / self.nz,
                )
        
        # Fallback: compute from MuJoCo body position
        # obstacle is a static body not connected via joints, so it may not
        # appear in link_config (auto-generated from BFS over joint tree).
        # Get its position from MuJoCo and convert to LBM normalized coords.
        obstacle_body_id = self._link_name_to_body_id.get(self._obstacle_body_name)
        if obstacle_body_id is None:
            raise ValueError(f"Obstacle body '{self._obstacle_body_name}' not found in MuJoCo model. "
                            f"Available bodies: {self.body_names}")
        
        # Get obstacle position in MuJoCo world frame (from mj_forward)
        obstacle_mj_pos = self.mj_data.xpos[obstacle_body_id]  # (3,)
        
        # Get root body position in MuJoCo world frame
        root_cfg = self.link_config[0]  # root link is first in config
        root_body_id = root_cfg['body_id']
        root_mj_pos = self.mj_data.xpos[root_body_id]  # (3,)
        root_lbm_pos = np.array(root_cfg.get('lbm_position', (self.nx * 0.5, self.ny * 0.5, self.nz * 0.5)))
        
        # Convert: lbm_pos = root_lbm_pos + (obstacle_mj - root_mj) * coordinate_scale
        offset = obstacle_mj_pos - root_mj_pos
        lbm_pos = root_lbm_pos + offset * self.coordinate_scale
        
        print(f"[Eel3DObstacle] Obstacle MuJoCo pos: {obstacle_mj_pos}, "
              f"LBM pos: {lbm_pos}")
        
        return (
            float(lbm_pos[0]) / self.nx,
            float(lbm_pos[1]) / self.ny,
            float(lbm_pos[2]) / self.nz,
        )
    
    def _get_obs(self) -> np.ndarray:
        """Get observations with obstacle info."""
        ox, oy, oz = self._obstacle_pos_normalized
        
        wp.launch(
            compute_obstacle_obs_kernel,
            dim=self.nworld,
            inputs=[
                self.mjw_data.qfrc_applied,
                self.mjw_data.qpos,
                self.mjw_data.qvel,
                self.lbm_solver.flows_wp,
                self._goal_positions_wp,
                self._obs_buffer,
                self.dynamic_solid_ids_wp,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.n_joints,
                ox,
                oy,
                oz,
            ],
            device=self.device,
        )
        return self._obs_buffer.numpy()
    
    def _compute_reward(self, instability_mask=None) -> np.ndarray:
        """Compute reward with obstacle avoidance penalty."""
        # Get base reward (goal distance + smoothness + wave)
        reward = super()._compute_reward(instability_mask)
        
        # Add obstacle avoidance penalty
        ox, oy, oz = self._obstacle_pos_normalized
        
        wp.launch(
            compute_obstacle_avoidance_reward_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._avoidance_rewards_buffer,
                self.dynamic_solid_ids_wp,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                ox,
                oy,
                oz,
                self.obstacle_safe_radius,
                self.obstacle_avoidance_weight,
                self.n_dynamic,
            ],
            device=self.device,
        )
        
        # Reload reward from buffer (super may have modified it via numpy)
        wp.copy(self._rewards_buffer, wp.array(reward, dtype=wp.float32, device=self.device))
        
        wp.launch(
            add_avoidance_rewards_kernel,
            dim=self.nworld,
            inputs=[self._rewards_buffer, self._avoidance_rewards_buffer],
            device=self.device,
        )
        
        return self._rewards_buffer.numpy()
    
    def _is_terminated(self, instability_mask=None) -> np.ndarray:
        """Check termination with obstacle collision."""
        # Base termination checks (boundary + instability)
        base_terminated = super()._is_terminated(instability_mask)
        
        # Additional: obstacle collision check
        ox, oy, oz = self._obstacle_pos_normalized
        
        wp.launch(
            check_obstacle_collision_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._terminated_buffer,
                self.dynamic_solid_ids_wp,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                ox,
                oy,
                oz,
                self.obstacle_collision_radius,
                self.n_dynamic,
            ],
            device=self.device,
        )
        
        return self._terminated_buffer.numpy().astype(bool)
