"""
3D Manta Ray LBM Environment with Static Terrain Geometry.

`Manta3DTerrainLBMEnv` extends the current manta goal-reaching environment with
static geometry that blocks the fluid and adds an avoidance penalty.
`Manta3DObstacleLBMEnv` is kept as a compatibility wrapper for the original
single-obstacle scene.
"""

from gym import spaces
import numpy as np
import warp as wp
import os
from typing import Optional, Tuple, List

from ..manta.manta_lbm_env_3d import Manta3DLBMEnv
from ..lbm_core_3d import HomeFlow3D


@wp.kernel
def compute_manta_terrain_obs_kernel(
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
    n_dynamic: int,
    n_solids: int,
):
    """Base manta observation plus relative position to the nearest static terrain body."""
    world_idx = wp.tid()
    flow = flows[world_idx]

    idx = 0

    for i in range(6):
        obs_out[world_idx, idx] = qfrc_applied[world_idx, i]
        idx = idx + 1

    for i in range(n_joints):
        obs_out[world_idx, idx] = qfrc_applied[world_idx, 6 + i]
        idx = idx + 1

    obs_out[world_idx, idx] = qpos[world_idx, 0]
    obs_out[world_idx, idx + 1] = qpos[world_idx, 1]
    obs_out[world_idx, idx + 2] = qpos[world_idx, 2]
    idx = idx + 3

    obs_out[world_idx, idx] = qpos[world_idx, 3]
    obs_out[world_idx, idx + 1] = qpos[world_idx, 4]
    obs_out[world_idx, idx + 2] = qpos[world_idx, 5]
    obs_out[world_idx, idx + 3] = qpos[world_idx, 6]
    idx = idx + 4

    obs_out[world_idx, idx] = qvel[world_idx, 0]
    obs_out[world_idx, idx + 1] = qvel[world_idx, 1]
    obs_out[world_idx, idx + 2] = qvel[world_idx, 2]
    idx = idx + 3

    obs_out[world_idx, idx] = qvel[world_idx, 3]
    obs_out[world_idx, idx + 1] = qvel[world_idx, 4]
    obs_out[world_idx, idx + 2] = qvel[world_idx, 5]
    idx = idx + 3

    for i in range(n_joints):
        obs_out[world_idx, idx] = qpos[world_idx, 7 + i]
        idx = idx + 1

    for i in range(n_joints):
        obs_out[world_idx, idx] = qvel[world_idx, 6 + i]
        idx = idx + 1

    center_pos = flow.solid_position[0]
    body_x = center_pos[0] / nx
    body_y = center_pos[1] / ny
    body_z = center_pos[2] / nz
    obs_out[world_idx, idx] = body_x
    obs_out[world_idx, idx + 1] = body_y
    obs_out[world_idx, idx + 2] = body_z
    idx = idx + 3

    obs_out[world_idx, idx] = goal_positions[world_idx, 0]
    obs_out[world_idx, idx + 1] = goal_positions[world_idx, 1]
    obs_out[world_idx, idx + 2] = goal_positions[world_idx, 2]
    idx = idx + 3

    nearest_dx = float(0.0)
    nearest_dy = float(0.0)
    nearest_dz = float(0.0)
    min_dist_sq = float(1.0e18)
    for static_idx in range(n_dynamic, n_solids):
        pos = flow.solid_position[static_idx]
        tx = pos[0] / nx
        ty = pos[1] / ny
        tz = pos[2] / nz
        dx = tx - body_x
        dy = ty - body_y
        dz = tz - body_z
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_dx = dx
            nearest_dy = dy
            nearest_dz = dz

    obs_out[world_idx, idx] = nearest_dx
    obs_out[world_idx, idx + 1] = nearest_dy
    obs_out[world_idx, idx + 2] = nearest_dz


@wp.kernel
def compute_terrain_avoidance_reward_manta_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    avoidance_rewards_out: wp.array(dtype=wp.float32),
    nx: float,
    ny: float,
    nz: float,
    safe_radius: float,
    avoidance_weight: float,
    n_dynamic: int,
    n_solids: int,
):
    """Quadratic penalty based on the closest dynamic-solid to static-terrain distance."""
    world_idx = wp.tid()
    flow = flows[world_idx]

    min_dist = float(1.0e9)

    for dyn_idx in range(n_dynamic):
        dyn_pos = flow.solid_position[dyn_idx]
        dxn = dyn_pos[0] / nx
        dyn_y = dyn_pos[1] / ny
        dyn_z = dyn_pos[2] / nz

        for static_idx in range(n_dynamic, n_solids):
            static_pos = flow.solid_position[static_idx]
            sx = static_pos[0] / nx
            sy = static_pos[1] / ny
            sz = static_pos[2] / nz

            dx = dxn - sx
            dy = dyn_y - sy
            dz = dyn_z - sz
            dist = wp.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < min_dist:
                min_dist = dist

    penalty = float(0.0)
    if min_dist < safe_radius:
        x = 1.0 - min_dist / safe_radius
        penalty = avoidance_weight * x * x

    avoidance_rewards_out[world_idx] = -penalty


@wp.kernel
def check_terrain_collision_manta_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    terminated_out: wp.array(dtype=wp.int32),
    terrain_collision_out: wp.array(dtype=wp.int32),
    nx: float,
    ny: float,
    nz: float,
    collision_radius: float,
    n_dynamic: int,
    n_solids: int,
):
    """Terminate when any dynamic manta body part is too close to any static terrain body."""
    world_idx = wp.tid()
    flow = flows[world_idx]
    terrain_collision_out[world_idx] = 0

    for dyn_idx in range(n_dynamic):
        dyn_pos = flow.solid_position[dyn_idx]
        dxn = dyn_pos[0] / nx
        dyn_y = dyn_pos[1] / ny
        dyn_z = dyn_pos[2] / nz

        for static_idx in range(n_dynamic, n_solids):
            static_pos = flow.solid_position[static_idx]
            sx = static_pos[0] / nx
            sy = static_pos[1] / ny
            sz = static_pos[2] / nz

            dx = dxn - sx
            dy = dyn_y - sy
            dz = dyn_z - sz
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq < collision_radius * collision_radius:
                terrain_collision_out[world_idx] = 1
                terminated_out[world_idx] = 1
                return


@wp.kernel
def add_terrain_rewards_manta_kernel(
    rewards: wp.array(dtype=wp.float32),
    terrain_rewards: wp.array(dtype=wp.float32),
):
    world_idx = wp.tid()
    rewards[world_idx] = rewards[world_idx] + terrain_rewards[world_idx]


_TERRAIN_XMLS = {
    'single_block': 'manta_obstacle_3d.xml',
    'gate': 'manta_terrain_gate_3d.xml',
    'slalom': 'manta_terrain_slalom_3d.xml',
    'figure8': 'manta_terrain_figure8_3d.xml',
}


class Manta3DTerrainLBMEnv(Manta3DLBMEnv):
    """Manta goal-reaching environment augmented with static terrain geometry."""

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        root_link: str = 'body',
        root_position: Optional[Tuple[float, float, float]] = None,
        nx: int = 300,
        ny: int = 300,
        nz: int = 100,
        lbm_scale: float = 1.0,
        nworld: int = 1,
        max_episode_steps: int = 2000,
        per_frame_steps: int = 10,
        fluid_density: float = 1000.0,
        device: Optional[str] = None,
        goal_threshold: float = 0.08,
        single_goal_mode: bool = True,
        goal_position: Optional[List[float]] = None,
        reward_w_dist: float = 100.0,
        reward_w_roll: float = 0.5,
        reward_w_heading: float = 0.2,
        reward_w_forward: float = 0.1,
        obstacle_avoidance_weight: float = 1.0,
        obstacle_safe_radius: float = 0.10,
        obstacle_collision_radius: float = 0.07,
        terrain_variant: str = 'single_block',
    ):
        if mjcf_path is None:
            xml_name = _TERRAIN_XMLS.get(terrain_variant)
            if xml_name is None:
                valid = ', '.join(sorted(_TERRAIN_XMLS))
                raise ValueError(f"Unknown terrain_variant '{terrain_variant}'. Choose from: {valid}")
            mjcf_path = os.path.join(os.path.dirname(__file__), xml_name)

        if root_position is None:
            root_position = (nx / 2, ny * 0.25, nz / 2)

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
            reward_w_dist=reward_w_dist,
            reward_w_roll=reward_w_roll,
            reward_w_heading=reward_w_heading,
            reward_w_forward=reward_w_forward,
        )

        if self.n_static < 1:
            raise ValueError(
                f"Terrain environment requires at least one static solid, got n_static={self.n_static}."
            )

        self.terrain_variant = terrain_variant
        self.obstacle_avoidance_weight = float(obstacle_avoidance_weight)
        self.obstacle_safe_radius = float(obstacle_safe_radius)
        self.obstacle_collision_radius = float(obstacle_collision_radius)
        self.terrain_body_names = [cfg['link_name'] for cfg in self.static_link_config]
        self._terrain_positions_normalized = np.array([
            [
                cfg['lbm_position'][0] / self.nx,
                cfg['lbm_position'][1] / self.ny,
                cfg['lbm_position'][2] / self.nz,
            ]
            for cfg in self.static_link_config
        ], dtype=np.float32)

        self.obs_dim = 25 + 3 * self.n_joints + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.nworld, self.obs_dim),
            dtype=np.float32,
        )
        self._obs_buffer = wp.zeros((nworld, self.obs_dim), dtype=wp.float32, device=self.device)
        self._terrain_rewards_buffer = wp.zeros(nworld, dtype=wp.float32, device=self.device)
        self._terrain_collision_buffer = wp.zeros(nworld, dtype=wp.int32, device=self.device)

        print(f"  Terrain variant: {self.terrain_variant}")
        print(f"  Terrain bodies: {self.terrain_body_names}")
        print(f"  Terrain safe radius: {self.obstacle_safe_radius}")
        print(f"  Terrain collision radius: {self.obstacle_collision_radius}")

    def _create_observation_space(self) -> spaces.Space:
        n_joints = self.mj_model.njnt - 1
        obs_dim = 25 + 3 * n_joints + 3
        return spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.nworld, obs_dim),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        wp.launch(
            compute_manta_terrain_obs_kernel,
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
                self.n_dynamic,
                self.solid_num,
            ],
            device=self.device,
        )
        return self._obs_buffer.numpy().copy()

    def _compute_reward(self, instability_mask=None) -> np.ndarray:
        reward = super()._compute_reward(instability_mask)

        wp.launch(
            compute_terrain_avoidance_reward_manta_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._terrain_rewards_buffer,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.obstacle_safe_radius,
                self.obstacle_avoidance_weight,
                self.n_dynamic,
                self.solid_num,
            ],
            device=self.device,
        )

        wp.copy(self._rewards_buffer, wp.array(reward, dtype=wp.float32, device=self.device))
        wp.launch(
            add_terrain_rewards_manta_kernel,
            dim=self.nworld,
            inputs=[self._rewards_buffer, self._terrain_rewards_buffer],
            device=self.device,
        )
        return self._rewards_buffer.numpy()

    def _is_terminated(self, instability_mask=None) -> np.ndarray:
        self._terrain_collision_buffer.zero_()
        super()._is_terminated(instability_mask)

        wp.launch(
            check_terrain_collision_manta_kernel,
            dim=self.nworld,
            inputs=[
                self.lbm_solver.flows_wp,
                self._terminated_buffer,
                self._terrain_collision_buffer,
                float(self.nx),
                float(self.ny),
                float(self.nz),
                self.obstacle_collision_radius,
                self.n_dynamic,
                self.solid_num,
            ],
            device=self.device,
        )
        return self._terminated_buffer.numpy().astype(bool)

    def step(self, action: np.ndarray):
        observation, reward, done, info = super().step(action)

        terrain_collision = self._terrain_collision_buffer.numpy().astype(bool).copy()
        term_reason = info.get('term_reason', ['running'] * self.nworld)
        if isinstance(term_reason, str):
            term_reason = [term_reason] * self.nworld

        patched_reasons = []
        for w in range(self.nworld):
            raw = term_reason[w]
            parts = [] if raw in (None, '', 'running') else [p for p in str(raw).split('|') if p and p != 'running']
            if terrain_collision[w] and 'terrain_collision' not in parts:
                parts.append('terrain_collision')
            patched_reasons.append('|'.join(parts) if parts else 'running')

        info['term_reason'] = patched_reasons
        info['terrain_collision'] = terrain_collision
        info['terrain_pos_normalized'] = self._terrain_positions_normalized.copy()
        info['terrain_body_names'] = list(self.terrain_body_names)
        if len(self._terrain_positions_normalized) == 1:
            info['obstacle_pos_normalized'] = self._terrain_positions_normalized[0].copy()

        return observation, reward, done, info


class Manta3DObstacleLBMEnv(Manta3DTerrainLBMEnv):
    """Compatibility wrapper for the original single-block manta obstacle scene."""

    def __init__(self, *args, terrain_variant: str = 'single_block', **kwargs):
        super().__init__(*args, terrain_variant=terrain_variant, **kwargs)
