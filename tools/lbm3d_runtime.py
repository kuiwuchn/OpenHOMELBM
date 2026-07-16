"""Lightweight runtime helpers for 3D LBM demos.

This module only provides:
- Eel and Kármán LBM environment construction
- MuJoCo/LBM visualization helpers
"""


from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, List

import cv2
import mujoco
import numpy as np
import warp as wp

from envs.lbm3d.lbm_core_3d import (
    HomeFlow3D,
    cx_d3q27,
    cy_d3q27,
    cz_d3q27,
    w_d3q27,
)
from envs.lbm3d.karman import Karman3DEnv


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@wp.kernel
def set_uniform_flow_3d_kernel(flows: wp.array(dtype=HomeFlow3D), ux: float, uy: float, uz: float):
    world_idx, x, y, z = wp.tid()
    flow = flows[world_idx]
    rho = 1.0
    pop = wp.types.vector(length=27, dtype=wp.float32)
    u_sqr = ux * ux + uy * uy + uz * uz
    for i in range(27):
        cu = cx_d3q27[i] * ux + cy_d3q27[i] * uy + cz_d3q27[i] * uz
        pop[i] = w_d3q27[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sqr)

    inv_rho = 1.0 / rho
    pixx = pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]
    pixy = (pop[7] + pop[8] + pop[19] + pop[20] + pop[21] + pop[22]) - (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26])
    pixz = (pop[9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26])
    piyy = pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]
    piyz = (pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24])
    pizz = pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26]
    cs2_local = pixx
    pixx = pixx * inv_rho - cs2_local
    pixy = pixy * inv_rho
    pixz = pixz * inv_rho
    piyy = piyy * inv_rho - cs2_local
    piyz = piyz * inv_rho
    pizz = pizz * inv_rho - cs2_local

    flow.rho[x, y, z] = rho
    flow.rho_post[x, y, z] = rho
    flow.u[x, y, z] = wp.vec3(ux, uy, uz)
    flow.u_post[x, y, z] = wp.vec3(ux, uy, uz)
    flow.Sxx[x, y, z] = pixx
    flow.Sxx_post[x, y, z] = pixx
    flow.Syy[x, y, z] = piyy
    flow.Syy_post[x, y, z] = piyy
    flow.Szz[x, y, z] = pizz
    flow.Szz_post[x, y, z] = pizz
    flow.Sxy[x, y, z] = pixy
    flow.Sxy_post[x, y, z] = pixy
    flow.Sxz[x, y, z] = pixz
    flow.Sxz_post[x, y, z] = pixz
    flow.Syz[x, y, z] = piyz
    flow.Syz_post[x, y, z] = piyz
    flow.forcex[x, y, z] = 0.0
    flow.forcey[x, y, z] = 0.0
    flow.forcez[x, y, z] = 0.0


@wp.kernel
def set_local_force_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    cx: float,
    cy: float,
    cz: float,
    rx: float,
    ry: float,
    rz: float,
    fx: float,
    fy: float,
    fz: float,
):
    world_idx, x, y, z = wp.tid()
    flow = flows[world_idx]
    dx = (float(x) - cx) / wp.max(rx, 1.0e-6)
    dy = (float(y) - cy) / wp.max(ry, 1.0e-6)
    dz = (float(z) - cz) / wp.max(rz, 1.0e-6)
    r2 = dx * dx + dy * dy + dz * dz
    weight = wp.max(0.0, 1.0 - r2)
    flow.forcex[x, y, z] = fx * weight
    flow.forcey[x, y, z] = fy * weight
    flow.forcez[x, y, z] = fz * weight


@wp.kernel
def set_boundary_velocity_3d_kernel(flows: wp.array(dtype=HomeFlow3D), boundary_idx: int, ux: float, uy: float, uz: float):
    world_idx = wp.tid()
    flows[world_idx].bc_value[boundary_idx] = wp.vec3(ux, uy, uz)


BOUNDARY_NAME_TO_INDEX_3D = {
    "left": 0,
    "right": 1,
    "top": 2,
    "bottom": 3,
    "front": 4,
    "back": 5,
}


def _runtime_signal(cfg: Dict[str, Any], step: int, default_period: float) -> float:
    active_steps = cfg.get("active_steps")
    if active_steps is not None and step > int(active_steps):
        return 0.0
    period_steps = max(1.0, float(cfg.get("period_steps", default_period)))
    phase = float(cfg.get("phase", 0.0))
    return float(np.sin(2.0 * np.pi * float(step) / period_steps + phase))


def apply_lbm_flow_config_3d(env, flow_cfg: Dict[str, Any]) -> None:
    if not flow_cfg:
        return
    for flow in env.lbm_solver.flows:
        if "viscosity" in flow_cfg:
            flow.vis_shear = float(flow_cfg["viscosity"])
        if "bc_type" in flow_cfg:
            bc_type = [int(v) for v in flow_cfg["bc_type"]]
            if len(bc_type) != 6:
                raise ValueError("3D lbm.flow.bc_type must contain 6 values: left, right, top, bottom, front, back")
            flow.bc_type = wp.types.vector(length=6, dtype=wp.int32)(*bc_type)
        if "bc_value" in flow_cfg:
            values = flow_cfg["bc_value"]
            if len(values) != 6:
                raise ValueError("3D lbm.flow.bc_value must contain 6 vectors")
            flow.bc_value = wp.array(tuple(wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in values), dtype=wp.vec3, device=env.device)
    env.lbm_solver.flows_wp = wp.array(env.lbm_solver.flows, dtype=HomeFlow3D, device=env.device)
    if "initial_velocity" in flow_cfg:
        ux, uy, uz = flow_cfg["initial_velocity"]
        wp.launch(
            set_uniform_flow_3d_kernel,
            dim=(env.nworld, env.nx, env.ny, env.nz),
            inputs=[env.lbm_solver.flows_wp, float(ux), float(uy), float(uz)],
            device=env.device,
        )
        wp.synchronize()


def apply_lbm_runtime_flow_config_3d(env, flow_cfg: Dict[str, Any], step: int) -> None:
    if not flow_cfg:
        return
    perturb = flow_cfg.get("inlet_perturbation") or flow_cfg.get("boundary_perturbation")
    if perturb:
        boundary = perturb.get("boundary", "left")
        boundary_idx = BOUNDARY_NAME_TO_INDEX_3D.get(str(boundary).lower(), int(boundary) if isinstance(boundary, int) else 0)
        base = perturb.get("base", flow_cfg.get("bc_value", [[0.0, 0.0, 0.0]] * 6)[boundary_idx])
        amp = perturb.get("amplitude", [0.0, 0.0, 0.0])
        signal = _runtime_signal(perturb, step, 900.0)
        wp.launch(
            set_boundary_velocity_3d_kernel,
            dim=env.nworld,
            inputs=[
                env.lbm_solver.flows_wp,
                int(boundary_idx),
                float(base[0]) + float(amp[0]) * signal,
                float(base[1]) + float(amp[1]) * signal,
                float(base[2]) + float(amp[2]) * signal,
            ],
            device=env.device,
        )
    wake = flow_cfg.get("wake_perturbation")
    if wake and bool(wake.get("enabled", True)):
        center = wake.get("center", [0.5 * env.nx, 0.5 * env.ny, 0.5 * env.nz])
        radius = wake.get("radius", [20.0, 20.0, 20.0])
        force = wake.get("force", [0.0, 0.0, 0.0])
        signal = _runtime_signal(wake, step, 240.0)
        wp.launch(
            set_local_force_3d_kernel,
            dim=(env.nworld, env.nx, env.ny, env.nz),
            inputs=[
                env.lbm_solver.flows_wp,
                float(center[0]),
                float(center[1]),
                float(center[2]),
                float(radius[0]),
                float(radius[1]),
                float(radius[2]),
                float(force[0]) * signal,
                float(force[1]) * signal,
                float(force[2]) * signal,
            ],
            device=env.device,
        )


class RuntimeVecEnvWrapper:
    """Minimal vector-env adapter used by realtime demos and wave tests."""


    def __init__(self, env):
        self._env = env
        self._num_envs = env.nworld

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        return self._env.reset()

    def _split_infos(self, infos):
        if isinstance(infos, list):
            return infos
        if not isinstance(infos, dict):
            return [{} for _ in range(self._num_envs)]
        result = [{} for _ in range(self._num_envs)]
        for key, value in infos.items():
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == self._num_envs:
                for i in range(self._num_envs):
                    result[i][key] = value[i]
            else:
                for i in range(self._num_envs):
                    result[i][key] = value
        return result

    def step(self, actions):
        obs, rewards, dones, infos = self._env.step(actions)
        return obs, list(rewards), dones, self._split_infos(infos)

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()


def make_multitask_env(config, nworld=1):
    env_type = getattr(config, "env_type", "eel_multitask")

    nx = getattr(config, "lbm_nx", 200)
    ny = getattr(config, "lbm_ny", 200)
    nz = getattr(config, "lbm_nz", 80)
    lbm_scale = getattr(config, "lbm_scale", 1.0)
    fluid_density = getattr(config, "fluid_density", 1000.0)
    per_frame_steps = getattr(config, "per_frame_steps", 10)
    task_switch_interval = getattr(config, "task_switch_interval", 0)
    k_harmonics = getattr(config, "k_harmonics", 2)
    b_bar = getattr(config, "b_bar", 1.0)
    use_reduced_order = getattr(config, "use_reduced_order", True)
    control_mode = getattr(config, "control_mode", "direct")
    mjcf_path = getattr(config, "mjcf_path", None)
    root_link = getattr(config, "root_link", None)
    root_position = getattr(config, "root_position", None)
    link_config = getattr(config, "link_config", None)
    flow_config = getattr(config, "flow_config", None)




    if env_type == "karman3d":
        env_class = Karman3DEnv
    elif env_type == "eel_multitask":
        from envs.lbm3d.eel.eel_multitask_env_3d import EelMultiTaskEnv
        env_class = EelMultiTaskEnv
    else:
        raise ValueError(
            f"Unsupported 3D environment type {env_type!r}; expected "
            "'eel_multitask' or a generic/Karman environment"
        )

    if env_type == "karman3d":
        env_kwargs = {
            "mjcf_path": str(mjcf_path),
            "link_config": link_config,
            "root_link": root_link,
            "root_position": tuple(root_position) if root_position is not None else None,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "lbm_scale": lbm_scale,
            "nworld": nworld,
            "max_episode_steps": config.time_limit,
            "per_frame_steps": per_frame_steps,
            "fluid_density": fluid_density,
            "flow_config": flow_config,
        }
    else:
        env_kwargs = {
            "nworld": nworld,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "lbm_scale": lbm_scale,
            "fluid_density": fluid_density,
            "max_episode_steps": config.time_limit,
            "per_frame_steps": per_frame_steps,
            "task_switch_interval": task_switch_interval,
            "k_harmonics": k_harmonics,
            "b_bar": b_bar,
            "use_reduced_order": use_reduced_order,
            "control_mode": control_mode,
        }
        if mjcf_path is not None:
            env_kwargs["mjcf_path"] = str(mjcf_path)
        if root_link is not None:
            env_kwargs["root_link"] = root_link
        if root_position is not None:
            env_kwargs["root_position"] = tuple(root_position)

    env_kwargs = {key: value for key, value in env_kwargs.items() if value is not None}
    env = env_class(**env_kwargs)


    return RuntimeVecEnvWrapper(env)


class MuJoCoRenderer:
    def __init__(self, mj_model, width=640, height=480,
                 camera_distance=1.5, camera_azimuth=45, camera_elevation=-35,
                 camera_lookat=None, show_position=True, show_fluid_force=False):
        self.model = mj_model
        self.width = width
        self.height = height
        self.show_position = show_position
        self.show_fluid_force = show_fluid_force
        self.fluid_forces = None
        self.fluid_torques = None
        self.renderer = mujoco.Renderer(mj_model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = camera_distance
        self.camera.azimuth = camera_azimuth
        self.camera.elevation = camera_elevation
        self.camera.lookat[:] = camera_lookat if camera_lookat is not None else [0, 0, 0]
        self.scene_option = mujoco.MjvOption()
        self.scene_option.frame = mujoco.mjtFrame.mjFRAME_NONE

    def set_fluid_forces(self, forces, torques=None):
        self.fluid_forces = forces
        self.fluid_torques = torques

    def render(self, mj_data):
        self.renderer.update_scene(mj_data, self.camera, self.scene_option)
        frame = self.renderer.render()
        return self._add_overlays(frame, mj_data)

    def _add_overlays(self, frame, mj_data):
        frame = frame.copy()
        if self.show_position and self.model.nbody > 1:
            pos = mj_data.xpos[1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            lines = [
                ("Position:", (200, 200, 200)),
                (f"  X: {pos[0]:+.3f}", (230, 100, 100)),
                (f"  Y: {pos[1]:+.3f}", (100, 230, 100)),
                (f"  Z: {pos[2]:+.3f}", (100, 100, 230)),
            ]
            box_width = 130
            box_height = len(lines) * 20 + 10
            box_x = self.width - box_width - 10
            box_y = 10
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
            for i, (text, color) in enumerate(lines):
                cv2.putText(frame, text, (box_x + 8, box_y + 22 + i * 20), font, 0.55, color, 1, cv2.LINE_AA)
        return frame

    def close(self):
        self.renderer.close()


def get_raw_frame_3d(env, world_idx=0, render_type="velocity", view_mode="topdown"):
    from envs.lbm3d.lbm_func_3d import (
        get_u_projection_topdown_3d,
        get_u_projection_max_topdown_3d,
        get_u_projection_side_3d,
        get_u_projection_front_3d,
        get_vorticity_projection_topdown_3d,
        get_vorticity_projection_side_3d,
        get_vorticity_projection_front_3d,
    )

    base_env = env._env
    solver = base_env.lbm_solver
    flow = solver.flows[world_idx]
    nx, ny, nz = flow.nx, flow.ny, flow.nz

    if view_mode == "topdown":
        kernel = get_vorticity_projection_topdown_3d if render_type == "vorticity" else get_u_projection_topdown_3d
        wp.launch(kernel, dim=(nx, ny), inputs=[flow], device=solver.device)
        wp.synchronize()
        return np.flipud(flow.u_img_xy.numpy().T)
    if view_mode == "max_topdown":
        kernel = get_vorticity_projection_topdown_3d if render_type == "vorticity" else get_u_projection_max_topdown_3d
        wp.launch(kernel, dim=(nx, ny), inputs=[flow], device=solver.device)
        wp.synchronize()
        return np.flipud(flow.u_img_xy.numpy().T)
    if view_mode == "side":
        kernel = get_vorticity_projection_side_3d if render_type == "vorticity" else get_u_projection_side_3d
        wp.launch(kernel, dim=(ny, nz), inputs=[flow], device=solver.device)
        wp.synchronize()
        return np.flipud(flow.u_img_xz.numpy().T)
    if view_mode == "front":
        kernel = get_vorticity_projection_front_3d if render_type == "vorticity" else get_u_projection_front_3d
        wp.launch(kernel, dim=(nx, nz), inputs=[flow], device=solver.device)
        wp.synchronize()
        return np.flipud(flow.u_img_xz_front.numpy().T)
    raise ValueError(f"Unknown view_mode: {view_mode}")


def get_mujoco_frame(env, mujoco_renderer, world_idx=0, with_fluid_force=False):
    base_env = env._env
    qpos = base_env.mjw_data.qpos.numpy()[world_idx]
    qvel = base_env.mjw_data.qvel.numpy()[world_idx]
    base_env.mj_data.qpos[:] = qpos
    base_env.mj_data.qvel[:] = qvel
    mujoco.mj_forward(base_env.mj_model, base_env.mj_data)
    if with_fluid_force:
        try:
            forces, torques = base_env.lbm_solver.get_forces_and_torques(world_idx)
            mujoco_renderer.set_fluid_forces(forces, torques)
        except Exception:
            mujoco_renderer.set_fluid_forces(None, None)
    return mujoco_renderer.render(base_env.mj_data)


def process_raw_to_frame(raw_img, vmax, render_type="velocity"):
    import matplotlib.pyplot as plt

    if render_type == "vorticity":
        solid_mask = raw_img >= 999.0
        fluid_vals = raw_img.copy()
        fluid_vals[solid_mask] = 0.0
        img_normalized = np.clip((fluid_vals / vmax + 1) / 2, 0, 1)
        cmap = plt.get_cmap("RdBu_r")
        img_rgb = (cmap(img_normalized)[:, :, :3] * 255).astype(np.uint8)
        img_rgb[solid_mask] = 200
    else:
        img_normalized = np.clip(raw_img / vmax, 0, 1)
        cmap = plt.get_cmap("magma")
        img_rgb = (cmap(img_normalized)[:, :, :3] * 255).astype(np.uint8)
    return img_rgb


def save_video(frames, output_path, fps=30):
    if len(frames) == 0:
        print("No frames to save!")
        return
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Video saved: {output_path}")


def combine_frames_left_right(mj_frames, lbm_frames, left_ratio=0.6, separator_width=2):
    combined: List[np.ndarray] = []
    for mj, lbm in zip(mj_frames, lbm_frames):
        h_mj, w_mj = mj.shape[:2]
        out_h = max(h_mj, lbm.shape[0])
        left_w = int(out_h * (w_mj / h_mj))
        right_w = int(left_w * (1.0 - left_ratio) / left_ratio)
        mj_resized = cv2.resize(mj, (left_w, out_h), interpolation=cv2.INTER_LINEAR)
        lbm_resized = cv2.resize(lbm, (right_w, out_h), interpolation=cv2.INTER_LINEAR)
        sep = np.zeros((out_h, separator_width, 3), dtype=np.uint8)
        combined.append(np.hstack([mj_resized, sep, lbm_resized]))
    return combined
