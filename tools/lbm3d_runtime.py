"""Lightweight runtime helpers for 3D LBM demos.

This module only provides:
- optional named config merging for legacy tools
- multi-task LBM environment construction
- MuJoCo/LBM visualization helpers
"""


from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import cv2
import mujoco
import numpy as np
import warp as wp
from ruamel.yaml import YAML

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _recursive_update(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _recursive_update(base[key], value)
        else:
            base[key] = value


def load_named_config(config_names, overrides=None, config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "configs.yaml"
    config_path = pathlib.Path(config_path)
    yaml = YAML(typ="safe", pure=True)
    configs = yaml.load(config_path.read_text(encoding="utf-8"))

    merged: Dict[str, Any] = {}
    for name in config_names:
        if name not in configs:
            raise KeyError(f"Config section '{name}' not found in {config_path}")
        _recursive_update(merged, configs[name])
    if overrides:
        _recursive_update(merged, overrides)
    return SimpleNamespace(**merged)


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
    env_type = getattr(config, "env_type", "manta_multitask")

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


    if env_type == "clownfish_multitask":

        from envs.lbm3d.clownfish.clownfish_multitask_env_3d import ClownfishMultiTaskEnv
        env_class = ClownfishMultiTaskEnv
    elif env_type == "tuna_multitask":
        from envs.lbm3d.tuna.tuna_multitask_env_3d import TunaMultiTaskEnv
        env_class = TunaMultiTaskEnv
    elif env_type == "eel_multitask":
        from envs.lbm3d.eel.eel_multitask_env_3d import EelMultiTaskEnv
        env_class = EelMultiTaskEnv
    elif env_type == "turtle_multitask":
        from envs.lbm3d.turtle.turtle_multitask_env_3d import TurtleMultiTaskEnv
        env_class = TurtleMultiTaskEnv
    else:
        from envs.lbm3d.manta.manta_multitask_env_3d import MantaMultiTaskEnv
        env_class = MantaMultiTaskEnv

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
