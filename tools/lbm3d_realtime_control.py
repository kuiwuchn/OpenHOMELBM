"""
3D realtime LBM + MuJoCo control demo.

Model selection, LBM parameters, keyboard bindings, task mapping, and preset
actions are loaded from a JSON config under `configs/realtime_3d`.

Examples:
    python tools/lbm3d_realtime_control.py --config configs/realtime_3d/eel3d.json --with-lbm
    python tools/lbm3d_realtime_control.py --animal eel --preset forward --export-lbm outputs/eel_lbm.mp4

Controls:
    W        forward
    A        turn_l
    D        turn_r
    F        fast
    S        freeze/glide/idle fallback
    Z        ascend
    C        descend
    Space    pause/resume
    R        reset
    Q/Esc    quit
"""

import argparse
import json
import pathlib
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

import cv2
import numpy as np


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from lbm3d_runtime import (
    MuJoCoRenderer,
    get_mujoco_frame,
    get_raw_frame_3d,
    make_multitask_env,
    process_raw_to_frame,
    save_video,
)


DEFAULT_TASK_BY_PRESET = {

    "forward": "forward",
    "fast": "forward",
    "tail_only": "forward",
    "cold_start": "forward",
    "reverse": "forward",
    "head_tail_swing": "forward",
    "glide": "forward",
    "freeze": "forward",
    "turn_l": "turn_left",
    "turn_r": "turn_right",
    "ascend": "ascend",
    "descend": "descend",
}


def resolve_config_path(config_arg: Optional[str], animal: str) -> pathlib.Path:
    if config_arg:
        path = pathlib.Path(config_arg)
    else:
        path = PROJECT_ROOT / "configs" / "realtime_3d" / f"{animal}3d.json"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json_config(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"3D realtime JSON config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_preset_section(config_data: Dict[str, Any]) -> tuple[Dict[str, dict], list[str]]:
    preset_section = config_data.get("presets", {})
    if "actions" in preset_section:
        presets = preset_section.get("actions", {})
        action_keys = preset_section.get("action_keys", [])
    else:
        presets = preset_section
        action_keys = config_data.get("action_keys", [])
    if not presets:
        raise ValueError("JSON config must define presets.actions")
    if not action_keys:
        first = next(iter(presets.values()))
        if isinstance(first, dict):
            action_keys = list(first.keys())
        else:
            raise ValueError("JSON config must define presets.action_keys when preset values are arrays")
    return presets, list(action_keys)


def preset_to_action(params: Any, action_keys: list[str]) -> np.ndarray:
    if isinstance(params, dict):
        if "values" in params:
            return np.asarray(params["values"], dtype=np.float32)
        missing = [key for key in action_keys if key not in params]
        if missing:
            raise ValueError(f"Preset is missing action keys: {missing}")
        return np.asarray([params[key] for key in action_keys], dtype=np.float32)
    return np.asarray(params, dtype=np.float32)


def build_runtime_config(config_data: Dict[str, Any], overrides: Dict[str, Any]) -> SimpleNamespace:
    model = dict(config_data.get("model", {}))
    lbm = dict(config_data.get("lbm", {}))
    control = dict(config_data.get("control", {}))

    config = {
        "env_type": model.get("env_type", config_data.get("env_type", "eel_multitask")),
        "time_limit": int(model.get("time_limit", config_data.get("time_limit", 2000))),
        "lbm_nx": int(lbm.get("nx", lbm.get("lbm_nx", 150))),
        "lbm_ny": int(lbm.get("ny", lbm.get("lbm_ny", 250))),
        "lbm_nz": int(lbm.get("nz", lbm.get("lbm_nz", 60))),
        "lbm_scale": float(lbm.get("lbm_scale", 0.5)),
        "fluid_density": float(lbm.get("fluid_density", 1000.0)),
        "per_frame_steps": int(lbm.get("per_frame_steps", 10)),
        "task_switch_interval": int(control.get("task_switch_interval", 0)),
        "control_mode": str(control.get("control_mode", model.get("control_mode", "wave"))),
        "k_harmonics": int(control.get("k_harmonics", 2)),
        "b_bar": float(control.get("b_bar", 1.0)),
        "use_reduced_order": bool(control.get("use_reduced_order", True)),
    }

    for key in ("mjcf_path", "root_link", "root_position", "link_config"):
        if key in model:
            config[key] = model[key]
    if "flow" in lbm:
        config["flow_config"] = lbm["flow"]

    if "mjcf_path" in config:
        mjcf_path = pathlib.Path(config["mjcf_path"])
        if not mjcf_path.is_absolute():
            config["mjcf_path"] = str(PROJECT_ROOT / mjcf_path)

    config.update(overrides)
    return SimpleNamespace(**config)


def make_env(config_data: Dict[str, Any], nworld: int, overrides: Dict[str, Any]):
    return make_multitask_env(build_runtime_config(config_data, overrides), nworld=nworld)


def choose_idle_preset(presets: Dict[str, dict]) -> str:

    for name in ("freeze", "glide", "idle", "forward"):
        if name in presets:
            return name
    return next(iter(presets))


def build_keymap(presets: Dict[str, dict], controls: Optional[Dict[str, str]] = None) -> Dict[int, str]:
    if controls:
        mapping: Dict[int, str] = {}
        for key_name, preset_name in controls.items():
            if len(key_name) != 1 or preset_name not in presets:
                continue
            mapping[ord(key_name.lower())] = preset_name
            mapping[ord(key_name.upper())] = preset_name
        return mapping

    idle = choose_idle_preset(presets)
    mapping = {
        ord("w"): "forward",
        ord("W"): "forward",
        ord("a"): "turn_l",
        ord("A"): "turn_l",
        ord("d"): "turn_r",
        ord("D"): "turn_r",
        ord("f"): "fast",
        ord("F"): "fast",
        ord("s"): idle,
        ord("S"): idle,
        ord("z"): "ascend",
        ord("Z"): "ascend",
        ord("c"): "descend",
        ord("C"): "descend",
    }
    return {k: v for k, v in mapping.items() if v in presets}



def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == height:
        return frame
    width = max(1, int(round(w * height / h)))
    interp = cv2.INTER_AREA if h > height else cv2.INTER_CUBIC
    return cv2.resize(frame, (width, height), interpolation=interp)


def combine_frames(left: np.ndarray, right: np.ndarray, output_height: int) -> np.ndarray:
    left = resize_to_height(left, output_height)
    right = resize_to_height(right, output_height)
    sep = np.full((output_height, 4, 3), 32, dtype=np.uint8)
    return np.concatenate([left, sep, right], axis=1)


def compute_lbm_vmax(raw: np.ndarray, render_type: str, previous: Optional[float]) -> float:
    if render_type == "vorticity":
        mask = raw < 999.0
        current = float(np.max(np.abs(raw[mask]))) * 0.2 + 1e-8 if np.any(mask) else 1.0
    else:
        current = float(np.max(raw)) * 0.6 + 1e-8
    if previous is None:
        return max(current, 1e-6)
    return max(previous * 0.96, current, 1e-6)


def draw_overlay(
    frame: np.ndarray,
    animal: str,
    mode: str,
    task: str,
    step: int,
    action: np.ndarray,
    reward: float,
    fps: float,
    paused: bool,
) -> np.ndarray:
    out = frame.copy()
    action_str = ", ".join(f"{v:+.2f}" for v in action.flatten())
    if len(action_str) > 72:
        action_str = action_str[:69] + "..."
    lines = [
        f"animal: {animal}  mode: {mode}  task: {task} {'[PAUSED]' if paused else ''}",
        f"step: {step}  reward: {reward:+.4f}  fps: {fps:.1f}",
        f"action: [{action_str}]",
        # "W forward | A left | D right | F fast | S idle | Z ascend | C descend",
        # "Space pause | R reset | Q/Esc quit",
    ]
    x, y0 = 12, 24
    for i, text in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 1, cv2.LINE_AA)
    return out


def set_task_if_supported(env, task_name: str) -> None:
    base_env = env._env
    if not hasattr(base_env, "_task_ids") or not hasattr(base_env, "_update_task_ids_wp"):
        return
    task_names = getattr(base_env, "TASK_NAMES", None)
    if task_names is None:
        task_names = ["forward", "turn_left", "turn_right", "ascend", "descend"]
    if task_name not in task_names:
        return
    task_id = task_names.index(task_name)
    base_env._task_ids[:] = task_id
    base_env._update_task_ids_wp()






def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def add_solid_meshes_to_pyvista(plotter, pv, env, flow) -> bool:
    base_env = getattr(env, "_env", env)
    solver = getattr(base_env, "lbm_solver", None)
    meshes = getattr(solver, "meshes", None)
    if not meshes:
        return False

    solid_pos = flow.solid_position.numpy().astype(np.float32)
    solid_quat = flow.solid_quaternion.numpy().astype(np.float32)
    rendered = False

    for solid_id, mesh in enumerate(meshes):
        if mesh is None or solid_id >= solid_pos.shape[0]:
            continue
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            continue

        mapping = getattr(solver, "mujoco_mappings", {}).get(solid_id, {})
        scale = float(mapping.get("scale", getattr(base_env, "coordinate_scale", 1.0)))
        rot = quat_wxyz_to_matrix(solid_quat[solid_id])
        world_vertices = (vertices * scale) @ rot.T + solid_pos[solid_id]
        pv_faces = np.hstack((np.full((faces.shape[0], 1), 3, dtype=np.int64), faces)).ravel()
        poly = pv.PolyData(world_vertices, pv_faces)
        plotter.add_mesh(
            poly,
            color="#4a4a4a",
            opacity=1.0,
            lighting=True,
            smooth_shading=False,
            show_edges=True,
            edge_color="#2f2f2f",
            ambient=0.35,
            diffuse=0.8,
            specular=0.08,
        )
        rendered = True


    return rendered


def render_vorticity_volume_frame(env, frame_idx: int, args) -> np.ndarray:

    """Render a rotating 3D vorticity-volume view from the coupled LBM field."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    flow = env._env.lbm_solver.flows[0]
    stride = max(1, int(args.volume_stride))
    u = flow.u.numpy()[::stride, ::stride, ::stride]
    if u.ndim != 4 or u.shape[-1] != 3:
        raise RuntimeError(f"Unexpected flow.u numpy shape: {u.shape}")

    ux = u[..., 0]
    uy = u[..., 1]
    uz = u[..., 2]
    dux_dx, dux_dy, dux_dz = np.gradient(ux)
    duy_dx, duy_dy, duy_dz = np.gradient(uy)
    duz_dx, duz_dy, duz_dz = np.gradient(uz)
    del dux_dx, duy_dy, duz_dz
    vort_x = duz_dy - duy_dz
    vort_y = dux_dz - duz_dx
    vort_z = duy_dx - dux_dy

    vort_mag = np.sqrt(vort_x * vort_x + vort_y * vort_y + vort_z * vort_z)

    azim = float(args.orbit_azim_start) + frame_idx * float(args.orbit_azim_speed)
    elev = float(args.orbit_elev)
    color_axis = str(args.volume_color_axis).lower()
    if color_axis == "x":
        signed_vort = vort_x
    elif color_axis == "y":
        signed_vort = vort_y
    else:
        signed_vort = vort_z

    abs_signed = np.abs(signed_vort)
    finite = np.isfinite(abs_signed)
    if np.any(finite):
        vmax = float(np.percentile(abs_signed[finite], float(args.volume_vmax_percentile)))
        if vmax <= 1.0e-12:
            vmax = float(np.max(abs_signed[finite]))
    else:
        vmax = 1.0
    vmax = max(vmax, 1.0e-8)

    width = int(args.volume_width)
    height = int(args.output_height)
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    cmap = plt.get_cmap("RdBu_r")

    if args.volume_render_mode == "points":
        threshold = float(np.percentile(abs_signed[finite], float(args.volume_percentile))) if np.any(finite) else 0.0
        mask = finite & (abs_signed >= threshold) & (abs_signed > 1.0e-12)
        coords = np.argwhere(mask)
        color_values = signed_vort[mask]
        rank_values = abs_signed[mask]
        max_points = max(1, int(args.volume_max_points))
        if rank_values.size > max_points:
            idx = np.argpartition(rank_values, -max_points)[-max_points:]
            coords = coords[idx]
            color_values = color_values[idx]
        if color_values.size > 0:
            xs = coords[:, 0] * stride
            ys = coords[:, 1] * stride
            zs = coords[:, 2] * stride
            ax.scatter(
                xs,
                ys,
                zs,
                c=color_values,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                s=float(args.volume_point_size),
                alpha=0.75,
                linewidths=0,
            )
    else:
        import pyvista as pv

        field = np.nan_to_num(signed_vort, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        nx_s, ny_s, nz_s = field.shape
        grid = pv.ImageData(dimensions=(nx_s, ny_s, nz_s), spacing=(stride, stride, stride), origin=(0.0, 0.0, 0.0))
        grid.point_data["vorticity"] = field.ravel(order="F")

        plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        plotter.set_background("white")

        if args.volume_render_mode == "isosurface":
            if np.any(finite):
                max_level = float(np.percentile(abs_signed[finite], float(args.volume_iso_percentile)))
                if max_level <= 1.0e-12:
                    max_level = float(np.max(abs_signed[finite]))
                min_level = float(np.percentile(abs_signed[finite], float(args.volume_iso_min_percentile)))
            else:
                max_level = 0.0
                min_level = 0.0
            max_level = min(max(max_level, 1.0e-8), vmax)
            if min_level <= 1.0e-12 or min_level >= max_level:
                min_level = max_level * 0.45

            iso_levels = np.linspace(min_level, max_level, max(1, int(args.volume_iso_levels)), dtype=np.float32)
            max_alpha = float(args.volume_iso_alpha)
            for level_idx, level in enumerate(iso_levels):
                strength = float(level_idx + 1) / float(len(iso_levels))
                alpha = max(0.04, max_alpha * (0.30 + 0.70 * strength * strength))
                for iso_value in (-float(level), float(level)):
                    surface = grid.contour(isosurfaces=[iso_value], scalars="vorticity")
                    if surface.n_points == 0:
                        continue
                    smooth_iter = max(0, int(args.volume_iso_smooth_iter))
                    if smooth_iter > 0:
                        try:
                            surface = surface.smooth(n_iter=smooth_iter, relaxation_factor=0.08)
                        except Exception:
                            pass
                    plotter.add_mesh(
                        surface,
                        scalars="vorticity",
                        cmap="RdBu_r",
                        clim=(-max_level, max_level),
                        opacity=alpha,
                        show_scalar_bar=False,
                        lighting=True,
                        smooth_shading=True,
                        ambient=0.25,
                        diffuse=0.75,
                        specular=0.25,
                        specular_power=18.0,
                    )

        else:
            slice_axis = str(args.volume_slice_axis).lower()
            count = max(1, int(args.volume_slice_count))
            if slice_axis == "x":
                indices = np.linspace(0, nx_s - 1, min(count, nx_s), dtype=int)
                slices = [grid.slice(normal="x", origin=(idx * stride, 0.0, 0.0)) for idx in indices]
            elif slice_axis == "y":
                indices = np.linspace(0, ny_s - 1, min(count, ny_s), dtype=int)
                slices = [grid.slice(normal="y", origin=(0.0, idx * stride, 0.0)) for idx in indices]
            else:
                indices = np.linspace(0, nz_s - 1, min(count, nz_s), dtype=int)
                slices = [grid.slice(normal="z", origin=(0.0, 0.0, idx * stride)) for idx in indices]

            for slc in slices:
                if slc.n_points == 0:
                    continue
                plotter.add_mesh(
                    slc,
                    scalars="vorticity",
                    cmap="RdBu_r",
                    clim=(-vmax, vmax),
                    opacity=float(args.volume_slice_alpha),
                    show_scalar_bar=False,
                    lighting=False,
                    interpolate_before_map=True,
                )


        try:
            if not add_solid_meshes_to_pyvista(plotter, pv, env, flow):
                solid_pos = flow.solid_position.numpy().astype(np.float32)
                if solid_pos.size > 0:
                    plotter.add_points(pv.PolyData(solid_pos), color="#4a4a4a", point_size=6.0, render_points_as_spheres=False)

        except Exception:
            pass


        center = np.array([flow.nx * 0.5, flow.ny * 0.5, flow.nz * 0.5], dtype=np.float32)
        radius = float(max(flow.nx, flow.ny, flow.nz)) * 1.8
        az = np.deg2rad(azim)
        el = np.deg2rad(elev)
        camera_pos = center + radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], dtype=np.float32)
        plotter.camera_position = (tuple(camera_pos), tuple(center), (0.0, 0.0, 1.0))
        plotter.camera.zoom(1.15)
        frame = plotter.screenshot(return_img=True)
        plotter.close()
        return np.asarray(frame[:, :, :3], dtype=np.uint8)

    try:



        solid_pos = flow.solid_position.numpy()
        if solid_pos.size > 0:
            ax.plot(solid_pos[:, 0], solid_pos[:, 1], solid_pos[:, 2], color="#888888", linewidth=3.0, alpha=0.95)
            ax.scatter(solid_pos[:, 0], solid_pos[:, 1], solid_pos[:, 2], color="#777777", s=16, alpha=0.95)
    except Exception:
        pass

    ax.set_xlim(0, flow.nx)
    ax.set_ylim(0, flow.ny)
    ax.set_zlim(0, flow.nz)
    ax.set_box_aspect((flow.nx, flow.ny, flow.nz))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    frame = rgba[:, :, :3].copy()
    plt.close(fig)
    return frame


def export_lbm_video(env, args, presets: Dict[str, dict], action_keys: list[str], task_by_preset: Dict[str, str], mode: str) -> None:

    task = task_by_preset.get(mode, "forward")
    set_task_if_supported(env, task)
    action_target = preset_to_action(presets[mode], action_keys).reshape(1, -1).astype(np.float32)


    raw_frames = []
    action_frames = []
    reward_frames = []
    video_frames = []
    total_reward = 0.0
    render_every = max(1, int(args.export_render_every))
    orbit_volume = args.view_mode == "orbit"

    print(
        f"[export-lbm] animal={args.animal} preset={mode} task={task} "
        f"steps={args.export_steps} render_every={render_every} view={args.view_mode} output={args.export_lbm}",
        flush=True,
    )
    start = time.perf_counter()
    for step_idx in range(args.export_steps):
        ramp = min(1.0, (step_idx + 1) / max(1, args.warmup_steps))
        action = np.clip(action_target * ramp, -1.0, 1.0).astype(np.float32)
        _obs, rewards, _dones, _infos = env.step(action)
        reward = float(rewards[0])
        total_reward += reward

        if step_idx % render_every == 0:
            if orbit_volume:
                frame = render_vorticity_volume_frame(env, len(video_frames), args)
                if not args.export_no_overlay:
                    frame = draw_overlay(frame, args.animal, mode, task, step_idx + 1, action, reward, 0.0, False)
                video_frames.append(frame)
            else:
                raw = get_raw_frame_3d(env, world_idx=0, render_type=args.render_type, view_mode=args.view_mode)
                raw_frames.append(raw.copy())
                action_frames.append(action.copy())
                reward_frames.append(reward)

        if (step_idx + 1) == 1 or (step_idx + 1) % max(1, args.benchmark_progress_every) == 0:
            elapsed = time.perf_counter() - start
            fps = (step_idx + 1) / max(elapsed, 1.0e-9)
            print(f"[export-lbm] step {step_idx + 1}/{args.export_steps} sim_fps={fps:.2f}", flush=True)

    if orbit_volume:
        if not video_frames:
            raise RuntimeError("No LBM volume frames captured; check --export-steps and --export-render-every")
        save_video(video_frames, pathlib.Path(args.export_lbm), fps=args.record_fps)
    else:
        if not raw_frames:
            raise RuntimeError("No LBM frames captured; check --export-steps and --export-render-every")

        all_raw = np.stack(raw_frames)
        if args.render_type == "vorticity":
            mask = all_raw < 999.0
            vmax = float(np.max(np.abs(all_raw[mask]))) * 0.2 + 1.0e-8 if np.any(mask) else 1.0
        else:
            vmax = float(np.max(all_raw)) * 0.6 + 1.0e-8

        for i, raw in enumerate(raw_frames):
            frame = process_raw_to_frame(raw, vmax, args.render_type)
            frame = resize_to_height(frame, args.output_height)
            if not args.export_no_overlay:
                sim_step = i * render_every + 1
                frame = draw_overlay(
                    frame,
                    args.animal,
                    mode,
                    task,
                    sim_step,
                    action_frames[i],
                    reward_frames[i],
                    0.0,
                    False,
                )
            video_frames.append(frame)

        save_video(video_frames, pathlib.Path(args.export_lbm), fps=args.record_fps)

    elapsed = time.perf_counter() - start
    print(
        f"[export-lbm] saved {len(video_frames)} frames, elapsed={elapsed:.3f}s, "
        f"avg_sim_fps={args.export_steps / max(elapsed, 1.0e-9):.2f}, total_reward={total_reward:.4f}",
        flush=True,
    )


def main() -> None:

    parser = argparse.ArgumentParser(description="3D realtime LBM + MuJoCo OpenGL control demo")
    parser.add_argument("--config", type=str, default=None, help="JSON config path; default is configs/realtime_3d/<animal>3d.json")
    parser.add_argument("--animal", type=str, default="eel", help="Animal name used to locate the default JSON config")
    parser.add_argument("--preset", type=str, default=None, help="Initial preset; default uses JSON control.start_mode or freeze/glide/forward fallback")

    parser.add_argument("--with-lbm", action="store_true", help="Show LBM 3D projection next to MuJoCo render")
    parser.add_argument("--render-type", type=str, default="vorticity", choices=["velocity", "vorticity"])
    parser.add_argument("--view-mode", type=str, default="topdown", choices=["topdown", "max_topdown", "side", "front", "orbit"])

    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--per-frame-steps", type=int, default=None)
    parser.add_argument("--lbm-scale", type=float, default=None)
    parser.add_argument("--output-height", type=int, default=720)
    parser.add_argument("--mujoco-width", type=int, default=720)
    parser.add_argument("--mujoco-height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=1.9)
    parser.add_argument("--camera-azimuth", type=float, default=45.0)
    parser.add_argument("--camera-elevation", type=float, default=-45.0)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--transition-steps", type=int, default=None, help="Smooth blending steps when switching presets")

    parser.add_argument("--window-name", type=str, default="LBM3D Realtime Control")

    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--export-lbm", type=str, default=None, help="Export LBM-only rendering video to this mp4 path and exit")
    parser.add_argument("--export-steps", type=int, default=120, help="Simulation steps for --export-lbm")
    parser.add_argument("--export-render-every", type=int, default=1, help="Capture one LBM frame every N simulation steps")
    parser.add_argument("--export-no-overlay", action="store_true", help="Do not draw text overlay on exported LBM video")
    parser.add_argument("--volume-width", type=int, default=960, help="Output width for --view-mode orbit volume rendering")
    parser.add_argument("--volume-stride", type=int, default=4, help="Downsample stride for orbit volume rendering")
    parser.add_argument("--volume-render-mode", type=str, default="slices", choices=["slices", "isosurface", "points"], help="Orbit volume renderer: slice stack, signed isosurfaces, or thresholded point cloud")
    parser.add_argument("--volume-slice-axis", type=str, default="z", choices=["x", "y", "z"], help="Slice stacking axis for --volume-render-mode slices")
    parser.add_argument("--volume-slice-count", type=int, default=9, help="Number of slices for --volume-render-mode slices")
    parser.add_argument("--volume-contour-levels", type=int, default=64, help="Number of filled contour levels for slice rendering")
    parser.add_argument("--volume-slice-alpha", type=float, default=0.82, help="Alpha value for filled contour slices")
    parser.add_argument("--volume-vmax-percentile", type=float, default=99.5, help="Color normalization percentile for orbit volume rendering")
    parser.add_argument("--volume-iso-percentile", type=float, default=97.0, help="Upper percentile of abs(vorticity) used for signed isosurfaces")
    parser.add_argument("--volume-iso-min-percentile", type=float, default=90.0, help="Lower percentile of abs(vorticity) used for weak transparent isosurfaces")
    parser.add_argument("--volume-iso-levels", type=int, default=4, help="Number of positive/negative isosurface levels for gradient-like rendering")
    parser.add_argument("--volume-iso-alpha", type=float, default=0.48, help="Maximum alpha value for strongest signed vorticity isosurfaces")
    parser.add_argument("--volume-iso-smooth-iter", type=int, default=8, help="Smoothing iterations applied to extracted isosurfaces")


    parser.add_argument("--volume-percentile", type=float, default=97.5, help="Vorticity percentile threshold for orbit point rendering")

    parser.add_argument("--volume-color-axis", type=str, default="z", choices=["x", "y", "z"], help="Global vorticity component used for red/blue coloring in orbit mode")
    parser.add_argument("--volume-max-points", type=int, default=50000, help="Maximum scatter points for orbit point rendering")
    parser.add_argument("--volume-point-size", type=float, default=2.0, help="Scatter point size for orbit point rendering")

    parser.add_argument("--orbit-elev", type=float, default=28.0, help="Camera elevation for orbit volume rendering")
    parser.add_argument("--orbit-azim-start", type=float, default=-60.0, help="Initial camera azimuth for orbit volume rendering")
    parser.add_argument("--orbit-azim-speed", type=float, default=0.0, help="Azimuth degrees advanced per exported frame; 0 disables camera rotation")


    parser.add_argument("--no-render", action="store_true", help="Run coupled LBM simulation without MuJoCo/LBM rendering and print sim FPS")
    parser.add_argument("--benchmark-steps", type=int, default=300, help="Number of steps for --no-render benchmark")
    parser.add_argument("--benchmark-progress-every", type=int, default=10, help="Print progress every N steps in --no-render mode")
    parser.add_argument("--dry-run", action="store_true", help="Load presets/config only and exit before creating env")



    args = parser.parse_args()

    config_path = resolve_config_path(args.config, args.animal)
    config_data = load_json_config(config_path)
    args.animal = str(config_data.get("animal", args.animal))
    presets, action_keys = get_preset_section(config_data)
    control_cfg = dict(config_data.get("control", {}))
    if args.warmup_steps is None:
        args.warmup_steps = int(control_cfg.get("warmup_steps", 20))
    if args.transition_steps is None:
        args.transition_steps = int(control_cfg.get("transition_steps", 30))
    task_by_preset = dict(DEFAULT_TASK_BY_PRESET)

    task_by_preset.update(control_cfg.get("task_by_preset", {}))
    keymap = build_keymap(presets, config_data.get("controls"))

    mode = args.preset or control_cfg.get("start_mode") or choose_idle_preset(presets)
    if mode not in presets:
        raise ValueError(f"Unknown preset '{mode}'. Choices: {list(presets.keys())}")

    if args.dry_run:
        model_cfg = config_data.get("model", {})
        print(
            f"config={config_path}, animal={args.animal}, env_type={model_cfg.get('env_type')}, "
            f"action_keys={action_keys}, presets={list(presets.keys())}"
        )
        return
    if args.view_mode == "orbit" and not args.export_lbm:
        raise ValueError("--view-mode orbit is only supported with --export-lbm")

    overrides = {}

    for key in ("nx", "ny", "nz", "per_frame_steps", "lbm_scale"):
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            overrides[f"lbm_{key}" if key in ("nx", "ny", "nz") else key] = value

    env = make_env(config_data, nworld=1, overrides=overrides)

    obs = env.reset()
    del obs

    base_env = env._env

    if args.export_lbm:
        export_lbm_video(env, args, presets, action_keys, task_by_preset, mode)
        return


    if args.no_render:

        task = task_by_preset.get(mode, "forward")
        set_task_if_supported(env, task)
        action_target = preset_to_action(presets[mode], action_keys).reshape(1, -1).astype(np.float32)

        total_reward = 0.0
        print(
            f"[no-render] starting benchmark: animal={args.animal} preset={mode} "
            f"steps={args.benchmark_steps} task={task}. First step may compile/capture CUDA graphs...",
            flush=True,
        )
        start = time.perf_counter()
        last_progress = start
        for step_idx in range(args.benchmark_steps):
            step_start = time.perf_counter()
            ramp = min(1.0, (step_idx + 1) / max(1, args.warmup_steps))
            action = np.clip(action_target * ramp, -1.0, 1.0).astype(np.float32)
            _obs, rewards, _dones, _infos = env.step(action)
            total_reward += float(rewards[0])
            if args.benchmark_progress_every > 0 and (
                (step_idx + 1) == 1 or (step_idx + 1) % args.benchmark_progress_every == 0
            ):
                now = time.perf_counter()
                recent_steps = 1 if (step_idx + 1) == 1 else args.benchmark_progress_every
                recent_fps = recent_steps / max(now - last_progress, 1.0e-9)
                print(
                    f"[no-render] step {step_idx + 1}/{args.benchmark_steps} "
                    f"last_step_ms={(now - step_start) * 1000.0:.2f} recent_fps={recent_fps:.2f}",
                    flush=True,
                )
                last_progress = now
        elapsed = time.perf_counter() - start

        sim_fps = args.benchmark_steps / max(elapsed, 1.0e-9)
        print(
            f"[no-render] animal={args.animal} preset={mode} steps={args.benchmark_steps} "
            f"elapsed={elapsed:.3f}s sim_fps={sim_fps:.2f} avg_step_ms={1000.0 / max(sim_fps, 1.0e-9):.2f} "
            f"total_reward={total_reward:.4f}"
        )
        return

    try:
        base_env.mj_model.vis.global_.offwidth = max(base_env.mj_model.vis.global_.offwidth, args.mujoco_width)
        base_env.mj_model.vis.global_.offheight = max(base_env.mj_model.vis.global_.offheight, args.mujoco_height)
    except Exception:
        pass

    renderer = MuJoCoRenderer(

        base_env.mj_model,
        width=args.mujoco_width,
        height=args.mujoco_height,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
    )

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    writer = None
    record_path = pathlib.Path(args.record) if args.record else None
    if record_path:
        record_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    mode_step = 0
    paused = False
    lbm_vmax = None
    last_reward = 0.0
    last_action = np.zeros_like(preset_to_action(presets[mode], action_keys).reshape(1, -1), dtype=np.float32)
    transition_from = last_action.copy()

    transition_step = args.transition_steps
    last_time = time.time()

    fps = 0.0

    print("Controls: W forward | A left | D right | F fast | S idle | Z ascend | C descend | Space pause | R reset | Q/Esc quit")

    try:
        while True:
            now = time.time()
            dt_wall = now - last_time
            last_time = now
            if dt_wall > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt_wall) if fps > 0 else 1.0 / dt_wall

            task = task_by_preset.get(mode, "forward")
            if not paused:
                set_task_if_supported(env, task)
                action_target = preset_to_action(presets[mode], action_keys).reshape(1, -1)

                if transition_step < args.transition_steps:
                    u = (transition_step + 1) / max(1, args.transition_steps)
                    alpha = u * u * (3.0 - 2.0 * u)  # smoothstep
                    action = (1.0 - alpha) * transition_from + alpha * action_target
                    transition_step += 1
                else:
                    ramp = min(1.0, (mode_step + 1) / max(1, args.warmup_steps))
                    action = action_target * ramp
                action = np.clip(action.astype(np.float32), -1.0, 1.0)
                _obs, rewards, _dones, _infos = env.step(action)
                last_reward = float(rewards[0])
                last_action = action
                step += 1
                mode_step += 1


            mj_frame = get_mujoco_frame(env, renderer, world_idx=0, with_fluid_force=False)
            if args.with_lbm:
                raw = get_raw_frame_3d(env, world_idx=0, render_type=args.render_type, view_mode=args.view_mode)
                lbm_vmax = compute_lbm_vmax(raw, args.render_type, lbm_vmax)
                lbm_frame = process_raw_to_frame(raw, lbm_vmax, args.render_type)
                combined = combine_frames(lbm_frame, mj_frame, args.output_height)
            else:
                combined = resize_to_height(mj_frame, args.output_height)

            combined = draw_overlay(combined, args.animal, mode, task, step, last_action, last_reward, fps, paused)

            if writer is None and record_path is not None:
                h, w = combined.shape[:2]
                writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"), args.record_fps, (w, h))
            if writer is not None:
                writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            cv2.imshow(args.window_name, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("r"), ord("R")):
                env.reset()
                step = 0
                mode_step = 0
                lbm_vmax = None
                last_reward = 0.0
                last_action = np.zeros_like(last_action, dtype=np.float32)
                transition_from = last_action.copy()
                transition_step = args.transition_steps
            elif key in keymap:
                new_mode = keymap[key]
                if new_mode != mode:
                    transition_from = last_action.copy()
                    transition_step = 0
                    mode = new_mode
                    mode_step = 0

    finally:
        if writer is not None:
            writer.release()
            print(f"Recorded video saved to: {record_path}")
        renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
