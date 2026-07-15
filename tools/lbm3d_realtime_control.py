"""
3D realtime LBM + MuJoCo control demo.

Model selection, LBM parameters, keyboard bindings, task mapping, and preset
actions are loaded from a JSON config under `configs/realtime_3d`.

Examples:
    python tools/lbm3d_realtime_control.py --config configs/realtime_3d/eel3d.json --with-lbm
    python tools/lbm3d_realtime_control.py --config configs/realtime_3d/eel3d.json --preset forward --view-mode orbit
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
    Mouse    drag to orbit, wheel to zoom (live orbit view)
"""

import argparse
import ctypes
import json
import math
import pathlib
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional

import cv2
import numpy as np
import warp as wp


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
from envs.lbm3d.lbm_core_3d import HomeFlow3D


_LIVE_HISTOGRAM_BINS = 256
_LIVE_LOG_VORT_MIN = -18.420680743952367  # log(1e-8)
_LIVE_LOG_VORT_MAX = 0.0                  # log(1.0)


@wp.kernel
def _clear_live_histogram(histogram: wp.array(dtype=wp.int32)):
    histogram[wp.tid()] = 0


@wp.kernel
def _extract_live_vorticity_slices(
    flow: HomeFlow3D,
    slice_positions: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.float32),
    histogram: wp.array(dtype=wp.int32),
    plane_width: int,
    plane_height: int,
    stride: int,
    slice_axis: int,
    color_axis: int,
    boundary_margin: int,
    collect_histogram: int,
):
    """Extract signed vorticity directly on the GPU for a stack of planes."""
    p, q, slice_id = wp.tid()
    x = int(0)
    y = int(0)
    z = int(0)
    plane_position = slice_positions[slice_id]

    if slice_axis == 0:
        x = plane_position
        y = wp.min(p * stride, flow.ny - 1)
        z = wp.min(q * stride, flow.nz - 1)
    elif slice_axis == 1:
        x = wp.min(p * stride, flow.nx - 1)
        y = plane_position
        z = wp.min(q * stride, flow.nz - 1)
    else:
        x = wp.min(p * stride, flow.nx - 1)
        y = wp.min(q * stride, flow.ny - 1)
        z = plane_position

    xm = wp.max(x - 1, 0)
    xp = wp.min(x + 1, flow.nx - 1)
    ym = wp.max(y - 1, 0)
    yp = wp.min(y + 1, flow.ny - 1)
    zm = wp.max(z - 1, 0)
    zp = wp.min(z + 1, flow.nz - 1)

    inv_dx = 1.0 / float(wp.max(xp - xm, 1))
    inv_dy = 1.0 / float(wp.max(yp - ym, 1))
    inv_dz = 1.0 / float(wp.max(zp - zm, 1))

    vort_x = (
        (flow.u[x, yp, z][2] - flow.u[x, ym, z][2]) * inv_dy
        - (flow.u[x, y, zp][1] - flow.u[x, y, zm][1]) * inv_dz
    )
    vort_y = (
        (flow.u[x, y, zp][0] - flow.u[x, y, zm][0]) * inv_dz
        - (flow.u[xp, y, z][2] - flow.u[xm, y, z][2]) * inv_dx
    )
    vort_z = (
        (flow.u[xp, y, z][1] - flow.u[xm, y, z][1]) * inv_dx
        - (flow.u[x, yp, z][0] - flow.u[x, ym, z][0]) * inv_dy
    )

    value = vort_z
    if color_axis == 0:
        value = vort_x
    elif color_axis == 1:
        value = vort_y
    if (
        x < boundary_margin
        or x >= flow.nx - boundary_margin
        or y < boundary_margin
        or y >= flow.ny - boundary_margin
        or z < boundary_margin
        or z >= flow.nz - boundary_margin
    ):
        value = 0.0

    offset = (slice_id * plane_height + q) * plane_width + p
    values[offset] = value

    if collect_histogram != 0:
        magnitude = wp.abs(value)
        if magnitude > 1.0e-8:
            normalized = (wp.log(magnitude) - _LIVE_LOG_VORT_MIN) / (_LIVE_LOG_VORT_MAX - _LIVE_LOG_VORT_MIN)
            bin_id = int(wp.clamp(normalized, 0.0, 0.999999) * float(_LIVE_HISTOGRAM_BINS))
            wp.atomic_add(histogram, bin_id, 1)


@wp.func
def _live_clamp01(value: float) -> float:
    return wp.min(wp.max(value, 0.0), 1.0)


@wp.kernel
def _live_vorticity_to_rgba(
    values: wp.array(dtype=wp.float32),
    rgba: wp.array(dtype=wp.uint8),
    vmax: float,
    alpha: float,
):
    """Apply a compact blue-white-red map in CUDA and write the GL atlas."""
    i = wp.tid()
    value = values[i]
    normalized = _live_clamp01(0.5 + 0.5 * value / wp.max(vmax, 1.0e-8))
    r = float(0.0)
    g = float(0.0)
    b = float(0.0)
    if normalized < 0.5:
        t = normalized * 2.0
        r = 255.0 * t
        g = 255.0 * t
        b = 255.0
    else:
        t = (normalized - 0.5) * 2.0
        r = 255.0
        g = 255.0 * (1.0 - t)
        b = 255.0 * (1.0 - t)
    base = i * 4
    rgba[base] = wp.uint8(r)
    rgba[base + 1] = wp.uint8(g)
    rgba[base + 2] = wp.uint8(b)
    rgba[base + 3] = wp.uint8(255.0 * _live_clamp01(alpha))


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


def draw_live_lbm_hud(
    mode: str,
    task: str,
    step: int,
    mode_step: int,
    action: np.ndarray,
    displacement: np.ndarray,
    reward: float,
    fps: float,
    vmax: float,
    paused: bool,
    controls_line: str,
) -> np.ndarray:
    """Create the transparent top-left status overlay used by live orbit."""
    width, height = 680, 166
    hud = np.zeros((height, width, 4), dtype=np.uint8)
    action_str = ", ".join(f"{value:+.2f}" for value in np.asarray(action).reshape(-1))
    lines = [
        f"mode: {mode}  task: {task} {'[PAUSED]' if paused else ''}",
        f"step: {step}  mode_step: {mode_step}",
        f"action: [{action_str}]",
        f"dx: {displacement[0]:+.2f}  dy: {displacement[1]:+.2f}  dz: {displacement[2]:+.2f}  reward: {reward:+.4f}",
        f"fps: {fps:.1f}  vorticity vmax: {vmax:.4g}",
        controls_line,
        "Left drag orbit/wheel zoom | Right bars drag action | Space/R/Q",
    ]
    for line_index, line in enumerate(lines):
        y = 20 + line_index * 22
        cv2.putText(hud, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(hud, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (18, 18, 18, 255), 1, cv2.LINE_AA)
    return hud


def live_action_panel_layout(width: int, height: int, action_count: int) -> Dict[str, Any]:
    """Shared bar geometry for drawing and mouse hit-testing."""
    top = 132
    bottom_reserved = 92
    available_height = max(100, height - top - bottom_reserved)
    row_height = max(42, min(72, available_height // max(action_count, 1)))
    label_width = min(112, max(72, width // 4))
    value_width = 62
    bar_x0 = label_width
    bar_x1 = width - value_width
    bar_width = max(80, bar_x1 - bar_x0)
    return {
        "top": top,
        "bottom_reserved": bottom_reserved,
        "row_height": row_height,
        "bar_x0": bar_x0,
        "bar_x1": bar_x1,
        "bar_width": bar_width,
        "center_x": bar_x0 + bar_width // 2,
    }


def draw_live_action_panel(
    width: int,
    height: int,
    mode: str,
    task: str,
    step: int,
    mode_step: int,
    action: np.ndarray,
    action_keys: list[str],
    reward: float,
    fps: float,
    paused: bool,
    controls_line: str,
) -> np.ndarray:
    """Draw the 2D-style action bar panel for the realtime 3D window."""
    panel = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (52, 58, 68), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        panel,
        f"Control  mode={mode} {'[PAUSED]' if paused else ''}",
        (14, 32),
        font,
        0.66,
        (235, 238, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(panel, f"task={task}", (14, 58), font, 0.48, (180, 190, 205), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"step={step}  mode_step={mode_step}  fps={fps:.1f}",
        (14, 82),
        font,
        0.46,
        (180, 190, 205),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(panel, f"reward={reward:+.4f}", (14, 106), font, 0.46, (160, 178, 198), 1, cv2.LINE_AA)

    raw_values = np.asarray(action, dtype=np.float32).reshape(-1)
    norm_values = np.clip(raw_values, -1.0, 1.0)
    count = int(raw_values.size)
    layout = live_action_panel_layout(width, height, count)
    top = layout["top"]
    bottom_reserved = layout["bottom_reserved"]
    row_height = layout["row_height"]
    bar_x0 = layout["bar_x0"]
    bar_x1 = layout["bar_x1"]
    bar_width = layout["bar_width"]
    center_x = layout["center_x"]

    for index, (normalized, raw_value) in enumerate(zip(norm_values, raw_values)):
        y = top + index * row_height
        if y + 30 >= height - bottom_reserved:
            break
        label = action_keys[index] if index < len(action_keys) else f"u{index}"
        cv2.putText(panel, label, (14, y + 21), font, 0.48, (215, 220, 230), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (bar_x0, y + 5), (bar_x1, y + 25), (54, 60, 70), -1)
        cv2.line(panel, (center_x, y + 1), (center_x, y + 29), (130, 140, 155), 1, cv2.LINE_AA)
        end_x = int(round(center_x + float(normalized) * (bar_width * 0.5)))
        color = (75, 190, 255) if normalized >= 0 else (255, 150, 90)
        cv2.rectangle(panel, (min(center_x, end_x), y + 8), (max(center_x, end_x), y + 22), color, -1)
        cv2.rectangle(panel, (bar_x0, y + 5), (bar_x1, y + 25), (100, 108, 122), 1)
        cv2.circle(panel, (end_x, y + 15), 5, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.putText(panel, f"{raw_value:+.2f}", (bar_x1 + 7, y + 21), font, 0.43, (215, 220, 230), 1, cv2.LINE_AA)

    cv2.putText(panel, "Mouse drag bars: manual action override", (14, height - 66), font, 0.38, (205, 215, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, controls_line, (14, height - 42), font, 0.36, (180, 190, 205), 1, cv2.LINE_AA)
    cv2.putText(panel, "Preset key exits manual | Space/R/Q", (14, height - 18), font, 0.38, (180, 190, 205), 1, cv2.LINE_AA)
    return panel


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


class OpenGLLiveSliceDisplay:
    """Persistent CUDA/OpenGL renderer for realtime 3D vorticity slices."""

    def __init__(self, env, flow: HomeFlow3D, args, action_keys: list[str]):
        try:
            import glfw
            from OpenGL import GL, GLU
            from cuda.bindings import driver as cu
        except Exception as exc:
            raise RuntimeError(
                "Realtime orbit rendering requires glfw, PyOpenGL, and cuda-python"
            ) from exc

        if str(env._env.lbm_solver.device).split(":", 1)[0] != "cuda":
            raise RuntimeError("Realtime orbit rendering requires a CUDA LBM device")
        if args.volume_render_mode != "slices":
            raise ValueError("Realtime orbit mode currently supports --volume-render-mode slices only")

        self.env = env
        self.flow = flow
        self.args = args
        self.action_keys = list(action_keys)
        self.glfw = glfw
        self.GL = GL
        self.GLU = GLU
        self.cu = cu
        self.device = env._env.lbm_solver.device
        self.slice_axis_name = str(args.volume_slice_axis).lower()
        self.color_axis_name = str(args.volume_color_axis).lower()
        self.slice_axis = {"x": 0, "y": 1, "z": 2}[self.slice_axis_name]
        self.color_axis = {"x": 0, "y": 1, "z": 2}[self.color_axis_name]
        self.stride = max(1, int(args.volume_stride))
        self.slice_count = max(1, int(args.volume_slice_count))
        self.show_mujoco = bool(args.orbit_with_mujoco)
        self.output_height = int(args.output_height)
        self.left_width = int(args.volume_width)
        self.right_width = (
            max(1, int(round(args.mujoco_width * self.output_height / args.mujoco_height)))
            if self.show_mujoco
            else int(args.action_panel_width)
        )
        self.separator = 4
        self.window_width = self.left_width + self.separator + self.right_width
        self.mj_width = int(args.mujoco_width)
        self.mj_height = int(args.mujoco_height)
        self.right_texture_width = self.mj_width if self.show_mujoco else self.right_width
        self.right_texture_height = self.mj_height if self.show_mujoco else self.output_height
        self.hud_width = min(680, self.left_width)
        self.hud_height = 166
        self.histogram_refresh = 8
        self.vmax = None
        self.frame_index = 0
        self.azimuth_offset = 0.0
        self.elevation_offset = 0.0
        self.zoom = 1.0
        self.dragging = False
        self.action_drag_index: Optional[int] = None
        self.pending_action_edits: Dict[int, float] = {}
        self.last_cursor = None
        self.prev_keys: Dict[int, bool] = {}

        if self.slice_axis == 0:
            self.plane_width = (flow.ny + self.stride - 1) // self.stride
            self.plane_height = (flow.nz + self.stride - 1) // self.stride
            axis_size = flow.nx
        elif self.slice_axis == 1:
            self.plane_width = (flow.nx + self.stride - 1) // self.stride
            self.plane_height = (flow.nz + self.stride - 1) // self.stride
            axis_size = flow.ny
        else:
            self.plane_width = (flow.nx + self.stride - 1) // self.stride
            self.plane_height = (flow.ny + self.stride - 1) // self.stride
            axis_size = flow.nz

        self.slice_count = min(self.slice_count, axis_size)
        self.slice_positions_np = np.linspace(0, axis_size - 1, self.slice_count, dtype=np.int32)
        self.slice_positions = wp.array(self.slice_positions_np, dtype=wp.int32, device=self.device)
        self.pixel_count = self.plane_width * self.plane_height * self.slice_count
        self.values = wp.empty(self.pixel_count, dtype=wp.float32, device=self.device)
        self.rgba = wp.empty(self.pixel_count * 4, dtype=wp.uint8, device=self.device)
        self.histogram = wp.zeros(_LIVE_HISTOGRAM_BINS, dtype=wp.int32, device=self.device)

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        self.window = glfw.create_window(self.window_width, self.output_height, args.window_name, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create realtime orbit window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        self.atlas_texture = self._create_texture(self.plane_width, self.plane_height * self.slice_count, GL.GL_RGBA)
        self.right_texture = self._create_texture(
            self.right_texture_width, self.right_texture_height, GL.GL_RGB
        )
        self.hud_texture = self._create_texture(self.hud_width, self.hud_height, GL.GL_RGBA)
        self.pbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self.pixel_count * 4, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        self._cu_check(cu.cuInit(0))
        err, cuda_device = cu.cuDeviceGet(0)
        self._cu_check(err)
        err, context = cu.cuDevicePrimaryCtxRetain(cuda_device)
        self._cu_check(err)
        self.cuda_context = context
        self._cu_check(cu.cuCtxSetCurrent(context))
        err, resource = cu.cuGraphicsGLRegisterBuffer(
            int(self.pbo), cu.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD
        )
        self._cu_check(err)
        self.cuda_resource = resource
        self.mesh_cache = self._build_mesh_cache()

    def _cu_check(self, result) -> None:
        err = result[0] if isinstance(result, tuple) else result
        if err != self.cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {err}")

    def _create_texture(self, width: int, height: int, fmt: int) -> int:
        GL = self.GL
        texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        internal = GL.GL_RGBA8 if fmt == GL.GL_RGBA else GL.GL_RGB8
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, width, height, 0, fmt, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return int(texture)

    def _build_mesh_cache(self) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
        base_env = self.env._env
        solver = getattr(base_env, "lbm_solver", None)
        meshes = getattr(solver, "meshes", None) or []
        cache = []
        for solid_id, mesh in enumerate(meshes):
            if mesh is None:
                continue
            vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
            faces = np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.uint32)
            mapping = getattr(solver, "mujoco_mappings", {}).get(solid_id, {})
            scale = float(mapping.get("scale", getattr(base_env, "coordinate_scale", 1.0)))
            cache.append((solid_id, vertices, faces, scale))
        return cache

    def _update_vmax_from_histogram(self) -> None:
        counts = self.histogram.numpy().astype(np.int64, copy=False)
        total = int(counts.sum())
        if total <= 0:
            current = 1.0e-6
        else:
            target = max(1, int(math.ceil(total * float(self.args.volume_vmax_percentile) / 100.0)))
            bin_id = int(np.searchsorted(np.cumsum(counts), target, side="left"))
            fraction = (min(bin_id, _LIVE_HISTOGRAM_BINS - 1) + 0.5) / _LIVE_HISTOGRAM_BINS
            current = math.exp(_LIVE_LOG_VORT_MIN + fraction * (_LIVE_LOG_VORT_MAX - _LIVE_LOG_VORT_MIN))
        self.vmax = max(current, 1.0e-8) if self.vmax is None else max(self.vmax * 0.96, current, 1.0e-8)

    def update_lbm_texture(self) -> None:
        collect = self.frame_index % self.histogram_refresh == 0
        if collect:
            wp.launch(
                _clear_live_histogram,
                dim=_LIVE_HISTOGRAM_BINS,
                inputs=[self.histogram],
                device=self.device,
            )
        wp.launch(
            _extract_live_vorticity_slices,
            dim=(self.plane_width, self.plane_height, self.slice_count),
            inputs=[
                self.flow,
                self.slice_positions,
                self.values,
                self.histogram,
                self.plane_width,
                self.plane_height,
                self.stride,
                self.slice_axis,
                self.color_axis,
                max(0, int(self.args.volume_boundary_margin)),
                int(collect),
            ],
            device=self.device,
        )
        if collect:
            wp.synchronize()
            self._update_vmax_from_histogram()
        if self.vmax is None:
            self.vmax = 1.0e-4
        wp.launch(
            _live_vorticity_to_rgba,
            dim=self.pixel_count,
            inputs=[self.values, self.rgba, float(self.vmax), float(self.args.volume_slice_alpha)],
            device=self.device,
        )
        wp.synchronize()

        self.glfw.make_context_current(self.window)
        GL = self.GL
        cu = self.cu
        GL.glFinish()
        self._cu_check(cu.cuGraphicsMapResources(1, self.cuda_resource, 0))
        try:
            err, pointer, size = cu.cuGraphicsResourceGetMappedPointer(self.cuda_resource)
            self._cu_check(err)
            byte_count = self.pixel_count * 4
            if int(size) < byte_count:
                raise RuntimeError("Mapped slice PBO is smaller than expected")
            self._cu_check(cu.cuMemcpyDtoD(int(pointer), int(self.rgba.ptr), byte_count))
        finally:
            self._cu_check(cu.cuGraphicsUnmapResources(1, self.cuda_resource, 0))

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.atlas_texture)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, int(self.pbo))
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.plane_width,
            self.plane_height * self.slice_count,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def update_right_texture(self, frame: np.ndarray) -> None:
        self.glfw.make_context_current(self.window)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape[:2] != (self.right_texture_height, self.right_texture_width):
            frame = cv2.resize(frame, (self.right_texture_width, self.right_texture_height), interpolation=cv2.INTER_AREA)
        GL = self.GL
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.right_texture)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, self.right_texture_width, self.right_texture_height,
            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, frame,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def update_hud_texture(self, hud: np.ndarray) -> None:
        self.glfw.make_context_current(self.window)
        hud = np.ascontiguousarray(hud, dtype=np.uint8)
        if hud.shape[:2] != (self.hud_height, self.hud_width):
            hud = cv2.resize(hud, (self.hud_width, self.hud_height), interpolation=cv2.INTER_AREA)
        GL = self.GL
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.hud_texture)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, self.hud_width, self.hud_height,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, hud,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _camera(self) -> tuple[np.ndarray, np.ndarray]:
        center = np.array([self.flow.nx * 0.5, self.flow.ny * 0.5, self.flow.nz * 0.5], dtype=np.float32)
        radius = (
            float(max(self.flow.nx, self.flow.ny, self.flow.nz))
            * 1.8
            * self.zoom
            / max(float(self.args.orbit_zoom), 1.0e-3)
        )
        azimuth = (
            float(self.args.orbit_azim_start)
            + self.frame_index * float(self.args.orbit_azim_speed)
            + self.azimuth_offset
        )
        elevation = float(self.args.orbit_elev) + self.elevation_offset
        az = np.deg2rad(azimuth)
        el = np.deg2rad(elevation)
        eye = center + radius * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], dtype=np.float32
        )
        return center, eye

    def _draw_slice(self, slice_id: int) -> None:
        GL = self.GL
        position = float(self.slice_positions_np[slice_id])
        # Keep bilinear filtering inside this slice's atlas tile. Without the
        # half-texel inset, adjacent planes bleed into each other at their edges.
        atlas_height = self.plane_height * self.slice_count
        v0 = (slice_id * self.plane_height + 0.5) / atlas_height
        v1 = ((slice_id + 1) * self.plane_height - 0.5) / atlas_height
        if self.slice_axis == 0:
            vertices = ((position, 0, 0), (position, self.flow.ny, 0), (position, self.flow.ny, self.flow.nz), (position, 0, self.flow.nz))
        elif self.slice_axis == 1:
            vertices = ((0, position, 0), (self.flow.nx, position, 0), (self.flow.nx, position, self.flow.nz), (0, position, self.flow.nz))
        else:
            vertices = ((0, 0, position), (self.flow.nx, 0, position), (self.flow.nx, self.flow.ny, position), (0, self.flow.ny, position))
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, v0); GL.glVertex3f(*vertices[0])
        GL.glTexCoord2f(1.0, v0); GL.glVertex3f(*vertices[1])
        GL.glTexCoord2f(1.0, v1); GL.glVertex3f(*vertices[2])
        GL.glTexCoord2f(0.0, v1); GL.glVertex3f(*vertices[3])
        GL.glEnd()

    def _draw_slices(self, eye: np.ndarray) -> None:
        GL = self.GL
        centers = []
        for slice_id, position in enumerate(self.slice_positions_np):
            center = np.array([self.flow.nx * 0.5, self.flow.ny * 0.5, self.flow.nz * 0.5], dtype=np.float32)
            center[self.slice_axis] = float(position)
            centers.append((float(np.linalg.norm(center - eye)), slice_id))
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.atlas_texture)
        GL.glColor4f(1.0, 1.0, 1.0, 1.0)
        for _, slice_id in sorted(centers, reverse=True):
            self._draw_slice(slice_id)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDisable(GL.GL_BLEND)

    def _draw_solids(self) -> None:
        if not self.mesh_cache:
            return
        GL = self.GL
        positions = self.flow.solid_position.numpy().astype(np.float32)
        quaternions = self.flow.solid_quaternion.numpy().astype(np.float32)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glColor3f(0.28, 0.28, 0.28)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        for solid_id, vertices, faces, scale in self.mesh_cache:
            if solid_id >= len(positions):
                break
            world_vertices = np.ascontiguousarray(
                (vertices * scale) @ quat_wxyz_to_matrix(quaternions[solid_id]).T + positions[solid_id],
                dtype=np.float32,
            )
            GL.glVertexPointer(3, GL.GL_FLOAT, 0, world_vertices)
            GL.glDrawElements(GL.GL_TRIANGLES, int(faces.size), GL.GL_UNSIGNED_INT, faces)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

    def _draw_box(self) -> None:
        GL = self.GL
        nx, ny, nz = self.flow.nx, self.flow.ny, self.flow.nz
        edges = (
            ((0, 0, 0), (nx, 0, 0)), ((0, ny, 0), (nx, ny, 0)), ((0, 0, nz), (nx, 0, nz)), ((0, ny, nz), (nx, ny, nz)),
            ((0, 0, 0), (0, ny, 0)), ((nx, 0, 0), (nx, ny, 0)), ((0, 0, nz), (0, ny, nz)), ((nx, 0, nz), (nx, ny, nz)),
            ((0, 0, 0), (0, 0, nz)), ((nx, 0, 0), (nx, 0, nz)), ((0, ny, 0), (0, ny, nz)), ((nx, ny, 0), (nx, ny, nz)),
        )
        GL.glColor3f(0.55, 0.55, 0.55)
        GL.glBegin(GL.GL_LINES)
        for start, end in edges:
            GL.glVertex3f(*start)
            GL.glVertex3f(*end)
        GL.glEnd()

    def draw(self, right_frame: np.ndarray, hud: np.ndarray, title: str) -> None:
        self.update_right_texture(right_frame)
        self.update_hud_texture(hud)
        glfw = self.glfw
        GL = self.GL
        glfw.make_context_current(self.window)
        glfw.set_window_title(self.window, title)
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        left_width = int(round(framebuffer_width * self.left_width / self.window_width))
        right_x = int(round(framebuffer_width * (self.left_width + self.separator) / self.window_width))
        center, eye = self._camera()

        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glViewport(0, 0, left_width, framebuffer_height)
        GL.glScissor(0, 0, left_width, framebuffer_height)
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU = self.GLU
        GLU.gluPerspective(42.0, left_width / max(framebuffer_height, 1), 1.0, float(max(self.flow.nx, self.flow.ny, self.flow.nz)) * 8.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GLU.gluLookAt(*eye, *center, 0.0, 0.0, 1.0)
        self._draw_slices(eye)
        self._draw_solids()
        if self.args.orbit_show_box:
            self._draw_box()

        # Transparent 2D-style parameter overlay in the upper-left corner.
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0.0, float(left_width), 0.0, float(framebuffer_height), -1.0, 1.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.hud_texture)
        GL.glColor4f(1.0, 1.0, 1.0, 1.0)
        hud_width = min(float(self.hud_width), max(float(left_width) - 20.0, 1.0))
        hud_height = hud_width * self.hud_height / self.hud_width
        x0 = 10.0
        y1 = float(framebuffer_height) - 8.0
        y0 = y1 - hud_height
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 1.0); GL.glVertex2f(x0, y0)
        GL.glTexCoord2f(1.0, 1.0); GL.glVertex2f(x0 + hud_width, y0)
        GL.glTexCoord2f(1.0, 0.0); GL.glVertex2f(x0 + hud_width, y1)
        GL.glTexCoord2f(0.0, 0.0); GL.glVertex2f(x0, y1)
        GL.glEnd()
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDisable(GL.GL_BLEND)

        # Right panel: action bars by default, optional MuJoCo with
        # --orbit-with-mujoco. Both are ordinary top-down RGB images.
        GL.glViewport(right_x, 0, framebuffer_width - right_x, framebuffer_height)
        GL.glScissor(right_x, 0, framebuffer_width - right_x, framebuffer_height)
        GL.glClearColor(0.02, 0.02, 0.02, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.right_texture)
        GL.glColor3f(1.0, 1.0, 1.0)
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 1.0); GL.glVertex2f(0.0, 0.0)
        GL.glTexCoord2f(1.0, 1.0); GL.glVertex2f(1.0, 0.0)
        GL.glTexCoord2f(1.0, 0.0); GL.glVertex2f(1.0, 1.0)
        GL.glTexCoord2f(0.0, 0.0); GL.glVertex2f(0.0, 1.0)
        GL.glEnd()
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        self.frame_index += 1

    def _on_mouse_button(self, _window, button, action, _mods) -> None:
        if button != self.glfw.MOUSE_BUTTON_LEFT:
            return
        if action == self.glfw.RELEASE:
            self.dragging = False
            self.action_drag_index = None
            self.last_cursor = None
            return
        if action != self.glfw.PRESS:
            return

        x, y = self.glfw.get_cursor_pos(self.window)
        action_index = self._action_index_at_cursor(x, y)
        if action_index is not None:
            self.action_drag_index = action_index
            self.dragging = False
            self._queue_action_edit(x)
        else:
            window_width, _ = self.glfw.get_window_size(self.window)
            left_boundary = window_width * self.left_width / self.window_width
            self.dragging = x < left_boundary
            self.action_drag_index = None
            self.last_cursor = None

    def _on_cursor(self, _window, x: float, y: float) -> None:
        if self.action_drag_index is not None:
            self._queue_action_edit(x)
            return
        if not self.dragging:
            return
        if self.last_cursor is not None:
            dx = x - self.last_cursor[0]
            dy = y - self.last_cursor[1]
            self.azimuth_offset += dx * 0.35
            self.elevation_offset = float(np.clip(self.elevation_offset - dy * 0.25, -80.0, 80.0))
        self.last_cursor = (x, y)

    def _on_scroll(self, _window, _xoffset: float, yoffset: float) -> None:
        self.zoom = float(np.clip(self.zoom * math.exp(-0.10 * yoffset), 0.35, 3.0))

    def _cursor_to_action_panel(self, x: float, y: float) -> Optional[tuple[float, float]]:
        if self.show_mujoco:
            return None
        window_width, window_height = self.glfw.get_window_size(self.window)
        if window_width <= 0 or window_height <= 0:
            return None
        right_x = window_width * (self.left_width + self.separator) / self.window_width
        if x < right_x:
            return None
        panel_x = (x - right_x) / max(window_width - right_x, 1.0) * self.right_texture_width
        panel_y = y / window_height * self.right_texture_height
        return panel_x, panel_y

    def _action_index_at_cursor(self, x: float, y: float) -> Optional[int]:
        panel_position = self._cursor_to_action_panel(x, y)
        if panel_position is None:
            return None
        panel_x, panel_y = panel_position
        layout = live_action_panel_layout(
            self.right_texture_width, self.right_texture_height, len(self.action_keys)
        )
        if panel_x < layout["bar_x0"] - 8 or panel_x > layout["bar_x1"] + 8:
            return None
        for index in range(len(self.action_keys)):
            row_y = layout["top"] + index * layout["row_height"]
            if row_y <= panel_y <= row_y + 32:
                return index
        return None

    def _queue_action_edit(self, cursor_x: float) -> None:
        if self.action_drag_index is None:
            return
        _, cursor_y = self.glfw.get_cursor_pos(self.window)
        panel_position = self._cursor_to_action_panel(cursor_x, cursor_y)
        if panel_position is None:
            # Keep dragging when the pointer leaves the panel horizontally by
            # projecting against its nearest edge.
            window_width, window_height = self.glfw.get_window_size(self.window)
            right_x = window_width * (self.left_width + self.separator) / self.window_width
            panel_x = (cursor_x - right_x) / max(window_width - right_x, 1.0) * self.right_texture_width
        else:
            panel_x = panel_position[0]
        layout = live_action_panel_layout(
            self.right_texture_width, self.right_texture_height, len(self.action_keys)
        )
        value = (panel_x - layout["center_x"]) / max(layout["bar_width"] * 0.5, 1.0)
        self.pending_action_edits[self.action_drag_index] = float(np.clip(value, -1.0, 1.0))

    def pop_action_edits(self) -> Dict[int, float]:
        edits = dict(self.pending_action_edits)
        self.pending_action_edits.clear()
        return edits

    def key_once(self, key_code: int) -> bool:
        pressed = self.glfw.get_key(self.window, key_code) == self.glfw.PRESS
        previous = self.prev_keys.get(key_code, False)
        self.prev_keys[key_code] = pressed
        return pressed and not previous

    def should_close(self) -> bool:
        return bool(self.glfw.window_should_close(self.window))

    def close(self) -> None:
        try:
            if hasattr(self, "window"):
                self.glfw.make_context_current(self.window)
            if hasattr(self, "cuda_resource"):
                self._cu_check(self.cu.cuGraphicsUnregisterResource(self.cuda_resource))
        finally:
            if hasattr(self, "pbo"):
                self.GL.glDeleteBuffers(1, [self.pbo])
            if hasattr(self, "atlas_texture"):
                self.GL.glDeleteTextures([self.atlas_texture, self.right_texture, self.hud_texture])
            if hasattr(self, "window"):
                self.glfw.destroy_window(self.window)
            self.glfw.terminate()


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

    parser.add_argument("--with-lbm", action="store_true", help="Show LBM next to MuJoCo; implied by live --view-mode orbit")
    parser.add_argument("--render-type", type=str, default="vorticity", choices=["velocity", "vorticity"])
    parser.add_argument("--view-mode", type=str, default="topdown", choices=["topdown", "max_topdown", "side", "front", "orbit"])

    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--per-frame-steps", type=int, default=None, help="Coupled substeps per displayed frame; live orbit defaults to 2, other modes use JSON")
    parser.add_argument("--lbm-scale", type=float, default=None)
    parser.add_argument("--output-height", type=int, default=720)
    parser.add_argument("--action-panel-width", type=int, default=420, help="Width of the right-side live action panel")
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
    parser.add_argument("--volume-stride", type=int, default=1, help="Downsample stride for orbit volume rendering")
    parser.add_argument("--volume-render-mode", type=str, default="slices", choices=["slices", "isosurface", "points"], help="Orbit volume renderer: slice stack, signed isosurfaces, or thresholded point cloud")
    parser.add_argument("--volume-slice-axis", type=str, default="z", choices=["x", "y", "z"], help="Slice stacking axis for --volume-render-mode slices")
    parser.add_argument("--volume-slice-count", type=int, default=15, help="Number of slices for --volume-render-mode slices")
    parser.add_argument("--volume-contour-levels", type=int, default=64, help="Number of filled contour levels for slice rendering")
    parser.add_argument("--volume-slice-alpha", type=float, default=0.15, help="Alpha value for filled contour slices")
    parser.add_argument("--volume-vmax-percentile", type=float, default=98.0, help="Color normalization percentile for orbit volume rendering")
    parser.add_argument("--volume-boundary-margin", type=int, default=2, help="Hide this many outer grid cells in live orbit mode")
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
    parser.add_argument("--orbit-zoom", type=float, default=1.7, help="Initial live orbit camera zoom")
    parser.add_argument("--orbit-show-box", action="store_true", help="Show the LBM domain wireframe in live orbit mode")
    parser.add_argument("--orbit-with-mujoco", action="store_true", help="Replace the right-side action panel with MuJoCo")


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
    if args.view_mode == "orbit" and not args.export_lbm and not args.no_render:
        args.with_lbm = True
        if args.volume_render_mode != "slices":
            raise ValueError("Live --view-mode orbit currently supports --volume-render-mode slices only")
        if args.per_frame_steps is None:
            # The eel export preset uses 10 coupled substeps per captured video frame.
            # A live preview uses a smaller batch so controls and rendering stay near
            # 30 FPS; callers can still request the export cadence explicitly.
            args.per_frame_steps = 2
            print("[live-orbit] using --per-frame-steps 2 for realtime preview (override explicitly to change it)")

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

    needs_mujoco_frame = args.view_mode != "orbit" or args.orbit_with_mujoco
    renderer = None
    if needs_mujoco_frame:
        renderer = MuJoCoRenderer(
            base_env.mj_model,
            width=args.mujoco_width,
            height=args.mujoco_height,
            camera_distance=args.camera_distance,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
        )

    live_slice_display = None
    if args.view_mode == "orbit":
        flow = base_env.lbm_solver.flows[0]
        live_slice_display = OpenGLLiveSliceDisplay(env, flow, args, action_keys)
        if args.record:
            print("[warn] --record is disabled for live orbit rendering; use --export-lbm for video output.")
            args.record = None
    else:
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
    manual_action_override: Optional[np.ndarray] = None
    transition_from = last_action.copy()
    controls_line = "W forward | A left | D right | F fast | S idle | Z/C vertical"
    initial_solid_position = (
        flow.solid_position.numpy()[0].astype(np.float32).copy()
        if live_slice_display is not None and flow.solid_position.shape[0] > 0
        else np.zeros(3, dtype=np.float32)
    )

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
                if manual_action_override is not None:
                    action = manual_action_override.copy()
                else:
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


            mj_frame = (
                get_mujoco_frame(env, renderer, world_idx=0, with_fluid_force=False)
                if renderer is not None
                else None
            )
            if live_slice_display is not None:
                live_slice_display.update_lbm_texture()
                display_mode = f"{mode}/mouse" if manual_action_override is not None else mode
                if mj_frame is not None:
                    right_frame = draw_overlay(
                        mj_frame, args.animal, display_mode, task, step, last_action, last_reward, fps, paused
                    )
                else:
                    right_frame = draw_live_action_panel(
                        args.action_panel_width,
                        args.output_height,
                        display_mode,
                        task,
                        step,
                        mode_step,
                        last_action,
                        action_keys,
                        last_reward,
                        fps,
                        paused,
                        controls_line,
                    )
                current_solid_position = flow.solid_position.numpy()[0].astype(np.float32)
                displacement = current_solid_position - initial_solid_position
                hud = draw_live_lbm_hud(
                    display_mode,
                    task,
                    step,
                    mode_step,
                    last_action,
                    displacement,
                    last_reward,
                    fps,
                    float(live_slice_display.vmax),
                    paused,
                    controls_line,
                )
                live_slice_display.draw(
                    right_frame,
                    hud,
                    f"{args.window_name} | {mode} | {fps:.1f} FPS | vmax={live_slice_display.vmax:.3g}",
                )
                action_edits = live_slice_display.pop_action_edits()
                if action_edits:
                    if manual_action_override is None:
                        manual_action_override = last_action.copy()
                    for action_index, action_value in action_edits.items():
                        if 0 <= action_index < manual_action_override.shape[1]:
                            manual_action_override[0, action_index] = action_value
                    manual_action_override = np.clip(manual_action_override, -1.0, 1.0).astype(np.float32)
                    last_action = manual_action_override.copy()
                combined = None
            elif args.with_lbm:
                raw = get_raw_frame_3d(env, world_idx=0, render_type=args.render_type, view_mode=args.view_mode)
                lbm_vmax = compute_lbm_vmax(raw, args.render_type, lbm_vmax)
                lbm_frame = process_raw_to_frame(raw, lbm_vmax, args.render_type)
                combined = combine_frames(lbm_frame, mj_frame, args.output_height)
            else:
                combined = resize_to_height(mj_frame, args.output_height)

            if live_slice_display is not None:
                glfw = live_slice_display.glfw
                key = -1
                if (
                    live_slice_display.should_close()
                    or live_slice_display.key_once(glfw.KEY_ESCAPE)
                    or live_slice_display.key_once(glfw.KEY_Q)
                ):
                    key = 27
                elif live_slice_display.key_once(glfw.KEY_SPACE):
                    key = ord(" ")
                elif live_slice_display.key_once(glfw.KEY_R):
                    key = ord("r")
                else:
                    for ascii_key in sorted(set(keymap)):
                        if not chr(ascii_key).isalpha() or not chr(ascii_key).islower():
                            continue
                        if live_slice_display.key_once(ord(chr(ascii_key).upper())):
                            key = ascii_key
                            break
            else:
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
                manual_action_override = None
                if live_slice_display is not None and flow.solid_position.shape[0] > 0:
                    initial_solid_position = flow.solid_position.numpy()[0].astype(np.float32).copy()
                step = 0
                mode_step = 0
                lbm_vmax = None
                last_reward = 0.0
                last_action = np.zeros_like(last_action, dtype=np.float32)
                transition_from = last_action.copy()
                transition_step = args.transition_steps
            elif key in keymap:
                new_mode = keymap[key]
                if new_mode != mode or manual_action_override is not None:
                    transition_from = last_action.copy()
                    transition_step = 0
                    mode = new_mode
                    mode_step = 0
                    manual_action_override = None

    finally:
        if writer is not None:
            writer.release()
            print(f"Recorded video saved to: {record_path}")
        if renderer is not None:
            renderer.close()
        if live_slice_display is not None:
            live_slice_display.close()
        if live_slice_display is None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
