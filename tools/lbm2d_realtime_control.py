"""
Generic 2D realtime LBM + MuJoCo control demo driven by JSON config.

The config selects the MuJoCo XML/environment, LBM parameters, render/camera
settings, keyboard bindings, and action presets.

Example:
    python tools/lbm2d_realtime_control.py --config configs/realtime_2d/fish2d.json
    python tools/lbm2d_realtime_control.py --config configs/realtime_2d/fish2d.json --record outputs/fish2d_realtime.mp4

"""

import argparse
import ctypes
import json
import pathlib
import sys
import time
from typing import Any, Dict, Optional, Tuple


import cv2
import mujoco
import numpy as np
import warp as wp


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from envs.lbm import ButterflyLBMEnv, FishLBMEnv, FishObstacleLBMEnv, LBMFluidEnv, StarfishLBMEnv
from envs.lbm.lbm_core import HomeFlow
from envs.lbm.lbm_func import get_solid_boundary_img, get_u_img, get_vorticity_with_solid_img
from lbm_wave_tester_2d import get_raw_frame_2d, raw_to_rgb



class GenericLBM2DEnv(LBMFluidEnv):
    """Generic 2D LBM wrapper for arbitrary MuJoCo XML + solid_config.

    This is for realtime demos only: it runs the same MuJoCo-Warp <-> 2D LBM
    coupling as the training envs, but uses zero reward and never terminates.
    """

    def _compute_reward(self, instability_mask=None):
        return np.zeros(self.nworld, dtype=np.float32)

    def _is_terminated(self, instability_mask=None):
        return np.zeros(self.nworld, dtype=bool)



ENV_CLASSES = {
    "GenericLBM2DEnv": GenericLBM2DEnv,
    "FishLBMEnv": FishLBMEnv,
    "FishObstacleLBMEnv": FishObstacleLBMEnv,
    "ButterflyLBMEnv": ButterflyLBMEnv,
    "StarfishLBMEnv": StarfishLBMEnv,
}


SPECIAL_KEYS = {
    "space": ord(" "),
    "esc": 27,
    "escape": 27,
}


def resolve_path(path_value: Optional[str], base_dir: pathlib.Path) -> Optional[str]:
    if not path_value:
        return None
    p = pathlib.Path(path_value)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def parse_vec3(value: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = [float(v.strip()) for v in value.split(",")]
    else:
        parts = [float(v) for v in value]
    if len(parts) != 3:
        raise ValueError("Expected a 3D vector")
    return tuple(parts)  # type: ignore[return-value]


def key_to_code(key: str) -> int:
    key = key.strip()
    lower = key.lower()
    if lower in SPECIAL_KEYS:
        return SPECIAL_KEYS[lower]
    if len(key) == 1:
        return ord(key.lower())
    raise ValueError(f"Unsupported key name: {key}")


def build_keymap(controls_cfg: Dict[str, str]) -> Dict[int, str]:
    keymap: Dict[int, str] = {}
    for key, mode in controls_cfg.items():
        code = key_to_code(key)
        keymap[code] = mode
        if len(key) == 1 and key.isalpha():
            keymap[ord(key.upper())] = mode
    return keymap


def controls_help(controls_cfg: Dict[str, str]) -> str:
    return " | ".join(f"{k.upper()} {v}" for k, v in controls_cfg.items())


def preset_action(
    step: int,
    dt: float,
    preset: Dict[str, Any],
    action_dim: int,
    warmup_steps: int,
    ctrl_range: Optional[np.ndarray] = None,
) -> np.ndarray:

    """Generate action with shape (1, action_dim) from a JSON preset."""
    preset_type = preset.get("type", "sine")
    ramp = min(1.0, (step + 1) / max(1, warmup_steps))

    if preset_type == "constant":
        values = np.array(preset.get("values", [0.0] * action_dim), dtype=np.float32)
        if values.size != action_dim:
            raise ValueError(f"constant preset has {values.size} values, expected {action_dim}")
        return np.clip(values.reshape(1, action_dim) * ramp, -1.0, 1.0)

    if preset_type == "eel_wave":
        if action_dim % 2 != 0:
            raise ValueError("eel_wave expects yaw/roll actuator pairs")
        n_pairs = action_dim // 2
        t = step * dt
        amp = float(preset.get("A", 0.8))
        omega_n = float(preset.get("omega", preset.get("freq", -0.5)))
        k_n = float(preset.get("k_wave", 0.5))
        head_bias = float(preset.get("head_bias", 0.0))
        roll_cmd = float(preset.get("roll", 0.0))
        omega_max = float(preset.get("omega_max", 2.0 * np.pi))
        k_max = float(preset.get("k_max", 1.5))
        head_amp = float(preset.get("head_amp", 0.05))

        values = np.zeros(action_dim, dtype=np.float32)
        for i in range(n_pairs):
            s = 0.0 if n_pairs <= 1 else i / (n_pairs - 1)
            envelope = head_amp + (1.0 - head_amp) * s
            phase = omega_n * omega_max * t + k_n * k_max * np.pi * s
            theta_norm = amp * envelope * np.sin(phase) + head_bias * (1.0 - s)
            theta_norm = float(np.clip(theta_norm, -1.0, 1.0))
            roll_norm = float(np.clip(roll_cmd, -1.0, 1.0))

            yaw_idx = 2 * i
            roll_idx = yaw_idx + 1
            if ctrl_range is not None and ctrl_range.shape[0] >= action_dim:
                yaw_lo, yaw_hi = ctrl_range[yaw_idx]
                roll_lo, roll_hi = ctrl_range[roll_idx]
                values[yaw_idx] = yaw_lo + (theta_norm + 1.0) * 0.5 * (yaw_hi - yaw_lo)
                values[roll_idx] = roll_lo + (roll_norm + 1.0) * 0.5 * (roll_hi - roll_lo)
            else:
                values[yaw_idx] = theta_norm
                values[roll_idx] = roll_norm
        return values.reshape(1, action_dim).astype(np.float32) * ramp


    if preset_type != "sine":
        raise ValueError(f"Unsupported preset type: {preset_type}")

    components = preset.get("components")

    if components is None:
        # Backward-compatible compact fish form: amp/freq/phase_lag/bias1/bias2/tail_ratio.
        if action_dim != 2:
            raise ValueError("compact sine preset only supports 2 actuators; use components for generic models")
        amp = float(preset.get("amp", 0.0))
        freq = float(preset.get("freq", 1.0))
        phase_lag = float(preset.get("phase_lag", 0.0))
        components = [
            {"amp": amp, "freq": freq, "phase": 0.0, "bias": float(preset.get("bias1", 0.0))},
            {
                "amp": amp * float(preset.get("tail_ratio", 1.0)),
                "freq": freq,
                "phase": phase_lag,
                "bias": float(preset.get("bias2", 0.0)),
            },
        ]

    if len(components) != action_dim:
        raise ValueError(f"sine preset has {len(components)} components, expected {action_dim}")

    t = step * dt
    values = []
    for comp in components:
        amp = float(comp.get("amp", 0.0))
        freq = float(comp.get("freq", preset.get("freq", 1.0)))
        phase = float(comp.get("phase", 0.0))
        bias = float(comp.get("bias", 0.0))
        values.append(bias + amp * np.sin(2.0 * np.pi * freq * t + phase))

    action = np.array(values, dtype=np.float32).reshape(1, action_dim) * ramp
    return np.clip(action, -1.0, 1.0)


@wp.func
def _clamp01(x: float) -> float:
    return wp.min(wp.max(x, 0.0), 1.0)


@wp.kernel
def lbm_uimg_to_rgba_kernel(
    flow: HomeFlow,
    out: wp.array3d(dtype=wp.uint8),
    render_mode: int,
    vmax: float,
):
    """Convert flow.u_img on GPU to an RGBA OpenGL PBO.

    render_mode: 0 velocity, 1 vorticity, 2 solid boundary.
    """
    x, y = wp.tid()
    val = flow.u_img[x, y]
    r = float(0.0)
    g = float(0.0)
    b = float(0.0)

    if render_mode == 2:
        n = _clamp01(val)
        c = (1.0 - n) * 255.0
        r = c
        g = c
        b = c
    elif render_mode == 1:
        if val >= 999.0:
            r = 200.0
            g = 200.0
            b = 200.0
        else:
            n = _clamp01(0.5 + 0.5 * val / wp.max(vmax, 1.0e-6))
            if n < 0.5:
                t = n * 2.0
                r = 255.0 * t
                g = 255.0 * t
                b = 255.0
            else:
                t = (n - 0.5) * 2.0
                r = 255.0
                g = 255.0 * (1.0 - t)
                b = 255.0 * (1.0 - t)
    else:
        n = _clamp01(val / wp.max(vmax, 1.0e-6))
        r = 255.0 * n
        g = 180.0 * n * n
        b = 60.0 * (1.0 - n)

    out[y, x, 0] = wp.uint8(r)
    out[y, x, 1] = wp.uint8(g)
    out[y, x, 2] = wp.uint8(b)
    out[y, x, 3] = wp.uint8(255)


class MuJoCoOpenGLRenderer:
    """Small renderer for the CPU MjData mirror of the Warp state."""


    def __init__(
        self,
        model: mujoco.MjModel,
        width: int,
        height: int,
        camera_distance: float,
        camera_azimuth: float,
        camera_elevation: float,
        camera_lookat: Tuple[float, float, float],
        follow_body_id: Optional[int] = None,
    ):
        self.model = model
        try:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        except Exception:
            pass
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = camera_distance
        self.camera.azimuth = camera_azimuth
        self.camera.elevation = camera_elevation
        self.camera.lookat[:] = camera_lookat
        self.follow_body_id = follow_body_id
        self.scene_option = mujoco.MjvOption()
        self.scene_option.frame = mujoco.mjtFrame.mjFRAME_NONE

    def render(self, data: mujoco.MjData) -> np.ndarray:
        if self.follow_body_id is not None:
            self.camera.lookat[:] = data.subtree_com[self.follow_body_id]
        self.renderer.update_scene(data, self.camera, self.scene_option)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == height:
        return frame
    width = max(1, int(round(w * height / h)))
    interp = cv2.INTER_AREA if h > height else cv2.INTER_CUBIC
    return cv2.resize(frame, (width, height), interpolation=interp)


def apply_mujoco_background(frame: np.ndarray, color: Optional[Tuple[int, int, int]], threshold: int) -> np.ndarray:
    """Replace empty black OpenGL background with a configurable RGB color."""
    if color is None:
        return frame
    out = frame.copy()
    mask = np.all(out <= threshold, axis=2)
    out[mask] = np.array(color, dtype=np.uint8)
    return out


def compute_lbm_vmax(raw: np.ndarray, render_type: str, previous: Optional[float], scale: float) -> float:
    if render_type == "solid_boundary":
        return 1.0

    if render_type == "vorticity":
        mask = raw < 999.0
        current = float(np.max(np.abs(raw[mask]))) * scale + 1e-8 if np.any(mask) else 1.0
    else:
        current = float(np.max(raw)) * scale + 1e-8
    if previous is None:
        return max(current, 1e-6)
    return max(previous * 0.96, current, 1e-6)


def draw_panel_overlay(
    frame: np.ndarray,
    mode: str,
    step: int,
    mode_step: int,
    action: np.ndarray,
    dx: float,
    dy: float,
    reward: float,
    fps: float,
    paused: bool,
    controls_line: str,
) -> np.ndarray:
    out = frame.copy()
    action_str = ", ".join(f"{v:+.2f}" for v in action[0])
    lines = [
        f"mode: {mode} {'[PAUSED]' if paused else ''}",
        f"step: {step}  mode_step: {mode_step}",
        f"action: [{action_str}]",
        f"dx: {dx:+.2f}  dy: {dy:+.2f}  reward: {reward:+.4f}",
        f"fps: {fps:.1f}",
        controls_line,
        "Space pause | R reset | Q/Esc quit",
    ]
    x, y0 = 12, 24
    for i, text in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 1, cv2.LINE_AA)
    return out


def sync_warp_to_cpu_mjdata(env: Any, mj_data: mujoco.MjData, world_idx: int = 0) -> None:
    mj_data.qpos[:] = env.data.qpos.numpy()[world_idx]
    mj_data.qvel[:] = env.data.qvel.numpy()[world_idx]
    mujoco.mj_forward(env.mujoco_model, mj_data)


def normalize_action_for_panel(action: np.ndarray, ctrl_range: Optional[np.ndarray]) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if ctrl_range is None or ctrl_range.shape[0] < values.size:
        return np.clip(values, -1.0, 1.0)

    ranges = np.asarray(ctrl_range[: values.size], dtype=np.float32)
    lo = ranges[:, 0]
    hi = ranges[:, 1]
    span = np.maximum(hi - lo, 1.0e-6)
    if np.nanmax(np.abs(values)) <= 1.05:
        return np.clip(values, -1.0, 1.0)
    normalized = 2.0 * (values - lo) / span - 1.0
    return np.clip(normalized, -1.0, 1.0)



def draw_control_signal_panel(
    width: int,
    height: int,
    mode: str,
    step: int,
    mode_step: int,
    action: np.ndarray,
    ctrl_range: Optional[np.ndarray],
    fps: float,
    paused: bool,
    controls_line: str,
    action_gain: float,
    gain_step: float,
) -> np.ndarray:
    panel = np.full((height, width, 3), (18, 20, 24), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (52, 58, 68), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.54 if width < 360 else 0.68
    info_scale = 0.42 if width < 360 else 0.50
    title = f"Control  mode={mode} {'[PAUSED]' if paused else ''}"
    cv2.putText(panel, title, (12, 30), font, title_scale, (235, 238, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, f"step={step}  gain={action_gain:.2f}  fps={fps:.1f}", (12, 56), font, info_scale, (180, 190, 205), 1, cv2.LINE_AA)
    cv2.putText(panel, f"+/- gain step={gain_step:.2f}", (12, 78), font, info_scale, (150, 165, 185), 1, cv2.LINE_AA)


    norm_values = normalize_action_for_panel(action, ctrl_range)
    raw_values = np.asarray(action, dtype=np.float32).reshape(-1)
    n = int(norm_values.size)
    if n == 0:
        cv2.putText(panel, "No action", (18, 100), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        return panel

    top = 104
    bottom_reserved = 74
    available_h = max(80, height - top - bottom_reserved)
    row_h = max(26, min(52, available_h // max(1, n)))
    bar_x0 = 58 if width < 360 else 92
    bar_x1 = width - (52 if width < 360 else 72)

    bar_w = max(60, bar_x1 - bar_x0)
    center_x = bar_x0 + bar_w // 2

    for i, (norm, raw) in enumerate(zip(norm_values, raw_values)):
        y = top + i * row_h
        if y + row_h > height - bottom_reserved + 12:
            break
        label = f"u{i}"
        cv2.putText(panel, label, (12, y + 18), font, 0.46 if width < 360 else 0.52, (210, 215, 225), 1, cv2.LINE_AA)

        cv2.rectangle(panel, (bar_x0, y + 5), (bar_x1, y + 22), (54, 60, 70), -1)
        cv2.line(panel, (center_x, y + 2), (center_x, y + 25), (130, 140, 155), 1, cv2.LINE_AA)
        end_x = int(round(center_x + float(norm) * (bar_w * 0.5)))
        color = (75, 190, 255) if norm >= 0 else (255, 150, 90)
        cv2.rectangle(panel, (min(center_x, end_x), y + 7), (max(center_x, end_x), y + 20), color, -1)
        cv2.rectangle(panel, (bar_x0, y + 5), (bar_x1, y + 22), (100, 108, 122), 1)
        cv2.circle(panel, (end_x, y + 14), 5, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.putText(panel, f"{raw:+.2f}", (bar_x1 + 6, y + 19), font, 0.38 if width < 360 else 0.45, (215, 220, 230), 1, cv2.LINE_AA)

    footer_scale = 0.36 if width < 360 else 0.45
    cv2.putText(panel, controls_line, (12, height - 42), font, footer_scale, (180, 190, 205), 1, cv2.LINE_AA)
    cv2.putText(panel, "Space/R reset/Q quit", (12, height - 18), font, footer_scale, (180, 190, 205), 1, cv2.LINE_AA)

    return panel


def make_combined_frame(lbm_frame: np.ndarray, right_panel: np.ndarray, output_height: int) -> np.ndarray:
    lbm_resized = resize_to_height(lbm_frame, output_height)
    panel_resized = resize_to_height(right_panel, output_height)
    sep = np.full((output_height, 4, 3), 32, dtype=np.uint8)
    return np.concatenate([lbm_resized, sep, panel_resized], axis=1)



class OpenGLInteropDisplay:
    """GLFW display with CUDA/OpenGL interop for the LBM panel.

    Left panel: Warp/CUDA writes LBM RGBA directly into an OpenGL PBO.
    Right panel: CPU-generated control signal panel is uploaded to an OpenGL texture.

    """

    def __init__(self, lbm_width: int, lbm_height: int, mj_width: int, mj_height: int, output_height: int, title: str, debug: bool = False):
        self.debug = debug
        self._dbg("init: importing glfw/OpenGL/CUDA")
        try:

            import glfw
            from OpenGL import GL
            from cuda.bindings import driver as cu
        except Exception as exc:
            raise RuntimeError("OpenGL backend requires glfw, PyOpenGL, and cuda-python") from exc

        self.glfw = glfw
        self.GL = GL
        self.cu = cu
        self.lbm_width = int(lbm_width)
        self.lbm_height = int(lbm_height)
        self.mj_width = int(mj_width)
        self.mj_height = int(mj_height)
        self.output_height = int(output_height)
        self.left_width = max(1, int(round(self.lbm_width * self.output_height / self.lbm_height)))
        self.right_width = max(1, int(round(self.mj_width * self.output_height / self.mj_height)))
        self.separator = 4
        self.window_width = self.left_width + self.separator + self.right_width

        self._dbg("init: glfw.init")
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        self._dbg(f"init: create window {self.window_width}x{self.output_height}")
        self.window = glfw.create_window(self.window_width, self.output_height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        self._dbg("init: create textures")
        self.lbm_tex = self._create_texture(self.lbm_width, self.lbm_height, GL.GL_RGBA)
        self.mj_tex = self._create_texture(self.mj_width, self.mj_height, GL.GL_RGB)
        self._dbg("init: create PBO")
        self.lbm_pbo = GL.glGenBuffers(1)

        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.lbm_pbo)
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self.lbm_width * self.lbm_height * 4, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        # Staging buffer: Warp writes normal CUDA memory first, then CUDA driver
        # copies device-to-device into the mapped OpenGL PBO. This is more robust
        # than launching a Warp kernel directly on the mapped GL pointer.
        self._dbg("init: allocate Warp RGBA staging buffer")
        self.lbm_rgba = wp.empty((self.lbm_height, self.lbm_width, 4), dtype=wp.uint8, device="cuda:0")

        self._dbg("init: cuInit")
        self._cu_check(cu.cuInit(0))

        self._dbg("init: retain/set CUDA primary context")
        err, dev = cu.cuDeviceGet(0)
        self._cu_check(err)
        err, ctx = cu.cuDevicePrimaryCtxRetain(dev)
        self._cu_check(err)
        self._cu_check(cu.cuCtxSetCurrent(ctx))
        self._dbg("init: register GL PBO with CUDA")
        err, resource = cu.cuGraphicsGLRegisterBuffer(

            int(self.lbm_pbo), cu.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD
        )
        self._cu_check(err)
        self.cuda_resource = resource

        self.prev_keys: Dict[int, bool] = {}
        self._dbg("init: done")

    def _dbg(self, message: str) -> None:
        if getattr(self, "debug", False):
            print(f"[opengl {time.perf_counter():.6f}] {message}", flush=True)

    def _cu_check(self, result):

        err = result[0] if isinstance(result, tuple) else result
        if err != self.cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {err}")

    def _create_texture(self, width: int, height: int, fmt: int) -> int:
        GL = self.GL
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        internal = GL.GL_RGBA8 if fmt == GL.GL_RGBA else GL.GL_RGB8
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, width, height, 0, fmt, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return int(tex)

    def update_lbm_texture(self, flow: HomeFlow, render_type: str, vmax: float) -> None:
        self._dbg("lbm: begin")
        # MuJoCo's offscreen renderer may switch the current OpenGL context.
        # Always restore our GLFW context before touching our PBO/texture.
        self._dbg("lbm: make GLFW context current")
        self.glfw.make_context_current(self.window)
        GL = self.GL
        cu = self.cu

        self._dbg(f"lbm: launch scalar field kernel render_type={render_type}")
        if render_type == "vorticity":
            wp.launch(get_vorticity_with_solid_img, dim=(flow.nx, flow.ny), inputs=[flow, 1.0])
            mode = 1
        elif render_type == "solid_boundary":
            wp.launch(get_solid_boundary_img, dim=(flow.nx, flow.ny), inputs=[flow, 1.0])
            mode = 2
        else:
            wp.launch(get_u_img, dim=(flow.nx, flow.ny), inputs=[flow])
            mode = 0
        self._dbg("lbm: synchronize scalar field kernel")
        wp.synchronize()

        self._dbg("lbm: launch RGBA staging kernel")
        wp.launch(lbm_uimg_to_rgba_kernel, dim=(flow.nx, flow.ny), inputs=[flow, self.lbm_rgba, mode, float(vmax)])
        self._dbg("lbm: synchronize RGBA staging kernel")
        wp.synchronize()

        # Ensure pending GL use of the PBO is complete before CUDA maps it.
        self._dbg("lbm: glFinish before CUDA map")
        GL.glFinish()
        self._dbg("lbm: cuGraphicsMapResources")
        self._cu_check(cu.cuGraphicsMapResources(1, self.cuda_resource, 0))
        try:
            self._dbg("lbm: cuGraphicsResourceGetMappedPointer")
            err, ptr, size = cu.cuGraphicsResourceGetMappedPointer(self.cuda_resource)
            self._cu_check(err)
            byte_count = self.lbm_width * self.lbm_height * 4
            if int(size) < byte_count:
                raise RuntimeError("Mapped PBO is smaller than expected")
            self._dbg(f"lbm: cuMemcpyDtoD byte_count={byte_count}")
            self._cu_check(cu.cuMemcpyDtoD(int(ptr), int(self.lbm_rgba.ptr), byte_count))
            self._dbg("lbm: cuMemcpyDtoD done")
        finally:
            self._dbg("lbm: cuGraphicsUnmapResources")
            self._cu_check(cu.cuGraphicsUnmapResources(1, self.cuda_resource, 0))

        self._dbg("lbm: upload PBO to texture")
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.lbm_tex)


        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, int(self.lbm_pbo))
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        # With a PBO bound, the last argument is a byte offset into the PBO.
        # PyOpenGL's None path can be ambiguous on Windows, so pass c_void_p(0).
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.lbm_width,
            self.lbm_height,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._dbg("lbm: end")

    def update_mujoco_texture(self, frame: np.ndarray) -> None:
        self._dbg("mujoco texture: begin")
        self.glfw.make_context_current(self.window)

        GL = self.GL
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.mj_tex)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.mj_width, self.mj_height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, frame)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        self._dbg("mujoco texture: end")

    def draw(self, title: str) -> None:
        self._dbg("draw: begin")
        glfw = self.glfw

        glfw.make_context_current(self.window)
        GL = self.GL

        glfw.set_window_title(self.window, title)
        GL.glViewport(0, 0, self.window_width, self.output_height)
        GL.glClearColor(0.02, 0.02, 0.02, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, self.window_width, 0, self.output_height, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glEnable(GL.GL_TEXTURE_2D)
        self._draw_texture(self.lbm_tex, 0, 0, self.left_width, self.output_height)
        self._draw_texture(self.mj_tex, self.left_width + self.separator, 0, self.right_width, self.output_height)
        GL.glDisable(GL.GL_TEXTURE_2D)
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        self._dbg("draw: end")

    def _draw_texture(self, tex: int, x: int, y: int, w: int, h: int) -> None:

        GL = self.GL
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 0.0); GL.glVertex2f(x, y)
        GL.glTexCoord2f(1.0, 0.0); GL.glVertex2f(x + w, y)
        GL.glTexCoord2f(1.0, 1.0); GL.glVertex2f(x + w, y + h)
        GL.glTexCoord2f(0.0, 1.0); GL.glVertex2f(x, y + h)
        GL.glEnd()
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def should_close(self) -> bool:
        return bool(self.glfw.window_should_close(self.window))

    def key_once(self, key_code: int) -> bool:
        pressed = self.glfw.get_key(self.window, key_code) == self.glfw.PRESS
        prev = self.prev_keys.get(key_code, False)
        self.prev_keys[key_code] = pressed
        return pressed and not prev

    def close(self) -> None:
        try:
            self._cu_check(self.cu.cuGraphicsUnregisterResource(self.cuda_resource))
        except Exception:
            pass
        try:
            self.glfw.destroy_window(self.window)
            self.glfw.terminate()
        except Exception:
            pass


def instantiate_env(config: Dict[str, Any], config_dir: pathlib.Path, cli_args: argparse.Namespace) -> Any:

    env_cfg = config.get("env", {})
    lbm_cfg = config.get("lbm", {})
    env_class_name = env_cfg.get("class", "FishLBMEnv")
    if env_class_name not in ENV_CLASSES:
        raise ValueError(f"Unsupported env class '{env_class_name}'. Choices: {list(ENV_CLASSES)}")

    kwargs = dict(env_cfg.get("kwargs", {}))
    kwargs.update(
        xml_path=resolve_path(env_cfg.get("xml_path"), PROJECT_ROOT),

        solid_config=env_cfg.get("solid_config"),
        nx=cli_args.nx if cli_args.nx is not None else int(lbm_cfg.get("nx", 400)),
        ny=cli_args.ny if cli_args.ny is not None else int(lbm_cfg.get("ny", 600)),
        lbm_scale=cli_args.lbm_scale if cli_args.lbm_scale is not None else float(lbm_cfg.get("lbm_scale", 0.2)),
        nworld=1,
        max_episode_steps=10_000_000,
        per_frame_steps=cli_args.per_frame_steps if cli_args.per_frame_steps is not None else int(lbm_cfg.get("per_frame_steps", 10)),
        render_mode=None,
    )
    # Let env defaults choose XML/solid config if omitted.
    if kwargs["xml_path"] is None:
        kwargs.pop("xml_path")
    if kwargs["solid_config"] is None:
        kwargs.pop("solid_config")

    return ENV_CLASSES[env_class_name](**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="JSON-driven 2D realtime LBM + MuJoCo OpenGL control demo")
    parser.add_argument("--config", type=str, default="configs/realtime_2d/fish2d.json")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--lbm-scale", type=float, default=None)
    parser.add_argument("--per-frame-steps", type=int, default=None)
    parser.add_argument("--render-type", type=str, default=None, choices=["velocity", "vorticity", "solid_boundary"])
    parser.add_argument("--render-backend", type=str, default=None, choices=["opencv", "opengl"], help="Display backend; opengl uses CUDA/OpenGL interop for the LBM panel")

    parser.add_argument("--control-dt", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--transition-steps", type=int, default=None, help="Smooth blending steps when switching presets")
    parser.add_argument("--start-mode", type=str, default=None)
    parser.add_argument("--action-gain", type=float, default=None, help="Initial multiplier applied to preset actions")
    parser.add_argument("--gain-step", type=float, default=None, help="Keyboard +/- adjustment step for action gain")


    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--record-fps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Load config and create env, then exit before opening the window")
    parser.add_argument("--debug-render", action="store_true", help="Print detailed OpenGL/CUDA backend progress with flush")
    args = parser.parse_args()



    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config_dir = config_path.parent

    render_cfg = config.get("render", {})
    camera_cfg = config.get("camera", {})
    control_cfg = config.get("control", {})
    controls_cfg = config.get("controls", {})
    presets = config.get("presets", {})
    if not presets:
        raise ValueError("Config must contain a non-empty 'presets' object")
    if not controls_cfg:
        raise ValueError("Config must contain keyboard 'controls'")

    env = instantiate_env(config, config_dir, args)
    env.reset()
    action_dim = env.action_space.shape[1]

    if args.dry_run:
        print(f"Loaded config: {config_path}")
        print(f"Env: {env.__class__.__name__}, action_dim={action_dim}, presets={list(presets.keys())}")
        return

    render_type = args.render_type or render_cfg.get("type", "vorticity")
    render_backend = args.render_backend or render_cfg.get("backend", "opencv")

    output_height = int(render_cfg.get("output_height", 720))
    control_panel_width = int(render_cfg.get("control_panel_width", max(220, int(output_height * 0.375))))
    window_name = render_cfg.get("window_name", "2D Realtime Control")
    vmax_scale = float(render_cfg.get("vmax_scale", 0.2))

    control_dt = args.control_dt if args.control_dt is not None else float(control_cfg.get("dt", 0.01))
    action_gain = args.action_gain if args.action_gain is not None else float(control_cfg.get("action_gain", 1.0))
    gain_step = args.gain_step if args.gain_step is not None else float(control_cfg.get("gain_step", 0.1))

    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(control_cfg.get("warmup_steps", 20))

    transition_steps = (
        args.transition_steps
        if args.transition_steps is not None
        else int(control_cfg.get("transition_steps", max(20, warmup_steps)))
    )
    start_mode = args.start_mode or control_cfg.get("start_mode", "idle")

    if start_mode not in presets:
        raise ValueError(f"start_mode '{start_mode}' not found in presets")

    keymap = build_keymap(controls_cfg)
    controls_line = controls_help(controls_cfg)
    ctrl_range = env.mujoco_model.actuator_ctrlrange.copy()

    gl_display = None

    if render_backend == "opengl":
        if args.record is not None:
            print("[warn] --record is disabled for --render-backend opengl (would require glReadPixels).")
            args.record = None
        flow0 = env.solver.flows[0]
        gl_display = OpenGLInteropDisplay(
            flow0.nx,
            flow0.ny,
            control_panel_width,
            output_height,
            output_height,
            window_name,
            debug=args.debug_render,
        )


    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    writer = None

    record_path = pathlib.Path(args.record) if args.record else None
    record_fps = args.record_fps if args.record_fps is not None else int(render_cfg.get("record_fps", 30))
    if record_path:
        record_path.parent.mkdir(parents=True, exist_ok=True)

    mode = start_mode
    step = 0
    mode_step = 0
    paused = False
    lbm_vmax: Optional[float] = None
    initial_head_pos = env.solver.flows[0].solid_position.numpy()[0].copy()
    last_time = time.time()
    fps = 0.0
    last_reward = 0.0
    last_action = np.zeros((1, action_dim), dtype=np.float32)
    transition_from = last_action.copy()
    transition_step = transition_steps

    print(f"Loaded config: {config_path}")

    print(f"Controls: {controls_line} | +/- gain | Space pause | R reset | Q/Esc quit")


    try:
        while True:
            now = time.time()
            dt_wall = now - last_time
            last_time = now
            if dt_wall > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt_wall) if fps > 0 else 1.0 / dt_wall

            if not paused:
                in_transition = transition_step < transition_steps
                target_warmup = 1 if in_transition else warmup_steps
                target_action = preset_action(mode_step, control_dt, presets[mode], action_dim, target_warmup, ctrl_range)

                if in_transition:
                    u = (transition_step + 1) / max(1, transition_steps)
                    alpha = u * u * (3.0 - 2.0 * u)  # smoothstep
                    action = (1.0 - alpha) * transition_from + alpha * target_action
                    transition_step += 1
                else:
                    action = target_action
                action = np.clip((action * action_gain).astype(np.float32), -1.0, 1.0)

                if args.debug_render:
                    print(f"[loop {time.perf_counter():.6f}] env.step begin step={step}", flush=True)
                _obs, reward, _done, _info = env.step(action)
                if args.debug_render:
                    print(f"[loop {time.perf_counter():.6f}] env.step end step={step}", flush=True)
                last_reward = float(reward[0])

                last_action = action
                step += 1
                mode_step += 1



            head_pos = env.solver.flows[0].solid_position.numpy()[0]

            dx = float(head_pos[0] - initial_head_pos[0])
            dy = float(head_pos[1] - initial_head_pos[1])

            control_panel = draw_control_signal_panel(
                control_panel_width,
                output_height,
                mode,
                step,
                mode_step,
                last_action,
                ctrl_range,
                fps,
                paused,
                controls_line,
                action_gain,
                gain_step,
            )


            if render_backend == "opengl":
                if gl_display is None:
                    raise RuntimeError("OpenGL backend was not initialized")
                fixed_vmax = float(render_cfg.get("opengl_lbm_vmax", render_cfg.get("lbm_vmax", 1.0)))
                gl_display.update_lbm_texture(env.solver.flows[0], render_type, fixed_vmax)
                gl_display.update_mujoco_texture(control_panel)
                title = f"{window_name} | mode={mode} step={step} dx={dx:+.2f} dy={dy:+.2f} fps={fps:.1f} {'PAUSED' if paused else ''}"
                gl_display.draw(title)
                if gl_display.should_close() or gl_display.key_once(gl_display.glfw.KEY_ESCAPE) or gl_display.key_once(ord("Q")):
                    break
                key = None
                if gl_display.key_once(ord(" ")):
                    key = ord(" ")
                elif gl_display.key_once(ord("R")):
                    key = ord("R")
                elif gl_display.key_once(ord("+")) or gl_display.key_once(ord("=")):
                    key = ord("+")
                elif gl_display.key_once(ord("-")) or gl_display.key_once(ord("_")):
                    key = ord("-")
                else:

                    for candidate in keymap:
                        if gl_display.key_once(candidate):
                            key = candidate
                            break
            else:
                raw = get_raw_frame_2d(env, render_type, world_idx=0)
                lbm_vmax = compute_lbm_vmax(raw, render_type, lbm_vmax, vmax_scale)
                lbm_frame = raw_to_rgb(raw, lbm_vmax, render_type)
                combined = make_combined_frame(lbm_frame, control_panel, output_height)


                if writer is None and record_path is not None:
                    h, w = combined.shape[:2]
                    writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"), record_fps, (w, h))
                if writer is not None:
                    writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

                cv2.imshow(window_name, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break

            if key in (ord("+"), ord("=")):
                action_gain = min(5.0, action_gain + gain_step)
            elif key in (ord("-"), ord("_")):
                action_gain = max(0.0, action_gain - gain_step)
            elif key == ord(" "):
                paused = not paused
            elif key in (ord("r"), ord("R")):

                env.reset()
                initial_head_pos = env.solver.flows[0].solid_position.numpy()[0].copy()
                step = 0
                mode_step = 0
                lbm_vmax = None
                last_reward = 0.0
                last_action = np.zeros((1, action_dim), dtype=np.float32)
                transition_from = last_action.copy()
                transition_step = transition_steps
            elif key in keymap:
                new_mode = keymap[key]
                if new_mode not in presets:
                    print(f"Ignoring key mapped to unknown preset: {new_mode}")
                elif new_mode != mode:
                    transition_from = last_action.copy() / max(abs(action_gain), 1.0e-6)
                    transition_step = 0
                    mode = new_mode
                    mode_step = 0




    finally:
        if writer is not None:
            writer.release()
            print(f"Recorded video saved to: {record_path}")
        if gl_display is not None:

            gl_display.close()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
