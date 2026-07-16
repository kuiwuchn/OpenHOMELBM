"""Minimal SAC training for LBM-RIGID.

Install if needed:
    pip install stable-baselines3 gymnasium

Run with realtime LBM rendering and right-side SAC panel:
    python train_sac_minimal.py --render --warmup-steps 50 --viscosity 0.05

Run projected 2D eel training with SAC controlling four planar parameters:
    python train_sac_minimal.py --animal eel --control-mode cpg --per-frame-steps 8 --cpg-ramp-steps 10 --cpg-hold-steps 60





Run without rendering:
    python train_sac_minimal.py
"""

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


import cv2
import gymnasium as gym
import numpy as np
import warp as wp
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.lbm.eel.eel_lbm_env import Eel2DLBMEnv
from envs.lbm.lbm_core import HomeFlow

from envs.lbm.lbm_func import get_vorticity_with_solid_img





class SingleEnvWrapper(gym.Env):
    """Convert project env API: (1, dim) batched Gym -> unbatched Gymnasium."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env,
        warmup_steps: int = 50,
        viscosity: float = 0.05,
        latest_observation_only: bool = False,
    ):
        super().__init__()
        self.env = env
        self.warmup_steps = max(0, int(warmup_steps))
        self.viscosity = float(viscosity)
        self.latest_observation_only = bool(latest_observation_only)


        act = env.action_space

        obs = env.observation_space

        self.action_space = spaces.Box(
            low=act.low[0].astype(np.float32),
            high=act.high[0].astype(np.float32),
            dtype=np.float32,
        )
        observation_dim = int(np.prod(obs.shape[1:]))
        if self.latest_observation_only:
            # Drop the duplicated pre-action frame in CPG mode.
            if observation_dim % 2 != 0:
                raise ValueError("Temporal observation must have an even dimension")
            observation_dim //= 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )

    def _single_observation(self, observation: np.ndarray) -> np.ndarray:
        result = np.asarray(observation[0], dtype=np.float32)
        if self.latest_observation_only:
            result = result[result.size // 2 :]
        return result

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs = self.env.reset(seed=seed, options=options)
        set_lbm_viscosity(self.env, self.viscosity)
        if self.warmup_steps > 0:
            # Settle the coupled solver with neutral controls.
            zero_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
            for _ in range(self.warmup_steps):
                obs, _, done, _ = self.env.step(zero_action)
                if bool(np.asarray(done).reshape(-1)[0]):
                    obs = self.env.reset(seed=seed, options=options)
                    break
            if hasattr(self.env, "current_steps"):
                self.env.current_steps[...] = 0
        return self._single_observation(obs), {}




    def step(self, action: np.ndarray):
        obs, reward, done, info = self.env.step(action.reshape(1, -1).astype(np.float32))
        info = info or {}
        terminated = bool(np.asarray(info.get("terminated", done))[0])
        truncated = bool(np.asarray(info.get("truncated", [False]))[0])
        return self._single_observation(obs), float(reward[0]), terminated, truncated, info

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


class EelCPGWrapper(gym.Env):
    """Drive the projected eel with four planar parameters from eel2d.json.

    SAC action dimensions are, in order: A, omega, k_wave and head_bias. Roll is
    fixed at zero because it has no useful degree of freedom in the 2D task.
    The generated actuator controls intentionally match preset_action() in
    tools/lbm2d_realtime_control.py, including its .01-second wave clock and
    per-actuator position-control ranges.
    """

    metadata = {"render_modes": []}

    PARAMETER_NAMES = ("A", "omega", "k_wave", "head_bias")
    PARAMETER_LOW = np.array([0.10, -1.0, 0.30, -0.30], dtype=np.float32)
    PARAMETER_HIGH = np.array([0.60, -0.3, 0.90, 0.30], dtype=np.float32)
    DEFAULT_PARAMETERS = np.array([0.36, -1.0, 0.65, 0.0], dtype=np.float32)
    OMEGA_MAX = 5.0 * np.pi
    K_MAX = 1.5
    HEAD_AMPLITUDE = 0.05
    WAVE_DT = 0.01

    def __init__(
        self,
        env: SingleEnvWrapper,
        smoothing: float = 1.0,
        parameter_ramp_steps: int = 10,
        parameter_hold_steps: int = 60,
    ):
        super().__init__()
        self.env = env
        self.raw_env = env.env
        # Retain the legacy argument for config compatibility.
        self.smoothing = 1.0
        self.legacy_smoothing_argument = float(smoothing)
        self.parameter_ramp_steps = max(0, int(parameter_ramp_steps))
        self.parameter_hold_steps = max(1, int(parameter_hold_steps))
        self.n_actuators = int(np.prod(env.action_space.shape))
        if self.n_actuators < 2 or self.n_actuators % 2 != 0:
            raise ValueError("Projected eel CPG requires yaw/roll actuator pairs")
        self.n_pairs = self.n_actuators // 2
        ctrl_range = np.asarray(self.raw_env.mujoco_model.actuator_ctrlrange, dtype=np.float32)
        if ctrl_range.shape != (self.n_actuators, 2):
            raise ValueError("Unexpected projected eel actuator control ranges")
        self.ctrl_low = ctrl_range[:, 0]
        self.ctrl_high = ctrl_range[:, 1]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        base_obs_dim = int(np.prod(env.observation_space.shape))
        # Append phase and the four active CPG parameters.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_obs_dim + 6,),
            dtype=np.float32,
        )

        mujoco_dt = float(self.raw_env.mujoco_model.opt.timestep)
        self.control_dt = mujoco_dt * int(self.raw_env.per_frame_steps)
        self.phase = 0.0
        self.current_normalized = self._parameters_to_normalized(self.DEFAULT_PARAMETERS)
        self.last_joint_action = np.zeros(self.n_actuators, dtype=np.float32)
        self.last_target_normalized = self.current_normalized.copy()
        self.last_executed_steps = 0
        self.render_hook: Optional[Callable[[float], bool]] = None
        self.render_substep_every = 1
        self.render_stop_requested = False

    @property
    def parameter_decision_steps(self) -> int:
        return self.parameter_ramp_steps + self.parameter_hold_steps

    def parameter_stage_text(self) -> str:
        if self.last_executed_steps <= self.parameter_ramp_steps and self.parameter_ramp_steps > 0:
            return f"RAMP {self.last_executed_steps}/{self.parameter_ramp_steps}"
        hold_step = max(0, self.last_executed_steps - self.parameter_ramp_steps)
        return f"HOLD {hold_step}/{self.parameter_hold_steps}"

    def set_render_hook(
        self,
        hook: Optional[Callable[[float], bool]],
        every: int = 1,
    ) -> None:
        """Render inside a held parameter decision without changing SAC semantics."""
        self.render_hook = hook
        self.render_substep_every = max(1, int(every))

    @classmethod
    def _normalized_to_parameters(cls, action: np.ndarray) -> np.ndarray:
        unit = 0.5 * (np.clip(action, -1.0, 1.0) + 1.0)
        return cls.PARAMETER_LOW + unit * (cls.PARAMETER_HIGH - cls.PARAMETER_LOW)

    @classmethod
    def _parameters_to_normalized(cls, parameters: np.ndarray) -> np.ndarray:
        unit = (parameters - cls.PARAMETER_LOW) / (cls.PARAMETER_HIGH - cls.PARAMETER_LOW)
        return np.clip(2.0 * unit - 1.0, -1.0, 1.0).astype(np.float32)

    def _physical_parameters(self) -> np.ndarray:
        return self._normalized_to_parameters(self.current_normalized)

    def _joint_wave(self, parameters: np.ndarray) -> np.ndarray:
        # Generate yaw waves while holding roll at zero.
        amplitude, _omega, k_wave, head_bias = parameters
        s = np.linspace(0.0, 1.0, self.n_pairs, dtype=np.float32)
        envelope = self.HEAD_AMPLITUDE + (1.0 - self.HEAD_AMPLITUDE) * s
        yaw_normalized = amplitude * envelope * np.sin(
            self.phase + k_wave * self.K_MAX * np.pi * s
        )
        yaw_normalized += head_bias * (1.0 - s)
        yaw_normalized = np.clip(yaw_normalized, -1.0, 1.0)
        roll_normalized = np.zeros(self.n_pairs, dtype=np.float32)

        normalized = np.empty(self.n_actuators, dtype=np.float32)
        normalized[0::2] = yaw_normalized
        normalized[1::2] = roll_normalized
        # Map normalized commands to physical actuator ranges.
        return (
            self.ctrl_low + 0.5 * (normalized + 1.0) * (self.ctrl_high - self.ctrl_low)
        ).astype(np.float32)

    def _augment_observation(self, observation: np.ndarray) -> np.ndarray:
        # Make oscillator state observable to SAC.
        cpg_state = np.concatenate(
            [
                np.array([np.sin(self.phase), np.cos(self.phase)], dtype=np.float32),
                self.current_normalized,
            ]
        )
        return np.concatenate([np.asarray(observation, dtype=np.float32), cpg_state]).astype(np.float32)

    def _info(self) -> Dict[str, Any]:
        parameters = self._physical_parameters()
        return {
            "cpg_parameters": {
                name: float(value) for name, value in zip(self.PARAMETER_NAMES, parameters)
            },
            "cpg_joint_action": self.last_joint_action.copy(),
            "cpg_parameter_ramp_steps": self.parameter_ramp_steps,
            "cpg_parameter_hold_steps": self.parameter_hold_steps,
            "cpg_parameter_decision_steps": self.parameter_decision_steps,
            "cpg_executed_steps": self.last_executed_steps,
        }

    def actual_body_points(self) -> Optional[np.ndarray]:
        """Return coupled LBM solid centers in world x/y coordinates."""
        try:
            points = self.raw_env.solver.flows[0].solid_position.numpy()
            points = np.asarray(points[: self.raw_env.solid_num, :2], dtype=np.float32)
            if points.shape[0] < 2 or not np.all(np.isfinite(points)):
                return None
            return points
        except Exception:
            return None

    def actual_body_polygons(self) -> Optional[list[np.ndarray]]:
        """Return the exact transformed LBM polygons used for coupling."""
        try:
            flow = self.raw_env.solver.flows[0]
            vertices = np.asarray(flow.solid_line_transformed.numpy(), dtype=np.float32)
            counts = np.asarray(flow.solid_line_num.numpy(), dtype=np.int32)
            polygons = [vertices[i, : int(count)].copy() for i, count in enumerate(counts)]
            polygons = [p for p in polygons if len(p) >= 3 and np.all(np.isfinite(p))]
            return polygons or None
        except Exception:
            return None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self.phase = 0.0
        self.current_normalized = self._parameters_to_normalized(self.DEFAULT_PARAMETERS)
        self.last_target_normalized = self.current_normalized.copy()
        self.last_joint_action.fill(0.0)
        self.last_executed_steps = 0
        info = dict(info)
        info.update(self._info())
        return self._augment_observation(observation), info

    def step(self, action: np.ndarray):
        target = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.last_target_normalized = target.copy()
        ramp_start = self.current_normalized.copy()
        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        observation = None
        self.last_executed_steps = 0

        # One SAC transition contains a parameter ramp and hold.
        for step_index in range(self.parameter_decision_steps):
            if step_index < self.parameter_ramp_steps:
                fraction = float(step_index + 1) / float(self.parameter_ramp_steps)
                self.current_normalized = (
                    ramp_start + fraction * (target - ramp_start)
                ).astype(np.float32)
            else:
                self.current_normalized = target.copy()

            parameters = self._physical_parameters()
            omega = float(parameters[1])
            self.last_joint_action = self._joint_wave(parameters)
            observation, reward, terminated, truncated, info = self.env.step(self.last_joint_action)
            total_reward += float(reward)
            self.last_executed_steps += 1
            self.phase = float(
                (self.phase + omega * self.OMEGA_MAX * self.WAVE_DT)
                % (2.0 * np.pi)
            )
            if (
                self.render_hook is not None
                and self.last_executed_steps % self.render_substep_every == 0
            ):
                if not self.render_hook(float(reward)):
                    self.render_stop_requested = True
                    break
            if terminated or truncated:
                break

        if observation is None:
            raise RuntimeError("CPG parameter hold executed zero control steps")
        info = dict(info)
        info.update(self._info())
        return self._augment_observation(observation), total_reward, terminated, truncated, info

    def config_dict(self) -> Dict[str, Any]:
        return {
            "parameter_names": list(self.PARAMETER_NAMES),
            "parameter_low": self.PARAMETER_LOW.tolist(),
            "parameter_high": self.PARAMETER_HIGH.tolist(),
            "default_parameters": self.DEFAULT_PARAMETERS.tolist(),
            "smoothing": self.smoothing,
            "control_dt": self.control_dt,
            "parameter_ramp_steps": self.parameter_ramp_steps,
            "parameter_hold_steps": self.parameter_hold_steps,
            "parameter_decision_steps": self.parameter_decision_steps,
            "parameter_decision_dt": self.control_dt * self.parameter_decision_steps,
            "actuator_count": self.n_actuators,
            "yaw_roll_pairs": self.n_pairs,
            "omega_max": self.OMEGA_MAX,
            "k_max": self.K_MAX,
            "wave_dt": self.WAVE_DT,
            "tail_envelope": [self.HEAD_AMPLITUDE, 1.0],
            "fixed_roll": 0.0,
        }

    def close(self):
        self.env.close()

class CPGWarmupSAC(SAC):
    """SAC with range-limited random CPG actions during replay-buffer warm-up."""

    def __init__(
        self,
        *args,
        warmup_low: np.ndarray,
        warmup_high: np.ndarray,
        warmup_seed: Optional[int] = None,
        **kwargs,
    ):
        self.cpg_warmup_low = np.asarray(warmup_low, dtype=np.float32).reshape(-1)
        self.cpg_warmup_high = np.asarray(warmup_high, dtype=np.float32).reshape(-1)
        if self.cpg_warmup_low.shape != self.cpg_warmup_high.shape:
            raise ValueError("Random warm-up bounds must have matching shapes")
        if np.any(self.cpg_warmup_low > self.cpg_warmup_high):
            raise ValueError("Random warm-up lower bounds must not exceed upper bounds")
        self.cpg_warmup_rng = np.random.default_rng(warmup_seed)
        super().__init__(*args, **kwargs)

    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        if self.num_timesteps < learning_starts:
            scaled_action = self.cpg_warmup_rng.uniform(
                self.cpg_warmup_low,
                self.cpg_warmup_high,
                size=(n_envs, self.cpg_warmup_low.size),
            ).astype(np.float32)
            action = self.policy.unscale_action(scaled_action)
            return action, scaled_action
        return super()._sample_action(learning_starts, action_noise, n_envs)


_VORTICITY_RGB_LUT: Optional[np.ndarray] = None


def _vorticity_rgb_lut() -> np.ndarray:
    """Build the RdBu lookup table once instead of running a float colormap per frame."""
    global _VORTICITY_RGB_LUT
    if _VORTICITY_RGB_LUT is None:
        import matplotlib.pyplot as plt

        values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        _VORTICITY_RGB_LUT = (
            plt.get_cmap("RdBu_r")(values)[:, :3] * 255.0
        ).astype(np.uint8)
    return _VORTICITY_RGB_LUT


def vorticity_frame(raw_env: Eel2DLBMEnv, height: int = 600, vorticity_vmax: float = 20.0) -> np.ndarray:
    """Render current 2D LBM vorticity as RGB uint8 with a fixed color range."""

    flow = raw_env.solver.flows[0]
    wp.launch(get_vorticity_with_solid_img, dim=(flow.nx, flow.ny), inputs=[flow, 1.0])
    wp.synchronize()

    raw = np.flipud(flow.u_img.numpy().T)
    solid = raw >= 999.0

    fluid = raw.copy()
    fluid[solid] = 0.0

    vmax = max(float(vorticity_vmax), 1.0e-6)
    normalized = np.clip(0.5 + 0.5 * fluid / vmax, 0.0, 1.0)
    color_indices = np.asarray(normalized * 255.0, dtype=np.uint8)
    frame = _vorticity_rgb_lut()[color_indices]
    frame[solid] = np.array([190, 190, 190], dtype=np.uint8)

    h, w = frame.shape[:2]
    if h != height:
        width = int(round(w * height / h))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA if h > height else cv2.INTER_CUBIC)
    return frame


def draw_target_marker(frame: np.ndarray, raw_env: Eel2DLBMEnv) -> None:
    """Overlay the current eel goal in the same coordinates as the LBM image."""
    if getattr(raw_env, "task_mode", "goal") != "goal":
        return
    targets = getattr(raw_env, "target_positions_lbm", None)
    if targets is None or len(targets) == 0:
        return
    target = np.asarray(targets[0], dtype=np.float32)
    if target.shape != (2,) or not np.all(np.isfinite(target)):
        return

    h, w = frame.shape[:2]
    px = int(round(float(target[0]) / float(raw_env.nx) * (w - 1)))
    py = int(round((1.0 - float(target[1]) / float(raw_env.ny)) * (h - 1)))
    radius_lbm = float(getattr(raw_env, "target_radius_fraction", 0.02)) * float(raw_env.ny)
    radius_px = max(
        7,
        int(round(radius_lbm * min(w / float(raw_env.nx), h / float(raw_env.ny)))),
    )
    px = int(np.clip(px, 0, w - 1))
    py = int(np.clip(py, 0, h - 1))
    green = (85, 255, 115)
    dark = (15, 45, 22)
    cv2.circle(frame, (px, py), radius_px + 2, dark, 3, cv2.LINE_AA)
    cv2.circle(frame, (px, py), radius_px, green, 2, cv2.LINE_AA)
    cv2.circle(frame, (px, py), 3, green, -1, cv2.LINE_AA)
    cv2.line(frame, (px - radius_px, py), (px + radius_px, py), green, 1, cv2.LINE_AA)
    cv2.line(frame, (px, py - radius_px), (px, py + radius_px), green, 1, cv2.LINE_AA)
    label_y = max(18, py - radius_px - 7)
    _draw_text(frame, "TARGET", (max(3, px - 27), label_y), 0.38, green)



def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return default


def _as_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    return arr.reshape(-1)


def set_lbm_viscosity(env: Eel2DLBMEnv, viscosity: float) -> None:
    for flow in env.solver.flows:
        flow.vis_shear = float(viscosity)
    env.solver.flows_wp = wp.array(env.solver.flows, dtype=HomeFlow, device=env.solver.device)
    env.solver.captured = False
    env.solver.captured_graph = None


def get_lbm_viscosity(env: Eel2DLBMEnv) -> float:
    try:
        return float(env.solver.flows[0].vis_shear)
    except Exception:
        return 0.0







def _draw_text(img: np.ndarray, text: str, xy: Tuple[int, int], scale: float = 0.55, color: Tuple[int, int, int] = (230, 235, 245)) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_sparkline(img: np.ndarray, values: deque, rect: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> None:
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (45, 50, 62), 1, cv2.LINE_AA)
    if len(values) < 2:
        return

    data = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(data))
    hi = float(np.max(data))
    span = max(hi - lo, 1.0e-6)
    xs = np.linspace(x + 2, x + w - 2, len(data)).astype(np.int32)
    ys = (y + h - 2 - (data - lo) / span * (h - 4)).astype(np.int32)
    pts = np.column_stack([xs, ys]).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
    _draw_text(img, f"{lo:+.2f}", (x + 4, y + h - 6), 0.38, (145, 150, 165))
    _draw_text(img, f"{hi:+.2f}", (x + 4, y + 14), 0.38, (145, 150, 165))


def _draw_action_bars(img: np.ndarray, action: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (45, 50, 62), 1, cv2.LINE_AA)
    if action.size == 0:
        _draw_text(img, "action: none", (x + 8, y + 24), 0.45, (160, 165, 178))
        return

    bar_gap = 8
    bar_h = max(12, (h - bar_gap * (len(action) + 1)) // max(len(action), 1))
    center_x = x + w // 2
    cv2.line(img, (center_x, y + 6), (center_x, y + h - 6), (85, 90, 105), 1, cv2.LINE_AA)
    for i, val in enumerate(np.clip(action, -1.0, 1.0)):
        yy = y + bar_gap + i * (bar_h + bar_gap)
        length = int(abs(float(val)) * (w * 0.43))
        if val >= 0.0:
            pt1, pt2 = (center_x, yy), (center_x + length, yy + bar_h)
            color = (88, 210, 140)
        else:
            pt1, pt2 = (center_x - length, yy), (center_x, yy + bar_h)
            color = (255, 130, 100)
        cv2.rectangle(img, pt1, pt2, color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x + 2, yy), (x + w - 2, yy + bar_h), (60, 66, 82), 1, cv2.LINE_AA)
        _draw_text(img, f"a{i}: {float(val):+.2f}", (x + 8, yy + bar_h - 3), 0.42, (235, 238, 245))


def build_sac_panel(frame: np.ndarray, stats: Dict[str, Any], panel_width: int = 360) -> np.ndarray:
    """Right-side SAC stats panel (text + sparklines).  Action bars live in their own column."""
    h, _w = frame.shape[:2]
    panel_width = max(300, int(panel_width))
    panel = np.full((h, panel_width, 3), (22, 25, 32), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, h - 1), (55, 60, 74), 1, cv2.LINE_AA)

    x = 18
    y = 32
    _draw_text(panel, "SAC Live", (x, y), 0.75, (250, 250, 255))
    y += 30
    mode = "learning" if stats["replay_size"] >= stats["learning_starts"] else "collecting"
    lines = [
        f"step: {stats['step']}",
        f"mode: {mode}",
        f"reward: {stats['reward']:+.4f}",
        f"episode return: {stats['episode_return']:+.3f}",
        f"episode len: {stats['episode_length']}",
        f"replay: {stats['replay_size']}/{stats['buffer_size']}",
        f"viscosity: {stats['viscosity']:.4f}",
        f"vort range: +/-{stats.get('vorticity_vmax', 20.0):.1f}",
        f"alpha: {stats['alpha']:.4f}",
    ]
    for line in lines:
        _draw_text(panel, line, (x, y), 0.5, (210, 218, 232))
        y += 23

    inner_w = panel_width - 2 * x
    y += 8
    _draw_text(panel, "reward trace", (x, y), 0.52, (250, 250, 255))
    y += 8
    _draw_sparkline(panel, stats["reward_history"], (x, y, inner_w, 92), (90, 190, 255))
    y += 118

    _draw_text(panel, "episode returns", (x, y), 0.52, (250, 250, 255))
    y += 8
    _draw_sparkline(panel, stats["episode_history"], (x, y, inner_w, 82), (250, 210, 85))

    _draw_text(panel, "Q/Esc: stop training", (x, h - 18), 0.45, (150, 156, 170))
    return panel


def build_action_panel(action: np.ndarray, height: int, action_width: int = 140) -> np.ndarray:
    """A narrow standalone column showing one action bar per dimension."""
    action_width = max(100, int(action_width))
    panel = np.full((height, action_width, 3), (22, 25, 32), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (action_width - 1, height - 1), (55, 60, 74), 1, cv2.LINE_AA)

    x = 8
    y = 32
    _draw_text(panel, "policy action", (x, y), 0.52, (250, 250, 255))
    y += 12

    bar_gap = 6
    n = max(len(action), 1)
    bar_h = max(14, (height - y - 22 - bar_gap * (n + 1)) // n)
    used_h = bar_gap * n + bar_h * n
    start_y = y + (height - y - 22 - used_h) // 2

    center_x = action_width // 2
    cv2.line(panel, (center_x, start_y - 4), (center_x, start_y + used_h + 4), (85, 90, 105), 1, cv2.LINE_AA)
    for i, val in enumerate(np.clip(action, -1.0, 1.0)):
        yy = start_y + i * (bar_h + bar_gap)
        length = int(abs(float(val)) * (action_width * 0.38))
        if val >= 0.0:
            pt1, pt2 = (center_x, yy), (center_x + length, yy + bar_h)
            color = (88, 210, 140)
        else:
            pt1, pt2 = (center_x - length, yy), (center_x, yy + bar_h)
            color = (255, 130, 100)
        cv2.rectangle(panel, pt1, pt2, color, -1, cv2.LINE_AA)
        cv2.rectangle(panel, (center_x - action_width // 2 + 4, yy), (center_x + action_width // 2 - 4, yy + bar_h), (60, 66, 82), 1, cv2.LINE_AA)
        _draw_text(panel, f"a{i}: {float(val):+.2f}", (x, yy + bar_h - 4), 0.38, (235, 238, 245))

    _draw_text(panel, "warmup active", (x, height - 18), 0.40, (150, 156, 170))
    return panel


_CPG_JOINT_COLORS = [
    (78, 205, 255),
    (80, 230, 185),
    (170, 225, 95),
    (255, 190, 75),
    (255, 105, 125),
]


def _draw_cpg_parameter_row(
    panel: np.ndarray,
    y: int,
    label: str,
    value_text: str,
    actual_normalized: float,
    target_normalized: float,
    width: int,
) -> None:
    """Draw one physical CPG value with executed fill and SAC-target marker."""
    x = 16
    right = width - 16
    _draw_text(panel, label, (x, y + 15), 0.43, (188, 198, 216))
    text_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
    _draw_text(panel, value_text, (right - text_size[0], y + 15), 0.48, (245, 248, 252))

    bar_y = y + 23
    bar_h = 9
    cv2.rectangle(panel, (x, bar_y), (right, bar_y + bar_h), (48, 55, 70), -1)
    cv2.rectangle(panel, (x, bar_y), (right, bar_y + bar_h), (75, 84, 103), 1, cv2.LINE_AA)
    actual_u = float(np.clip(0.5 * (actual_normalized + 1.0), 0.0, 1.0))
    target_u = float(np.clip(0.5 * (target_normalized + 1.0), 0.0, 1.0))
    actual_x = x + int(actual_u * (right - x))
    target_x = x + int(target_u * (right - x))
    cv2.rectangle(panel, (x + 1, bar_y + 1), (actual_x, bar_y + bar_h - 1), (76, 205, 230), -1)
    cv2.line(panel, (target_x, bar_y - 3), (target_x, bar_y + bar_h + 3), (255, 105, 190), 2, cv2.LINE_AA)


def _draw_cpg_wave_chart(
    panel: np.ndarray,
    rect: Tuple[int, int, int, int],
    cpg_env: EelCPGWrapper,
    parameters: np.ndarray,
) -> None:
    """Preview all joint commands over one oscillator cycle."""
    x, y, w, h = rect
    cv2.rectangle(panel, (x, y), (x + w, y + h), (48, 55, 70), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (75, 84, 103), 1, cv2.LINE_AA)
    mid_y = y + h // 2
    cv2.line(panel, (x + 1, mid_y), (x + w - 1, mid_y), (82, 90, 108), 1, cv2.LINE_AA)
    for fraction in (0.25, 0.5, 0.75):
        grid_x = x + int(fraction * w)
        cv2.line(panel, (grid_x, y + 1), (grid_x, y + h - 1), (58, 65, 80), 1)

    amplitude, omega, k_wave, head_bias = parameters
    cycle = np.linspace(0.0, 1.0, max(100, w - 4), dtype=np.float32)
    xs = np.linspace(x + 2, x + w - 2, cycle.size).astype(np.int32)
    body_s = np.linspace(0.0, 1.0, cpg_env.n_pairs, dtype=np.float32)
    temporal_direction = -1.0 if float(omega) < 0.0 else 1.0
    for joint_idx, s in enumerate(body_s):
        envelope = cpg_env.HEAD_AMPLITUDE + (1.0 - cpg_env.HEAD_AMPLITUDE) * s
        values = amplitude * envelope * np.sin(
            temporal_direction * 2.0 * np.pi * cycle
            + k_wave * cpg_env.K_MAX * np.pi * s
        )
        values += head_bias * (1.0 - s)
        values = np.clip(values, -1.0, 1.0)
        ys = (mid_y - values * (h * 0.43)).astype(np.int32)
        points = np.column_stack([xs, ys]).reshape((-1, 1, 2))
        color = _CPG_JOINT_COLORS[joint_idx % len(_CPG_JOINT_COLORS)]
        cv2.polylines(panel, [points], False, color, 1 if joint_idx < cpg_env.n_pairs - 1 else 2, cv2.LINE_AA)

    phase_u = float(cpg_env.phase % (2.0 * np.pi) / (2.0 * np.pi))
    phase_x = x + int(phase_u * w)
    cv2.line(panel, (phase_x, y), (phase_x, y + h), (250, 250, 255), 1, cv2.LINE_AA)
    cv2.circle(panel, (phase_x, y + 7), 3, (250, 250, 255), -1, cv2.LINE_AA)
    physical_frequency = (
        abs(float(omega))
        * cpg_env.OMEGA_MAX
        / (2.0 * np.pi)
        * cpg_env.WAVE_DT
        / cpg_env.control_dt
    )
    period = 1.0 / max(physical_frequency, 1.0e-6)
    _draw_text(panel, "now", (min(phase_x + 4, x + w - 28), y + 13), 0.32, (235, 238, 245))
    _draw_text(panel, "0", (x + 3, y + h - 5), 0.32, (135, 145, 165))
    _draw_text(panel, f"one cycle  T={period:.2f}s", (x + w - 120, y + h - 5), 0.32, (155, 165, 184))


def _draw_actual_body_pose(
    panel: np.ndarray,
    rect: Tuple[int, int, int, int],
    cpg_env: EelCPGWrapper,
) -> None:
    """Show actual coupled solid centers using the same LBM world x/y axes."""
    x, y, w, h = rect
    cv2.rectangle(panel, (x, y), (x + w, y + h), (48, 55, 70), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (75, 84, 103), 1, cv2.LINE_AA)
    polygons = cpg_env.actual_body_polygons()
    points = cpg_env.actual_body_points()
    if polygons is None or points is None:
        _draw_text(panel, "body pose unavailable", (x + 10, y + h // 2), 0.40, (155, 165, 184))
        return

    all_vertices = np.vstack(polygons)
    center = np.mean(all_vertices, axis=0)
    span = np.ptp(all_vertices, axis=0)
    drawable_w = max(1.0, float(w - 52))
    drawable_h = max(1.0, float(h - 32))
    scale_x = drawable_w / max(float(span[0]), 1.0e-5)
    scale_y = drawable_h / max(float(span[1]), 1.0e-5)
    scale = min(scale_x, scale_y)
    def to_screen(vertices: np.ndarray) -> np.ndarray:
        centered = vertices - center
        screen_x = x + w * 0.5 + centered[:, 0] * scale
        # LBM render flips the image vertically, so larger world y is higher on screen.
        screen_y = y + h * 0.5 - centered[:, 1] * scale
        return np.column_stack([screen_x, screen_y]).astype(np.int32)

    for i, polygon in enumerate(polygons):
        screen_polygon = to_screen(polygon).reshape((-1, 1, 2))
        color = (225, 235, 245) if i == 0 else _CPG_JOINT_COLORS[(i - 1) % len(_CPG_JOINT_COLORS)]
        cv2.fillPoly(panel, [screen_polygon], color, cv2.LINE_AA)
        cv2.polylines(panel, [screen_polygon], True, (35, 42, 55), 1, cv2.LINE_AA)

    screen_points = to_screen(points)
    cv2.polylines(panel, [screen_points.reshape((-1, 1, 2))], False, (35, 42, 55), 1, cv2.LINE_AA)

    _draw_text(panel, "HEAD", (x + 5, y + 13), 0.32, (210, 220, 235))
    label = "TAIL"
    text_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0]
    _draw_text(panel, label, (x + w - text_w - 5, y + 13), 0.32, _CPG_JOINT_COLORS[-1])
    _draw_text(panel, "LBM world XY / equal scale", (x + 5, y + h - 6), 0.30, (135, 145, 165))


def build_cpg_action_panel(
    cpg_env: EelCPGWrapper,
    target_action: np.ndarray,
    height: int,
    panel_width: int = 400,
) -> np.ndarray:
    """CPG-specific policy panel with physical parameters and generated waves."""
    panel_width = max(340, int(panel_width))
    panel = np.full((height, panel_width, 3), (22, 25, 32), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, height - 1), (55, 60, 74), 1, cv2.LINE_AA)

    target_normalized = np.asarray(target_action, dtype=np.float32).reshape(-1)
    if target_normalized.size != len(cpg_env.PARAMETER_NAMES):
        target_normalized = cpg_env.current_normalized.copy()
    actual_parameters = cpg_env._physical_parameters()

    _draw_text(panel, "CPG policy", (16, 28), 0.70, (250, 250, 255))
    _draw_text(panel, "cyan: executed   pink: SAC target", (16, 48), 0.36, (160, 170, 190))

    labels = ("A", "omega", "k_wave", "head_bias")
    value_texts = (
        f"{actual_parameters[0]:.3f}",
        f"{actual_parameters[1]:+.3f}",
        f"{actual_parameters[2]:.3f}",
        f"{actual_parameters[3]:+.3f}",
    )
    row_y = 58
    for index, (label, value_text) in enumerate(zip(labels, value_texts)):
        _draw_cpg_parameter_row(
            panel,
            row_y + index * 41,
            label,
            value_text,
            float(cpg_env.current_normalized[index]),
            float(target_normalized[index]),
            panel_width,
        )

    wave_title_y = row_y + len(labels) * 41 + 15
    _draw_text(panel, "Yaw commands / physical cycle", (16, wave_title_y), 0.48, (235, 240, 248))
    hold_text = cpg_env.parameter_stage_text()
    hold_w = cv2.getTextSize(hold_text, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0][0]
    _draw_text(panel, hold_text, (panel_width - hold_w - 16, wave_title_y), 0.34, (155, 165, 184))
    wave_y = wave_title_y + 10
    remaining = height - wave_y - 24
    if remaining < 180:
        # Compact render heights keep the physically useful cycle chart and omit
        # the secondary body snapshot rather than drawing outside the panel.
        wave_h = max(60, remaining)
        _draw_cpg_wave_chart(panel, (16, wave_y, panel_width - 32, wave_h), cpg_env, actual_parameters)
        return panel

    wave_h = max(90, int(remaining * 0.58))
    snapshot_h = max(60, remaining - wave_h - 30)
    _draw_cpg_wave_chart(panel, (16, wave_y, panel_width - 32, wave_h), cpg_env, actual_parameters)

    snapshot_title_y = wave_y + wave_h + 22
    _draw_text(panel, "Actual coupled body pose", (16, snapshot_title_y), 0.48, (235, 240, 248))
    snapshot_y = snapshot_title_y + 9
    _draw_actual_body_pose(
        panel,
        (16, snapshot_y, panel_width - 32, min(snapshot_h, height - snapshot_y - 10)),
        cpg_env,
    )
    return panel




class RealtimeLBMCallback(BaseCallback):
    """Show live LBM vorticity and SAC training stats while SB3 is training."""

    def __init__(
        self,
        raw_env: Eel2DLBMEnv,
        every: int = 5,
        height: int = 600,
        panel_width: int = 360,
        vorticity_vmax: float = 20.0,
        cpg_env: Optional[EelCPGWrapper] = None,
        cpg_panel_width: int = 400,
    ):
        super().__init__()
        self.raw_env = raw_env
        self.every = max(1, int(every))
        self.height = int(height)
        self.panel_width = int(panel_width)
        self.vorticity_vmax = float(vorticity_vmax)
        self.cpg_env = cpg_env
        self.cpg_panel_width = int(cpg_panel_width)

        self.window = "SAC training - LBM + SAC"
        self.reward_history = deque(maxlen=240)
        self.episode_history = deque(maxlen=120)
        self.episode_return = 0.0
        self.episode_length = 0
        self.last_action = np.zeros(0, dtype=np.float32)
        self.last_reward = 0.0
        self.stop_requested = False

    def _on_training_start(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

    def _read_alpha(self) -> float:
        try:
            log_ent_coef = getattr(self.model, "log_ent_coef", None)
            if log_ent_coef is not None:
                return float(log_ent_coef.exp().detach().cpu().item())
            ent_coef = getattr(self.model, "ent_coef", 0.0)
            return float(ent_coef) if not isinstance(ent_coef, str) else 0.0
        except Exception:
            return 0.0

    def _build_stats(self) -> Dict[str, Any]:
        replay_buffer = getattr(self.model, "replay_buffer", None)
        replay_size = int(replay_buffer.size()) if replay_buffer is not None else 0
        return {
            "step": int(self.num_timesteps),
            "reward": self.last_reward,
            "episode_return": self.episode_return,
            "episode_length": self.episode_length,
            "reward_history": self.reward_history,
            "episode_history": self.episode_history,
            "action": self.last_action,
            "alpha": self._read_alpha(),
            "viscosity": get_lbm_viscosity(self.raw_env),
            "vorticity_vmax": self.vorticity_vmax,
            "replay_size": replay_size,
            "buffer_size": int(getattr(self.model, "buffer_size", 0)),

            "learning_starts": int(getattr(self.model, "learning_starts", 0)),
        }

    def _on_step(self) -> bool:
        self.last_reward = _as_float(self.locals.get("rewards", [0.0]))
        self.last_action = _as_vector(self.locals.get("actions"))
        self.reward_history.append(self.last_reward)
        self.episode_return += self.last_reward
        self.episode_length += 1

        dones = np.asarray(self.locals.get("dones", [False])).reshape(-1)
        if dones.size > 0 and bool(dones[0]):
            self.episode_history.append(self.episode_return)
            self.episode_return = 0.0
            self.episode_length = 0

        # CPG mode renders from inside the held low-level steps. Rendering again
        # here would duplicate the final frame and reintroduce a long pause.
        if self.cpg_env is not None and self.cpg_env.render_hook is not None:
            return not self.stop_requested

        if self.n_calls % self.every != 0:
            return True

        return self._render_current_frame()

    def _render_current_frame(self) -> bool:
        """Render one current coupled state; shared by callback and CPG substeps."""

        lbm_frame = vorticity_frame(self.raw_env, self.height, self.vorticity_vmax)
        draw_target_marker(lbm_frame, self.raw_env)

        if self.cpg_env is not None:
            text = (
                f"SAC step={self.num_timesteps}  {self.cpg_env.parameter_stage_text()}  "
                f"reward={self.last_reward:+.3f}"
            )
        else:
            text = f"step={self.num_timesteps}  ep_step={self.episode_length}  reward={self.last_reward:+.3f}"
        cv2.putText(lbm_frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(lbm_frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)

        sac_panel = build_sac_panel(lbm_frame, self._build_stats(), self.panel_width)
        if self.cpg_env is not None:
            action_panel = build_cpg_action_panel(
                self.cpg_env,
                self.cpg_env.last_target_normalized,
                self.height,
                self.cpg_panel_width,
            )
        else:
            action_panel = build_action_panel(self.last_action, self.height, 140)
        frame = np.concatenate([lbm_frame, sac_panel, action_panel], axis=1)


        cv2.imshow(self.window, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"), ord("Q"))

    def render_cpg_substep(self, reward: float) -> bool:
        """Render while one SAC-selected CPG parameter set is being held."""
        self.last_reward = float(reward)
        keep_running = self._render_current_frame()
        if not keep_running:
            self.stop_requested = True
        return keep_running

    def _on_training_end(self) -> None:
        cv2.destroyWindow(self.window)


class LBMVideoRecorder:
    """Record only the LBM view with compact policy-evaluation annotations."""

    def __init__(
        self,
        raw_env: Eel2DLBMEnv,
        cpg_env: EelCPGWrapper,
        output_path: Path,
        height: int,
        vorticity_vmax: float,
        playback_speed: float,
        frame_stride: int,
    ):
        self.raw_env = raw_env
        self.cpg_env = cpg_env
        self.output_path = output_path
        self.height = int(height)
        self.vorticity_vmax = float(vorticity_vmax)
        self.playback_speed = float(playback_speed)
        self.frame_stride = max(1, int(frame_stride))
        self.output_fps = self.playback_speed / (
            self.cpg_env.control_dt * self.frame_stride
        )
        self.writer: Optional[cv2.VideoWriter] = None
        self.episode = 0
        self.total_reward = 0.0
        self.frame_count = 0

    def begin_episode(self, episode: int) -> None:
        self.episode = int(episode)
        self.total_reward = 0.0

    def _annotate(self, frame: np.ndarray, reward: float) -> None:
        height, width = frame.shape[:2]
        band_height = min(118, height)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, band_height), (8, 12, 20), -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0.0, dst=frame)

        white = (245, 248, 252)
        green = (100, 255, 140)
        cv2.putText(
            frame, "Task: Forward", (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, white, 2, cv2.LINE_AA,
        )
        arrow_x = min(width - 18, 176)
        cv2.arrowedLine(
            frame, (arrow_x, 30), (arrow_x, 7), green, 2, cv2.LINE_AA,
            tipLength=0.35,
        )
        cv2.putText(
            frame, f"Reward: {reward:+.4f}", (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"Total Reward: {self.total_reward:+.4f}", (12, 82),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv2.LINE_AA,
        )
        simulation_step = int(np.asarray(self.raw_env.current_steps).reshape(-1)[0])
        cv2.putText(
            frame, f"Sim Step: {simulation_step}", (12, 108),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv2.LINE_AA,
        )
        speed_text = f"Speed: {self.playback_speed:g}x"
        text_width = cv2.getTextSize(
            speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )[0][0]
        cv2.putText(
            frame, speed_text, (max(12, width - text_width - 12), 108),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, green, 1, cv2.LINE_AA,
        )

    def record(self, reward: float) -> bool:
        self.total_reward += float(reward)
        frame = vorticity_frame(self.raw_env, self.height, self.vorticity_vmax)
        self._annotate(frame, float(reward))
        if self.writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.output_fps,
                (width, height),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {self.output_path}")
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frame_count += 1
        return True

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        print(
            f"Video saved: {self.output_path} "
            f"({self.frame_count} frames, {self.output_fps:.2f} fps)"
        )


def evaluate_loaded_policy(
    model: SAC,
    env: gym.Env,
    renderer: Optional[RealtimeLBMCallback],
    video_recorder: Optional[LBMVideoRecorder],
    episodes: int,
    seed: int,
    deterministic: bool,
) -> None:
    """Run a loaded policy without gradient updates or checkpoint writes."""
    if renderer is not None:
        renderer.init_callback(model)
        renderer.on_training_start({}, {})

    stop_requested = False
    try:
        for episode in range(max(1, int(episodes))):
            observation, _ = env.reset(seed=seed + episode)
            if video_recorder is not None:
                video_recorder.begin_episode(episode + 1)
            episode_return = 0.0
            episode_length = 0
            terminated = False
            truncated = False
            info: Dict[str, Any] = {}

            while not (terminated or truncated or stop_requested):
                action, _ = model.predict(observation, deterministic=deterministic)
                if renderer is not None:
                    renderer.last_action = np.asarray(action, dtype=np.float32)
                    renderer.episode_return = episode_return
                    renderer.episode_length = episode_length
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                if renderer is not None:
                    renderer.reward_history.append(float(reward))
                    stop_requested = renderer.stop_requested

            if renderer is not None:
                renderer.episode_history.append(episode_return)
            metrics = ""
            if "forward_progress_lbm" in info:
                metrics = (
                    f", forward={float(info['forward_progress_lbm']):+.3f}, "
                    f"lateral={float(info['lateral_drift_lbm']):+.3f}"
                )
            print(
                f"[eval] episode={episode + 1}, return={episode_return:+.4f}, "
                f"length={episode_length}{metrics}"
            )
    finally:
        if renderer is not None:
            renderer.on_training_end()
        if video_recorder is not None:
            video_recorder.close()


def main():

    parser = argparse.ArgumentParser(description="Minimal SAC training for LBM-RIGID")

    parser.add_argument(
        "--animal",
        choices=["eel"],
        default="eel",
        help="Animal environment to train (only eel is retained)",
    )
    parser.add_argument(
        "--control-mode",
        choices=["direct", "cpg"],
        default="direct",
        help="Direct actuator actions, or four planar eel CPG parameters (eel only)",
    )
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Save an intermediate policy zip every N SAC decisions; 0 disables checkpoints")
    parser.add_argument("--learning-starts", type=int, default=250, help="Replay warm-up SAC decisions before gradient updates")
    parser.add_argument("--load-model", type=Path, default=None, help="Load an existing SAC ZIP instead of creating a new model")
    parser.add_argument("--eval-only", action="store_true", help="Run a loaded policy without training")
    parser.add_argument("--eval-episodes", type=int, default=3, help="Episodes to run in evaluation mode")
    parser.add_argument("--stochastic-eval", action="store_true", help="Sample actions during evaluation instead of using the deterministic mean")
    parser.add_argument("--record-video", type=Path, default=None, help="Write an eval-only MP4 containing only the annotated LBM view")
    parser.add_argument("--playback-speed", type=float, default=5.0, help="Encoded video playback speed multiplier")

    parser.add_argument("--render", action="store_true", help="Show realtime LBM vorticity and SAC panel during training")


    parser.add_argument("--render-every", type=int, default=5, help="Render every N SAC steps in direct-control mode")
    parser.add_argument("--render-substep-every", type=int, default=1, help="Render every N low-level steps while CPG parameters are held")
    parser.add_argument("--render-height", type=int, default=600)
    parser.add_argument("--vorticity-vmax", type=float, default=20.0, help="Fixed vorticity color range: [-vmax, +vmax]")
    parser.add_argument("--panel-width", type=int, default=360, help="Right-side SAC stats panel width in pixels")
    parser.add_argument("--cpg-panel-width", type=int, default=400, help="Width of the CPG parameter/wave panel")



    parser.add_argument("--warmup-steps", type=int, default=50, help="Zero-action LBM warmup steps after each reset")
    parser.add_argument("--viscosity", type=float, default=0.05, help="LBM kinematic viscosity; larger value lowers Reynolds number")
    parser.add_argument("--action-scale", type=float, default=0.8, help="Scale SAC actions before applying them to MuJoCo actuators")
    parser.add_argument("--per-frame-steps", type=int, default=None, help="Coupled physics substeps per low-level action; defaults to 8 for projected eel CPG and 10 otherwise")
    parser.add_argument("--episode-steps", type=int, default=100, help="Episode length in SAC decisions (or raw actions in direct mode)")
    parser.add_argument("--task", choices=["goal", "forward"], default="goal", help="Reach a target point, or learn a straight forward swimming gait")
    parser.add_argument("--target-mode", choices=["random", "ahead"], default="random", help="Random front-sector goals for a reusable policy, or a fixed straight-ahead goal")
    parser.add_argument("--target-distance-range", type=float, nargs=2, default=[0.12, 0.25], metavar=("MIN", "MAX"), help="Random target distance range as fractions of LBM ny")
    parser.add_argument("--target-angle-range-deg", type=float, nargs=2, default=[-70.0, 70.0], metavar=("MIN", "MAX"), help="Random target bearing range relative to the eel heading")
    parser.add_argument("--target-radius-fraction", type=float, default=0.02, help="Success radius as a fraction of LBM ny")
    parser.add_argument("--forward-progress-weight", type=float, default=100.0, help="Forward displacement reward weight")
    parser.add_argument("--forward-lateral-weight", type=float, default=20.0, help="Lateral displacement penalty weight")
    parser.add_argument("--forward-heading-weight", type=float, default=0.0001, help="Per-step penalty weight for turning away from the initial heading")
    parser.add_argument("--cpg-smoothing", type=float, default=1.0, help="Deprecated compatibility option; explicit ramp now reaches the SAC target")
    parser.add_argument("--cpg-ramp-steps", type=int, default=10, help="Low-level steps used to linearly reach each new SAC CPG target")
    parser.add_argument("--cpg-hold-steps", type=int, default=80, help="Low-level steps to hold the reached CPG target")
    parser.add_argument("--warmup-exploration", choices=["rand", "uniform"], default="rand", help="Range-limited random CPG warm-up, or SB3 full-action uniform warm-up")
    parser.add_argument("--warmup-head-bias-range", type=float, nargs=2, default=[-0.30, 0.30], metavar=("MIN", "MAX"), help="Physical head_bias range used by random CPG warm-up")
    args = parser.parse_args()

    if args.eval_only and args.load_model is None:
        parser.error("--eval-only requires --load-model")
    if args.load_model is not None and not args.load_model.exists():
        parser.error(f"Model ZIP not found: {args.load_model}")
    if args.record_video is not None and not args.eval_only:
        parser.error("--record-video requires --eval-only")
    if args.record_video is not None and args.control_mode != "cpg":
        parser.error("--record-video currently requires --control-mode cpg")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    if not (
        0.0 < args.target_distance_range[0] <= args.target_distance_range[1]
    ):
        parser.error("--target-distance-range must satisfy 0 < MIN <= MAX")
    if not (
        -90.0 <= args.target_angle_range_deg[0]
        <= args.target_angle_range_deg[1]
        <= 90.0
    ):
        parser.error("--target-angle-range-deg must stay in the forward half-plane [-90, 90]")
    if not (0.0 < args.target_radius_fraction < 0.5):
        parser.error("--target-radius-fraction must be between 0 and 0.5")
    if not (-0.30 <= args.warmup_head_bias_range[0] <= args.warmup_head_bias_range[1] <= 0.30):
        parser.error("--warmup-head-bias-range must stay within [-0.30, 0.30]")




    outdir = Path("outputs/sac_minimal")
    outdir.mkdir(parents=True, exist_ok=True)

    per_frame_steps = args.per_frame_steps
    if per_frame_steps is None:
        per_frame_steps = 8 if args.control_mode == "cpg" else 10
    raw_episode_steps = int(args.episode_steps)
    if args.control_mode == "cpg":
        # Express each episode length in low-level control steps.
        raw_episode_steps *= int(args.cpg_ramp_steps + args.cpg_hold_steps)

    if args.control_mode == "cpg":
        # Reuse the projected geometry from the realtime preset.
        project_root = Path(__file__).resolve().parent
        projected_config_path = project_root / "configs" / "realtime_2d" / "eel2d.json"
        projected_config = json.loads(projected_config_path.read_text(encoding="utf-8"))
        env_config = projected_config["env"]
        lbm_config = projected_config["lbm"]
        solid_config = [dict(item) for item in env_config["solid_config"]]
        raw_env = Eel2DLBMEnv(
            xml_path=str(project_root / env_config["xml_path"]),
            solid_config=solid_config,
            nworld=1,
            nx=int(lbm_config["nx"]),
            ny=int(lbm_config["ny"]),
            lbm_scale=float(lbm_config["lbm_scale"]),
            per_frame_steps=int(per_frame_steps),
            max_episode_steps=raw_episode_steps,
            include_image=False,
            render_mode=None,
        )
        # The CPG wrapper applies each actuator's physical range.
        raw_env.action_scale = 1.0
        print(
            f"[eel CPG] projected model={env_config['xml_path']} solids={raw_env.solid_num} "
            f"actuators={raw_env.model.nu} grid={raw_env.nx}x{raw_env.ny} "
            f"per_frame_steps={per_frame_steps}"
        )
    else:
        raw_env = Eel2DLBMEnv(
            nworld=1,
            nx=320,
            ny=480,
            lbm_scale=0.2,
            per_frame_steps=int(per_frame_steps),
            max_episode_steps=raw_episode_steps,
            include_image=False,
            render_mode=None,
        )
        raw_env.action_scale = float(args.action_scale)

    if args.animal == "eel":
        raw_env.task_mode = args.task
        raw_env.randomize_target = args.target_mode == "random"
        raw_env.target_distance_range_fraction = tuple(args.target_distance_range)
        raw_env.target_angle_range_deg = tuple(args.target_angle_range_deg)
        raw_env.target_radius_fraction = float(args.target_radius_fraction)
        raw_env.forward_progress_weight = float(args.forward_progress_weight)
        raw_env.forward_lateral_weight = float(args.forward_lateral_weight)
        raw_env.forward_heading_weight = float(args.forward_heading_weight)

    set_lbm_viscosity(raw_env, args.viscosity)
    base_env = SingleEnvWrapper(
        raw_env,
        warmup_steps=args.warmup_steps,
        viscosity=args.viscosity,
        latest_observation_only=args.control_mode == "cpg",
    )
    cpg_env = None
    if args.control_mode == "cpg":
        cpg_env = EelCPGWrapper(
            base_env,
            smoothing=args.cpg_smoothing,
            parameter_ramp_steps=args.cpg_ramp_steps,
            parameter_hold_steps=args.cpg_hold_steps,
        )
        train_env = cpg_env
    else:
        train_env = base_env

    task_suffix = "_forward" if args.task == "forward" else "_goal"
    monitor_suffix = (
        f"{task_suffix}_cpg"
        if args.control_mode == "cpg"
        else ("_forward" if args.task == "forward" else "")
    )
    if args.task == "forward":
        monitor_info_keywords = ("forward_progress_lbm", "lateral_drift_lbm")
    elif cpg_env is not None:
        monitor_info_keywords = ("is_success", "target_distance_lbm")
    else:
        monitor_info_keywords = ()
    env = Monitor(
        train_env,
        filename=str(
            outdir
            / f"{args.animal}2d{monitor_suffix}{'.eval' if args.eval_only else ''}.monitor.csv"
        ),
        info_keywords=monitor_info_keywords,
    )
    model_name = f"sac_{args.animal}2d{monitor_suffix}"



    model_class = SAC
    model_extra_kwargs: Dict[str, Any] = {}
    warmup_parameter_low = None
    warmup_parameter_high = None
    if (
        cpg_env is not None
        and args.warmup_exploration == "rand"
        and args.load_model is None
    ):
        warmup_parameter_low = cpg_env.PARAMETER_LOW.copy()
        warmup_parameter_high = cpg_env.PARAMETER_HIGH.copy()
        warmup_parameter_low[3] = args.warmup_head_bias_range[0]
        warmup_parameter_high[3] = args.warmup_head_bias_range[1]
        model_class = CPGWarmupSAC
        model_extra_kwargs = {
            "warmup_low": cpg_env._parameters_to_normalized(warmup_parameter_low),
            "warmup_high": cpg_env._parameters_to_normalized(warmup_parameter_high),
            "warmup_seed": args.seed,
        }
        print(
            "[CPG warmup] independent uniform random parameters: "
            f"low={warmup_parameter_low.tolist()}, "
            f"high={warmup_parameter_high.tolist()}"
        )

    if args.load_model is not None:
        model = SAC.load(str(args.load_model), env=env, device="cuda")
        print(f"Loaded: {args.load_model}")
    else:
        model = model_class(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=args.learning_starts,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=str(outdir / "tb"),
            device="cuda",
            seed=args.seed,
            **model_extra_kwargs,
        )

    realtime_callback = (
        RealtimeLBMCallback(
            raw_env,
            every=args.render_every,
            height=args.render_height,
            panel_width=args.panel_width,
            vorticity_vmax=args.vorticity_vmax,
            cpg_env=cpg_env,
            cpg_panel_width=args.cpg_panel_width,
        )

        if args.render
        else None
    )
    video_recorder = (
        LBMVideoRecorder(
            raw_env,
            cpg_env,
            output_path=args.record_video,
            height=args.render_height,
            vorticity_vmax=args.vorticity_vmax,
            playback_speed=args.playback_speed,
            frame_stride=args.render_substep_every,
        )
        if args.record_video is not None and cpg_env is not None
        else None
    )
    render_hooks: list[Callable[[float], bool]] = []
    if realtime_callback is not None:
        render_hooks.append(realtime_callback.render_cpg_substep)
    if video_recorder is not None:
        render_hooks.append(video_recorder.record)
    if render_hooks and cpg_env is not None:
        def render_outputs(reward: float) -> bool:
            keep_running = True
            for hook in render_hooks:
                keep_running = bool(hook(reward)) and keep_running
            return keep_running

        cpg_env.set_render_hook(
            render_outputs,
            every=args.render_substep_every,
        )

    if args.eval_only:
        evaluate_loaded_policy(
            model,
            env,
            realtime_callback,
            video_recorder,
            episodes=args.eval_episodes,
            seed=args.seed,
            deterministic=not args.stochastic_eval,
        )
        env.close()
        return

    # Keep checkpoints independent of optional rendering.
    callbacks: list[BaseCallback] = []
    if args.checkpoint_every > 0:
        checkpoint_dir = outdir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, int(args.checkpoint_every)),
                save_path=str(checkpoint_dir),
                name_prefix=model_name,
            )
        )
    if realtime_callback is not None:
        callbacks.append(realtime_callback)
    learn_callback: Optional[BaseCallback]
    if len(callbacks) == 0:
        learn_callback = None
    elif len(callbacks) == 1:
        learn_callback = callbacks[0]
    else:
        learn_callback = CallbackList(callbacks)

    if cpg_env is not None:
        config_path = outdir / f"{model_name}_config.json"
        saved_config = cpg_env.config_dict()
        saved_config["warmup_exploration"] = args.warmup_exploration
        saved_config["learning_starts"] = args.learning_starts
        saved_config["warmup_parameter_low"] = (
            warmup_parameter_low.tolist() if warmup_parameter_low is not None else None
        )
        saved_config["warmup_parameter_high"] = (
            warmup_parameter_high.tolist() if warmup_parameter_high is not None else None
        )
        saved_config["target_mode"] = args.target_mode
        saved_config["task"] = args.task
        saved_config["target_distance_range_fraction"] = list(args.target_distance_range)
        saved_config["target_angle_range_deg"] = list(args.target_angle_range_deg)
        saved_config["target_radius_fraction"] = args.target_radius_fraction
        saved_config["forward_reward_weights"] = {
            "progress": args.forward_progress_weight,
            "lateral": args.forward_lateral_weight,
            "heading": args.forward_heading_weight,
        }
        saved_config["policy_observation_dim"] = int(np.prod(cpg_env.observation_space.shape))
        saved_config["seed"] = args.seed
        saved_config["checkpoint_every"] = args.checkpoint_every
        config_path.write_text(json.dumps(saved_config, indent=2), encoding="utf-8")
    model.learn(total_timesteps=args.total_steps, log_interval=10, callback=learn_callback)
    model.save(str(outdir / model_name))
    env.close()
    print(f"Saved: {outdir / (model_name + '.zip')}")



if __name__ == "__main__":
    main()
