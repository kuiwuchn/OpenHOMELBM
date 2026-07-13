"""
3D realtime LBM + MuJoCo control demo.

Uses the existing 3D multitask environments and preset wave actions from
`tools/lbm_wave_tester.py`. The window shows MuJoCo OpenGL rendering and,
optionally, a 3D LBM projection side-by-side.

Examples:
    python tools/lbm3d_realtime_control.py --animal eel --with-lbm
    python tools/lbm3d_realtime_control.py --animal turtle --preset forward
    python tools/lbm3d_realtime_control.py --animal tuna --record outputs/tuna_realtime.mp4

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
import pathlib
import sys
import time
from typing import Dict, Optional

import cv2
import mujoco
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
    load_named_config,
    make_multitask_env,
    process_raw_to_frame,
)
from lbm_wave_tester import ANIMAL_REGISTRY



TASK_BY_PRESET = {
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


def choose_idle_preset(presets: Dict[str, dict]) -> str:
    for name in ("freeze", "glide", "idle", "forward"):
        if name in presets:
            return name
    return next(iter(presets))


def build_keymap(presets: Dict[str, dict]) -> Dict[int, str]:
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
        "W forward | A left | D right | F fast | S idle | Z ascend | C descend",
        "Space pause | R reset | Q/Esc quit",
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


def make_env(animal: str, nworld: int, overrides: dict):
    animal_info = ANIMAL_REGISTRY[animal]
    config = load_named_config(["defaults", *animal_info["configs"]], overrides=overrides)
    return make_multitask_env(config, nworld=nworld)


def main() -> None:
    parser = argparse.ArgumentParser(description="3D realtime LBM + MuJoCo OpenGL control demo")
    parser.add_argument("--animal", type=str, default="eel", choices=list(ANIMAL_REGISTRY.keys()))
    parser.add_argument("--preset", type=str, default=None, help="Initial preset; default uses freeze/glide/forward fallback")
    parser.add_argument("--with-lbm", action="store_true", help="Show LBM 3D projection next to MuJoCo render")
    parser.add_argument("--render-type", type=str, default="vorticity", choices=["velocity", "vorticity"])
    parser.add_argument("--view-mode", type=str, default="topdown", choices=["topdown", "max_topdown", "side", "front"])
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
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--transition-steps", type=int, default=30, help="Smooth blending steps when switching presets")
    parser.add_argument("--window-name", type=str, default="LBM3D Realtime Control")

    parser.add_argument("--record", type=str, default=None)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--no-render", action="store_true", help="Run coupled LBM simulation without MuJoCo/LBM rendering and print sim FPS")
    parser.add_argument("--benchmark-steps", type=int, default=300, help="Number of steps for --no-render benchmark")
    parser.add_argument("--benchmark-progress-every", type=int, default=10, help="Print progress every N steps in --no-render mode")
    parser.add_argument("--dry-run", action="store_true", help="Load presets/config only and exit before creating env")


    args = parser.parse_args()

    animal_info = ANIMAL_REGISTRY[args.animal]
    presets = animal_info["presets"]
    preset_to_action = animal_info["preset_to_action"]
    keymap = build_keymap(presets)

    mode = args.preset or choose_idle_preset(presets)
    if mode not in presets:
        raise ValueError(f"Unknown preset '{mode}'. Choices: {list(presets.keys())}")

    if args.dry_run:
        print(f"animal={args.animal}, configs={animal_info['configs']}, presets={list(presets.keys())}")
        return

    overrides = {}
    for key in ("nx", "ny", "nz", "per_frame_steps", "lbm_scale"):
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            overrides[f"lbm_{key}" if key in ("nx", "ny", "nz") else key] = value

    env = make_env(args.animal, nworld=1, overrides=overrides)
    obs = env.reset()
    del obs

    base_env = env._env

    if args.no_render:
        task = TASK_BY_PRESET.get(mode, "forward")
        set_task_if_supported(env, task)
        action_target = preset_to_action(presets[mode]).reshape(1, -1).astype(np.float32)
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
    last_action = np.zeros_like(preset_to_action(presets[mode]).reshape(1, -1), dtype=np.float32)
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

            task = TASK_BY_PRESET.get(mode, "forward")
            if not paused:
                set_task_if_supported(env, task)
                action_target = preset_to_action(presets[mode]).reshape(1, -1)
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
