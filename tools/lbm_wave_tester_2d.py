"""
2D LBM Wave Tester — preset open-loop controls for FishLBMEnv.

Runs simple preset joint-control signals through the real 2D MuJoCo-Warp + LBM
coupled simulator and exports an MP4 video plus CSV metrics. No trained agent is
required.

Examples:
    python tools/lbm_wave_tester_2d.py --preset forward --steps 300
    python tools/lbm_wave_tester_2d.py --preset all --steps 300 --render-type vorticity
    python tools/lbm_wave_tester_2d.py --preset forward --amp 0.8 --freq 1.2 --phase-lag 1.57
"""

import argparse
import csv
import math
import pathlib
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import warp as wp

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# Add project root to path when running as: python tools/lbm_wave_tester_2d.py
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.lbm import FishLBMEnv
from envs.lbm.lbm_func import get_solid_boundary_img, get_u_img, get_vorticity_with_solid_img


FISH_PRESETS: Dict[str, Dict[str, float]] = {
    # joint1 and joint2 use a traveling-wave-like phase offset.
    # Tail-dominant gaits: smaller front joint, stronger tail joint.
    "forward": dict(amp=0.65, freq=2.4, phase_lag=1.35, bias1=0.0, bias2=0.0, tail_ratio=1.46),
    "slow": dict(amp=0.45, freq=1.5, phase_lag=1.35, bias1=0.0, bias2=0.0, tail_ratio=1.56),
    "fast": dict(amp=0.75, freq=3.1, phase_lag=1.35, bias1=0.0, bias2=0.0, tail_ratio=1.33),
    "turn_left": dict(amp=0.55, freq=2.0, phase_lag=1.25, bias1=0.28, bias2=0.18, tail_ratio=1.55),
    "turn_right": dict(amp=0.55, freq=2.0, phase_lag=1.25, bias1=-0.28, bias2=-0.18, tail_ratio=1.55),
    "idle": dict(amp=0.0, freq=1.0, phase_lag=1.20, bias1=0.0, bias2=0.0, tail_ratio=1.0),
}



def make_action(step: int, dt: float, params: Dict[str, float], warmup_steps: int) -> np.ndarray:
    """Return action with shape (1, 2) for FishLBMEnv."""
    t = step * dt
    amp = float(params["amp"])
    freq = float(params["freq"])
    phase_lag = float(params["phase_lag"])
    bias1 = float(params["bias1"])
    bias2 = float(params["bias2"])
    tail_ratio = float(params["tail_ratio"])

    phase = 2.0 * math.pi * freq * t
    ramp = min(1.0, (step + 1) / max(1, warmup_steps))

    joint1 = bias1 + amp * math.sin(phase)
    joint2 = bias2 + amp * tail_ratio * math.sin(phase + phase_lag)
    action = np.array([[joint1, joint2]], dtype=np.float32) * ramp
    return np.clip(action, -1.0, 1.0)


def get_raw_frame_2d(env: FishLBMEnv, render_type: str = "vorticity", world_idx: int = 0) -> np.ndarray:
    """Read raw LBM field as (ny, nx). Vorticity mode includes solid overlay marker."""
    flow = env.solver.flows[world_idx]

    if render_type == "vorticity":
        wp.launch(get_vorticity_with_solid_img, dim=(flow.nx, flow.ny), inputs=[flow, 1.0])
    elif render_type == "solid_boundary":
        wp.launch(get_solid_boundary_img, dim=(flow.nx, flow.ny), inputs=[flow, 1.0])
    else:
        wp.launch(get_u_img, dim=(flow.nx, flow.ny), inputs=[flow])

    wp.synchronize()
    raw = flow.u_img.numpy().T
    return np.flipud(raw)


def raw_to_rgb(raw: np.ndarray, vmax: float, render_type: str = "vorticity") -> np.ndarray:
    """Convert raw scalar field to RGB uint8 frame."""
    import matplotlib.pyplot as plt

    if render_type == "solid_boundary":
        img = np.clip(raw, 0.0, 1.0)
        rgb = (plt.get_cmap("gray_r")(img)[:, :, :3] * 255).astype(np.uint8)
        return rgb

    if render_type == "vorticity":
        solid_mask = raw >= 999.0
        fluid = raw.copy()
        fluid[solid_mask] = 0.0
        normalized = np.clip((fluid / max(vmax, 1e-8) + 1.0) * 0.5, 0.0, 1.0)
        rgb = (plt.get_cmap("RdBu_r")(normalized)[:, :, :3] * 255).astype(np.uint8)
        rgb[solid_mask] = np.array([200, 200, 200], dtype=np.uint8)
        return rgb

    normalized = np.clip(raw / max(vmax, 1e-8), 0.0, 1.0)
    return (plt.get_cmap("magma")(normalized)[:, :, :3] * 255).astype(np.uint8)


def draw_overlay(frame: np.ndarray, preset: str, step: int, action: np.ndarray, reward: float) -> np.ndarray:
    """Draw status text and two action bars on an RGB frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    lines = [
        f"preset: {preset}",
        f"step: {step}",
        f"action: [{action[0, 0]:+.2f}, {action[0, 1]:+.2f}]",
        f"reward: {reward:+.4f}",
    ]
    x, y0 = 12, 24
    for i, text in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    # Draw action bars near bottom-left. Values in [-1, 1].
    bar_x, bar_y = 14, h - 54
    bar_w, bar_h = min(180, max(80, w // 3)), 12
    for idx, val in enumerate([float(action[0, 0]), float(action[0, 1])]):
        y = bar_y + idx * 22
        cv2.rectangle(out, (bar_x, y), (bar_x + bar_w, y + bar_h), (35, 35, 35), 1)
        mid = bar_x + bar_w // 2
        cv2.line(out, (mid, y), (mid, y + bar_h), (220, 220, 220), 1)
        end = int(mid + np.clip(val, -1.0, 1.0) * (bar_w // 2))
        color = (80, 220, 120) if val >= 0 else (230, 120, 80)
        cv2.rectangle(out, (min(mid, end), y + 2), (max(mid, end), y + bar_h - 2), color, -1)
        cv2.putText(out, f"j{idx + 1}", (bar_x + bar_w + 8, y + bar_h), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def save_video(frames: List[np.ndarray], output_path: pathlib.Path, fps: int) -> None:
    if not frames:
        print("No frames to save.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Video saved to: {output_path}")


def save_metrics_csv(path: pathlib.Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Metrics saved to: {path}")


def run_preset(args: argparse.Namespace, preset_name: str, preset_params: Dict[str, float]) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    print(f"\n=== Running preset: {preset_name} ===")
    print(f"params: {preset_params}")

    env = FishLBMEnv(
        nx=args.nx,
        ny=args.ny,
        lbm_scale=args.lbm_scale,
        nworld=1,
        max_episode_steps=args.steps + 1,
        per_frame_steps=args.per_frame_steps,
        include_image=False,
        render_mode=None,
    )
    env.reset()
    initial_head_pos = env.solver.flows[0].solid_position.numpy()[0].copy()

    raw_frames: List[np.ndarray] = []

    actions: List[np.ndarray] = []
    metrics: List[Dict[str, float]] = []
    total_reward = 0.0

    iterator = range(args.steps)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"{preset_name}", leave=False)

    start = time.time()
    dt = args.dt
    done = np.array([False])

    for step in iterator:
        action = make_action(step, dt, preset_params, args.warmup_steps)
        _obs, reward, done, _info = env.step(action)
        reward_val = float(reward[0])
        total_reward += reward_val
        head_pos = env.solver.flows[0].solid_position.numpy()[0].copy()
        dx = float(head_pos[0] - initial_head_pos[0])
        dy = float(head_pos[1] - initial_head_pos[1])

        if step % args.render_every == 0:
            raw_frames.append(get_raw_frame_2d(env, args.render_type, world_idx=0))
            actions.append(action.copy())

        metrics.append(
            {
                "step": step,
                "time": step * dt,
                "action_joint1": float(action[0, 0]),
                "action_joint2": float(action[0, 1]),
                "head_x": float(head_pos[0]),
                "head_y": float(head_pos[1]),
                "dx_from_start": dx,
                "dy_from_start": dy,
                "reward": reward_val,
                "total_reward": total_reward,
                "done": int(bool(done[0])),
            }
        )


        if bool(done[0]) and not args.ignore_done:
            print(f"Episode done at step {step}.")
            break

    elapsed = time.time() - start
    final_head_pos = env.solver.flows[0].solid_position.numpy()[0].copy()
    final_dx = float(final_head_pos[0] - initial_head_pos[0])
    final_dy = float(final_head_pos[1] - initial_head_pos[1])
    print(
        f"Finished {len(metrics)} steps in {elapsed:.2f}s, "
        f"total_reward={total_reward:.4f}, dx={final_dx:.3f}, dy={final_dy:.3f}"
    )

    if not raw_frames:

        return [], metrics

    all_raw = np.stack(raw_frames)
    if args.render_type == "vorticity":
        fluid_mask = all_raw < 999.0
        vmax = np.max(np.abs(all_raw[fluid_mask])) * args.vmax_scale + 1e-8 if np.any(fluid_mask) else 1.0
    elif args.render_type == "solid_boundary":
        vmax = 1.0
    else:
        vmax = np.max(all_raw) * args.vmax_scale + 1e-8

    frames = []
    for i, raw in enumerate(raw_frames):
        frame = raw_to_rgb(raw, vmax, args.render_type)
        render_step = i * args.render_every
        action = actions[i]
        reward_val = metrics[min(render_step, len(metrics) - 1)]["reward"]
        frame = draw_overlay(frame, preset_name, render_step, action, reward_val)
        frames.append(frame)

    return frames, metrics


def parse_override(text: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not text:
        return result
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        result[key.strip()] = float(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="2D Fish preset-action LBM video exporter")
    parser.add_argument("--preset", type=str, default="forward", help=f"Preset name or 'all'. Choices: {list(FISH_PRESETS.keys())}")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--render-type", type=str, default="vorticity", choices=["velocity", "vorticity", "solid_boundary"])
    parser.add_argument("--output-dir", type=str, default="outputs/lbm_wave_test_2d/fish")
    parser.add_argument("--output", type=str, default=None, help="Optional output mp4 path for a single preset")

    # Environment options
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=600)
    parser.add_argument("--lbm-scale", type=float, default=0.2)
    parser.add_argument("--per-frame-steps", type=int, default=10)


    # Control options. If omitted, values come from the selected preset.
    parser.add_argument("--amp", type=float, default=None)
    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--phase-lag", type=float, default=None)
    parser.add_argument("--bias1", type=float, default=None)
    parser.add_argument("--bias2", type=float, default=None)
    parser.add_argument("--tail-ratio", type=float, default=None)
    parser.add_argument("--override", type=str, default=None, help="Override preset params: 'amp=0.8,freq=1.2,...'")
    parser.add_argument("--dt", type=float, default=0.01, help="Control signal time step used by waveform")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--vmax-scale", type=float, default=0.2)
    parser.add_argument("--ignore-done", action="store_true", help="Continue even if environment reports done")

    args = parser.parse_args()

    if args.preset == "all":
        preset_names = list(FISH_PRESETS.keys())
    else:
        if args.preset not in FISH_PRESETS:
            raise ValueError(f"Unknown preset '{args.preset}'. Choices: {list(FISH_PRESETS.keys())} or 'all'")
        preset_names = [args.preset]

    output_dir = pathlib.Path(args.output_dir)
    for preset_name in preset_names:
        params = dict(FISH_PRESETS[preset_name])
        for key in ["amp", "freq", "phase_lag", "bias1", "bias2", "tail_ratio"]:
            cli_value = getattr(args, key)
            if cli_value is not None:
                params[key] = float(cli_value)
        params.update(parse_override(args.override))

        frames, metrics = run_preset(args, preset_name, params)

        if args.output and len(preset_names) == 1:
            video_path = pathlib.Path(args.output)
        else:
            video_path = output_dir / f"{preset_name}_{args.render_type}.mp4"
        csv_path = output_dir / f"{preset_name}_{args.render_type}.csv"

        save_video(frames, video_path, args.fps)
        save_metrics_csv(csv_path, metrics)


if __name__ == "__main__":
    main()
