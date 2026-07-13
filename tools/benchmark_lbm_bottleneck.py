"""Benchmark 2D/3D LBM bottlenecks: simulation vs GPU readback/render vs disk writes.

This script avoids MuJoCo window rendering and focuses on the coupled LBM step,
LBM visualization readback, CPU frame conversion, and video/CSV writes.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
import warp as wp

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))


@dataclass
class SectionTimer:
    totals: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, name: str, elapsed: float, count: int = 1) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + count

    def timed(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                timer.add(name, time.perf_counter() - self.start)

        return _Ctx()


def write_csv(path: pathlib.Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_video(path: pathlib.Path, frames: List[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def raw_frames_to_rgb(raw_frames: List[np.ndarray], render_type: str, is_3d: bool) -> List[np.ndarray]:
    if not raw_frames:
        return []
    all_raw = np.stack(raw_frames)
    if render_type == "vorticity":
        fluid_mask = all_raw < 999.0
        vmax = np.max(np.abs(all_raw[fluid_mask])) * 0.2 + 1e-8 if np.any(fluid_mask) else 1.0
    else:
        vmax = np.max(all_raw) * 0.2 + 1e-8
    if is_3d:
        from lbm3d_runtime import process_raw_to_frame

        return [process_raw_to_frame(raw, vmax, render_type) for raw in raw_frames]
    from lbm_wave_tester_2d import raw_to_rgb


    return [raw_to_rgb(raw, vmax, render_type) for raw in raw_frames]


def benchmark_2d(args: argparse.Namespace) -> Tuple[SectionTimer, int, pathlib.Path]:
    from envs.lbm.fish.fish_lbm_env import FishLBMEnv
    from lbm_wave_tester_2d import FISH_PRESETS, get_raw_frame_2d, make_action

    out_dir = pathlib.Path(args.output_dir)
    env = FishLBMEnv(
        nx=args.nx2d,
        ny=args.ny2d,
        lbm_scale=args.lbm_scale2d,
        nworld=1,
        max_episode_steps=args.steps2d + args.warmup + 10,
        per_frame_steps=args.per_frame_steps2d,
        include_image=False,
        render_mode=None,
    )
    env.reset()
    params = dict(FISH_PRESETS[args.preset2d])
    timer = SectionTimer()

    for i in range(args.warmup):
        action = make_action(i, args.dt2d, params, max(1, args.warmup))
        env.step(action)
    wp.synchronize()

    raw_frames: List[np.ndarray] = []
    rows: List[Dict[str, float]] = []
    flow = env.solver.flows[0]
    loop_start = time.perf_counter()
    for step in range(args.steps2d):
        action = make_action(step, args.dt2d, params, max(1, args.warmup))
        with timer.timed("sim_step"):
            _obs, reward, done, _info = env.step(action)
            wp.synchronize()
        with timer.timed("state_readback"):
            head = flow.solid_position.numpy()[0].copy()
        if args.render_every2d > 0 and step % args.render_every2d == 0:
            with timer.timed("lbm_render_readback"):
                raw_frames.append(get_raw_frame_2d(env, args.render_type2d, world_idx=0).copy())
        rows.append({"step": step, "reward": float(reward[0]), "head_x": float(head[0]), "head_y": float(head[1]), "done": int(bool(done[0]))})
    timer.add("loop_total", time.perf_counter() - loop_start)

    with timer.timed("frame_convert"):
        frames = raw_frames_to_rgb(raw_frames, args.render_type2d, is_3d=False)
    with timer.timed("video_write"):
        write_video(out_dir / "bench_2d.mp4", frames, args.fps)
    with timer.timed("csv_write"):
        write_csv(out_dir / "bench_2d.csv", rows)
    return timer, len(raw_frames), out_dir


def benchmark_3d(args: argparse.Namespace) -> Tuple[SectionTimer, int, pathlib.Path]:
    from envs.lbm3d.manta.manta_multitask_env_3d import TASK_NAMES
    from lbm3d_runtime import get_raw_frame_3d, load_named_config, make_multitask_env
    from lbm_wave_tester import ANIMAL_REGISTRY


    out_dir = pathlib.Path(args.output_dir)
    info = ANIMAL_REGISTRY[args.animal3d]
    overrides = {
        "lbm_nx": args.nx3d,
        "lbm_ny": args.ny3d,
        "lbm_nz": args.nz3d,
        "per_frame_steps": args.per_frame_steps3d,
        "lbm_scale": args.lbm_scale3d,
        "time_limit": args.steps3d + args.warmup + 20,
        "envs": 1,
    }
    config = load_named_config(["defaults", *info["configs"]], overrides=overrides)
    env = make_multitask_env(config, nworld=1)
    env.reset()
    base_env = env._env
    task_id = TASK_NAMES.index("forward")
    base_env._task_ids[:] = task_id
    base_env._update_task_ids_wp()

    action_target = info["preset_to_action"](info["presets"][args.preset3d]).reshape(1, -1).astype(np.float32)
    timer = SectionTimer()

    for i in range(args.warmup):
        ramp = min(1.0, (i + 1) / max(1, args.warmup))
        env.step(np.clip(action_target * ramp, -1.0, 1.0).astype(np.float32))
    wp.synchronize()

    raw_frames: List[np.ndarray] = []
    rows: List[Dict[str, float]] = []
    flow = base_env.lbm_solver.flows[0]
    loop_start = time.perf_counter()
    for step in range(args.steps3d):
        ramp = min(1.0, (step + 1) / max(1, args.warmup))
        action = np.clip(action_target * ramp, -1.0, 1.0).astype(np.float32)
        with timer.timed("sim_step"):
            _obs, rewards, dones, _infos = env.step(action)
            wp.synchronize()
        with timer.timed("state_readback"):
            qpos = base_env.mjw_data.qpos.numpy()[0].copy()
            qvel = base_env.mjw_data.qvel.numpy()[0].copy()
            pos = flow.solid_position.numpy()[0].copy()
        if args.render_every3d > 0 and step % args.render_every3d == 0:
            with timer.timed("lbm_render_readback"):
                raw_frames.append(get_raw_frame_3d(env, world_idx=0, render_type=args.render_type3d, view_mode=args.view_mode3d).copy())
        rows.append({"step": step, "reward": float(rewards[0]), "pos_x": float(pos[0]), "pos_y": float(pos[1]), "pos_z": float(pos[2]), "qpos0": float(qpos[0]), "qvel0": float(qvel[0]), "done": int(bool(dones[0]))})
    timer.add("loop_total", time.perf_counter() - loop_start)

    with timer.timed("frame_convert"):
        frames = raw_frames_to_rgb(raw_frames, args.render_type3d, is_3d=True)
    with timer.timed("video_write"):
        write_video(out_dir / "bench_3d.mp4", frames, args.fps)
    with timer.timed("csv_write"):
        write_csv(out_dir / "bench_3d.csv", rows)
    return timer, len(raw_frames), out_dir


def print_report(name: str, timer: SectionTimer, steps: int, frames: int, out_dir: pathlib.Path) -> None:
    print(f"\n=== {name} bottleneck report ===")
    print(f"steps={steps}, frames={frames}, output_dir={out_dir}")
    loop_total = timer.totals.get("loop_total", 0.0)
    grand_total = sum(v for k, v in timer.totals.items() if k != "loop_total")
    for key in ["sim_step", "state_readback", "lbm_render_readback", "frame_convert", "video_write", "csv_write"]:
        value = timer.totals.get(key, 0.0)
        count = timer.counts.get(key, 0)
        avg_ms = 1000.0 * value / max(1, count)
        denom = grand_total if grand_total > 0 else loop_total
        pct = 100.0 * value / max(denom, 1e-9)
        print(f"{key:>20s}: total={value:8.3f}s  avg={avg_ms:8.2f}ms  share={pct:6.2f}%  count={count}")
    print(f"{'loop_total':>20s}: total={loop_total:8.3f}s  avg={1000.0 * loop_total / max(1, steps):8.2f}ms/step")
    if loop_total > 0:
        print(f"{'loop_fps':>20s}: {steps / loop_total:8.3f} steps/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile 2D/3D LBM simulation vs readback/render/write bottlenecks")
    parser.add_argument("--mode", choices=["2d", "3d", "both"], default="both")
    parser.add_argument("--output-dir", default="outputs/_bench_bottleneck")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=2)

    parser.add_argument("--steps2d", type=int, default=30)
    parser.add_argument("--nx2d", type=int, default=400)
    parser.add_argument("--ny2d", type=int, default=600)
    parser.add_argument("--lbm-scale2d", type=float, default=0.2)
    parser.add_argument("--per-frame-steps2d", type=int, default=10)
    parser.add_argument("--preset2d", default="forward")
    parser.add_argument("--dt2d", type=float, default=0.01)
    parser.add_argument("--render-every2d", type=int, default=1, help="0 disables LBM visualization readback")
    parser.add_argument("--render-type2d", default="vorticity", choices=["velocity", "vorticity", "solid_boundary"])

    parser.add_argument("--steps3d", type=int, default=8)
    parser.add_argument("--animal3d", default="eel")
    parser.add_argument("--preset3d", default="forward")
    parser.add_argument("--nx3d", type=int, default=150)
    parser.add_argument("--ny3d", type=int, default=250)
    parser.add_argument("--nz3d", type=int, default=60)
    parser.add_argument("--lbm-scale3d", type=float, default=0.5)
    parser.add_argument("--per-frame-steps3d", type=int, default=10)
    parser.add_argument("--render-every3d", type=int, default=1, help="0 disables LBM projection readback")
    parser.add_argument("--render-type3d", default="vorticity", choices=["velocity", "vorticity"])
    parser.add_argument("--view-mode3d", default="topdown", choices=["topdown", "max_topdown", "side", "front"])
    args = parser.parse_args()

    if args.mode in ("2d", "both"):
        timer, frames, out_dir = benchmark_2d(args)
        print_report("2D", timer, args.steps2d, frames, out_dir)
    if args.mode in ("3d", "both"):
        timer, frames, out_dir = benchmark_3d(args)
        print_report("3D", timer, args.steps3d, frames, out_dir)


if __name__ == "__main__":
    main()
