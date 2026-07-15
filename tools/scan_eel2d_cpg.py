"""Coarse parameter scan for the real Eel2D MuJoCo-Warp/LBM coupling.

The scan bypasses SAC and evaluates fixed traveling-wave controls. Ranking uses
steady whole-body COM velocity estimated by linear regression, not endpoint head
displacement, so oscillator phase at the final frame does not bias the result.

Example:
    python tools/scan_eel2d_cpg.py --samples 24 --batch-size 4
"""

import argparse
import csv
import json
import math
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.lbm.eel.eel_lbm_env import Eel2DLBMEnv
from train_sac_minimal import set_lbm_viscosity


def make_candidates(args) -> List[Dict[str, float]]:
    """Latin-hypercube samples plus useful deterministic anchor points."""
    count = max(4, int(args.samples))
    rng = np.random.default_rng(args.seed)
    dimensions = []
    for _ in range(3):
        values = (np.arange(count, dtype=np.float32) + rng.random(count)) / count
        rng.shuffle(values)
        dimensions.append(values)

    amplitudes = args.amp_range[0] + dimensions[0] * (
        args.amp_range[1] - args.amp_range[0]
    )
    frequencies = args.freq_range[0] + dimensions[1] * (
        args.freq_range[1] - args.freq_range[0]
    )
    phase_pi = args.phase_pi_range[0] + dimensions[2] * (
        args.phase_pi_range[1] - args.phase_pi_range[0]
    )
    if args.directions == "both":
        directions = np.where(np.arange(count) % 2 == 0, -1.0, 1.0)
        rng.shuffle(directions)
    elif args.directions == "positive":
        directions = np.ones(count, dtype=np.float32)
    else:
        directions = -np.ones(count, dtype=np.float32)

    candidates = [
        {
            "candidate": int(i),
            "amplitude": float(amplitudes[i]),
            "frequency_hz": float(frequencies[i]),
            "spatial_phase_pi": float(phase_pi[i]),
            "direction": float(directions[i]),
        }
        for i in range(count)
    ]
    if args.include_baseline:
        candidates[0] = {
            "candidate": 0,
            "amplitude": 0.28,
            "frequency_hz": 2.0,
            "spatial_phase_pi": 0.825,
            "direction": -1.0,
        }
    if args.scan_preset == "fast":
        # configs/realtime_2d/eel2d.json uses omega=-1, omega_max=5*pi,
        # k_wave=.65 and k_max=1.5.  Its wave clock advances by .01 s per
        # outer step while that environment advances 8*.01 s of simulation,
        # so the equivalent physical frequency here is 2.5/8=.3125 Hz.
        candidates[0] = {
            "candidate": 0,
            "amplitude": 0.36,
            "frequency_hz": 0.3125,
            "spatial_phase_pi": 0.975,
            "direction": -1.0,
        }
        # The realtime preset drives position actuators.  Eel2D instead has
        # five low-geared torque motors, so also test a stronger amplitude at
        # the same temporal/spatial wave before relying on random samples.
        candidates[1] = {
            "candidate": 1,
            "amplitude": 0.80,
            "frequency_hz": 0.3125,
            "spatial_phase_pi": 0.975,
            "direction": -1.0,
        }
    return candidates


def solid_com_positions(env: Eel2DLBMEnv) -> np.ndarray:
    return np.asarray(
        [
            np.mean(
                env.solver.flows[w].solid_position.numpy()[: env.solid_num, :2],
                axis=0,
            )
            for w in range(env.nworld)
        ],
        dtype=np.float32,
    )


def regression_slope(values: np.ndarray, dt: float) -> np.ndarray:
    """Least-squares slope along axis 0 for values shaped (time, world, ...)."""
    sample_count = values.shape[0]
    times = np.arange(sample_count, dtype=np.float64) * float(dt)
    centered_time = times - np.mean(times)
    denominator = float(np.sum(centered_time * centered_time))
    centered_values = values - np.mean(values, axis=0, keepdims=True)
    return np.tensordot(centered_time, centered_values, axes=(0, 0)) / max(
        denominator, 1.0e-12
    )


def evaluate_batch(candidates: List[Dict[str, float]], args) -> List[Dict[str, float]]:
    nworld = len(candidates)
    env = Eel2DLBMEnv(
        nworld=nworld,
        nx=args.nx,
        ny=args.ny,
        lbm_scale=args.lbm_scale,
        per_frame_steps=args.per_frame_steps,
        max_episode_steps=args.warmup_steps + args.control_steps + 1,
        include_image=False,
        render_mode=None,
    )
    env.action_scale = float(args.action_scale)
    set_lbm_viscosity(env, args.viscosity)
    env.reset()

    zero_action = np.zeros((nworld, env.model.nu), dtype=np.float32)
    for _ in range(args.warmup_steps):
        env.step(zero_action)
    env.current_steps[...] = 0

    start = solid_com_positions(env)
    phases = np.zeros(nworld, dtype=np.float32)
    body_s = np.linspace(0.0, 1.0, env.model.nu, dtype=np.float32)
    envelope = 0.10 + 0.90 * body_s
    amplitudes = np.asarray([c["amplitude"] for c in candidates], dtype=np.float32)
    frequencies = np.asarray([c["frequency_hz"] for c in candidates], dtype=np.float32)
    spatial_phase = np.asarray(
        [c["spatial_phase_pi"] * np.pi for c in candidates], dtype=np.float32
    )
    directions = np.asarray([c["direction"] for c in candidates], dtype=np.float32)
    control_dt = float(env.mujoco_model.opt.timestep) * int(env.per_frame_steps)

    joint_min = np.full((nworld, env.model.nu), np.inf, dtype=np.float32)
    joint_max = np.full((nworld, env.model.nu), -np.inf, dtype=np.float32)
    effort_sum = np.zeros(nworld, dtype=np.float64)
    saturation_count = np.zeros(nworld, dtype=np.float64)
    fit_sample_count = np.zeros(nworld, dtype=np.int32)
    active = np.ones(nworld, dtype=bool)
    terminated_early = np.zeros(nworld, dtype=bool)
    executed_steps = np.zeros(nworld, dtype=np.int32)

    settle_steps = min(
        max(0, int(round(args.settle_seconds / control_dt))),
        max(0, args.control_steps - 3),
    )
    com_history = []
    yaw_history = []
    joint_limits = np.deg2rad(
        np.asarray([50.0, 55.0, 60.0, 65.0, 70.0], dtype=np.float32)
    )[: env.model.nu]

    for step_index in range(args.control_steps):
        actions = amplitudes[:, None] * envelope[None, :] * np.sin(
            phases[:, None] + spatial_phase[:, None] * body_s[None, :]
        )
        actions = np.asarray(np.clip(actions, -1.0, 1.0), dtype=np.float32)
        actions[~active] = 0.0
        _, _, done, _ = env.step(actions)

        qpos = env.data.qpos.numpy().astype(np.float32)
        joint_angles = qpos[:, 7 : 7 + env.model.nu]
        executed_steps += active.astype(np.int32)

        if step_index >= settle_steps:
            joint_min = np.minimum(joint_min, joint_angles)
            joint_max = np.maximum(joint_max, joint_angles)
            effort_sum += np.mean(actions * actions, axis=1) * active
            saturation_count += (
                np.any(np.abs(joint_angles) >= 0.90 * joint_limits[None, :], axis=1)
                * active
            )
            fit_sample_count += active.astype(np.int32)
            com_history.append(solid_com_positions(env))
            qw, qx, qy, qz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
            yaw_history.append(
                np.arctan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz),
                )
            )

        phases += directions * 2.0 * np.pi * frequencies * control_dt
        newly_done = np.asarray(done, dtype=bool) & active
        terminated_early |= newly_done
        active &= ~newly_done

    end = solid_com_positions(env)
    com_samples = np.asarray(com_history, dtype=np.float64)
    yaw_samples = np.unwrap(np.asarray(yaw_history, dtype=np.float64), axis=0)
    com_velocity = regression_slope(com_samples, control_dt)
    yaw_rate = regression_slope(yaw_samples, control_dt)
    joint_pp_deg = np.rad2deg(np.maximum(joint_max - joint_min, 0.0))

    forward_speed = com_velocity[:, 1]
    lateral_speed = com_velocity[:, 0]
    mean_effort = effort_sum / np.maximum(fit_sample_count, 1)
    saturation_fraction = saturation_count / np.maximum(fit_sample_count, 1)
    valid_gait = (
        (~terminated_early)
        & (forward_speed > args.min_forward_speed)
        & (
            np.abs(lateral_speed)
            <= np.maximum(
                args.max_lateral_ratio * forward_speed,
                args.min_forward_speed,
            )
        )
        & (np.abs(np.rad2deg(yaw_rate)) <= args.max_yaw_rate_deg_s)
        & (saturation_fraction <= args.max_saturation_fraction)
    )
    # Normalized steady velocities dominate. Invalid gaits remain in the CSV for
    # diagnosis but receive a large ranking penalty and never define OU ranges.
    score = (
        100.0 * forward_speed / float(args.ny)
        - 30.0 * np.abs(lateral_speed) / float(args.nx)
        - 0.10 * np.abs(yaw_rate)
        - 0.02 * mean_effort
        - 2.0 * saturation_fraction
        - 100.0 * (~valid_gait).astype(np.float32)
    )

    results: List[Dict[str, float]] = []
    for i, candidate in enumerate(candidates):
        row = dict(candidate)
        row.update(
            {
                "score": float(score[i]),
                "valid_gait": bool(valid_gait[i]),
                "forward_speed_lbm_s": float(forward_speed[i]),
                "lateral_speed_lbm_s": float(lateral_speed[i]),
                "yaw_rate_deg_s": float(np.rad2deg(yaw_rate[i])),
                "com_forward_displacement_lbm": float(end[i, 1] - start[i, 1]),
                "com_lateral_displacement_lbm": float(end[i, 0] - start[i, 0]),
                "mean_joint_pp_deg": float(np.mean(joint_pp_deg[i])),
                "tail_joint_pp_deg": float(joint_pp_deg[i, -1]),
                "mean_action_sq": float(mean_effort[i]),
                "joint_saturation_fraction": float(saturation_fraction[i]),
                "executed_steps": int(executed_steps[i]),
                "terminated_early": bool(terminated_early[i]),
            }
        )
        results.append(row)

    env.close()
    return results


def recommended_range(results: List[Dict[str, float]], top_k: int) -> Dict[str, object]:
    valid = [r for r in results if r["valid_gait"]]
    if not valid:
        return {
            "status": "no_valid_forward_gait",
            "amplitude": None,
            "frequency_hz": None,
            "spatial_phase_pi": None,
            "temporal_direction": None,
            "steering_bias": [0.0, 0.0],
            "top_k": 0,
        }

    best = max(valid, key=lambda row: row["score"])
    direction = float(best["direction"])
    same_direction = sorted(
        [r for r in valid if r["direction"] == direction],
        key=lambda row: row["score"],
        reverse=True,
    )[: max(1, top_k)]
    return {
        "status": "ok",
        "amplitude": [
            min(r["amplitude"] for r in same_direction),
            max(r["amplitude"] for r in same_direction),
        ],
        "frequency_hz": [
            min(r["frequency_hz"] for r in same_direction),
            max(r["frequency_hz"] for r in same_direction),
        ],
        "spatial_phase_pi": [
            min(r["spatial_phase_pi"] for r in same_direction),
            max(r["spatial_phase_pi"] for r in same_direction),
        ],
        "temporal_direction": direction,
        "steering_bias": [0.0, 0.0],
        "top_k": len(same_direction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan fixed CPG parameters in Eel2D LBM")
    parser.add_argument(
        "--scan-preset",
        choices=["broad", "fast"],
        default="broad",
        help=(
            "'fast' scans near configs/realtime_2d/eel2d.json's fast gait "
            "after converting its wave clock to physical simulation time"
        ),
    )
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--amp-range", type=float, nargs=2, default=[0.35, 1.0], metavar=("MIN", "MAX"))
    parser.add_argument("--freq-range", type=float, nargs=2, default=[0.30, 1.50], metavar=("MIN", "MAX"))
    parser.add_argument("--phase-pi-range", type=float, nargs=2, default=[0.30, 1.60], metavar=("MIN", "MAX"))
    parser.add_argument("--directions", choices=["both", "negative", "positive"], default="both")
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-steps", type=int, default=600)
    parser.add_argument("--settle-seconds", type=float, default=4.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--per-frame-steps", type=int, default=2)
    parser.add_argument("--nx", type=int, default=320)
    parser.add_argument("--ny", type=int, default=480)
    parser.add_argument("--lbm-scale", type=float, default=0.2)
    parser.add_argument("--viscosity", type=float, default=0.05)
    parser.add_argument("--action-scale", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--min-forward-speed", type=float, default=0.02, help="Minimum steady COM +y speed in LBM cells/s")
    parser.add_argument("--max-lateral-ratio", type=float, default=1.0, help="Maximum abs lateral/forward steady speed ratio")
    parser.add_argument("--max-yaw-rate-deg-s", type=float, default=15.0)
    parser.add_argument("--max-saturation-fraction", type=float, default=0.05)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("outputs/cpg_scan/eel2d_scan.csv"))
    args = parser.parse_args()

    if args.scan_preset == "fast":
        args.amp_range = [0.30, 1.00]
        args.freq_range = [0.20, 0.55]
        args.phase_pi_range = [0.75, 1.20]
        args.directions = "negative"
        args.include_baseline = False
        print(
            "Fast neighborhood: A=[0.30, 1.00], f=[0.20, 0.55] Hz, "
            "phase=[0.75, 1.20] pi, direction=-1; anchors A=0.36/0.80, "
            "f=0.3125 Hz, phase=0.975 pi"
        )

    candidates = make_candidates(args)
    results: List[Dict[str, float]] = []
    started = time.perf_counter()
    for start in range(0, len(candidates), max(1, args.batch_size)):
        batch = candidates[start : start + max(1, args.batch_size)]
        print(f"Evaluating candidates {start + 1}-{start + len(batch)} / {len(candidates)}")
        results.extend(evaluate_batch(batch, args))

    results.sort(key=lambda row: (row["valid_gait"], row["score"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    recommendation = recommended_range(results, args.top_k)
    valid_results = [row for row in results if row["valid_gait"]]
    diagnostic_best = results[0]
    recommendation.update(
        {
            "best": valid_results[0] if valid_results else None,
            "diagnostic_best": diagnostic_best,
            "samples": len(results),
            "valid_gaits": len(valid_results),
            "scan_preset": args.scan_preset,
            "control_steps": args.control_steps,
            "per_frame_steps": args.per_frame_steps,
            "settle_seconds": args.settle_seconds,
            "fit_seconds": max(0.0, args.control_steps * args.per_frame_steps * 0.01 - args.settle_seconds),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    print("\nTop candidates")
    print("rank  ok  amp   freq   phase/pi  dir  vfwd     vlat     tail_pp  score")
    for rank, row in enumerate(results[: args.top_k], start=1):
        print(
            f"{rank:>4}  {int(row['valid_gait'])}   {row['amplitude']:.3f}  {row['frequency_hz']:.3f}  "
            f"{row['spatial_phase_pi']:.3f}    {row['direction']:+.0f}  "
            f"{row['forward_speed_lbm_s']:+.4f}  {row['lateral_speed_lbm_s']:+.4f}  "
            f"{row['tail_joint_pp_deg']:.2f}   {row['score']:+.4f}"
        )
    print(f"\nCSV:  {args.output}")
    print(f"JSON: {json_path}")
    print("Recommended range:", json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
