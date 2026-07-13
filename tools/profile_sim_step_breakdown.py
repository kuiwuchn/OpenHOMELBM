"""Fine-grained profiling for LBM/MuJoCo coupled simulation step."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from dataclasses import dataclass, field
from types import MethodType
from typing import Dict

import numpy as np
import warp as wp
import mujoco_warp as mjw

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))


@dataclass
class Profiler:
    totals: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, name: str, elapsed: float, count: int = 1) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + count

    def time_sync(self, name: str, fn, count: int = 1):
        start = time.perf_counter()
        result = fn()
        wp.synchronize()
        self.add(name, time.perf_counter() - start, count)
        return result


def install_2d_profiler(env, profiler: Profiler):
    from envs.lbm.lbm_fluid_env_func import (
        convert_and_update_solid_batch_2d,
        extract_body_states,
        extract_forces_torques_batch,
        fill_xfrc_kernel,
    )
    from envs.lbm.lbm_func import (
        Swap_Mom,
        apply_bc,
        init_force,
        precompute_transformed_segments,
        stream_and_collide,
    )


    def profiled_simulation_step(self):
        n_bodies = len(self.solid_config)
        start_total = time.perf_counter()
        for _ in range(self.per_frame_steps):
            xipos_full = self.data.xipos
            xquat_full = self.data.xquat

            profiler.time_sync(
                "extract_body_states",
                lambda: wp.launch(
                    extract_body_states,
                    dim=(self.nworld, n_bodies),
                    inputs=[xipos_full, xquat_full, self.body_ids_wp, self.positions_buffer, self.quaternions_buffer],
                ),
            )
            profiler.time_sync(
                "update_lbm_solid",
                lambda: wp.launch(
                    convert_and_update_solid_batch_2d,
                    dim=(self.nworld, n_bodies),
                    inputs=[
                        self.solver.flows_wp,
                        self.solid_ids_wp,
                        self.positions_buffer,
                        self.quaternions_buffer,
                        self.solver.mujoco_origins_wp,
                        self.solver.lbm_origins_wp,
                        self.solver.scales_wp,
                    ],
                ),
            )
            def lbm_init_force():
                for flow in self.solver.flows:
                    init_force(flow)

            profiler.time_sync("lbm_init_force", lbm_init_force)
            profiler.time_sync(
                "lbm_precompute_segments",
                lambda: wp.launch(
                    precompute_transformed_segments,
                    dim=(self.nworld, self.solver.flows[0].n_objects, self.solver.flows[0].max_segments_per_object),
                    inputs=[self.solver.flows_wp],
                    device=self.solver.device,
                ),
            )
            profiler.time_sync(
                "lbm_stream_collide",
                lambda: wp.launch(
                    stream_and_collide,
                    dim=(self.nworld, self.solver.nx, self.solver.ny),
                    inputs=[self.solver.flows_wp],
                    device=self.solver.device,
                ),
            )
            profiler.time_sync(
                "lbm_apply_bc",
                lambda: wp.launch(
                    apply_bc,
                    dim=(self.nworld, self.solver.nx, self.solver.ny),
                    inputs=[self.solver.flows_wp],
                    device=self.solver.device,
                ),
            )
            profiler.time_sync(
                "lbm_swap",
                lambda: wp.launch(Swap_Mom, dim=(self.nworld,), inputs=[self.solver.flows_wp], device=self.solver.device),
            )

            if not hasattr(self, "forces_buffer"):

                self.forces_buffer = wp.zeros((self.nworld, n_bodies, 3), dtype=wp.float32)
                self.torques_buffer = wp.zeros((self.nworld, n_bodies, 3), dtype=wp.float32)

            profiler.time_sync(
                "extract_fluid_forces",
                lambda: wp.launch(
                    extract_forces_torques_batch,
                    dim=(self.nworld, n_bodies),
                    inputs=[self.solver.flows_wp, self.solid_ids_wp, self.solver.scales_wp, self.forces_buffer, self.torques_buffer],
                ),
            )
            profiler.time_sync("zero_xfrc", self.data.xfrc_applied.zero_)
            profiler.time_sync(
                "fill_xfrc",
                lambda: wp.launch(
                    fill_xfrc_kernel,
                    dim=(self.nworld, n_bodies),
                    inputs=[self.data.xfrc_applied, self.body_ids_wp, self.forces_buffer, self.torques_buffer],
                ),
            )
            profiler.time_sync("zero_qfrc", self.data.qfrc_applied.zero_)
            profiler.time_sync("xfrc_accumulate", lambda: mjw.xfrc_accumulate(self.model, self.data, self.data.qfrc_applied))

            def mujoco_step():
                if not self.graph_initialized:
                    with wp.ScopedCapture() as capture:
                        mjw.step(self.model, self.data)
                    self.graph_initialized = True
                    self.mujoco_single_step_graph = capture.graph
                else:
                    wp.capture_launch(self.mujoco_single_step_graph)

            profiler.time_sync("mujoco_step", mujoco_step)
        profiler.add("simulation_step_total", time.perf_counter() - start_total)

    env._simulation_step = MethodType(profiled_simulation_step, env)


def install_3d_profiler(env, profiler: Profiler):
    from envs.lbm3d.lbm_fluid_env_3d_func import (
        convert_and_update_solid_batch_3d,
        extract_body_states_3d,
        extract_forces_torques_physical_3d,
        fill_xfrc_3d_kernel,
    )
    from envs.lbm3d.lbm_func_3d import (
        Swap_Mom_3D,
        apply_bc_3d,
        init_force_3d_batch,
        stream_and_collide_3d,
    )


    def profiled_simulation_step(self):
        n_all = len(self.link_config)
        start_total = time.perf_counter()
        for _ in range(self.per_frame_steps):
            xipos_full = self.mjw_data.xipos
            xquat_full = self.mjw_data.xquat

            profiler.time_sync(
                "extract_body_states",
                lambda: wp.launch(
                    extract_body_states_3d,
                    dim=(self.nworld, n_all),
                    inputs=[xipos_full, xquat_full, self.body_ids_wp, self.positions_buffer, self.quaternions_buffer],
                    device=self.device,
                ),
            )
            profiler.time_sync(
                "update_lbm_solid",
                lambda: wp.launch(
                    convert_and_update_solid_batch_3d,
                    dim=(self.nworld, n_all),
                    inputs=[
                        self.lbm_solver.flows_wp,
                        self.solid_ids_wp,
                        self.positions_buffer,
                        self.quaternions_buffer,
                        self._mujoco_origins_wp,
                        self._lbm_origins_wp,
                        self._scales_wp,
                    ],
                    device=self.device,
                ),
            )
            profiler.time_sync(
                "lbm_init_force",
                lambda: wp.launch(
                    init_force_3d_batch,
                    dim=(self.lbm_solver.nworld,),
                    inputs=[self.lbm_solver.flows_wp],
                    device=self.lbm_solver.device,
                ),
            )
            profiler.time_sync(
                "lbm_stream_collide",
                lambda: wp.launch(
                    stream_and_collide_3d,
                    dim=(self.lbm_solver.nworld, self.lbm_solver.nx, self.lbm_solver.ny, self.lbm_solver.nz),
                    inputs=[self.lbm_solver.flows_wp],
                    device=self.lbm_solver.device,
                ),
            )
            profiler.time_sync(
                "lbm_apply_bc",
                lambda: wp.launch(
                    apply_bc_3d,
                    dim=(self.lbm_solver.nworld, self.lbm_solver.nx, self.lbm_solver.ny, self.lbm_solver.nz),
                    inputs=[self.lbm_solver.flows_wp],
                    device=self.lbm_solver.device,
                ),
            )
            profiler.time_sync(
                "lbm_swap",
                lambda: wp.launch(Swap_Mom_3D, dim=(self.lbm_solver.nworld,), inputs=[self.lbm_solver.flows_wp], device=self.lbm_solver.device),
            )

            if self.n_dynamic > 0:

                profiler.time_sync(
                    "extract_fluid_forces",
                    lambda: wp.launch(
                        extract_forces_torques_physical_3d,
                        dim=(self.nworld, self.n_dynamic),
                        inputs=[
                            self.lbm_solver.flows_wp,
                            self.dynamic_solid_ids_wp,
                            self.force_conversion,
                            self.torque_conversion,
                            self.forces_buffer,
                            self.torques_buffer,
                        ],
                        device=self.device,
                    ),
                )
            profiler.time_sync("zero_xfrc", self.mjw_data.xfrc_applied.zero_)
            if self.n_dynamic > 0:
                profiler.time_sync(
                    "fill_xfrc",
                    lambda: wp.launch(
                        fill_xfrc_3d_kernel,
                        dim=(self.nworld, self.n_dynamic),
                        inputs=[self.mjw_data.xfrc_applied, self.dynamic_body_ids_wp, self.forces_buffer, self.torques_buffer],
                        device=self.device,
                    ),
                )
            profiler.time_sync("zero_qfrc", self.mjw_data.qfrc_applied.zero_)
            profiler.time_sync("xfrc_accumulate", lambda: mjw.xfrc_accumulate(self.mjw_model, self.mjw_data, self.mjw_data.qfrc_applied))

            def mujoco_step():
                if not self.graph_initialized:
                    with wp.ScopedCapture() as capture:
                        mjw.step(self.mjw_model, self.mjw_data)
                    self.graph_initialized = True
                    self.mujoco_single_step_graph = capture.graph
                else:
                    wp.capture_launch(self.mujoco_single_step_graph)

            profiler.time_sync("mujoco_step", mujoco_step)
        profiler.add("simulation_step_total", time.perf_counter() - start_total)

    env._simulation_step = MethodType(profiled_simulation_step, env)


def run_2d(args):
    from envs.lbm.fish.fish_lbm_env import FishLBMEnv
    from lbm_wave_tester_2d import FISH_PRESETS, make_action

    env = FishLBMEnv(
        nx=args.nx2d,
        ny=args.ny2d,
        lbm_scale=args.lbm_scale2d,
        nworld=1,
        max_episode_steps=args.steps + args.warmup + 10,
        per_frame_steps=args.per_frame_steps2d,
        include_image=False,
        render_mode=None,
    )
    env.reset()
    params = dict(FISH_PRESETS[args.preset2d])
    for i in range(args.warmup):
        env.step(make_action(i, args.dt2d, params, max(1, args.warmup)))
    wp.synchronize()

    profiler = Profiler()
    install_2d_profiler(env, profiler)
    for i in range(args.steps):
        action = make_action(i, args.dt2d, params, max(1, args.warmup))
        start = time.perf_counter()
        env.step(action)
        wp.synchronize()
        profiler.add("env_step_total", time.perf_counter() - start)
    return profiler, args.per_frame_steps2d, args.steps, f"2D fish {args.nx2d}x{args.ny2d}"


def run_3d(args):
    from envs.lbm3d.manta.manta_multitask_env_3d import TASK_NAMES
    from lbm3d_runtime import load_named_config, make_multitask_env
    from lbm_wave_tester import ANIMAL_REGISTRY

    info = ANIMAL_REGISTRY[args.animal3d]
    config = load_named_config(
        ["defaults", *info["configs"]],
        overrides={
            "lbm_nx": args.nx3d,
            "lbm_ny": args.ny3d,
            "lbm_nz": args.nz3d,
            "lbm_scale": args.lbm_scale3d,
            "per_frame_steps": args.per_frame_steps3d,
            "time_limit": args.steps + args.warmup + 20,
            "envs": 1,
        },
    )
    wrapped = make_multitask_env(config, nworld=1)
    wrapped.reset()
    env = wrapped._env
    if hasattr(env, "_task_ids"):
        env._task_ids[:] = TASK_NAMES.index("forward")
        env._update_task_ids_wp()
    action_target = info["preset_to_action"](info["presets"][args.preset3d]).reshape(1, -1).astype(np.float32)
    for i in range(args.warmup):
        ramp = min(1.0, (i + 1) / max(1, args.warmup))
        env.step(np.clip(action_target * ramp, -1.0, 1.0).astype(np.float32))
    wp.synchronize()

    profiler = Profiler()
    install_3d_profiler(env, profiler)
    for i in range(args.steps):
        ramp = min(1.0, (i + 1) / max(1, args.warmup))
        action = np.clip(action_target * ramp, -1.0, 1.0).astype(np.float32)
        start = time.perf_counter()
        env.step(action)
        wp.synchronize()
        profiler.add("env_step_total", time.perf_counter() - start)
    return profiler, args.per_frame_steps3d, args.steps, f"3D {args.animal3d} {args.nx3d}x{args.ny3d}x{args.nz3d}"


def print_report(profiler: Profiler, per_frame_steps: int, env_steps: int, title: str):
    print(f"\n=== sim_step breakdown: {title} ===")
    print(f"env_steps={env_steps}, per_frame_steps={per_frame_steps}, substeps={env_steps * per_frame_steps}")
    sim_total = profiler.totals.get("simulation_step_total", 0.0)
    env_total = profiler.totals.get("env_step_total", 0.0)
    phase_names = [
        "extract_body_states",
        "update_lbm_solid",
        "lbm_init_force",
        "lbm_precompute_segments",
        "lbm_stream_collide",
        "lbm_apply_bc",
        "lbm_swap",
        "extract_fluid_forces",
        "zero_xfrc",
        "fill_xfrc",
        "zero_qfrc",
        "xfrc_accumulate",
        "mujoco_step",
    ]

    for name in phase_names:
        total = profiler.totals.get(name, 0.0)
        count = profiler.counts.get(name, 0)
        share = 100.0 * total / max(sim_total, 1e-9)
        print(f"{name:>22s}: total={total:8.3f}s  avg/substep={1000.0 * total / max(1, count):8.2f}ms  share={share:6.2f}%  count={count}")
    other = max(0.0, env_total - sim_total)
    print(f"{'simulation_step_total':>22s}: total={sim_total:8.3f}s  avg/env_step={1000.0 * sim_total / max(1, env_steps):8.2f}ms")
    print(f"{'env_step_total':>22s}: total={env_total:8.3f}s  avg/env_step={1000.0 * env_total / max(1, env_steps):8.2f}ms")
    print(f"{'outside_sim_step':>22s}: total={other:8.3f}s  avg/env_step={1000.0 * other / max(1, env_steps):8.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Profile phases inside LBM/MuJoCo simulation step")
    parser.add_argument("--mode", choices=["2d", "3d", "both"], default="both")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)

    parser.add_argument("--nx2d", type=int, default=400)
    parser.add_argument("--ny2d", type=int, default=600)
    parser.add_argument("--lbm-scale2d", type=float, default=0.2)
    parser.add_argument("--per-frame-steps2d", type=int, default=10)
    parser.add_argument("--preset2d", default="forward")
    parser.add_argument("--dt2d", type=float, default=0.01)

    parser.add_argument("--animal3d", default="eel")
    parser.add_argument("--preset3d", default="forward")
    parser.add_argument("--nx3d", type=int, default=150)
    parser.add_argument("--ny3d", type=int, default=250)
    parser.add_argument("--nz3d", type=int, default=60)
    parser.add_argument("--lbm-scale3d", type=float, default=0.5)
    parser.add_argument("--per-frame-steps3d", type=int, default=10)
    args = parser.parse_args()

    if args.mode in ("2d", "both"):
        print_report(*run_2d(args))
    if args.mode in ("3d", "both"):
        print_report(*run_3d(args))


if __name__ == "__main__":
    main()
