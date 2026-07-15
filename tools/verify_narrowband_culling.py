"""Correctness check for the narrow-band bounding-sphere culling in
`stream_and_collide_3d` / `get_cutcell_multi_3d`.

The culling skips mesh ray queries for fluid cells farther than
`solid_bound_radius + 2` from every solid center. Because a cut-cell hit lies
within |c_i|*cutcell <= sqrt(3) < 2 of the cell, such cells can only take the
plain-streaming branch — identical to a ray miss. This script proves it:

Run one `stream_and_collide_3d` launch on an identical warmed-up state with
culling ON (real radii) vs OFF (radius = 1e9, i.e. the original full-query
path), and compare:
  * per-cell field outputs (*_post) — written without atomics, must be bit-identical
  * solid_force / solid_torque — atomic sums, differ only by summation order

A second OFF run establishes the atomic-order noise floor for reference.

Usage:
    python tools/verify_narrowband_culling.py [--config configs/realtime_3d/eel3d.json]
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import warp as wp

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(1, str(PROJECT_ROOT))

from lbm3d_runtime import make_multitask_env
from envs.lbm3d.lbm_core_3d import HomeFlow3D
from envs.lbm3d.lbm_func_3d import stream_and_collide_3d, init_force_3d_batch

_spec = importlib.util.spec_from_file_location("rtc", str(SCRIPT_DIR / "lbm3d_realtime_control.py"))
rtc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rtc)

POST_FIELDS = ["rho_post", "u_post", "Sxx_post", "Syy_post", "Szz_post",
               "Sxy_post", "Sxz_post", "Syz_post"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/realtime_3d/eel3d.json")
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = json.load(open(PROJECT_ROOT / args.config))
    np.random.seed(args.seed)
    env = make_multitask_env(rtc.build_runtime_config(cfg, {}), nworld=1)
    env.reset()
    b = env._env
    s = b.lbm_solver

    act = np.zeros((1, b.action_dim), dtype=np.float32)
    act[0, 0] = 0.8
    act[0, 2] = 0.5
    for _ in range(args.warmup):
        env.step(act)
    wp.synchronize()
    flow = s.flows[0]

    def run_once():
        flow.solid_force = wp.array(np.zeros((s.solid_num, 3), np.float32), dtype=wp.vec3, device=s.device)
        flow.solid_torque = wp.array(np.zeros((s.solid_num, 3), np.float32), dtype=wp.vec3, device=s.device)
        s.flows_wp = wp.array(s.flows, dtype=HomeFlow3D, device=s.device)
        wp.launch(init_force_3d_batch, dim=(s.nworld,), inputs=[s.flows_wp], device=s.device)
        wp.launch(stream_and_collide_3d, dim=(s.nworld, s.nx, s.ny, s.nz), inputs=[s.flows_wp], device=s.device)
        wp.synchronize()
        out = {n: getattr(flow, n).numpy().copy() for n in POST_FIELDS}
        out["solid_force"] = flow.solid_force.numpy().copy()
        out["solid_torque"] = flow.solid_torque.numpy().copy()
        return out

    real_r = flow.solid_bound_radius.numpy().copy()

    flow.solid_bound_radius = wp.array(real_r, dtype=wp.float32, device=s.device)
    on = run_once()

    flow.solid_bound_radius = wp.array(np.full_like(real_r, 1e9), dtype=wp.float32, device=s.device)
    off = run_once()
    off2 = run_once()

    dfield = max(float(np.abs(on[n] - off[n]).max()) for n in POST_FIELDS)
    dfrc = float(np.abs(on["solid_force"] - off["solid_force"]).max())
    dtrq = float(np.abs(on["solid_torque"] - off["solid_torque"]).max())
    noise_field = max(float(np.abs(off[n] - off2[n]).max()) for n in POST_FIELDS)
    noise_frc = float(np.abs(off["solid_force"] - off2["solid_force"]).max())
    noise_trq = float(np.abs(off["solid_torque"] - off2["solid_torque"]).max())

    print(f"[verify] ON vs OFF : max|Dfield_post|={dfield:.3e} max|Dforce|={dfrc:.3e} max|Dtorque|={dtrq:.3e}")
    print(f"[verify] noise floor: max|Dfield_post|={noise_field:.3e} max|Dforce|={noise_frc:.3e} max|Dtorque|={noise_trq:.3e}")

    ok = (dfield == 0.0) and (dfrc <= max(noise_frc, 1e-4)) and (dtrq <= max(noise_trq, 1e-4))
    print("[verify] RESULT:", "PASS - culling is result-preserving" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
