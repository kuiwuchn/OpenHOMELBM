"""Kármán vortex-street environment for the 3D D3Q27 LBM solver."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import warp as wp

from ..lbm_core_3d import HomeFlow3D, cx_d3q27, cy_d3q27, cz_d3q27, w_d3q27
from ..lbm_fluid_env_3d import LBMFluidEnv3D


@wp.kernel
def set_uniform_flow_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D), ux: float, uy: float, uz: float
):
    """Initialize every lattice cell with a uniform D3Q27 equilibrium flow."""
    world_idx, x, y, z = wp.tid()
    flow = flows[world_idx]
    rho = 1.0
    population = wp.types.vector(length=27, dtype=wp.float32)
    speed_squared = ux * ux + uy * uy + uz * uz
    for i in range(27):
        cu = cx_d3q27[i] * ux + cy_d3q27[i] * uy + cz_d3q27[i] * uz
        population[i] = w_d3q27[i] * rho * (
            1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared
        )

    inv_rho = 1.0 / rho
    pixx = population[1] + population[2] + population[7] + population[8] + population[9] + population[10] + population[13] + population[14] + population[15] + population[16] + population[19] + population[20] + population[21] + population[22] + population[23] + population[24] + population[25] + population[26]
    pixy = (population[7] + population[8] + population[19] + population[20] + population[21] + population[22]) - (population[13] + population[14] + population[23] + population[24] + population[25] + population[26])
    pixz = (population[9] + population[10] + population[19] + population[20] + population[23] + population[24]) - (population[15] + population[16] + population[21] + population[22] + population[25] + population[26])
    piyy = population[3] + population[4] + population[7] + population[8] + population[11] + population[12] + population[13] + population[14] + population[17] + population[18] + population[19] + population[20] + population[21] + population[22] + population[23] + population[24] + population[25] + population[26]
    piyz = (population[11] + population[12] + population[19] + population[20] + population[25] + population[26]) - (population[17] + population[18] + population[21] + population[22] + population[23] + population[24])
    pizz = population[5] + population[6] + population[9] + population[10] + population[11] + population[12] + population[15] + population[16] + population[17] + population[18] + population[19] + population[20] + population[21] + population[22] + population[23] + population[24] + population[25] + population[26]
    cs2_local = pixx
    pixx = pixx * inv_rho - cs2_local
    pixy = pixy * inv_rho
    pixz = pixz * inv_rho
    piyy = piyy * inv_rho - cs2_local
    piyz = piyz * inv_rho
    pizz = pizz * inv_rho - cs2_local

    flow.rho[x, y, z] = rho
    flow.rho_post[x, y, z] = rho
    flow.u[x, y, z] = wp.vec3(ux, uy, uz)
    flow.u_post[x, y, z] = wp.vec3(ux, uy, uz)
    flow.Sxx[x, y, z] = pixx
    flow.Sxx_post[x, y, z] = pixx
    flow.Syy[x, y, z] = piyy
    flow.Syy_post[x, y, z] = piyy
    flow.Szz[x, y, z] = pizz
    flow.Szz_post[x, y, z] = pizz
    flow.Sxy[x, y, z] = pixy
    flow.Sxy_post[x, y, z] = pixy
    flow.Sxz[x, y, z] = pixz
    flow.Sxz_post[x, y, z] = pixz
    flow.Syz[x, y, z] = piyz
    flow.Syz_post[x, y, z] = piyz
    flow.forcex[x, y, z] = 0.0
    flow.forcey[x, y, z] = 0.0
    flow.forcez[x, y, z] = 0.0


@wp.kernel
def set_local_force_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    center_x: float,
    center_y: float,
    center_z: float,
    radius_x: float,
    radius_y: float,
    radius_z: float,
    force_x: float,
    force_y: float,
    force_z: float,
):
    """Apply a smooth ellipsoidal perturbation to seed vortex shedding."""
    world_idx, x, y, z = wp.tid()
    flow = flows[world_idx]
    dx = (float(x) - center_x) / wp.max(radius_x, 1.0e-6)
    dy = (float(y) - center_y) / wp.max(radius_y, 1.0e-6)
    dz = (float(z) - center_z) / wp.max(radius_z, 1.0e-6)
    weight = wp.max(0.0, 1.0 - dx * dx - dy * dy - dz * dz)
    flow.forcex[x, y, z] = force_x * weight
    flow.forcey[x, y, z] = force_y * weight
    flow.forcez[x, y, z] = force_z * weight


@wp.kernel
def set_boundary_velocity_3d_kernel(
    flows: wp.array(dtype=HomeFlow3D),
    boundary_idx: int,
    ux: float,
    uy: float,
    uz: float,
):
    """Update one velocity boundary for every world."""
    world_idx = wp.tid()
    flows[world_idx].bc_value[boundary_idx] = wp.vec3(ux, uy, uz)


BOUNDARY_NAME_TO_INDEX_3D = {
    "left": 0,
    "right": 1,
    "top": 2,
    "bottom": 3,
    "front": 4,
    "back": 5,
}


def _runtime_signal(
    config: Dict[str, Any], step: int, default_period: float
) -> float:
    active_steps = config.get("active_steps")
    if active_steps is not None and step > int(active_steps):
        return 0.0
    period = max(1.0, float(config.get("period_steps", default_period)))
    phase = float(config.get("phase", 0.0))
    return float(np.sin(2.0 * np.pi * float(step) / period + phase))


def apply_lbm_flow_config_3d(
    env: "Karman3DEnv", flow_config: Dict[str, Any]
) -> None:
    """Apply viscosity, boundary, and initial-flow settings after reset."""
    if not flow_config:
        return
    for flow in env.lbm_solver.flows:
        if "viscosity" in flow_config:
            flow.vis_shear = float(flow_config["viscosity"])
        if "bc_type" in flow_config:
            boundary_types = [int(value) for value in flow_config["bc_type"]]
            if len(boundary_types) != 6:
                raise ValueError("3D bc_type must contain six boundary values")
            flow.bc_type = wp.types.vector(length=6, dtype=wp.int32)(
                *boundary_types
            )
        if "bc_value" in flow_config:
            values = flow_config["bc_value"]
            if len(values) != 6:
                raise ValueError("3D bc_value must contain six velocity vectors")
            flow.bc_value = wp.array(
                tuple(
                    wp.vec3(float(value[0]), float(value[1]), float(value[2]))
                    for value in values
                ),
                dtype=wp.vec3,
                device=env.device,
            )
    env.lbm_solver.flows_wp = wp.array(
        env.lbm_solver.flows, dtype=HomeFlow3D, device=env.device
    )
    if "initial_velocity" in flow_config:
        ux, uy, uz = flow_config["initial_velocity"]
        wp.launch(
            set_uniform_flow_3d_kernel,
            dim=(env.nworld, env.nx, env.ny, env.nz),
            inputs=[env.lbm_solver.flows_wp, float(ux), float(uy), float(uz)],
            device=env.device,
        )
        wp.synchronize()


def apply_lbm_runtime_flow_config_3d(
    env: "Karman3DEnv", flow_config: Dict[str, Any], step: int
) -> None:
    """Apply time-dependent inlet and wake perturbations."""
    if not flow_config:
        return
    perturbation = flow_config.get("inlet_perturbation") or flow_config.get(
        "boundary_perturbation"
    )
    if perturbation:
        boundary = perturbation.get("boundary", "left")
        boundary_idx = BOUNDARY_NAME_TO_INDEX_3D.get(
            str(boundary).lower(),
            int(boundary) if isinstance(boundary, int) else 0,
        )
        default_values = flow_config.get("bc_value", [[0.0, 0.0, 0.0]] * 6)
        base = perturbation.get("base", default_values[boundary_idx])
        amplitude = perturbation.get("amplitude", [0.0, 0.0, 0.0])
        signal = _runtime_signal(perturbation, step, 900.0)
        wp.launch(
            set_boundary_velocity_3d_kernel,
            dim=env.nworld,
            inputs=[
                env.lbm_solver.flows_wp,
                boundary_idx,
                float(base[0]) + float(amplitude[0]) * signal,
                float(base[1]) + float(amplitude[1]) * signal,
                float(base[2]) + float(amplitude[2]) * signal,
            ],
            device=env.device,
        )

    wake = flow_config.get("wake_perturbation")
    if wake and bool(wake.get("enabled", True)):
        center = wake.get("center", [0.5 * env.nx, 0.5 * env.ny, 0.5 * env.nz])
        radius = wake.get("radius", [20.0, 20.0, 20.0])
        force = wake.get("force", [0.0, 0.0, 0.0])
        signal = _runtime_signal(wake, step, 240.0)
        wp.launch(
            set_local_force_3d_kernel,
            dim=(env.nworld, env.nx, env.ny, env.nz),
            inputs=[
                env.lbm_solver.flows_wp,
                float(center[0]),
                float(center[1]),
                float(center[2]),
                float(radius[0]),
                float(radius[1]),
                float(radius[2]),
                float(force[0]) * signal,
                float(force[1]) * signal,
                float(force[2]) * signal,
            ],
            device=env.device,
        )


class Karman3DEnv(LBMFluidEnv3D):
    """Simulate three-dimensional flow around a static MuJoCo cylinder."""

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        link_config: Optional[List[Dict[str, Any]]] = None,
        nx: int = 240,
        ny: int = 120,
        nz: int = 40,
        lbm_scale: float = 0.05,
        flow_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if mjcf_path is None:
            mjcf_path = os.path.join(os.path.dirname(__file__), "karman_cylinder_3d.xml")
        if link_config is None:
            link_config = [
                {
                    "link_name": "cylinder",
                    "lbm_position": (50.0 / 240.0 * nx, 61.0 / 120.0 * ny, 0.5 * nz),
                    "is_static": True,
                }
            ]
        if flow_config is None:
            flow_config = {
                "initial_velocity": [0.08, 0.0, 0.0],
                "viscosity": 0.0048,
                "bc_type": [0, 1, 1, 1, 1, 1],
                "bc_value": [
                    [0.08, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            }
        self.flow_config = dict(flow_config)
        super().__init__(
            mjcf_path=mjcf_path,
            link_config=link_config,
            nx=nx,
            ny=ny,
            nz=nz,
            lbm_scale=lbm_scale,
            **kwargs,
        )

    def reset(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Reset the coupled state and reapply the configured Karman flow."""
        observation = super().reset(*args, **kwargs)
        apply_lbm_flow_config_3d(self, self.flow_config)
        return observation

    def set_viscosity(self, viscosity: float) -> None:
        """Set the lattice kinematic viscosity for subsequent solver steps.

        Rebuilds the batched Warp flow array and invalidates the captured LBM
        graph so the next step uses the new positive viscosity.

        Args:
            viscosity: Positive kinematic viscosity in lattice units.

        Raises:
            ValueError: If ``viscosity`` is not finite and positive.
        """
        value = float(viscosity)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("viscosity must be finite and positive")

        wp.synchronize()
        self.flow_config["viscosity"] = value
        for flow in self.lbm_solver.flows:
            flow.vis_shear = value
        self.lbm_solver.flows_wp = wp.array(
            self.lbm_solver.flows, dtype=HomeFlow3D, device=self.device
        )
        # The captured graph closes over the previous flows_wp allocation.
        self.lbm_solver.captured = False
        self.lbm_solver.captured_graph = None

    def _simulation_step(self) -> None:
        for substep in range(self.per_frame_steps):
            step = int(self.step_counts[0]) * self.per_frame_steps + substep
            apply_lbm_runtime_flow_config_3d(self, self.flow_config, step)
            self.lbm_solver.step()
            wp.synchronize()

    def _compute_reward(self, instability_mask=None) -> np.ndarray:
        return np.zeros(self.nworld, dtype=np.float32)

    def _is_terminated(self, instability_mask=None) -> np.ndarray:
        if instability_mask is None:
            return np.zeros(self.nworld, dtype=bool)
        return np.asarray(instability_mask, dtype=bool)
