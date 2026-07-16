"""Kármán vortex-street environment for the 2D HOME LBM solver."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import warp as wp

from ..lbm_core import HomeFlow
from ..lbm_fluid_env import LBMFluidEnv


@wp.kernel
def set_uniform_flow_kernel(
    flows: wp.array(dtype=HomeFlow), ux: float, uy: float
):
    """Initialize every lattice cell with a uniform D2Q9 equilibrium flow."""
    world_idx, x, y = wp.tid()
    flow = flows[world_idx]
    rho = 1.0
    population = wp.types.vector(length=9, dtype=wp.float32)
    speed_squared = ux * ux + uy * uy
    for i in range(9):
        cu = flow.cx_d2q9[i] * ux + flow.cy_d2q9[i] * uy
        population[i] = (
            rho
            * flow.w_d2q9[i]
            * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared)
        )

    inv_rho = 1.0 / rho
    pixx = population[1] + population[2] + population[5] + population[6] + population[7] + population[8]
    piyy = population[3] + population[4] + population[5] + population[6] + population[7] + population[8]
    pixy = population[5] + population[7] - population[6] - population[8]
    cs2_local = pixx
    pixx = pixx * inv_rho - cs2_local
    piyy = piyy * inv_rho - cs2_local
    pixy = pixy * inv_rho

    flow.rho[x, y] = rho
    flow.rho_post[x, y] = rho
    flow.u[x, y] = wp.vec2(ux, uy)
    flow.u_post[x, y] = wp.vec2(ux, uy)
    flow.Sxx[x, y] = pixx
    flow.Sxx_post[x, y] = pixx
    flow.Syy[x, y] = piyy
    flow.Syy_post[x, y] = piyy
    flow.Sxy[x, y] = pixy
    flow.Sxy_post[x, y] = pixy
    flow.forcex[x, y] = 0.0
    flow.forcey[x, y] = 0.0


@wp.kernel
def set_boundary_velocity_kernel(
    flows: wp.array(dtype=HomeFlow), boundary_idx: int, ux: float, uy: float
):
    """Update one velocity boundary for every world."""
    world_idx = wp.tid()
    flows[world_idx].bc_value[boundary_idx] = wp.vec2(ux, uy)


@wp.kernel
def advance_prescribed_solid_motion_kernel(
    flows: wp.array(dtype=HomeFlow),
    solid_id: int,
    vx: float,
    vy: float,
    omega: float,
):
    """Advance an immersed boundary directly in LBM coordinates."""
    world_idx = wp.tid()
    flow = flows[world_idx]
    flow.solid_position[solid_id] = flow.solid_position[solid_id] + wp.vec2(vx, vy)
    flow.solid_angle[solid_id] = flow.solid_angle[solid_id] + omega


@wp.kernel
def set_local_force_kernel(
    flows: wp.array(dtype=HomeFlow),
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    force_x: float,
    force_y: float,
):
    """Apply a smooth elliptical perturbation to seed vortex shedding."""
    world_idx, x, y = wp.tid()
    flow = flows[world_idx]
    dx = (float(x) - center_x) / wp.max(radius_x, 1.0e-6)
    dy = (float(y) - center_y) / wp.max(radius_y, 1.0e-6)
    weight = wp.max(0.0, 1.0 - dx * dx - dy * dy)
    flow.forcex[x, y] = force_x * weight
    flow.forcey[x, y] = force_y * weight


BOUNDARY_NAME_TO_INDEX = {"left": 0, "top": 1, "right": 2, "bottom": 3}


def _perturbation_signal(
    config: Dict[str, Any], step: int, default_period: float
) -> float:
    active_steps = config.get("active_steps")
    if active_steps is not None and step > int(active_steps):
        return 0.0
    period = max(1.0, float(config.get("period_steps", default_period)))
    phase = float(config.get("phase", 0.0))
    return float(np.sin(2.0 * np.pi * float(step) / period + phase))


def apply_lbm_runtime_flow_config(
    env: "Karman2DEnv", flow_config: Optional[Dict[str, Any]], step: int
) -> None:
    """Apply time-dependent inlet and wake perturbations."""
    if not flow_config:
        return
    perturbation = flow_config.get("inlet_perturbation") or flow_config.get(
        "boundary_perturbation"
    )
    if perturbation:
        boundary = perturbation.get("boundary", "left")
        boundary_idx = BOUNDARY_NAME_TO_INDEX.get(
            str(boundary).lower(),
            int(boundary) if isinstance(boundary, int) else 0,
        )
        default_values = flow_config.get("bc_value", [[0.0, 0.0]] * 4)
        base = perturbation.get("base", default_values[boundary_idx])
        amplitude = perturbation.get("amplitude", [0.0, 0.0])
        signal = _perturbation_signal(perturbation, step, 900.0)
        wp.launch(
            set_boundary_velocity_kernel,
            dim=env.nworld,
            inputs=[
                env.solver.flows_wp,
                boundary_idx,
                float(base[0]) + float(amplitude[0]) * signal,
                float(base[1]) + float(amplitude[1]) * signal,
            ],
            device=env.solver.device,
        )

    wake = flow_config.get("wake_perturbation")
    if wake and bool(wake.get("enabled", True)):
        center = wake.get("center", [0.5 * env.nx, 0.5 * env.ny])
        radius = wake.get("radius", [20.0, 20.0])
        force = wake.get("force", [0.0, 0.0])
        signal = _perturbation_signal(wake, step, 240.0)
        wp.launch(
            set_local_force_kernel,
            dim=(env.nworld, env.nx, env.ny),
            inputs=[
                env.solver.flows_wp,
                float(center[0]),
                float(center[1]),
                float(radius[0]),
                float(radius[1]),
                float(force[0]) * signal,
                float(force[1]) * signal,
            ],
            device=env.solver.device,
        )


def apply_lbm_flow_config(
    env: "Karman2DEnv", flow_config: Optional[Dict[str, Any]]
) -> None:
    """Apply viscosity, boundary, and initial-flow settings after reset."""
    if not flow_config:
        return
    for flow in env.solver.flows:
        if "viscosity" in flow_config:
            flow.vis_shear = float(flow_config["viscosity"])
        if "bc_type" in flow_config:
            boundary_types = [int(value) for value in flow_config["bc_type"]]
            if len(boundary_types) != 4:
                raise ValueError("bc_type must contain left, top, right, bottom")
            flow.bc_type = wp.types.vector(length=4, dtype=wp.int32)(
                *boundary_types
            )
        if "bc_value" in flow_config:
            values = flow_config["bc_value"]
            if len(values) != 4:
                raise ValueError("bc_value must contain four velocity vectors")
            flow.bc_value = wp.array(
                tuple(wp.vec2(float(value[0]), float(value[1])) for value in values),
                dtype=wp.vec2,
                device=env.solver.device,
            )
    env.solver.flows_wp = wp.array(
        env.solver.flows, dtype=HomeFlow, device=env.solver.device
    )
    if "initial_velocity" in flow_config:
        ux, uy = flow_config["initial_velocity"]
        wp.launch(
            set_uniform_flow_kernel,
            dim=(env.nworld, env.nx, env.ny),
            inputs=[env.solver.flows_wp, float(ux), float(uy)],
            device=env.solver.device,
        )
        wp.synchronize()


class Karman2DEnv(LBMFluidEnv):
    """Simulate flow around a projected three-dimensional cylinder in 2D LBM."""

    def __init__(
        self,
        xml_path: Optional[str] = None,
        solid_config: Optional[List[Dict[str, Any]]] = None,
        nx: int = 600,
        ny: int = 200,
        lbm_scale: float = 0.016,
        flow_config: Optional[Dict[str, Any]] = None,
        prescribed_motion: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "karman_cylinder_2d.xml")
        if solid_config is None:
            solid_config = [
                {
                    "solid_id": 0,
                    "body_id": 1,
                    "body_or_geom_name": "cylinder_geom",
                    "lbm_position": (0.25 * nx, 0.505 * ny),
                    "is_body": False,
                    "n_samples": 96,
                }
            ]
        if flow_config is None:
            flow_config = {
                "initial_velocity": [0.1, 0.0],
                "viscosity": 0.01,
                "bc_type": [0, 1, 1, 1],
                "bc_value": [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            }
        self.flow_config = dict(flow_config)
        self.prescribed_motion = dict(
            prescribed_motion
            or {
                "enabled": True,
                "solid_id": 0,
                "velocity": [0.0, 0.0],
                "angular_velocity": 0.0,
            }
        )
        super().__init__(
            xml_path=xml_path,
            solid_config=solid_config,
            nx=nx,
            ny=ny,
            lbm_scale=lbm_scale,
            **kwargs,
        )

    def reset(self, *args: Any, **kwargs: Any) -> np.ndarray:
        observation = super().reset(*args, **kwargs)
        apply_lbm_flow_config(self, self.flow_config)
        return observation

    def _simulation_step(self) -> None:
        if not self.prescribed_motion.get("enabled", False):
            super()._simulation_step()
            return
        solid_id = int(self.prescribed_motion.get("solid_id", 0))
        velocity = self.prescribed_motion.get("velocity", [0.0, 0.0])
        omega = float(self.prescribed_motion.get("angular_velocity", 0.0))
        for substep in range(self.per_frame_steps):
            step = int(self.current_steps[0]) * self.per_frame_steps + substep
            apply_lbm_runtime_flow_config(self, self.flow_config, step)
            wp.launch(
                advance_prescribed_solid_motion_kernel,
                dim=self.nworld,
                inputs=[
                    self.solver.flows_wp,
                    solid_id,
                    float(velocity[0]),
                    float(velocity[1]),
                    omega,
                ],
                device=self.solver.device,
            )
            self.solver.step()
            wp.synchronize()

    def _compute_reward(self, instability_mask=None) -> np.ndarray:
        return np.zeros(self.nworld, dtype=np.float32)

    def _is_terminated(self, instability_mask=None) -> np.ndarray:
        if instability_mask is None:
            return np.zeros(self.nworld, dtype=bool)
        return np.asarray(instability_mask, dtype=bool)
