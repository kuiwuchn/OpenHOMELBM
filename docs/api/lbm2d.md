# 2D LBM solver and projected-rigid environment

## Typical imports

```python
from envs.lbm import HomeFlow, LBMFluidEnv, LBM_Solver
```

`LBM_Solver` is the high-level 2D fluid entry point. `HomeFlow` exposes
per-world Warp buffers for projected boundaries, coupling, and visualization.
`LBMFluidEnv` keeps the rigid-body state in 3D MuJoCo-Warp and projects selected
bodies onto the 2D LBM plane. Instantiate a task-specific subclass or use a
checked-in JSON demo rather than using the base directly.

## LBM solver

::: envs.lbm.lbm_solver.LBM_Solver
    options:
      members:
        - step
        - create_solid_from_mujoco
        - finalize_mappings

## Flow state

::: envs.lbm.lbm_core.HomeFlow
    options:
      members:
        - Initialize
        - configure_solid

## Environment base

::: envs.lbm.lbm_fluid_env.LBMFluidEnv
    options:
      members:
        - reset
        - partial_reset
        - step
        - render
        - save_video
        - close

## Kármán environment

::: envs.lbm.karman.Karman2DEnv
    options:
      members:
        - reset
        - step
