# 3D solver and environment

## Typical imports

```python
from envs.lbm3d import HomeFlow3D, LBMFluidEnv3D, LBM_Solver3D
```

Three-dimensional grids store D3Q27 state and mesh-based immersed boundaries.
Start with the checked-in 3D configurations before increasing lattice sizes or
the number of parallel worlds.

## LBM solver

::: envs.lbm3d.lbm_solver_3d.LBM_Solver3D
    options:
      members:
        - step
        - create_solid_from_mesh
        - update_solids_batch
        - get_forces_and_torques
        - reset_world
        - reset_worlds
        - finalize_mappings

## Flow state

::: envs.lbm3d.lbm_core_3d.HomeFlow3D
    options:
      members:
        - Initialize

## Environment base

::: envs.lbm3d.lbm_fluid_env_3d.LBMFluidEnv3D
    options:
      members:
        - reset
        - partial_reset
        - step
        - render
        - close

## Kármán environment

::: envs.lbm3d.karman.Karman3DEnv
    options:
      members:
        - reset
        - step
