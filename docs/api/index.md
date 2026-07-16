# API reference

The API reference documents the stable, high-level objects used to construct
fluid simulations and Gym environments.

## Supported layers

- [2D LBM and projected-rigid environment](lbm2d.md): `HomeFlow`, `LBM_Solver`,
  and `LBMFluidEnv`. The rigid bodies remain 3D in MuJoCo.
- [3D solver and environment](lbm3d.md): `HomeFlow3D`, `LBM_Solver3D`, and
  `LBMFluidEnv3D`.
- [Environment catalog](environments.md): checked-in task environments and
  their public import paths.

The 2D solver class retains the historical name `LBM_Solver`. Renaming it would
break existing configs and imports, so documentation uses the compatible name.
