# Environment catalog

Use environment classes when building training code. Use the JSON-driven tools
when running interactive or export demos.

## 2D LBM / projected-rigid convenience exports

Every class in this table advances 3D MuJoCo rigid-body state. “2D” describes
the planar LBM field and the projected coupling boundary.

| Class | Import path | Purpose |
| --- | --- | --- |
| `LBMFluidEnv` | `envs.lbm.LBMFluidEnv` | Base coupled environment |
| `Eel2DLBMEnv` | `envs.lbm.Eel2DLBMEnv` | Projected eel control |
| `Karman2DEnv` | `envs.lbm.Karman2DEnv` | Planar cylinder-wake benchmark |

## 3D convenience exports

| Class | Import path | Purpose |
| --- | --- | --- |
| `LBMFluidEnv3D` | `envs.lbm3d.LBMFluidEnv3D` | Base coupled environment |
| `Eel3DLBMEnv` | `envs.lbm3d.Eel3DLBMEnv` | Eel locomotion |
| `EelMultiTaskEnv` | `envs.lbm3d.EelMultiTaskEnv` | Multi-task eel control |
| `Karman3DEnv` | `envs.lbm3d.Karman3DEnv` | Three-dimensional cylinder-wake benchmark |
