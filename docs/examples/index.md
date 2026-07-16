# Examples

The repository's executable demos live in `tools/` and at the repository root.
They use checked-in JSON configurations so model paths, grid sizes, control
presets, and rendering parameters remain reproducible.

| Demo | Entry point | Typical result |
| --- | --- | --- |
| Realtime 2D | `tools/lbm2d_realtime_control.py` | 2D flow around projected 3D MuJoCo bodies |
| Realtime 3D | `tools/lbm3d_realtime_control.py` | Orbiting vorticity slices or exported video |
| SAC training | `train_sac_minimal.py` | Checkpoints, monitor metrics, optional live panel |

Start with the 2D Kármán or eel demo to verify the environment. Three-dimensional
rendering and SAC training allocate more memory and run for longer.

All commands on the following pages assume the repository root as the working
directory. Output paths are explicit and live under `outputs/`.
