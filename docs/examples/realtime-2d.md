# Realtime 2D

“2D” refers only to the LBM fluid grid and its immersed boundary. MuJoCo always
simulates the rigid bodies in 3D. At every coupling step, the relevant 3D body
geometry, pose, and velocity are projected onto the 2D LBM plane; the resulting
planar polygons are used by the fluid solver, and the computed planar fluid
loads are coupled back to the 3D MuJoCo bodies.

The controller combines the projected 2D LBM field, the original 3D MuJoCo
rigid-body rendering, and an action panel. A JSON configuration selects the
environment class, model, grid, flow, keyboard bindings, and named action presets.

!!! important "The rigid-body simulation is not 2D"

    Eel and cylinder bodies remain 3D MuJoCo bodies. Only their coupling
    boundary is projected into the 2D LBM domain.

## Projected eel

```powershell
python tools/lbm2d_realtime_control.py --config configs/realtime_2d/eel2d.json
```

This configuration projects the articulated 3D eel model into twelve planar
immersed boundaries. The right panel exposes the traveling-wave control
parameters; MuJoCo continues to evolve the full 3D articulated state.

## Kármán vortex street

```powershell
python tools/lbm2d_realtime_control.py --config configs/realtime_2d/karman2d.json
```

The JSON instantiates `Karman2DEnv`, which projects the 3D MuJoCo cylinder onto
the LBM plane. It uses a uniform inlet plus small perturbations to break the
symmetric wake. The
live panel reports Reynolds number; `+` and `-` adjust it, while `W`, `A`, `S`,
and `D` move the projected cylinder boundary in LBM space.

Record the same scene:

```powershell
python tools/lbm2d_realtime_control.py `
  --config configs/realtime_2d/karman2d.json `
  --record outputs/karman_vortex_2d.mp4
```

## Configuration map

| Section | Purpose |
| --- | --- |
| `env` | 3D MuJoCo model and its mapping to projected LBM boundaries |
| `lbm` | Grid size, scale, coupling substeps, and flow parameters |
| `render` | Field type, output size, color scaling, and recording FPS |
| `camera` | MuJoCo camera tracking and viewpoint |
| `control` | Action interpretation and start behavior |
| `keyboard_control` | Keyboard-to-preset mapping |
| `presets` | Named action vectors or parameter dictionaries |

Paths in checked-in configs are relative to the repository root. Keep new
configs portable; do not introduce workstation-specific absolute paths.
