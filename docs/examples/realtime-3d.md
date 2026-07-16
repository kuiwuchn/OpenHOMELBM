# Realtime 3D

The 3D controller renders D3Q27 flow around articulated or fixed meshes. Orbit
mode displays transparent signed-vorticity slices and transfers the slice atlas
from CUDA to OpenGL.

## Live eel view

```powershell
python tools/lbm3d_realtime_control.py `
  --config configs/realtime_3d/eel3d.json `
  --preset forward `
  --view-mode orbit
```

Drag to orbit, use the mouse wheel to zoom, press `Space` to pause, `R` to
reset, and `Q` or `Esc` to quit. Preset keys include `W` for forward, `A`/`D`
for turns, `F` for fast motion, and `S` for idle.

The action panel exposes `A`, `omega`, `k_wave`, and `head_bias`. Roll is fixed
at zero; the realtime demo has no ascend/descend presets.
The checked-in `fast` preset stays below the normalized parameter limits to
leave numerical headroom during long interactive runs.

## Export vorticity slices

```powershell
python tools/lbm3d_realtime_control.py `
  --config configs/realtime_3d/eel3d.json `
  --preset forward `
  --export-lbm outputs/eel_lbm_orbit.mp4 `
  --export-steps 480 `
  --record-fps 30 `
  --view-mode orbit `
  --volume-render-mode slices `
  --volume-stride 1 `
  --volume-slice-count 15 `
  --volume-color-axis z `
  --volume-slice-alpha 0.15 `
  --volume-vmax-percentile 98
```

Export mode does not open a realtime window and exits after the requested
simulation steps.

## Live 3D Kármán scene

```powershell
python tools/lbm3d_realtime_control.py `
  --config configs/realtime_3d/karman3d.json `
  --preset steady `
  --view-mode orbit `
  --volume-render-mode slices `
  --volume-slice-count 9
```

This configuration instantiates `Karman3DEnv`, a static-cylinder environment
that advances the D3Q27 flow without rigid-body control or training rewards.
The checked-in scene uses five solver substeps per rendered frame and an inlet
speed of `0.12`, so the alternating wake becomes visible sooner while the
initial Reynolds number remains `400`.
It opens the same realtime orbit viewer as the eel scene; drag to rotate, use
the mouse wheel to zoom, and press `Space` to pause. The simplified status
panel shows the Reynolds number, lattice viscosity, inlet speed, and cylinder
diameter. Press `+` or `-` to adjust Reynolds number from 100 to 600 in steps
of 50; the controller updates viscosity using `Re = U D / nu`.

## Performance controls

- Reduce `--volume-slice-count` for a faster preview.
- Increase `--volume-stride` to downsample the volume.
- For offline export, increase `--export-render-every` to simulate more steps
  per output frame.
- Use the checked-in grid sizes first; larger D3Q27 grids increase memory use
  quickly.
