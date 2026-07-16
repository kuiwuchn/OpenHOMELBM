# LBM-RIGID

LBM-RIGID couples GPU lattice-Boltzmann fluid simulation with 3D MuJoCo-Warp
rigid-body dynamics. Its 2D mode projects the 3D bodies onto a planar LBM grid;
its 3D mode couples their meshes directly to a D3Q27 fluid domain. Both modes
support interactive flow visualization, aquatic locomotion control, and
reinforcement-learning experiments.

## Choose a starting point

| Goal | Start here |
| --- | --- |
| Run an existing scene | [Getting started](getting-started.md) |
| Explore a live 2D flow with projected 3D bodies | [Realtime 2D](examples/realtime-2d.md) |
| Explore a live 3D flow with 3D bodies | [Realtime 3D](examples/realtime-3d.md) |
| Train an eel controller | [SAC training](examples/sac-training.md) |
| Integrate the solver in Python | [API reference](api/index.md) |
| Understand the coupling loop | [Architecture](architecture.md) |

## Minimal 2D run

After installing the runtime dependencies, launch the projected eel demo:

```powershell
python tools/lbm2d_realtime_control.py --config configs/realtime_2d/eel2d.json
```

The JSON file selects the MuJoCo model, lattice resolution, immersed solids,
fluid parameters, renderer, and keyboard presets. Use a Kármán configuration
when you want a smaller fixed-geometry flow experiment:

```powershell
python tools/lbm2d_realtime_control.py --config configs/realtime_2d/karman2d.json
```

!!! note "GPU-oriented runtime"

    The simulation code uses Warp and MuJoCo-Warp and is designed around CUDA.
    Building this documentation site does not initialize the solver or require a
    GPU because mkdocstrings reads the Python source statically.

## Public API

The supported high-level entry points are the 2D and 3D solver classes,
their flow-state structs, and Gym-compatible environment bases. Warp kernels in
`lbm_func.py` and environment-specific kernel modules are implementation
details unless an API page explicitly documents them.
