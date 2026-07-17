# Open HOME-LBM

Open HOME-LBM is an open-source research codebase based on the High-Order
Moment-Encoded Lattice Boltzmann Method (HOME-LBM) introduced by
[Li et al. (2023)](https://kuiwuchn.github.io/homelbm.html). It extends the original
implementation with MuJoCo-Warp coupled environments and realtime demonstrations.
It also provides Python scripts for training and evaluating SAC controllers for
articulated swimmers.

## HOME-LBM foundation

HOME-LBM is a moment-encoded lattice Boltzmann solver. Instead of persistently
storing all directional distributions, it stores the first three orders of
velocity moments: density, momentum, and a stress-related second-order tensor.
The directional distributions are reconstructed during each solver update.

Open HOME-LBM supports both D2Q9 and D3Q27 lattice models. D2Q9 represents a
two-dimensional lattice with nine discrete velocity directions and is used for
planar flow with projected 3D MuJoCo bodies. D3Q27 represents a three-dimensional
lattice with 27 directions and is used for volumetric flow coupled directly to
3D body meshes.

![Alternating two-dimensional Karman wake behind a projected cylinder](assets/demos/karman2d.jpg)

*2D Kármán wake. A three-dimensional MuJoCo cylinder is projected into the
D2Q9 flow while signed vorticity reveals the alternating vortex street.*

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

## Reference

Wei Li, Tongtong Wang, Zherong Pan, Xifeng Gao, Kui Wu, and Mathieu Desbrun.
"High-Order Moment-Encoded Kinetic Simulation of Turbulent Flows."
*ACM Transactions on Graphics*, 42(6), Article 190, 2023.
[HOME-LBM project page](https://kuiwuchn.github.io/homelbm.html) |
[DOI: 10.1145/3618341](https://doi.org/10.1145/3618341)
