# Getting started

## Requirements

- Python 3.11, matching the repository's current setup instructions.
- A CUDA-capable environment for the simulation and interactive demos.
- MuJoCo Warp and the Python packages listed in `requirements.txt`.

## Install the runtime

Create and activate an isolated environment:

```powershell
conda create -n openhomelbm python=3.11 -y
conda activate openhomelbm
```

Install the CUDA build of PyTorch used by the current project setup, then
MuJoCo Warp and the remaining dependencies:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install mujoco-warp
pip install -e .
```

The editable installation keeps imports linked to the current checkout. Keep
the working directory at the repository root when using the relative config
paths shown in these docs.

## Verify the command-line entry points

Argument parsing is a quick way to verify that Python can import the runtime
dependencies without allocating a full simulation:

```powershell
python tools/lbm2d_realtime_control.py --help
python tools/lbm3d_realtime_control.py --help
python train_sac_minimal.py --help
```

Then run a configured scene:

```powershell
python tools/lbm2d_realtime_control.py --config configs/realtime_2d/eel2d.json
```

## Common failure modes

### Imports fail before a window opens

Confirm that `torch`, `warp-lang`, `mujoco`, and `mujoco-warp` were installed in
the active environment. Run the `--help` checks above to identify the first
missing import.

### CUDA allocation fails

Start with the smaller 2D Kármán configuration. The 3D scenes allocate D3Q27
state and per-solid mesh buffers and therefore require substantially more GPU
memory.

### No interactive window appears

Check the configuration's `run.headless` and recording/export options. Export
commands intentionally run without a live window.
