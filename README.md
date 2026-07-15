# LBM-RIGID

## Environment Setup

### Step 1: Create a Conda environment

```powershell
conda create -n dreamer python=3.11 -y
conda activate dreamer
```

### Step 2: Install PyTorch for CUDA 12.8

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Install MuJoCo Warp

```powershell
pip install mujoco-warp
```

### Step 4: Install project dependencies

```powershell
pip install -r requirements.txt
```

## Run Commands

### Train the 2D eel with SAC-controlled CPG parameters

The CPG mode makes SAC output four normalized parameters—wave amplitude, frequency,
spatial phase lag, and head bias—and expands them into a traveling wave over the five
eel joints. The oscillator phase and current CPG parameters are appended to the policy
observation.

```powershell
python train_sac_minimal.py --animal eel --control-mode cpg --per-frame-steps 4 --cpg-hold-steps 10 --warmup-steps 10 --total-steps 20000
```

Use `--render` only for short visual checks; headless training is faster. The model and
the physical CPG parameter ranges used for training are saved under
`outputs/sac_minimal/` as `sac_eel2d_cpg.zip` and `sac_eel2d_cpg_config.json`.
In CPG render mode, the right panel shows executed versus SAC-target parameters, the
five generated motor commands over one cycle, and the actual coupled body centerline
read from the LBM solid positions. One SAC action is held for 10 low-level control
steps by default; their rewards are accumulated into one SAC transition.

### Run the 2D realtime LBM controller:

```powershell
python tools/lbm2d_realtime_control.py --config configs/realtime_2d/eel2d.json
```
Run a realtime 2D Karman vortex street around a fixed cylinder using the project 2D rigid-body/LBM coupled solver. The right panel shows Reynolds number and `+` / `-` adjusts it live; `W/A/S/D` moves the cylinder in LBM space:


```powershell
python tools\lbm2d_realtime_control.py --config configs\realtime_2d\karman2d.json
```

To record while viewing, add for example:

```powershell
python tools\lbm2d_realtime_control.py --config configs\realtime_2d\karman2d.json --record outputs\karman_vortex_2d.mp4
```


Alternative moving-cylinder setup: the fluid starts still, and the circular solid translates through the domain using the same project LBM solid boundary solver:

```powershell
python tools\lbm2d_realtime_control.py --config configs\realtime_2d\karman2d_moving.json
```

The JSON configs define the cylinder model, LBM grid, velocity/viscosity, boundary conditions or prescribed solid motion, and output video path. The default outputs are:

```text
outputs/karman_vortex_2d.mp4
outputs/karman_vortex_2d_moving.mp4
```


Important JSON fields:

- `env.xml_path`: MuJoCo cylinder model used by the existing rigid-body coupling path.
- `env.solid_config`: maps `cylinder_geom` into the LBM solver as a fixed circular solid.
- `lbm.flow.initial_velocity`: initial uniform inflow.
- `lbm.flow.viscosity`: lattice viscosity. In realtime mode this is updated when `+` / `-` changes Reynolds number.
- `lbm.flow.reynolds_control`: enables the right-panel Reynolds control, including initial value, range, step size, velocity, and diameter.

- `lbm.flow.bc_type` / `bc_value`: left velocity inlet and zero-gradient-style copy boundaries elsewhere.
- `lbm.flow.inlet_perturbation`: small transverse inlet perturbation used to break the perfectly symmetric wake and seed alternating vortex shedding.
- `lbm.flow.wake_perturbation`: local downstream body-force perturbation near the cylinder wake; this is used to push the symmetric shear layers into alternating vortex shedding.
- `env.prescribed_motion`: optional LBM-space rigid-body motion, used by `karman2d_moving.json` to translate the cylinder through still fluid.

- `run.headless`, `run.record`, `run.steps`, `run.record_start_step`, `run.record_every`: optional headless export settings. For realtime viewing, keep `run.headless=false`.






### Run the realtime 3D LBM slice controller

```powershell
python tools\lbm3d_realtime_control.py --config configs\realtime_3d\eel3d.json --preset forward --view-mode orbit
```

Live `orbit` mode computes signed vorticity slices on CUDA and transfers the RGBA
atlas directly to OpenGL. The default view uses 15 full-resolution transparent
`z` slices, a top-left LBM status overlay, and a right-side action panel for
`A`, `omega`, `k_wave`, `head_bias`, and `roll`. Realtime preview defaults to two
coupled substeps per displayed frame; pass `--per-frame-steps 10` to use the JSON
export cadence at a lower display FPS.

Drag the left view to orbit the camera and use the mouse wheel to zoom. Drag any
right-side action bar to enter manual override and set that action in `[-1, 1]`;
the other actions keep their current values. Pressing a preset key exits manual
override: `W` forward, `A` left, `D` right, `F` fast, `S` idle, and `Z`/`C`
vertical control. `Space` pauses, `R` resets, and `Q` or `Esc` quits.

Useful live-view options include `--action-panel-width`, `--orbit-zoom`,
`--orbit-show-box`, and `--orbit-with-mujoco` (replace the action panel with a
MuJoCo view).

### Export a 3D LBM rendering video. 


2D projected vorticity video:

```powershell
python tools\lbm3d_realtime_control.py --config configs\realtime_3d\eel3d.json --preset forward --export-lbm outputs\eel_lbm_topdown.mp4 --export-steps 120 --record-fps 30 --render-type vorticity --view-mode topdown

```

Recommended 3D eel vorticity slice rendering:

```powershell
python tools\lbm3d_realtime_control.py --config configs\realtime_3d\eel3d.json --preset forward --export-lbm outputs\eel_lbm_orbit_slice9.mp4 --export-steps 480 --record-fps 30 --view-mode orbit --volume-render-mode slices --volume-stride 1 --volume-slice-count 15 --volume-color-axis z --volume-slice-alpha 0.15 --volume-vmax-percentile 98

```

This preset uses full-resolution volume sampling (`--volume-stride 1`), 15 transparent slices, fixed global `z` vorticity coloring, and a lower color percentile so weaker vortices remain visible.

3D Karman vortex street around a fixed cylinder:

```powershell
python tools\lbm3d_realtime_control.py --config configs\realtime_3d\karman3d.json --preset steady --export-lbm outputs\karman_vortex_3d_slice.mp4 --export-steps 9000 --export-render-every 10 --record-fps 30 --view-mode orbit --volume-render-mode slices --volume-stride 1 --volume-slice-axis z --volume-slice-count 9 --volume-color-axis z --volume-slice-alpha 0.25 --volume-vmax-percentile 98
```

The 3D Karman JSON uses `env_type: karman3d`, a fixed MuJoCo cylinder mesh, uniform x-direction inflow, and a small local wake perturbation to seed alternating vortex shedding. With `--export-render-every 10` and `--record-fps 30`, `--export-steps 9000` produces about 30 seconds of video.

Common export options:


- `--export-lbm`: Output `.mp4` path. When enabled, no realtime window is opened; the program exits after export.
- `--export-steps`: Number of simulation steps.
- `--export-render-every`: Capture one LBM frame every N simulation steps.
- `--record-fps`: Exported video FPS.
- `--render-type`: `vorticity` or `velocity`. `orbit` mode always renders a 3D vorticity volume.
- `--view-mode`: `topdown`, `max_topdown`, `side`, `front`, or `orbit`.
- `--volume-stride`: Downsampling stride for `orbit` volume rendering. Use `1` for best quality, `2` for preview, and larger values for speed.
- `--volume-render-mode`: `slices` renders stacked full-field vorticity slices; `isosurface` renders signed red/blue vorticity surfaces; `points` renders only thresholded high-vorticity points.
- `--volume-slice-axis`: Slice stacking axis for `slices` mode: `x`, `y`, or `z`.
- `--volume-slice-count`: Number of slices rendered in `slices` mode.
- `--volume-slice-alpha`: Slice opacity. Use lower values when rendering many slices, for example `0.15` for 15 slices.
- `--volume-vmax-percentile`: Color normalization percentile for the red/white/blue vorticity map. Lower values make weaker vortices more visible.
- `--volume-color-axis`: Fixed global vorticity component used for red/blue coloring in `orbit` mode: `x`, `y`, or `z`. Default is `z`, matching the usual 2D vorticity convention.
- `--volume-iso-min-percentile`, `--volume-iso-percentile`, `--volume-iso-levels`, `--volume-iso-alpha`: Control the multi-layer transparent isosurface renderer.
- `--volume-percentile`: In `points` mode, only vorticity points above this percentile are shown.
- `--volume-max-points`: Maximum number of vorticity points drawn in `points` mode.
- `--orbit-azim-speed`: Camera rotation angle per exported frame in `orbit` mode. Default is `0`, which keeps the camera fixed.
- `--export-no-overlay`: Disable text overlay.

### 3D JSON config

`tools\lbm3d_realtime_control.py` does not read the legacy YAML config. The 3D model, LBM parameters, keyboard bindings, task mapping, and preset actions are loaded from JSON. The default eel config is:

```text
configs/realtime_3d/eel3d.json
```

Important fields:

- `model.env_type`: 3D environment type, for example `eel_multitask`.
- `model.mjcf_path`: MuJoCo XML model path.
- `model.root_link`: Root body used for LBM/MuJoCo alignment.
- `lbm`: 3D grid size, `lbm_scale`, fluid density, and coupling substeps.
- `control.control_mode`: Eel uses `wave` for 5-dimensional traveling-wave actions.
- `control.task_by_preset`: Maps preset names to multitask targets such as `forward` or `turn_left`.
- `controls`: Keyboard-to-preset mapping.
- `presets.action_keys`: Defines the order used to convert preset dictionaries into action vectors.
- `presets.actions`: Named preset action dictionaries.

## JSON Configuration Guide

The 2D realtime control entry point uses a JSON file to configure the scene, LBM grid, rendering, keyboard bindings, and preset actions. Example configs are located at:


```text
configs/realtime_2d/fish2d.json
configs/realtime_2d/eel2d.json
```

Top-level structure:

```json
{
  "name": "eel2d_projected",
  "env": {},
  "lbm": {},
  "render": {},
  "control": {},
  "controls": {},
  "presets": {}
}
```

### `env`: Environment and model

`env` selects the environment class, MuJoCo XML, and the rigid bodies that participate in the 2D LBM projection.

Fish example:

```json
"env": {
  "class": "FishLBMEnv",
  "xml_path": "envs/lbm/fish/fish_2d_v3.xml"
}
```

Eel example:

```json
"env": {
  "class": "GenericLBM2DEnv",
  "xml_path": "envs/lbm3d/eel/eel_3d.xml",
  "solid_config": [
    {"solid_id": 0, "body_id": 1, "body_or_geom_name": "seg1", "lbm_position": [200, 350], "is_body": true}
  ]
}
```

Fields:

- `class`: Environment class name. Common values: `FishLBMEnv`, `GenericLBM2DEnv`.
- `xml_path`: MuJoCo XML path, relative to the project root.
- `solid_config`: Used by `GenericLBM2DEnv` to map XML bodies into 2D LBM solids.
  - `solid_id`: Solid index in the LBM solver, starting from `0`.
  - `body_id`: MuJoCo body id.
  - `body_or_geom_name`: Body or geom name.
  - `lbm_position`: Initial LBM grid position `[x, y]`.
  - `is_body`: `true` means the entry refers to a body.

### `lbm`: LBM grid and simulation substeps

```json
"lbm": {
  "nx": 400,
  "ny": 600,
  "lbm_scale": 0.25,
  "per_frame_steps": 8
}
```

Fields:

- `nx`: LBM grid width.
- `ny`: LBM grid height.
- `lbm_scale`: Scale factor from MuJoCo coordinates to LBM grid coordinates.
- `per_frame_steps`: Number of LBM/MuJoCo coupling substeps per control step. Larger values can improve stability but are slower.

Command-line overrides:

```powershell
--nx 400 --ny 600 --lbm-scale 0.25 --per-frame-steps 8
```

### `render`: Display window

The current 2D UI shows LBM rendering on the left and control-signal bars on the right.

```json
"render": {
  "type": "vorticity",
  "output_height": 720,
  "control_panel_width": 270,
  "vmax_scale": 0.2,
  "opengl_lbm_vmax": 1.0,
  "window_name": "Eel2D Projected Realtime Control",
  "record_fps": 30
}
```

Fields:

- `type`: LBM visualization type. Options: `vorticity`, `velocity`, `solid_boundary`.
- `output_height`: Output window height.
- `control_panel_width`: Width of the right-side control-signal panel. Default is approximately `output_height * 0.375`.
- `vmax_scale`: Colormap intensity scale for the OpenCV backend.
- `opengl_lbm_vmax`: LBM color range for the OpenGL backend.
- `window_name`: Window title.
- `record_fps`: Video recording FPS.

Note: Old fields such as `mujoco_width`, `mujoco_height`, `mujoco_background_rgb`, and `camera` are not required for the default LBM + control-signal UI.

### `control`: Control timing and action gain

```json
"control": {
  "dt": 0.01,
  "warmup_steps": 15,
  "transition_steps": 36,
  "start_mode": "idle",
  "action_gain": 1.0,
  "gain_step": 0.1
}
```

Fields:

- `dt`: Time step used by preset waveform generation.
- `warmup_steps`: Number of steps used to ramp action amplitude from zero at startup.
- `transition_steps`: Number of smoothing steps used when switching presets.
- `start_mode`: Preset used at startup. It must exist in `presets`.
- `action_gain`: Multiplier applied to generated actions before clipping to `[-1, 1]`.
- `gain_step`: Step size for runtime `+` / `-` action-gain adjustment.

Runtime keys:

- `+` or `=`: Increase `action_gain`.
- `-` or `_`: Decrease `action_gain`.
- `Space`: Pause/resume.
- `R`: Reset.
- `Q` or `Esc`: Quit.

Command-line overrides:

```powershell
--control-dt 0.01 --warmup-steps 15 --transition-steps 36 --start-mode idle --action-gain 1.0 --gain-step 0.1
```

### `controls`: Keyboard-to-preset mapping

```json
"controls": {
  "w": "forward",
  "a": "turn_l",
  "d": "turn_r",
  "s": "idle",
  "f": "fast",
  "x": "reverse"
}
```

The key is the keyboard input, and the value is the preset name from `presets`.

### `presets`: Preset actions

`presets` defines how each action mode is generated. Any name referenced by `controls` or `start_mode` must exist here.

#### `constant`: Constant action

```json
"idle": {
  "type": "constant",
  "values": [0, 0, 0, 0]
}
```

- `values` length must match the actuator count.
- Commonly used for `idle`.

#### `sine`: Generic sine action

Fish example:

```json
"forward": {
  "type": "sine",
  "components": [
    {"amp": 0.65, "freq": 2.4, "phase": 0.0, "bias": 0.0},
    {"amp": 0.95, "freq": 2.4, "phase": 1.35, "bias": 0.0}
  ]
}
```

- `components` length must match the actuator count.
- `amp`: Amplitude.
- `freq`: Frequency.
- `phase`: Phase offset.
- `bias`: Offset, often used for turning.

#### `eel_wave`: Eel traveling-wave action

```json
"forward": {
  "type": "eel_wave",
  "A": 0.28,
  "omega": -1.0,
  "omega_max": 12.566370614,
  "k_wave": 0.55,
  "head_bias": 0.0,
  "roll": 0.0
}
```

`eel_wave` assumes actuators are arranged in yaw/roll pairs:

```text
u0  = joint1_yaw
u1  = joint1_roll
u2  = joint2_yaw
u3  = joint2_roll
...
```

Fields:

- `A`: Yaw wave amplitude.
- `omega`: Normalized frequency direction and magnitude.
- `omega_max`: Maximum angular frequency.
- `k_wave`: Normalized wave number.
- `head_bias`: Head bias, used for turning.
- `roll`: Constant roll command. The default is `0.0`, so odd-numbered channels such as `u1/u3/...` do not oscillate.
- `head_amp`: Optional head amplitude ratio. Default: `0.05`.
- `k_max`: Optional maximum wave-number scale. Default: `1.5`.

### Minimal config template

```json
{
  "name": "my_case",
  "env": {
    "class": "FishLBMEnv",
    "xml_path": "envs/lbm/fish/fish_2d_v3.xml"
  },
  "lbm": {
    "nx": 400,
    "ny": 600,
    "lbm_scale": 0.2,
    "per_frame_steps": 10
  },
  "render": {
    "type": "vorticity",
    "output_height": 720,
    "control_panel_width": 270,
    "window_name": "LBM 2D Control"
  },
  "control": {
    "dt": 0.01,
    "warmup_steps": 10,
    "transition_steps": 24,
    "start_mode": "idle",
    "action_gain": 1.0,
    "gain_step": 0.1
  },
  "controls": {
    "w": "forward",
    "s": "idle"
  },
  "presets": {
    "forward": {
      "type": "sine",
      "components": [
        {"amp": 0.65, "freq": 2.4, "phase": 0.0, "bias": 0.0},
        {"amp": 0.95, "freq": 2.4, "phase": 1.35, "bias": 0.0}
      ]
    },
    "idle": {
      "type": "constant",
      "values": [0.0, 0.0]
    }
  }
}
```
