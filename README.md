# LBM-RIGID

## 环境配置

### 第一步：创建 Conda 环境

```powershell
conda create -n dreamer python=3.11 -y
conda activate dreamer
```

### 第二步：安装 PyTorch（CUDA 12.8）

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 第三步：安装 MuJoCo Warp

```powershell
pip install mujoco-warp
```

### 第四步：安装项目依赖

```powershell
pip install -r requirements.txt
```

## 运行命令

```powershell
python tools/lbm2d_realtime_control.py --config configs/realtime_2d/eel2d.json
```


## JSON 配置说明

2D 实时控制入口使用 JSON 文件配置场景、LBM 网格、渲染、键盘控制和预设动作。配置文件位于：

```text
configs/realtime_2d/fish2d.json
configs/realtime_2d/eel2d.json
```

顶层结构：

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

### `env`：环境和模型

`env` 指定使用哪个环境类、哪个 MuJoCo XML，以及哪些刚体参与 2D LBM 投影。

fish 示例：

```json
"env": {
  "class": "FishLBMEnv",
  "xml_path": "envs/lbm/fish/fish_2d_v3.xml"
}
```

 eel 示例：

```json
"env": {
  "class": "GenericLBM2DEnv",
  "xml_path": "envs/lbm3d/eel/eel_3d.xml",
  "solid_config": [
    {"solid_id": 0, "body_id": 1, "body_or_geom_name": "seg1", "lbm_position": [200, 350], "is_body": true}
  ]
}
```

字段说明：

- `class`：环境类名。常用：`FishLBMEnv`、`GenericLBM2DEnv`。
- `xml_path`：MuJoCo XML 路径，相对于项目根目录。
- `solid_config`：`GenericLBM2DEnv` 使用，用来把 XML 里的多个 body 映射到 2D LBM 固体。
  - `solid_id`：LBM 里的固体编号，从 `0` 开始。
  - `body_id`：MuJoCo body id。
  - `body_or_geom_name`：body 或 geom 名字。
  - `lbm_position`：初始 LBM 网格坐标 `[x, y]`。
  - `is_body`：`true` 表示按 body 读取。

### `lbm`：LBM 网格和仿真步数

```json
"lbm": {
  "nx": 400,
  "ny": 600,
  "lbm_scale": 0.25,
  "per_frame_steps": 8
}
```

字段说明：

- `nx`：LBM 网格宽度。
- `ny`：LBM 网格高度。
- `lbm_scale`：MuJoCo 到 LBM 网格的尺度系数。
- `per_frame_steps`：每个控制 step 内执行多少个 LBM/MuJoCo 耦合子步。越大越稳定但越慢。

命令行可覆盖：

```powershell
--nx 400 --ny 600 --lbm-scale 0.25 --per-frame-steps 8
```

### `render`：显示窗口

当前 2D 界面左侧显示 LBM，右侧显示控制信号进度条。

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

字段说明：

- `type`：LBM 可视化类型，可选：`vorticity`、`velocity`、`solid_boundary`。
- `output_height`：窗口输出高度。
- `control_panel_width`：右侧控制信号面板宽度。默认约为 `output_height * 0.375`。
- `vmax_scale`：OpenCV 后端颜色映射强度缩放。
- `opengl_lbm_vmax`：OpenGL 后端 LBM 颜色范围。
- `window_name`：窗口标题。
- `record_fps`：录制视频帧率。

说明：旧配置里的 `mujoco_width`、`mujoco_height`、`mujoco_background_rgb`、`camera` 等字段现在对默认 LBM+控制信号界面不是必需项。

### `control`：控制时间和动作系数

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

字段说明：

- `dt`：预设动作波形的时间步长。
- `warmup_steps`：启动时动作幅度从 0 逐步增大的步数。
- `transition_steps`：切换 preset 时平滑过渡步数。
- `start_mode`：启动时使用的 preset 名称，必须存在于 `presets`。
- `action_gain`：动作倍率，最终动作会乘这个系数后再裁剪到 `[-1, 1]`。
- `gain_step`：运行时按 `+` / `-` 调整动作倍率的步长。

运行时快捷键：

- `+` 或 `=`：增大 `action_gain`
- `-` 或 `_`：减小 `action_gain`
- `Space`：暂停/继续
- `R`：重置
- `Q` 或 `Esc`：退出

命令行可覆盖：

```powershell
--control-dt 0.01 --warmup-steps 15 --transition-steps 36 --start-mode idle --action-gain 1.0 --gain-step 0.1
```

### `controls`：键盘到动作模式的映射

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

左边是键盘按键，右边是 `presets` 里的 preset 名称。

### `presets`：预设动作

`presets` 定义每种模式下的动作生成方式。`controls` 和 `start_mode` 引用的名称必须存在于这里。

#### `constant`：常量动作

```json
"idle": {
  "type": "constant",
  "values": [0, 0, 0, 0]
}
```

- `values` 长度必须等于 actuator 数量。
- 常用于 `idle`。

#### `sine`：通用正弦动作

fish 示例：

```json
"forward": {
  "type": "sine",
  "components": [
    {"amp": 0.65, "freq": 2.4, "phase": 0.0, "bias": 0.0},
    {"amp": 0.95, "freq": 2.4, "phase": 1.35, "bias": 0.0}
  ]
}
```

- `components` 数量必须等于 actuator 数量。
- `amp`：振幅。
- `freq`：频率。
- `phase`：相位。
- `bias`：偏置，用于转向等。

#### `eel_wave`：eel traveling wave 动作

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

`eel_wave` 假设 actuator 按 yaw/roll 成对排列：

```text
u0  = joint1_yaw
u1  = joint1_roll
u2  = joint2_yaw
u3  = joint2_roll
...
```

字段说明：

- `A`：yaw 波幅。
- `omega`：归一化频率方向和大小。
- `omega_max`：最大角频率。
- `k_wave`：归一化波数。
- `head_bias`：头部偏置，用于转向。
- `roll`：roll 通道常量。当前默认 `0.0`，所以奇数 `u1/u3/...` 不摆动。
- `head_amp`：可选，头部波幅比例，默认 `0.05`。
- `k_max`：可选，最大波数比例，默认 `1.5`。

### 最小配置模板

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
