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
