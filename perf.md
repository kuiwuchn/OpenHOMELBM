# LBM 3D Realtime 性能优化记录

运行命令：
```
python tools/lbm3d_realtime_control.py --config configs/realtime_3d/eel3d.json
```
硬件：RTX 4090 (24GB, sm_89) / Warp 1.15.0 / CUDA 12.9。
场景：eel（鳗鱼）多任务环境，网格 150×250×60 = 2.25M cells，D3Q27，`nworld=1`，`per_frame_steps=10`。

初始现象：交互窗口约 **3 FPS**。

---

## 问题分析（Profiling — before）

用 `--no-render` 隔离仿真开销，并对各阶段单独计时（每次 `wp.synchronize()` 后计时）。

每次 `env.step()`（= 10 个流体子步）实测分解：

| 阶段 | 每子步 | 每帧(×10) | 占比 |
|---|---|---|---|
| `lbm_solver.step()` | **24.3 ms** | 243 ms | **94%** |
| ↳ 其中 mesh 光线投射 | ~10.8 ms | 108 ms | 42% |
| ↳ 其中 核心 stream-collide | ~13.5 ms | 135 ms | 52% |
| MuJoCo `step`（CUDA graph） | 1.1 ms | 11 ms | 4% |
| 耦合 kernel + Python | ~0.5 ms | ~5 ms | 2% |
| **合计** | | **259 ms → 3.9 FPS** | |

- `--no-render` benchmark：`sim_fps=3.84, avg_step_ms=260.75`。
- MuJoCo 渲染额外 ~70 ms/帧（交互窗口 ~3 FPS vs 无头 3.9 FPS）。
- **结论：瓶颈是流体求解器，不是渲染。**

mesh 光线投射开销的验证：把 `mesh_ids` 清零（禁用光线查询）后重新捕获图，`lbm_solver.step()` 从 **24.3 ms → 13.5 ms**，即光线投射占 ~11 ms/子步。

### 根因

`stream_and_collide_3d`（`envs/lbm3d/lbm_func_3d.py`）里，**每个流体格点、每个格子方向**都调用 `get_cutcell_multi_3d`，后者**遍历全部 solid** 做 `wp.mesh_query_ray` BVH 查询。

实测 eel 是 **12 个独立 solid**（每个 8 顶点小盒子，缩放后包围半径仅 ~5 格）：

> 2.25M cells × 27 方向 × 12 solid ≈ **每子步 7.3 亿次 `mesh_query_ray`**，每帧 73 亿次。

而 12 个 solid 的窄带（半径+2）合计只占全域 **~1–2%**，其余 98%+ 的格点纯属陪跑。

---

## Fix #1：窄带包围球裁剪 mesh 光线投射

### 思路与正确性论证

命中点 = `ray_origin + ray_direction * cutcell`，`cutcell ∈ [0,1]`，`|ray_direction| ≤ √3`，所以**任何 cut cell 必然离固体表面不超过 √3 ≈ 1.73 格**。

给每个 solid 存一个**包围球**：
- 球心 = `solid_position`（正是 mesh 变换的旋转中心）；
- 半径 = 缩放后顶点到局部原点的最大范数（建 mesh 时算一次）。

因为顶点经任意旋转 `R` 后为 `solid_position + R·v_local`，到球心距离恒为 `‖v_local‖`——**球对旋转天然不变**，任何姿态都严格包住整个 solid。

于是：离所有 solid 距离 > `radius + 2`（2 > √3）的格点**在物理上不可能命中**，跳过其光线查询、直接走原有 `else` 普通 streaming 分支 —— 与"查询后未命中"是**完全同一条代码路径**，故结果**逐比特一致**。

### 改动点

1. `HomeFlow3D` 增加 `solid_bound_radius` 数组字段（`lbm_core_3d.py`）。
2. `create_solid_from_mesh` 计算并写入每个 solid 的包围半径（`lbm_solver_3d.py`）。
3. `stream_and_collide_3d` 加 per-cell `near_solid` 门控；`get_cutcell_multi_3d` 加 per-solid 距离早退（`lbm_func_3d.py`）。

涉及文件：
- `envs/lbm3d/lbm_core_3d.py`：`HomeFlow3D` 增加 `solid_bound_radius` 字段并在 `Initialize` 里置零。
- `envs/lbm3d/lbm_solver_3d.py`：`create_solid_from_mesh` 用 `max‖scaled_vertices‖` 算包围半径写入每个 flow。
- `envs/lbm3d/lbm_func_3d.py`：`get_cutcell_multi_3d` 加 per-solid 距离早退；`stream_and_collide_3d` 加 per-cell `near_solid` 门控（远离所有 solid 的格点整段跳过 27 方向的光线查询）。

### 正确性验证

脚本：`tools/verify_narrowband_culling.py`。

方法：在同一个预热状态（warmup 12 步）上，对**单次** `stream_and_collide_3d` 启动做 A/B——
裁剪 ON（真实半径） vs 裁剪 OFF（半径设为 `1e9`，等价于原始"全查询"路径），比对输出。
（注意：不能用"两次完整耦合仿真跑 N 步"来比对——LBM+刚体耦合是混沌的，力的 `atomic_add` 求和次序本就有 run-to-run 抖动，会被放大。必须在**相同输入、单次 kernel** 下比对。）

结果：

| 比较 | max\|Δ field_post\| | max\|Δ solid_force\| | max\|Δ solid_torque\| |
|---|---|---|---|
| ON vs OFF | **0.000e+00** | 4.3e-06 | 9.5e-06 |
| OFF vs OFF（atomic 次序噪声地板） | 0.000e+00 | 3.0e-06 | 8.3e-06 |

- 逐格场输出（`*_post`，无 atomic）**逐比特一致**。
- 力/力矩差异（~1e-5）与"OFF 跑两次"的 atomic 求和次序噪声地板同量级 → 差异**完全由原本就存在的 atomic 非确定性**解释，非裁剪引入。
- 结论：**裁剪保结果正确**。`RESULT: PASS`。

---

## Profiling — after（Fix #1 生效）

| 阶段 | 每子步 before | 每子步 after | 说明 |
|---|---|---|---|
| `lbm_solver.step()` | 24.3 ms | **13.95 ms** | mesh 开销几乎归零 |
| ↳ mesh 光线投射 | ~10.8 ms | **~0.04 ms** | `step()` 已等于 no-mesh 地板（13.91 ms），约 **270× 缩减** |
| ↳ 核心 stream-collide | ~13.5 ms | ~13.9 ms | 不变（预期的地板） |
| MuJoCo `step` | 1.1 ms | 1.0 ms | |
| **full `env.step()`（×10）** | 259 ms | **155 ms** | |

`--no-render` benchmark：`sim_fps 3.84 → 6.32`，`avg_step_ms 260.75 → 158.11`。

**收益汇总**：
- LBM 单步 24.3 → 13.95 ms（**1.74×**）
- 整帧 259 → 155 ms（**1.67×**）
- 无头仿真 3.84 → 6.32 FPS（**1.65×**）
- mesh 光线投射分量 ~108 ms/帧 → ~0.4 ms/帧（基本消除）

复现：
```
# before/after 阶段计时见思路描述；标准无头基准：
python tools/lbm3d_realtime_control.py --config configs/realtime_3d/eel3d.json --no-render --benchmark-steps 60
# 正确性：
python tools/verify_narrowband_culling.py
```

**剩余瓶颈**：核心 stream-collide 约 13.9 ms/子步（135 ms/帧）成为新的地板——矩基 LBM 的邻居 gather 访存不合并，安全提速需算法/访存重构（后续项）。此外交互路径的 MuJoCo 渲染 ~70 ms/帧、`EelMultiTaskEnv.step()` 里每步 `.numpy()` 主机同步、以及子步循环的单 CUDA graph 融合，均为下一步候选。

---

## Fix #1 之后的进一步排查（结论：已接近安全优化的极限）

对 Fix #1 之后的帧做了细粒度归因，**修正了之前对渲染开销的估计**，并逐项排除了其余候选：

| 分量 | 每帧耗时 | 占比 | 结论 |
|---|---|---|---|
| `_simulation_step`（10 子步+耦合） | 152.6 ms | 98.5% | 真正的大头 |
| ↳ 核心 stream-collide（×10） | ~139.5 ms | 90% | **地板，访存受限** |
| ↳ MuJoCo step（×10） | ~10 ms | 6% | |
| ↳ 耦合 kernel + 每子步 sync | ~3 ms | 2% | |
| 子类 Python（disp_vel/ema/reward/obs/stats） | ~2.3 ms | 1.5% | 各 0.05–0.12 ms |
| **MuJoCo 渲染（交互路径）** | **~2.2 ms** | — | **之前估的 ~70 ms 是错的** |

排查记录：
1. **渲染并非瓶颈**：offscreen 720×720 渲染仅 ~1.1 ms（GL）+ `get_mujoco_frame` 全程 ~2.2 ms；`qpos.numpy()`/`mj_forward`/`update_scene` 均 <0.1 ms。之前的 ~70 ms 来自"用户口述 3 fps"与 3.84 fps 基准的差值，不可靠。Fix #1 后交互 FPS ≈ 仿真 FPS ≈ **6.3**。
2. **`block_dim` 已最优**：stream_collide 在 block_dim=256（默认）最快（13.68 ms）；128→14.6、64→17.1 更差，512 寄存器溢出。启动级无可榨取。
3. **去掉每子步 `wp.synchronize()`**：仅省 ~1.3 ms/帧（154.6→153.3，<1%）。流内本已保序，主机端循环中无 `.numpy()` 读取，故安全但收益极小，**不改**。
4. **子类每步 `.numpy()` 主机同步**：nworld=1 时合计仅 ~2.3 ms，不值得改（多世界训练时才有意义）。

### 结论

Fix #1 之后，**96% 的帧时间是核心 stream-collide kernel**（矩基/正则化 LBM，每格从 26 个邻居各读 10 个矩 ≈ 260 次读，约为群体（population）法的 ~10× 访存量），受**显存带宽/占用率**限制。在**保结果完全一致**的约束下，已无更多"低风险、大收益"的空间（其余候选合计 <5%）。

进一步提速只剩两条路，都**不属于**"安全快赢"，需用户决策：
- **A. 可调物理档位（会改结果）**：`per_frame_steps` 10→5（≈2×）、网格分辨率下调（≈线性）。适合 demo，需接受数值差异。
- **B. LBM 存储方案重构（大改，需重验证）**：改存 27 个群体分布（two-lattice / AA-pattern），访存从 ~270→~54 次/格，理论 ~3–5× 潜力。理论上与当前矩重构在同一公式下数值等价，但跨如此重构做到逐比特一致极难保证，需大量回归验证。

---

## Fix #2 调查：kernel 融合（工程角度）——实测后判定不值得做

用户要求从工程角度"把多个 kernel 合一"。逐项实测了 `lbm_solver.step()` 内部与整帧的融合上限：

`lbm_solver.step()` 内部各 kernel（eel，nworld=1）：

| kernel | 启动维度 | 耗时 | 占比 |
|---|---|---|---|
| `stream_and_collide_3d` | nworld×nx×ny×nz | **13.67 ms** | **96.6%** |
| `apply_bc_3d` | nworld×nx×ny×nz | 0.244 ms | 1.7% |
| `init_force_3d_batch` | nworld | 0.019 ms | — |
| `Swap_Mom_3D` | nworld | 0.016 ms | — |

整帧 CUDA graph 融合实测（把 10 个子步的全部耦合+LBM+MuJoCo kernel 录成**单个 graph** 回放）：

| 方案 | 每帧 | 说明 |
|---|---|---|
| python 循环（现状，MuJoCo 用预捕获子图） | 153.4 ms | |
| **单 graph 回放（全部 inline 后录制）** | **150.6 ms** | 融合上限，仅省 ~3 ms（**~2%**） |
| python 循环但 `mjw.step` 不捕获直接调 | 295 ms | 反例：MuJoCo 不用 graph 会因启动开销暴涨 ~2× |

**结论：融合不值得做。**
- 96.6% 的帧是**单个** `stream_collide` kernel（本就一次启动），其余所有 kernel + 每子步 sync 合计仅 ~6 ms/帧，融合上限 ~2%。
- 要拿到这 ~2% 必须把 `mujoco_warp` 的内部 kernel inline 进外层 capture 并重捕整个循环——跨 mjw 版本脆弱、且打乱现有干净的子图结构与 reset/partial_reset 路径。**为 ~2% 引入这种脆弱性属于糟糕的工程取舍，不做。**
- 附带发现：`mjw.step` 若不走预捕获 graph，每帧从 153→295 ms。现有"MuJoCo 单步捕获成子图"已是正确做法，勿动。

因此，在保结果一致的前提下，**Fix #1 已是安全优化的终点**；后续只剩前述 A（改物理档位）/ B（群体法重构）两条需权衡的路。
