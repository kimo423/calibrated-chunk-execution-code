#!/usr/bin/env python
"""Assemble docs/reports/CCE_sim_probe_executor.md from the prose below + generated tables."""

import subprocess
import sys
from pathlib import Path

ROOT = Path("/opt/cce")
OUT = ROOT / "docs" / "reports" / "CCE_sim_probe_executor.md"

PROSE = r"""# CCE 仿真侧实现与首轮结果：接口层 plant 族、策略动作空间探针、ARX 辨识与执行器反演

适用范围：方案 B2（CCE）中不依赖 VLA 的部分。本文覆盖 plant 族、探针、辨识、执行器反演，以及两个不需要策略的早期实验——合成参考轨迹跟踪基准与专家演示回放。冻结 VLA 的锚定、Gate 0 与策略侧对照臂不在本文范围。

证据等级：DEVELOPMENT_ONLY。全部结论限于 ManiSkill3 PickCube-v1、`pd_ee_delta_pose` 共享控制接口、physx_cpu 后端；不构成对真机的外推，theta-hat 也不声称等于任何物理参数。

日期：2026-09-05。机器：03 服务器（127.0.0.1）。全程未使用 GPU。

---

## 1. 结论摘要

1. **plant 族已建成且可辨识。** 26 个接口层 plant（含标称、D1-D4 单因子、10 个混合、2 个过响应）叠在 `pd_ee_delta_pose` 命令之上。7.00 s 探针在 Panda 与 xArm6-Robotiq 上分别辨识：命令延迟 26/26 精确命中（MAE 0 ms），夹爪时序提前量 26/26 精确命中（MAE 0 步）；相对时间常数在 xArm6 上 MAE 0.008 s、相关 0.999，在 Panda 上 MAE 0.078 s、相关 0.827；相对增益在 xArm6 上 MAE 0.029、Panda 上 MAE 0.196。156 个轴全部通过 R2 >= 0.7 质量门。
2. **K1（跟踪 RMSE 降 >= 30%）通过，且余量很大。** 在含延迟/滞后的 plant 子集上，标定后跟踪 RMSE 相对 naive 平均下降 **76.3%（Panda）/ 93.2%（xArm6）**，plant 级配对 t 分别为 3.44 与 3.53，25 个 plant 中 24 个变好。夹爪相位误差从 0.25 s / 0.34 s 降到 0.001 s / -0.019 s。
3. **K2（回放恢复率 R >= 0.5）通过。** 专家演示回放中，标称 plant 上 naive 成功率 1.00（两个本体均满足 >= 0.9 的前置门）；在 naive 明显下降的 plant 上，宏恢复率 **R = 0.796（Panda，n = 10）/ 0.972（xArm6，n = 8）**；若把分母换成两种 naive 变体中更强的一种，仍有 0.610 / 0.967。
4. **优于最强通用基线。** 全族宏成功率：ours-P 0.952 / 0.982，Slow-down x2 为 0.867 / 0.825，naive 0.708 / 0.736。ours-P 与 Slow-down x2 的差为 +8.5 pp（Panda，配对 t = 1.41，CI95 含 0）与 +15.8 pp（xArm6，配对 t = 2.61，CI95 [0.053, 0.283]）。同时时间到成功从 7.23 s / 7.41 s 缩短到 4.04 s / 4.18 s，**缩短约 44%**，远超"SR 持平时缩短 >= 25%"的替代判据。
5. **与预期不符的一条。** 任务书预期"D1 d = 150 ms 与 D2 tau = 250 ms 上 naive 应明显下降"未复现：以**无误差的专家动作序列**作策略时，两个本体在这两档上的 naive 成功率是 0.86 / 1.00（Panda）与 1.00 / 1.00（xArm6）。崩塌发生在 d = 250 ms、T_g = 1.0 s 以及混合 plant 上（26 个 plant 中有 7 个 naive 相对标称下降 >= 25 pp）。学生此前 0.6 -> 0.0 的数据来自标称成功率仅 0.6 的学习型锚定策略，余量远小于专家回放，两者不可直接对齐。详见 9.2。
6. **方法唯一的负例已定位并修好。** 主配方在 D4（每步位置限幅）上会变差：xArm6 上 D4_c05 回放成功率 1.00 -> 0.80，跟踪 RMSE 0.0009 -> 0.0074 m。根因不是"限幅未知"，而是限幅段进入了 ARX 拟合窗口、把增益压低，执行器随即过量下发命令。加一条饱和检测后，D4_c05 完全恢复（xArm6 SR 1.00、RMSE 0.00016 m），其余 25 个 plant 一字未变。见 10.1。

---

## 2. 环境、入口与复现命令

- 仓库 `/opt/cce`（本文未运行任何 git 命令）。学生目录 `CalibrationToPrompt` 全程只读引用。
- 入口 `cce/env.sh <script.py> [args]`，内部为学生的 `tools/run_clean_env.sh` + `conda run -p /opt/CalibrationToPrompt/envs/ctp-e0 python`（py3.10、mani_skill 3.0.1、sapien 3.0.3、torch 2.1.2）。
- 对 `cce/env.sh` 做过一处最小改动：追加 `export PATH="/opt/runtime/miniconda3/bin:$PATH"`。非交互 ssh 的 PATH 里没有 `conda`，原脚本会以 exit 127 失败。
- 仿真后端 `sim_backend="physx_cpu"`、`render_backend="none"`、`obs_mode="state_dict"`，**全程未占用 GPU**。单步耗时约 4.7 ms；26 plant x 6 臂 x 50 集的回放实验分 13 片并行约 7 min。

复现顺序：

```bash
cd /opt/cce

# 0) 单元自检：numpy 编解码 vs 学生 torch 编解码；oracle 执行器闭合自身模型
./cce/env.sh cce/validate_codec.py

# 1) 探针（每 plant 7.00 s）
./cce/env.sh cce/probe.py --uid panda
./cce/env.sh cce/probe.py --uid xarm6_robotiq

# 2) ARX 辨识（主配方 / 饱和感知版）
./cce/env.sh cce/identify.py --uid panda
./cce/env.sh cce/identify.py --uid xarm6_robotiq
./cce/env.sh cce/identify.py --uid panda          --sat-aware --out-dir data/cce/theta_sat
./cce/env.sh cce/identify.py --uid xarm6_robotiq  --sat-aware --out-dir data/cce/theta_sat

# 3) 两个基准，按 plant 分 13 片
./cce/run_shards.sh replay   panda           13 -
./cce/run_shards.sh replay   xarm6_robotiq   13 -
./cce/run_shards.sh tracking panda           13 -
./cce/run_shards.sh tracking xarm6_robotiq   13 -

# 4) 三组消融（xArm6 同理，替换 uid）
./cce/run_shards.sh replay   panda 13 sat  --theta-dir data/cce/theta_sat --methods ours_p
./cce/run_shards.sh tracking panda 13 sat  --theta-dir data/cce/theta_sat --methods ours_p
./cce/run_shards.sh replay   panda 13 clip --clip-from-burst --methods ours_p
./cce/run_shards.sh tracking panda 13 clip --clip-from-burst --methods ours_p
./cce/run_shards.sh tracking panda  5 qp   --methods ours_qp --plants P00_nominal,D1_d150,D1_d250,D2_t250,MX05

# 5) 汇总
./cce/env.sh cce/merge_results.py
./cce/env.sh cce/make_tables.py > /tmp/cce_tables.md
```

分片日志在 `data/logs/<bench><tag>_<uid>_<i>.log`，退出码在同名 `.exit`；本轮 104 + 52 + 52 个分片全部 exit 0。

## 3. 代码与数据落点

| 文件 | 职责 |
|---|---|
| `cce/common.py` | 归一化动作编解码（纯 numpy 复刻学生 torch 版）、位姿误差、slerp、环境构造、夹爪开度观测 |
| `cce/plants.py` | `PlantSpec` / `PlantWrapper`，26 个 plant 的冻结定义 |
| `cce/probe.py` | 7.00 s 探针的参考序列与命令序列、执行与落盘 |
| `cce/identify.py` | 每轴 ARX(1,1) + 整数延迟、自由仿真 R2、夹爪 90% 到达时间、饱和检测 |
| `cce/executor.py` | `Theta`、`shape_reference`、`InversionExecutor`（参数版与 QP 版）、`build_arm` |
| `cce/bench_tracking.py` | 实验 a：合成参考轨迹跟踪 |
| `cce/bench_replay.py` | 实验 b：专家演示回放 |
| `cce/merge_results.py` / `cce/make_tables.py` | 分片汇总、配对统计、Markdown 表 |
| `cce/run_shards.sh` | 按 plant 分片并发 |
| `cce/validate_codec.py` | 单元自检 |

数据：探针轨迹 `data/cce/probes/probe_<uid>_<pid>.npz`；theta-hat 在 `data/cce/theta/` 与 `data/cce/theta_sat/`；意图轨迹缓存 `data/cce/intent_<uid>.npz`；分片原始结果 `data/cce/parts/`。结果 JSON：`cce/results/{identify_summary,tracking_summary,replay_summary}.json`（合计约 160 KiB）。

单元自检结果：numpy 编解码与学生 `pilot2_task_plan` 的 torch 实现最大偏差 3.0e-8（学生侧返回 float32）；oracle 执行器在纯模型闭环上的稳态跟踪 RMSE 为 1.0e-10 m，即反演代数本身无误。

## 4. 接口层 plant 族

plant 是加在策略输出与 `env.step` 之间的一层，不改物理引擎。每个控制步（20 Hz）依次执行：命令延迟队列（按步取整）-> 一阶滞后 -> 稳态增益 -> 每步限幅；夹爪通道单独走"传输延迟 + 执行延迟"的 FIFO，不做滤波。这是学生 `tools/dev_residual_benchmark.py:ResidualPlant` 的直接扩展。

夹爪的总提前量定义为 `d + T_g`（传输延迟同样作用于夹爪指令，符合物理；D3 单因子 plant 的 `d = 0`，因此该轴仍然干净）。

族内同时含欠响应（g = 0.70 / 0.85）与过响应（g = 1.15，以及大 tau 在闭环里造成的过冲）。

## 5. 探针：两处必须记录的实现决定

**（1）探针必须开环。** 最初按"绝对 EE 目标 -> `shared_ee_delta_action`(当前实测位姿) -> plant -> env"实现，即闭环残差适配器。该回路在 d >= 100 ms 时发散：D1_d250 上 TCP 的 z 最低到 0.0095 m，机械臂直接撞进桌面，探针数据被接触污染。改为把适配器的"当前位姿"换成完美 plant 下的**虚拟参考位姿**后，命令序列与 plant 无关（保证不同 plant 的 theta-hat 可比），26 个 plant 的 z 最低值稳定在 0.163-0.171 m。这一现象本身是方法论证据：把绝对位姿锚点转成 delta 命令的标准执行器，在几百毫秒延迟下是不稳定的。

**（2）时序编排要给限幅段留缓冲。** 第一版把限幅 burst 放在姿态正弦之前，burst 的平动瞬态在有延迟时溢进姿态窗口，旋转轴的一步 R2 掉到 0.0-0.06。改为"chirp -> burst -> 静默间隔 -> 姿态正弦 -> 保持 + 夹爪"后，旋转轴自由仿真 R2 稳定在 0.97-1.00。

最终编排（140 步 = 7.00 s）：

| 步区间 | 时长 | 内容 |
|---|---|---|
| [0, 4) | 0.20 s | 在复位位姿静置 |
| [4, 72) | 3.40 s | xyz 三轴线性 chirp，0.2 -> 2.0 Hz，+-3 cm，相位互差 120 度 |
| [72, 80) | 0.40 s | 水平饱和 burst（+-x、+-y 满量程，避开桌面方向） |
| [80, 88) | 0.40 s | 静默间隔，让 burst 在 d = 250 ms 下也已冲刷完毕 |
| [88, 102) | 0.70 s | 三轴姿态正弦，+-0.10 rad |
| [102, 140) | 1.90 s | 保持位姿；夹爪在 102 步闭合并保持 |

## 6. 辨识

对每个轴 i（xyz 与 rx/ry/rz 共 6 轴）拟合

    w_i[t] = a * w_i[t-1] + b * u_i[t-d]

其中 `u` 是交给 plant 的命令，`w[t] = norm_delta(y[t], y[t+1])` 是**实际达成**的归一化位移。d 在 0-6 步（0-300 ms）格点搜索，(a, b) 闭式最小二乘，再换算 alpha = a、tau = -dt/ln a、g = b/(1-a)。夹爪由闭合阶跃的 90% 到达时间给出，减去同一本体标称 plant 的同一量，得到提前步数。

两处偏离直觉但必要的做法：

- **延迟按一步预测 R2 选，拟合窗口按自由仿真 R2 选。** 一步预测 R2 在所有 plant 上都 >= 0.98，对 (tau, g) 的错误分解毫无鉴别力；执行器却是多步预测。改用自由仿真（不喂实测反馈）R2 作窗口选择后，26 个 plant 一律选中含 burst 的窗口（burst 提供了 chirp 缺乏的低频成分）。但若把延迟也交给自由仿真选，延迟与滞后会互相混叠，延迟 MAE 从 0 步涨到 0.35 步——所以两级分开。
- **theta-hat 是"本体自身响应 x 注入 plant"的复合量，必须相对化后才能与真值比。** 标称 plant 上 g-hat = 0.408（Panda）/ 0.390（xArm6）：仿真器自己在 50 ms 内只完成约 40% 的命令位移。表 T2 中的 `rel-hat` 列即除以（增益）或减去（时间常数）同一本体标称 plant 的估计值。纯增益 plant 的相对增益几乎精确（0.70 / 0.85 / 1.15 全部命中两位小数）；大 tau plant 上 (tau, g) 存在互换自由度，Panda 尤为明显，因为它自身的 tau-hat = 0.056 s 与注入量同量级，两个一阶环节被单极点模型合并。这也是不把 theta-hat 写成物理参数的直接理由。

## 7. 执行器反演

**参数版（ours-P）** 每步做四件事：

1. **相位超前 / Smith 式预测**：把仍在 plant 管线里的 d 条命令按模型冲刷一遍，得到 t+d 时刻的预测位姿 y-hat；
2. **增益预补**：由 y-hat 到参考的期望达成增量 w_des 除以 g-hat 得到 z_req；
3. **超前校正**：一阶滞后的一步逆滤波 `u = (z_req - alpha_hat * z_pred) / (1 - alpha_hat)`，再截到 +-1；
4. **夹爪提前**：夹爪通道喂 `grip[t + 提前步数]`。

`invert(c_seq, theta_hat, y_now) -> u_seq` 提供开环批式接口（只滚动内部模型、不碰环境），`InversionExecutor.step` 提供闭环单步接口，二者共用同一套模型记账。

**一个关键设计选择：执行器的设定点不是意图本身，而是"标称本体对该意图的预测达成轨迹"。** 意图 `c` 是绝对 TCP 目标序列，而标称本体一步只走约 40%，从来没真正到达过 `c`。若让标定后的执行器精确落在 `c` 上，它会比演示快一倍以上，直接把任务做坏。因此先用标称 plant 的 theta-hat_N 闭环仿真出 `R = shape_reference(c, theta_hat_N, y0)`，再用本 plant 的 theta-hat 反演去跟踪 `R`。在标称 plant 上该配方自然退化为 naive，这与恢复率 R 的定义（分母是标称行为）完全对齐，实测标称 plant 上 ours-P 的成功率 1.00、跟踪 RMSE 与 oracle 同级。

**对照臂定义（务必在论文里写清）：**

- `naive`：恒等 theta，跟踪意图 `c`，即 `u = c` 的标准落地方式（用实测位姿把绝对锚点转成 delta）。
- `naive_open`：把演示记录的归一化动作逐条原样重放，序列耗尽后保持不动。开环变体，对纯延迟免疫，对增益/滞后极脆。
- `ours_p`：探针辨识的 theta-hat + 反演，跟踪 `R`。
- `oracle`：**注入 plant 的真值参数**与**同一份标称本体响应模型**级联（两个一阶环节按时间常数相加近似为一个），跟踪 `R`。它不知道仿真器自身的真实响应，只是把 (d, tau, g, clip, T_g) 的辨识误差归零，因此 ours-P 与 oracle 的差就是辨识质量的读数。
- `slow15` / `slow20`：恒等 theta，跟踪时间缩放 x1.5 / x2 后的意图。

**QP 版**用 `scipy.optimize.lsq_linear` 在 H = 30 步的滚动窗口上解带界最小二乘 `min ||A u + f - r||^2 + lam*||du||^2 + mu*||u||^2`（仅平动通道，旋转与夹爪沿用闭式解），每 5 步重解一次。作为可选实现已跑通，但在本族上劣于闭式版（见 10.3）。

## 8. 实验 a：合成参考轨迹跟踪

每条参考 8.0 s（160 步）：前 4 s 是复位位姿附近安全盒内的平滑随机三次样条加慢姿态扰动；后 4 s 是 PickCube 式的接近 -> 下降 -> 闭合夹爪 -> 抬起，锚在该 episode 真实的方块位置上。每个 plant 20 条轨迹、5 条对照臂，共 26 x 20 x 5 = 2600 集。

误差同时对两个基准报告：`rmse_pos_m` 相对 `R`（标称本体在同一意图下的达成轨迹，即"恢复"的机制读数，也是下表主列），`rmse_pos_vs_intent_m` 相对原始意图 `c`。Slow-down 臂按各自缩放后的参考在自己的时间轴上计分，其时长为 12 s / 16 s，须与 8 s 的其它臂并列理解。

## 9. 实验 b：专家演示回放

### 9.1 意图轨迹的构造

演示是 `pd_ee_delta_pose` 的归一化 delta 序列（Panda 50 条、长度 57-91 步；xArm6-Robotiq 50 条、长度 56-200 步）。用同 seed 复位后在标称 plant 上重放一遍，逐步记录 `c[t] = apply_norm_delta(y_t, a_t[:6])`，得到绝对 TCP 目标序列与夹爪指令序列，缓存在 `data/cce/intent_<uid>.npz`。该定义使 naive 在标称 plant 上逐步复现演示，标称成功率 1.00，满足"标称 plant 上 naive 回放 SR >= 0.9"的前置门。

每个 plant 50 集（对应 50 条演示、同 seed 复位），预算 200 步（PickCube 注册地平线）。恢复率 `R = (SR_calib - SR_naive)/(SR_nominal - SR_naive)`，仅在 `SR_nominal - SR_naive > 10 pp` 的 plant 上计算，避免分母噪声。

### 9.2 关于"naive 应该崩掉"的偏差

任务书按学生既有数据预期 d = 150 ms -> 0.0、tau = 250 ms -> 0.2。本轮以专家动作序列作策略，这两档上 naive 分别是 0.86 / 1.00（Panda）与 1.00 / 1.00（xArm6）。差异的来源是余量：学生的锚定策略在标称 plant 上本就只有约 0.6 的成功率，任何额外扰动都能把它推过失败阈值；专家回放的标称成功率是 1.00。

因此本文的证据链不依赖"任意一档 D1/D2 都能打崩 naive"，而依赖族内实际出现的崩塌：26 个 plant 中有 7 个 naive 相对标称下降 >= 25 pp（`D1_d250`、`D3_g10`、`MX04`、`MX05`、`MX07`、`MX09`、`MX10`，两个本体一致），恢复率就在这些 plant 上统计。装上真实冻结策略后余量会显著变小，naive 的崩塌面预期会更宽，本轮结果应视为**对方法的保守估计**。

### 9.3 两种 naive 的分工

`naive`（闭环）在 d = 250 ms 上直接归零（Panda 0.00、xArm6 0.06），而 `naive_open`（开环重放）在同一 plant 上是 0.98-1.00；反过来在增益/滞后 plant 上 `naive_open` 崩得更厉害（`D2_g70` 上 0.00、`D2_t400` 上 0.04-0.28、多个混合 plant 上 0.00）。两者都不是全域可用的，ours-P 同时压过两者：与更强那一支的配对差为 +15.2 pp（Panda，t = 2.37）与 +17.8 pp（xArm6，t = 2.49），CI95 均不含 0。审稿时若被要求"用开环重放做 naive"，这一列是现成的回答。

## 10. 消融

### 10.1 饱和感知辨识（建议纳入配方）

主配方把每步限幅留作未辨识（`clip = 1.0`），是唯一让标定变差的地方：`D4_c05` 上 xArm6 回放 1.00 -> 0.80、跟踪 RMSE 0.0009 -> 0.0074 m，Panda 跟踪 RMSE 0.0011 -> 0.0035 m。

根因不是"上界未知"，而是**限幅段落在拟合窗口里、把 ARX 增益压低了**：`D4_c05` 的相对增益被估成 0.84（Panda）与 0.60（xArm6），真值 1.00。执行器随即按偏小的增益过量下发命令，预测与实际脱节。

饱和检测规则：用不含 burst 的窗口（chirp only）拟合出的模型自由仿真预测 burst 段；仅在该轴 burst 被真正激励（预测峰值 > 0.15）且实测峰值 < 0.75 x 预测峰值时判定饱和，此时改用 chirp-only 拟合结果，并把实测峰值作为**达成域**上界交给执行器。

结果：两个本体各只有 `D4_c05` 被判定饱和（增益修正到 0.408 / 0.388，即回到标称水平；限幅估计 0.196 / 0.199，真值 0.204 / 0.195），其余 25 个 plant 的 theta-hat 与结果一字未变。`D4_c05` 上 xArm6 回放回到 1.00、跟踪 RMSE 0.00016 m，Panda 跟踪 RMSE 0.00063 m（与 oracle 同级）。全族宏值：xArm6 回放 0.982 -> 0.990，跟踪 RMSE 0.00088 -> 0.00059 m；Panda 回放持平 0.952，跟踪 RMSE 0.00402 -> 0.00391 m。

顺带修掉的一个模型域错误：plant 的限幅作用在**命令域**，而执行器模型最初把它当作**达成域**的界。二者相差一个本体增益（约 0.4）。已改为在达成域统一表达（oracle 用 `clip_cmd x g_nominal`，饱和估计直接用观测到的达成峰值）。

### 10.2 直接用 burst 峰值当限幅（否定）

若不做饱和判定、无条件把 burst 观测峰值当限幅上界，结果显著变差：Panda 全族宏成功率 0.952 -> 0.874（`MX06` 0.88 -> 0.06、`MX08` 0.96 -> 0.26、`D2_t400` 0.96 -> 0.64）。原因是大 tau plant 上 2 步 burst 远未到稳态，观测峰值严重低估可达速率，执行器把自己憋死。**限幅估计必须由饱和判定把门。**

### 10.3 QP 版反演（否定）

在 5 个 plant x 20 条轨迹的子集上，QP 版跟踪 RMSE 一律劣于闭式版（标称 0.00189 vs 0.00060；`D1_d150` 0.00274 vs 0.00179；`D1_d250` 0.00853 vs 0.00648；`D2_t250` 0.01310 vs 0.00321；`MX05` 0.01881 vs 0.01303）。滚动窗口每 5 步重解一次带来的锯齿、以及 du 正则对超前校正的抑制，在这一族上都没有回报。本族 plant 是分段线性的，闭式三步解已经是最优反演；QP 的价值应留给 C3 所指向的非线性 plant。

## 11. 异常、限制与不确定性

1. **确定性不完全。** 同一配置两次全量重跑，`MX06` 的 Panda naive 成功率为 0.80 与 0.82，其余单元差异 <= 0.02、跟踪 RMSE 差异 <= 2e-6 m。属进程级浮点/调度噪声，不影响任何结论方向，但报告 SR 时应记为 +-0.02。
2. **(tau, g) 不可分辨。** Panda 上大 tau plant 的相对增益估到 1.5-2.2（真值 1.00），相对时间常数同步偏大；这是单极点模型合并两个一阶环节的必然结果，一步预测 R2 仍 >= 0.98，自由仿真 R2 为 0.85-0.89。它没有伤害闭环性能（Panda 全族 R = 0.796），但**禁止把 theta-hat 当物理参数报告**。xArm6 因自身 tau-hat 仅 0.019 s，分解干净得多。
3. **平动自由仿真 R2 只有 0.81-0.89（Panda）。** 剩余误差来自仿真器自身命令到达成映射的非线性（IK 与 PD 在 50 ms 内的收敛行为），不是注入 plant 的成分。xArm6 上是 0.92-1.00。这是 C3（学习型执行器）的直接动机，也是 ours-P 与 oracle 在 Panda 上仍差 4.8 pp 成功率的主要来源。
4. **恢复率的样本单元少。** 满足"naive 下降 > 10 pp"的 plant 只有 10 个（Panda）与 8 个（xArm6）。宏 R 的点估计可靠，但 plant 级配对检验的功效有限；接上真实策略后崩塌面变宽，样本单元会自然增加。
5. **Slow-down 的计分口径。** Slow-down 按自己缩放后的参考计分，时长 12 s / 16 s，其它臂 8 s。跟踪 RMSE 上它与 ours-P 的配对差在 Panda 上是"胜 13 负 12"的均势（均值仍利于 ours-P，CI95 [0.004, 0.036] m），成功率上则被 ours-P 稳定压过，且时间到成功多花约 78%。论文里必须把这两列并排给出，不能只报 RMSE。
6. **D4 的辨识依赖 burst。** 若最终探针出于安全考虑删掉饱和 burst，限幅将完全不可辨识，`D4_c05` 那一类 plant 上标定会重新变成负收益。burst 已改为纯水平方向（+-x、+-y），不朝桌面，实测 26 个 plant 的 TCP 最低 z 为 0.163 m。
7. **未覆盖。** 关节速度限缩放（D4 的第二个来源）未实现；真机、非线性 plant、学习型执行器（C3）、策略侧对照臂（prompt-fit / codec-fit / FT-demo）均不在本轮范围。

## 12. Kill 判定

| 门 | 判据 | 结果 |
|---|---|---|
| 前置门 | 标称 plant 上 naive 回放 SR >= 0.9 | **通过**：1.00 / 1.00 |
| G1 | 强档 plant 上 naive 相对标称下降 >= 25 pp | **部分通过**：26 个 plant 中 7 个满足；但指定的 d = 150 ms、tau = 250 ms 两档不满足（见 9.2） |
| **K1** | lag plant 上标定后跟踪 RMSE 相对 naive 降 >= 30% | **通过**：76.3%（Panda）/ 93.2%（xArm6）；oracle 89.5% / 91.4% |
| **K2** | 回放恢复率 R >= 0.5 | **通过**：0.796（n = 10）/ 0.972（n = 8）；相对更强 naive 变体为 0.610 / 0.967 |
| 门 3 | ours-P vs Slow-down >= 5 pp，或 SR 持平时时间到成功 -25% | **通过**：+8.5 pp / +15.8 pp，且时间到成功 -44% |
| E2 | ours-P 与 oracle 的 SR 差 <= 5 pp | **通过**：4.8 pp（Panda，配对 t = 2.86）/ 0.8 pp（xArm6） |

结论：CCE 在接口层 plant 族上的机制成立，且不依赖 VLA 就能给出。下一步风险不在"反演能不能补"，而在两处：一是装上真实冻结策略后，意图 chunk 由策略而非专家给出，naive 的失败模式会更杂（策略自身误差与 plant 跟踪差距叠加）；二是 Panda 上 ours-P 与 oracle 仍差 4.8 pp，说明单极点复合模型在该本体上不够，这是 C3 的入口。

---

附：下列表格由 `cce/make_tables.py` 从 `cce/results/*.json` 生成。

"""


def main() -> int:
    tables = subprocess.run(
        [str(ROOT / "cce" / "env.sh"), str(ROOT / "cce" / "make_tables.py")],
        capture_output=True, text=True, cwd=str(ROOT))
    if tables.returncode != 0:
        sys.stderr.write(tables.stderr[-2000:])
        return tables.returncode
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PROSE + tables.stdout, encoding="utf-8")
    print(f"{OUT}  {OUT.stat().st_size / 1024:.1f} KiB  {len(OUT.read_text(encoding='utf-8').splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
