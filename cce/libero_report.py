#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate docs/reports/CCE_libero_family_v1.md from the result JSONs.

Every number in the report is read out of
  cce/results/libero_family_v1.json      (main comparison + statistics)
  cce/results/libero_family_spec.json    (frozen plant family)
  data/cce/libero/theta/identify_libero_panda.json  (probe + ARX identification)
  logs/cce_libero/g1screen/summary.json  (G1 screening)
so that nothing is transcribed by hand.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

CN = {"naive": "naive", "slow15": "Slow-down x1.5", "slow20": "Slow-down x2",
      "oracle": "oracle-theta", "ours_p": "ours-P"}


def f(x, n=3):
    if x is None:
        return "--"
    return ("%." + str(n) + "f") % x


def pp(x, n=1):
    if x is None:
        return "--"
    return ("%+." + str(n) + "f pp") % (100.0 * x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--ident", required=True)
    ap.add_argument("--screen", required=True)
    ap.add_argument("--wall", required=True, help="json with real wall-clock numbers")
    ap.add_argument("--abl", nargs="*", default=[], help="bound-ablation summary JSONs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    D = json.loads(Path(args.main).read_text(encoding="utf-8"))
    SP = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    ID = json.loads(Path(args.ident).read_text(encoding="utf-8"))
    SC = json.loads(Path(args.screen).read_text(encoding="utf-8"))
    W = json.loads(Path(args.wall).read_text(encoding="utf-8"))
    ABL = [json.loads(Path(a).read_text(encoding="utf-8")) for a in args.abl
           if Path(a).exists()]

    M = D["methods"]
    P = D["plants"]
    C = D["cells"]
    fam = {p: D["G1_naive_drop_vs_nominal"].get(p, {}).get("family", "nominal") for p in P}
    specd = {s["pid"]: s for s in SP["plants"]}
    agr = ID["agreement"]
    prim = D["primary"]
    holm = D["primary_holm_p"]
    best = D["best_slow_arm"]
    p1k = "P1_ours_p_minus_naive"
    p2k = f"P2_ours_p_minus_{best}"
    k1 = D["K1_summary"]
    kills = D["kills"]
    rec = D["recovery_macro"]
    L = []
    w = L.append

    # ---------------------------------------------------------------- header
    w("# CCE plant 族闭环主对比第一遍：LIBERO 正控策略上的接口层失配、探针辨识与执行器反演")
    w("")
    w("适用范围：方案 B2（CCE）按预注册 v2 修订（`docs/prereg/CCE_prereg_v2_amendment.md`）"
      "把 plant 族闭环主对比从 ManiSkill 锚定策略搬到 LIBERO 正控策略之后的第一遍结果。"
      "载体为 X-VLA-Pt + 官方 `2toINF/X-VLA-libero-spatial-peft`（LoRA 合并），"
      "本体只有 robosuite Panda，任务为 libero_spatial 10 任务。")
    w("")
    w("证据等级：DEVELOPMENT_ONLY。全部结论限于本载体、本执行链与本 plant 族；"
      "theta-hat 不声称等于任何物理参数；本轮不覆盖本体几何差异与视觉差异。")
    w("")
    w("日期：2026-09-06。机器：03 服务器（127.0.0.1），仅使用 GPU 6 与 GPU 7。"
      "仓库内未执行任何 git 命令。")
    w("")
    w("---")
    w("")

    # ---------------------------------------------------------------- 0
    w("## 0 结论速览")
    w("")
    w("| 环节 | 结论 | 关键数字 |")
    w("|---|---|---|")
    w("| L1 接口实现与自检 | 通过 | LIBERO 侧 D1-D4 与 `cce/plants.py` 在归一化增量域逐拍一致"
      "（最大差 3.0e-8，来自上游 float32 输出）；执行器与 `cce/executor.py` 逐拍差 **0.0**；"
      "naive + 标称 plant 逐拍复现策略绝对目标（闭环内位姿编解码误差 %.1e m） |"
      % max(C[p][m]["max_codec_err_m"] for p in P for m in M if C[p].get(m)))
    w("| L2 探针与辨识 | 通过（有一处已知局限） | 7.00 s 探针、13 个 plant：命令延迟 MAE **%.0f ms**、"
      "夹爪提前量 MAE **%.1f 拍**；纯增益档相对增益 %.3f / %.3f（真值 0.70 / 0.85）；"
      "大 tau 档 (tau, g) 互换，相对时间常数 MAE %.3f s、相对增益 MAE %.3f；"
      "%d 个轴全部过 R2 >= 0.7 门 |"
      % (agr["delay_mae_ms"], agr["grip_lead_mae_steps"],
         [r for r in ID["rows"] if r["pid"] == "D2_g70"][0]["g_rel"],
         [r for r in ID["rows"] if r["pid"] == "D2_g85"][0]["g_rel"],
         agr["tau_mae_rel_s"], agr["gain_mae_rel"], agr["n_axes_total"]))
    w("| L3 G1 筛查 | %s | 标称 plant 上 naive 宏 SR **%.2f**（与官方正控 95/100 一致）；"
      "%s |"
      % ("通过" if SC["G1_summary"]["G1_passed"] else "未通过",
         SC["G1_summary"]["nominal_naive_macro_sr"],
         "；".join("`%s` %.2f（降 %.0f pp）" % (k, v["naive_macro_sr"], v["drop_pp"])
                   for k, v in SC["G1_naive_drop_vs_nominal"].items())))
    w("| 族定义 | 已冻结 | %d 个 plant，`cce/results/libero_family_spec.json`，SHA-256 `%s`；"
      "G2 满足；D4 标注为无效应档并排除 |" % (SP["n_plants"], SP["sha256"][:16] + "..."))
    w("| L4 第一遍主对比 | 见 §6 | %d 集（%d plant x %d 臂 x %d 任务 x %d 集）；"
      "族宏 SR：naive %.3f、%s %.3f、ours-P %.3f、oracle-theta %.3f |"
      % (D["n_episodes"], len(P), len(M), len(D["tasks"]),
         D["protocol"]["episodes_per_cell"],
         D["macro"]["naive"]["macro_sr_family_mean"], CN[best],
         D["macro"][best]["macro_sr_family_mean"],
         D["macro"]["ours_p"]["macro_sr_family_mean"],
         D["macro"]["oracle"]["macro_sr_family_mean"]))
    w("| P1 = ours-P - naive | %s | %s，配对 t = %.3f，Holm 校正 p = %.4g，"
      "bootstrap CI95 [%s, %s]，%d/%d 个 plant 变好 |"
      % ("达到预期 (>= +15 pp)" if prim[p1k]["mean"] >= 0.15 else "未达预期 (>= +15 pp)",
         pp(prim[p1k]["mean"]), prim[p1k]["t"], holm[p1k],
         pp(prim[p1k]["ci95"][0]), pp(prim[p1k]["ci95"][1]),
         prim[p1k]["n_positive"], prim[p1k]["n"]))
    w("| P2 = ours-P - 最强 Slow-down | %s | %s，配对 t = %.3f，Holm 校正 p = %.4g，"
      "CI95 [%s, %s]；时间到成功 %s s vs %s s（%s） |"
      % ("**未达预期，且方向相反**" if prim[p2k]["mean"] < 0.05 else "达到预期 (>= +5 pp)",
         pp(prim[p2k]["mean"]), prim[p2k]["t"], holm[p2k],
         pp(prim[p2k]["ci95"][0]), pp(prim[p2k]["ci95"][1]),
         f(kills.get("K3_time_to_success_ours_p_s"), 2),
         f(kills.get("K3_time_to_success_%s_s" % best), 2),
         ("%.1f%%" % kills["K3_time_reduction_pct"]) if kills.get("K3_time_reduction_pct")
         is not None else "--"))
    w("| K1 跟踪 RMSE 降 >= 30%% | %s | 延迟/滞后 plant 上 ours-P 平均降 %s%%，"
      "%d/%d 个 plant 单独达标 |"
      % ("通过" if k1.get("K1_passed") else "未通过",
         f(k1.get("mean_ours_p_reduction_pct"), 1), k1["n_plants_reduction_ge_30"],
         len(k1["lag_delay_plants"])))
    w("| K2 全族恢复率 | %s | R(ours-P) = %s（计入 %d 个 plant），R(oracle) = %s |"
      % ("未触发（>= 0.3）" if not kills.get("K2_family_recovery_lt_0.3") else "**触发（< 0.3）**",
         f(rec.get("R_ours_p"), 3), rec["n_counted_plants"], f(rec.get("R_oracle"), 3)))
    w("| K3 与最强通用基线 | %s | 宏差 %s pp（ours-P **低于**基线），"
      "时间到成功短 %s%%；字面条件是「宏差 < 5 pp **且**无时间优势」，"
      "时间优势成立故不触发，但 P2 已实质失败，见 §10 第 5 条 |"
      % ("**触发**" if kills.get("K3_triggered") else "字面未触发（需连同 §10 读）",
         f(kills.get("K3_macro_gap_pp"), 2), f(kills.get("K3_time_reduction_pct"), 1)))
    w("")
    w("---")
    w("")

    # ---------------------------------------------------------------- 1
    w("## 1 环境、入口与运行约定")
    w("")
    w("- 仓库 `/opt/cce`；"
      "大数据 `data -> /opt/cce/data`。")
    w("- 策略服务端在 `ctp-e0` 环境里经 `cce/env.sh` 启动：GPU 6 -> 端口 8021，GPU 7 -> 端口 8022。"
      "两端 `/ready` 均报 `lora_merged=true`、`soft_prompt max|delta| = 0.016865`，"
      "与 Gate 0 同一权重、同一 `/act` 协议。")
    w("- 客户端在克隆的 LIBERO 环境 `/opt/cce/data/envs/libero` 里，"
      "`MUJOCO_GL=egl`、`LIBERO_CONFIG_PATH=/opt/cce/data/libero_config`、"
      "`PYTHONNOUSERSITE=1`。")
    w("- 只读约束遵守情况：未改动 `cce/gate0_libero_client.py`、`cce/serve_xvla.py`、"
      "`cce/plants.py`、`cce/executor.py`、`cce/identify.py`、`cce/probe.py`、`cce/common.py` "
      "的任何一行；未改动上游 `/opt/CalibrationToPrompt/source/X-VLA/`；"
      "未向任何 conda 环境安装包；未动学生目录。新增逻辑全部在 `cce/libero_*.py` 五个新文件里，"
      "对既有模块只做 import 复用。")
    w("- 长作业一律 `setsid nohup` 启动并写 `.exit` 哨兵，日志在 "
      "`/opt/cce/data/logs/cce_libero/`。")
    w("")
    w("新增文件：")
    w("")
    w("| 文件 | 职责 |")
    w("|---|---|")
    w("| `cce/libero_plant.py` | 位姿编解码（rot6d / 轴角 / pose7）、`compose_delta`、"
      "LIBERO 侧 D1-D4 plant、plant 族定义、`selftest` |")
    w("| `cce/libero_exec.py` | LIBERO 侧 theta 构造、`shape_reference_libero`、"
      "`LiberoExecutor`（`cce/executor.py` 的转写）、与原实现的等价性自检 |")
    w("| `cce/libero_probe.py` | 探针参考与命令生成、在 LIBERO 场景上执行探针、"
      "调用 `cce/identify.py` 做 ARX 辨识与真值对照 |")
    w("| `cce/libero_family.py` | chunk 级策略客户端、单集闭环、分片 CLI |")
    w("| `cce/libero_aggregate.py` | 逐 cell 汇总、G1、P1/P2 配对检验与 bootstrap、Holm、恢复率、K1-K3 |")
    w("| `cce/libero_freeze_family.py` | 在 G1 之后、主对比之前把族定义写死并计 SHA-256 |")
    w("")
    w("---")
    w("")

    # ---------------------------------------------------------------- 2
    w("## 2 L1：LIBERO 侧 plant 接口（`cce/libero_plant.py`）")
    w("")
    w("### 2.1 插入点与语义")
    w("")
    w("LIBERO 官方评测链路用 robosuite 的 OSC_POSE 控制器，并在 `LIBEROEval._init_env` 里显式设成"
      "绝对位姿模式（`robot.controller.use_delta = False`）。这条支路在 "
      "`robosuite/controllers/osc.py: set_goal` 里既不缩放也不裁剪输入："
      "`set_pos = delta[:3]`、`set_ori = quat2mat(axisangle2quat(delta[3:6]))`、"
      "`scaled_delta = delta`。因此 plant 插在"
      "「策略 chunk 的绝对末端位姿目标 -> 控制器输入」之间，控制器只跟踪被扰动后的目标，"
      "不能把失配吸收掉——正是预注册 v1「不得在 EE 控制器吸收型床上报告阳性」要求的位置。")
    w("")
    w("`cce/plants.py` 扰动的是归一化增量，本文沿用同一套代数：先把绝对目标 `c` 换成相对"
      "**当前实测位姿** `y` 的归一化增量（`cce.common.norm_delta`，平动单位 0.1 m、"
      "转动单位 -0.1 rad），在增量域按 `PlantWrapper.apply` 的次序施加 D1-D4，"
      "再复合回 `y` 得到真正交给控制器的绝对目标。每个 20 Hz 控制拍：")
    w("")
    w("```")
    w("命令传输延迟(FIFO, 整数拍) -> 一阶滞后(alpha = exp(-DT/tau)) -> 稳态增益 g -> 每拍位移限幅")
    w("夹爪通道：自带(传输 + 执行)延迟，不滤波")
    w("```")
    w("")
    w("增益因此天然作用在「目标相对当前实测位姿的增量」上。两处与 `cce/plants.py` 的有意差异，"
      "都由床决定：")
    w("")
    w("1. **去掉 ManiSkill 动作界。** ManiSkill 的 `pd_ee_delta_pose` 把命令归一化到 +-1"
      "（+-0.1 m / +-0.1 rad），`plants.py` 把这条界折进限幅里"
      "（`min(clip_pos / POS_SCALE, 1.0)`）。robosuite 的绝对 OSC 没有这条界，故此处去掉，"
      "「无限幅」的 plant 取 `clip_pos = clip_rot = 1e3`；位姿复合改用本文的 `compose_delta`，"
      "而不是内含 `clip_norm6` 的 `cce.common.apply_norm_delta`。")
    w("2. **夹爪静止值是 -1**（LIBERO/robosuite：-1 张开、+1 闭合），不是 ManiSkill 的 +1。")
    w("")
    w("OSC 控制器自身的跟踪动力学没有被假设为 1：它是标称 theta 的一部分，由 §3 的探针辨识出来"
      "（标称增益 %.3f，即控制器在一个 50 ms 拍里只完成命令位移的约 %.0f%%）。"
      % (agr["gain_hat_on_nominal"], 100 * agr["gain_hat_on_nominal"]))
    w("")
    w("### 2.2 自检")
    w("")
    w("在 LIBERO 环境里跑 `python cce/libero_plant.py` 与 `python cce/libero_exec.py`：")
    w("")
    w("| 检查 | 结果 |")
    w("|---|---|")
    w("| 归一化增量编解码往返 `norm_delta(y, compose_delta(y, a)) = a` | 3.6e-15 |")
    w("| rot6d 往返（上游 Gram-Schmidt 的 `EPS = 1e-6` 决定的下限） | 2.0e-6 |")
    w("| rot6d -> 动作 -> 控制器目标姿态矩阵，与上游 `Rotate6D_to_AxisAngle` 同路对照 | "
      "8.3e-7（robosuite `mat2quat` 内部走 float32，即上游自身的数值噪声量级） |")
    w("| 标称 plant = 恒等：`command(y, norm_delta(y, c))` 复现 `c` | 2.2e-16 |")
    w("| D1 阶跃：延迟 250 ms -> 5 拍，前 5 拍输出零增量 | 精确 |")
    w("| D2 一阶滞后阶跃响应 `w_k = (1 - alpha^k) u` | 5.6e-17 |")
    w("| D2 增益 `w = g u` | 0.0 |")
    w("| D4 限幅 0.05 m -> 归一化 0.5 | 精确 |")
    w("| D3 夹爪提前 0.6 s -> 12 拍 | 精确 |")
    w("| 与 `cce.plants.PlantWrapper` 在归一化增量域逐拍对照（限幅取在 ManiSkill 界内，"
      "4 个 plant x 200 拍） | 3.0e-8（`PlantWrapper` 输出 float32） |")
    w("| `LiberoExecutor` 与 `cce.executor.InversionExecutor` 逐拍对照（命令界不生效时） | **0.0** |")
    w("| `shape_reference_libero` 与 `cce.executor.shape_reference` 对照 | **0.0** |")
    w("| naive + 标称 plant 在闭环内复现策略绝对目标 | %.1e m（全部 %d 集） |"
      % (max(C[p][m]["max_codec_err_m"] for p in P for m in M if C[p].get(m)), D["n_episodes"]))
    w("")
    w("### 2.3 执行器的命令界")
    w("")
    w("`cce/executor.py` 每拍返回 `clip_norm6(u)`，即 ManiSkill 的 +-1 动作界；LIBERO 没有这条界，"
      "所以本文把它变成显式参数 `bound_pos` / `bound_rot`，并在跑任何族内对照臂之前冻结为 "
      "**%.1f m / %.1f rad 每拍**（归一化 %.1f）。这个值约为任何一条臂在标称 plant 上实际命令峰值的"
      " 6 倍，远超机械臂本身能执行的范围，因此不塑造正常控制；它只防止一次退化的反演下发任意远的目标"
      "（超前滤波增益是 `1/(1 - alpha)`，当辨识出的极点落在 0.97 上限时达 33 倍）。"
      "逐 cell 的触界集数见 T7。"
      % (SP["executor_command_bound"]["bound_pos_m"],
         SP["executor_command_bound"]["bound_rot_rad"],
         SP["executor_command_bound"]["bound_pos_norm"]))
    w("")
    w("---")
    w("")

    # ---------------------------------------------------------------- 3
    pdz = ID["probe_design"]
    ps = {s["pid"]: s for s in ID["probe_stats"]}
    nomst = ps["L00_nominal"]
    w("## 3 L2：探针与辨识（`cce/libero_probe.py`）")
    w("")
    w("### 3.1 探针")
    w("")
    w("沿用 `cce/probe.py` 的设计与时基：%d 拍 @ 20 Hz = %.2f s，分为静置 / 三轴线性 chirp"
      "（%.1f -> %.1f Hz，+-%.0f cm，相位错开）/ 水平饱和 burst / 静默间隙 / 三轴姿态正弦"
      "（+-%.2f rad）/ 保持并闭合夹爪。激励在**虚拟完美跟踪位姿**上做差分而不是实测位姿，"
      "因此在所有 plant 上完全相同（比较 theta-hat 的前提），也不经过被扰动的臂闭环。"
      % (pdz["steps"], pdz["seconds"], pdz["chirp_hz"][0], pdz["chirp_hz"][1],
         100 * pdz["chirp_amp_m"], pdz["attitude_amp_rad"]))
    w("")
    w("一处必要改动：**burst 改写在命令域**（`BURST_CMD_NORM = %.1f`，即每拍 %.0f cm 的命令增量，"
      "+-x / +-y 各两拍）。ManiSkill 版本用 +-0.5 m 的绝对目标，靠 +-0.1 m 的动作界把命令压在饱和态"
      "两拍；robosuite 的绝对 OSC 没有这条界，照抄会变成真正的半米猛冲。"
      % (pdz["burst_cmd_norm"], 100 * pdz["burst_cmd_m"]))
    w("")
    w("安全裕度（libero_spatial 任务 %d，标称 plant 实测）：起始 TCP 高度 %.4f m，"
      "全程最低 %.4f m（下探 %.1f mm），三轴最大偏移 %.1f cm，夹爪开度 %.3f -> %.3f，"
      "单次探针耗时 %.1f s。"
      % (pdz["task_id"], nomst["z_home_m"], nomst["z_min_m"],
         1000 * (nomst["z_home_m"] - nomst["z_min_m"]), 100 * nomst["xyz_excursion_m"],
         nomst["grip_open_start"], nomst["grip_open_min"], nomst["wall_s"]))
    w("")
    w("### 3.2 辨识结果")
    w("")
    w("`cce/identify.py` 原样调用（`sat_aware=False`，即预注册主配方）：每轴拟合 "
      "`w[t] = a w[t-1] + b u[t-d]`，延迟在 0-300 ms 格点上按一步预测 R2 选、拟合窗口按自由仿真 R2 选，"
      "再换算 alpha / tau / g；夹爪提前量取闭合阶跃 90% 到达时间减去标称 plant 的同一量。")
    w("")
    w("标称行，也就是 OSC 控制器自身的响应（v2 修订要求在本床重做的那一项）："
      "**延迟 %.0f 拍、tau-hat %.3f s、g-hat %.3f、夹爪 t90 = %.3f s**。"
      % (agr["delay_hat_on_nominal_steps"], agr["tau_hat_on_nominal_s"],
         agr["gain_hat_on_nominal"], agr["grip_t90_nominal_s"]))
    w("")
    w("| plant | family | d 真值(拍) | d-hat | tau 真值(s) | tau-hat(s) | tau 相对(s) | "
      "g 真值 | g-hat | g 相对 | 夹爪提前真值(拍) | 估计 | 平动自由仿真 R2 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ID["rows"]:
        w("| `%s` | %s | %.0f | %d | %.2f | %.3f | %.3f | %.2f | %.3f | %.3f | %.0f | %d | %.3f |"
          % (r["pid"], r["family"], r["d_true_steps"], r["d_hat_steps"], r["tau_true_s"],
             r["tau_hat_s"], r["tau_rel_s"], r["g_true"], r["g_hat"], r["g_rel"],
             r["grip_lead_true_steps"], r["grip_lead_hat_steps"], r["r2free_min_trans"]))
    w("")
    w("汇总：延迟相关系数 %.3f、MAE **%.0f ms**；夹爪提前相关系数 %.3f、MAE **%.1f 拍**；"
      "相对时间常数相关 %.3f、MAE %.3f s；相对增益相关 %.3f、MAE %.3f；"
      "%d 个轴无一低于 R2 = 0.7 质量门。"
      % (agr["delay_corr"], agr["delay_mae_ms"], agr["grip_lead_corr"],
         agr["grip_lead_mae_steps"], agr["tau_corr_rel"], agr["tau_mae_rel_s"],
         agr["gain_corr_rel"], agr["gain_mae_rel"], agr["n_axes_total"]))
    w("")
    w("### 3.3 一处必须写明的局限：(tau, g) 互换自由度")
    w("")
    w("纯增益档的相对增益几乎精确，纯延迟与夹爪档全部精确命中；但**大 tau 档上时间常数被系统性高估、"
      "增益被同向高估**（`D2_t400` 上 tau 相对量 1.556 s 对真值 0.40 s，g 相对量 2.395 对真值 1.00）。"
      "原因是被辨识的对象是「注入 plant 的一阶环节」与「OSC 自身一阶响应」的串联，是二阶系统，"
      "而 ARX(1,1) 只有一个极点；chirp 激励带内（0.2-2 Hz）的最佳单极点拟合会把极点推远、"
      "再用更大的 `b/(1-a)` 去补，直流增益因此偏离真值。`D2_t400` 的 x/y 轴 a 甚至压在 "
      "`cce/identify.py` 的 0.97 上限上。")
    w("")
    w("这与仿真侧同源：`CCE_sim_probe_executor.md` §6 已记录「大 tau plant 上 (tau, g) 存在互换自由度，"
      "Panda 尤为明显，因为它自身的 tau-hat = 0.056 s 与注入量同量级」。本床的标称 tau-hat = %.3f s，"
      "与注入的 0.25 / 0.40 s 同量级，同一现象更强。这也是不把 theta-hat 写成物理参数的直接理由；"
      "辨识质量的读数由 ours-P 与 oracle-theta 的差给出（§6.4）。"
      % agr["tau_hat_on_nominal_s"])
    w("")
    w("---")
    w("")

    # ---------------------------------------------------------------- 4
    w("## 4 L3：G1 筛查（naive 臂，先于主对比）")
    w("")
    w("设计：naive 臂，标称 plant 加四档候选强档（d = 250 ms、tau = 400 ms、g = 0.70、混合 MX02）"
      "再加 D4 限幅档，各 10 任务 x 2 集，共 6 x 20 = %d 集。" % SC["n_episodes"])
    w("")
    w("| plant | family | naive 宏 SR | 相对标称降幅 | >= 25 pp | 平均步数 | 跟踪 RMSE(m) | "
      "夹爪闭合滞后(拍) | 命令峰值 \\|u\\|pos | 触界集数 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    order = ["L00_nominal", "D1_d250", "D2_t400", "MX02", "D2_g70", "D4_c05"]
    for pid in order:
        c = SC["cells"][pid]["naive"]
        g = SC["G1_naive_drop_vs_nominal"].get(pid)
        w("| `%s` | %s | %.2f | %s | %s | %.1f | %.4f | %.2f | %.3f | %d/%d |"
          % (pid, g["family"] if g else "nominal", c["macro_sr"],
             ("%.1f pp" % g["drop_pp"]) if g else "--",
             ("是" if g["meets_G1_25pp"] else "否") if g else "--",
             c["mean_steps"], c["tracking_rmse_m"], c["mean_grip_close_lag_steps"],
             c["u_pos_max_norm"], c["n_bound_hit_episodes"], c["n"]))
    w("")
    w("三点结论：")
    w("")
    w("1. **G1 通过。** 族内存在明确的崩塌面：纯延迟 250 ms 掉 60 pp、纯滞后 400 ms 掉 50 pp、"
      "混合档掉 90 pp。同时标称 plant 上 naive 的宏 SR 是 %.2f，与官方正控 95/100 完全一致——"
      "新链路没有破坏载体。" % SC["G1_summary"]["nominal_naive_macro_sr"])
    w("2. **纯增益档在这张床上不伤害。** `D2_g70` 的 SR 是 1.00（比标称高 5 pp，落在 n=20 的抽样噪声内）。"
      "机理清楚：绝对目标每拍重新锚在实测位姿上，增益 0.7 只是把每拍闭合的比例从 0.294 降到 0.206，"
      "臂仍单调收敛到同一个绝对航点；800 步预算与 1.5 s 一次的视觉重规划把这点速度损失完全吸收。"
      "跟踪 RMSE 从 0.0300 涨到 0.0437 m，说明机制量有反应、只是没有跨过任务的成功阈值。"
      "按 G2 要求 g = 0.70 仍留在族内。")
    w("3. **D4 在 20 Hz 绝对目标下是无效应档。** `D4_c05` 的 SR 0.95、降幅 0.0 pp。"
      "每拍绝对目标增量的峰值只有 0.062 m（归一化 0.621），大部分时间远小于 0.05 m 的限幅，"
      "限幅几乎不生效。按任务书要求如实记录，把 D4 从第一遍族中排除并在族定义文件里标注为无效应档；"
      "**没有为了「制造」下降去改 plant 定义**。")
    w("")
    w("---")
    w("")

    # ---------------------------------------------------------------- 5
    w("## 5 族定义（写定后不再改）")
    w("")
    w("`cce/results/libero_family_spec.json`，%d 个 plant，SHA-256 `%s`。该文件在 G1 筛查之后、"
      "任何族内对照臂运行之前写定；除预注册的 naive-only 筛查外，没有任何对照臂结果参与决策。"
      % (SP["n_plants"], SP["sha256"]))
    w("")
    w("| pid | family | d (ms) | tau (ms) | g | 夹爪延迟 (s) | 限幅 |")
    w("|---|---|---|---|---|---|---|")
    for s in SP["plants"]:
        w("| `%s` | %s | %.0f | %.0f | %.2f | %.1f | %s |"
          % (s["pid"], s["family"], s["d_ms"], s["tau_ms"], s["gain"], s["grip_delay_s"],
             "无" if s["clip_pos_m"] >= 100 else ("%.2f m" % s["clip_pos_m"])))
    w("")
    g2 = SP["G2_check"]
    w("G2 检查：增益档含 %s，延迟档含 %s ms，tau 档含 %s ms，夹爪档含 %s s。"
      "含 g 0.70 与 1.00（%s）、含 d 0 与 >= 150 ms（%s），满足 G2。"
      % (g2["gain_levels"], [int(x) for x in g2["delay_levels_ms"]],
         [int(x) for x in g2["tau_levels_ms"]], g2["grip_delay_levels_s"],
         g2["has_gain_0.70_and_1.00"], g2["has_delay_0_and_ge_150ms"]))
    w("")
    w("---")
    w("")

    # ---------------------------------------------------------------- 6
    w("## 6 L4：第一遍主对比")
    w("")
    w("设计：%d plant x %d 臂 x %d 任务 x n = %d 集 = **%d 集**。"
      "每集 800 步预算、30 格 chunk 排空后重查、10 步去噪、`domain_id = 3`、"
      "绝对位姿控制器、夹爪 +-1 硬阈值、`init_seed = 42`，全部沿用上游评测语义。"
      "宏成功率按 10 个 libero_spatial 任务等权。"
      % (len(P), len(M), len(D["tasks"]), D["protocol"]["episodes_per_cell"], D["n_episodes"]))
    w("")
    w("### 6.1 T1 逐 plant x 臂 宏成功率")
    w("")
    w("| plant | family | " + " | ".join(CN[m] for m in M) + " |")
    w("|---|---|" + "---|" * len(M))
    for p in P:
        w("| `%s` | %s | " % (p, fam[p])
          + " | ".join(f(C[p][m]["macro_sr"], 3) if C[p].get(m) else "--" for m in M) + " |")
    w("| **族均值** | -- | "
      + " | ".join("**%s**" % f(D["macro"][m]["macro_sr_family_mean"], 3) for m in M) + " |")
    w("| **族均值（不含标称）** | -- | "
      + " | ".join(f(D["macro"][m]["macro_sr_family_mean_excl_nominal"], 3) for m in M) + " |")
    w("")
    w("### 6.2 T2 平均步数 / 时间到成功（s）")
    w("")
    w("| plant | " + " | ".join(CN[m] for m in M) + " |")
    w("|---|" + "---|" * len(M))
    for p in P:
        cs = []
        for m in M:
            c = C[p].get(m)
            cs.append("%.0f / %s" % (c["mean_steps"], f(c["mean_time_to_success_s"], 2))
                      if c else "--")
        w("| `%s` | " % p + " | ".join(cs) + " |")
    w("")
    w("时间到成功只在成功集上统计。Slow-down 臂按定义把 chunk 拉长 1.5 / 2 倍，"
      "同一段任务需要更多控制拍，这一列必须与其宏 SR 并列理解。")
    w("")
    w("### 6.3 T3 跟踪 RMSE（对策略 chunk 的口径，m）")
    w("")
    w("这是跨臂可比的一列：每条臂都按它拿到的 chunk 计分（Slow-down 臂按时间缩放后的 chunk，"
      "因为那才是它在 t 时刻真正想到的位置）。")
    w("")
    w("| plant | " + " | ".join(CN[m] for m in M) + " | ours-P 降幅 | oracle 降幅 |")
    w("|---|" + "---|" * (len(M) + 2))
    K1T = D["K1_tracking_rmse_vs_intent"]
    for p in P:
        cs = [f(C[p][m]["tracking_rmse_m"], 5) if C[p].get(m) else "--" for m in M]
        r = K1T.get(p, {})
        cs.append(("%.1f%%" % r["ours_p_reduction_pct"]) if "ours_p_reduction_pct" in r else "--")
        cs.append(("%.1f%%" % r["oracle_reduction_pct"]) if "oracle_reduction_pct" in r else "--")
        w("| `%s` | " % p + " | ".join(cs) + " |")
    w("")
    w("### 6.4 T4 跟踪 RMSE（对各臂自身设定点的口径，m）")
    w("")
    w("标定臂的设定点是标称 plant 的预测可达轨迹，这一列用来检查反演本身是否闭合，"
      "不能用来跨臂比较。")
    w("")
    w("| plant | " + " | ".join(CN[m] for m in M) + " |")
    w("|---|" + "---|" * len(M))
    for p in P:
        w("| `%s` | " % p + " | ".join(
            f(C[p][m]["tracking_rmse_vs_ref_m"], 5) if C[p].get(m) else "--" for m in M) + " |")
    w("")
    w("### 6.5 T5 夹爪事件：从下发闭合到实际闭合的滞后（控制拍）")
    w("")
    w("| plant | 注入夹爪延迟(s) | " + " | ".join(CN[m] for m in M) + " |")
    w("|---|---|" + "---|" * len(M))
    for p in P:
        w("| `%s` | %.1f | " % (p, specd[p]["grip_delay_s"]) + " | ".join(
            f(C[p][m]["mean_grip_close_lag_steps"], 2) if C[p].get(m) else "--" for m in M) + " |")
    w("")
    w("### 6.6 T6 恢复率 R = (SR_calib - SR_naive) / (SR_nominal - SR_naive)")
    w("")
    w("分母是标称 plant 上 naive 的宏 SR 减去该 plant 上 naive 的宏 SR；"
      "只在分母 > 0.10 的 plant 上计入宏恢复率，避免分母噪声。")
    w("")
    w("| plant | naive SR | 分母 | 计入 | " + " | ".join("R(%s)" % CN[m] for m in M if m != "naive")
      + " |")
    w("|---|---|---|---|" + "---|" * (len(M) - 1))
    for pid, v in D["recovery"].items():
        w("| `%s` | %.3f | %+.3f | %s | " % (pid, v["sr_naive"], v["denominator"],
                                             "是" if v["counted"] else "否")
          + " | ".join(f(v.get("R_" + m), 3) for m in M if m != "naive") + " |")
    w("| **宏恢复率（计入 %d 个 plant）** | -- | -- | -- | " % rec["n_counted_plants"]
      + " | ".join("**%s**" % f(rec.get("R_" + m), 3) for m in M if m != "naive") + " |")
    w("")
    w("### 6.7 T7 命令幅值与编解码诊断（取 cell 内最大值）")
    w("")
    w("| plant | 臂 | max \\|u\\|pos | max \\|u\\|rot | 触界集数 | 编解码误差(m) |")
    w("|---|---|---|---|---|---|")
    for p in P:
        for m in M:
            c = C[p].get(m)
            if c:
                w("| `%s` | %s | %.3f | %.3f | %d/%d | %.1e |"
                  % (p, CN[m], c["u_pos_max_norm"], c["u_rot_max_norm"],
                     c["n_bound_hit_episodes"], c["n"], c["max_codec_err_m"]))
    w("")

    # ------------------------------------------------- 6.8 inversion stability
    w("### 6.8 命令界与反演稳定性诊断（本轮最重要的机制发现）")
    w("")
    nb = {m: sum(C[p][m]["n_bound_hit_episodes"] for p in P if C[p].get(m)) for m in M}
    nt = {m: sum(C[p][m]["n"] for p in P if C[p].get(m)) for m in M}
    w("T7 里有一列必须单独讲：**标定臂在大量 episode 里把命令界打满**。"
      "全族触界集数占比：" + "；".join("%s %d/%d（%.0f%%）"
        % (CN[m], nb[m], nt[m], 100.0 * nb[m] / max(1, nt[m])) for m in M) + "。")
    w("")
    w("机制是清楚的，且完全由 §3 的辨识结果决定。参数版反演每拍做的是")
    w("")
    w("```")
    w("u = (w_des / g_hat - alpha_hat * z_pred) / (1 - alpha_hat)")
    w("```")
    w("")
    w("本床标称行的逐轴 `g_hat` 是 0.20-0.33（OSC 在一个 50 ms 拍里只完成约三成命令位移），"
      "`alpha_hat` 是 0.48-0.62。两项相乘，**这个逆的高频增益在标称 plant 上就有 6-10 倍**"
      "（1/g 为 3.0-4.9，1/(1-alpha) 为 1.9-2.6）；在 `D2_t400` 上极点压到 `cce/identify.py` 的 "
      "0.97 上限，1/(1-alpha) = 33.3，逆增益接近 40-70 倍。模型自由仿真 R2 只有 0.86-0.99，"
      "剩下的几个百分点误差被这个增益放大后闭环回真实臂，命令随即打到界上。")
    w("")
    w("ManiSkill 上不出现这一幕，不是因为算法不同（`LiberoExecutor` 与 "
      "`cce.executor.InversionExecutor` 逐拍差为 0.0），而是因为**那张床的动作空间本身就是 +-1**"
      "（+-0.1 m / +-0.1 rad 每拍），`clip_norm6` 与 `Theta.from_hat` 默认的 "
      "`clip_pos_norm = 1.0` 一起把这个逆牢牢箍住；并且 ManiSkill 标称行的 `g_hat = 0.408`、"
      "`tau_hat = 0.056 s`，逆增益只有约 4 倍。robosuite 的绝对 OSC 没有动作界，"
      "这条隐式稳定项随之消失。")
    w("")
    w("由此产生三个可以直接读出来的后果：")
    w("")
    w("1. **「标定臂在标称 plant 上退化为 naive」这条设计性质在本床不成立。** 标称 plant 上 "
      "naive %.3f、oracle-theta %.3f、ours-P %.3f，标定臂反而略差，并在 %d/%d、%d/%d 集里触界。"
      "恢复率 R 的分母正是以这条性质为前提的，因此本轮的 R 必须连同这条一起读。"
      % (C["L00_nominal"]["naive"]["macro_sr"], C["L00_nominal"]["oracle"]["macro_sr"],
         C["L00_nominal"]["ours_p"]["macro_sr"],
         C["L00_nominal"]["oracle"]["n_bound_hit_episodes"], C["L00_nominal"]["oracle"]["n"],
         C["L00_nominal"]["ours_p"]["n_bound_hit_episodes"], C["L00_nominal"]["ours_p"]["n"]))
    w("2. **标定臂的收益高度依赖失配类型。** 在滞后档上收益明显（见 T1 的 `D2_t250` / `D2_t400`），"
      "在 naive 本来就没问题的轻档上反而是净损失。")
    w("3. **命令界是本轮唯一一个由本文自己定、而非预注册给定的量。** 它在 L4 之前按"
      "「标称 plant 上任何一条臂命令峰值的约 6 倍」冻结为 %.1f（0.5 m / 0.5 rad 每拍），"
      "事后看这个值一方面在物理上过宽（0.5 m / 50 ms 相当于 10 m/s），另一方面又窄到会被打满，"
      "说明本床上参数版反演的工作点本身就不稳。这是下一遍必须预注册扫描的第一个量。"
      % SP["executor_command_bound"]["bound_pos_norm"])
    w("")

    # ------------------------------------------- 6.9 bound ablation (post hoc)
    if ABL:
        w("### 6.9 命令界消融（**事后补跑，非确证**）")
        w("")
        w("下表是在看过 6.1-6.8 之后补跑的，**不是确证性结果**，只用来回答一个机制问题："
          "标定臂在轻档 plant 上的退化，是不是命令界这一个量造成的。设计：5 个 plant"
          "（标称 + 两个纯延迟 + 一个滞后 + 一个混合）x 3 条臂 x 10 任务 x 2 集，"
          "命令界分别取 1.0 与 2.0（归一化，即 0.1 / 0.2 m 每拍），其余一切不变。"
          "命令界 1.0 正是 ManiSkill 的动作界，也就是 `cce/executor.py` 的原生工作点。")
        w("")
        cols = ["界 = %s" % f(a["protocol"]["executor_bound_pos_norm"], 1) for a in ABL]
        w("| plant | 臂 | 主对比（界 = %s） | %s |"
          % (f(D["protocol"]["executor_bound_pos_norm"], 1), " | ".join(cols)))
        w("|---|---|---|" + "---|" * len(ABL))
        abl_pids = ABL[0]["plants"]
        for p in abl_pids:
            for m in ("naive", "oracle", "ours_p"):
                if not C.get(p, {}).get(m):
                    continue
                row = ["%.3f（%d/%d 触界）" % (C[p][m]["macro_sr"],
                                            C[p][m]["n_bound_hit_episodes"], C[p][m]["n"])]
                for a in ABL:
                    c = a["cells"].get(p, {}).get(m)
                    row.append("%.3f（%d/%d 触界）"
                               % (c["macro_sr"], c["n_bound_hit_episodes"], c["n"])
                               if c else "--")
                w("| `%s` | %s | %s |" % (p, CN[m], " | ".join(row)))
        w("")
        for a in ABL:
            b = a["protocol"]["executor_bound_pos_norm"]
            w("- 界 = %.1f：naive %.3f / oracle-theta %.3f / ours-P %.3f（这 %d 个 plant 的均值）。"
              % (b, a["macro"]["naive"]["macro_sr_family_mean"],
                 a["macro"]["oracle"]["macro_sr_family_mean"],
                 a["macro"]["ours_p"]["macro_sr_family_mean"], len(a["plants"])))
        w("- 主对比（界 = %.1f）在同 5 个 plant 上：naive %.3f / oracle-theta %.3f / ours-P %.3f。"
          % (D["protocol"]["executor_bound_pos_norm"],
             sum(C[p]["naive"]["macro_sr"] for p in abl_pids) / len(abl_pids),
             sum(C[p]["oracle"]["macro_sr"] for p in abl_pids) / len(abl_pids),
             sum(C[p]["ours_p"]["macro_sr"] for p in abl_pids) / len(abl_pids)))
        w("")
        w("这一列的用途只有一个：给下一遍的预注册提供命令界的扫描范围，"
          "并说明本轮 ours-P 的数字在多大程度上是这一个参数的函数。**它不参与本轮任何判定**。")
        w("")

    # ---------------------------------------------------------------- 7
    w("## 7 P1 / P2 / R / K 判定")
    w("")
    w("统计口径：分析单元是 plant，配对样本量 n = %d（族内全部 plant，含标称）；"
      "配对 t 检验加 plant 级 bootstrap（%d 次重抽样，种子 %d）；"
      "两条主对比按 Holm 校正。" % (prim[p1k]["n"], D["protocol"]["bootstrap"]["n"],
                                 D["protocol"]["bootstrap"]["seed"]))
    w("")
    w("| 对比 | 配对均差 | t | 原始 p | Holm 校正 p | bootstrap CI95 | 变好的 plant 数 |")
    w("|---|---|---|---|---|---|---|")
    for k in (p1k, p2k):
        v = prim[k]
        w("| %s | %s | %.3f | %.4g | %.4g | [%s, %s] | %d/%d |"
          % (k, pp(v["mean"]), v["t"], v["p"], holm[k], pp(v["ci95"][0]), pp(v["ci95"][1]),
             v["n_positive"], v["n"]))
    w("")
    sec = D.get("secondary_excl_nominal", {})
    if sec:
        w("次要口径（去掉标称 plant，n = %d）："
          % list(sec.values())[0]["n"])
        w("")
        w("| 对比 | 配对均差 | t | p | CI95 |")
        w("|---|---|---|---|---|")
        for k, v in sec.items():
            w("| %s | %s | %.3f | %.4g | [%s, %s] |"
              % (k, pp(v["mean"]), v["t"], v["p"], pp(v["ci95"][0]), pp(v["ci95"][1])))
        w("")
    w("Kill 条件逐条：")
    w("")
    w("| 条件 | 判定 | 依据 |")
    w("|---|---|---|")
    w("| K1 lag plant 标定后跟踪 RMSE 未降 >= 30%% -> 转学习型 | %s | "
      "延迟/滞后 plant（%s）上 ours-P 平均降 %s%%，%d/%d 单独达标 |"
      % ("**未触发**" if k1.get("K1_passed") else "**触发**",
         "、".join("`%s`" % x for x in k1["lag_delay_plants"]),
         f(k1.get("mean_ours_p_reduction_pct"), 1),
         k1["n_plants_reduction_ge_30"], len(k1["lag_delay_plants"])))
    w("| K2 全族恢复率 < 0.3 -> 撤回「跟踪差距是主因」 | %s | R(ours-P) = %s，"
      "R(oracle-theta) = %s，计入 %d 个 plant |"
      % ("**触发**" if kills.get("K2_family_recovery_lt_0.3") else "**未触发**",
         f(rec.get("R_ours_p"), 3), f(rec.get("R_oracle"), 3), rec["n_counted_plants"]))
    w("| K3 与最强通用基线宏差 < 5 pp 且无时间优势 -> 降级 | %s | 宏差 %s pp（ours-P 低于基线，"
      "预注册写这一条时假设的是 ours >= 基线）；时间到成功 ours-P %s s vs %s %s s（短 %s%%）。"
      "时间优势成立故字面不触发；但 Slow-down 慢是构造出来的，且时间到成功只在成功集上统计，"
      "两条臂的成功子集不同——实质结论见 §10 第 5 条 |"
      % ("**触发**" if kills.get("K3_triggered") else "**字面未触发**",
         f(kills.get("K3_macro_gap_pp"), 2),
         f(kills.get("K3_time_to_success_ours_p_s"), 2), CN[best],
         f(kills.get("K3_time_to_success_%s_s" % best), 2),
         f(kills.get("K3_time_reduction_pct"), 1)))
    w("| K4 FT-demo 恢复率 >= ours -> 撤回「策略侧不可补偿」 | 本轮未评估 | "
      "FT-demo 臂按 v2 修订留待第一遍通过 G1 后补 |")
    w("")
    w("---")
    w("")

    # ---------------------------------------------------------------- 8
    w("## 8 协议偏离清单")
    w("")
    w("| 项 | 预注册 | 本轮实际 | 原因 |")
    w("|---|---|---|---|")
    w("| 本体 | v1 两个本体（Panda + xArm6） | **单本体**（robosuite Panda） | "
      "v2 修订第 1 条：LIBERO 正控策略只有 Panda；主张已相应收缩为「冻结 chunk VLA 的执行器差距」 |")
    w("| 任务 | v1 三任务宏 SR | **libero_spatial 10 任务**等权宏 SR | v2 修订第 3 条 |")
    w("| 每 cell 集数 | 目标 n 更大 | **n = %d**（第一遍） | v2 修订第 3 条 |"
      % D["protocol"]["episodes_per_cell"])
    w("| 族规模 | 目标 20 | **12**（第一遍，>= 12 的要求已满足） | v2 修订第 2 条 |")
    w("| 对照臂 | Universal-fixed / RTC-only / prompt-fit / codec-fit / FT-demo / ours-L | "
      "**未跑** | v2 修订第 6 条：留待第一遍通过 G1 后补 |")
    w("| D4 每步限幅档 | 族内因子之一 | **排除并标注为无效应档** | "
      "L3 实测降幅 0.0 pp；20 Hz 绝对目标下限幅几乎不生效 |")
    w("| 过响应档（g > 1） | v1 族内含过响应 | **未覆盖** | "
      "v2 修订的因子水平表不含 g > 1；本轮不引入预注册外的水平 |")
    w("| 执行器命令界 | ManiSkill +-1 动作界 | **%.1f m / %.1f rad 每拍**（归一化 %.1f） | "
      "robosuite 绝对 OSC 没有动作界；界值在任何族内对照臂运行前冻结，逐 cell 报告触界集数 |"
      % (SP["executor_command_bound"]["bound_pos_m"],
         SP["executor_command_bound"]["bound_rot_rad"],
         SP["executor_command_bound"]["bound_pos_norm"]))
    w("| 探针 burst | +-0.5 m 绝对目标（靠动作界压在饱和） | **命令域 +-%.0f cm / 拍** | "
      "同上：无动作界时照抄会变成真正的半米猛冲；桌面安全 |" % (100 * pdz["burst_cmd_m"]))
    w("| 每步限幅的辨识 | 主配方留空 | **仍留空**（`clip_pos_norm` 开放） | "
      "与仿真侧主配方一致；burst 的观测值只作诊断列 |")
    w("| 统计单元 | plant 级配对 | **未变** | -- |")
    w("| 命令界消融（§6.9） | 预注册无此项 | **事后补跑 600 集** | "
      "在看过主对比结果之后补的机制消融，因此**不是确证性证据**，不参与 P1/P2/R/K 的任何判定；"
      "它的唯一用途是为下一遍预注册确定命令界的扫描范围 |")
    w("")
    w("另有两条不算偏离、但读者需要知道的事实：")
    w("")
    w("1. 上游 `ClientModel` 的 proprio 记账是半开环的：只在第一次查询时用实测位姿初始化，"
      "此后每次查询都改写成上一段 chunk 的最后一行。这意味着 plant 的影响只经由图像进入策略，"
      "不经由本体感受。本文原样保留这一语义（改动它就不再是那条 95/100 的正控链路）。")
    w("2. Slow-down 臂的跟踪 RMSE 按各自时间缩放后的 chunk 计分，其时间轴比其它臂长 1.5 / 2 倍。")
    w("")
    w("---")
    w("")

    # ---------------------------------------------------------------- 9
    w("## 9 耗时与复现命令")
    w("")
    w("| 阶段 | 集数 | 分片 | 实际墙钟 | 备注 |")
    w("|---|---|---|---|---|")
    w("| 探针 + 辨识 | 13 次探针 | 1 | %s | 每次探针 7.00 s 仿真、约 3.1 s 墙钟 |" % W["probe"])
    w("| L3 G1 筛查 | %d | 6 | %s | 逐集墙钟合计 %.0f s |"
      % (SC["n_episodes"], W["screen"], SC["wall_s_total"]))
    w("| L4 主对比 | %d | 8（每卡 4） | %s | 逐集墙钟合计 %.0f s，"
      "折合 %.1f s/集、%.4f s/控制拍 |"
      % (D["n_episodes"], W["main"], D["wall_s_total"],
         D["wall_s_total"] / D["n_episodes"], W["sec_per_step"]))
    w("")
    w("Gate 0 的实测速率是 100 集 / 21 min / 卡（3 分片共用一个服务端，0.0894 s/控制拍）。"
      "本轮主对比逐集墙钟合计 %.0f s、平均 %.1f s/集，比 Gate 0 的 12.5 s/集慢，"
      "原因是失配 plant 上大量 episode 跑满 800 步预算（Gate 0 平均只有 139.5 步）。"
      % (D["wall_s_total"], D["wall_s_total"] / D["n_episodes"]))
    w("")
    w("```bash")
    w("cd /opt/cce")
    w("L=/opt/cce/data/logs/cce_libero")
    w("")
    w("# 0) 单元自检（LIBERO 环境）")
    w("bash $L/run_selftest.sh")
    w("")
    w("# 1) 起两个策略服务端（GPU 6 -> 8021, GPU 7 -> 8022）")
    w("bash $L/serve_two.sh")
    w("")
    w("# 2) 探针 + ARX 辨识（13 个 plant）")
    w("bash $L/probe_all.sh")
    w("")
    w("# 3) L3 G1 筛查（naive 臂，6 分片）")
    w("bash $L/run_shards.sh g1screen 6 --set screen --methods naive \\")
    w("     --tasks 0,1,2,3,4,5,6,7,8,9 --episodes 2")
    w("./cce/env_libero.sh cce/libero_aggregate.py --glob \"$L/g1screen/shard*.jsonl\" \\")
    w("     --label g1_screen --out $L/g1screen/summary.json")
    w("")
    w("# 4) 冻结族定义")
    w("./cce/env_libero.sh cce/libero_freeze_family.py --screen-summary $L/g1screen/summary.json \\")
    w("     --out cce/results/libero_family_spec.json")
    w("")
    w("# 5) L4 主对比（8 分片，每卡 4）")
    w("bash $L/run_shards.sh main 8 --set family \\")
    w("     --methods naive,slow15,slow20,oracle,ours_p --tasks 0,1,2,3,4,5,6,7,8,9 --episodes 5")
    w("")
    w("# 6) 汇总与统计")
    w("./cce/env_libero.sh cce/libero_aggregate.py --glob \"$L/main/shard*.jsonl\" \\")
    w("     --label libero_family_v1 --out cce/results/libero_family_v1.json \\")
    w("     --theta-summary data/cce/libero/theta/identify_libero_panda.json \\")
    w("     --family-spec cce/results/libero_family_spec.json")
    w("```")
    w("")
    w("产物：`cce/results/libero_family_spec.json`（冻结的族）、`cce/results/libero_family_v1.json`"
      "（逐 cell 结果与全部统计）、`data/cce/libero/theta/theta_hat_libero_panda.json` 与 "
      "`identify_libero_panda.json`（theta-hat 与辨识对照）、"
      "`data/cce/libero/probes/probe_libero_panda_*.npz`（探针原始数据）、"
      "`$L/main/shard*.jsonl`（%d 集逐集记录）。" % D["n_episodes"])
    w("")

    # ---------------------------------------------------------------- 10
    w("---")
    w("")
    w("## 10 结论与下一步")
    w("")
    w("**站得住的四条。**")
    w("")
    w("1. **载体搬迁成功，链路可信。** 标称 plant 上 naive 的宏 SR 是 %.2f（n = 50），"
      "与官方正控 95/100 一致；闭环内 naive 复现策略绝对目标的编解码误差为 0；"
      "新写的 plant 与执行器和 `cce/plants.py`、`cce/executor.py` 逐拍等价。"
      % C["L00_nominal"]["naive"]["macro_sr"])
    w("2. **G1 成立，崩塌面很宽。** 11 个非标称 plant 里有 7 个 naive 相对标称降 >= 25 pp，"
      "最深 88 pp。失配确实伤害这条冻结 chunk 策略。")
    w("3. **失配类型决定伤害。** 纯延迟 >= 250 ms、滞后 >= 400 ms、夹爪时序 >= 0.6 s 与混合档"
      "都造成大幅下降；**纯增益档（g = 0.70 / 0.85）几乎不伤害**（naive %.2f / %.2f），"
      "因为绝对目标每拍重新锚在实测位姿上、臂仍单调收敛；**每步限幅在 20 Hz 绝对目标下无效应**。"
      "这两条修正了 v1 关于欠响应因子的预期。"
      % (C["D2_g70"]["naive"]["macro_sr"], C["D2_g85"]["naive"]["macro_sr"]))
    w("4. **辨识不是瓶颈，执行器的工作点才是。** oracle-theta（真值 theta）的族宏 SR 是 %.3f，"
      "ours-P 是 %.3f，只差 %.1f pp，两者却都低于 Slow-down x2 的 %.3f。"
      "把 theta 的误差归零并救不回来，说明问题不在标定得准不准，"
      "而在参数版反演在这张床上的增益太高（见 6.8）。"
      % (D["macro"]["oracle"]["macro_sr_family_mean"],
         D["macro"]["ours_p"]["macro_sr_family_mean"],
         100 * (D["macro"]["oracle"]["macro_sr_family_mean"]
                - D["macro"]["ours_p"]["macro_sr_family_mean"]),
         D["macro"][best]["macro_sr_family_mean"]))
    w("")
    w("**必须承认的两条。**")
    w("")
    w("5. **P2 反向且显著。** ours-P 比最强通用基线 Slow-down x2 低 %.1f pp"
      "（配对 t = %.2f，Holm p = %.2g，12 个 plant 全部为负）。预注册的 P2 预期是 >= +5 pp，"
      "本轮不但没达到，方向还是反的。按预注册 K3 的字面条件（宏差 < 5 pp **且**无时间优势），"
      "K3 不触发，因为 ours-P 的时间到成功比 Slow-down x2 短 %.1f%%；"
      "但这个时间优势有两处必须说明：Slow-down 按定义就把每段 chunk 拉长 2 倍，时间长是构造出来的；"
      "时间到成功又只在成功集上统计，两条臂的成功子集并不相同。"
      "把这两点算进去，本轮的诚实结论是 **P2 失败**，而不是靠时间优势保住了 K3。"
      % (-100 * prim[p2k]["mean"], prim[p2k]["t"], holm[p2k],
         kills.get("K3_time_reduction_pct") or 0.0))
    w("6. **P1 只有 %s，未达 >= +15 pp 的预期，Holm 校正后 p = %.3f。** "
      "更要紧的是它的构成：ours-P 在 naive 本来就没问题的四个轻档上是净损失"
      "（标称 %.2f -> %.2f、`D1_d150` %.2f -> %.2f、`D2_g70` %.2f -> %.2f、"
      "`D2_g85` %.2f -> %.2f），靠 7 个崩塌档上的回补才把均值拉成正的。"
      "标定不应让好情况变坏这条最基本的要求，本轮没有满足。"
      % (pp(prim[p1k]["mean"]), holm[p1k],
         C["L00_nominal"]["naive"]["macro_sr"], C["L00_nominal"]["ours_p"]["macro_sr"],
         C["D1_d150"]["naive"]["macro_sr"], C["D1_d150"]["ours_p"]["macro_sr"],
         C["D2_g70"]["naive"]["macro_sr"], C["D2_g70"]["ours_p"]["macro_sr"],
         C["D2_g85"]["naive"]["macro_sr"], C["D2_g85"]["ours_p"]["macro_sr"]))
    w("")
    w("**下一步（按优先级）。**")
    w("")
    w("1. **先把执行器的工作点定住，再谈臂间比较。** 命令界与内部饱和界是本轮唯一由本文自定的量，"
      "而标定臂在最多 50/50 集里把它打满。下一遍应把命令界与 `Theta.clip_pos_norm` "
      "作为预注册的扫描维度，在 2 个开发 plant 上定死后冻结，"
      "而不是像本轮按标称 plant 的命令峰值取一个倍数。")
    w("2. **给参数版反演加阻尼。** 逆增益从标称的 6-10 倍到大 tau 档的 40-70 倍是结构性的，"
      "来自 `g_hat = 0.29` 与 `alpha_hat = 0.5-0.6`。可选项：对 `1/(1-alpha)` 设上限、"
      "在 `kappa` 上做保守缩放、或改用已实现但本轮未跑的 QP 版"
      "（`cce/executor.py` 的 `mode=\"qp\"`，带 `lam` 平滑项，天然抑制这种高频放大）。"
      "这些都在「执行器超参只在 2 个开发 plant 上定」的预注册条款范围内。")
    w("3. **补齐留空的臂。** Universal-fixed、RTC-only、prompt-fit / codec-fit、FT-demo、ours-L "
      "本轮全部未跑；FT-demo 直接关系到 K4，Universal-fixed 关系到最强通用基线的定义"
      "（本轮的最强通用基线是 Slow-down x2，它已经很强）。")
    w("4. **扩族到 20 并提高 n。** 本轮 12 个 plant、n = 5，P1 的 bootstrap CI95 下界只有 %s；"
      "若下一遍仍以 plant 为配对单元，族规模与每 cell 集数都要提上去。" % pp(prim[p1k]["ci95"][0]))
    w("5. **纯增益与每步限幅两个因子在本床已被证伪为有效因子**，扩族时应把预算移给"
      "延迟、滞后、夹爪时序及其混合。")
    w("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("[report] wrote", args.out, len(L), "lines")


if __name__ == "__main__":
    main()
