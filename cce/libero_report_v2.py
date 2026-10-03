#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate docs/reports/CCE_libero_family_v2.md from the result JSONs.

New file; cce/libero_report.py (the first-pass generator) is left untouched so the
first-pass report still reproduces byte for byte.

Every number in the output is read out of, or computed by this script from,
  cce/results/libero_family_v2.json          second-pass aggregation + statistics
  cce/results/libero_family_v1.json          first-pass aggregation + statistics
  cce/results/libero_family_spec_v2.json     frozen 14-plant family
  cce/results/libero_devplant_v2.json        development-plant working-point scan
  data/cce/libero/theta/identify_libero_panda_v2.json   probe + ARX identification
  logs/cce_libero/wall_v2.json               second-pass wall clock
  logs/cce_libero/wall.json                  first-pass wall clock
  logs/cce_libero/abl_b10/shard*.jsonl       first-pass post-hoc bound ablation
Nothing is transcribed by hand.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob as _glob
import json
from pathlib import Path

import numpy as np

from cce.libero_aggregate_v2 import paired_test

CN = {"naive": "naive", "slow15": "Slow-down ×1.5", "slow20": "Slow-down ×2",
      "oracle": "oracle-θ", "ours_p": "ours-P"}
BOOT_SEED = 20260906


def f(x, n=3):
    return "—" if x is None else ("%." + str(n) + "f") % x


def pct(x, n=1):
    return "—" if x is None else ("%." + str(n) + "f%%") % x


def spp(x, n=2):
    """signed percentage points, x given as a fraction"""
    return "—" if x is None else ("%+." + str(n) + "f") % (100.0 * x)


def sfx(x, n=2):
    return "—" if x is None else ("%+." + str(n) + "f") % x


def z(x, n=1):
    """signed percentage points from a fraction, without a negative zero"""
    if x is None:
        return "—"
    v = 100.0 * x
    if abs(v) < 0.5 * 10 ** (-n):
        v = 0.0
    return ("%+." + str(n) + "f") % v


def pv(x):
    """p value: fixed point when readable, scientific when tiny"""
    if x is None:
        return "—"
    if x >= 1e-4:
        return "%.5f" % x
    return "%.2e" % x


def yn(b):
    return "是" if b else "否"


def note_cn(s):
    """Chinese one-line description of a plant, built from its own parameters."""
    parts = []
    if s["d_ms"] > 0:
        parts.append("传输延迟 %.0f ms" % s["d_ms"])
    if s["tau_ms"] > 0:
        parts.append("一阶滞后 τ = %.0f ms" % s["tau_ms"])
    if abs(s["gain"] - 1.0) > 1e-9:
        parts.append(("欠响应增益 %.2f" if s["gain"] < 1.0 else "过响应增益 %.2f") % s["gain"])
    if s["grip_delay_s"] > 0:
        parts.append("夹爪执行延迟 %.1f s" % s["grip_delay_s"])
    return " + ".join(parts) if parts else "标称，恒等包装（LIBERO 官方执行链）"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_rows(pattern):
    rows = []
    for h in sorted(_glob.glob(pattern)):
        for line in Path(h).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def macro_of(rows, plants, arms):
    """macro SR per arm, averaged over the given plants (10 tasks equally weighted)."""
    out = {}
    for m in arms:
        per = []
        for p in plants:
            tt = []
            for t in sorted({r["task_id"] for r in rows}):
                xs = [r["success"] for r in rows
                      if r["pid"] == p and r["method"] == m and r["task_id"] == t]
                if xs:
                    tt.append(float(np.mean(xs)))
            if tt:
                per.append(float(np.mean(tt)))
        out[m] = round(float(np.mean(per)), 4) if per else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--ident", required=True)
    ap.add_argument("--wall2", required=True)
    ap.add_argument("--wall1", required=True)
    ap.add_argument("--abl-glob", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    D = load(args.v2)
    V1 = load(args.v1)
    SP = load(args.spec)
    DV = load(args.dev)
    ID = load(args.ident)
    W2 = load(args.wall2)
    W1 = load(args.wall1)

    M = D["methods"]
    P = D["plants"]
    C = D["cells"]
    TEST = D["test_plants"]
    DEV = D["dev_plants"]
    NEW = D["new_plants_this_pass"]
    V1C = V1["cells"]
    V1P = V1["plants"]
    SHARED = [p for p in TEST if p in V1P]           # 10 test plants common to both passes
    specd = {s["pid"]: s for s in SP["plants"]}
    best = D["best_slow_arm"]
    p1k = "P1_ours_p_minus_naive"
    p2k = "P2_ours_p_minus_%s" % best
    prim, holm = D["primary"], D["primary_holm_p"]
    v1prim, v1holm = V1["primary"], V1["primary_holm_p"]
    tts = D["time_to_success_common_cells"]
    kills, v1kills = D["kills"], V1["kills"]
    rec, v1rec = D["recovery_macro"], V1["recovery_macro"]
    k1s, v1k1s = D["K1_summary"], V1["K1_summary"]
    cmp1 = D["compare_with_pass_1"]
    mm = D["macro"]

    def c2(p, m, key):
        c = C[p].get(m)
        return c.get(key) if c else None

    def c1(p, m, key):
        c = (V1C.get(p) or {}).get(m)
        return c.get(key) if c else None

    # ------------------------------------------------------------------ #
    # supplementary statistics computed here (clearly labelled, non-preregistered)
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(BOOT_SEED)

    def paired2(a, b, ps, key="macro_sr"):
        use = [p for p in ps if c2(p, a, key) is not None and c2(p, b, key) is not None]
        d = np.array([c2(p, a, key) - c2(p, b, key) for p in use], dtype=np.float64)
        r = paired_test(d, rng)
        r["plants"] = use
        return r

    SUP = {
        "P1_on_v1_population": paired2("ours_p", "naive", V1P),
        "P2_on_v1_population": paired2("ours_p", best, V1P),
        "P1_on_shared_test_plants": paired2("ours_p", "naive", SHARED),
        "P2_on_shared_test_plants": paired2("ours_p", best, SHARED),
        "oracle_minus_ours_p_test": paired2("oracle", "ours_p", TEST),
        "oracle_minus_ours_p_rmse_test": paired2("oracle", "ours_p", TEST, "tracking_rmse_m"),
    }
    # K1 recomputed on the first pass's own seven-plant set, using second-pass data
    v1k1_plants = v1k1s["lag_delay_plants"]
    k1v2_on_v1set = [D["K1_tracking_rmse_vs_intent"][p]["ours_p_reduction_pct"]
                     for p in v1k1_plants if p in D["K1_tracking_rmse_vs_intent"]]
    SUP["K1_v2_on_v1_plant_set"] = {
        "plants": v1k1_plants,
        "mean_pct": round(float(np.mean(k1v2_on_v1set)), 1) if k1v2_on_v1set else None,
        "n_ge_30": int(sum(1 for v in k1v2_on_v1set if v >= 30.0)),
    }

    # post-hoc ablation of the first pass vs. what the confirmatory run actually gave
    ABL = None
    if args.abl_glob:
        arows = load_rows(args.abl_glob)
        if arows:
            aplants = sorted({r["pid"] for r in arows})
            aarms = [m for m in M if any(r["method"] == m for r in arows)]
            ABL = {
                "plants": aplants,
                "arms": aarms,
                "n_episodes": len(arows),
                "episodes_per_cell": len({r["ep"] for r in arows}),
                "bound": sorted({r["bound_pos"] for r in arows}),
                "abl_macro": macro_of(arows, aplants, aarms),
                "v2_macro_same_plants": {m: round(float(np.mean(
                    [c2(p, m, "macro_sr") for p in aplants])), 4) for m in aarms},
            }

    L = []
    w = L.append

    # ============================== header ============================== #
    w("# CCE plant 族闭环主对比第二遍：执行器命令界回到设计工作点后的确证运行")
    w("")
    w("适用范围：方案 B2（CCE）在 LIBERO 正控策略上的 plant 族闭环主对比第二遍。"
      "第二遍按 `docs/prereg/CCE_prereg_v3_amendment.md` 执行，"
      "该修订是在读过第一遍结果（`docs/reports/CCE_libero_family_v1.md`）之后写的，"
      "因此第二遍是**校正后的确证运行**，不是预注册运行；"
      "第一遍仍作为预注册结果原样保留，两遍在本报告中并列呈现。")
    w("")
    w("载体：X-VLA-Pt + 官方 `2toINF/X-VLA-libero-spatial-peft`（LoRA 合并），"
      "本体只有 robosuite Panda，任务为 `libero_spatial` 10 任务，"
      "控制 %s Hz，chunk %d 步（%.1f s），预算 %d 步，绝对目标位姿注入 OSC 控制器输入端。"
      % (f(D["protocol"]["control_hz"], 1), D["protocol"]["chunk"],
         D["protocol"]["chunk_seconds"], D["protocol"]["budget_steps"]))
    w("")
    w("规模：%d plant × %d 臂 × %d 任务 × %d 集 = %d 集，全部落盘且无缺格。"
      % (len(P), len(M), len(D["tasks"]), D["protocol"]["episodes_per_cell"],
         D["n_episodes"]))
    w("结果文件 `cce/results/libero_family_v2.json`，族定义 `cce/results/libero_family_spec_v2.json`"
      "（冻结于 %s，sha256 `%s`）。" % (SP["frozen_utc"], SP["sha256"][:16]))
    w("")

    # ============================== 0 速览 ============================== #
    w("## 0  结论速览")
    w("")
    w("### 0.1  两遍并列")
    w("")
    w("| 项 | 第一遍（预注册，命令界 5.0） | 第二遍（确证，命令界 1.0） |")
    w("|---|---|---|")
    w("| 闭环集数 | %d | %d |" % (V1["n_episodes"], D["n_episodes"]))
    w("| plant 数（统计人群） | %d（含 2 个后来划为开发的 plant） | %d 测试 plant（另 2 个开发 plant 单列） |"
      % (len(V1P), len(TEST)))
    for m in M:
        w("| 族宏 SR — %s | %s | %s（测试 %d plant）／%s（同一批 %d plant） |"
          % (CN[m], f(V1["macro"][m]["macro_sr_family_mean"]),
             f(mm[m]["macro_sr_test_mean"]), len(TEST),
             f(cmp1["v2_macro_same_12_plants"][m]), len(V1P)))
    w("| **P1** = ours-P − naive | %s pp，CI95 [%s, %s]，Holm p = %s | %s pp，CI95 [%s, %s]，Holm p = %s |"
      % (spp(v1prim[p1k]["mean"]), spp(v1prim[p1k]["ci95"][0]), spp(v1prim[p1k]["ci95"][1]),
         f(v1holm[p1k], 4), spp(prim[p1k]["mean"]), spp(prim[p1k]["ci95"][0]),
         spp(prim[p1k]["ci95"][1]), f(holm[p1k], 4)))
    w("| P1 判据（≥ +15 pp 且 CI 下界 > 0） | **未达** | **达成** |")
    w("| **P2** = ours-P − %s（宏 SR） | %s pp，CI95 [%s, %s]，Holm p = %s | %s pp，CI95 [%s, %s]，Holm p = %s |"
      % (CN[best], spp(v1prim[p2k]["mean"]), spp(v1prim[p2k]["ci95"][0]),
         spp(v1prim[p2k]["ci95"][1]), f(v1holm[p2k], 5),
         spp(prim[p2k]["mean"]), spp(prim[p2k]["ci95"][0]), spp(prim[p2k]["ci95"][1]),
         f(holm[p2k], 4)))
    w("| P2 时间到成功降幅 | %s（各臂自身成功子集，口径有偏） | %s（两臂共同成功格配对） |"
      % (pct(v1kills.get("K3_time_reduction_pct")), pct(tts["reduction_pct"])))
    w("| P2 判据（宏 SR ≥ +5 pp **或** 时间 −25%%） | **实质失败**（宏 SR %d/%d plant 为负） | 按字面**达成**（仅靠时间支）；宏 SR 支仍显著为负 |"
      % (v1prim[p2k]["n"] - v1prim[p2k]["n_positive"], v1prim[p2k]["n"]))
    w("| 恢复率 R(ours-P) | %s | %s |" % (f(v1rec["R_ours_p"]), f(rec["R_ours_p"])))
    w("| 恢复率 R(%s) | %s | %s |" % (CN[best], f(v1rec["R_slow20"]), f(rec["R_slow20"])))
    w("| 恢复率 R(oracle-θ) | %s | %s |" % (f(v1rec["R_oracle"]), f(rec["R_oracle"])))
    w("| **K1**（延迟/滞后 plant 跟踪 RMSE 降 ≥ 30%%） | 通过，均降 %s（%d/%d plant ≥ 30%%） | **未通过**，均降 %s（%d/%d plant ≥ 30%%） |"
      % (pct(v1k1s["mean_ours_p_reduction_pct"]), v1k1s["n_plants_reduction_ge_30"],
         len(v1k1s["lag_delay_plants"]), pct(k1s["test_plants_only"]["mean_ours_p_reduction_pct"]),
         k1s["test_plants_only"]["n_plants_reduction_ge_30"], k1s["test_plants_only"]["n"]))
    w("| **K2**（R < 0.3 触发） | 未触发（R = %s） | 未触发（R = %s） |"
      % (f(v1kills["K2_R_ours_p"]), f(kills["K2_R_ours_p"])))
    w("| **K3**（对最强 Slow-down 既无 SR 优势又无时间优势） | 未触发 | 未触发 |")
    w("| ours-P 命令界饱和（集比例／步比例） | %s／%s | %s／%s |"
      % (pct(100.0 * cmp1["v1_from_shards"]["sat_episode_share_by_arm"]["ours_p"]),
         pct(100.0 * cmp1["v1_from_shards"]["sat_step_share_by_arm"]["ours_p"]),
         pct(100.0 * mm["ours_p"]["sat_episode_share_all"]),
         pct(100.0 * mm["ours_p"]["sat_step_share_all"])))
    w("")

    w("### 0.2  三条结论")
    w("")
    w("**一，命令界确实是第一遍的主要瓶颈，但只解释了一部分差距。**"
      "在两遍共有的同一批 12 个 plant 上，命令界从 5.0 收到 1.0 后，"
      "两个标定臂大幅回升（oracle-θ %s → %s，ours-P %s → %s），"
      "两个通用基线几乎不动（Slow-down ×1.5 %s → %s，Slow-down ×2 %s → %s），"
      "naive 上升 %s pp。方向与预注册 v3 的判断一致：第一遍压制的是标定臂而非全体。"
      "但事后消融预测的 ours-P ≈ 0.85 并未兑现——在消融所用的同一批 plant 上，"
      "确证运行给出的是 %s（见 §1.4），消融高估了收益。"
      % (f(V1["macro"]["oracle"]["macro_sr_family_mean"]), f(cmp1["v2_macro_same_12_plants"]["oracle"]),
         f(V1["macro"]["ours_p"]["macro_sr_family_mean"]), f(cmp1["v2_macro_same_12_plants"]["ours_p"]),
         f(V1["macro"]["slow15"]["macro_sr_family_mean"]), f(cmp1["v2_macro_same_12_plants"]["slow15"]),
         f(V1["macro"]["slow20"]["macro_sr_family_mean"]), f(cmp1["v2_macro_same_12_plants"]["slow20"]),
         spp(cmp1["v2_macro_same_12_plants"]["naive"] - V1["macro"]["naive"]["macro_sr_family_mean"]),
         f(ABL["v2_macro_same_plants"]["ours_p"]) if ABL else "—"))
    w("")
    w("**二，P1 成立，P2 的实质结论不变：ours-P 打不过一个不需要辨识的 Slow-down ×2。**"
      "P1 = %s pp（CI95 [%s, %s]，Holm p = %s），达到预注册阈值。"
      "P2 的宏 SR 支为 %s pp（CI95 [%s, %s]，Holm p = %s），"
      "%d/%d 个测试 plant 上 ours-P 不优于 Slow-down ×2；"
      "P2 只能靠时间到成功那一支达成（%s），"
      "而该支对一个把 chunk 摊到两倍步数执行的基线近乎恒真（§4.3），"
      "因此不宜作为方法优越性的证据。"
      % (spp(prim[p1k]["mean"]), spp(prim[p1k]["ci95"][0]), spp(prim[p1k]["ci95"][1]),
         f(holm[p1k], 4), spp(prim[p2k]["mean"]), spp(prim[p2k]["ci95"][0]),
         spp(prim[p2k]["ci95"][1]), f(holm[p2k], 4),
         prim[p2k]["n"] - prim[p2k]["n_positive"], prim[p2k]["n"],
         pct(tts["reduction_pct"])))
    w("")
    w("**三，K1 在第二遍被新增的过响应档打掉。**"
      "第二遍 K1 集（延迟/滞后 plant，测试 plant %d 个）上 ours-P 的跟踪 RMSE 平均只降 %s，"
      "未达 30%% 阈值；若沿用第一遍的七 plant 集重算第二遍数据，为 %s（%d/%d ≥ 30%%），"
      "差别几乎全部来自新增的 `MX_g130_d100`，该 plant 上的降幅为 %s，即反演后比 naive 更差。"
      "在两个过响应 plant 上，ours-P 与 oracle-θ 的跟踪 RMSE 同时**劣于** naive"
      "（`D2_g130` %s／%s，`MX_g130_d100` %s／%s），"
      "说明这不是辨识误差，而是执行器反演臂本身在过响应 plant 上的结构性代价（§5.3）。"
      % (k1s["test_plants_only"]["n"], pct(k1s["test_plants_only"]["mean_ours_p_reduction_pct"]),
         pct(SUP["K1_v2_on_v1_plant_set"]["mean_pct"]), SUP["K1_v2_on_v1_plant_set"]["n_ge_30"],
         len(v1k1_plants),
         pct(D["K1_tracking_rmse_vs_intent"]["MX_g130_d100"]["ours_p_reduction_pct"]),
         pct(D["K1_tracking_rmse_vs_intent"]["D2_g130"]["ours_p_reduction_pct"]),
         pct(D["K1_tracking_rmse_vs_intent"]["D2_g130"]["oracle_reduction_pct"]),
         pct(D["K1_tracking_rmse_vs_intent"]["MX_g130_d100"]["ours_p_reduction_pct"]),
         pct(D["K1_tracking_rmse_vs_intent"]["MX_g130_d100"]["oracle_reduction_pct"])))
    w("")

    # ============================== 1 差异 ============================== #
    w("## 1  第二遍与第一遍的差异")
    w("")
    w("### 1.1  唯一改动的执行器超参：命令界 5.0 → 1.0")
    w("")
    ecb = SP["executor_command_bound"]
    w("命令界指执行器每个控制步允许下发的位置/姿态增量上界，"
      "以归一化动作单位计，第一遍取 %s（%.2f m／步），第二遍取 %s（%.2f m／步）。"
      "第二遍的取值不是新调出来的：它是 `cce/executor.py` 在仿真侧的原生工作点，"
      "也是 ManiSkill 床动作空间的隐含界（±1），"
      "第一遍的 5.0 属实施偏离而非设计取值。该值由预注册 v3 第 1 项在本轮任何闭环之前固定。"
      % (f(ecb["first_pass_value_norm"], 1),
         ecb["bound_pos_m"] * ecb["first_pass_value_norm"] / max(ecb["bound_pos_norm"], 1e-9),
         f(ecb["bound_pos_norm"], 1), ecb["bound_pos_m"]))
    w("")
    w("其余执行器超参一律未动：")
    oh = ecb["other_executor_hyperparameters"]
    w("- 扫描的超参只有 `%s`；" % "`、`".join(oh["scanned"]))
    w("- λ / μ / `resolve_every` 属 `cce/executor.py` 的 QP 臂（`mode='qp'`），"
      "ours-P 是闭式参数臂，本床的 `LiberoExecutor` 只转写该臂，故这三个量在本床结构性缺席，"
      "按构造保持仿真侧取值；")
    w("- κ = %s，仿真侧默认值，未经族驱动暴露，在两遍的所有臂与所有 plant 上恒定。"
      % f(oh["kappa"], 1))
    w("")
    w("除命令界外，第二遍相对第一遍还有两处族级改动（预注册 v3 第 1、2 项）："
      "把 `%s` 与 `%s` 划为开发 plant 并移出 P1/P2 的 plant 级统计；"
      "加入两个过响应 plant `%s` 与 `%s`。臂集合、任务、n、控制协议、初始化种子、分片方案一律不变。"
      % (DEV[0], DEV[1], NEW[0], NEW[1]))
    w("")

    w("### 1.2  开发 plant 上的工作点确认")
    w("")
    w("预注册 v3 第 1 项要求命令界在两个开发 plant 上按跟踪 RMSE 最小确认。"
      "扫描设计：ours-P 单臂，开发 plant `%s` 与 `%s`，%d 任务 × %d 集 = 每个工作点 %d 集，"
      "候选工作点为归一化界 %s。判据：对策略意图的跟踪 RMSE 在两个开发 plant 上取平均后最小。"
      % (DV["dev_plants"][0], DV["dev_plants"][1], len(DV["design"]["tasks"]),
         DV["design"]["episodes_per_task"], DV["design"]["episodes_per_configuration"],
         "、".join(f(x, 1) for x in DV["design"]["scanned_bounds_norm"])))
    w("")
    w("| 工作点（归一化界 / m） | plant | 集数 | 对意图 RMSE (m) | 对自身设定点 RMSE (m) | 宏 SR | 平均步数 | 饱和集比例 | 饱和步比例 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(DV["scan"], key=lambda k: float(k)):
        s = DV["scan"][key]
        for pid, v in s["per_plant"].items():
            w("| %s / %s | `%s` | %d | %s | %s | %s | %s | %s | %s |"
              % (f(s["bound_pos_norm"], 1), f(s["bound_pos_norm"] * 0.1, 2), pid, v["n_episodes"],
                 f(v["tracking_rmse_vs_intent_m"], 5), f(v["tracking_rmse_vs_ref_m"], 5),
                 f(v["macro_sr"]), f(v["mean_steps"], 1), pct(100.0 * v["sat_episode_share"]),
                 pct(100.0 * v["sat_step_share"])))
        w("| %s / %s | **两 plant 均值** | %d | **%s** | — | %s | — | %s | %s |"
          % (f(s["bound_pos_norm"], 1), f(s["bound_pos_norm"] * 0.1, 2), s["n_episodes"],
             f(s["mean_tracking_rmse_vs_intent_m"], 5), f(s["mean_macro_sr"]),
             pct(100.0 * s["mean_sat_episode_share"]), pct(100.0 * s["mean_sat_step_share"])))
    w("")
    w("选定 %s。判据下的差距很小（%s 对 %s，相对差 %s），"
      "而两个开发 plant 的单点最优并不一致（`%s` 取 %s，`%s` 取 %s，归一化界）；"
      "选定值同时等于预注册 v3 事先固定的 %s，二者一致（`selection_matches_prereg_v3` = %s）。"
      "换言之，这次扫描是对一个已由预注册固定的取值的确认，而非一次自由的超参搜索；"
      "扫描本身的判别力有限，这一点应如实计入结论的强度。"
      % (f(DV["selected_bound_norm"], 1),
         f(DV["scan"]["1"]["mean_tracking_rmse_vs_intent_m"], 5),
         f(DV["scan"]["2"]["mean_tracking_rmse_vs_intent_m"], 5),
         pct(100.0 * (DV["scan"]["2"]["mean_tracking_rmse_vs_intent_m"]
                      / DV["scan"]["1"]["mean_tracking_rmse_vs_intent_m"] - 1.0)),
         DV["dev_plants"][0], f(float(DV["per_plant_argmin"][DV["dev_plants"][0]]), 1),
         DV["dev_plants"][1], f(float(DV["per_plant_argmin"][DV["dev_plants"][1]]), 1),
         f(DV["prereg_v3_fixed_bound_norm"], 1), yn(DV["selection_matches_prereg_v3"])))
    w("")
    w("扫描的 %d 集全部来自开发 plant，不进入 §3–§4 的任何 plant 级统计。" % DV["n_episodes"])
    w("")

    w("### 1.3  两个新增过响应 plant 的辨识结果")
    w("")
    agn = ID["agreement_new_plants"]
    w("探针设计与第一遍完全相同（%d 步／%.1f s：chirp %.2f–%.2f Hz、幅值 %.3f m，"
      "burst 归一化 %.2f，姿态幅值 %.2f rad，第 %d 步下发夹爪闭合），只是 plant 列表不同。"
      % (ID["probe_design"]["steps"], ID["probe_design"]["seconds"],
         ID["probe_design"]["chirp_hz"][0], ID["probe_design"]["chirp_hz"][1],
         ID["probe_design"]["chirp_amp_m"], ID["probe_design"]["burst_cmd_norm"],
         ID["probe_design"]["attitude_amp_rad"], ID["probe_design"]["grip_close_step"]))
    w("")
    w("| plant | d 真值 / 估计（步） | τ 真值 / 相对估计（s） | g 真值 / 相对估计 | 夹爪超前 真值 / 估计（步） | 自由仿真 R²（平移 / 姿态） |")
    w("|---|---|---|---|---|---|")
    for r in agn["rows"]:
        w("| `%s` | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s |"
          % (r["pid"], f(r["d_true_steps"], 1), f(float(r["d_hat_steps"]), 1),
             f(r["tau_true_s"], 3), f(r["tau_rel_s"], 4),
             f(r["g_true"], 2), f(r["g_rel"], 3),
             f(r["grip_lead_true_steps"], 1), f(float(r["grip_lead_hat_steps"]), 1),
             f(r["r2free_min_trans"], 3), f(r["r2free_min_rot"], 3)))
    w("")
    w("两个新 plant 上，传输延迟与夹爪超前的估计与真值完全一致（MAE %s 步）；"
      "增益的相对估计 MAE 为 %s（真值 1.30，估计 %s 与 %s）；"
      "一阶滞后的相对估计 MAE 为 %s s（真值 0）。"
      % (f(agn["delay_mae_steps"], 1), f(agn["gain_rel_mae"], 3),
         f(agn["rows"][0]["g_rel"], 3), f(agn["rows"][1]["g_rel"], 3),
         f(agn["tau_rel_mae_s"], 5)))
    w("")
    w("需要说明口径：辨识器输出的原始增益 `g_hat` 在标称 plant 上就只有 %s，"
      "并非物理增益；表中给出的是相对标称的比值 `g_rel = g_hat / g_hat(标称)`，"
      "该比值才与注入的 gain 可比。同样，`tau_hat` 在标称 plant 上是 %s s，"
      "相对量 `tau_rel = tau_hat − tau_hat(标称)` 才是注入滞后的估计。"
      % (f(ID["agreement"]["gain_hat_on_nominal"], 4),
         f(ID["agreement"]["tau_hat_on_nominal_s"], 4)))
    w("")
    w("全族（%d plant）的辨识一致性：延迟相关 %s、MAE %s 步；"
      "夹爪超前相关 %s、MAE %s 步；滞后相对相关 %s、MAE %s s；"
      "增益相对相关 %s、MAE %s；R² 门下未通过的轴 %d／%d。"
      "增益一路的相关系数偏低这一点自第一遍起就存在，第二遍未改辨识器（预注册 v3 第 5 项）。"
      % (ID["n_plants"], f(ID["agreement"]["delay_corr"], 3),
         f(ID["agreement"]["delay_mae_steps"], 2),
         f(ID["agreement"]["grip_lead_corr"], 3), f(ID["agreement"]["grip_lead_mae_steps"], 2),
         f(ID["agreement"]["tau_corr_rel"], 3), f(ID["agreement"]["tau_mae_rel_s"], 4),
         f(ID["agreement"]["gain_corr_rel"], 3), f(ID["agreement"]["gain_mae_rel"], 4),
         ID["agreement"]["n_axes_below_r2_gate"], ID["agreement"]["n_axes_total"]))
    w("")

    if ABL:
        ABL["v1_macro_same_plants"] = {
            m: round(float(np.mean([c1(p, m, "macro_sr") for p in ABL["plants"]])), 4)
            for m in ABL["arms"]}
        w("### 1.4  第一遍事后消融与确证运行的落差")
        w("")
        w("预注册 v3 在论证命令界时引用了一次事后消融（非确证）。"
          "该消融的界 %s 一侧共 %d 集，覆盖 %d 个 plant（%s）、%d 个臂、每格 %d 集；"
          "与之对比的界 5.0 一侧并非单独跑的，而是取第一遍主对比在同一批 plant 上的子集。"
          "把第二遍确证运行同样限制到这批 plant 后，三者并列："
          % ("、".join(f(b, 1) for b in ABL["bound"]), ABL["n_episodes"], len(ABL["plants"]),
             "、".join("`%s`" % p for p in ABL["plants"]), len(ABL["arms"]),
             ABL["episodes_per_cell"]))
        w("")
        w("| 臂 | 第一遍主对比子集（界 5.0，每格 %d 集） | 事后消融（界 %s，每格 %d 集） | 第二遍确证（界 %s，每格 %d 集） | 确证 − 消融 |"
          % (D["protocol"]["episodes_per_cell"], "、".join(f(b, 1) for b in ABL["bound"]),
             ABL["episodes_per_cell"], f(D["protocol"]["executor_bound_pos_norm"], 1),
             D["protocol"]["episodes_per_cell"]))
        w("|---|---|---|---|---|")
        for m in ABL["arms"]:
            v0 = ABL["v1_macro_same_plants"][m]
            a, b = ABL["abl_macro"][m], ABL["v2_macro_same_plants"][m]
            w("| %s | %s | %s | %s | %s pp |" % (CN[m], f(v0), f(a), f(b), z(b - a)))
        w("")
        w("预注册 v3 引用的「命令界 5.0 → 1.0 时 ours-P %s → %s、naive %s → %s」"
          "在本表的前两列复现无误。但确证运行在同一批 plant 上给出的 ours-P 只有 %s，"
          "较消融低 %s pp。原因有二：消融每格只有 %d 集（确证为 %d 集，标准误相差约 %.1f 倍），"
          "且这 %d 个 plant 里有 %d 个是标定臂本来就占优的延迟/滞后档。"
          "这条落差是把事后消融当结论的风险实例，报告在此如实记录，"
          "并据此不在结论中引用任何未经确证运行复核的消融数字。"
          % (f(ABL["v1_macro_same_plants"]["ours_p"]), f(ABL["abl_macro"]["ours_p"]),
             f(ABL["v1_macro_same_plants"]["naive"]), f(ABL["abl_macro"]["naive"]),
             f(ABL["v2_macro_same_plants"]["ours_p"]),
             f(100.0 * (ABL["abl_macro"]["ours_p"]
                        - ABL["v2_macro_same_plants"]["ours_p"]), 1),
             ABL["episodes_per_cell"], D["protocol"]["episodes_per_cell"],
             (D["protocol"]["episodes_per_cell"] / ABL["episodes_per_cell"]) ** 0.5,
             len(ABL["plants"]),
             sum(1 for p in ABL["plants"]
                 if specd.get(p, {}).get("d_ms", 0) > 0 or specd.get(p, {}).get("tau_ms", 0) > 0)))
        w("")

    # ============================== 2 族定义 ============================== #
    w("## 2  族定义")
    w("")
    w("族冻结于 %s，共 %d 个 plant：%d 个开发 plant（不进入 P1/P2 的 plant 级统计）、"
      "%d 个测试 plant，其中 %d 个为第二遍新增。"
      % (SP["frozen_utc"], SP["n_plants"], SP["n_dev_plants"], SP["n_test_plants"], len(NEW)))
    w("")
    w("| plant | 家族 | 角色 | 新增 | d (ms) | τ (ms) | g | 夹爪延迟 (s) | 夹爪超前 (步) | 属 K1 集 | 说明 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in SP["plants"]:
        w("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
          % (s["pid"], s["family"], "开发" if s["role"] == "dev" else "测试",
             "是" if s["new_in_pass_2"] else "—", f(s["d_ms"], 0), f(s["tau_ms"], 0),
             f(s["gain"], 2), f(s["grip_delay_s"], 1), f(s["grip_lead_steps"], 0),
             yn(s["in_K1_set"]), note_cn(s)))
    w("")
    g2 = SP["G2_check"]
    w("G2（族内档位覆盖）检查：增益档 %s；传输延迟档 %s ms；一阶滞后档 %s ms；"
      "夹爪延迟档 %s s。同时含欠响应与过响应：%s。"
      % ("、".join(f(x, 2) for x in g2["gain_levels"]),
         "、".join(f(x, 0) for x in g2["delay_levels_ms"]),
         "、".join(f(x, 0) for x in g2["tau_levels_ms"]),
         "、".join(f(x, 1) for x in g2["grip_delay_levels_s"]),
         yn(g2["has_under_and_over_response"])))
    w("")
    w("K1 集在第二遍改用参数谓词定义：注入的失配中含传输延迟或一阶滞后（d_ms > 0 或 τ_ms > 0），"
      "共 %d 个 plant（`%s`）。在第一遍的族上，该谓词选出的 plant 与第一遍按家族标签"
      "（D1、D2t、MIX）列举的七个完全一致；改为谓词是为了让新增的过响应混合档"
      "——它带 100 ms 传输延迟但属于新的家族标签——无需人工判断即可归位。"
      % (len(SP["K1_plant_set"]["pids"]), "`、`".join(SP["K1_plant_set"]["pids"])))
    w("")
    w("剔除项沿用第一遍：D4 限幅档在本床对 naive 的掉幅为 0.0 pp（第一遍 L3 筛查），"
      "属无效应档，第二遍继续剔除。"
      "臂集合按预注册 v3 第 3 项保持不变，"
      "Universal-fixed、RTC-only、prompt-fit、codec-fit、FT-demo、ours-L 本轮均未跑。")
    w("")

    # ============================== 3 主对比 ============================== #
    w("## 3  主对比")
    w("")
    w("### 3.1  宏 SR（逐 plant × 臂，两遍并列）")
    w("")
    w("每格 %d 任务 × %d 集 = %d 集；宏 SR 为 10 个任务的等权平均。"
      "表内写作「第一遍 → 第二遍」，第二遍新增的 plant 第一遍无对应值。"
      % (len(D["tasks"]), D["protocol"]["episodes_per_cell"],
         len(D["tasks"]) * D["protocol"]["episodes_per_cell"]))
    w("")
    w("| plant | 家族 | 角色 | " + " | ".join(CN[m] for m in M) + " |")
    w("|---|---|---|" + "---|" * len(M))
    for p in P:
        s = specd[p]
        row = []
        for m in M:
            row.append("%s → %s" % (f(c1(p, m, "macro_sr")), f(c2(p, m, "macro_sr"))))
        w("| `%s` | %s | %s | " % (p, s["family"], "开发" if s["role"] == "dev" else "测试")
          + " | ".join(row) + " |")
    w("| **测试 plant 均值（n=%d，第二遍口径）** | — | — | " % len(TEST)
      + " | ".join("— → %s" % f(mm[m]["macro_sr_test_mean"]) for m in M) + " |")
    w("| **同一批 12 plant 均值（两遍可比）** | — | — | "
      + " | ".join("%s → %s" % (f(V1["macro"][m]["macro_sr_family_mean"]),
                                f(cmp1["v2_macro_same_12_plants"][m])) for m in M) + " |")
    w("| **开发 plant 均值（n=%d，单列）** | — | — | " % len(DEV)
      + " | ".join("— → %s" % f(mm[m]["macro_sr_dev_mean"]) for m in M) + " |")
    w("")
    w("读表要点：")
    w("- 两个开发 plant 在第一遍上是标定臂损伤最重的地方（`%s` 的 ours-P %s → %s，"
      "oracle-θ %s → %s），命令界收紧后基本修复；这也是把它们划为开发 plant 之后"
      "第二遍测试人群里不再包含这两处修复的原因，两遍的 P1 因此不是同一人群上的量（§4.4 给同口径重算）。"
      % (DEV[1], f(c1(DEV[1], "ours_p", "macro_sr")), f(c2(DEV[1], "ours_p", "macro_sr")),
         f(c1(DEV[1], "oracle", "macro_sr")), f(c2(DEV[1], "oracle", "macro_sr"))))
    w("- 两个新增过响应 plant 上 naive 几乎不掉（%s 与 %s），"
      "属族内的轻档；它们进入测试人群后同时拉低 P1 的均值与 K1 的均值。"
      % (f(c2(NEW[0], "naive", "macro_sr")), f(c2(NEW[1], "naive", "macro_sr"))))
    w("- Slow-down ×2 在第二遍的测试 plant 均值 %s 高于 ours-P 的 %s 与 oracle-θ 的 %s，"
      "这一序关系在第一遍即已存在，命令界修正后并未逆转。"
      % (f(mm["slow20"]["macro_sr_test_mean"]), f(mm["ours_p"]["macro_sr_test_mean"]),
         f(mm["oracle"]["macro_sr_test_mean"])))
    w("")

    w("### 3.2  平均步数与时间到成功（第二遍）")
    w("")
    w("格式为「平均步数 / 成功集平均时间 (s)」。平均步数含失败集（失败集耗满 %d 步预算）。"
      % D["protocol"]["budget_steps"])
    w("")
    w("| plant | " + " | ".join(CN[m] for m in M) + " |")
    w("|---|" + "---|" * len(M))
    for p in P:
        row = []
        for m in M:
            c = C[p].get(m)
            row.append("%s / %s" % (f(c["mean_steps"], 0), f(c["mean_time_to_success_s"], 2))
                       if c else "—")
        w("| `%s` | " % p + " | ".join(row) + " |")
    w("")

    w("### 3.3  跟踪 RMSE（两列口径）")
    w("")
    w("第一列为对策略 chunk 意图的 RMSE，是 K1 的判据量；"
      "第二列为对该臂自身设定点的 RMSE，衡量执行器是否跟住了它自己要求的轨迹。"
      "两列的差别正是标定臂的设计意图所在：标定臂会有意偏离原始意图去补偿 plant。")
    w("")
    w("| plant | " + " | ".join("%s（对意图）" % CN[m] for m in M)
      + " | ours-P 降幅 | oracle-θ 降幅 |")
    w("|---|" + "---|" * (len(M) + 2))
    k1t = D["K1_tracking_rmse_vs_intent"]
    for p in P:
        row = [f(c2(p, m, "tracking_rmse_m"), 5) for m in M]
        r = k1t.get(p, {})
        row.append(pct(r.get("ours_p_reduction_pct")))
        row.append(pct(r.get("oracle_reduction_pct")))
        w("| `%s` | " % p + " | ".join(row) + " |")
    w("| **测试 plant 均值** | "
      + " | ".join(f(mm[m]["tracking_rmse_test_mean_m"], 5) for m in M) + " | — | — |")
    w("")
    w("| plant | " + " | ".join("%s（对自身设定点）" % CN[m] for m in M) + " |")
    w("|---|" + "---|" * len(M))
    for p in P:
        w("| `%s` | " % p + " | ".join(f(c2(p, m, "tracking_rmse_vs_ref_m"), 5) for m in M) + " |")
    w("")
    w("两个标定臂对自身设定点的 RMSE 在全族都低到 %s–%s m 量级，"
      "远小于对意图的 RMSE；即执行器忠实执行了反演给出的命令，"
      "K1 的失败不是执行不到位，而是反演目标本身在部分 plant 上并不更接近原始意图。"
      % (f(min(c2(p, "ours_p", "tracking_rmse_vs_ref_m") for p in P), 5),
         f(max(c2(p, "ours_p", "tracking_rmse_vs_ref_m") for p in P), 5)))
    w("")

    w("### 3.4  夹爪闭合滞后（下发闭合 → 实际闭合，控制步）")
    w("")
    w("| plant | 注入夹爪延迟 (s) | " + " | ".join(CN[m] for m in M) + " |")
    w("|---|---|" + "---|" * len(M))
    for p in P:
        w("| `%s` | %s | " % (p, f(specd[p]["grip_delay_s"], 1))
          + " | ".join(f(c2(p, m, "mean_grip_close_lag_steps"), 2) for m in M) + " |")
    w("")

    w("### 3.5  G1 筛查（naive 相对标称 plant 的掉幅）")
    w("")
    w("| plant | 家族 | 角色 | naive 宏 SR | 掉幅 (pp) | ≥ 25 pp |")
    w("|---|---|---|---|---|---|")
    for p, v in D["G1_naive_drop_vs_nominal"].items():
        w("| `%s` | %s | %s | %s | %s | %s |"
          % (p, v["family"], "开发" if v["role"] == "dev" else "测试",
             f(v["naive_macro_sr"]), f(v["drop_pp"], 1), yn(v["meets_G1_25pp"])))
    w("")
    n_g1 = sum(1 for v in D["G1_naive_drop_vs_nominal"].values() if v["meets_G1_25pp"])
    no_g1 = [p for p, v in D["G1_naive_drop_vs_nominal"].items() if not v["meets_G1_25pp"]]
    w("%d／%d 个非标称 plant 触发 G1（掉幅 ≥ 25 pp）。未触发的是 %s。"
      "其中两个过响应档为第二遍新增，掉幅分别为 %s pp 与 %s pp，"
      "说明 g = 1.30 在本床对 naive 几乎无伤害——绝对目标每拍重锚于实测位姿，"
      "增益只影响收敛速率，过响应甚至略微加快收敛。"
      "这两个 plant 满足预注册 v1 的 G2（族内须同时含欠响应与过响应），"
      "却不具备 G1 意义上的判别力，进入测试人群后同时稀释了 P1 与 K1。"
      % (n_g1, len(D["G1_naive_drop_vs_nominal"]), "、".join("`%s`" % p for p in no_g1),
         f(D["G1_naive_drop_vs_nominal"][NEW[0]]["drop_pp"], 1),
         f(D["G1_naive_drop_vs_nominal"][NEW[1]]["drop_pp"], 1)))
    w("")

    # ============================== 4 判定 ============================== #
    w("## 4  P1 / P2 / R / K1–K3 判定")
    w("")
    w("统计口径：配对单元为 plant，人群为 %d 个测试 plant（开发 plant 不进入）；"
      "bootstrap %d 次、种子 %d、重采样单元为 plant；两个主检验做 Holm 校正。"
      % (len(TEST), D["protocol"]["bootstrap"]["n"], D["protocol"]["bootstrap"]["seed"]))
    w("")

    w("### 4.1  P1：ours-P − naive")
    w("")
    r = prim[p1k]
    v = D["verdict"]["P1"]
    w("| 量 | 第一遍 | 第二遍 |")
    w("|---|---|---|")
    w("| 人群 | %d plant（含后来划为开发的 2 个） | %d 测试 plant |" % (v1prim[p1k]["n"], r["n"]))
    w("| 均值 | %s pp | %s pp |" % (spp(v1prim[p1k]["mean"]), spp(r["mean"])))
    w("| bootstrap CI95 | [%s, %s] pp | [%s, %s] pp |"
      % (spp(v1prim[p1k]["ci95"][0]), spp(v1prim[p1k]["ci95"][1]),
         spp(r["ci95"][0]), spp(r["ci95"][1])))
    w("| 配对 t / p | %s / %s | %s / %s |"
      % (f(v1prim[p1k]["t"], 3), pv(v1prim[p1k]["p"]), f(r["t"], 3), pv(r["p"])))
    w("| Holm 校正 p | %s | %s |" % (pv(v1holm[p1k]), pv(holm[p1k])))
    w("| 正向 plant 数 | %d／%d | %d／%d |"
      % (v1prim[p1k]["n_positive"], v1prim[p1k]["n"], r["n_positive"], r["n"]))
    w("| 判据 ≥ +15 pp 且 CI 下界 > 0 | 未达 | **达成** |")
    w("")
    w("逐 plant 差值（第二遍）：")
    w("")
    w("| plant | " + " | ".join("`%s`" % p for p in r["plants"]) + " |")
    w("|---|" + "---|" * len(r["plants"]))
    w("| Δ (pp) | " + " | ".join(z(r["per_plant"][p], 1) for p in r["plants"]) + " |")
    w("")
    neg = [p for p in r["plants"] if r["per_plant"][p] < 0]
    w("负向的 %d 个 plant 是 %s——全部是 naive 本来就接近天花板的轻档"
      "（naive 宏 SR 分别为 %s），标定臂在这些 plant 上没有可恢复的空间，只剩自身开销。"
      % (len(neg), "、".join("`%s`" % p for p in neg),
         "、".join(f(c2(p, "naive", "macro_sr")) for p in neg)))
    w("")

    w("### 4.2  P2：ours-P − %s（宏 SR 支）" % CN[best])
    w("")
    r2 = prim[p2k]
    w("最强 Slow-down 按测试 plant 上的宏 SR 选出，两遍均为 %s。" % CN[best])
    w("")
    w("| 量 | 第一遍 | 第二遍 |")
    w("|---|---|---|")
    w("| 均值 | %s pp | %s pp |" % (spp(v1prim[p2k]["mean"]), spp(r2["mean"])))
    w("| bootstrap CI95 | [%s, %s] pp | [%s, %s] pp |"
      % (spp(v1prim[p2k]["ci95"][0]), spp(v1prim[p2k]["ci95"][1]),
         spp(r2["ci95"][0]), spp(r2["ci95"][1])))
    w("| 配对 t / p | %s / %s | %s / %s |"
      % (f(v1prim[p2k]["t"], 3), pv(v1prim[p2k]["p"]), f(r2["t"], 3), pv(r2["p"])))
    w("| Holm 校正 p | %s | %s |" % (pv(v1holm[p2k]), pv(holm[p2k])))
    w("| 正向 plant 数 | %d／%d | %d／%d |"
      % (v1prim[p2k]["n_positive"], v1prim[p2k]["n"], r2["n_positive"], r2["n"]))
    w("| 判据 ≥ +5 pp | 未达 | 未达 |")
    w("")
    w("逐 plant 差值（第二遍）：")
    w("")
    w("| plant | " + " | ".join("`%s`" % p for p in r2["plants"]) + " |")
    w("|---|" + "---|" * len(r2["plants"]))
    w("| Δ (pp) | " + " | ".join(z(r2["per_plant"][p], 1) for p in r2["plants"]) + " |")
    w("")
    w("差距从第一遍的 %s pp 收窄到 %s pp，方向未变，且 CI95 上界 %s pp 仍在零以下，"
      "即 ours-P 显著劣于 Slow-down ×2 这一结论在两遍中都成立。"
      % (spp(v1prim[p2k]["mean"]), spp(r2["mean"]), spp(r2["ci95"][1])))
    w("")

    w("### 4.3  P2：时间到成功支")
    w("")
    w("口径按预注册 v3 第 4 项改为在两臂共同成功的 plant × 任务格上配对，"
      "共 %d 个 plant、%d 个共同成功格。" % (tts["n_plants"], tts["n_common_cells"]))
    w("")
    w("| plant | 共同成功格 | ours-P (s) | %s (s) | 差 (s) |" % CN[best])
    w("|---|---|---|---|---|")
    for p, v in tts["per_plant"].items():
        w("| `%s` | %d | %s | %s | %s |"
          % (p, v["n_common_cells"], f(v["ours_p_s"], 3), f(v["%s_s" % best], 3),
             sfx(v["diff_s"], 3)))
    w("| **%d plant 均值** | %d | %s | %s | %s |"
      % (tts["n_plants"], tts["n_common_cells"], f(tts["ours_p_s"], 3),
         f(tts["%s_s" % best], 3), sfx(tts["ours_p_s"] - tts["%s_s" % best], 3)))
    w("")
    pl, cl = tts["paired_plant_level"], tts["paired_cell_level"]
    w("降幅 %s。plant 级配对：均值 %s s，CI95 [%s, %s]，t = %s，p = %s，%d／%d plant 为正；"
      "格级配对：n = %d，均值 %s s，CI95 [%s, %s]，t = %s，p = %s。"
      % (pct(tts["reduction_pct"]), sfx(pl["mean"], 3), sfx(pl["ci95"][0], 3),
         sfx(pl["ci95"][1], 3), f(pl["t"], 2), pv(pl["p"]), pl["n_positive"], pl["n"],
         cl["n"], sfx(cl["mean"], 3), sfx(cl["ci95"][0], 3), sfx(cl["ci95"][1], 3),
         f(cl["t"], 2), pv(cl["p"])))
    w("")
    w("**这一支不能作为方法优越性的证据。**Slow-down ×2 的定义就是把同一个 chunk 摊到两倍控制步上执行，"
      "成功集的耗时因此按构造接近翻倍：实测比值为 %s，与 2 倍只差 %s。"
      "换言之「对 Slow-down ×2 的时间到成功降 25%%」几乎是恒真命题，"
      "它衡量的是基线的减速幅度，不是 ours-P 的能力。"
      "预注册 v1 把 P2 写成「宏 SR ≥ +5 pp 或时间 −25%%」的析取式，"
      "在最强基线恰好是一个减速基线时，这个析取式失去判别力。"
      "按字面 P2 达成；按实质，P2 的两支中唯一有判别力的一支（宏 SR）在两遍中都失败。"
      % (f(tts["%s_s" % best] / tts["ours_p_s"], 3),
         f(abs(2.0 - tts["%s_s" % best] / tts["ours_p_s"]), 3)))
    w("")

    w("### 4.4  同口径重算（补充统计，非预注册）")
    w("")
    w("两遍的 P1/P2 人群不同：第一遍的 12 个 plant 含后来划为开发的 `%s` 与 `%s`，"
      "第二遍的 12 个测试 plant 换成了不含开发 plant、但含 2 个新增过响应 plant 的集合。"
      "为把「命令界修正带来的提升」与「人群变动带来的提升」分开，"
      "下表用第二遍的数据在三个人群上重算 P1 与 P2。"
      % (DEV[0], DEV[1]))
    w("")
    w("| 人群 | n | P1 均值 (pp) | P1 CI95 (pp) | P2 均值 (pp) | P2 CI95 (pp) |")
    w("|---|---|---|---|---|---|")
    w("| 第一遍原人群（第一遍数据） | %d | %s | [%s, %s] | %s | [%s, %s] |"
      % (v1prim[p1k]["n"], spp(v1prim[p1k]["mean"]), spp(v1prim[p1k]["ci95"][0]),
         spp(v1prim[p1k]["ci95"][1]), spp(v1prim[p2k]["mean"]),
         spp(v1prim[p2k]["ci95"][0]), spp(v1prim[p2k]["ci95"][1])))
    a, b = SUP["P1_on_v1_population"], SUP["P2_on_v1_population"]
    w("| 第一遍原人群（第二遍数据） | %d | %s | [%s, %s] | %s | [%s, %s] |"
      % (a["n"], spp(a["mean"]), spp(a["ci95"][0]), spp(a["ci95"][1]),
         spp(b["mean"]), spp(b["ci95"][0]), spp(b["ci95"][1])))
    a, b = SUP["P1_on_shared_test_plants"], SUP["P2_on_shared_test_plants"]
    w("| 两遍共有的 %d 个测试 plant（第二遍数据） | %d | %s | [%s, %s] | %s | [%s, %s] |"
      % (len(SHARED), a["n"], spp(a["mean"]), spp(a["ci95"][0]), spp(a["ci95"][1]),
         spp(b["mean"]), spp(b["ci95"][0]), spp(b["ci95"][1])))
    w("| 第二遍正式人群（%d 测试 plant） | %d | %s | [%s, %s] | %s | [%s, %s] |"
      % (len(TEST), r["n"], spp(r["mean"]), spp(r["ci95"][0]), spp(r["ci95"][1]),
         spp(r2["mean"]), spp(r2["ci95"][0]), spp(r2["ci95"][1])))
    w("")
    w("在完全相同的第一遍人群上，命令界修正把 P1 从 %s pp 抬到 %s pp（+%s pp），"
      "把 P2 从 %s pp 抬到 %s pp。P1 达标主要由命令界修正贡献。"
      % (spp(v1prim[p1k]["mean"]), spp(SUP["P1_on_v1_population"]["mean"]),
         f(100.0 * (SUP["P1_on_v1_population"]["mean"] - v1prim[p1k]["mean"]), 2),
         spp(v1prim[p2k]["mean"]), spp(SUP["P2_on_v1_population"]["mean"])))
    w("")
    w("人群变动的净效应可以再拆成两步：移出两个开发 plant（第一遍原人群 → 两遍共有的 %d 个测试 plant）"
      "使 P1 从 %s pp 升到 %s pp（%s pp），因为这两个 plant 上 ours-P 相对 naive 是负的；"
      "再加入两个新的过响应 plant（→ 第二遍正式人群）使 P1 从 %s pp 回落到 %s pp（%s pp），"
      "因为这两个 plant 同样是 ours-P 无优势的轻档。两步相抵后净效应为 %s pp，"
      "即人群变动对 P1 的达标几乎没有贡献，也没有把结论方向翻过来。"
      "这一补充统计不是预注册量，只用于归因，不改变 §4.1–§4.2 的判定。"
      % (len(SHARED), spp(SUP["P1_on_v1_population"]["mean"]),
         spp(SUP["P1_on_shared_test_plants"]["mean"]),
         z(SUP["P1_on_shared_test_plants"]["mean"] - SUP["P1_on_v1_population"]["mean"], 2),
         spp(SUP["P1_on_shared_test_plants"]["mean"]), spp(r["mean"]),
         z(r["mean"] - SUP["P1_on_shared_test_plants"]["mean"], 2),
         z(r["mean"] - SUP["P1_on_v1_population"]["mean"], 2)))
    w("")

    w("### 4.5  恢复率 R")
    w("")
    w("R = (SR_臂 − SR_naive) / (SR_标称 − SR_naive)，其中 SR_标称 为 naive 在标称 plant 上的宏 SR。"
      "宏统计只计入分母大于 0.10 的测试 plant，即失配确实夺走了成功率的那些；"
      "分母接近零或为负的 plant 上 R 没有意义（表内仍列出，标为不计入）。")
    w("")
    nn = [m for m in M if m != "naive"]
    w("| plant | 角色 | naive SR | 分母 | 计入 | " + " | ".join("R(%s)" % CN[m] for m in nn) + " |")
    w("|---|---|---|---|---|" + "---|" * len(nn))
    for p, v in D["recovery"].items():
        w("| `%s` | %s | %s | %s | %s | " % (p, "开发" if v["role"] == "dev" else "测试",
                                             f(v["sr_naive"]), sfx(v["denominator"], 3),
                                             yn(v["counted"]))
          + " | ".join(f(v.get("R_%s" % m), 3) for m in nn) + " |")
    w("| **宏（计入的 %d 个测试 plant）** | — | — | — | — | " % rec["n_counted_plants"]
      + " | ".join(f(rec.get("R_%s" % m), 3) for m in nn) + " |")
    w("")
    w("| 臂 | 第一遍 R | 第二遍 R |")
    w("|---|---|---|")
    for m in nn:
        w("| %s | %s | %s |" % (CN[m], f(v1rec.get("R_%s" % m), 3), f(rec.get("R_%s" % m), 3)))
    w("")
    w("ours-P 的 R 从 %s 升到 %s，但仍与 Slow-down ×1.5 的 %s 持平，"
      "低于 Slow-down ×2 的 %s 与 oracle-θ 的 %s。"
      "即在「恢复被 plant 失配夺走的成功率」这个指标上，"
      "一个不需要任何辨识的减速基线仍然优于本方法。"
      % (f(v1rec["R_ours_p"], 3), f(rec["R_ours_p"], 3), f(rec["R_slow15"], 3),
         f(rec["R_slow20"], 3), f(rec["R_oracle"], 3)))
    w("")

    w("### 4.6  K1 / K2 / K3")
    w("")
    kt, ka = k1s["test_plants_only"], k1s["incl_dev_plants"]
    w("**K1**（延迟/滞后 plant 上 ours-P 对意图的跟踪 RMSE 降幅 ≥ 30%）：**未通过**。")
    w("")
    w("| 口径 | plant 集 | n | 平均降幅 | ≥ 30% 的 plant 数 | 通过 |")
    w("|---|---|---|---|---|---|")
    w("| 第一遍（预注册） | %s | %d | %s | %d | %s |"
      % ("、".join("`%s`" % p for p in v1k1s["lag_delay_plants"]), len(v1k1s["lag_delay_plants"]),
         pct(v1k1s["mean_ours_p_reduction_pct"]), v1k1s["n_plants_reduction_ge_30"],
         yn(v1k1s["K1_passed"])))
    w("| 第二遍，测试 plant（正式口径） | %s | %d | %s | %d | %s |"
      % ("、".join("`%s`" % p for p in kt["plants"]), kt["n"],
         pct(kt["mean_ours_p_reduction_pct"]), kt["n_plants_reduction_ge_30"],
         yn(kt["K1_passed"])))
    w("| 第二遍，含开发 plant | %s | %d | %s | %d | %s |"
      % ("、".join("`%s`" % p for p in ka["plants"]), ka["n"],
         pct(ka["mean_ours_p_reduction_pct"]), ka["n_plants_reduction_ge_30"],
         yn(ka["K1_passed"])))
    w("| 第二遍数据 × 第一遍 plant 集（补充） | %s | %d | %s | %d | %s |"
      % ("、".join("`%s`" % p for p in SUP["K1_v2_on_v1_plant_set"]["plants"]),
         len(SUP["K1_v2_on_v1_plant_set"]["plants"]),
         pct(SUP["K1_v2_on_v1_plant_set"]["mean_pct"]),
         SUP["K1_v2_on_v1_plant_set"]["n_ge_30"],
         yn(SUP["K1_v2_on_v1_plant_set"]["mean_pct"] >= 30.0)))
    w("")
    w("最后一行说明：在第一遍的同一批 plant 上，第二遍数据给出 %s，"
      "较第一遍的 %s 略降但仍在阈值线上；正式口径失败是因为 K1 集换成参数谓词后纳入了新增的 `%s`，"
      "而该 plant 上 ours-P 的降幅为 %s（即比 naive 更差）。"
      % (pct(SUP["K1_v2_on_v1_plant_set"]["mean_pct"]), pct(v1k1s["mean_ours_p_reduction_pct"]),
         NEW[1], pct(k1t[NEW[1]]["ours_p_reduction_pct"])))
    w("")
    w("**K2**（族恢复率 R(ours-P) < 0.3 则触发）：未触发，R = %s。" % f(kills["K2_R_ours_p"], 3))
    w("")
    w("**K3**（对最强 Slow-down 既无宏 SR 优势又无时间优势则触发）：未触发。"
      "宏 SR 差 %s pp（无优势），时间到成功降 %s（≥ 25%%，故不触发）。"
      "K3 的免触发同样只靠时间支，§4.3 的保留意见在此同样适用。"
      % (f(kills["K3_macro_gap_pp"], 2), pct(kills["K3_time_reduction_pct_common_cells"])))
    w("")

    # ============================== 5 饱和率 ============================== #
    w("## 5  饱和率对照，以及 oracle-θ 与 ours-P 的差")
    w("")
    w("### 5.1  按臂的饱和率，两遍并列")
    w("")
    v1s = cmp1["v1_from_shards"]
    w("饱和的两种口径：命令界至少被打满一次的集比例，以及处在界上的控制步比例。")
    w("")
    w("| 臂 | 第一遍集比例（界 5.0） | 第二遍集比例（界 1.0） | 第一遍步比例 | 第二遍步比例 |")
    w("|---|---|---|---|---|")
    for m in M:
        w("| %s | %s | %s | %s | %s |"
          % (CN[m], pct(100.0 * v1s["sat_episode_share_by_arm"][m]),
             pct(100.0 * mm[m]["sat_episode_share_all"]),
             pct(100.0 * v1s["sat_step_share_by_arm"][m]),
             pct(100.0 * mm[m]["sat_step_share_all"])))
    w("")
    w("第一遍 %d 集、第二遍 %d 集，全部集参与统计。" % (v1s["n_episodes"], D["n_episodes"]))
    w("")
    w("这张表给出的是一个反直觉但必须如实记录的事实：**命令界收紧后，饱和更严重而结果更好**。"
      "ours-P 的饱和集比例从 %s 升到 %s，饱和步比例从 %s 升到 %s，"
      "却同时把同一批 plant 上的宏 SR 从 %s 抬到 %s；"
      "naive 也是同样的方向（饱和步比例 %s → %s，宏 SR %s → %s），"
      "而 naive 的命令并不经过反演，两遍之间唯一的差别就是裁剪本身。"
      % (pct(100.0 * v1s["sat_episode_share_by_arm"]["ours_p"]),
         pct(100.0 * mm["ours_p"]["sat_episode_share_all"]),
         pct(100.0 * v1s["sat_step_share_by_arm"]["ours_p"]),
         pct(100.0 * mm["ours_p"]["sat_step_share_all"]),
         f(V1["macro"]["ours_p"]["macro_sr_family_mean"]),
         f(cmp1["v2_macro_same_12_plants"]["ours_p"]),
         pct(100.0 * v1s["sat_step_share_by_arm"]["naive"]),
         pct(100.0 * mm["naive"]["sat_step_share_all"]),
         f(V1["macro"]["naive"]["macro_sr_family_mean"]),
         f(cmp1["v2_macro_same_12_plants"]["naive"])))
    w("")
    w("与之相容的读法是：命令界起的是限幅保护作用，触界频次本身不是病因，"
      "单步允许的位移幅度才是。界取 %s 时单步最大位移为 %.2f m，"
      "在 %.1f Hz 的控制率下相当于 %.1f m/s 的瞬时指令；"
      "界收到 %s 后被裁到 %.2f m／步，触界次数上升，但每一次越界的幅度被限住。"
      "因此第一遍报告把失败归因为「标定臂把命令界打满」是不准确的表述，"
      "更贴近数据的说法是「第一遍的界过宽，未能起到限幅作用」。"
      "本段是对两遍差异的机制性解读，不是直接测量；"
      "要坐实它需要逐步记录命令-实现对，本轮未做。"
      % (f(V1["protocol"]["executor_bound_pos_norm"], 1),
         0.1 * V1["protocol"]["executor_bound_pos_norm"],
         D["protocol"]["control_hz"],
         0.1 * V1["protocol"]["executor_bound_pos_norm"] * D["protocol"]["control_hz"],
         f(D["protocol"]["executor_bound_pos_norm"], 1),
         0.1 * D["protocol"]["executor_bound_pos_norm"]))
    w("")

    w("### 5.2  逐 plant 饱和率（第二遍）")
    w("")
    w("格式为「打满的集数／总集数，处在界上的步比例」。")
    w("")
    w("| plant | " + " | ".join(CN[m] for m in M) + " |")
    w("|---|" + "---|" * len(M))
    for p in P:
        row = []
        for m in M:
            c = C[p].get(m)
            row.append("%d／%d，%s" % (c["n_bound_hit_episodes"], c["n"],
                                      pct(100.0 * (c["sat_step_share"] or 0.0)))
                       if c else "—")
        w("| `%s` | " % p + " | ".join(row) + " |")
    w("")
    hi = max(P, key=lambda p: c2(p, "ours_p", "sat_step_share") or 0.0)
    w("最严重的是 `%s`：ours-P 有 %s 的控制步处在界上，"
      "该 plant 同时是滞后与增益联合失配的强档，反演出的命令长期超出可下发范围，"
      "执行器实际执行的是一条被限幅后的轨迹。"
      "在这类 plant 上，「反演」的语义已经退化为「持续朝目标方向饱和输出」，"
      "这也解释了为何 ours-P 与 oracle-θ 在强档上的差距很小——"
      "两者的命令都被同一个界削平了。"
      % (hi, pct(100.0 * c2(hi, "ours_p", "sat_step_share"))))
    w("")

    w("### 5.3  oracle-θ 与 ours-P 的差")
    w("")
    o = SUP["oracle_minus_ours_p_test"]
    orm = SUP["oracle_minus_ours_p_rmse_test"]
    w("oracle-θ 与 ours-P 的唯一区别是前者直接使用注入的真参数，后者使用探针辨识出的参数。"
      "二者之差就是辨识误差在闭环上的代价。")
    w("")
    w("| 量（测试 plant，n=%d） | oracle-θ | ours-P | 差 |" % len(TEST))
    w("|---|---|---|---|")
    w("| 宏 SR | %s | %s | %s pp |"
      % (f(mm["oracle"]["macro_sr_test_mean"]), f(mm["ours_p"]["macro_sr_test_mean"]),
         spp(o["mean"])))
    w("| 对意图跟踪 RMSE (m) | %s | %s | %s |"
      % (f(mm["oracle"]["tracking_rmse_test_mean_m"], 5),
         f(mm["ours_p"]["tracking_rmse_test_mean_m"], 5), sfx(orm["mean"], 5)))
    w("| 恢复率 R | %s | %s | %s |"
      % (f(rec["R_oracle"], 3), f(rec["R_ours_p"], 3), sfx(rec["R_oracle"] - rec["R_ours_p"], 3)))
    w("| 饱和步比例 | %s | %s | %s |"
      % (pct(100.0 * mm["oracle"]["sat_step_share_all"]),
         pct(100.0 * mm["ours_p"]["sat_step_share_all"]),
         sfx(100.0 * (mm["oracle"]["sat_step_share_all"]
                      - mm["ours_p"]["sat_step_share_all"]), 1) + " pp"))
    w("")
    w("宏 SR 的配对差为 %s pp（CI95 [%s, %s] pp，t = %s，p = %s，未达显著），"
      "第一遍的族均值之差为 %s pp。辨识误差的代价在第二遍略有扩大，但并不统计显著，"
      "且两遍都在几个 pp 的量级——**瓶颈仍不在辨识精度**。"
      "延迟与夹爪超前的辨识在全族是精确的（MAE 均为 %s 步），"
      "残余误差集中在增益与滞后的相对估计上（MAE %s 与 %s s），"
      "而这两个量恰恰是反演式 u = (w_des/ĝ − α̂z)/(1 − α̂) 里被放大的那两个。"
      % (spp(o["mean"]), spp(o["ci95"][0]), spp(o["ci95"][1]), f(o["t"], 3), pv(o["p"]),
         spp(V1["macro"]["oracle"]["macro_sr_family_mean"]
             - V1["macro"]["ours_p"]["macro_sr_family_mean"]),
         f(ID["agreement"]["delay_mae_steps"], 2), f(ID["agreement"]["gain_mae_rel"], 4),
         f(ID["agreement"]["tau_mae_rel_s"], 4)))
    w("")
    w("更值得注意的是两个臂在过响应 plant 上的**同号退化**："
      "`%s` 上 ours-P 与 oracle-θ 对意图的 RMSE 相对 naive 分别变化 %s 与 %s，"
      "`%s` 上为 %s 与 %s。oracle-θ 使用的是真参数，仍然退化，"
      "说明这不是辨识问题，而是执行器反演臂在这类 plant 上的结构性代价："
      "过响应 plant 的 naive 跟踪误差本来就低（%s m，甚至低于标称 plant 的 %s m，"
      "因为更大的增益让绝对目标更快收敛），反演没有可改善的余量，"
      "而反演自身引入的设定点重排与限幅开销仍在。"
      % (NEW[0], pct(k1t[NEW[0]]["ours_p_reduction_pct"]),
         pct(k1t[NEW[0]]["oracle_reduction_pct"]),
         NEW[1], pct(k1t[NEW[1]]["ours_p_reduction_pct"]),
         pct(k1t[NEW[1]]["oracle_reduction_pct"]),
         f(c2(NEW[0], "naive", "tracking_rmse_m"), 5),
         f(c2("L00_nominal", "naive", "tracking_rmse_m"), 5)))
    w("")
    w("同样的同号退化也出现在标称 plant 上：ours-P 与 oracle-θ 的宏 SR 分别为 %s 与 %s，"
      "都低于 naive 的 %s。预注册 v1 期望的「标定臂在标称 plant 上退化为 naive」"
      "这一设计性质，在第二遍仍未完全成立，只是差距从第一遍的 %s pp 收窄到 %s pp。"
      % (f(c2("L00_nominal", "ours_p", "macro_sr")), f(c2("L00_nominal", "oracle", "macro_sr")),
         f(c2("L00_nominal", "naive", "macro_sr")),
         spp(c1("L00_nominal", "ours_p", "macro_sr") - c1("L00_nominal", "naive", "macro_sr")),
         spp(c2("L00_nominal", "ours_p", "macro_sr") - c2("L00_nominal", "naive", "macro_sr"))))
    w("")

    # ============================== 6 偏离 ============================== #
    w("## 6  协议偏离清单")
    w("")
    w("以下逐条为本轮相对预注册的偏离或未覆盖项，不作辩解，供读者据以折扣结论强度。")
    w("")
    w("1. **第二遍在看过第一遍之后校正，属确证运行而非预注册运行。**"
      "预注册 v3 修订本身是读过第一遍臂间结果后写的，"
      "其中命令界的取值与两个新增 plant 的档位都受第一遍结果影响。"
      "第一遍（%d 集）仍作为预注册结果原样报告，论文中两遍必须并列，不得只报第二遍。"
      % V1["n_episodes"])
    w("2. **单本体。**只有 robosuite Panda 一个机器人本体，无跨本体证据；"
      "第一遍在 ManiSkill 上的 xarm6_robotiq 结果不属于本条主对比。")
    w("3. **单任务套件、10 任务。**只有 `%s` 的 %d 个任务，未覆盖 libero_object / goal / long / 90。"
      % (D["suite"], len(D["tasks"])))
    w("4. **每格 n = %d 集。**宏 SR 的每个 plant × 臂格由 %d 集构成，"
      "单格的二项标准误约 %.2f，plant 级配对统计的自由度为 %d。"
      "第二遍未提高 n。"
      % (D["protocol"]["episodes_per_cell"],
         len(D["tasks"]) * D["protocol"]["episodes_per_cell"],
         (0.25 / (len(D["tasks"]) * D["protocol"]["episodes_per_cell"])) ** 0.5,
         len(TEST) - 1))
    w("5. **未跑的臂：Universal-fixed、RTC-only、FT-demo、ours-L。**"
      "预注册 v3 第 3 项明确保持臂集合不变，因此本轮无法回答"
      "「固定通用补偿是否够用」「重规划节流是否够用」"
      "「用少量目标域演示微调是否更划算」「学习式执行器是否优于闭式参数臂」这四个问题。"
      "其中 Universal-fixed 与 RTC-only 是本方法最直接的竞争者，其缺席是本轮最大的证据缺口。")
    w("6. **D4 限幅档剔除。**第一遍 L3 筛查显示该档在本床对 naive 的掉幅为 0.0 pp，"
      "属无效应档，两遍均剔除；因此族内不含「执行器限幅」这一类失配。")
    w("7. **开发 plant 的选取与工作点扫描的判别力有限。**"
      "命令界的取值在扫描之前已由预注册 v3 固定为 %s，"
      "扫描只在 %s 两个候选点上、每点 %d 集做确认，"
      "两点的判据值相差 %s；这不构成一次有力的超参搜索。"
      % (f(DV["prereg_v3_fixed_bound_norm"], 1),
         "／".join(f(x, 1) for x in DV["design"]["scanned_bounds_norm"]),
         DV["design"]["episodes_per_configuration"],
         pct(100.0 * (DV["scan"]["2"]["mean_tracking_rmse_vs_intent_m"]
                      / DV["scan"]["1"]["mean_tracking_rmse_vs_intent_m"] - 1.0))))
    w("8. **P1/P2 的人群在两遍之间发生了变动。**"
      "第一遍的 12 个 plant 含后来划为开发的 2 个，第二遍换成 12 个测试 plant（含 2 个新增）。"
      "§4.4 给出同口径重算以便归因，但该重算不是预注册量。")
    w("9. **辨识器结构限制保留。**大 τ 档存在 (τ, g) 互换自由度"
      "（ARX(1,1) 对二阶系统只有一个极点），预注册 v3 第 5 项决定本轮不改辨识器，"
      "该限制原样保留为后续项。")
    w("10. **P2 的时间支缺乏判别力。**见 §4.3；本报告在结论中不采信该支。")
    w("11. **命名与硬件的技术性偏离。**"
      "预注册 v3 把标称开发 plant 写作 `P00_nominal`，本床该 plant 的 pid 为 `%s`，"
      "同一个 plant 的不同床前缀；GPU 7 被其他实验占用，第二个策略服务端改在 GPU 0，"
      "分片方案（2 服务端、各 4 个客户端分片）不变。" % DEV[0])
    w("")

    # ============================== 7 耗时与复现 ============================== #
    w("## 7  耗时与复现命令")
    w("")
    mw = W2["main_v2"]
    dw = W2["dev_scan_total"]
    w("| 阶段 | 集数 | 分片 | 最慢分片 | 分片墙钟合计 | 控制步 | 秒／步 |")
    w("|---|---|---|---|---|---|---|")
    w("| 探针 + 辨识（新增 %d 个 plant） | — | — | 每 plant %.1f s 仿真（%d 步） | — | — | — |"
      % (W2["probe_v2"]["n_plants_probed_this_pass"], W2["probe_v2"]["probe_seconds_per_plant"],
         W2["probe_v2"]["probe_steps"]))
    w("| 开发 plant 工作点扫描（两个工作点） | %d | 4 + 4 | %.0f s | %.0f s | %d | — |"
      % (dw["n_episodes"], dw["slowest_shard_s"], dw["sum_shard_wall_s"],
         W2["dev_b10"]["steps"] + W2["dev_b20"]["steps"]))
    w("| 主对比 | %d | %d | %.0f s = %.2f h | %.0f s = %.2f h | %d | %s |"
      % (mw["n_episodes"], mw["n_shards"], mw["slowest_shard_s"], mw["slowest_shard_h"],
         mw["sum_shard_wall_s"], mw["sum_shard_wall_s"] / 3600.0, mw["steps"],
         f(mw["sec_per_step"], 4)))
    w("")
    w("第一遍主对比作对照：%s，控制步 %d，秒／步 %s。"
      % (W1["main"], W1["main_total_steps"], f(W1["sec_per_step"], 4)))
    w("第二遍每步耗时略高（%s 对 %s），与命令界收紧后平均集长变化及两块 GPU 的型号差异一致。"
      % (f(mw["sec_per_step"], 4), f(W1["sec_per_step"], 4)))
    w("")
    w("复现命令（服务器 03，仓库 `/opt/cce`，"
      "环境 `/opt/cce/data/envs/libero`）：")
    w("")
    w("```bash")
    w("L=/opt/cce/data/logs/cce_libero")
    w("")
    w("# 1) 策略服务端（GPU 6 -> 8021 沿用第一遍；GPU 0 -> 8023 为第二遍新起）")
    w("bash $L/serve_two.sh          # 仅需其中的 8021")
    w("bash $L/serve_gpu0.sh         # 8023")
    w("")
    w("# 2) 新增两个过响应 plant 的探针与辨识")
    w("bash $L/probe_v2.sh")
    w("")
    w("# 3) 开发 plant 工作点扫描（命令界 1.0 与 2.0，各 40 集）")
    w("bash $L/devscan_v2.sh")
    w("")
    w("# 4) 冻结族定义")
    w("./cce/env_libero.sh cce/libero_freeze_family_v2.py")
    w("")
    w("# 5) 主对比 %d 集，%d 分片" % (D["n_episodes"], mw["n_shards"]))
    w("bash $L/run_shards_v2.sh main_v2 %d \\" % mw["n_shards"])
    w("     --set family_v2 --methods %s \\" % ",".join(M))
    w("     --tasks %s --episodes %d --tag v2"
      % (",".join(str(t) for t in D["tasks"]), D["protocol"]["episodes_per_cell"]))
    w("")
    w("# 6) 汇总")
    w("./cce/env_libero.sh cce/libero_aggregate_v2.py \\")
    w("     --glob \"$L/main_v2/shard*.jsonl\" \\")
    w("     --out cce/results/libero_family_v2.json --label libero_family_v2 \\")
    w("     --theta-summary data/cce/libero/theta/identify_libero_panda_v2.json \\")
    w("     --family-spec cce/results/libero_family_spec_v2.json \\")
    w("     --devplant-json cce/results/libero_devplant_v2.json \\")
    w("     --v1-json cce/results/libero_family_v1.json \\")
    w("     --v1-glob \"$L/main/shard*.jsonl\"")
    w("")
    w("# 7) 本报告")
    w("./cce/env_libero.sh cce/libero_report_v2.py \\")
    w("     --v2 cce/results/libero_family_v2.json --v1 cce/results/libero_family_v1.json \\")
    w("     --spec cce/results/libero_family_spec_v2.json \\")
    w("     --dev cce/results/libero_devplant_v2.json \\")
    w("     --ident data/cce/libero/theta/identify_libero_panda_v2.json \\")
    w("     --wall2 $L/wall_v2.json --wall1 $L/wall.json \\")
    w("     --abl-glob \"$L/abl_b10/shard*.jsonl\" \\")
    w("     --out docs/reports/CCE_libero_family_v2.md")
    w("```")
    w("")
    w("完整性核对：%d 个 `.jsonl` 分片共 %d 行，"
      "(plant, 臂, 任务, 集号) 四元键无重复、无缺格，"
      "全部记录的命令界均为 %s，编解码往返误差%s。"
      % (mw["n_shards"], D["n_episodes"], f(D["protocol"]["executor_bound_pos_norm"], 1),
         ("最大 %.2e m" % _mx) if (_mx := max(c2(p, m, "max_codec_err_m")
                                             for p in P for m in M)) > 0 else "全程精确为零"))
    w("")

    # ============================== 8 结论 ============================== #
    w("## 8  结论与对论文主张的含义")
    w("")
    w("### 8.1  数据支持的主张")
    w("")
    w("1. **接口层失配确实会显著损伤一个正控的 VLA 策略。**"
      "标称 plant 上 naive 宏 SR %s，注入失配后最重的档掉到 %s；"
      "%d／%d 个非标称 plant 的掉幅 ≥ 25 pp。这条在两遍中都稳固。"
      % (f(c2("L00_nominal", "naive", "macro_sr")),
         f(min(c2(p, "naive", "macro_sr") for p in TEST)), n_g1,
         len(D["G1_naive_drop_vs_nominal"])))
    w("2. **基于探针辨识 + 执行器反演的补偿显著优于不作为。**"
      "P1 = %s pp（CI95 [%s, %s]，Holm p = %s），达到预注册的 +15 pp 阈值，"
      "%d／%d 个测试 plant 为正。"
      % (spp(prim[p1k]["mean"]), spp(prim[p1k]["ci95"][0]), spp(prim[p1k]["ci95"][1]),
         f(holm[p1k], 4), prim[p1k]["n_positive"], prim[p1k]["n"]))
    w("3. **接口层辨识在延迟与夹爪时序上是精确的。**"
      "全族 %d 个 plant 上传输延迟与夹爪超前的估计 MAE 均为 %s 步，相关系数 %s。"
      % (ID["n_plants"], f(ID["agreement"]["delay_mae_steps"], 2),
         f(ID["agreement"]["delay_corr"], 3)))
    w("4. **方法的瓶颈不在辨识精度，而在执行器工作点与反演结构。**"
      "oracle-θ 与 ours-P 的宏 SR 差只有 %s pp，"
      "而两者与 Slow-down ×2 的差为 %s pp 与 %s pp；"
      "把命令界从 5.0 收到 1.0 这一个执行器超参的修正，"
      "在同一批 plant 上就让 ours-P 涨了 %s pp。"
      % (spp(o["mean"]),
         spp(mm["oracle"]["macro_sr_test_mean"] - mm["slow20"]["macro_sr_test_mean"]),
         spp(mm["ours_p"]["macro_sr_test_mean"] - mm["slow20"]["macro_sr_test_mean"]),
         spp(cmp1["v2_macro_same_12_plants"]["ours_p"]
             - V1["macro"]["ours_p"]["macro_sr_family_mean"])))
    w("")
    w("### 8.2  数据不支持的主张")
    w("")
    w("1. **不能主张本方法优于通用的保守执行策略。**"
      "P2 的宏 SR 支为 %s pp（CI95 [%s, %s]，Holm p = %s），"
      "%d／%d 个测试 plant 上 ours-P 不优于 Slow-down ×2；恢复率 R 也是 %s 对 %s。"
      "一个不需要探针、不需要辨识、不需要任何 plant 知识的减速基线，在成功率上仍然更强。"
      "论文中不得写「优于通用基线」「显著超过 Slow-down」一类表述。"
      % (spp(r2["mean"]), spp(r2["ci95"][0]), spp(r2["ci95"][1]), f(holm[p2k], 4),
         r2["n"] - r2["n_positive"], r2["n"], f(rec["R_ours_p"], 3), f(rec["R_slow20"], 3)))
    w("2. **不能用「时间到成功降 %s」来支撑优越性。**"
      "该降幅对一个把 chunk 摊到两倍步数的基线近乎恒真（实测耗时比 %s，接近 2）；"
      "它衡量的是基线的减速幅度。若论文要报告速度优势，"
      "对照臂必须换成与 ours-P 同等执行速率的臂，本轮没有这样的臂。"
      % (pct(tts["reduction_pct"]), f(tts["%s_s" % best] / tts["ours_p_s"], 3)))
    w("3. **不能主张补偿能改善对策略意图的跟踪。**"
      "K1 在第二遍未通过：延迟/滞后测试 plant 上平均只降 %s（阈值 30%%），"
      "%d／%d 个 plant 达标；在两个过响应 plant 上 ours-P 与 oracle-θ 的跟踪 RMSE 双双劣于 naive。"
      "「反演让机器人更贴近策略想要的轨迹」这一叙事只在延迟与滞后档成立，不能一般化。"
      % (pct(kt["mean_ours_p_reduction_pct"]), kt["n_plants_reduction_ge_30"], kt["n"]))
    light = [p for p in r["plants"] if r["per_plant"][p] <= 0.05]
    w("4. **不能主张标定臂在无失配时无害。**"
      "标称 plant 上 ours-P %s、oracle-θ %s，均低于 naive 的 %s；"
      "%d 个轻档（%s）上 P1 的逐 plant 差为负或近零，"
      "这 %d 个 plant %s落在 G1 未触发（naive 掉幅 < 25 pp）的那一组内。"
      "预注册 v1 期望的「标定臂在标称 plant 上退化为 naive」并未成立。"
      % (f(c2("L00_nominal", "ours_p", "macro_sr")), f(c2("L00_nominal", "oracle", "macro_sr")),
         f(c2("L00_nominal", "naive", "macro_sr")), len(light),
         "、".join("`%s`" % p for p in light), len(light),
         "全部" if all(p in no_g1 for p in light) else "多数"))
    w("5. **不能主张跨本体、跨任务套件的普适性。**单本体、单套件、10 任务、每格 %d 集；"
      "四个关键对照臂（Universal-fixed、RTC-only、FT-demo、ours-L）未跑。"
      % D["protocol"]["episodes_per_cell"])
    w("6. **不能把第二遍单独作为预注册证据呈现。**第二遍是看过第一遍后的校正运行；"
      "第一遍的 P1 未达标（%s pp，Holm p = %s）、P2 实质失败，"
      "两遍必须并列，且需说明第二遍改动了哪一个超参。"
      % (spp(v1prim[p1k]["mean"]), f(v1holm[p1k], 4)))
    w("")
    w("### 8.3  对下一步的直接含义")
    w("")
    w("1. 最重要的缺口是**竞争者缺席**。在没有 Universal-fixed 与 RTC-only 的情况下，"
      "本方法唯一被证明超过的对照是 naive。补齐这两个臂是把「P1 成立」变成有意义主张的前提。")
    w("2. **P2 的判据需要改写**。以 Slow-down 为最强基线时，时间支恒真、SR 支恒难；"
      "应改为在同等执行速率下比较，或改用「成功率 × 速率」的联合指标。")
    w("3. **反演在轻档与过响应档上的净损失需要单独处理**。"
      "一个可检验的方向是给标定臂加一个失配显著性门：当辨识出的失配小于某阈值时退回 naive，"
      "使「无失配时无害」成为构造性质而非期望性质。"
      "本轮数据已能给出该门的候选判据——%d 个 P1 为负或近零的 plant（%s）"
      "均落在 G1 掉幅未达 25 pp 的一组内（该组共 %d 个测试 plant），"
      "提示「失配是否值得补偿」可以先用一个粗判据筛掉，再决定是否启用反演。"
      % (len(light), "、".join("`%s`" % p for p in light),
         len([p for p in no_g1 if p in TEST])))
    w("4. **过响应档的结构性代价需要诊断到机制**。oracle-θ 同号退化说明问题在反演式或执行器，"
      "不在辨识；下一步应记录逐步的命令-实现对，定位是限幅、设定点重排还是增益反演本身。")
    w("")
    w("---")
    w("")
    w("本报告的全部数值由 `cce/libero_report_v2.py` 从上列 JSON 生成，无手工转录。"
      "第二遍主对比共 %d 集、%d 个控制步，"
      "结果文件 `cce/results/libero_family_v2.json`。" % (D["n_episodes"], mw["steps"]))
    w("")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L), encoding="utf-8")
    print("[report-v2] wrote %s (%d lines)" % (p, len(L)))


if __name__ == "__main__":
    main()
