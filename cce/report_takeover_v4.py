#!/usr/bin/env python
"""Generate the takeover report directly from audits and open-loop results."""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path
from cce.libero_plant_v3 import FAMILY_V3, DEV_PIDS, TEST_PIDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    root = Path(args.run_dir)
    read = lambda name: json.loads((root/name).read_text())
    audit = read('audit.json')
    grey14, grey19 = read('greybox.json'), read('greybox_19.json')
    ex, ident = read('executor_v4.json'), read('identify_selftest.json')
    probes = read('new_probes/probe_summary.json')
    assert audit['ok'] and ex['implementation_checks_passed'] and ident['ok'] and probes['complete']
    assert len(grey19['plants']) == len(FAMILY_V3) == 19
    assert (root/'new_probes.exit').read_text().strip() == '0'
    code = ['audit_takeover_v4.py', 'libero_identify_v4.py', 'libero_exec_v4.py',
            'libero_plant_v3.py', 'libero_probe_v3.py', 'report_takeover_v4.py']
    manifest = {f'cce/{p}': hashlib.sha256(Path('cce', p).read_bytes()).hexdigest() for p in code}
    result = {'status': 'W0_reproduction_and_W1_open_loop_development_complete',
        'formal_closedloop_started': False, 'formal_gate_ready': False, 'run_dir': str(root),
        'baseline_commit': audit['source_commit'], 'implementation_sha256': manifest,
        'reproduction': audit, 'identification_14': grey14['agreement'],
        'identification_19': grey19['agreement'], 'identification_selftest': ident,
        'executor_selftest': ex, 'new_probes': probes,
        'candidate_family': {'n': len(FAMILY_V3), 'dev': list(DEV_PIDS), 'test': list(TEST_PIDS),
                             'main_episodes': len(FAMILY_V3)*7*10*10,
                             'secondary_episodes': len(FAMILY_V3)*8*10*5},
        'identification_rows': grey19['plants']}
    Path('cce/results/takeover_v4_20260908.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# CCE 接手审计与首批 v4 开发实验', '',
        '日期：2026-09-08。状态：历史复算完成；v4开环开发阶段；正式闭环未启动，预注册未冻结。', '',
        f'历史基线提交：`{audit["source_commit"]}`。新实现文件的完整SHA-256见同名结果JSON；原始产物根目录：`{root}`。', '',
        '## 1 已有两遍独立复算', '',
        '| 遍次 | 集数/唯一键 | 缺格 | primary / Holm / macro逐数一致 |',
        '|---|---:|---:|---|']
    for ver, r in audit['passes'].items():
        lines.append(f'| {ver} | {r["n_episodes"]}/{r["n_unique_keys"]} | {r["missing_keys"]} | {all(r["exact_matches"].values())} |')
    lines += ['', '原始16份分片的SHA及完整复算命令已保存。每个四元键为plant×method×task_id×ep；全部shard的.exit为0。旧结果与旧代码未修改。', '',
        '旧LIBERO plant、executor自检退出0；ManiSkill codec数值误差按1e−7阈值验收（学生接口返回float32，不能误写成1e−9）。旧validate_codec脚本只打印指标而不assert，故本次增加独立数值断言。其原nominal identity随机输入含范数>1的旋转，产生约0.377的合法限幅差，不是标称plant错误；已在合法动作域补验。',
        f'合法域2000个动作最大偏差：{audit["checks"]["valid_domain_nominal_identity"]["max_error"]:.3g}。', '',
        '结论保持原样：第二遍P1成立，ours-P成功率仍低于slow20；原跨本体和策略侧不可补偿主张没有补出新证据。', '',
        '## 2 灰盒辨识（只有开环数据）', '',
        '| 人群 | 延迟MAE/拍 | 夹爪提前MAE/拍 | τ MAE/s | gain MAE | R²<0.7轴 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, obj in [('旧14个', grey14), ('全部候选19个', grey19)]:
        r = obj['agreement']
        lines.append(f'| {name} | {r["delay_mae_steps"]:.4f} | {r["grip_mae_steps"]:.4f} | {r["tau_mae_s"]:.6f} | {r["gain_mae"]:.6f} | {r["unidentifiable_axes"]} |')
    lines += ['', '| plant | d真/灰盒 | τ真/灰盒/s | g真/灰盒 | 夹爪提前真/估计/拍 | 最低轴R² |',
              '|---|---|---|---|---|---|']
    for pid, r in grey19['plants'].items():
        t, g = r['truth_for_report_only'], r['greybox']
        lines.append(f'| {pid} | {t["delay_steps"]:g}/{g["delay_steps"]} | {t["tau_s"]:.2f}/{g["tau_s"]:.2f} | {t["gain"]:.2f}/{g["gain"]:.4f} | {t["grip_lead_steps"]:g}/{g["grip_lead_steps"]} | {min(a["r2_freerun"] for a in g["axes"]):.4f} |')
    r = grey14['plants']['D2_t400']; a, g = r['arx_relative'], r['greybox']
    lines += ['', f'D2_t400：τ由旧ARX的{a["tau_s"]:.6f}s变为{g["tau_s"]:.2f}s，gain由{a["gain"]:.6f}变为{g["gain"]:.6f}；最低轴自由仿真R²由{min(x["r2_freerun"] for x in a["axes"]):.4f}提高到{min(x["r2_freerun"] for x in g["axes"]):.4f}。τ位于规划容差边界，gain仍未达到1±0.15。保留限制，不按真值调整格点或窗口。',
        '标称灰盒gain也不逐轴严格为1；因此“给定恒等参数”的代数性质与“实际标称探针后无害”是两项不同验收。',
        f'二阶精确模型辨识自检：{ident["cases"]}组(d,τ,g)，全部精确恢复格点；最大gain误差{ident["gain_max_abs_error"]:.3g}。这只验算法与时序索引，不代表真实OSC拟合无偏。', '',
        '## 3 v4执行器原型与反例', '',
        f'四个ρ候选下，恒等相对参数与旧naive的最大逐拍命令差{ex["nominal_max_command_error"]:.3g}，小于1e−9。测试跨三个chunk保持内部状态，并覆盖饱和。',
        f'12种二阶合成压力配置各执行600拍，均有限且命令不超界；最大位置误差{max(x["max_error_m"] for x in ex["synthetic_stability"]):.6f}m。这是有限时域合成检查，不是全局稳定性证明或ManiSkill完整回归。',
        '低R²轴使用当前naive命令；部分旋转轴回退时保持该轴值，再约束其余旋转轴，避免二次范数裁剪破坏回退。',
        f'纯夹爪20拍延迟的米制H={ex["pure_gripper_H_m"]:.1f}m；正阈值下整plant退回naive，取消本来需要的夹爪提前，反例已assert复现。当前原型忠实保留原规划门的行为并标记formal_gate_ready=false。', '',
        '## 4 新增开环实验', '',
        f'完成{len(probes["stats"])}个新增plant，各140拍/7秒，共{len(probes["stats"])*140}控制拍、{len(probes["stats"])*7}秒仿真。仅控制步墙钟之和{sum(r["wall_s"] for r in probes["stats"]):.2f}秒（不含环境加载，不能当作作业总墙钟）。policy_calls=0。',
        '使用libero_spatial task0、ep0、init_seed42，与旧探针相同；夹爪以旧标称t90作基准。输入、达成位姿、实际命令和夹爪开度均存NPZ并记SHA。渲染配置为GPU5；未加载VLA，没有训练，没有正式成功率结果。GPU·h未测量，不伪造精确开销。', '',
        '## 5 协议与实现审查', '',
        f'完整表格为{len(FAMILY_V3)}个plant，开发{len(DEV_PIDS)}、测试{len(TEST_PIDS)}；主臂{len(FAMILY_V3)*700}集、次臂{len(FAMILY_V3)*400}集。原规划18/15是计数冲突，原文件保留，候选代码与新蓝图已校正。',
        '正式冻结前还需解决：米制H遗漏夹爪；P2a非劣/等价的定义与Holm family；冻结前naive筛查与§4.0的冲突；灰盒验收/回退条件；标称估计无害；跨chunk与策略随机输入复用。细项见预注册草案首页。',
        '不把sim-only合成检查写成完整门2通过。ManiSkill26plant×双本体跟踪/回放回归、开发扫描、ρ/H0选择、正式runner与统计仍未完成。',
        '未找到本机《实现进展_真源.md》，已以当前git及原始记录接手；8021/8022/8023无策略服务，不为CPU开发空载启动模型。文献逐篇核验和投稿截止核验未做，蓝图条目标为待核实。',
        '提交脚本原先硬编码另一个模型的Co-Author，改为可选IDEA_COMMIT_COAUTHOR，避免本次提交带入错误署名。', '',
        '## 6 下一批工作与复现', '',
        '先在冻结草案中解决门的通道定义及统计歧义；保留本次字面实现与反例。再完成实际ManiSkill回归和开发扫描；只有验收齐全并冻结提交后才进入第三遍测试plant对照。策略侧与第二套件按原周计划后续推进。', '',
        '```bash',
        f'CUDA_VISIBLE_DEVICES="" ./cce/env_libero.sh cce/audit_takeover_v4.py --out-dir {root}',
        './cce/env_libero.sh cce/libero_identify_v4.py --selftest',
        './cce/env_libero.sh cce/libero_exec_v4.py',
        f'./cce/env_libero.sh cce/libero_identify_v4.py --family v3 --additional-probe-dir {root}/new_probes --additional-arx {root}/new_probes/probe_summary.json --out {root}/greybox_19.json',
        f'./cce/env_libero.sh cce/report_takeover_v4.py --run-dir {root}',
        '```', '',
        '新探针采集命令：`CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 ./cce/env_libero.sh cce/libero_probe_v3.py --out-dir <新的/mnt/sda/anonymous目录>`；输出目录若已有探针则拒绝覆盖。重新采集前必须重查GPU实时状态。', '',
        '本报告数值由`cce/report_takeover_v4.py`生成，机器可读摘要为`cce/results/takeover_v4_20260908.json`。']
    Path('docs/reports/CCE_takeover_v4_20260908.md').write_text('\n'.join(lines)+'\n')
    self_report = '# CCE executor v4 前置自检（开发阶段）\n\n'
    self_report += '\n'.join(lines[lines.index('## 3 v4执行器原型与反例'):lines.index('## 4 新增开环实验')])
    self_report += '\n完整ManiSkill双本体26plant回归尚未执行，门2前半未全部通过。详见`CCE_takeover_v4_20260908.md`。\n'
    Path('docs/reports/CCE_executor_v4_selftest.md').write_text(self_report)
    print(json.dumps({'report': 'docs/reports/CCE_takeover_v4_20260908.md',
                      'family': result['candidate_family'], 'identification_19': grey19['agreement']}, indent=2))


if __name__ == '__main__':
    main()
