"""Promote the audited diagnostic summary, preserving raw runs on /mnt/sda."""
import json
from pathlib import Path

def main():
    root=Path('/opt/cce/data/cce/v4r1_20260908/cpu_gate_diagnostic_r2')
    assert (root/'supervisor.exit').read_text().strip()=='0'
    a=json.loads((root/'results.json').read_text())
    n=sum(r[b]['n_rows'] for r in a['robots'].values() for b in ['tracking','replay'])
    assert n==7280 and a['status']=='complete'
    a['n_rows']=n
    lines=['# v4r1 H门事后诊断：完整结果','',
           '7,280条CPU记录、52分片均正常完成，完整网格与唯一键审计通过。固定ρ=2，只切换机械臂位置门；夹爪提前、辨识包、控制预算与初态保持一致。此为已知门2失败后提出的机制诊断，不新增确证主张。', '',
           '| 机器人 | 关门−开门：宏回放SR | 开门−关门：平均意图RMSE(mm) |', '|---|---:|---:|']
    for uid,r in a['robots'].items():
        lines.append(f"| {uid} | {r['replay']['macro_nogate_minus_gate']*100:+.2f} pp | {-r['tracking']['macro_nogate_minus_gate']*1000:+.3f} |")
    lines+=['',
            '人群为原CPU基准全部26plant（包括标称），先各plant内平均，再plant等权。不能把这些数字当作LIBERO策略成功率或跨本体迁移结果。', '',
            '关门后改善不局限于纯滞后。Panda的MX02/MX03/MX04以及xArm的MX04有明显回放改善；完整逐plant表如下。该干预支持“当前位置H阈值关闭了部分有用补偿”的解释，但不说明任意失配门都无效，也不支持按此批结果重调主实验阈值。', '',
            '冻结LIBERO主批保持原ρ与H0；甲分支门2判决不追溯修改。对于原本不退回的plant，两个配置的物理跟踪仍有极小数值差异，完整表保留，不能将无显著差异写成逐步物理完全相等。', '',
            '首次启动仅发生参数解析错误，零有效轨迹；修复与冒烟在新目录重跑前提交（f10c718）。原失败记录`cpu_gate_diagnostic/`保留。', '',
            f'原始结果与日志：`{root}`。', '', (root/'report.md').read_text()]
    Path('cce/results/cce_v4r1_gate_diagnostic.json').write_text(json.dumps(a,indent=2,allow_nan=False)+'\n')
    Path('docs/reports/CCE_v4r1_gate_diagnostic.md').write_text('\n'.join(lines))
    print('Promoted audited 7280-case diagnostic report')

if __name__=='__main__':main()
