"""Generate the results draft from audited historical, development and main data."""
import json
import hashlib
from pathlib import Path

def secondary_section():
    s=json.loads(Path('cce/results/libero_secondary_completed_v4r1.json').read_text())
    d=json.loads(Path('cce/results/libero_demos_completed_v4r1.json').read_text())
    assert s['status']=='complete' and s['secondary_rows']==5700
    lines=['## 六项次级比较与门控消融','',
           '六项次臂共5,700集，456个分片全部完成并通过自动网格、源码、参数和原始SHA审计。实施细节在主批结果已知后独立冻结。下表将次臂与主批相同ep=0..4匹配，每个plant×任务均为5集；不与上文10集主表混合比较。', '',
           '| 方法 | 16测试plant宏SR | 匹配5集宏R |', '|---|---:|---:|']
    for m,v in s['macro'].items():
        lines.append(f"| {m} | {v['sr']:.4f} | {v['R']['mean']:.4f} |")
    g=s['comparisons']['v4_no_gate__vs__ours_v4']['sr_difference']
    te=s['comparisons']['te_requery__vs__naive']['sr_difference']
    lines += ['',
              f"取消机械臂H门后，宏SR由{s['macro']['ours_v4']['sr']:.4f}变为{s['macro']['v4_no_gate']['sr']:.4f}，plant级差{g['mean']*100:.3f} pp，描述性bootstrap 95% CI [{g['ci95'][0]*100:.3f}, {g['ci95'][1]*100:.3f}] pp。该结果与CPU关门诊断方向一致，支持当前位置门会抑制某些有效补偿的解释。它属于次级消融，不能用来追认原方法已经通过P2或门2，也不能以其与slow20相同的SR点估计宣称等价成立。", '',
              f"固定补偿宏SR为{s['macro']['universal_fixed']['sr']:.4f}。TE重查询宏SR为{s['macro']['te_requery']['sr']:.4f}，相对naive差{te['mean']*100:.3f} pp，描述性95% CI [{te['ci95'][0]*100:.3f}, {te['ci95'][1]*100:.3f}] pp。本次TE同时改变重查频率与集成，且沿用原半开环proprio；结果只描述这一具体组合，不说明所有连续性方法或RTC无效。", '',
              '取消单侧约束、取消逆增益封顶、替换灰盒估计的宏SR点估计变化分别为+0.25、+0.25、+1.00 pp，不能据小幅变化宣称这些部件普遍无用。所有次级区间未经主检验家族的多重比较校正，不加入原Holm家族。', '',
              '## 策略侧演示采集的数据可得性','',
              f"在与评测隔离的初态上，固定采集预算共执行{d['total_attempts']}次尝试。每任务最多10次，选最先5成功；MX02不足时按原规划追加MX03。", '',
              '| plant | 尝试数 | 成功演示 | 每任务5条条件 |', '|---|---:|---:|---|']
    for p,v in d['plants'].items():
        lines.append(f"| {p} | {v['attempts']} | {v['success_demos']} | {'通过' if v['status']=='complete' else '不足'} |")
    lines += ['',
              '以上为原预算结果：当时仅D1_d250采齐50条并完成两种训练。第一次补充预算使MX03达标；随后多种子采集使D2_t400达标并完成训练，补充结果见后文。原预算的失败记录不改写，数据不足不证明策略适配不可补偿。', '']
    return lines


COMPLETION_REPORTS = [
    'docs/reports/CCE_policyside_completed_v4r1.md',
    'docs/reports/CCE_object_completed_v4r1.md',
    'docs/reports/CCE_overresponse_matched_v4r1.md',
    'docs/reports/CCE_probe_duration_completed_w4r2.md',
    'docs/reports/CCE_mechanism_w4r2.md',
    'docs/reports/CCE_nominal_acceptance_w4r2.md',
    'docs/reports/CCE_policyside_supplement_w4r2.md',
    'docs/reports/CCE_policyside_supplement_w4r4.md',
    'docs/reports/CCE_policyside_matched_table_w4r4.md',
]
COMPLETION_INPUTS = COMPLETION_REPORTS + [
    'cce/results/libero_policyside_completed_v4r1.json',
    'cce/results/libero_object_completed_v4r1.json',
    'cce/results/CCE_overresponse_matched_v4r1.json',
    'cce/results/CCE_probe_duration_completed_w4r2.json',
    'cce/results/CCE_mechanism_w4r2.json',
    'cce/results/CCE_nominal_acceptance_w4r2.json',
    'cce/results/CCE_demo_supplement_w4r2.json',
    'cce/results/CCE_policyside_supplement_w4r2.json',
    'cce/results/CCE_demo_supplement_w4r4.json',
    'cce/results/CCE_policyside_supplement_w4r4.json',
    'cce/results/CCE_multiseed_data_audit_w4r4.json',
    'cce/results/CCE_policyside_matched_table_w4r4.json',
    'cce/results/CCE_planned_figures_v4r5.json',
    'cce/results/CCE_planned_figures_render_v4r5.json',
    'cce/results/CCE_experiment_delivery_v4r5.json',
    'docs/reports/CCE_planned_figures_v4r5.md',
    'docs/reports/CCE_planned_tables_v4r5.md',
    'docs/reports/CCE_experiment_delivery_v4r5.md',
]

def completion_sections():
    lines=[]
    for path in COMPLETION_REPORTS:
        for line in Path(path).read_text().splitlines():
            lines.append('#'+line if line.startswith('#') else line)
        lines.append('')
    return lines

def supplement_section(main_result):
    data=json.loads(Path('cce/results/CCE_demo_supplement_w4r2.json').read_text())
    result=json.loads(Path('cce/results/CCE_policyside_supplement_w4r2.json').read_text())
    assert data['status']==result['status']=='execution_complete'
    model=result['plants']['MX03']
    assert model['independent_audit']['status']=='passed'
    for p,h in model['independent_audit']['inputs_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    baseline={}
    for p,h in main_result['raw_sha256'].items():
        raw=Path(p).read_bytes();assert hashlib.sha256(raw).hexdigest()==h,p
        for line in raw.splitlines():
            r=json.loads(line)
            if r['pid'] in ['L00_nominal','MX03'] and r['method'] in ['naive','ours_v4','slow20']:
                baseline.setdefault((r['pid'],r['method']),[]).append(r['success'])
    assert all(len(v)==100 for v in baseline.values()) and len(baseline)==6
    lines=['## 补充采集与MX03匹配比较','',
           f"固定补充协议实际执行{data['new_attempts']}次尝试，上限{data['max_attempts']}次；已补足任务提前停止，未补足任务耗尽各自20候选初态，不把剩余全局名额转移追加。D2_t400由34增至{data['plants']['D2_t400']['total']}条，任务0仅1条、任务3仅3条，未训练；MX03由49增至50条，完成40/10划分、两臂各5,000更新、G0′和各200集在线。两套数据的官方handler格式检查均通过，格式通过不等于数量达标。",'',
           '| 执行/策略 | 标称SR | MX03 SR |','|---|---:|---:|']
    for m in ['naive','ours_v4','slow20']:
        lines.append(f"| {m} | {sum(baseline['L00_nominal',m])/100:.2%} | {sum(baseline['MX03',m])/100:.2%} |")
    for m in ['ft_demo','prompt_fit']:
        r=model['modes'][m];assert r['status']=='complete' and r['online_episodes']==200
        lines.append(f"| {m} | {r['sr']['L00_nominal']:.2%} | {r['sr']['MX03']:.2%} |")
    lines += ['', '各单元使用同一10任务×10评测初态；策略侧为最终模型的naive执行。目标plant上的小幅恢复伴随标称性能下降，不能仅报告恢复而省略标称损伤。两种适配均通过teacher-forced G0′，该门不保证闭环保持能力。该批结束时D1与MX03已观测FT-demo均低于r1，D2尚未完成；D2后来完成的结果见多种子补采一节。新增采集预算与单训练种子限制需要披露，不加入原Holm家族。','']
    return lines


def multiseed_section():
    data=json.loads(Path('cce/results/CCE_demo_supplement_w4r4.json').read_text())
    audit=json.loads(Path('cce/results/CCE_multiseed_data_audit_w4r4.json').read_text())
    completed=json.loads(Path('cce/results/CCE_policyside_supplement_w4r4.json').read_text())
    assert data['plants']['D2_t400']['total']==50 and audit['status']=='passed'
    assert completed['plants']['D2_t400']['independent_audit']['status']=='passed'
    jobs=[]
    for p,h in audit['attempt_log_sha256'].items():
        raw=Path(p).read_bytes();assert hashlib.sha256(raw).hexdigest()==h
        rows=[json.loads(x) for x in raw.splitlines()]
        jobs.append((rows[0]['task_id'],len(rows),sum(r['accepted'] for r in rows)))
    lines=['## D2多种子补采与闭环负结果','',
           f"在两个困难任务中，通过不同策略生成种子补采，共{data['new_attempts']}次有效尝试得到6条新成功演示。教师、物理条件、成功判据不变；每任务累计5条便停止。此前不足记录保持。",'',
           '| 任务 | 本批尝试 | 新接受演示 |','|---|---:|---:|']
    for task,n,k in sorted(jobs):lines.append(f'| {task} | {n} | {k} |')
    lines+=['',f"最终40条训练演示覆盖{audit['train_initial_groups']}个任务×初态组，10条验证覆盖{audit['validation_initial_groups']}组，组交集为{audit['group_overlap']}。验证演示及初态在本批之前固定，补采排除验证初态，评测初态0–9保持隔离。多采集种子不是多训练种子复现；新增尝试成本和数据覆盖与D1原预算不同。",'',
            '两种适配均完成5,000更新并通过预定30拍xyz与all10平均MAE低于persistence的G0′，冻结参数审计及各200集在线网格通过。目标与标称表现均见统一表：G0′通过不等于闭环策略保持能力，训练损失下降也不能单独说明适配成功。', '',
            'D2上r1与naive均为45%，slow20为84%；FT-demo与prompt-fit分别19%和13%。r1在该条件下机械臂门关闭，不能由此断言所有反演结构均无效。当前训练配方造成闭环退化，不能泛化为所有策略适配不可补偿；补采也没有改变原P1/P2结果。','']
    return lines


def planned_figure_section():
    f=json.loads(Path('cce/results/CCE_planned_figures_v4r5.json').read_text())
    d=json.loads(Path('cce/results/CCE_experiment_delivery_v4r5.json').read_text())
    assert f['status']==d['status']=='audited' and d['main_tests_unchanged']
    for row in d['offline_acceptance'].values():
        for mode,v in row.items():
            assert v['strict_better_dimension_count']['30']==10
            assert v['strict_better_dimension_count']['1']==(1 if mode=='ft_demo' else 0)
    errors=f['F2_T1_F6']['macro_absolute_error']
    lines=['## 规划图表补齐：辨识、共同意图与三遍对照','',
           'F1采用事前已留存的MX02 task0/ep0。三种方法初始位姿逐值相同；naive/r1首30拍意图逐值一致，slow20为同一chunk插值至60拍。图中第一chunk展示共同目标下的TCP、跟踪误差和夹爪，完整rollout夹爪与失败标记单独展示。三集均在800拍失败；后续策略观测不同导致重查意图分叉，不将完整rollout当作固定意图因果对照。', '',
           f"F2/T1与F6辨识对比使用实际冻结r1包。19条件灰盒/ARX的平移轴中位滞后参数MAE分别为{errors['greybox']['tau_s']*1000:.2f}/{errors['arx_relative']['tau_s']*1000:.2f} ms，增益MAE为{errors['greybox']['gain']:.4f}/{errors['arx_relative']['gain']:.4f}；延迟与夹爪提前MAE均为0。这里衡量给定探针下的估计准确性，不是结构可辨识性证明。相同标称串联模型与窗口下的自由仿真拟合另列；没有新增拟合或参数选择。", '',
           '尽管灰盒的参数误差较小，D2_t400等条件仍出现r1无恢复，说明参数准确性不能直接替代闭环补偿有效性证据。F2各参数分列并保留单位，不事后构造有利的误差加权分数。', '',
           '| 同12条件、每任务5初态 | 第一遍SR | 第二遍SR | 第三遍SR |', '|---|---:|---:|---:|']
    for method in f['F4']['macro']['1']:
        lines.append('| '+method+' | '+' | '.join(f"{f['F4']['macro'][str(k)][method]['sr']:.2%}" for k in [1,2,3])+' |')
    lines += ['', 'F4含历史开发条件，区间为描述性的plant bootstrap。三遍同时存在命令界、参数化和策略随机数实现差异，因此不能分别量化“命令界贡献”与“相对反演贡献”；原规划这一因果措辞超出现有对照能支持的范围。跨遍对齐不替代原16测试条件主检验。', '',
              '六个最终模型的30拍teacher-forced误差已补做逐维核对：有效机械臂10维全部低于persistence，既满足冻结均值门，也满足该视界的逐维条件。1拍视界仅FT-demo各1/10维、prompt-fit各0/10维优于persistence；10拍也有部分模型未全部优于。视界条件必须明确，不能把30拍验收扩展为每一拍都更好。', '',
              'F1–F6与T1–T5的文件、重建命令、原规划修订与未通过验收集中在[实验交付索引](../reports/CCE_experiment_delivery_v4r5.md)。逐条件主表及配对区间见[定量表](../reports/CCE_planned_tables_v4r5.md)，参数和三遍口径见[图表证据](../reports/CCE_planned_figures_v4r5.md)。原主检验数值保持不变。', '']
    return lines


def main():
    a=json.loads(Path('cce/results/libero_family_v1.json').read_text())
    b=json.loads(Path('cce/results/libero_family_v2.json').read_text())
    d=json.loads(Path('cce/results/cce_v4r1_development.json').read_text())
    main_result=json.loads(Path('cce/results/libero_fallback_v4r1.json').read_text())
    assert main_result['independent_audit']['status']=='passed'
    p=b['primary']['P1_ours_p_minus_naive'];q=b['primary']['P2_ours_p_minus_slow20']
    t=b['time_to_success_common_cells']
    lines=['# CCE 结果节工作稿：减速对照与相对反演的失败边界','',
           '本稿汇集历史两遍、r1开发验收、15,200集主批、5,700集次级比较、D1及补充预算MX03、D2策略适配、object第二套件和W4诊断。它是可追溯的结果工作稿，不是完整投稿稿；策略侧三个条件六次训练和1,200集在线评测均完成。各历史报告中的状态以其原预算为准。', '',
           '## 受控失配下的成功率与减速对照','',
           f"已有两遍共{a['n_episodes']+b['n_episodes']:,}集。第一遍报告全12个plant，含随后被指定为开发的标称与150 ms延迟；第二遍主推断使用排除开发plant后的12个测试plant。为避免人群变化被误读为改进，下面同时列出第二遍在第一遍原12个plant上的宏均值。两遍命令界及其他实施差异依原记录保留，不以跨遍差值作单因素因果解释。", '',
           '| 臂 | 第一遍：原12 plant | 第二遍：相同原12 plant | 第二遍：预定12测试plant |', '|---|---:|---:|---:|']
    for m in ['naive','slow15','slow20','oracle','ours_p']:
        lines.append(f"| {m} | {a['macro'][m]['macro_sr_family_mean']:.4f} | {b['compare_with_pass_1']['v2_macro_same_12_plants'][m]:.4f} | {b['macro'][m]['macro_sr_test_mean']:.4f} |")
    lines+=['',
            f"第二遍中，参数反演相对naive的plant级平均改善为{p['mean']*100:.2f} pp，bootstrap 95% CI为[{p['ci95'][0]*100:.2f}, {p['ci95'][1]*100:.2f}] pp，双侧配对t经原两假设Holm校正p={b['primary_holm_p']['P1_ours_p_minus_naive']:.6f}。这一结果支持在该受控族中恢复部分失配损失。", '',
            f"但反演相对slow20的差为{q['mean']*100:.2f} pp，95% CI [{q['ci95'][0]*100:.2f}, {q['ci95'][1]*100:.2f}] pp，Holm p={b['primary_holm_p']['P2_ours_p_minus_slow20']:.6f}。简单减速的成功率更高，因而不能将相对naive的改善写成优于通用保守执行。", '',
            '## 时间优势的条件化边界','',
            f"第二遍在双方均至少一次成功的{t['n_common_cells']}个plant×任务格上，先平均格内成功集，再plant等权，反演与slow20的平均成功耗时分别为{t['ours_p_s']:.3f} s和{t['slow20_s']:.3f} s，相对缩短{t['reduction_pct']:.1f}%。plant级平均差为{t['paired_plant_level']['mean']:.4f} s。", '',
            '这是成功条件下的时间比较：不能把失败任务或失败episode排除后的速度优势等同于总体效率提高，也不能单独作为方法优越性证据。r1组合臂见下文独立实验，不以跨遍差值作因果比较。', '',
            '## 相对反演的开发验收','',
            f"r1执行器只反演注入差异，并将机械臂位置门与夹爪提前分开。完整无策略回归覆盖双机器人、26 plant、20条合成参考及50个回放初态、6个方法，共{d['cpu_rows']:,}条记录。所有ρ候选均未达到既定门2。", '',
            '| 机器人 | ρ=2 跟踪改善 | ρ=2 回放R | 旧v3同批跟踪改善 | 旧v3同批回放R |', '|---|---:|---:|---:|---:|']
    for uid,x in d['cpu_metrics'].items():
        v,o=x['v4_rho2'],x['ours_v3']
        lines.append(f"| {uid} | {v['lag_tracking_reduction']['rmse_pos_m']:.2%} | {v['macro_R']:.4f} | {o['lag_tracking_reduction']['rmse_pos_m']:.2%} | {o['macro_R']:.4f} |")
    lines+=['',
            '跟踪改善此表使用原门槛的shaped-reference口径；完整机器报告也保留对原意图的误差，二者不能混同。旧v3本次回放R与历史底线存在小幅差异，原底线仍原样保留；相对反演的退步远大于此偏差。', '',
            f"LIBERO正控为{d['G0']['success']}/{d['G0']['n']}。开发集240集、四ρ候选逐集相同：三个开发plant估计滞后均为0，无法识别滞后逆封顶的影响，按事前平局规则取ρ=2，H0={d['H0']:.17f} m。因此不能宣称ρ经实验确定为最优。标称两集naive/r1逐步一致，仅是该两条初始化下的端到端检查。", '',
            '门2失败触发既定乙分支，相对反演作为未达到预期的尝试保留。主策略对照仍按冻结设计执行，以报告伤害面、减速前沿和标定的代价；不根据新的成功率读数撤销已失败的门2。', '']
    r=main_result; pr=r['primary']
    lines += ['## r1冻结主批：相对改善与速率匹配检验','',
              '主批覆盖19个plant（3开发、16测试）、10任务，七主臂每任务10集；另两个次臂slow30与旧执行器每任务5集，共15,200集。684个分片全部正常结束，原始文件SHA、冻结源码/参数、唯一键与完整网格经独立复核；推断单位为测试plant。', '',
              '| 方法 | 测试宏SR | 宏恢复R | 共同成功T(s) | 每任务集数 |', '|---|---:|---:|---:|---:|']
    for m,x in r['macro'].items():
        lines.append(f"| {m} | {x['sr']:.4f} | {x['R']:.4f} | {x['T_common_all_s']:.3f} | {5 if m in ('slow30','ours_v3') else 10} |")
    p1,p2=pr['P1'],pr['P2b'];eq=pr['P2a']['equivalence']
    lines += ['',
              f"R仅在标称naive SR减去失配naive SR大于10 pp的{len(r['R_population'])}个测试plant上定义，按plant等权；T采用九臂均至少一次成功的{len(r['frontier_common_cells'])}个任务格。时间人群与R人群不同，图中连线仅展示固定减速因子的描述性折中，不证明总体效率或支配关系。", '',
              f"相对naive，r1的宏SR从{r['macro']['naive']['sr']:.4f}提高至{r['macro']['ours_v4']['sr']:.4f}，plant级改善{p1['mean']*100:.4f} pp，bootstrap 95% CI [{p1['ci95'][0]*100:.3f}, {p1['ci95'][1]*100:.3f}] pp，三项Holm校正p={pr['holm']['P1']:.6f}，达到预定P1。", '',
              f"速率匹配的ours_v4_slow20与slow20分别达到{r['macro']['ours_v4_slow20']['sr']:.4f}和{r['macro']['slow20']['sr']:.4f}，差{p2['mean']*100:.3f} pp，未校正bootstrap 95% CI [{p2['ci95'][0]*100:.3f}, {p2['ci95'][1]*100:.3f}] pp。尽管点估计与区间为正，Holm p={pr['holm']['P2b']:.6f}未达到0.05，故P2b不成立，不能写成显著优于减速基线。", '',
              f"ours_v4_slow15相对slow20的SR差为{eq['mean']*100:.4f} pp；预定±5 pp等价TOST p={eq['p']:.6f}，不通过。该对照的共同成功耗时缩短{pr['P2a']['time']['reduction']:.2%}，达到时间幅度要求，但不能替代未通过的等价检验，P2a整体仍不成立。不在看到结果后将等价改成非劣检验。", '',
              f"P3在由naive损失小于25 pp定义的{pr['P3']['n']}个轻失配测试plant上通过±3 pp等价检验（p={pr['P3']['p']:.6f}）；这只支持该条件子集。标称100个初态上四组方法配对、共400对的七项汇总字段均一致；此前两初态还有逐步一致检查，不能将二者混写为400条完整轨迹验证。", '',
              f"延迟/滞后测试plant的意图跟踪RMSE相对改善为{r['K1_prime']['reduction']:.2%}，低于预定30%（K1′触发）；两条P2路径均未通过（K3触发）。结合此前门2失败，本批支持部分损失恢复与速率折中的分析，不支持恢复方法优越性的甲分支。", '',
              '## 当前证据不支持的表述','',
            'ManiSkill五条策略锚定路线未过G0，不能写跨本体策略迁移成功。三个条件各一个训练种子的结果不能支持“策略侧不可补偿”；原预算K4不可判定的历史记录保留，补充完成后K4未触发，不撤销门2或P2失败。探针估计为接口有效模型量，不等于机器人真实物理参数。当前位置H对夹爪无感的原型反例、灰盒参数残差、开发扫描不可辨识和回归失败均须进入局限与方法学记录。', '',
            '## 可追溯来源','',
            '输入：`cce/results/libero_family_v1.json`、`libero_family_v2.json`、`cce_v4r1_development.json`、`libero_fallback_v4r1.json`；历史与r1主批原始JSONL分别验证6,500及15,200个唯一键；次级与采集来源另见`libero_secondary_completed_v4r1.json`和`libero_demos_completed_v4r1.json`。', '',
            '图表：`data/cce/v4r1_20260908/figures/historical_v2_controls.pdf`、`v4r1_gate_report.pdf`；原始图与数据在/mnt/sda。', '']
    idx=lines.index('## 当前证据不支持的表述');lines[idx:idx]=secondary_section()+completion_sections()+supplement_section(main_result)+multiseed_section()+planned_figure_section()
    target=Path('docs/paper/CCE_results_draft_v4r1.md');target.parent.mkdir(exist_ok=True)
    target.write_text('\n'.join(lines))
    paths=['cce/results/libero_family_v1.json','cce/results/libero_family_v2.json','cce/results/cce_v4r1_development.json','cce/results/libero_fallback_v4r1.json']
    paths += ['cce/results/libero_secondary_completed_v4r1.json','cce/results/libero_demos_completed_v4r1.json']
    paths += COMPLETION_INPUTS
    Path('docs/paper/CCE_results_draft_v4r1_inputs.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(target)

if __name__=='__main__':main()
