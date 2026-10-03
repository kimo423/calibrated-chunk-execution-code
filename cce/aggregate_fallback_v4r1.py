"""Complete-grid audits and plant-level statistics for the frozen fallback study."""
import _bootstrap  # noqa: F401
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from cce.stats_v4r1 import paired, tost, holm


def summarize(rows, spec):
    methods = list(spec['episodes'])
    pids, test, tasks = spec['plant_ids'], spec['test'], spec['tasks']
    expected = {(p,m,t,e) for p,m,t in itertools.product(pids,methods,tasks)
                for e in range(spec['episodes'][m])}
    counts = Counter((r['pid'],r['method'],r['task_id'],r['ep']) for r in rows)
    assert set(counts) == expected, f'Missing {len(expected-set(counts))}; extra {len(set(counts)-expected)}'
    assert max(counts.values()) == 1, 'Duplicate episodes'
    for r in rows:
        assert r['success'] in (0,1) and 1 <= r['steps'] <= 800
        assert r['success'] or r['steps'] == 800, 'Failed episode stopped before budget'
        assert np.isfinite(r['tracking_rmse_m']) and r['tracking_rmse_m'] >= 0
        assert 0 <= r['n_bound_hits'] <= r['steps']
        if r['success']:
            assert abs(r['time_to_success_s'] - r['steps']*.05) < 1e-9
    groups = {(p,m,t): [] for p,m,t in itertools.product(pids,methods,tasks)}
    for r in rows:
        groups[r['pid'],r['method'],r['task_id']].append(r)
    cells = {}
    for p in pids:
        cells[p] = {}
        for m in methods:
            sub = [r for t in tasks for r in groups[p,m,t]]
            cells[p][m] = {
                'n':len(sub), 'sr':float(np.mean([r['success'] for r in sub])),
                'rmse_m':float(np.mean([r['tracking_rmse_m'] for r in sub])),
                'bound_step_share':sum(r['n_bound_hits'] for r in sub)/sum(r['steps'] for r in sub)}
    nominal = cells['L00_nominal']['naive']['sr']
    rpop = [p for p in test if nominal-cells[p]['naive']['sr'] > .10+1e-12]
    light = [p for p in test if nominal-cells[p]['naive']['sr'] < .25-1e-12]
    def diffs(a,b,pop=test):
        return [cells[p][a]['sr']-cells[p][b]['sr'] for p in pop]
    def time_pair(a,b):
        byplant, common = {}, []
        for p in test:
            aa,bb = [],[]
            for t in tasks:
                ra = [r['time_to_success_s'] for r in groups[p,a,t] if r['success']]
                rb = [r['time_to_success_s'] for r in groups[p,b,t] if r['success']]
                if ra and rb:
                    va,vb = float(np.mean(ra)),float(np.mean(rb))
                    aa.append(va); bb.append(vb)
                    common.append({'pid':p,'task_id':t,'a_s':va,'b_s':vb,'n_success_a':len(ra),'n_success_b':len(rb)})
            if aa:
                byplant[p] = {'a_s':float(np.mean(aa)),'b_s':float(np.mean(bb)),'n_tasks':len(aa)}
        stat = paired([v['b_s']-v['a_s'] for v in byplant.values()])
        reduction = 1-np.mean([v['a_s'] for v in byplant.values()])/np.mean([v['b_s'] for v in byplant.values()]) if byplant else None
        return {'arms':[a,b], 'plants':byplant, 'cells':common,'time_saved_test':stat,
                'reduction':float(reduction) if reduction is not None else None}
    p1 = paired(diffs('ours_v4','naive'),alternative='two-sided')
    p2b = paired(diffs('ours_v4_slow20','slow20'),alternative='two-sided')
    eq = tost(diffs('ours_v4_slow15','slow20'),.05)
    tp = time_pair('ours_v4_slow15','slow20')
    joint = max(eq['p'],tp['time_saved_test']['p']) if eq['p'] is not None and tp['time_saved_test']['p'] is not None else None
    adj = holm({'P1':p1['p'],'P2a':joint,'P2b':p2b['p']})
    p1['criterion_met'] = bool(p1['mean'] >= .15-1e-12 and p1['ci95'] and p1['ci95'][0]>0 and adj['P1']<.05)
    p2b['criterion_met'] = bool(p2b['mean'] >= .05-1e-12 and p2b['ci95'] and p2b['ci95'][0]>0 and adj['P2b']<.05)
    p2a = {'equivalence':eq, 'time':tp, 'joint_p':joint,
           'criterion_met':bool(eq['equivalent'] and tp['reduction'] is not None and tp['reduction']>=.20-1e-12 and adj['P2a']<.05)}
    p3 = tost(diffs('ours_v4','naive',light),.03)
    macro = {}
    # The time frontier uses exactly the same successful task cells for all arms.
    all_common = []
    frontier_times = {m:{} for m in methods}
    for p in test:
        pt = {m:[] for m in methods}
        for t in tasks:
            values = {m:[r['time_to_success_s'] for r in groups[p,m,t] if r['success']] for m in methods}
            if all(values.values()):
                all_common.append([p,t])
                for m in methods:
                    pt[m].append(float(np.mean(values[m])))
        for m in methods:
            if pt[m]:
                frontier_times[m][p] = float(np.mean(pt[m]))
    for m in methods:
        rs = {p:(cells[p][m]['sr']-cells[p]['naive']['sr'])/(nominal-cells[p]['naive']['sr']) for p in rpop}
        macro[m] = {'sr':float(np.mean([cells[p][m]['sr'] for p in test])),
                    'R':float(np.mean(list(rs.values()))) if rs else None,
                    'R_by_plant':rs, 'R_ci95':paired(list(rs.values()))['ci95'],
                    'T_common_all_s':float(np.mean(list(frontier_times[m].values()))) if frontier_times[m] else None,
                    'T_by_plant':frontier_times[m]}
    lag = [p['pid'] for p in spec['plants'] if p['pid'] in test and p['family'] in ('D1','D2t','MIX')]
    reductions = {p:1-cells[p]['ours_v4']['rmse_m']/cells[p]['naive']['rmse_m'] for p in lag if cells[p]['naive']['rmse_m']>0}
    k1 = float(np.mean(list(reductions.values()))) if reductions else None
    # Exact ep=0..4 pairing for the old executor and slow30 comparisons.
    matched5 = {}
    for a in ['ours_v3','slow30']:
        matched5[a] = {}
        for b in ['naive','slow20','ours_v4']:
            dd=[]
            for p in test:
                av=[r['success'] for t in tasks for r in groups[p,a,t] if r['ep']<5]
                bv=[r['success'] for t in tasks for r in groups[p,b,t] if r['ep']<5]
                dd.append(float(np.mean(av)-np.mean(bv)))
            matched5[a][b]=paired(dd,alternative='two-sided')
    return {'status':'complete','n_rows':len(rows),'n_test_plants':len(test),'cells':cells,'macro':macro,
            'R_population':rpop,'P3_population':light,'primary':{'P1':p1,'P2a':p2a,'P2b':p2b,'P3':p3,'holm':adj},
            'K1_prime':{'population':lag,'per_plant':reductions,'reduction':k1,'triggered':k1 is None or k1<.30},
            'K2_triggered':macro['ours_v4']['R'] is None or macro['ours_v4']['R']<.30,
            'K3_triggered':not(p2a['criterion_met'] or p2b['criterion_met']), 'K5_triggered':not p3['equivalent'],
            'frontier_common_cells':all_common,'time_ours_v3_vs_slow20':time_pair('ours_v3','slow20'),
            'matched_first5_descriptive':matched5,
            'method_claim_authorized':False, 'reason':'Gate 2 failed before this fallback run; no post-hoc reversal'}


def report(out):
    lines=['# CCE v4r1 乙分支完整结果', '',
           f"完整性审计通过：{out['n_rows']:,}集，{out['n_test_plants']}个测试plant。门2此前失败，甲主张保持关闭。", '',
           '| 臂 | 宏SR | R | 共同成功T(s) |', '|---|---:|---:|---:|']
    for m,x in out['macro'].items():
        r='不可估计' if x['R'] is None else f"{x['R']:.4f}"
        t='不可估计' if x['T_common_all_s'] is None else f"{x['T_common_all_s']:.3f}"
        lines.append(f"| {m} | {x['sr']:.4f} | {r} | {t} |")
    lines += ['', 'T仅针对9臂均有成功的共同任务格，不能替代失败惩罚后的总体效用。slow30/ours_v3每格5集，主臂10集。', '',
              '完整逐plant表、共同成功人群、三项Holm检验、P3条件子集及前5集匹配比较见同名JSON。', '',
              '```json', json.dumps({'holm':out['primary']['holm'], 'P1':out['primary']['P1'],
                                    'P2a_met':out['primary']['P2a']['criterion_met'],'P2b':out['primary']['P2b'],
                                    'P3':out['primary']['P3'],'K1_prime':out['K1_prime']},ensure_ascii=False,indent=2), '```', '']
    return '\n'.join(lines)


def selftest():
    methods=['naive','slow15','slow20','ours_v4','ours_v4_slow15','ours_v4_slow20','oracle_v4','slow30','ours_v3']
    ps=['L00_nominal','a','b']
    spec={'plant_ids':ps,'test':['a','b'],'tasks':[0,1],'episodes':dict.fromkeys(methods,2),
          'plants':[{'pid':p,'family':'D1'} for p in ps]}
    rows=[{'pid':p,'method':m,'task_id':t,'ep':e,'success':1,'steps':100,'time_to_success_s':5.,
           'tracking_rmse_m':.02,'n_bound_hits':0} for p,m,t,e in itertools.product(ps,methods,[0,1],[0,1])]
    x=summarize(rows,spec)
    assert x['primary']['P3']['equivalent'] and not x['primary']['P2a']['criterion_met']
    assert x['macro']['naive']['R'] is None and x['macro']['naive']['T_common_all_s']==5
    for bad in [rows[:-1],rows+[rows[0]]]:
        try:summarize(bad,spec)
        except AssertionError:pass
        else:raise AssertionError('Failed to reject incomplete/duplicate grid')
    bad=[dict(r) for r in rows];bad[0]['success']=0
    try:summarize(bad,spec)
    except AssertionError:pass
    else:raise AssertionError('Accepted early failure')
    print('aggregate selftest: complete, no-R, conditional-time, duplicate, missing, early-exit checks passed')


def figures(out, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=list(out['macro'])
    pids=list(out['cells'])
    fig,axes=plt.subplots(1,2,figsize=(17,7),gridspec_kw={'width_ratios':[1.5,1]})
    heat=np.array([[out['cells'][p][m]['sr'] for m in methods] for p in pids])
    im=axes[0].imshow(heat,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    axes[0].set_xticks(range(len(methods)),methods,rotation=45,ha='right')
    axes[0].set_yticks(range(len(pids)),pids)
    axes[0].set_title('All 19 plants; test-plant inference excludes 3 development plants')
    fig.colorbar(im,ax=axes[0],label='Success rate',fraction=.035)
    for j,m in enumerate(methods):
        x=out['macro'][m]
        if x['R'] is None or x['T_common_all_s'] is None:continue
        axes[1].scatter(x['T_common_all_s'],x['R'],marker='s' if m in ('slow30','ours_v3') else 'o')
        axes[1].annotate(m,(x['T_common_all_s'],x['R']),xytext=(4,5+(j%3)*9),textcoords='offset points',fontsize=8)
    slow=[out['macro'][m] for m in ['naive','slow15','slow20','slow30']]
    if all(x['R'] is not None and x['T_common_all_s'] is not None for x in slow):
        axes[1].plot([x['T_common_all_s'] for x in slow],[x['R'] for x in slow],color='gray',alpha=.5,linestyle='--')
    axes[1].set(xlabel='Time on all-arm common-success cells (s)',ylabel='Macro recovery R',
                title='Fallback study: success-conditioned time, descriptive frontier')
    axes[1].grid(alpha=.2)
    fig.suptitle('CCE v4r1: Gate 2 failed; no superior-method claim',fontsize=14)
    fig.text(.5,.01,'Main arms: 10 episodes/task; slow30 and ours_v3: 5. R and T populations are listed in results.json. Lines connect fixed slowdown factors.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.96])
    fig.savefig(root/'frontier.pdf');fig.savefig(root/'frontier.png',dpi=180);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir');ap.add_argument('--freeze');ap.add_argument('--selftest',action='store_true')
    args=ap.parse_args()
    if args.selftest:
        selftest();return
    from cce.freeze_v4r1 import validate_freeze, sha
    spec=validate_freeze(args.freeze)
    root=Path(args.run_dir)
    assert (root/'execution.exit').read_text().strip()=='0'
    paths=sorted((root/'jobs').glob('*.jsonl'))
    rows=[json.loads(l) for p in paths for l in p.read_text().splitlines()]
    out=summarize(rows,spec)
    out['freeze_sha256']=sha(args.freeze)
    out['raw_sha256']={str(p):sha(p) for p in paths}
    (root/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (root/'report.md').write_text(report(out))
    figures(out,root)
    print(json.dumps({'n_rows':len(rows),'macro':{m:{k:v for k,v in r.items() if k not in ('R_by_plant','T_by_plant')} for m,r in out['macro'].items()}},indent=2))

if __name__=='__main__':
    main()
