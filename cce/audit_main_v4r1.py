"""Independent post-run audit and promotion; original frozen analysis remains intact."""
import _bootstrap  # noqa: F401
from collections import Counter
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel
from cce.freeze_v4r1 import validate_freeze,sha

ROOT=Path('/opt/cce/data/cce/v4r1_20260908/fallback_main')

def main():
    frozen='cce/results/libero_family_spec_v4r1_fallback.json'
    spec=validate_freeze(frozen)
    assert (ROOT/'supervisor.exit').read_text().strip()=='0'
    a=json.loads((ROOT/'results.json').read_text())
    paths=sorted((ROOT/'jobs').glob('*.jsonl'));assert len(paths)==684
    rows=[]
    for p in paths:
        assert sha(p)==a['raw_sha256'][str(p)]
        assert p.with_suffix('.exit').read_text().strip()=='0'
        meta=json.loads(p.with_suffix('.meta.json').read_text())
        assert meta['freeze_sha256']==sha(frozen)
        assert meta['source_sha256']==spec['inputs_sha256']['cce/libero_family_v3.py']
        assert meta['pack_sha256']==spec['pack_sha256']
        rows.extend(json.loads(l) for l in p.read_text().splitlines())
    expected={(p,m,t,e) for p,m,t in itertools.product(spec['plant_ids'],spec['episodes'],spec['tasks']) for e in range(spec['episodes'][m])}
    count=Counter((r['pid'],r['method'],r['task_id'],r['ep']) for r in rows)
    assert set(count)==expected and set(count.values())=={1} and len(rows)==15200
    lookup={(r['pid'],r['method'],r['task_id'],r['ep']):r for r in rows}
    sr={}
    for p,m in itertools.product(spec['plant_ids'],spec['episodes']):
        tasks=[np.mean([lookup[p,m,t,e]['success'] for e in range(spec['episodes'][m])]) for t in spec['tasks']]
        sr[p,m]=float(np.mean(tasks));assert abs(sr[p,m]-a['cells'][p][m]['sr'])<1e-12
    tests={}
    for k,m,b in [('P1','ours_v4','naive'),('P2b','ours_v4_slow20','slow20')]:
        pv=float(ttest_rel([sr[p,m] for p in spec['test']],[sr[p,b] for p in spec['test']]).pvalue)
        assert abs(pv-a['primary'][k]['p'])<1e-12;tests[k]=pv
    fields=['success','steps','tracking_rmse_m','policy_queries','n_bound_hits','u_pos_max_norm','u_rot_max_norm']
    pairs=[('ours_v4','naive'),('oracle_v4','naive'),('ours_v4_slow15','slow15'),('ours_v4_slow20','slow20')]
    discrepancies=[]
    for m,b in pairs:
        for t,e,f in itertools.product(spec['tasks'],range(10),fields):
            if lookup['L00_nominal',m,t,e][f]!=lookup['L00_nominal',b,t,e][f]:discrepancies.append([m,b,t,e,f])
    audit={'status':'passed','n_rows':15200,'n_files':684,'freeze_sha256':sha(frozen),'independent_t_p':tests,
           'nominal_paired_checks':400,'nominal_differences':discrepancies,
           'wall_hours':((ROOT/'supervisor.exit').stat().st_mtime-(ROOT/'plan.json').stat().st_mtime)/3600}
    a['independent_audit']=audit
    Path('cce/results/libero_fallback_v4r1.json').write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    p=a['primary']
    head=['# CCE v4r1 15,200集主批：完成审计','',
          f"684分片全部退出0，15,200唯一键与预定网格完全一致；原始SHA、分片源码/参数/冻结SHA全部通过。独立任务等权SR和SciPy配对t复算一致。墙钟{audit['wall_hours']:.2f}小时。",'',
          f"P1成立：+{p['P1']['mean']*100:.4f} pp，Holm p={p['holm']['P1']:.6f}。P2b虽为+{p['P2b']['mean']*100:.3f} pp，且未校正95% CI下界为正，但Holm p={p['holm']['P2b']:.6f}，未过0.05，不写显著胜出。",'',
          f"P2a共同成功耗时缩短{p['P2a']['time']['reduction']:.2%}；SR差+{p['P2a']['equivalence']['mean']*100:.4f} pp，未通过预定±5pp TOST。不能事后改成单侧非劣检验。",'',
          f"P3在{p['P3']['n']}个由naive筛选的轻失配测试plant上通过±3pp等价，p={p['P3']['p']:.6f}；只对该条件人群作结论。",'',
          f"标称100集的四组匹配比较、共400对汇总字段逐项检查，差异数{len(discrepancies)}。这不等于400条独立初态，也不替代逐步轨迹证明。",'',
          f"K1′意图跟踪改善仅{a['K1_prime']['reduction']:.2%}，仍低于30%；CPU门2也已失败。保留乙分支，不能因组合臂点估计较高恢复甲主张。",'',
          '后续六项次臂与策略侧/第二套件结果尚未包含。以下为冻结汇总原文，保持检验不变。','', (ROOT/'report.md').read_text()]
    Path('docs/reports/CCE_libero_fallback_v4r1.md').write_text('\n'.join(head))
    (ROOT/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
