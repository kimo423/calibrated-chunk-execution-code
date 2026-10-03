"""Post-gate-failure CPU mechanism study; does not change the frozen policy study."""
import _bootstrap  # noqa: F401
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from cce.bench_relative_v4r1 import SimAdapter, gate_values
from cce.libero_exec_v4r1 import RelativeExecutor, RelativeTheta
from cce.plants import PLANTS

ROOT=Path('/opt/cce/data/cce/v4r1_20260908')

def bench(args,rest):
    pack=json.loads((ROOT/f'theta_{args.uid}.json').read_text())
    an,gn=np.array(pack['nominal_alpha']),np.array(pack['nominal_gain'])
    with np.load(f'data/cce/probes/probe_{args.uid}_P00_nominal.npz') as z:
        gates=gate_values(pack,z['targets'])
    def factory(method,spec,hat,theta_nominal,**kw):
        assert method in ('v4_gate2','v4_nogate2')
        ex=RelativeExecutor(RelativeTheta.from_fit(pack['plants'][spec.pid]),an,gn,rho_max=2.,
                            h=gates['H'][spec.pid],h0=gates['H0'],gate=method=='v4_gate2')
        return SimAdapter(ex),False
    module=importlib.import_module('cce.bench_'+args.bench)
    module.build_arm=factory
    sys.argv=[sys.argv[0],'--uid',args.uid,*rest]
    module.main()

def aggregate(root):
    methods=['v4_gate2','v4_nogate2'];out={}
    for uid in ['panda','xarm6_robotiq']:
        out[uid]={}
        pack=json.loads((ROOT/f'theta_{uid}.json').read_text())
        with np.load(f'data/cce/probes/probe_{uid}_P00_nominal.npz') as z:
            gates=gate_values(pack,z['targets'])
        for name,n in [('tracking',20),('replay',50)]:
            paths=sorted(root.glob(f'{name}_{uid}_*.json'))
            assert len(paths)==13
            rows=[r for p in paths for r in json.loads(p.read_text())]
            key='traj_seed' if name=='tracking' else 'seed'
            seeds=sorted({r[key] for r in rows});assert len(seeds)==n
            keys=Counter((r['pid'],r['method'],r[key]) for r in rows)
            assert set(keys)==set(itertools.product([p.pid for p in PLANTS],methods,seeds)) and max(keys.values())==1
            field='rmse_pos_vs_intent_m' if name=='tracking' else 'success'
            cells={p.pid:{m:float(np.mean([r[field] for r in rows if r['pid']==p.pid and r['method']==m])) for m in methods} for p in PLANTS}
            for p,v in cells.items():
                v['gate_bypasses_arm']=gates['H'][p]<gates['H0']
                v['nogate_minus_gate']=v['v4_nogate2']-v['v4_gate2']
            out[uid][name]={'n_rows':len(rows),'field':field,'cells':cells,
                             'macro_nogate_minus_gate':float(np.mean([v['nogate_minus_gate'] for v in cells.values()]))}
        out[uid]['gate']=gates
    result={'status':'complete','analysis':'post-hoc mechanism diagnostic; no tuning, not a new gate pass',
            'robots':out,'raw_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('tracking_*.json')} |
                                    {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('replay_*.json')}}
    (root/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# v4r1 米制门的事后机制诊断','',
           '固定ρ=2，唯一变化为机械臂H门开/关；CPU无策略。结果不用于更改正在运行的LIBERO冻结参数。', '',
           '| 机器人 | plant | H门退回 | 门开意图RMSE | 门关意图RMSE | 门开回放SR | 门关回放SR |',
           '|---|---|---|---:|---:|---:|---:|']
    for uid,r in out.items():
        for p,t in r['tracking']['cells'].items():
            s=r['replay']['cells'][p]
            lines.append(f"| {uid} | {p} | {t['gate_bypasses_arm']} | {t['v4_gate2']:.6f} | {t['v4_nogate2']:.6f} | {s['v4_gate2']:.3f} | {s['v4_nogate2']:.3f} |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({u:{b:r[b]['macro_nogate_minus_gate'] for b in ['tracking','replay']} for u,r in out.items()}),flush=True)

def run(args):
    root=Path(args.out_dir);root.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for uid,name,shard in itertools.product(['panda','xarm6_robotiq'],['tracking','replay'],range(13)):
        stem=f'{name}_{uid}_{shard}'
        jobs.append({'name':stem,'cmd':['bash','cce/env.sh',__file__,'bench','--bench',name,'--uid',uid,
                    '--methods','v4_gate2,v4_nogate2','--theta-dir','data/cce/theta_sat','--shard',str(shard),'--nshards','13',
                    '--out',str(root/(stem+'.json'))]})
    tracked=['cce/diagnose_gate_v4r1.py','docs/prereg/CCE_gate_diagnostic_v4r1.md',
             'cce/bench_relative_v4r1.py','cce/libero_exec_v4r1.py','cce/bench_tracking.py','cce/bench_replay.py']
    inputs=tracked+[str(ROOT/f'theta_{uid}.json') for uid in ['panda','xarm6_robotiq']]
    (root/'plan.json').write_text(json.dumps({'started':datetime.datetime.now().astimezone().isoformat(),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'expected_rows':7280,
        'jobs':jobs,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs}},indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def work(j):
        start=time.monotonic()
        with (root/(j['name']+'.log')).open('w') as log:
            proc=subprocess.run(j['cmd'],env=env,stdout=log,stderr=subprocess.STDOUT)
        (root/(j['name']+'.exit')).write_text(str(proc.returncode)+'\n')
        return {'name':j['name'],'exit':proc.returncode,'wall_s':time.monotonic()-start}
    done=[];rc=1
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for f in as_completed([pool.submit(work,j) for j in jobs]):
                done.append(f.result())
                (root/'status.json').write_text(json.dumps({'done':len(done),'total':52,'jobs':done},indent=2)+'\n')
                print(json.dumps(done[-1]),flush=True)
        assert all(r['exit']==0 for r in done)
        aggregate(root);rc=0
    finally:(root/'supervisor.exit').write_text(str(rc)+'\n')

def main():
    # --out belongs to the historical benchmark, never abbreviates --out-dir.
    ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('mode',choices=['bench','run'])
    ap.add_argument('--uid');ap.add_argument('--bench',choices=['tracking','replay']);ap.add_argument('--out-dir')
    args,rest=ap.parse_known_args()
    bench(args,rest) if args.mode=='bench' else run(args)

if __name__=='__main__':main()
