"""Frozen object-suite transfer study with complete-grid descriptive analysis."""
import _bootstrap  # noqa: F401
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import numpy as np
from cce.freeze_v4r1 import sha
from cce.run_object_g0_v4r1 import validate as validate_g0
from cce.run_policyside_training_v4r1 import gpu_snapshot
from cce.aggregate_secondary_v4r1 import interval

ROOT=Path('/opt/cce/data/cce/object_v4r1_20260909')
G0='cce/results/libero_object_g0_spec_v4r1.json'
METHODS=['naive','slow20','ours_v4','ours_v4_slow15']
PLANTS=['D1_d250','D2_t400','D3_g10','MX01','MX02','MX03','D2_g130','MX04']


def validate(path):
    spec=json.loads(Path(path).read_text());assert spec['status']=='FROZEN_OBJECT_SUBSET'
    validate_g0(G0)
    for p,h in spec['inputs_sha256'].items():assert sha(p)==h,p
    return spec


def freeze(path):
    parent=validate_g0(G0)
    assert (ROOT/'g0/supervisor.exit').read_text().strip()=='0'
    assert json.loads((ROOT/'g0/results.json').read_text())['G0_pass']
    assert json.loads((ROOT/'subset_smoke_audit.json').read_text())['status']=='passed'
    files=['cce/run_object_subset_v4r1.py','cce/libero_object_subset_v4r1.py',G0,
           'cce/aggregate_secondary_v4r1.py','cce/run_policyside_training_v4r1.py',
           'docs/prereg/CCE_object_subset_v4r1.md']
    files += [str(ROOT/p) for p in ['g0/results.json','subset_smoke_audit.json','nominal_theta/theta_hat_libero_panda.json']]
    spec={'status':'FROZEN_OBJECT_SUBSET','frozen_at':datetime.datetime.now().astimezone().isoformat(),
          'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'plants':PLANTS,'methods':METHODS,'tasks':list(range(10)),'episodes':5,'total_episodes':1600,'nshards':32,
          'adapter':parent['adapter'],'pack':parent['pack'],'inputs_sha256':{p:sha(p) for p in files}}
    with Path(path).open('x') as f:f.write(json.dumps(spec,indent=2)+'\n')
    validate(path);print('Frozen object subset',sha(path))


def summarize(rows,spec):
    counts=Counter((r['pid'],r['method'],r['task_id'],r['ep']) for r in rows)
    expected={(p,m,t,e) for p in PLANTS for m in METHODS for t in range(10) for e in range(5)}
    assert set(counts)==expected and set(counts.values())=={1} and len(rows)==1600
    groups={(p,m,t):[] for p in PLANTS for m in METHODS for t in range(10)}
    for r in rows:
        assert r['suite']=='libero_object' and r['success'] in (0,1) and 1<=r['steps']<=800
        assert r['success'] or r['steps']==800
        assert np.isfinite(r['tracking_rmse_m']) and r['tracking_rmse_m']>=0
        freq={'slow20':60,'ours_v4_slow15':45}.get(r['method'],30)
        assert r['policy_queries']==(r['steps']+freq-1)//freq
        if r['success']:assert abs(r['time_to_success_s']-.05*r['steps'])<1e-9
        groups[r['pid'],r['method'],r['task_id']].append(r)
    cells={p:{m:{'sr':float(np.mean([np.mean([r['success'] for r in groups[p,m,t]]) for t in range(10)]))}
              for m in METHODS} for p in PLANTS}
    g0=json.loads((ROOT/'g0/results.json').read_text())
    nominal=[]
    for p,h in g0['raw_sha256'].items():
        assert sha(p)==h
        nominal.extend(json.loads(l) for l in Path(p).read_text().splitlines())
    nominal=[r for r in nominal if r['ep']<5];assert len(nominal)==50
    nominal_sr=float(np.mean([r['success'] for r in nominal]))
    rpop=[p for p in PLANTS if nominal_sr-cells[p]['naive']['sr']>.10+1e-12]
    common=[];times={m:{} for m in METHODS}
    for p in PLANTS:
        pt={m:[] for m in METHODS}
        for t in range(10):
            values={m:[r['time_to_success_s'] for r in groups[p,m,t] if r['success']] for m in METHODS}
            if all(values.values()):
                common.append({'pid':p,'task_id':t,'n_success':{m:len(v) for m,v in values.items()}})
                for m,v in values.items():pt[m].append(float(np.mean(v)))
        for m,v in pt.items():
            if v:times[m][p]=float(np.mean(v))
    macro={}
    for m in METHODS:
        rv={p:(cells[p][m]['sr']-cells[p]['naive']['sr'])/(nominal_sr-cells[p]['naive']['sr']) for p in rpop}
        macro[m]={'sr':float(np.mean([cells[p][m]['sr'] for p in PLANTS])),
                  'R':interval(list(rv.values())),'R_by_plant':rv,
                  'T_common_s':float(np.mean(list(times[m].values()))) if times[m] else None}
    differences={f'{a}__vs__{b}':interval([cells[p][a]['sr']-cells[p][b]['sr'] for p in PLANTS])
                 for a,b in [('ours_v4','naive'),('ours_v4_slow15','slow20')]}
    return {'status':'complete','n':1600,'cells':cells,'macro':macro,'differences':differences,
            'nominal_sr_matched5':nominal_sr,'R_population':rpop,'common_success_cells':common,
            'time_by_plant':times,'inference':'descriptive only; frozen spatial executor transferred without retuning'}


def main():
    ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('--freeze-out');ap.add_argument('--freeze');args=ap.parse_args()
    if args.freeze_out:freeze(args.freeze_out);return
    spec=validate(args.freeze)
    subprocess.run(['git','ls-files','--error-unmatch',args.freeze],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--exit-code','HEAD','--',args.freeze],check=True,stdout=subprocess.DEVNULL)
    root=ROOT/'subset';root.mkdir(exist_ok=False)
    owned=json.loads((ROOT/'subset_service_launcher.json').read_text());code=1
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',MUJOCO_EGL_DEVICE_ID='7',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        def shard(i):
            cmd=['bash','cce/env_libero.sh','cce/libero_object_subset_v4r1.py','--freeze',args.freeze,
                 '--shard',str(i),'--nshards',str(spec['nshards']),'--port','8057','--out',str(root/f'shard{i}.jsonl')]
            with (root/f'shard{i}.log').open('x') as log:
                p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (root/f'shard{i}.exit').write_text(str(p.returncode)+'\n');assert p.returncode==0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(shard,i) for i in range(spec['nshards'])]
            for f in futures:f.result()
        paths=sorted(root.glob('shard*.jsonl'));rows=[]
        for p in paths:
            meta=json.loads(p.with_suffix('.meta.json').read_text())
            assert meta['source_sha256']==sha('cce/libero_object_subset_v4r1.py') and meta['freeze_sha256']==sha(args.freeze)
            assert meta['pack_sha256']==sha(spec['pack']) and meta['server']['lora_path']==spec['adapter']['path']
            rows.extend(json.loads(l) for l in p.read_text().splitlines())
        out=summarize(rows,spec);out.update(freeze_sha256=sha(args.freeze),raw_sha256={str(p):sha(p) for p in paths})
        (root/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
        lines=['# LIBERO-object 固定执行器迁移结果','','1,600唯一键与原始SHA审计通过；仅次级描述，不改变主检验。','',
               '| 方法 | 8plant宏SR | 宏R | 共同成功T(s) |','|---|---:|---:|---:|']
        for m,v in out['macro'].items():lines.append(f"| {m} | {v['sr']:.4f} | {v['R']['mean']} | {v['T_common_s']} |")
        lines+=['','R使用标称G0匹配前5集，人群与时间共同成功任务格名单见JSON。条件成功耗时不代表总体效率。','']
        (root/'report.md').write_text('\n'.join(lines));print(json.dumps(out['macro']),flush=True);code=0
    except BaseException as error:
        (root/'failure.txt').write_text(repr(error)+'\n');raise
    finally:
        pid=owned['pid'];proc=Path(f'/proc/{pid}')
        if proc.exists():
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if (proc/'stat').read_text().split()[21]==owned['start_ticks'] and 'serve_xvla_v4r1.py' in cmd and '--port 8057' in cmd:
                os.killpg(pid,signal.SIGTERM)
        (root/'supervisor.exit').write_text(str(code)+'\n')
        gpu_snapshot(root,'object_subset')


if __name__=='__main__':main()
