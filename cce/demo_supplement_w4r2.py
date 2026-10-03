"""Finite supplemental demo collection; original data gate failures stay intact."""
import _bootstrap  # noqa: F401
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import requests
from cce.freeze_v4r1 import sha, validate_freeze
from cce.freeze_demos_v4r1 import validate_demo_freeze

ROOT=Path('/opt/cce/data/cce/demo_supplement_w4r2_20260911')
OLD=Path('/opt/cce/data/cce/policyside_v4r1_20260909/collection')
FREEZE='cce/results/demo_supplement_w4r2_spec.json'
PARENT='cce/results/libero_family_spec_v4r1_fallback.json'
DEMOS='cce/results/libero_demo_spec_v4r1.json'


def read(p):return json.loads(Path(p).read_text())


def write(p,s):Path(p).write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')


def validate():
    s=read(FREEZE);assert s['status']=='FROZEN'
    validate_freeze(PARENT);validate_demo_freeze(DEMOS)
    for p,h in s['inputs_sha256'].items():assert sha(p)==h,p
    return s


def freeze():
    parent=validate_freeze(PARENT,weights=True);validate_demo_freeze(DEMOS)
    ROOT.mkdir(exist_ok=False)
    files=[PARENT,DEMOS,'cce/demo_supplement_w4r2.py','docs/prereg/CCE_demo_supplement_w4r2.md']
    jobs=[];previous={}
    for plant in ['D2_t400','MX03']:
        old=OLD/plant
        previous[plant]=read(old/'dataset_meta.json')['datalist']
        files+=previous[plant]+[str(old/name) for name in ['dataset_meta.json','collection.json','attempts.jsonl','meta.json']]
        for task,n in read(old/'collection.json')['per_task'].items():
            if n<5:jobs.append(dict(plant=plant,task=int(task),previous=n,required=5-n))
    assert len(jobs)==7 and sum(x['required'] for x in jobs)==17
    from cce.libero_collect_plant_demos_v4r1 import configure
    _,_,_,_,suite=configure()
    state_hashes={}
    for task in range(10):
        states=suite.get_task_init_states(task);assert len(states)>=40
        hh={str(e):hashlib.sha256(np.asarray(states[e]).tobytes()).hexdigest() for e in range(40)}
        assert len(set(hh.values()))==40
        state_hashes[str(task)]=hh
    write(FREEZE,dict(status='FROZEN',date=datetime.datetime.now().astimezone().isoformat(),implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      jobs=jobs,candidates=list(range(20,40)),max_attempts=140,previous=previous,initial_state_sha256=state_hashes,server=parent['server'],inputs_sha256={p:sha(p) for p in files}))
    validate();print('Supplement frozen; all initial-state hashes disjoint.')


def worker(i):
    s=validate();job=s['jobs'][i];plant,task=job['plant'],job['task'];out=ROOT/'run'/f'job{i}';out.mkdir()
    from cce.libero_collect_plant_demos_v4r1 import configure,RecordingEval,save_episode
    from cce import libero_family_v3 as F
    _,hats,LC,ev,suite=configure()
    info=requests.get('http://127.0.0.1:8067/ready',timeout=5).json()
    assert info['lora_path']==s['server']['lora_path'] and info['supports_request_seed']
    recorder=RecordingEval(ev,LC._flip_agentview)
    client=F.ChunkClient('127.0.0.1',8067);client.suite_name='libero_spatial';client.verify_repeat=True
    selected=[];rows=[]
    with (out/'attempts.jsonl').open('x') as f:
        for ep in s['candidates']:
            hh=hashlib.sha256(np.asarray(suite.get_task_init_states(task)[ep]).tobytes()).hexdigest()
            assert hh==s['initial_state_sha256'][str(task)][str(ep)]
            row=F.run_episode(recorder,suite,task,ep,F.BY_ID_V3[plant],'oracle_v4',client,hats[plant],F.theta_from_hat(hats[F.NOMINAL_PID]),800,1.,1.)
            row.update(initial_state_sha256=hh,h5_path=None)
            if row['success']:
                p=out/f'{plant}__t{task}__e{ep}.h5';save_episode(p,recorder.record,row)
                row.update(h5_path=str(p),h5_sha256=sha(p));selected.append(str(p))
            rows.append(row);f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps({'plant':plant,'task':task,'ep':ep,'success':row['success'],'collected':len(selected),'required':job['required']}),flush=True)
            if len(selected)==job['required']:break
    write(out/'result.json',dict(job=job,attempts=len(rows),selected=selected,status='complete' if len(selected)==job['required'] else 'insufficient_successes'))


def aggregate(env):
    s=validate();results={};attempts=0
    for plant in s['previous']:
        out=ROOT/'run'/plant;out.mkdir()
        paths=list(s['previous'][plant]);counts=dict(read(OLD/plant/'collection.json')['per_task'])
        for i,j in enumerate(s['jobs']):
            if j['plant']!=plant:continue
            job=ROOT/'run'/f'job{i}';r=read(job/'result.json');paths+=r['selected'];counts[str(j['task'])]+=len(r['selected']);attempts+=r['attempts']
            for row in map(json.loads,(job/'attempts.jsonl').read_text().splitlines()):
                if row['h5_path']:assert sha(row['h5_path'])==row['h5_sha256']
        assert all(v<=5 for v in counts.values()) and len(paths)==sum(counts.values())
        meta=read(OLD/plant/'dataset_meta.json');meta['datalist']=paths;write(out/'dataset_meta.json',meta)
        with (out/'handler_audit.log').open('x') as log:
            p=subprocess.run(['bash','cce/env.sh','cce/audit_demo_handler_v4r1.py','--demo-dir',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/'handler_audit.exit').write_text(str(p.returncode)+'\n');assert p.returncode==0
        results[plant]=dict(status='complete' if all(v==5 for v in counts.values()) else 'insufficient_successes',per_task=counts,total=len(paths),meta=str(out/'dataset_meta.json'),handler_audit=read(out/'handler_audit.json')['status'],h5_sha256={p:sha(p) for p in paths})
    assert attempts<=s['max_attempts']
    out=dict(status='execution_complete',new_attempts=attempts,max_attempts=140,plants=results,training_started=False)
    write(ROOT/'results.json',out);write('cce/results/CCE_demo_supplement_w4r2.json',out)


def run():
    s=validate();r=ROOT/'run';r.mkdir(exist_ok=False);server=None;code=1
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',MUJOCO_EGL_DEVICE_ID='7',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    write(r/'plan.json',dict(started=datetime.datetime.now().astimezone().isoformat(),freeze_sha256=sha(FREEZE),max_attempts=140))
    try:
        with (r/'server.log').open('x') as log:
            server=subprocess.Popen(['bash','cce/env.sh','cce/serve_xvla_v4r1.py','--model-path',s['server']['model_path'],'--lora-path',s['server']['lora_path'],'--port','8067'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(r/'owned_server.json',dict(pid=server.pid))
        for _ in range(120):
            if server.poll() is not None:raise RuntimeError('server died before ready')
            try:
                info=requests.get('http://127.0.0.1:8067/ready',timeout=2).json()
                assert info['lora_path']==s['server']['lora_path'];break
            except requests.RequestException:time.sleep(2)
        else:raise RuntimeError('server readiness timeout')
        def job(i):
            with (r/f'job{i}.log').open('x') as log:
                p=subprocess.run(['bash','cce/env_libero.sh','cce/demo_supplement_w4r2.py','--worker',str(i)],env=env,stdout=log,stderr=subprocess.STDOUT)
            (r/f'job{i}.exit').write_text(str(p.returncode)+'\n')
            if p.returncode:raise RuntimeError(f'job{i} exit {p.returncode}')
        with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(job,range(len(s['jobs']))))
        aggregate(env);code=0
    except BaseException as e:write(r/'failure.json',dict(error=repr(e)));raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        (r/'supervisor.exit').write_text(str(code)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--worker',type=int);a=ap.parse_args()
    if a.freeze:freeze()
    elif a.worker is not None:worker(a.worker)
    elif a.run:run()
