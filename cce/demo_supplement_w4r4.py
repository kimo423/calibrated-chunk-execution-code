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
import h5py
import re
from collections import Counter
from cce.freeze_v4r1 import sha, validate_freeze
from cce.freeze_demos_v4r1 import validate_demo_freeze

ROOT=Path('/opt/cce/data/cce/demo_supplement_w4r4_20260911')
OLD=Path('/opt/cce/data/cce/demo_supplement_w4r2_20260911/run')
FREEZE='cce/results/demo_supplement_w4r4_spec.json'
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
    previous_result=read(OLD.parent/'results.json')
    assert (OLD/'supervisor.exit').read_text().strip()=='0'
    assert previous_result['plants']['D2_t400']['total']==44
    for p,h in previous_result['plants']['D2_t400']['h5_sha256'].items():assert sha(p)==h,p
    ROOT.mkdir(exist_ok=False)
    files=[PARENT,DEMOS,'cce/demo_supplement_w4r4.py','docs/prereg/CCE_demo_supplement_w4r4.md']
    jobs=[];previous={}
    for plant in ['D2_t400']:
        old=OLD/plant
        previous[plant]=read(old/'dataset_meta.json')['datalist']
        files+=previous[plant]+[str(old/'dataset_meta.json'),str(OLD.parent/'results.json'),'cce/results/demo_supplement_w4r2_spec.json']
        for task,n in read(OLD.parent/'results.json')['plants'][plant]['per_task'].items():
            if n<5:jobs.append(dict(plant=plant,task=int(task),previous=n,required=5-n))
    assert len(jobs)==2 and sum(x['required'] for x in jobs)==6
    validation = {}
    for task in range(10):
        paths=[p for p in previous['D2_t400'] if f'__t{task}__' in p]
        validation[str(task)]=max(paths,key=lambda p:int(re.search(r'__e(\d+)\.h5$',p)[1]))
    from cce.libero_collect_plant_demos_v4r1 import configure
    _,_,_,_,suite=configure()
    state_hashes={}
    for task in range(10):
        states=suite.get_task_init_states(task);assert len(states)==50
        hh={str(e):hashlib.sha256(np.asarray(states[e]).tobytes()).hexdigest() for e in range(50)}
        assert len(set(hh.values()))==50
        state_hashes[str(task)]=hh
    for job in jobs:
        task=job['task'];held=int(re.search(r'__e(\d+)\.h5$',validation[str(task)])[1])
        pool=[ep for ep in range(10,50) if ep!=held]
        job['validation_ep']=held;job['schedule']=[]
        rng=np.random.default_rng(20260911+task)
        while len(job['schedule'])<200:job['schedule']+=rng.permutation(pool).tolist()
        job['schedule']=job['schedule'][:200]
    write(FREEZE,dict(status='FROZEN',date=datetime.datetime.now().astimezone().isoformat(),implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      jobs=jobs,max_attempts=400,max_attempts_per_task=200,max_successes_per_training_initial_state=2,validation=validation,previous=previous,initial_state_sha256=state_hashes,server=parent['server'],inputs_sha256={p:sha(p) for p in files}))
    validate();print('Supplement frozen; all initial-state hashes disjoint.')


class SeededClient:
    """Wrap the frozen client; alter only its diffusion request seed namespace."""
    def __new__(cls, *args):
        from cce.libero_family_v3 import ChunkClient
        class Client(ChunkClient):
            def query(self, obs, goal):
                original=self.episode_key
                self.episode_key=f"{original}:demo_w4r4:trial{self.trial}"
                try:
                    a=super().query(obs,goal)
                    if self.query_index==1:self.first_chunk_sha256=hashlib.sha256(a.tobytes()).hexdigest()
                    return a
                finally:self.episode_key=original
        return Client(*args)


def trajectory_hash(a):
    return hashlib.sha256(np.asarray(a,dtype=np.float64).tobytes()).hexdigest()


def worker(i):
    s=validate();job=s['jobs'][i];plant,task=job['plant'],job['task'];out=ROOT/'run'/f'job{i}';out.mkdir()
    from cce.libero_collect_plant_demos_v4r1 import configure,RecordingEval,save_episode
    from cce import libero_family_v3 as F
    _,hats,LC,ev,suite=configure()
    info=requests.get('http://127.0.0.1:8067/ready',timeout=5).json()
    assert info['lora_path']==s['server']['lora_path'] and info['supports_request_seed']
    recorder=RecordingEval(ev,LC._flip_agentview)
    client=SeededClient('127.0.0.1',8067);client.suite_name='libero_spatial';client.verify_repeat=True
    selected=[];rows=[];used_hashes=set();initial_counts=Counter()
    for p in s['previous'][plant]:
        with h5py.File(p) as f:
            if int(f.attrs['task_id'])==task:
                used_hashes.add(trajectory_hash(f['abs_action_6d'][()]))
                initial_counts[int(f.attrs['initial_state_index'])]+=1
    first_seeds=set()
    with (out/'attempts.jsonl').open('x') as f:
        for trial,ep in enumerate(job['schedule']):
            if initial_counts[ep]>=s['max_successes_per_training_initial_state']:continue
            assert ep!=job['validation_ep'] and 10<=ep<50
            hh=hashlib.sha256(np.asarray(suite.get_task_init_states(task)[ep]).tobytes()).hexdigest()
            assert hh==s['initial_state_sha256'][str(task)][str(ep)]
            client.trial=trial
            key=f'libero_spatial:{task}:{ep}:demo_w4r4:trial{trial}'
            first=int.from_bytes(hashlib.sha256(f'{key}:0'.encode()).digest()[:4],'little')%(2**31-1)
            assert first not in first_seeds;first_seeds.add(first)
            row=F.run_episode(recorder,suite,task,ep,F.BY_ID_V3[plant],'oracle_v4',client,hats[plant],F.theta_from_hat(hats[F.NOMINAL_PID]),800,1.,1.)
            row.update(initial_state_sha256=hh,h5_path=None,accepted=False,trial=trial,seed_episode_key=key,first_request_seed=first,first_chunk_sha256=client.first_chunk_sha256)
            if row['success']:
                digest=trajectory_hash(recorder.record.actions)
                row['trajectory_sha256']=digest
                if digest in used_hashes:row['rejection_reason']='exact_duplicate_action_trajectory'
                else:
                    p=out/f'{plant}__t{task}__e{ep}__s{trial}.h5';save_episode(p,recorder.record,row)
                    with h5py.File(p,'r+') as hf:
                        hf.attrs.update(collector_sha256=sha(__file__),seed_episode_key=key,first_request_seed=first,trajectory_sha256=digest,split_role='train')
                    row.update(h5_path=str(p),h5_sha256=sha(p),accepted=True);selected.append(str(p));used_hashes.add(digest);initial_counts[ep]+=1
            rows.append(row);f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps({'plant':plant,'task':task,'ep':ep,'trial':trial,'success':row['success'],'accepted':row['accepted'],'collected':len(selected),'required':job['required']}),flush=True)
            if len(selected)==job['required']:break
    write(out/'result.json',dict(job=job,attempts=len(rows),selected=selected,status='complete' if len(selected)==job['required'] else 'insufficient_successes'))


def aggregate(env):
    s=validate();results={};attempts=0
    for plant in s['previous']:
        out=ROOT/'run'/plant;out.mkdir()
        paths=list(s['previous'][plant]);counts=dict(read(OLD.parent/'results.json')['plants'][plant]['per_task'])
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
        training=[p for p in paths if p not in s['validation'].values()];validation=list(s['validation'].values())
        train_groups=set();val_groups=set();digests=[]
        for p in paths:
            with h5py.File(p) as hf:
                group=(int(hf.attrs['task_id']),str(hf.attrs['initial_state_sha256']))
                (val_groups if p in validation else train_groups).add(group)
                digests.append((group[0],trajectory_hash(hf['abs_action_6d'][()])))
        assert not train_groups & val_groups and len(set(digests))==len(paths)
        results[plant]=dict(split=dict(train=training,validation=validation,initial_state_groups_disjoint=True),status='complete' if all(v==5 for v in counts.values()) else 'insufficient_successes',per_task=counts,total=len(paths),meta=str(out/'dataset_meta.json'),handler_audit=read(out/'handler_audit.json')['status'],h5_sha256={p:sha(p) for p in paths})
    assert attempts<=s['max_attempts']
    out=dict(status='execution_complete',new_attempts=attempts,max_attempts=400,plants=results,training_started=False)
    write(ROOT/'results.json',out);write('cce/results/CCE_demo_supplement_w4r4.json',out)


def run():
    s=validate();r=ROOT/'run';r.mkdir(exist_ok=False);server=None;code=1
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',MUJOCO_EGL_DEVICE_ID='7',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    write(r/'plan.json',dict(started=datetime.datetime.now().astimezone().isoformat(),freeze_sha256=sha(FREEZE),max_attempts=400))
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
                p=subprocess.run(['bash','cce/env_libero.sh','cce/demo_supplement_w4r4.py','--worker',str(i)],env=env,stdout=log,stderr=subprocess.STDOUT)
            (r/f'job{i}.exit').write_text(str(p.returncode)+'\n')
            if p.returncode:raise RuntimeError(f'job{i} exit {p.returncode}')
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(job,range(len(s['jobs']))))
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
