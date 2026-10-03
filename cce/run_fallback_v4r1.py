"""Detached finite two-phase queue with frozen inputs, audit and owned-server cleanup."""
import _bootstrap  # noqa: F401
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from cce.freeze_v4r1 import validate_freeze, sha


def atomic(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def stop_owned(services):
    results=[]
    for s in services:
        pid=s['pid']
        try:
            proc=Path(f'/proc/{pid}')
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            start=(proc/'stat').read_text().split()[21]
            if start != s['start_ticks'] or 'cce/serve_xvla_v4r1.py' not in cmd or f"--port {s['port']} " not in cmd:
                results.append({'pid':pid,'action':'identity_mismatch_no_signal'});continue
            os.kill(pid,signal.SIGTERM)
            results.append({'pid':pid,'action':'SIGTERM_owned_server'})
        except FileNotFoundError:
            results.append({'pid':pid,'action':'already_exited'})
    return results

def update_gpu_log(root, label):
    snap=subprocess.check_output(['nvidia-smi'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv'],text=True)
    pids=[]
    for line in apps.splitlines()[1:]:
        try:pids.append(str(int(line.split(',')[0])))
        except ValueError:pass
    owners=subprocess.check_output(['ps','-o','pid,ppid,user,args','-p',','.join(pids)],text=True) if pids else 'no compute processes'
    log=Path('/opt/runtime/all_users/GPU_Usage_Log.md')
    note=f"\n## {datetime.datetime.now().astimezone().isoformat()} anonymous CCE v4r1 — {label}\n\n本会话实验队列：`{root}`。GPU 0/5/7服务已执行收尾；以下实时输出为准，不宣称其他用户进程空闲。\n\n```text\n{snap}\n{apps}\n{owners}\n```\n"
    (root/'gpu_final.txt').write_text(note)
    with log.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        old=f.read();f.seek(0);f.write('# GPU 使用共享记录\n'+note+old);f.truncate()
        fcntl.flock(f,fcntl.LOCK_UN)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--freeze',required=True);ap.add_argument('--out-dir',required=True)
    ap.add_argument('--services',required=True)
    args=ap.parse_args()
    root=Path(args.out_dir);root.mkdir(parents=True,exist_ok=False);(root/'jobs').mkdir()
    services=json.loads(Path(args.services).read_text())
    rc=1
    try:
        spec=validate_freeze(args.freeze,weights=True)
        subprocess.run(['git','ls-files','--error-unmatch',args.freeze],check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','diff','--exit-code','HEAD','--',args.freeze],check=True,stdout=subprocess.DEVNULL)
        import requests
        for service in services:
            info=requests.get(f"http://127.0.0.1:{service['port']}/ready",timeout=5).json()
            for k in ['model_path','lora_path','lora_merged','supports_request_seed','num_actions']:
                assert info[k]==spec['server'][k], f'Server mismatch: {k}'
        jobs=[]
        for method,n in spec['episodes'].items():
            for pid in spec['plant_ids']:
                for shard in range(4):
                    jobs.append({'name':f'{pid}__{method}__s{shard}','pid':pid,'method':method,'n':n,'shard':shard,
                                 'expected_rows':sum(1 for j in range(10*n) if j%4==shard)})
        atomic(root/'plan.json',{'started':datetime.datetime.now().astimezone().isoformat(),
                               'freeze':args.freeze,'freeze_sha256':sha(args.freeze),'jobs':jobs,'services':services,
                               'expected_rows':spec['expected_episodes']})
        lock=threading.Lock();failed=threading.Event();done=[]
        def phase(name, selected):
            work=queue.Queue()
            for j in selected:work.put(j)
            def worker(service):
                while not failed.is_set():
                    try:j=work.get_nowait()
                    except queue.Empty:return
                    output=root/'jobs'/f"{j['name']}.jsonl"
                    command=['bash','cce/env_libero.sh','cce/libero_family_v3.py','--stage','fallback',
                             '--freeze',args.freeze,'--pack',spec['pack'],'--plants',j['pid'],'--methods',j['method'],
                             '--episodes',str(j['n']),'--port',str(service['port']),'--shard',str(j['shard']),
                             '--nshards','4','--out',str(output),'--trace-first']
                    start=time.monotonic()
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(service['gpu']),MUJOCO_EGL_DEVICE_ID=str(service['gpu']),
                             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                    with output.with_suffix('.log').open('w') as log:
                        proc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    code=proc.returncode
                    if code==0:
                        try:
                            rows=[json.loads(l) for l in output.read_text().splitlines()]
                            expected={(t,e) for i,(t,e) in enumerate((t,e) for t in spec['tasks'] for e in range(j['n'])) if i%4==j['shard']}
                            assert len(rows)==j['expected_rows'] and {(r['task_id'],r['ep']) for r in rows}==expected
                            assert all(r['pid']==j['pid'] and r['method']==j['method'] for r in rows)
                        except Exception as error:
                            code=90
                            output.with_suffix('.audit_error.txt').write_text(repr(error)+'\n')
                    output.with_suffix('.exit').write_text(str(code)+'\n')
                    if code:failed.set()
                    with lock:
                        item={**j,'exit':code,'gpu':service['gpu'],'wall_s':time.monotonic()-start}
                        done.append(item)
                        atomic(root/'status.json',{'phase':name,'done_jobs':len(done),'total_jobs':len(jobs),
                                                  'completed_rows':sum(x['expected_rows'] for x in done if x['exit']==0),
                                                  'expected_rows':spec['expected_episodes'],'failed':failed.is_set(),'jobs':done})
                        print(json.dumps(item),flush=True)
            with ThreadPoolExecutor(max_workers=12) as pool:
                futures=[pool.submit(worker,s) for s in services for _ in range(4)]
                for future in futures:future.result()
            if failed.is_set():raise RuntimeError(f'{name}: a job failed; remaining queue cancelled')
        phase('naive_first',[j for j in jobs if j['method']=='naive'])
        phase('comparisons',[j for j in jobs if j['method']!='naive'])
        (root/'execution.exit').write_text('0\n')
        with (root/'aggregate.log').open('w') as log:
            subprocess.run(['bash','cce/env_libero.sh','cce/aggregate_fallback_v4r1.py','--run-dir',str(root),
                            '--freeze',args.freeze],check=True,stdout=log,stderr=subprocess.STDOUT)
        rc=0
    except BaseException as error:
        (root/'failure.txt').write_text(repr(error)+'\n')
        if not (root/'execution.exit').exists():(root/'execution.exit').write_text('1\n')
        raise
    finally:
        atomic(root/'cleanup.json',stop_owned(services))
        time.sleep(3)
        try:update_gpu_log(root,'完成' if rc==0 else '失败后停止')
        except Exception as error:(root/'gpu_log_error.txt').write_text(repr(error)+'\n')
        (root/'supervisor.exit').write_text(str(rc)+'\n')

if __name__=='__main__':
    main()
