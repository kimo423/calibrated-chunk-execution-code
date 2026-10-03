"""Two finite train -> offline gate -> conditional online evaluation chains."""
import _bootstrap  # noqa: F401
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import requests
from cce.freeze_training_v4r1 import validate_training_freeze, sha


def run(command, logfile, env):
    with logfile.open('x') as log:
        proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    logfile.with_suffix('.exit').write_text(str(proc.returncode)+'\n')
    if proc.returncode: raise RuntimeError(f'{logfile}: exit {proc.returncode}')


def gpu_snapshot(root, mode):
    snap=subprocess.check_output(['nvidia-smi'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv'],text=True)
    pids=[str(int(l.split(',')[0])) for l in apps.splitlines()[1:] if l.strip()]
    owners=subprocess.check_output(['ps','-o','pid,user,args','-p',','.join(pids)],text=True) if pids else '无计算进程'
    note=f'## {datetime.datetime.now().astimezone().isoformat()} CCE {mode} 状态记录\n\nGPU具体占用与其他用户任务以下实时输出为准；本次链启动或收尾状态见标题及独立运行目录。\n\n```text\n{snap}\n{apps}\n{owners}\n```\n'
    (root/'gpu_final.txt').write_text(note)
    p=Path('/opt/runtime/all_users/GPU_Usage_Log.md')
    with p.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);old=f.read()
        archive=Path('/opt/cce/data/admin')/f'GPU_Usage_Log_training_{mode}_{time.time_ns()}.md'
        archive.write_text(old);f.seek(0);f.write('# GPU 使用共享记录\n\n'+note+f'\n历史：`{archive}`\n');f.truncate()


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--freeze', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--gpus', type=int, nargs=2, required=True)
    args = ap.parse_args(); assert len(set(args.gpus)) == 2
    spec = validate_training_freeze(args.freeze, data=True)
    subprocess.run(['git','ls-files','--error-unmatch',args.freeze],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--exit-code','HEAD','--',args.freeze],check=True,stdout=subprocess.DEVNULL)
    root = Path(args.out_dir); root.mkdir(parents=True, exist_ok=False)
    (root/'plan.json').write_text(json.dumps({'freeze':args.freeze,'freeze_sha256':sha(args.freeze),
                                             'started':datetime.datetime.now().astimezone().isoformat(),
                                             'chains':[{'mode':'ft_demo','gpu':args.gpus[0],'port':8070},
                                                       {'mode':'prompt_fit','gpu':args.gpus[1],'port':8077}]},indent=2)+'\n')

    def chain(mode, gpu, port):
        dest = root/mode; dest.mkdir()
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
                   OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        phase = 'initial_offline'; server = None; code = 1
        def status(value):
            temp=dest/'status.tmp';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(dest/'status.json')
        try:
            status({'phase':phase})
            run(['bash','cce/env.sh','cce/eval_policyside_offline_v4r1.py','--mode',mode,
                 '--samples',spec['validation_samples'],'--out',str(dest/'initial_offline.json')],dest/'initial_offline.log',env)
            phase='training';status({'phase':phase})
            run(['bash','cce/env.sh','cce/train_policyside_v4r1.py','--mode',mode,'--freeze',args.freeze,
                 '--meta',spec['train_meta'],'--out-dir',str(dest/'train')],dest/'training.log',env)
            audit=json.loads((dest/'train/training_audit.json').read_text())
            assert audit['status']=='passed' and audit['steps']==5000
            checkpoint=dest/'train/ckpt-5000'
            phase='offline';status({'phase':phase})
            run(['bash','cce/env.sh','cce/eval_policyside_offline_v4r1.py','--mode',mode,'--checkpoint',str(checkpoint),
                 '--samples',spec['validation_samples'],'--out',str(dest/'offline.json')],dest/'offline.log',env)
            offline=json.loads((dest/'offline.json').read_text())
            if not offline['G0_prime_pass']:
                result={'status':'G0_prime_failed','training_complete':True,'online_started':False,'offline':offline['horizons']}
                status(result);code=0;return result
            phase='online';status({'phase':phase,'G0_prime_pass':True})
            logfile=(dest/'server.log').open('x')
            command=['bash','cce/env.sh','cce/serve_policyside_v4r1.py','--mode',mode,'--checkpoint',str(checkpoint),'--port',str(port)]
            server=subprocess.Popen(command,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
            logfile.close()
            for _ in range(120):
                if server.poll() is not None: raise RuntimeError('Model server exited before ready')
                try:
                    info=requests.get(f'http://127.0.0.1:{port}/ready',timeout=2).json()
                    assert info['checkpoint']==str(checkpoint.resolve()) and info['policy_mode']==mode
                    break
                except requests.RequestException: time.sleep(2)
            else: raise RuntimeError('Model server readiness timeout')
            (dest/'online').mkdir()
            def shard(i):
                run(['bash','cce/env_libero.sh','cce/eval_policyside_online_v4r1.py','--freeze',args.freeze,
                     '--mode',mode,'--checkpoint',str(checkpoint),'--port',str(port),'--shard',str(i),
                     '--out',str(dest/f'online/shard{i}.jsonl')],dest/f'online/shard{i}.log',env)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(shard,i) for i in range(4)]
                for future in futures: future.result()
            paths=sorted((dest/'online').glob('*.jsonl'))
            rows=[json.loads(l) for p in paths for l in p.read_text().splitlines()]
            count=Counter((r['pid'],r['task_id'],r['ep']) for r in rows)
            expected={(p,t,e) for p in spec['evaluation_plants'] for t in range(10) for e in range(10)}
            assert set(count)==expected and set(count.values())=={1} and len(rows)==200
            for r in rows:
                assert r['success'] in (0,1) and 1<=r['steps']<=800
                assert r['success'] or r['steps']==800
                assert r['policy_queries']==(r['steps']+29)//30
            result={'status':'complete','training_complete':True,'G0_prime_pass':True,'online_episodes':200,
                    'sr':{p:sum(r['success'] for r in rows if r['pid']==p)/100 for p in spec['evaluation_plants']},
                    'task_sr':{p:{str(t):sum(r['success'] for r in rows if r['pid']==p and r['task_id']==t)/10 for t in range(10)} for p in spec['evaluation_plants']},
                    'raw_sha256':{str(p):sha(p) for p in paths}}
            (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n');status(result);code=0;return result
        except BaseException as error:
            status({'phase':phase,'status':'error','error':repr(error)});raise
        finally:
            if server is not None and server.poll() is None:
                # This process group was created above; do not signal any shared service.
                os.killpg(server.pid,signal.SIGTERM)
                try: server.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(server.pid,signal.SIGKILL);server.wait()
            (dest/'chain.exit').write_text(str(code)+'\n')
            try: gpu_snapshot(dest,mode)
            except Exception as error: (dest/'gpu_log_error.txt').write_text(repr(error)+'\n')
    code=1
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs={m:pool.submit(chain,m,g,p) for m,g,p in [('ft_demo',args.gpus[0],8070),('prompt_fit',args.gpus[1],8077)]}
            results={m:f.result() for m,f in jobs.items()}
        (root/'results.json').write_text(json.dumps(results,indent=2)+'\n');code=0
    finally:
        (root/'supervisor.exit').write_text(str(code)+'\n')


if __name__ == '__main__': main()
