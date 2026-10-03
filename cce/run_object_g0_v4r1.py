"""Pinned object adapter: finite 100-episode nominal positive control."""
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
import time
import requests
from cce.freeze_v4r1 import sha, validate_freeze

ROOT=Path('/opt/cce/data/cce/object_v4r1_20260909')
PARENT='cce/results/libero_family_spec_v4r1_fallback.json'


def validate(path):
    spec=json.loads(Path(path).read_text());assert spec['status']=='FROZEN_OBJECT_G0'
    validate_freeze(PARENT,weights=True)
    for p,h in spec['inputs_sha256'].items():assert sha(p)==h,p
    return spec


def freeze(path):
    parent=validate_freeze(PARENT)
    source=json.loads((ROOT/'adapter_source.json').read_text())
    files=['cce/run_object_g0_v4r1.py','cce/xvla_bridge.py','docs/prereg/CCE_object_g0_v4r1.md',PARENT,str(ROOT/'adapter_source.json')]
    files += [str(p) for p in Path(source['path']).glob('*') if p.is_file()]
    spec={'status':'FROZEN_OBJECT_G0','adapter':source,'model_path':parent['server']['model_path'],
          'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'frozen_at':datetime.datetime.now().astimezone().isoformat(),
          'inputs_sha256':{p:sha(p) for p in files},'episodes':100,'G0_min_success':91,'port':8057,'gpu':7,
          'pack':parent['pack']}
    with Path(path).open('x') as f:f.write(json.dumps(spec,indent=2)+'\n')
    validate(path);print('Frozen object G0',sha(path))


def main():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--freeze-out');ap.add_argument('--freeze');args=ap.parse_args()
    if args.freeze_out:freeze(args.freeze_out);return
    spec=validate(args.freeze)
    subprocess.run(['git','ls-files','--error-unmatch',args.freeze],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','diff','--exit-code','HEAD','--',args.freeze],check=True,stdout=subprocess.DEVNULL)
    root=ROOT/'g0';root.mkdir(exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',MUJOCO_EGL_DEVICE_ID='7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    server=None;code=1
    try:
        with (root/'server.log').open('x') as log:
            server=subprocess.Popen(['bash','cce/env.sh','cce/serve_xvla_v4r1.py','--lora-path',spec['adapter']['path'],
                                     '--port',str(spec['port']),'--info-json',str(root/'server_info.json')],
                                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(120):
            if server.poll() is not None:raise RuntimeError('Object server exited')
            try:
                info=requests.get(f"http://127.0.0.1:{spec['port']}/ready",timeout=2).json()
                assert info['lora_path']==spec['adapter']['path'] and info['lora_merged'] and info['supports_request_seed']
                break
            except requests.RequestException:time.sleep(2)
        else:raise RuntimeError('Object server timeout')
        def shard(i):
            output=root/f'shard{i}.jsonl'
            cmd=['bash','cce/env_libero.sh','cce/libero_family_v3.py','--stage','g0','--suite','libero_object',
                 '--pack',spec['pack'],'--plants','L00_nominal','--methods','naive','--episodes','10',
                 '--port',str(spec['port']),'--shard',str(i),'--nshards','4','--out',str(output),'--trace-first','--verify-repeat']
            with (root/f'shard{i}.log').open('x') as log:
                p=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (root/f'shard{i}.exit').write_text(str(p.returncode)+'\n')
            assert p.returncode==0,f'G0 shard {i} failed'
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(shard,i) for i in range(4)]
            for future in futures:future.result()
        paths=sorted(root.glob('shard*.jsonl'))
        rows=[json.loads(l) for p in paths for l in p.read_text().splitlines()]
        count=Counter((r['task_id'],r['ep']) for r in rows)
        assert set(count)=={(t,e) for t in range(10) for e in range(10)} and set(count.values())=={1}
        for p in paths:
            meta=json.loads(p.with_suffix('.meta.json').read_text())
            assert meta['source_sha256']==sha('cce/libero_family_v3.py')
            assert meta['pack_sha256']==sha(spec['pack']) and meta['server']['lora_path']==spec['adapter']['path']
        for r in rows:
            assert r['suite']=='libero_object' and r['pid']=='L00_nominal' and r['method']=='naive'
            assert r['success'] in (0,1) and 1<=r['steps']<=800 and (r['success'] or r['steps']==800)
        n=sum(r['success'] for r in rows)
        result={'status':'complete','n':len(rows),'success':n,'sr':n/100,'G0_pass':n>=spec['G0_min_success'],
                'by_task':{str(t):sum(r['success'] for r in rows if r['task_id']==t) for t in range(10)},
                'freeze_sha256':sha(args.freeze),'raw_sha256':{str(p):sha(p) for p in paths},'adapter':spec['adapter']}
        (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);code=0
    except BaseException as error:
        (root/'failure.txt').write_text(repr(error)+'\n');raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        (root/'supervisor.exit').write_text(str(code)+'\n')
        # The training controller supplies only the shared logging helper; no
        # training state or GPU ownership is changed here.
        from cce.run_policyside_training_v4r1 import gpu_snapshot
        gpu_snapshot(root,'object_G0')


if __name__=='__main__':main()
