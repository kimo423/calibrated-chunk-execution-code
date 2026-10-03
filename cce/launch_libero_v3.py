#!/usr/bin/env python
"""Finite development/G0 LIBERO launch; per-shard exit codes are authoritative."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--stage', choices=['dev','g0'], required=True)
    ap.add_argument('--plants', default='L00_nominal')
    ap.add_argument('--methods', default='naive')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--shards', type=int, default=4)
    ap.add_argument('--port', type=int, default=8035)
    ap.add_argument('--gpu', type=int, default=5)
    args = ap.parse_args()
    root = Path(args.out_dir); root.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for i in range(args.shards):
        cmd=['bash','cce/env_libero.sh','cce/libero_family_v3.py','--stage',args.stage,
             '--pack',args.pack,'--plants',args.plants,'--methods',args.methods,'--episodes',str(args.episodes),
             '--port',str(args.port),'--shard',str(i),'--nshards',str(args.shards),'--out',str(root/f'shard{i}.jsonl')]
        jobs.append({'shard':i,'command':cmd})
    manifest={'started':datetime.datetime.now().isoformat(),'jobs':jobs,'args':vars(args),
              'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['cce/libero_family_v3.py','cce/libero_exec_v4r1.py','cce/serve_xvla_v4r1.py']}}
    (root/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),MUJOCO_EGL_DEVICE_ID=str(args.gpu),
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def run(j):
        start=time.monotonic()
        with (root/f'shard{j["shard"]}.log').open('w') as log:
            p=subprocess.run(j['command'],stdout=log,stderr=subprocess.STDOUT,env=env)
        (root/f'shard{j["shard"]}.exit').write_text(str(p.returncode)+'\n')
        return {'shard':j['shard'],'exit':p.returncode,'wall_s':time.monotonic()-start}
    rows=[]
    with ThreadPoolExecutor(max_workers=args.shards) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            rows.append(f.result())
            (root/'status.json').write_text(json.dumps({'done':len(rows),'total':len(jobs),'jobs':rows},indent=2)+'\n')
            print(json.dumps(rows[-1]),flush=True)
    rc=int(any(r['exit'] for r in rows))
    (root/'supervisor.exit').write_text(str(rc)+'\n')
    raise SystemExit(rc)


if __name__=='__main__':
    main()
