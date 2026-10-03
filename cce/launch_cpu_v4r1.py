#!/usr/bin/env python
"""Finite CPU regression supervisor with immutable plans and exit sentinels."""
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
    ap.add_argument('--theta-dir', required=True)
    ap.add_argument('--workers', type=int, default=13)
    ap.add_argument('--shards', type=int, default=13)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=False)
    methods = 'naive,ours_v3,v4_rho2,v4_rho4,v4_rho8,v4_rhoinf'
    jobs = []
    for uid in ['panda', 'xarm6_robotiq']:
        for bench in ['tracking', 'replay']:
            for shard in range(1 if args.smoke else args.shards):
                name = f'{bench}_{uid}_{shard}'
                cmd = ['bash', 'cce/env.sh', 'cce/bench_relative_v4r1.py', '--bench', bench,
                       '--relative-pack', str(Path(args.theta_dir)/f'theta_{uid}.json'),
                       '--uid', uid, '--methods', methods, '--theta-dir', 'data/cce/theta_sat',
                       '--out', str(root/(name+'.json'))]
                if args.smoke:
                    cmd += ['--plants', 'P00_nominal,D2_t400',
                            '--n-traj' if bench == 'tracking' else '--n-episodes', '2']
                else:
                    cmd += ['--shard', str(shard), '--nshards', str(args.shards)]
                jobs.append({'name': name, 'command': cmd})
    paths = ['cce/bench_relative_v4r1.py', 'cce/libero_exec_v4r1.py', 'cce/identify_v4r1.py',
             'cce/bench_tracking.py', 'cce/bench_replay.py', 'cce/launch_cpu_v4r1.py']
    plan = {'started': datetime.datetime.now().isoformat(), 'jobs': jobs, 'smoke': args.smoke,
            'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    def run(job):
        start = time.monotonic()
        with (root/(job['name']+'.log')).open('w') as log:
            p = subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT, env=env)
        (root/(job['name']+'.exit')).write_text(str(p.returncode)+'\n')
        return {'name': job['name'], 'exit': p.returncode, 'wall_s': time.monotonic()-start}
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for fut in as_completed(futures):
            rows.append(fut.result())
            (root/'status.json').write_text(json.dumps({'done':len(rows), 'total':len(jobs), 'jobs':rows}, indent=2)+'\n')
            print(json.dumps(rows[-1]), flush=True)
    rc = int(any(r['exit'] for r in rows))
    (root/'supervisor.exit').write_text(str(rc)+'\n')
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
