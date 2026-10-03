"""Five full nominal trajectory pairs with owned-service lifecycle and audit."""
import _bootstrap  # noqa: F401
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import requests
from cce.freeze_v4r1 import sha, validate_freeze

ROOT=Path('/opt/cce/data/cce/nominal_acceptance_w4r2_20260911')


def main():
    spec=validate_freeze('cce/results/libero_family_spec_v4r1_fallback.json',weights=True)
    ROOT.mkdir(exist_ok=False)
    from cce.libero_collect_plant_demos_v4r1 import configure
    from cce import libero_family_v3 as F
    _,hats,_,ev,suite=configure()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',MUJOCO_EGL_DEVICE_ID='7',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    server=None; code=1; started=time.time(); port=8067
    try:
        with (ROOT/'server.log').open('x') as log:
            server=subprocess.Popen(['bash','cce/env.sh','cce/serve_xvla_v4r1.py','--model-path',spec['server']['model_path'],'--lora-path',spec['server']['lora_path'],'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'plan.json').write_text(json.dumps({'source_sha256':sha(__file__),'parent_sha256':sha('cce/results/libero_family_spec_v4r1_fallback.json'),'started':started,'task':0,'episodes':list(range(5)),'pairs':['naive','ours_v4'],'owned_server_pid':server.pid},indent=2)+'\n')
        for _ in range(120):
            if server.poll() is not None:raise RuntimeError('server exited before readiness')
            try:
                info=requests.get(f'http://127.0.0.1:{port}/ready',timeout=2).json()
                assert info['lora_path']==spec['server']['lora_path'] and info['supports_request_seed'];break
            except requests.RequestException:time.sleep(2)
        else:raise RuntimeError('readiness timeout')
        client=F.ChunkClient('127.0.0.1',port);client.suite_name='libero_spatial';client.verify_repeat=True
        F.STEP_DIR=ROOT/'traces';F.STEP_DIR.mkdir()
        base={}
        source=json.loads(Path('cce/results/libero_fallback_v4r1.json').read_text())
        for path,h in source['raw_sha256'].items():
            if Path(path).name.startswith('L00_nominal__'):
                assert sha(path)==h
                for r in map(json.loads,Path(path).read_text().splitlines()):
                    if r['task_id']==0 and r['ep']<5:base[r['method'],r['ep']]=r
        pairs=[]
        with (ROOT/'episodes.jsonl').open('x') as f:
            for ep in range(5):
                traces={}
                for method in ['naive','ours_v4']:
                    r=F.run_episode(ev,suite,0,ep,F.BY_ID_V3['L00_nominal'],method,client,hats['L00_nominal'],F.theta_from_hat(hats['L00_nominal']),800,1.,1.)
                    f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
                    for k in ['success','steps','policy_queries','n_bound_hits','tracking_rmse_m']:
                        assert r[k]==base[method,ep][k],(ep,method,k)
                    p=F.STEP_DIR/f'L00_nominal__{method}__t0__e{ep}.npz'
                    with np.load(p) as z:traces[method]=z['steps']
                assert np.array_equal(traces['naive'],traces['ours_v4']),ep
                pair={'task':0,'ep':ep,'steps':len(traces['naive']),'max_abs_difference':float(np.max(abs(traces['naive']-traces['ours_v4'])))}
                pairs.append(pair);print(json.dumps(pair),flush=True)
        out={'status':'passed','n_pairs':5,'n_episodes':10,'all_36_step_columns_exact':True,'original_summary_reproduced':True,'pairs':pairs,'wall_s':time.time()-started,
             'raw_sha256':{str(p):sha(p) for p in F.STEP_DIR.glob('*.npz')},'source_sha256':sha(__file__)}
        (ROOT/'results.json').write_text(json.dumps(out,indent=2)+'\n')
        Path('cce/results/CCE_nominal_acceptance_w4r2.json').write_text(json.dumps(out,indent=2)+'\n')
        Path('docs/reports/CCE_nominal_acceptance_w4r2.md').write_text('# 标称五组逐步验收\n\n任务0、初态ep0..4，naive/r1各5集。全部36列逐步完全一致（最大绝对差0），并复现主批相同初态汇总。原规划5集抽查要求已满足，不外推为全体初态的逐步证明。\n')
        code=0
    except BaseException as error:
        (ROOT/'failure.txt').write_text(repr(error)+'\n');raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        (ROOT/'supervisor.exit').write_text(str(code)+'\n')


if __name__=='__main__':main()
