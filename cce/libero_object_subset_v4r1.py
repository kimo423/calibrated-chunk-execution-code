"""Transfer the frozen spatial executor to the object suite without retuning."""
import _bootstrap  # noqa: F401
import argparse
import json
import os
from pathlib import Path
import requests
from cce import libero_family_v3 as F
from cce.libero_collect_plant_demos_v4r1 import configure
from cce.libero_probe import make_evaluator
from cce.freeze_v4r1 import sha


def main():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--freeze');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--port',type=int,default=8057);ap.add_argument('--out',required=True)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--nshards',type=int,default=4)
    args=ap.parse_args()
    parent, hats, _, _, _=configure()
    methods=['naive','slow20','ours_v4','ours_v4_slow15']
    if args.smoke:
        plants=['L00_nominal'];tasks=[0];episodes=1
        assert args.shard==0 and args.nshards==1
        source=json.loads(Path('/opt/cce/data/cce/object_v4r1_20260909/adapter_source.json').read_text())
    else:
        from cce.run_object_subset_v4r1 import validate
        spec=validate(args.freeze);plants=spec['plants'];tasks=spec['tasks'];episodes=spec['episodes']
        source=spec['adapter'];assert 0<=args.shard<args.nshards==spec['nshards']
    ev,suites,LC=make_evaluator('libero_object');suite=suites[0]
    client=F.ChunkClient('127.0.0.1',args.port);client.suite_name='libero_object';client.verify_repeat=True
    info=requests.get(f'http://127.0.0.1:{args.port}/ready',timeout=5).json()
    assert info['lora_path']==source['path'] and info['lora_merged'] and info['supports_request_seed']
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
    meta={'suite':'libero_object','source_sha256':sha(__file__),'pack_sha256':sha(parent['pack']),
          'server':info,'args':vars(args),'freeze_sha256':sha(args.freeze) if args.freeze else None,
          'executor_parameters':'frozen spatial pack, rho and H0 transferred unchanged'}
    out.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.smoke:
        F.STEP_DIR=out.parent/'smoke_traces';F.STEP_DIR.mkdir(exist_ok=False)
    work=[(p,m,t,e) for p in plants for m in methods for t in tasks for e in range(episodes)]
    with out.open('x') as log:
        for p,m,t,e in work[args.shard::args.nshards]:
            row=F.run_episode(ev,suite,t,e,F.BY_ID_V3[p],m,client,hats[p],F.theta_from_hat(hats[F.NOMINAL_PID]),800,1.,1.)
            row.update(suite='libero_object',rho=2.,H0=F.GATE['H0'])
            log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno())
            print(json.dumps({k:row[k] for k in ['pid','method','task_id','ep','success','steps']}),flush=True)


if __name__=='__main__':main()
