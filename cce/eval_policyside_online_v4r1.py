"""Naive execution of adapted policies on disjoint evaluation initial states."""
import _bootstrap  # noqa: F401
import argparse
import json
import os
from pathlib import Path
import requests
from cce import libero_family_v3 as F
from cce.libero_collect_plant_demos_v4r1 import configure
from cce.freeze_training_v4r1 import validate_training_freeze


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--freeze'); ap.add_argument('--mode', choices=['ft_demo','prompt_fit'], required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--shard', type=int, required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); assert 0 <= args.shard < 4
    if args.smoke:
        assert Path(args.checkpoint).name=='ckpt-3' and args.shard==0
    else: spec = validate_training_freeze(args.freeze)
    _, hats, LC, ev, suite = configure()
    info = requests.get(f'http://127.0.0.1:{args.port}/ready', timeout=5).json()
    assert info['supports_request_seed'] and info['policy_mode'] == args.mode
    assert info['checkpoint'] == str(Path(args.checkpoint).resolve())
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); assert not out.exists()
    client = F.ChunkClient('127.0.0.1', args.port); client.suite_name = 'libero_spatial'
    client.verify_repeat = True
    work = [('L00_nominal',0,10)] if args.smoke else [(p, t, e) for p in spec['evaluation_plants'] for t in range(10) for e in range(10)]
    with out.open('x') as f:
        for p, t, e in work[args.shard::4]:
            row = F.run_episode(ev, suite, t, e, F.BY_ID_V3[p], 'naive', client,
                                hats[p], F.theta_from_hat(hats[F.NOMINAL_PID]), 800, 1., 1.)
            row.update(policy_mode=args.mode, checkpoint=info['checkpoint'], training_freeze=args.freeze,
                       evaluation_protocol='dev smoke ep10' if args.smoke else 'unchanged seeded half-open-loop proprio; ep0..9 held out from demo collection')
            f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
            print(json.dumps({k:row[k] for k in ['pid','task_id','ep','success','steps']}), flush=True)


if __name__ == '__main__': main()
