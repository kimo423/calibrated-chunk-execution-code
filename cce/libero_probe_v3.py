#!/usr/bin/env python
"""Collect the five planned open-loop probes; never invoke a policy server."""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cce.common import DT, json_default
from cce.identify import identify_plant
from cce.libero_plant import NOMINAL_PID
from cce.libero_plant_v3 import NEW_V3
from cce.libero_probe import make_evaluator, run_probe, probe_stats, UID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--nominal-arx', default='data/cce/libero/theta/theta_hat_libero_panda_v2.json')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob('probe_*.npz')):
        raise FileExistsError('probe output already exists; choose a new run directory')
    old = json.loads(Path(args.nominal_arx).read_text())
    t90 = old[NOMINAL_PID]['grip_t90_s']
    ev, suites, _ = make_evaluator('libero_spatial', 42)
    results = {'suite': 'libero_spatial', 'task_id': 0, 'ep': 0, 'init_seed': 42,
               'policy_calls': 0, 'status': 'open_loop_development', 'hats': {}, 'stats': [],
               'sha256': {}, 'nominal_arx_sha256': hashlib.sha256(Path(args.nominal_arx).read_bytes()).hexdigest()}
    for plant in NEW_V3:
        rec = run_probe(ev, suites[0], plant, task_id=0, ep=0)
        path = out / f'probe_{UID}_{plant.pid}.npz'
        np.savez_compressed(path, **{k: rec[k] for k in ['u', 'y', 'grip_open', 'grip_cmd', 'home', 'targets', 'applied']})
        hat = identify_plant(rec)
        lead = hat['grip_t90_s']-t90
        if not np.isfinite(lead):
            raise ValueError('gripper lead is not identifiable')
        hat.update(grip_t90_nominal_s=t90, grip_lead_s_hat=lead,
                   grip_lead_steps_hat=max(0, round(lead/DT)))
        results['hats'][plant.pid] = hat
        results['stats'].append(probe_stats(rec))
        results['sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out/'probe_summary.json').write_text(json.dumps(results, indent=2, default=json_default)+'\n')
        print(json.dumps(results['stats'][-1]), flush=True)
    results['complete'] = True
    (out/'probe_summary.json').write_text(json.dumps(results, indent=2, default=json_default)+'\n')


if __name__ == '__main__':
    main()
