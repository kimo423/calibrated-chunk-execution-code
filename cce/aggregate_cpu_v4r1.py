#!/usr/bin/env python
"""Audit complete physical regression cells and report both tracking conventions."""
import _bootstrap  # noqa: F401
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import numpy as np
from cce.plants import PLANTS

METHODS = ['naive','ours_v3','v4_rho2','v4_rho4','v4_rho8','v4_rhoinf']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.run_dir)
    assert (root/'supervisor.exit').read_text().strip() == '0'
    by_robot = {}
    for uid, k1, k2 in [('panda', .763, .796), ('xarm6_robotiq', .932, .972)]:
        data = {}
        for bench, n in [('tracking',20), ('replay',50)]:
            paths = sorted(root.glob(f'{bench}_{uid}_*.json'))
            assert len(paths) == 13
            rows = [r for p in paths for r in json.loads(p.read_text())]
            keyfield = 'traj_seed' if bench == 'tracking' else 'seed'
            seeds = sorted({r[keyfield] for r in rows})
            assert len(seeds) == n
            count = Counter((r['pid'],r['method'],r[keyfield]) for r in rows)
            expected = set(itertools.product([p.pid for p in PLANTS],METHODS,seeds))
            assert set(count) == expected and max(count.values()) == 1
            cells = {}
            for spec in PLANTS:
                cells[spec.pid] = {}
                for method in METHODS:
                    sub = [r for r in rows if r['pid']==spec.pid and r['method']==method]
                    fields = ['rmse_pos_m','rmse_pos_vs_intent_m'] if bench=='tracking' else ['success']
                    cells[spec.pid][method] = {f:float(np.mean([r[f] for r in sub])) for f in fields}
            data[bench] = {'n_rows':len(rows),'cells':cells}
        metrics = {}
        lag = [s.pid for s in PLANTS if s.family in ('D1','D2t','MIX')]
        for method in METHODS:
            track = data['tracking']['cells']
            reduction = {}
            for field in ['rmse_pos_m','rmse_pos_vs_intent_m']:
                reduction[field] = float(np.mean([(track[p]['naive'][field]-track[p][method][field])/track[p]['naive'][field] for p in lag]))
            replay = data['replay']['cells']
            nominal = replay['P00_nominal']['naive']['success']
            valid = [s.pid for s in PLANTS if nominal-replay[s.pid]['naive']['success'] > .10]
            recovery = float(np.mean([(replay[p][method]['success']-replay[p]['naive']['success'])/(nominal-replay[p]['naive']['success']) for p in valid]))
            metrics[method] = {'lag_tracking_reduction':reduction, 'macro_R':recovery,
                               'R_population':valid, 'nominal_SR':replay['P00_nominal'][method]['success'],
                               'macro_SR':float(np.mean([replay[s.pid][method]['success'] for s in PLANTS if s.pid!='P00_nominal'])),
                               'historical_K1_floor_met':reduction['rmse_pos_m'] >= k1,
                               'historical_K2_floor_met':recovery >= k2}
        by_robot[uid] = {'metrics':metrics, 'thresholds':{'K1':k1,'K2':k2}, **data}
    out = {'status':'complete', 'robots':by_robot,
           'all_rho_pass_both_robots':{m:all(by_robot[u]['metrics'][m]['historical_K1_floor_met'] and by_robot[u]['metrics'][m]['historical_K2_floor_met'] for u in by_robot) for m in METHODS[2:]}}
    Path(args.out).write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'all_rho_pass':out['all_rho_pass_both_robots'], 'metrics':{u:r['metrics'] for u,r in by_robot.items()}}, indent=2))


if __name__ == '__main__':
    main()
