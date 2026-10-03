#!/usr/bin/env python
"""Anchor the nominal gain in the same free-run LS metric, then fit relative plants."""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cce.libero_identify_v4 import simulate, fit_axis, relative_arx
from cce.identify import realized_deltas, TRANS_EVAL_WINDOW, ROT_WINDOW
from cce.common import json_default


def identify(uid, specs, nominal_pid, hats, probe_paths):
    nom = hats[nominal_pid]
    with np.load(probe_paths[nominal_pid]) as rec:
        u, w = rec['u'], realized_deltas(rec['y'])
    an = np.array([a['alpha'] for a in nom['axes']])
    gn = np.zeros(6)
    for i in range(6):
        lo, hi = TRANS_EVAL_WINDOW if i < 3 else ROT_WINDOW
        basis = simulate(u[:, i], an[i], 1., 0, 0.)[lo:hi]
        gn[i] = (basis @ w[lo:hi, i])/(basis @ basis)
    out = {'uid': uid, 'nominal_alpha': an, 'nominal_gain': gn, 'plants': {},
           'probe_sha256': {p: hashlib.sha256(Path(path).read_bytes()).hexdigest() for p, path in probe_paths.items()}}
    for s in specs:
        with np.load(probe_paths[s.pid]) as rec:
            w = realized_deltas(rec['y'])
            axes = [fit_axis(rec['u'][:, i], w[:, i], an[i], gn[i],
                             TRANS_EVAL_WINDOW if i < 3 else ROT_WINDOW) for i in range(6)]
        for axis in axes:
            if abs(axis['gain']-1.) < 1e-12:
                axis['gain'] = 1.  # numerical identity, not a fitted tolerance
        out['plants'][s.pid] = {
            'axes': axes, 'delay_steps': int(round(np.median([a['delay_steps'] for a in axes[:3]]))),
            'grip_lead_steps': hats[s.pid]['grip_lead_steps_hat'],
            'arx_relative': relative_arx(hats[s.pid], nom), 'truth_report_only': s.as_dict()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bed', choices=['libero', 'panda', 'xarm6_robotiq'], required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.bed == 'libero':
        from cce.libero_plant_v3 import FAMILY_V3 as specs
        nominal_pid, uid = 'L00_nominal', 'libero_panda'
        arx = Path('data/cce/libero/theta/theta_hat_libero_panda_v2.json')
        hats = json.loads(arx.read_text())
        extra = Path('data/cce/takeover_v4_20260908/new_probes')
        extra_arx = extra/'probe_summary.json'
        hats.update(json.loads(extra_arx.read_text())['hats'])
        paths = {s.pid: Path('data/cce/libero/probes')/f'probe_{uid}_{s.pid}.npz' for s in specs}
        paths = {p: v if v.exists() else extra/v.name for p,v in paths.items()}
    else:
        from cce.plants import PLANTS as specs
        nominal_pid, uid = 'P00_nominal', args.bed
        arx = Path(f'data/cce/theta_sat/theta_hat_{uid}.json')
        hats = json.loads(arx.read_text())
        paths = {s.pid: Path('data/cce/probes')/f'probe_{uid}_{s.pid}.npz' for s in specs}
    out = identify(uid, specs, nominal_pid, hats, paths)
    out['arx_sha256'] = hashlib.sha256(arx.read_bytes()).hexdigest()
    if args.bed == 'libero':
        out['additional_arx_sha256'] = hashlib.sha256(extra_arx.read_bytes()).hexdigest()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=json_default, allow_nan=False)+'\n')
    print(json.dumps({'bed': args.bed, 'nominal': out['plants'][nominal_pid], 'n':len(specs)}, default=json_default))


if __name__ == '__main__':
    main()
