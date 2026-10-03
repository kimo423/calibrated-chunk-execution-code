#!/usr/bin/env python
"""Run historical physical benchmarks through an additive v4r1 arm factory."""
import _bootstrap  # noqa: F401
import argparse
import importlib
import json
import sys
from pathlib import Path
import numpy as np
from cce.executor import build_arm as old_build_arm
from cce.libero_exec_v4r1 import RelativeExecutor, RelativeTheta, predicted_h


def gate_values(pack, reference):
    an, gn = np.array(pack['nominal_alpha']), np.array(pack['nominal_gain'])
    h = {p: predicted_h(RelativeTheta.from_fit(v), an, gn, reference) for p, v in pack['plants'].items()}
    h0 = (h['D1_d100']+h['D1_d150'])/2
    nominal = 'L00_nominal' if 'L00_nominal' in h else 'P00_nominal'
    return {'H': h, 'H0': h0, 'dev_separable': h[nominal] < h0 and h['D1_d100'] < h0 <= h['D1_d150']}


class SimAdapter:
    def __init__(self, ex):
        self.ex = ex

    def step(self, y, c, grip, t):
        u, g = self.ex.step(y, c, grip, t)
        return np.array([*u, g], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', choices=['tracking', 'replay'], required=True)
    ap.add_argument('--relative-pack', required=True)
    args, rest = ap.parse_known_args()
    pack = json.loads(Path(args.relative_pack).read_text())
    an, gn = np.array(pack['nominal_alpha']), np.array(pack['nominal_gain'])
    with np.load(f'data/cce/probes/probe_{pack["uid"]}_P00_nominal.npz') as z:
        gate = gate_values(pack, z['targets'])
    print(json.dumps({'gate': gate, 'relative_pack': args.relative_pack}), flush=True)

    def factory(method, spec, hat, theta_nominal, **kw):
        if method == 'ours_v3':
            return old_build_arm('ours_p', spec, hat, theta_nominal, **kw)
        if not method.startswith('v4_rho'):
            return old_build_arm(method, spec, hat, theta_nominal, **kw)
        rho = float(method.removeprefix('v4_rho'))
        rel = RelativeTheta.from_fit(pack['plants'][spec.pid])
        ex = RelativeExecutor(rel, an, gn, rho_max=rho, h=gate['H'][spec.pid], h0=gate['H0'])
        return SimAdapter(ex), False

    bench = importlib.import_module('cce.bench_'+args.bench)
    bench.build_arm = factory  # process-local injection; historical source untouched
    sys.argv = [sys.argv[0], *rest]
    bench.main()


if __name__ == '__main__':
    main()
