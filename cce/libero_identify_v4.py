#!/usr/bin/env python
"""Grey-box relative identification: fixed nominal pole followed by injected pole.

Fits only recorded commands and achieved increments. Injection truth is used by
the reporting code, never by fit_axis. Historical ARX artifacts stay read-only.
"""
from __future__ import annotations
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cce.common import DT
from cce.identify import realized_deltas, TRANS_EVAL_WINDOW, ROT_WINDOW
from cce.libero_plant import NOMINAL_PID
from cce.libero_plant_v2 import FAMILY_V2

TAU_GRID = (0., .05, .1, .15, .2, .25, .3, .4, .5, .6)
R2_GATE = .7


def simulate(u, alpha_n, gain_n, delay, tau, gain=1.):
    """Zero initial states; output[t] is the increment y[t] -> y[t+1]."""
    a = np.exp(-DT / tau) if tau > 0 else 0.
    z, v = 0., 0.
    out = np.zeros(len(u), dtype=np.float64)
    for t in range(len(u)):
        z = a*z + (1-a)*(u[t-delay] if t >= delay else 0.)
        v = alpha_n*v + (1-alpha_n)*gain*z
        out[t] = gain_n*v
    return out


def r2(pred, truth):
    denom = float(np.sum((truth-truth.mean())**2))
    return float(1-np.sum((truth-pred)**2)/denom) if denom > 1e-12 else None


def fit_axis(u, w, alpha_n, gain_n, window):
    if not (0 <= alpha_n < 1 and gain_n > 0):
        raise ValueError('invalid nominal response')
    u, w = np.asarray(u, dtype=float), np.asarray(w, dtype=float)
    if u.shape != w.shape or not np.isfinite(u).all() or not np.isfinite(w).all():
        raise ValueError('invalid probe series')
    lo, hi = window
    target = w[lo:hi]
    best = None
    for delay in range(7):
        for tau in TAU_GRID:
            basis = simulate(u, alpha_n, gain_n, delay, tau)[lo:hi]
            denom = float(basis @ basis)
            if denom <= 1e-12:
                continue
            gain = float(basis @ target / denom)
            if gain <= 0:
                continue
            score = r2(gain*basis, target)
            if score is not None and (best is None or score > best['r2_freerun']):
                best = dict(delay_steps=delay, tau_s=tau, gain=gain,
                            r2_freerun=score, identifiable=bool(score >= R2_GATE))
    return best or dict(delay_steps=0, tau_s=0., gain=1., r2_freerun=None, identifiable=False)


def relative_arx(hat, nominal):
    return {'axes': [dict(delay_steps=max(0, a['delay_steps']-n['delay_steps']),
                         tau_s=max(0., a['tau_s']-n['tau_s']), gain=a['gain']/n['gain'],
                         r2_freerun=a['r2_freerun'], identifiable=a['identifiable'])
                     for a, n in zip(hat['axes'], nominal['axes'])],
            'delay_steps': max(0, hat['delay_steps_agg']-nominal['delay_steps_agg']),
            'tau_s': max(0., hat['tau_s_agg']-nominal['tau_s_agg']),
            'gain': hat['gain_agg']/nominal['gain_agg'],
            'grip_lead_steps': hat['grip_lead_steps_hat']}


def selftest():
    rng = np.random.default_rng(20260908)
    u = rng.normal(size=400)
    worst = 0.
    count = 0
    for delay in range(7):
        for tau in TAU_GRID:
            for gain in (.7, 1., 1.3):
                w = simulate(u, .57, .29, delay, tau, gain)
                fit = fit_axis(u, w, .57, .29, (0, len(u)))
                assert fit['delay_steps'] == delay and fit['tau_s'] == tau
                worst = max(worst, abs(fit['gain']-gain))
                assert fit['identifiable'] and fit['r2_freerun'] > 1-1e-12
                count += 1
    assert worst < 1e-12
    assert not fit_axis(np.zeros(40), np.zeros(40), .5, .3, (0, 40))['identifiable']
    return dict(cases=count, gain_max_abs_error=worst, ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--probe-dir', default='data/cce/libero/probes')
    ap.add_argument('--arx', default='data/cce/libero/theta/theta_hat_libero_panda_v2.json')
    ap.add_argument('--out', default='data/cce/libero/theta/theta_hat_libero_panda_v4.json')
    ap.add_argument('--family', choices=['v2', 'v3'], default='v2')
    ap.add_argument('--additional-probe-dir')
    ap.add_argument('--additional-arx', help='probe_summary.json produced by libero_probe_v3')
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2)); return
    hats = json.loads(Path(args.arx).read_text())
    family = FAMILY_V2
    if args.family == 'v3':
        from cce.libero_plant_v3 import FAMILY_V3
        family = FAMILY_V3
    if args.additional_arx:
        extra = json.loads(Path(args.additional_arx).read_text())['hats']
        if set(extra) & set(hats):
            raise ValueError('additional probes would overwrite historical estimates')
        hats.update(extra)
    nominal = hats[NOMINAL_PID]
    output = {'status': 'development_only_not_frozen', 'tau_grid_s': TAU_GRID,
              'r2_gate': R2_GATE, 'nominal_model': nominal['axes'], 'plants': {}, 'inputs_sha256': {}}
    for spec in family:
        p = Path(args.probe_dir)/f'probe_libero_panda_{spec.pid}.npz'
        if not p.exists() and args.additional_probe_dir:
            p = Path(args.additional_probe_dir)/p.name
        output['inputs_sha256'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p) as rec:
            w = realized_deltas(rec['y'])
            axes = [fit_axis(rec['u'][:, i], w[:, i], n['alpha'], n['gain'],
                             TRANS_EVAL_WINDOW if i < 3 else ROT_WINDOW)
                    for i, n in enumerate(nominal['axes'])]
        grey = {'axes': axes, 'delay_steps': int(round(np.median([a['delay_steps'] for a in axes[:3]]))),
                'tau_s': float(np.median([a['tau_s'] for a in axes[:3]])),
                'gain': float(np.median([a['gain'] for a in axes[:3]])),
                'grip_lead_steps': hats[spec.pid]['grip_lead_steps_hat']}
        output['plants'][spec.pid] = {'greybox': grey, 'arx_relative': relative_arx(hats[spec.pid], nominal),
                                     'truth_for_report_only': spec.as_dict()}
    rows = list(output['plants'].values())
    output['agreement'] = {
        'delay_mae_steps': float(np.mean([abs(r['greybox']['delay_steps']-r['truth_for_report_only']['delay_steps']) for r in rows])),
        'grip_mae_steps': float(np.mean([abs(r['greybox']['grip_lead_steps']-r['truth_for_report_only']['grip_lead_steps']) for r in rows])),
        'tau_mae_s': float(np.mean([abs(r['greybox']['tau_s']-r['truth_for_report_only']['tau_s']) for r in rows])),
        'gain_mae': float(np.mean([abs(r['greybox']['gain']-r['truth_for_report_only']['gain']) for r in rows])),
        'unidentifiable_axes': sum(not a['identifiable'] for r in rows for a in r['greybox']['axes'])}
    output['inputs_sha256'][args.arx] = hashlib.sha256(Path(args.arx).read_bytes()).hexdigest()
    if args.additional_arx:
        output['inputs_sha256'][args.additional_arx] = hashlib.sha256(Path(args.additional_arx).read_bytes()).hexdigest()
    target = Path(args.out); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'agreement': output['agreement'], 'D2_t400': output['plants']['D2_t400'],
                      'nominal': output['plants'][NOMINAL_PID]['greybox']}, indent=2))


if __name__ == '__main__':
    main()
