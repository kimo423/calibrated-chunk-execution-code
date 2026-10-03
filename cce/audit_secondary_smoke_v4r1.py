"""Audit dev-only integration checks without selecting any method parameter."""
import _bootstrap  # noqa: F401
import json
from pathlib import Path
import numpy as np
from cce.freeze_v4r1 import sha, validate_freeze


def main():
    validate_freeze('cce/results/libero_family_spec_v4r1_fallback.json')
    root = Path('/opt/cce/data/cce/secondary_v4r1_20260909/dev_smoke')
    oldroot = Path('/opt/cce/data/cce/v4r1_20260908/fallback_main/jobs')
    paths = [root / f'{s}.jsonl' for s in ['nominal', 'd150', 'te_tasks']]
    rows = [json.loads(l) for p in paths for l in p.read_text().splitlines()]
    assert len(rows) == 37
    assert {r['pid'] for r in rows} == {'L00_nominal', 'D1_d150'}
    trace_checks = 0
    for p in paths:
        for r in [json.loads(l) for l in p.read_text().splitlines()]:
            frequency = 10 if r['method'] == 'te_requery' else 30
            assert r['policy_queries'] == (r['steps'] + frequency - 1) // frequency
            trace = p.parent / f'{p.stem}_steps' / f"{r['pid']}__{r['method']}__t{r['task_id']}__e{r['ep']}.npz"
            with np.load(trace) as z:
                a = z['steps']
                assert len(a) == r['steps'] and np.isfinite(a).all()
                assert np.max(np.abs(a[:, 14:17])) <= 1 + 1e-12
                assert np.max(np.linalg.norm(a[:, 17:20], axis=1)) <= 1 + 1e-12
                assert np.allclose(np.linalg.norm(a[:, 10:14], axis=1), 1)
            trace_checks += 1
    nominal = [r for r in rows if r['pid'] == 'L00_nominal' and r['task_id'] == 0]
    fields = ['success', 'steps', 'tracking_rmse_m', 'policy_queries', 'n_bound_hits']
    for method in ['naive', 'ours_v4']:
        previous = [json.loads(l) for p in oldroot.glob(f'L00_nominal__{method}__s*.jsonl')
                    for l in p.read_text().splitlines()]
        for ep in range(2):
            r = next(r for r in nominal if r['method'] == method and r['ep'] == ep)
            b = next(r for r in previous if r['task_id'] == 0 and r['ep'] == ep)
            assert all(r[k] == b[k] for k in fields)
        original_trace = next(oldroot.glob(f'L00_nominal__{method}__s*_steps/L00_nominal__{method}__t0__e0.npz'))
        new_trace = root / 'nominal_steps' / f'L00_nominal__{method}__t0__e0.npz'
        with np.load(original_trace) as z, np.load(new_trace) as w:
            assert np.array_equal(z['steps'], w['steps'])
    te_errors = []
    for ep in range(2):
        with np.load(root / 'nominal_steps' / f'L00_nominal__naive__t0__e{ep}.npz') as a, \
             np.load(root / 'nominal_steps' / f'L00_nominal__te_requery__t0__e{ep}.npz') as b:
            error = float(np.max(np.abs(a['steps'][:10] - b['steps'][:10])))
            te_errors.append(error)
            assert np.allclose(a['steps'][:10], b['steps'][:10], rtol=0, atol=1e-12)
    out = {'status': 'passed', 'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
           'trace_checks': trace_checks, 'query_cadence_checked_all': True,
           'nominal_summary_reproductions': 4, 'nominal_full_trace_reproductions': 2,
           'te_first_ten_steps_max_abs_error': te_errors, 'parameter_selection': False,
           'raw_sha256': {str(p): sha(p) for p in paths},
           'note': 'Nominal smoke preceded a docstring-only edit. Initial strict TE byte-equality check found 1.1e-15 rounding in one episode; numerical comparison uses atol=1e-12, rtol=0. No experiment or controller parameter changed.'}
    (root / 'audit.json').write_text(json.dumps(out, indent=2) + '\n')
    Path('cce/results/libero_secondary_smoke_v4r1.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
