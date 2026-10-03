#!/usr/bin/env python
"""Reproduce both historical passes without modifying any historical artifact."""
import _bootstrap  # noqa: F401
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'passes': {}, 'checks': {}}
    for version, folder, count in [(1, 'main', 3000), (2, 'main_v2', 3500)]:
        files = sorted(Path('data/logs/cce_libero', folder).glob('shard*.jsonl'))
        assert len(files) == 8, files
        rows = [json.loads(line) for p in files for line in p.read_text().splitlines() if line.strip()]
        keys = Counter((r['pid'], r['method'], r['task_id'], r['ep']) for r in rows)
        expected = set(itertools.product(sorted({r['pid'] for r in rows}),
                        ['naive', 'slow15', 'slow20', 'oracle', 'ours_p'], range(10), range(5)))
        assert len(rows) == count and set(keys) == expected and max(keys.values()) == 1
        exits = {str(p.with_suffix('.exit')): p.with_suffix('.exit').read_text().strip() for p in files}
        assert set(exits.values()) == {'0'}, exits
        old = Path(f'cce/results/libero_family_v{version}.json')
        old_sha = sha(old)
        target = out / f'check_v{version}.json'
        suffix = '' if version == 1 else '_v2'
        cmd = [sys.executable, f'cce/libero_aggregate{suffix}.py', '--glob', *map(str, files),
               '--out', str(target), '--label', f'check_v{version}',
               '--theta-summary', f'data/cce/libero/theta/identify_libero_panda{suffix}.json',
               '--family-spec', f'cce/results/libero_family_spec{suffix}.json']
        if version == 2:
            cmd += ['--devplant-json', 'cce/results/libero_devplant_v2.json',
                    '--v1-json', 'cce/results/libero_family_v1.json', '--v1-glob',
                    'data/logs/cce_libero/main/shard*.jsonl']
        with (out / f'aggregate_v{version}.log').open('w') as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        (out / f'aggregate_v{version}.exit').write_text(str(proc.returncode)+'\n')
        assert proc.returncode == 0, cmd
        a, b = json.loads(old.read_text()), json.loads(target.read_text())
        matches = {key: a[key] == b[key] for key in ['primary', 'primary_holm_p', 'macro']}
        assert all(matches.values()), matches
        assert sha(old) == old_sha
        results['passes'][str(version)] = {'n_episodes': len(rows), 'n_unique_keys': len(keys),
            'missing_keys': len(expected - set(keys)), 'exact_matches': matches,
            'historical_summary_sha256': old_sha, 'raw_sha256': {str(p): sha(p) for p in files},
            'command': cmd, 'primary': b['primary'], 'macro': b['macro']}
    for script in ['libero_plant.py', 'libero_exec.py', 'validate_codec.py']:
        cmd = [sys.executable, 'cce/'+script]
        if script == 'validate_codec.py':
            cmd = ['bash', 'cce/env.sh', 'cce/'+script]
        with (out / (script+'.log')).open('w') as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        (out / (script+'.exit')).write_text(str(proc.returncode)+'\n')
        results['checks'][script] = {'exit_code': proc.returncode, 'command': cmd}
        assert proc.returncode == 0, script
        if script == 'validate_codec.py':
            content = (out / (script+'.log')).read_text()
            metrics = json.loads(content[content.index('{'):])
            assert max(metrics['codec'].values()) < 1e-7, metrics['codec']
            assert metrics['worst_oracle_tail_rmse_m'] < 1e-8
            results['checks'][script]['metrics'] = metrics
            results['checks'][script]['identity_test_caveat'] = (
                'Historical random actions include rotation vectors with norm > 1; '
                'the nominal wrapper clips them. Its printed deviation is not an identity assertion.')
    from cce.plants import PlantWrapper, get_plant
    rng = np.random.default_rng(20260908)
    plant = PlantWrapper(get_plant('P00_nominal'))
    err = 0.
    for _ in range(2000):
        action = rng.uniform(-1, 1, 7)
        action[3:6] /= max(1., np.linalg.norm(action[3:6]))
        err = max(err, float(np.max(np.abs(plant.apply(action)-action))))
    assert err < 1e-7, err
    results['checks']['valid_domain_nominal_identity'] = {'max_error': err, 'cases': 2000,
                                                         'tolerance': 1e-7}
    results['ok'] = True
    (out / 'audit.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({'ok': True, 'episodes': 6500, 'checks': results['checks']}, indent=2))


if __name__ == '__main__':
    main()
