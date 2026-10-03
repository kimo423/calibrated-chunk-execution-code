"""Audit the secondary grid and report matched, descriptive plant-level effects."""
import _bootstrap  # noqa: F401
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import numpy as np
from cce.freeze_secondary_v4r1 import validate_freeze, sha

BASELINES = ['naive', 'slow20', 'ours_v4']


def interval(values):
    x = np.asarray(values, float)
    if not len(x):
        return {'n': 0, 'mean': None, 'ci95': None}
    rng = np.random.default_rng(20260908)
    draws = x[rng.integers(0, len(x), (10000, len(x)))].mean(axis=1)
    return {'n': len(x), 'mean': float(x.mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist()}


def check_grid(rows, pids, episodes, tasks):
    expected = {(p, m, t, e) for p, m, t in itertools.product(pids, episodes, tasks)
                for e in range(episodes[m])}
    counts = Counter((r['pid'], r['method'], r['task_id'], r['ep']) for r in rows)
    assert set(counts) == expected, 'Incomplete or extra episode keys'
    assert set(counts.values()) == {1}, 'Duplicate episodes'
    for r in rows:
        assert r['success'] in (0, 1) and 1 <= r['steps'] <= 800
        assert r['success'] or r['steps'] == 800, 'Premature failed episode'
        assert np.isfinite(r['tracking_rmse_m']) and r['tracking_rmse_m'] >= 0
        assert 0 <= r['n_bound_hits'] <= r['steps']
        if r['success']:
            assert abs(r['time_to_success_s'] - r['steps'] * .05) < 1e-9


def summarize(secondary, main, spec):
    methods = list(spec['episodes'])
    pids, tasks, test = spec['plant_ids'], spec['tasks'], spec['test']
    check_grid(secondary, pids, spec['episodes'], tasks)
    # Exactly the same five initial states, never compare 5 vs 10 here.
    base = [r for r in main if r['method'] in BASELINES and r['ep'] < 5]
    check_grid(base, pids, dict.fromkeys(BASELINES, 5), tasks)
    arms = BASELINES + methods
    groups = {(p, m, t): [] for p, m, t in itertools.product(pids, arms, tasks)}
    for r in secondary + base:
        groups[r['pid'], r['method'], r['task_id']].append(r)
    cells = {p: {} for p in pids}
    for p, m in itertools.product(pids, arms):
        sub = [r for t in tasks for r in groups[p, m, t]]
        cells[p][m] = {
            'n': len(sub),
            'sr': float(np.mean([np.mean([r['success'] for r in groups[p, m, t]]) for t in tasks])),
            'rmse_m': float(np.mean([r['tracking_rmse_m'] for r in sub])),
            'queries_per_episode': float(np.mean([r['policy_queries'] for r in sub])),
            'bound_step_share': sum(r['n_bound_hits'] for r in sub) / sum(r['steps'] for r in sub)}
    nominal = cells['L00_nominal']['naive']['sr']
    rpop = [p for p in test if nominal - cells[p]['naive']['sr'] > .10 + 1e-12]
    macro = {}
    for m in arms:
        recovery = {p: (cells[p][m]['sr'] - cells[p]['naive']['sr']) /
                    (nominal - cells[p]['naive']['sr']) for p in rpop}
        macro[m] = {'sr': float(np.mean([cells[p][m]['sr'] for p in test])),
                    'R': interval(list(recovery.values())), 'R_by_plant': recovery}
    comparisons = {}
    for m, b in itertools.product(methods, BASELINES):
        delta = {p: cells[p][m]['sr'] - cells[p][b]['sr'] for p in test}
        time_cells, time_plants = [], {}
        for p in test:
            aa, bb = [], []
            for t in tasks:
                a = [r['time_to_success_s'] for r in groups[p, m, t] if r['success']]
                v = [r['time_to_success_s'] for r in groups[p, b, t] if r['success']]
                if a and v:
                    am, bm = float(np.mean(a)), float(np.mean(v))
                    aa.append(am); bb.append(bm)
                    time_cells.append({'pid': p, 'task_id': t, 'a_s': am, 'b_s': bm,
                                       'n_success_a': len(a), 'n_success_b': len(v)})
            if aa:
                time_plants[p] = {'a_s': float(np.mean(aa)), 'b_s': float(np.mean(bb)), 'n_tasks': len(aa)}
        reduction = (1 - np.mean([x['a_s'] for x in time_plants.values()]) /
                     np.mean([x['b_s'] for x in time_plants.values()])) if time_plants else None
        comparisons[f'{m}__vs__{b}'] = {
            'arms': [m, b], 'sr_difference': interval(list(delta.values())), 'sr_by_plant': delta,
            'conditional_time': {'reduction': float(reduction) if reduction is not None else None,
                                 'plants': time_plants, 'cells': time_cells}}
    return {'status': 'complete', 'secondary_rows': len(secondary), 'matched_main_rows': len(base),
            'episodes_per_task': 5, 'test_plants': test, 'R_population_matched5': rpop,
            'cells': cells, 'macro': macro, 'comparisons': comparisons,
            'inference': 'Descriptive bootstrap only; primary tests and Holm family unchanged',
            'method_claim_authorized': False}


def load_audited(root, manifest, source, expected_files):
    spec = json.loads(Path(manifest).read_text())
    assert (root / 'execution.exit').read_text().strip() == '0'
    paths = sorted((root / 'jobs').glob('*.jsonl'))
    assert len(paths) == expected_files
    rows, hashes = [], {}
    for p in paths:
        assert p.with_suffix('.exit').read_text().strip() == '0', p
        meta = json.loads(p.with_suffix('.meta.json').read_text())
        assert meta['freeze_sha256'] == sha(manifest), p
        assert meta['source_sha256'] == spec['inputs_sha256'][source], p
        assert meta['pack_sha256'] == spec['pack_sha256'], p
        for k in ['model_path', 'lora_path', 'lora_merged', 'supports_request_seed', 'num_actions']:
            assert meta['server'][k] == spec['server'][k], (p, k)
        rows.extend(json.loads(l) for l in p.read_text().splitlines())
        hashes[str(p)] = sha(p)
    return rows, hashes


def report(out):
    lines = ['# CCE v4r1 六项次级实验', '',
             f"完整网格审计通过：{out['secondary_rows']:,}集；主批匹配前5集：{out['matched_main_rows']:,}集。", '',
             '在主结果已知后冻结实施细节，以下均为描述性次级比较。CPU门2与主批P2判定保持不变。', '',
             '| 臂 | 16测试plant宏SR | 匹配5集宏R |', '|---|---:|---:|']
    for m, x in out['macro'].items():
        rv = x['R']['mean']
        rtext = '不可估计' if rv is None else f'{rv:.4f}'
        lines.append(f"| {m} | {x['sr']:.4f} | {rtext} |")
    lines += ['', '| 次臂 | 参照 | SR差(pp) | plant bootstrap 95% CI(pp) | 共同成功耗时缩短 |',
              '|---|---|---:|---|---:|']
    for x in out['comparisons'].values():
        d = x['sr_difference']; ci = d['ci95']; t = x['conditional_time']['reduction']
        tx = '不可估计' if t is None else f'{t:.2%}'
        lines.append(f"| {x['arms'][0]} | {x['arms'][1]} | {d['mean']*100:.3f} | [{ci[0]*100:.3f}, {ci[1]*100:.3f}] | {tx} |")
    lines += ['', '时间仅在成对都有成功的任务格上估计，成功初态不必相同；不能作为含失败成本的总体效率。各比较人群、成功数、逐plant差与原始SHA见JSON。',
              'TE改变重查频率与时间集成两项因素，不能将其差异单独归因为连续性。no_unilateral仍把负τ置0。', '']
    return '\n'.join(lines)


def selftest():
    spec = {'plant_ids': ['L00_nominal', 'a', 'b'], 'test': ['a', 'b'],
            'tasks': [0, 1], 'episodes': {'te_requery': 5}}
    rows = [{'pid': p, 'method': m, 'task_id': t, 'ep': e, 'success': 1, 'steps': 100,
             'time_to_success_s': 5., 'tracking_rmse_m': .02, 'policy_queries': 10, 'n_bound_hits': 0}
            for p, m, t, e in itertools.product(spec['plant_ids'], BASELINES + ['te_requery'], [0, 1], range(5))]
    sec = [r for r in rows if r['method'] == 'te_requery']
    main = [r for r in rows if r['method'] in BASELINES]
    out = summarize(sec, main, spec)
    assert out['macro']['te_requery']['R']['mean'] is None
    assert out['comparisons']['te_requery__vs__naive']['sr_difference']['ci95'] == [0., 0.]
    assert out['comparisons']['te_requery__vs__naive']['conditional_time']['reduction'] == 0
    for bad in [sec[:-1], sec + [sec[0]], [{**r, 'success': 0} for r in sec]]:
        try:
            summarize(bad, main, spec)
        except AssertionError:
            pass
        else:
            raise AssertionError('Accepted invalid grid')
    print('secondary aggregation: matched grid, no R population, conditional time, duplicates, missing, premature failure passed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir'); ap.add_argument('--freeze'); ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    spec = validate_freeze(args.freeze)
    root = Path(args.run_dir)
    sec, hashes = load_audited(root, args.freeze, 'cce/libero_secondary_v4r1.py', 456)
    oldroot = Path(spec['main_run'])
    old, oldhashes = load_audited(oldroot, spec['parent_manifest'], 'cce/libero_family_v3.py', 684)
    previous = json.loads((oldroot / 'results.json').read_text())
    assert oldhashes == previous['raw_sha256'], 'Main raw results changed'
    parent = json.loads(Path(spec['parent_manifest']).read_text())
    check_grid(old, parent['plant_ids'], parent['episodes'], parent['tasks'])
    assert len(sec) == spec['expected_episodes']
    for r in sec:
        frequency = 10 if r['method'] == 'te_requery' else 30
        assert r['policy_queries'] == (r['steps'] + frequency - 1) // frequency, 'Incorrect query cadence'
    out = summarize(sec, old, spec)
    out.update(freeze_sha256=sha(args.freeze), raw_sha256=hashes, main_raw_sha256=oldhashes)
    (root / 'results.json').write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (root / 'report.md').write_text(report(out))
    print(json.dumps({'secondary_rows': len(sec), 'macro_sr': {m: x['sr'] for m, x in out['macro'].items()}}, indent=2))


if __name__ == '__main__':
    main()
