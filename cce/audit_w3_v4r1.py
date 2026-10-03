"""Post-run independent W3 audit; never changes frozen runners or hypotheses."""
import _bootstrap  # noqa: F401
from collections import Counter
import itertools
import json
from pathlib import Path
import numpy as np
from cce.freeze_training_v4r1 import validate_training_freeze, sha
from cce.run_object_subset_v4r1 import validate as validate_object

DATA = Path('/opt/cce/data/cce')
TRAIN = DATA/'policyside_train_v4r1_20260909/main'
OBJECT = DATA/'object_v4r1_20260909/subset'
PLANTS = ['L00_nominal', 'D1_d250']
BASE = ['naive', 'slow20', 'ours_v4', 'ours_v4_slow15']


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    raw = Path(path).read_bytes()
    assert raw.endswith(b'\n'), path
    return [json.loads(line) for line in raw.splitlines()]


def interval(values):
    values = np.asarray(values, float)
    draws = np.random.default_rng(20260908).choice(values, (10000, len(values)), replace=True).mean(1)
    return {'n': len(values), 'mean': float(values.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist()}


def grid(records, expected):
    keys = Counter((r['pid'], r['method'], r['task_id'], r['ep']) for r in records)
    assert set(keys) == expected and set(keys.values()) == {1}
    for r in records:
        assert r['success'] in (0, 1) and 1 <= r['steps'] <= 800
        assert r['success'] or r['steps'] == 800
        assert np.isfinite(r['tracking_rmse_m']) and r['tracking_rmse_m'] >= 0
        assert 0 <= r['n_bound_hits'] <= r['steps']
        if r['success']:
            assert abs(r['time_to_success_s'] - r['steps']*.05) < 1e-9
        else:
            assert r['time_to_success_s'] is None


def audit_training():
    freeze = 'cce/results/libero_training_spec_v4r1.json'
    validate_training_freeze(freeze, data=True)
    assert (TRAIN/'supervisor.exit').read_text().strip() == '0'
    main = read('cce/results/libero_fallback_v4r1.json')
    records = []
    sources = {freeze: sha(freeze)}
    for path, digest in main['raw_sha256'].items():
        assert sha(path) == digest, path
        records += [r for r in rows(path) if r['pid'] in PLANTS and r['method'] in BASE]
        sources[path] = digest
    grid(records, set(itertools.product(PLANTS, BASE, range(10), range(10))))
    out = {'status': 'passed', 'online_rows': 400, 'modes': {}, 'cells': {}, 'comparisons': {},
           'K4': 'not_evaluable: only 1 of the required 3 plants has a complete training dataset',
           'inference': 'Descriptive paired task bootstrap; one plant and one training seed; no Holm-family additions.'}
    for mode in ['ft_demo', 'prompt_fit']:
        root = TRAIN/mode
        for name in ['chain.exit', 'training.exit', 'initial_offline.exit', 'offline.exit']:
            assert (root/name).read_text().strip() == '0'
        audit = read(root/'train/training_audit.json')
        init = read(root/'train/initialization.json')
        assert audit['status'] == 'passed' and audit['steps'] == 5000 and not audit['smoke']
        assert audit['frozen_sha256_before'] == audit['frozen_sha256_after'] == init['frozen_sha256']
        assert audit['changed_trainable_tensors'] > 0 and init['adapter_loaded_exactly']
        losses = rows(root/'train/losses.jsonl')
        assert [r['step'] for r in losses] == list(range(1, 5001))
        assert all(np.isfinite([r['loss'], r['lr']]).all() for r in losses)
        before, after = read(root/'initial_offline.json'), read(root/'offline.json')
        assert after['n_samples'] == 200 and after['n_tasks'] == 10 and not after['smoke']
        for h, v in after['horizons'].items():
            a, b = np.array(v['model_mae_by_dim']), np.array(v['persistence_mae_by_dim'])
            assert a.shape == b.shape == (10,) and np.isfinite([a, b]).all()
            assert np.isclose(a[:3].mean(), v['position_mae_m'])
            assert np.isclose(a.mean(), v['all10_mae'])
            assert np.allclose(b, before['horizons'][h]['persistence_mae_by_dim'])
        h = after['horizons']['30']
        assert h['position_mae_m'] < h['persistence_position_mae_m']
        assert h['all10_mae'] < h['persistence_all10_mae'] and after['G0_prime_pass']
        assert np.isclose(np.mean([s['position_mae_m'] for s in after['samples']]), h['position_mae_m'])
        result = read(root/'results.json')
        online = []
        assert len(result['raw_sha256']) == 4
        for path, digest in result['raw_sha256'].items():
            assert sha(path) == digest and Path(path).with_suffix('.exit').read_text().strip() == '0'
            sources[path] = digest
            online.extend(rows(path))
        grid(online, set(itertools.product(PLANTS, ['naive'], range(10), range(10))))
        for r in online:
            assert r['policy_mode'] == mode and r['training_freeze'] == freeze
            assert r['checkpoint'] == str(root/'train/ckpt-5000')
            assert r['policy_queries'] == (r['steps']+29)//30
        for p in PLANTS:
            assert abs(np.mean([r['success'] for r in online if r['pid'] == p]) - result['sr'][p]) < 1e-12
            for t in range(10):
                assert abs(np.mean([r['success'] for r in online if r['pid'] == p and r['task_id'] == t]) - result['task_sr'][p][str(t)]) < 1e-12
        records += [dict(r, method=mode) for r in online]
        for path in [root/'train/training_audit.json', root/'train/initialization.json', root/'train/losses.jsonl',
                     root/'initial_offline.json', root/'offline.json', root/'results.json'] + list((root/'train/ckpt-5000').glob('*')):
            if path.is_file(): sources[str(path)] = sha(path)
        out['modes'][mode] = {'training_audit': audit, 'trainable_parameters': init['trainable_parameters'],
                              'wall_s': read(root/'train/status.json')['wall_s'],
                              'loss_first100': float(np.mean([r['loss'] for r in losses[:100]])),
                              'loss_last100': float(np.mean([r['loss'] for r in losses[-100:]])),
                              'initial_horizons': before['horizons'], 'final_horizons': after['horizons']}
    for p in PLANTS:
        out['cells'][p] = {}
        for m in BASE + ['ft_demo', 'prompt_fit']:
            task = [float(np.mean([r['success'] for r in records if r['pid'] == p and r['method'] == m and r['task_id'] == t])) for t in range(10)]
            out['cells'][p][m] = {'n': 100, 'sr': float(np.mean(task)), 'task_sr': task}
        out['comparisons'][p] = {}
        for a, b in itertools.product(['ft_demo', 'prompt_fit'], ['naive', 'ours_v4']):
            out['comparisons'][p][a+'__vs__'+b] = interval(np.array(out['cells'][p][a]['task_sr']) - out['cells'][p][b]['task_sr'])
    nominal, target = out['cells']['L00_nominal']['naive']['sr'], out['cells']['D1_d250']['naive']['sr']
    out['target_R'] = {m: (v['sr']-target)/(nominal-target) for m, v in out['cells']['D1_d250'].items()}
    out['inputs_sha256'] = sources
    return out


def audit_object():
    freeze = 'cce/results/libero_object_subset_spec_v4r1.json'
    spec = validate_object(freeze)
    original = read(OBJECT/'results.json')
    assert (OBJECT/'supervisor.exit').read_text().strip() == '0'
    assert len(original['raw_sha256']) == 32
    records = []
    for path, digest in original['raw_sha256'].items():
        assert sha(path) == digest and Path(path).with_suffix('.exit').read_text().strip() == '0'
        meta = read(Path(path).with_suffix('.meta.json'))
        assert meta['freeze_sha256'] == sha(freeze) and meta['source_sha256'] == sha('cce/libero_object_subset_v4r1.py')
        assert meta['pack_sha256'] == sha(spec['pack']) and meta['server']['lora_path'] == spec['adapter']['path']
        records += rows(path)
    grid(records, set(itertools.product(spec['plants'], spec['methods'], range(10), range(5))))
    for r in records:
        assert r['suite'] == 'libero_object'
        f = {'slow20': 60, 'ours_v4_slow15': 45}.get(r['method'], 30)
        assert r['policy_queries'] == (r['steps']+f-1)//f
    cells = {}
    for p in spec['plants']:
        cells[p] = {m: float(np.mean([r['success'] for r in records if r['pid'] == p and r['method'] == m])) for m in spec['methods']}
        for m in spec['methods']: assert abs(cells[p][m] - original['cells'][p][m]['sr']) < 1e-12
    nominal = []
    for path, digest in read(DATA/'object_v4r1_20260909/g0/results.json')['raw_sha256'].items():
        assert sha(path) == digest
        nominal += [r for r in rows(path) if r['ep'] < 5]
    assert len(nominal) == 50
    sr0 = np.mean([r['success'] for r in nominal])
    assert sr0 == original['nominal_sr_matched5']
    population = [p for p in spec['plants'] if sr0-cells[p]['naive'] > .10+1e-12]
    assert population == original['R_population']
    for m in spec['methods']:
        assert abs(np.mean([cells[p][m] for p in spec['plants']])-original['macro'][m]['sr']) < 1e-12
        R = interval([(cells[p][m]-cells[p]['naive'])/(sr0-cells[p]['naive']) for p in population])
        assert np.allclose([R['mean']]+R['ci95'], [original['macro'][m]['R']['mean']]+original['macro'][m]['R']['ci95'])
    common = []
    times = {m: {} for m in spec['methods']}
    for p in spec['plants']:
        pt = {m: [] for m in spec['methods']}
        for t in range(10):
            vals = {m: [r['time_to_success_s'] for r in records if r['pid'] == p and r['method'] == m and r['task_id'] == t and r['success']] for m in spec['methods']}
            if all(vals.values()):
                common.append({'pid': p, 'task_id': t, 'n_success': {m: len(v) for m, v in vals.items()}})
                for m, v in vals.items(): pt[m].append(np.mean(v))
        for m, v in pt.items():
            if v: times[m][p] = float(np.mean(v))
    assert common == original['common_success_cells']
    for m, v in times.items(): assert abs(np.mean(list(v.values())) - original['macro'][m]['T_common_s']) < 1e-12
    for key, value in original['differences'].items():
        a, b = key.split('__vs__'); x = interval([cells[p][a]-cells[p][b] for p in spec['plants']])
        assert np.allclose([x['mean']]+x['ci95'], [value['mean']]+value['ci95'])
    original['independent_audit'] = {'status': 'passed', 'n_rows': len(records), 'n_files': 32,
                                   'checks': ['SHA/source/meta/exit', 'unique complete grid', 'SR', 'matched G0', 'R population', 'bootstrap', 'conditional time population']}
    return original


def reports(t, o):
    lines = ['# CCE D1策略适配：完成审计', '',
             '两次5,000步训练、400集在线评测均完整，训练/数据冻结SHA、原始逐集SHA、退出码、唯一键、最终checkpoint路径及离线聚合复核通过。所有冻结参数训练前后SHA不变。', '',
             '| 方法 | 标称SR | D1_d250 SR | D1恢复R |', '|---|---:|---:|---:|']
    for m in BASE + ['ft_demo', 'prompt_fit']:
        lines.append(f"| {m} | {t['cells']['L00_nominal'][m]['sr']:.2%} | {t['cells']['D1_d250'][m]['sr']:.2%} | {t['target_R'][m]:.3f} |")
    lines += ['', '所有臂同10任务×ep0..9；微调50条采集预算中40条训练、10条验证，单训练种子。原策略标称96%、D1 46%作为共同恢复率分母，不能使用微调后已经下降的标称率人为抬高R。', '',
              '## 配对比较', '', '| 条件 | 差值 | SR差(pp) | 任务配对bootstrap 95% CI(pp) |', '|---|---|---:|---|']
    for p, comparisons in t['comparisons'].items():
        for name, v in comparisons.items():
            lines.append(f"| {p} | {name} | {100*v['mean']:.2f} | [{100*v['ci95'][0]:.2f}, {100*v['ci95'][1]:.2f}] |")
    lines += ['', '描述性分析以任务为重采样单位（10个），不把100集当独立任务；不并入主Holm家族。两种微调均提高目标延迟条件的成功率，同时损伤标称。不能从这个单plant/单种子结果推断策略侧不可补偿。K4原要求3个plant中至少2个，当前仅1个具备完整训练数据，故K4不可判定，不能写“通过”。', '', '## 离线误差与训练', '']
    for m, v in t['modes'].items():
        lines += [f"### {m}", '', f"更新5,000步，训练耗时{v['wall_s']/3600:.3f}小时，可训练参数{v['trainable_parameters']:,}，前/后100步平均loss {v['loss_first100']:.6f} / {v['loss_last100']:.6f}。", '',
                  '| 视界 | 训练前xyz MAE(mm) | 训练后xyz MAE(mm) | 平凡预测器(mm) | 优于平凡的维数/10 |', '|---|---:|---:|---:|---:|']
        for h, x in v['final_horizons'].items():
            n = sum(a < b for a, b in zip(x['model_mae_by_dim'], x['persistence_mae_by_dim']))
            lines.append(f"| {h} | {1000*v['initial_horizons'][h]['position_mae_m']:.3f} | {1000*x['position_mae_m']:.3f} | {1000*x['persistence_position_mae_m']:.3f} | {n} |")
        lines += ['', '| 30拍维度 | 模型MAE | 平凡预测器MAE |', '|---|---:|---:|']
        h = v['final_horizons']['30']
        for name, a, b in zip(['x','y','z','r6d_0','r6d_1','r6d_2','r6d_3','r6d_4','r6d_5','gripper'], h['model_mae_by_dim'], h['persistence_mae_by_dim']):
            lines.append(f'| {name} | {a:.6f} | {b:.6f} |')
        lines += ['']
    lines += ['两臂30拍的10个维度事实上均优于平凡预测器，但1拍并非如此。冻结G0′采用30拍xyz平均与all10平均；其通过不代表闭环正控或全视界逐维胜出。', '',
              'checkpoint与完整SHA在`cce/results/libero_policyside_completed_v4r1.json`。原始资料在`data/cce/policyside_train_v4r1_20260909/main/`。']
    Path('docs/reports/CCE_policyside_completed_v4r1.md').write_text('\n'.join(lines)+'\n')
    lines = ['# LIBERO-object第二套件：完成独立审计', '', '32分片、1,600唯一键、原始及冻结SHA、SR/恢复率/时间人群/区间均通过独立复算。G0为97/100；扩展每任务仅ep0..4，与G0匹配的50集标称成功率为100%，恢复R按这个匹配分母计算。', '',
             '| plant | naive | slow20 | r1 | r1+slow15 |', '|---|---:|---:|---:|---:|']
    for p, c in o['cells'].items(): lines.append('| '+p+' | '+' | '.join(f"{c[m]['sr']:.2%}" for m in BASE)+' |')
    lines += ['| **8plant等权宏均值** | '+' | '.join(f"{o['macro'][m]['sr']:.2%}" for m in BASE)+' |', '',
              '| 比较 | 差(pp) | plant bootstrap 95% CI(pp) |', '|---|---:|---|']
    for name, v in o['differences'].items(): lines.append(f"| {name} | {100*v['mean']:.2f} | [{100*v['ci95'][0]:.2f}, {100*v['ci95'][1]:.2f}] |")
    lines += ['', f"恢复R的7个条件：{', '.join(o['R_population'])}。D2_g130无naive损失，不进入R分母，但保留在8plant SR均值。共同成功时间仅覆盖{o['common_success_cells'].__len__()}个plant×任务格；不能宣称全体效率提高。", '',
              '这是同一Panda本体的跨套件、不同官方策略适配器验证。独立object标称探针已记录，但执行器沿用冻结spatial参数，不在object上重选H0/ρ/θ。只作预先冻结的次级描述，不改变spatial的P2检验或CPU门2结论，不称跨本体迁移。', '',
              '完整时间、恢复率与逐文件SHA：`cce/results/libero_object_completed_v4r1.json`。']
    Path('docs/reports/CCE_object_completed_v4r1.md').write_text('\n'.join(lines)+'\n')


def main():
    t, o = audit_training(), audit_object()
    for name, value in [('policyside', t), ('object', o)]:
        Path(f'cce/results/libero_{name}_completed_v4r1.json').write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    reports(t, o)
    print(json.dumps({'training': t['status'], 'object': o['independent_audit'], 'target_R': t['target_R'], 'K4': t['K4']}, indent=2))


if __name__ == '__main__':
    main()
