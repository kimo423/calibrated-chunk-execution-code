"""Finite collection -> data gate -> frozen training -> conditional evaluation."""
import _bootstrap  # noqa: F401
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from cce.freeze_training_v4r1 import validate_training_freeze, sha

ROOT = Path('/opt/cce/data/cce/policyside_supplement_w4r3_20260911')
COLLECTION = Path('/opt/cce/data/cce/demo_supplement_w4r3_20260911')
ORIGINAL = 'cce/results/libero_training_spec_v4r1.json'
SUPPLEMENT = 'cce/results/demo_supplement_w4r3_spec.json'
PREFLIGHT = 'cce/results/policyside_supplement_w4r3_preflight.json'
SOURCES = ['cce/continue_policyside_w4r3.py', 'cce/prepare_validation_w4r2.py',
           'cce/run_policyside_training_w4r2.py', 'docs/prereg/CCE_policyside_supplement_w4r3.md']


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(p)


def check_preflight():
    s = read(PREFLIGHT)
    assert s['status'] == 'FROZEN_PREFLIGHT'
    for p, h in s['sha256'].items():
        assert sha(p) == h, p
    validate_training_freeze(ORIGINAL)


def commit(paths, message, root):
    msg = root/'commit_message.txt'
    msg.write_text(message+'\n')
    subprocess.run(['bash', 'scripts/commit.sh', str(msg), *paths], check=True)


def prepare(plant, result):
    check_preflight()
    assert result['status'] == 'complete' and result['total'] == 50
    assert result['handler_audit'] == 'passed'
    assert set(result['per_task'].values()) == {5}
    root = ROOT/plant
    root.mkdir()
    meta = read(result['meta'])
    train, val = [], []
    for task in range(10):
        pairs = []
        for p in meta['datalist']:
            match = re.fullmatch(re.escape(plant)+r'__t(\d+)__e(\d+)\.h5', Path(p).name)
            assert match, p
            if int(match[1]) == task:
                pairs.append((int(match[2]), p))
        pairs.sort()
        assert len(pairs) == 5 and len({ep for ep, _ in pairs}) == 5
        assert all(10 <= ep < 50 for ep, _ in pairs)
        train.extend(p for _, p in pairs[:4]); val.append(pairs[4][1])
    assert len(set(train+val)) == 50 and not set(train) & set(val)
    for p, h in result['h5_sha256'].items():
        assert sha(p) == h, p
    assert set(train+val) == set(result['h5_sha256'])
    write(root/'split.json', dict(train=train, validation=val, raw_sha256=result['h5_sha256']))
    write(root/'train_meta.json', {**meta, 'datalist': train})
    write(root/'validation_meta.json', {**meta, 'datalist': val})
    with (root/'prepare_validation.log').open('x') as log:
        p = subprocess.run(['bash', 'cce/env.sh', 'cce/prepare_validation_w4r2.py', '--root', str(root)], stdout=log, stderr=subprocess.STDOUT)
    (root/'prepare_validation.exit').write_text(str(p.returncode)+'\n')
    assert p.returncode == 0
    parent = validate_training_freeze(ORIGINAL, data=True)
    files = SOURCES+[PREFLIGHT, ORIGINAL, SUPPLEMENT, str(COLLECTION/'results.json')]
    files += [str(root/p) for p in ['split.json', 'train_meta.json', 'validation_meta.json', 'validation_panel.json']]
    spec = {**parent, 'frozen_at': datetime.datetime.now().astimezone().isoformat(),
            'implementation_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'train_meta': str(root/'train_meta.json'), 'validation_samples': str(root/'validation_samples.pt'),
            'evaluation_plants': ['L00_nominal', plant], 'supplemental_collection': True,
            'inputs_sha256': {**parent['inputs_sha256'], **{p:sha(p) for p in files}},
            'data_sha256': {**result['h5_sha256'], str(root/'validation_samples.pt'):sha(root/'validation_samples.pt')}}
    frozen = f'cce/results/policyside_supplement_{plant}_w4r3_spec.json'
    assert not Path(frozen).exists()
    write(frozen, spec); validate_training_freeze(frozen, data=True)
    commit([frozen], f'Freeze supplemental {plant} adaptation data and evaluation', root)
    return frozen, root


def free_gpus():
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'], text=True)
    occupied = set(apps.splitlines())
    free = {int(i) for i,u,m in (line.split(',') for line in raw.splitlines()) if int(m) < 256 and u.strip() not in occupied}
    if 7 in free:
        for i in [0,3,4,5]:
            if i in free:
                return [i,7]
    return None


def audit_result(root, frozen, results):
    spec = validate_training_freeze(frozen, data=True)
    hashes = {frozen: sha(frozen)}
    for mode, result in results.items():
        d = root/'main'/mode
        for name in ['chain.exit','training.exit','initial_offline.exit','offline.exit']:
            assert (d/name).read_text().strip() == '0'
        a, initial = read(d/'train/training_audit.json'), read(d/'train/initialization.json')
        assert a['status'] == 'passed' and a['steps'] == 5000 and not a['smoke']
        assert a['frozen_sha256_before'] == a['frozen_sha256_after'] == initial['frozen_sha256']
        assert initial['adapter_loaded_exactly'] and a['changed_trainable_tensors'] > 0
        losses = [json.loads(x) for x in (d/'train/losses.jsonl').read_text().splitlines()]
        assert [x['step'] for x in losses] == list(range(1,5001))
        assert all(math.isfinite(x['loss']) for x in losses)
        off = read(d/'offline.json'); h = off['horizons']['30']
        passed = h['position_mae_m'] < h['persistence_position_mae_m'] and h['all10_mae'] < h['persistence_all10_mae']
        assert off['G0_prime_pass'] == passed and off['n_samples'] == 200
        if passed:
            assert result['status'] == 'complete'
            rows = []
            for p, digest in result['raw_sha256'].items():
                assert sha(p) == digest and Path(p).with_suffix('.exit').read_text().strip() == '0'
                rows.extend(json.loads(x) for x in Path(p).read_text().splitlines()); hashes[p] = digest
            expected = {(p,t,e) for p in spec['evaluation_plants'] for t in range(10) for e in range(10)}
            assert len(rows) == 200 and {(r['pid'],r['task_id'],r['ep']) for r in rows} == expected
            for r in rows:
                assert r['policy_mode'] == mode and r['training_freeze'] == frozen
                assert r['checkpoint'] == str(d/'train/ckpt-5000')
            for p in spec['evaluation_plants']:
                assert sum(r['success'] for r in rows if r['pid'] == p)/100 == result['sr'][p]
        else:
            assert result['status'] == 'G0_prime_failed' and not result['online_started']
        files = [d/'train/training_audit.json',d/'train/initialization.json',d/'train/losses.jsonl',d/'initial_offline.json',d/'offline.json']
        files += list((d/'train/ckpt-5000').glob('*'))
        hashes.update({str(p):sha(p) for p in files if p.is_file()})
    return dict(status='passed', inputs_sha256=hashes)


def run():
    check_preflight(); ROOT.mkdir(exist_ok=False)
    write(ROOT/'launch_state.json', dict(pid=os.getpid(), started=datetime.datetime.now().astimezone().isoformat()))
    code = 1
    try:
        write(ROOT/'status.json', dict(phase='waiting_collection', collection=str(COLLECTION)))
        while not (COLLECTION/'run/supervisor.exit').exists():
            pid = read(COLLECTION/'launch.json')['pid']
            cmd = Path(f'/proc/{pid}/cmdline')
            assert cmd.exists() and b'demo_supplement_w4r3.py' in cmd.read_bytes(), 'Collection supervisor disappeared without exit'
            time.sleep(20)
        assert (COLLECTION/'run/supervisor.exit').read_text().strip() == '0'
        result = read(COLLECTION/'results.json')
        assert result['status'] == 'execution_complete' and result['new_attempts'] <= 20
        results = {}
        for plant in ['D2_t400']:
            data = result['plants'][plant]
            if data['status'] != 'complete':
                results[plant] = dict(status='data_insufficient', total=data['total'], per_task=data['per_task'], training_started=False)
                continue
            write(ROOT/'status.json', dict(phase='preparing_validation_and_freeze', plant=plant))
            frozen, root = prepare(plant, data)
            write(ROOT/'status.json', dict(phase='waiting_free_gpus', plant=plant))
            gpus = free_gpus()
            while gpus is None:
                time.sleep(30); gpus = free_gpus()
            write(ROOT/'status.json', dict(phase='training_evaluation_chain', plant=plant, gpus=gpus))
            write(root/'gpu_assignment.json', dict(gpus=gpus, snapshot=subprocess.check_output(['nvidia-smi'], text=True)))
            from cce.run_policyside_training_w4r2 import gpu_snapshot
            gpu_snapshot(root, f'{plant}启动前，计划GPU{gpus}')
            with (root/'supervisor.log').open('x') as log:
                p = subprocess.run(['bash','cce/env.sh','cce/run_policyside_training_w4r2.py','--freeze',frozen,
                                    '--out-dir',str(root/'main'),'--gpus',*map(str,gpus)],stdout=log,stderr=subprocess.STDOUT)
            (root/'supervisor.exit').write_text(str(p.returncode)+'\n')
            assert p.returncode == 0
            modes = read(root/'main/results.json')
            audit = audit_result(root, frozen, modes)
            results[plant] = dict(status='executed', freeze=frozen, modes=modes, independent_audit=audit)
        output = dict(status='execution_complete', plants=results, completed_at=datetime.datetime.now().astimezone().isoformat())
        write(ROOT/'results.json', output)
        write('cce/results/CCE_policyside_supplement_w4r3.json', output)
        lines = ['# W3固定补充预算执行结果', '', '原预算内的数据不足和主检验乙分支不变。本报告为新增采集预算下的补充分析，单训练种子。', '', '| plant | 数据/训练状态 | FT-demo nominal/target SR | prompt-fit nominal/target SR |', '|---|---|---|---|']
        for plant, r in results.items():
            vals = []
            for mode in ['ft_demo','prompt_fit']:
                m = r.get('modes',{}).get(mode,{})
                vals.append(f"{m['sr']['L00_nominal']:.2%} / {m['sr'][plant]:.2%}" if m.get('status') == 'complete' else m.get('status','未训练'))
            lines.append(f"| {plant} | {r['status']} | {' | '.join(vals)} |")
        lines += ['', '每个通过数据门的plant为40训练/10验证，最终5,000步模型；G0′通过才进行200集在线。数据不足或G0′失败不计作有效适配后的闭环负结果。结果JSON包含逐文件哈希、冻结参数审计与完整在线网格核对。']
        Path('docs/reports/CCE_policyside_supplement_w4r3.md').write_text('\n'.join(lines)+'\n')
        write(ROOT/'status.json', output); code = 0
    except BaseException as e:
        write(ROOT/'failure.json', dict(error=repr(e))); raise
    finally:
        (ROOT/'supervisor.exit').write_text(str(code)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--freeze-preflight', action='store_true'); a = ap.parse_args()
    if a.freeze_preflight:
        assert not Path(PREFLIGHT).exists()
        validate_training_freeze(ORIGINAL)
        write(PREFLIGHT, dict(status='FROZEN_PREFLIGHT', frozen_at=datetime.datetime.now().astimezone().isoformat(),
                              sha256={p:sha(p) for p in SOURCES+[ORIGINAL,SUPPLEMENT]}))
        check_preflight()
    else:
        run()
