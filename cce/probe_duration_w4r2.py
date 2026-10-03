"""Explicitly frozen prefix-probe ablation and finite closed-loop evaluation."""
import _bootstrap  # noqa: F401
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import requests
from cce.freeze_v4r1 import sha, validate_freeze
from cce.libero_identify_v4 import fit_axis
from cce.identify import realized_deltas, TRANS_EVAL_WINDOW, ROT_WINDOW
from cce.libero_exec_v4r1 import RelativeTheta, predicted_h

ROOT = Path('/opt/cce/data/cce/probe_duration_w4r2_20260911')
PARENT = 'cce/results/libero_family_spec_v4r1_fallback.json'
FREEZE = 'cce/results/probe_duration_w4r2_spec.json'
PLANTS = ['D1_d250', 'D2_t400', 'D3_g10', 'MX02']
PORT = 8067


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def validate():
    spec = read(FREEZE)
    validate_freeze(PARENT)
    assert spec['status'] == 'FROZEN'
    for p, h in spec['inputs_sha256'].items():
        assert sha(p) == h, p
    return spec


def prepare():
    parent = validate_freeze(PARENT, weights=True)
    ROOT.mkdir(exist_ok=False)
    original = read(parent['pack'])
    hats = read('data/cce/libero/theta/theta_hat_libero_panda_v2.json')
    hashes = {PARENT: sha(PARENT), parent['pack']: sha(parent['pack'])}
    packs, errors = {}, {}
    with np.load('data/cce/libero/probes/probe_libero_panda_L00_nominal.npz') as rec:
        reference = rec['targets']
    for sec in [2, 4, 7]:
        pack = copy.deepcopy(original)
        errors[str(sec)] = {}
        for p in PLANTS:
            path = Path('data/cce/libero/probes')/f'probe_libero_panda_{p}.npz'
            assert sha(path) == original['probe_sha256'][p]
            hashes[str(path)] = sha(path)
            with np.load(path) as rec:
                n = sec*20
                u = rec['u'][:n].copy()
                w = realized_deltas(rec['y'][:n+1].copy())
            axes = []
            for i in range(6):
                lo, hi = TRANS_EVAL_WINDOW if i < 3 else ROT_WINDOW
                hi = min(hi, n)
                if hi-lo < 12:
                    axis = dict(delay_steps=0, tau_s=0., gain=1., r2_freerun=None, identifiable=False)
                else:
                    axis = fit_axis(u[:, i], w[:, i], original['nominal_alpha'][i], original['nominal_gain'][i], (lo, hi))
                    if abs(axis['gain']-1.) < 1e-12:
                        axis['gain'] = 1.
                axes.append(axis)
            fit = dict(axes=axes, delay_steps=int(round(np.median([a['delay_steps'] for a in axes[:3]]))),
                       grip_lead_steps=hats[p]['grip_lead_steps_hat'] if sec == 7 else 0)
            if sec == 7:
                assert fit['delay_steps'] == original['plants'][p]['delay_steps']
                assert fit['grip_lead_steps'] == original['plants'][p]['grip_lead_steps']
                for a, b in zip(axes, original['plants'][p]['axes']):
                    assert a == b, (p, a, b)
            pack['plants'][p].update(fit)
            truth = original['plants'][p]['truth_report_only']
            errors[str(sec)][p] = dict(delay_abs_error_steps=abs(fit['delay_steps']-truth['delay_steps']),
                                      tau_median_abs_error_s=abs(float(np.median([a['tau_s'] for a in axes[:3]]))-truth['tau_s']),
                                      gain_median_abs_error=abs(float(np.median([a['gain'] for a in axes[:3]]))-truth['gain']),
                                      grip_estimate_observed=sec == 7,
                                      grip_default_or_estimate_abs_error_steps=abs(fit['grip_lead_steps']-truth['grip_lead_steps']),
                                      identifiable_axes=sum(a['identifiable'] for a in axes))
        pack['fixed_gate'] = {'H0': parent['H0'], 'H': {p: predicted_h(RelativeTheta.from_fit(pack['plants'][p]), np.array(pack['nominal_alpha']), np.array(pack['nominal_gain']), reference) for p in PLANTS}}
        path = ROOT/f'pack_{sec}s.json'; write(path, pack); packs[str(sec)] = str(path); hashes[str(path)] = sha(path)
    for p in ['cce/probe_duration_w4r2.py', 'cce/libero_collect_plant_demos_v4r1.py', 'docs/prereg/CCE_probe_duration_w4r2.md', 'cce/results/libero_fallback_v4r1.json']:
        hashes[p] = sha(p)
    write(ROOT/'identification.json', errors)
    hashes[str(ROOT/'identification.json')] = sha(ROOT/'identification.json')
    write(FREEZE, dict(status='FROZEN', frozen_at=datetime.datetime.now().astimezone().isoformat(),
                       implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                       plants=PLANTS, seconds=[2,4,7], new_episodes=400, reused_7s_episodes=200,
                       packs=packs, server=parent['server'], inputs_sha256=hashes))
    validate()
    print('Freeze prepared; 7s estimated parameters exactly match original.')


def worker(shard):
    spec = validate()
    from cce.libero_collect_plant_demos_v4r1 import configure
    from cce import libero_family_v3 as F
    parent, hats, _, ev, suite = configure()
    info = requests.get(f'http://127.0.0.1:{PORT}/ready', timeout=5).json()
    assert info['supports_request_seed'] and info['lora_path'] == spec['server']['lora_path']
    assert info['model_path'] == spec['server']['model_path']
    client = F.ChunkClient('127.0.0.1', PORT); client.suite_name = 'libero_spatial'; client.verify_repeat = True
    packs = {s: read(p) for s, p in spec['packs'].items()}
    work = [(s,p,t,e) for s in [2,4] for p in PLANTS for t in range(10) for e in range(5)]
    out = ROOT/'run'/f'shard{shard}.jsonl'
    write(out.with_suffix('.meta.json'), dict(freeze_sha256=sha(FREEZE), server=info, shard=shard))
    with out.open('x') as f:
        for s,p,t,e in work[shard::8]:
            F.PACK = packs[str(s)]; F.GATE = F.PACK['fixed_gate']; F.RHO = parent['rho']
            row = F.run_episode(ev, suite, t, e, F.BY_ID_V3[p], 'ours_v4', client, hats[p], F.theta_from_hat(hats[F.NOMINAL_PID]), 800, 1., 1.)
            row.update(probe_seconds=s, suite='libero_spatial', freeze_sha256=sha(FREEZE))
            f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
            print(json.dumps({k:row[k] for k in ['probe_seconds','pid','task_id','ep','success','steps']}), flush=True)


def aggregate():
    spec = validate(); records = []; hashes = {}
    for i in range(8):
        p = ROOT/'run'/f'shard{i}.jsonl'
        assert p.with_suffix('.exit').read_text().strip() == '0'
        assert read(p.with_suffix('.meta.json'))['freeze_sha256'] == sha(FREEZE)
        records.extend(map(json.loads, p.read_text().splitlines())); hashes[str(p)] = sha(p)
    main = read('cce/results/libero_fallback_v4r1.json')
    for path, h in main['raw_sha256'].items():
        assert sha(path) == h
        rr = [dict(r, probe_seconds=7) for r in map(json.loads, Path(path).read_text().splitlines()) if r['pid'] in PLANTS and r['method'] == 'ours_v4' and r['ep'] < 5]
        if rr: records.extend(rr); hashes[path] = h
    keys = Counter((r['probe_seconds'],r['pid'],r['task_id'],r['ep']) for r in records)
    expected = {(s,p,t,e) for s in [2,4,7] for p in PLANTS for t in range(10) for e in range(5)}
    assert set(keys) == expected and set(keys.values()) == {1}
    for r in records:
        assert r['success'] in (0,1) and 1 <= r['steps'] <= 800
        assert r['success'] or r['steps'] == 800
        assert r['policy_queries'] == (r['steps']+29)//30
    cells = {p: {str(s): dict(n=50, sr=float(np.mean([r['success'] for r in records if r['pid']==p and r['probe_seconds']==s]))) for s in [2,4,7]} for p in PLANTS}
    out = dict(status='complete', n=600, new_rows=400, reused_7s_rows=200, cells=cells,
               identification=read(ROOT/'identification.json'), raw_sha256=hashes, freeze_sha256=sha(FREEZE),
               interpretation='Descriptive prefix ablation; nominal model fixed; short prefixes contain no dedicated rotation or gripper excitation.')
    write(ROOT/'results.json', out)
    write('cce/results/CCE_probe_duration_completed_w4r2.json', out)
    lines = ['# 探针前缀时长消融', '', '400条新增闭环与200条完全相同7秒参数主批记录，唯一键及SHA审计通过。4个条件、10任务、相同ep0..4。', '', '| plant | 2秒SR | 4秒SR | 7秒SR |','|---|---:|---:|---:|']
    for p, c in cells.items(): lines.append('| '+p+' | '+' | '.join(f"{c[str(s)]['sr']:.2%}" for s in [2,4,7])+' |')
    lines += ['', '采用同一7秒记录的时间前缀，固定已知标称模型、H0与ρ，不把7秒旋转或夹爪估计泄漏给短前缀。短前缀缺少相应激励时使用不可辨识掩码及零夹爪提前。该比较同时改变持续时间与可见激励阶段，不代表最优短探针设计。', '', '完整估计误差与审计来源见JSON；本补充不改变原P2/K结论。']
    Path('docs/reports/CCE_probe_duration_completed_w4r2.md').write_text('\n'.join(lines)+'\n')


def run():
    spec = validate()
    root = ROOT/'run'; root.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='7', MUJOCO_EGL_DEVICE_ID='7', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    server = None; code = 1
    write(root/'plan.json', dict(started=datetime.datetime.now().astimezone().isoformat(), freeze_sha256=sha(FREEZE), workers=4, shards=8))
    try:
        with (root/'server.log').open('x') as log:
            server = subprocess.Popen(['bash','cce/env.sh','cce/serve_xvla_v4r1.py','--model-path',spec['server']['model_path'],'--lora-path',spec['server']['lora_path'],'--port',str(PORT)], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(root/'owned_server.json', dict(pid=server.pid, pgid=server.pid))
        for _ in range(120):
            if server.poll() is not None: raise RuntimeError('server failed')
            try:
                info = requests.get(f'http://127.0.0.1:{PORT}/ready',timeout=2).json()
                assert info['lora_path'] == spec['server']['lora_path']
                break
            except requests.RequestException: time.sleep(2)
        else: raise RuntimeError('readiness timeout')
        def shard(i):
            with (root/f'shard{i}.log').open('x') as log:
                p = subprocess.run(['bash','cce/env_libero.sh','cce/probe_duration_w4r2.py','--worker',str(i)], env=env, stdout=log, stderr=subprocess.STDOUT)
            (root/f'shard{i}.exit').write_text(str(p.returncode)+'\n')
            if p.returncode: raise RuntimeError(f'shard {i} failed: {p.returncode}')
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(shard, range(8)))
        aggregate(); code = 0
    except BaseException as error:
        write(root/'failure.json', dict(error=repr(error))); raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try: server.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(server.pid, signal.SIGKILL); server.wait()
        (root/'supervisor.exit').write_text(str(code)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--prepare',action='store_true'); ap.add_argument('--worker',type=int); ap.add_argument('--run',action='store_true'); ap.add_argument('--aggregate',action='store_true'); args = ap.parse_args()
    if args.prepare: prepare()
    elif args.worker is not None:
        assert 0 <= args.worker < 8
        worker(args.worker)
    elif args.run: run()
    elif args.aggregate: aggregate()
