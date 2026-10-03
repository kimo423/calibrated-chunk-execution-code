"""Create and verify the prospectively frozen fallback study; fail closed on drift."""
import _bootstrap  # noqa: F401
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from cce.libero_plant_v3 import FAMILY_V3, DEV_PIDS, TEST_PIDS

ROOT = Path('/opt/cce/data/cce/v4r1_20260908')
MAIN = ['naive', 'slow15', 'slow20', 'ours_v4', 'ours_v4_slow15', 'ours_v4_slow20', 'oracle_v4']
SECONDARY = ['slow30', 'ours_v3']
SOURCES = ['cce/libero_family_v3.py', 'cce/libero_exec_v4r1.py', 'cce/identify_v4r1.py',
           'cce/bench_relative_v4r1.py', 'cce/libero_plant_v3.py', 'cce/libero_plant_v2.py',
           'cce/libero_plant.py', 'cce/libero_exec.py', 'cce/common.py', 'cce/plants.py',
           'cce/libero_probe.py', 'cce/libero_identify_v4.py', 'cce/env_libero.sh', 'cce/env.sh',
           'cce/serve_xvla_v4r1.py', 'cce/freeze_v4r1.py', 'cce/run_fallback_v4r1.py',
           'cce/aggregate_fallback_v4r1.py', 'cce/stats_v4r1.py',
           'docs/prereg/CCE_fallback_v4r1_frozen.md',
           'docs/prereg/CCE_v4r1_execution_amendment.md']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def validate_freeze(path, weights=False):
    if not path:
        raise ValueError('A frozen manifest is required')
    spec = json.loads(Path(path).read_text())
    assert spec['status'] == 'FROZEN_FALLBACK' and spec['method_claim_authorized'] is False
    for p, digest in spec['inputs_sha256'].items():
        assert sha(p) == digest, f'Frozen input changed: {p}'
    if weights:
        for p, digest in spec['weights_sha256'].items():
            assert sha(p) == digest, f'Frozen weight changed: {p}'
    return spec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    path = Path(args.out)
    assert not path.exists(), 'Never replace a frozen manifest'
    cpu = json.loads((ROOT/'cpu_report.json').read_text())
    dev = json.loads((ROOT/'dev_audit.json').read_text())
    g0 = json.loads((ROOT/'g0/audit.json').read_text())
    assert not any(cpu['all_rho_pass_both_robots'].values())
    assert dev['status'] == 'complete' and dev['rho'] == 2
    # G0 is independently re-counted to avoid depending on report field names.
    rows = [json.loads(l) for p in (ROOT/'g0').glob('shard*.jsonl') for l in p.read_text().splitlines()]
    assert len(rows) == 100 and sum(r['success'] for r in rows) >= 90
    extra = ['data/cce/libero/theta/theta_hat_libero_panda_v2.json',
             'data/cce/takeover_v4_20260908/new_probes/probe_summary.json',
             'data/cce/libero/probes/probe_libero_panda_L00_nominal.npz']
    extra += [str(ROOT/p) for p in ['theta_libero.json', 'cpu_report.json', 'dev_audit.json', 'g0/audit.json']]
    extra += ['/opt/CalibrationToPrompt/source/X-VLA/evaluation/libero/libero_client.py']
    info = json.loads((ROOT/'server_info.json').read_text())
    assets = []
    for key in ['model_path', 'lora_path']:
        assets += sorted(p for p in Path(info[key]).rglob('*') if p.is_file())
    assert any(p.suffix == '.safetensors' for p in assets)
    spec = {'status': 'FROZEN_FALLBACK', 'frozen_at': datetime.datetime.now().astimezone().isoformat(),
            'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'method_claim_authorized': False, 'reason': 'Gate 2 physical regression failed for every cap',
            'suite': 'libero_spatial', 'plant_ids': [p.pid for p in FAMILY_V3],
            'plants': [p.as_dict() for p in FAMILY_V3], 'dev': list(DEV_PIDS), 'test': list(TEST_PIDS),
            'tasks': list(range(10)), 'episodes': {**dict.fromkeys(MAIN, 10), **dict.fromkeys(SECONDARY, 5)},
            'expected_episodes': 15200, 'rho': 2.,
            'H0': json.loads((ROOT/'dev_design_audit.json').read_text())['gate']['H0'],
            'pack': str(ROOT/'theta_libero.json'), 'pack_sha256': sha(ROOT/'theta_libero.json'),
            'inputs_sha256': {p: sha(p) for p in SOURCES+extra},
            'weights_sha256': {str(p): sha(p) for p in assets}, 'server': info}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(spec, indent=2, ensure_ascii=False)+'\n')
    validate_freeze(path)
    print(json.dumps({'freeze': str(path), 'sha256': sha(path), 'episodes': spec['expected_episodes']}))

if __name__ == '__main__':
    main()
