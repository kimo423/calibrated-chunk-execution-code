"""Freeze bounded teacher-demo acquisition, independent of later training."""
import _bootstrap  # noqa: F401
import argparse
import datetime
import json
from pathlib import Path
import subprocess
from cce.freeze_v4r1 import validate_freeze, sha

PARENT = 'cce/results/libero_family_spec_v4r1_fallback.json'
UPSTREAM = Path('/opt/CalibrationToPrompt/source/X-VLA/datasets')
SOURCES = ['cce/libero_collect_plant_demos_v4r1.py', 'cce/audit_demo_handler_v4r1.py',
           'cce/freeze_demos_v4r1.py', 'cce/run_demo_collection_v4r1.py',
           'docs/prereg/CCE_demo_collection_v4r1.md']


def validate_demo_freeze(path):
    spec = json.loads(Path(path).read_text())
    assert spec['status'] == 'FROZEN_DEMO_COLLECTION'
    validate_freeze(spec['parent_manifest'])
    assert sha(spec['parent_manifest']) == spec['parent_sha256']
    for p, h in spec['inputs_sha256'].items():
        assert sha(p) == h, p
    return spec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); args = ap.parse_args()
    validate_freeze(PARENT)
    assert not Path(args.out).exists()
    dev = Path('/opt/cce/data/cce/policyside_v4r1_20260909/dev_demo')
    assert json.loads((dev / 'collection.json').read_text())['recording_replay_exact']
    assert json.loads((dev / 'handler_audit.json').read_text())['status'] == 'passed'
    paths = SOURCES + [str(UPSTREAM / p) for p in ['dataset.py', 'utils.py', 'domain_handler/base.py',
                                                                  'domain_handler/simulations.py']]
    paths += [str(dev / p) for p in ['collection.json', 'handler_audit.json']]
    spec = {'status': 'FROZEN_DEMO_COLLECTION', 'parent_manifest': PARENT, 'parent_sha256': sha(PARENT),
            'frozen_at': datetime.datetime.now().astimezone().isoformat(),
            'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'plants': ['D1_d250', 'D2_t400', 'MX02'], 'fallback_plant': 'MX03',
            'tasks': list(range(10)), 'candidate_episodes': list(range(10, 20)), 'target_per_task': 5,
            'main_results_known': True, 'inputs_sha256': {p: sha(p) for p in paths}}
    with Path(args.out).open('x') as f:
        f.write(json.dumps(spec, indent=2) + '\n')
    validate_demo_freeze(args.out); print('Frozen demo collection:', sha(args.out))


if __name__ == '__main__':
    main()
