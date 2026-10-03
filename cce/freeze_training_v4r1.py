"""Freeze both adaptation chains, their data split and evaluation rules."""
import _bootstrap  # noqa: F401
import argparse
import datetime
import json
from pathlib import Path
import subprocess
from cce.freeze_v4r1 import sha, validate_freeze
from cce.freeze_demos_v4r1 import validate_demo_freeze

ROOT = Path('/opt/cce/data/cce/policyside_train_v4r1_20260909')
PARENT = 'cce/results/libero_family_spec_v4r1_fallback.json'
DEMOS = 'cce/results/libero_demo_spec_v4r1.json'
SOURCES = ['cce/policyside_model_v4r1.py', 'cce/train_policyside_v4r1.py',
           'cce/prepare_validation_v4r1.py', 'cce/eval_policyside_offline_v4r1.py',
           'cce/serve_policyside_v4r1.py', 'cce/eval_policyside_online_v4r1.py',
           'cce/freeze_training_v4r1.py', 'cce/run_policyside_training_v4r1.py',
           'cce/xvla_bridge.py', 'cce/_bootstrap.py', 'docs/prereg/CCE_policyside_training_v4r1.md']


def validate_training_freeze(path, data=False):
    spec = json.loads(Path(path).read_text()); assert spec['status'] == 'FROZEN_TRAINING'
    validate_freeze(PARENT, weights=data); validate_demo_freeze(DEMOS)
    for p, h in spec['inputs_sha256'].items(): assert sha(p) == h, p
    if data:
        for p, h in spec['data_sha256'].items(): assert sha(p) == h, p
    return spec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); args = ap.parse_args()
    validate_freeze(PARENT); validate_demo_freeze(DEMOS)
    assert not Path(args.out).exists()
    assert json.loads((ROOT/'smoke_audit.json').read_text())['status'] == 'passed'
    split = json.loads((ROOT/'split.json').read_text())
    assert len(split['train']) == 40 and len(split['validation']) == 10
    files = SOURCES + [PARENT, DEMOS]
    files += [str(p) for p in Path('cce/xvla_patch').rglob('*.py')]
    upstream = Path('/opt/CalibrationToPrompt/source/X-VLA')
    for d in ['models', 'datasets']:
        files += [str(p) for p in (upstream/d).rglob('*.py')]
    files += [str(ROOT/p) for p in ['split.json','train_meta.json','validation_meta.json','validation_panel.json','smoke_audit.json']]
    spec = {'status': 'FROZEN_TRAINING', 'frozen_at': datetime.datetime.now().astimezone().isoformat(),
            'implementation_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'train_meta': str(ROOT/'train_meta.json'), 'validation_samples': str(ROOT/'validation_samples.pt'),
            'modes': ['ft_demo','prompt_fit'], 'steps': 5000, 'effective_batch': 32, 'microbatch': 16,
            'evaluation_plants': ['L00_nominal','D1_d250'], 'primary_results_known': True,
            'inputs_sha256': {p:sha(p) for p in files},
            'data_sha256': {**split['raw_sha256'],str(ROOT/'validation_samples.pt'):sha(ROOT/'validation_samples.pt')}}
    with Path(args.out).open('x') as f: f.write(json.dumps(spec,indent=2)+'\n')
    validate_training_freeze(args.out); print('Training freeze:',sha(args.out))


if __name__ == '__main__': main()
