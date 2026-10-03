"""Freeze the remaining six secondary arms after acknowledging known main results."""
import _bootstrap  # noqa: F401
import argparse
import datetime
import json
from pathlib import Path
import subprocess
from cce.freeze_v4r1 import sha,validate_freeze as validate_parent

PARENT='cce/results/libero_family_spec_v4r1_fallback.json'
METHODS=['universal_fixed','te_requery','v4_no_gate','v4_no_unilateral','v4_no_cap','v4_no_greybox']
SOURCES=['cce/freeze_secondary_v4r1.py','cce/libero_secondary_v4r1.py','cce/temporal_ensemble_v4r1.py',
         'cce/run_secondary_v4r1.py','cce/aggregate_secondary_v4r1.py','docs/prereg/CCE_secondary_v4r1.md']

def validate_freeze(path,weights=False):
    spec=json.loads(Path(path).read_text())
    assert spec['status']=='FROZEN_SECONDARY'
    validate_parent(spec['parent_manifest'],weights=weights)
    assert sha(spec['parent_manifest'])==spec['parent_sha256']
    for p,h in spec['inputs_sha256'].items():assert sha(p)==h,f'Changed secondary input: {p}'
    assert sha(Path(spec['main_run'])/'results.json')==spec['main_results_sha256']
    return spec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    parent=validate_parent(PARENT)
    assert not Path(args.out).exists()
    out={k:parent[k] for k in ['suite','plant_ids','plants','dev','test','tasks','rho','H0','pack','pack_sha256','server']}
    out.update(status='FROZEN_SECONDARY',parent_manifest=PARENT,parent_sha256=sha(PARENT),
               frozen_at=datetime.datetime.now().astimezone().isoformat(),
               implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               episodes=dict.fromkeys(METHODS,5),expected_episodes=5700,
               primary_results_known=True,method_claim_authorized=False,
               inputs_sha256={p:sha(p) for p in SOURCES},
               main_run='/opt/cce/data/cce/v4r1_20260908/fallback_main',
               main_results_sha256=sha('/opt/cce/data/cce/v4r1_20260908/fallback_main/results.json'))
    with Path(args.out).open('x') as f:f.write(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    validate_freeze(args.out);print('Frozen secondary episodes:',5700,'SHA:',sha(args.out))

if __name__=='__main__':main()
