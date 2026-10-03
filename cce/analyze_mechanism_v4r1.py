"""Aggregate pre-recorded step traces for the W4 mechanism diagnosis."""
import _bootstrap  # noqa: F401
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

ROOT=Path('/opt/cce/data/cce/v4r1_20260908/fallback_main')
OUT=Path('cce/results/CCE_mechanism_v4r1.json')

def main():
    raise RuntimeError("Deprecated residual indexing; use cce/mechanism_w4r2.py. Historical output is invalid and retained for provenance.")
    groups=defaultdict(lambda:{'episodes':0,'steps':0,'bound_steps':0,'residual':[],'residual_by_bin':defaultdict(list)})
    files=list(ROOT.rglob('*_steps/*.npz'))
    for p in files:
        parts=p.stem.split('__'); plant,method=parts[0],parts[1]
        with np.load(p) as z: a=z['steps']
        g=groups[plant,method];g['episodes']+=1;g['steps']+=len(a)
        u=a[:,14:20];ap=a[:,27:33]
        pos=np.max(np.abs(u[:,:3]),axis=1);rot=np.linalg.norm(u[:,3:],axis=1)
        hit=(pos>=1-1e-9)|(rot>=1-1e-9);g['bound_steps']+=int(hit.sum())
        # applied-vs-command residual, retaining separate temporal bins.
        rr=np.linalg.norm(ap[:,:3]-u[:,:3],axis=1);g['residual'].extend(rr.tolist())
        n=len(a)
        for i,x in enumerate(rr):
            k=min(3,4*i//max(1,n));g['residual_by_bin'][k].append((float(x),bool(hit[i])))
    out={'status':'complete','source_root':str(ROOT),'n_trace_files':len(files),'groups':{}}
    for (plant,method),g in sorted(groups.items()):
        out['groups'].setdefault(plant,{})[method]={
          'episodes':g['episodes'],'steps':g['steps'],'bound_step_share':g['bound_steps']/g['steps'],
          'mean_command_residual_m':float(np.mean(g['residual'])) if g['residual'] else None,
          'p95_command_residual_m':float(np.quantile(g['residual'],.95)) if g['residual'] else None,
          'bound_step_share_by_quarter':{str(k):float(np.mean([h for _,h in v])) for k,v in g['residual_by_bin'].items()},
          'mean_command_residual_m_by_quarter':{str(k):float(np.mean([x for x,_ in v])) for k,v in g['residual_by_bin'].items()}}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':out['status'],'trace_files':len(files),'plants':len(out['groups']),'groups':sum(len(v) for v in out['groups'].values())},ensure_ascii=False))
if __name__=='__main__':main()
