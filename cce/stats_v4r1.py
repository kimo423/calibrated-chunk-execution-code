"""Plant-level paired superiority and equivalence; never resample episodes."""
import math
import numpy as np
from scipy.stats import t as student_t


def paired(diffs, margin=0., alternative='greater'):
    if alternative not in ('greater', 'less', 'two-sided'):
        raise ValueError(alternative)
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError('one finite difference per plant is required')
    n = len(d)
    if n < 2:
        return {'n': n, 'mean': float(d.mean()) if n else None, 'p': None, 'ci95': None}
    mean = float(d.mean())
    se = float(d.std(ddof=1)/math.sqrt(n))
    if se == 0:
        p = float(mean == margin) if alternative == 'two-sided' else float(not (mean > margin if alternative=='greater' else mean < margin))
    else:
        stat = (mean-margin)/se
        p = float(2*student_t.sf(abs(stat),n-1) if alternative=='two-sided' else
                  student_t.sf(stat,n-1) if alternative=='greater' else student_t.cdf(stat,n-1))
    rng = np.random.default_rng(20260908)
    boot = d[rng.integers(0,n,(10000,n))].mean(axis=1)
    return {'n':n, 'mean':mean, 'p':p, 'ci95':np.quantile(boot,[.025,.975]).tolist()}


def tost(diffs, delta):
    if delta <= 0:
        raise ValueError('positive equivalence margin required')
    lower = paired(diffs,-delta,'greater')
    upper = paired(diffs,delta,'less')
    p = None if lower['p'] is None else max(lower['p'],upper['p'])
    return {'n':lower['n'], 'mean':lower['mean'], 'margin':delta, 'lower_p':lower['p'],
            'upper_p':upper['p'], 'p':p, 'equivalent':p is not None and p < .05}


def holm(pvals):
    # Unestimable registered hypotheses retain p=1 in the registered family.
    items = sorted((k,1. if v is None else float(v)) for k,v in pvals.items())
    items.sort(key=lambda kv:kv[1])
    result, previous = {}, 0.
    for i,(k,p) in enumerate(items):
        previous = max(previous,min(1.,(len(items)-i)*p))
        result[k] = previous
    return result


def selftest():
    assert tost([0.]*16,.03)['equivalent']
    assert not tost([.03]*16,.03)['equivalent']
    assert not tost([-.06]*16,.05)['equivalent']
    assert not tost([.06]*16,.05)['equivalent']
    assert not tost([0.],.03)['equivalent']
    assert not tost([],.03)['equivalent']
    p = holm({'P1':.01,'P2a':.04,'P2b':None})
    assert p == {'P1':.03,'P2a':.08,'P2b':1.}
    assert paired([-.1]*16)['p'] == 1.
    assert paired([.1]*16)['p'] == 0.
    assert paired([-.1]*16, alternative='two-sided')['p'] == 0.
    assert paired([0.]*16, alternative='two-sided')['p'] == 1.
    return {'ok':True,'checks':11}


if __name__ == '__main__':
    print(selftest())
