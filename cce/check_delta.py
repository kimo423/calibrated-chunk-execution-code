#!/usr/bin/env python
"""Are `cce.common.norm_delta` (executor path) and `xvla_bridge.pose_to_normalized_ee6`
(eval_xvla path) the same map?

It matters because the T4 `naive` arm runs the policy chunk through
`InversionExecutor` with identity theta, while the T1 own-nominal evaluator runs
it through `XB.shared_ee_delta_action`. If the two normalizations disagreed, the
T4 nominal cell would not be comparable to the T1 number and the whole
plant-family comparison would rest on two different executors.
"""

import _bootstrap  # noqa: F401

import json
from pathlib import Path

import numpy as np

import xvla_bridge as XB
from cce.common import apply_norm_delta, norm_delta

rng = np.random.default_rng(0)
worst = {"pos": 0.0, "rot": 0.0}
worst_round = 0.0
for _ in range(500):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    cur = np.concatenate([rng.normal(0.5, 0.1, 3), q])
    d = rng.normal(0, 0.01, 3)
    dq = rng.normal(size=4)
    dq /= np.linalg.norm(dq)
    dq = 0.999 * np.array([1.0, 0.0, 0.0, 0.0]) + 0.001 * dq
    dq /= np.linalg.norm(dq)
    tgt = np.concatenate([
        cur[:3] + d,
        XB.matrix_to_quat_wxyz(XB.quat_wxyz_to_matrix(dq) @ XB.quat_wxyz_to_matrix(cur[3:])),
    ])
    a = np.asarray(norm_delta(cur, tgt), np.float64).reshape(6)
    b = np.asarray(XB.pose_to_normalized_ee6(cur, tgt, clip=True), np.float64).reshape(6)
    worst["pos"] = max(worst["pos"], float(np.abs(a[:3] - b[:3]).max()))
    worst["rot"] = max(worst["rot"], float(np.abs(a[3:] - b[3:]).max()))
    back = apply_norm_delta(cur, b)
    worst_round = max(worst_round, float(np.abs(back[:3] - tgt[:3]).max()))

# One normalized position unit is 0.1 m, so 1e-6 normalized is 0.1 um -- far
# below the millimetre scale anything in this experiment resolves.
out = {
    "n_pairs": 500,
    "max_abs_diff_normalized_pos": float(f"{worst['pos']:.3e}"),
    "max_abs_diff_normalized_rot": float(f"{worst['rot']:.3e}"),
    "max_abs_diff_metres": float(f"{worst['pos'] * XB.POS_LIMIT_M:.3e}"),
    "apply_norm_delta_roundtrip_pos_m": float(f"{worst_round:.3e}"),
    "verdict": "identical up to float64 round-off" if max(worst.values()) < 1e-6 else "DIFFERENT",
}
print(json.dumps(out, indent=2))
Path("cce/results/check_delta_conventions.json").write_text(json.dumps(out, indent=2) + "\n")
