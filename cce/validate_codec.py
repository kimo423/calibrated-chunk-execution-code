#!/usr/bin/env python
"""Unit checks: numpy codec == external torch codec; executor inverts its own model."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json

import numpy as np

from cce.common import apply_norm_delta, norm_delta, pose_error, qnorm
from cce.executor import InversionExecutor, Theta, _clip_to_limits
from cce.plants import PLANTS, PlantWrapper, get_plant


def rand_pose(rng):
    p = rng.uniform(-0.5, 0.5, 3)
    q = qnorm(rng.normal(size=4))
    return np.concatenate([p, q])


def check_codec(n=2000) -> dict:
    from calibration_to_prompt.pilot2_task_plan import (
        reconstruct_target_pose_from_normalized_ee6d,
        target_pose_to_normalized_ee6d,
    )

    rng = np.random.default_rng(0)
    e_fwd, e_bwd, e_rt = 0.0, 0.0, 0.0
    for _ in range(n):
        cur = rand_pose(rng)
        a = rng.uniform(-1, 1, 6)
        a[3:] *= 0.5
        tgt_ref = reconstruct_target_pose_from_normalized_ee6d(cur, a)
        tgt_mine = apply_norm_delta(cur, a)
        e_bwd = max(e_bwd, float(np.abs(tgt_ref - tgt_mine).max()))
        a_ref = target_pose_to_normalized_ee6d(cur, tgt_ref, clip=False)
        a_mine = norm_delta(cur, tgt_ref)
        e_fwd = max(e_fwd, float(np.abs(np.asarray(a_ref, dtype=np.float64) - a_mine).max()))
        back = apply_norm_delta(cur, norm_delta(cur, tgt_ref))
        dp, dr = pose_error(back, tgt_ref)
        e_rt = max(e_rt, dp + dr)
    return {"max_abs_err_reconstruct": e_bwd, "max_abs_err_norm_delta": e_fwd, "max_roundtrip_err": e_rt}


def check_plant_identity() -> dict:
    rng = np.random.default_rng(1)
    p = PlantWrapper(get_plant("P00_nominal"))
    err = 0.0
    for _ in range(200):
        a = rng.uniform(-1, 1, 7)
        err = max(err, float(np.abs(p.apply(a) - a).max()))
    return {"nominal_plant_max_abs_dev": err}


def check_executor_closes_model_loop() -> dict:
    """Executor + oracle theta driving an exact model plant: tracking error must vanish."""
    rng = np.random.default_rng(2)
    ref = np.zeros((120, 7))
    base_q = qnorm(np.array([1.0, 0.0, 0.02, 0.0]))
    for t in range(120):
        ref[t, :3] = np.array([0.02 * np.sin(0.08 * t), 0.02 * np.cos(0.05 * t), 0.15 + 0.01 * np.sin(0.03 * t)])
        ref[t, 3:] = base_q
    grip = np.ones(120)
    rows = {}
    for spec in PLANTS:
        plant = PlantWrapper(spec)
        theta = Theta.from_spec(spec)
        ex = InversionExecutor(theta)
        y = ref[0].copy()
        errs = []
        for t in range(120):
            u = ex.step(y, ref, grip, t)
            applied = plant.apply(u)
            y = apply_norm_delta(y, applied[:6])
            errs.append(pose_error(y, ref[t])[0])
        rows[spec.pid] = {"rmse_tail_m": float(np.sqrt(np.mean(np.square(errs[40:])))),
                          "rmse_all_m": float(np.sqrt(np.mean(np.square(errs))))}
    naive_rows = {}
    for spec in PLANTS:
        plant = PlantWrapper(spec)
        ex = InversionExecutor(Theta.identity())
        y = ref[0].copy()
        errs = []
        for t in range(120):
            u = ex.step(y, ref, grip, t)
            y = apply_norm_delta(y, plant.apply(u)[:6])
            errs.append(pose_error(y, ref[t])[0])
        naive_rows[spec.pid] = float(np.sqrt(np.mean(np.square(errs[40:]))))
    return {"oracle_tail_rmse": rows, "naive_tail_rmse": naive_rows,
            "worst_oracle_tail_rmse_m": max(v["rmse_tail_m"] for v in rows.values())}


def main() -> None:
    out = {"codec": check_codec(), "plant": check_plant_identity(),
           "executor": check_executor_closes_model_loop()}
    ex = out["executor"]
    print(json.dumps({"codec": out["codec"], "plant": out["plant"],
                      "worst_oracle_tail_rmse_m": ex["worst_oracle_tail_rmse_m"],
                      "per_plant": {k: {"oracle": round(v["rmse_tail_m"], 6),
                                        "naive": round(ex["naive_tail_rmse"][k], 6)}
                                    for k, v in ex["oracle_tail_rmse"].items()}}, indent=2))


if __name__ == "__main__":
    main()
