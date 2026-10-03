#!/usr/bin/env python
"""How much end-effector prediction error does PickCube tolerate?

Replays the ground-truth demonstration poses through the same
`shared_ee_delta_action` -> `to_plant_action` -> `pd_ee_delta_pose` path the
evaluator uses (i.e. `replay_check.py`), but perturbs the commanded absolute
pose by a controlled amount before it is issued. Sweeping the perturbation
magnitude turns "the policy's chunk has ~1.5 cm of xyz error" into a statement
about success rate, which is what decides whether the failure is an accuracy
problem or a protocol problem.

Noise models
  white  : i.i.d. N(0, s) on xyz at every control tick
  chunk  : one N(0, s) offset per 30-tick block, held constant inside the block
           (this is the shape of a chunk policy's error: a bias that persists
           for a whole chunk, not tick-to-tick jitter)

A gripper-timing shift can be swept independently with `--grip-shift`.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import xvla_bridge as XB
from eval_xvla import make_eval_env


def load_demo(path: Path):
    with h5py.File(path, "r") as f:
        pose = np.asarray(f["tcp_root_pose"][()], np.float64)
        closed = np.asarray(f["grip_closed"][()], np.float64).reshape(-1)
        seed = int(f.attrs["seed"])
    enc = np.stack([XB.pose7_to_ee6d20(pose[i], closed[i]) for i in range(len(pose))], 0)
    dec = [XB.ee6d20_to_pose7_closed(a) for a in enc]
    pose = np.stack([d[0] for d in dec], 0)
    closed = np.array([d[1] for d in dec], np.float64)
    return pose, closed, seed


def offsets(n: int, std: float, mode: str, chunk: int, rng) -> np.ndarray:
    if std <= 0:
        return np.zeros((n, 3))
    if mode == "white":
        return rng.normal(0.0, std, size=(n, 3))
    blocks = int(np.ceil(n / chunk))
    per_block = rng.normal(0.0, std, size=(blocks, 3))
    return np.repeat(per_block, chunk, axis=0)[:n]


def replay_one(env, uid: str, pose, closed, seed: int, budget: int,
               std: float, mode: str, chunk: int, grip_shift: int, rng) -> dict:
    base = env.unwrapped
    env.reset(seed=seed)
    cur = XB.tcp_root_pose(base)
    n = min(len(pose) - 1, budget)
    off = offsets(n, std, mode, chunk, rng)
    success = False
    errs = []
    for i in range(n):
        tgt = pose[i + 1].copy()
        tgt[:3] += off[i]
        gi = int(np.clip(i + 1 + grip_shift, 0, len(closed) - 1))
        g = XB.closed_to_shared_gripper(1.0 if closed[gi] > 0.5 else 0.0)
        a7 = XB.shared_ee_delta_action(cur, tgt, g)
        _, _, term, _, info = env.step(XB.to_plant_action(uid, a7))
        cur = XB.tcp_root_pose(base)
        errs.append(float(np.linalg.norm(cur[:3] - tgt[:3])))
        success = bool(np.asarray(XB.np_of(info["success"])).reshape(-1)[0])
        if success or bool(np.asarray(XB.np_of(term)).reshape(-1)[0]):
            break
    return {"seed": seed, "success": int(success), "steps": len(errs),
            "mean_tcp_tracking_error_m": round(float(np.mean(errs)), 5) if errs else None}


def main() -> None:
    ap = argparse.ArgumentParser("perturbed ground-truth replay: accuracy budget of PickCube")
    ap.add_argument("--uid", default="panda", choices=["panda", "xarm6_robotiq"])
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--demo-dir", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--sim-backend", default="physx_cpu")
    ap.add_argument("--stds", default="0,0.005,0.010,0.015,0.020,0.030",
                    help="comma-separated xyz perturbation std in metres")
    ap.add_argument("--modes", default="chunk,white")
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--grip-shifts", default="0", help="comma-separated gripper index shift in ticks")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the perturbation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    demo_dir = Path(args.demo_dir or (_bootstrap.DATA / "demos_heldout"))
    files = sorted(demo_dir.glob(f"{args.uid}_{args.task}_seed*.h5"))[: args.n]
    if not files:
        raise SystemExit(f"no demos for {args.uid} in {demo_dir}")
    demos = [load_demo(p) for p in files]

    stds = [float(x) for x in args.stds.split(",") if x != ""]
    modes = [m for m in args.modes.split(",") if m]
    shifts = [int(x) for x in args.grip_shifts.split(",") if x != ""]

    env = make_eval_env(args.task, args.uid, args.img, args.budget, args.sim_backend)
    cells = []
    try:
        for mode in modes:
            for std in stds:
                for sh in shifts:
                    if std == 0 and mode != modes[0] and sh == 0:
                        continue  # the std=0 cell is identical across noise models
                    rng = np.random.default_rng(args.seed)
                    rows = [replay_one(env, args.uid, p, c, s, args.budget,
                                       std, mode, args.chunk, sh, rng)
                            for (p, c, s) in demos]
                    cell = {"mode": mode, "pos_noise_std_m": std, "grip_shift_ticks": sh,
                            "n": len(rows),
                            "success_rate": round(float(np.mean([r["success"] for r in rows])), 4),
                            "mean_steps": round(float(np.mean([r["steps"] for r in rows])), 1)}
                    cells.append(cell)
                    print(json.dumps(cell), flush=True)
    finally:
        env.close()

    out = {"control": "ground-truth replay with a controlled xyz perturbation",
           "uid": args.uid, "task": args.task, "demo_dir": str(demo_dir),
           "n_demos": len(demos), "chunk": args.chunk, "rng_seed": args.seed,
           "sim_backend": args.sim_backend, "cells": cells}
    print(json.dumps(out, indent=2), flush=True)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[diag] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
