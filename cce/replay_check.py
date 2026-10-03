#!/usr/bin/env python
"""Control for the closed-loop evaluator: replay ground-truth demo poses.

Feeds the *recorded* absolute EE trajectory of a held-out demonstration through
exactly the path `eval_xvla.py` uses for a prediction --
`shared_ee_delta_action` -> `to_plant_action` -> `pd_ee_delta_pose` -- in the
evaluation environment at the same seed.

If this succeeds, the demo format, the rot6d/pose codec, the EE bridge and the
evaluation env are all sound, and any closed-loop failure is attributable to the
policy. If it fails, the carrier itself is at fault. Same role for the ManiSkill
domains that Gate 0 plays for LIBERO.
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


def replay_one(env, uid: str, path: Path, budget: int, use_recorded_rot: bool = True) -> dict:
    with h5py.File(path, "r") as f:
        pose = np.asarray(f["tcp_root_pose"][()], np.float64)
        closed = np.asarray(f["grip_closed"][()], np.float64).reshape(-1)
        seed = int(f.attrs["seed"])
    # Round-trip the recorded pose through the ee6d 20-d encoding the policy
    # emits, so the codec is exercised exactly as it is at inference time.
    if use_recorded_rot:
        enc = np.stack([XB.pose7_to_ee6d20(pose[i], closed[i]) for i in range(len(pose))], 0)
        dec = [XB.ee6d20_to_pose7_closed(a) for a in enc]
        pose = np.stack([d[0] for d in dec], 0)
        closed = np.array([d[1] for d in dec], np.float64)

    base = env.unwrapped
    env.reset(seed=seed)
    cur = XB.tcp_root_pose(base)
    success = False
    errs = []
    n = min(len(pose) - 1, budget)
    for i in range(n):
        tgt = pose[i + 1]
        g = XB.closed_to_shared_gripper(1.0 if closed[i + 1] > 0.5 else 0.0)
        a7 = XB.shared_ee_delta_action(cur, tgt, g)
        _, _, term, trunc, info = env.step(XB.to_plant_action(uid, a7))
        cur = XB.tcp_root_pose(base)
        errs.append(float(np.linalg.norm(cur[:3] - tgt[:3])))
        success = bool(np.asarray(XB.np_of(info["success"])).reshape(-1)[0])
        if success or bool(np.asarray(XB.np_of(term)).reshape(-1)[0]):
            break
    return {
        "file": path.name,
        "seed": seed,
        "success": int(success),
        "steps": len(errs),
        "demo_len": int(len(pose)),
        "mean_tcp_tracking_error_m": round(float(np.mean(errs)), 5) if errs else None,
        "max_tcp_tracking_error_m": round(float(np.max(errs)), 5) if errs else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser("ground-truth replay control")
    ap.add_argument("--uid", default="panda", choices=["panda", "xarm6_robotiq"])
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--demo-dir", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--sim-backend", default="physx_cpu")
    ap.add_argument("--no-codec-roundtrip", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    demo_dir = Path(args.demo_dir or (_bootstrap.DATA / "demos_heldout"))
    files = sorted(demo_dir.glob(f"{args.uid}_{args.task}_seed*.h5"))[: args.n]
    if not files:
        raise SystemExit(f"no demos for {args.uid} in {demo_dir}")

    env = make_eval_env(args.task, args.uid, args.img, args.budget, args.sim_backend)
    rows = []
    try:
        for p in files:
            r = replay_one(env, args.uid, p, args.budget, not args.no_codec_roundtrip)
            rows.append(r)
            print(json.dumps(r), flush=True)
    finally:
        env.close()

    out = {
        "control": "ground-truth demo replay through the shared EE-delta bridge",
        "uid": args.uid,
        "task": args.task,
        "demo_dir": str(demo_dir),
        "codec_roundtrip": not args.no_codec_roundtrip,
        "sim_backend": args.sim_backend,
        "n": len(rows),
        "success_rate": round(float(np.mean([r["success"] for r in rows])), 4),
        "mean_tcp_tracking_error_m": round(float(np.mean([r["mean_tcp_tracking_error_m"] for r in rows])), 5),
        "max_tcp_tracking_error_m": round(float(np.max([r["max_tcp_tracking_error_m"] for r in rows])), 5),
        "episodes": rows,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "episodes"}, indent=2), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
