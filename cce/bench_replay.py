#!/usr/bin/env python
"""Experiment (b): expert-demonstration replay on the plant family (Gate 1).

The 50 motion-planned PickCube demonstrations act as a "perfect policy": their
recorded normalized EE-delta actions are first re-expanded into an *intent*
trajectory (absolute TCP target per control step) by replaying them on the
nominal plant, then re-executed on every plant with

  naive       u = c   (closed-loop residual against the intent, the standard arm)
  naive_open  verbatim replay of the recorded action series (open loop)
  ours_p      executor inversion with the probe-identified theta
  oracle      executor inversion with ground-truth theta
  slow15/20   naive tracking of a x1.5 / x2 time-scaled intent

Budget 200 steps per episode (the registered PickCube horizon).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from cce.common import (
    DT,
    apply_norm_delta,
    first_bool,
    interp_pose_seq,
    json_default,
    make_env,
    tcp_pose,
)
from cce.executor import Theta, build_arm, shape_reference
from cce.plants import NOMINAL_PID, PlantWrapper, plant_ids, select

DATA = Path("/opt/cce/data/cce")
DEMO_DIR = Path("/opt/CalibrationToPrompt/pilot2_runs/dev_route_r_demos")
BUDGET = 200
METHODS = ("naive", "naive_open", "ours_p", "oracle", "slow15", "slow20")


def load_demos(uid: str) -> dict[int, np.ndarray]:
    z = np.load(DEMO_DIR / f"{uid}_pickcube_d50.npz")
    seeds = sorted(int(k[4:-8]) for k in z.files if k.endswith("_actions"))
    return {s: np.asarray(z[f"seed{s}_actions"], dtype=np.float64) for s in seeds}


def build_intent(uid: str, env, demos: dict[int, np.ndarray], cache: Path) -> dict[int, dict]:
    """Replay each demo on the nominal plant; record the absolute intent it commands."""
    from calibration_to_prompt.dev_anchor import to_plant_action

    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        seeds = sorted(int(k[4:-2]) for k in z.files if k.endswith("_c"))
        return {s: {"c": z[f"seed{s}_c"], "grip": z[f"seed{s}_g"],
                    "success": bool(z[f"seed{s}_s"][0] > 0.5)} for s in seeds}
    out, store = {}, {}
    for seed, acts in demos.items():
        env.reset(seed=int(seed))
        base = env.unwrapped
        y = tcp_pose(base)
        n = acts.shape[0]
        c = np.zeros((n, 7))
        ok = False
        for t in range(n):
            c[t] = apply_norm_delta(y, acts[t, :6])
            _, _, term, _, info = env.step(to_plant_action(uid, acts[t]))
            y = tcp_pose(base)
            ok = first_bool(info["success"])
            if ok or first_bool(term):
                c = c[: t + 1]
                break
        out[seed] = {"c": c, "grip": acts[: c.shape[0], 6].copy(), "success": ok}
        store[f"seed{seed}_c"] = c
        store[f"seed{seed}_g"] = acts[: c.shape[0], 6].copy()
        store[f"seed{seed}_s"] = np.array([1.0 if ok else 0.0])
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **store)
    os.replace(tmp, cache)
    return out


def time_scale(c: np.ndarray, grip: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(math.ceil(c.shape[0] * s))
    cs = np.stack([interp_pose_seq(c, k / s) for k in range(n)])
    gs = np.array([grip[min(int(k / s), grip.shape[0] - 1)] for k in range(n)])
    return cs, gs


def run_episode(uid, env, spec, method, seed, intent, demo_actions, hat, theta_n, clip_burst=False) -> dict:
    from calibration_to_prompt.dev_anchor import to_plant_action

    c, g = intent["c"], intent["grip"]
    if method == "slow15":
        c, g = time_scale(c, g, 1.5)
    elif method == "slow20":
        c, g = time_scale(c, g, 2.0)

    ex, shaped = ((None, False) if method == "naive_open"
                  else build_arm(method, spec, hat, theta_n, clip_from_burst=clip_burst))

    env.reset(seed=int(seed))
    base = env.unwrapped
    plant = PlantWrapper(spec)
    y = tcp_pose(base)
    if shaped:
        c = shape_reference(c, theta_n, y)
    n_demo = demo_actions.shape[0]
    ok, steps = False, BUDGET
    for t in range(BUDGET):
        if method == "naive_open":
            if t < n_demo:
                u = demo_actions[t].astype(np.float32)
            else:
                u = np.zeros(7, dtype=np.float32)
                u[6] = demo_actions[-1, 6]
        else:
            u = ex.step(y, c, g, t)
        _, _, term, _, info = env.step(to_plant_action(uid, plant.apply(u)))
        y = tcp_pose(base)
        ok = first_bool(info["success"])
        if ok:
            steps = t + 1
            break
        if first_bool(term):
            steps = t + 1
            break
    return {"success": bool(ok), "steps": int(steps), "time_to_success_s": float(steps * DT) if ok else float("nan"),
            "ref_len": int(c.shape[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default="panda")
    ap.add_argument("--plants", default="")
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--theta-dir", default=str(DATA / "theta"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-from-burst", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--prep-only", action="store_true")
    args = ap.parse_args()

    hats = json.loads((Path(args.theta_dir) / f"theta_hat_{args.uid}.json").read_text())
    theta_n = Theta.from_hat(hats[NOMINAL_PID])
    pids = [p for p in args.plants.split(",") if p] or plant_ids()
    pids = pids[args.shard:: args.nshards]
    methods = [m for m in args.methods.split(",") if m]
    demos = load_demos(args.uid)
    seeds = sorted(demos)[: args.n_episodes]
    env = make_env(args.uid, max_episode_steps=BUDGET)
    intents = build_intent(args.uid, env, {s: demos[s] for s in seeds}, DATA / f"intent_{args.uid}.npz")
    print(json.dumps({"uid": args.uid, "n_intent": len(intents),
                      "nominal_rebuild_sr": float(np.mean([intents[s]["success"] for s in seeds])),
                      "ref_len_mean": float(np.mean([intents[s]["c"].shape[0] for s in seeds]))},
                     default=json_default), flush=True)
    if args.prep_only:
        env.close()
        return

    rows = []
    for spec in select(pids):
        for m in methods:
            for s in seeds:
                res = run_episode(args.uid, env, spec, m, s, intents[s], demos[s], hats.get(spec.pid), theta_n,
                                  clip_burst=args.clip_from_burst)
                rows.append({"uid": args.uid, "pid": spec.pid, "family": spec.family,
                             "seed": int(s), "method": m, **res})
        done = [x for x in rows if x["pid"] == spec.pid]
        print(json.dumps({"pid": spec.pid, "sr": {
            m: round(float(np.mean([x["success"] for x in done if x["method"] == m])), 3) for m in methods}},
            default=json_default), flush=True)
    env.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, default=json_default) + "\n")


if __name__ == "__main__":
    main()
