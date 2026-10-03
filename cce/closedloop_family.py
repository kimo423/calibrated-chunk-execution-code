#!/usr/bin/env python
"""T4: the real X-VLA policy in the loop over the CCE plant family.

This is the policy-in-the-loop counterpart of `cce/bench_replay.py`. There the
"perfect policy" was a recorded expert demonstration; here the chunk comes from
the trained domain row, so the whole CCE claim (interface mismatch is a tracking
gap that a seconds-long calibration can invert) is exercised end to end.

Per control step, at 20 Hz:

    policy chunk (30 absolute TCP targets, root frame)
      -> [optional time scaling]                       Slow-down arms
      -> [optional nominal-shaping]                    calibrated arms
      -> InversionExecutor.step(measured pose, ref, t) -> normalized 7-vector
      -> PlantWrapper.apply                            D1-D4 interface mismatch
      -> XB.to_plant_action -> env.step (pd_ee_delta_pose)

The chunk is drained (one grid step per control step, the pre-registered
protocol) and re-queried when exhausted; the executor's internal pipeline state
is *not* reset between chunks, only the reference index restarts.

Arms
  naive   identity theta, set point = the policy chunk itself
  slow15  naive tracking of a x1.5 time-scaled chunk
  slow20  naive tracking of a x2 time-scaled chunk
  oracle  ground-truth theta cascaded with the nominal response, shaped reference
  ours_p  probe-identified theta (`data/cce/theta/theta_hat_<uid>.json`), shaped
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import xvla_bridge as XB
from cce.common import DT, interp_pose_seq, json_default
from cce.executor import Theta, build_arm, shape_reference
from cce.plants import NOMINAL_PID, PlantWrapper, plant_ids, select
from eval_xvla import ChunkPolicy, load_cce_state, load_lora, make_eval_env, obs_rgb

DATA = Path(_bootstrap.DATA)
BUDGET = 200
CHUNK = 30

# Reduced first-pass family: spans g in {0.70, 1.00} and d in {0, 150, 250} ms,
# and carries one mixed plant. Applied to both arms, so 5 specs = 10 plant cells.
REDUCED = ("P00_nominal", "D1_d150", "D1_d250", "D2_g70", "MX03")
METHODS = ("naive", "slow20", "oracle", "ours_p")


def time_scale(c: np.ndarray, grip: np.ndarray, s: float):
    n = int(math.ceil(c.shape[0] * s))
    cs = np.stack([interp_pose_seq(c, k / s) for k in range(n)])
    gs = np.array([grip[min(int(k / s), grip.shape[0] - 1)] for k in range(n)])
    return cs, gs


def chunk_from_policy(policy: ChunkPolicy, rgb, pose7, closed):
    """One policy query -> (absolute pose targets [30,7], shared gripper [30])."""
    proprio = XB.pose7_to_ee6d20(pose7, closed)
    rows = policy._query(rgb, proprio)
    c = np.zeros((rows.shape[0], 7))
    g = np.zeros(rows.shape[0])
    for i, r in enumerate(rows):
        p, cl = XB.ee6d20_to_pose7_closed(r)
        c[i] = p
        g[i] = XB.closed_to_shared_gripper(1.0 if cl > 0.5 else 0.0)
    return c, g


def run_episode(uid, env, policy, spec, method, seed, hat, theta_n) -> dict:
    base = env.unwrapped
    obs, _ = env.reset(seed=int(seed))
    plant = PlantWrapper(spec)
    ex, shaped = build_arm(method, spec, hat, theta_n)
    scale = {"slow15": 1.5, "slow20": 2.0}.get(method, 1.0)

    y = XB.tcp_root_pose(base)
    closed = 0.0
    c_ref = None      # the arm's own set point (shaped for the calibrated arms)
    c_intent = None   # the policy chunk itself, time-scaled but never shaped
    g_ref = None
    t_in = 0
    ok = False
    steps = BUDGET
    n_queries = 0
    track_ref = []
    track_intent = []
    tcp_cube = []
    cube_z = []
    grasped_steps = 0

    def _p(name):
        o = getattr(base, name, None)
        return None if o is None else XB.np_of(o.pose.p).reshape(-1)[:3]

    cube0 = _p("cube")
    z0 = float(cube0[2]) if cube0 is not None else None

    for t in range(BUDGET):
        if c_ref is None or t_in >= c_ref.shape[0]:
            c_raw, g_raw = chunk_from_policy(policy, obs_rgb(obs, "base_camera"), y, closed)
            n_queries += 1
            if scale != 1.0:
                c_raw, g_raw = time_scale(c_raw, g_raw, scale)
            c_intent = c_raw
            c_ref = shape_reference(c_raw, theta_n, y) if shaped else c_raw
            g_ref = g_raw
            t_in = 0
        u = ex.step(y, c_ref, g_ref, t_in)
        applied = plant.apply(u)
        obs, _, term, _, info = env.step(XB.to_plant_action(uid, applied))
        y = XB.tcp_root_pose(base)
        closed = 1.0 if float(g_ref[min(t_in, len(g_ref) - 1)]) < 0.0 else 0.0
        j = min(t_in, c_ref.shape[0] - 1)
        track_ref.append(float(np.linalg.norm(y[:3] - c_ref[j][:3])))
        track_intent.append(float(np.linalg.norm(y[:3] - c_intent[j][:3])))
        cube = _p("cube")
        if cube is not None:
            tcp_cube.append(float(np.linalg.norm(XB.np_of(base.agent.tcp.pose.p).reshape(-1)[:3] - cube)))
            cube_z.append(float(cube[2]))
        if "is_grasped" in info:
            grasped_steps += int(bool(np.asarray(XB.np_of(info["is_grasped"])).reshape(-1)[0]))
        t_in += 1
        ok = bool(np.asarray(XB.np_of(info["success"])).reshape(-1)[0])
        if ok or bool(np.asarray(XB.np_of(term)).reshape(-1)[0]):
            steps = t + 1
            break

    return {
        "success": int(ok),
        "steps": int(steps),
        "time_to_success_s": float(steps * DT) if ok else None,
        # vs_intent is the cross-arm comparable one: every arm is scored against the
        # policy chunk it was given (time-scaled for the slow-down arms, since that is
        # genuinely where the arm intends to be at time t). vs_ref scores each arm
        # against its own set point, which for the calibrated arms is the shaped
        # reference -- useful for checking the inversion, useless for comparing arms.
        "tracking_rmse_m": round(float(np.sqrt(np.mean(np.square(track_intent)))), 5) if track_intent else None,
        "tracking_rmse_vs_ref_m": round(float(np.sqrt(np.mean(np.square(track_ref)))), 5) if track_ref else None,
        "min_tcp_cube_dist_m": round(float(np.min(tcp_cube)), 4) if tcp_cube else None,
        "max_cube_z_m": round(float(np.max(cube_z)), 4) if cube_z else None,
        "lifted": int(z0 is not None and cube_z and (max(cube_z) - z0) > 0.01),
        "grasped_steps": grasped_steps,
        "policy_queries": n_queries,
    }


def main() -> None:
    ap = argparse.ArgumentParser("CCE plant-family closed loop with the real policy")
    ap.add_argument("--uid", default="panda", choices=["panda", "xarm6_robotiq"])
    ap.add_argument("--ckpt", default=None, help="phase-1 domain-state checkpoint dir")
    ap.add_argument("--lora", default=None, help="peft adapter dir (LoRA route)")
    ap.add_argument("--model-path", default=str(XB.XVLA_PT))
    ap.add_argument("--plants", default=",".join(REDUCED))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=20280101)
    ap.add_argument("--theta-dir", default=str(DATA / "theta"))
    ap.add_argument("--denoise-steps", type=int, default=10)
    ap.add_argument("--sim-backend", default="physx_cpu")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hats = json.loads((Path(args.theta_dir) / f"theta_hat_{args.uid}.json").read_text())
    theta_n = Theta.from_hat(hats[NOMINAL_PID])
    pids = [p for p in args.plants.split(",") if p] or plant_ids()
    methods = [m for m in args.methods.split(",") if m]
    seeds = [args.seed_base + i for i in range(args.n)]

    model, processor, info = XB.load_xvla(model_path=args.model_path, device=args.device)
    model, lora_info = load_lora(model, args.lora, args.device)
    ck = load_cce_state(model, args.ckpt)
    ck.update(lora_info)
    domain_id = XB.DOMAIN_IDS[args.uid]
    policy = ChunkPolicy(model, processor, domain_id, XB.TASK_INSTRUCTIONS["PickCube-v1"],
                         args.device, args.denoise_steps)

    env = make_eval_env("PickCube-v1", args.uid, 256, BUDGET, args.sim_backend)
    rows = []
    t0 = time.time()
    try:
        for spec in select(pids):
            for m in methods:
                for s in seeds:
                    r = run_episode(args.uid, env, policy, spec, m, s, hats.get(spec.pid), theta_n)
                    rows.append({"uid": args.uid, "pid": spec.pid, "family": spec.family,
                                 "method": m, "seed": int(s), **r})
            done = [x for x in rows if x["pid"] == spec.pid]
            print(json.dumps({"pid": spec.pid, "sr": {
                m: round(float(np.mean([x["success"] for x in done if x["method"] == m])), 3)
                for m in methods}}, default=json_default), flush=True)
    finally:
        env.close()

    # per-cell aggregate + the G1 statistic (naive drop relative to nominal)
    def cell(pid, m):
        xs = [x for x in rows if x["pid"] == pid and x["method"] == m]
        if not xs:
            return None
        return {
            "n": len(xs),
            "success_rate": round(float(np.mean([x["success"] for x in xs])), 4),
            "mean_steps": round(float(np.mean([x["steps"] for x in xs])), 1),
            "tracking_rmse_m": round(float(np.mean([x["tracking_rmse_m"] for x in xs
                                                    if x["tracking_rmse_m"] is not None])), 5),
            "tracking_rmse_vs_ref_m": round(float(np.mean([x["tracking_rmse_vs_ref_m"] for x in xs
                                                           if x["tracking_rmse_vs_ref_m"] is not None])), 5),
            "n_lifted": int(sum(x["lifted"] for x in xs)),
            "mean_min_tcp_cube_dist_m": round(float(np.mean([x["min_tcp_cube_dist_m"] for x in xs
                                                             if x["min_tcp_cube_dist_m"] is not None])), 4),
            "mean_time_to_success_s": (round(float(np.mean([x["time_to_success_s"] for x in xs
                                                            if x["time_to_success_s"] is not None])), 3)
                                       if any(x["time_to_success_s"] is not None for x in xs) else None),
        }

    cells = {pid: {m: cell(pid, m) for m in methods} for pid in pids}
    nominal_naive = (cells.get(NOMINAL_PID, {}).get("naive") or {}).get("success_rate")
    g1 = {}
    for pid in pids:
        if pid == NOMINAL_PID or nominal_naive is None:
            continue
        c = cells[pid].get("naive")
        if c is not None:
            g1[pid] = {"naive_sr": c["success_rate"],
                       "drop_pp": round(100.0 * (nominal_naive - c["success_rate"]), 1),
                       "meets_G1_25pp": bool(100.0 * (nominal_naive - c["success_rate"]) >= 25.0)}

    # K1: on a lag/delay plant the calibrated arms must cut tracking RMSE by >= 30%
    # (measured against the shared intent, so the arms are comparable).
    k1 = {}
    for pid in pids:
        base_c = cells[pid].get("naive")
        if base_c is None or not base_c["tracking_rmse_m"]:
            continue
        row = {"naive_tracking_rmse_m": base_c["tracking_rmse_m"]}
        for m in methods:
            if m == "naive":
                continue
            c = cells[pid].get(m)
            if c and c["tracking_rmse_m"] is not None:
                row[f"{m}_tracking_rmse_m"] = c["tracking_rmse_m"]
                row[f"{m}_reduction_pct"] = round(
                    100.0 * (1.0 - c["tracking_rmse_m"] / base_c["tracking_rmse_m"]), 1)
        row["meets_K1_30pct"] = bool(row.get("ours_p_reduction_pct", -1) >= 30.0)
        k1[pid] = row

    out = {
        "experiment": "T4 plant-family closed loop, policy in the loop",
        "uid": args.uid,
        "checkpoint": ck,
        "protocol": {
            "budget_steps": BUDGET, "chunk": CHUNK, "chunk_seconds": CHUNK * DT,
            "control_hz": round(1.0 / DT, 1), "denoise_steps": args.denoise_steps,
            "chunk_execution": "drained, re-queried when exhausted",
            "seed_base": args.seed_base, "n_per_cell": args.n,
            "sim_backend": args.sim_backend,
            "theta_source": str(Path(args.theta_dir) / f"theta_hat_{args.uid}.json"),
            "plant_family": "reduced first pass" if args.plants == ",".join(REDUCED) else args.plants,
            "deviations": [
                "reduced plant family (first pass); the pre-registered family has 23 plants",
                "1 task only (PickCube-v1); the pre-registered macro SR is over 3 tasks",
            ],
        },
        "cells": cells,
        "K1_tracking_rmse_vs_intent": k1,
        "G1_naive_drop_vs_nominal": g1,
        "nominal_naive_sr": nominal_naive,
        "wall_s": round(time.time() - t0, 1),
        "episodes": rows,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "episodes"}, indent=2, default=json_default),
          flush=True)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=json_default) + "\n")
    print(f"[T4] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
