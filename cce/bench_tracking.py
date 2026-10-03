#!/usr/bin/env python
"""Experiment (a): synthetic-reference tracking benchmark on the plant family.

No policy involved.  Each reference is 8 s (160 control steps):
  * 4 s smooth random cubic spline in a safe box around the reset TCP pose,
    with a slow attitude perturbation;
  * 4 s PickCube-shaped segment: approach -> descend -> close gripper -> lift,
    anchored on the actual cube position of that episode.

Arms compared per plant: naive (u = c), ours-P (identified theta), oracle-theta
(ground-truth theta), Slow-down x1.5 / x2 (naive on a time-scaled reference),
and optionally ours-QP.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
from pathlib import Path

import numpy as np

from cce.common import (
    DT,
    apply_norm_delta,
    cube_pose_root,
    euler_xyz_to_mat,
    gripper_opening,
    interp_pose_seq,
    json_default,
    make_env,
    mat_to_quat,
    norm_delta,
    pose_error,
    qnorm,
    quat_mul,
    tcp_pose,
)
from cce.executor import Theta, build_arm, shape_reference
from cce.plants import NOMINAL_PID, PlantWrapper, plant_ids, select

DATA = Path("/opt/cce/data/cce")
N_SPLINE, N_PICK = 80, 80
N_REF = N_SPLINE + N_PICK
TRAJ_SEED0 = 20268000
ENV_CAP = 420
METHODS = ("naive", "ours_p", "oracle", "slow15", "slow20")


def _ease(f: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * float(np.clip(f, 0.0, 1.0)))


def _quat_perturb(home_q: np.ndarray, e: np.ndarray) -> np.ndarray:
    return qnorm(quat_mul(mat_to_quat(euler_xyz_to_mat(e)), qnorm(home_q)))


def build_reference(home: np.ndarray, cube: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (c_seq[N_REF,7], grip_seq[N_REF], gripper close step)."""
    from scipy.interpolate import CubicSpline

    home = np.asarray(home, dtype=np.float64).reshape(7)
    c = np.tile(home, (N_REF, 1))
    grip = np.ones(N_REF, dtype=np.float64)

    # --- segment A: smooth random spline -----------------------------------
    n_knot = 5
    knots = np.zeros((n_knot, 3))
    knots[0] = home[:3]
    for k in range(1, n_knot):
        knots[k] = home[:3] + np.array([
            rng.uniform(-0.08, 0.08), rng.uniform(-0.10, 0.10), rng.uniform(-0.03, 0.10)])
    tk = np.linspace(0.0, 1.0, n_knot)
    spl = CubicSpline(tk, knots, axis=0, bc_type="clamped")
    amp = 0.08
    f1, f2 = rng.uniform(0.12, 0.30), rng.uniform(0.12, 0.30)
    ph1, ph2 = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
    for k in range(N_SPLINE):
        u = k / (N_SPLINE - 1)
        c[k, :3] = spl(u)
        t = k * DT
        e = np.array([amp * math.sin(2 * math.pi * f1 * t + ph1),
                      amp * math.sin(2 * math.pi * f2 * t + ph2), 0.0])
        c[k, 3:] = _quat_perturb(home[3:], e)

    # --- segment B: PickCube-shaped ----------------------------------------
    start = c[N_SPLINE - 1, :3].copy()
    pre = np.array([cube[0], cube[1], cube[2] + 0.10])
    grasp = np.array([cube[0], cube[1], cube[2] + 0.005])
    lift = np.array([cube[0], cube[1], cube[2] + 0.20])
    segs = [(20, start, pre), (20, pre, grasp), (15, grasp, grasp), (25, grasp, lift)]
    close_at = N_SPLINE + 40
    i = N_SPLINE
    for n, p0, p1 in segs:
        for k in range(n):
            c[i, :3] = p0 + (p1 - p0) * _ease((k + 1) / n)
            c[i, 3:] = home[3:]
            i += 1
    grip[close_at:] = -1.0
    return c, grip, close_at


def time_scale(c: np.ndarray, grip: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray, int]:
    n = int(math.ceil(c.shape[0] * s))
    cs = np.stack([interp_pose_seq(c, k / s) for k in range(n)])
    gs = np.array([grip[min(int(k / s), grip.shape[0] - 1)] for k in range(n)])
    close = int(np.argmax(gs < 0)) if np.any(gs < 0) else -1
    return cs, gs, close


def run_episode(uid, env, spec, method, c_seq, grip_seq, close_at, seed, hat, theta_n,
                qp_kw=None, clip_burst=False) -> dict:
    """Errors are reported both against the raw intent and against the trajectory the
    nominal robot would have achieved from the same intent (the recovery set point)."""
    from calibration_to_prompt.dev_anchor import to_plant_action

    ex, shaped = build_arm(method, spec, hat, theta_n, clip_from_burst=clip_burst,
                           **(qp_kw or {} if method == "ours_qp" else {}))
    env.reset(seed=int(seed))
    base = env.unwrapped
    plant = PlantWrapper(spec)
    n = c_seq.shape[0]
    y = tcp_pose(base)
    r_seq = shape_reference(c_seq, theta_n, y)
    ref = r_seq if shaped else c_seq
    ep_i, er_i, ep_n, er_n = (np.zeros(n) for _ in range(4))
    grip_hit = -1
    for t in range(n):
        u = ex.step(y, ref, grip_seq, t)
        env.step(to_plant_action(uid, plant.apply(u)))
        y = tcp_pose(base)
        ep_i[t], er_i[t] = pose_error(y, c_seq[t])
        ep_n[t], er_n[t] = pose_error(y, r_seq[t])
        if grip_hit < 0 and gripper_opening(uid, base) < 0.5:
            grip_hit = t
    lat = float((grip_hit - close_at) * DT) if (grip_hit >= 0 and close_at >= 0) else float("nan")
    return {
        "rmse_pos_m": float(np.sqrt((ep_n ** 2).mean())),
        "rmse_rot_rad": float(np.sqrt((er_n ** 2).mean())),
        "rmse_pos_vs_intent_m": float(np.sqrt((ep_i ** 2).mean())),
        "rmse_rot_vs_intent_rad": float(np.sqrt((er_i ** 2).mean())),
        "max_pos_err_m": float(ep_n.max()),
        "final_pos_err_m": float(ep_n[-1]),
        "grip_latency_s": lat,
        "steps": int(n),
        "duration_s": float(n * DT),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default="panda")
    ap.add_argument("--plants", default="")
    ap.add_argument("--n-traj", type=int, default=20)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--theta-dir", default=str(DATA / "theta"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--qp-horizon", type=int, default=30)
    ap.add_argument("--clip-from-burst", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    hats = json.loads((Path(args.theta_dir) / f"theta_hat_{args.uid}.json").read_text())
    theta_n = Theta.from_hat(hats[NOMINAL_PID])
    pids = [p for p in args.plants.split(",") if p] or plant_ids()
    pids = pids[args.shard:: args.nshards]
    methods = [m for m in args.methods.split(",") if m]
    env = make_env(args.uid, max_episode_steps=ENV_CAP)

    # references are fixed per (uid, traj) and shared by every plant / method
    refs = []
    for k in range(args.n_traj):
        seed = TRAJ_SEED0 + k
        env.reset(seed=seed)
        base = env.unwrapped
        home = tcp_pose(base)
        cube = cube_pose_root(base)
        rng = np.random.default_rng(seed)
        c, g, close = build_reference(home, cube, rng)
        variants = {"base": (c, g, close)}
        for s, name in ((1.5, "slow15"), (2.0, "slow20")):
            variants[name] = time_scale(c, g, s)
        refs.append({"seed": seed, "variants": variants})

    rows = []
    for spec in select(pids):
        for r in refs:
            for m in methods:
                key = m if m in ("slow15", "slow20") else "base"
                c, g, close = r["variants"][key]
                res = run_episode(args.uid, env, spec, m, c, g, close, r["seed"],
                                  hats.get(spec.pid), theta_n, qp_kw={"horizon": args.qp_horizon},
                                  clip_burst=args.clip_from_burst)
                rows.append({"uid": args.uid, "pid": spec.pid, "family": spec.family,
                             "traj_seed": r["seed"], "method": m, **res})
        done = [x for x in rows if x["pid"] == spec.pid]
        print(json.dumps({"pid": spec.pid, "n_rows": len(done), "rmse_by_method": {
            m: round(float(np.mean([x["rmse_pos_m"] for x in done if x["method"] == m])), 5)
            for m in methods}}, default=json_default), flush=True)
    env.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, default=json_default) + "\n")


if __name__ == "__main__":
    main()
