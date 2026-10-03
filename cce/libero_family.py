#!/usr/bin/env python
"""CCE plant-family closed loop with the LIBERO positive-control policy (prereg v2).

Per control step, at 20 Hz:

    policy chunk (30 absolute EE targets + hard gripper, from the X-VLA server)
      -> [optional time scaling]                        Slow-down arms
      -> [optional nominal shaping]                     calibrated arms
      -> LiberoExecutor.step(measured pose, ref, t)     normalized command increment
      -> LiberoPlant.command                            D1-D4 interface mismatch
      -> absolute target + gripper -> env.step          robosuite absolute OSC

The chunk is drained one grid step per control step and re-queried when exhausted
(the pre-registered protocol, and upstream's own); the executor's pipeline state is
*not* reset between chunks, only the reference index restarts.

Everything that defines the positive control is taken from upstream
``evaluation/libero/libero_client.py``: task loading and init states, the agentview
double flip, the wrist view, ``robot.controller.use_delta = False``, the proprio
bookkeeping (initialised once from the first observation, then overwritten with the
last row of each returned chunk), the 30-step drain, the +-1 gripper threshold, the
800-step horizon and ``domain_id = 3`` / 10 denoising steps.

Arms
  naive   identity theta, set point = the policy chunk itself
  slow15  naive tracking of a x1.5 time-scaled chunk
  slow20  naive tracking of a x2 time-scaled chunk
  oracle  ground-truth theta cascaded with the nominal response, shaped set point
  ours_p  probe-identified theta (cce/libero_probe.py), shaped set point
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("LIBERO_CONFIG_PATH", "/opt/cce/data/libero_config")
os.environ.setdefault("MUJOCO_GL", "egl")

XVLA_LIBERO = "/opt/CalibrationToPrompt/source/X-VLA/evaluation/libero"

from cce.common import DT, json_default, interp_pose_seq, norm_delta  # noqa: E402
from cce.libero_exec import (  # noqa: E402
    ACT_BOUND,
    METHODS,
    build_arm,
    shape_reference_libero,
    theta_from_hat,
)
from cce.libero_plant import (  # noqa: E402
    FAMILY_V1,
    GRIP_CLOSE,
    GRIP_OPEN,
    NOMINAL_PID,
    SCREEN,
    LiberoPlant,
    compose_delta,
    pose7_from_row,
    pose7_to_action,
)
from cce.libero_probe import DATA, UID, gripper_opening, measured_pose7  # noqa: E402

CHUNK = 30
HORIZON = 800
# Frozen executor command bound (declared before any plant-family arm was run).  It
# replaces ManiSkill's +-1 action bound, which LIBERO's absolute OSC does not have:
# 0.5 m / 0.5 rad per 50 ms step is ~6x the largest command any arm issues on the
# nominal plant and far beyond what the arm can physically execute, so it never
# shapes normal control -- it only stops a degenerate inversion from emitting an
# arbitrarily distant target.  `n_bound_hits` reports how often it binds.
BOUND_POS = 5.0
BOUND_ROT = 5.0
DENOISE_STEPS = 10
DOMAIN_ID = 3


# --------------------------------------------------------------------------- #
# policy client (upstream ClientModel, but returning the whole chunk)
# --------------------------------------------------------------------------- #
class ChunkClient:
    """Upstream ``ClientModel`` semantics, exposing the 30-row chunk instead of one row."""

    def __init__(self, host: str, port: int, timeout: float = 600.0) -> None:
        self.url = f"http://{host}:{port}/act"
        self.timeout = float(timeout)
        self.n_calls = 0
        self.reset()

    def reset(self) -> None:
        self.proprio = None

    def query(self, obs, goal: str) -> np.ndarray:
        import json_numpy
        import requests

        sys.path.insert(0, XVLA_LIBERO)
        import libero_client as LC

        main_view = LC._flip_agentview(obs["agentview_image"])
        wrist_view = obs["robot0_eye_in_hand_image"]
        clp = np.concatenate([obs["robo_pos"], obs["robo_ori"], np.array([0.0])], axis=-1)
        clp = np.concatenate([clp, np.zeros_like(clp)], axis=-1)
        if self.proprio is None:
            self.proprio = clp
        payload = {
            "proprio": json_numpy.dumps(self.proprio),
            "language_instruction": goal,
            "image0": json_numpy.dumps(main_view),
            "image1": json_numpy.dumps(wrist_view),
            "domain_id": DOMAIN_ID,
            "steps": DENOISE_STEPS,
        }
        resp = requests.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        action = np.array(resp.json()["action"])
        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"unexpected action shape {action.shape}")
        self.proprio[:9] = action[-1, :9].copy()
        self.n_calls += 1
        return action


def chunk_to_targets(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(30, 10) policy rows -> (absolute pose7 targets, hard gripper in {-1, +1})."""
    c = np.stack([pose7_from_row(r) for r in action])
    g = np.where(action[:, 9] > 0.5, GRIP_CLOSE, GRIP_OPEN).astype(np.float64)
    return c, g


def time_scale(c: np.ndarray, grip: np.ndarray, s: float):
    """x s time stretch of one chunk (linear position, slerp rotation)."""
    n = int(math.ceil(c.shape[0] * s))
    cs = np.stack([interp_pose_seq(c, k / s) for k in range(n)])
    gs = np.array([grip[min(int(k / s), grip.shape[0] - 1)] for k in range(n)])
    return cs, gs


# --------------------------------------------------------------------------- #
# one episode
# --------------------------------------------------------------------------- #
def run_episode(ev, suite, task_id: int, ep: int, spec, method: str, client: ChunkClient,
                hat, theta_n, horizon: int, bound_pos: float, bound_rot: float) -> dict:
    t_wall = time.time()
    client.reset()
    env, lang, obs = ev._init_env(suite, task_id, ep)
    plant = LiberoPlant(spec)
    ex, shaped = build_arm(method, spec, hat, theta_n, bound_pos=bound_pos, bound_rot=bound_rot)
    scale = {"slow15": 1.5, "slow20": 2.0}.get(method, 1.0)
    proc = ev.processor

    c_ref = c_intent = g_ref = None
    t_in = 0
    success = 0.0
    steps = horizon
    n_queries = 0
    track_intent: list[float] = []
    track_ref: list[float] = []
    u_pos_max = 0.0
    u_rot_max = 0.0
    n_bound_hits = 0
    codec_err = 0.0
    grip_cmd_prev = GRIP_OPEN
    n_grip_cmd_changes = 0
    first_cmd_close = None
    first_applied_close = None
    first_realized_close = None
    grip_applied_prev = GRIP_OPEN

    try:
        for t in range(horizon):
            y = measured_pose7(env)
            if c_ref is None or t_in >= c_ref.shape[0]:
                obs["robo_ori"] = proc.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
                obs["robo_pos"] = env.env.robots[0].controller.ee_pos
                rows = client.query(obs, lang)
                n_queries += 1
                c_raw, g_raw = chunk_to_targets(rows)
                if scale != 1.0:
                    c_raw, g_raw = time_scale(c_raw, g_raw, scale)
                c_intent = c_raw
                c_ref = shape_reference_libero(c_raw, theta_n, y) if shaped else c_raw
                g_ref = g_raw
                t_in = 0

            u, g_cmd = ex.step(y, c_ref, g_ref, t_in)
            cmd_pose, g_app, w = plant.command(y, u, g_cmd)
            action = pose7_to_action(cmd_pose, g_app)

            up = float(np.abs(u[:3]).max())
            ur = float(np.linalg.norm(u[3:]))
            u_pos_max = max(u_pos_max, up)
            u_rot_max = max(u_rot_max, ur)
            if up >= bound_pos - 1e-9 or ur >= bound_rot - 1e-9:
                n_bound_hits += 1
            j = min(t_in, c_intent.shape[0] - 1)
            codec_err = max(codec_err, float(np.linalg.norm(
                compose_delta(y, norm_delta(y, c_intent[j]))[:3] - c_intent[j][:3])))
            if g_cmd != grip_cmd_prev:
                n_grip_cmd_changes += 1
                grip_cmd_prev = g_cmd
            if first_cmd_close is None and g_cmd > 0:
                first_cmd_close = t
            if first_applied_close is None and g_app > 0:
                first_applied_close = t
            grip_applied_prev = g_app

            obs, reward, done, info = env.step(action)
            steps = t + 1
            if first_realized_close is None and gripper_opening(obs) < 0.5:
                first_realized_close = t
            y_next = measured_pose7(env)
            track_intent.append(float(np.linalg.norm(y_next[:3] - c_intent[j][:3])))
            track_ref.append(float(np.linalg.norm(y_next[:3] - c_ref[min(t_in, c_ref.shape[0] - 1)][:3])))
            t_in += 1
            if done:
                success = 1.0
                break
    finally:
        env.close()

    return {
        "task_id": int(task_id), "ep": int(ep), "task": lang,
        "pid": spec.pid, "family": spec.family, "method": method,
        "success": int(success), "steps": int(steps),
        "time_to_success_s": round(float(steps * DT), 3) if success else None,
        # vs_intent is the cross-arm comparable column: every arm is scored against
        # the policy chunk it was given (time-scaled for the Slow-down arms, since
        # that is genuinely where they intend to be at time t).  vs_ref scores each
        # arm against its own set point, which for the calibrated arms is the shaped
        # reference -- a check on the inversion, not a cross-arm comparison.
        "tracking_rmse_m": round(float(np.sqrt(np.mean(np.square(track_intent)))), 5) if track_intent else None,
        "tracking_rmse_vs_ref_m": round(float(np.sqrt(np.mean(np.square(track_ref)))), 5) if track_ref else None,
        "policy_queries": int(n_queries),
        "u_pos_max_norm": round(u_pos_max, 4),
        "u_rot_max_norm": round(u_rot_max, 4),
        "n_bound_hits": int(n_bound_hits),
        "codec_err_m": round(codec_err, 9),
        "n_grip_cmd_changes": int(n_grip_cmd_changes),
        "first_cmd_close_step": first_cmd_close,
        "first_applied_close_step": first_applied_close,
        "first_realized_close_step": first_realized_close,
        "grip_close_lag_steps": (None if (first_cmd_close is None or first_realized_close is None)
                                 else int(first_realized_close - first_cmd_close)),
        "wall_s": round(time.time() - t_wall, 2),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser("CCE LIBERO plant-family closed loop")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--set", default="family", choices=["family", "screen"])
    ap.add_argument("--plants", default="", help="csv override")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--init-seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--theta-dir", default=str(DATA / "theta"))
    ap.add_argument("--bound-pos", type=float, default=BOUND_POS)
    ap.add_argument("--bound-rot", type=float, default=BOUND_ROT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool = {s.pid: s for s in (list(FAMILY_V1) + list(SCREEN))}
    if args.plants:
        specs = [pool[p] for p in args.plants.split(",") if p]
    else:
        specs = list(FAMILY_V1) if args.set == "family" else list(SCREEN)
    methods = [m for m in args.methods.split(",") if m]
    tasks = [int(x) for x in args.tasks.split(",") if x.strip() != ""]

    hats = json.loads((Path(args.theta_dir) / f"theta_hat_{UID}.json").read_text())
    theta_n = theta_from_hat(hats[NOMINAL_PID])

    sys.path.insert(0, XVLA_LIBERO)
    import libero_client as LC

    ev = LC.LIBEROEval(task_suite_name=args.suite, eval_horizon=args.horizon, act_type="abs",
                       num_episodes=args.episodes, init_seed=args.init_seed)
    ev.base_dir = Path(args.out).parent
    ev.base_dir.mkdir(parents=True, exist_ok=True)
    ev._log_results = lambda metrics: None
    suite = [LC.benchmark_dict[t]() for t in LC.LIBERO_DATASETS[args.suite]][0]

    client = ChunkClient(args.host, args.port)

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["pid"], r["method"], r["task_id"], r["ep"]))
            except Exception:
                pass

    work = []
    idx = 0
    for sp in specs:
        for m in methods:
            for tid in tasks:
                for ep in range(args.episodes):
                    if idx % args.nshards == args.shard:
                        work.append((sp, m, tid, ep))
                    idx += 1

    fh = out_path.open("a", encoding="utf-8")
    t0 = time.time()
    n_run = 0
    for sp, m, tid, ep in work:
        if (sp.pid, m, tid, ep) in done:
            continue
        rec = run_episode(ev, suite, tid, ep, sp, m, client, hats.get(sp.pid), theta_n,
                          args.horizon, args.bound_pos, args.bound_rot)
        rec["suite"] = args.suite
        rec["shard"] = args.shard
        rec["bound_pos"] = args.bound_pos
        rec["bound_rot"] = args.bound_rot
        fh.write(json.dumps(rec, ensure_ascii=False, default=json_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        n_run += 1
        print(json.dumps({k: rec[k] for k in ("pid", "method", "task_id", "ep", "success",
                                              "steps", "tracking_rmse_m", "wall_s")},
                         default=json_default), flush=True)
    fh.close()
    print(json.dumps({"shard": args.shard, "n_run": n_run, "n_planned": len(work),
                      "wall_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
