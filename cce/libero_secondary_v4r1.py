#!/usr/bin/env python
"""CCE r1 secondary controls; frozen primary runner remains unchanged.

Per control step, at 20 Hz:

    policy chunk (30 absolute EE targets + hard gripper, from the X-VLA server)
      -> [optional time scaling]                        Slow-down arms
      -> [optional nominal shaping]                     calibrated arms
      -> LiberoExecutor.step(measured pose, ref, t)     normalized command increment
      -> LiberoPlant.command                            D1-D4 interface mismatch
      -> absolute target + gripper -> env.step          robosuite absolute OSC

Chunks are drained at one row per step except TE, which queries every ten steps
and averages overlapping predictions. Executor state persists across chunks.

Everything that defines the positive control is taken from upstream
``evaluation/libero/libero_client.py``: task loading and init states, the agentview
double flip, the wrist view, ``robot.controller.use_delta = False``, the proprio
bookkeeping (initialised once from the first observation, then overwritten with the
last row of each returned chunk), the 30-step drain, the +-1 gripper threshold, the
800-step horizon and ``domain_id = 3`` / 10 denoising steps.

The six secondary arms are universal_fixed, te_requery, v4_no_gate,
v4_no_unilateral, v4_no_cap and v4_no_greybox. Naive and ours_v4 are available
only for development implementation checks. See CCE_secondary_v4r1.md.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import hashlib
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
    build_arm as old_build_arm,
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
from cce.libero_exec_v4r1 import RelativeExecutor, RelativeTheta
from cce.libero_plant_v3 import FAMILY_V3, BY_ID_V3, DEV_PIDS
from cce.bench_relative_v4r1 import gate_values

STEP_DIR = None
PACK = None
GATE = None
RHO = None


def build_arm(method, spec, hat, theta_n, **kw):
    if method in ('naive', 'slow15', 'slow20', 'slow30', 'te_requery'):
        return old_build_arm('naive', spec, hat, theta_n, **kw)
    if method == 'ours_v3':
        return old_build_arm('ours_p', spec, hat, theta_n, **kw)
    rel = RelativeTheta.from_fit(PACK['plants'][spec.pid])
    if method == 'oracle_v4':
        rel = RelativeTheta(spec.delay_steps, np.full(6, spec.tau_s), np.full(6, spec.gain), spec.grip_delay_steps)
    if method == 'universal_fixed':
        rel = RelativeTheta(3, np.full(6, .1), np.full(6, .85), 10)
    if method == 'v4_no_greybox':
        rel = RelativeTheta.from_fit(PACK['plants'][spec.pid]['arx_relative'])
    rho = float(method.removeprefix('v4_rho')) if method.startswith('v4_rho') else RHO
    if method == 'v4_no_cap':
        rho = float('inf')
    from cce.libero_exec_v4r1 import predicted_h
    # Oracle/fixed/ARX gate is computed from that arm's own information only.
    h = GATE['H'][spec.pid]
    if method in ('oracle_v4', 'universal_fixed', 'v4_no_greybox'):
        h = predicted_h(rel, np.array(PACK['nominal_alpha']), np.array(PACK['nominal_gain']), GATE_REFERENCE)
    ex = RelativeExecutor(rel, PACK['nominal_alpha'], PACK['nominal_gain'], rho_max=rho,
                          h=h, h0=GATE['H0'], gate=method != 'v4_no_gate',
                          unilateral=method != 'v4_no_unilateral')
    return ex, False
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
        self.query_index = 0

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
            "seed": int.from_bytes(hashlib.sha256(f"{self.episode_key}:{self.query_index}".encode()).digest()[:4], "little") % (2**31-1),
        }
        resp = requests.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        action = np.array(resp.json()["action"])
        if self.query_index == 0 and getattr(self, 'verify_repeat', False):
            repeat = requests.post(self.url, json=payload, timeout=self.timeout)
            repeat.raise_for_status()
            assert np.array_equal(action, np.array(repeat.json()['action'])), 'seeded response mismatch'
        self.query_index += 1
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
    step_rows = []
    from cce.temporal_ensemble_v4r1 import TemporalEnsemble
    ensemble = TemporalEnsemble(m=.1) if method == "te_requery" else None
    client.episode_key = f"{client.suite_name}:{task_id}:{ep}"
    client.reset()
    env, lang, obs = ev._init_env(suite, task_id, ep)
    plant = LiberoPlant(spec)
    ex, shaped = build_arm(method, spec, hat, theta_n, bound_pos=bound_pos, bound_rot=bound_rot)
    scale = {"slow15": 1.5, "slow20": 2.0, "slow30": 3.0, "ours_v4_slow15": 1.5, "ours_v4_slow20": 2.0}.get(method, 1.0)
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
            query_due = (t % 10 == 0) if ensemble is not None else (c_ref is None or t_in >= c_ref.shape[0])
            if query_due:
                obs["robo_ori"] = proc.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
                obs["robo_pos"] = env.env.robots[0].controller.ee_pos
                rows = client.query(obs, lang)
                n_queries += 1
                c_raw, g_raw = chunk_to_targets(rows)
                if ensemble is not None:
                    ensemble.add(t, c_raw, g_raw)
                if scale != 1.0:
                    c_raw, g_raw = time_scale(c_raw, g_raw, scale)
                c_intent = c_raw
                c_ref = shape_reference_libero(c_raw, theta_n, y) if shaped else c_raw
                g_ref = g_raw
                t_in = 0

            if ensemble is not None:
                target, target_grip = ensemble.target(t)
                c_ref = c_intent = target[None, :]
                g_ref = np.array([target_grip]); t_in = 0
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
            if STEP_DIR is not None:
                step_rows.append(np.concatenate([y, c_intent[j], u, w, y_next, [g_cmd, g_app, gripper_opening(obs)]]))
            track_intent.append(float(np.linalg.norm(y_next[:3] - c_intent[j][:3])))
            track_ref.append(float(np.linalg.norm(y_next[:3] - c_ref[min(t_in, c_ref.shape[0] - 1)][:3])))
            t_in += 1
            if done:
                success = 1.0
                break
    finally:
        env.close()

    if STEP_DIR is not None:
        step_path = STEP_DIR / f'{spec.pid}__{method}__t{task_id}__e{ep}.npz'
        np.savez_compressed(step_path, steps=np.array(step_rows), columns=np.array(['y:7', 'intent:7', 'u:6', 'applied:6', 'y_next:7', 'g_cmd:1', 'g_applied:1', 'opening:1']))
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
        "tracking_rmse_m": float(np.sqrt(np.mean(np.square(track_intent)))) if track_intent else None,
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
def main():
    global PACK, GATE, RHO, STEP_DIR, GATE_REFERENCE
    import requests
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', required=True)
    ap.add_argument('--stage', choices=['dev', 'g0', 'formal', 'secondary'], required=True)
    ap.add_argument('--freeze')
    ap.add_argument('--plants', default='L00_nominal')
    ap.add_argument('--methods', default='naive')
    ap.add_argument('--tasks', default='0,1,2,3,4,5,6,7,8,9')
    ap.add_argument('--episodes', type=int, default=2)
    ap.add_argument('--rho', type=float, default=2.)
    ap.add_argument('--suite', default='libero_spatial')
    ap.add_argument('--port', type=int, default=8035)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--log-steps', action='store_true')
    ap.add_argument('--trace-first', action='store_true', help='Record task 0 episode 0 only, for every arm/plant')
    ap.add_argument('--verify-repeat', action='store_true')
    ap.add_argument('--horizon', type=int, default=800)
    args = ap.parse_args()
    if args.stage == 'formal':
        raise ValueError('Formal execution remains disabled until protocol and full gates are frozen')
    pids, methods = args.plants.split(','), args.methods.split(',')
    allowed = {'naive', 'ours_v4', 'te_requery', 'universal_fixed', 'v4_no_greybox', 'v4_no_gate', 'v4_no_cap', 'v4_no_unilateral'}
    assert set(methods) <= allowed
    if args.stage != 'secondary':
        assert set(pids) <= set(DEV_PIDS), 'Nondevelopment plants require a frozen run'
    else:
        from cce.freeze_secondary_v4r1 import validate_freeze
        frozen = validate_freeze(args.freeze)
        assert args.suite == frozen['suite'] and args.rho == frozen['rho']
        assert set(pids) <= set(frozen['plant_ids'])
        assert args.tasks == ','.join(map(str, frozen['tasks']))
        assert all(frozen['episodes'][m] == args.episodes for m in methods)
        assert hashlib.sha256(Path(args.pack).read_bytes()).hexdigest() == frozen['pack_sha256']
    if args.stage == 'g0':
        assert pids == ['L00_nominal'] and methods == ['naive']
    assert args.horizon == 800
    PACK = json.loads(Path(args.pack).read_text())
    with np.load('data/cce/libero/probes/probe_libero_panda_L00_nominal.npz') as z:
        GATE_REFERENCE = z['targets']
    GATE = gate_values(PACK, GATE_REFERENCE)
    if args.stage == 'secondary':
        assert GATE['H0'] == frozen['H0'], 'Frozen gate threshold mismatch'
    RHO = args.rho
    hats = json.loads(Path('data/cce/libero/theta/theta_hat_libero_panda_v2.json').read_text())
    hats.update(json.loads(Path('data/cce/takeover_v4_20260908/new_probes/probe_summary.json').read_text())['hats'])
    theta_n = theta_from_hat(hats[NOMINAL_PID])
    info = requests.get(f'http://127.0.0.1:{args.port}/ready', timeout=5).json()
    assert info.get('supports_request_seed'), 'Wrong policy server'
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Choose a fresh output path; implicit resume is disabled')
    if args.log_steps or args.trace_first:
        STEP_DIR = path.parent/(path.stem+'_steps')
        STEP_DIR.mkdir(exist_ok=False)
    sys.path.insert(0, XVLA_LIBERO)
    import libero_client as LC
    ev = LC.LIBEROEval(task_suite_name=args.suite, eval_horizon=800, act_type='abs', num_episodes=args.episodes, init_seed=42)
    ev.base_dir = path.parent
    ev._log_results = lambda metrics: None
    suite = [LC.benchmark_dict[t]() for t in LC.LIBERO_DATASETS[args.suite]][0]
    client = ChunkClient('127.0.0.1', args.port)
    client.suite_name, client.verify_repeat = args.suite, args.verify_repeat
    meta = {'stage': args.stage, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'pack_sha256': hashlib.sha256(Path(args.pack).read_bytes()).hexdigest(), 'gate': GATE,
            'rho': str(RHO), 'server': info, 'args': vars(args)}
    if args.freeze:
        meta['freeze_sha256'] = hashlib.sha256(Path(args.freeze).read_bytes()).hexdigest()
    path.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2)+'\n')
    work = [(p,m,t,e) for p in pids for m in methods for t in map(int,args.tasks.split(',')) for e in range(args.episodes)]
    with path.open('x') as f:
        for p,m,t,e in work[args.shard::args.nshards]:
            if args.trace_first:
                STEP_DIR = path.parent/(path.stem+'_steps') if t == 0 and e == 0 else None
            rec = run_episode(ev, suite, t, e, BY_ID_V3[p], m, client, hats.get(p), theta_n, 800, 1., 1.)
            rec.update(suite=args.suite, stage=args.stage, rho=str(RHO), init_seed=42,
                       bound_pos=1., bound_rot=1., seed_protocol='sha256(suite:task:ep:query_index)')
            f.write(json.dumps(rec, ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
            print(json.dumps({k:rec[k] for k in ['pid','method','task_id','ep','success','steps','tracking_rmse_m']}), flush=True)


if __name__ == '__main__':
    main()
