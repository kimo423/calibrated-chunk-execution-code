#!/usr/bin/env python
"""CCE probe: <=7 s task-agnostic excitation in the policy's own action space.

The probe never touches joints or the simulator: it emits *absolute* TCP targets,
converts them with the shared adapter (``shared_ee_delta_action``), pushes them
through the plant wrapper and then ``to_plant_action`` -> ``env.step`` -- exactly
the deployment execution path.

Layout (140 steps @ 20 Hz = 7.00 s):
    [  0,   4) 0.20 s  settle at the reset pose
    [  4,  72) 3.40 s  3-axis linear chirp 0.2 -> 2.0 Hz, +-3 cm, phase-staggered
    [ 72,  80) 0.40 s  horizontal saturation burst (+-x, +-y at full command scale)
    [ 80,  88) 0.40 s  quiet gap so the burst has flushed even at d = 250 ms
    [ 88, 102) 0.70 s  3-axis attitude sine, +-0.10 rad
    [102, 140) 1.90 s  hold pose; gripper closes at 102 and stays closed
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
from pathlib import Path

import numpy as np

from cce.common import DT, POS_SCALE, apply_norm_delta, gripper_opening, make_env, tcp_pose
from cce.plants import PlantWrapper, plant_ids, select

PROBE_SEED = 20267100
N_SETTLE, N_CHIRP, N_BURST, N_GAP, N_ATT, N_HOLD = 4, 68, 8, 8, 14, 38
N_PROBE = N_SETTLE + N_CHIRP + N_BURST + N_GAP + N_ATT + N_HOLD  # 140 steps = 7.00 s
CHIRP_START = N_SETTLE
BURST_START = CHIRP_START + N_CHIRP          # 72
GAP_START = BURST_START + N_BURST            # 80
ATT_START = GAP_START + N_GAP                # 88
HOLD_START = ATT_START + N_ATT               # 102
CHIRP_AMP_M = 0.03
CHIRP_F0, CHIRP_F1 = 0.2, 2.0
ATT_AMP_RAD = 0.10
GRIP_CLOSE_STEP = HOLD_START  # 102
DATA = Path("/opt/cce/data/cce")


def _quat_from_euler_xyz(e):
    from cce.common import euler_xyz_to_mat, mat_to_quat

    return mat_to_quat(euler_xyz_to_mat(np.asarray(e, dtype=np.float64)))


def probe_reference(home: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute TCP target sequence + gripper command sequence for the probe."""
    from cce.common import quat_mul, qnorm

    home = np.asarray(home, dtype=np.float64).reshape(7)
    targets = np.tile(home, (N_PROBE, 1))
    grip = np.ones(N_PROBE, dtype=np.float64)

    # chirp
    T = N_CHIRP * DT
    for k in range(N_CHIRP):
        t = k * DT
        phase = 2 * math.pi * (CHIRP_F0 * t + 0.5 * (CHIRP_F1 - CHIRP_F0) / T * t * t)
        for ax in range(3):
            off = CHIRP_AMP_M * math.sin(phase + 2 * math.pi * ax / 3.0)
            targets[CHIRP_START + k, ax] = home[ax] + off

    # saturation burst: +-x, +-y at full command scale (horizontal only -> table safe)
    burst = [(0, +1), (0, -1), (1, +1), (1, -1)]
    for k in range(N_BURST):
        ax, sgn = burst[k // 2]
        targets[BURST_START + k, ax] = home[ax] + sgn * 0.5

    # 3-axis attitude sine (one full cycle, phase-staggered so rx/ry/rz are all excited)
    for k in range(N_ATT):
        base_phase = 2 * math.pi * k / N_ATT
        e = np.array([ATT_AMP_RAD * math.sin(base_phase + 2 * math.pi * ax / 3.0) for ax in range(3)])
        dq = _quat_from_euler_xyz(e)
        targets[ATT_START + k, 3:] = qnorm(quat_mul(dq, qnorm(home[3:])))

    grip[GRIP_CLOSE_STEP:] = -1.0
    return targets, grip


def probe_commands(home: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the (open-loop) probe command series in the policy action space.

    The excitation is generated exactly as a chunk policy's output would be executed
    -- absolute TCP target -> ``shared_ee_delta_action`` -> normalized command -- but
    the pose the adapter differences against is the *virtual* reference produced by a
    perfect plant, not the measured pose.  That keeps the excitation identical across
    plants (a requirement for comparing theta-hat) and open loop.

    A measured-pose (closed-loop) probe was tried first and is not usable: with the
    deadbeat residual adapter and >=100 ms of command delay the loop rings and the arm
    drives into the table (min TCP z fell to 0.010 m on D1_d250).
    """
    from calibration_to_prompt.pilot2_ee_adapter import shared_ee_delta_action

    targets, grip_cmd = probe_reference(home)
    u = np.zeros((N_PROBE, 7), dtype=np.float64)
    virt = np.asarray(home, dtype=np.float64).reshape(7).copy()
    for t in range(N_PROBE):
        cmd = shared_ee_delta_action(virt, targets[t], gripper=float(grip_cmd[t]))
        u[t] = cmd
        virt = apply_norm_delta(virt, cmd[:6])
    return u, targets, grip_cmd


def run_probe(uid: str, spec, env=None, seed: int = PROBE_SEED) -> dict:
    """Execute the probe on one plant; returns commands, achieved poses and gripper."""
    from calibration_to_prompt.dev_anchor import to_plant_action

    own_env = env is None
    if own_env:
        env = make_env(uid, max_episode_steps=N_PROBE + 10)
    env.reset(seed=int(seed))
    base = env.unwrapped
    plant = PlantWrapper(spec)
    home = tcp_pose(base)
    u, targets, grip_cmd = probe_commands(home)

    y = np.zeros((N_PROBE + 1, 7), dtype=np.float64)
    gopen = np.zeros(N_PROBE + 1, dtype=np.float64)
    y[0] = home
    gopen[0] = gripper_opening(uid, base)
    for t in range(N_PROBE):
        env.step(to_plant_action(uid, plant.apply(u[t])))
        y[t + 1] = tcp_pose(base)
        gopen[t + 1] = gripper_opening(uid, base)
    if own_env:
        env.close()
    return {
        "uid": uid,
        "pid": spec.pid,
        "seed": int(seed),
        "u": u,
        "y": y,
        "grip_open": gopen,
        "grip_cmd": grip_cmd,
        "home": home,
        "targets": targets,
        "probe_seconds": N_PROBE * DT,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default="panda")
    ap.add_argument("--plants", default="")
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pids = [p for p in args.plants.split(",") if p] or plant_ids()
    out_dir = Path(args.out) if args.out else DATA / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args.uid, max_episode_steps=N_PROBE + 10)
    for spec in select(pids):
        rec = run_probe(args.uid, spec, env=env, seed=args.seed)
        np.savez_compressed(
            out_dir / f"probe_{args.uid}_{spec.pid}.npz",
            u=rec["u"], y=rec["y"], grip_open=rec["grip_open"],
            grip_cmd=rec["grip_cmd"], home=rec["home"], targets=rec["targets"],
        )
        dy = np.linalg.norm(rec["y"][1:, :3] - rec["y"][:-1, :3], axis=1)
        print(json.dumps({
            "uid": args.uid, "pid": spec.pid, "probe_s": rec["probe_seconds"],
            "max_step_disp_m": round(float(dy.max()), 5),
            "rms_step_disp_m": round(float(np.sqrt((dy ** 2).mean())), 5),
            "grip_open_min": round(float(rec["grip_open"].min()), 3),
            "z_min_m": round(float(rec["y"][:, 2].min()), 4),
        }), flush=True)
    env.close()


if __name__ == "__main__":
    main()
