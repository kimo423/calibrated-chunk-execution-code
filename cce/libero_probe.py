#!/usr/bin/env python
"""CCE probe + ARX identification on the LIBERO bed (prereg v2 amendment, L2).

Same probe design as ``cce/probe.py`` -- 140 steps at 20 Hz = 7.00 s, laid out as
settle / 3-axis linear chirp / horizontal saturation burst / quiet gap / 3-axis
attitude sine / hold-and-close -- and the same ARX fit (``cce/identify.py``: per axis
one integer input delay grid-searched 0..300 ms plus a closed-form ``(a, b)``, read
out as delay / tau / gain, with the gripper lead taken from the 90 %-arrival time of
the closing step relative to the nominal plant).

Two adaptations, both forced by the bed and both recorded in the output JSON:

* the excitation is emitted as *absolute* end-effector targets (the plant converts
  the target into the increment relative to the measured pose, applies D1-D4, and
  composes it back), because LIBERO's OSC controller is in absolute mode;
* the saturation burst is specified in the command domain (``BURST_CMD_NORM``)
  instead of as a +-0.5 m target.  On ManiSkill the +-0.5 m target was clipped to the
  +-0.1 m action bound and therefore held the command at saturation for two steps;
  robosuite's absolute OSC has no such bound, so a +-0.5 m target would be a real
  half-metre lunge.  The burst amplitude is chosen to stay table-safe.

What is identified is the *composite* of the injected plant and the OSC controller's
own command-to-achieved response.  The composite is what the executor inverts; the
nominal-plant row is the controller's own response and is the "nominal theta" the v2
amendment requires to be re-identified on this bed.
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

from cce.common import DT, json_default, norm_delta, qnorm, quat_mul  # noqa: E402
from cce.identify import identify_plant  # noqa: E402
from cce.libero_plant import (  # noqa: E402
    FAMILY_V1,
    NOMINAL_PID,
    SCREEN,
    GRIP_CLOSE,
    GRIP_OPEN,
    LiberoPlant,
    compose_delta,
    pose7_from_pos_mat,
    pose7_to_action,
)
from cce.probe import (  # noqa: E402
    ATT_START,
    BURST_START,
    CHIRP_F0,
    CHIRP_F1,
    CHIRP_START,
    GAP_START,
    GRIP_CLOSE_STEP,
    N_BURST,
    N_CHIRP,
    N_PROBE,
)

CHIRP_AMP_M = 0.03
ATT_AMP_RAD = 0.10
BURST_CMD_NORM = 0.6  # 6 cm commanded increment per step (table-safe on this bed)
BURST_AXES = [(0, +1), (0, -1), (1, +1), (1, -1)]
GRIP_TRAVEL = 0.08  # Panda finger pair, fully open q0 - q1
DATA = Path("/opt/cce/data/cce/libero")
UID = "libero_panda"


# --------------------------------------------------------------------------- #
# env access
# --------------------------------------------------------------------------- #
def measured_pose7(env) -> np.ndarray:
    c = env.env.robots[0].controller
    return pose7_from_pos_mat(c.ee_pos, c.ee_ori_mat)


def gripper_opening(obs) -> float:
    q = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1)
    return float(np.clip((q[0] - q[1]) / GRIP_TRAVEL, 0.0, 1.0))


def make_evaluator(suite_name: str, init_seed: int = 42):
    sys.path.insert(0, XVLA_LIBERO)
    import libero_client as LC

    ev = LC.LIBEROEval(task_suite_name=suite_name, eval_horizon=800, act_type="abs",
                       num_episodes=1, init_seed=init_seed)
    ev.base_dir = Path("/tmp")
    ev._log_results = lambda metrics: None
    suites = [LC.benchmark_dict[t]() for t in LC.LIBERO_DATASETS[suite_name]]
    return ev, suites, LC


# --------------------------------------------------------------------------- #
# probe design
# --------------------------------------------------------------------------- #
def probe_reference(home: np.ndarray):
    """Absolute end-effector target series + gripper command series (LIBERO units)."""
    from cce.common import euler_xyz_to_mat, mat_to_quat

    home = np.asarray(home, dtype=np.float64).reshape(7)
    targets = np.tile(home, (N_PROBE, 1))
    grip = np.full(N_PROBE, GRIP_OPEN, dtype=np.float64)

    T = N_CHIRP * DT
    for k in range(N_CHIRP):
        t = k * DT
        phase = 2 * math.pi * (CHIRP_F0 * t + 0.5 * (CHIRP_F1 - CHIRP_F0) / T * t * t)
        for ax in range(3):
            targets[CHIRP_START + k, ax] = home[ax] + CHIRP_AMP_M * math.sin(
                phase + 2 * math.pi * ax / 3.0)

    for k in range(14):  # attitude sine, one full cycle
        base_phase = 2 * math.pi * k / 14
        e = np.array([ATT_AMP_RAD * math.sin(base_phase + 2 * math.pi * ax / 3.0) for ax in range(3)])
        dq = mat_to_quat(euler_xyz_to_mat(e))
        targets[ATT_START + k, 3:] = qnorm(quat_mul(dq, qnorm(home[3:])))

    grip[GRIP_CLOSE_STEP:] = GRIP_CLOSE
    return targets, grip


def probe_commands(home: np.ndarray):
    """Open-loop command series in the normalized increment domain.

    Differencing is against a *virtual* perfectly-tracking pose, not the measured
    pose, so the excitation is identical on every plant (a requirement for comparing
    theta-hat) and never closes a loop through the disturbed arm.
    """
    targets, grip = probe_reference(home)
    u = np.zeros((N_PROBE, 7), dtype=np.float64)
    virt = np.asarray(home, dtype=np.float64).reshape(7).copy()
    for t in range(N_PROBE):
        if BURST_START <= t < BURST_START + N_BURST:
            k = t - BURST_START
            ax, sgn = BURST_AXES[k // 2]
            a6 = np.zeros(6)
            a6[ax] = sgn * BURST_CMD_NORM
        else:
            a6 = norm_delta(virt, targets[t])
        u[t, :6] = a6
        u[t, 6] = grip[t]
        virt = compose_delta(virt, a6)
    return u, targets, grip


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def run_probe(ev, suite, spec, task_id: int = 0, ep: int = 0, z_floor: float | None = None) -> dict:
    env, lang, obs = ev._init_env(suite, task_id, ep)
    try:
        y0 = measured_pose7(env)
        u, targets, grip_cmd = probe_commands(y0)
        plant = LiberoPlant(spec)
        y = np.zeros((N_PROBE + 1, 7), dtype=np.float64)
        gopen = np.zeros(N_PROBE + 1, dtype=np.float64)
        cmd_log = np.zeros((N_PROBE, 7), dtype=np.float64)
        y[0] = y0
        gopen[0] = gripper_opening(obs)
        t0 = time.time()
        for t in range(N_PROBE):
            yt = measured_pose7(env)
            cmd, g, w = plant.command(yt, u[t, :6], u[t, 6])
            cmd_log[t, :6] = w
            cmd_log[t, 6] = g
            obs, _, done, _ = env.step(pose7_to_action(cmd, g))
            y[t + 1] = measured_pose7(env)
            gopen[t + 1] = gripper_opening(obs)
            if z_floor is not None and y[t + 1, 2] < z_floor:
                raise RuntimeError(f"probe unsafe: z={y[t + 1, 2]:.4f} < {z_floor} at t={t}")
        wall = time.time() - t0
    finally:
        env.close()
    return {"uid": UID, "pid": spec.pid, "u": u, "y": y, "grip_open": gopen,
            "grip_cmd": grip_cmd, "home": y0, "targets": targets, "applied": cmd_log,
            "probe_seconds": N_PROBE * DT, "wall_s": round(wall, 2), "task": lang}


def probe_stats(rec: dict) -> dict:
    y = rec["y"]
    dy = np.linalg.norm(y[1:, :3] - y[:-1, :3], axis=1)
    return {
        "pid": rec["pid"],
        "probe_s": rec["probe_seconds"],
        "wall_s": rec["wall_s"],
        "z_min_m": round(float(y[:, 2].min()), 4),
        "z_home_m": round(float(rec["home"][2]), 4),
        "xyz_excursion_m": round(float(np.abs(y[:, :3] - rec["home"][:3]).max()), 4),
        "max_step_disp_m": round(float(dy.max()), 5),
        "rms_step_disp_m": round(float(np.sqrt((dy ** 2).mean())), 5),
        "grip_open_start": round(float(rec["grip_open"][0]), 3),
        "grip_open_min": round(float(rec["grip_open"].min()), 3),
    }


# --------------------------------------------------------------------------- #
# identification + report
# --------------------------------------------------------------------------- #
def summarize(hats: dict, specs) -> dict:
    nom = hats[NOMINAL_PID]
    g_nom, tau_nom = nom["gain_agg"], nom["tau_s_agg"]
    by_id = {s.pid: s for s in specs}
    rows = []
    for pid, h in hats.items():
        th = by_id[pid].theta()
        rows.append({
            "pid": pid, "family": by_id[pid].family,
            "d_true_steps": th["delay_steps"], "d_hat_steps": h["delay_steps_agg"],
            "d_err_steps": h["delay_steps_agg"] - th["delay_steps"],
            "tau_true_s": th["tau_s"], "tau_hat_s": h["tau_s_agg"],
            "tau_rel_s": float(max(0.0, h["tau_s_agg"] - tau_nom)),
            "g_true": th["gain"], "g_hat": h["gain_agg"],
            "g_rel": float(h["gain_agg"] / g_nom) if g_nom > 1e-9 else float("nan"),
            "grip_lead_true_steps": th["grip_lead_steps"],
            "grip_lead_hat_steps": h["grip_lead_steps_hat"],
            "grip_t90_s": h["grip_t90_s"],
            "r2_min_trans": h["r2_min_trans"], "r2_min_rot": h["r2_min_rot"],
            "r2free_min_trans": h["r2free_min_trans"], "r2free_min_rot": h["r2free_min_rot"],
            "w_max_burst_trans": h["w_max_burst_trans"],
            "clip_pos_true_m": th["clip_pos_m"],
            "clip_pos_norm_burst": h["clip_pos_norm_burst"],
        })
    arr = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in
           ("d_true_steps", "d_hat_steps", "tau_true_s", "tau_rel_s", "g_true", "g_rel",
            "grip_lead_true_steps", "grip_lead_hat_steps")}

    def corr(a, b):
        if np.std(arr[a]) < 1e-12 or np.std(arr[b]) < 1e-12:
            return float("nan")
        return float(np.corrcoef(arr[a], arr[b])[0, 1])

    return {
        "uid": UID,
        "n_plants": len(rows),
        "probe_seconds": N_PROBE * DT,
        "rows": rows,
        "agreement": {
            "delay_corr": corr("d_true_steps", "d_hat_steps"),
            "delay_mae_steps": float(np.abs(arr["d_true_steps"] - arr["d_hat_steps"]).mean()),
            "delay_mae_ms": float(np.abs(arr["d_true_steps"] - arr["d_hat_steps"]).mean() * DT * 1000),
            "tau_corr_rel": corr("tau_true_s", "tau_rel_s"),
            "tau_mae_rel_s": float(np.abs(arr["tau_true_s"] - arr["tau_rel_s"]).mean()),
            "gain_corr_rel": corr("g_true", "g_rel"),
            "gain_mae_rel": float(np.abs(arr["g_true"] - arr["g_rel"]).mean()),
            "gain_hat_on_nominal": float(nom["gain_agg"]),
            "tau_hat_on_nominal_s": float(nom["tau_s_agg"]),
            "delay_hat_on_nominal_steps": float(nom["delay_steps_agg"]),
            "grip_t90_nominal_s": float(nom["grip_t90_s"]),
            "grip_lead_corr": corr("grip_lead_true_steps", "grip_lead_hat_steps"),
            "grip_lead_mae_steps": float(np.abs(arr["grip_lead_true_steps"]
                                                - arr["grip_lead_hat_steps"]).mean()),
            "n_axes_below_r2_gate": int(sum(1 for h in hats.values() for a in h["axes"]
                                            if not a["identifiable"])),
            "n_axes_total": int(6 * len(rows)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser("CCE probe + ARX identification on LIBERO")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--set", default="family", choices=["family", "screen", "all", "nominal"])
    ap.add_argument("--z-floor", type=float, default=None)
    ap.add_argument("--out-dir", default=str(DATA / "theta"))
    ap.add_argument("--probe-dir", default=str(DATA / "probes"))
    ap.add_argument("--dry", action="store_true", help="nominal plant only, report geometry")
    args = ap.parse_args()

    if args.set == "family":
        specs = list(FAMILY_V1)
    elif args.set == "screen":
        specs = list(SCREEN)
    elif args.set == "nominal":
        specs = [s for s in FAMILY_V1 if s.pid == NOMINAL_PID]
    else:
        seen, specs = set(), []
        for s in FAMILY_V1 + SCREEN:
            if s.pid not in seen:
                seen.add(s.pid)
                specs.append(s)
    if args.dry:
        specs = [s for s in specs if s.pid == NOMINAL_PID]

    ev, suites, LC = make_evaluator(args.suite)
    suite = suites[0]
    out_dir, probe_dir = Path(args.out_dir), Path(args.probe_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    hats, stats = {}, []
    for spec in specs:
        rec = run_probe(ev, suite, spec, args.task_id, args.ep, args.z_floor)
        np.savez_compressed(probe_dir / f"probe_{UID}_{spec.pid}.npz",
                            u=rec["u"], y=rec["y"], grip_open=rec["grip_open"],
                            grip_cmd=rec["grip_cmd"], home=rec["home"],
                            targets=rec["targets"], applied=rec["applied"])
        st = probe_stats(rec)
        stats.append(st)
        print(json.dumps(st, default=json_default), flush=True)
        if not args.dry:
            hats[spec.pid] = identify_plant(rec, sat_aware=False)

    if args.dry:
        print(json.dumps({"dry": True, "stats": stats}, indent=2, default=json_default))
        return

    t90_nom = hats[NOMINAL_PID]["grip_t90_s"]
    for h in hats.values():
        lead = h["grip_t90_s"] - t90_nom
        h["grip_t90_nominal_s"] = t90_nom
        h["grip_lead_s_hat"] = float(lead) if np.isfinite(lead) else 0.0
        h["grip_lead_steps_hat"] = int(max(0, round(h["grip_lead_s_hat"] / DT))) if np.isfinite(lead) else 0

    (out_dir / f"theta_hat_{UID}.json").write_text(
        json.dumps(hats, indent=2, default=json_default) + "\n")
    summary = summarize(hats, specs)
    summary["probe_stats"] = stats
    summary["probe_design"] = {
        "steps": int(N_PROBE), "seconds": N_PROBE * DT,
        "chirp_amp_m": CHIRP_AMP_M, "chirp_hz": [CHIRP_F0, CHIRP_F1],
        "burst_cmd_norm": BURST_CMD_NORM, "burst_cmd_m": BURST_CMD_NORM * 0.1,
        "attitude_amp_rad": ATT_AMP_RAD, "grip_close_step": int(GRIP_CLOSE_STEP),
        "suite": args.suite, "task_id": args.task_id, "ep": args.ep,
    }
    (out_dir / f"identify_{UID}.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n")
    print(json.dumps({"n_plants": len(hats), "agreement": summary["agreement"]},
                     indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
