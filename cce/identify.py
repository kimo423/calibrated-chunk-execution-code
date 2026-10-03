#!/usr/bin/env python
"""ARX identification of the interface-level plant from a <=7 s probe.

Per axis (6 = xyz + rx ry rz of the normalized action) we fit

    w[t] = a * w[t-1] + b * u[t-d]

where ``u`` is the command handed to the plant and ``w[t] = norm_delta(y[t], y[t+1])``
is the *realized* normalized displacement.  ``d`` is grid-searched over 0..6 steps
(0..300 ms); ``(a, b)`` are closed-form least squares.  Then

    alpha = a,  tau = -DT / ln(a),  gain = b / (1 - a).

The identified quantity is the composite of the injected plant *and* the
simulator's own command-to-achieved response; that composite is what the executor
must invert, so no attempt is made to claim these are physical parameters.

The gripper lead is the 90 %-arrival time of the closing step response minus the
same quantity measured on the nominal plant of the same robot.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
from pathlib import Path

import numpy as np

from cce.common import DT, json_default, norm_delta
from cce.plants import NOMINAL_PID, get_plant, plant_ids
from cce.probe import (
    ATT_START,
    BURST_START,
    DATA,
    GAP_START,
    GRIP_CLOSE_STEP,
    N_PROBE,
)

MAX_DELAY_STEPS = 6  # 300 ms
R2_GATE = 0.7
TRANS_FIT_WINDOWS = [(0, BURST_START), (0, ATT_START)]  # chirp only / chirp + step burst
TRANS_EVAL_WINDOW = (0, ATT_START)  # whole arm phase, free-run scored here
ROT_WINDOW = (ATT_START - 2, N_PROBE)  # attitude sine + free decay
BURST_WINDOW = (BURST_START, GAP_START + 6)  # burst response incl. max delay


def realized_deltas(y: np.ndarray) -> np.ndarray:
    """Achieved pose sequence -> realized normalized 6-vector per step."""
    y = np.asarray(y, dtype=np.float64)
    return np.stack([norm_delta(y[t], y[t + 1]) for t in range(y.shape[0] - 1)])


def _freerun_r2(u: np.ndarray, w: np.ndarray, a: float, b: float, d: int,
                window: tuple[int, int]) -> float:
    """R2 of a pure simulation (no measured feedback) -- the regime the executor uses."""
    lo, hi = window
    wh = 0.0
    pred, truth = [], []
    for t in range(lo, hi):
        wh = a * wh + b * (u[t - d] if t - d >= 0 else 0.0)
        pred.append(wh)
        truth.append(w[t])
    pred, truth = np.asarray(pred), np.asarray(truth)
    sst = float(((truth - truth.mean()) ** 2).sum())
    if sst < 1e-12:
        return float("nan")
    return 1.0 - float(((truth - pred) ** 2).sum()) / sst


def _freerun_series(u: np.ndarray, a: float, b: float, d: int, hi: int) -> np.ndarray:
    out = np.zeros(hi)
    wh = 0.0
    for t in range(hi):
        wh = a * wh + b * (u[t - d] if t - d >= 0 else 0.0)
        out[t] = wh
    return out


def fit_axis(u: np.ndarray, w: np.ndarray, fit_windows: list[tuple[int, int]],
             eval_window: tuple[int, int], sat_aware: bool = False,
             burst_window: tuple[int, int] | None = None) -> dict:
    """ARX(1,1) + integer input delay.

    Two stages.  The integer input delay is chosen by one-step R2 -- that is the
    standard, and unbiased, way to read a transport delay out of an ARX model.  The
    excitation window (chirp only vs chirp + step burst) is then chosen by *free-run*
    R2, because the executor predicts several steps ahead and one-step R2 stays above
    0.98 even for a (tau, gain) pair whose multi-step behaviour is wrong.
    """
    best, cands = None, {}
    for lo, hi in fit_windows:
        cand = None
        for d in range(MAX_DELAY_STEPS + 1):
            idx = np.arange(max(lo + 1, d), hi)
            if idx.size < 12:
                continue
            yv = w[idx]
            sst = float(((yv - yv.mean()) ** 2).sum())
            if sst < 1e-12:
                continue
            X = np.stack([w[idx - 1], u[idx - d]], axis=1)
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            a = float(np.clip(coef[0], 0.0, 0.97))
            ui = u[idx - d]
            denom = float((ui * ui).sum())
            b = float(((yv - a * w[idx - 1]) * ui).sum() / denom) if denom > 1e-12 else float(coef[1])
            r2_one = 1.0 - float(((yv - (a * w[idx - 1] + b * ui)) ** 2).sum()) / sst
            if cand is None or r2_one > cand["r2"]:
                cand = {"delay_steps": d, "a": a, "b": b, "r2": r2_one,
                        "fit_window": [int(lo), int(hi)], "n": int(idx.size),
                        "u_rms": float(np.sqrt((u[idx] ** 2).mean()))}
        if cand is None:
            continue
        cand["r2_freerun"] = float(_freerun_r2(u, w, cand["a"], cand["b"], cand["delay_steps"], eval_window))
        cand["_score"] = cand["r2_freerun"] if np.isfinite(cand["r2_freerun"]) else cand["r2"]
        cands[(lo, hi)] = cand
        if best is None or cand["_score"] > best["_score"]:
            best = cand

    # Saturation guard: a per-step clip both hides itself from a +-3 cm chirp and, if
    # the burst is inside the fit window, biases the identified gain downwards.  Detect
    # it by predicting the burst with the chirp-only model; if the arm delivers much
    # less than predicted, the clip is active -> refit without the burst and hand the
    # observed ceiling to the executor as an achieved-domain bound.
    saturated, clip_obs = False, 1.0
    if sat_aware and burst_window is not None and fit_windows and fit_windows[0] in cands:
        c0 = cands[fit_windows[0]]
        b0, b1 = burst_window
        pred = _freerun_series(u, c0["a"], c0["b"], c0["delay_steps"], b1)
        pred_max = float(np.abs(pred[b0:b1]).max())
        obs_max = float(np.abs(w[b0:b1]).max())
        # only axes the burst actually drives can testify about saturation
        if pred_max > 0.15 and obs_max < 0.75 * pred_max:
            saturated, clip_obs, best = True, obs_max, c0
            best["r2_freerun"] = float(_freerun_r2(u, w, c0["a"], c0["b"],
                                                   c0["delay_steps"], fit_windows[0]))

    if best is None:
        return {"delay_steps": 0, "a": 0.0, "b": 1.0, "r2": float("nan"), "r2_freerun": float("nan"),
                "alpha": 0.0, "tau_s": 0.0, "gain": 1.0, "n": 0, "u_rms": 0.0, "identifiable": False}
    best.pop("_score")
    a = best["a"]
    best["alpha"] = a
    best["tau_s"] = float(-DT / math.log(a)) if a > 1e-6 else 0.0
    best["gain"] = float(np.clip(best["b"] / (1.0 - a) if a < 1.0 else best["b"], 0.2, 3.0))
    best["identifiable"] = bool(best["r2_freerun"] >= R2_GATE)
    best["saturated"] = bool(saturated)
    best["clip_obs"] = float(clip_obs)
    return best


def gripper_t90(grip_open: np.ndarray) -> float:
    """Seconds from the close command to 90 % of the finger travel."""
    g = np.asarray(grip_open, dtype=np.float64)
    base = float(g[max(0, GRIP_CLOSE_STEP - 4): GRIP_CLOSE_STEP + 1].mean())
    tail = g[GRIP_CLOSE_STEP:]
    final = float(tail.min())
    if base - final < 0.2:
        return float("nan")
    thr = base - 0.9 * (base - final)
    hits = np.nonzero(tail <= thr)[0]
    return float(hits[0] * DT) if hits.size else float("nan")


def identify_plant(rec: dict, sat_aware: bool = False) -> dict:
    u, y = np.asarray(rec["u"], dtype=np.float64), np.asarray(rec["y"], dtype=np.float64)
    w = realized_deltas(y)
    axes = []
    for i in range(6):
        if i < 3:
            axes.append(fit_axis(u[:, i], w[:, i], TRANS_FIT_WINDOWS, TRANS_EVAL_WINDOW,
                                 sat_aware=sat_aware, burst_window=BURST_WINDOW))
        else:
            axes.append(fit_axis(u[:, i], w[:, i], [ROT_WINDOW], ROT_WINDOW))
    trans_delays = [axes[i]["delay_steps"] for i in range(3)]
    b0, b1 = BURST_WINDOW
    gain_agg = float(np.median([axes[i]["gain"] for i in range(3)]))
    w_burst = float(np.abs(w[b0:b1, :3]).max())
    return {
        "uid": rec["uid"],
        "pid": rec["pid"],
        "axes": axes,
        "delay_steps_agg": int(round(float(np.median(trans_delays)))),
        "tau_s_agg": float(np.median([axes[i]["tau_s"] for i in range(3)])),
        "gain_agg": gain_agg,
        "r2_min_trans": float(min(axes[i]["r2"] for i in range(3))),
        "r2_min_rot": float(min(axes[i]["r2"] for i in range(3, 6))),
        "r2free_min_trans": float(min(axes[i]["r2_freerun"] for i in range(3))),
        "r2free_min_rot": float(min(axes[i]["r2_freerun"] for i in range(3, 6))),
        "w_max_obs_trans": float(np.abs(w[:, :3]).max()),
        "w_max_burst_trans": float(np.abs(w[b0:b1, :3]).max()),
        "grip_t90_s": gripper_t90(rec["grip_open"]),
        # Main recipe leaves the per-step clip unidentified (a +-3 cm / <=2 Hz chirp
        # never saturates it).  The saturation burst does see it; that estimate is
        # carried separately and only used by the clip-from-burst ablation.
        "clip_pos_norm_hat": float(np.clip(np.median([a["clip_obs"] for a in axes[:3]
                                                      if a.get("saturated")]), 0.05, 1.0))
        if any(a.get("saturated") for a in axes[:3]) else 1.0,
        "clip_rot_norm_hat": 1.0,
        "n_axes_saturated": int(sum(1 for a in axes[:3] if a.get("saturated"))),
        "clip_pos_norm_burst": float(np.clip(w_burst, 0.05, 1.0)),
    }


def load_probe(uid: str, pid: str, probe_dir: Path) -> dict:
    z = np.load(probe_dir / f"probe_{uid}_{pid}.npz")
    return {"uid": uid, "pid": pid, "u": z["u"], "y": z["y"], "grip_open": z["grip_open"],
            "grip_cmd": z["grip_cmd"]}


def identify_all(uid: str, pids: list[str], probe_dir: Path, sat_aware: bool = False) -> dict:
    hats = {}
    for pid in pids:
        hats[pid] = identify_plant(load_probe(uid, pid, probe_dir), sat_aware=sat_aware)
    t90_nom = hats[NOMINAL_PID]["grip_t90_s"] if NOMINAL_PID in hats else float("nan")
    for pid, h in hats.items():
        lead_s = h["grip_t90_s"] - t90_nom
        h["grip_t90_nominal_s"] = t90_nom
        h["grip_lead_s_hat"] = float(lead_s) if np.isfinite(lead_s) else 0.0
        h["grip_lead_steps_hat"] = int(max(0, round(h["grip_lead_s_hat"] / DT))) if np.isfinite(lead_s) else 0
    return hats


def summarize(uid: str, hats: dict) -> dict:
    """theta-hat vs theta.

    ``*_hat`` columns are the raw composite ARX estimate (robot response x injected
    plant) -- that composite is what the executor inverts.  ``*_rel`` columns divide /
    subtract the nominal-plant estimate of the same robot and are the ones comparable
    with the injected ground truth.
    """
    nom = hats[NOMINAL_PID]
    g_nom, tau_nom = nom["gain_agg"], nom["tau_s_agg"]
    rows = []
    for pid, h in hats.items():
        spec = get_plant(pid)
        th = spec.theta()
        rows.append({
            "pid": pid, "family": spec.family,
            "d_true_steps": th["delay_steps"], "d_hat_steps": h["delay_steps_agg"],
            "tau_true_s": th["tau_s"], "tau_hat_s": h["tau_s_agg"],
            "tau_rel_s": float(max(0.0, h["tau_s_agg"] - tau_nom)),
            "g_true": th["gain"], "g_hat": h["gain_agg"],
            "g_rel": float(h["gain_agg"] / g_nom) if g_nom > 1e-9 else float("nan"),
            "grip_lead_true_steps": th["grip_lead_steps"], "grip_lead_hat_steps": h["grip_lead_steps_hat"],
            "r2_min_trans": h["r2_min_trans"], "r2_min_rot": h["r2_min_rot"],
            "r2free_min_trans": h["r2free_min_trans"], "r2free_min_rot": h["r2free_min_rot"],
            "w_max_obs_trans": h["w_max_obs_trans"], "w_max_burst_trans": h["w_max_burst_trans"],
            "clip_pos_true_m": th["clip_pos_m"],
        })
    arr = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in
           ("d_true_steps", "d_hat_steps", "tau_true_s", "tau_hat_s", "tau_rel_s",
            "g_true", "g_hat", "g_rel", "grip_lead_true_steps", "grip_lead_hat_steps")}

    def corr(a, b):
        if np.std(arr[a]) < 1e-12 or np.std(arr[b]) < 1e-12:
            return float("nan")
        return float(np.corrcoef(arr[a], arr[b])[0, 1])

    return {
        "uid": uid,
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
            "gain_hat_on_nominal": float(next(r["g_hat"] for r in rows if r["pid"] == NOMINAL_PID)),
            "tau_hat_on_nominal_s": float(next(r["tau_hat_s"] for r in rows if r["pid"] == NOMINAL_PID)),
            "grip_lead_corr": corr("grip_lead_true_steps", "grip_lead_hat_steps"),
            "grip_lead_mae_steps": float(np.abs(arr["grip_lead_true_steps"] - arr["grip_lead_hat_steps"]).mean()),
            "n_axes_below_r2_gate": int(sum(1 for _, h in hats.items() for a in h["axes"] if not a["identifiable"])),
            "n_axes_total": int(6 * len(rows)),
            "r2_gate": R2_GATE,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default="panda")
    ap.add_argument("--probe-dir", default=str(DATA / "probes"))
    ap.add_argument("--out-dir", default=str(DATA / "theta"))
    ap.add_argument("--sat-aware", action="store_true")
    args = ap.parse_args()

    probe_dir = Path(args.probe_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pids = [p for p in plant_ids() if (probe_dir / f"probe_{args.uid}_{p}.npz").exists()]
    hats = identify_all(args.uid, pids, probe_dir, sat_aware=args.sat_aware)
    (out_dir / f"theta_hat_{args.uid}.json").write_text(
        json.dumps(hats, indent=2, default=json_default) + "\n")
    summary = summarize(args.uid, hats)
    (out_dir / f"identify_{args.uid}.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n")
    print(json.dumps({"uid": args.uid, "n_plants": len(pids),
                      "agreement": summary["agreement"]}, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
