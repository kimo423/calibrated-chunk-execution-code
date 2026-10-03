#!/usr/bin/env python
"""Merge shard outputs into the three small result JSONs under cce/results/."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np

from cce.common import json_default
from cce.plants import NOMINAL_PID, PLANTS, get_plant

ROOT = Path("/opt/cce")
DATA = ROOT / "data" / "cce"
RESULTS = ROOT / "cce" / "results"
LAG_FAMILIES = ("D1", "D2t", "MIX", "OVER")


def _load_parts(pattern: str) -> list[dict]:
    rows = []
    for p in sorted((DATA / "parts").glob(pattern)):
        rows.extend(json.loads(p.read_text()))
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def paired(a: list, b: list, n_boot: int = 10000, seed: int = 0) -> dict:
    """Plant-level paired contrast: mean difference, paired t, bootstrap 95 % CI."""
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return {"n": len(pairs)}
    d = np.array([x - y for x, y in pairs], dtype=np.float64)
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / np.sqrt(d.size))) if sd > 1e-12 else float("inf")
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return {"n": int(d.size), "mean_diff": float(d.mean()), "sd": sd, "t": t,
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_plants_better": int((d > 0).sum()), "n_plants_worse": int((d < 0).sum())}


def _ablation(pattern: str, uid: str, key: str) -> dict:
    rows = [r for r in _load_parts(pattern) if r["uid"] == uid]
    out = {}
    for r in rows:
        out.setdefault(r["pid"], []).append(r[key])
    return {pid: _mean(v) for pid, v in out.items()}


# --------------------------------------------------------------------- replay
def merge_replay(uids: list[str]) -> dict:
    rows = _load_parts("replay_*.json")
    out = {"n_rows": len(rows), "budget_steps": 200, "by_uid": {}}
    for uid in uids:
        ru = [r for r in rows if r["uid"] == uid]
        if not ru:
            continue
        methods = sorted({r["method"] for r in ru})
        pids = [p.pid for p in PLANTS if any(r["pid"] == p.pid for r in ru)]
        cell = {}
        for pid in pids:
            for m in methods:
                sel = [r for r in ru if r["pid"] == pid and r["method"] == m]
                if not sel:
                    continue
                cell[(pid, m)] = {
                    "n": len(sel),
                    "sr": float(np.mean([r["success"] for r in sel])),
                    "mean_steps": float(np.mean([r["steps"] for r in sel])),
                    "tts_s": _mean([r["time_to_success_s"] for r in sel]),
                }
        sr_nom = cell.get((NOMINAL_PID, "naive"), {}).get("sr", float("nan"))
        plants = []
        for pid in pids:
            spec = get_plant(pid)
            row = {"pid": pid, "family": spec.family, **{f"sr_{m}": cell[(pid, m)]["sr"]
                                                         for m in methods if (pid, m) in cell}}
            row["n"] = cell[(pid, methods[0])]["n"] if (pid, methods[0]) in cell else 0
            base = cell.get((pid, "naive"), {}).get("sr", float("nan"))
            base_open = cell.get((pid, "naive_open"), {}).get("sr", float("nan"))
            base_best = float(np.nanmax([base, base_open]))
            row["sr_naive"] = base
            row["sr_naive_best"] = base_best
            denom, denom_b = sr_nom - base, sr_nom - base_best
            for m in methods:
                if (pid, m) not in cell:
                    continue
                row[f"tts_{m}"] = cell[(pid, m)]["tts_s"]
                row[f"R_{m}"] = float((cell[(pid, m)]["sr"] - base) / denom) if denom > 0.10 else None
                row[f"Rb_{m}"] = float((cell[(pid, m)]["sr"] - base_best) / denom_b) if denom_b > 0.10 else None
            plants.append(row)

        def macro(key, subset=None):
            sel = [p for p in plants if p["pid"] != NOMINAL_PID and
                   (subset is None or p["family"] in subset)]
            return _mean([p.get(key) for p in sel])

        out["by_uid"][uid] = {
            "sr_nominal_naive": sr_nom,
            "methods": methods,
            "plants": plants,
            "macro_sr": {m: macro(f"sr_{m}") for m in methods},
            "macro_R": {m: macro(f"R_{m}") for m in methods},
            "macro_R_lag_only": {m: macro(f"R_{m}", LAG_FAMILIES) for m in methods},
            "macro_R_vs_best_naive": {m: macro(f"Rb_{m}") for m in methods},
            "n_plants_R_defined": int(sum(1 for p in plants if p["pid"] != NOMINAL_PID and p.get("R_ours_p") is not None)),
            "n_plants_Rb_defined": int(sum(1 for p in plants if p["pid"] != NOMINAL_PID and p.get("Rb_ours_p") is not None)),
            "macro_tts_s": {m: macro(f"tts_{m}") for m in methods},
            "ablation_clip_from_burst_sr": _ablation("replayclip_*.json", uid, "success"),
            "ablation_sat_aware_sr": _ablation("replaysat_*.json", uid, "success"),
            "paired": {
                "ours_p_minus_naive": paired([p.get("sr_ours_p") for p in plants if p["pid"] != NOMINAL_PID],
                                             [p.get("sr_naive") for p in plants if p["pid"] != NOMINAL_PID]),
                "ours_p_minus_best_naive": paired([p.get("sr_ours_p") for p in plants if p["pid"] != NOMINAL_PID],
                                                  [p.get("sr_naive_best") for p in plants if p["pid"] != NOMINAL_PID]),
                "ours_p_minus_slow20": paired([p.get("sr_ours_p") for p in plants if p["pid"] != NOMINAL_PID],
                                              [p.get("sr_slow20") for p in plants if p["pid"] != NOMINAL_PID]),
                "ours_p_minus_slow15": paired([p.get("sr_ours_p") for p in plants if p["pid"] != NOMINAL_PID],
                                              [p.get("sr_slow15") for p in plants if p["pid"] != NOMINAL_PID]),
                "oracle_minus_ours_p": paired([p.get("sr_oracle") for p in plants if p["pid"] != NOMINAL_PID],
                                              [p.get("sr_ours_p") for p in plants if p["pid"] != NOMINAL_PID]),
            },
            "gates": {
                "G_nominal_naive_sr_ge_0p9": bool(sr_nom >= 0.9),
                "sr_naive_D1_d150": cell.get(("D1_d150", "naive"), {}).get("sr"),
                "sr_naive_D2_t250": cell.get(("D2_t250", "naive"), {}).get("sr"),
                "n_plants_with_naive_drop_ge_25pp": int(sum(
                    1 for p in plants if p["pid"] != NOMINAL_PID and sr_nom - p["sr_naive"] >= 0.25)),
                "K2_macro_R_ours_p_ge_0p5": bool(macro("R_ours_p") >= 0.5)
                if np.isfinite(macro("R_ours_p")) else None,
            },
        }
    return out


# ------------------------------------------------------------------- tracking
def merge_tracking(uids: list[str]) -> dict:
    rows = _load_parts("tracking_*.json")
    out = {"n_rows": len(rows), "by_uid": {}}
    for uid in uids:
        ru = [r for r in rows if r["uid"] == uid]
        if not ru:
            continue
        methods = sorted({r["method"] for r in ru})
        pids = [p.pid for p in PLANTS if any(r["pid"] == p.pid for r in ru)]
        cell = {}
        for pid in pids:
            for m in methods:
                sel = [r for r in ru if r["pid"] == pid and r["method"] == m]
                if not sel:
                    continue
                cell[(pid, m)] = {
                    "n": len(sel),
                    "rmse_pos_m": _mean([r["rmse_pos_m"] for r in sel]),
                    "rmse_rot_rad": _mean([r["rmse_rot_rad"] for r in sel]),
                    "rmse_pos_vs_intent_m": _mean([r["rmse_pos_vs_intent_m"] for r in sel]),
                    "max_pos_err_m": _mean([r["max_pos_err_m"] for r in sel]),
                    "grip_latency_s": _mean([r["grip_latency_s"] for r in sel]),
                    "duration_s": _mean([r["duration_s"] for r in sel]),
                }
        lat_ref = cell.get((NOMINAL_PID, "naive"), {}).get("grip_latency_s", float("nan"))
        plants = []
        for pid in pids:
            spec = get_plant(pid)
            base = cell.get((pid, "naive"), {}).get("rmse_pos_m", float("nan"))
            row = {"pid": pid, "family": spec.family, "n": cell[(pid, methods[0])]["n"]}
            for m in methods:
                if (pid, m) not in cell:
                    continue
                c = cell[(pid, m)]
                row[f"rmse_{m}"] = c["rmse_pos_m"]
                row[f"rmserot_{m}"] = c["rmse_rot_rad"]
                row[f"rmseintent_{m}"] = c["rmse_pos_vs_intent_m"]
                row[f"gripphase_{m}"] = float(c["grip_latency_s"] - lat_ref)
                row[f"dur_{m}"] = c["duration_s"]
                row[f"red_{m}"] = float((base - c["rmse_pos_m"]) / base) if base > 1e-9 else None
            plants.append(row)

        def macro(key, subset=None):
            sel = [p for p in plants if p["pid"] != NOMINAL_PID and
                   (subset is None or p["family"] in subset)]
            return _mean([p.get(key) for p in sel])

        out["by_uid"][uid] = {
            "methods": methods,
            "nominal_naive_grip_latency_s": lat_ref,
            "plants": plants,
            "macro_rmse": {m: macro(f"rmse_{m}") for m in methods},
            "macro_rmse_rot": {m: macro(f"rmserot_{m}") for m in methods},
            "macro_reduction_vs_naive": {m: macro(f"red_{m}") for m in methods},
            "macro_reduction_lag_only": {m: macro(f"red_{m}", LAG_FAMILIES) for m in methods},
            "macro_grip_phase_err_s": {m: macro(f"gripphase_{m}") for m in methods},
            "ablation_clip_from_burst_rmse": _ablation("trackingclip_*.json", uid, "rmse_pos_m"),
            "ablation_sat_aware_rmse": _ablation("trackingsat_*.json", uid, "rmse_pos_m"),
            "ablation_qp_rmse": _ablation("trackingqp_*.json", uid, "rmse_pos_m"),
            "paired": {
                "naive_minus_ours_p_rmse": paired([p.get("rmse_naive") for p in plants if p["pid"] != NOMINAL_PID],
                                                  [p.get("rmse_ours_p") for p in plants if p["pid"] != NOMINAL_PID]),
                "slow20_minus_ours_p_rmse": paired([p.get("rmse_slow20") for p in plants if p["pid"] != NOMINAL_PID],
                                                   [p.get("rmse_ours_p") for p in plants if p["pid"] != NOMINAL_PID]),
                "ours_p_minus_oracle_rmse": paired([p.get("rmse_ours_p") for p in plants if p["pid"] != NOMINAL_PID],
                                                   [p.get("rmse_oracle") for p in plants if p["pid"] != NOMINAL_PID]),
            },
            "gates": {
                "K1_lag_rmse_reduction_ge_30pct": bool(macro("red_ours_p", LAG_FAMILIES) >= 0.30)
                if np.isfinite(macro("red_ours_p", LAG_FAMILIES)) else None,
                "reduction_ours_p_lag": macro("red_ours_p", LAG_FAMILIES),
                "reduction_oracle_lag": macro("red_oracle", LAG_FAMILIES),
            },
        }
    return out


# ------------------------------------------------------------------- identify
def merge_identify(uids: list[str]) -> dict:
    out = {"plant_family": [p.as_dict() for p in PLANTS], "by_uid": {}}
    for uid in uids:
        f = DATA / "theta" / f"identify_{uid}.json"
        if f.exists():
            out["by_uid"][uid] = json.loads(f.read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", default="panda,xarm6_robotiq")
    args = ap.parse_args()
    uids = [u for u in args.uids.split(",") if u]
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, obj in (("identify_summary", merge_identify(uids)),
                      ("tracking_summary", merge_tracking(uids)),
                      ("replay_summary", merge_replay(uids))):
        p = RESULTS / f"{name}.json"
        p.write_text(json.dumps(obj, indent=1, default=json_default) + "\n")
        print(f"{p}  {p.stat().st_size/1024:.1f} KiB", flush=True)


if __name__ == "__main__":
    main()
