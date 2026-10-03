#!/usr/bin/env python
"""Aggregation and pre-registered statistics for the LIBERO CCE plant family.

Reads the shard JSONLs written by ``cce/libero_family.py`` and produces the
pre-registered read-outs:

* per (plant, arm, task) cells and the 10-task-equally-weighted macro success rate;
* G1: the naive arm's drop relative to the nominal plant;
* P1 = ours-P - naive and P2 = ours-P - strongest Slow-down, both paired at the
  plant level (paired t plus a plant-level bootstrap CI), Holm-corrected across the
  two primary comparisons;
* recovery rate R = (SR_calib - SR_naive) / (SR_nominal - SR_naive);
* K1 (tracking RMSE cut by >= 30 % on delay/lag plants), K2 (family recovery rate
  >= 0.3), K3 (ours-P vs the strongest general baseline: >= 5 pp macro or >= 25 %
  shorter time to success).

Tracking RMSE is reported in both pre-registered columns: against the policy chunk
(``tracking_rmse_m``, the cross-arm comparable one) and against each arm's own set
point (``tracking_rmse_vs_ref_m``).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from cce.common import DT, json_default
from cce.libero_plant import NOMINAL_PID

N_BOOT = 10000
BOOT_SEED = 20260906
LAG_FAMILIES = ("D1", "D2t", "MIX")  # plants carrying delay and/or lag


def load(paths) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def macro_sr(rows, pid, method, tasks) -> float | None:
    per = []
    for tid in tasks:
        xs = [r["success"] for r in rows if r["pid"] == pid and r["method"] == method
              and r["task_id"] == tid]
        if xs:
            per.append(float(np.mean(xs)))
    return float(np.mean(per)) if per else None


def cell(rows, pid, method, tasks) -> dict | None:
    xs = [r for r in rows if r["pid"] == pid and r["method"] == method]
    if not xs:
        return None
    ok = [r for r in xs if r["success"]]
    return {
        "n": len(xs),
        "n_tasks": len({r["task_id"] for r in xs}),
        "macro_sr": macro_sr(rows, pid, method, tasks),
        "pooled_sr": round(float(np.mean([r["success"] for r in xs])), 4),
        "mean_steps": round(float(np.mean([r["steps"] for r in xs])), 1),
        "mean_time_to_success_s": (round(float(np.mean([r["steps"] * DT for r in ok])), 3)
                                   if ok else None),
        "tracking_rmse_m": round(_mean([r["tracking_rmse_m"] for r in xs]), 5),
        "tracking_rmse_vs_ref_m": round(_mean([r["tracking_rmse_vs_ref_m"] for r in xs]), 5),
        "mean_grip_close_lag_steps": (round(_mean([r["grip_close_lag_steps"] for r in xs]), 2)
                                      if any(r["grip_close_lag_steps"] is not None for r in xs)
                                      else None),
        "mean_n_grip_cmd_changes": round(_mean([r["n_grip_cmd_changes"] for r in xs]), 2),
        "u_pos_max_norm": round(max(r["u_pos_max_norm"] for r in xs), 3),
        "u_rot_max_norm": round(max(r["u_rot_max_norm"] for r in xs), 3),
        "n_bound_hit_episodes": int(sum(1 for r in xs if r.get("n_bound_hits", 0) > 0)),
        "max_codec_err_m": max(r["codec_err_m"] for r in xs),
        "wall_s": round(float(np.sum([r["wall_s"] for r in xs])), 1),
    }


def paired_test(diffs: np.ndarray, rng) -> dict:
    d = np.asarray(diffs, dtype=np.float64)
    n = d.size
    mean = float(d.mean())
    if n < 2:
        return {"n": n, "mean": mean, "t": None, "p": None, "ci95": [None, None],
                "n_positive": int((d > 0).sum())}
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    t = float(mean / se) if se > 0 else (float("inf") if mean > 0 else 0.0)
    try:
        from scipy import stats

        p = float(2.0 * stats.t.sf(abs(t), df=n - 1)) if se > 0 else (0.0 if mean != 0 else 1.0)
    except Exception:
        p = float(2.0 * 0.5 * math.erfc(abs(t) / math.sqrt(2.0))) if se > 0 else 1.0
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return {
        "n": n, "mean": mean, "sd": sd, "t": t, "p": p,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n_positive": int((d > 0).sum()),
    }


def holm(pvals: dict) -> dict:
    items = sorted(((k, v) for k, v in pvals.items() if v is not None), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    for k, v in pvals.items():
        if v is None:
            out[k] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser("aggregate the LIBERO CCE plant family")
    ap.add_argument("--glob", nargs="+", required=True, help="shard JSONL paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="libero_family_v1")
    ap.add_argument("--theta-summary", default=None)
    ap.add_argument("--family-spec", default=None)
    args = ap.parse_args()

    import glob as _glob

    paths = []
    for g in args.glob:
        hits = sorted(_glob.glob(g))
        paths.extend(Path(h) for h in hits) if hits else paths.append(Path(g))
    rows = load(paths)
    if not rows:
        raise SystemExit("no rows")

    pids = sorted({r["pid"] for r in rows}, key=lambda p: (p != NOMINAL_PID, p))
    methods = [m for m in ("naive", "slow15", "slow20", "oracle", "ours_p")
               if any(r["method"] == m for r in rows)]
    tasks = sorted({r["task_id"] for r in rows})
    fam = {r["pid"]: r["family"] for r in rows}

    cells = {pid: {m: cell(rows, pid, m, tasks) for m in methods} for pid in pids}

    # ---- G1 ---------------------------------------------------------------- #
    nom_naive = (cells.get(NOMINAL_PID, {}).get("naive") or {}).get("macro_sr")
    g1 = {}
    for pid in pids:
        if pid == NOMINAL_PID or nom_naive is None:
            continue
        c = cells[pid].get("naive")
        if c and c["macro_sr"] is not None:
            drop = 100.0 * (nom_naive - c["macro_sr"])
            g1[pid] = {"family": fam[pid], "naive_macro_sr": round(c["macro_sr"], 4),
                       "drop_pp": round(drop, 1), "meets_G1_25pp": bool(drop >= 25.0)}
    g1_summary = {
        "nominal_naive_macro_sr": nom_naive,
        "n_plants_meeting_G1": int(sum(1 for v in g1.values() if v["meets_G1_25pp"])),
        "n_plants_tested": len(g1),
        "max_drop_pp": max((v["drop_pp"] for v in g1.values()), default=None),
        "G1_passed": bool(any(v["meets_G1_25pp"] for v in g1.values())),
    }

    # ---- primary comparisons ---------------------------------------------- #
    rng = np.random.default_rng(BOOT_SEED)
    primary, pvals = {}, {}
    best_slow = None
    slow_arms = [m for m in ("slow15", "slow20") if m in methods]
    if slow_arms:
        best_slow = max(slow_arms, key=lambda m: np.mean(
            [cells[p][m]["macro_sr"] for p in pids if cells[p].get(m)]))

    def paired(a, b, key="macro_sr"):
        ps = [p for p in pids if cells[p].get(a) and cells[p].get(b)
              and cells[p][a][key] is not None and cells[p][b][key] is not None]
        d = np.array([cells[p][a][key] - cells[p][b][key] for p in ps], dtype=np.float64)
        r = paired_test(d, rng)
        r["plants"] = ps
        r["per_plant"] = {p: round(float(x), 4) for p, x in zip(ps, d)}
        return r

    if "ours_p" in methods and "naive" in methods:
        primary["P1_ours_p_minus_naive"] = paired("ours_p", "naive")
        pvals["P1_ours_p_minus_naive"] = primary["P1_ours_p_minus_naive"]["p"]
    if "ours_p" in methods and best_slow:
        primary[f"P2_ours_p_minus_{best_slow}"] = paired("ours_p", best_slow)
        pvals[f"P2_ours_p_minus_{best_slow}"] = primary[f"P2_ours_p_minus_{best_slow}"]["p"]
    holm_p = holm(pvals) if pvals else {}

    # non-nominal subset (secondary view)
    secondary = {}
    nn = [p for p in pids if p != NOMINAL_PID]
    for name, (a, b) in {"P1": ("ours_p", "naive"),
                         "P2": ("ours_p", best_slow or "slow20")}.items():
        if a in methods and b in methods:
            ps = [p for p in nn if cells[p].get(a) and cells[p].get(b)]
            d = np.array([cells[p][a]["macro_sr"] - cells[p][b]["macro_sr"] for p in ps])
            if d.size:
                secondary[f"{name}_excl_nominal_{a}_minus_{b}"] = paired_test(d, rng)

    # ---- recovery rate ----------------------------------------------------- #
    recovery = {}
    if nom_naive is not None:
        for pid in nn:
            base = cells[pid].get("naive")
            if not base or base["macro_sr"] is None:
                continue
            den = nom_naive - base["macro_sr"]
            row = {"family": fam[pid], "sr_naive": round(base["macro_sr"], 4),
                   "denominator": round(den, 4), "counted": bool(den > 0.10)}
            for m in methods:
                if m in ("naive",):
                    continue
                c = cells[pid].get(m)
                if c and c["macro_sr"] is not None and den > 1e-9:
                    row[f"R_{m}"] = round(float((c["macro_sr"] - base["macro_sr"]) / den), 3)
            recovery[pid] = row
    counted = [p for p, v in recovery.items() if v["counted"]]
    recovery_macro = {
        "n_counted_plants": len(counted),
        "criterion": "plants with SR_nominal - SR_naive > 0.10",
    }
    for m in methods:
        if m == "naive":
            continue
        vals = [recovery[p][f"R_{m}"] for p in counted if f"R_{m}" in recovery[p]]
        recovery_macro[f"R_{m}"] = round(float(np.mean(vals)), 3) if vals else None

    # ---- K1 ---------------------------------------------------------------- #
    k1 = {}
    for pid in pids:
        base = cells[pid].get("naive")
        if not base or not base["tracking_rmse_m"]:
            continue
        row = {"family": fam[pid], "naive_tracking_rmse_m": base["tracking_rmse_m"]}
        for m in methods:
            if m == "naive":
                continue
            c = cells[pid].get(m)
            if c and c["tracking_rmse_m"]:
                row[f"{m}_tracking_rmse_m"] = c["tracking_rmse_m"]
                row[f"{m}_reduction_pct"] = round(
                    100.0 * (1.0 - c["tracking_rmse_m"] / base["tracking_rmse_m"]), 1)
        k1[pid] = row
    lag_pids = [p for p in nn if fam[p] in LAG_FAMILIES]
    k1_summary = {
        "lag_delay_plants": lag_pids,
        "mean_ours_p_reduction_pct": (round(float(np.mean(
            [k1[p]["ours_p_reduction_pct"] for p in lag_pids
             if p in k1 and "ours_p_reduction_pct" in k1[p]])), 1) if lag_pids else None),
        "n_plants_reduction_ge_30": int(sum(1 for p in lag_pids if p in k1
                                            and k1[p].get("ours_p_reduction_pct", -1) >= 30.0)),
    }
    if k1_summary["mean_ours_p_reduction_pct"] is not None:
        k1_summary["K1_passed"] = bool(k1_summary["mean_ours_p_reduction_pct"] >= 30.0)

    # ---- K2 / K3 ----------------------------------------------------------- #
    kills = {}
    if recovery_macro.get("R_ours_p") is not None:
        kills["K2_family_recovery_lt_0.3"] = bool(recovery_macro["R_ours_p"] < 0.3)
        kills["K2_R_ours_p"] = recovery_macro["R_ours_p"]
    if best_slow and "ours_p" in methods:
        key = f"P2_ours_p_minus_{best_slow}"
        macro_gap_pp = 100.0 * primary[key]["mean"] if key in primary else None
        t_ours = _mean([cells[p]["ours_p"]["mean_time_to_success_s"] for p in pids
                        if cells[p].get("ours_p")])
        t_slow = _mean([cells[p][best_slow]["mean_time_to_success_s"] for p in pids
                        if cells[p].get(best_slow)])
        t_gain = (100.0 * (1.0 - t_ours / t_slow)) if (t_ours and t_slow) else None
        kills["K3_macro_gap_pp"] = round(macro_gap_pp, 2) if macro_gap_pp is not None else None
        kills["K3_time_to_success_ours_p_s"] = round(t_ours, 3) if t_ours else None
        kills[f"K3_time_to_success_{best_slow}_s"] = round(t_slow, 3) if t_slow else None
        kills["K3_time_reduction_pct"] = round(t_gain, 1) if t_gain is not None else None
        if macro_gap_pp is not None:
            kills["K3_triggered"] = bool(macro_gap_pp < 5.0 and (t_gain is None or t_gain < 25.0))

    macro = {m: {"macro_sr_family_mean": round(float(np.mean(
        [cells[p][m]["macro_sr"] for p in pids if cells[p].get(m)])), 4),
        "macro_sr_family_mean_excl_nominal": (round(float(np.mean(
            [cells[p][m]["macro_sr"] for p in nn if cells[p].get(m)])), 4) if nn else None)}
        for m in methods}

    out = {
        "label": args.label,
        "policy": "X-VLA-Pt + 2toINF/X-VLA-libero-spatial-peft (LoRA merged)",
        "suite": rows[0]["suite"],
        "n_episodes": len(rows),
        "plants": pids,
        "methods": methods,
        "tasks": tasks,
        "protocol": {
            "control_hz": round(1.0 / DT, 1),
            "chunk": 30, "chunk_seconds": 30 * DT,
            "budget_steps": 800, "denoise_steps": 10, "domain_id": 3,
            "act_type": "abs (robot.controller.use_delta = False)",
            "chunk_execution": "drained, re-queried when exhausted",
            "episodes_per_cell": max(len({r["ep"] for r in rows if r["pid"] == p
                                          and r["method"] == m}) for p in pids for m in methods),
            "macro_sr": "10 libero_spatial tasks, equally weighted",
            "executor_bound_pos_norm": rows[0].get("bound_pos"),
            "executor_bound_rot_norm": rows[0].get("bound_rot"),
            "paired_unit": "plant",
            "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "resampled_unit": "plant"},
        },
        "macro": macro,
        "cells": cells,
        "G1_naive_drop_vs_nominal": g1,
        "G1_summary": g1_summary,
        "primary": primary,
        "primary_holm_p": holm_p,
        "secondary_excl_nominal": secondary,
        "recovery": recovery,
        "recovery_macro": recovery_macro,
        "K1_tracking_rmse_vs_intent": k1,
        "K1_summary": k1_summary,
        "kills": kills,
        "best_slow_arm": best_slow,
        "wall_s_total": round(float(np.sum([r["wall_s"] for r in rows])), 1),
    }
    if args.theta_summary and Path(args.theta_summary).exists():
        th = json.loads(Path(args.theta_summary).read_text())
        out["identification"] = {"agreement": th["agreement"], "rows": th["rows"],
                                 "probe_design": th.get("probe_design")}
    if args.family_spec and Path(args.family_spec).exists():
        out["family_spec"] = json.loads(Path(args.family_spec).read_text())

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=json_default) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("label", "n_episodes", "macro", "G1_summary", "primary_holm_p",
                               "recovery_macro", "K1_summary", "kills")},
                     indent=2, default=json_default))
    print(f"[agg] wrote {p}")


if __name__ == "__main__":
    main()
