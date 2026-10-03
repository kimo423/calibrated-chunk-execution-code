#!/usr/bin/env python
"""Aggregation and pre-registered statistics for the second pass (prereg v3).

Same read-outs as ``cce/libero_aggregate.py`` -- which is left untouched, so the
first pass still reproduces byte for byte -- with the four changes prereg v3 makes:

1. the P1 / P2 plant-level population is the **12 test plants**; the two
   development plants are aggregated and reported separately and never enter the
   paired statistics;
2. time to success is paired on **plant x task cells where both compared arms have
   at least one success**, instead of comparing each arm's own success subset;
3. the K1 plant set is stated as a parameter predicate (delay and/or lag injected)
   rather than a family-label list, so the new mixed plant resolves without a
   judgement call.  On the first-pass family the two definitions coincide;
4. saturation is reported both as the share of episodes in which the executor
   command bound binds at least once and as the share of control steps at the
   bound, side by side with the first pass.

``--v1-json`` / ``--v1-glob`` are optional; when given, the output carries the
first-pass numbers next to the second-pass ones so the report can print both.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob as _glob
import json
import math
from pathlib import Path

import numpy as np

from cce.common import DT, json_default
from cce.libero_plant import NOMINAL_PID
from cce.libero_plant_v2 import DEV_PIDS, FAMILY_V2, K1_PIDS, NEW_V2, TEST_PIDS

N_BOOT = 10000
BOOT_SEED = 20260906
ARMS = ("naive", "slow15", "slow20", "oracle", "ours_p")


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def load(patterns) -> list[dict]:
    rows = []
    for g in patterns:
        hits = sorted(_glob.glob(g)) or [g]
        for h in hits:
            p = Path(h)
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


# --------------------------------------------------------------------------- #
# cells
# --------------------------------------------------------------------------- #
def macro_sr(rows, tasks) -> float | None:
    per = []
    for tid in tasks:
        xs = [r["success"] for r in rows if r["task_id"] == tid]
        if xs:
            per.append(float(np.mean(xs)))
    return float(np.mean(per)) if per else None


def cell(rows, tasks) -> dict | None:
    if not rows:
        return None
    ok = [r for r in rows if r["success"]]
    hits = float(np.sum([r.get("n_bound_hits", 0) for r in rows]))
    steps = float(np.sum([r["steps"] for r in rows]))
    return {
        "n": len(rows),
        "n_tasks": len({r["task_id"] for r in rows}),
        "macro_sr": macro_sr(rows, tasks),
        "pooled_sr": round(float(np.mean([r["success"] for r in rows])), 4),
        "mean_steps": round(float(np.mean([r["steps"] for r in rows])), 1),
        "mean_time_to_success_s": (round(float(np.mean([r["steps"] * DT for r in ok])), 3)
                                   if ok else None),
        "tracking_rmse_m": round(_mean([r["tracking_rmse_m"] for r in rows]), 5),
        "tracking_rmse_vs_ref_m": round(_mean([r["tracking_rmse_vs_ref_m"] for r in rows]), 5),
        "mean_grip_close_lag_steps": (round(_mean([r["grip_close_lag_steps"] for r in rows]), 2)
                                      if any(r["grip_close_lag_steps"] is not None for r in rows)
                                      else None),
        "u_pos_max_norm": round(max(r["u_pos_max_norm"] for r in rows), 3),
        "u_rot_max_norm": round(max(r["u_rot_max_norm"] for r in rows), 3),
        "n_bound_hit_episodes": int(sum(1 for r in rows if r.get("n_bound_hits", 0) > 0)),
        "sat_episode_share": round(float(np.mean([1.0 if r.get("n_bound_hits", 0) > 0 else 0.0
                                                  for r in rows])), 4),
        "sat_step_share": round(hits / steps, 5) if steps else None,
        "max_codec_err_m": max(r["codec_err_m"] for r in rows),
        "wall_s": round(float(np.sum([r["wall_s"] for r in rows])), 1),
    }


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def paired_test(diffs, rng) -> dict:
    d = np.asarray(diffs, dtype=np.float64)
    n = d.size
    mean = float(d.mean()) if n else float("nan")
    if n < 2:
        return {"n": int(n), "mean": mean, "t": None, "p": None, "ci95": [None, None],
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
    return {"n": int(n), "mean": mean, "sd": sd, "t": t, "p": p,
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_positive": int((d > 0).sum())}


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


def common_success_time(rows, pids, a, b, tasks) -> dict:
    """Time to success paired on plant x task cells where both arms succeed at least once."""
    per_plant, cell_rows = {}, []
    for pid in pids:
        pa, pb, tids = [], [], []
        for tid in tasks:
            oa = [r["steps"] * DT for r in rows
                  if r["pid"] == pid and r["method"] == a and r["task_id"] == tid and r["success"]]
            ob = [r["steps"] * DT for r in rows
                  if r["pid"] == pid and r["method"] == b and r["task_id"] == tid and r["success"]]
            if oa and ob:
                ma, mb = float(np.mean(oa)), float(np.mean(ob))
                pa.append(ma)
                pb.append(mb)
                tids.append(tid)
                cell_rows.append({"pid": pid, "task_id": tid, f"{a}_s": round(ma, 3),
                                  f"{b}_s": round(mb, 3), "diff_s": round(ma - mb, 3),
                                  "n_succ_a": len(oa), "n_succ_b": len(ob)})
        if tids:
            per_plant[pid] = {f"{a}_s": round(float(np.mean(pa)), 3),
                              f"{b}_s": round(float(np.mean(pb)), 3),
                              "diff_s": round(float(np.mean(pa)) - float(np.mean(pb)), 3),
                              "n_common_cells": len(tids), "tasks": tids}
    return {"per_plant": per_plant, "cells": cell_rows}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser("aggregate the LIBERO CCE plant family, second pass")
    ap.add_argument("--glob", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="libero_family_v2")
    ap.add_argument("--theta-summary", default=None)
    ap.add_argument("--family-spec", default=None)
    ap.add_argument("--devplant-json", default=None)
    ap.add_argument("--v1-json", default=None)
    ap.add_argument("--v1-glob", nargs="*", default=None)
    args = ap.parse_args()

    rows = load(args.glob)
    if not rows:
        raise SystemExit("no rows")

    order = [p.pid for p in FAMILY_V2]
    pids = [p for p in order if any(r["pid"] == p for r in rows)]
    methods = [m for m in ARMS if any(r["method"] == m for r in rows)]
    tasks = sorted({r["task_id"] for r in rows})
    fam = {r["pid"]: r["family"] for r in rows}
    dev = [p for p in pids if p in DEV_PIDS]
    test = [p for p in pids if p in TEST_PIDS]

    cells = {pid: {m: cell([r for r in rows if r["pid"] == pid and r["method"] == m], tasks)
                   for m in methods} for pid in pids}

    def mean_over(ps, m, key="macro_sr"):
        vals = [cells[p][m][key] for p in ps if cells[p].get(m) and cells[p][m][key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    macro = {m: {"macro_sr_test_mean": mean_over(test, m),
                 "macro_sr_family_mean": mean_over(pids, m),
                 "macro_sr_dev_mean": mean_over(dev, m),
                 "tracking_rmse_test_mean_m": mean_over(test, m, "tracking_rmse_m"),
                 "sat_episode_share_all": round(float(np.mean(
                     [1.0 if r.get("n_bound_hits", 0) > 0 else 0.0
                      for r in rows if r["method"] == m])), 4),
                 "sat_step_share_all": round(float(np.sum(
                     [r.get("n_bound_hits", 0) for r in rows if r["method"] == m])
                     / max(1.0, float(np.sum([r["steps"] for r in rows if r["method"] == m])))), 5)}
             for m in methods}

    # ---- G1 (naive drop vs the nominal plant) ------------------------------ #
    nom_naive = (cells.get(NOMINAL_PID, {}).get("naive") or {}).get("macro_sr")
    g1 = {}
    for pid in pids:
        if pid == NOMINAL_PID or nom_naive is None:
            continue
        c = cells[pid].get("naive")
        if c and c["macro_sr"] is not None:
            drop = 100.0 * (nom_naive - c["macro_sr"])
            g1[pid] = {"family": fam[pid], "role": "dev" if pid in DEV_PIDS else "test",
                       "naive_macro_sr": round(c["macro_sr"], 4), "drop_pp": round(drop, 1),
                       "meets_G1_25pp": bool(drop >= 25.0)}

    # ---- primary comparisons on the test plants ---------------------------- #
    rng = np.random.default_rng(BOOT_SEED)
    slow_arms = [m for m in ("slow15", "slow20") if m in methods]
    best_slow = (max(slow_arms, key=lambda m: mean_over(test, m)) if slow_arms else None)

    def paired(a, b, ps, key="macro_sr"):
        use = [p for p in ps if cells[p].get(a) and cells[p].get(b)
               and cells[p][a][key] is not None and cells[p][b][key] is not None]
        d = np.array([cells[p][a][key] - cells[p][b][key] for p in use], dtype=np.float64)
        r = paired_test(d, rng)
        r["plants"] = use
        r["per_plant"] = {p: round(float(x), 4) for p, x in zip(use, d)}
        return r

    primary, pvals = {}, {}
    if "ours_p" in methods and "naive" in methods:
        primary["P1_ours_p_minus_naive"] = paired("ours_p", "naive", test)
        pvals["P1_ours_p_minus_naive"] = primary["P1_ours_p_minus_naive"]["p"]
    if "ours_p" in methods and best_slow:
        primary[f"P2_ours_p_minus_{best_slow}"] = paired("ours_p", best_slow, test)
        pvals[f"P2_ours_p_minus_{best_slow}"] = primary[f"P2_ours_p_minus_{best_slow}"]["p"]
    holm_p = holm(pvals) if pvals else {}

    # ---- P2 time-to-success on common-success cells ------------------------ #
    tts = {}
    if best_slow and "ours_p" in methods:
        cs = common_success_time(rows, test, "ours_p", best_slow, tasks)
        pp = cs["per_plant"]
        ps = list(pp.keys())
        d_plant = np.array([pp[p]["diff_s"] for p in ps], dtype=np.float64)
        stat_plant = paired_test(d_plant, rng)
        a_mean = float(np.mean([pp[p]["ours_p_s"] for p in ps])) if ps else None
        b_mean = float(np.mean([pp[p][f"{best_slow}_s"] for p in ps])) if ps else None
        d_cell = np.array([c["diff_s"] for c in cs["cells"]], dtype=np.float64)
        stat_cell = paired_test(d_cell, rng) if d_cell.size else None
        tts = {
            "arms": ["ours_p", best_slow],
            "definition": "mean time to success on plant x task cells where both arms have at "
                          "least one success; averaged within a plant, then paired across plants",
            "n_plants": len(ps),
            "n_common_cells": len(cs["cells"]),
            "ours_p_s": round(a_mean, 3) if a_mean else None,
            f"{best_slow}_s": round(b_mean, 3) if b_mean else None,
            "reduction_pct": (round(100.0 * (1.0 - a_mean / b_mean), 1)
                              if a_mean and b_mean else None),
            "paired_plant_level": stat_plant,
            "paired_cell_level": stat_cell,
            "per_plant": pp,
        }

    # ---- recovery rate ----------------------------------------------------- #
    recovery = {}
    if nom_naive is not None:
        for pid in pids:
            if pid == NOMINAL_PID:
                continue
            base = cells[pid].get("naive")
            if not base or base["macro_sr"] is None:
                continue
            den = nom_naive - base["macro_sr"]
            row = {"family": fam[pid], "role": "dev" if pid in DEV_PIDS else "test",
                   "sr_naive": round(base["macro_sr"], 4), "denominator": round(den, 4),
                   "counted": bool(den > 0.10 and pid in TEST_PIDS)}
            for m in methods:
                if m == "naive":
                    continue
                c = cells[pid].get(m)
                if c and c["macro_sr"] is not None and abs(den) > 1e-9:
                    row[f"R_{m}"] = round(float((c["macro_sr"] - base["macro_sr"]) / den), 3)
            recovery[pid] = row
    counted = [p for p, v in recovery.items() if v["counted"]]
    recovery_macro = {"n_counted_plants": len(counted), "counted_plants": counted,
                      "criterion": "test plants with SR_nominal - SR_naive > 0.10 "
                                   "(SR_nominal = naive on the nominal plant, a development plant)"}
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
        row = {"family": fam[pid], "role": "dev" if pid in DEV_PIDS else "test",
               "in_K1_set": pid in K1_PIDS, "naive_tracking_rmse_m": base["tracking_rmse_m"]}
        for m in methods:
            if m == "naive":
                continue
            c = cells[pid].get(m)
            if c and c["tracking_rmse_m"]:
                row[f"{m}_tracking_rmse_m"] = c["tracking_rmse_m"]
                row[f"{m}_reduction_pct"] = round(
                    100.0 * (1.0 - c["tracking_rmse_m"] / base["tracking_rmse_m"]), 1)
        k1[pid] = row
    k1_test = [p for p in test if p in K1_PIDS and p in k1]
    k1_all = [p for p in pids if p in K1_PIDS and p in k1]

    def k1_block(ps):
        vals = [k1[p]["ours_p_reduction_pct"] for p in ps if "ours_p_reduction_pct" in k1[p]]
        return {"plants": ps, "n": len(vals),
                "mean_ours_p_reduction_pct": round(float(np.mean(vals)), 1) if vals else None,
                "n_plants_reduction_ge_30": int(sum(1 for v in vals if v >= 30.0)),
                "K1_passed": bool(np.mean(vals) >= 30.0) if vals else None}

    k1_summary = {"definition": "plants with injected delay and/or lag (d_ms > 0 or tau_ms > 0)",
                  "test_plants_only": k1_block(k1_test),
                  "incl_dev_plants": k1_block(k1_all)}

    # ---- K2 / K3 ----------------------------------------------------------- #
    kills = {}
    if recovery_macro.get("R_ours_p") is not None:
        kills["K2_R_ours_p"] = recovery_macro["R_ours_p"]
        kills["K2_triggered_lt_0.3"] = bool(recovery_macro["R_ours_p"] < 0.3)
    if best_slow and "ours_p" in methods:
        key = f"P2_ours_p_minus_{best_slow}"
        gap_pp = 100.0 * primary[key]["mean"] if key in primary else None
        t_red = tts.get("reduction_pct")
        kills["K3_macro_gap_pp"] = round(gap_pp, 2) if gap_pp is not None else None
        kills["K3_time_reduction_pct_common_cells"] = t_red
        if gap_pp is not None:
            kills["K3_triggered"] = bool(gap_pp < 5.0 and (t_red is None or t_red < 25.0))
    if k1_summary["test_plants_only"]["K1_passed"] is not None:
        kills["K1_passed_test_plants"] = k1_summary["test_plants_only"]["K1_passed"]

    verdict = {}
    if "P1_ours_p_minus_naive" in primary:
        r = primary["P1_ours_p_minus_naive"]
        verdict["P1"] = {"mean_pp": round(100.0 * r["mean"], 2),
                         "ci95_pp": [round(100.0 * r["ci95"][0], 2), round(100.0 * r["ci95"][1], 2)],
                         "holm_p": holm_p.get("P1_ours_p_minus_naive"),
                         "meets_ge_15pp_and_ci_lower_gt_0":
                             bool(100.0 * r["mean"] >= 15.0 and r["ci95"][0] > 0.0)}
    if best_slow:
        key = f"P2_ours_p_minus_{best_slow}"
        r = primary.get(key)
        if r:
            verdict["P2"] = {"best_general_baseline": best_slow,
                             "mean_pp": round(100.0 * r["mean"], 2),
                             "ci95_pp": [round(100.0 * r["ci95"][0], 2),
                                         round(100.0 * r["ci95"][1], 2)],
                             "holm_p": holm_p.get(key),
                             "time_reduction_pct_common_cells": tts.get("reduction_pct"),
                             "meets_ge_5pp_or_time_minus_25pct":
                                 bool(100.0 * r["mean"] >= 5.0
                                      or (tts.get("reduction_pct") or -1e9) >= 25.0)}

    # ---- first-pass side by side ------------------------------------------- #
    compare = {}
    if args.v1_json and Path(args.v1_json).exists():
        v1 = json.loads(Path(args.v1_json).read_text(encoding="utf-8"))
        v1cells = v1.get("cells", {})
        compare["v1_label"] = v1.get("label")
        compare["v1_bound_pos_norm"] = v1.get("protocol", {}).get("executor_bound_pos_norm")
        compare["v1_macro"] = {m: v1["macro"][m]["macro_sr_family_mean"] for m in v1.get("macro", {})}
        compare["v2_macro_same_12_plants"] = {
            m: mean_over([p for p in pids if p in v1cells], m) for m in methods}
        compare["per_plant"] = {
            pid: {m: {"v1": (v1cells.get(pid, {}).get(m) or {}).get("macro_sr"),
                      "v2": (cells[pid].get(m) or {}).get("macro_sr")}
                  for m in methods}
            for pid in pids}
        compare["v1_primary"] = v1.get("primary")
        compare["v1_primary_holm_p"] = v1.get("primary_holm_p")
        compare["v1_recovery_macro"] = v1.get("recovery_macro")
        compare["v1_K1_summary"] = v1.get("K1_summary")
        compare["v1_kills"] = v1.get("kills")
        compare["v1_sat_episode_share_by_arm"] = {
            m: round(float(np.mean([(v1cells[p][m] or {}).get("n_bound_hit_episodes", 0)
                                    / max(1, (v1cells[p][m] or {}).get("n", 1))
                                    for p in v1cells if v1cells[p].get(m)])), 4)
            for m in methods}
    if args.v1_glob:
        v1rows = load(args.v1_glob)
        if v1rows:
            compare["v1_from_shards"] = {
                "n_episodes": len(v1rows),
                "sat_episode_share_by_arm": {
                    m: round(float(np.mean([1.0 if r.get("n_bound_hits", 0) > 0 else 0.0
                                            for r in v1rows if r["method"] == m])), 4)
                    for m in methods},
                "sat_step_share_by_arm": {
                    m: round(float(np.sum([r.get("n_bound_hits", 0) for r in v1rows
                                           if r["method"] == m])
                                   / max(1.0, float(np.sum([r["steps"] for r in v1rows
                                                            if r["method"] == m])))), 5)
                    for m in methods},
            }

    out = {
        "label": args.label,
        "pass": 2,
        "status": ("confirmatory run, corrected after the first pass was read; the first pass "
                   "remains the pre-registered result and is reported alongside"),
        "policy": "X-VLA-Pt + 2toINF/X-VLA-libero-spatial-peft (LoRA merged)",
        "suite": rows[0]["suite"],
        "n_episodes": len(rows),
        "plants": pids,
        "dev_plants": dev,
        "test_plants": test,
        "new_plants_this_pass": [p.pid for p in NEW_V2],
        "methods": methods,
        "tasks": tasks,
        "protocol": {
            "control_hz": round(1.0 / DT, 1),
            "chunk": 30, "chunk_seconds": 30 * DT,
            "budget_steps": 800, "denoise_steps": 10, "domain_id": 3,
            "act_type": "abs (robot.controller.use_delta = False)",
            "chunk_execution": "drained, re-queried when exhausted",
            "episodes_per_cell": max(len({r["ep"] for r in rows if r["pid"] == p
                                          and r["method"] == m})
                                     for p in pids for m in methods),
            "macro_sr": "10 libero_spatial tasks, equally weighted",
            "executor_bound_pos_norm": rows[0].get("bound_pos"),
            "executor_bound_rot_norm": rows[0].get("bound_rot"),
            "executor_bound_source": "prereg v3 item 1 (cce/executor.py's native working point); "
                                     "first pass used 5.0",
            "paired_unit": "plant",
            "P1_P2_population": "12 test plants; the 2 development plants are excluded",
            "time_to_success": "paired on plant x task cells where both arms succeed at least once",
            "K1_set_definition": "d_ms > 0 or tau_ms > 0",
            "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "resampled_unit": "plant"},
            "deviations": [
                "the second pass was designed after reading the first pass; it is a corrected "
                "confirmatory run, not a pre-registered one, and the first pass is reported "
                "unchanged next to it",
                "prereg v3 names the development plants 'P00_nominal' and 'D1_d150'; on this bed "
                "the nominal plant's pid is 'L00_nominal' (same plant, bed-specific prefix)",
                "GPU 7 was occupied by another experiment, so the second policy server ran on "
                "GPU 0; the sharding scheme (2 servers, 4 client shards each) is unchanged",
            ],
        },
        "macro": macro,
        "cells": cells,
        "G1_naive_drop_vs_nominal": g1,
        "primary": primary,
        "primary_holm_p": holm_p,
        "time_to_success_common_cells": tts,
        "verdict": verdict,
        "recovery": recovery,
        "recovery_macro": recovery_macro,
        "K1_tracking_rmse_vs_intent": k1,
        "K1_summary": k1_summary,
        "kills": kills,
        "best_slow_arm": best_slow,
        "compare_with_pass_1": compare,
        "wall_s_total": round(float(np.sum([r["wall_s"] for r in rows])), 1),
    }
    for name, path in (("identification", args.theta_summary), ("family_spec", args.family_spec),
                       ("devplant_working_point", args.devplant_json)):
        if path and Path(path).exists():
            blob = json.loads(Path(path).read_text(encoding="utf-8"))
            if name == "identification":
                blob = {"agreement": blob["agreement"], "rows": blob["rows"],
                        "agreement_new_plants": blob.get("agreement_new_plants"),
                        "probe_design": blob.get("probe_design")}
            out[name] = blob

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=json_default) + "\n")
    print(json.dumps({k: out[k] for k in ("label", "n_episodes", "macro", "primary_holm_p",
                                          "verdict", "recovery_macro", "K1_summary", "kills")},
                     indent=2, default=json_default))
    print(f"[agg-v2] wrote {p}")


if __name__ == "__main__":
    main()
