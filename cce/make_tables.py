#!/usr/bin/env python
"""Emit the markdown tables used by docs/reports/CCE_sim_probe_executor.md."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from pathlib import Path

import numpy as np

from cce.plants import PLANTS

ROOT = Path("/opt/cce")
RES = ROOT / "cce" / "results"
NOM = "P00_nominal"


def f(x, n=3):
    if x is None:
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "-" if not np.isfinite(v) else f"{v:.{n}f}"


def plant_table() -> None:
    print("### T1 plant family\n")
    print("| id | family | d (ms) | tau (ms) | g | clip (m/step) | T_g (s) | gripper lead (steps) |")
    print("|---|---|---|---|---|---|---|---|")
    for p in PLANTS:
        print(f"| `{p.pid}` | {p.family} | {p.d_ms:.0f} | {p.tau_ms:.0f} | {p.gain:.2f} | "
              f"{p.clip_pos:.2f} | {p.grip_delay_s:.1f} | {p.grip_delay_steps} |")
    print()


def identify_table() -> None:
    d = json.loads((RES / "identify_summary.json").read_text())
    for uid, b in d["by_uid"].items():
        print(f"### T2 identification, {uid}\n")
        print("| plant | d true/hat (steps) | tau true/rel-hat (s) | g true/rel-hat | "
              "grip lead true/hat (steps) | R2 free-run xyz | R2 free-run rot |")
        print("|---|---|---|---|---|---|---|")
        for r in b["rows"]:
            print(f"| `{r['pid']}` | {r['d_true_steps']:.0f} / {r['d_hat_steps']:.0f} | "
                  f"{f(r['tau_true_s'])} / {f(r['tau_rel_s'])} | {f(r['g_true'], 2)} / {f(r['g_rel'], 2)} | "
                  f"{r['grip_lead_true_steps']:.0f} / {r['grip_lead_hat_steps']:.0f} | "
                  f"{f(r['r2free_min_trans'])} | {f(r['r2free_min_rot'])} |")
        a = b["agreement"]
        print()
        print(f"aggregate: delay corr {f(a['delay_corr'])}, MAE {f(a['delay_mae_ms'], 1)} ms; "
              f"tau(rel) corr {f(a['tau_corr_rel'])}, MAE {f(a['tau_mae_rel_s'])} s; "
              f"gain(rel) corr {f(a['gain_corr_rel'])}, MAE {f(a['gain_mae_rel'])}; "
              f"gripper lead corr {f(a['grip_lead_corr'])}, MAE {f(a['grip_lead_mae_steps'], 2)} steps; "
              f"axes below R2 gate {a['n_axes_below_r2_gate']}/{a['n_axes_total']}; "
              f"nominal g_hat {f(a['gain_hat_on_nominal'])}, nominal tau_hat {f(a['tau_hat_on_nominal_s'])} s\n")


def tracking_table() -> None:
    d = json.loads((RES / "tracking_summary.json").read_text())
    for uid, b in d["by_uid"].items():
        sat = b["ablation_sat_aware_rmse"]
        clip = b["ablation_clip_from_burst_rmse"]
        print(f"### T3 tracking RMSE (m), {uid}  [20 trajectories x 8 s per plant]\n")
        print("| plant | naive | ours-P | ours-P+sat | oracle | Slow x1.5 | Slow x2 | ours-P clip-burst | "
              "reduction ours-P | rot RMSE naive/ours-P (rad) | gripper phase err ours-P (s) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for p in b["plants"]:
            pid = p["pid"]
            print(f"| `{pid}` | {f(p['rmse_naive'],5)} | {f(p['rmse_ours_p'],5)} | {f(sat.get(pid),5)} | "
                  f"{f(p['rmse_oracle'],5)} | {f(p['rmse_slow15'],5)} | {f(p['rmse_slow20'],5)} | "
                  f"{f(clip.get(pid),5)} | {f(p.get('red_ours_p'),2)} | "
                  f"{f(p['rmserot_naive'],4)} / {f(p['rmserot_ours_p'],4)} | {f(p.get('gripphase_ours_p'),3)} |")
        nn = [p for p in b["plants"] if p["pid"] != NOM]
        print()
        print("macro over the 25 non-nominal plants: "
              + ", ".join(f"{m} {f(np.mean([p['rmse_' + m] for p in nn]), 5)}" for m in b["methods"])
              + f", ours-P+sat {f(np.mean([sat[p['pid']] for p in nn]), 5)}"
              + f", ours-P clip-burst {f(np.mean([clip[p['pid']] for p in nn]), 5)}")
        print("macro RMSE reduction vs naive: "
              + ", ".join(f"{m} {f(v)}" for m, v in b["macro_reduction_vs_naive"].items())
              + f"; lag-plant subset: " + ", ".join(f"{m} {f(v)}" for m, v in b["macro_reduction_lag_only"].items()))
        print("macro gripper phase error (s): "
              + ", ".join(f"{m} {f(v)}" for m, v in b["macro_grip_phase_err_s"].items()))
        for k, v in b["paired"].items():
            print(f"paired ({k}): mean {f(v.get('mean_diff'),5)}, t({v.get('n',0)-1}) {f(v.get('t'),2)}, "
                  f"bootstrap CI95 [{f(v.get('ci95',[None,None])[0],5)}, {f(v.get('ci95',[None,None])[1],5)}], "
                  f"better/worse {v.get('n_plants_better')}/{v.get('n_plants_worse')}")
        if b["ablation_qp_rmse"]:
            main = {p["pid"]: p["rmse_ours_p"] for p in b["plants"]}
            print("QP arm (subset): " + ", ".join(
                f"{k} {f(v,5)} vs ours-P {f(main[k],5)}" for k, v in b["ablation_qp_rmse"].items()))
        print()


def replay_table() -> None:
    d = json.loads((RES / "replay_summary.json").read_text())
    for uid, b in d["by_uid"].items():
        sat = b["ablation_sat_aware_sr"]
        clip = b["ablation_clip_from_burst_sr"]
        print(f"### T4 expert-replay success rate, {uid}  [50 episodes / plant, 200-step budget]\n")
        print("| plant | naive | naive-open | ours-P | ours-P+sat | oracle | Slow x1.5 | Slow x2 | "
              "ours-P clip-burst | R (ours-P) | R vs best naive |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for p in b["plants"]:
            pid = p["pid"]
            print(f"| `{pid}` | {f(p['sr_naive'],2)} | {f(p['sr_naive_open'],2)} | {f(p['sr_ours_p'],2)} | "
                  f"{f(sat.get(pid),2)} | {f(p['sr_oracle'],2)} | {f(p['sr_slow15'],2)} | {f(p['sr_slow20'],2)} | "
                  f"{f(clip.get(pid),2)} | {f(p.get('R_ours_p'),2)} | {f(p.get('Rb_ours_p'),2)} |")
        nn = [p for p in b["plants"] if p["pid"] != NOM]
        print()
        print("macro SR over the 25 non-nominal plants: "
              + ", ".join(f"{m} {f(v)}" for m, v in b["macro_sr"].items())
              + f", ours-P+sat {f(np.mean([sat[p['pid']] for p in nn]))}"
              + f", ours-P clip-burst {f(np.mean([clip[p['pid']] for p in nn]))}")
        print(f"macro recovery R (plants whose naive drop > 10 pp, n = {b['n_plants_R_defined']}): "
              + ", ".join(f"{m} {f(v)}" for m, v in b["macro_R"].items()))
        print(f"macro recovery vs the better naive variant (n = {b['n_plants_Rb_defined']}): "
              + ", ".join(f"{m} {f(v)}" for m, v in b["macro_R_vs_best_naive"].items()))
        print("macro time-to-success (s): " + ", ".join(f"{m} {f(v,2)}" for m, v in b["macro_tts_s"].items()))
        for k, v in b["paired"].items():
            print(f"paired ({k}): mean {f(v.get('mean_diff'),4)}, t({v.get('n',0)-1}) {f(v.get('t'),2)}, "
                  f"bootstrap CI95 [{f(v.get('ci95',[None,None])[0],4)}, {f(v.get('ci95',[None,None])[1],4)}], "
                  f"better/worse {v.get('n_plants_better')}/{v.get('n_plants_worse')}")
        print(f"gates: {json.dumps(b['gates'])}")
        print()


if __name__ == "__main__":
    plant_table()
    identify_table()
    tracking_table()
    replay_table()
