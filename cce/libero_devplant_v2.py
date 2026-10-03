#!/usr/bin/env python
"""Development-plant working point for the second pass (prereg v3 item 1).

Prereg v1 requires the executor hyper-parameters to be fixed on two development
plants only; prereg v3 adds the command bound to that list and names the two
plants (``P00_nominal`` = this bed's ``L00_nominal``, and ``D1_d150``).  This
script reads the ours-P scan shards, ranks the scanned bounds by the *tracking
RMSE against the policy intent* averaged over the two development plants, and
writes the selection to ``cce/results/libero_devplant_v2.json``.

The scan is run before the confirmatory family run and its plants are excluded
from the P1 / P2 plant-level statistics.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob as _glob
import json
from pathlib import Path

import numpy as np

from cce.common import DT, json_default
from cce.libero_exec import LiberoExecutor
from cce.libero_plant_v2 import DEV_PIDS


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


def macro_sr(rows) -> float | None:
    tasks = sorted({r["task_id"] for r in rows})
    per = [float(np.mean([r["success"] for r in rows if r["task_id"] == t])) for t in tasks]
    return float(np.mean(per)) if per else None


def cell(rows) -> dict:
    ok = [r for r in rows if r["success"]]
    hits = float(np.sum([r["n_bound_hits"] for r in rows]))
    steps = float(np.sum([r["steps"] for r in rows]))
    return {
        "n_episodes": len(rows),
        "n_tasks": len({r["task_id"] for r in rows}),
        "tracking_rmse_vs_intent_m": round(float(np.mean([r["tracking_rmse_m"] for r in rows])), 5),
        "tracking_rmse_vs_ref_m": round(float(np.mean([r["tracking_rmse_vs_ref_m"] for r in rows])), 5),
        "macro_sr": macro_sr(rows),
        "mean_steps": round(float(np.mean([r["steps"] for r in rows])), 1),
        "mean_time_to_success_s": (round(float(np.mean([r["steps"] * DT for r in ok])), 3)
                                   if ok else None),
        "sat_episode_share": round(float(np.mean([1.0 if r["n_bound_hits"] > 0 else 0.0
                                                  for r in rows])), 4),
        "sat_step_share": round(hits / steps, 4) if steps else None,
        "u_pos_max_norm": round(max(r["u_pos_max_norm"] for r in rows), 3),
        "wall_s": round(float(np.sum([r["wall_s"] for r in rows])), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser("dev-plant working point selection (prereg v3)")
    ap.add_argument("--glob", nargs="+", required=True)
    ap.add_argument("--method", default="ours_p")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [r for r in load(args.glob) if r["method"] == args.method]
    if not rows:
        raise SystemExit("no rows")
    bounds = sorted({float(r["bound_pos"]) for r in rows})
    pids = [p for p in DEV_PIDS if any(r["pid"] == p for r in rows)]

    scan = {}
    for b in bounds:
        sub = [r for r in rows if float(r["bound_pos"]) == b]
        per_plant = {p: cell([r for r in sub if r["pid"] == p]) for p in pids}
        scan[f"{b:g}"] = {
            "bound_pos_norm": b,
            "bound_pos_m": round(b * 0.1, 4),
            "bound_rot_norm": float(sub[0]["bound_rot"]),
            "per_plant": per_plant,
            "mean_tracking_rmse_vs_intent_m": round(float(np.mean(
                [per_plant[p]["tracking_rmse_vs_intent_m"] for p in pids])), 5),
            "mean_macro_sr": round(float(np.mean([per_plant[p]["macro_sr"] for p in pids])), 4),
            "mean_sat_episode_share": round(float(np.mean(
                [per_plant[p]["sat_episode_share"] for p in pids])), 4),
            "mean_sat_step_share": round(float(np.mean(
                [per_plant[p]["sat_step_share"] for p in pids])), 4),
            "n_episodes": sum(per_plant[p]["n_episodes"] for p in pids),
        }

    best = min(scan, key=lambda k: scan[k]["mean_tracking_rmse_vs_intent_m"])
    per_plant_best = {p: min(scan, key=lambda k: scan[k]["per_plant"][p]["tracking_rmse_vs_intent_m"])
                      for p in pids}

    # lambda / mu / resolve_every are parameters of the *QP* arm in cce/executor.py
    # (InversionExecutor(mode="qp")).  ours-P is the closed-form parameter arm, and
    # LiberoExecutor transcribes only that arm, so those three do not exist on this
    # path; kappa is the only remaining executor hyper-parameter and is carried over
    # from the simulation side at its default 1.0 (cce/libero_family.py does not
    # expose it, so the confirmatory run cannot vary it).
    probe = LiberoExecutor.__init__.__code__.co_varnames
    hyper = {
        "scanned": ["bound_pos", "bound_rot"],
        "lambda_mu_exposed_by_libero_exec": bool("lam" in probe or "mu" in probe),
        "lambda_mu_note": ("lam / mu / resolve_every belong to cce/executor.py's QP arm "
                           "(mode='qp'); ours-P is the closed-form parameter arm, which "
                           "LiberoExecutor transcribes, so they are structurally absent on "
                           "this bed and keep their simulation-side values by construction"),
        "kappa": 1.0,
        "kappa_note": ("simulation-side default, unchanged; not exposed by the family driver, "
                       "so it is constant across every arm and plant in both passes"),
    }

    out = {
        "label": "CCE LIBERO dev-plant working point (prereg v3 item 1)",
        "prereg": "docs/prereg/CCE_prereg_v3_amendment.md",
        "arm": args.method,
        "dev_plants": pids,
        "dev_plant_name_mapping": {
            "P00_nominal (prereg v3 wording)": "L00_nominal (this bed's nominal plant pid)",
        },
        "design": {
            "tasks": sorted({r["task_id"] for r in rows}),
            "episodes_per_task": len({r["ep"] for r in rows}),
            "episodes_per_configuration": sum(1 for r in rows
                                              if float(r["bound_pos"]) == bounds[0]) // max(len(pids), 1),
            "scanned_bounds_norm": bounds,
            "scanned_bounds_m": [round(b * 0.1, 4) for b in bounds],
            "selection_criterion": "minimum tracking RMSE against the policy intent, "
                                   "averaged over the two development plants",
        },
        "hyperparameters": hyper,
        "scan": scan,
        "selected_bound_norm": float(scan[best]["bound_pos_norm"]),
        "selected_by_criterion": best,
        "per_plant_argmin": per_plant_best,
        "prereg_v3_fixed_bound_norm": 1.0,
        "selection_matches_prereg_v3": bool(abs(float(scan[best]["bound_pos_norm"]) - 1.0) < 1e-9),
        "n_episodes": len(rows),
        "note": ("Development-plant results only.  Excluded from the P1 / P2 plant-level "
                 "statistics of the confirmatory run, reported separately."),
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=json_default) + "\n")
    print(json.dumps({k: out[k] for k in ("scan", "selected_bound_norm",
                                          "selection_matches_prereg_v3", "n_episodes")},
                     indent=2, default=json_default))
    print(f"[dev] wrote {p}")


if __name__ == "__main__":
    main()
