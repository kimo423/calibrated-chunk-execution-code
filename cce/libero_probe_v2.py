#!/usr/bin/env python
"""Probe + ARX identification for the two over-responding plants (prereg v3 item 2).

The probe design, the excitation, the ARX fit and the gripper-lead read-out are
imported unchanged from ``cce/libero_probe.py``; only the plant list differs.  The
first-pass theta-hat file is read and merged, never overwritten: the second-pass
outputs carry a ``_v2`` suffix

    data/cce/libero/theta/theta_hat_libero_panda_v2.json
    data/cce/libero/theta/identify_libero_panda_v2.json

so that ``cce/libero_family.py`` at its own defaults still reproduces the first pass.

The gripper lead is referenced to the *first-pass nominal* 90 %-arrival time, i.e.
exactly the same reference the first-pass plants used.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("LIBERO_CONFIG_PATH", "/opt/cce/data/libero_config")
os.environ.setdefault("MUJOCO_GL", "egl")

from cce.common import DT, json_default  # noqa: E402
from cce.identify import identify_plant  # noqa: E402
from cce.libero_plant import NOMINAL_PID  # noqa: E402
from cce.libero_plant_v2 import FAMILY_V2, NEW_V2  # noqa: E402
from cce.libero_probe import (  # noqa: E402
    ATT_AMP_RAD,
    BURST_CMD_NORM,
    CHIRP_AMP_M,
    DATA,
    UID,
    make_evaluator,
    probe_stats,
    run_probe,
    summarize,
)
from cce.probe import CHIRP_F0, CHIRP_F1, GRIP_CLOSE_STEP, N_PROBE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser("CCE probe + ARX identification, second pass (new plants only)")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--z-floor", type=float, default=1.05)
    ap.add_argument("--theta-dir", default=str(DATA / "theta"))
    ap.add_argument("--probe-dir", default=str(DATA / "probes"))
    ap.add_argument("--base-hats", default=str(DATA / "theta" / f"theta_hat_{UID}.json"))
    args = ap.parse_args()

    theta_dir, probe_dir = Path(args.theta_dir), Path(args.probe_dir)
    theta_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    base = json.loads(Path(args.base_hats).read_text())
    if NOMINAL_PID not in base:
        raise SystemExit(f"first-pass nominal hat missing in {args.base_hats}")
    t90_nom = float(base[NOMINAL_PID]["grip_t90_s"])

    ev, suites, LC = make_evaluator(args.suite)
    suite = suites[0]

    new_hats, stats = {}, []
    for spec in NEW_V2:
        rec = run_probe(ev, suite, spec, args.task_id, args.ep, args.z_floor)
        np.savez_compressed(probe_dir / f"probe_{UID}_{spec.pid}.npz",
                            u=rec["u"], y=rec["y"], grip_open=rec["grip_open"],
                            grip_cmd=rec["grip_cmd"], home=rec["home"],
                            targets=rec["targets"], applied=rec["applied"])
        st = probe_stats(rec)
        stats.append(st)
        print(json.dumps(st, default=json_default), flush=True)
        h = identify_plant(rec, sat_aware=False)
        lead = h["grip_t90_s"] - t90_nom
        h["grip_t90_nominal_s"] = t90_nom
        h["grip_lead_s_hat"] = float(lead) if np.isfinite(lead) else 0.0
        h["grip_lead_steps_hat"] = (int(max(0, round(h["grip_lead_s_hat"] / DT)))
                                    if np.isfinite(lead) else 0)
        new_hats[spec.pid] = h

    merged = dict(base)
    merged.update(new_hats)
    out_hats = theta_dir / f"theta_hat_{UID}_v2.json"
    out_hats.write_text(json.dumps(merged, indent=2, default=json_default) + "\n")

    fam_hats = {p.pid: merged[p.pid] for p in FAMILY_V2 if p.pid in merged}
    missing = [p.pid for p in FAMILY_V2 if p.pid not in merged]
    if missing:
        raise SystemExit(f"missing theta-hat for {missing}")
    summary = summarize(fam_hats, list(FAMILY_V2))
    summary["label"] = "CCE LIBERO identification, second pass family (14 plants)"
    summary["new_plants_this_pass"] = [p.pid for p in NEW_V2]
    summary["base_hats"] = str(args.base_hats)
    summary["probe_stats_new_plants"] = stats
    summary["probe_design"] = {
        "steps": int(N_PROBE), "seconds": N_PROBE * DT,
        "chirp_amp_m": CHIRP_AMP_M, "chirp_hz": [CHIRP_F0, CHIRP_F1],
        "burst_cmd_norm": BURST_CMD_NORM, "burst_cmd_m": BURST_CMD_NORM * 0.1,
        "attitude_amp_rad": ATT_AMP_RAD, "grip_close_step": int(GRIP_CLOSE_STEP),
        "suite": args.suite, "task_id": args.task_id, "ep": args.ep,
        "note": "identical to the first pass; only the plant list differs",
    }
    new_rows = [r for r in summary["rows"] if r["pid"] in new_hats]
    summary["agreement_new_plants"] = {
        "rows": new_rows,
        "delay_mae_steps": float(np.mean([abs(r["d_hat_steps"] - r["d_true_steps"])
                                          for r in new_rows])),
        "gain_rel_mae": float(np.mean([abs(r["g_rel"] - r["g_true"]) for r in new_rows])),
        "tau_rel_mae_s": float(np.mean([abs(r["tau_rel_s"] - r["tau_true_s"]) for r in new_rows])),
    }
    out_sum = theta_dir / f"identify_{UID}_v2.json"
    out_sum.write_text(json.dumps(summary, indent=2, default=json_default) + "\n")

    print(json.dumps({"n_plants_in_file": len(merged),
                      "n_plants_in_family_v2": len(fam_hats),
                      "new": summary["agreement_new_plants"],
                      "agreement_family_v2": summary["agreement"]},
                     indent=2, default=json_default), flush=True)
    print(f"[probe-v2] wrote {out_hats}")
    print(f"[probe-v2] wrote {out_sum}")


if __name__ == "__main__":
    main()
