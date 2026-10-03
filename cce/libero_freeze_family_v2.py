#!/usr/bin/env python
"""Freeze the second-pass LIBERO plant family (prereg v3 amendment).

Written after the development-plant working-point scan and the identification of
the two new over-responding plants, and before any arm of the confirmatory family
run.  Nothing here is decided by a test-plant arm result: the family is the frozen
first-pass 12 plus the two plants the amendment names, the development / test split
is the amendment's, and the command bound is the amendment's fixed 1.0 (the
development scan only confirms it).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from cce.common import DT, json_default
from cce.libero_exec import METHODS
from cce.libero_family_v2 import BOUND_POS_V2, BOUND_ROT_V2, HORIZON
from cce.libero_plant_v2 import DEV_PIDS, FAMILY_V2, K1_PIDS, NEW_V2, TEST_PIDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devplant-json", required=True)
    ap.add_argument("--identify-json", required=True)
    ap.add_argument("--v1-spec", default="cce/results/libero_family_spec.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = json.loads(Path(args.devplant_json).read_text(encoding="utf-8"))
    ident = json.loads(Path(args.identify_json).read_text(encoding="utf-8"))
    v1 = json.loads(Path(args.v1_spec).read_text(encoding="utf-8")) if Path(args.v1_spec).exists() else {}

    plants = []
    for p in FAMILY_V2:
        d = p.as_dict()
        d["role"] = "dev" if p.pid in DEV_PIDS else "test"
        d["new_in_pass_2"] = p.pid in {q.pid for q in NEW_V2}
        d["in_K1_set"] = p.pid in K1_PIDS
        plants.append(d)
    payload = json.dumps(plants, sort_keys=True, default=json_default).encode()

    gains = {p["gain"] for p in plants}
    delays = {p["d_ms"] for p in plants}
    taus = {p["tau_ms"] for p in plants}
    grips = {p["grip_delay_s"] for p in plants}

    ident_rows = {r["pid"]: r for r in ident["rows"]}
    out = {
        "label": "CCE LIBERO plant family, second pass (v2)",
        "frozen_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "prereg": ["docs/prereg/CCE_prereg_v1.md",
                   "docs/prereg/CCE_prereg_v2_amendment.md",
                   "docs/prereg/CCE_prereg_v3_amendment.md"],
        "status": ("confirmatory run corrected after seeing the first pass; the first pass "
                   "remains the pre-registered result and is reported alongside"),
        "bed": {
            "policy": "X-VLA-Pt + 2toINF/X-VLA-libero-spatial-peft (LoRA merged)",
            "suite": "libero_spatial (10 tasks)",
            "control_hz": round(1.0 / DT, 1),
            "chunk": 30,
            "budget_steps": HORIZON,
            "act_type": "abs (robot.controller.use_delta = False)",
            "insertion_point": "policy chunk absolute EE target -> OSC controller input",
        },
        "n_plants": len(plants),
        "n_dev_plants": len(DEV_PIDS),
        "n_test_plants": len(TEST_PIDS),
        "dev_plants": list(DEV_PIDS),
        "test_plants": list(TEST_PIDS),
        "plants": plants,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "arms": list(METHODS),
        "episodes_per_cell": 5,
        "tasks": list(range(10)),
        "n_episodes_planned": len(plants) * len(METHODS) * 10 * 5,
        "inherited_from_v1": {
            "spec_file": args.v1_spec,
            "sha256": v1.get("sha256"),
            "n_plants": v1.get("n_plants"),
            "note": "the 12 first-pass plants are carried over unchanged (same pids, same "
                    "parameters); the second pass adds two plants and changes nothing else",
        },
        "new_plants": [
            {**p.as_dict(),
             "reason": "prereg v1 G2 requires the family to contain over-response as well as "
                       "under-response; prereg v3 item 2 adds these two levels",
             "identification": {
                 k: ident_rows.get(p.pid, {}).get(k)
                 for k in ("d_true_steps", "d_hat_steps", "tau_true_s", "tau_rel_s",
                           "g_true", "g_hat", "g_rel", "grip_lead_true_steps",
                           "grip_lead_hat_steps", "r2free_min_trans", "r2free_min_rot")}}
            for p in NEW_V2
        ],
        "executor_command_bound": {
            "bound_pos_norm": BOUND_POS_V2, "bound_pos_m": BOUND_POS_V2 * 0.1,
            "bound_rot_norm": BOUND_ROT_V2, "bound_rot_rad": BOUND_ROT_V2 * 0.1,
            "source": "prereg v3 item 1: cce/executor.py's native working point "
                      "(ManiSkill's +-1 action bound), fixed before this run",
            "first_pass_value_norm": 5.0,
            "dev_plant_confirmation": {
                "file": args.devplant_json,
                "dev_plants": dev.get("dev_plants"),
                "scanned_bounds_norm": dev.get("design", {}).get("scanned_bounds_norm"),
                "criterion": dev.get("design", {}).get("selection_criterion"),
                "argmin_bound_norm": dev.get("selected_bound_norm"),
                "matches_prereg_v3": dev.get("selection_matches_prereg_v3"),
                "per_bound_mean_tracking_rmse_m": {
                    k: v.get("mean_tracking_rmse_vs_intent_m") for k, v in
                    (dev.get("scan") or {}).items()},
            },
            "other_executor_hyperparameters": dev.get("hyperparameters"),
        },
        "G2_check": {
            "gain_levels": sorted(gains),
            "delay_levels_ms": sorted(delays),
            "tau_levels_ms": sorted(taus),
            "grip_delay_levels_s": sorted(grips),
            "has_gain_0.70_and_1.00": bool({0.70, 1.00} <= gains),
            "has_delay_0_and_ge_150ms": bool(0.0 in delays and any(d >= 150 for d in delays)),
            "has_over_response_gain_gt_1": bool(any(g > 1.0 for g in gains)),
            "has_under_and_over_response": bool(any(g < 1.0 for g in gains)
                                                and any(g > 1.0 for g in gains)),
        },
        "K1_plant_set": {
            "pids": list(K1_PIDS),
            "definition": "plants whose injected mismatch contains transport delay and/or "
                          "first-order lag (d_ms > 0 or tau_ms > 0)",
            "note": "on the first-pass family this selects exactly the same seven plants as "
                    "the first pass's family-label list ('D1', 'D2t', 'MIX'); stated as a "
                    "predicate so the new mixed plant, which carries a 100 ms delay under a "
                    "new family label, is resolved without a judgement call",
        },
        "statistics": {
            "paired_unit": "plant",
            "P1_P2_population": "the 12 test plants (development plants excluded)",
            "time_to_success": "paired on plant x task cells where both compared arms have at "
                               "least one success (prereg v3 item 4)",
        },
        "excluded": {
            "D4 clip": "no-effect level on this bed (first-pass L3 screening, 0.0 pp drop); "
                       "excluded again",
            "arms Universal-fixed / RTC-only / prompt-fit / codec-fit / FT-demo / ours-L":
                "prereg v3 item 3 keeps the arm set unchanged for this pass",
        },
        "note": "Frozen before any arm of the confirmatory family run.  The only measurements "
                "that informed this file are the naive-only G1 screening of the first pass, the "
                "development-plant scan above (ours-P on two development plants, excluded from "
                "the P1 / P2 population) and the open-loop probes.",
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=json_default) + "\n")
    print(json.dumps({"n_plants": out["n_plants"], "n_test": out["n_test_plants"],
                      "sha256": out["sha256"], "G2_check": out["G2_check"],
                      "bound_norm": BOUND_POS_V2,
                      "n_episodes_planned": out["n_episodes_planned"]}, indent=2))
    print(f"[freeze-v2] wrote {p}")


if __name__ == "__main__":
    main()
