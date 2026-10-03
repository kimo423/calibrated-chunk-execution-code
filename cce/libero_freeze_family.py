#!/usr/bin/env python
"""Freeze the first-pass LIBERO plant family (prereg v2 amendment, step L3 -> L4).

Written once, after the naive-only G1 screening and before any arm comparison on the
family.  The screening decides one thing only: whether D4 (per-step displacement
clip) has a measurable effect on this bed and therefore belongs in the family.  It
does not touch the rest of the family, which was fixed in ``cce/libero_plant.py``
before the screening ran.
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
from cce.libero_family import BOUND_POS, BOUND_ROT, HORIZON
from cce.libero_plant import FAMILY_V1, NOMINAL_PID, SCREEN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scr = json.loads(Path(args.screen_summary).read_text(encoding="utf-8"))
    plants = [p.as_dict() for p in FAMILY_V1]
    payload = json.dumps(plants, sort_keys=True, default=json_default).encode()

    g1 = scr["G1_naive_drop_vs_nominal"]
    d4 = g1.get("D4_c05", {})
    d4_cell = scr["cells"].get("D4_c05", {}).get("naive", {})

    out = {
        "label": "CCE LIBERO plant family, first pass (v1)",
        "frozen_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "prereg": "docs/prereg/CCE_prereg_v2_amendment.md",
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
        "plants": plants,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "arms": list(METHODS),
        "episodes_per_cell": 5,
        "tasks": list(range(10)),
        "executor_command_bound": {
            "bound_pos_norm": BOUND_POS, "bound_pos_m": BOUND_POS * 0.1,
            "bound_rot_norm": BOUND_ROT, "bound_rot_rad": BOUND_ROT * 0.1,
            "role": "replaces ManiSkill's +-1 action bound, which LIBERO's absolute OSC "
                    "does not have; declared before any family arm was run",
        },
        "G2_check": {
            "gain_levels": sorted({p["gain"] for p in plants}),
            "delay_levels_ms": sorted({p["d_ms"] for p in plants}),
            "tau_levels_ms": sorted({p["tau_ms"] for p in plants}),
            "grip_delay_levels_s": sorted({p["grip_delay_s"] for p in plants}),
            "has_gain_0.70_and_1.00": bool({0.70, 1.00} <= {p["gain"] for p in plants}),
            "has_delay_0_and_ge_150ms": bool(0.0 in {p["d_ms"] for p in plants}
                                             and any(p["d_ms"] >= 150 for p in plants)),
        },
        "G1_screen": {
            "source": args.screen_summary,
            "design": "naive arm only, 10 tasks x 2 episodes per condition, 120 episodes",
            "nominal_naive_macro_sr": scr["G1_summary"]["nominal_naive_macro_sr"],
            "per_plant": g1,
            "passed": scr["G1_summary"]["G1_passed"],
        },
        "D4_decision": {
            "plant": "D4_c05 (per-step displacement clip 0.05 m)",
            "naive_macro_sr": d4.get("naive_macro_sr"),
            "drop_pp": d4.get("drop_pp"),
            "mean_steps": d4_cell.get("mean_steps"),
            "max_commanded_increment_norm": d4_cell.get("u_pos_max_norm"),
            "verdict": "no-effect level on this bed: at 20 Hz the absolute-target increment "
                       "rarely exceeds 0.05 m, so the clip almost never binds; excluded from "
                       "the first-pass family and recorded as such",
            "in_family": False,
        },
        "excluded_from_first_pass": {
            "over-responding gain (g > 1)": "not in the v2 amendment's factor levels; "
                                            "over-response is not covered in this pass",
            "D4 clip": "no effect (see D4_decision)",
            "d = 50 ms / tau = 100 ms as standalone plants":
                "carried only inside the mixed plants, to keep the family at 12 for the "
                "first pass (the pre-registered target is 20)",
        },
        "note": "Frozen before any plant-family arm comparison was run.  No arm result other "
                "than the pre-registered naive-only G1 screening informed this file.",
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=json_default) + "\n")
    print(json.dumps({"n_plants": out["n_plants"], "sha256": out["sha256"],
                      "G2_check": out["G2_check"], "D4_in_family": False}, indent=2))
    print(f"[freeze] wrote {p}")


if __name__ == "__main__":
    main()
