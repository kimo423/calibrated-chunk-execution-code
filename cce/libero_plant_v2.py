#!/usr/bin/env python
"""Second-pass (prereg v3 amendment) LIBERO plant family.

Additive only: ``cce/libero_plant.py`` is untouched, so the first pass stays
reproducible.  This module contributes

* the two over-responding plants required by prereg v1's G2 clause ("the family
  must contain both under- and over-response") and added by prereg v3 item 2:
  a pure gain plant ``g = 1.3`` and a mixed plant ``g = 1.3 + d = 100 ms``;
* ``FAMILY_V2`` = the frozen first-pass 12 plus those two (14 plants);
* the two development plants named by prereg v3 item 1.  The amendment writes the
  nominal one with the ManiSkill-side pid ``P00_nominal``; on this bed the same
  plant is ``L00_nominal`` (``cce/libero_plant.py: NOMINAL_PID``).  The prefix is
  a bed name, not a different plant;
* the K1 plant set, stated as a parameter predicate (``d_ms > 0 or tau_ms > 0``)
  instead of the first pass's family-label list ``("D1", "D2t", "MIX")``.  On the
  first-pass family the two definitions select exactly the same seven plants; the
  predicate additionally resolves the new mixed plant, which carries a 100 ms
  transport delay under a new family label.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from cce.libero_plant import FAMILY_V1, NOMINAL_PID, spec
from cce.plants import PlantSpec

# --------------------------------------------------------------------------- #
# new levels (prereg v3 item 2)
# --------------------------------------------------------------------------- #
D2_G130 = spec("D2_g130", "OVER", gain=1.30, note="over-responding gain 1.30")
MX_G130_D100 = spec("MX_g130_d100", "OVERMIX", d_ms=100, gain=1.30,
                    note="over-responding gain 1.30 + transport delay 100 ms")

NEW_V2: list[PlantSpec] = [D2_G130, MX_G130_D100]
FAMILY_V2: list[PlantSpec] = list(FAMILY_V1) + NEW_V2

# --------------------------------------------------------------------------- #
# development / test split (prereg v3 item 1)
# --------------------------------------------------------------------------- #
DEV_PIDS: tuple[str, ...] = (NOMINAL_PID, "D1_d150")
TEST_PIDS: tuple[str, ...] = tuple(p.pid for p in FAMILY_V2 if p.pid not in DEV_PIDS)

DEV_SPECS: list[PlantSpec] = [p for p in FAMILY_V2 if p.pid in DEV_PIDS]
TEST_SPECS: list[PlantSpec] = [p for p in FAMILY_V2 if p.pid not in DEV_PIDS]

BY_ID_V2: dict[str, PlantSpec] = {p.pid: p for p in FAMILY_V2}


def carries_delay_or_lag(s: PlantSpec) -> bool:
    """K1 set: plants whose injected mismatch contains transport delay and/or lag."""
    return bool(s.d_ms > 0.0 or s.tau_ms > 0.0)


K1_PIDS: tuple[str, ...] = tuple(p.pid for p in FAMILY_V2 if carries_delay_or_lag(p))


def get(pid: str) -> PlantSpec:
    return BY_ID_V2[pid]


if __name__ == "__main__":
    import json

    from cce.libero_aggregate import LAG_FAMILIES

    v1_lag_by_label = [p.pid for p in FAMILY_V1 if p.family in LAG_FAMILIES]
    v1_lag_by_pred = [p.pid for p in FAMILY_V1 if carries_delay_or_lag(p)]
    print(json.dumps({
        "n_family_v2": len(FAMILY_V2),
        "dev_pids": list(DEV_PIDS),
        "test_pids": list(TEST_PIDS),
        "n_test": len(TEST_PIDS),
        "new_plants": [p.as_dict() for p in NEW_V2],
        "K1_pids_v2": list(K1_PIDS),
        "K1_definition_matches_v1_on_family_v1": v1_lag_by_label == v1_lag_by_pred,
        "K1_v1_by_label": v1_lag_by_label,
        "K1_v1_by_predicate": v1_lag_by_pred,
    }, indent=2))
