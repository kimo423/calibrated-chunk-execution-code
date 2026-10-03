"""Candidate family from every row of plan v4: 19 plants = 3 dev + 16 test.

Not a frozen family. The plan's prose count of 18 conflicts with its 19 table
rows and five explicitly requested additions; no listed plant is silently cut.
"""
import _bootstrap  # noqa: F401
from cce.libero_plant import spec, NOMINAL_PID
from cce.libero_plant_v2 import FAMILY_V2

NEW_V3 = [
    spec('D1_d100', 'D1', d_ms=100, note='new development plant'),
    spec('D2_t150', 'D2t', tau_ms=150),
    spec('MX04', 'MIX', d_ms=100, tau_ms=400, grip_delay_s=.2),
    spec('MX05', 'MIX', d_ms=200, tau_ms=100, gain=.85, grip_delay_s=.6),
    spec('OV2', 'OVER', tau_ms=250, gain=1.15),
]
FAMILY_V3 = list(FAMILY_V2) + NEW_V3
DEV_PIDS = (NOMINAL_PID, 'D1_d150', 'D1_d100')
TEST_PIDS = tuple(p.pid for p in FAMILY_V3 if p.pid not in DEV_PIDS)
BY_ID_V3 = {p.pid: p for p in FAMILY_V3}
assert len(FAMILY_V3) == len(BY_ID_V3) == 19
assert len(TEST_PIDS) == 16
