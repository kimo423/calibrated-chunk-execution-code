#!/usr/bin/env python
"""Launcher for cce/xvla_patch_v3/train_cce.py (CCE variant v3, relative xyz).

Identical to `cce/run_train_cce.py` except that the patched `datasets` package
put ahead of the upstream one is `cce/xvla_patch_v3` rather than
`cce/xvla_patch`. `train_cce.py` itself is a byte-for-byte copy of the mainline
file, so the training recipe is unchanged; the only difference between this run
and the mainline LoRA run lives in the dataset package (the action chunk carries
`gain * (xyz_target - xyz_proprio)` instead of `xyz_target`).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import runpy
import sys
from pathlib import Path

import xvla_bridge as XB

PATCH = Path(__file__).resolve().parent / "xvla_patch_v3"

if __name__ == "__main__":
    XB.add_xvla_to_path()          # provides `models.*`
    sys.path.insert(0, str(PATCH))  # provides the v3 `datasets` package
    sys.argv = [str(PATCH / "train_cce.py")] + sys.argv[1:]
    runpy.run_path(str(PATCH / "train_cce.py"), run_name="__main__")
