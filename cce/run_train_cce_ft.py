#!/usr/bin/env python
"""Launcher for cce/xvla_patch_ft/train_cce_ft.py (CCE phase-2 full fine-tune).

Same plumbing as `cce/run_train_cce.py`: the patched `datasets` package under
`cce/xvla_patch` goes ahead of the upstream one on sys.path, and the upstream
X-VLA source provides `models.*`. Neither is modified -- `cce/xvla_patch` is
imported read-only so the phase-2 run sees exactly the phase-1 / LoRA data
pipeline (absolute xyz, `qdur = 1.5 s`, `index_margin = 30`).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import runpy
import sys
from pathlib import Path

import xvla_bridge as XB

HERE = Path(__file__).resolve().parent
PATCH = HERE / "xvla_patch"          # read-only: supplies the patched `datasets`
SCRIPT = HERE / "xvla_patch_ft" / "train_cce_ft.py"

if __name__ == "__main__":
    XB.add_xvla_to_path()            # provides `models.*`
    sys.path.insert(0, str(PATCH))   # provides the patched `datasets` package
    sys.argv = [str(SCRIPT)] + sys.argv[1:]
    runpy.run_path(str(SCRIPT), run_name="__main__")
