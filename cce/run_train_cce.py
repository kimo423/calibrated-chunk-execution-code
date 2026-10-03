#!/usr/bin/env python
"""Launcher for cce/xvla_patch/train_cce.py.

Puts the patched `datasets` package ahead of the upstream one on sys.path so the
forked training script stays a verbatim-ish fork (`from datasets import ...`,
`from models... import ...`) with no path plumbing of its own.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import runpy
import sys
from pathlib import Path

import xvla_bridge as XB

PATCH = Path(__file__).resolve().parent / "xvla_patch"

if __name__ == "__main__":
    XB.add_xvla_to_path()          # provides `models.*`
    sys.path.insert(0, str(PATCH))  # provides the patched `datasets` package
    sys.argv = [str(PATCH / "train_cce.py")] + sys.argv[1:]
    runpy.run_path(str(PATCH / "train_cce.py"), run_name="__main__")
