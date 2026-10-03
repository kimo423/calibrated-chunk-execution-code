"""Import first in every cce script: repo root + external CtP src/tools on sys.path (read-only reuse)."""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("IDEA_V2_ROOT", Path(__file__).resolve().parents[1]))
CTP = Path(os.environ.get("CTP_ROOT", "/opt/CalibrationToPrompt"))
for p in (ROOT, CTP / "src", CTP / "tools", CTP):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
os.environ.setdefault("MS_ASSET_DIR", str(CTP / "maniskill-assets"))
os.environ.setdefault("HF_HOME", str(CTP / "cache/huggingface"))
DATA = ROOT / "data" / "cce"
DATA.mkdir(parents=True, exist_ok=True)
