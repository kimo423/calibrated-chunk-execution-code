#!/usr/bin/env python
"""Evaluate a CCE phase-2 *full-model* checkpoint with the unmodified `cce/eval_xvla.py`.

`eval_xvla.py` knows two checkpoint shapes: a three-module domain state
(`--ckpt`) and a peft adapter (`--lora`). A phase-2 checkpoint is neither -- it
is a complete `save_pretrained` directory -- but `eval_xvla.py` already loads
the policy through `XB.load_xvla(model_path=...)`, which is exactly
`XVLA.from_pretrained` + `XVLAProcessor.from_pretrained`. So this wrapper only
validates that the directory really is a loadable checkpoint and forwards it as
`--model-path`; the evaluation protocol, the G0' code and the result schema are
byte-for-byte the ones used for the phase-1, LoRA, v2 and v3 routes.

Usage:
    cce/eval_xvla_ft.py --ft-ckpt data/cce/ckpt/phase2_fullft/ckpt-4000 \
        --uid panda --n 20 --g0-metas data/cce/metas_heldout --out ...
Every other flag is passed straight through to `cce/eval_xvla.py`.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE / "eval_xvla.py"


def main() -> None:
    argv = sys.argv[1:]
    passthrough: list[str] = []
    ckpt: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--ft-ckpt":
            if i + 1 >= len(argv):
                raise SystemExit("--ft-ckpt needs a directory")
            ckpt, i = argv[i + 1], i + 2
            continue
        if a.startswith("--ft-ckpt="):
            ckpt, i = a.split("=", 1)[1], i + 1
            continue
        if a in ("--model-path", "--ckpt", "--lora") or a.startswith(("--model-path=", "--ckpt=", "--lora=")):
            raise SystemExit(f"{a} is incompatible with --ft-ckpt (a phase-2 checkpoint is a whole model)")
        passthrough.append(a)
        i += 1

    if ckpt is None:
        raise SystemExit("usage: eval_xvla_ft.py --ft-ckpt <save_pretrained dir> [eval_xvla.py flags]")

    p = Path(ckpt).resolve()
    if not p.is_dir():
        raise SystemExit(f"not a directory: {p}")
    for f in ("config.json", "preprocessor_config.json", "tokenizer.json"):
        if not (p / f).exists():
            raise SystemExit(f"{p} is not a complete checkpoint: {f} missing")
    weights = sorted(x.name for x in p.iterdir()
                     if x.name.startswith(("model", "pytorch_model")) and x.suffix in (".safetensors", ".bin"))
    if not weights:
        raise SystemExit(f"{p} holds no model weights")
    print(f"[ft] evaluating full phase-2 checkpoint {p} (weights: {weights})", flush=True)

    sys.argv = [str(EVAL), "--model-path", str(p)] + passthrough
    runpy.run_path(str(EVAL), run_name="__main__")


if __name__ == "__main__":
    main()
