#!/usr/bin/env python
"""Is the chunk prediction error sampling variance or bias?

`generate_actions` starts from a fresh `x1 = randn(...)` every call, so two calls
on the same observation give different chunks. This script separates the two
error sources on held-out demonstrations, using the training data pipeline:

  * `steps`  - number of denoising iterations (protocol default 10)
  * `K`      - number of independent samples averaged into one prediction

For each (steps, K) cell it reports the xyz MAE at grid step 1, over grid steps
1-10 and over the full 1-30 chunk, plus the mean across-sample standard
deviation of the xyz prediction. If the error falls roughly as 1/sqrt(K), it is
sampling variance and averaging (or more denoising steps) fixes it for free; if
it plateaus, the error is bias and only more model capacity can remove it.

The VLM encoding is computed once per observation and reused across all cells,
so the sweep costs little more than a single evaluation pass.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import xvla_bridge as XB
from eval_xvla import load_cce_state, load_lora

PATCH = Path(__file__).resolve().parent / "xvla_patch"
POS_IDX = [0, 1, 2]
ROT_IDX = [3, 4, 5, 6, 7, 8]


@torch.no_grad()
def sample_chunks(raw, enc, proprio, domain_id, steps: int, K: int) -> torch.Tensor:
    """`XVLA.generate_actions` with a pre-computed encoding and K parallel draws."""
    encK = {k: v.expand(K, *v.shape[1:]).contiguous() for k, v in enc.items()}
    propK = proprio.expand(K, -1).contiguous()
    domK = domain_id.expand(K).contiguous()
    D = raw.action_space.dim_action
    x1 = torch.randn(K, raw.num_actions, D, device=proprio.device, dtype=proprio.dtype)
    action = torch.zeros_like(x1)
    for i in range(steps, 0, -1):
        t = torch.full((K,), i / steps, device=proprio.device, dtype=proprio.dtype)
        x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        prop_m, x_t_m = raw.action_space.preprocess(propK, x_t)
        action = raw.transformer(domain_id=domK, action_with_noise=x_t_m,
                                 proprio=prop_m, t=t, **encK)
    return raw.action_space.postprocess(action)  # [K, 30, 20]


def main() -> None:
    ap = argparse.ArgumentParser("denoising-steps / sample-averaging sweep")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--model-path", default=str(XB.XVLA_PT))
    ap.add_argument("--metas", default="data/cce/metas_heldout")
    ap.add_argument("--n", type=int, default=24, help="observations per meta file")
    ap.add_argument("--stride", type=int, default=17)
    ap.add_argument("--cells", default="10x1,10x4,10x16,30x1,30x4,50x1",
                    help="comma-separated <steps>x<K> cells")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = []
    for c in args.cells.split(","):
        s, k = c.lower().split("x")
        cells.append((int(s), int(k)))

    model, processor, _ = XB.load_xvla(model_path=args.model_path, device=args.device)
    model, lora_info = load_lora(model, args.lora, args.device)
    ck = load_cce_state(model, args.ckpt)
    raw = XB.unwrap_xvla(model)

    XB.add_xvla_to_path()
    if str(PATCH) not in sys.path:
        sys.path.insert(0, str(PATCH))
    from datasets.dataset import InfiniteDataReader

    p = Path(args.metas)
    metas = sorted(str(x) for x in p.glob("*.json")) if p.is_dir() else [str(p)]

    torch.manual_seed(0)
    out = {"checkpoint": {k: v for k, v in ck.items() if k != "max_abs_delta_vs_base"},
           "lora": lora_info, "metas": args.metas, "cells": args.cells, "per_domain": {}}
    t_start = time.time()

    for m in metas:
        reader = InfiniteDataReader(m, num_actions=30, training=False, action_mode="ee6d")
        acc = {}
        n = seen = 0
        it = iter(reader)
        dom_key = None
        while n < args.n:
            try:
                s = next(it)
            except StopIteration:
                break
            seen += 1
            if (seen - 1) % max(1, args.stride) != 0:
                continue
            dom = int(s["domain_id"])
            dom_key = f"domain_{dom}"
            enc = raw.forward_vlm(
                processor.encode_language([s["language_instruction"]])["input_ids"].to(args.device),
                s["image_input"].unsqueeze(0).to(args.device),
                s["image_mask"].unsqueeze(0).to(args.device),
            )
            proprio = s["proprio"].unsqueeze(0).to(args.device)
            target = s["action"].unsqueeze(0).to(args.device)
            domain_id = torch.tensor([dom], device=args.device)
            for (steps, K) in cells:
                draws = sample_chunks(raw, enc, proprio, domain_id, steps, K)      # [K,30,20]
                pred = draws.mean(0, keepdim=True)                                  # [1,30,20]
                d = acc.setdefault((steps, K), {"n": 0, "g1": 0.0, "g10": 0.0, "g30": 0.0,
                                                "rot30": 0.0, "spread": 0.0})
                e = (pred - target).abs()
                d["n"] += 1
                d["g1"] += float(e[:, :1, POS_IDX].mean())
                d["g10"] += float(e[:, :10, POS_IDX].mean())
                d["g30"] += float(e[:, :, POS_IDX].mean())
                d["rot30"] += float(e[:, :, ROT_IDX].mean())
                d["spread"] += float(draws[..., POS_IDX].std(0).mean()) if K > 1 else 0.0
            # trivial reference on the same observations
            triv = proprio.unsqueeze(1).expand_as(target)
            e = (triv - target).abs()
            d0 = acc.setdefault(("trivial", 0), {"n": 0, "g1": 0.0, "g10": 0.0, "g30": 0.0,
                                                 "rot30": 0.0, "spread": 0.0})
            d0["n"] += 1
            d0["g1"] += float(e[:, :1, POS_IDX].mean())
            d0["g10"] += float(e[:, :10, POS_IDX].mean())
            d0["g30"] += float(e[:, :, POS_IDX].mean())
            d0["rot30"] += float(e[:, :, ROT_IDX].mean())
            n += 1

        rows = []
        for key, d in acc.items():
            nn_ = max(d["n"], 1)
            rows.append({
                "steps": key[0], "K": key[1], "n": d["n"],
                "xyz_mae_g1_m": round(d["g1"] / nn_, 5),
                "xyz_mae_g1_10_m": round(d["g10"] / nn_, 5),
                "xyz_mae_g1_30_m": round(d["g30"] / nn_, 5),
                "rot6d_mae_g1_30": round(d["rot30"] / nn_, 5),
                "xyz_sample_std_m": round(d["spread"] / nn_, 5),
            })
        out["per_domain"][dom_key or m] = rows
        for r in rows:
            print(json.dumps({"domain": dom_key, **r}), flush=True)

    out["wall_s"] = round(time.time() - t_start, 1)
    print(json.dumps(out, indent=2), flush=True)
    if args.out:
        q = Path(args.out)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[diag] wrote {q}", flush=True)


if __name__ == "__main__":
    main()
