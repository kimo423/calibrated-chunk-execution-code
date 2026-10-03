#!/usr/bin/env python
"""Write the X-VLA meta JSONs for the ManiSkill CCE domains."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import xvla_bridge as XB

NUM_ACTIONS = 30  # X-VLA-Pt config


def main() -> None:
    ap = argparse.ArgumentParser("build CCE metas")
    ap.add_argument("--demo-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--index-margin", type=int, default=NUM_ACTIONS)
    ap.add_argument("--limit", type=int, default=0, help="cap episodes per domain (0 = all)")
    args = ap.parse_args()

    demo_dir = Path(args.demo_dir or (_bootstrap.DATA / "demos"))
    out_dir = Path(args.out_dir or (_bootstrap.DATA / "metas"))
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for uid, ds_name in XB.DATASET_NAMES.items():
        files = sorted(str(p) for p in demo_dir.glob(f"{uid}_{args.task}_seed*.h5"))
        if args.limit:
            files = files[: args.limit]
        if not files:
            print(f"[warn] no episodes for {uid} in {demo_dir}")
            continue
        freq = 20.0
        meta = {
            "dataset_name": ds_name,
            "robot_uid": uid,
            "domain_id": XB.DOMAIN_IDS[uid],
            "task": args.task,
            "datalist": files,
            "observation_key": ["rgb"],
            "language_instruction_key": "language_instruction",
            "freq": freq,
            "qdur": NUM_ACTIONS / freq,
            "index_margin": args.index_margin,
        }
        p = out_dir / f"{ds_name}.json"
        p.write_text(json.dumps(meta, indent=2) + "\n")
        written[ds_name] = {"file": str(p), "episodes": len(files),
                            "domain_id": meta["domain_id"], "qdur": meta["qdur"]}
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
