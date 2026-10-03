#!/usr/bin/env python
"""Write the X-VLA meta JSONs for the CCE v3 (relative-xyz) run.

Same fields as `cce/build_metas.py`, plus two differences:

  * `--seed-prefix` restricts the episode list to one seed band, so the
    tail-padded corpus in `data/cce/demos_tp/` can be cut back to exactly the
    100 episodes per arm that the mainline LoRA run used (band 20260101+).
  * the relative-xyz contract is written into the meta itself
    (`rel_xyz`, `rel_xyz_gain`, `rel_xyz_idx`) so that the parameterization a
    checkpoint was trained under is recoverable from the data side alone.

The v3 dataset package reads those three keys; every other field is byte-for-byte
the same shape as the mainline metas.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import xvla_bridge as XB

NUM_ACTIONS = 30  # X-VLA-Pt config

# v3 parameterization defaults; must match `xvla_patch_v3/datasets/domain_handler/maniskill.py`.
REL_XYZ_IDX = [0, 1, 2]
REL_XYZ_GAIN = 6.0


def main() -> None:
    ap = argparse.ArgumentParser("build CCE v3 metas (relative xyz)")
    ap.add_argument("--demo-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--index-margin", type=int, default=NUM_ACTIONS)
    ap.add_argument("--seed-prefix", default="seed",
                    help="keep only episodes whose file name carries this seed prefix "
                         "(e.g. seed2026 selects the 20260101+ training band)")
    ap.add_argument("--limit", type=int, default=0, help="cap episodes per domain (0 = all)")
    ap.add_argument("--rel-xyz", type=int, default=1)
    ap.add_argument("--rel-xyz-gain", type=float, default=REL_XYZ_GAIN)
    args = ap.parse_args()

    demo_dir = Path(args.demo_dir or (_bootstrap.DATA / "demos_tp")).resolve()
    out_dir = Path(args.out_dir or (_bootstrap.DATA / "metas_v3")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for uid, ds_name in XB.DATASET_NAMES.items():
        files = sorted(str(p) for p in demo_dir.glob(f"{uid}_{args.task}_{args.seed_prefix}*.h5"))
        if args.limit:
            files = files[: args.limit]
        if not files:
            print(f"[warn] no episodes for {uid} in {demo_dir} (prefix {args.seed_prefix})")
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
            # --- v3 additions -------------------------------------------------
            "rel_xyz": bool(args.rel_xyz),
            "rel_xyz_idx": list(REL_XYZ_IDX),
            "rel_xyz_gain": float(args.rel_xyz_gain),
        }
        p = out_dir / f"{ds_name}.json"
        p.write_text(json.dumps(meta, indent=2) + "\n")
        written[ds_name] = {"file": str(p), "episodes": len(files),
                            "domain_id": meta["domain_id"], "qdur": meta["qdur"],
                            "rel_xyz": meta["rel_xyz"], "rel_xyz_gain": meta["rel_xyz_gain"],
                            "first": Path(files[0]).name, "last": Path(files[-1]).name}
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
