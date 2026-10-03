#!/usr/bin/env python
"""Count how many conditioning indices the tail padding actually buys.

`datasets/domain_handler/base.py` yields a training sample for index `idx` only
when both

  (a) `idx in ManiSkillEE6DHandler.index_candidates(T, training)` = `range(0, T - 30)`, and
  (b) the first grid step is not static:
      `not ((lseq[1] - lseq[0]).abs().max() < 1e-5 and (rseq[1] - rseq[0]).abs().max() < 1e-5)`

and the second arm block is identically zero for these single-arm domains, so (b)
reduces to "the left 10-d vector changes between `idx` and `idx + 1`". PickCube
success requires `is_robot_static`, so the expert decelerates at the end and some
of the frames the padding exposes are static anyway. This script reports the real
gain instead of assuming 30 per episode.

Reads only the small float arrays, never the RGB stack.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

MARGIN = 30
NUM_ACTIONS = 30
EPS = 1e-5


def quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    """[T,4] wxyz -> [T,6]. Local copy; only the *magnitude of change* matters here,
    so any fixed choice of the two retained basis vectors gives the same verdict."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3), np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r[:, :, :2].reshape(q.shape[0], 6)


def left_block(f: h5py.File) -> np.ndarray:
    pose = np.asarray(f["tcp_root_pose"][()], np.float64)
    closed = np.asarray(f["grip_closed"][()], np.float64).reshape(-1, 1)
    rot6d = quat_wxyz_to_rot6d(pose[:, 3:7])
    return np.concatenate([pose[:, :3], rot6d, closed], axis=-1)


def survivors(left: np.ndarray, margin: int) -> tuple[int, int]:
    """(#candidate indices, #that survive the static-segment skip)."""
    T = left.shape[0]
    cand = max(0, T - margin)
    if cand == 0:
        return 0, 0
    step = np.abs(left[1:cand + 1] - left[:cand]).max(axis=1)
    return cand, int((step >= EPS).sum())


def main() -> None:
    ap = argparse.ArgumentParser("tail-pad index gain")
    ap.add_argument("--src", action="append", default=None)
    ap.add_argument("--tp", default=None, help="tail-padded dir")
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--pad", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    srcs = [Path(s) for s in (args.src or [str(_bootstrap.DATA / "demos"),
                                           str(_bootstrap.DATA / "demos_extra")])]
    tp = Path(args.tp or (_bootstrap.DATA / "demos_tp"))

    t0 = time.time()
    per_uid: dict = {}
    for d, tag in [(srcs, "orig"), ([tp], "tailpad")]:
        for root in d:
            for p in sorted(root.glob(f"*_{args.task}_seed*.h5")):
                with h5py.File(p, "r") as f:
                    uid = str(f.attrs["robot_uid"])
                    lb = left_block(f)
                cand, surv = survivors(lb, MARGIN)
                slot = per_uid.setdefault(uid, {})
                s = slot.setdefault(tag, {"episodes": 0, "frames": 0,
                                          "candidates": 0, "surviving": 0})
                s["episodes"] += 1
                s["frames"] += int(lb.shape[0])
                s["candidates"] += cand
                s["surviving"] += surv

    for uid, slot in per_uid.items():
        o, t = slot.get("orig"), slot.get("tailpad")
        if o and t:
            slot["gain"] = {
                "candidates_abs": t["candidates"] - o["candidates"],
                "surviving_abs": t["surviving"] - o["surviving"],
                "surviving_ratio": round(t["surviving"] / max(o["surviving"], 1), 4),
                "surviving_per_episode_orig": round(o["surviving"] / max(o["episodes"], 1), 2),
                "surviving_per_episode_tailpad": round(t["surviving"] / max(t["episodes"], 1), 2),
            }

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources_orig": [str(s) for s in srcs],
        "source_tailpad": str(tp),
        "index_margin": MARGIN,
        "static_skip_eps": EPS,
        "pad": args.pad,
        "note": "orig rows include only the 100/arm original demos plus the 200/arm extra "
                "demos; tailpad rows are the padded copies of the same episodes, so the "
                "difference isolates the padding, not the extra data.",
        "per_uid": per_uid,
        "wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
