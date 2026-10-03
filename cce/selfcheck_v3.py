#!/usr/bin/env python
"""Self-checks for the CCE v3 (relative-xyz) action parameterization.

Three subcommands:

  stats    Replay the `datasets/domain_handler/base.py` windowing rule directly
           on the HDF5 files listed in a metas dir (no image decode), and report
           the magnitude of the absolute xyz targets against the relative ones.
           This is what fixes the fixed xyz gain of the v3 action layout.

  dump     Run the real training data pipeline (mainline `xvla_patch` or the v3
           `xvla_patch_v3`) with `training=False` -- deterministic order, no
           colour jitter -- and dump `proprio` / `action` for the first N samples.

  compare  Take one mainline dump and one v3 dump of the same metas and verify
           (i)   proprio is untouched,
           (ii)  every non-xyz action channel is untouched,
           (iii) rel_xyz / gain + proprio_xyz reproduces the absolute target
                 element-wise, and report the worst error in metres.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import xvla_bridge as XB

REPO = Path(__file__).resolve().parent.parent
PATCH_MAIN = REPO / "cce" / "xvla_patch"
PATCH_V3 = REPO / "cce" / "xvla_patch_v3"


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def _episode_chunks(path: str, freq: float, qdur: float, margin: int, num_actions: int = 30):
    """Reproduce base.py's query-window construction without decoding images."""
    import h5py
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation as R

    with h5py.File(path, "r") as f:
        pose = np.asarray(f["tcp_root_pose"][()], np.float64)
        closed = np.asarray(f["grip_closed"][()], np.float64).reshape(-1, 1)
    rot6d = R.from_quat(pose[:, 3:7], scalar_first=True).as_matrix()[..., :, :2].reshape(-1, 6)
    left = np.concatenate([pose[:, :3], rot6d, closed], axis=-1)          # [T, 10]
    T = left.shape[0]
    lt = np.arange(T, dtype=np.float64) / float(freq)
    L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
    ref = lt
    for idx in range(0, max(0, T - margin)):
        cur = ref[idx]
        q = np.linspace(cur, min(cur + qdur, float(ref.max())), num_actions + 1, dtype=np.float32)
        lseq = np.asarray(L(q))
        if np.abs(lseq[1] - lseq[0]).max() < 1e-5:   # right block is all-zero, so its
            continue                                  # own diff test is always satisfied
        yield lseq[0], lseq[1:]


def cmd_stats(args) -> None:
    metas = sorted(Path(args.metas).glob("*.json"))
    out = {"metas": str(args.metas), "per_domain": {}}
    for m in metas:
        meta = json.loads(m.read_text())
        freq, qdur = float(meta["freq"]), float(meta["qdur"])
        margin = int(meta.get("index_margin", 30))
        files = meta["datalist"]
        if args.limit_episodes:
            files = files[: args.limit_episodes]
        abs_all, rel_all, first_rel, first_abs = [], [], [], []
        n_chunks = 0
        for p in files:
            for proprio, action in _episode_chunks(p, freq, qdur, margin):
                rel = action[:, 0:3] - proprio[0:3]
                abs_all.append(action[:, 0:3].reshape(-1))
                rel_all.append(rel.reshape(-1))
                first_rel.append(rel[0])
                first_abs.append(action[0, 0:3])
                n_chunks += 1
        a = np.concatenate(abs_all)
        r = np.concatenate(rel_all)
        fr = np.stack(first_rel)
        fa = np.stack(first_abs)
        p0 = np.stack([np.zeros(3)])  # placeholder, unused
        rec = {
            "n_episodes": len(files),
            "n_chunks": n_chunks,
            "abs_xyz": {
                "rms": float(np.sqrt(np.mean(a ** 2))),
                "mean_abs": float(np.mean(np.abs(a))),
                "p50_abs": float(np.percentile(np.abs(a), 50)),
                "p99_abs": float(np.percentile(np.abs(a), 99)),
                "max_abs": float(np.max(np.abs(a))),
            },
            "rel_xyz": {
                "rms": float(np.sqrt(np.mean(r ** 2))),
                "mean_abs": float(np.mean(np.abs(r))),
                "p50_abs": float(np.percentile(np.abs(r), 50)),
                "p99_abs": float(np.percentile(np.abs(r), 99)),
                "max_abs": float(np.max(np.abs(r))),
            },
            "grid1": {
                "rel_l2_median_mm": float(np.median(np.linalg.norm(fr, axis=-1)) * 1000.0),
                "rel_l2_p90_mm": float(np.percentile(np.linalg.norm(fr, axis=-1), 90) * 1000.0),
                "rel_percomp_median_mm": float(np.median(np.abs(fr)) * 1000.0),
                "abs_l2_median_m": float(np.median(np.linalg.norm(fa, axis=-1))),
            },
        }
        rec["rms_ratio_abs_over_rel"] = rec["abs_xyz"]["rms"] / max(rec["rel_xyz"]["rms"], 1e-12)
        out["per_domain"][meta["dataset_name"]] = rec
        del p0
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# dump
# --------------------------------------------------------------------------- #
def cmd_dump(args) -> None:
    XB.add_xvla_to_path()
    patch = PATCH_V3 if args.pkg == "v3" else PATCH_MAIN
    sys.path.insert(0, str(patch))
    from datasets.dataset import InfiniteDataReader  # noqa: E402

    reader = InfiniteDataReader(args.metas, num_actions=30, training=False, action_mode="ee6d")
    pro, act, dom = [], [], []
    for i, s in enumerate(iter(reader)):
        if i >= args.n:
            break
        pro.append(s["proprio"].numpy())
        act.append(s["action"].numpy())
        dom.append(int(s["domain_id"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, proprio=np.stack(pro), action=np.stack(act), domain=np.asarray(dom))
    print(json.dumps({"pkg": args.pkg, "patch": str(patch), "metas": args.metas,
                      "n": len(pro), "out": args.out,
                      "action_shape": list(np.stack(act).shape)}, indent=2))


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def cmd_compare(args) -> None:
    A = np.load(args.a)   # mainline: absolute
    B = np.load(args.b)   # v3: xyz relative x gain
    g = float(args.gain)
    assert A["action"].shape == B["action"].shape, (A["action"].shape, B["action"].shape)

    d_pro = float(np.abs(A["proprio"] - B["proprio"]).max())
    d_dom = int(np.abs(A["domain"] - B["domain"]).max())

    other = [i for i in range(20) if i not in (0, 1, 2)]
    d_other = float(np.abs(A["action"][:, :, other] - B["action"][:, :, other]).max())

    rec_xyz = B["action"][:, :, 0:3] / g + A["proprio"][:, None, 0:3]
    d_xyz = float(np.abs(rec_xyz - A["action"][:, :, 0:3]).max())

    grid1 = np.linalg.norm(B["action"][:, 0, 0:3] / g, axis=-1)
    out = {
        "n": int(A["action"].shape[0]),
        "gain": g,
        "max_abs_proprio_diff": d_pro,
        "max_abs_domain_diff": d_dom,
        "max_abs_non_xyz_action_diff": d_other,
        "max_abs_reconstruction_error_m": d_xyz,
        "reconstruction_pass_1e-6": bool(d_xyz <= 1e-6),
        "proprio_untouched_pass": bool(d_pro == 0.0),
        "non_xyz_untouched_pass": bool(d_other == 0.0),
        "grid1_rel_l2_median_mm": float(np.median(grid1) * 1000.0),
        "grid1_rel_l2_p90_mm": float(np.percentile(grid1, 90) * 1000.0),
        "v3_target_xyz_rms": float(np.sqrt(np.mean(B["action"][:, :, 0:3] ** 2))),
        "mainline_target_xyz_rms": float(np.sqrt(np.mean(A["action"][:, :, 0:3] ** 2))),
        "domains": sorted({int(x) for x in A["domain"]}),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser("CCE v3 self-checks")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats")
    s.add_argument("--metas", default="data/cce/metas_v3")
    s.add_argument("--limit-episodes", type=int, default=0)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_stats)

    d = sub.add_parser("dump")
    d.add_argument("--pkg", choices=["main", "v3"], required=True)
    d.add_argument("--metas", default="data/cce/metas_v3")
    d.add_argument("--n", type=int, default=32)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_dump)

    c = sub.add_parser("compare")
    c.add_argument("--a", required=True, help="mainline (absolute) dump")
    c.add_argument("--b", required=True, help="v3 (relative) dump")
    c.add_argument("--gain", type=float, required=True)
    c.add_argument("--out", default=None)
    c.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
