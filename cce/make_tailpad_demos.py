#!/usr/bin/env python
"""Build tail-padded copies of the CCE ManiSkill demonstrations.

Motivation
----------
`ManiSkillEE6DHandler.index_candidates` returns `range(0, T - index_margin)` with
`index_margin = 30`, so on a 78-90 frame episode the last 30 frames never appear
as a *conditioning observation* during training. Those frames are exactly the
"cube already grasped, carrying it to the goal" phase, which is the phase the
phase-1 policy fails at (see docs/reports/CCE_anchor_train_eval.md, §2.2 / §7.2).

Rather than touch the handler (it is in use by a running job), this script
rewrites the data: every episode gets `--pad` extra frames appended that repeat
its final frame verbatim. Physically that is an honest "the arm reached the goal
and holds still": the recorded state is static, the gripper command is
unchanged, and the rendered image would not change either. With the padding in
place, every *real* frame of the original episode becomes a legal conditioning
index under the unchanged `index_margin = 30`, and the query window that starts
inside the carry phase is filled with the honest static tail.

Per-frame datasets (leading axis == T) are repeated; scalars such as
`language_instruction` are copied verbatim. HDF5 attributes are copied verbatim
except that `num_frames` is updated to the true new frame count (leaving it stale
would be a false number in the file), with the original preserved as
`num_frames_orig` and the padding recorded as `tail_pad`.

The RGB stack is copied chunk-by-chunk at the HDF5 filter level
(`read_direct_chunk` / `write_direct_chunk`), so the gzip payload is moved
byte-identically and no re-compression happens; a decode/re-encode fallback is
used if the source layout does not permit it.

Usage
-----
  ./cce/env.sh cce/make_tailpad_demos.py \
      --src data/cce/demos --src data/cce/demos_extra \
      --out data/cce/demos_tp --pad 30 --workers 12
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

# Datasets written by cce/collect_demos.py that carry one row per control tick.
PER_FRAME_KNOWN = ("rgb", "tcp_root_pose", "grip_closed", "grip_cmd", "action_raw", "qpos")
SCALAR_KNOWN = ("language_instruction",)


# --------------------------------------------------------------------------- #
# copying
# --------------------------------------------------------------------------- #
def _chunk_copyable(ds: h5py.Dataset) -> bool:
    """True when the dataset is chunked one-frame-per-chunk, so raw chunks move."""
    ch = ds.chunks
    if ch is None or ds.ndim < 1:
        return False
    if ch[0] != 1:
        return False
    return all(ch[k] == ds.shape[k] for k in range(1, ds.ndim))


def _copy_padded_raw(src: h5py.Dataset, dst_f: h5py.File, name: str, pad: int) -> None:
    """Move gzip chunks verbatim, then repeat the last chunk `pad` times."""
    T = src.shape[0]
    dst = dst_f.create_dataset(
        name,
        shape=(T + pad,) + src.shape[1:],
        dtype=src.dtype,
        chunks=src.chunks,
        compression=src.compression,
        compression_opts=src.compression_opts,
        shuffle=src.shuffle,
    )
    zeros = (0,) * (src.ndim - 1)
    for i in range(T):
        filt, raw = src.id.read_direct_chunk((i,) + zeros)
        dst.id.write_direct_chunk((i,) + zeros, raw, filter_mask=filt)
    filt, raw = src.id.read_direct_chunk((T - 1,) + zeros)
    for j in range(pad):
        dst.id.write_direct_chunk((T + j,) + zeros, raw, filter_mask=filt)


def _copy_padded_array(src: h5py.Dataset, dst_f: h5py.File, name: str, pad: int) -> None:
    """Decode / re-encode fallback and the path used for the small float arrays."""
    arr = np.asarray(src[()])
    tail = np.repeat(arr[-1:], pad, axis=0)
    out = np.concatenate([arr, tail], axis=0)
    kw = {}
    if src.chunks is not None:
        kw["chunks"] = src.chunks
    if src.compression is not None:
        kw["compression"] = src.compression
        kw["compression_opts"] = src.compression_opts
    dst_f.create_dataset(name, data=out, dtype=src.dtype, **kw)


# --------------------------------------------------------------------------- #
# per-episode worker
# --------------------------------------------------------------------------- #
def process_one(src_path_s: str, dst_path_s: str, pad: int, full_rgb_check: bool) -> dict:
    src_path, dst_path = Path(src_path_s), Path(dst_path_s)
    tmp_path = dst_path.with_suffix(".h5.tmp")
    rec: dict = {"file": dst_path.name, "src": str(src_path)}
    raw_copy = {}
    try:
        with h5py.File(src_path, "r") as sf:
            T = int(sf["rgb"].shape[0])
            names = sorted(sf.keys())
            per_frame, scalars, unknown = [], [], []
            for n in names:
                ds = sf[n]
                if ds.ndim >= 1 and ds.shape and int(ds.shape[0]) == T:
                    per_frame.append(n)
                elif ds.ndim == 0:
                    scalars.append(n)
                else:
                    unknown.append(n)
            if unknown:
                raise RuntimeError(f"dataset(s) neither per-frame nor scalar: {unknown}")
            missing = [k for k in PER_FRAME_KNOWN if k not in per_frame]
            if missing:
                raise RuntimeError(f"expected per-frame dataset(s) missing: {missing}")

            if tmp_path.exists():
                tmp_path.unlink()
            with h5py.File(tmp_path, "w") as df:
                for n in per_frame:
                    ds = sf[n]
                    if _chunk_copyable(ds):
                        _copy_padded_raw(ds, df, n, pad)
                        raw_copy[n] = True
                    else:
                        _copy_padded_array(ds, df, n, pad)
                        raw_copy[n] = False
                for n in scalars:
                    df.create_dataset(n, data=sf[n][()], dtype=sf[n].dtype)
                for k, v in sf.attrs.items():
                    df.attrs[k] = v
                df.attrs["num_frames"] = T + pad
                df.attrs["num_frames_orig"] = T
                df.attrs["tail_pad"] = pad
                df.attrs["tail_pad_source"] = str(src_path)

            # ---------------- self-checks ----------------
            with h5py.File(tmp_path, "r") as df:
                chk: dict = {}
                for n in per_frame:
                    if int(df[n].shape[0]) != T + pad:
                        raise RuntimeError(f"{n}: got {df[n].shape[0]} frames, want {T + pad}")
                    if tuple(df[n].shape[1:]) != tuple(sf[n].shape[1:]):
                        raise RuntimeError(f"{n}: trailing shape changed")

                # small float arrays: exact equality on the head, exact repeat on the tail
                for n in ("tcp_root_pose", "grip_closed", "grip_cmd", "action_raw", "qpos"):
                    a, b = np.asarray(sf[n][()]), np.asarray(df[n][()])
                    if not np.array_equal(a, b[:T]):
                        raise RuntimeError(f"{n}: head differs from source")
                    if not np.array_equal(np.repeat(a[-1:], pad, axis=0), b[T:]):
                        raise RuntimeError(f"{n}: tail is not a verbatim repeat of the last frame")

                pose_new = np.asarray(df["tcp_root_pose"][()], np.float64)
                qn = np.linalg.norm(pose_new[:, 3:7], axis=1)
                chk["quat_norm_max_dev"] = float(np.max(np.abs(qn - 1.0)))
                chk["quat_norm_max_dev_src"] = float(
                    np.max(np.abs(np.linalg.norm(
                        np.asarray(sf["tcp_root_pose"][()], np.float64)[:, 3:7], axis=1) - 1.0))
                )
                gs_src = sorted(float(x) for x in np.unique(np.asarray(sf["grip_closed"][()])))
                gs_new = sorted(float(x) for x in np.unique(np.asarray(df["grip_closed"][()])))
                if gs_src != gs_new:
                    raise RuntimeError(f"grip_closed value set changed: {gs_src} -> {gs_new}")
                chk["grip_closed_values"] = gs_new
                chk["grip_cmd_values_equal"] = bool(
                    np.array_equal(np.unique(np.asarray(sf["grip_cmd"][()])),
                                   np.unique(np.asarray(df["grip_cmd"][()]))))

                # rgb: full byte equality on the audited subset, spot checks otherwise
                idx = sorted({0, T // 2, T - 1, T, T + pad - 1})
                last_src = np.asarray(sf["rgb"][T - 1])
                for i in idx:
                    got = np.asarray(df["rgb"][i])
                    want = np.asarray(sf["rgb"][i]) if i < T else last_src
                    if not np.array_equal(got, want):
                        raise RuntimeError(f"rgb frame {i} differs")
                chk["rgb_spot_frames"] = idx
                if full_rgb_check:
                    if not np.array_equal(np.asarray(sf["rgb"][()]), np.asarray(df["rgb"][()])[:T]):
                        raise RuntimeError("rgb head differs from source (full check)")
                    tail = np.asarray(df["rgb"][()])[T:]
                    if not np.array_equal(np.repeat(last_src[None], pad, axis=0), tail):
                        raise RuntimeError("rgb tail is not a verbatim repeat (full check)")
                    chk["rgb_full_check"] = True
                for k in ("robot_uid", "task", "seed", "control_freq", "language_instruction"):
                    if k in sf.attrs and str(df.attrs[k]) != str(sf.attrs[k]):
                        raise RuntimeError(f"attr {k} changed")
                chk["attrs_num_frames"] = int(df.attrs["num_frames"])
                chk["attrs_tail_pad"] = int(df.attrs["tail_pad"])

        if dst_path.exists():
            dst_path.unlink()
        os.replace(tmp_path, dst_path)
        rec.update({
            "ok": True,
            "uid": None,
            "T_src": T,
            "T_dst": T + pad,
            "bytes": dst_path.stat().st_size,
            "raw_chunk_copy": raw_copy,
            "checks": chk,
        })
        with h5py.File(dst_path, "r") as df:
            rec["uid"] = str(df.attrs["robot_uid"])
            rec["seed"] = int(df.attrs["seed"])
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-2000:]})
    return rec


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser("tail-pad the CCE ManiSkill demos")
    ap.add_argument("--src", action="append", default=None,
                    help="source demo dir; repeatable (default: demos + demos_extra)")
    ap.add_argument("--out", default=None, help="output dir (default: data/cce/demos_tp)")
    ap.add_argument("--pad", type=int, default=30, help="frames appended per episode")
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--full-check-every", type=int, default=20,
                    help="every k-th episode gets a full byte-equality RGB audit (0 = none)")
    ap.add_argument("--limit", type=int, default=0, help="cap episodes (debug)")
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    srcs = [Path(s) for s in (args.src or [str(_bootstrap.DATA / "demos"),
                                           str(_bootstrap.DATA / "demos_extra")])]
    out_dir = Path(args.out or (_bootstrap.DATA / "demos_tp"))
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs, seen = [], {}
    for d in srcs:
        if not d.is_dir():
            raise SystemExit(f"source dir not found: {d}")
        for p in sorted(d.glob(f"*_{args.task}_seed*.h5")):
            if p.name in seen:
                raise SystemExit(f"filename collision between {seen[p.name]} and {p}")
            seen[p.name] = p
            jobs.append(p)
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        raise SystemExit(f"no episodes found under {[str(s) for s in srcs]}")

    print(json.dumps({"sources": [str(s) for s in srcs], "out": str(out_dir),
                      "episodes": len(jobs), "pad": args.pad,
                      "workers": args.workers}, indent=2), flush=True)

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for i, p in enumerate(jobs):
            full = bool(args.full_check_every) and (i % args.full_check_every == 0)
            futs[ex.submit(process_one, str(p), str(out_dir / p.name), args.pad, full)] = p
        done = 0
        for fu in as_completed(futs):
            rows.append(fu.result())
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(json.dumps({"done": done, "of": len(jobs),
                                  "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    rows.sort(key=lambda r: r["file"])
    bad = [r for r in rows if not r.get("ok")]
    per_uid: dict = {}
    for r in rows:
        if not r.get("ok"):
            continue
        u = r["uid"]
        d = per_uid.setdefault(u, {"episodes": 0, "frames_src": 0, "frames_dst": 0,
                                   "len_src": [], "bytes": 0,
                                   "quat_norm_max_dev": 0.0, "grip_values": set()})
        d["episodes"] += 1
        d["frames_src"] += r["T_src"]
        d["frames_dst"] += r["T_dst"]
        d["len_src"].append(r["T_src"])
        d["bytes"] += r["bytes"]
        d["quat_norm_max_dev"] = max(d["quat_norm_max_dev"], r["checks"]["quat_norm_max_dev"])
        d["grip_values"].update(r["checks"]["grip_closed_values"])
    for u, d in per_uid.items():
        L = np.asarray(d.pop("len_src"), np.int64)
        d["len_src_min"] = int(L.min())
        d["len_src_max"] = int(L.max())
        d["len_src_mean"] = round(float(L.mean()), 2)
        d["len_src_p50"] = int(np.median(L))
        d["len_dst_min"] = int(L.min()) + args.pad
        d["len_dst_max"] = int(L.max()) + args.pad
        d["len_dst_mean"] = round(float(L.mean()) + args.pad, 2)
        d["grip_values"] = sorted(d["grip_values"])
        d["gb"] = round(d["bytes"] / 2**30, 2)
        d["trainable_index_gain_frames"] = int(d["episodes"]) * args.pad

    n_full = sum(1 for r in rows if r.get("ok") and r["checks"].get("rgb_full_check"))
    n_raw = sum(1 for r in rows if r.get("ok") and all(r["raw_chunk_copy"].get(k, False)
                                                       for k in ("rgb",)))
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": [str(s) for s in srcs],
        "out_dir": str(out_dir),
        "task": args.task,
        "pad": args.pad,
        "episodes_total": len(rows),
        "episodes_ok": len(rows) - len(bad),
        "episodes_failed": len(bad),
        "rgb_raw_chunk_copy_episodes": n_raw,
        "rgb_full_byte_audit_episodes": n_full,
        "wall_s": round(time.time() - t0, 1),
        "per_uid": per_uid,
        "failures": [{k: v for k, v in r.items() if k != "traceback"} for r in bad][:20],
        "protocol": {
            "change": "tail padding of every episode by repeating its final frame",
            "why": "index_margin=30 hides the last 30 real frames from index_candidates; "
                   "padding restores them as conditioning observations without touching "
                   "the running handler",
            "attrs": "copied verbatim except num_frames, which is set to the true new "
                     "frame count; num_frames_orig and tail_pad added",
            "not_changed": "freq=20 Hz, qdur=1.5 s, index_margin=30, chunk length 30",
        },
    }
    sp = Path(args.summary or (out_dir / "summary_tailpad.json"))
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if bad:
        raise SystemExit(f"{len(bad)} episode(s) failed")


if __name__ == "__main__":
    main()
