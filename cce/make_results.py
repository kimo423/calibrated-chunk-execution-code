#!/usr/bin/env python
"""Aggregate CCE artifacts into idea_v2/cce/results/*.json."""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("IDEA_V2_ROOT", "/opt/cce"))
RESULTS = ROOT / "cce" / "results"


def gate0(args) -> None:
    rows = []
    for f in sorted(glob.glob(args.jsonl_glob)):
        for line in Path(f).read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    per_task = {}
    for r in rows:
        d = per_task.setdefault(r["task"], {"task_id": r["task_id"], "n": 0, "succ": 0, "steps": []})
        d["n"] += 1
        d["succ"] += int(r["success"])
        d["steps"].append(r["steps"])
    tasks = {
        k: {
            "task_id": v["task_id"],
            "episodes": v["n"],
            "successes": v["succ"],
            "success_rate": round(v["succ"] / max(v["n"], 1), 4),
            "mean_steps": round(float(np.mean(v["steps"])), 1),
        }
        for k, v in sorted(per_task.items(), key=lambda kv: kv[1]["task_id"])
    }
    n = len(rows)
    succ = int(sum(r["success"] for r in rows))
    info_p = ROOT / "data" / "logs" / "serve_libero_info.json"
    out = {
        "gate": "G0 (LIBERO positive control)",
        "suite": rows[0]["suite"] if rows else None,
        "policy": "X-VLA-Pt + 2toINF/X-VLA-libero-spatial-peft (LoRA merged)",
        "server": json.loads(info_p.read_text()) if info_p.exists() else None,
        "client": "upstream evaluation/libero/libero_client.py (sharded by cce/gate0_libero_client.py)",
        "protocol": {
            "act_type": "abs",
            "domain_id": 3,
            "denoise_steps": 10,
            "eval_horizon": 800,
            "episodes_per_task": 10,
            "init_seed": 42,
            "camera": "agentview(flipped) + eye_in_hand, 256x256, third view zero-padded",
        },
        "episodes": n,
        "successes": succ,
        "success_rate": round(succ / max(n, 1), 4),
        "reference_success_rate": 0.96,
        "per_task": tasks,
        "wall_s_total": round(float(sum(r["wall_s"] for r in rows)), 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "gate0_libero.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_task"}, indent=2))
    print(json.dumps(tasks, indent=2))


def demos(args) -> None:
    import h5py

    demo_dir = Path(args.demo_dir or (_bootstrap.DATA / "demos"))
    summaries = {}
    for p in sorted(demo_dir.glob("summary_*.json")):
        s = json.loads(p.read_text())
        eps = s.pop("episodes", [])
        s["episodes_recorded"] = len(eps)
        summaries[f"{s['uid']}/{s['task']}"] = s
    # verify a couple of files end-to-end
    checks = {}
    for uid in ("panda", "xarm6_robotiq"):
        files = sorted(demo_dir.glob(f"{uid}_*_seed*.h5"))
        if not files:
            continue
        sizes = [p.stat().st_size for p in files]
        with h5py.File(files[0], "r") as f:
            rgb = f["rgb"]
            pose = f["tcp_root_pose"][()]
            closed = f["grip_closed"][()]
            sample = f["rgb"][0]
            checks[uid] = {
                "n_files": len(files),
                "total_bytes": int(sum(sizes)),
                "mean_file_mb": round(float(np.mean(sizes)) / 1024**2, 2),
                "rgb_shape": list(rgb.shape),
                "rgb_dtype": str(rgb.dtype),
                "rgb_mean": round(float(sample.mean()), 2),
                "rgb_min_max": [int(sample.min()), int(sample.max())],
                "tcp_xyz_min": [round(float(x), 4) for x in pose[:, :3].min(0)],
                "tcp_xyz_max": [round(float(x), 4) for x in pose[:, :3].max(0)],
                "quat_norm_max_dev": round(float(np.abs(np.linalg.norm(pose[:, 3:], axis=1) - 1).max()), 8),
                "grip_closed_values": sorted({round(float(x), 4) for x in closed.reshape(-1)}),
                "language_instruction": f["language_instruction"][()].decode(),
                "attrs": {k: (v.item() if hasattr(v, "item") else str(v)) for k, v in f.attrs.items()},
            }
    out = {
        "demo_dir": str(demo_dir),
        "task_coverage_note": (
            "Motion-planning solvers exist for xarm6 on PickCube/PushCube/StackCube/PlugCharger, but "
            "PushCube-v1, StackCube-v1 and PlugCharger-v1 declare SUPPORTED_ROBOTS without xarm6_robotiq "
            "(and PegInsertionSide-v1 only supports panda_wristcam), so PickCube-v1 is the only task both "
            "nominal plants can run. Both domains therefore use PickCube-v1 only, keeping the two rows "
            "task-matched."
        ),
        "per_domain": summaries,
        "file_checks": checks,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "demos_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2)[:4000])


def main() -> None:
    ap = argparse.ArgumentParser("CCE result aggregation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate0")
    g.add_argument("--jsonl-glob", default="/opt/cce/data/logs/gate0_shard*.jsonl")
    g.set_defaults(fn=gate0)
    d = sub.add_parser("demos")
    d.add_argument("--demo-dir", default=None)
    d.set_defaults(fn=demos)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
