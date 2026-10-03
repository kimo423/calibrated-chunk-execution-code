#!/usr/bin/env python
"""LIBERO Gate 0 client -- thin shard-able wrapper around the official client.

Runs in the cloned LIBERO env (`idea_v2/data/envs/libero`), talks to the X-VLA
policy server started by `cce/serve_xvla.py`. All evaluation semantics (image
flips, absolute controller, proprio bookkeeping, 30-step chunk drain, gripper
threshold, per-task init states) come from the upstream
`evaluation/libero/libero_client.py`; this file only adds task sharding,
JSONL bookkeeping and video suppression so several shards can share one server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

XVLA_LIBERO = "/opt/CalibrationToPrompt/source/X-VLA/evaluation/libero"

# Keep LIBERO's path config inside idea_v2 (never touch the shared ~/.libero).
os.environ.setdefault("LIBERO_CONFIG_PATH", "/opt/cce/data/libero_config")
os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
    ap = argparse.ArgumentParser("LIBERO Gate 0 shard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-ids", default="", help="comma separated; empty = all")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--init-seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=0, help="0 = official per-suite horizon")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-video-dir", default=None)
    args = ap.parse_args()

    sys.path.insert(0, XVLA_LIBERO)
    import libero_client as LC  # noqa: E402

    horizon = args.horizon or LC.LIBERO_DATASETS_HORIZON[args.suite]
    suites = [LC.benchmark_dict[t]() for t in LC.LIBERO_DATASETS[args.suite]]
    policy = LC.ClientModel(args.host, args.port)
    ev = LC.LIBEROEval(
        task_suite_name=args.suite,
        eval_horizon=horizon,
        act_type="abs",
        num_episodes=args.episodes,
        init_seed=args.init_seed,
    )
    ev.base_dir = Path(args.out).parent
    ev.base_dir.mkdir(parents=True, exist_ok=True)
    ev._log_results = lambda metrics: None  # bookkeeping handled below

    wanted = [int(x) for x in args.task_ids.split(",") if x.strip() != ""] if args.task_ids else None
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["suite_idx"], r["task_id"], r["ep"]))
            except Exception:
                pass

    fh = out_path.open("a", encoding="utf-8")
    for si, suite in enumerate(suites):
        ids = range(len(suite.tasks)) if wanted is None else wanted
        for task_id in ids:
            for ep in range(args.episodes):
                if (si, task_id, ep) in done:
                    continue
                t0 = time.time()
                policy.reset()
                env, lang, obs = ev._init_env(suite, task_id, ep)
                frames = []
                success = 0.0
                steps = 0
                try:
                    for _ in range(horizon):
                        obs["robo_ori"] = ev.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
                        obs["robo_pos"] = env.env.robots[0].controller.ee_pos
                        action = policy.step(obs, lang)
                        if args.save_video_dir:
                            frames.append(np.ascontiguousarray(LC._flip_agentview(obs["agentview_image"])))
                        obs, reward, dn, info = env.step(action)
                        steps += 1
                        if dn:
                            success = 1.0
                            break
                finally:
                    env.close()
                if args.save_video_dir and frames:
                    vd = Path(args.save_video_dir)
                    vd.mkdir(parents=True, exist_ok=True)
                    import imageio

                    imageio.mimsave((vd / f"t{task_id}_ep{ep}.mp4").as_posix(), frames, fps=30)
                rec = {
                    "suite": args.suite,
                    "suite_idx": si,
                    "task_id": task_id,
                    "task": lang,
                    "ep": ep,
                    "success": success,
                    "steps": steps,
                    "wall_s": round(time.time() - t0, 2),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                print(json.dumps(rec, ensure_ascii=False), flush=True)
    fh.close()


if __name__ == "__main__":
    main()
