#!/usr/bin/env python
"""Collect image-bearing ManiSkill demonstrations for the CCE new-domain rows.

One env, one pass: the official motion-planning solver drives the plant in
`pd_joint_pos` while a wrapper records, at every control tick, the base_camera
RGB frame, the root-frame TCP pose and the commanded gripper. That is exactly
the (image, absolute-EE-trajectory) pairing the X-VLA ee6d handler consumes, so
no separate EE replay pass is needed.

Output: one HDF5 per episode under `data/cce/demos/`, plus a summary JSON.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

import xvla_bridge as XB

SOLVERS = {
    ("panda", "PickCube-v1"): "mani_skill.examples.motionplanning.panda.solutions.pick_cube",
    ("panda", "StackCube-v1"): "mani_skill.examples.motionplanning.panda.solutions.stack_cube",
    ("panda", "PushCube-v1"): "mani_skill.examples.motionplanning.panda.solutions.push_cube",
    ("xarm6_robotiq", "PickCube-v1"): "mani_skill.examples.motionplanning.xarm6.solutions.pick_cube",
    ("xarm6_robotiq", "StackCube-v1"): "mani_skill.examples.motionplanning.xarm6.solutions.stack_cube",
    ("xarm6_robotiq", "PushCube-v1"): "mani_skill.examples.motionplanning.xarm6.solutions.push_cube",
}


def obs_rgb(obs, cam: str) -> np.ndarray:
    rgb = XB.np_of(obs["sensor_data"][cam]["rgb"])
    if rgb.ndim == 4:
        rgb = rgb[0]
    return np.ascontiguousarray(rgb.astype(np.uint8))


def make_env(task: str, uid: str, img: int, max_steps: int):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        task,
        robot_uids=uid,
        num_envs=1,
        obs_mode="rgb",
        reward_mode="none",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
        render_backend="sapien_cuda",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
        sensor_configs=dict(width=img, height=img),
    )


def build_recorder(cam: str):
    import gymnasium as gym

    class Recorder(gym.Wrapper):
        """Records (image, root-frame TCP pose, gripper command) per control tick."""

        def __init__(self, env):
            super().__init__(env)
            self.reset_log()

        def reset_log(self):
            self.rgb = []
            self.pose = []
            self.grip = []
            self.act = []
            self.qpos = []
            self._obs = None

        def reset(self, **kwargs):
            out = super().reset(**kwargs)
            self.reset_log()
            self._obs = out[0]
            return out

        def _snap(self, grip_cmd, action):
            base = self.env.unwrapped
            self.rgb.append(obs_rgb(self._obs, cam))
            self.pose.append(XB.tcp_root_pose(base).astype(np.float32))
            self.grip.append(np.float32(grip_cmd))
            self.act.append(np.asarray(action, np.float32).reshape(-1))
            self.qpos.append(XB.np_of(base.agent.robot.get_qpos()).reshape(-1).astype(np.float32))

        def step(self, action):
            a = np.asarray(action, np.float32).reshape(-1)
            self._snap(float(a[-1]) if a.size else 0.0, a)
            out = super().step(action)
            self._obs = out[0]
            return out

        def finalize(self):
            """Append the terminal observation, holding the last gripper command."""
            if not self.rgb:
                return
            self._snap(float(self.grip[-1]), self.act[-1])

    return Recorder


def write_h5(path: Path, rec, meta: dict) -> None:
    import h5py

    rgb = np.stack(rec.rgb, 0)
    pose = np.stack(rec.pose, 0)
    grip_cmd = np.stack(rec.grip, 0).reshape(-1, 1)
    closed = np.array(
        [[XB.grip_cmd_to_closed(meta["robot_uid"], float(g))] for g in grip_cmd.reshape(-1)], np.float32
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("rgb", data=rgb, compression="gzip", compression_opts=4,
                         chunks=(1, rgb.shape[1], rgb.shape[2], 3))
        f.create_dataset("tcp_root_pose", data=pose)
        f.create_dataset("grip_closed", data=closed)
        f.create_dataset("grip_cmd", data=grip_cmd.astype(np.float32))
        f.create_dataset("action_raw", data=np.stack(rec.act, 0))
        f.create_dataset("qpos", data=np.stack(rec.qpos, 0))
        f.create_dataset("language_instruction", data=np.bytes_(meta["language_instruction"]))
        for k, v in meta.items():
            f.attrs[k] = v


def main() -> None:
    ap = argparse.ArgumentParser("ManiSkill demo collection for CCE")
    ap.add_argument("--uid", required=True, choices=["panda", "xarm6_robotiq"])
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--n", type=int, default=100, help="successful episodes wanted")
    ap.add_argument("--seed-base", type=int, default=20260101)
    ap.add_argument("--max-attempts", type=int, default=0, help="0 = 3x n")
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--camera", default="base_camera")
    ap.add_argument("--fresh-env-every", type=int, default=0, help="rebuild env every k episodes (0=never)")
    args = ap.parse_args()

    import importlib

    out_dir = Path(args.out_dir or (_bootstrap.DATA / "demos"))
    out_dir.mkdir(parents=True, exist_ok=True)
    key = (args.uid, args.task)
    if key not in SOLVERS:
        raise SystemExit(f"no motion-planning solver registered for {key}")
    solve = importlib.import_module(SOLVERS[key]).solve
    instruction = XB.TASK_INSTRUCTIONS[args.task]
    Recorder = build_recorder(args.camera)

    env = make_env(args.task, args.uid, args.img, args.max_steps)
    base = env.unwrapped
    sensors = sorted(base._sensors.keys()) if hasattr(base, "_sensors") else []
    control_freq = int(getattr(base, "control_freq", 20))
    print(json.dumps({"uid": args.uid, "task": args.task, "sensors": sensors,
                      "control_freq": control_freq, "img": args.img}), flush=True)
    if args.camera not in sensors:
        env.close()
        raise SystemExit(f"camera {args.camera!r} not in {sensors}")
    wrapped = Recorder(env)

    max_attempts = args.max_attempts or 3 * args.n
    rows, kept, attempt = [], 0, 0
    t_start = time.time()
    while kept < args.n and attempt < max_attempts:
        seed = args.seed_base + attempt
        attempt += 1
        ok, err = False, None
        try:
            solve(wrapped, seed=seed, debug=False, vis=False)
            ok = bool(np.asarray(XB.np_of(wrapped.unwrapped.evaluate()["success"])).reshape(-1)[0])
        except Exception as exc:  # planner failures are expected on some seeds
            err = f"{type(exc).__name__}: {exc}"
        if ok and len(wrapped.rgb) >= 40:
            wrapped.finalize()
            T = len(wrapped.rgb)
            name = f"{args.uid}_{args.task}_seed{seed}.h5"
            meta = {
                "robot_uid": args.uid,
                "task": args.task,
                "seed": seed,
                "control_freq": control_freq,
                "num_frames": T,
                "image_size": args.img,
                "camera": args.camera,
                "language_instruction": instruction,
                "success": 1,
                "sim_backend": "physx_cpu",
                "render_backend": "sapien_cuda",
                "control_mode": "pd_joint_pos",
                "expert": SOLVERS[key],
            }
            write_h5(out_dir / name, wrapped, meta)
            kept += 1
            rows.append({"seed": seed, "file": name, "T": T, "success": 1})
        else:
            rows.append({"seed": seed, "file": None, "T": len(wrapped.rgb), "success": 0, "error": err})
        if attempt % 10 == 0 or kept == args.n:
            print(json.dumps({"attempt": attempt, "kept": kept,
                              "elapsed_s": round(time.time() - t_start, 1)}), flush=True)
        if args.fresh_env_every and attempt % args.fresh_env_every == 0 and kept < args.n:
            env.close()
            env = make_env(args.task, args.uid, args.img, args.max_steps)
            wrapped = Recorder(env)
    env.close()

    lens = [r["T"] for r in rows if r["success"]]
    summary = {
        "uid": args.uid,
        "task": args.task,
        "requested": args.n,
        "collected": kept,
        "attempts": attempt,
        "expert_success_rate": round(kept / max(attempt, 1), 4),
        "control_freq": control_freq,
        "image_size": [args.img, args.img],
        "camera": args.camera,
        "sensors": sensors,
        "language_instruction": instruction,
        "frames_total": int(sum(lens)),
        "len_min": int(min(lens)) if lens else 0,
        "len_max": int(max(lens)) if lens else 0,
        "len_mean": round(float(np.mean(lens)), 2) if lens else 0.0,
        "len_p50": int(np.median(lens)) if lens else 0,
        "seed_base": args.seed_base,
        "wall_s": round(time.time() - t_start, 1),
        "out_dir": str(out_dir),
        "episodes": rows,
    }
    sp = out_dir / f"summary_{args.uid}_{args.task}.json"
    sp.write_text(json.dumps(summary, indent=2) + "\n")
    brief = {k: v for k, v in summary.items() if k != "episodes"}
    print(json.dumps(brief, indent=2), flush=True)


if __name__ == "__main__":
    main()
