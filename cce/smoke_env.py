#!/usr/bin/env python
"""CCE smoke: env creation on CPU for both uids, demo npz inventory, naive nominal replay."""
import _bootstrap  # noqa: F401

import json
import time
from pathlib import Path

import numpy as np

DEMO_DIR = Path("/opt/CalibrationToPrompt/pilot2_runs/dev_route_r_demos")


def main() -> None:
    from calibration_to_prompt.dev_anchor import make_env, to_plant_action
    from calibration_to_prompt.pilot2_maniskill_q0 import _tcp_root_pose

    for uid in ("panda", "xarm6_robotiq"):
        path = DEMO_DIR / f"{uid}_pickcube_d50.npz"
        data = np.load(path)
        seeds = sorted(int(k[4:-8]) for k in data.files if k.endswith("_actions"))
        acts = [data[f"seed{s}_actions"] for s in seeds]
        lens = [a.shape[0] for a in acts]
        print(json.dumps({
            "uid": uid, "n_seeds": len(seeds), "seed_min": seeds[0], "seed_max": seeds[-1],
            "action_shape": list(acts[0].shape), "state_shape": list(data[f"seed{seeds[0]}_states"].shape),
            "len_min": int(min(lens)), "len_max": int(max(lens)), "len_mean": float(np.mean(lens)),
            "grip_uniq": sorted({float(v) for a in acts[:3] for v in a[:, 6]})[:6],
        }), flush=True)

        t0 = time.time()
        try:
            env = make_env(uid, "pd_ee_delta_pose")
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"uid": uid, "make_env": "FAIL", "err": str(exc)[:300]}), flush=True)
            continue
        t_make = time.time() - t0
        env.reset(seed=seeds[0])
        base = env.unwrapped
        pose0 = _tcp_root_pose(base)
        root = np.asarray(base.agent.robot.pose.raw_pose).reshape(-1)[:7]
        tcp_world = np.asarray(base.agent.tcp.pose.p).reshape(-1)[:3]
        print(json.dumps({
            "uid": uid, "make_env_s": round(t_make, 2),
            "action_space": str(env.action_space),
            "tcp_root_pose": [round(float(v), 4) for v in pose0],
            "tcp_world_p": [round(float(v), 4) for v in tcp_world],
            "robot_root": [round(float(v), 4) for v in root],
        }), flush=True)

        # timing + naive nominal replay of first 5 demos
        n_ok, steps_tot, t_step = 0, 0, 0.0
        for s in seeds[:5]:
            a = data[f"seed{s}_actions"]
            env.reset(seed=int(s))
            ok = False
            t1 = time.time()
            for i in range(a.shape[0]):
                _, _, term, _, info = env.step(to_plant_action(uid, a[i]))
                ok = bool(np.asarray(info["success"]).reshape(-1)[0])
                steps_tot += 1
                if ok or bool(np.asarray(term).any()):
                    break
            t_step += time.time() - t1
            n_ok += int(ok)
        print(json.dumps({
            "uid": uid, "naive_nominal_sr_5": n_ok / 5.0,
            "ms_per_step": round(1000.0 * t_step / max(steps_tot, 1), 2),
        }), flush=True)
        env.close()


if __name__ == "__main__":
    main()
