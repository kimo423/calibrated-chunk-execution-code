#!/usr/bin/env python
"""Inspect qpos layout / gripper opening observable for both uids."""
import _bootstrap  # noqa: F401

import json

import numpy as np


def main() -> None:
    from calibration_to_prompt.dev_anchor import make_env, to_plant_action

    for uid in ("panda", "xarm6_robotiq"):
        env = make_env(uid, "pd_ee_delta_pose")
        env.reset(seed=20262000)
        base = env.unwrapped
        robot = base.agent.robot
        names = [j.name for j in robot.get_active_joints()]
        qpos = np.asarray(robot.get_qpos()).reshape(-1)
        print(json.dumps({"uid": uid, "n_active": len(names), "joints": names,
                          "qpos0": [round(float(v), 4) for v in qpos]}), flush=True)
        # close then open, log tail joints
        a = np.zeros(7, dtype=np.float32)
        traj = []
        for i in range(60):
            a[6] = -1.0 if i < 30 else 1.0
            env.step(to_plant_action(uid, a))
            q = np.asarray(robot.get_qpos()).reshape(-1)
            traj.append([round(float(v), 4) for v in q[-4:]])
        print(json.dumps({"uid": uid, "tail4_t0": traj[0], "t5": traj[5], "t15": traj[15],
                          "t29": traj[29], "t35": traj[35], "t45": traj[45], "t59": traj[59]}), flush=True)
        env.close()


if __name__ == "__main__":
    main()
