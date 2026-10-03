"""Shared math / env helpers for the CCE simulation stack (B2).

Everything lives in the policy action space of ManiSkill3's ``pd_ee_delta_pose``
controller: a 7-vector ``[dx, dy, dz, rx, ry, rz, gripper]`` in shared normalized
units.  Translation unit = 0.1 m, rotation unit = -0.1 rad (ManiSkill rot_lower),
gripper in [-1, 1] with +1 = open.

The quaternion / euler helpers reproduce ManiSkill's (pytorch3d) "XYZ" convention
in pure numpy; ``validate_codec.py`` checks them against the external torch
implementation to 1e-9.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

HZ = 20.0
DT = 1.0 / HZ
POS_SCALE = 0.1
ROT_SCALE = -0.1
MAX_EPISODE_STEPS_DEFAULT = 200

# ---------------------------------------------------------------- quaternions


def qnorm(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("zero quaternion")
    q = q / n
    return -q if q[0] < 0 else q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64) / float(np.dot(q, q))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = qnorm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return qnorm(q)


def euler_xyz_to_mat(e: np.ndarray) -> np.ndarray:
    """R = Rx(e0) @ Ry(e1) @ Rz(e2) -- pytorch3d ``euler_angles_to_matrix(.., 'XYZ')``."""
    a, b, c = (float(v) for v in np.asarray(e, dtype=np.float64).reshape(3))
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    return np.array(
        [
            [cb * cc, -cb * sc, sb],
            [ca * sc + sa * sb * cc, ca * cc - sa * sb * sc, -sa * cb],
            [sa * sc - ca * sb * cc, sa * cc + ca * sb * sc, ca * cb],
        ],
        dtype=np.float64,
    )


def mat_to_euler_xyz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    sb = float(np.clip(m[0, 2], -1.0, 1.0))
    b = math.asin(sb)
    if abs(sb) < 1.0 - 1e-9:
        a = math.atan2(-m[1, 2], m[2, 2])
        c = math.atan2(-m[0, 1], m[0, 0])
    else:  # gimbal lock: fold into a, keep c = 0
        a = math.atan2(m[2, 1], m[1, 1])
        c = 0.0
    return np.array([a, b, c], dtype=np.float64)


def quat_geodesic(qa: np.ndarray, qb: np.ndarray) -> float:
    d = float(abs(np.dot(qnorm(qa), qnorm(qb))))
    return float(2.0 * math.acos(float(np.clip(d, -1.0, 1.0))))


# ------------------------------------------------------ normalized-action codec


def norm_delta(current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """Absolute pose pair -> normalized 6-vector (no bound check, no clip)."""
    cur = np.asarray(current_pose, dtype=np.float64).reshape(7)
    tgt = np.asarray(target_pose, dtype=np.float64).reshape(7)
    dq = quat_mul(qnorm(tgt[3:]), quat_inv(qnorm(cur[3:])))
    euler = mat_to_euler_xyz(quat_to_mat(dq))
    out = np.empty(6, dtype=np.float64)
    out[:3] = (tgt[:3] - cur[:3]) / POS_SCALE
    out[3:] = euler / ROT_SCALE
    return out


def clip_norm6(a6: np.ndarray) -> np.ndarray:
    """ManiSkill controller preprocessing: clip translation, renormalize rotation."""
    a = np.asarray(a6, dtype=np.float64).reshape(6).copy()
    a[:3] = np.clip(a[:3], -1.0, 1.0)
    n = float(np.linalg.norm(a[3:]))
    if n > 1.0:
        a[3:] /= n
    return a


def apply_norm_delta(pose: np.ndarray, a6: np.ndarray) -> np.ndarray:
    """Pose composition performed by the pinned pd_ee_delta_pose controller."""
    p = np.asarray(pose, dtype=np.float64).reshape(7)
    a = clip_norm6(a6)
    dq = mat_to_quat(euler_xyz_to_mat(a[3:] * ROT_SCALE))
    return np.concatenate([p[:3] + a[:3] * POS_SCALE, qnorm(quat_mul(dq, qnorm(p[3:])))])


def pose_error(y: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=np.float64).reshape(7)
    c = np.asarray(c, dtype=np.float64).reshape(7)
    return float(np.linalg.norm(y[:3] - c[:3])), quat_geodesic(y[3:], c[3:])


def slerp(qa: np.ndarray, qb: np.ndarray, f: float) -> np.ndarray:
    a, b = qnorm(qa), qnorm(qb)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    d = float(np.clip(d, -1.0, 1.0))
    if d > 0.9995:
        return qnorm((1 - f) * a + f * b)
    ang = math.acos(d)
    s = math.sin(ang)
    return qnorm(math.sin((1 - f) * ang) / s * a + math.sin(f * ang) / s * b)


def interp_pose_seq(seq: np.ndarray, t: float) -> np.ndarray:
    """Continuous-time lookup into a pose sequence (linear position, slerp rotation)."""
    seq = np.asarray(seq, dtype=np.float64)
    n = seq.shape[0]
    t = float(np.clip(t, 0.0, n - 1))
    i = int(math.floor(t))
    if i >= n - 1:
        return seq[n - 1].copy()
    f = t - i
    return np.concatenate([(1 - f) * seq[i, :3] + f * seq[i + 1, :3], slerp(seq[i, 3:], seq[i + 1, 3:], f)])


# ------------------------------------------------------------------ env access

_ARM_DOF = {"panda": 7, "xarm6_robotiq": 6}


def make_env(uid: str, max_episode_steps: int = MAX_EPISODE_STEPS_DEFAULT):
    """Mirror of ``dev_anchor.make_env`` with a configurable step cap (CPU, no render)."""
    from calibration_to_prompt.dev_headless_urdf import install_headless_urdf_visual_skip

    install_headless_urdf_visual_skip()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    return gym.make(
        "PickCube-v1",
        robot_uids=uid,
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="none",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cpu",
        render_backend="none",
        render_mode=None,
        max_episode_steps=int(max_episode_steps),
    )


def tcp_pose(base: Any) -> np.ndarray:
    from calibration_to_prompt.pilot2_maniskill_q0 import _tcp_root_pose

    return _tcp_root_pose(base)


def gripper_opening(uid: str, base: Any) -> float:
    """Normalized finger opening in [0, 1]; 1 = fully open."""
    q = np.asarray(base.agent.robot.get_qpos()).reshape(-1)
    if uid == "panda":
        return float(np.clip(0.5 * (q[7] + q[8]) / 0.04, 0.0, 1.0))
    if uid == "xarm6_robotiq":
        return float(np.clip(1.0 - q[6] / 0.81, 0.0, 1.0))
    raise ValueError(uid)


def cube_pose_root(base: Any) -> np.ndarray:
    """Cube position expressed in the robot root frame (same frame as tcp_pose)."""
    import sapien

    root_raw = np.asarray(base.agent.robot.pose.raw_pose).reshape(-1)[:7]
    root = sapien.Pose(root_raw[:3], root_raw[3:])
    cube_p = np.asarray(base.cube.pose.p).reshape(-1)[:3]
    return np.asarray((root.inv() * sapien.Pose(cube_p, [1, 0, 0, 0])).p, dtype=np.float64)


def goal_pos_root(base: Any) -> np.ndarray:
    import sapien

    root_raw = np.asarray(base.agent.robot.pose.raw_pose).reshape(-1)[:7]
    root = sapien.Pose(root_raw[:3], root_raw[3:])
    goal_p = np.asarray(base.goal_site.pose.p).reshape(-1)[:3]
    return np.asarray((root.inv() * sapien.Pose(goal_p, [1, 0, 0, 0])).p, dtype=np.float64)


def first_bool(value: Any) -> bool:
    arr = value
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return bool(np.asarray(arr).reshape(-1)[0])


def json_default(obj: Any):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")
