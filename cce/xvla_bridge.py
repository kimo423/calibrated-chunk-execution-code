"""Shared X-VLA access + EE6D <-> ManiSkill pose codec for the CCE carrier.

Everything here is self-contained on purpose: the external CalibrationToPrompt
package is read-only and still moving, so the load-bearing conversions are
re-implemented (and round-trip tested in `selftest`) rather than imported.

Conventions pinned here
-----------------------
* rot6d layout follows X-VLA `datasets/utils.py`: ``R[..., :, :2].reshape(6)``,
  i.e. the interleaved row-major order [m00, m01, m10, m11, m20, m21].
  `rot6d_to_quat` is the exact inverse used by X-VLA (`rotate6d_to_quat`).
* quaternions are SAPIEN/ManiSkill order (w, x, y, z).
* ee6d action vector is 20-d: first arm = xyz(3) + rot6d(6) + gripper(1),
  second arm = zeros. gripper 1.0 == closed (X-VLA convention).
* ManiSkill `pd_ee_delta_pose` normalized action semantics are inverted with the
  official controller scales (pos +/-0.1 m, rot -0.1 rad).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root on sys.path)

import sys
from pathlib import Path

import numpy as np

XVLA_SRC = Path("/opt/CalibrationToPrompt/source/X-VLA")
XVLA_PT = Path("/opt/CalibrationToPrompt/models/x-vla-pt-c1c4a64a")

# Official Panda PDEEPoseControllerConfig limits (shared by both plants here).
POS_LIMIT_M = 0.1
ROT_LIMIT_RAD = -0.1

# --------------------------------------------------------------------------- #
# CCE new-domain registration table
#   rows 10-17 of X-VLA-Pt are the trained ones; 19/20 are untouched rows.
# --------------------------------------------------------------------------- #
DOMAIN_IDS = {"panda": 19, "xarm6_robotiq": 20}
DATASET_NAMES = {"panda": "maniskill-panda", "xarm6_robotiq": "maniskill-xarm6"}
TASK_INSTRUCTIONS = {
    "PickCube-v1": "pick up the cube and move it to the goal position",
    "StackCube-v1": "stack the red cube on the green cube",
    "PushCube-v1": "push the cube to the goal region",
}

# ManiSkill gripper action semantics per plant.
#   panda:          [-1, 1], +1 open, -1 closed
#   xarm6_robotiq:  [0, 0.81], 0 open, 0.81 closed
XARM_GRIP_CLOSED = 0.81


def add_xvla_to_path() -> str:
    s = str(XVLA_SRC)
    if s not in sys.path:
        sys.path.insert(0, s)
    return s


# --------------------------------------------------------------------------- #
# rot6d codec (X-VLA interleaved layout)
# --------------------------------------------------------------------------- #
def quat_wxyz_to_matrix(q) -> np.ndarray:
    q = np.asarray(q, np.float64).reshape(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(m) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    q = R.from_matrix(np.asarray(m, np.float64)).as_quat()  # xyzw
    out = np.empty(4, np.float64)
    out[0] = q[3]
    out[1:] = q[:3]
    n = np.linalg.norm(out)
    return out / (n + 1e-12)


def matrix_to_rot6d(m) -> np.ndarray:
    """X-VLA layout: take the first two columns, flatten row-major."""
    m = np.asarray(m, np.float64)
    return m[..., :, :2].reshape(m.shape[:-2] + (6,))


def rot6d_to_matrix(v6) -> np.ndarray:
    """Exact inverse of `matrix_to_rot6d` (mirrors X-VLA `rotate6d_to_quat`)."""
    v6 = np.asarray(v6, np.float64)
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def quat_wxyz_to_rot6d(q) -> np.ndarray:
    return matrix_to_rot6d(quat_wxyz_to_matrix(q))


def rot6d_to_quat_wxyz(v6) -> np.ndarray:
    return matrix_to_quat_wxyz(rot6d_to_matrix(v6))


# --------------------------------------------------------------------------- #
# ee6d 20-d packing
# --------------------------------------------------------------------------- #
def grip_cmd_to_closed(uid: str, g: float) -> float:
    """Plant gripper command -> X-VLA closed-ness in [0, 1] (1 == closed)."""
    g = float(g)
    if uid == "xarm6_robotiq":
        return float(np.clip(g / XARM_GRIP_CLOSED, 0.0, 1.0))
    return float(np.clip(0.5 * (1.0 - g), 0.0, 1.0))


def closed_to_shared_gripper(closed: float) -> float:
    """X-VLA closed-ness -> shared [-1, 1] command (+1 open, -1 closed)."""
    return float(np.clip(1.0 - 2.0 * float(closed), -1.0, 1.0))


def to_plant_action(uid: str, action7) -> np.ndarray:
    """Shared 7-d [-1,1] EE-delta action -> the plant's own action range."""
    out = np.asarray(action7, np.float32).copy()
    if uid == "xarm6_robotiq":
        out[-1] = np.float32(0.5 * (1.0 - float(out[-1])) * XARM_GRIP_CLOSED)
    return out


def pose7_to_ee6d20(pose7, closed: float) -> np.ndarray:
    """(xyz + quat_wxyz, closed-ness) -> 20-d ee6d row (second arm zeros)."""
    p = np.asarray(pose7, np.float64).reshape(7)
    a = np.zeros(20, np.float32)
    a[0:3] = p[:3]
    a[3:9] = quat_wxyz_to_rot6d(p[3:])
    a[9] = float(np.clip(closed, 0.0, 1.0))
    return a


def ee6d20_to_pose7_closed(a20):
    a = np.asarray(a20, np.float64).reshape(-1)[:20]
    pose = np.zeros(7, np.float64)
    pose[:3] = a[:3]
    pose[3:] = rot6d_to_quat_wxyz(a[3:9])
    return pose, float(a[9])


# --------------------------------------------------------------------------- #
# ManiSkill runtime helpers
# --------------------------------------------------------------------------- #
def np_of(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def tcp_root_pose(base) -> np.ndarray:
    """TCP pose expressed in the robot root frame: [xyz, quat_wxyz]."""
    import sapien

    root_raw = np_of(base.agent.robot.pose.raw_pose).reshape(-1, 7)[0]
    tcp_raw = np_of(base.agent.tcp.pose.raw_pose).reshape(-1, 7)[0]
    rel = sapien.Pose(root_raw[:3], root_raw[3:]).inv() * sapien.Pose(tcp_raw[:3], tcp_raw[3:])
    out = np.concatenate([rel.p, rel.q]).astype(np.float64)
    if out.shape != (7,) or not np.all(np.isfinite(out)):
        raise RuntimeError("TCP root-frame pose is invalid")
    return out


def _norm_quat_wxyz(q) -> np.ndarray:
    q = np.asarray(q, np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        raise ValueError("degenerate quaternion")
    q = q / n
    return q if q[0] >= 0 else -q


def pose_to_normalized_ee6(current_pose7, target_pose7, clip: bool = True) -> np.ndarray:
    """Invert the pinned PDEEPoseController: (current, target) -> 6-d in [-1,1]."""
    import torch
    from mani_skill.utils.geometry.rotation_conversions import (
        matrix_to_euler_angles,
        quaternion_invert,
        quaternion_multiply,
        quaternion_to_matrix,
    )

    cur = np.asarray(current_pose7, np.float64).reshape(7)
    tgt = np.asarray(target_pose7, np.float64).reshape(7)
    cq = torch.as_tensor(_norm_quat_wxyz(cur[3:]), dtype=torch.float64).reshape(1, 4)
    tq = torch.as_tensor(_norm_quat_wxyz(tgt[3:]), dtype=torch.float64).reshape(1, 4)
    dq = quaternion_multiply(tq, quaternion_invert(cq))
    euler = matrix_to_euler_angles(quaternion_to_matrix(dq), "XYZ")[0].cpu().numpy()
    act = np.concatenate([(tgt[:3] - cur[:3]) / POS_LIMIT_M, euler / ROT_LIMIT_RAD])
    if clip:
        act[:3] = np.clip(act[:3], -1, 1)
        n = float(np.linalg.norm(act[3:]))
        if n > 1:
            act[3:] /= n
    return act.astype(np.float32)


def normalized_ee6_to_pose(current_pose7, action6) -> np.ndarray:
    """Reproduce the controller's own preprocess + root-aligned composition."""
    import torch
    from mani_skill.utils.geometry.rotation_conversions import (
        euler_angles_to_matrix,
        matrix_to_quaternion,
        quaternion_multiply,
    )

    cur = np.asarray(current_pose7, np.float64).reshape(7)
    act = np.asarray(action6, np.float64).reshape(6)
    translation = np.clip(act[:3], -1, 1) * POS_LIMIT_M
    rot = act[3:].copy()
    n = float(np.linalg.norm(rot))
    if n > 1:
        rot /= n
    dq = matrix_to_quaternion(
        euler_angles_to_matrix(torch.as_tensor(rot * ROT_LIMIT_RAD, dtype=torch.float64).reshape(1, 3), "XYZ")
    )
    cq = torch.as_tensor(_norm_quat_wxyz(cur[3:]), dtype=torch.float64).reshape(1, 4)
    tq = quaternion_multiply(dq, cq)[0].cpu().numpy()
    return np.concatenate([cur[:3] + translation, _norm_quat_wxyz(tq)])


def shared_ee_delta_action(current_pose7, target_pose7, gripper: float) -> np.ndarray:
    """7-d shared action: normalized EE delta + shared [-1,1] gripper."""
    a = np.empty(7, np.float32)
    a[:6] = pose_to_normalized_ee6(current_pose7, target_pose7, clip=True)
    a[6] = np.float32(np.clip(gripper, -1.0, 1.0))
    if not np.all(np.isfinite(a)) or np.any(np.abs(a) > 1 + 1e-6):
        raise RuntimeError("shared EE adapter emitted an invalid action")
    return a


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def load_xvla(model_path=None, lora_path=None, device: str = "cuda:0", merge_lora: bool = True):
    """Load X-VLA (+ optional PEFT adapter) and its processor."""
    import torch

    add_xvla_to_path()
    from models.modeling_xvla import XVLA  # noqa: E402
    from models.processing_xvla import XVLAProcessor  # noqa: E402

    model_path = str(model_path or XVLA_PT)
    model = XVLA.from_pretrained(model_path, torch_dtype=torch.float32).to(device).to(torch.float32)
    processor = XVLAProcessor.from_pretrained(model_path)
    info = {"model_path": model_path, "lora_path": None, "lora_merged": False}

    if lora_path:
        from peft import PeftModel

        before = model.transformer.soft_prompt_hub.weight.detach().clone()
        peft_model = PeftModel.from_pretrained(model, str(lora_path), torch_dtype=torch.float32).to(device)
        merged = False
        if merge_lora:
            try:
                model = peft_model.merge_and_unload()
                merged = True
            except Exception as exc:  # pragma: no cover - fallback path
                print(f"[warn] merge_and_unload failed ({exc}); keeping the PeftModel wrapper", flush=True)
                model = peft_model
        else:
            model = peft_model
        after = _soft_prompt_weight(model)
        delta = float((after - before).abs().max().item())
        info.update(
            {
                "lora_path": str(lora_path),
                "lora_merged": merged,
                "soft_prompt_max_abs_delta": delta,
            }
        )
        print(f"[lora] merged={merged} soft_prompt_hub max|delta|={delta:.6f}", flush=True)
        if delta < 1e-8:
            print("[warn] adapter did not change soft_prompt_hub -- check modules_to_save wiring", flush=True)

    model = model.to(device).eval()
    return model, processor, info


def unwrap_xvla(model):
    """Return the raw XVLA module regardless of peft wrapping."""
    mod = model
    for _ in range(6):
        if hasattr(mod, "transformer") and hasattr(mod, "action_space"):
            return mod
        nxt = None
        for name in ("base_model", "model"):
            cand = getattr(mod, name, None)
            if cand is not None and cand is not mod:
                nxt = cand
                break
        if nxt is None:
            break
        mod = nxt
    return mod


def _soft_prompt_weight(model):
    """Fetch soft_prompt_hub weight through whatever wrapper peft installed."""
    hub = unwrap_xvla(model).transformer.soft_prompt_hub
    if hasattr(hub, "modules_to_save"):
        hub = hub.modules_to_save[hub.active_adapter]
    return hub.weight.detach()


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest() -> dict:
    rng = np.random.default_rng(0)
    errs = {"rot6d_roundtrip": 0.0, "ee6d_roundtrip": 0.0, "vs_xvla_utils": None}
    for _ in range(200):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        back = rot6d_to_quat_wxyz(quat_wxyz_to_rot6d(q))
        if back[0] < 0:
            back = -back
        errs["rot6d_roundtrip"] = max(errs["rot6d_roundtrip"], float(np.abs(back - q).max()))
        pose = np.concatenate([rng.normal(size=3), q])
        p2, c2 = ee6d20_to_pose7_closed(pose7_to_ee6d20(pose, 1.0))
        if p2[3] < 0:
            p2[3:] = -p2[3:]
        errs["ee6d_roundtrip"] = max(errs["ee6d_roundtrip"], float(np.abs(p2 - pose).max()), abs(c2 - 1.0))
    try:  # cross-check against the upstream implementation when importable
        add_xvla_to_path()
        from datasets.utils import quat_to_rotate6d  # type: ignore

        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        mine = quat_wxyz_to_rot6d(q)
        theirs = quat_to_rotate6d(np.asarray(q), scalar_first=True)
        errs["vs_xvla_utils"] = float(np.abs(mine - theirs).max())
    except Exception as exc:  # pragma: no cover
        errs["vs_xvla_utils"] = f"skipped: {type(exc).__name__}"
    return errs


if __name__ == "__main__":
    import json

    print(json.dumps(selftest(), indent=2))
