# ------------------------------------------------------------------------------
# CCE addition (idea_v2). Not part of upstream X-VLA.
# ------------------------------------------------------------------------------

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image

from ..utils import quat_to_rotate6d
from .base import BaseHDF5Handler

# --- CCE variant v3: relative-xyz action parameterization ---------------------
# The action chunk carries `gain * (xyz_target[k] - xyz_proprio)` instead of
# `xyz_target[k]`; rotation and gripper stay absolute and `proprio` is untouched.
# The mechanism is upstream's own `idx_for_delta` (datasets/utils.py:action_slice,
# used by `lerobotv21.py` and `x2robot.py` for their joint channels); the fixed
# gain is the CCE addition, so that the relative targets keep the numeric range
# the absolute ones had (measured RMS ratio 7.07 on domain 19 and 5.68 on domain
# 20, see cce/results/v3_relxyz_stats.json).
#
# Only the first arm's xyz is listed. The second 10-d block (10..19) is
# identically zero for these single-arm plants, so relative and absolute
# coincide there and leaving it out keeps the transform exactly invertible.
REL_XYZ_IDX = [0, 1, 2]
REL_XYZ_GAIN = 6.0


class ManiSkillEE6DHandler(BaseHDF5Handler):
    """ManiSkill nominal-plant domains (`maniskill-panda`, `maniskill-xarm6`).

    HDF5 layout written by `cce/collect_demos.py`:
      /rgb            uint8 [T, H, W, 3]  base_camera frames, already RGB
      /tcp_root_pose  f32   [T, 7]        xyz + quat(w,x,y,z) in the robot root frame
      /grip_closed    f32   [T, 1]        1.0 == closed (X-VLA gripper convention)

    Single arm, so the second 10-d block of the 20-d ee6d vector is zeros, the
    same way `robomind-franka` packs a one-arm plant.

    Timing: `freq` is the ManiSkill control frequency (20 Hz) and `qdur` is the
    chunk duration in seconds. It must equal num_actions / freq (30 / 20 = 1.5 s)
    so that one predicted action grid step == one control step at execution time,
    which is the same relation LIBERO uses upstream (30 actions, 30 Hz, 1.0 s).
    """

    dataset_name = "maniskill-*"

    # --- images -------------------------------------------------------------
    @staticmethod
    def _pil_from_arr(arr: Any) -> Image.Image:
        # Frames are stored decoded and in RGB order, so bypass the cv2 (BGR)
        # decode path that `BaseHDF5Handler` uses for byte-encoded datasets.
        if isinstance(arr, Image.Image):
            return arr
        return Image.fromarray(np.ascontiguousarray(np.asarray(arr, np.uint8)))

    def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
        keys: Sequence[str] = self.meta.get("observation_key", ["rgb"])
        return [f[k][()] for k in keys]

    def read_instruction(self, f: h5py.File) -> str:
        key: str = self.meta.get("language_instruction_key", "language_instruction")
        v = f[key][()]
        if isinstance(v, (bytes, bytearray)):
            return v.decode()
        if isinstance(v, np.ndarray) and v.dtype.kind in "SO":
            return v.reshape(-1)[0].decode()
        return str(v)

    # --- kinematics ---------------------------------------------------------
    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq = float(self.meta.get("freq", f.attrs.get("control_freq", 20.0)))
        qdur = float(self.meta.get("qdur", 30.0 / freq))
        pose = np.asarray(f["tcp_root_pose"][()], np.float64)          # [T, 7]
        closed = np.asarray(f["grip_closed"][()], np.float64).reshape(-1, 1)
        rot6d = quat_to_rotate6d(pose[:, 3:7], scalar_first=True)      # SAPIEN is wxyz
        left = np.concatenate([pose[:, :3], rot6d, closed], axis=-1)   # [T, 10]
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        # Same rule as the upstream sim handlers: leave one full chunk of margin
        # so the linspace query window is never compressed near the episode end.
        margin = int(self.meta.get("index_margin", 30))
        return range(0, max(0, T_left - margin))

    # --- CCE v3: tag every sample with the relative-xyz contract --------------
    def iter_episode(self, traj_idx: int, **kwargs) -> Iterable[dict]:
        """Base iteration, plus the `idx_for_delta` / `scale_for_delta` keys that
        `datasets/dataset.py` hands to `action_slice`.

        `rel_xyz` defaults to True in this package: a meta written by
        `cce/build_metas_v3.py` states it explicitly, and a meta that predates v3
        (e.g. `data/cce/metas_heldout`, reused verbatim for G0' so the number
        stays comparable with the mainline run) still gets the v3 target layout.
        Set `"rel_xyz": false` in a meta to fall back to absolute targets.
        """
        if not bool(self.meta.get("rel_xyz", True)):
            yield from super().iter_episode(traj_idx, **kwargs)
            return
        idx = list(self.meta.get("rel_xyz_idx", REL_XYZ_IDX))
        gain = float(self.meta.get("rel_xyz_gain", REL_XYZ_GAIN))
        for sample in super().iter_episode(traj_idx, **kwargs):
            sample["idx_for_delta"] = idx
            sample["scale_for_delta"] = gain
            yield sample
