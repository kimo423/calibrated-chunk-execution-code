#!/usr/bin/env python
"""LIBERO-side interface plant (D1-D4) for CCE, prereg v2 amendment.

Where it sits
-------------
LIBERO's evaluation stack drives the arm with robosuite's OSC_POSE controller in
*absolute* mode (``robot.controller.use_delta = False``): every 50 ms the policy
hands the controller one absolute end-effector target ``[xyz, axis-angle]`` plus a
hard gripper command in ``{-1, +1}``.  The CCE plant is inserted between the policy
chunk's absolute target and that controller input -- the controller then tracks the
*perturbed* target and cannot absorb the mismatch (prereg v1 "do not report a
positive on an EE-controller-absorbing bed").

Semantics (identical to ``cce/plants.py``)
------------------------------------------
``cce/plants.py`` perturbs a *normalized increment*.  The same algebra is used here:
the absolute target ``c`` is first turned into the increment relative to the
**current measured pose** ``y`` (``cce.common.norm_delta``, translation unit 0.1 m,
rotation unit -0.1 rad), the D1-D4 chain is applied to that increment exactly as in
``PlantWrapper.apply``, and the result is composed back onto ``y`` to give the
absolute target actually handed to the controller.  Per control step (20 Hz):

    command transport delay (FIFO, integer steps)
      -> first-order lag (alpha = exp(-DT/tau))
      -> steady-state gain g              (acts on the increment wrt the measured pose)
      -> per-step displacement clip
    gripper channel: its own (transport + actuation) delay, no filtering

Two deliberate differences from ``cce/plants.py``, both forced by the bed:

* no ManiSkill action bound.  ManiSkill's ``pd_ee_delta_pose`` normalizes the command
  to +-1 (= +-0.1 m / +-0.1 rad) and ``plants.py`` folds that bound into the clip
  (``min(clip_pos / POS_SCALE, 1.0)``).  robosuite's absolute OSC neither scales nor
  clips its input (``osc.py: set_goal``, ``use_delta=False`` branch), so the bound is
  dropped here and the "no clip" plants get ``clip_pos = clip_rot = 1e3``.
* the gripper rest value is -1 (LIBERO/robosuite: -1 open, +1 close), not +1.

``selftest()`` checks the pose codec round-trip, the analytic D1/D2/D3/D4 responses,
and byte-for-byte agreement with ``cce.plants.PlantWrapper`` on the normalized
increment series whenever the clips stay inside the ManiSkill bound.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cce.common import (
    DT,
    POS_SCALE,
    ROT_SCALE,
    euler_xyz_to_mat,
    mat_to_quat,
    norm_delta,
    qnorm,
    quat_mul,
    quat_to_mat,
)
from cce.plants import ROT_UNIT_RAD, PlantSpec

try:  # robosuite is present in the LIBERO env; absent in ctp-e0
    import robosuite.utils.transform_utils as _T
except Exception:  # pragma: no cover
    _T = None

EPS_R6D = 1e-6  # upstream libero_client.py EPS, reproduced bit for bit
NO_CLIP_M = 1.0e3  # "clip disabled" in metres
NO_CLIP_RAD = 1.0e3  # "clip disabled" in radians
GRIP_OPEN = -1.0  # LIBERO gripper action convention
GRIP_CLOSE = 1.0

NOMINAL_PID = "L00_nominal"


# --------------------------------------------------------------------------- #
# pose codec: LIBERO (pos, R) / rot6d / axis-angle  <->  cce pose7 (xyz + wxyz)
# --------------------------------------------------------------------------- #
def rot6d_to_mat(r6) -> np.ndarray:
    """LIBERO block layout [R[:,0], R[:,1]] -> rotation matrix (upstream Gram-Schmidt)."""
    r = np.asarray(r6, dtype=np.float64).reshape(6)
    a1, a2 = r[0:3], r[3:6]
    b1 = a1 / (np.linalg.norm(a1) + EPS_R6D)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + EPS_R6D)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def mat_to_rot6d(R) -> np.ndarray:
    """Inverse of :func:`rot6d_to_mat` (upstream ``Mat_to_Rotate6D``)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return np.concatenate([R[:3, 0], R[:3, 1]])


def mat_to_axisangle(R) -> np.ndarray:
    """Rotation matrix -> axis-angle, through robosuite's own path when available."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    if _T is not None:
        return np.asarray(_T.quat2axisangle(_T.mat2quat(R)), dtype=np.float64)
    q = mat_to_quat(R)
    w = float(np.clip(q[0], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if den < 1e-9:
        return np.zeros(3)
    return q[1:] / den * (2.0 * math.acos(w))


def mat_to_quat_wxyz(R) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z).

    robosuite's ``mat2quat`` is preferred when available so that the pose codec
    orthonormalizes a Gram-Schmidt rot6d matrix exactly the way upstream's action
    path does; otherwise the two disagree by ~2e-6 rad (the size of upstream's own
    ``EPS = 1e-6`` in the Gram-Schmidt denominators).
    """
    if _T is not None:
        q = np.asarray(_T.mat2quat(np.asarray(R, dtype=np.float64).reshape(3, 3)), dtype=np.float64)
        return qnorm(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))
    return mat_to_quat(R)


def pose7_from_pos_mat(pos, R) -> np.ndarray:
    return np.concatenate([np.asarray(pos, dtype=np.float64).reshape(3), mat_to_quat_wxyz(R)])


def pose7_from_row(row10) -> np.ndarray:
    """One policy chunk row ``[xyz(3), rot6d(6), gripper(1)]`` -> absolute pose7."""
    r = np.asarray(row10, dtype=np.float64).reshape(-1)
    return pose7_from_pos_mat(r[:3], rot6d_to_mat(r[3:9]))


def pose7_to_action(pose7, grip) -> np.ndarray:
    """Absolute pose7 + gripper -> the 7-vector robosuite's absolute OSC consumes."""
    p = np.asarray(pose7, dtype=np.float64).reshape(7)
    aa = mat_to_axisangle(quat_to_mat(p[3:]))
    out = np.empty(7, dtype=np.float32)
    out[:3] = p[:3]
    out[3:6] = aa
    out[6] = float(grip)
    return out


def compose_delta(pose7, a6) -> np.ndarray:
    """``cce.common.apply_norm_delta`` without the ManiSkill +-1 action bound."""
    p = np.asarray(pose7, dtype=np.float64).reshape(7)
    a = np.asarray(a6, dtype=np.float64).reshape(6)
    dq = mat_to_quat(euler_xyz_to_mat(a[3:] * ROT_SCALE))
    return np.concatenate([p[:3] + a[:3] * POS_SCALE, qnorm(quat_mul(dq, qnorm(p[3:])))])


# --------------------------------------------------------------------------- #
# the plant
# --------------------------------------------------------------------------- #
@dataclass
class LiberoPlant:
    """D1-D4 between the policy's absolute target and the OSC controller input."""

    spec: PlantSpec
    _buf: deque = field(default_factory=deque, init=False)
    _gbuf: deque = field(default_factory=deque, init=False)
    _z: np.ndarray = field(default_factory=lambda: np.zeros(6), init=False)
    _g_hold: float = field(default=GRIP_OPEN, init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = deque()
        self._gbuf = deque()
        self._z = np.zeros(6, dtype=np.float64)
        self._g_hold = GRIP_OPEN  # LIBERO resets with the gripper open

    # -- normalized-increment domain (the domain of cce/plants.py) ----------
    def apply_u(self, u6, grip) -> tuple[np.ndarray, float]:
        a = np.asarray(u6, dtype=np.float64).reshape(6).copy()
        s = self.spec

        d = s.delay_steps
        if d > 0:
            self._buf.append(a)
            a = self._buf.popleft() if len(self._buf) > d else np.zeros(6)

        if s.alpha > 0.0:
            self._z = s.alpha * self._z + (1.0 - s.alpha) * a
            a = self._z.copy()

        a = s.gain * a

        cp = s.clip_pos / POS_SCALE
        a[:3] = np.clip(a[:3], -cp, cp)
        cr = s.clip_rot / ROT_UNIT_RAD
        n = float(np.linalg.norm(a[3:]))
        if n > cr:
            a[3:] *= cr / n

        g = float(grip)
        gd = s.grip_delay_steps
        if gd > 0:
            self._gbuf.append(g)
            if len(self._gbuf) > gd:
                self._g_hold = float(self._gbuf.popleft())
            g = self._g_hold
        return a, g

    # -- absolute-pose domain (what the controller sees) --------------------
    def command(self, y_pose7, u6, grip) -> tuple[np.ndarray, float, np.ndarray]:
        """(measured pose, commanded increment, gripper) -> (target pose, gripper, applied increment)."""
        w, g = self.apply_u(u6, grip)
        return compose_delta(y_pose7, w), g, w


# --------------------------------------------------------------------------- #
# plant specs
# --------------------------------------------------------------------------- #
def spec(pid, family, d_ms=0.0, tau_ms=0.0, gain=1.0, clip_pos=NO_CLIP_M,
         clip_rot=NO_CLIP_RAD, grip_delay_s=0.0, note="") -> PlantSpec:
    return PlantSpec(pid=pid, family=family, d_ms=d_ms, tau_ms=tau_ms, gain=gain,
                     clip_pos=clip_pos, clip_rot=clip_rot, grip_delay_s=grip_delay_s, note=note)


NOMINAL = spec(NOMINAL_PID, "nominal", note="identity wrapper (LIBERO official execution chain)")

# ---- L3 screening set: the strongest candidate level of each factor -------- #
SCREEN: list[PlantSpec] = [
    NOMINAL,
    spec("D1_d250", "D1", d_ms=250, note="pure transport delay 250 ms"),
    spec("D2_t400", "D2t", tau_ms=400, note="first-order lag tau = 400 ms"),
    spec("D2_g70", "D2g", gain=0.70, note="under-responding gain 0.70"),
    spec("MX02", "MIX", d_ms=250, tau_ms=250, gain=0.70, grip_delay_s=1.0,
         note="mixed D1+D2+D3 (strong)"),
    spec("D4_c05", "D4", clip_pos=0.05, note="per-step displacement clip 0.05 m"),
]

# ---- first-pass family (12 plants; frozen after L3, before any arm run) ---- #
FAMILY_V1: list[PlantSpec] = [
    NOMINAL,
    spec("D1_d150", "D1", d_ms=150, note="pure transport delay 150 ms"),
    spec("D1_d250", "D1", d_ms=250, note="pure transport delay 250 ms"),
    spec("D2_t250", "D2t", tau_ms=250, note="first-order lag tau = 250 ms"),
    spec("D2_t400", "D2t", tau_ms=400, note="first-order lag tau = 400 ms"),
    spec("D2_g70", "D2g", gain=0.70, note="under-responding gain 0.70"),
    spec("D2_g85", "D2g", gain=0.85, note="under-responding gain 0.85"),
    spec("D3_g06", "D3", grip_delay_s=0.6, note="gripper actuation delay 0.6 s"),
    spec("D3_g10", "D3", grip_delay_s=1.0, note="gripper actuation delay 1.0 s"),
    spec("MX01", "MIX", d_ms=100, tau_ms=250, gain=0.85, grip_delay_s=0.6, note="mixed D1-D3"),
    spec("MX02", "MIX", d_ms=250, tau_ms=250, gain=0.70, grip_delay_s=1.0, note="mixed D1-D3 (strong)"),
    spec("MX03", "MIX", d_ms=150, tau_ms=100, gain=0.70, grip_delay_s=0.6, note="mixed D1-D3"),
]

BY_ID: dict[str, PlantSpec] = {p.pid: p for p in (SCREEN + FAMILY_V1)}


def get(pid: str) -> PlantSpec:
    return BY_ID[pid]


def select(pids) -> list[PlantSpec]:
    return [BY_ID[p] for p in pids]


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def _rand_pose(rng) -> np.ndarray:
    p = rng.normal(scale=0.3, size=3)
    q = rng.normal(size=4)
    return np.concatenate([p, qnorm(q)])


def selftest(verbose: bool = True) -> dict:
    rng = np.random.default_rng(20260906)
    out: dict[str, float] = {}

    # 1. pose codec round-trips ------------------------------------------------
    e_codec = 0.0
    e_r6d = 0.0
    for _ in range(400):
        y = _rand_pose(rng)
        a = np.concatenate([rng.normal(scale=0.3, size=3), rng.normal(scale=0.3, size=3)])
        c = compose_delta(y, a)
        e_codec = max(e_codec, float(np.abs(norm_delta(y, c) - a).max()))
        R = quat_to_mat(_rand_pose(rng)[3:])
        e_r6d = max(e_r6d, float(np.abs(rot6d_to_mat(mat_to_rot6d(R)) - R).max()))
    out["codec_delta_roundtrip"] = e_codec
    # upstream's Gram-Schmidt divides by (norm + 1e-6), so the rot6d round-trip is
    # only accurate to ~1e-6 by construction; reproduced here on purpose.
    out["rot6d_roundtrip"] = e_r6d
    assert e_codec < 1e-9, e_codec
    assert e_r6d < 5e-6, e_r6d

    # The rot6d -> action leg must give the controller the same goal orientation as
    # upstream (`set_ori = quat2mat(axisangle2quat(action[3:6]))` in osc.py), so the
    # comparison is made on that reconstructed matrix, not on raw axis-angle components.
    if _T is not None:
        e_up = 0.0
        for _ in range(300):
            R = quat_to_mat(_rand_pose(rng)[3:])
            r6 = mat_to_rot6d(R)
            Rg = rot6d_to_mat(r6)  # what upstream feeds its own converter
            mine = pose7_to_action(pose7_from_row(np.concatenate([np.zeros(3), r6, [1.0]])), 1.0)
            theirs = np.asarray(_T.quat2axisangle(_T.mat2quat(Rg)), dtype=np.float64)
            gm = _T.quat2mat(_T.axisangle2quat(np.asarray(mine[3:6], dtype=np.float64)))
            gt = _T.quat2mat(_T.axisangle2quat(theirs))
            e_up = max(e_up, float(np.abs(gm - gt).max()))
        # robosuite's ``mat2quat`` casts to float32, so bit-exact agreement through an
        # extra matrix<->quaternion round trip is not attainable; the residual is the
        # size of upstream's own numerical noise (~1e-6 rad on the goal orientation).
        out["goal_ori_vs_upstream"] = e_up
        assert e_up < 5e-6, e_up

    # target -> pose7 -> action -> back must reproduce the target exactly
    e_act = 0.0
    for _ in range(200):
        R = quat_to_mat(_rand_pose(rng)[3:])
        p7 = pose7_from_row(np.concatenate([rng.normal(size=3), mat_to_rot6d(R), [1.0]]))
        act = pose7_to_action(p7, 1.0)
        if _T is not None:
            R_back = _T.quat2mat(_T.axisangle2quat(act[3:6]))
        else:
            ang = float(np.linalg.norm(act[3:6]))
            if ang < 1e-12:
                R_back = np.eye(3)
            else:
                ax = act[3:6] / ang
                K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                R_back = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
        e_act = max(e_act, float(np.abs(R_back - R).max()))
    out["pose7_to_action_roundtrip"] = e_act
    assert e_act < 1e-6

    # 2. nominal plant is the identity ----------------------------------------
    pl = LiberoPlant(NOMINAL)
    e_id = 0.0
    for _ in range(200):
        y, c = _rand_pose(rng), _rand_pose(rng)
        c = compose_delta(y, np.concatenate([rng.normal(scale=0.2, size=3), rng.normal(scale=0.2, size=3)]))
        cmd, g, _ = pl.command(y, norm_delta(y, c), GRIP_OPEN)
        e_id = max(e_id, float(np.abs(cmd - c).max()))
    out["nominal_identity"] = e_id
    assert e_id < 1e-9

    # 3. D1: pure delay, measured pose held fixed ------------------------------
    y = _rand_pose(rng)
    s = spec("t_d250", "D1", d_ms=250)
    pl = LiberoPlant(s)
    u = np.array([0.3, -0.2, 0.1, 0.05, -0.02, 0.03])
    seq = [pl.apply_u(u, GRIP_CLOSE)[0] for _ in range(12)]
    d = s.delay_steps
    assert d == 5
    assert all(float(np.abs(seq[k]).max()) == 0.0 for k in range(d))
    assert float(np.abs(seq[d] - u).max()) < 1e-15
    out["D1_delay_steps"] = float(d)

    # 4. D2 lag: analytic step response ---------------------------------------
    s = spec("t_t400", "D2t", tau_ms=400)
    pl = LiberoPlant(s)
    a = s.alpha
    err = 0.0
    for k in range(1, 25):
        w = pl.apply_u(u, GRIP_CLOSE)[0]
        err = max(err, float(np.abs(w - (1.0 - a ** k) * u).max()))
    out["D2_lag_step_response"] = err
    assert err < 1e-12

    # 5. D2 gain ---------------------------------------------------------------
    pl = LiberoPlant(spec("t_g70", "D2g", gain=0.70))
    w = pl.apply_u(u, GRIP_CLOSE)[0]
    out["D2_gain"] = float(np.abs(w - 0.70 * u).max())
    assert out["D2_gain"] < 1e-15

    # 6. D4 clip: ramp input saturates at clip_pos / POS_SCALE -----------------
    pl = LiberoPlant(spec("t_c05", "D4", clip_pos=0.05))
    tops = []
    for k in range(1, 15):
        w = pl.apply_u(np.array([0.05 * k, 0.0, 0.0, 0, 0, 0]), GRIP_CLOSE)[0]
        tops.append(float(w[0]))
    out["D4_clip_norm"] = max(tops)
    assert abs(max(tops) - 0.5) < 1e-12  # 0.05 m / 0.1 m per normalized unit

    # 7. D3 gripper timing -----------------------------------------------------
    s = spec("t_grip", "D3", grip_delay_s=0.6)
    pl = LiberoPlant(s)
    lead = s.grip_delay_steps
    assert lead == 12
    gs = [pl.apply_u(np.zeros(6), GRIP_CLOSE if k >= 3 else GRIP_OPEN)[1] for k in range(30)]
    first_close = next(k for k, g in enumerate(gs) if g > 0)
    out["D3_grip_lead_steps"] = float(first_close - 3)
    assert first_close - 3 == lead

    # 8. agreement with cce/plants.py in the normalized-increment domain -------
    from cce.plants import PlantWrapper

    worst = 0.0
    cases = [
        spec("x1", "D1", d_ms=150),
        spec("x2", "D2t", tau_ms=250, clip_pos=0.10, clip_rot=0.10),
        spec("x3", "D2g", gain=0.70, clip_pos=0.05, clip_rot=0.05),
        spec("x4", "MIX", d_ms=100, tau_ms=250, gain=0.85, clip_pos=0.05,
             clip_rot=0.05, grip_delay_s=0.6),
    ]
    for sp in cases:
        # cce/plants.py caps the clip at the ManiSkill action bound; keep the test
        # inside that bound so the two implementations must agree exactly.
        sp_ms = PlantSpec(pid=sp.pid, family=sp.family, d_ms=sp.d_ms, tau_ms=sp.tau_ms,
                          gain=sp.gain, clip_pos=min(sp.clip_pos, 0.10),
                          clip_rot=min(sp.clip_rot, 0.10), grip_delay_s=sp.grip_delay_s)
        mine = LiberoPlant(sp_ms)
        mine._g_hold = 1.0  # test only: match ManiSkill's "+1 = open" rest value
        theirs = PlantWrapper(sp_ms)
        for _ in range(200):
            uu = rng.normal(scale=0.25, size=6)
            gg = float(rng.choice([-1.0, 1.0]))
            w_mine, g_mine = mine.apply_u(uu, gg)
            out_theirs = theirs.apply(np.concatenate([uu, [gg]]))
            worst = max(worst, float(np.abs(w_mine - out_theirs[:6].astype(np.float64)).max()))
            worst = max(worst, abs(g_mine - float(out_theirs[6])))
    out["vs_cce_plants_max_abs_diff"] = worst
    assert worst < 1e-6, worst

    out["ok"] = 1.0
    if verbose:
        print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    selftest()
