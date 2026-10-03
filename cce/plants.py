"""Interface-level plant family (D1-D4) for CCE.

A plant sits between the policy's normalized ``pd_ee_delta_pose`` command and
``env.step``; it never touches the simulator's physics.  Pipeline per control
step (20 Hz), in the order fixed by the design doc:

    command delay (FIFO, integer steps) -> first-order lag -> steady-state gain
    -> per-step clip;  the gripper channel carries its own (transport + actuation)
    delay and is not filtered.

``PlantWrapper.apply`` is a drop-in extension of the student's
``tools/dev_residual_benchmark.py:ResidualPlant``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

import _bootstrap  # noqa: F401  (sys.path for cce.*)

from cce.common import DT, POS_SCALE

ROT_UNIT_RAD = 0.1  # |ROT_SCALE|; one normalized rotation unit


@dataclass(frozen=True)
class PlantSpec:
    pid: str
    family: str  # nominal | D1 | D2g | D2t | D3 | D4 | MIX | OVER
    d_ms: float = 0.0
    tau_ms: float = 0.0
    gain: float = 1.0
    clip_pos: float = 0.10  # metres per control step
    clip_rot: float = 0.10  # radians per control step
    grip_delay_s: float = 0.0
    note: str = ""

    @property
    def delay_steps(self) -> int:
        return int(round(self.d_ms / 1000.0 / DT))

    @property
    def tau_s(self) -> float:
        return float(self.tau_ms) / 1000.0

    @property
    def alpha(self) -> float:
        return math.exp(-DT / self.tau_s) if self.tau_s > 1e-9 else 0.0

    @property
    def grip_delay_steps(self) -> int:
        """Total gripper lead the executor must recover: transport delay + actuation."""
        return self.delay_steps + int(round(self.grip_delay_s / DT))

    def theta(self) -> dict[str, float]:
        return {
            "d_ms": float(self.d_ms),
            "delay_steps": float(self.delay_steps),
            "tau_ms": float(self.tau_ms),
            "tau_s": float(self.tau_s),
            "alpha": float(self.alpha),
            "gain": float(self.gain),
            "clip_pos_m": float(self.clip_pos),
            "clip_rot_rad": float(self.clip_rot),
            "grip_delay_s": float(self.grip_delay_s),
            "grip_lead_steps": float(self.grip_delay_steps),
        }

    def as_dict(self) -> dict[str, object]:
        return {"pid": self.pid, "family": self.family, "note": self.note, **self.theta()}


@dataclass
class PlantWrapper:
    spec: PlantSpec
    _buf: deque = field(default_factory=deque, init=False)
    _gbuf: deque = field(default_factory=deque, init=False)
    _z: np.ndarray = field(default_factory=lambda: np.zeros(6), init=False)
    _g_hold: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = deque()
        self._gbuf = deque()
        self._z = np.zeros(6, dtype=np.float64)
        self._g_hold = 1.0  # gripper starts open at env reset

    def apply(self, action7: np.ndarray) -> np.ndarray:
        """Shared normalized 7-vector command -> shared normalized 7-vector applied."""
        a = np.asarray(action7, dtype=np.float64).reshape(7)
        arm, grip = a[:6].copy(), float(a[6])
        s = self.spec

        # 1. command transport delay (arm)
        d = s.delay_steps
        if d > 0:
            self._buf.append(arm)
            arm = self._buf.popleft() if len(self._buf) > d else np.zeros(6)

        # 2. first-order lag
        if s.alpha > 0.0:
            self._z = s.alpha * self._z + (1.0 - s.alpha) * arm
            arm = self._z.copy()

        # 3. steady-state gain
        arm = s.gain * arm

        # 4. per-step clip (actuator saturation) then action-space bound
        cp = min(s.clip_pos / POS_SCALE, 1.0)
        arm[:3] = np.clip(arm[:3], -cp, cp)
        cr = min(s.clip_rot / ROT_UNIT_RAD, 1.0)
        n = float(np.linalg.norm(arm[3:]))
        if n > cr:
            arm[3:] *= cr / n

        # 5. gripper: transport delay + actuation delay, no filtering
        gd = s.grip_delay_steps
        if gd > 0:
            self._gbuf.append(grip)
            if len(self._gbuf) > gd:
                self._g_hold = float(self._gbuf.popleft())
            grip = self._g_hold

        out = np.empty(7, dtype=np.float32)
        out[:6] = arm
        out[6] = grip
        return out


# ----------------------------------------------------------------- the family

NOMINAL_PID = "P00_nominal"


def _family() -> list[PlantSpec]:
    S = PlantSpec
    out: list[PlantSpec] = [S(NOMINAL_PID, "nominal", note="identity wrapper (control arm / denominator)")]

    # D1 pure command delay
    for d in (50, 100, 150, 250):
        out.append(S(f"D1_d{d}", "D1", d_ms=d, note="pure transport delay"))

    # D2 bandwidth (tau) and steady-state gain
    for t in (100, 250, 400):
        out.append(S(f"D2_t{t}", "D2t", tau_ms=t, note="first-order lag (over-responding: overshoot)"))
    for g in (0.70, 0.85):
        out.append(S(f"D2_g{int(g * 100)}", "D2g", gain=g, note="under-responding gain"))

    # D3 gripper timing
    for tg in (0.2, 0.6, 1.0):
        out.append(S(f"D3_g{int(tg * 10)}", "D3", grip_delay_s=tg, note="gripper actuation delay"))

    # D4 per-step clip
    out.append(S("D4_c05", "D4", clip_pos=0.05, note="half-rate position clip"))

    # over-responding gain (family must span under- and over-response)
    out.append(S("OV_g115", "OVER", gain=1.15, note="over-responding gain"))
    out.append(S("OV_g115_t250_d50", "OVER", d_ms=50, tau_ms=250, gain=1.15, grip_delay_s=0.2,
                 note="over-responding + lag + delay"))

    # mixed plants (frozen combination sample)
    mixes = [
        (50, 100, 0.85, 0.10, 0.2),
        (100, 250, 0.85, 0.10, 0.6),
        (150, 100, 0.70, 0.05, 0.6),
        (100, 400, 1.00, 0.10, 0.2),
        (250, 250, 0.70, 0.05, 1.0),
        (50, 400, 0.70, 0.10, 1.0),
        (150, 250, 1.00, 0.10, 0.6),
        (0, 400, 0.85, 0.05, 0.2),
        (100, 100, 1.00, 0.05, 1.0),
        (250, 100, 0.85, 0.10, 0.2),
    ]
    for i, (d, t, g, cp, tg) in enumerate(mixes, start=1):
        out.append(
            S(f"MX{i:02d}", "MIX", d_ms=d, tau_ms=t, gain=g, clip_pos=cp, grip_delay_s=tg,
              note="mixed D1-D4")
        )
    return out


PLANTS: list[PlantSpec] = _family()
PLANT_BY_ID: dict[str, PlantSpec] = {p.pid: p for p in PLANTS}


def plant_ids() -> list[str]:
    return [p.pid for p in PLANTS]


def get_plant(pid: str) -> PlantSpec:
    return PLANT_BY_ID[pid]


def select(pids: Iterable[str] | None = None) -> list[PlantSpec]:
    if pids is None:
        return list(PLANTS)
    return [PLANT_BY_ID[p] for p in pids]


if __name__ == "__main__":
    import json

    print(json.dumps({"n_plants": len(PLANTS), "plants": [p.as_dict() for p in PLANTS]}, indent=2))
