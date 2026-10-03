"""Executor-side inversion for CCE.

Given an intent trajectory ``c`` (absolute TCP poses on the 50 ms control grid,
plus a gripper command series) and a plant estimate ``theta``, produce the
command series ``u`` whose *predicted achieved* trajectory matches ``c``.

Parametric version (ours-P), the three closed-form terms of the design doc:

* phase lead   -- the reference consumed at step ``t`` is ``c[t + d]``, and the
  pose the command will act on is predicted by flushing the ``d`` commands still
  in the plant's pipeline (Smith-style prediction);
* lead compensation -- one-step inverse of the first-order lag,
  ``u = (z_req - alpha * z_pred) / (1 - alpha)``;
* gain pre-compensation -- ``z_req = w_des / g``;
* gripper lead -- the gripper channel is fed ``grip[t + grip_lead]``.

Everything is clipped to the identified per-step limit and to the action bound.
``mode="naive"`` degenerates to ``u = c`` (identity theta).  ``mode="qp"`` solves
a small bounded least-squares problem over a receding horizon instead of using
the closed form.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import _bootstrap  # noqa: F401  (sys.path for cce.*)

from cce.common import DT, POS_SCALE, apply_norm_delta, clip_norm6, norm_delta

ROT_UNIT_RAD = 0.1
ALPHA_CAP = 0.97


@dataclass
class Theta:
    """Plant parameters consumed by the executor (normalized action space)."""

    delay_steps: int = 0
    alpha: np.ndarray = None  # (6,)
    gain: np.ndarray = None  # (6,)
    # Saturation bounds expressed in the ACHIEVED-delta domain (the domain of w),
    # i.e. plant command clip x robot command-to-achieved gain.  The plant clips the
    # command, so a command-domain number must be converted before it lands here.
    clip_pos_norm: float = 1.0
    clip_rot_norm: float = 1.0
    grip_lead_steps: int = 0
    source: str = "identity"

    def __post_init__(self) -> None:
        if self.alpha is None:
            self.alpha = np.zeros(6)
        if self.gain is None:
            self.gain = np.ones(6)
        self.alpha = np.clip(np.asarray(self.alpha, dtype=np.float64).reshape(6), 0.0, ALPHA_CAP)
        self.gain = np.clip(np.asarray(self.gain, dtype=np.float64).reshape(6), 0.2, 3.0)
        self.delay_steps = int(max(0, round(self.delay_steps)))
        self.grip_lead_steps = int(max(0, round(self.grip_lead_steps)))

    # -- constructors -------------------------------------------------------
    @staticmethod
    def identity() -> "Theta":
        return Theta(source="identity")

    @staticmethod
    def from_spec(spec: Any) -> "Theta":
        """Injected ground-truth theta only (no robot-side response model)."""
        return Theta(
            delay_steps=spec.delay_steps,
            alpha=np.full(6, spec.alpha),
            gain=np.full(6, spec.gain),
            clip_pos_norm=min(spec.clip_pos / POS_SCALE, 1.0),
            clip_rot_norm=min(spec.clip_rot / ROT_UNIT_RAD, 1.0),
            grip_lead_steps=spec.grip_delay_steps,
            source="oracle_injected",
        )

    @staticmethod
    def oracle(spec: Any, theta_nominal: "Theta") -> "Theta":
        """oracle-theta arm: ground-truth injected parameters cascaded with the *same*
        robot-side response model that every arm gets from the nominal probe.

        Two first-order lags in series are approximated by one whose time constant is
        the sum; that is the only approximation, and it is shared with ours-P, so the
        ours-P / oracle gap isolates identification error of the injected parameters.
        """
        tau_n = np.array([(-DT / np.log(a)) if a > 1e-6 else 0.0 for a in theta_nominal.alpha])
        tau_o = tau_n + spec.tau_s
        alpha_o = np.array([math.exp(-DT / t) if t > 1e-9 else 0.0 for t in tau_o])
        g_pos = float(np.median(theta_nominal.gain[:3]))
        g_rot = float(np.median(theta_nominal.gain[3:]))
        return Theta(
            delay_steps=spec.delay_steps + theta_nominal.delay_steps,
            alpha=alpha_o,
            gain=theta_nominal.gain * spec.gain,
            clip_pos_norm=min(spec.clip_pos / POS_SCALE, 1.0) * g_pos,
            clip_rot_norm=min(spec.clip_rot / ROT_UNIT_RAD, 1.0) * g_rot,
            grip_lead_steps=spec.grip_delay_steps,
            source="oracle",
        )

    @staticmethod
    def from_hat(hat: dict, r2_gate: float = 0.7, fallback: "Theta" = None,
                 clip_from_burst: bool = False) -> "Theta":
        """Theta from ``identify.identify_plant`` output.

        Axes whose ARX fit falls below the R2 gate are declared non-identifiable and
        inherit the median of the identifiable axes of the same channel (translation or
        rotation); if a whole channel fails, it degenerates to no compensation (g = 1,
        alpha = 0), i.e. that axis is executed naively.
        """
        axes = hat["axes"]
        alpha = np.array([axes[i]["alpha"] for i in range(6)], dtype=np.float64)
        gain = np.array([axes[i]["gain"] for i in range(6)], dtype=np.float64)
        ok = np.array([bool(axes[i]["r2"] >= r2_gate) for i in range(6)])
        subs = []
        for lo, hi in ((0, 3), (3, 6)):
            sel = np.arange(lo, hi)
            good = sel[ok[lo:hi]]
            bad = sel[~ok[lo:hi]]
            if bad.size == 0:
                continue
            subs.extend(int(b) for b in bad)
            if good.size:
                alpha[bad] = float(np.median(alpha[good]))
                gain[bad] = float(np.median(gain[good]))
            elif fallback is not None and lo == 3:
                alpha[bad] = float(np.median(fallback.alpha[3:6]))
                gain[bad] = float(np.median(fallback.gain[3:6]))
            else:
                alpha[bad] = 0.0
                gain[bad] = 1.0
        th = Theta(
            delay_steps=int(hat["delay_steps_agg"]),
            alpha=alpha,
            gain=gain,
            # clip_pos_norm_burst is already an achieved-domain observation (the largest
            # realized step during the saturation burst); the default recipe leaves the
            # bound open at 1.0 because the +-3 cm chirp never saturates anything.
            clip_pos_norm=float(hat.get("clip_pos_norm_burst", 1.0)) if clip_from_burst
            else float(hat.get("clip_pos_norm_hat", 1.0)),
            clip_rot_norm=float(hat.get("clip_rot_norm_hat", 1.0)),
            grip_lead_steps=int(hat["grip_lead_steps_hat"]),
            source="identified_clipburst" if clip_from_burst else "identified",
        )
        th.substituted_axes = subs
        return th

    def as_dict(self) -> dict:
        return {
            "delay_steps": self.delay_steps,
            "alpha": self.alpha.tolist(),
            "gain": self.gain.tolist(),
            "tau_s": [(-DT / np.log(a)) if a > 1e-6 else 0.0 for a in self.alpha],
            "clip_pos_norm": self.clip_pos_norm,
            "clip_rot_norm": self.clip_rot_norm,
            "grip_lead_steps": self.grip_lead_steps,
            "source": self.source,
        }


def shape_reference(c_seq: np.ndarray, theta_nominal: Theta, y0: np.ndarray) -> np.ndarray:
    """Intent (command targets) -> the trajectory the *nominal* robot would achieve.

    The intent chunk ``c`` is a series of absolute TCP targets; the nominal robot does
    not reach them within one control step (its identified command-to-achieved gain is
    well below 1).  Asking a calibrated executor to land exactly on ``c`` would make it
    move far faster than the demonstration ever did, so the executor's set point is the
    nominal system's *predicted achieved* trajectory instead.  On the nominal plant the
    inversion then degenerates to the naive arm, which is the behaviour the recovery
    rate R is defined against.
    """
    c_seq = np.asarray(c_seq, dtype=np.float64)
    n = c_seq.shape[0]
    th = theta_nominal
    r = np.asarray(y0, dtype=np.float64).reshape(7).copy()
    z = np.zeros(6)
    out = np.zeros((n, 7))
    for t in range(n):
        u = np.clip(norm_delta(r, c_seq[t]), -1.0, 1.0)
        z = th.alpha * z + (1.0 - th.alpha) * u
        r = apply_norm_delta(r, _clip_to_limits(th.gain * z, th))
        out[t] = r
    return out


def _clip_to_limits(w: np.ndarray, theta: Theta) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(6).copy()
    w[:3] = np.clip(w[:3], -theta.clip_pos_norm, theta.clip_pos_norm)
    n = float(np.linalg.norm(w[3:]))
    if n > theta.clip_rot_norm:
        w[3:] *= theta.clip_rot_norm / n
    return w


class InversionExecutor:
    """Closed-loop executor: measured pose in, plant command out, one step at a time."""

    def __init__(
        self,
        theta: Theta,
        mode: str = "param",
        kappa: float = 1.0,
        horizon: int = 30,
        lam: float = 0.05,
        mu: float = 0.0,
        resolve_every: int = 5,
    ) -> None:
        self.theta = theta
        self.mode = mode
        self.kappa = float(kappa)
        self.horizon = int(horizon)
        self.lam = float(lam)
        self.mu = float(mu)
        self.resolve_every = int(resolve_every)
        self.reset()

    def reset(self) -> None:
        d = self.theta.delay_steps
        self._buf: deque[np.ndarray] = deque(np.zeros(6) for _ in range(d))
        self._z = np.zeros(6, dtype=np.float64)
        self._qp_cache: np.ndarray | None = None
        self._qp_t0 = -10**9

    # -- model bookkeeping --------------------------------------------------
    def _predict(self, y_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flush the pending pipeline: returns (pose at t+d, lag state z_{t+d-1})."""
        th = self.theta
        zz = self._z.copy()
        yy = np.asarray(y_now, dtype=np.float64).reshape(7).copy()
        for u_old in self._buf:
            zz = th.alpha * zz + (1.0 - th.alpha) * u_old
            yy = apply_norm_delta(yy, _clip_to_limits(th.gain * zz, th))
        return yy, zz

    def _commit(self, u6: np.ndarray) -> None:
        th = self.theta
        self._buf.append(np.asarray(u6, dtype=np.float64).reshape(6))
        if len(self._buf) > th.delay_steps:
            oldest = self._buf.popleft()
            self._z = th.alpha * self._z + (1.0 - th.alpha) * oldest

    # -- policies -----------------------------------------------------------
    def _param_arm(self, y_now: np.ndarray, c_seq: np.ndarray, t: int) -> np.ndarray:
        th = self.theta
        n = c_seq.shape[0]
        y_pred, z_pred = self._predict(y_now)
        target = c_seq[min(t + th.delay_steps, n - 1)]
        w_des = _clip_to_limits(self.kappa * norm_delta(y_pred, target), th)
        z_req = w_des / th.gain
        denom = np.maximum(1.0 - th.alpha, 1e-3)
        u = (z_req - th.alpha * z_pred) / denom
        return np.clip(u, -1.0, 1.0)

    def _qp_arm(self, y_now: np.ndarray, c_seq: np.ndarray, t: int) -> np.ndarray:
        """Receding-horizon bounded least squares on the translation channel."""
        if self._qp_cache is not None and 0 <= t - self._qp_t0 < min(self.resolve_every, self.horizon):
            u = self._qp_cache[t - self._qp_t0].copy()
        else:
            u = self._solve_qp(y_now, c_seq, t)
        # rotation + fallback from the closed form
        u_par = self._param_arm(y_now, c_seq, t)
        out = u_par.copy()
        out[:3] = u[:3]
        return np.clip(out, -1.0, 1.0)

    def _solve_qp(self, y_now: np.ndarray, c_seq: np.ndarray, t: int) -> np.ndarray:
        from scipy.optimize import lsq_linear

        th = self.theta
        H, d, n = self.horizon, th.delay_steps, c_seq.shape[0]
        y_pred, z_pred = self._predict(y_now)  # pose at t+d, z_{t+d-1}

        # free response is already inside y_pred; build cumulative-response matrix
        U = np.zeros((H, 3))
        sols = np.zeros((H, 6))
        for ax in range(3):
            a, g = th.alpha[ax], th.gain[ax]
            hcum = np.array([g * (1.0 - a ** (m + 1)) for m in range(H)])
            A = np.zeros((H, H))
            for k in range(H):
                for j in range(k + 1):
                    A[k, j] = hcum[k - j]
            free = g * a * z_pred[ax] * np.array([(1.0 - a ** (k + 1)) / (1.0 - a) for k in range(H)])
            r = np.array(
                [(c_seq[min(t + d + k, n - 1), ax] - y_pred[ax]) / POS_SCALE for k in range(H)]
            ) - free
            D = np.zeros((H, H))
            for k in range(H):
                D[k, k] = 1.0
                if k > 0:
                    D[k, k - 1] = -1.0
            M = np.vstack([A, np.sqrt(self.lam) * D, np.sqrt(self.mu) * np.eye(H)])
            b = np.concatenate([r, np.zeros(H), np.zeros(H)])
            res = lsq_linear(M, b, bounds=(-1.0, 1.0), max_iter=40, tol=1e-8)
            U[:, ax] = res.x
        sols[:, :3] = U
        self._qp_cache = sols
        self._qp_t0 = t
        return sols[0]

    # -- public API ---------------------------------------------------------
    def step(self, y_now: np.ndarray, c_seq: np.ndarray, grip_seq: np.ndarray, t: int) -> np.ndarray:
        th = self.theta
        n = c_seq.shape[0]
        if self.mode == "qp":
            u6 = self._qp_arm(y_now, c_seq, t)
        else:
            u6 = self._param_arm(y_now, c_seq, t)
        self._commit(u6)
        g_idx = min(t + th.grip_lead_steps, len(grip_seq) - 1)
        out = np.empty(7, dtype=np.float32)
        out[:6] = clip_norm6(u6)
        out[6] = float(np.clip(grip_seq[g_idx], -1.0, 1.0))
        return out

    def invert(
        self,
        c_seq: np.ndarray,
        theta_hat: Theta | None = None,
        y_now: np.ndarray | None = None,
        grip_seq: np.ndarray | None = None,
        horizon: int | None = None,
    ) -> np.ndarray:
        """Open-loop batch inversion: (c_seq, theta_hat, y_now) -> u_seq [H, 7].

        Rolls the internal plant model forward, so no environment is touched.
        Signature matches the interface required by the design doc.
        """
        if theta_hat is not None:
            self.theta = theta_hat
        self.reset()
        c_seq = np.asarray(c_seq, dtype=np.float64)
        n = c_seq.shape[0]
        H = int(horizon) if horizon is not None else n
        if grip_seq is None:
            grip_seq = np.ones(n, dtype=np.float64)
        y = np.asarray(y_now if y_now is not None else c_seq[0], dtype=np.float64).reshape(7).copy()
        th = self.theta
        buf_true: deque[np.ndarray] = deque(np.zeros(6) for _ in range(th.delay_steps))
        z_true = np.zeros(6)
        out = np.zeros((H, 7), dtype=np.float32)
        for t in range(H):
            u = self.step(y, c_seq, grip_seq, t)
            out[t] = u
            # advance the *model* of the plant to get the next predicted pose
            buf_true.append(np.asarray(u[:6], dtype=np.float64))
            applied = np.zeros(6)
            if len(buf_true) > th.delay_steps:
                z_true = th.alpha * z_true + (1.0 - th.alpha) * buf_true.popleft()
                applied = _clip_to_limits(th.gain * z_true, th)
            y = apply_norm_delta(y, applied)
        return out


def build_arm(method: str, spec: Any, hat: dict | None, theta_nominal: Theta,
              clip_from_burst: bool = False, **kw):
    """(executor, uses_shaped_reference) for one comparison arm.

    naive / slow*  -> identity theta, set point = the intent itself.
    ours_p / oracle / qp -> plant model, set point = the nominal-shaped reference.
    """
    if method == "naive" or method.startswith("slow"):
        return InversionExecutor(Theta.identity(), mode="param"), False
    if method == "oracle":
        return InversionExecutor(Theta.oracle(spec, theta_nominal), mode="param", **kw), True
    if method == "ours_p":
        return InversionExecutor(Theta.from_hat(hat, fallback=theta_nominal,
                                               clip_from_burst=clip_from_burst), mode="param", **kw), True
    if method == "ours_qp":
        return InversionExecutor(Theta.from_hat(hat, fallback=theta_nominal,
                                               clip_from_burst=clip_from_burst), mode="qp", **kw), True
    raise ValueError(method)
