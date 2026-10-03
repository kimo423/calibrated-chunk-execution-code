#!/usr/bin/env python
"""Executor-side inversion on the LIBERO bed (CCE ours-P / oracle-theta arms).

This is a transcription of ``cce/executor.py`` -- same three closed-form terms
(phase lead by Smith-style pipeline flushing, one-step inverse of the first-order
lag, gain pre-compensation, plus the gripper lead) and the same set point
(``shape_reference``: the trajectory the *nominal* plant is predicted to achieve).
The only changes are the ones the bed forces:

* ``cce.executor`` composes poses with ``cce.common.apply_norm_delta`` and returns
  ``clip_norm6(u)``, both of which impose ManiSkill's +-1 normalized action bound
  (+-0.1 m / +-0.1 rad per control step).  robosuite's absolute OSC has no such
  bound, so ``compose_delta`` (unclipped) is used and the command bound becomes a
  parameter, set high enough never to bind (``ACT_BOUND``).
* ``Theta.from_spec`` / ``Theta.oracle`` cap the saturation bound at the same
  ManiSkill limit; the LIBERO constructors below drop that cap.

``selftest()`` shows the transcription is faithful: with the bound set to 1.0 and
motions inside it, ``LiberoExecutor`` and ``cce.executor.InversionExecutor`` produce
identical command series, and ``shape_reference_libero`` reproduces
``cce.executor.shape_reference``.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import math
from collections import deque

import numpy as np

from cce.common import DT, POS_SCALE, norm_delta, qnorm
from cce.executor import ALPHA_CAP, Theta, _clip_to_limits  # noqa: F401  (shared semantics)
from cce.libero_plant import compose_delta
from cce.plants import ROT_UNIT_RAD, PlantSpec

# Command bound in normalized units.  50 = 5 m / 5 rad per 50 ms step: never binding,
# but still a finite guard against a NaN or a degenerate identification.
ACT_BOUND = 50.0
# "no saturation" in the achieved-delta domain (LIBERO's controller does not clip).
NO_CLIP_NORM = 1.0e4

METHODS = ("naive", "slow15", "slow20", "oracle", "ours_p")


# --------------------------------------------------------------------------- #
# theta constructors (LIBERO bounds)
# --------------------------------------------------------------------------- #
def _open_clips(th: Theta) -> Theta:
    th.clip_pos_norm = NO_CLIP_NORM
    th.clip_rot_norm = NO_CLIP_NORM
    return th


def theta_identity() -> Theta:
    th = _open_clips(Theta.identity())
    th.source = "identity_libero"
    return th


def theta_from_hat(hat: dict, fallback: Theta | None = None) -> Theta:
    """Probe-identified theta.  The per-step clip is left unidentified (as in the
    pre-registered main recipe) and therefore left open on this bed."""
    th = _open_clips(Theta.from_hat(hat, fallback=fallback))
    th.source = "identified_libero"
    return th


def theta_oracle(spec: PlantSpec, theta_nominal: Theta) -> Theta:
    """Ground-truth injected parameters cascaded with the nominal-probe response.

    Same approximation as ``cce.executor.Theta.oracle`` (two first-order lags in
    series -> one lag whose time constant is the sum), shared with ours-P, so the
    ours-P / oracle gap isolates identification error.
    """
    tau_n = np.array([(-DT / math.log(a)) if a > 1e-6 else 0.0 for a in theta_nominal.alpha])
    tau_o = tau_n + spec.tau_s
    alpha_o = np.array([math.exp(-DT / t) if t > 1e-9 else 0.0 for t in tau_o])
    g_pos = float(np.median(theta_nominal.gain[:3]))
    g_rot = float(np.median(theta_nominal.gain[3:]))
    th = Theta(
        delay_steps=spec.delay_steps + theta_nominal.delay_steps,
        alpha=alpha_o,
        gain=theta_nominal.gain * spec.gain,
        clip_pos_norm=(spec.clip_pos / POS_SCALE) * g_pos,
        clip_rot_norm=(spec.clip_rot / ROT_UNIT_RAD) * g_rot,
        grip_lead_steps=spec.grip_delay_steps,
        source="oracle_libero",
    )
    return th


# --------------------------------------------------------------------------- #
# set point shaping
# --------------------------------------------------------------------------- #
def shape_reference_libero(c_seq, theta_nominal: Theta, y0, bound: float = ACT_BOUND) -> np.ndarray:
    """Policy chunk -> the trajectory the nominal plant is predicted to achieve."""
    c_seq = np.asarray(c_seq, dtype=np.float64)
    n = c_seq.shape[0]
    th = theta_nominal
    r = np.asarray(y0, dtype=np.float64).reshape(7).copy()
    z = np.zeros(6)
    out = np.zeros((n, 7))
    for t in range(n):
        u = np.clip(norm_delta(r, c_seq[t]), -bound, bound)
        z = th.alpha * z + (1.0 - th.alpha) * u
        r = compose_delta(r, _clip_to_limits(th.gain * z, th))
        out[t] = r
    return out


# --------------------------------------------------------------------------- #
# executor
# --------------------------------------------------------------------------- #
class LiberoExecutor:
    """Closed-loop executor: measured pose in, normalized command increment out.

    ``bound_pos`` / ``bound_rot`` play the role of ``clip_norm6`` in
    ``cce/executor.py``: translation is clipped element-wise, rotation is limited in
    norm.  On ManiSkill that bound is the controller's own action range (+-0.1 m /
    +-0.1 rad per step); LIBERO's absolute OSC has none, so it becomes an explicit
    safety net against a divergent inversion (the lead filter has gain
    ``1/(1-alpha)``, which reaches 33 when the identified pole sits at the 0.97 cap).
    """

    def __init__(self, theta: Theta, kappa: float = 1.0,
                 bound_pos: float = ACT_BOUND, bound_rot: float = ACT_BOUND) -> None:
        self.theta = theta
        self.kappa = float(kappa)
        self.bound_pos = float(bound_pos)
        self.bound_rot = float(bound_rot)
        self.reset()

    def reset(self) -> None:
        d = self.theta.delay_steps
        self._buf: deque = deque(np.zeros(6) for _ in range(d))
        self._z = np.zeros(6, dtype=np.float64)

    def _predict(self, y_now) -> tuple[np.ndarray, np.ndarray]:
        """Flush the pending pipeline: (pose at t+d, lag state z_{t+d-1})."""
        th = self.theta
        zz = self._z.copy()
        yy = np.asarray(y_now, dtype=np.float64).reshape(7).copy()
        for u_old in self._buf:
            zz = th.alpha * zz + (1.0 - th.alpha) * u_old
            yy = compose_delta(yy, _clip_to_limits(th.gain * zz, th))
        return yy, zz

    def _commit(self, u6) -> None:
        th = self.theta
        self._buf.append(np.asarray(u6, dtype=np.float64).reshape(6))
        if len(self._buf) > th.delay_steps:
            oldest = self._buf.popleft()
            self._z = th.alpha * self._z + (1.0 - th.alpha) * oldest

    def step(self, y_now, c_seq, grip_seq, t: int) -> tuple[np.ndarray, float]:
        th = self.theta
        n = c_seq.shape[0]
        y_pred, z_pred = self._predict(y_now)
        target = c_seq[min(t + th.delay_steps, n - 1)]
        w_des = _clip_to_limits(self.kappa * norm_delta(y_pred, target), th)
        z_req = w_des / th.gain
        denom = np.maximum(1.0 - th.alpha, 1e-3)
        u = (z_req - th.alpha * z_pred) / denom
        u[:3] = np.clip(u[:3], -self.bound_pos, self.bound_pos)
        nr = float(np.linalg.norm(u[3:]))
        if nr > self.bound_rot:
            u[3:] *= self.bound_rot / nr
        self._commit(u)
        gi = min(t + th.grip_lead_steps, len(grip_seq) - 1)
        return u, float(grip_seq[gi])


def build_arm(method: str, spec: PlantSpec, hat: dict | None, theta_nominal: Theta,
              kappa: float = 1.0, bound_pos: float = ACT_BOUND,
              bound_rot: float = ACT_BOUND) -> tuple[LiberoExecutor, bool]:
    """(executor, uses_shaped_reference) for one comparison arm."""
    kw = {"bound_pos": bound_pos, "bound_rot": bound_rot}
    if method == "naive" or method.startswith("slow"):
        return LiberoExecutor(theta_identity(), **kw), False
    if method == "oracle":
        return LiberoExecutor(theta_oracle(spec, theta_nominal), kappa=kappa, **kw), True
    if method == "ours_p":
        if hat is None:
            raise ValueError(f"no theta-hat for plant {spec.pid}")
        return LiberoExecutor(theta_from_hat(hat, fallback=theta_nominal),
                              kappa=kappa, **kw), True
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# selftest: byte-level agreement with cce/executor.py inside the ManiSkill bound
# --------------------------------------------------------------------------- #
def selftest(verbose: bool = True) -> dict:
    from cce.executor import InversionExecutor, shape_reference

    rng = np.random.default_rng(20260906)
    out: dict[str, float] = {}

    def rand_theta():
        # kept mild on purpose: the equivalence check must exercise the algebra with
        # no bound binding on either side (cce/executor.py clips element-wise to +-1,
        # LiberoExecutor clips the rotation channel in norm -- the two differ only
        # once a bound actually binds, which is exactly the intended difference).
        th = Theta(
            delay_steps=int(rng.integers(0, 6)),
            alpha=rng.uniform(0.0, 0.6, size=6),
            gain=rng.uniform(0.8, 1.2, size=6),
            clip_pos_norm=1.0,
            clip_rot_norm=1.0,
            grip_lead_steps=int(rng.integers(0, 8)),
            source="test",
        )
        return th

    worst_exec = 0.0
    worst_shape = 0.0
    max_u = 0.0
    for _ in range(30):
        th = rand_theta()
        y0 = np.concatenate([rng.normal(scale=0.2, size=3), qnorm(rng.normal(size=4))])
        # a smooth 30-step chunk with small increments, so no bound binds in either
        # implementation and the two must agree exactly
        c = np.zeros((30, 7))
        p = y0.copy()
        for k in range(30):
            p = compose_delta(p, np.concatenate([rng.normal(scale=0.005, size=3),
                                                 rng.normal(scale=0.005, size=3)]))
            c[k] = p
        grip = np.where(np.arange(30) < 15, -1.0, 1.0)

        a = LiberoExecutor(th, bound_pos=1e4, bound_rot=1e4)
        b = InversionExecutor(th, mode="param")
        a.reset()
        b.reset()
        y = y0.copy()
        for t in range(30):
            # cce/executor.py returns clip_norm6(u) (the ManiSkill action bound, which
            # LIBERO's absolute OSC does not have) but *commits* the pre-bound value;
            # compare against that pre-bound value.
            u_par = b._param_arm(y, c, t)
            ua, _ = a.step(y, c, grip, t)
            b.step(y, c, grip, t)
            max_u = max(max_u, float(np.abs(ua).max()))
            worst_exec = max(worst_exec, float(np.abs(ua - u_par).max()))
            y = compose_delta(y, np.concatenate([rng.normal(scale=0.002, size=3),
                                                 rng.normal(scale=0.002, size=3)]))

        th2 = rand_theta()
        s_mine = shape_reference_libero(c, th2, y0, bound=1.0)
        s_theirs = shape_reference(c, th2, y0)
        worst_shape = max(worst_shape, float(np.abs(s_mine - s_theirs).max()))

    out["executor_vs_cce_executor"] = worst_exec
    out["max_abs_u_in_test"] = max_u  # must stay < 1 so no bound binds either side
    out["shape_reference_vs_cce"] = worst_shape
    assert max_u < 0.95, max_u
    assert worst_exec < 1e-12, worst_exec
    assert worst_shape < 1e-9, worst_shape

    # identity theta on a nominal plant must reproduce the policy target verbatim
    from cce.libero_plant import LiberoPlant, NOMINAL

    ex = LiberoExecutor(theta_identity())
    pl = LiberoPlant(NOMINAL)
    y = np.concatenate([rng.normal(scale=0.2, size=3), qnorm(rng.normal(size=4))])
    c = np.stack([compose_delta(y, np.concatenate([rng.normal(scale=0.1, size=3),
                                                   rng.normal(scale=0.1, size=3)]))
                  for _ in range(30)])
    grip = np.full(30, -1.0)
    worst = 0.0
    for t in range(30):
        u, g = ex.step(y, c, grip, t)
        cmd, g_out, _ = pl.command(y, u, g)
        worst = max(worst, float(np.abs(cmd - c[t]).max()))
    out["naive_on_nominal_reproduces_target"] = worst
    assert worst < 1e-9, worst

    out["ok"] = 1.0
    if verbose:
        print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    selftest()
