#!/usr/bin/env python
"""Relative executor r1: position gate and gripper compensation are independent.

The literal v4 prototype and its counterexample remain in libero_exec_v4.py.
This revision is declared before any new policy comparison in the r1 amendment.
"""
from __future__ import annotations
import _bootstrap  # noqa: F401
from collections import deque
from dataclasses import dataclass
import json
import math
import numpy as np
from cce.common import DT, norm_delta
from cce.libero_exec import LiberoExecutor, theta_identity
from cce.libero_plant import compose_delta


@dataclass
class RelativeTheta:
    delay_steps: int
    tau_s: np.ndarray
    gain: np.ndarray
    grip_lead_steps: int = 0
    identifiable: np.ndarray | None = None

    @classmethod
    def identity(cls):
        return cls(0, np.zeros(6), np.ones(6))

    @classmethod
    def from_fit(cls, fit):
        return cls(int(fit['delay_steps']), np.array([a['tau_s'] for a in fit['axes']]),
                   np.array([a['gain'] for a in fit['axes']]), int(fit['grip_lead_steps']),
                   np.array([a['identifiable'] for a in fit['axes']], dtype=bool))


def clip_command(u, bound=1.):
    u = np.array(u, dtype=float, copy=True)
    u[:3] = np.clip(u[:3], -bound, bound)
    nr = float(np.linalg.norm(u[3:]))
    if nr > bound:
        u[3:] *= bound/nr
    return u


class RelativeExecutor:
    def __init__(self, relative, nominal_alpha, nominal_gain, *, rho_max,
                 h=None, h0=None, gate=True, unilateral=True):
        self.d = int(relative.delay_steps)
        if self.d < 0 or self.d != relative.delay_steps:
            raise ValueError('delay must be a nonnegative integer')
        tau = np.asarray(relative.tau_s, dtype=float).reshape(6).copy()
        gain = np.asarray(relative.gain, dtype=float).reshape(6).copy()
        self.an = np.asarray(nominal_alpha, dtype=float).reshape(6).copy()
        self.gn = np.asarray(nominal_gain, dtype=float).reshape(6).copy()
        if not all(np.isfinite(x).all() for x in (tau, gain, self.an, self.gn)):
            raise ValueError('nonfinite model parameter')
        if (gain <= 0).any() or (self.gn <= 0).any() or (self.an < 0).any() or (self.an >= 1).any():
            raise ValueError('invalid gain/pole')
        if math.isnan(rho_max) or rho_max < 1:
            raise ValueError('rho_max must be >= 1')
        if gate and (h is None or h0 is None or not np.isfinite([h, h0]).all()):
            raise ValueError('the gate requires measured H and a chosen H0')
        self.bypass = bool(gate and h < h0)
        self.mask = (np.ones(6, dtype=bool) if relative.identifiable is None
                     else np.asarray(relative.identifiable, dtype=bool).reshape(6).copy())
        if unilateral:
            gain = np.minimum(gain, 1.)
        self.gain = gain
        # Negative lag has no causal one-pole inverse, including in the ablation.
        self.alpha = np.array([math.exp(-DT/t) if t > 0 else 0. for t in tau])
        self.alpha = np.minimum(self.alpha, 1.-1./rho_max)
        self.lead = max(0, int(relative.grip_lead_steps)) if unilateral else int(relative.grip_lead_steps)
        self.reset()

    def reset(self):
        self.pipe = deque(np.zeros(6) for _ in range(self.d))
        self.z = np.zeros(6)
        self.zn = np.zeros(6)
        self.last_y = None

    def _advance(self, z, zn, u):
        z = self.alpha*z+(1-self.alpha)*u
        zn = self.an*zn+(1-self.an)*self.gain*z
        return z, zn

    def step(self, y, c, grip, t):
        y = np.asarray(y, dtype=float).reshape(7)
        c, grip = np.asarray(c, dtype=float), np.asarray(grip, dtype=float)
        if c.ndim != 2 or c.shape[1] != 7 or len(c) != len(grip) or not t >= 0 or len(c) == 0:
            raise ValueError('invalid chunk/index')
        if not np.isfinite(y).all() or not np.isfinite(c).all() or not np.isfinite(grip).all():
            raise ValueError('nonfinite observation or chunk')
        naive = clip_command(norm_delta(y, c[min(t, len(c)-1)]))
        if self.bypass:
            return naive, float(grip[np.clip(t+self.lead, 0, len(grip)-1)])
        # The observed achieved increment initializes the nominal filter state;
        # neither pose nor nominal memory is reset when a new chunk arrives.
        if self.last_y is not None:
            self.zn = norm_delta(self.last_y, y)/self.gn
        yp, z, zn = y.copy(), self.z.copy(), self.zn.copy()
        for queued in self.pipe:
            z, zn = self._advance(z, zn, queued)
            yp = compose_delta(yp, self.gn*zn)
        v = norm_delta(yp, c[min(t+self.d, len(c)-1)])
        u = (v/self.gain-self.alpha*z)/(1-self.alpha)
        u = clip_command(u)
        # Failed axes use the bounded current-target command, not a future target.
        u[~self.mask] = naive[~self.mask]
        # Keep failed rotation coordinates unchanged while fitting the remaining
        # coordinates into the shared rotation ball.
        good_rot = np.flatnonzero(self.mask[3:])+3
        bad_rot = np.flatnonzero(~self.mask[3:])+3
        budget = math.sqrt(max(0., 1.-float(u[bad_rot] @ u[bad_rot])))
        n_good = float(np.linalg.norm(u[good_rot]))
        if n_good > budget:
            u[good_rot] *= budget/n_good
        self.pipe.append(u.copy())
        old = self.pipe.popleft()
        self.z, self.zn = self._advance(self.z, self.zn, old)
        self.last_y = y.copy()
        return u, float(grip[np.clip(t+self.lead, 0, len(grip)-1)])


def predicted_h(relative, nominal_alpha, nominal_gain, reference):
    """Literal positional H in metres; gripper delay cannot affect this value."""
    def rollout(th):
        y = reference[0].copy()
        a = np.array([math.exp(-DT/t) if t > 0 else 0. for t in th.tau_s])
        pipe = deque(np.zeros(6) for _ in range(th.delay_steps))
        z, zn = np.zeros(6), np.zeros(6)
        errors = []
        for target in reference:
            pipe.append(clip_command(norm_delta(y, target)))
            delayed = pipe.popleft()
            z = a*z+(1-a)*delayed
            zn = nominal_alpha*zn+(1-nominal_alpha)*th.gain*z
            y = compose_delta(y, nominal_gain*zn)
            errors.append(np.sum((y[:3]-target[:3])**2))
        return float(np.sqrt(np.mean(errors)))
    return rollout(relative)-rollout(RelativeTheta.identity())


def selftest():
    rng = np.random.default_rng(20260908)
    an, gn = np.full(6, .57), np.full(6, .29)
    y0 = np.array([0., 0., .4, 1., 0., 0., 0.])
    c = np.stack([compose_delta(y0, rng.normal(0, .5, 6)) for _ in range(90)])
    grip = np.where(np.arange(90)%30 < 15, -1., 1.)
    worst = 0.
    for rho in (2., 4., 8., math.inf):
        ex = RelativeExecutor(RelativeTheta.identity(), an, gn, rho_max=rho, gate=False)
        old = LiberoExecutor(theta_identity(), bound_pos=1., bound_rot=1.)
        for t in range(90):
            y = compose_delta(y0, rng.normal(0, .5, 6))
            start = (t//30)*30
            a, ga = ex.step(y, c[start:start+30], grip[start:start+30], t%30)
            b, gb = old.step(y, c[start:start+30], grip[start:start+30], t%30)
            worst = max(worst, float(np.max(np.abs(a-b))))
            assert ga == gb
    assert worst < 1e-9, worst
    # Exact cascade pipeline prediction, with an independent synthetic plant.
    stability = []
    for d, tau, gain in [(5, .4, .7), (2, .25, 1.3), (0, .6, .2)]:
        for rho in (2., 4., 8., math.inf):
            th = RelativeTheta(d, np.full(6, tau), np.full(6, gain), d+12)
            ex = RelativeExecutor(th, an, gn, rho_max=rho, gate=False)
            y, z, zn = y0.copy(), np.zeros(6), np.zeros(6)
            pipe = deque(np.zeros(6) for _ in range(d))
            a = math.exp(-DT/tau)
            max_error = 0.
            for t in range(600):
                target = y0.copy(); target[0] += .03*math.sin(t*.04)
                seq = np.tile(target, (30, 1))
                u, _ = ex.step(y, seq, np.full(30, -1.), t%30)
                assert np.max(np.abs(u[:3])) <= 1+1e-12 and np.linalg.norm(u[3:]) <= 1+1e-12
                pipe.append(u.copy()); delayed = pipe.popleft()
                z = a*z+(1-a)*delayed
                zn = an*zn+(1-an)*gain*z
                y = compose_delta(y, gn*zn)
                assert np.isfinite(y).all()
                max_error = max(max_error, float(np.linalg.norm(y[:3]-target[:3])))
            assert max_error < .5, max_error
            stability.append({'d': d, 'tau': tau, 'gain': gain, 'rho': str(rho), 'max_error_m': max_error})
    pure_grip = RelativeTheta.identity(); pure_grip.grip_lead_steps = 20
    h = predicted_h(pure_grip, an, gn, c)
    assert h == 0.
    bypass = RelativeExecutor(pure_grip, an, gn, rho_max=4., h=h, h0=.001)
    _, g = bypass.step(y0, c, grip, 0)
    assert g == grip[20] and g != grip[0]
    bad = RelativeTheta(5, np.ones(6)*.4, np.ones(6)*.7, identifiable=np.zeros(6, bool))
    ex = RelativeExecutor(bad, an, gn, rho_max=4., gate=False)
    u, _ = ex.step(y0, c, grip, 0)
    assert np.allclose(u, clip_command(norm_delta(y0, c[0])), rtol=0, atol=1e-12)
    bad.identifiable = np.array([False, True, True, False, True, True])
    ex = RelativeExecutor(bad, an, gn, rho_max=4., gate=False)
    u, _ = ex.step(y0, c, grip, 0)
    naive = clip_command(norm_delta(y0, c[0]))
    assert np.array_equal(u[~bad.identifiable], naive[~bad.identifiable])
    assert np.linalg.norm(u[3:]) <= 1.+1e-12
    return {'nominal_max_command_error': worst, 'synthetic_stability': stability,
            'all_axes_quality_fallback': True, 'pure_gripper_H_m': h,
            'arm_gate_preserves_needed_gripper_lead': True,
            'formal_gate_ready': False, 'implementation_checks_passed': True}


if __name__ == '__main__':
    print(json.dumps(selftest(), indent=2, allow_nan=False))
