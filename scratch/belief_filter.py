"""Kalman-filter tracks of neighbors and of ownship for the DoubleIntegrator noise experiments. Pure JAX, no side
effects on import; used by scratch/di_ekf_sweep.py.

State x = [px, py, vx, vy]. Reports are z = x + n with n first-order Gauss-Markov (phi, sigma per component).
Two neighbor models:
  "cv_gm": state [x, n] (8). x: constant velocity, unknown acceleration ~ white, std sigma_a (piecewise constant over
           a step); n: n+ = phi n + w, Q_n = (1 - phi^2) sigma^2. Measurement z = [I I] s (exact up to a tiny R floor).
           This is the consistent model for the GM noise.
  "cv":    state x (4), measurement noise treated as white, R = sigma^2 I. Overconfident under GM noise (ablation).
Ownship uses "cv_gm" but predicts x with the env's own Euler step and the known clipped action (double_integrator.py
agent_step_euler: p+ = p + dt v, v+ = clip(v + dt u / m, +-v_max)); the covariance uses that step's Jacobian, where
the clip contributes 1 inside the limits and 0 when saturated -- the "extended" part of the EKF.
Everything else is linear, so for neighbors this EKF is exactly a linear Kalman filter.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

R_FLOOR_REL = 1e-2  # cv_gm measurement-noise floor, as a fraction of sigma (keeps S invertible in float32)


class LinModel(NamedTuple):
    F: jax.Array  # (d, d)
    Q: jax.Array  # (d, d)
    H: jax.Array  # (4, d)
    R: jax.Array  # (4, 4)


class Tracks(NamedTuple):
    mean: jax.Array  # (..., d)
    cov: jax.Array  # (..., d, d)
    active: jax.Array  # (...) bool
    age: jax.Array  # (...) int32, steps since the last report used


# ---- model matrices (all inputs may be traced) -------------------------------------------------------------------
def cv_F(dt) -> jax.Array:
    I2, Z2 = jnp.eye(2), jnp.zeros((2, 2))
    return jnp.block([[I2, dt * I2], [Z2, I2]])


def cv_Q(dt, sigma_a) -> jax.Array:
    """Unknown acceleration with std sigma_a per axis, discretized like the env's Euler step (p+ = p + dt v,
    v+ = v + dt a): G = [0; dt I], Q = sigma_a^2 G G^T. The textbook exact-discretization G = [dt^2/2 I; dt I] does
    NOT match the env: with a rank-2 Q it forces a position jump dt/2 * dv with every velocity jump, the position
    report contradicts it, and the noise state absorbs the mismatch (~5e-3 errors even with zero report noise)."""
    G = jnp.concatenate([jnp.zeros((2, 2)), dt * jnp.eye(2)], axis=0)
    return sigma_a ** 2 * G @ G.T


def cv_gm_model(dt, sigma_a, phi, sigma) -> LinModel:
    """sigma: (4,) stationary std of n per component."""
    Z4 = jnp.zeros((4, 4))
    F = jnp.block([[cv_F(dt), Z4], [Z4, phi * jnp.eye(4)]])
    Q = jnp.block([[cv_Q(dt, sigma_a), Z4], [Z4, jnp.diag((1 - phi ** 2) * sigma ** 2)]])
    H = jnp.concatenate([jnp.eye(4), jnp.eye(4)], axis=1)
    R = jnp.diag((R_FLOOR_REL * sigma) ** 2)
    return LinModel(F, Q, H, R)


def cv_model(dt, sigma_a, sigma) -> LinModel:
    return LinModel(cv_F(dt), cv_Q(dt, sigma_a), jnp.eye(4), jnp.diag(sigma ** 2))


def init_mean_cov(z, sigma, d: int):
    """Posterior after the first report z (4,). cv_gm: x = z, n = 0, P_xx = P_nn = S, P_xn = -S (x + n = z exactly)."""
    S = jnp.diag(sigma ** 2)
    if d == 4:
        return z, S
    return jnp.concatenate([z, jnp.zeros(4)]), jnp.block([[S, -S], [-S, S]])


# ---- single-track predict / update --------------------------------------------------------------------------------
def kf_predict(mean, cov, F, Q, mean_new=None):
    """Linear predict; pass mean_new for a nonlinear mean step (then F is its Jacobian)."""
    m = F @ mean if mean_new is None else mean_new
    P = F @ cov @ F.T + Q
    return m, 0.5 * (P + P.T)


def kf_update(mean, cov, z, H, R):
    """Joseph-form update. Returns mean, cov, innovation y, innovation covariance S."""
    S = H @ cov @ H.T + R
    S = 0.5 * (S + S.T)
    K = jnp.linalg.solve(S, H @ cov).T  # P H^T S^-1 (S symmetric)
    y = z - H @ mean
    m = mean + K @ y
    IKH = jnp.eye(mean.shape[0]) - K @ H
    P = IKH @ cov @ IKH.T + K @ R @ K.T
    return m, 0.5 * (P + P.T), y, S


# ---- neighbor tracks: one per (receiver i, sender j), arrays (n, n, ...) --------------------------------------------
def nbr_tracks_empty(n: int, d: int) -> Tracks:
    return Tracks(jnp.zeros((n, n, d)), jnp.tile(jnp.eye(d), (n, n, 1, 1)), jnp.zeros((n, n), dtype=bool),
                  jnp.zeros((n, n), dtype=jnp.int32))


def _where_tracks(mask, a: Tracks, b: Tracks) -> Tracks:
    return Tracks(jnp.where(mask[..., None], a.mean, b.mean), jnp.where(mask[..., None, None], a.cov, b.cov),
                  jnp.where(mask, a.active, b.active), jnp.where(mask, a.age, b.age))


def nbr_tracks_update(tr: Tracks, a_z, aa_heard, model: LinModel, sigma, t_drop) -> Tracks:
    """Receiver i hears report a_z[j] where aa_heard[i, j]. New tracks start from the report; heard active tracks get a
    KF update; unheard tracks coast (age + 1) and are dropped after t_drop steps without a report."""
    n, d = tr.mean.shape[0], tr.mean.shape[-1]
    aa_z = jnp.broadcast_to(a_z[None], (n, n, 4))
    upd = jax.vmap(jax.vmap(lambda m, P, z: kf_update(m, P, z, model.H, model.R)[:2]))
    m_u, P_u = upd(tr.mean, tr.cov, aa_z)
    m_i, P_i = jax.vmap(jax.vmap(lambda z: init_mean_cov(z, sigma, d)))(aa_z)
    start = aa_heard & ~tr.active
    mean = jnp.where(start[..., None], m_i, jnp.where(aa_heard[..., None], m_u, tr.mean))
    cov = jnp.where(start[..., None, None], P_i, jnp.where(aa_heard[..., None, None], P_u, tr.cov))
    age = jnp.where(aa_heard, 0, tr.age + 1)
    active = (tr.active | aa_heard) & (age <= t_drop)
    return Tracks(mean, cov, active, age)


def nbr_tracks_predict(tr: Tracks, model: LinModel) -> Tracks:
    m, P = jax.vmap(jax.vmap(lambda m, P: kf_predict(m, P, model.F, model.Q)))(tr.mean, tr.cov)
    return tr._replace(mean=m, cov=P)


# ---- ownship track: one per agent, arrays (n, 8); always cv_gm, always gets its own nav fix ------------------------
def own_track_init(a_z, sigma) -> tuple[jax.Array, jax.Array]:
    return jax.vmap(lambda z: init_mean_cov(z, sigma, 8))(a_z)


def own_track_update(a_mean, a_cov, a_z, model: LinModel):
    return jax.vmap(lambda m, P, z: kf_update(m, P, z, model.H, model.R)[:2])(a_mean, a_cov, a_z)


def own_track_predict(a_mean, a_cov, a_u, dt, mass, v_max, model: LinModel):
    """EKF predict with the known clipped action a_u (n, 2), mirroring agent_step_euler including the velocity clip."""
    def one(m, P, u):
        p, v, nse = m[:2], m[2:4], m[4:]
        v_raw = v + dt * u / mass
        v_new = jnp.clip(v_raw, -v_max, v_max)
        mean_new = jnp.concatenate([p + dt * v, v_new, model.F[4:, 4:] @ nse])
        D = (jnp.abs(v_raw) < v_max).astype(m.dtype)  # d clip / d v
        F = model.F.at[2:4, 2:4].set(jnp.diag(D))
        return kf_predict(m, P, F, model.Q, mean_new)
    return jax.vmap(one)(a_mean, a_cov, a_u)


def nees(err, P):
    """err^T P^-1 err over the trailing dims."""
    return jnp.einsum("...i,...i->...", err, jnp.linalg.solve(P, err[..., None])[..., 0])
