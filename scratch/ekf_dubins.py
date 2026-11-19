# Standalone EKF demo: Dubins car, noisy position (or range/bearing) measurements, FilterPy's ExtendedKalmanFilter.
# Not used by anything else in the repo. Run cell-by-cell in the VS Code Interactive Window, or:
#   python scratch/ekf_dubins.py
#
# EKF in one screen (x = state estimate, P = its covariance):
#   predict:  x⁻ = f(x, u)                 nonlinear motion model
#             F  = ∂f/∂x at (x, u)          linearize the motion model
#             P⁻ = F P Fᵀ + Q               uncertainty grows (Q = process noise)
#   update:   H  = ∂h/∂x at x⁻              linearize the measurement model
#             y  = z - h(x⁻)                innovation (measurement surprise)
#             S  = H P⁻ Hᵀ + R              predicted innovation covariance (R = measurement noise)
#             K  = P⁻ Hᵀ S⁻¹                Kalman gain: how much to trust z vs the prediction
#             x  = x⁻ + K y                 corrected estimate
#             P  = (I-KH) P⁻ (I-KH)ᵀ + K R Kᵀ   uncertainty shrinks (Joseph form, used by FilterPy)

# %% setup / config
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from matplotlib.patches import Ellipse

SEED = 0
DT = 0.1  # [s] simulation / filter step
N_STEPS = 300  # 30 s
MEAS_EVERY = 5  # take a measurement every 5 steps (0.5 s, ADS-B-like); covariance grows in between
MEAS_MODEL = "position"  # "position" (GNSS-like, linear) or "range_bearing" (from a fixed station, nonlinear)

# true initial state x = [px, py, θ, v] and the nominal (known) control u = [ω, a]
X0_TRUE = np.array([0.0, 0.0, 0.0, 1.0])


def nominal_control(t: float) -> np.ndarray:
    """S-curve: turn rate oscillates, speed held constant."""
    return np.array([0.4 * np.sin(2 * np.pi * t / 10.0), 0.0])


# noise levels
SIGMA_W = np.array([0.05, 0.1])  # process noise on [ω (rad/s), a (m/s²)]
SIGMA_POS = 0.5  # [m] position measurement noise (position model)
SIGMA_RANGE, SIGMA_BEARING = 0.3, np.deg2rad(2.0)  # range-bearing model
STATION = np.array([5.0, -5.0])  # fixed station for the range-bearing model

# filter initialization: deliberately wrong initial guess, with a covariance that admits it
X0_EST = X0_TRUE + np.array([1.0, -1.0, 0.3, -0.3])
P0 = np.diag([1.5 ** 2, 1.5 ** 2, 0.5 ** 2, 0.5 ** 2])

rng = np.random.default_rng(SEED)


# %% motion model: Dubins car (Euler step) and its Jacobians
def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """x_{k+1} = f(x_k, u_k). Works on flat (4,) or column (4, 1) arrays."""
    px, py, th, v = np.ravel(x)
    om, a = np.ravel(u)
    x_next = np.array([px + v * np.cos(th) * DT,
                       py + v * np.sin(th) * DT,
                       th + om * DT,
                       v + a * DT])
    return x_next.reshape(np.shape(x))


def F_jacobian(x: np.ndarray) -> np.ndarray:
    """F = ∂f/∂x: how a small error in each state component propagates through one step."""
    _, _, th, v = np.ravel(x)
    return np.array([[1, 0, -v * np.sin(th) * DT, np.cos(th) * DT],
                     [0, 1, v * np.cos(th) * DT, np.sin(th) * DT],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


# G = ∂f/∂u maps control noise into the state; Q = G diag(σ_w²) Gᵀ (+ tiny jitter so Q is not singular)
G = np.array([[0, 0], [0, 0], [DT, 0], [0, DT]])
Q = G @ np.diag(SIGMA_W ** 2) @ G.T + 1e-6 * np.eye(4)


# %% measurement models: h(x), its Jacobian H, noise R, and the residual (innovation) function
def h_position(x):
    return np.ravel(x)[:2].reshape(2, 1)


def H_position(x):
    return np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


def h_range_bearing(x):
    dx, dy = np.ravel(x)[:2] - STATION
    return np.array([[np.hypot(dx, dy)], [np.arctan2(dy, dx)]])


def H_range_bearing(x):
    dx, dy = np.ravel(x)[:2] - STATION
    r2 = dx ** 2 + dy ** 2
    r = np.sqrt(r2)
    return np.array([[dx / r, dy / r, 0, 0],
                     [-dy / r2, dx / r2, 0, 0]])


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def residual_range_bearing(z, hx):
    y = z - hx
    y[1] = wrap(y[1])  # bearing difference must be wrapped to [-π, π)
    return y


if MEAS_MODEL == "position":
    h, H_fn, R, residual = h_position, H_position, np.diag([SIGMA_POS ** 2] * 2), np.subtract
    Z_LABELS = ["x [m]", "y [m]"]
elif MEAS_MODEL == "range_bearing":
    h, H_fn, R, residual = (h_range_bearing, H_range_bearing,
                            np.diag([SIGMA_RANGE ** 2, SIGMA_BEARING ** 2]), residual_range_bearing)
    Z_LABELS = ["range [m]", "bearing [rad]"]
else:
    raise ValueError(MEAS_MODEL)


# %% the filter: FilterPy's EKF with a nonlinear prediction
class DubinsEKF(ExtendedKalmanFilter):
    """FilterPy's predict() calls predict_x(u) and then does P = F P Fᵀ + Q. The default predict_x is linear
    (x = F x + B u), so override it with the nonlinear motion model. F must be set before each predict()."""

    def predict_x(self, u=0):
        self.x = f(self.x, u)


ekf = DubinsEKF(dim_x=4, dim_z=2)
ekf.x = X0_EST.reshape(4, 1)  # FilterPy uses column vectors
ekf.P = P0.copy()
ekf.Q = Q
ekf.R = R

# %% simulate truth + measurements and run the filter
x_true = np.zeros((N_STEPS + 1, 4))
x_est = np.zeros((N_STEPS + 1, 4))
P_est = np.zeros((N_STEPS + 1, 4, 4))
x_true[0], x_est[0], P_est[0] = X0_TRUE, X0_EST, P0
meas_k, meas_z, meas_err = [], [], []  # measurement step index, value, and noise actually drawn

printed = False
for k in range(N_STEPS):
    u = nominal_control(k * DT)

    # truth: the real vehicle gets the nominal control plus random disturbances
    x_true[k + 1] = f(x_true[k], u + rng.normal(0, SIGMA_W))

    # EKF predict: linearize at the current estimate, then propagate mean and covariance
    ekf.F = F_jacobian(ekf.x)
    ekf.predict(u=u)

    # EKF update, only when a measurement arrives
    if (k + 1) % MEAS_EVERY == 0:
        z_true = h(x_true[k + 1])
        noise = rng.multivariate_normal(np.zeros(2), R).reshape(2, 1)
        z = z_true + noise
        ekf.update(z, HJacobian=H_fn, Hx=h, residual=residual)
        meas_k.append(k + 1)
        meas_z.append(np.ravel(z))
        meas_err.append(np.ravel(residual(z, z_true)))

        if not printed:  # show every quantity of one update step
            printed = True
            np.set_printoptions(precision=3, suppress=True)
            print(f"--- first update at step {k + 1} (t = {(k + 1) * DT:.1f} s) ---")
            print("prior x⁻          ", np.ravel(ekf.x_prior))
            print("measurement z     ", np.ravel(z))
            print("innovation y      ", np.ravel(ekf.y))
            print("innovation cov S\n", ekf.S)
            print("Kalman gain K\n", ekf.K)
            print("posterior x       ", np.ravel(ekf.x_post))
            print("true x            ", x_true[k + 1])
            print("prior std    ", np.sqrt(np.diag(ekf.P_prior)))
            print("posterior std", np.sqrt(np.diag(ekf.P_post)))

    x_est[k + 1] = np.ravel(ekf.x)
    P_est[k + 1] = ekf.P

meas_k, meas_z, meas_err = np.array(meas_k), np.array(meas_z), np.array(meas_err)


# %% summary numbers
def meas_to_xy(z):
    """Measurement expressed as a position (for plotting / comparing with the estimate)."""
    if MEAS_MODEL == "position":
        return z
    return STATION + z[:, :1] * np.column_stack([np.cos(z[:, 1]), np.sin(z[:, 1])])


err = x_est - x_true
err[:, 2] = wrap(err[:, 2])
sig = np.sqrt(np.einsum("kii->ki", P_est))
meas_xy = meas_to_xy(meas_z)
rmse_meas = np.sqrt(np.mean(np.sum((meas_xy - x_true[meas_k, :2]) ** 2, axis=1)))
rmse_ekf = np.sqrt(np.mean(np.sum(err[meas_k, :2] ** 2, axis=1)))
within = (np.abs(err) <= 2 * sig).mean(axis=0)
print(f"\nmeasurement model: {MEAS_MODEL}, {len(meas_k)} measurements over {N_STEPS * DT:.0f} s")
print(f"position RMSE at measurement times: raw measurements {rmse_meas:.3f} m, EKF {rmse_ekf:.3f} m")
print(f"fraction of errors within ±2σ [px, py, θ, v]: {np.round(within, 3)}  (≈0.95 if consistent)")


# %% plots
def cov_ellipse(ax, mean, cov, n_sigma=2.0, **kw):
    """n-sigma ellipse of a 2x2 covariance: axes along its eigenvectors, half-lengths n·sqrt(eigenvalue)."""
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mean, 2 * n_sigma * np.sqrt(vals[1]), 2 * n_sigma * np.sqrt(vals[0]), angle=angle,
                         fill=False, **kw))


fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(3, 4)
t = np.arange(N_STEPS + 1) * DT

# (1) x-y view
ax = fig.add_subplot(gs[:, :2])
ax.plot(x_true[:, 0], x_true[:, 1], "k-", lw=2, label="truth")
ax.plot(x_est[:, 0], x_est[:, 1], "C0-", lw=1.5, label="EKF estimate")
ax.plot(meas_xy[:, 0], meas_xy[:, 1], "C3.", ms=6, label="measurements")
for k in range(0, N_STEPS + 1, 2 * MEAS_EVERY):
    cov_ellipse(ax, x_est[k, :2], P_est[k, :2, :2], color="C0", lw=1, alpha=0.7)
for i in range(0, len(meas_k), 6):  # measurement-noise ellipse (R, mapped to x-y) around some measurements
    if MEAS_MODEL == "position":
        R_xy = R
    else:  # J R Jᵀ with J = ∂(x, y)/∂(range, bearing)
        r_, b_ = meas_z[i]
        J = np.array([[np.cos(b_), -r_ * np.sin(b_)], [np.sin(b_), r_ * np.cos(b_)]])
        R_xy = J @ R @ J.T
    cov_ellipse(ax, meas_xy[i], R_xy, color="C3", lw=1, ls="--", alpha=0.7)
if MEAS_MODEL == "range_bearing":
    ax.plot(*STATION, "k^", ms=10, label="station")
ax.plot([], [], "C0-", lw=1, label="EKF 2σ position covariance")
ax.plot([], [], "C3--", lw=1, label="measurement noise 2σ (R)")
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"Dubins car EKF ({MEAS_MODEL}, measurement every {MEAS_EVERY * DT:.1f} s)")
ax.legend(loc="best", fontsize=9)

# (2) measurement noise: histogram of the noise actually drawn vs the Gaussian the filter assumes
for i in range(2):
    ax = fig.add_subplot(gs[0, 2 + i])
    s = np.sqrt(R[i, i])
    ax.hist(meas_err[:, i], bins=15, density=True, color="C3", alpha=0.5, label="drawn noise")
    xs = np.linspace(-4 * s, 4 * s, 200)
    ax.plot(xs, np.exp(-0.5 * (xs / s) ** 2) / (s * np.sqrt(2 * np.pi)), "k-", label=f"N(0, {s:.3g}²)")
    ax.set_title(f"measurement noise: {Z_LABELS[i]}")
    ax.legend(fontsize=8)

# (3) estimation error vs time with ±2σ bounds from P
names = ["px [m]", "py [m]", "θ [rad]", "v [m/s]"]
for i in range(4):
    ax = fig.add_subplot(gs[1 + i // 2, 2 + i % 2])
    ax.fill_between(t, -2 * sig[:, i], 2 * sig[:, i], color="C0", alpha=0.2, label="±2σ from P")
    ax.plot(t, err[:, i], "C0-", lw=1, label="estimate − truth")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"error in {names[i]} ({within[i] * 100:.0f}% within ±2σ)")
    ax.set_xlabel("t [s]")
    if i == 0:
        ax.legend(fontsize=8)

fig.tight_layout()
out_dir = pathlib.Path(__file__).resolve().parent / "figures" if "__file__" in globals() else pathlib.Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / f"ekf_dubins_{MEAS_MODEL}.png", dpi=110)
print("saved", out_dir / f"ekf_dubins_{MEAS_MODEL}.png")
plt.show()

# %% heading and speed estimates vs truth (neither is measured directly; the EKF infers them from position)
fig2, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, i, name in zip(axs, [2, 3], ["heading θ [rad]", "speed v [m/s]"]):
    truth, est = x_true[:, i], x_est[:, i]
    if i == 2:  # unwrap so heading never jumps by 2π
        truth, est = np.unwrap(truth), np.unwrap(est)
    ax.fill_between(t, est - 2 * sig[:, i], est + 2 * sig[:, i], color="C0", alpha=0.2, label="EKF ±2σ")
    ax.plot(t, truth, "k-", lw=2, label="truth")
    ax.plot(t, est, "C0-", lw=1.5, label="EKF estimate")
    for tm in meas_k * DT:  # measurement times: the band narrows at each update
        ax.axvline(tm, color="C3", lw=0.5, alpha=0.25)
    ax.set_ylabel(name)
    ax.set_title(f"{name}: estimate vs truth")
axs[0].plot([], [], "C3-", lw=0.5, alpha=0.5, label="measurement times")
axs[0].legend(loc="upper right", fontsize=9)
axs[-1].set_xlabel("t [s]")
fig2.tight_layout()
fig2.savefig(out_dir / f"ekf_dubins_{MEAS_MODEL}_theta_v.png", dpi=110)
print("saved", out_dir / f"ekf_dubins_{MEAS_MODEL}_theta_v.png")
plt.show()

# %%
