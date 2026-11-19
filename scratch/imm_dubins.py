# Standalone IMM demo: track another aircraft (Dubins, straight legs + standard-rate turns) from ADS-B-like
# reports, without knowing its control. Compares FilterPy's IMMEstimator (CV + CT models) against single-model
# EKFs. Not used by anything else in the repo. Run cell-by-cell in the VS Code Interactive Window, or:
#   python scratch/imm_dubins.py
#
# IMM in one screen (M models, each its own EKF with estimate x_j, P_j; μ_j = probability model j is active;
# Π[i, j] = probability of switching from model i to model j in one step):
#   1. mix:      c̄_j    = Σ_i Π[i,j] μ_i                     predicted model probabilities
#                μ_i|j  = Π[i,j] μ_i / c̄_j                    how much filter i feeds filter j's starting point
#                x̄_j    = Σ_i μ_i|j x_i
#                P̄_j    = Σ_i μ_i|j [P_i + (x_i - x̄_j)(x_i - x̄_j)ᵀ]
#   2. filter:   each EKF predicts (and updates, if a measurement arrived) from (x̄_j, P̄_j)
#                Λ_j    = N(y_j; 0, S_j)                       how well model j predicted the measurement
#   3. weights:  μ_j    ∝ Λ_j c̄_j                              (normalize to sum to 1)
#   4. combine:  x      = Σ_j μ_j x_j                          output only (not fed back)
#                P      = Σ_j μ_j [P_j + (x_j - x)(x_j - x)ᵀ]  second term = "the models disagree" spread
# FilterPy's IMMEstimator does steps 1 and 4 in predict(), and steps 2 and 3 in update().

# %% setup / config
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter, IMMEstimator
from matplotlib.patches import Ellipse
from scipy.stats import chi2

SEED = 0
N_MC = 30  # Monte Carlo runs (same turn schedule, fresh noise) for the consistency statistics
DT = 0.1  # [s] simulation / filter prediction step
T_END = 150.0  # [s]
N_STEPS = int(round(T_END / DT))
MEAS_EVERY = 5  # an ADS-B report every 5 steps (0.5 s, 1090ES nominal rate)
MEAS_MODEL = "pos_vel"  # "pos_vel" (ADS-B: position + ground velocity) or "position" (position only)
T_SKIP = 5.0  # [s] ignore the initial convergence transient in the printed summary

# truth: state x = [px, py, θ, v, ω] (ω = turn rate). Constant 40 m/s, standard-rate turns (3°/s).
V0 = 40.0
STD_RATE = np.deg2rad(3.0)
TURNS = [(20.0, 50.0, +STD_RATE),  # (t_start, t_end, ω): 90° left
         (70.0, 100.0, -STD_RATE),  # 90° right
         (115.0, 130.0, +STD_RATE)]  # 45° left
SIGMA_A_TRUE = 0.2  # [m/s²] speed disturbance
SIGMA_THDOT_TRUE = np.deg2rad(0.1)  # [rad/s] heading disturbance

# ADS-B accuracy categories -> per-axis σ. 95% radius of a circular 2D Gaussian = 2.45σ.
# NACp 9: EPU < 30 m (95%). NACv 2: < 3 m/s (95%). Real ADS-B position error is time-correlated (GNSS);
# here it is white, so this demo isolates the maneuver-model problem.
SIGMA_POS = 30.0 / 2.45
SIGMA_VEL = 3.0 / 2.45

# filter process noise
SIGMA_A = 0.5  # [m/s²] along-track acceleration (both models)
SIGMA_THDOT = np.deg2rad(0.2)  # [rad/s] heading wander (both models)
SIGMA_OMDOT = np.deg2rad(1.0)  # [rad/s²] turn-rate random walk (CT model only); lets ω reach 3°/s in a few s

# IMM Markov switching matrix (per DT step): mean time in a mode = DT / (1 - p_stay) ≈ 33 s
P_STAY = 0.997
PI = np.array([[P_STAY, 1 - P_STAY],
               [1 - P_STAY, P_STAY]])
MODEL_NAMES = ["CV", "CT"]

# filter initialization: error drawn from P0 so the filters start consistent
P0 = np.diag([SIGMA_POS ** 2, SIGMA_POS ** 2, np.deg2rad(10.0) ** 2, 2.0 ** 2, np.deg2rad(1.0) ** 2])
X0_TRUE = np.array([0.0, 0.0, 0.0, V0, 0.0])


def omega_true(t: float) -> float:
    for t0, t1, om in TURNS:
        if t0 <= t < t1:
            return om
    return 0.0


# %% motion models on the common state [px, py, θ, v, ω] and their Jacobians F = ∂f/∂x
def f_cv(x):
    """Constant velocity: fly straight, turn rate forced to 0."""
    px, py, th, v, _ = np.ravel(x)
    return np.array([px + v * np.cos(th) * DT, py + v * np.sin(th) * DT, th, v, 0.0]).reshape(np.shape(x))


def f_ct(x):
    """Coordinated turn: heading changes at the (estimated) turn rate ω, which is a random walk."""
    px, py, th, v, om = np.ravel(x)
    return np.array([px + v * np.cos(th) * DT, py + v * np.sin(th) * DT, th + om * DT, v, om]).reshape(np.shape(x))


def _F_common(x):
    _, _, th, v, _ = np.ravel(x)
    F = np.eye(5)
    F[0, 2], F[0, 3] = -v * np.sin(th) * DT, np.cos(th) * DT
    F[1, 2], F[1, 3] = v * np.cos(th) * DT, np.sin(th) * DT
    return F


def F_cv(x):
    F = _F_common(x)
    F[4, 4] = 0.0  # ω is reset, so its old value (and uncertainty) does not carry over
    return F


def F_ct(x):
    F = _F_common(x)
    F[2, 4] = DT  # θ_{k+1} depends on ω_k
    return F


# process noise Q = G diag(σ²) Gᵀ, G = how each noise source enters the state (+ tiny jitter so Q is not singular)
Q_CV = np.diag([0, 0, (SIGMA_THDOT * DT) ** 2, (SIGMA_A * DT) ** 2, 0]) + 1e-6 * np.eye(5)
Q_CT = np.diag([0, 0, (SIGMA_THDOT * DT) ** 2, (SIGMA_A * DT) ** 2, (SIGMA_OMDOT * DT) ** 2]) + 1e-6 * np.eye(5)


# %% measurement models
def h_position(x):
    return np.ravel(x)[:2].reshape(2, 1)


def H_position(x):
    return np.eye(2, 5)


def h_pos_vel(x):
    px, py, th, v, _ = np.ravel(x)
    return np.array([[px], [py], [v * np.cos(th)], [v * np.sin(th)]])


def H_pos_vel(x):
    _, _, th, v, _ = np.ravel(x)
    H = np.zeros((4, 5))
    H[0, 0] = H[1, 1] = 1.0
    H[2, 2], H[2, 3] = -v * np.sin(th), np.cos(th)
    H[3, 2], H[3, 3] = v * np.cos(th), np.sin(th)
    return H


if MEAS_MODEL == "position":
    h, H_fn, R = h_position, H_position, np.diag([SIGMA_POS ** 2] * 2)
elif MEAS_MODEL == "pos_vel":
    h, H_fn, R = h_pos_vel, H_pos_vel, np.diag([SIGMA_POS ** 2] * 2 + [SIGMA_VEL ** 2] * 2)
else:
    raise ValueError(MEAS_MODEL)
DIM_Z = R.shape[0]


# %% FilterPy adapter: an EKF that IMMEstimator can drive
class IMMReadyEKF(ExtendedKalmanFilter):
    """IMMEstimator calls f.predict(u) and f.update(z) with no other arguments (filterpy/kalman/IMM.py:173,219),
    but ExtendedKalmanFilter.update needs HJacobian and Hx, and predict needs F set beforehand. This subclass
    stores the model functions and supplies them, so the same class works alone or inside an IMM."""

    def __init__(self, f, F_jac, Q, x0, P0):
        super().__init__(dim_x=5, dim_z=DIM_Z)
        self.f, self.F_jac = f, F_jac
        self.Q, self.R = Q.copy(), R.copy()
        self.x, self.P = x0.reshape(5, 1).copy(), P0.copy()

    def predict_x(self, u=0):
        self.x = self.f(self.x)  # nonlinear prediction; the target's control is unknown, so u is ignored

    def predict(self, u=0):
        self.F = self.F_jac(self.x)  # linearize at the current estimate, before moving it
        super().predict(u)

    def update(self, z):
        super().update(z, HJacobian=H_fn, Hx=h)


FILTERS = ["EKF-CV", "EKF-CT", "IMM"]
COLORS = {"EKF-CV": "C3", "EKF-CT": "C2", "IMM": "C0"}


# %% one run: simulate truth + measurements and run all three filters on the same data
def run_once(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x0_est = X0_TRUE + np.linalg.cholesky(P0) @ rng.normal(size=5)

    ekf_cv = IMMReadyEKF(f_cv, F_cv, Q_CV, x0_est, P0)
    ekf_ct = IMMReadyEKF(f_ct, F_ct, Q_CT, x0_est, P0)
    imm = IMMEstimator([IMMReadyEKF(f_cv, F_cv, Q_CV, x0_est, P0), IMMReadyEKF(f_ct, F_ct, Q_CT, x0_est, P0)],
                       mu=np.array([0.5, 0.5]), M=PI)
    est = {"EKF-CV": ekf_cv, "EKF-CT": ekf_ct, "IMM": imm}

    T_x_true = np.zeros((N_STEPS + 1, 5))
    T_x_true[0] = X0_TRUE
    T_x = {name: np.zeros((N_STEPS + 1, 5)) for name in FILTERS}
    T_P = {name: np.zeros((N_STEPS + 1, 5, 5)) for name in FILTERS}
    T_mu = np.zeros((N_STEPS + 1, 2))
    for name in FILTERS:
        T_x[name][0], T_P[name][0] = x0_est, P0
    T_mu[0] = imm.mu
    meas_k, meas_z = [], []

    for k in range(N_STEPS):
        # truth: commanded turn rate + small disturbances
        px, py, th, v, _ = T_x_true[k]
        om = omega_true(k * DT)
        T_x_true[k + 1] = [px + v * np.cos(th) * DT,
                           py + v * np.sin(th) * DT,
                           th + (om + rng.normal(0, SIGMA_THDOT_TRUE)) * DT,
                           v + rng.normal(0, SIGMA_A_TRUE) * DT,
                           omega_true((k + 1) * DT)]

        for f in est.values():
            f.predict()
        if (k + 1) % MEAS_EVERY == 0:
            z = h(T_x_true[k + 1]) + rng.multivariate_normal(np.zeros(DIM_Z), R).reshape(DIM_Z, 1)
            for f in est.values():
                f.update(z)
            meas_k.append(k + 1)
            meas_z.append(np.ravel(z))

        for name, f in est.items():
            T_x[name][k + 1], T_P[name][k + 1] = np.ravel(f.x), f.P
        T_mu[k + 1] = imm.mu

    return dict(T_x_true=T_x_true, T_x=T_x, T_P=T_P, T_mu=T_mu, meas_k=np.array(meas_k), meas_z=np.array(meas_z))


# %% Monte Carlo
runs = [run_once(SEED + i) for i in range(N_MC)]
run0 = runs[0]
t = np.arange(N_STEPS + 1) * DT
T_is_turn = np.array([omega_true(tk) != 0 for tk in t])

# position error and position NEES eᵀ P_pos⁻¹ e (2 DOF): shape (N_MC, T)
pos_err, pos_nees = {}, {}
for name in FILTERS:
    e = np.stack([r["T_x"][name][:, :2] - r["T_x_true"][:, :2] for r in runs])  # (N_MC, T, 2)
    P = np.stack([r["T_P"][name][:, :2, :2] for r in runs])  # (N_MC, T, 2, 2)
    pos_err[name] = e
    pos_nees[name] = np.einsum("mti,mti->mt", e, np.linalg.solve(P, e[..., None])[..., 0])
mu_ct = np.stack([r["T_mu"][:, 1] for r in runs])  # (N_MC, T)

# 95% band for the Monte Carlo average NEES: N_MC · avg ~ χ²(2 N_MC) if the filter is consistent
nees_lo, nees_hi = chi2.ppf([0.025, 0.975], 2 * N_MC) / N_MC
in_2sig_ideal, chi95 = 1 - np.exp(-2.0), chi2.ppf(0.95, 2)

# %% summary numbers (after the first T_SKIP seconds)
keep = t >= T_SKIP
print(f"measurement model: {MEAS_MODEL}, {N_MC} Monte Carlo runs, {T_END:.0f} s each")
print(f"position NEES: ideal mean 2; 95% band for the run-averaged NEES [{nees_lo:.2f}, {nees_hi:.2f}]")
print(f"fraction inside 2σ ellipse: ideal {in_2sig_ideal:.3f}; inside 95% ellipse: ideal 0.950\n")
for seg, mask in [("straight", keep & ~T_is_turn), ("turning", keep & T_is_turn)]:
    print(f"--- {seg} segments ---")
    print(f"{'filter':<8}{'RMSE [m]':>10}{'NEES':>8}{'in 2σ':>8}{'in 95%':>8}")
    for name in FILTERS:
        rmse = np.sqrt(np.mean(np.sum(pos_err[name][:, mask] ** 2, axis=-1)))
        nees = pos_nees[name][:, mask]
        print(f"{name:<8}{rmse:>10.1f}{nees.mean():>8.2f}{(nees <= 4).mean():>8.3f}{(nees <= chi95).mean():>8.3f}")
    print()


# %% plots
def cov_ellipse(ax, mean, cov, n_sigma=2.0, **kw):
    """n-sigma ellipse of a 2x2 covariance: axes along its eigenvectors, half-lengths n·sqrt(eigenvalue)."""
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mean, 2 * n_sigma * np.sqrt(vals[1]), 2 * n_sigma * np.sqrt(vals[0]), angle=angle,
                         fill=False, **kw))


def shade_turns(ax):
    for t0, t1, _ in TURNS:
        ax.axvspan(t0, t1, color="0.85", zorder=0)


fig = plt.figure(figsize=(17, 11))
gs = fig.add_gridspec(4, 4)
xt = run0["T_x_true"]

# (1) x-y view of run 0, with a zoom on the entry to the first turn
ax = fig.add_subplot(gs[:, :2])
ax.plot(xt[:, 0], xt[:, 1], "k-", lw=2, label="truth")
for name in FILTERS:
    ax.plot(run0["T_x"][name][:, 0], run0["T_x"][name][:, 1], "-", color=COLORS[name], lw=1, label=name)
ax.plot(run0["meas_z"][:, 0], run0["meas_z"][:, 1], ".", color="0.5", ms=2, label="ADS-B position reports")
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(f"run 0 ({MEAS_MODEL}); inset: entering the first turn, 2σ ellipses every 1 s")
ax.legend(loc="lower right", fontsize=9)

t_zoom = (TURNS[0][0] - 2.0, TURNS[0][0] + 10.0)
kz = np.flatnonzero((t >= t_zoom[0]) & (t <= t_zoom[1]))
axin = ax.inset_axes([0.03, 0.45, 0.5, 0.5])
axin.plot(xt[kz, 0], xt[kz, 1], "k-", lw=2)
mz = np.isin(run0["meas_k"], kz)
axin.plot(run0["meas_z"][mz, 0], run0["meas_z"][mz, 1], ".", color="0.5", ms=4)
for name in ["EKF-CV", "IMM"]:
    axin.plot(run0["T_x"][name][kz, 0], run0["T_x"][name][kz, 1], "-", color=COLORS[name], lw=1.5)
    for k in kz[:: int(round(1.0 / DT))]:
        cov_ellipse(axin, run0["T_x"][name][k, :2], run0["T_P"][name][k, :2, :2], color=COLORS[name], lw=1)
        axin.plot(*xt[k, :2], "k.", ms=5)  # truth at the same instant: should sit inside the ellipse
axin.set_aspect("equal")
axin.tick_params(labelsize=7)
ax.indicate_inset_zoom(axin, edgecolor="0.4")

# (2) IMM mode probability
ax = fig.add_subplot(gs[0, 2:])
shade_turns(ax)
ax.plot(t, mu_ct.mean(axis=0), "C0-", lw=2, label=f"mean over {N_MC} runs")
ax.plot(t, run0["T_mu"][:, 1], "C0-", lw=0.7, alpha=0.6, label="run 0")
ax.set_ylim(-0.02, 1.02)
ax.set_ylabel("μ_CT")
ax.set_title("IMM probability of the coordinated-turn model (grey = true turns)")
ax.legend(fontsize=8, loc="upper right")

# (3) turn rate
ax = fig.add_subplot(gs[1, 2:])
shade_turns(ax)
ax.plot(t, np.rad2deg(xt[:, 4]), "k-", lw=2, label="truth")
for name in ["EKF-CT", "IMM"]:
    ax.plot(t, np.rad2deg(run0["T_x"][name][:, 4]), "-", color=COLORS[name], lw=1, label=name)
ax.set_ylabel("ω [deg/s]")
ax.set_title("turn rate (run 0); EKF-CV has ω ≡ 0 by construction")
ax.legend(fontsize=8, loc="upper right")

# (4) position error
ax = fig.add_subplot(gs[2, 2:])
shade_turns(ax)
for name in FILTERS:
    ax.plot(t, np.linalg.norm(pos_err[name], axis=-1).mean(axis=0), "-", color=COLORS[name], lw=1.2, label=name)
ax.set_ylabel("‖position error‖ [m]")
ax.set_title(f"position error, mean over {N_MC} runs")
ax.legend(fontsize=8, loc="upper right")

# (5) consistency
ax = fig.add_subplot(gs[3, 2:])
shade_turns(ax)
for name in FILTERS:
    ax.semilogy(t, pos_nees[name].mean(axis=0), "-", color=COLORS[name], lw=1.2, label=name)
ax.axhline(2.0, color="k", lw=0.8, label="ideal (2)")
ax.axhspan(nees_lo, nees_hi, color="k", alpha=0.1, label="95% band")
ax.set_ylabel("position NEES")
ax.set_xlabel("t [s]")
ax.set_title("consistency: above the band = overconfident (P too small for the actual error)")
ax.legend(fontsize=8, loc="upper right", ncol=2)

fig.tight_layout()
out_dir = pathlib.Path(__file__).resolve().parent / "figures" if "__file__" in globals() else pathlib.Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / f"imm_dubins_{MEAS_MODEL}.png", dpi=110)
print("saved", out_dir / f"imm_dubins_{MEAS_MODEL}.png")
plt.show()

# %%
