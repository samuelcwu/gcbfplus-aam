"""How conservative is the half-space (line-of-sight) chance constraint of the EKF belief CBF?

Setup (Vahs et al., RA-L 2023, Sec. V): the relative position p ~ N(mu, sigma^2 I) in 2-D, and collision means
||p|| < rho (rho = 2r). Each method gives the smallest mean distance d = ||mu|| that keeps P(collision) <= delta:
  half-space  d = rho + kappa*sigma, kappa = Phi^-1(1 - delta)  (tangent line on the line of sight; sound, conservative)
  exact       solve P(||p|| < rho) = delta; ||p||^2/sigma^2 is noncentral chi^2 (2 dof, nc = (d/sigma)^2)
  ellipse     d = rho + sqrt(-2 ln delta)*sigma  (inflate rho by the 1-delta 2-D confidence-ellipse radius)
Everything is in units of rho. sigma is the RELATIVE sigma per axis (= sqrt(2)*sigma_agent if independent).

Run as a script or cell by cell in the VS Code Interactive Window, like the other scratch files.
"""
# %% imports and core functions
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2, norm


def p_coll(d, s, rho=1.0):
    """Exact P(||p|| < rho) for p ~ N(d*e1, s^2 I) in 2-D."""
    return ncx2.cdf((rho / s) ** 2, 2, (d / s) ** 2)


def d_half(s, delta, rho=1.0):
    return rho + norm.ppf(1 - delta) * s


def d_ellipse(s, delta, rho=1.0):
    return rho + np.sqrt(-2 * np.log(delta)) * s


def d_exact(s, delta, rho=1.0):
    """Smallest distance with P(collision) <= delta. Returns 0 if even d = 0 is "safe" (probability dilution)."""
    if p_coll(0.0, s, rho) <= delta:
        return 0.0
    return brentq(lambda d: p_coll(d, s, rho) - delta, 0.0, rho + 50 * s)


# %% table
S_TABLE = [0.1, 0.25, 0.35, 0.5, 0.7, 1.0, 2.0, 4.0]
DELTAS = [1e-1, 1e-2, 1e-3, 1e-4]
print(f"{'delta':>7} {'s/rho':>5} | {'d_half':>6} {'d_exact':>7} {'d_ell':>6} | {'Pc@d_half/delta':>15}")
for dl in DELTAS:
    for s in S_TABLE:
        dh = d_half(s, dl)
        print(f"{dl:7.0e} {s:5.2f} | {dh:6.3f} {d_exact(s, dl):7.3f} {d_ellipse(s, dl):6.3f} | "
              f"{p_coll(dh, s) / dl:15.2f}")
    print()

# %% your numbers: sim units (rho = 2r = 0.1), sigma per agent, independent errors
RHO = 0.1
for name, s_agent in [("WAAS-like", 0.001), ("NACp 8", 0.025), ("sweep", 0.05)]:
    s_rel = np.sqrt(2) * s_agent
    for dl in [1e-2, 1e-5]:
        print(f"{name:10s} sigma_agent={s_agent:.3f} sigma_rel={s_rel:.3f} delta={dl:.0e}: "
              f"half-space radius {d_half(s_rel, dl, RHO):.3f}, exact {d_exact(s_rel, dl, RHO):.3f}")

# %% figure
s_grid = np.linspace(0.05, 2.5, 200)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated default categorical slots 1-3
INK, MUTED, GRID = "#222222", "#6b6b66", "#e4e3df"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.1), constrained_layout=True)
dl = 1e-2
for y, c, lab in [([d_ellipse(s, dl) for s in s_grid], AQUA, "2-D ellipse"),
                  ([d_half(s, dl) for s in s_grid], BLUE, "half-space"),
                  ([d_exact(s, dl) for s in s_grid], ORANGE, "exact disk")]:
    ax1.plot(s_grid, y, color=c, lw=2, label=lab)
ax1.set(xlabel="relative σ / ρ", ylabel="required mean distance / ρ", title="Required distance, δ = 1e-2")
ax1.legend(frameon=False, fontsize=8, loc="lower right")

for dl, c in [(1e-2, BLUE), (1e-4, ORANGE)]:
    ax2.plot(s_grid, [p_coll(d_half(s, dl), s) / dl for s in s_grid], color=c, lw=2, label=f"δ = {dl:.0e}")
ax2.axhline(1.0, color=MUTED, lw=1, ls="--")
ax2.set(xlabel="relative σ / ρ", ylabel="true P(collision) / δ", title="Half-space: actual risk vs target",
        ylim=(0, 1.05))
ax2.legend(frameon=False, fontsize=8, loc="upper right")
ax1.set_ylim(0, None)

for ax in (ax1, ax2):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)
    ax.title.set_fontsize(10)
    for v, lab in [(0.35, "NACp 8"), (0.71, "sweep σ=0.05")]:
        ax.axvline(v, color=MUTED, lw=0.8, ls=":")
        y0, y1 = ax.get_ylim()
        top = ax is ax1  # left panel: label at the top (lines are low there); right panel: at the bottom
        ax.text(v, y1 - 0.03 * (y1 - y0) if top else y0 + 0.03 * (y1 - y0), lab + " ", color=MUTED, fontsize=7,
                rotation=90, va="top" if top else "bottom", ha="right")

out = pathlib.Path(__file__).resolve().parent / "figures" / "halfspace_conservatism.png"
fig.savefig(out, dpi=200)
print("saved", out)
