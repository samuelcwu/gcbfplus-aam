# Compare the 100-map and 1000-map runs of scratch/di_noise_sweep.py: does the safety curve hold up, how much do the
# confidence intervals tighten, and what can be said about points with zero collisions.
# Run cell-by-cell in the VS Code Interactive Window, or: python scratch/di_noise_sweep_compare.py

# %% load
import pathlib

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd()
RUNS = {100: REPO_ROOT / "scratch/results/di_noise_sweep_epi100.npz",
        1000: REPO_ROOT / "scratch/results/di_noise_sweep_epi1000.npz"}
FIGURE_PATH = REPO_ROOT / "scratch/figures/di_noise_sweep_compare.png"

res = {n_epi: np.load(path) for n_epi, path in RUNS.items()}
r_small, r_big = res[100], res[1000]
sigmas, noise_names, n_agents_list = r_big["sigmas"], list(r_big["noise_names"]), list(r_big["n_agents_list"])
assert np.array_equal(sigmas, r_small["sigmas"]) and list(r_small["noise_names"]) == noise_names

# episodes 0-99 of the 1000-map run use the same maps and noise draws as the 100-map run
for n_agents in n_agents_list:
    for k in ["ever_unsafe", "ever_finish"]:
        same = np.array_equal(r_big[f"{k}_n{n_agents}"][:, :, :100], r_small[f"{k}_n{n_agents}"])
        print(f"n={n_agents} {k}: first 100 episodes identical to the 100-map run: {same}")


def safety_stats(r, n_agents: int):
    """Per-episode safety rate -> (mean, 95% CI half-width), each (noise type, sigma), in %; plus unsafe agent count."""
    unsafe = r[f"ever_unsafe_n{n_agents}"]  # (noise type, sigma, episode, agent)
    rate = 100 * (~unsafe).mean(-1)
    return rate.mean(-1), 1.96 * rate.std(-1) / np.sqrt(rate.shape[-1]), unsafe.sum((-1, -2)), unsafe.shape[-2] * n_agents


# %% table: safety % and CI half-width for both runs; for zero-collision points, the 95% upper bound on the per-agent
# collision probability from the rule of three (0 events in n trials -> p < 3/n). Agents in one episode are not
# independent (collisions come in pairs), so treat this bound as optimistic.
print(f"\n{'N':>2} {'noise':>5} {'sigma':>6} {'safe% (100)':>13} {'safe% (1000)':>14} {'unsafe/agent-runs':>18} {'p_coll 95% UB':>14}")
for n_agents in n_agents_list:
    m_s, ci_s, _, _ = safety_stats(r_small, n_agents)
    m_b, ci_b, bad_b, runs_b = safety_stats(r_big, n_agents)
    for i_type, name in enumerate(noise_names):
        for i_sig, s in enumerate(sigmas):
            ub = f"< {100 * 3 / runs_b:.2f}%" if bad_b[i_type, i_sig] == 0 else ""
            print(f"{n_agents:>2} {name:>5} {s:6.3f} {m_s[i_type, i_sig]:7.2f}±{ci_s[i_type, i_sig]:<5.2f}"
                  f" {m_b[i_type, i_sig]:8.2f}±{ci_b[i_type, i_sig]:<5.2f}"
                  f" {f'{bad_b[i_type, i_sig]}/{runs_b}':>18} {ub:>14}")

# %% figure: safety rate vs sigma, 100-map vs 1000-map, rows = agent count, columns = noise type
COLORS = {100: "0.5", 1000: "C0"}
fig, axs = plt.subplots(len(n_agents_list), len(noise_names), figsize=(12, 4.2 * len(n_agents_list)), sharex=True,
                        squeeze=False)
for i_n, n_agents in enumerate(n_agents_list):
    for n_epi, r in res.items():
        mean, ci, _, _ = safety_stats(r, n_agents)
        for i_type, name in enumerate(noise_names):
            ax = axs[i_n, i_type]
            ls = "--" if n_epi == 100 else "-"
            ax.plot(sigmas, mean[i_type], "o" + ls, color=COLORS[n_epi], ms=4, label=f"{n_epi} maps")
            ax.fill_between(sigmas, mean[i_type] - ci[i_type], np.minimum(mean[i_type] + ci[i_type], 100),
                            color=COLORS[n_epi], alpha=0.2)
    for i_type, (name, tau) in enumerate(zip(noise_names, r_big["noise_taus"])):
        ax = axs[i_n, i_type]
        ax.set_title(f"{n_agents} agents, " + (f"Gauss-Markov (τ = {tau:g} s)" if tau > 0 else "white noise"))
        ax.set_ylabel("safety rate %")
        ax.grid(alpha=0.3)
        if i_n == len(n_agents_list) - 1:
            ax.set_xlabel("noise σ (position and velocity, per axis)")
axs[0, 0].legend(fontsize=9, loc="lower left")
fig.suptitle("Pretrained GCBF+ (DoubleIntegrator) safety vs noise: 100 vs 1000 random maps per point "
             "(bands = 95% CI over maps)")
fig.tight_layout()
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE_PATH, dpi=110)
print("saved", FIGURE_PATH)
plt.show()

# %%
