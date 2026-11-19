# Scratch experiment: how the pretrained GCBF+ DoubleIntegrator policy degrades under per-agent navigation noise.
# Two noise arms, same model as scratch/di_gm_noise.py (each agent broadcasts x_i + n_i; with OWNSHIP_NOISE it also
# believes it is at x_i + n_i):
#   "white": n_i redrawn independently every step (Gauss-Markov with phi = 0)
#   "gm":    first-order Gauss-Markov, correlation time TAU
# Evaluation mirrors the paper / test.py: each episode is a different random map (obstacles, starts, goals from
# env.reset(key)), keys from a fixed seed, so every (noise type, sigma) point is scored on the SAME maps. The noise
# draws are shared too (for a fixed key the noise is linear in sigma), so points differ only by the noise setting.
# Metrics per episode (test.py:184-186): safety = fraction of agents never colliding, reach = fraction ever reaching
# the goal, success = both. Note: the paper also averages over 3 policy seeds; only one pretrained policy exists here.
# Run cell-by-cell in the VS Code Interactive Window, or: python scratch/di_noise_sweep.py

# %% setup / config
import csv
import os
import pathlib
import time

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import yaml

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.utils.graph import GraphsTuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd()
MODEL_PATH = REPO_ROOT / "pretrained/DoubleIntegrator/gcbf+"

RUN_SWEEP = True  # False: skip the simulations and only re-plot from RESULTS_PATH
N_AGENTS_LIST = [4, 8]
AREA_SIZE = 4
N_OBS = 4
MAX_STEP = 1024  # paper uses 4096; test.py default is 256
N_EPI = 1000  # <= 1000. Map and noise keys come from fixed pools of 1000, so N_EPI = 100 is a subset of N_EPI = 1000
CHUNK = 100  # episodes per vmapped batch; N_EPI must be a multiple. Fixed batch shape -> one compile per agent count
SEED = 1234  # map keys, built exactly as in test.py
NOISE_SEED = 0
NOISE_TAUS = {"white": 0.0, "gm": 1.0}  # correlation time [s]; 0 -> phi = 0 -> white
# stationary std of each agent's error, applied to [x, y] and [vx, vy] alike. For scale: agent radius r = 0.05,
# collision distance 2r = 0.1, sensing radius R = 0.5. Capped at 2r: beyond that white noise is unrealistically jumpy.
SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
OWNSHIP_NOISE = True  # True: agent uses x_i + n_i for itself too. False: broadcast-only (receiver uses true state)

RESULTS_PATH = REPO_ROOT / f"scratch/results/di_noise_sweep_epi{N_EPI}.npz"
SUMMARY_PATH = REPO_ROOT / f"scratch/results/di_noise_sweep_epi{N_EPI}_summary.csv"
FIGURE_PATH = REPO_ROOT / f"scratch/figures/di_noise_sweep_epi{N_EPI}.png"

with open(MODEL_PATH / "config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)


def load_env_algo(n_agents: int):
    """env + pretrained algo for one agent count (mirrors test.py)."""
    env = make_env(env_id="DoubleIntegrator", num_agents=n_agents, area_size=AREA_SIZE, max_step=MAX_STEP,
                   num_obs=N_OBS)
    algo = make_algo(
        algo=config.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        gnn_layers=config.gnn_layers,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        horizon=config.horizon,
        lr_actor=config.lr_actor,
        lr_cbf=config.lr_cbf,
        alpha=config.alpha,
        eps=0.02,
        inner_epoch=8,
        loss_action_coef=config.loss_action_coef,
        loss_unsafe_coef=config.loss_unsafe_coef,
        loss_safe_coef=config.loss_safe_coef,
        loss_h_dot_coef=config.loss_h_dot_coef,
        max_grad_norm=2.0,
        seed=config.seed,
    )
    model_dir = MODEL_PATH / "models"
    step = max(int(m) for m in os.listdir(model_dir) if m.isdigit())
    algo.load(str(model_dir), step)
    return env, algo


# %% noise model (copied from di_gm_noise.py, with n_agents passed in and phi traced so white / GM share one compile)
def gm_init(key: jax.Array, sigma: jax.Array, n_agents: int) -> jax.Array:
    """Draw a_noise from the stationary distribution N(0, sigma^2). Shape (n_agents, 4)."""
    return jr.normal(key, (n_agents, 4)) * sigma


def gm_step(a_noise: jax.Array, key: jax.Array, sigma: jax.Array, phi: jax.Array) -> jax.Array:
    """n_{t+1} = phi * n_t + sqrt(1 - phi^2) * sigma * w_t; keeps the variance at sigma^2. phi = 0 is white noise."""
    return phi * a_noise + jnp.sqrt(1 - phi ** 2) * sigma * jr.normal(key, a_noise.shape)


def apply_broadcast_noise(graph: GraphsTuple, a_noise: jax.Array) -> GraphsTuple:
    """Agent-agent edges become state[recv] - (state[send] + a_noise[send]). Goal / lidar edges, states stay true."""
    n_agents = a_noise.shape[0]
    is_agent_edge = (graph.node_type[graph.receivers] == 0) & (graph.node_type[graph.senders] == 0)
    send_noise = a_noise[jnp.minimum(graph.senders, n_agents - 1)]  # agents are nodes 0..n-1; pad ids are masked
    return graph._replace(edges=graph.edges - is_agent_edge[:, None] * send_noise)


def apply_nav_noise(graph: GraphsTuple, a_noise: jax.Array) -> GraphsTuple:
    """Agent believes it is at x_i + n_i: agent-agent / agent-goal edges get node_noise[recv] - node_noise[send].
    Noise is added to the env's own edges rather than rebuilding them with env.add_edge_feats, because the env's goal
    edges are not actually clipped (edge_blocks slices goals, not columns, double_integrator.py:239-244) while
    add_edge_feats clips them. LiDAR is an onboard relative sensor, so its edges stay true.
    Agent rows of states also get the noise, so u_ref uses the noisy own state."""
    n_agents = a_noise.shape[0]
    node_noise = jnp.zeros_like(graph.states).at[:n_agents].set(a_noise)
    is_nav_edge = (graph.node_type[graph.receivers] == 0) & jnp.isin(graph.node_type[graph.senders], jnp.array([0, 1]))
    delta = node_noise[graph.receivers] - node_noise[graph.senders]
    return graph._replace(edges=graph.edges + is_nav_edge[:, None] * delta, states=graph.states + node_noise)


apply_noise = apply_nav_noise if OWNSHIP_NOISE else apply_broadcast_noise


# %% batched rollout: only per-agent "ever collided" / "ever reached goal" flags are kept (no graphs stored)
def make_batch_rollout(env, algo):
    n_agents = env.num_agents

    def episode(key_x0, key_noise, sigma, phi):
        graph0 = env.reset(key_x0)
        key_init, key_steps = jr.split(key_noise)
        a_noise0 = gm_init(key_init, sigma, n_agents)

        def body(carry, key):
            graph, a_noise = carry
            action = algo.act(apply_noise(graph, a_noise))  # policy sees the noisy graph
            graph_new = env.step(graph, action)[0]  # env steps the true graph
            a_noise_new = gm_step(a_noise, key, sigma, phi)
            return (graph_new, a_noise_new), (env.collision_mask(graph_new), env.finish_mask(graph_new))

        _, (Ta_unsafe, Ta_finish) = lax.scan(body, (graph0, a_noise0), jr.split(key_steps, MAX_STEP))
        # include the initial state, as test.py does (its masks run over Tp1_graph)
        a_ever_unsafe = Ta_unsafe.any(axis=0) | env.collision_mask(graph0)
        a_ever_finish = Ta_finish.any(axis=0) | env.finish_mask(graph0)
        return a_ever_unsafe, a_ever_finish

    return jax.jit(jax.vmap(episode, in_axes=(0, 0, None, None)))


# %% sweep
NOISE_NAMES = list(NOISE_TAUS)
if RUN_SWEEP:
    assert N_EPI <= 1000 and N_EPI % CHUNK == 0
    n_points, i_point = len(N_AGENTS_LIST) * len(NOISE_NAMES) * len(SIGMAS), 0
    # run time per point is roughly proportional to the agent count; use that to weight the ETA
    work_total, work_done = sum(N_AGENTS_LIST) * len(NOISE_NAMES) * len(SIGMAS), 0
    test_keys = jr.split(jr.PRNGKey(SEED), 1_000)[:N_EPI]  # same reset keys as test.py
    key_x0s = jax.vmap(lambda k: jr.split(k, 2)[0])(test_keys)  # test.py: key_x0, _ = jr.split(test_keys[i_epi], 2)
    noise_keys = jr.split(jr.PRNGKey(NOISE_SEED), 1_000)[:N_EPI]

    saved = {}
    t_start = time.perf_counter()
    for n_agents in N_AGENTS_LIST:
        env, algo = load_env_algo(n_agents)
        batch_rollout = make_batch_rollout(env, algo)
        # (noise type, sigma, episode, agent)
        ever_unsafe = np.zeros((len(NOISE_NAMES), len(SIGMAS), N_EPI, n_agents), dtype=bool)
        ever_finish = np.zeros_like(ever_unsafe)
        for i_type, name in enumerate(NOISE_NAMES):
            tau = NOISE_TAUS[name]
            phi = float(np.exp(-env.dt / tau)) if tau > 0 else 0.0
            for i_sig, s in enumerate(SIGMAS):
                t0 = time.perf_counter()
                sigma = jnp.array([s, s, s, s])
                for c in range(0, N_EPI, CHUNK):
                    u, fin = batch_rollout(key_x0s[c:c + CHUNK], noise_keys[c:c + CHUNK], sigma, jnp.float32(phi))
                    ever_unsafe[i_type, i_sig, c:c + CHUNK] = np.asarray(u)
                    ever_finish[i_type, i_sig, c:c + CHUNK] = np.asarray(fin)
                i_point, work_done = i_point + 1, work_done + n_agents
                elapsed = time.perf_counter() - t_start
                eta = elapsed / work_done * (work_total - work_done)
                print(f"[{i_point}/{n_points}] n={n_agents} {name:>5} sigma={s:<6g} "
                      f"safe={100 * (1 - ever_unsafe[i_type, i_sig].mean()):6.2f}% "
                      f"reach={100 * ever_finish[i_type, i_sig].mean():6.2f}%  ({time.perf_counter() - t0:.0f} s; "
                      f"elapsed {elapsed / 60:.0f} min, ETA {eta / 60:.0f} min)", flush=True)
        # sanity: with sigma = 0 the noise type cannot matter (same maps, zero noise)
        i0 = SIGMAS.index(0.0)
        assert np.array_equal(ever_unsafe[:, i0], ever_unsafe[:1, i0].repeat(len(NOISE_NAMES), 0))
        assert np.array_equal(ever_finish[:, i0], ever_finish[:1, i0].repeat(len(NOISE_NAMES), 0))
        saved[f"ever_unsafe_n{n_agents}"], saved[f"ever_finish_n{n_agents}"] = ever_unsafe, ever_finish
    print(f"sweep done in {time.perf_counter() - t_start:.0f} s")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS_PATH, sigmas=np.array(SIGMAS), noise_names=np.array(NOISE_NAMES),
             noise_taus=np.array([NOISE_TAUS[k] for k in NOISE_NAMES]), n_agents_list=np.array(N_AGENTS_LIST),
             max_step=MAX_STEP, n_epi=N_EPI, area_size=AREA_SIZE, n_obs=N_OBS, ownship_noise=OWNSHIP_NOISE,
             seed=SEED, noise_seed=NOISE_SEED, **saved)
    print("saved", RESULTS_PATH)

# %% summary: per-episode rates, then mean / std / 95% CI over episodes
res = np.load(RESULTS_PATH)
sigmas, noise_names, n_agents_list = res["sigmas"], list(res["noise_names"]), list(res["n_agents_list"])
n_epi = int(res["n_epi"])
METRICS = ["safety", "reach", "success"]


def episode_rates(n_agents: int) -> np.ndarray:
    """(metric, noise type, sigma, episode) fraction of agents, in [0, 1]."""
    unsafe, finish = res[f"ever_unsafe_n{n_agents}"], res[f"ever_finish_n{n_agents}"]
    return np.stack([(~unsafe).mean(-1), finish.mean(-1), (~unsafe & finish).mean(-1)])


rows = []
print(f"\n{n_epi} maps, {int(res['max_step'])} steps, area {float(res['area_size']):g}, {int(res['n_obs'])} obstacles")
print(f"{'N':>2} {'noise':>5} {'sigma':>6} {'safe%':>7} {'reach%':>7} {'succ%':>7} {'unsafe agents':>14}")
for n_agents in n_agents_list:
    rates = episode_rates(n_agents)
    unsafe = res[f"ever_unsafe_n{n_agents}"]
    for i_type, name in enumerate(noise_names):
        for i_sig, s in enumerate(sigmas):
            r = rates[:, i_type, i_sig]  # (metric, episode)
            mean, std = r.mean(-1), r.std(-1)
            for i_m, m in enumerate(METRICS):
                rows.append([n_agents, name, s, m, mean[i_m], std[i_m], 1.96 * std[i_m] / np.sqrt(n_epi)])
            n_bad = int(unsafe[i_type, i_sig].sum())
            print(f"{n_agents:>2} {name:>5} {s:6.3f} {100 * mean[0]:7.2f} {100 * mean[1]:7.2f} {100 * mean[2]:7.2f} "
                  f"{f'{n_bad}/{unsafe[i_type, i_sig].size}':>14}")

SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SUMMARY_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n_agents", "noise", "sigma", "metric", "mean", "std", "ci95_halfwidth"])
    writer.writerows(rows)
print("saved", SUMMARY_PATH)

# %% plot: rows = agent count, columns = metric; one line per noise type with a 95% CI band
COLORS = {"white": "C1", "gm": "C0"}
LABELS = {name: (f"Gauss-Markov (τ = {tau:g} s)" if tau > 0 else "white (redrawn every step)")
          for name, tau in zip(noise_names, res["noise_taus"])}
fig, axs = plt.subplots(len(n_agents_list), len(METRICS), figsize=(15, 4.2 * len(n_agents_list)), sharex=True,
                        squeeze=False)
for i_n, n_agents in enumerate(n_agents_list):
    rates = 100 * episode_rates(n_agents)
    for i_m, m in enumerate(METRICS):
        ax = axs[i_n, i_m]
        for i_type, name in enumerate(noise_names):
            r = rates[i_m, i_type]  # (sigma, episode)
            mean, ci = r.mean(-1), 1.96 * r.std(-1) / np.sqrt(n_epi)
            ax.plot(sigmas, mean, "o-", color=COLORS.get(name), ms=4, label=LABELS[name])
            ax.fill_between(sigmas, mean - ci, np.minimum(mean + ci, 100), color=COLORS.get(name), alpha=0.2)
        ax.axvline(0.05, color="0.5", lw=0.8, ls="--")
        ax.text(0.051, 0.03, "σ = agent radius", transform=ax.get_xaxis_transform(), fontsize=8, color="0.4")
        ax.set_title(f"{m} rate, {n_agents} agents")
        ax.set_ylabel("%")
        ax.grid(alpha=0.3)
        if i_n == len(n_agents_list) - 1:
            ax.set_xlabel("noise σ (position and velocity, per axis)")
axs[0, 0].legend(fontsize=9, loc="lower left")
fig.suptitle(f"Pretrained GCBF+ (DoubleIntegrator) under per-agent navigation noise"
             f"{' incl. ownship' if bool(res['ownship_noise']) else ' (broadcast only)'}: "
             f"{n_epi} random maps per point, {int(res['max_step'])} steps, area {float(res['area_size']):g}, "
             f"{int(res['n_obs'])} obstacles; bands = 95% CI over maps")
fig.tight_layout()
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE_PATH, dpi=110)
print("saved", FIGURE_PATH)
plt.show()

# %%
