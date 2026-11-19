# Scratch experiment: test-time line-of-sight tightening of the pretrained GCBF+ DoubleIntegrator policy under
# navigation noise, plus a statistical certificate on the per-episode collision probability.
#
# Noise model is the same as scratch/di_noise_sweep.py: each agent broadcasts x_i + n_i and (OWNSHIP_NOISE) also
# believes it is at x_i + n_i; n_i is stationary Gauss-Markov (or white), so every report error is N(0, sigma^2).
# The relative-position belief is then N(noisy edge, 2 sigma^2 I) without any filter, and the tightening is
#   position: each neighbor is shown m_pos = kappa * sigma_rel closer along the line of sight (sigma_rel = sqrt(2) sigma)
#   velocity: (arm "posvel") the neighbor is also shown closing m_vel = kappa * sigma_rel faster along the line of sight
# This is the belief-CBF margin of Vahs et al. (RA-L 2023) applied to the GNN's input, not to h. Because the GCBF is
# learned (not convex), its effect on the collision probability is checked statistically:
#   per-episode collision = any agent collides; one-sided 95% Clopper-Pearson upper bound;
#   kappa* = smallest kappa whose bound <= TARGET on the CALIBRATION episodes, then certified on the held-out TEST
#   episodes (choosing and certifying on the same episodes would invalidate the bound).
# Same map keys (seed 1234) and noise keys (seed 0) as di_noise_sweep.py, so kappa = 0 must reproduce its flags.
# Run cell-by-cell in the VS Code Interactive Window, or: python scratch/di_tighten_sweep.py

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
from scipy.stats import beta

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.env.utils import inside_obstacles
from gcbfplus.utils.graph import GraphsTuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd()
MODEL_PATH = REPO_ROOT / "pretrained/DoubleIntegrator/gcbf+"

RUN_SWEEP = True  # False: skip the simulations and only re-plot from RESULTS_PATH
N_AGENTS = 8
AREA_SIZE = 4
N_OBS = 4
MAX_STEP = 1024
# Keys come from fixed pools of 1000 (same as di_noise_sweep.py). The 100-episode pilot (both arms, 7 kappas, white
# check) is saved as *_epi100.*; the full run below drops the velocity arm and the extreme kappas (pilot findings).
N_EPI = int(os.environ.get("N_EPI", 1000))
CHUNK = 100  # episodes per vmapped batch; N_EPI must be a multiple
SEED = 1234  # map keys, built exactly as in test.py
NOISE_SEED = 0
OWNSHIP_NOISE = True
# (noise name, tau [s], sigma, arm). tau = 0 -> white. arm "pos": position margin only; "posvel": position + velocity
CELLS = [("gm", 1.0, s, "pos") for s in [0.025, 0.05, 0.075]]
KAPPAS = [float(k) for k in os.environ.get("KAPPAS", "0,1,1.5,2,2.5").split(",")]
TARGET = 0.05  # per-episode collision probability to certify
CONF = 0.95  # one-sided confidence of the Clopper-Pearson upper bound
OLD_RESULTS_PATH = REPO_ROOT / "scratch/results/di_noise_sweep_epi1000.npz"  # for the kappa = 0 reproduction check
RUN_TAG = os.environ.get("RUN_TAG", "")  # e.g. "_smoke" for a quick test that must not overwrite real results

RESULTS_PATH = REPO_ROOT / f"scratch/results/di_tighten_sweep_epi{N_EPI}{RUN_TAG}.npz"
SUMMARY_PATH = REPO_ROOT / f"scratch/results/di_tighten_sweep_epi{N_EPI}{RUN_TAG}_summary.csv"
FIGURE_PATH = REPO_ROOT / f"scratch/figures/di_tighten_sweep_epi{N_EPI}{RUN_TAG}.png"

with open(MODEL_PATH / "config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)


def load_env_algo(n_agents: int):
    """env + pretrained algo for one agent count (mirrors test.py; copied from di_noise_sweep.py)."""
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


# %% noise model (copied from di_noise_sweep.py)
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
    LiDAR edges stay true. Agent rows of states also get the noise, so u_ref uses the noisy own state.
    (See di_noise_sweep.py for why the env's own edges are perturbed instead of rebuilt.)"""
    n_agents = a_noise.shape[0]
    node_noise = jnp.zeros_like(graph.states).at[:n_agents].set(a_noise)
    is_nav_edge = (graph.node_type[graph.receivers] == 0) & jnp.isin(graph.node_type[graph.senders], jnp.array([0, 1]))
    delta = node_noise[graph.receivers] - node_noise[graph.senders]
    return graph._replace(edges=graph.edges + is_nav_edge[:, None] * delta, states=graph.states + node_noise)


apply_noise = apply_nav_noise if OWNSHIP_NOISE else apply_broadcast_noise
SIGMA_REL_FACTOR = np.sqrt(2.0) if OWNSHIP_NOISE else 1.0  # sigma_rel / sigma per axis


# %% tightening
def tighten_edges(graph: GraphsTuple, m_pos: jax.Array, m_vel: jax.Array) -> GraphsTuple:
    """Show each neighbor m_pos closer along the line of sight and closing m_vel faster. Agent-agent edges only.
    Edge features are x_recv - x_send = [dp, dv]; u = dp / |dp| points from the sender to the receiver, and the range
    rate is u . dv, so subtracting m_vel * u makes the neighbor look like it is closing faster.
    With m_pos = m_vel = 0 the edges are returned bit-for-bit unchanged (d / d == 1 exactly)."""
    is_agent_edge = (graph.node_type[graph.receivers] == 0) & (graph.node_type[graph.senders] == 0)
    dp, dv = graph.edges[:, :2], graph.edges[:, 2:4]
    d = jnp.linalg.norm(dp, axis=-1, keepdims=True)
    d_safe = jnp.maximum(d, 1e-6)
    u = dp / d_safe
    dp_new = dp * (jnp.maximum(d - m_pos, 1e-3) / d_safe)
    dv_new = dv - m_vel * u
    edges_new = jnp.concatenate([dp_new, dv_new, graph.edges[:, 4:]], axis=-1)
    return graph._replace(edges=jnp.where(is_agent_edge[:, None], edges_new, graph.edges))


# %% collision split: the two terms that env.collision_mask ORs together (double_integrator.py:419-434)
def collision_split(env, graph: GraphsTuple) -> tuple[jax.Array, jax.Array]:
    """(a_hit_agent, a_hit_obs): agent within 2r of another agent / agent inside an obstacle (inflated by r)."""
    r = env._params["car_radius"]
    agent_pos = graph.type_states(type_idx=0, n_type=env.num_agents)[:, :2]
    dist = jnp.linalg.norm(agent_pos[:, None, :] - agent_pos[None, :, :], axis=-1)
    dist = dist + jnp.eye(dist.shape[1]) * (r * 2 + 1)  # remove self connection
    a_hit_agent = jnp.less(dist, r * 2).max(axis=1)
    a_hit_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, r)
    return a_hit_agent, a_hit_obs


# %% batched rollout: per-agent "ever collided" (with agent / obstacle split) / "ever reached goal" flags
def make_batch_rollout(env, algo):
    n_agents = env.num_agents

    def flags(graph):
        a_hit_agent, a_hit_obs = collision_split(env, graph)
        return env.collision_mask(graph), a_hit_agent, a_hit_obs, env.finish_mask(graph)

    def episode(key_x0, key_noise, sigma, phi, m_pos, m_vel):
        graph0 = env.reset(key_x0)
        key_init, key_steps = jr.split(key_noise)
        a_noise0 = gm_init(key_init, sigma, n_agents)

        def body(carry, key):
            graph, a_noise = carry
            action = algo.act(tighten_edges(apply_noise(graph, a_noise), m_pos, m_vel))
            graph_new = env.step(graph, action)[0]  # env steps the true graph
            a_noise_new = gm_step(a_noise, key, sigma, phi)
            return (graph_new, a_noise_new), flags(graph_new)

        _, T_flags = lax.scan(body, (graph0, a_noise0), jr.split(key_steps, MAX_STEP))
        # include the initial state, as test.py does; order: unsafe, hit_agent, hit_obs, finish
        return tuple(Ta.any(axis=0) | a0 for Ta, a0 in zip(T_flags, flags(graph0)))

    return jax.jit(jax.vmap(episode, in_axes=(0, 0, None, None, None, None)))


env, algo = load_env_algo(N_AGENTS)

# %% unit check of tighten_edges on a real graph that has agent-agent edges
def first_graph_with_agent_edges(max_tries: int = 20, max_steps: int = 400) -> GraphsTuple:
    act_step = jax.jit(lambda g: env.step(g, algo.act(g))[0])
    for i in range(max_tries):
        graph = env.reset(jr.PRNGKey(10_000 + i))
        for _ in range(max_steps):
            is_ae = (graph.node_type[graph.receivers] == 0) & (graph.node_type[graph.senders] == 0)
            if int(is_ae.sum()) > 0:
                return graph
            graph = act_step(graph)
    raise RuntimeError("no graph with agent-agent edges found")


g = first_graph_with_agent_edges()
is_ae = np.asarray((g.node_type[g.receivers] == 0) & (g.node_type[g.senders] == 0))
m = 0.02
gt = tighten_edges(g, jnp.float32(m), jnp.float32(m))
e, et = np.asarray(g.edges), np.asarray(gt.edges)
d, dt_ = np.linalg.norm(e[is_ae, :2], axis=-1), np.linalg.norm(et[is_ae, :2], axis=-1)
ok_d = d > m + 1e-3
assert np.allclose(dt_[ok_d], d[ok_d] - m, atol=1e-6), "distance must shrink by exactly m_pos"
u, ut = e[is_ae, :2] / d[:, None], et[is_ae, :2] / dt_[:, None]
assert np.allclose(u, ut, atol=1e-5), "bearing must be unchanged"
assert np.allclose((et[is_ae, 2:4] * u).sum(-1), (e[is_ae, 2:4] * u).sum(-1) - m, atol=1e-6), "range rate - m_vel"
assert np.array_equal(e[~is_ae], et[~is_ae]), "non agent-agent edges must be untouched"
assert np.array_equal(np.asarray(tighten_edges(g, jnp.float32(0), jnp.float32(0)).edges), e), "m = 0 is identity"
print(f"tighten_edges unit check passed ({int(is_ae.sum())} agent-agent edges)")

# %% sweep
if RUN_SWEEP:
    assert N_EPI <= 1000 and N_EPI % CHUNK == 0
    test_keys = jr.split(jr.PRNGKey(SEED), 1_000)[:N_EPI]  # same reset keys as test.py / di_noise_sweep.py
    key_x0s = jax.vmap(lambda k: jr.split(k, 2)[0])(test_keys)
    noise_keys = jr.split(jr.PRNGKey(NOISE_SEED), 1_000)[:N_EPI]
    batch_rollout = make_batch_rollout(env, algo)

    # (cell, kappa, episode, agent)
    ever_unsafe = np.zeros((len(CELLS), len(KAPPAS), N_EPI, N_AGENTS), dtype=bool)
    ever_hit_agent, ever_hit_obs, ever_finish = [np.zeros_like(ever_unsafe) for _ in range(3)]
    n_points, i_point, t_start = len(CELLS) * len(KAPPAS), 0, time.perf_counter()
    for i_c, (name, tau, s, arm) in enumerate(CELLS):
        phi = float(np.exp(-env.dt / tau)) if tau > 0 else 0.0
        sigma = jnp.array([s, s, s, s])
        for i_k, kappa in enumerate(KAPPAS):
            t0 = time.perf_counter()
            m = kappa * SIGMA_REL_FACTOR * s
            m_pos, m_vel = jnp.float32(m), jnp.float32(m if arm == "posvel" else 0.0)
            for c in range(0, N_EPI, CHUNK):
                out = batch_rollout(key_x0s[c:c + CHUNK], noise_keys[c:c + CHUNK], sigma, jnp.float32(phi),
                                    m_pos, m_vel)
                for arr, o in zip((ever_unsafe, ever_hit_agent, ever_hit_obs, ever_finish), out):
                    arr[i_c, i_k, c:c + CHUNK] = np.asarray(o)
            i_point += 1
            elapsed = time.perf_counter() - t_start
            print(f"[{i_point}/{n_points}] {name:>5} sigma={s:<6g} {arm:>6} kappa={kappa:<4g} margin={m:.3f} "
                  f"epi-collision={100 * ever_unsafe[i_c, i_k].any(-1).mean():6.2f}% "
                  f"agent-safe={100 * (1 - ever_unsafe[i_c, i_k].mean()):6.2f}% "
                  f"(agent-agent {100 * ever_hit_agent[i_c, i_k].any(-1).mean():.1f}%, "
                  f"obstacle {100 * ever_hit_obs[i_c, i_k].any(-1).mean():.1f}%) "
                  f"reach={100 * ever_finish[i_c, i_k].mean():6.2f}%  ({time.perf_counter() - t0:.0f} s; "
                  f"ETA {elapsed / i_point * (n_points - i_point) / 60:.0f} min)", flush=True)
    print(f"sweep done in {time.perf_counter() - t_start:.0f} s")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS_PATH, cell_names=np.array([c[0] for c in CELLS]), cell_taus=np.array([c[1] for c in CELLS]),
             cell_sigmas=np.array([c[2] for c in CELLS]), cell_arms=np.array([c[3] for c in CELLS]),
             kappas=np.array(KAPPAS), sigma_rel_factor=SIGMA_REL_FACTOR, n_agents=N_AGENTS, max_step=MAX_STEP,
             n_epi=N_EPI, area_size=AREA_SIZE, n_obs=N_OBS, ownship_noise=OWNSHIP_NOISE, seed=SEED,
             noise_seed=NOISE_SEED, ever_unsafe=ever_unsafe, ever_hit_agent=ever_hit_agent,
             ever_hit_obs=ever_hit_obs, ever_finish=ever_finish)
    print("saved", RESULTS_PATH)

# %% sanity: kappa = 0 reproduces di_noise_sweep.py (same keys, edges bit-for-bit unchanged); both arms agree at 0
res = np.load(RESULTS_PATH)
cell_names, cell_sigmas, cell_arms = list(res["cell_names"]), res["cell_sigmas"], list(res["cell_arms"])
kappas, n_epi = res["kappas"], int(res["n_epi"])
unsafe_all, finish_all = res["ever_unsafe"], res["ever_finish"]
has_split = "ever_hit_agent" in res.files  # the 100-episode pilot predates the split
if has_split:
    hit_agent_all, hit_obs_all = res["ever_hit_agent"], res["ever_hit_obs"]
    assert np.array_equal(hit_agent_all | hit_obs_all, unsafe_all), "split must OR back to collision_mask"
    print("collision split ORs back to the total: OK")
i_k0 = list(kappas).index(0.0)
if OLD_RESULTS_PATH.exists():
    old = np.load(OLD_RESULTS_PATH)
    old_names, old_sigmas = list(old["noise_names"]), list(np.round(old["sigmas"], 6))
    for i_c, (name, s) in enumerate(zip(cell_names, cell_sigmas)):
        if round(float(s), 6) not in old_sigmas:
            continue
        o_u = old[f"ever_unsafe_n{int(res['n_agents'])}"][old_names.index(name), old_sigmas.index(round(float(s), 6))]
        o_f = old[f"ever_finish_n{int(res['n_agents'])}"][old_names.index(name), old_sigmas.index(round(float(s), 6))]
        same = np.array_equal(unsafe_all[i_c, i_k0], o_u[:n_epi]) and np.array_equal(finish_all[i_c, i_k0], o_f[:n_epi])
        print(f"kappa=0 vs di_noise_sweep: {name} sigma={s:g} {cell_arms[i_c]}: {'identical' if same else 'DIFFERENT'}")
for i_c in range(len(cell_names)):
    for j_c in range(i_c + 1, len(cell_names)):
        if cell_names[i_c] == cell_names[j_c] and cell_sigmas[i_c] == cell_sigmas[j_c]:
            assert np.array_equal(unsafe_all[i_c, i_k0], unsafe_all[j_c, i_k0]), "arms must agree at kappa = 0"


# %% summary + certification (calibration = first half of the episodes, test = second half)
def cp_upper(k: int, n: int, conf: float = CONF) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial probability after k events in n trials."""
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


epi_coll = unsafe_all.any(-1)  # (cell, kappa, episode)
n_cal = n_epi // 2
cal, tst = slice(0, n_cal), slice(n_cal, n_epi)
rows, kappa_star = [], {}
print(f"\n{n_epi} maps ({n_cal} calibration + {n_epi - n_cal} test), {int(res['max_step'])} steps, "
      f"target P(collision per episode) <= {TARGET:g} at {100 * CONF:g}% confidence")
print(f"{'noise':>5} {'sigma':>6} {'arm':>6} {'kappa':>5} {'margin':>6} | {'epi-coll':>8} {'CP-up':>6} | "
      f"{'ag-ag':>6} {'obst':>6} | {'agent-safe':>10} {'reach':>6}")
for i_c, (name, s, arm) in enumerate(zip(cell_names, cell_sigmas, cell_arms)):
    for i_k, kappa in enumerate(kappas):
        k_all = int(epi_coll[i_c, i_k].sum())
        k_cal, k_tst = int(epi_coll[i_c, i_k, cal].sum()), int(epi_coll[i_c, i_k, tst].sum())
        safe = 1 - unsafe_all[i_c, i_k].mean()
        reach = finish_all[i_c, i_k].mean()
        margin = kappa * float(res["sigma_rel_factor"]) * s
        r_ag = hit_agent_all[i_c, i_k].any(-1).mean() if has_split else np.nan
        r_ob = hit_obs_all[i_c, i_k].any(-1).mean() if has_split else np.nan
        rows.append([name, s, arm, kappa, margin, k_all / n_epi, cp_upper(k_all, n_epi), k_cal, cp_upper(k_cal, n_cal),
                     k_tst, cp_upper(k_tst, n_epi - n_cal), r_ag, r_ob, safe, reach])
        print(f"{name:>5} {s:6.3f} {arm:>6} {kappa:5.1f} {margin:6.3f} | {100 * k_all / n_epi:7.1f}% "
              f"{100 * cp_upper(k_all, n_epi):5.1f}% | {100 * r_ag:5.1f}% {100 * r_ob:5.1f}% | "
              f"{100 * safe:9.2f}% {100 * reach:5.1f}%")
    ok = [i_k for i_k in range(len(kappas)) if cp_upper(int(epi_coll[i_c, i_k, cal].sum()), n_cal) <= TARGET]
    kappa_star[i_c] = ok[0] if ok else None
    if ok:
        k_tst = int(epi_coll[i_c, ok[0], tst].sum())
        print(f"  -> kappa* = {kappas[ok[0]]:g}; test: {k_tst}/{n_epi - n_cal} episodes with a collision, "
              f"CP upper bound {100 * cp_upper(k_tst, n_epi - n_cal):.1f}% "
              f"({'certified' if cp_upper(k_tst, n_epi - n_cal) <= TARGET else 'NOT certified'})")
    else:
        best = min(cp_upper(int(epi_coll[i_c, i_k, cal].sum()), n_cal) for i_k in range(len(kappas)))
        print(f"  -> no kappa reaches the target on calibration (best CP upper bound {100 * best:.1f}%; "
              f"the smallest possible with {n_cal} episodes is {100 * cp_upper(0, n_cal):.1f}%)")

SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SUMMARY_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["noise", "sigma", "arm", "kappa", "margin", "epi_coll_rate", "epi_coll_cp_upper", "k_cal",
                     "cp_upper_cal", "k_test", "cp_upper_test", "epi_rate_agent_agent", "epi_rate_obstacle",
                     "agent_safety", "agent_reach"])
    writer.writerows(rows)
print("saved", SUMMARY_PATH)

# %% plot: rows = noise setting (sigma); columns = P(collision per episode), split by type, reach
BLUE, ORANGE, AQUA, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b66", "#e4e3df"  # validated slots 1-3
ARM_STYLE = {"pos": (BLUE, "position margin"), "posvel": (ORANGE, "position + velocity margin")}
groups = []  # one row per (noise, sigma)
for name, s in zip(cell_names, cell_sigmas):
    if (name, s) not in groups:
        groups.append((name, s))
fig, axs = plt.subplots(len(groups), 3, figsize=(13, 3.1 * len(groups)), squeeze=False)
for i_g, (name, s) in enumerate(groups):
    ax_c, ax_s, ax_r = axs[i_g]
    tag = f"{'Gauss-Markov' if name == 'gm' else 'white'} σ = {s:g}"
    for i_c in [i for i in range(len(cell_names)) if (cell_names[i], cell_sigmas[i]) == (name, s)]:
        color, label = ARM_STYLE[cell_arms[i_c]]
        rate = 100 * epi_coll[i_c].mean(-1)
        upper = [100 * cp_upper(int(epi_coll[i_c, i_k].sum()), n_epi) for i_k in range(len(kappas))]
        ax_c.plot(kappas, rate, "o-", color=color, ms=4, lw=2, label=label)
        ax_c.plot(kappas, upper, "--", color=color, lw=1, label="95% upper bound")
        if kappa_star[i_c] is not None:
            ax_c.plot(kappas[kappa_star[i_c]], rate[kappa_star[i_c]], "*", color=color, ms=13, label="κ* (calibration)")
        ax_r.plot(kappas, 100 * finish_all[i_c].mean((-1, -2)), "o-", color=color, ms=4, lw=2, label=label)
        if has_split and cell_arms[i_c] == "pos":
            ax_s.plot(kappas, rate, "o-", color=BLUE, ms=4, lw=2, label="any collision")
            ax_s.plot(kappas, 100 * hit_agent_all[i_c].any(-1).mean(-1), "s-", color=ORANGE, ms=4, lw=2,
                      label="agent-agent")
            ax_s.plot(kappas, 100 * hit_obs_all[i_c].any(-1).mean(-1), "^-", color=AQUA, ms=5, lw=2,
                      label="agent-obstacle")
    ax_c.axhline(100 * TARGET, color=MUTED, lw=1, ls=":", label=f"target {100 * TARGET:g}%")
    ax_c.set(title=f"{tag}: episodes with a collision", xlabel="κ (margin = κ·σ_rel)", ylabel="%", ylim=(0, None))
    ax_s.set(title=f"{tag}: by collision type", xlabel="κ", ylabel="% of episodes", ylim=(0, None))
    ax_r.set(title=f"{tag}: reach (per agent)", xlabel="κ", ylabel="%")
    if not has_split:
        ax_s.text(0.5, 0.5, "no split in this file", ha="center", va="center", transform=ax_s.transAxes, color=MUTED)
    for ax in (ax_c, ax_s, ax_r):
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
axs[0, 0].legend(frameon=False, fontsize=8)
axs[0, 1].legend(frameon=False, fontsize=8)
fig.suptitle(f"Pretrained GCBF+ with line-of-sight tightening of neighbor edges, n = {int(res['n_agents'])}, "
             f"{n_epi} maps per point\nupper bound = one-sided {100 * CONF:g}% Clopper-Pearson over all episodes; "
             f"κ* chosen on the first {n_cal} episodes, certified on the last {n_epi - n_cal}", fontsize=10)
fig.tight_layout()
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE_PATH, dpi=110)
print("saved", FIGURE_PATH)
plt.show()

# %%
