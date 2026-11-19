# Scratch experiment: a filter layer between the noisy reports and the pretrained GCBF+ DoubleIntegrator policy.
#
# Same noise model as di_noise_sweep.py / di_tighten_sweep.py: agent j broadcasts z_j = x_j + n_j and also navigates
# on z_j (ownship noise); n_j is stationary Gauss-Markov. Instead of feeding z straight into the GNN, every agent i runs
#   a neighbor track of every other agent j (receiver x sender, belief_filter.nbr_tracks_*), and
#   an ownship track of itself (belief_filter.own_track_*, predicts with its own known action),
# and the GNN sees the tracks' MEAN: agent-agent edge = x_hat_i(own) - x_hat_{i->j}, goal edge and the agent's state
# row use x_hat_i(own). LiDAR stays true; which neighbors are adjacent is still decided from true positions.
# Arms: raw (no filter; bit-identical to the old sweeps), nbr (neighbor tracks only), own (ownship only), both.
# Neighbor reports can come every MEAS_EVERY steps (ADS-B-like ~1 Hz is 33); ownship always gets a fix every step.
# Same map keys (seed 1234) and noise keys (seed 0) as before, so every comparison is paired.
# Run cell-by-cell in the VS Code Interactive Window, or: python scratch/di_ekf_sweep.py
#   env knobs: N_EPI (default 1000), RUN_TAG (e.g. "_pilot")

# %% setup / config
import csv
import os
import pathlib
import sys
import time

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import beta, chi2

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.env.utils import inside_obstacles
from gcbfplus.utils.graph import GraphsTuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "scratch"))
import belief_filter as bf  # noqa: E402

MODEL_PATH = REPO_ROOT / "pretrained/DoubleIntegrator/gcbf+"

RUN_SWEEP = True  # False: skip the simulations and only summarize / re-plot from RESULTS_PATH
N_AGENTS = 8
AREA_SIZE = 4
N_OBS = 4
MAX_STEP = 1024
N_EPI = int(os.environ.get("N_EPI", 1000))
CHUNK = 100  # episodes per vmapped batch; N_EPI must be a multiple
SEED = 1234  # map keys, built exactly as in test.py
NOISE_SEED = 0
TAU = 1.0  # GM correlation time [s]
SIGMAS = [0.025, 0.05, 0.075, 0.1]
CONF = 0.95  # one-sided confidence of the Clopper-Pearson upper bound
R_TRACK = np.inf  # a receiver hears j only if its own report says j is within R_TRACK
T_DROP = 100  # steps without a report before a neighbor track is dropped
OWN_SIGMA_A = 0.05  # ownship process noise floor (the sim has no disturbance, so this is a near-perfect "IMU")
RUN_TAG = os.environ.get("RUN_TAG", "")

# runs: (label, sigma, use_nbr, use_own, nbr_model, sigma_a^2 scale, meas_every)
RUNS = [(arm, s, arm in ("nbr", "both"), arm in ("own", "both"), "cv_gm", 1.0, 1)
        for s in SIGMAS for arm in ("nbr", "own", "both")]
RUNS += [("both q x0.1", 0.05, True, True, "cv_gm", 0.1, 1),
         ("both q x10", 0.05, True, True, "cv_gm", 10.0, 1),
         ("both nbr-cv", 0.05, True, True, "cv", 1.0, 1),
         ("both 1Hz", 0.05, True, True, "cv_gm", 1.0, 33),
         ("raw-hold 1Hz", 0.05, False, False, "cv_gm", 1.0, 33)]

NOISE_SWEEP_PATH = REPO_ROOT / "scratch/results/di_noise_sweep_epi1000.npz"  # raw arm, all sigmas
TIGHTEN_PATH = REPO_ROOT / "scratch/results/di_tighten_sweep_epi1000.npz"  # raw (kappa = 0) and kappa = 2 overlay
RESULTS_PATH = REPO_ROOT / f"scratch/results/di_ekf_sweep_epi{N_EPI}{RUN_TAG}.npz"
SUMMARY_PATH = REPO_ROOT / f"scratch/results/di_ekf_sweep_epi{N_EPI}{RUN_TAG}_summary.csv"
FIGURE_PATH = REPO_ROOT / f"scratch/figures/di_ekf_sweep_epi{N_EPI}{RUN_TAG}.png"
DIAG_FIGURE_PATH = REPO_ROOT / f"scratch/figures/di_ekf_sweep_epi{N_EPI}{RUN_TAG}_diag.png"

with open(MODEL_PATH / "config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)


def load_env_algo(n_agents: int):
    """env + pretrained algo for one agent count (mirrors test.py; copied from di_tighten_sweep.py)."""
    env = make_env(env_id="DoubleIntegrator", num_agents=n_agents, area_size=AREA_SIZE, max_step=MAX_STEP,
                   num_obs=N_OBS)
    algo = make_algo(
        algo=config.algo, env=env, node_dim=env.node_dim, edge_dim=env.edge_dim, state_dim=env.state_dim,
        action_dim=env.action_dim, n_agents=env.num_agents, gnn_layers=config.gnn_layers,
        batch_size=config.batch_size, buffer_size=config.buffer_size, horizon=config.horizon,
        lr_actor=config.lr_actor, lr_cbf=config.lr_cbf, alpha=config.alpha, eps=0.02, inner_epoch=8,
        loss_action_coef=config.loss_action_coef, loss_unsafe_coef=config.loss_unsafe_coef,
        loss_safe_coef=config.loss_safe_coef, loss_h_dot_coef=config.loss_h_dot_coef, max_grad_norm=2.0,
        seed=config.seed,
    )
    model_dir = MODEL_PATH / "models"
    step = max(int(m) for m in os.listdir(model_dir) if m.isdigit())
    algo.load(str(model_dir), step)
    return env, algo


# %% noise model and graph perturbation (copied from di_tighten_sweep.py) + the belief version
def gm_init(key: jax.Array, sigma: jax.Array, n_agents: int) -> jax.Array:
    return jr.normal(key, (n_agents, 4)) * sigma


def gm_step(a_noise: jax.Array, key: jax.Array, sigma: jax.Array, phi: jax.Array) -> jax.Array:
    return phi * a_noise + jnp.sqrt(1 - phi ** 2) * sigma * jr.normal(key, a_noise.shape)


def apply_nav_noise(graph: GraphsTuple, a_noise: jax.Array) -> GraphsTuple:
    """Raw arm: agent believes it is at x_i + n_i and sees neighbors at x_j + n_j (bit-identical to the old sweeps)."""
    n_agents = a_noise.shape[0]
    node_noise = jnp.zeros_like(graph.states).at[:n_agents].set(a_noise)
    is_nav_edge = (graph.node_type[graph.receivers] == 0) & jnp.isin(graph.node_type[graph.senders], jnp.array([0, 1]))
    delta = node_noise[graph.receivers] - node_noise[graph.senders]
    return graph._replace(edges=graph.edges + is_nav_edge[:, None] * delta, states=graph.states + node_noise)


def apply_belief(graph: GraphsTuple, a_own_err: jax.Array, aa_nbr_err: jax.Array) -> GraphsTuple:
    """Belief graph from estimation errors: a_own_err[i] = x_hat_i(own) - x_i, aa_nbr_err[i, j] = x_hat_{i->j} - x_j.
    Edges are x_recv - x_send, so agent-agent edges get a_own_err[i] - aa_nbr_err[i, j], goal edges a_own_err[i], and
    agent state rows a_own_err. With a_own_err = n_i and aa_nbr_err[i, j] = n_j this equals apply_nav_noise."""
    n = a_own_err.shape[0]
    recv, send = graph.receivers, graph.senders
    is_agent_recv = graph.node_type[recv] == 0
    is_aa = is_agent_recv & (graph.node_type[send] == 0)
    is_goal = is_agent_recv & (graph.node_type[send] == 1)
    r, s = jnp.minimum(recv, n - 1), jnp.minimum(send, n - 1)  # agents are nodes 0..n-1; others are masked below
    delta = jnp.where(is_aa[:, None], a_own_err[r] - aa_nbr_err[r, s],
                      jnp.where(is_goal[:, None], a_own_err[r], 0.0))
    node_err = jnp.zeros_like(graph.states).at[:n].set(a_own_err)
    return graph._replace(edges=graph.edges + delta, states=graph.states + node_err)


def collision_split(env, graph: GraphsTuple) -> tuple[jax.Array, jax.Array]:
    """(a_hit_agent, a_hit_obs): the two terms env.collision_mask ORs together (double_integrator.py:419-434)."""
    r = env._params["car_radius"]
    agent_pos = graph.type_states(type_idx=0, n_type=env.num_agents)[:, :2]
    dist = jnp.linalg.norm(agent_pos[:, None, :] - agent_pos[None, :, :], axis=-1)
    dist = dist + jnp.eye(dist.shape[1]) * (r * 2 + 1)
    return jnp.less(dist, r * 2).max(axis=1), inside_obstacles(agent_pos, graph.env_states.obstacle, r)


def cp_upper(k: int, n: int, conf: float = CONF) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial probability after k events in n trials."""
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


# %% batched rollout with the filter layer
def make_batch_rollout(env, algo, nbr_model: str, diag: bool = False):
    """Returns jit(vmap(episode)). episode(key_x0, key_noise, sigma(4,), phi, sigma_a, use_nbr, use_own, raw_exact,
    meas_every). diag=False: per-agent ever-flags (unsafe, hit_agent, hit_obs, finish). diag=True: per-step error
    statistics (see the end of body)."""
    n = env.num_agents
    d_nbr = 8 if nbr_model == "cv_gm" else 4
    dt, mass = env.dt, env._params["m"]
    v_max = float(env.state_lim()[1][2])
    offdiag = ~jnp.eye(n, dtype=bool)

    def flags(graph):
        a_hit_agent, a_hit_obs = collision_split(env, graph)
        return env.collision_mask(graph), a_hit_agent, a_hit_obs, env.finish_mask(graph)

    def episode(key_x0, key_noise, sigma, phi, sigma_a, use_nbr, use_own, raw_exact, meas_every):
        graph0 = env.reset(key_x0)
        key_init, key_steps = jr.split(key_noise)
        a_noise0 = gm_init(key_init, sigma, n)
        m_nbr = (bf.cv_gm_model(dt, sigma_a, phi, sigma) if nbr_model == "cv_gm" else bf.cv_model(dt, sigma_a, sigma))
        m_own = bf.cv_gm_model(dt, OWN_SIGMA_A, phi, sigma)
        a_x0 = graph0.type_states(type_idx=0, n_type=n)
        own0 = bf.own_track_init(a_x0 + a_noise0, sigma)
        carry0 = (graph0, a_noise0, own0, bf.nbr_tracks_empty(n, d_nbr), jnp.zeros((n, n, 4)), jnp.int32(0))

        def body(carry, key):
            graph, a_noise, (own_m, own_P), nbr, aa_held, t = carry
            a_x = graph.type_states(type_idx=0, n_type=n)
            a_z = a_x + a_noise  # this step's reports (= each agent's own nav fix)
            # ownship: initialized from z_0 before the scan, KF update from t = 1 on
            own_mu, own_Pu = bf.own_track_update(own_m, own_P, a_z, m_own)
            own_m = jnp.where(t == 0, own_m, own_mu)
            own_P = jnp.where(t == 0, own_P, own_Pu)
            # neighbors: receiver i hears j on report steps if its own report says j is within R_TRACK
            dist_z = jnp.linalg.norm(a_z[:, None, :2] - a_z[None, :, :2], axis=-1)
            aa_heard = (t % meas_every == 0) & (dist_z < R_TRACK)
            nbr = bf.nbr_tracks_update(nbr, a_z, aa_heard, m_nbr, sigma, T_DROP)
            aa_held = jnp.where(aa_heard[..., None], a_z[None], aa_held)  # last report heard (raw-hold arm)
            # estimation errors seen by the GNN
            a_own_err = jnp.where(use_own, own_m[:, :4] - a_x, a_noise)
            aa_trk = jnp.where(nbr.active[..., None], nbr.mean[..., :4], aa_held)
            aa_nbr_err = jnp.where(use_nbr, aa_trk, aa_held) - a_x[None]
            g_raw, g_bel = apply_nav_noise(graph, a_noise), apply_belief(graph, a_own_err, aa_nbr_err)
            g_in = jax.tree_util.tree_map(lambda a, b: jnp.where(raw_exact, a, b), g_raw, g_bel)
            action = algo.act(g_in)
            graph_new = env.step(graph, action)[0]  # env steps the TRUE graph
            # predict to t + 1: ownship with its own clipped action, neighbors with constant velocity
            own_m_p, own_P_p = bf.own_track_predict(own_m, own_P, env.clip_action(action), dt, mass, v_max, m_own)
            nbr_p = bf.nbr_tracks_predict(nbr, m_nbr)
            a_noise_new = gm_step(a_noise, key, sigma, phi)
            new_carry = (graph_new, a_noise_new, (own_m_p, own_P_p), nbr_p, aa_held, t + 1)
            if not diag:
                return new_carry, flags(graph_new)
            # diagnostics at time t (posterior = what the GNN saw); means over off-diagonal pairs / agents
            own_f, own_r = own_m[:, :2] - a_x[:, :2], a_noise[:, :2]
            nbr_f, nbr_r = aa_trk[..., :2] - a_x[None, :, :2], jnp.broadcast_to(a_noise[None, :, :2], (n, n, 2))

            def edge_mse(o, nb):  # squared error of the relative position the GNN sees
                e = o[:, None, :] - nb
                return ((e ** 2).sum(-1) * offdiag).sum() / offdiag.sum()

            edge = jnp.stack([edge_mse(own_r, nbr_r), edge_mse(own_r, nbr_f), edge_mse(own_f, nbr_r),
                              edge_mse(own_f, nbr_f)])  # raw, nbr, own, both
            P_x = nbr.cov[..., :4, :4]
            nees = bf.nees(nbr.mean[..., :4] - a_x[None], P_x)
            nees_mean = (nees * offdiag).sum() / offdiag.sum()
            own_nees = bf.nees(own_m[:, :4] - a_x, own_P[:, :4, :4]).mean()
            spread = jnp.abs(nbr.mean - nbr.mean[:1]).max()  # receivers' tracks of the same j must agree
            max_err = jnp.maximum(jnp.abs(own_m[:, :4] - a_x).max(),
                                  jnp.abs((aa_trk - a_x[None]) * offdiag[..., None]).max())
            accel = jnp.sqrt((env.clip_action(action) / mass) ** 2).mean()
            return new_carry, dict(edge=edge, own_mse=jnp.stack([(own_r ** 2).sum(-1).mean(),
                                                                  (own_f ** 2).sum(-1).mean()]),
                                   nees=nees_mean, own_nees=own_nees, spread=spread, max_err=max_err,
                                   accel_ms=((env.clip_action(action) / mass) ** 2).mean(), accel=accel)

        _, out = lax.scan(body, carry0, jr.split(key_steps, MAX_STEP))
        if diag:
            return out
        return tuple(Ta.any(axis=0) | a0 for Ta, a0 in zip(out, flags(graph0)))

    return jax.jit(jax.vmap(episode, in_axes=(0, 0) + (None,) * 7))


env, algo = load_env_algo(N_AGENTS)
PHI = float(np.exp(-env.dt / TAU))
test_keys = jr.split(jr.PRNGKey(SEED), 1_000)[:N_EPI]  # same reset keys as test.py / the earlier sweeps
key_x0s = jax.vmap(lambda k: jr.split(k, 2)[0])(test_keys)
noise_keys = jr.split(jr.PRNGKey(NOISE_SEED), 1_000)[:N_EPI]
rollout = {m: make_batch_rollout(env, algo, m) for m in ("cv_gm", "cv")}
rollout_diag = {m: make_batch_rollout(env, algo, m, diag=True) for m in ("cv_gm", "cv")}


def args(s, sigma_a, use_nbr=False, use_own=False, raw_exact=False, meas_every=1, phi=PHI):
    return (jnp.full(4, s, dtype=jnp.float32), jnp.float32(phi), jnp.float32(sigma_a), jnp.bool_(use_nbr),
            jnp.bool_(use_own), jnp.bool_(raw_exact), jnp.int32(meas_every))


# %% unit check 1: a cv_gm track on a straight line (truth inside the model) is consistent: NEES ~ 4
def synthetic_nees(n_mc=200, n_steps=600, s=0.05):
    sigma = jnp.full(4, s)
    model = bf.cv_gm_model(env.dt, 1e-3, PHI, sigma)
    v = jnp.array([0.3, -0.2])

    def one(key):
        k0, ks = jr.split(key)
        noise0 = jr.normal(k0, (4,)) * sigma
        x0 = jnp.concatenate([jnp.zeros(2), v])
        m, P = bf.init_mean_cov(x0 + noise0, sigma, 8)

        def step(c, k):
            x, nse, m, P = c
            m, P = bf.kf_predict(m, P, model.F, model.Q)
            x = jnp.concatenate([x[:2] + env.dt * x[2:], x[2:]])
            nse = PHI * nse + jnp.sqrt(1 - PHI ** 2) * sigma * jr.normal(k, (4,))
            m, P, _, _ = bf.kf_update(m, P, x + nse, model.H, model.R)
            return (x, nse, m, P), bf.nees(m[:4] - x, P[:4, :4])
        return lax.scan(step, (x0, noise0, m, P), jr.split(ks, n_steps))[1]

    return np.asarray(jax.jit(jax.vmap(one))(jr.split(jr.PRNGKey(7), n_mc)))


T_nees = synthetic_nees()
nees_late = T_nees[:, 300:].mean()
lo, hi = chi2.ppf([0.025, 0.975], 4 * 200) / 200
print(f"synthetic cv_gm NEES (4 dof): mean over steps 300-600 = {nees_late:.2f} "
      f"(ideal 4; 95% band for one step's MC average [{lo:.2f}, {hi:.2f}])")
assert 2.5 < nees_late < 5.5, "cv_gm track is badly inconsistent on its own model"

# %% unit check 2: raw arm reproduces the stored flags bit for bit (100 maps, GM sigma = 0.05)
old = np.load(NOISE_SWEEP_PATH)
old_sig = list(np.round(old["sigmas"], 6))
i_gm = list(old["noise_names"]).index("gm")
out = rollout["cv_gm"](key_x0s[:CHUNK], noise_keys[:CHUNK], *args(0.05, 1.0, raw_exact=True))
same = np.array_equal(np.asarray(out[0]), old["ever_unsafe_n8"][i_gm, old_sig.index(0.05), :CHUNK]) and \
    np.array_equal(np.asarray(out[3]), old["ever_finish_n8"][i_gm, old_sig.index(0.05), :CHUNK])
print(f"raw arm vs di_noise_sweep (sigma = 0.05, {CHUNK} maps): {'identical' if same else 'DIFFERENT'}")
assert same, "the raw path must reproduce the stored runs"

# %% diagnostics (CHUNK maps): sigma_a from noise-free rollouts; tiny-noise check; errors and NEES at sigma = 0.05
t0 = time.perf_counter()
dg0 = rollout_diag["cv_gm"](key_x0s[:CHUNK], noise_keys[:CHUNK], *args(1e-6, 1.0, True, True))
SIGMA_A = float(np.sqrt(np.asarray(dg0["accel_ms"]).mean()))  # RMS acceleration per axis, noise-free policy
print(f"empirical RMS acceleration per axis (sigma ~ 0): sigma_a = {SIGMA_A:.3f}  ({time.perf_counter() - t0:.0f} s)")
print(f"tiny noise (sigma = 1e-6): max |belief error| = {np.asarray(dg0['max_err']).max():.2e}; "
      f"max spread between receivers' tracks of the same agent = {np.asarray(dg0['spread']).max():.1e}")
assert np.asarray(dg0["max_err"]).max() < 1e-4, "filters must track the truth when the noise is ~0"
if np.isinf(R_TRACK):
    assert np.asarray(dg0["spread"]).max() == 0.0, "with R_TRACK = inf all receivers' tracks of j must be identical"

DIAG_SIGMA = 0.05
diag = {m: rollout_diag[m](key_x0s[:CHUNK], noise_keys[:CHUNK], *args(DIAG_SIGMA, SIGMA_A, True, True))
        for m in ("cv_gm", "cv")}
edge_rmse = np.sqrt(np.asarray(diag["cv_gm"]["edge"])[:, 50:].mean((0, 1)))  # skip the first 50 steps
own_rmse = np.sqrt(np.asarray(diag["cv_gm"]["own_mse"])[:, 50:].mean((0, 1)))
print(f"\ndiagnostics, GM sigma = {DIAG_SIGMA} (position errors; theory raw edge = sqrt(2)*sqrt(2)*sigma = "
      f"{2 * DIAG_SIGMA:.3f}, raw own = sqrt(2)*sigma = {np.sqrt(2) * DIAG_SIGMA:.3f})")
for name, v in zip(("raw", "nbr", "own", "both"), edge_rmse):
    print(f"  edge position RMSE, {name:>4}: {v:.4f}")
print(f"  ownship position RMSE: raw {own_rmse[0]:.4f}, filtered {own_rmse[1]:.4f}")
for m in ("cv_gm", "cv"):
    print(f"  neighbor NEES ({m}, 4 dof, ideal 4): {np.asarray(diag[m]['nees'])[:, 50:].mean():.2f}")
print(f"  ownship NEES (4 dof, ideal 4): {np.asarray(diag['cv_gm']['own_nees'])[:, 50:].mean():.2f}")

# %% sweep
if RUN_SWEEP:
    assert N_EPI <= 1000 and N_EPI % CHUNK == 0
    shape = (len(RUNS), N_EPI, N_AGENTS)
    ever_unsafe, ever_hit_agent, ever_hit_obs, ever_finish = [np.zeros(shape, dtype=bool) for _ in range(4)]
    t_start = time.perf_counter()
    for i_r, (label, s, use_nbr, use_own, nbr_model, qs, me) in enumerate(RUNS):
        t0 = time.perf_counter()
        a = args(s, SIGMA_A * np.sqrt(qs), use_nbr, use_own, False, me)
        for c in range(0, N_EPI, CHUNK):
            out = rollout[nbr_model](key_x0s[c:c + CHUNK], noise_keys[c:c + CHUNK], *a)
            for arr, o in zip((ever_unsafe, ever_hit_agent, ever_hit_obs, ever_finish), out):
                arr[i_r, c:c + CHUNK] = np.asarray(o)
        elapsed = time.perf_counter() - t_start
        print(f"[{i_r + 1}/{len(RUNS)}] {label:>13} sigma={s:<6g} "
              f"epi-collision={100 * ever_unsafe[i_r].any(-1).mean():6.2f}% "
              f"(agent-agent {100 * ever_hit_agent[i_r].any(-1).mean():.1f}%, "
              f"obstacle {100 * ever_hit_obs[i_r].any(-1).mean():.1f}%) "
              f"reach={100 * ever_finish[i_r].mean():6.2f}%  ({time.perf_counter() - t0:.0f} s; "
              f"ETA {elapsed / (i_r + 1) * (len(RUNS) - i_r - 1) / 60:.0f} min)", flush=True)
    print(f"sweep done in {time.perf_counter() - t_start:.0f} s")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(RESULTS_PATH, run_labels=np.array([r[0] for r in RUNS]), run_sigmas=np.array([r[1] for r in RUNS]),
             run_use_nbr=np.array([r[2] for r in RUNS]), run_use_own=np.array([r[3] for r in RUNS]),
             run_nbr_model=np.array([r[4] for r in RUNS]), run_q_scale=np.array([r[5] for r in RUNS]),
             run_meas_every=np.array([r[6] for r in RUNS]), sigma_a=SIGMA_A, own_sigma_a=OWN_SIGMA_A, tau=TAU,
             r_track=R_TRACK, t_drop=T_DROP, n_agents=N_AGENTS, max_step=MAX_STEP, n_epi=N_EPI, seed=SEED,
             noise_seed=NOISE_SEED, ever_unsafe=ever_unsafe, ever_hit_agent=ever_hit_agent,
             ever_hit_obs=ever_hit_obs, ever_finish=ever_finish, diag_edge_rmse=edge_rmse, diag_own_rmse=own_rmse)
    print("saved", RESULTS_PATH)

# %% summary: filter runs + the raw arm from the stored sweeps (same maps and noise, so paired)
res = np.load(RESULTS_PATH)
n_epi = int(res["n_epi"])
labels, sigmas = list(res["run_labels"]), res["run_sigmas"]
U, A, O, Fn = res["ever_unsafe"], res["ever_hit_agent"], res["ever_hit_obs"], res["ever_finish"]
assert np.array_equal(A | O, U), "split must OR back to collision_mask"

tight = np.load(TIGHTEN_PATH)
t_sig, t_kap = list(np.round(tight["cell_sigmas"], 6)), list(tight["kappas"])


def raw_flags(s):
    """(unsafe, hit_agent, hit_obs, finish) for the raw arm at sigma s; the split exists only in the tighten sweep."""
    if round(s, 6) in t_sig:
        i = t_sig.index(round(s, 6))
        k0 = t_kap.index(0.0)
        return tuple(tight[k][i, k0, :n_epi] for k in ("ever_unsafe", "ever_hit_agent", "ever_hit_obs", "ever_finish"))
    j = old_sig.index(round(s, 6))
    u = old["ever_unsafe_n8"][i_gm, j, :n_epi]
    return u, None, None, old["ever_finish_n8"][i_gm, j, :n_epi]


rows = []


def add_row(label, s, u, a, o, fin):
    k = int(u.any(-1).sum())
    rows.append([label, s, k / n_epi, cp_upper(k, n_epi), a.any(-1).mean() if a is not None else np.nan,
                 o.any(-1).mean() if o is not None else np.nan, 1 - u.mean(), fin.mean()])


for s in SIGMAS:
    add_row("raw", s, *raw_flags(s))
for i_r, (lab, s) in enumerate(zip(labels, sigmas)):
    add_row(lab, float(s), U[i_r], A[i_r], O[i_r], Fn[i_r])
rows.sort(key=lambda r: (r[0] not in ("raw", "nbr", "own", "both"), r[1], ["raw", "nbr", "own", "both"].index(r[0])
                         if r[0] in ("raw", "nbr", "own", "both") else 0))
print(f"\n{n_epi} maps, GM tau = {TAU:g} s, n = {N_AGENTS}; sigma_a = {float(res['sigma_a']):.3f}")
print(f"{'arm':>13} {'sigma':>6} | {'epi-coll':>8} {'CP-up':>6} | {'ag-ag':>6} {'obst':>6} | {'safe':>7} {'reach':>6}")
for lab, s, rate, ub, ra, ro, safe, reach in rows:
    print(f"{lab:>13} {s:6.3f} | {100 * rate:7.1f}% {100 * ub:5.1f}% | {100 * ra:5.1f}% {100 * ro:5.1f}% | "
          f"{100 * safe:6.2f}% {100 * reach:5.1f}%")
SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SUMMARY_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "sigma", "epi_coll_rate", "epi_coll_cp_upper", "epi_rate_agent_agent", "epi_rate_obstacle",
                "agent_safety", "agent_reach"])
    w.writerows(rows)
print("saved", SUMMARY_PATH)

# %% plots: main comparison (rate vs sigma per arm) and diagnostics
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated default categorical slots 1-3
INK, MUTED, GRID = "#222222", "#6b6b66", "#e4e3df"
ARMS = {"raw": (MUTED, "o", "raw reports"), "nbr": (BLUE, "s", "neighbor filter"),
        "own": (ORANGE, "^", "ownship filter"), "both": (AQUA, "D", "both filters")}
FLOOR = 0.1


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
cols = {"epi": 2, "aa": 4, "obs": 5, "reach": 7}
for arm, (col, mk, lab) in ARMS.items():
    rr = [r for r in rows if r[0] == arm]
    x = [r[1] for r in rr]
    axs[0].plot(x, [max(100 * r[2], FLOOR) for r in rr], marker=mk, color=col, lw=2, ms=5, label=lab)
    for ax, c in ((axs[1], "aa"), (axs[2], "obs")):
        ok = [r for r in rr if not np.isnan(r[cols[c]])]
        ax.plot([r[1] for r in ok], [max(100 * r[cols[c]], FLOOR) for r in ok], marker=mk, color=col, lw=2, ms=5,
                label=lab)
    axs[3].plot(x, [100 * r[7] for r in rr], marker=mk, color=col, lw=2, ms=5, label=lab)
k2 = t_kap.index(2.0)
xt = [float(s) for s in tight["cell_sigmas"]]
yt = [max(100 * tight["ever_unsafe"][i, k2, :n_epi].any(-1).mean(), FLOOR) for i in range(len(xt))]
axs[0].plot(xt, yt, ls="--", marker="x", color=INK, lw=1.2, ms=6, label="raw + LOS margin κ = 2")
axs[0].axhline(5, color=INK, lw=1, ls=":")
for ax, t in zip(axs, ("episodes with any collision", "episodes with an agent-agent collision",
                       "episodes with an obstacle collision", "reach (per agent)")):
    ax.set(title=t, xlabel="GM noise σ per axis")
    style(ax)
for ax in axs[:3]:
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_ylim(FLOOR * 0.9, 100)
    ax.minorticks_off()
    ax.set_ylabel("%")
axs[3].set_ylabel("%")
axs[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle(f"Pretrained GCBF+ with a Kalman filter layer (mean belief into the GNN), n = {N_AGENTS}, "
             f"{n_epi} maps per point, GM τ = {TAU:g} s", fontsize=10)
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE_PATH, dpi=130)
print("saved", FIGURE_PATH)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
tt = np.arange(MAX_STEP) * env.dt
for i, (arm, (col, mk, lab)) in enumerate(ARMS.items()):
    ax1.plot(tt, np.sqrt(np.asarray(diag["cv_gm"]["edge"])[:, :, i].mean(0)), color=col, lw=1.5, label=lab)
ax1.set(title=f"edge position RMSE, GM σ = {DIAG_SIGMA}", xlabel="time [s]", ylabel="RMSE", ylim=(0, None))
band = chi2.ppf([0.025, 0.975], 4 * CHUNK) / CHUNK
for m, col in (("cv_gm", AQUA), ("cv", ORANGE)):
    ax2.semilogy(tt, np.asarray(diag[m]["nees"]).mean(0), color=col, lw=1.5, label=f"neighbor track, {m}")
ax2.semilogy(tt, np.asarray(diag["cv_gm"]["own_nees"]).mean(0), color=BLUE, lw=1.5, label="ownship track")
ax2.axhspan(*band, color=MUTED, alpha=0.2, lw=0, label="95% band (consistent)")
ax2.set(title="NEES (4 dof, ideal 4)", xlabel="time [s]")
for ax in (ax1, ax2):
    style(ax)
    ax.legend(frameon=False, fontsize=8)
fig.savefig(DIAG_FIGURE_PATH, dpi=130)
print("saved", DIAG_FIGURE_PATH)
plt.show()

# %%
