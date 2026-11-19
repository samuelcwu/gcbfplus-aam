# Scratch experiment: ADS-B-like per-agent navigation error (DoubleIntegrator).
# Each agent has one first-order Gauss-Markov navigation error n_i. It broadcasts x_i + n_i, and every neighbor
# receives the SAME noisy report. With OWNSHIP_NOISE the agent also believes it is at x_i + n_i (neighbor edges,
# goal edge, u_ref), so relative estimates carry n_i - n_j. The env is always stepped with the TRUE graph.
# Run cell-by-cell in the VS Code Interactive Window, or: python scratch/di_gm_noise.py

# %% setup / config
import os
import pathlib
import sys

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, jax_vmap, tree_concat_at_front

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "scratch"))  # for render_noisy.py next to this file
from render_noisy import render_video_noisy

MODEL_PATH = REPO_ROOT / "pretrained/DoubleIntegrator/gcbf+"
N_AGENTS = 16
AREA_SIZE = 4
N_OBS = 0
MAX_STEP = 256
N_EPI = 10
SEED = 1234
NOISE_SEED = 0
# (sigma_pos, sigma_vel) pairs: stationary std of each agent's broadcast error on [x, y] and [vx, vy]
SIGMAS = [(0.0, 0.0), (0.01, 0.01), (0.05, 0.05)]
OWNSHIP_NOISE = True  # True: agent uses x_i + n_i for itself too. False: broadcast-only (receiver uses true state)
TAU = 1.0  # Gauss-Markov correlation time [s]; TAU -> 0 gives white per-agent noise. Episode = MAX_STEP * 0.03 s

# %% env + algo (mirrors test.py)
env = make_env(env_id="DoubleIntegrator", num_agents=N_AGENTS, area_size=AREA_SIZE, max_step=MAX_STEP, num_obs=N_OBS)

with open(MODEL_PATH / "config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)

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
print("loaded step", step)

PHI = float(np.exp(-env.dt / TAU))  # per-step correlation
print(f"TAU = {TAU} s, dt = {env.dt} s, phi = {PHI:.4f}")


# %% Gauss-Markov broadcast error
def gm_init(key: jax.Array, sigma: jax.Array) -> jax.Array:
    """Draw a_noise from the stationary distribution N(0, sigma^2). Shape (n_agents, 4)."""
    return jr.normal(key, (N_AGENTS, 4)) * sigma


def gm_step(a_noise: jax.Array, key: jax.Array, sigma: jax.Array, phi: float) -> jax.Array:
    """n_{t+1} = phi * n_t + sqrt(1 - phi^2) * sigma * w_t; keeps the variance at sigma^2."""
    return phi * a_noise + jnp.sqrt(1 - phi ** 2) * sigma * jr.normal(key, a_noise.shape)


def apply_broadcast_noise(graph: GraphsTuple, a_noise: jax.Array) -> GraphsTuple:
    """Agent-agent edges become state[recv] - (state[send] + a_noise[send]). Goal / lidar edges, states stay true."""
    is_agent_edge = (graph.node_type[graph.receivers] == 0) & (graph.node_type[graph.senders] == 0)
    send_noise = a_noise[jnp.minimum(graph.senders, N_AGENTS - 1)]  # agents are nodes 0..n-1; pad ids are masked
    return graph._replace(edges=graph.edges - is_agent_edge[:, None] * send_noise)


def apply_nav_noise(graph: GraphsTuple, a_noise: jax.Array) -> GraphsTuple:
    """Agent believes it is at x_i + n_i. Edges are state[recv] - state[send], so each agent-agent / agent-goal edge
    gets node_noise[recv] - node_noise[send] (goals have no noise): (x_i + n_i) - (x_j + n_j) and (x_i + n_i) - g.
    Noise is added to the env's own edges rather than rebuilding them with env.add_edge_feats, because the env's goal
    edges are not actually clipped (edge_blocks slices goals, not columns, double_integrator.py:239-244) while
    add_edge_feats clips them. LiDAR is an onboard relative sensor, so its edges stay true.
    Agent rows of states also get the noise, so u_ref uses the noisy own state."""
    node_noise = jnp.zeros_like(graph.states).at[:N_AGENTS].set(a_noise)  # agents are nodes 0..n-1; pad row is 0
    is_nav_edge = (graph.node_type[graph.receivers] == 0) & jnp.isin(graph.node_type[graph.senders], jnp.array([0, 1]))
    delta = node_noise[graph.receivers] - node_noise[graph.senders]
    return graph._replace(edges=graph.edges + is_nav_edge[:, None] * delta, states=graph.states + node_noise)


apply_noise = apply_nav_noise if OWNSHIP_NOISE else apply_broadcast_noise
NOISE_TAG = "own" if OWNSHIP_NOISE else "bcast"

# sanity checks on one graph
_g = env.reset(jr.PRNGKey(0))
_a_noise = gm_init(jr.PRNGKey(1), jnp.array([0.05, 0.05, 0.1, 0.1]))
_is_aa = (_g.node_type[_g.receivers] == 0) & (_g.node_type[_g.senders] == 0)
_s, _r = _g.senders[_is_aa], _g.receivers[_is_aa]
# broadcast-only: every receiver sees sender j at the same reported position pos_j + noise_j
_g_b = apply_broadcast_noise(_g, _a_noise)
assert jnp.array_equal(_g_b.states, _g.states)
assert jnp.array_equal(_g_b.edges[~_is_aa], _g.edges[~_is_aa])
assert jnp.allclose(_g.states[_r, :2] - _g_b.edges[_is_aa, :2], _g.states[_s, :2] + _a_noise[_s, :2], atol=1e-5)
# ownship: agent-agent edges = (x_i + n_i) - (x_j + n_j); goal edges shifted by n_i; goal rows true; u_ref changes
_g_o = apply_nav_noise(_g, _a_noise)
_expected = (_g.states[_r] + _a_noise[_r]) - (_g.states[_s] + _a_noise[_s])
assert jnp.allclose(_g_o.edges[_is_aa], _expected, atol=1e-5)
_is_ag = (_g.node_type[_g.receivers] == 0) & (_g.node_type[_g.senders] == 1)
assert jnp.allclose(_g_o.edges[_is_ag], _g.edges[_is_ag] + _a_noise[_g.receivers[_is_ag]], atol=1e-5)
assert jnp.array_equal(_g_o.states[N_AGENTS:], _g.states[N_AGENTS:])
assert not jnp.allclose(env.u_ref(_g_o), env.u_ref(_g))
# zero noise reproduces the env's graph exactly
_g_0 = apply_nav_noise(_g, jnp.zeros_like(_a_noise))
assert jnp.array_equal(_g_0.edges, _g.edges) and jnp.array_equal(_g_0.states, _g.states)
print(f"sanity ok: {int(_is_aa.sum())} agent-agent edges at reset, noise model = {NOISE_TAG}")


# %% rollout: policy sees the noisy graph, env steps the true graph
def make_noisy_rollout(sigma_pos: float, sigma_vel: float, rollout_length: int = MAX_STEP):
    sigma = jnp.array([sigma_pos, sigma_pos, sigma_vel, sigma_vel])

    def body(carry, key):
        graph, a_noise = carry
        obs = apply_noise(graph, a_noise)
        action = algo.act(obs)
        graph_new, reward, cost, done, info = env.step(graph, action, get_eval_info=True)
        a_noise_new = gm_step(a_noise, key, sigma, PHI)
        return (graph_new, a_noise_new), (graph_new, action, reward, cost, done, info, obs.edges, a_noise)

    def fn(key_x0, key_noise):
        graph0 = env.reset(key_x0)
        key_init, key_steps = jr.split(key_noise)
        a_noise0 = gm_init(key_init, sigma)
        T_keys = jr.split(key_steps, rollout_length)
        _, (T_graph, T_action, T_reward, T_cost, T_done, T_info, T_obs_edges, Ta_noise) = lax.scan(
            body, (graph0, a_noise0), T_keys
        )
        Tp1_graph = tree_concat_at_front(graph0, T_graph, axis=0)
        # T_obs_edges[t] / Ta_noise[t] = noisy edges / broadcast error the policy saw for Tp1_graph[t]
        return RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info), T_obs_edges, Ta_noise

    return jax_jit_np(fn)


is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))

# %% sweep
test_keys = jr.split(jr.PRNGKey(SEED), 1_000)[:N_EPI]  # same reset keys as test.py
noise_keys = jr.split(jr.PRNGKey(NOISE_SEED), N_EPI)

results = {}
for sigma_pos, sigma_vel in SIGMAS:
    rollout_fn = make_noisy_rollout(sigma_pos, sigma_vel)
    rates, costs, rollouts = [], [], []
    for i_epi in range(N_EPI):
        key_x0, _ = jr.split(test_keys[i_epi], 2)
        rollout, T_obs_edges, Ta_noise = rollout_fn(key_x0, noise_keys[i_epi])
        Ta_is_unsafe = is_unsafe_fn(rollout.Tp1_graph)
        Ta_is_finish = is_finish_fn(rollout.Tp1_graph)
        a_unsafe, a_finish = Ta_is_unsafe.max(axis=0), Ta_is_finish.max(axis=0)
        rates.append([1 - a_unsafe.mean(), a_finish.mean(), ((1 - a_unsafe) * a_finish).mean()])
        costs.append(rollout.T_cost.sum())
        rollouts.append((rollout, Ta_is_unsafe, T_obs_edges, Ta_noise))
    results[(sigma_pos, sigma_vel)] = (np.array(rates), np.array(costs), rollouts)

print(f"\nTAU = {TAU} s, noise model = {NOISE_TAG}")
print(f"{'sig_pos':>8} {'sig_vel':>8} {'safe%':>8} {'finish%':>8} {'success%':>9} {'cost':>8}")
for (sigma_pos, sigma_vel), (rates, costs, _) in results.items():
    safe, finish, success = rates.mean(axis=0) * 100
    print(f"{sigma_pos:8.3f} {sigma_vel:8.3f} {safe:8.2f} {finish:8.2f} {success:9.2f} {costs.mean():8.3f}")

# %% noise statistics check: std ~ sigma, lag-k autocorrelation ~ phi^k
_sig = next(s for s in SIGMAS if s[0] > 0)
_n = np.concatenate([r[3] for r in results[_sig][2]], axis=1)  # (T, N_EPI * n_agents, 4)
print(f"\nnoise stats at sigma = {_sig}:  std per dim = {_n.std(axis=(0, 1)).round(4)}")
for k in [1, 10, 33]:
    ac = (_n[k:] * _n[:-k]).mean() / (_n ** 2).mean()
    print(f"  lag {k:3d}: autocorr = {ac:.3f}, phi^k = {PHI ** k:.3f}")
_rel = _n[:, :, None, :] - _n[:, None, :, :]  # pairwise n_i - n_j (pairs across episodes are also independent)
_iu = np.triu_indices(_n.shape[1], k=1)
print(f"  relative error n_i - n_j: std per dim = {_rel[:, _iu[0], _iu[1]].std(axis=(0, 1)).round(4)}"
      f" (sqrt(2) * sigma = {np.sqrt(2) * np.array(_sig)})")

# %% videos: one per sigma for each episode in RENDER_EPIS (same reset key across sigmas -> same start)
RENDER = True
RENDER_EPIS = [0]
GHOST_TYPES = (0,)  # sender types to draw perceived positions for: 0 = agent, 1 = goal, 2 = lidar hit
if RENDER:
    videos_dir = REPO_ROOT / "scratch/videos/gm"
    videos_dir.mkdir(parents=True, exist_ok=True)
    for (sigma_pos, sigma_vel), (rates, _, rollouts) in results.items():
        for i_epi in RENDER_EPIS:
            rollout, Ta_is_unsafe, T_obs_edges, Ta_noise = rollouts[i_epi]
            Ta_own_offset = Ta_noise[..., :2] if OWNSHIP_NOISE else None
            safe, finish, _ = rates[i_epi] * 100
            video_name = (f"{NOISE_TAG}_tau{TAU:g}_epi{i_epi:02}_pos{sigma_pos:.3f}_vel{sigma_vel:.3f}"
                          f"_safe{safe:.0f}_finish{finish:.0f}.mp4")
            render_video_noisy(env, rollout, T_obs_edges, videos_dir / video_name, Ta_is_unsafe, GHOST_TYPES,
                               Ta_own_offset=Ta_own_offset)
            print("saved", video_name)
