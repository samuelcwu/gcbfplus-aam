# Scratch experiment: Gaussian noise on the edge features the GCBF+ policy observes (DoubleIntegrator).
# The env is always stepped with the TRUE graph; only the policy input is corrupted.
# Run cell-by-cell in the VS Code Interactive Window, or: python scratch/di_edge_noise.py

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

MODEL_PATH =REPO_ROOT / "pretrained/DoubleIntegrator/gcbf+"
N_AGENTS = 16
AREA_SIZE = 4
N_OBS = 0
MAX_STEP = 256
N_EPI = 1
SEED = 1234
NOISE_SEED = 0
# (sigma_pos, sigma_vel) pairs; edge layout is [x_rel, y_rel, vx_rel, vy_rel]
SIGMAS = [(0.0, 0.0), (0.01, 0.01), (0.05, 0.05)]
#SIGMAS = [(0.0, 0.0), (1.00, 1.00)]
AGENT_EDGES_ONLY = True  # True: only perturb agent-agent edges (not goal / lidar edges)

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


# %% noise
def add_edge_noise(
    graph: GraphsTuple, key: jax.Array, sigma_pos: float, sigma_vel: float, agent_edges_only: bool = False
) -> GraphsTuple:
    """Return a copy of graph with Gaussian noise on valid edges. states / env_states stay true."""
    recv_type = graph.node_type[graph.receivers]
    send_type = graph.node_type[graph.senders]
    e_valid = (recv_type != -1) & (send_type != -1)
    if agent_edges_only:
        e_valid = e_valid & (send_type == 0)

    sigma = jnp.array([sigma_pos, sigma_pos, sigma_vel, sigma_vel])
    noise = jr.normal(key, graph.edges.shape) * sigma
    return graph._replace(edges=graph.edges + e_valid[:, None] * noise)


# sanity check on one graph
_g = env.reset(jr.PRNGKey(0))
_g_noisy = add_edge_noise(_g, jr.PRNGKey(1), 0.05, 0.1)
_pad = (_g.node_type[_g.receivers] == -1) | (_g.node_type[_g.senders] == -1)
assert jnp.array_equal(_g_noisy.states, _g.states)
assert jnp.array_equal(_g_noisy.edges[_pad], _g.edges[_pad])
print(f"edges: {_g.edges.shape[0]} total, {int((~_pad).sum())} valid, "
      f"max |noise| = {float(jnp.abs(_g_noisy.edges - _g.edges).max()):.3f}")


# %% rollout: policy sees the noisy graph, env steps the true graph
def make_noisy_rollout(sigma_pos: float, sigma_vel: float, rollout_length: int = MAX_STEP):
    def body(graph, key):
        obs = add_edge_noise(graph, key, sigma_pos, sigma_vel, AGENT_EDGES_ONLY)
        action = algo.act(obs)
        graph_new, reward, cost, done, info = env.step(graph, action, get_eval_info=True)
        return graph_new, (graph_new, action, reward, cost, done, info, obs.edges)

    def fn(key_x0, key_noise) -> tuple[RolloutResult, jax.Array]:
        graph0 = env.reset(key_x0)
        T_keys = jr.split(key_noise, rollout_length)
        _, (T_graph, T_action, T_reward, T_cost, T_done, T_info, T_obs_edges) = lax.scan(body, graph0, T_keys)
        Tp1_graph = tree_concat_at_front(graph0, T_graph, axis=0)
        # T_obs_edges[t] = noisy edge features the policy saw for Tp1_graph[t]
        return RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info), T_obs_edges

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
        rollout, T_obs_edges = rollout_fn(key_x0, noise_keys[i_epi])
        Ta_is_unsafe = is_unsafe_fn(rollout.Tp1_graph)
        Ta_is_finish = is_finish_fn(rollout.Tp1_graph)
        a_unsafe, a_finish = Ta_is_unsafe.max(axis=0), Ta_is_finish.max(axis=0)
        rates.append([1 - a_unsafe.mean(), a_finish.mean(), ((1 - a_unsafe) * a_finish).mean()])
        costs.append(rollout.T_cost.sum())
        rollouts.append((rollout, Ta_is_unsafe, T_obs_edges))
    results[(sigma_pos, sigma_vel)] = (np.array(rates), np.array(costs), rollouts)

print(f"\n{'sig_pos':>8} {'sig_vel':>8} {'safe%':>8} {'finish%':>8} {'success%':>9} {'cost':>8}")
for (sigma_pos, sigma_vel), (rates, costs, _) in results.items():
    safe, finish, success = rates.mean(axis=0) * 100
    print(f"{sigma_pos:8.3f} {sigma_vel:8.3f} {safe:8.2f} {finish:8.2f} {success:9.2f} {costs.mean():8.3f}")

# %% videos: one per sigma for each episode in RENDER_EPIS (same reset key across sigmas -> same start)
RENDER = True
#RENDER_EPIS = range(N_EPI)
RENDER_EPIS = [0]
GHOST_TYPES = (0,)  # sender types to draw perceived positions for: 0 = agent, 1 = goal, 2 = lidar hit
if RENDER:
    videos_dir = REPO_ROOT / "scratch/videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    for (sigma_pos, sigma_vel), (rates, _, rollouts) in results.items():
        for i_epi in RENDER_EPIS:
            rollout, Ta_is_unsafe, T_obs_edges = rollouts[i_epi]
            safe, finish, _ = rates[i_epi] * 100
            video_name = f"epi{i_epi:02}_pos{sigma_pos:.3f}_vel{sigma_vel:.3f}_safe{safe:.0f}_finish{finish:.0f}.mp4"
            render_video_noisy(env, rollout, T_obs_edges, videos_dir / video_name, Ta_is_unsafe, GHOST_TYPES)
            print("saved", video_name)
