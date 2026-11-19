# Visualize individual episodes of the 1000-map noise sweep (scratch/di_noise_sweep.py). The sweep only stored per-agent
# "ever collided" / "ever reached goal" flags, so episodes are re-simulated here from their stored keys (map key from
# PRNGKey(SEED), noise key from PRNGKey(NOISE_SEED), both pools of 1000) with the full trajectory kept.
# Outputs: a trajectory-grid PNG of many episodes at one (agents, noise, sigma) point plus the same maps at
# COMPARE_SIGMA, and videos (render_noisy.py) of a few episodes per category (collision / no_reach / success).
# Run cell-by-cell in the VS Code Interactive Window, or: python scratch/di_noise_sweep_videos.py

# %% setup / config
import os
import pathlib
import sys

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import yaml

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.env.plot import get_obs_collection
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, jax_vmap, tree_concat_at_front

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "scratch"))  # for render_noisy.py next to this file
from render_noisy import OBS_COLOR, render_video_noisy

MODEL_PATH = REPO_ROOT / "pretrained/DoubleIntegrator/gcbf+"
RESULTS_PATH = REPO_ROOT / "scratch/results/di_noise_sweep_epi1000.npz"

POINT = dict(n_agents=8, noise="gm", sigma=0.05)  # the sweep point to inspect
COMPARE_SIGMA = 0.0  # same maps re-simulated at this sigma, for comparison (None to skip)
N_PER_CATEGORY = dict(collision=3, no_reach=3, success=1)  # episodes per category to render as videos
GRID_PER_CATEGORY = dict(collision=10, no_reach=10, success=4)  # episodes per category in the PNG grid
GRID_COLS = 6
MANUAL_EPISODES = []  # episode ids; if non-empty, replaces the automatic pick (grid and videos) at POINT
RENDER_VIDEOS = True
GHOST_TYPES = (0,)  # sender types to draw perceived positions for: 0 = agent, 1 = goal, 2 = lidar hit
DRAW_SCALE = 1.0  # display-only circle size multiplier for the videos
PLOT_CBF = True  # add a strip chart of the learned barrier h_i(t) (on the graph the policy saw, and on the true graph)
GRID_PLOTS = True  # trajectory-grid PNGs (set False to only render videos)

# must match the sweep (scratch/di_noise_sweep.py config cell)
AREA_SIZE = 4
N_OBS = 4
MAX_STEP = 1024
SEED = 1234
NOISE_SEED = 0
NOISE_TAUS = {"white": 0.0, "gm": 1.0}
OWNSHIP_NOISE = True

FIG_DIR = REPO_ROOT / "scratch/figures"
VIDEO_DIR = REPO_ROOT / ("scratch/videos/sweep_samples_cbf" if PLOT_CBF else "scratch/videos/sweep_samples")

with open(MODEL_PATH / "config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)

# %% pick episodes from the sweep results
res = np.load(RESULTS_PATH)
n_agents, noise_name, sigma = POINT["n_agents"], POINT["noise"], POINT["sigma"]
i_type = list(res["noise_names"]).index(noise_name)
i_sig = int(np.argmin(np.abs(res["sigmas"] - sigma)))
assert np.isclose(res["sigmas"][i_sig], sigma), f"sigma {sigma} not in sweep grid {res['sigmas']}"
ea_unsafe = res[f"ever_unsafe_n{n_agents}"][i_type, i_sig]  # (episode, agent)
ea_finish = res[f"ever_finish_n{n_agents}"][i_type, i_sig]


def categorize(a_unsafe: np.ndarray, a_finish: np.ndarray) -> str:
    """collision: any agent collided; no_reach: none collided but some agent never reached its goal; else success."""
    if a_unsafe.any():
        return "collision"
    return "no_reach" if not a_finish.all() else "success"


sweep_cats = np.array([categorize(u, f) for u, f in zip(ea_unsafe, ea_finish)])
print(f"{n_agents} agents, {noise_name}, sigma = {res['sigmas'][i_sig]:g}: episodes per category (of {len(sweep_cats)})")
for cat in N_PER_CATEGORY:
    print(f"  {cat:>9}: {np.sum(sweep_cats == cat)}")

if MANUAL_EPISODES:
    grid_epis = video_epis = list(MANUAL_EPISODES)
else:
    by_cat = {cat: np.flatnonzero(sweep_cats == cat) for cat in GRID_PER_CATEGORY}
    grid_epis = [int(e) for cat, k in GRID_PER_CATEGORY.items() for e in by_cat[cat][:k]]
    video_epis = [int(e) for cat, k in N_PER_CATEGORY.items() for e in by_cat[cat][:k]]
print("grid episodes: ", grid_epis)
print("video episodes:", video_epis)


# %% model, env, noise (copied from scratch/di_noise_sweep.py, which runs the whole sweep on import)
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
    LiDAR edges stay true. Agent rows of states also get the noise, so u_ref uses the noisy own state."""
    n_agents = a_noise.shape[0]
    node_noise = jnp.zeros_like(graph.states).at[:n_agents].set(a_noise)
    is_nav_edge = (graph.node_type[graph.receivers] == 0) & jnp.isin(graph.node_type[graph.senders], jnp.array([0, 1]))
    delta = node_noise[graph.receivers] - node_noise[graph.senders]
    return graph._replace(edges=graph.edges + is_nav_edge[:, None] * delta, states=graph.states + node_noise)


apply_noise = apply_nav_noise if OWNSHIP_NOISE else apply_broadcast_noise

env, algo = load_env_algo(n_agents)
tau = NOISE_TAUS[noise_name]
phi = float(np.exp(-env.dt / tau)) if tau > 0 else 0.0


# %% full-trajectory rollout (adapted from scratch/di_gm_noise.py; sigma and phi traced -> one compile for all sigmas)
def episode(key_x0, key_noise, sigma, phi):
    graph0 = env.reset(key_x0)
    key_init, key_steps = jr.split(key_noise)
    a_noise0 = gm_init(key_init, sigma, n_agents)

    def body(carry, key):
        graph, a_noise = carry
        obs = apply_noise(graph, a_noise)
        action = algo.act(obs)
        graph_new, reward, cost, done, info = env.step(graph, action, get_eval_info=True)
        a_noise_new = gm_step(a_noise, key, sigma, phi)
        # barrier h_i on the graph the policy saw and on the true graph (diagnostic only; h is not used by algo.act)
        a_h = (algo.get_cbf(obs)[:, 0], algo.get_cbf(graph)[:, 0]) if PLOT_CBF else (a_noise[:, 0], a_noise[:, 0])
        return (graph_new, a_noise_new), (graph_new, action, reward, cost, done, info, obs.edges, a_noise, a_h)

    _, (T_graph, T_action, T_reward, T_cost, T_done, T_info, T_obs_edges, Ta_noise, Ta_h) = lax.scan(
        body, (graph0, a_noise0), jr.split(key_steps, MAX_STEP)
    )
    Tp1_graph = tree_concat_at_front(graph0, T_graph, axis=0)
    # T_obs_edges[t] / Ta_noise[t] / Ta_h[.][t] = noisy edges / noise / barrier values for Tp1_graph[t]
    return RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info), T_obs_edges, Ta_noise, Ta_h


rollout_fn = jax_jit_np(episode)
is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))

# same keys as the sweep
test_keys = jr.split(jr.PRNGKey(SEED), 1_000)
key_x0s = jax.vmap(lambda k: jr.split(k, 2)[0])(test_keys)
noise_keys = jr.split(jr.PRNGKey(NOISE_SEED), 1_000)


def simulate(i_epi: int, s: float, keep_rollout: bool):
    """Re-simulate one episode. Returns a dict with the agent paths, per-step masks, obstacles, and optionally the
    full rollout (needed for the video)."""
    sig = jnp.array([s, s, s, s])
    rollout, T_obs_edges, Ta_noise, (Ta_h_seen, Ta_h_true) = rollout_fn(key_x0s[i_epi], noise_keys[i_epi], sig,
                                                                         jnp.float32(phi))
    Ta_unsafe = is_unsafe_fn(rollout.Tp1_graph)  # (T + 1, agent), includes graph0 as the sweep does
    Ta_finish = is_finish_fn(rollout.Tp1_graph)
    out = dict(
        Ta_pos=np.asarray(rollout.Tp1_graph.env_states.agent[..., :2]),
        a_goal=np.asarray(rollout.Tp1_graph.env_states.goal[0, :, :2]),
        obstacle=jax.tree_util.tree_map(lambda x: x[0], rollout.Tp1_graph.env_states.obstacle),
        Ta_unsafe=Ta_unsafe, a_unsafe=Ta_unsafe.any(0), a_finish=Ta_finish.any(0),
    )
    if keep_rollout:
        out.update(rollout=rollout, T_obs_edges=T_obs_edges, Ta_noise=Ta_noise, Ta_h_seen=Ta_h_seen,
                   Ta_h_true=Ta_h_true)
    return out


# %% simulate grid + video episodes at POINT and COMPARE_SIGMA; check they reproduce the sweep's flags
sigmas_run = [sigma] + ([COMPARE_SIGMA] if COMPARE_SIGMA is not None else [])
sims = {}  # (episode, sigma) -> simulate() output
for s in sigmas_run:
    j_sig = int(np.argmin(np.abs(res["sigmas"] - s)))
    for i_epi in dict.fromkeys(grid_epis + video_epis):
        sim = simulate(i_epi, s, keep_rollout=i_epi in video_epis)
        stored_u = res[f"ever_unsafe_n{n_agents}"][i_type, j_sig, i_epi]
        stored_f = res[f"ever_finish_n{n_agents}"][i_type, j_sig, i_epi]
        sim["matches_sweep"] = np.array_equal(sim["a_unsafe"], stored_u) and np.array_equal(sim["a_finish"], stored_f)
        sim["category"] = categorize(sim["a_unsafe"], sim["a_finish"])
        sims[(i_epi, s)] = sim

mismatch = [k for k, v in sims.items() if not v["matches_sweep"]]
print(f"re-simulated {len(sims)} episodes; reproduce the sweep's per-agent flags: {len(sims) - len(mismatch)}"
      f"/{len(sims)}")
for i_epi, s in mismatch:
    print(f"  MISMATCH episode {i_epi}, sigma {s:g}: re-simulated as {sims[(i_epi, s)]['category']}")


# %% trajectory grid: one panel per episode, paths + starts + goals + first collision points
def plot_grid(s: float, path: pathlib.Path):
    n_rows = int(np.ceil(len(grid_epis) / GRID_COLS))
    fig, axs = plt.subplots(n_rows, GRID_COLS, figsize=(3.2 * GRID_COLS, 3.4 * n_rows), squeeze=False)
    colors = plt.cm.tab10(np.arange(n_agents) % 10)
    for ax, i_epi in zip(axs.flat, grid_epis):
        sim = sims[(i_epi, s)]
        ax.add_collection(get_obs_collection(sim["obstacle"], OBS_COLOR, alpha=0.8))
        for a in range(n_agents):
            path_a = sim["Ta_pos"][:, a]
            ax.plot(*path_a.T, "-", color=colors[a], lw=0.8)
            ax.plot(*path_a[0], "o", color=colors[a], ms=3)  # start
            ax.plot(*sim["a_goal"][a], "*", color=colors[a], ms=7, mec="k", mew=0.3)  # goal
            # end position: filled if the agent ever reached its goal, hollow if it never did
            ax.plot(*path_a[-1], "o", ms=5, mec=colors[a], mfc=colors[a] if sim["a_finish"][a] else "none", mew=1.2)
            if sim["a_unsafe"][a]:
                t_hit = int(np.argmax(sim["Ta_unsafe"][:, a]))
                ax.plot(*path_a[t_hit], "x", color="r", ms=9, mew=2)
        n_safe, n_reach = int((~sim["a_unsafe"]).sum()), int(sim["a_finish"].sum())
        flag = "" if sim["matches_sweep"] else " (!)"
        ax.set_title(f"ep {i_epi}: {sim['category']}{flag}\nsafe {n_safe}/{n_agents}  reach {n_reach}/{n_agents}",
                     fontsize=9)
        ax.set_xlim(0, AREA_SIZE), ax.set_ylim(0, AREA_SIZE), ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
    for ax in axs.flat[len(grid_epis):]:
        ax.axis("off")
    fig.suptitle(f"{n_agents} agents, {noise_name} noise (tau = {tau:g} s), sigma = {s:g}.  "
                 f"o start, * goal, filled/hollow end = reached/never reached, red x = first collision, "
                 f"(!) = differs from sweep", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # leave room for the suptitle
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    print("saved", path)
    plt.show()


if GRID_PLOTS:
    for s in sigmas_run:
        plot_grid(s, FIG_DIR / f"sweep_samples_n{n_agents}_{noise_name}_s{sigma:.3f}_at_s{s:.3f}.png")

# %% videos of the selected episodes at each sigma (same map, same noise draws scaled by sigma)
if RENDER_VIDEOS:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for s in sigmas_run:
        for i_epi in video_epis:
            sim = sims[(i_epi, s)]
            n_safe, n_reach = int((~sim["a_unsafe"]).sum()), int(sim["a_finish"].sum())
            name = (f"n{n_agents}_{noise_name}_epi{i_epi:03}_s{s:.3f}_{sim['category']}"
                    f"_safe{n_safe}of{n_agents}_reach{n_reach}of{n_agents}.mp4")
            Ta_own_offset = sim["Ta_noise"][..., :2] if OWNSHIP_NOISE else None
            Ta_h = {"seen": sim["Ta_h_seen"], "true": sim["Ta_h_true"]} if PLOT_CBF else None
            render_video_noisy(env, sim["rollout"], sim["T_obs_edges"], VIDEO_DIR / name, sim["Ta_unsafe"],
                               GHOST_TYPES, Ta_own_offset=Ta_own_offset, draw_scale=DRAW_SCALE, Ta_h=Ta_h)
            print("saved", name)

# %%
