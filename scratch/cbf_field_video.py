# Barrier-field video: for one agent k, color the map around it by the learned GCBF+ barrier h_k it would have if it
# were at each grid point, with everything else frozen at that frame (other agents, goals, obstacles, and agent k's own
# velocity). The h_k = 0 line is the edge of the unsafe zone as agent k sees it.
# Unlike test.py --cbf (trainer/utils.py:get_bb_cbf), each grid point rebuilds the whole graph with env.get_graph, so the
# LiDAR is re-cast and the neighbor set recomputed from the moved position (otherwise obstacles show up only as the
# frozen scan points, and the neighbor set is wrong away from the agent's real position).
# Noise-free for now (sigma = 0); episodes keyed exactly like scratch/di_noise_sweep.py.
# Run cell-by-cell in the VS Code Interactive Window, or: python scratch/cbf_field_video.py

# %% config
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
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.env.plot import get_obs_collection
from gcbfplus.utils.utils import save_anim, tree_concat_at_front, tree_index

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd()
MODEL_PATH = REPO_ROOT / "pretrained/DoubleIntegrator/gcbf+"
SWEEP_PATH = REPO_ROOT / "scratch/results/di_noise_sweep_epi1000.npz"  # for the flag check at sigma = 0
VIDEO_DIR = REPO_ROOT / "scratch/videos/cbf_field"
FIG_DIR = REPO_ROOT / "scratch/figures"

N_AGENTS = 8
AREA_SIZE = 4
N_OBS = 4
MAX_STEP = 1024
SEED = 1234  # map keys, as in the sweep

# env vars override these, so a cluster job (cluster/cbf_field.sbatch) can pick the episode / agent without edits
EPISODE = int(os.environ.get("EPISODE", 2))
AGENT = int(os.environ.get("AGENT", 1))  # agent whose barrier field is drawn
HALF_WIN = 0.75  # field window: agent position +- HALF_WIN
N_GRID = int(os.environ.get("N_GRID", 61))  # grid points per side (61 -> spacing 2 * HALF_WIN / 60 = 0.025)
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", 2))  # render every FRAME_STRIDE-th step
N_STILLS = 4  # still-frame check figure: the steps where agent AGENT's own h is lowest (its close encounters)
STILL_SEP = 60  # minimum steps between stills

# Full float32 matmuls. On recent NVIDIA GPUs jax's default float32 matmul precision uses TF32 (~1e-3 relative error),
# which makes GPU results drift from CPU / Mac runs; "highest" keeps them comparable at negligible cost here.
jax.config.update("jax_default_matmul_precision", os.environ.get("JAX_MATMUL_PRECISION", "highest"))

with open(MODEL_PATH / "config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.UnsafeLoader)
print(f"jax {jax.__version__} on {jax.devices()}; episode {EPISODE}, agent {AGENT}, grid {N_GRID}, "
      f"stride {FRAME_STRIDE}; matmul precision {jax.config.jax_default_matmul_precision}")


# %% env + algo (as in scratch/di_noise_sweep_videos.py)
def load_env_algo(n_agents: int):
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


env, algo = load_env_algo(N_AGENTS)
test_keys = jr.split(jr.PRNGKey(SEED), 1_000)
key_x0s = jax.vmap(lambda k: jr.split(k, 2)[0])(test_keys)


# %% noise-free rollout
@jax.jit
def rollout(key_x0):
    graph0 = env.reset(key_x0)

    def body(graph, _):
        graph_new = env.step(graph, algo.act(graph))[0]
        return graph_new, graph_new

    _, T_graph = lax.scan(body, graph0, None, length=MAX_STEP)
    return tree_concat_at_front(graph0, T_graph, axis=0)


Tp1_graph = rollout(key_x0s[EPISODE])
Ta_unsafe = np.asarray(jax.vmap(env.collision_mask)(Tp1_graph))
Ta_finish = np.asarray(jax.vmap(env.finish_mask)(Tp1_graph))
Ta_state = np.asarray(Tp1_graph.env_states.agent)  # (T + 1, n, 4)
a_goal = np.asarray(Tp1_graph.env_states.goal[0])
obstacle = tree_index(Tp1_graph.env_states.obstacle, 0)
Ta_h_true = np.asarray(jax.vmap(algo.get_cbf)(Tp1_graph)[..., 0])  # h of every agent at its real position

match = "n/a (no stored sweep file)"
if SWEEP_PATH.exists():
    res = np.load(SWEEP_PATH)
    i_gm, i_s0 = list(res["noise_names"]).index("gm"), int(np.flatnonzero(res["sigmas"] == 0.0)[0])
    match = (np.array_equal(Ta_unsafe.any(0), res["ever_unsafe_n8"][i_gm, i_s0, EPISODE])
             and np.array_equal(Ta_finish.any(0), res["ever_finish_n8"][i_gm, i_s0, EPISODE]))
print(f"episode {EPISODE}, sigma 0: collided {np.flatnonzero(Ta_unsafe.any(0))}, reached {Ta_finish.any(0).sum()}/"
      f"{N_AGENTS}; matches the stored sweep flags: {match}")

STILL_STEPS = []
for kk in np.argsort(Ta_h_true[:, AGENT]):
    if all(abs(int(kk) - s) >= STILL_SEP for s in STILL_STEPS):
        STILL_STEPS.append(int(kk))
    if len(STILL_STEPS) == N_STILLS:
        break
STILL_STEPS = sorted(STILL_STEPS)
print("stills at steps", STILL_STEPS, "h =", np.round(Ta_h_true[STILL_STEPS, AGENT], 3))


# %% barrier field of agent AGENT around its position, rebuilding the full graph at each grid point
offs = np.linspace(-HALF_WIN, HALF_WIN, N_GRID)


@jax.jit
def field(env_state):
    """h_AGENT on an (N_GRID, N_GRID) grid centered on AGENT (rows = y, cols = x); velocity kept."""
    center = env_state.agent[AGENT, :2]
    gx, gy = jnp.meshgrid(center[0] + offs, center[1] + offs)
    pts = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)

    def h_at(p):
        agent = env_state.agent.at[AGENT, :2].set(p)
        g = env.get_graph(env_state._replace(agent=agent))
        return algo.get_cbf(g)[AGENT, 0]

    return lax.map(h_at, pts, batch_size=1024).reshape(N_GRID, N_GRID)


# sanity: the grid center is the real position, so it must equal h on the rollout's own graph
mid = N_GRID // 2
for kk in STILL_STEPS[:2]:
    h_c = float(field(tree_index(Tp1_graph.env_states, kk))[mid, mid])
    print(f"step {kk}: field center {h_c:+.5f} vs rollout graph {Ta_h_true[kk, AGENT]:+.5f}")

t0 = time.time()
still_fields = {kk: np.asarray(field(tree_index(Tp1_graph.env_states, kk))) for kk in STILL_STEPS}
t_frame = (time.time() - t0) / len(STILL_STEPS)
frames = np.arange(0, MAX_STEP + 1, FRAME_STRIDE)
print(f"field: {t_frame:.2f} s per frame -> about {t_frame * len(frames) / 60:.0f} min for {len(frames)} video frames")


# %% drawing: left = full map (window outlined), right = zoom on the window with the field
A_COLOR, G_COLOR, K_COLOR, OBS_COLOR = "#0068ff", "#2fdd00", "#111111", "#8a0000"
r = env.params["car_radius"]
CMAP = "RdBu"  # red = h < 0 (unsafe for agent k), blue = h > 0
HMAX = 1.0  # h = tanh(.) is in [-1, 1]; fixed scale so colors mean the same in every frame and figure
norm = TwoSlopeNorm(vcenter=0.0, vmin=-HMAX, vmax=HMAX)


def _scene(ax, kk: int, label_size: float):
    ax.add_collection(get_obs_collection(obstacle, OBS_COLOR, alpha=0.8))
    for ii in range(N_AGENTS):
        p = Ta_state[kk, ii, :2]
        ax.add_patch(plt.Circle(a_goal[ii, :2], r, color=G_COLOR, lw=0, zorder=4))
        ax.add_patch(plt.Circle(p, r, color=K_COLOR if ii == AGENT else A_COLOR, lw=0, zorder=5))
        ax.text(*p, f"{ii}", color="w", ha="center", va="center", fontsize=label_size, zorder=6, clip_on=True)


def draw_frame(ax_map, ax_zoom, kk: int, h_grid: np.ndarray):
    """Step kk: overview on ax_map, agent AGENT's barrier field h_grid on ax_zoom (both redrawn from scratch)."""
    c = Ta_state[kk, AGENT, :2]
    v = Ta_state[kk, AGENT, 2:]
    ext = (c[0] - HALF_WIN, c[0] + HALF_WIN, c[1] - HALF_WIN, c[1] + HALF_WIN)

    ax_map.clear()
    ax_map.set_xlim(0, AREA_SIZE), ax_map.set_ylim(0, AREA_SIZE), ax_map.set_aspect("equal")
    ax_map.set_xticks([]), ax_map.set_yticks([])
    _scene(ax_map, kk, 6)
    ax_map.add_patch(plt.Rectangle(ext[::2], 2 * HALF_WIN, 2 * HALF_WIN, fill=False, ec="k", lw=1.2, zorder=7))
    unsafe = np.flatnonzero(Ta_unsafe[kk])
    ax_map.set_title(f"t = {kk * env.dt:5.2f} s (step {kk})" + (f"   colliding: {list(unsafe)}" if len(unsafe) else ""),
                     fontsize=11, loc="left")

    ax_zoom.clear()
    ax_zoom.set_xlim(ext[0], ext[1]), ax_zoom.set_ylim(ext[2], ext[3]), ax_zoom.set_aspect("equal")
    ax_zoom.tick_params(labelsize=8)
    im = ax_zoom.imshow(h_grid, origin="lower", extent=ext, cmap=CMAP, norm=norm, interpolation="bilinear", zorder=1)
    if h_grid.min() < 0 < h_grid.max():
        ax_zoom.contour(c[0] + offs, c[1] + offs, h_grid, levels=[0.0], colors="k", linewidths=2.0, zorder=2)
    for edge in (0.0, AREA_SIZE):  # arena boundary, if inside the window
        ax_zoom.axvline(edge, color="0.3", lw=1, ls=":"), ax_zoom.axhline(edge, color="0.3", lw=1, ls=":")
    _scene(ax_zoom, kk, 9)
    ax_zoom.add_patch(plt.Circle(c, env.params["comm_radius"], fill=False, ec="0.3", lw=1, ls="--", zorder=3))
    ax_zoom.annotate("", xy=c + 0.6 * v, xytext=c, arrowprops=dict(arrowstyle="->", color=K_COLOR, lw=1.8), zorder=7)
    ax_zoom.set_title(f"agent {AGENT}: h = {Ta_h_true[kk, AGENT]:+.3f},  |v| = {np.linalg.norm(v):.2f}",
                      fontsize=11, loc="left")
    return im


LEGEND = (f"Right: color = h_{AGENT} if agent {AGENT} (black) were at that point with its current velocity (arrow); "
          f"everything else frozen. Black line: h = 0. Dashed circle: sensing radius.")

# %% still frames (check before rendering the video)
fig, axs = plt.subplots(2, len(STILL_STEPS), figsize=(4.6 * len(STILL_STEPS), 9.6))
for j, kk in enumerate(STILL_STEPS):
    im = draw_frame(axs[0, j], axs[1, j], kk, still_fields[kk])
fig.colorbar(im, ax=axs[1, :], shrink=0.9, label=f"h_{AGENT}")
fig.suptitle(f"Episode {EPISODE}, sigma = 0: barrier field of agent {AGENT} at its {len(STILL_STEPS)} lowest-h moments"
             f"\n{LEGEND}", fontsize=11)
FIG_DIR.mkdir(parents=True, exist_ok=True)  # scratch/figures/ is gitignored, so a fresh clone does not have it
still_path = FIG_DIR / f"cbf_field_epi{EPISODE:03}_agent{AGENT}_s0.000_stills.png"
fig.savefig(still_path, dpi=100, bbox_inches="tight")
print("saved", still_path)
plt.show()

# %% video: field for every FRAME_STRIDE-th step, then render
RENDER_VIDEO = os.environ.get("RENDER_VIDEO", "1") == "1"
if not RENDER_VIDEO:
    raise SystemExit(0)
t0 = time.time()
F_field = []
for i_f, kk in enumerate(frames):
    F_field.append(np.asarray(field(tree_index(Tp1_graph.env_states, int(kk)))))
    if (i_f + 1) % 50 == 0:
        print(f"  field frames {i_f + 1}/{len(frames)} ({time.time() - t0:.0f} s)")
F_field = np.stack(F_field)
print(f"field done in {time.time() - t0:.0f} s; h range [{F_field.min():.3f}, {F_field.max():.3f}]")

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
fig, (ax_map, ax_zoom) = plt.subplots(1, 2, figsize=(15, 7.6), gridspec_kw=dict(width_ratios=(1, 1.15)))
im = draw_frame(ax_map, ax_zoom, 0, F_field[0])
fig.colorbar(im, ax=ax_zoom, shrink=0.85, label=f"h_{AGENT}  (red < 0 < blue)")
fig.text(0.01, 0.01, LEGEND, fontsize=9, color="0.3")
ani = FuncAnimation(fig, lambda i: draw_frame(ax_map, ax_zoom, int(frames[i]), F_field[i]), frames=len(frames),
                    interval=1000 / 30 * FRAME_STRIDE, blit=False)
video_path = VIDEO_DIR / f"cbf_field_epi{EPISODE:03}_agent{AGENT}_s0.000.mp4"
save_anim(ani, video_path)
plt.close(fig)
print("saved", video_path)

# %%
