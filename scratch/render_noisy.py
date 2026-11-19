# Scratch copy of the 2D path of gcbfplus/env/plot.py:render_video, plus an overlay of where each agent
# PERCEIVED its neighbors given the noisy edge features the policy received. Works for 2D and 3D envs by drawing
# one or more 2D projections ("views") side by side, e.g. top (x-y) and side (x-z) for LinearDrone.
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection

from gcbfplus.env.base import MultiAgentEnv, RolloutResult
from gcbfplus.env.obstacle import Rectangle, Sphere
from gcbfplus.env.plot import get_obs_collection
from gcbfplus.utils.utils import MutablePatchCollection, save_anim, tree_index

GHOST_COLOR = "#ff8c00"
AGENT_COLOR = "#0068ff"
GOAL_COLOR = "#2fdd00"
OBS_COLOR = "#8a0000"
HIDDEN_POS = (-100.0, -100.0)  # off-screen parking spot for unused circles
DIM_NAMES = "xyz"


def _obstacle_artist(obstacles, dims: tuple):
    """Obstacles projected onto the view dims, or None if they can't be drawn in this view."""
    if isinstance(obstacles, Rectangle):
        return get_obs_collection(obstacles, OBS_COLOR, alpha=0.8) if dims == (0, 1) else None
    if isinstance(obstacles, Sphere):
        centers, radii = np.asarray(obstacles.center), np.asarray(obstacles.radius)
        circs = [plt.Circle(centers[ii, list(dims)], radii[ii]) for ii in range(len(radii))]
        return PatchCollection(circs, color=OBS_COLOR, alpha=0.5, zorder=2)
    return None


class _Panel:
    """All per-frame artists of one projection view."""

    def __init__(self, ax, dims: tuple, n_agent: int, n_edges: int, r: float, side_length: float, obstacles,
                 title: str = None):
        self.dims, self.n_agent = list(dims), n_agent
        ax.set_xlim(0., side_length)
        ax.set_ylim(0., side_length)
        ax.set(aspect="equal")
        ax.axis("off")
        if title is not None:
            ax.add_patch(plt.Rectangle((0, 0), side_length, side_length, fill=False, edgecolor="0.6", linewidth=1.0))
            ax.set_title(title, size=18, family="cursive", pad=40)

        obs_artist = _obstacle_artist(obstacles, dims)
        if obs_artist is not None:
            ax.add_collection(obs_artist)

        n_color = [AGENT_COLOR] * n_agent + [GOAL_COLOR] * n_agent
        self.agent_circs = [plt.Circle(HIDDEN_POS, r, color=n_color[ii], linewidth=0.0) for ii in range(n_agent * 2)]
        self.agent_col = MutablePatchCollection(list(reversed(self.agent_circs)), match_original=True, zorder=6)
        ax.add_collection(self.agent_col)

        # true edges (drawn between true node positions, as in the stock renderer)
        self.edge_col = LineCollection([], linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(self.edge_col)

        # ghosts: one hollow circle per observed edge (pool sized to all edges, unused ones parked off-screen)
        self.ghost_circs = [plt.Circle(HIDDEN_POS, r, fill=False, edgecolor=GHOST_COLOR, linewidth=1.5)
                            for _ in range(n_edges)]
        self.ghost_col = MutablePatchCollection(self.ghost_circs, match_original=True, zorder=7)
        ax.add_collection(self.ghost_col)
        self.ghost_line_col = LineCollection([], colors=GHOST_COLOR, linewidths=1.0, linestyles="dashed", alpha=0.8,
                                             zorder=4)
        ax.add_collection(self.ghost_line_col)

        # own-state estimates: hollow agent-colored circle at the believed position, thin line from the true agent
        self.own_circs = [plt.Circle(HIDDEN_POS, r, fill=False, edgecolor=AGENT_COLOR, linewidth=1.5)
                          for _ in range(n_agent)]
        self.own_col = MutablePatchCollection(self.own_circs, match_original=True, zorder=7)
        ax.add_collection(self.own_col)
        self.own_line_col = LineCollection([], colors=AGENT_COLOR, linewidths=1.0, alpha=0.8, zorder=4)
        ax.add_collection(self.own_line_col)

        label_font_opts = dict(size=20, color="k", family="cursive", weight="normal", ha="center", va="center",
                               transform=ax.transData, clip_on=True, zorder=8)
        self.agent_labels = [ax.text(*HIDDEN_POS, f"{ii}", **label_font_opts) for ii in range(n_agent)]

    @property
    def artists(self) -> list[plt.Artist]:
        return [self.agent_col, self.edge_col, self.ghost_col, self.ghost_line_col, self.own_col, self.own_line_col,
                *self.agent_labels]

    def update(self, n_pos, e_send, e_recv, e_is_goal, own_pos, g_recv_pos, g_pos) -> list[plt.Artist]:
        """Positions are full-dimensional; this panel projects them onto its dims. own_pos is None when hidden."""
        n_agent, d = self.n_agent, self.dims
        p = n_pos[:, d]
        for ii in range(n_agent * 2):
            self.agent_circs[ii].set_center(tuple(p[ii]))
        for ii in range(n_agent):
            self.agent_labels[ii].set_position(p[ii])

        self.edge_col.set_segments(np.stack([p[e_send], p[e_recv]], axis=1))
        self.edge_col.set_colors([GOAL_COLOR if g else "0.2" for g in e_is_goal])

        if own_pos is not None:
            o = own_pos[:, d]
            self.own_line_col.set_segments(np.stack([p[:n_agent], o], axis=1))
            for ii, circ in enumerate(self.own_circs):
                circ.set_center(tuple(o[ii]))
        else:
            self.own_line_col.set_segments([])
            for circ in self.own_circs:
                circ.set_center(HIDDEN_POS)

        g_recv, g = g_recv_pos[:, d], g_pos[:, d]
        for ii, circ in enumerate(self.ghost_circs):
            circ.set_center(tuple(g[ii]) if ii < len(g) else HIDDEN_POS)
        self.ghost_line_col.set_segments(np.stack([g_recv, g], axis=1))
        return self.artists


def render_video_noisy(
        env: MultiAgentEnv,
        rollout: RolloutResult,
        T_obs_edges: np.ndarray,
        video_path: pathlib.Path,
        Ta_is_unsafe: np.ndarray,
        ghost_types: tuple = (0,),
        dpi: int = 100,
        Ta_own_offset: np.ndarray = None,
        views: tuple = ((0, 1),),
        draw_scale: float = 1.0,
        Ta_h: dict = None,
):
    """
    T_obs_edges: (T, n_edges, edge_dim) noisy edge features the policy saw at each step; T_obs_edges[kk] observes
        rollout.Tp1_graph[kk]. Edge features are state[receiver] - state[sender] with positions first, so the
        receiver's perceived sender position is pos[receiver] - obs_edge[:pos_dim]. Note: features may be clipped to
        comm_radius, so a clipped edge's ghost sits at comm_radius along the edge direction, which is what the policy
        actually sees.
    ghost_types: sender node types to draw ghosts for. 0 = agent, 1 = goal, 2 = lidar hit.
    Ta_own_offset: optional (T, n_agents, pos_dim) error in each agent's estimate of its OWN position. If given, each
        agent's believed position (true + offset) is drawn as a hollow blue circle, and ghosts are anchored there
        (believed pos_i - obs_edge), so each dashed line is exactly the relative vector agent i computed.
    views: pairs of position dims, one panel each. ((0, 1),) = top-down only; ((0, 1), (0, 2)) = top + side.
    draw_scale: multiplies the drawn agent / goal / ghost radius (display only). With draw_scale > 1, circles overlap
        before agents actually collide; the "Unsafe" text is still the true collision check.
    Ta_h: optional {label: (T, n_agents) array} of barrier values h_i(t), e.g. {"seen": h on the graph the policy
        saw, "true": h on the true graph}. Adds a strip chart under the map(s): one color per agent (agent numbers on
        the map use the same color), first label solid, second dashed, x = first collision, moving cursor = current
        frame. On the map, a red ring marks agents whose first-label h is < 0.
    """
    n_agent = env.num_agents
    r = env.params.get("car_radius", env.params.get("drone_radius")) * draw_scale
    side_length = env.area_size
    pos_dim = max(max(v) for v in views) + 1

    T_graph = rollout.Tp1_graph
    graph0 = tree_index(T_graph, 0)
    pad_id = graph0.states.shape[0] - 1  # the padding node is always last
    n_edges = graph0.edges.shape[0]

    n_views = len(views)
    if Ta_h is None:
        fig, axs = plt.subplots(1, n_views, figsize=(10 * n_views, 10), dpi=dpi, squeeze=False)
        axs = axs[0]
    else:
        fig = plt.figure(figsize=(10 * n_views, 14), dpi=dpi)
        gs = fig.add_gridspec(2, n_views, height_ratios=(10, 3.2), hspace=0.12, top=0.93, bottom=0.1)
        axs = [fig.add_subplot(gs[0, ii]) for ii in range(n_views)]
        ax_h = fig.add_subplot(gs[1, :])
    panels = [
        _Panel(ax, tuple(v), n_agent, n_edges, r, side_length, graph0.env_states.obstacle,
               title=f"{'top' if v == (0, 1) else 'side'} ({DIM_NAMES[v[0]]}-{DIM_NAMES[v[1]]})" if n_views > 1
               else None)
        for ax, v in zip(axs, views)
    ]

    # text: status on the first panel, legend on the last
    text_font_opts = dict(size=16, color="k", family="cursive", weight="normal")
    ax0, ax1 = axs[0], axs[-1]
    cost_text = ax0.text(0.02, 1.04, "Cost: 1.0, Reward: 1.0", va="bottom", transform=ax0.transAxes, **text_font_opts)
    safe_text = ax0.text(0.02, 1.00, "Unsafe: {}", va="bottom", transform=ax0.transAxes, **text_font_opts)
    kk_text = ax1.text(0.99, 0.99, "kk=0", va="top", ha="right", transform=ax1.transAxes, **text_font_opts)
    ax1.text(0.99, 1.04, "orange = perceived neighbor", va="bottom", ha="right", transform=ax1.transAxes,
             **{**text_font_opts, "color": GHOST_COLOR})
    if Ta_own_offset is not None:
        ax1.text(0.99, 1.00, "hollow blue = own estimate", va="bottom", ha="right", transform=ax1.transAxes,
                 **{**text_font_opts, "color": AGENT_COLOR})
    if draw_scale != 1.0:
        ax1.text(0.99, 0.96, f"circles drawn {draw_scale:g}x actual size", va="top", ha="right",
                 transform=ax1.transAxes, **{**text_font_opts, "size": 12, "color": "0.4"})
    texts = [cost_text, safe_text, kk_text]

    # barrier strip chart (static lines; only the cursor and the map rings move)
    h_artists, h_rings = [], []
    if Ta_h is not None:
        import matplotlib.patheffects as pe
        a_color = plt.cm.tab10(np.arange(n_agent) % 10)
        labels = list(Ta_h)
        dt = float(env.dt)
        for li, (lab, Ta_hl) in enumerate(Ta_h.items()):
            Ta_hl = np.asarray(Ta_hl)
            t_axis = np.arange(len(Ta_hl)) * dt
            for ii in range(n_agent):
                ax_h.plot(t_axis, Ta_hl[:, ii], color=a_color[ii], lw=1.6 if li == 0 else 1.2,
                          ls="-" if li == 0 else "--", alpha=1.0 if li == 0 else 0.7,
                          label=f"{ii}" if li == 0 else None)
            for ii in range(n_agent):  # first collision of each agent, on every label's line
                hit = np.flatnonzero(np.asarray(Ta_is_unsafe)[:len(Ta_hl), ii])
                if len(hit):
                    ax_h.plot(t_axis[hit[0]], Ta_hl[hit[0], ii], "x", color="r", ms=12 if li == 0 else 9,
                              mew=3 if li == 0 else 2, zorder=5)
        ax_h.axhline(0.0, color="k", lw=1.0)
        ax_h.set_xlim(0, (len(T_graph.n_node) - 1) * dt)
        ax_h.set_xlabel("time [s]", size=14)
        style = f"{labels[0]} (solid)" + (f", {labels[1]} (dashed)" if len(labels) > 1 else "")
        ax_h.set_ylabel("h", size=14)
        ax_h.set_title(f"barrier h per agent: {style};  red x = first collision", size=13, loc="left")
        leg = ax_h.legend(title="agent", ncol=n_agent, loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=10,
                          title_fontsize=10, frameon=False)
        leg._legend_box.align = "left"
        ax_h.grid(alpha=0.3)
        h_cursor = ax_h.axvline(0.0, color="0.3", lw=1.5)
        h_artists = [h_cursor]
        h_first = np.asarray(Ta_h[labels[0]])
        # agent numbers on the map take the agent's line color; red ring = first-label h < 0
        for panel in panels:
            for ii, lbl in enumerate(panel.agent_labels):
                lbl.set_color(a_color[ii])
                lbl.set_path_effects([pe.withStroke(linewidth=3, foreground="w")])
            rings = [plt.Circle(HIDDEN_POS, 2.2 * r, fill=False, edgecolor="r", linewidth=2.0) for _ in range(n_agent)]
            ring_col = MutablePatchCollection(rings, match_original=True, zorder=7)
            panel_ax = panel.agent_col.axes
            panel_ax.add_collection(ring_col)
            h_rings.append((panel, rings, ring_col))
            h_artists.append(ring_col)

    def init_fn() -> list[plt.Artist]:
        return [a for panel in panels for a in panel.artists] + texts + h_artists

    def update(kk: int) -> list[plt.Artist]:
        graph = tree_index(T_graph, kk)
        n_pos_t = np.asarray(graph.states[:-1, :pos_dim])
        senders, receivers = np.asarray(graph.senders), np.asarray(graph.receivers)

        # true edges
        is_pad = (senders == pad_id) | (receivers == pad_id)
        e_send, e_recv = senders[~is_pad], receivers[~is_pad]
        e_is_goal = (n_agent <= e_send) & (e_send < n_agent * 2)

        # own-state estimates (believed position = true + offset); hidden on the last frame (no observation)
        anchor_pos = n_pos_t.copy()
        own_pos = None
        if Ta_own_offset is not None and kk < len(Ta_own_offset):
            anchor_pos[:n_agent] += Ta_own_offset[kk][:, :pos_dim]
            own_pos = anchor_pos[:n_agent]

        # ghosts: perceived sender position = receiver's (believed) position - noisy relative position
        if kk < len(T_obs_edges):
            node_type = np.asarray(graph.node_type)
            is_ghost = ~is_pad & np.isin(node_type[senders], ghost_types)
            g_recv_pos = anchor_pos[receivers[is_ghost]]
            g_pos = g_recv_pos - T_obs_edges[kk][is_ghost, :pos_dim]
        else:
            g_recv_pos = g_pos = np.zeros((0, pos_dim))

        artists = []
        for panel in panels:
            artists += panel.update(n_pos_t, e_send, e_recv, e_is_goal, own_pos, g_recv_pos, g_pos)

        # text
        if kk < len(rollout.T_cost):
            cost_text.set_text("Cost: {:5.4f}, Reward: {:5.4f}".format(rollout.T_cost[kk], rollout.T_reward[kk]))
        else:
            cost_text.set_text("")
        safe_text.set_text("Unsafe: {}".format(np.where(Ta_is_unsafe[kk])[0] if kk < len(Ta_is_unsafe) else ""))
        kk_text.set_text("kk={:04}".format(kk))

        if Ta_h is not None:
            h_cursor.set_xdata([kk * dt, kk * dt])
            a_neg = h_first[kk] < 0 if kk < len(h_first) else np.zeros(n_agent, dtype=bool)
            for panel, rings, _ in h_rings:
                p = n_pos_t[:n_agent][:, panel.dims]
                for ii, ring in enumerate(rings):
                    ring.set_center(tuple(p[ii]) if a_neg[ii] else HIDDEN_POS)
        return artists + texts + h_artists

    fps = 30.0
    ani = FuncAnimation(fig, update, frames=len(T_graph.n_node), init_func=init_fn, interval=1_000 / fps, blit=True)
    save_anim(ani, video_path)
    plt.close(fig)
