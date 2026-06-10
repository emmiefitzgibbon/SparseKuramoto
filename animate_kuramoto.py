"""
Kuramoto comparison animation. Run after kuramoto.py has set up the system:
    python animate_kuramoto.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from kuramoto import (
    N, A, A_sparse_ER, omega,
    sol_full, sol_ER, t_eval,
    edges_from_A, top_weight_edges,
    K, k_over_n, q, random_weights_std, edge_density, A_name, ring_k,
)


def _A_extra():
    parts = []
    if A_name.startswith('A_ring'):
        parts.append(f'k={ring_k}')
    if 'random' in A_name:
        if 'edges' in A_name:
            parts.append(f'edge density={edge_density}')
        if 'weights' in A_name or A_name == 'A_random_weights':
            parts.append(f'lognormal σ={random_weights_std}')
    return (' — ' + ', '.join(parts)) if parts else ''


def param_summary():
    n_full = len(edges_from_A(A)[0])
    n_sparse = len(edges_from_A(A_sparse_ER)[0])
    return '\n'.join([
        f'N={N}  K={K:.2g} (k/N={k_over_n})',
        f'full: {A_name}{_A_extra()}  ({n_full:,} edges)',
        f'sparse: ER q={q}  ({n_sparse:,} edges)',
    ])

# perf knobs — subsample solved trajectory; don't re-integrate
anim_frames_base = 120
t_end_ref = 5.0
sim_dt_per_frame = t_end_ref / anim_frames_base
anim_max_edges = 1500
anim_interval_ms = 50
anim_blit = True

L = int(np.sqrt(N))
R = 0.35
RING_R = L / 2
USE_RING_LAYOUT = A_name.startswith('A_ring')
norm = plt.Normalize(omega.min(), omega.max())
ball_colors = plt.cm.viridis(norm(omega))

t_end = float(t_eval[-1])
anim_frames = max(anim_frames_base, int(round(t_end / sim_dt_per_frame)))
frame_idx = np.linspace(0, sol_full.shape[1] - 1, anim_frames, dtype=int)
t_anim = t_eval[frame_idx]
sol_f = sol_full[:, frame_idx]
sol_s = sol_ER[:, frame_idx]

z = np.mean(np.exp(1j * sol_f), axis=0)
z_sparse = np.mean(np.exp(1j * sol_s), axis=0)
diff = np.degrees(np.angle(np.exp(1j * (sol_f - sol_s))))
dz = np.abs(z - z_sparse)
r_full = np.abs(z)
r_sparse = np.abs(z_sparse)
err_r = np.abs(r_full - r_sparse)
err_angle = np.degrees(np.abs(np.angle(z * np.conj(z_sparse))))

rows = np.arange(N) // L
cols = np.arange(N) % L


def nodes_xy(indices):
    idx = np.asarray(indices)
    if USE_RING_LAYOUT:
        ang = 2 * np.pi * idx / N - np.pi / 2
        return RING_R * np.cos(ang), RING_R * np.sin(ang)
    return (idx % L).astype(float), -(idx // L).astype(float)


def panel_limits():
    if USE_RING_LAYOUT:
        pad = R + 0.5
        return (-RING_R - pad, RING_R + pad), (-RING_R - pad, RING_R + pad)
    return (-0.6, L - 0.4), (-L + 0.4, 0.6)


def aligned_edge_lists(A_full, A_sparse, max_edges):
    """
    top-weight edges per graph; sparse list first in both panels, then full-only.
    matching index = matching z-order (LineCollection draws later segments on top).
    """
    ei_s, ej_s, _ = top_weight_edges(A_sparse, max_edges)
    ei_f, ej_f, _ = top_weight_edges(A_full, max_edges)
    sparse_pairs = set(zip(ei_s.tolist(), ej_s.tolist()))
    full_only = [(i, j) for i, j in zip(ei_f, ej_f) if (i, j) not in sparse_pairs]
    n_extra = max(0, max_edges - len(ei_s))
    full_only = full_only[:n_extra]
    ei_fo = np.fromiter((p[0] for p in full_only), dtype=int, count=len(full_only))
    ej_fo = np.fromiter((p[1] for p in full_only), dtype=int, count=len(full_only))
    return (ei_s, ej_s), (np.concatenate([ei_s, ei_fo]), np.concatenate([ej_s, ej_fo]))


def edge_segments(ei, ej):
    x1, y1 = nodes_xy(ei)
    x2, y2 = nodes_xy(ej)
    dx, dy = x2 - x1, y2 - y1
    d = np.maximum(np.hypot(dx, dy), 1e-12)
    p1 = np.column_stack([x1 + R * dx / d, y1 + R * dy / d])
    p2 = np.column_stack([x2 - R * dx / d, y2 - R * dy / d])
    return np.stack([p1, p2], axis=1)


def ball_offsets(sol):
    cx, cy = nodes_xy(np.arange(N))
    return np.stack(
        (cx[:, None] + R * np.cos(sol), cy[:, None] + R * np.sin(sol)),
        axis=-1,
    )


def edge_sin(sol, ei, ej):
    return np.sin(sol[ej, :] - sol[ei, :])


def unit_circle_offsets(sol):
    # (N, T, 2) — each oscillator on the unit circle at its phase angle
    return np.stack((np.cos(sol), np.sin(sol)), axis=-1)


(ei_s, ej_s), (ei_f, ej_f) = aligned_edge_lists(A, A_sparse_ER, anim_max_edges)
ball_f = ball_offsets(sol_f)
ball_s = ball_offsets(sol_s)

fig, axes = plt.subplots(
    2, 3, figsize=(13, 8.8),
    gridspec_kw={'height_ratios': [2.2, 0.9], 'width_ratios': [1, 1, 1.4]},
)
fig.suptitle(param_summary(), fontsize=9, y=0.98, ha='center', va='top')
print(param_summary())
ax_f, ax_s, ax_d = axes[0]
ax_zf, ax_zs, ax_rd = axes[1]

panels = []
_xlim, _ylim = panel_limits()
for ax, A_mat, ei, ej, balls, sol_sub, title in [
    (ax_f, A, ei_f, ej_f, ball_f, sol_f, 'full'),
    (ax_s, A_sparse_ER, ei_s, ej_s, ball_s, sol_s, 'sparse'),
]:
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    ax.set_xlim(*_xlim)
    ax.set_ylim(*_ylim)
    n_total = len(edges_from_A(A_mat)[0])
    ax.text(
        0.5, -0.04,
        f'{n_total:,} edges ({len(ei):,} shown)',
        transform=ax.transAxes, ha='center', va='top', fontsize=9,
    )
    for i in range(N):
        x, y = nodes_xy(i)
        ax.add_patch(plt.Circle((x, y), R, fill=False, ec='k', lw=0.8))
    lc = LineCollection(
        edge_segments(ei, ej), linewidths=0.1, cmap='coolwarm', clim=(-1, 1), zorder=1,
    )
    ax.add_collection(lc)
    sc = ax.scatter(
        balls[:, 0, 0], balls[:, 0, 1], s=18, c=ball_colors, zorder=3, edgecolors='none',
    )
    panels.append((lc, sc, edge_sin(sol_sub, ei, ej), balls))

op_panels = []
for ax, z_t, sol_sub, title in [
    (ax_zf, z, sol_f, 'full order param'),
    (ax_zs, z_sparse, sol_s, 'sparse order param'),
]:
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec='gray', lw=1, zorder=1))
    phase_pos = unit_circle_offsets(sol_sub)
    sc = ax.scatter(
        phase_pos[:, 0, 0], phase_pos[:, 0, 1],
        s=10, c=ball_colors, alpha=0.75, zorder=2, edgecolors='none',
    )
    ln, = ax.plot([0, z_t[0].real], [0, z_t[0].imag], 'k-', lw=2, zorder=3)
    dot, = ax.plot(z_t[0].real, z_t[0].imag, 'o', color='crimson', ms=6, zorder=4)
    op_panels.append((ln, dot, sc, phase_pos, z_t))

ax_rd.set_title('order param error (full − sparse)')
ax_rd.set_xlim(0, t_anim[-1])
ax_rd.set_ylim(0, 1.05)
ax_rd.set_xlabel('time (s)')
ax_rd.set_ylabel('|Δr|', color='crimson')
ax_rd.tick_params(axis='y', labelcolor='crimson')
bg_err_r, = ax_rd.plot(t_anim, err_r, color='crimson', alpha=0.3)
ax_rd_twin = ax_rd.twinx()
ax_rd_twin.set_ylim(0, max(180, err_angle.max() * 1.05))
ax_rd_twin.set_ylabel('|Δψ| (°)', color='royalblue')
ax_rd_twin.tick_params(axis='y', labelcolor='royalblue')
bg_err_angle, = ax_rd_twin.plot(t_anim, err_angle, color='royalblue', alpha=0.3)
op_err_r, = ax_rd.plot([], [], color='crimson', lw=2)
op_err_angle, = ax_rd_twin.plot([], [], color='royalblue', lw=2)
ax_rd.legend([bg_err_r, bg_err_angle], ['|Δr|', '|Δψ|'], loc='upper left', fontsize=8)

ax_d.set_title('phase diff per oscillator (full − sparse)')
phase_diff_segs = np.dstack([np.broadcast_to(t_anim, (N, anim_frames)), diff])
ax_d.add_collection(LineCollection(
    phase_diff_segs, colors=ball_colors, alpha=0.5, linewidths=0.4,
))
ax_d.set_xlim(0, t_anim[-1])
ax_d.set_xlabel('time (s)')
ax_d.set_ylabel('Δθ (°)')
ax_d.set_ylim(-180, 180)
vline = ax_d.axvline(t_anim[0], color='crimson', lw=2)
err_txt = ax_d.text(0.02, 0.95, '', transform=ax_d.transAxes, va='top')


def update(k):
    artists = []
    for lc, sc, s_all, balls in panels:
        s = s_all[:, k]
        lc.set_array(s)
        lc.set_linewidths(0.1 + 1.9 * np.abs(s))
        artists.append(lc)
        sc.set_offsets(balls[:, k, :])
        artists.append(sc)
    for ln, dot, sc, phase_pos, z_t in op_panels:
        ln.set_data([0, z_t[k].real], [0, z_t[k].imag])
        artists.append(ln)
        dot.set_data([z_t[k].real], [z_t[k].imag])
        artists.append(dot)
        sc.set_offsets(phase_pos[:, k, :])
        artists.append(sc)
    op_err_r.set_data(t_anim[: k + 1], err_r[: k + 1])
    op_err_angle.set_data(t_anim[: k + 1], err_angle[: k + 1])
    artists.append(op_err_r)
    artists.append(op_err_angle)
    vline.set_xdata([t_anim[k], t_anim[k]])
    artists.append(vline)
    err_txt.set_text(
        f'mean |Δθ| = {np.mean(np.abs(diff[:, k])):.1f}°  |  |Δr| = {err_r[k]:.3f}  |  |Δψ| = {err_angle[k]:.1f}°'
    )
    artists.append(err_txt)
    return artists


ani = FuncAnimation(
    fig, update, frames=anim_frames, interval=anim_interval_ms, blit=anim_blit,
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plots_dir = Path('plots')
plots_dir.mkdir(parents=True, exist_ok=True)
gif_path = plots_dir / f'sparse_kuramoto_{A_name.removeprefix("A_")}.gif'
ani.save(gif_path, writer='pillow', fps=1000 // anim_interval_ms)
print(f'saved {gif_path}')
plt.show()
