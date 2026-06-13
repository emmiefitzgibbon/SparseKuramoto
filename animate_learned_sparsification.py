"""
Animate full vs learned sparse Kuramoto from saved training checkpoint.

    python animate_learned_sparsification.py
    python animate_learned_sparsification.py plots/best_sparsification_logits.pt
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from optimized_sparsification import (
    N,
    K,
    Q,
    EDGE_BUDGET,
    T_END,
    fc_adj,
    adjacency_from_logits as adjacency_gumbel,
    integrate_torch,
)
from optimized_sparsification_simple import adjacency_from_logits as adjacency_simple


def sparse_adjacency_np(sigma_np, ei, ej, n, threshold=0.5):
    mask = sigma_np > threshold
    a = np.zeros((n, n))
    a[ei[mask], ej[mask]] = sigma_np[mask]
    a[ej[mask], ei[mask]] = sigma_np[mask]
    return a

_args = [a for a in sys.argv[1:] if a != "--no-show"]
SHOW = "--no-show" not in sys.argv
CKPT_PATH = Path(_args[0]) if _args else Path("plots/best_sparsification_logits.pt")

# perf knobs — subsample solved trajectory; don't re-integrate
anim_frames_base = 120
t_end_ref = 5.0
sim_dt_per_frame = t_end_ref / anim_frames_base
anim_max_edges = 1500
anim_interval_ms = 50
anim_blit = True
n_anim_steps = 1000


def edges_from_A(A):
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    ei, ej = np.where(np.triu(A, 1))
    return ei, ej, A[ei, ej].astype(float)


def top_weight_edges(A, max_edges):
    ei, ej, w = edges_from_A(A)
    if len(ei) <= max_edges:
        return ei, ej, w
    idx = np.argpartition(w, -max_edges)[-max_edges:]
    return ei[idx], ej[idx], w[idx]


def load_from_checkpoint(ckpt_path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}; run optimized_sparsification.py first")

    ckpt = torch.load(ckpt_path, weights_only=False)
    seed = int(ckpt["seed"])
    logits = ckpt["logits"]
    print(
        f"loaded {ckpt_path} | epoch {ckpt['epoch']} | "
        f"circ loss {float(ckpt['loss']):.4e} | seed {seed}"
    )

    n_full_edges = N * (N - 1) // 2
    edge_budget = EDGE_BUDGET if EDGE_BUDGET is not None else int(Q * n_full_edges)

    ei, ej = np.triu_indices(N, k=1)
    ei_t = torch.tensor(ei)
    ej_t = torch.tensor(ej)
    if "omega" in ckpt and "theta0" in ckpt:
        omega = ckpt["omega"].numpy()
        theta0 = ckpt["theta0"].numpy()
        print("using omega/theta0 stored in checkpoint")
    else:
        rng = np.random.default_rng(seed)
        omega = rng.normal(5.0, 0.5, N)
        theta0 = rng.uniform(0, 2 * np.pi, N)
        print("rebuilt omega/theta0 from seed (older checkpoint — re-run training to store ICs)")
    t_eval = np.linspace(0, T_END, n_anim_steps)
    omega_t = torch.tensor(omega, dtype=torch.float64)
    theta0_t = torch.tensor(theta0, dtype=torch.float64)
    t_t = torch.tensor(t_eval, dtype=torch.float64)

    a_full = torch.tensor(fc_adj(N), dtype=torch.float64)
    with torch.no_grad():
        if "simple" in ckpt_path.stem:
            a_learned_t, sigma = adjacency_simple(logits, ei_t, ej_t, N)
            a_learned = sparse_adjacency_np(sigma.numpy(), ei, ej, N)
            n_active = int(np.count_nonzero(np.triu(a_learned, 1)))
        else:
            a_learned_t, z = adjacency_gumbel(logits, ei_t, ej_t, N, beta=None)
            a_learned = a_learned_t.numpy()
            n_active = int((z > 0).sum().item())

    # same torchdiffeq integrator as training
    with torch.no_grad():
        sol_full = integrate_torch(theta0_t, omega_t, a_full, K, t_t).numpy().T
        sol_learned = integrate_torch(theta0_t, omega_t, a_learned_t, K, t_t).numpy().T

    return {
        "seed": seed,
        "epoch": int(ckpt["epoch"]),
        "loss": float(ckpt["loss"]),
        "edge_budget": edge_budget,
        "n_active": n_active,
        "omega": omega,
        "a_full": a_full.numpy(),
        "a_learned": a_learned,
        "t_eval": t_eval,
        "sol_full": sol_full,
        "sol_learned": sol_learned,
    }


data = load_from_checkpoint(CKPT_PATH)
omega = data["omega"]
A = data["a_full"]
A_learned = data["a_learned"]
t_eval = data["t_eval"]
sol_full = data["sol_full"]
sol_learned = data["sol_learned"]


def param_summary():
    n_full = len(edges_from_A(A)[0])
    n_sparse = len(edges_from_A(A_learned)[0])
    return "\n".join([
        f"N={N}  K={K:.2g}  q={Q}  budget={data['edge_budget']:,}",
        f"full: FC  ({n_full:,} edges)",
        f"learned sparse: {n_sparse:,} active edges (epoch {data['epoch']}, loss {data['loss']:.3e})",
    ])


L = int(np.sqrt(N))
R = 0.35
norm = plt.Normalize(omega.min(), omega.max())
ball_colors = plt.cm.viridis(norm(omega))

t_end = float(t_eval[-1])
anim_frames = max(anim_frames_base, int(round(t_end / sim_dt_per_frame)))
frame_idx = np.linspace(0, sol_full.shape[1] - 1, anim_frames, dtype=int)
t_anim = t_eval[frame_idx]
sol_f = sol_full[:, frame_idx]
sol_s = sol_learned[:, frame_idx]

z = np.mean(np.exp(1j * sol_f), axis=0)
z_learned = np.mean(np.exp(1j * sol_s), axis=0)
diff = np.degrees(np.angle(np.exp(1j * (sol_f - sol_s))))
r_full = np.abs(z)
r_learned = np.abs(z_learned)
err_r = np.abs(r_full - r_learned)
err_angle = np.degrees(np.abs(np.angle(z * np.conj(z_learned))))


def nodes_xy(indices):
    idx = np.asarray(indices)
    return (idx % L).astype(float), -(idx // L).astype(float)


def panel_limits():
    return (-0.6, L - 0.4), (-L + 0.4, 0.6)


def aligned_edge_lists(A_full, A_sparse, max_edges):
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
    return np.stack((np.cos(sol), np.sin(sol)), axis=-1)


(ei_s, ej_s), (ei_f, ej_f) = aligned_edge_lists(A, A_learned, anim_max_edges)
ball_f = ball_offsets(sol_f)
ball_s = ball_offsets(sol_s)

fig, axes = plt.subplots(
    2, 3, figsize=(13, 8.8),
    gridspec_kw={"height_ratios": [2.2, 0.9], "width_ratios": [1, 1, 1.4]},
)
fig.suptitle(param_summary(), fontsize=9, y=0.98, ha="center", va="top")
print(param_summary())
ax_f, ax_s, ax_d = axes[0]
ax_zf, ax_zs, ax_rd = axes[1]

panels = []
_xlim, _ylim = panel_limits()
for ax, A_mat, ei, ej, balls, sol_sub, title in [
    (ax_f, A, ei_f, ej_f, ball_f, sol_f, "full"),
    (ax_s, A_learned, ei_s, ej_s, ball_s, sol_s, "learned sparse"),
]:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    ax.set_xlim(*_xlim)
    ax.set_ylim(*_ylim)
    n_total = len(edges_from_A(A_mat)[0])
    ax.text(
        0.5, -0.04,
        f"{n_total:,} edges ({len(ei):,} shown)",
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
    )
    for i in range(N):
        x, y = nodes_xy(i)
        ax.add_patch(plt.Circle((x, y), R, fill=False, ec="k", lw=0.8))
    lc = LineCollection(
        edge_segments(ei, ej), linewidths=0.1, cmap="coolwarm", clim=(-1, 1), zorder=1,
    )
    ax.add_collection(lc)
    sc = ax.scatter(
        balls[:, 0, 0], balls[:, 0, 1], s=18, c=ball_colors, zorder=3, edgecolors="none",
    )
    panels.append((lc, sc, edge_sin(sol_sub, ei, ej), balls))

op_panels = []
for ax, z_t, sol_sub, title in [
    (ax_zf, z, sol_f, "full order param"),
    (ax_zs, z_learned, sol_s, "learned order param"),
]:
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec="gray", lw=1, zorder=1))
    phase_pos = unit_circle_offsets(sol_sub)
    sc = ax.scatter(
        phase_pos[:, 0, 0], phase_pos[:, 0, 1],
        s=10, c=ball_colors, alpha=0.75, zorder=2, edgecolors="none",
    )
    ln, = ax.plot([0, z_t[0].real], [0, z_t[0].imag], "k-", lw=2, zorder=3)
    dot, = ax.plot(z_t[0].real, z_t[0].imag, "o", color="crimson", ms=6, zorder=4)
    op_panels.append((ln, dot, sc, phase_pos, z_t))

ax_rd.set_title("order param error (full − learned)")
ax_rd.set_xlim(0, t_anim[-1])
ax_rd.set_ylim(0, 1.05)
ax_rd.set_xlabel("time (s)")
ax_rd.set_ylabel("|Δr|", color="crimson")
ax_rd.tick_params(axis="y", labelcolor="crimson")
bg_err_r, = ax_rd.plot(t_anim, err_r, color="crimson", alpha=0.3)
ax_rd_twin = ax_rd.twinx()
ax_rd_twin.set_ylim(0, max(180, err_angle.max() * 1.05))
ax_rd_twin.set_ylabel("|Δψ| (°)", color="royalblue")
ax_rd_twin.tick_params(axis="y", labelcolor="royalblue")
bg_err_angle, = ax_rd_twin.plot(t_anim, err_angle, color="royalblue", alpha=0.3)
op_err_r, = ax_rd.plot([], [], color="crimson", lw=2)
op_err_angle, = ax_rd_twin.plot([], [], color="royalblue", lw=2)
ax_rd.legend([bg_err_r, bg_err_angle], ["|Δr|", "|Δψ|"], loc="upper left", fontsize=8)

ax_d.set_title("phase diff per oscillator (full − learned)")
phase_diff_segs = np.dstack([np.broadcast_to(t_anim, (N, anim_frames)), diff])
ax_d.add_collection(LineCollection(
    phase_diff_segs, colors=ball_colors, alpha=0.5, linewidths=0.4,
))
ax_d.set_xlim(0, t_anim[-1])
ax_d.set_xlabel("time (s)")
ax_d.set_ylabel("Δθ (°)")
ax_d.set_ylim(-180, 180)
vline = ax_d.axvline(t_anim[0], color="crimson", lw=2)
err_txt = ax_d.text(0.02, 0.95, "", transform=ax_d.transAxes, va="top")


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
        f"mean |Δθ| = {np.mean(np.abs(diff[:, k])):.1f}°  |  "
        f"|Δr| = {err_r[k]:.3f}  |  |Δψ| = {err_angle[k]:.1f}°"
    )
    artists.append(err_txt)
    return artists


ani = FuncAnimation(
    fig, update, frames=anim_frames, interval=anim_interval_ms, blit=anim_blit,
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plots_dir = Path("plots")
plots_dir.mkdir(parents=True, exist_ok=True)
gif_path = plots_dir / f"learned_sparsification_n={N}_q={Q}.gif"
ani.save(gif_path, writer="pillow", fps=1000 // anim_interval_ms)
print(f"saved {gif_path}")
if SHOW:
    plt.show()
