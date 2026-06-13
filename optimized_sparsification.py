"""Learn a sparse adjacency matrix (fixed edge budget) to match full-network Kuramoto trajectories."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from torchdiffeq import odeint

N = 64
K = 0.1 * N
Q = 0.25  # edge budget = Q * (edges in full FC graph); set EDGE_BUDGET directly to override
EDGE_BUDGET = None
T_END = 5.0
N_STEPS = 200
ODE_METHOD = "dopri5"  # used for target generation and evaluation plots
ODE_RTOL = 1e-4
ODE_ATOL = 1e-4
TRAIN_ODE_METHOD = "euler"  # fast fixed-step for training; simpler backprop, no adaptive-step graph
EPOCHS = 2000
LR = 0.01
WARMUP_EPOCHS = 50  # linear LR warmup from 0 to LR
RUN_SEED = int.from_bytes(os.urandom(8), "big") % (2**63)
N_PLOT_SAMPLE = 100  # oscillators shown in diagnostic plot (of N total)
N_ICS = 100  # number of initial conditions in the training batch
NETWORK_PLOT_EVERY = 50  # refresh learned_sparsification_network.png every N epochs
NETWORK_PLOT_ITERATIONS = 150  # force-layout steps for network shape plot
NETWORK_ALIGNMENT = "vertical"  # "vertical", "horizontal", or None
NETWORK_EDGE_POWER = 0.5  # edge width ~ (weight/max)^power; 0.5 = sqrt, gentler than squared
NETWORK_MIN_LINEWIDTH = 0.15  # floor so weak edges stay visible
NETWORK_FULL_EDGE_ALPHA = 0.12
NETWORK_LEARNED_EDGE_ALPHA = 0.55
BUDGET_LAMBDA = 0.1  # weight on soft budget penalty: keeps expected L0 norm ≈ max_edges
HC_GAMMA = -0.1  # hard-concrete stretch interval lower bound
HC_ZETA = 1.1    # hard-concrete stretch interval upper bound
BETA_START = 1.0  # initial hard-concrete temperature: soft gates
BETA_END = 0.3    # final hard-concrete temperature: kept well above 0 so late gradients stay usable


def fc_adj(n):
    return np.ones((n, n), dtype=float) - np.eye(n)


def random_subgraph_adj(n, ei, ej, max_edges, rng):
    """Random subgraph baseline: max_edges edges (of the full graph) with unit weight."""
    idx = rng.choice(len(ei), size=max_edges, replace=False)
    a = np.zeros((n, n), dtype=float)
    a[ei[idx], ej[idx]] = 1.0
    a[ej[idx], ei[idx]] = 1.0
    return a


def ring_lattice_adj(n, k):
    """Degree-preserving baseline: k-regular ring lattice (k/2 nearest neighbors per side)."""
    a = np.zeros((n, n), dtype=float)
    idx = np.arange(n)
    for d in range(1, k // 2 + 1):
        a[idx, (idx + d) % n] = 1.0
        a[(idx + d) % n, idx] = 1.0
    return a


def kuramoto_np(t, theta, omega, a, k):
    ei, ej = np.where(np.triu(a, 1))
    w = a[ei, ej]
    coupling = np.zeros_like(theta)
    np.add.at(coupling, ei, w * np.sin(theta[ej] - theta[ei]))
    np.add.at(coupling, ej, w * np.sin(theta[ei] - theta[ej]))
    deg = np.maximum(a.sum(1), 1e-12)
    return omega + k * coupling / deg


def kuramoto_torch(theta, omega, a, k):
    # theta: (..., N) — works for single (N,) or batched (B, N) states
    diff = theta.unsqueeze(-1) - theta.unsqueeze(-2)  # (..., N, N): diff[...,i,j] = θ_i − θ_j
    coupling = (a * torch.sin(-diff)).sum(-1)  # (..., N)
    deg = a.sum(-1).clamp_min(1e-12)  # (N,)
    return omega + k * coupling / deg  # (..., N)


def integrate_np(theta0, omega, a, k, t):
    sol = solve_ivp(
        kuramoto_np, (t[0], t[-1]), theta0, args=(omega, a, k),
        t_eval=t, method="DOP853", rtol=1e-8, atol=1e-10,
    )
    return sol.y.T


class _KuramotoODE(nn.Module):
    def __init__(self, omega, a, k):
        super().__init__()
        self.omega = omega
        self.a = a
        self.k = k

    def forward(self, t, theta):
        return kuramoto_torch(theta, self.omega, self.a, self.k)


def integrate_torch(theta0, omega, a, k, t, method=None):
    m = method if method is not None else ODE_METHOD
    kw = {} if m == "euler" else {"rtol": ODE_RTOL, "atol": ODE_ATOL}
    return odeint(_KuramotoODE(omega, a, k), theta0, t, method=m, **kw)


def hard_concrete_gate(logits, beta=None, gamma=HC_GAMMA, zeta=HC_ZETA):
    """Hard-concrete gate (Louizos, Welling & Kingma, 2018).

    beta is the temperature: smaller beta -> closer to a true Bernoulli(sigmoid(logits)).
    beta=None gives the deterministic eval gate (no sampling).
    """
    if beta is not None:
        u = torch.zeros_like(logits).uniform_().clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + logits) / beta)
    else:
        s = torch.sigmoid(logits)
    s_bar = s * (zeta - gamma) + gamma
    return s_bar.clamp(0.0, 1.0)


def l0_expected(logits, beta, gamma=HC_GAMMA, zeta=HC_ZETA):
    """Expected number of active gates: sum_i P(gate_i > 0)."""
    return torch.sigmoid(logits - beta * float(np.log(-gamma / zeta))).sum()


def adjacency_from_logits(logits, ei, ej, n, beta=None, top_k=None):
    z = hard_concrete_gate(logits, beta)
    if top_k is not None:
        k = min(top_k, z.numel())
        _, top = torch.topk(logits, k)
        mask = torch.zeros_like(z)
        mask[top] = 1.0
        z = z * mask
    a = torch.zeros(n, n, dtype=logits.dtype, device=logits.device)
    a[ei, ej] = z
    a[ej, ei] = z
    return a, z


def init_logits(num_edges, rng, target_density=Q, gamma=HC_GAMMA, zeta=HC_ZETA):
    # Set logit so the deterministic eval gate ≈ target_density at init.
    target_s = float(np.clip((target_density - gamma) / (zeta - gamma), 1e-6, 1 - 1e-6))
    logit_center = float(np.log(target_s / (1 - target_s)))
    logits = torch.full((num_edges,), logit_center, dtype=torch.float64)
    logits += 0.1 * torch.tensor(rng.standard_normal(num_edges), dtype=torch.float64)
    return logits


def predict_trajectory(logits, ei_t, ej_t, theta0_t, omega_t, t_t, top_k=None):
    with torch.no_grad():
        a, _ = adjacency_from_logits(logits, ei_t, ej_t, N, beta=None, top_k=top_k)
        return integrate_torch(theta0_t, omega_t, a, K, t_t).numpy()


def save_plot(t, target, init_pred, pred, sample, epoch, budget, active, train_loss, out, n_total=N):
    n_show = len(sample)
    lw = 0.8 if n_show > 20 else 1.2
    alpha = 0.35 if n_show > 20 else 0.85
    cmap = plt.cm.viridis if n_show > 10 else plt.cm.tab10
    colors = cmap(np.linspace(0.05, 0.95, n_show))

    fig, (ax_init, ax_r, ax_learned) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    init_err = np.degrees(init_pred - target)
    for k, i in enumerate(sample):
        c = colors[k]
        ax_init.plot(t, init_err[:, i], color=c, alpha=0.5, linewidth=lw)
        ax_learned.plot(t, np.degrees(np.angle(np.exp(1j * (pred[:, i] - target[:, i])))), color=c, alpha=alpha, linewidth=lw)

    ax_init.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
    ax_init.set_ylabel("init − full (deg)")
    ax_init.set_title("fixed reference (before training)", fontsize=8, loc="left", pad=2)

    z_full = np.exp(1j * target).mean(axis=1)
    z_learned = np.exp(1j * pred).mean(axis=1)
    r_full = np.abs(z_full)
    r_learned = np.abs(z_learned)
    psi_full = np.degrees(np.unwrap(np.angle(z_full)))
    psi_learned = np.degrees(np.unwrap(np.angle(z_learned)))

    ax_r.plot(t, r_full, color="0.3", linewidth=1.2, label="R full")
    ax_r.plot(t, r_learned, color="C0", linewidth=1.2, label=f"R learned @ {budget}")
    ax_r.set_ylabel("R(t)")
    ax_r.set_ylim(0, 1.05)
    ax_r.legend(loc="upper left", fontsize=8)

    ax_psi = ax_r.twinx()
    ax_psi.plot(t, psi_full, color="0.3", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_psi.plot(t, psi_learned, color="C0", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_psi.set_ylabel("ψ(t) (deg)", rotation=-90, labelpad=15)

    ax_learned.axhline(0, color="0.5", linewidth=0.5, linestyle=":")
    ax_learned.set_ylabel("learned − full (deg)")
    ax_learned.set_xlabel("time")

    title = "before training" if epoch < 0 else f"epoch {epoch}"
    loss_str = f" | circ loss {train_loss:.3e}" if train_loss is not None else ""
    fig.suptitle(
        f"{title}{loss_str} | active {active}/{budget} | {n_show}/{n_total} oscs",
        fontsize=10,
    )
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_batch_plot(t, target_batch, init_batch, pred_batch, epoch, budget, active, train_loss, out, n_total=N):
    # All batch arrays: (T, B, N)
    B = target_batch.shape[1]

    wrapped_err = np.degrees(np.angle(np.exp(1j * (pred_batch - target_batch))))  # (T, B, N)
    mean_abs_err_per_ic = np.abs(wrapped_err).mean(axis=2)   # (T, B)
    mean_abs_err = mean_abs_err_per_ic.mean(axis=1)           # (T,)

    init_wrapped = np.degrees(np.angle(np.exp(1j * (init_batch - target_batch))))
    init_mean_abs_err = np.abs(init_wrapped).mean(axis=(1, 2))  # (T,)

    z_full_ic = np.exp(1j * target_batch).mean(axis=2)             # (T, B) complex
    z_learned_ic = np.exp(1j * pred_batch).mean(axis=2)            # (T, B) complex
    r_full_ic = np.abs(z_full_ic)                                   # (T, B)
    r_learned_ic = np.abs(z_learned_ic)                             # (T, B)
    psi_full_ic = np.degrees(np.unwrap(np.angle(z_full_ic), axis=0))    # (T, B)
    psi_learned_ic = np.degrees(np.unwrap(np.angle(z_learned_ic), axis=0))  # (T, B)

    fig, (ax_err, ax_r) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for b in range(B):
        ax_err.plot(t, mean_abs_err_per_ic[:, b], color="C0", alpha=0.25, linewidth=0.6)
    ax_err.plot(t, mean_abs_err, color="C0", linewidth=1.8, label="learned (mean over ICs)")
    ax_err.plot(t, init_mean_abs_err, color="0.5", linewidth=1.2, linestyle="--", label="init (mean over ICs)")
    ax_err.set_ylabel("|learned − full| (deg)")
    ax_err.set_ylim(bottom=0)
    ax_err.legend(fontsize=8)

    for b in range(B):
        ax_r.plot(t, r_full_ic[:, b], color="0.4", alpha=0.15, linewidth=0.5)
        ax_r.plot(t, r_learned_ic[:, b], color="C0", alpha=0.15, linewidth=0.5)
    ax_r.plot(t, r_full_ic.mean(axis=1), color="0.3", linewidth=1.5, label="R full (mean)")
    ax_r.plot(t, r_learned_ic.mean(axis=1), color="C0", linewidth=1.5, label="R learned (mean)")
    ax_r.set_ylabel("R(t)")
    ax_r.set_ylim(0, 1.05)
    ax_r.legend(loc="upper left", fontsize=8)

    ax_psi = ax_r.twinx()
    for b in range(B):
        ax_psi.plot(t, psi_full_ic[:, b], color="0.4", alpha=0.15, linewidth=0.5, linestyle="--")
        ax_psi.plot(t, psi_learned_ic[:, b], color="C0", alpha=0.15, linewidth=0.5, linestyle="--")
    ax_psi.plot(t, psi_full_ic.mean(axis=1), color="0.3", linewidth=1.2, linestyle="--", label="ψ full (mean)")
    ax_psi.plot(t, psi_learned_ic.mean(axis=1), color="C0", linewidth=1.2, linestyle="--", label="ψ learned (mean)")
    ax_psi.set_ylabel("ψ(t) (deg)", rotation=-90, labelpad=15)
    ax_psi.legend(loc="lower right", fontsize=8)

    ax_r.set_xlabel("time")

    title = "before training" if epoch < 0 else f"epoch {epoch}"
    loss_str = f" | circ loss {train_loss:.3e}" if train_loss is not None else ""
    fig.suptitle(f"{title}{loss_str} | active {active}/{budget} | {B} ICs | {n_total} oscs", fontsize=10)
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_network_plot(omega, a_full, a_learned, epoch, train_loss, out):
    from network_plotter import compute_layout, plot_network

    n = len(omega)
    color_norm = (float(omega.min()), float(omega.max()))
    align = NETWORK_ALIGNMENT if NETWORK_ALIGNMENT else None

    layout_kw = dict(
        alignment=align,
        iterations=NETWORK_PLOT_ITERATIONS,
        seed=RUN_SEED,
    )
    edge_kw = dict(
        edge_width_power=NETWORK_EDGE_POWER,
        min_linewidth=NETWORK_MIN_LINEWIDTH,
    )
    pos_full = compute_layout(omega, a_full, **layout_kw)
    pos_learned = compute_layout(omega, a_learned, **layout_kw)

    fig, (ax_full, ax_learned) = plt.subplots(1, 2, figsize=(14, 6))
    title = "before training" if epoch < 0 else f"epoch {epoch}"
    loss_str = f" | circ loss {train_loss:.3e}" if train_loss is not None else ""
    fig.suptitle(f"{title}{loss_str} | {n} nodes", fontsize=10)

    plot_network(
        omega, a_full, ax=ax_full, initial_pos=pos_full, iterations=0,
        alignment=align, add_colorbar=False, color_norm=color_norm,
        edge_thickness=0.15, edge_alpha=NETWORK_FULL_EDGE_ALPHA, node_radius=None, **edge_kw,
    )
    n_full_edges = int(np.count_nonzero(np.triu(a_full, 1)))
    ax_full.set_title(f"full network ({n_full_edges:,} edges)", fontsize=9)

    plot_network(
        omega, a_learned, ax=ax_learned, initial_pos=pos_learned, iterations=0,
        alignment=align, add_colorbar=False, color_norm=color_norm,
        edge_thickness=0.2, edge_alpha=NETWORK_LEARNED_EDGE_ALPHA, node_radius=None, **edge_kw,
    )
    n_edges = int(np.count_nonzero(np.triu(a_learned, 1)))
    ax_learned.set_title(f"learned sparse ({n_edges:,} edges)", fontsize=9)

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdBu, norm=plt.Normalize(*color_norm),
    )
    sm.set_array([])
    fig.subplots_adjust(top=0.88, left=0.04, right=0.84, wspace=0.18)
    cbar_ax = fig.add_axes([0.87, 0.14, 0.02, 0.72])
    fig.colorbar(sm, cax=cbar_ax, label=r"$\omega_i$")
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_training_dashboard(
    t,
    omega,
    a_full,
    a_learned,
    target_batch,
    init_batch,
    pred_batch,
    epoch,
    budget,
    active,
    train_loss,
    out,
    n_total=N,
):
    """stacked network + batch metrics in one png for live training monitoring."""
    from network_plotter import compute_layout, plot_network

    B = target_batch.shape[1]
    n = len(omega)
    color_norm = (float(omega.min()), float(omega.max()))
    align = NETWORK_ALIGNMENT if NETWORK_ALIGNMENT else None
    layout_kw = dict(
        alignment=align,
        iterations=NETWORK_PLOT_ITERATIONS,
        seed=RUN_SEED,
    )
    edge_kw = dict(
        edge_width_power=NETWORK_EDGE_POWER,
        min_linewidth=NETWORK_MIN_LINEWIDTH,
    )
    pos_full = compute_layout(omega, a_full, **layout_kw)
    pos_learned = compute_layout(omega, a_learned, **layout_kw)

    wrapped_err = np.degrees(np.angle(np.exp(1j * (pred_batch - target_batch))))
    mean_abs_err_per_ic = np.abs(wrapped_err).mean(axis=2)
    mean_abs_err = mean_abs_err_per_ic.mean(axis=1)
    init_wrapped = np.degrees(np.angle(np.exp(1j * (init_batch - target_batch))))
    init_mean_abs_err = np.abs(init_wrapped).mean(axis=(1, 2))

    z_full_ic = np.exp(1j * target_batch).mean(axis=2)
    z_learned_ic = np.exp(1j * pred_batch).mean(axis=2)
    r_full_ic = np.abs(z_full_ic)
    r_learned_ic = np.abs(z_learned_ic)
    psi_full_ic = np.degrees(np.unwrap(np.angle(z_full_ic), axis=0))
    psi_learned_ic = np.degrees(np.unwrap(np.angle(z_learned_ic), axis=0))

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 0.9, 0.9], hspace=0.38, wspace=0.16)
    ax_full = fig.add_subplot(gs[0, 0])
    ax_learned = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, :])
    ax_r = fig.add_subplot(gs[2, :])

    plot_network(
        omega, a_full, ax=ax_full, initial_pos=pos_full, iterations=0,
        alignment=align, add_colorbar=False, color_norm=color_norm,
        edge_thickness=0.15, edge_alpha=NETWORK_FULL_EDGE_ALPHA, node_radius=None, **edge_kw,
    )
    n_full_edges = int(np.count_nonzero(np.triu(a_full, 1)))
    ax_full.set_title(f"full network ({n_full_edges:,} edges)", fontsize=9)

    plot_network(
        omega, a_learned, ax=ax_learned, initial_pos=pos_learned, iterations=0,
        alignment=align, add_colorbar=False, color_norm=color_norm,
        edge_thickness=0.2, edge_alpha=NETWORK_LEARNED_EDGE_ALPHA, node_radius=None, **edge_kw,
    )
    n_edges = int(np.count_nonzero(np.triu(a_learned, 1)))
    ax_learned.set_title(f"learned sparse ({n_edges:,} edges)", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu, norm=plt.Normalize(*color_norm))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.72, 0.015, 0.16])
    fig.colorbar(sm, cax=cbar_ax, label=r"$\omega_i$")

    for b in range(B):
        ax_err.plot(t, mean_abs_err_per_ic[:, b], color="C0", alpha=0.25, linewidth=0.6)
    ax_err.plot(t, mean_abs_err, color="C0", linewidth=1.8, label="learned (mean over ICs)")
    ax_err.plot(t, init_mean_abs_err, color="0.5", linewidth=1.2, linestyle="--", label="init (mean over ICs)")
    ax_err.set_ylabel("|learned − full| (deg)")
    ax_err.set_ylim(bottom=0)
    ax_err.legend(fontsize=8, loc="upper right")

    for b in range(B):
        ax_r.plot(t, r_full_ic[:, b], color="0.4", alpha=0.15, linewidth=0.5)
        ax_r.plot(t, r_learned_ic[:, b], color="C0", alpha=0.15, linewidth=0.5)
    ax_r.plot(t, r_full_ic.mean(axis=1), color="0.3", linewidth=1.5, label="R full (mean)")
    ax_r.plot(t, r_learned_ic.mean(axis=1), color="C0", linewidth=1.5, label="R learned (mean)")
    ax_r.set_ylabel("R(t)")
    ax_r.set_ylim(0, 1.05)
    ax_r.legend(loc="upper left", fontsize=8)

    ax_psi = ax_r.twinx()
    for b in range(B):
        ax_psi.plot(t, psi_full_ic[:, b], color="0.4", alpha=0.15, linewidth=0.5, linestyle="--")
        ax_psi.plot(t, psi_learned_ic[:, b], color="C0", alpha=0.15, linewidth=0.5, linestyle="--")
    ax_psi.plot(t, psi_full_ic.mean(axis=1), color="0.3", linewidth=1.2, linestyle="--", label="ψ full (mean)")
    ax_psi.plot(t, psi_learned_ic.mean(axis=1), color="C0", linewidth=1.2, linestyle="--", label="ψ learned (mean)")
    ax_psi.set_ylabel("ψ(t) (deg)", rotation=-90, labelpad=15)
    ax_psi.legend(loc="lower right", fontsize=8)
    ax_r.set_xlabel("time")

    title = "before training" if epoch < 0 else f"epoch {epoch}"
    loss_str = f" | circ loss {train_loss:.3e}" if train_loss is not None else ""
    fig.suptitle(
        f"{title}{loss_str} | active {active}/{budget} | {B} ICs | {n_total} oscs",
        fontsize=10,
    )
    out.parent.mkdir(exist_ok=True)
    fig.subplots_adjust(top=0.94, left=0.06, right=0.90, bottom=0.05)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    print(f"run seed: {RUN_SEED}")
    rng = np.random.default_rng(RUN_SEED)
    a_full = fc_adj(N)
    n_full_edges = N * (N - 1) // 2
    max_edges = EDGE_BUDGET if EDGE_BUDGET is not None else int(Q * n_full_edges)
    print(f"full edges: {n_full_edges} | edge budget: {max_edges} (Q={Q})")

    ei, ej = np.triu_indices(N, k=1)
    ei_t = torch.tensor(ei)
    ej_t = torch.tensor(ej)
    omega = rng.normal(5.0, 0.5, N)
    # IC batch: (N_ICS, N); IC 0 is the fixed reference for single-IC diagnostic plot
    theta0_batch = rng.uniform(0, 2 * np.pi, (N_ICS, N))

    t = np.linspace(0, T_END, N_STEPS)
    omega_t = torch.tensor(omega, dtype=torch.float64)
    theta0_batch_t = torch.tensor(theta0_batch, dtype=torch.float64)  # (N_ICS, N)
    t_t = torch.tensor(t, dtype=torch.float64)
    a_full_t = torch.tensor(a_full, dtype=torch.float64)

    # dopri5 targets: accurate, used for plots and evaluation only
    target_batch_t = integrate_torch(theta0_batch_t, omega_t, a_full_t, K, t_t).detach()  # (T, N_ICS, N)
    target_batch = target_batch_t.numpy().copy()
    target_ref = target_batch[:, 0, :]  # (T, N) — reference IC for per-IC plot
    # Euler targets: same ICs as plots, consistent with training integrator
    train_target_batch_t = integrate_torch(
        theta0_batch_t, omega_t, a_full_t, K, t_t, method=TRAIN_ODE_METHOD,
    ).detach()  # (T, N_ICS, N)

    out = Path("plots") / "learned_sparsification.png"
    batch_out = Path("plots") / "learned_sparsification_batch.png"
    net_out = Path("plots") / "learned_sparsification_network.png"
    dashboard_out = Path("plots") / "learned_sparsification_dashboard.png"
    sample = np.random.default_rng(RUN_SEED + 1).choice(N, size=min(N_PLOT_SAMPLE, N), replace=False)

    # Baselines: how good is a sparse network with no learning at all?
    rng_base = np.random.default_rng(RUN_SEED + 2)
    a_rand = random_subgraph_adj(N, ei, ej, max_edges, rng_base)
    a_rand_t = torch.tensor(a_rand, dtype=torch.float64)
    rand_pred_t = integrate_torch(theta0_batch_t, omega_t, a_rand_t, K, t_t, method=TRAIN_ODE_METHOD)
    rand_loss = (1 - torch.cos(rand_pred_t - train_target_batch_t)).mean().item()

    k_ring = max(2, 2 * round(max_edges / N))
    a_ring = ring_lattice_adj(N, k_ring)
    a_ring_t = torch.tensor(a_ring, dtype=torch.float64)
    ring_pred_t = integrate_torch(theta0_batch_t, omega_t, a_ring_t, K, t_t, method=TRAIN_ODE_METHOD)
    ring_loss = (1 - torch.cos(ring_pred_t - train_target_batch_t)).mean().item()
    n_ring_edges = int(np.count_nonzero(np.triu(a_ring, 1)))
    print(f"baseline | random {max_edges}-edge subgraph: circ loss {rand_loss:.4e}")
    print(f"baseline | {k_ring}-regular ring lattice ({n_ring_edges} edges): circ loss {ring_loss:.4e}")

    init_logits_frozen = init_logits(len(ei), rng)
    logits = init_logits_frozen.clone()
    with torch.no_grad():
        a_init, z_init = adjacency_from_logits(init_logits_frozen, ei_t, ej_t, N, beta=None, top_k=max_edges)
        init_active = int((z_init > 0).sum().item())
        init_pred_batch = integrate_torch(theta0_batch_t, omega_t, a_init, K, t_t).numpy().copy()  # (T, N_ICS, N)
    init_pred_ref = init_pred_batch[:, 0, :]  # (T, N)

    save_plot(t, target_ref, init_pred_ref, init_pred_ref, sample, -1, max_edges, init_active, None, out)
    save_batch_plot(t, target_batch, init_pred_batch, init_pred_batch, -1, max_edges, init_active, None, batch_out)
    save_network_plot(omega, a_full, a_init.numpy(), -1, None, net_out)
    save_training_dashboard(
        t, omega, a_full, a_init.numpy(), target_batch, init_pred_batch, init_pred_batch,
        -1, max_edges, init_active, None, dashboard_out,
    )
    print(f"init state | active edges {init_active}/{max_edges} | {N_ICS} ICs")

    logits.requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=LR)
    best_loss = float("inf")
    best_logits = logits.detach().clone()
    best_epoch = -1
    ckpt_path = Path("plots") / "best_sparsification_logits.pt"

    for epoch in range(EPOCHS):
        if epoch < WARMUP_EPOCHS:
            for g in opt.param_groups:
                g["lr"] = LR * (epoch + 1) / WARMUP_EPOCHS
        opt.zero_grad()
        beta = BETA_START * (BETA_END / BETA_START) ** (epoch / max(1, EPOCHS - 1))
        a, z = adjacency_from_logits(logits, ei_t, ej_t, N, beta=beta)
        pred_t = integrate_torch(theta0_batch_t, omega_t, a, K, t_t, method=TRAIN_ODE_METHOD)  # (T, N_ICS, N)
        circ_loss = (1 - torch.cos(pred_t - train_target_batch_t)).mean()
        budget_penalty = BUDGET_LAMBDA * (l0_expected(logits, beta) - max_edges) ** 2 / max_edges
        loss = circ_loss + budget_penalty
        loss.backward()
        opt.step()
        if (epoch > 0 and epoch % 25 == 0) or epoch == EPOCHS - 1:
            with torch.no_grad():
                a_plot, z_plot = adjacency_from_logits(logits, ei_t, ej_t, N, beta=None, top_k=max_edges)
                pred_plot_t = integrate_torch(theta0_batch_t, omega_t, a_plot, K, t_t, method=TRAIN_ODE_METHOD)
                active = int((z_plot > 0).sum().item())
                train_loss = circ_loss.item()
                eval_loss = (1 - torch.cos(pred_plot_t - train_target_batch_t)).mean().item()
                pred_np = pred_plot_t.numpy()   # (T, N_ICS, N)
                pred_ref = pred_np[:, 0, :]
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_logits = logits.detach().clone()
                best_epoch = epoch
                torch.save(
                    {
                        "logits": best_logits,
                        "epoch": best_epoch,
                        "loss": best_loss,
                        "seed": RUN_SEED,
                        "omega": omega_t,
                        "theta0_batch": theta0_batch_t,
                    },
                    ckpt_path,
                )
            print(f"epoch {epoch:4d} | train loss {train_loss:.4e} | eval loss {eval_loss:.4e} | beta {beta:.3f} | active {active}/{max_edges}")
            save_plot(t, target_ref, init_pred_ref, pred_ref, sample, epoch, max_edges, active, train_loss, out)
            save_batch_plot(t, target_batch, init_pred_batch, pred_np, epoch, max_edges, active, train_loss, batch_out)
            save_training_dashboard(
                t, omega, a_full, a_plot.numpy(), target_batch, init_pred_batch, pred_np,
                epoch, max_edges, active, train_loss, dashboard_out,
            )
            if epoch % NETWORK_PLOT_EVERY == 0:
                save_network_plot(omega, a_full, a_plot.numpy(), epoch, train_loss, net_out)

    with torch.no_grad():
        a_best, z_best = adjacency_from_logits(best_logits, ei_t, ej_t, N, beta=None, top_k=max_edges)
        n_learned = int((z_best > 0).sum().item())
    best_pred_batch = predict_trajectory(best_logits, ei_t, ej_t, theta0_batch_t, omega_t, t_t, top_k=max_edges)  # (T, N_ICS, N)
    best_pred_ref = best_pred_batch[:, 0, :]
    save_plot(t, target_ref, init_pred_ref, best_pred_ref, sample, best_epoch, max_edges, n_learned, best_loss, out)
    save_batch_plot(t, target_batch, init_pred_batch, best_pred_batch, best_epoch, max_edges, n_learned, best_loss, batch_out)
    save_network_plot(omega, a_full, a_best.numpy(), best_epoch, best_loss, net_out)
    save_training_dashboard(
        t, omega, a_full, a_best.numpy(), target_batch, init_pred_batch, best_pred_batch,
        best_epoch, max_edges, n_learned, best_loss, dashboard_out,
    )
    print(f"best epoch {best_epoch} | circ loss {best_loss:.4e} | active edges {n_learned}")
    print(f"saved best logits {ckpt_path}, plot {out}, batch plot {batch_out}, network plot {net_out}, dashboard {dashboard_out}")

    from compare_learned_sparsification import run_comparison
    run_comparison(ckpt_path, no_show=True)


def regenerate_plot():
    ckpt_path = Path("plots/best_sparsification_logits.pt")
    if not ckpt_path.exists():
        print(f"no checkpoint at {ckpt_path}; run training first")
        return False
    ckpt = torch.load(ckpt_path, weights_only=False)
    seed = int(ckpt["seed"])
    print(f"regenerating plot from checkpoint (epoch {ckpt['epoch']}, seed {seed})")
    rng = np.random.default_rng(seed)
    a_full = fc_adj(N)
    max_edges = EDGE_BUDGET if EDGE_BUDGET is not None else int(Q * N * (N - 1) // 2)
    ei, ej = np.triu_indices(N, k=1)
    ei_t = torch.tensor(ei)
    ej_t = torch.tensor(ej)
    omega = rng.normal(5.0, 0.5, N)
    if "theta0_batch" in ckpt:
        theta0_batch_t = ckpt["theta0_batch"]
    else:
        theta0_batch = rng.uniform(0, 2 * np.pi, (N_ICS, N))
        theta0_batch_t = torch.tensor(theta0_batch, dtype=torch.float64)
    t = np.linspace(0, T_END, N_STEPS)
    omega_t = torch.tensor(omega, dtype=torch.float64)
    t_t = torch.tensor(t, dtype=torch.float64)
    a_full_t = torch.tensor(a_full, dtype=torch.float64)
    target_batch_t = integrate_torch(theta0_batch_t, omega_t, a_full_t, K, t_t)
    target_batch = target_batch_t.numpy().copy()
    target_ref = target_batch[:, 0, :]
    logits = ckpt["logits"]
    init_logits_frozen = init_logits(len(ei), rng)
    with torch.no_grad():
        a_init, z_init = adjacency_from_logits(init_logits_frozen, ei_t, ej_t, N, beta=None, top_k=max_edges)
        init_active = int((z_init > 0).sum().item())
        init_pred_batch = integrate_torch(theta0_batch_t, omega_t, a_init, K, t_t).numpy().copy()
        a_best, z_best = adjacency_from_logits(logits, ei_t, ej_t, N, beta=None, top_k=max_edges)
        active = int((z_best > 0).sum().item())
    init_pred_ref = init_pred_batch[:, 0, :]
    best_pred_batch = predict_trajectory(logits, ei_t, ej_t, theta0_batch_t, omega_t, t_t, top_k=max_edges)
    best_pred_ref = best_pred_batch[:, 0, :]
    sample = np.random.default_rng(seed + 1).choice(N, size=min(N_PLOT_SAMPLE, N), replace=False)
    ckpt_loss = float(ckpt.get("loss", ckpt.get("mse", 0.0)))
    out = Path("plots") / "learned_sparsification.png"
    batch_out = Path("plots") / "learned_sparsification_batch.png"
    dashboard_out = Path("plots") / "learned_sparsification_dashboard.png"
    save_plot(t, target_ref, init_pred_ref, best_pred_ref, sample, int(ckpt["epoch"]), max_edges, active, ckpt_loss, out)
    save_batch_plot(t, target_batch, init_pred_batch, best_pred_batch, int(ckpt["epoch"]), max_edges, active, ckpt_loss, batch_out)
    save_training_dashboard(
        t, omega, a_full, a_best.numpy(), target_batch, init_pred_batch, best_pred_batch,
        int(ckpt["epoch"]), max_edges, active, ckpt_loss, dashboard_out,
    )
    print(f"saved {out}, {batch_out}, and {dashboard_out}")
    return True


def regenerate_network_plot():
    ckpt_path = Path("plots") / "best_sparsification_logits.pt"
    if not ckpt_path.exists():
        print(f"no checkpoint at {ckpt_path}; wait until epoch 25 (first log step)")
        return False
    ckpt = torch.load(ckpt_path, weights_only=False)
    seed = int(ckpt["seed"])
    epoch = int(ckpt["epoch"])
    loss = float(ckpt.get("loss", ckpt.get("mse", 0.0)))
    print(f"regenerating network plot from checkpoint (epoch {epoch}, seed {seed})")
    max_edges = EDGE_BUDGET if EDGE_BUDGET is not None else int(Q * N * (N - 1) // 2)
    ei, ej = np.triu_indices(N, k=1)
    ei_t = torch.tensor(ei)
    ej_t = torch.tensor(ej)
    logits = ckpt["logits"]
    if "omega" in ckpt:
        omega = ckpt["omega"].detach().cpu().numpy()
    else:
        omega = np.random.default_rng(seed).normal(5.0, 0.5, N)
    with torch.no_grad():
        a_best, _ = adjacency_from_logits(logits, ei_t, ej_t, N, beta=None, top_k=max_edges)
    out = Path("plots") / "learned_sparsification_network.png"
    save_network_plot(omega, fc_adj(N), a_best.numpy(), epoch, loss, out)
    print(f"saved {out}")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--plot-only":
        regenerate_plot()
    elif len(sys.argv) > 1 and sys.argv[1] == "--network-only":
        regenerate_network_plot()
    else:
        main()
