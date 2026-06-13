"""Train a fixed-edge-budget reduced Kuramoto network to match full-graph dynamics.

Reference: a single fully-connected (FC) graph of N oscillators with random
omega ~ N(5, 0.5). We learn a sparse weighted adjacency matrix with exactly
Q * E_full edges (E_full = N(N-1)/2) that best reproduces the full-graph phase
trajectories, averaged over many random initial conditions.

Edge budget: each undirected edge has a weight w_e = softplus(raw_e) >= 0.
Forward pass keeps only the top `MAX_EDGES` edges by weight (straight-through
estimator), so the network is ALWAYS exactly at budget -- no soft penalty, no
train/eval mismatch from later thresholding.

Run:
    python train_sparse_kuramoto.py
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdiffeq import odeint

from network_plotter import compute_layout, plot_network

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
N = 64
K = 0.1 * N
Q = 0.25  # edge budget as a fraction of E_full = N(N-1)/2

T_END = 5.0
N_STEPS = 150
N_ICS = 128      # size of each training IC batch
REFRESH_EVERY = 10**9  # resample a fresh training IC batch every this many epochs (effectively never)
VAL_ICS = 64     # fixed held-out ICs used for model selection / early-stop checks
DISPLAY_ICS = 64  # fixed ICs used only for dashboard visualization

EPOCHS = 300
LR = 0.01
LR_MIN_FRAC = 0.05  # cosine-decay LR down to LR * LR_MIN_FRAC
TRAIN_METHOD = "euler"   # fast fixed-step integrator used during training
EVAL_METHOD = "dopri5"   # accurate adaptive integrator for the reference target

LOG_EVERY = 20
PLOT_DIR = Path("plots/sparse_train")

RUN_SEED = int.from_bytes(os.urandom(8), "big") % (2**63)


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------
def kuramoto_rhs(theta, omega, a, k):
    """theta: (..., N); a: (N, N) symmetric adjacency; returns dtheta/dt, shape (..., N)."""
    diff = theta.unsqueeze(-1) - theta.unsqueeze(-2)  # diff[..., i, j] = theta_i - theta_j
    coupling = (a * torch.sin(-diff)).sum(-1)
    degree = a.sum(-1).clamp_min(1e-12)
    return omega + k * coupling / degree


class KuramotoODE(torch.nn.Module):
    def __init__(self, omega, a, k):
        super().__init__()
        self.omega, self.a, self.k = omega, a, k

    def forward(self, t, theta):
        return kuramoto_rhs(theta, self.omega, self.a, self.k)


def integrate(theta0, omega, a, k, t, method):
    kw = {} if method == "euler" else {"rtol": 1e-7, "atol": 1e-9}
    return odeint(KuramotoODE(omega, a, k), theta0, t, method=method, **kw)


# ---------------------------------------------------------------------------
# budget-constrained adjacency parameterization
# ---------------------------------------------------------------------------
def adjacency_from_raw(raw, ei, ej, n, max_edges):
    """raw: (E,) unconstrained params -> (A, weight, mask).

    weight = softplus(raw) >= 0 for every possible edge.
    mask selects the top `max_edges` by weight (hard budget, exact every step).
    A is built with a straight-through estimator: forward value is the masked
    (hard) weight, but gradients flow to `weight` as if no masking occurred.
    """
    weight = torch.nn.functional.softplus(raw)
    with torch.no_grad():
        mask = torch.zeros_like(weight)
        mask[torch.topk(weight, max_edges).indices] = 1.0
    masked_weight = weight + (mask * weight - weight).detach()
    a = torch.zeros(n, n, dtype=raw.dtype)
    a[ei, ej] = masked_weight
    a[ej, ei] = masked_weight
    return a, weight, mask


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def save_dashboard(t, omega, a_full, a_learned, target, pred, epoch, loss, max_edges, out):
    """target, pred: (T, N_ICS, N)."""
    n_edges = int(np.count_nonzero(np.triu(a_learned, 1)))

    wrapped_err = np.degrees(np.angle(np.exp(1j * (pred - target))))
    mean_abs_err_per_ic = np.abs(wrapped_err).mean(axis=2)  # (T, B)
    mean_abs_err = mean_abs_err_per_ic.mean(axis=1)

    z_full = np.exp(1j * target).mean(axis=2)
    z_learned = np.exp(1j * pred).mean(axis=2)
    r_full, r_learned = np.abs(z_full), np.abs(z_learned)

    color_norm = (float(omega.min()), float(omega.max()))
    layout_kw = dict(alignment="vertical", iterations=150, seed=RUN_SEED)
    pos_full = compute_layout(omega, a_full, **layout_kw)
    pos_learned = compute_layout(omega, a_learned, **layout_kw)

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.15)
    ax_full, ax_learned = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_err, ax_r = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    edge_kw = dict(edge_width_power=0.5, min_linewidth=0.15)
    plot_network(omega, a_full, ax=ax_full, initial_pos=pos_full, iterations=0,
                  alignment="vertical", add_colorbar=False, color_norm=color_norm,
                  edge_thickness=0.15, edge_alpha=0.12, node_radius=None, **edge_kw)
    ax_full.set_title(f"full network ({a_full.shape[0]*(a_full.shape[0]-1)//2:,} edges)", fontsize=9)

    plot_network(omega, a_learned, ax=ax_learned, initial_pos=pos_learned, iterations=0,
                  alignment="vertical", add_colorbar=False, color_norm=color_norm,
                  edge_thickness=0.2, edge_alpha=0.55, node_radius=None, **edge_kw)
    ax_learned.set_title(f"learned sparse ({n_edges}/{max_edges} edges)", fontsize=9)

    B = target.shape[1]
    for b in range(B):
        ax_err.plot(t, mean_abs_err_per_ic[:, b], color="C0", alpha=0.2, linewidth=0.5)
    ax_err.plot(t, mean_abs_err, color="C0", linewidth=1.8, label="mean over ICs")
    ax_err.set_ylabel("|learned - full| (deg)")
    ax_err.set_ylim(bottom=0)
    ax_err.legend(fontsize=8)

    for b in range(B):
        ax_r.plot(t, r_full[:, b], color="0.4", alpha=0.15, linewidth=0.5)
        ax_r.plot(t, r_learned[:, b], color="C0", alpha=0.15, linewidth=0.5)
    ax_r.plot(t, r_full.mean(axis=1), color="0.3", linewidth=1.5, label="R full (mean)")
    ax_r.plot(t, r_learned.mean(axis=1), color="C0", linewidth=1.5, label="R learned (mean)")
    ax_r.set_ylabel("R(t)")
    ax_r.set_ylim(0, 1.05)
    ax_r.legend(loc="upper left", fontsize=8)

    title = "before training" if epoch < 0 else f"epoch {epoch}"
    loss_str = f" | circ loss {loss:.4e}" if loss is not None else ""
    fig.suptitle(f"{title}{loss_str} | {B} ICs | N={omega.shape[0]}", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_network_plot(omega, a_full, a_learned, epoch, loss, max_edges, out):
    n_edges = int(np.count_nonzero(np.triu(a_learned, 1)))
    color_norm = (float(omega.min()), float(omega.max()))
    layout_kw = dict(alignment="vertical", iterations=150, seed=RUN_SEED)
    pos_full = compute_layout(omega, a_full, **layout_kw)
    pos_learned = compute_layout(omega, a_learned, **layout_kw)

    fig, (ax_full, ax_learned) = plt.subplots(1, 2, figsize=(13, 6))
    edge_kw = dict(edge_width_power=0.5, min_linewidth=0.15)
    plot_network(omega, a_full, ax=ax_full, initial_pos=pos_full, iterations=0,
                  alignment="vertical", add_colorbar=False, color_norm=color_norm,
                  edge_thickness=0.15, edge_alpha=0.12, node_radius=None, **edge_kw)
    ax_full.set_title(f"full network ({a_full.shape[0]*(a_full.shape[0]-1)//2:,} edges)", fontsize=9)

    plot_network(omega, a_learned, ax=ax_learned, initial_pos=pos_learned, iterations=0,
                  alignment="vertical", add_colorbar=False, color_norm=color_norm,
                  edge_thickness=0.2, edge_alpha=0.55, node_radius=None, **edge_kw)
    ax_learned.set_title(f"learned sparse ({n_edges}/{max_edges} edges)", fontsize=9)

    title = "before training" if epoch < 0 else f"epoch {epoch}"
    loss_str = f" | circ loss {loss:.4e}" if loss is not None else ""
    fig.suptitle(f"{title}{loss_str} | N={omega.shape[0]}", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print(f"run seed: {RUN_SEED}")
    rng = np.random.default_rng(RUN_SEED)

    a_full = np.ones((N, N)) - np.eye(N)
    n_full_edges = N * (N - 1) // 2
    max_edges = int(Q * n_full_edges)
    print(f"N={N} | full edges={n_full_edges} | budget={max_edges} (Q={Q})")

    ei, ej = np.triu_indices(N, k=1)
    ei_t, ej_t = torch.tensor(ei), torch.tensor(ej)

    omega = rng.normal(5.0, 0.5, N)

    t = np.linspace(0, T_END, N_STEPS)
    omega_t = torch.tensor(omega, dtype=torch.float64)
    t_t = torch.tensor(t, dtype=torch.float64)
    a_full_t = torch.tensor(a_full, dtype=torch.float64)

    # fixed validation ICs (model selection) and display ICs (dashboard viz), independent of training stream
    val_rng = np.random.default_rng(RUN_SEED + 1)
    theta0_val = torch.tensor(val_rng.uniform(0, 2 * np.pi, (VAL_ICS, N)), dtype=torch.float64)
    disp_rng = np.random.default_rng(RUN_SEED + 2)
    theta0_disp = torch.tensor(disp_rng.uniform(0, 2 * np.pi, (DISPLAY_ICS, N)), dtype=torch.float64)
    train_rng = np.random.default_rng(RUN_SEED + 3)  # drives the per-epoch fresh training ICs

    print("integrating full-graph validation target (euler)...")
    target_val = integrate(theta0_val, omega_t, a_full_t, K, t_t, TRAIN_METHOD).detach()
    print("integrating full-graph display target (dopri5)...")
    target_disp = integrate(theta0_disp, omega_t, a_full_t, K, t_t, EVAL_METHOD).detach()

    # init: weight = softplus(raw); raw ~ N(0.5, 0.3) -> initial weights cluster near 1
    raw = torch.tensor(0.5 + 0.3 * rng.standard_normal(len(ei)), dtype=torch.float64)
    raw.requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * LR_MIN_FRAC)

    dashboard_out = PLOT_DIR / f"dashboard_n={N}_q={Q}.png"
    network_out = PLOT_DIR / f"reduced_network_n={N}_q={Q}.png"
    ckpt_path = PLOT_DIR / f"best_raw_n={N}_q={Q}.pt"

    with torch.no_grad():
        a_init, _, _ = adjacency_from_raw(raw, ei_t, ej_t, N, max_edges)
        pred_disp_init = integrate(theta0_disp, omega_t, a_init, K, t_t, TRAIN_METHOD).numpy()
    save_dashboard(t, omega, a_full, a_init.numpy(), target_disp.numpy(), pred_disp_init, -1, None, max_edges, dashboard_out)
    save_network_plot(omega, a_full, a_init.numpy(), -1, None, max_edges, network_out)

    best_val_loss = float("inf")
    best_raw = raw.detach().clone()
    best_epoch = -1

    theta0_train, target_train = None, None
    for epoch in range(EPOCHS):
        if epoch % REFRESH_EVERY == 0:
            theta0_train = torch.tensor(train_rng.uniform(0, 2 * np.pi, (N_ICS, N)), dtype=torch.float64)
            with torch.no_grad():
                target_train = integrate(theta0_train, omega_t, a_full_t, K, t_t, TRAIN_METHOD)

        opt.zero_grad()
        a, weight, mask = adjacency_from_raw(raw, ei_t, ej_t, N, max_edges)
        pred = integrate(theta0_train, omega_t, a, K, t_t, TRAIN_METHOD)
        loss = (1 - torch.cos(pred - target_train)).mean()
        loss.backward()
        opt.step()
        sched.step()

        if epoch % LOG_EVERY == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                n_active = int(mask.sum().item())
                pred_val = integrate(theta0_val, omega_t, a, K, t_t, TRAIN_METHOD)
                val_loss = (1 - torch.cos(pred_val - target_val)).mean().item()
                pred_disp = integrate(theta0_disp, omega_t, a, K, t_t, TRAIN_METHOD).numpy()
            print(f"epoch {epoch:4d} | train loss {loss.item():.4e} | val loss {val_loss:.4e} | "
                  f"active edges {n_active}/{max_edges} | lr {sched.get_last_lr()[0]:.4f}")
            save_dashboard(t, omega, a_full, a.detach().numpy(), target_disp.numpy(), pred_disp, epoch, val_loss, max_edges, dashboard_out)
            if epoch % (LOG_EVERY * 5) == 0:
                save_network_plot(omega, a_full, a.detach().numpy(), epoch, val_loss, max_edges, network_out)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_raw = raw.detach().clone()
                best_epoch = epoch
                torch.save({"raw": best_raw, "epoch": best_epoch, "loss": best_val_loss,
                            "seed": RUN_SEED, "omega": omega_t}, ckpt_path)

    with torch.no_grad():
        a_best, _, _ = adjacency_from_raw(best_raw, ei_t, ej_t, N, max_edges)
        pred_disp_best = integrate(theta0_disp, omega_t, a_best, K, t_t, TRAIN_METHOD).numpy()
    save_dashboard(t, omega, a_full, a_best.numpy(), target_disp.numpy(), pred_disp_best, best_epoch, best_val_loss, max_edges, dashboard_out)
    save_network_plot(omega, a_full, a_best.numpy(), best_epoch, best_val_loss, max_edges, network_out)
    print(f"best epoch {best_epoch} | val loss {best_val_loss:.4e}")
    print(f"saved checkpoint {ckpt_path}, dashboard {dashboard_out}, network plot {network_out}")


if __name__ == "__main__":
    main()
