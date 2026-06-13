"""Compare the learned sparse network against ER, weight-based, and random
sparsification baselines, on both in-distribution (training) ICs and fresh
out-of-distribution ICs.

Run after train_sparse_kuramoto.py has produced a checkpoint:
    python compare_sparsification_methods.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import train_sparse_kuramoto as tsk
from network_plotter import compute_layout, plot_network

CKPT_PATH = tsk.PLOT_DIR / f"best_raw_n={tsk.N}_q={tsk.Q}.pt"
N_OOD_ICS = 64  # fresh ICs not seen during training
OUT_DIR = tsk.PLOT_DIR


# ---------------------------------------------------------------------------
# baseline sparsifications (numpy, operate on the full FC adjacency)
# ---------------------------------------------------------------------------
def effective_resistance_sparsification(a, q, rng):
    n = a.shape[0]
    degree_matrix = np.diag(a.sum(1))
    laplacian = degree_matrix - a
    laplacian_pinv = np.linalg.pinv(laplacian)
    diag = np.diag(laplacian_pinv)
    eff_res = diag[:, None] + diag[None, :] - 2 * laplacian_pinv
    ei, ej = np.where(np.triu(a, 1))
    we = a[ei, ej].astype(float)
    re = we * eff_res[ei, ej]
    pe = re / re.sum()
    s = max(1, int(q * len(ei)))
    out = np.zeros((n, n))
    for k in rng.choice(len(ei), s, replace=True, p=pe):
        w = we[k] / (s * pe[k])
        out[ei[k], ej[k]] += w
        out[ej[k], ei[k]] += w
    return out


def weight_based_sparsification(a, q, rng):
    n = a.shape[0]
    ei, ej = np.where(np.triu(a, 1))
    we = a[ei, ej].astype(float)
    pe = we / we.sum()
    s = max(1, int(q * len(ei)))
    out = np.zeros((n, n))
    for k in rng.choice(len(ei), s, replace=True, p=pe):
        w = we[k] / (s * pe[k])
        out[ei[k], ej[k]] += w
        out[ej[k], ei[k]] += w
    return out


def random_sparsification(a, max_edges, rng):
    n = a.shape[0]
    ei, ej = np.where(np.triu(a, 1))
    idx = rng.choice(len(ei), size=max_edges, replace=False)
    out = np.zeros((n, n))
    out[ei[idx], ej[idx]] = a[ei[idx], ej[idx]]
    out[ej[idx], ei[idx]] = a[ej[idx], ei[idx]]
    return out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def circ_loss_curve(pred, target):
    """mean circular error per time step, averaged over ICs and oscillators. pred/target: (T,B,N)."""
    return (1 - np.cos(pred - target)).mean(axis=(1, 2))


def main():
    if not CKPT_PATH.exists():
        print(f"no checkpoint at {CKPT_PATH}; run train_sparse_kuramoto.py first")
        return
    ckpt = torch.load(CKPT_PATH, weights_only=False)
    seed = int(ckpt["seed"])
    epoch = int(ckpt["epoch"])
    print(f"loaded checkpoint: epoch {epoch}, loss {ckpt['loss']:.4e}, seed {seed}")

    N = tsk.N
    a_full = np.ones((N, N)) - np.eye(N)
    n_full_edges = N * (N - 1) // 2
    max_edges = int(tsk.Q * n_full_edges)
    ei, ej = np.triu_indices(N, k=1)
    ei_t, ej_t = torch.tensor(ei), torch.tensor(ej)

    omega_t = ckpt["omega"]
    omega = omega_t.numpy()

    # in-distribution = the same fixed validation ICs used for model selection during training
    val_rng = np.random.default_rng(seed + 1)
    theta0_id_t = torch.tensor(val_rng.uniform(0, 2 * np.pi, (tsk.VAL_ICS, N)), dtype=torch.float64)

    rng = np.random.default_rng(seed + 1000)  # independent of training rng
    theta0_ood = rng.uniform(0, 2 * np.pi, (N_OOD_ICS, N))
    theta0_ood_t = torch.tensor(theta0_ood, dtype=torch.float64)

    t = np.linspace(0, tsk.T_END, tsk.N_STEPS)
    t_t = torch.tensor(t, dtype=torch.float64)
    a_full_t = torch.tensor(a_full, dtype=torch.float64)

    # learned sparse adjacency
    with torch.no_grad():
        a_learned_t, _, _ = tsk.adjacency_from_raw(ckpt["raw"], ei_t, ej_t, N, max_edges)
    a_learned = a_learned_t.numpy()

    # baselines, all at the same edge budget
    base_rng = np.random.default_rng(seed + 2000)
    a_er = effective_resistance_sparsification(a_full, tsk.Q, base_rng)
    a_weight = weight_based_sparsification(a_full, tsk.Q, base_rng)
    a_random = random_sparsification(a_full, max_edges, base_rng)

    networks = {
        "learned": a_learned,
        "ER": a_er,
        "weight-based": a_weight,
        "random": a_random,
    }

    results = {}
    for name, a in networks.items():
        a_t = torch.tensor(a, dtype=torch.float64)
        n_edges = int(np.count_nonzero(np.triu(a, 1)))
        with torch.no_grad():
            pred_id = tsk.integrate(theta0_id_t, omega_t, a_t, tsk.K, t_t, tsk.EVAL_METHOD).numpy()
            pred_ood = tsk.integrate(theta0_ood_t, omega_t, a_t, tsk.K, t_t, tsk.EVAL_METHOD).numpy()
        results[name] = dict(a=a, n_edges=n_edges, pred_id=pred_id, pred_ood=pred_ood)
        print(f"{name:14s} | edges {n_edges:4d}")

    # reference (full graph) trajectories
    with torch.no_grad():
        target_id = tsk.integrate(theta0_id_t, omega_t, a_full_t, tsk.K, t_t, tsk.EVAL_METHOD).numpy()
        target_ood = tsk.integrate(theta0_ood_t, omega_t, a_full_t, tsk.K, t_t, tsk.EVAL_METHOD).numpy()

    for name, r in results.items():
        r["loss_id"] = circ_loss_curve(r["pred_id"], target_id)
        r["loss_ood"] = circ_loss_curve(r["pred_ood"], target_ood)
        r["mean_loss_id"] = float(r["loss_id"].mean())
        r["mean_loss_ood"] = float(r["loss_ood"].mean())
        print(f"{name:14s} | mean circ loss  ID {r['mean_loss_id']:.4e}  |  OOD {r['mean_loss_ood']:.4e}")

    # -------------------------------------------------------------- plotting
    colors = {"learned": "C0", "ER": "C1", "weight-based": "C2", "random": "C3"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for name, r in results.items():
        ax.plot(t, r["loss_id"], color=colors[name], label=f"{name} ({r['n_edges']} edges)")
    ax.set_title(f"in-distribution ICs (n={theta0_id_t.shape[0]})")
    ax.set_ylabel("circ loss (1-cos)")
    ax.set_xlabel("time")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for name, r in results.items():
        ax.plot(t, r["loss_ood"], color=colors[name], label=name)
    ax.set_title(f"out-of-distribution ICs (n={N_OOD_ICS})")
    ax.set_xlabel("time")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, [results[n]["mean_loss_id"] for n in names], width, label="ID", color="C0", alpha=0.8)
    ax.bar(x + width / 2, [results[n]["mean_loss_ood"] for n in names], width, label="OOD", color="C1", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("mean circ loss (time-averaged)")
    ax.set_title("overall comparison")
    ax.legend()

    # learned-network visualization
    ax = axes[1, 1]
    color_norm = (float(omega.min()), float(omega.max()))
    pos = compute_layout(omega, a_learned, alignment="vertical", iterations=150, seed=seed)
    plot_network(omega, a_learned, ax=ax, initial_pos=pos, iterations=0,
                  alignment="vertical", color_norm=color_norm,
                  edge_thickness=0.2, edge_alpha=0.55, edge_width_power=0.5, min_linewidth=0.15)
    ax.set_title(f"learned network ({results['learned']['n_edges']} edges)")

    fig.suptitle(f"sparsification comparison | N={N} | Q={tsk.Q} | epoch {epoch} | train loss {ckpt['loss']:.3e}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / f"comparison_n={N}_q={tsk.Q}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
