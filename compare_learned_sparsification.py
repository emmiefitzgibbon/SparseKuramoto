"""
Compare learned (Gumbel) vs learned_simple (sigmoid) vs ER / weight-based / random / freq-based.

Each panel uses its own ω, training ICs, and test ICs (seed+5000).

    python compare_learned_sparsification.py
    python compare_learned_sparsification.py plots/best_sparsification_logits.pt \\
        --simple-checkpoint plots/best_sparsification_logits_simple.pt
    python compare_learned_sparsification.py --simple --no-show
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from optimized_sparsification import (
    N,
    K,
    Q,
    EDGE_BUDGET,
    T_END,
    N_STEPS,
    fc_adj,
    adjacency_from_logits as adjacency_gumbel,
    integrate_torch,
)
from optimized_sparsification_simple import (
    adjacency_from_logits as adjacency_simple,
)

TEST_IC_SEED_OFFSET = 5000
TEST_N_ICS = 500
DEFAULT_SPARSIFY_SEEDS = 10
DEFAULT_CKPT = Path("plots/best_sparsification_logits.pt")
DEFAULT_SIMPLE_CKPT = Path("plots/best_sparsification_logits_simple.pt")


def sparse_adjacency_np(sigma_np, ei, ej, n, threshold=0.5):
    mask = sigma_np > threshold
    a = np.zeros((n, n))
    a[ei[mask], ej[mask]] = sigma_np[mask]
    a[ej[mask], ei[mask]] = sigma_np[mask]
    return a


def precompute_er(A):
    L = np.diag(A.sum(1)) - A
    Lp = np.linalg.pinv(L)
    diag = np.diag(Lp)
    R = diag[:, None] + diag[None, :] - 2 * Lp
    edge_i, edge_j = np.where(np.triu(A, 1))
    we = A[edge_i, edge_j].astype(float)
    Re = we * R[edge_i, edge_j]
    pe = Re / Re.sum()
    return edge_i, edge_j, we, pe


def sparsify_stochastic(edge_i, edge_j, we, pe, q_frac, rng):
    s = max(1, int(q_frac * len(edge_i)))
    n = int(np.max(np.concatenate([edge_i, edge_j])) + 1)
    A = np.zeros((n, n))
    for idx in rng.choice(len(edge_i), s, replace=True, p=pe):
        w = we[idx] / (s * pe[idx])
        A[edge_i[idx], edge_j[idx]] += w
        A[edge_j[idx], edge_i[idx]] += w
    return A


def sparsify_er(edge_i, edge_j, we, pe, q_frac, rng):
    return sparsify_stochastic(edge_i, edge_j, we, pe, q_frac, rng)


def sparsify_weight(edge_i, edge_j, we, q_frac, rng):
    pe = we / we.sum()
    return sparsify_stochastic(edge_i, edge_j, we, pe, q_frac, rng)


def sparsify_random(edge_i, edge_j, we, q_frac, rng):
    pe = np.full(len(edge_i), 1.0 / len(edge_i))
    return sparsify_stochastic(edge_i, edge_j, we, pe, q_frac, rng)


def sparsify_freq(edge_i, edge_j, we, omega, q_frac, rng):
    d = np.abs(omega[edge_i] - omega[edge_j]).astype(float)
    d_peak = float(np.median(d))
    width = max(2.0 * float(np.std(d)), 1e-6)
    pe = np.exp(-0.5 * ((d - d_peak) / width) ** 2)
    pe /= pe.sum()
    return sparsify_stochastic(edge_i, edge_j, we, pe, q_frac, rng)


def circ_loss(pred, target):
    return float((1.0 - np.cos(pred - target)).mean())


def _as_batch(pred, target):
    if pred.ndim == 2:
        pred = pred[:, None, :]
        target = target[:, None, :]
    return pred, target


def order_param_mean_errors(pred, target):
    pred, target = _as_batch(pred, target)
    dr_vals, dpsi_vals = [], []
    for b in range(pred.shape[1]):
        z_full = np.mean(np.exp(1j * target[:, b, :]), axis=1)
        z_pred = np.mean(np.exp(1j * pred[:, b, :]), axis=1)
        dr_vals.append(float(np.mean(np.abs(np.abs(z_full) - np.abs(z_pred)))))
        dpsi_vals.append(float(np.mean(np.degrees(np.abs(np.angle(z_full * np.conj(z_pred)))))))
    return float(np.mean(dr_vals)), float(np.mean(dpsi_vals))


def trajectory_metrics(pred, target):
    dr, dpsi = order_param_mean_errors(pred, target)
    return {
        "circ_loss_mean": circ_loss(pred, target),
        "dr_mean": dr,
        "dpsi_mean": dpsi,
    }


def load_omega_theta0(ckpt):
    seed = int(ckpt["seed"])
    ic_source = "checkpoint"

    if "omega" in ckpt:
        omega = ckpt["omega"].detach().cpu().numpy()
    else:
        ic_source = "seed (omega)"
        omega = np.random.default_rng(seed).normal(5.0, 0.5, N)

    if "theta0_batch" in ckpt:
        theta0_batch = ckpt["theta0_batch"].detach().cpu().numpy()
    elif "theta0" in ckpt:
        theta0_batch = ckpt["theta0"].detach().cpu().numpy()[None, :]
    else:
        ic_source = "seed (legacy single IC)"
        rng = np.random.default_rng(seed)
        _ = rng.normal(5.0, 0.5, N)
        theta0_batch = rng.uniform(0, 2 * np.pi, N)[None, :]
    return omega, theta0_batch, ic_source


def load_learned_model(ckpt_path, *, variant="gumbel"):
    ckpt = torch.load(ckpt_path, weights_only=False)
    logits = ckpt["logits"]
    omega, theta0_batch, ic_source = load_omega_theta0(ckpt)
    edge_budget = EDGE_BUDGET if EDGE_BUDGET is not None else int(Q * N * (N - 1) // 2)

    ei, ej = np.triu_indices(N, k=1)
    ei_t = torch.tensor(ei)
    ej_t = torch.tensor(ej)

    with torch.no_grad():
        if variant == "simple":
            _, sigma = adjacency_simple(logits, ei_t, ej_t, N)
            sigma_np = sigma.detach().cpu().numpy()
            a_sparse = sparse_adjacency_np(sigma_np, ei, ej, N)
            n_edges = int((sigma_np > 0.5).sum())
        else:
            a_t, z = adjacency_gumbel(logits, ei_t, ej_t, N, beta=None, top_k=edge_budget)
            a_sparse = a_t.detach().cpu().numpy()
            n_edges = int((z > 0).sum().item())
        a_sparse_t = torch.tensor(a_sparse, dtype=logits.dtype)

    label = "learned_simple" if variant == "simple" else "learned"
    return {
        "label": label,
        "variant": variant,
        "ckpt_path": ckpt_path,
        "seed": int(ckpt["seed"]),
        "epoch": int(ckpt["epoch"]),
        "ckpt_loss": float(ckpt["loss"]),
        "edge_budget": edge_budget,
        "n_ics": theta0_batch.shape[0],
        "omega": omega,
        "theta0_batch": theta0_batch,
        "ic_source": ic_source,
        "a_learned_t": a_sparse_t,
        "a_learned": a_sparse,
        "n_learned_edges": n_edges,
    }


def load_trial(ckpt_path, simple_ckpt_path=None):
    """single-checkpoint view for freq_sparsify_sweep and legacy callers."""
    primary = load_learned_model(ckpt_path, variant="gumbel")
    t = np.linspace(0, T_END, N_STEPS)
    er_data = precompute_er(fc_adj(N))
    return {
        "ckpt_path": ckpt_path,
        "seed": primary["seed"],
        "epoch": primary["epoch"],
        "edge_budget": primary["edge_budget"],
        "n_ics": primary["n_ics"],
        "t": t,
        "omega": primary["omega"],
        "theta0_batch": primary["theta0_batch"],
        "a_learned_t": primary["a_learned_t"],
        "a_learned": primary["a_learned"],
        "n_learned_edges": primary["n_learned_edges"],
        "er_data": er_data,
        "q": Q,
        "ic_source": primary["ic_source"],
        "primary": primary,
    }


def make_test_theta0(seed, n_ics):
    rng = np.random.default_rng(seed + TEST_IC_SEED_OFFSET)
    return rng.uniform(0, 2 * np.pi, (n_ics, N))


def count_train_test_overlap(train_theta0, test_theta0):
    n = 0
    for i in range(test_theta0.shape[0]):
        for j in range(train_theta0.shape[0]):
            if np.allclose(test_theta0[i], train_theta0[j]):
                n += 1
                break
    return n


def build_baseline_graphs(omega, seed, er_data, n_sparsify_seeds, q=Q):
    er_i, er_j, er_w, er_pe = er_data
    out = {}
    for method, offset in [
        ("er", 1000),
        ("weight_based", 2000),
        ("random", 3000),
        ("freq_based", 4000),
    ]:
        graphs = []
        for si in range(n_sparsify_seeds):
            rng = np.random.default_rng(seed + offset + si)
            if method == "er":
                graphs.append(sparsify_er(er_i, er_j, er_w, er_pe, q, rng))
            elif method == "weight_based":
                graphs.append(sparsify_weight(er_i, er_j, er_w, q, rng))
            elif method == "freq_based":
                graphs.append(sparsify_freq(er_i, er_j, er_w, omega, q, rng))
            else:
                graphs.append(sparsify_random(er_i, er_j, er_w, q, rng))
        out[method] = graphs
    return out


def build_eval_panel(model, t, n_test_ics, n_sparsify_seeds):
    test_theta0 = make_test_theta0(model["seed"], n_test_ics)
    overlap = count_train_test_overlap(model["theta0_batch"], test_theta0)
    er_data = precompute_er(fc_adj(N))
    baselines = build_baseline_graphs(
        model["omega"], model["seed"], er_data, n_sparsify_seeds,
    )
    return {
        "model": model,
        "t": t,
        "omega": model["omega"],
        "train_theta0": model["theta0_batch"],
        "test_theta0": test_theta0,
        "baselines": baselines,
        "test_overlap": overlap,
        "panel_title": (
            f"{model['label']} | seed {model['seed']} | epoch {model['epoch']}"
        ),
    }


def evaluate_graphs(theta0_batch_t, omega_t, t_t, target, graphs):
    metrics = []
    for A in graphs:
        a_t = torch.tensor(A, dtype=torch.float64)
        with torch.no_grad():
            pred = integrate_torch(theta0_batch_t, omega_t, a_t, K, t_t).numpy()
        metrics.append(trajectory_metrics(pred, target))
    return metrics


def _aggregate_metrics(metrics_list):
    return {
        "circ_loss_mean": float(np.mean([m["circ_loss_mean"] for m in metrics_list])),
        "circ_loss_std": float(np.std([m["circ_loss_mean"] for m in metrics_list]))
        if len(metrics_list) > 1 else 0.0,
        "dr_mean": float(np.mean([m["dr_mean"] for m in metrics_list])),
        "dr_std": float(np.std([m["dr_mean"] for m in metrics_list]))
        if len(metrics_list) > 1 else 0.0,
        "dpsi_mean": float(np.mean([m["dpsi_mean"] for m in metrics_list])),
        "dpsi_std": float(np.std([m["dpsi_mean"] for m in metrics_list]))
        if len(metrics_list) > 1 else 0.0,
    }


def evaluate_panel(panel, theta0_batch):
    model = panel["model"]
    omega = panel["omega"]
    t = panel["t"]
    omega_t = torch.tensor(omega, dtype=torch.float64)
    theta0_batch_t = torch.tensor(theta0_batch, dtype=torch.float64)
    t_t = torch.tensor(t, dtype=torch.float64)
    a_full_t = torch.tensor(fc_adj(N), dtype=torch.float64)

    with torch.no_grad():
        target = integrate_torch(theta0_batch_t, omega_t, a_full_t, K, t_t).numpy()
        pred_learned = integrate_torch(
            theta0_batch_t, omega_t, model["a_learned_t"], K, t_t,
        ).numpy()

    rows = [{
        "method": model["label"],
        "n_edges": model["n_learned_edges"],
        **_aggregate_metrics([trajectory_metrics(pred_learned, target)]),
    }]
    for method, graphs in panel["baselines"].items():
        graph_metrics = evaluate_graphs(theta0_batch_t, omega_t, t_t, target, graphs)
        rows.append({
            "method": method,
            "n_edges": int(np.mean([count_edges(g) for g in graphs])),
            **_aggregate_metrics(graph_metrics),
        })
    return rows


def print_panel_rows(panel, train_rows, test_rows, n_sparsify_seeds, n_test_ics):
    model = panel["model"]
    print(
        f"\n=== {panel['panel_title']} | ω+ICs from this checkpoint ==="
    )
    print(
        f"train {model['n_ics']} ICs ({model['ic_source']}) | "
        f"test {n_test_ics} ICs (rng seed {model['seed']}+{TEST_IC_SEED_OFFSET}) | "
        f"train/test overlap {panel['test_overlap']}"
    )
    print_rows("training ICs", train_rows, n_sparsify_seeds)
    print_rows("out-of-training ICs", test_rows, n_sparsify_seeds)


def print_rows(title, rows, n_sparsify_seeds):
    print(f"\n{title}")
    print(
        f"{'method':<16} {'circ loss':>12} {'|Δr|':>10} {'|Δψ|°':>10} {'edges':>8}"
    )
    print("-" * 62)
    for row in rows:
        print(
            f"{row['method']:<16} {row['circ_loss_mean']:>12.4e} "
            f"{row['dr_mean']:>10.4e} {row['dpsi_mean']:>10.2f} {row['n_edges']:>8}"
        )


METRIC_SPECS = [
    ("circ_loss_mean", "circ_loss_std", "circ loss"),
    ("dr_mean", "dr_std", "|Δr|"),
    ("dpsi_mean", "dpsi_std", "|Δψ| (°)"),
]
METHOD_COLORS = ["C2", "C0", "C1", "C3", "C4"]


def plot_metric_panel(ax, rows, n_sparsify_seeds, mean_key, std_key):
    labels = [r["method"] for r in rows]
    means = [r[mean_key] for r in rows]
    stds = [r[std_key] for r in rows]
    x = np.arange(len(labels))
    colors = [METHOD_COLORS[i % len(METHOD_COLORS)] for i in range(len(labels))]
    ax.bar(
        x,
        means,
        color=colors,
        yerr=stds if n_sparsify_seeds > 1 else None,
        capsize=3,
        edgecolor="0.2",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_yscale("log")


def plot_comparison_grid(panels_eval, n_test_ics, n_sparsify_seeds):
    n_panels = len(panels_eval)
    n_cols = 2 * n_panels  # train + test column per panel

    fig, axes = plt.subplots(
        len(METRIC_SPECS),
        n_cols,
        figsize=(5.5 * n_cols, 3.8 * len(METRIC_SPECS)),
        sharey="row",
    )
    # Normalise shape: always (n_metrics, n_cols)
    if axes.ndim == 1:
        axes = axes.reshape(len(METRIC_SPECS), n_cols)

    for pi, (panel, train_rows, test_rows) in enumerate(panels_eval):
        col_train = pi * 2
        col_test = pi * 2 + 1
        model = panel["model"]

        axes[0, col_train].set_title(
            f"{panel['panel_title']}\ntraining ICs ({model['n_ics']})",
            fontsize=8,
        )
        axes[0, col_test].set_title(
            f"{panel['panel_title']}\nout-of-training ICs ({n_test_ics})",
            fontsize=8,
        )

        for mi, (mean_key, std_key, metric_label) in enumerate(METRIC_SPECS):
            plot_metric_panel(axes[mi, col_train], train_rows, n_sparsify_seeds, mean_key, std_key)
            plot_metric_panel(axes[mi, col_test], test_rows, n_sparsify_seeds, mean_key, std_key)
            if col_train == 0:
                axes[mi, 0].set_ylabel(metric_label)

    fig.suptitle(
        f"sparsification comparison | N={N} q={Q} | {n_sparsify_seeds} sparsify seeds | "
        "each panel uses its own ω and ICs",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def count_edges(A):
    return int(np.count_nonzero(np.triu(A, 1)))


def run_comparison(
    ckpt_path=DEFAULT_CKPT,
    *,
    simple_ckpt_path=DEFAULT_SIMPLE_CKPT,
    n_sparsify_seeds=DEFAULT_SPARSIFY_SEEDS,
    test_n_ics=TEST_N_ICS,
    no_show=True,
):
    t = np.linspace(0, T_END, N_STEPS)
    panels = [build_eval_panel(load_learned_model(ckpt_path, variant="gumbel"), t, test_n_ics, n_sparsify_seeds)]

    if simple_ckpt_path.exists():
        panels.append(
            build_eval_panel(
                load_learned_model(simple_ckpt_path, variant="simple"),
                t, test_n_ics, n_sparsify_seeds,
            ),
        )
    else:
        print(f"warning: no simple checkpoint at {simple_ckpt_path} — skipping learned_simple panel")

    panels_eval = []
    print(f"checkpoint: {ckpt_path}")
    if len(panels) > 1:
        print(f"simple checkpoint: {simple_ckpt_path}")
    print(
        f"N={N} K={K:.2g} q={Q} | n_sparsify_seeds={n_sparsify_seeds} | test_n_ics={test_n_ics}"
    )

    for panel in panels:
        train_rows = evaluate_panel(panel, panel["train_theta0"])
        test_rows = evaluate_panel(panel, panel["test_theta0"])
        print_panel_rows(panel, train_rows, test_rows, n_sparsify_seeds, test_n_ics)
        panels_eval.append((panel, train_rows, test_rows))

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    fig = plot_comparison_grid(panels_eval, test_n_ics, n_sparsify_seeds)
    plot_path = out_dir / f"learned_sparsification_compare_n={N}_q={Q}.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nsaved {plot_path}")
    if no_show:
        plt.close(fig)
    else:
        plt.show()
    return plot_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=str(DEFAULT_CKPT),
        type=Path,
    )
    parser.add_argument(
        "--simple-checkpoint",
        default=str(DEFAULT_SIMPLE_CKPT),
        type=Path,
        help="checkpoint from optimized_sparsification_simple.py",
    )
    parser.add_argument("--n-sparsify-seeds", type=int, default=DEFAULT_SPARSIFY_SEEDS)
    parser.add_argument("--test-n-ics", type=int, default=TEST_N_ICS)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    run_comparison(
        args.checkpoint,
        simple_ckpt_path=args.simple_checkpoint,
        n_sparsify_seeds=args.n_sparsify_seeds,
        test_n_ics=args.test_n_ics,
        no_show=args.no_show,
    )


if __name__ == "__main__":
    main()
