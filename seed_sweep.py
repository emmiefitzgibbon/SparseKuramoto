"""
Compare ER vs weight-based sparsification across seeds and K values.
Metrics: steady-state and time-averaged |Δr|, |Δψ|, |Δz|,
|Δf_drift|, |Δr_lock|, |Δr_drift| (lock/drift split matches interactive_kuramoto.py).
Run: python seed_sweep.py
"""
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from kuramoto import (
    N, q, RUN_SEED, omega, theta_0, kuramoto_rhs, solver_kw,
    A_fc, A_random_weights, edges_from_A, weighted_degree,
)

q = 0.08
# sweep settings
n_seeds = 10
K_values = np.linspace(0, 10, 21)
t_span = (0, 20)
t_eval = np.linspace(*t_span, 300)
steady_fraction = 0.2
freq_lock_cutoff = 0.1  # |dθ/dt − Ω_pack| below this ⇒ in the pack


def laplacian(A):
    return np.diag(A.sum(axis=1)) - A


def edge_probs(A, method):
    edge_i, edge_j = np.where(np.triu(A, 1))
    we = A[edge_i, edge_j].astype(float)
    if method == 'wts':
        u = we
    elif method == 'er':
        degree_matrix = np.diag(A.sum(1))
        L = degree_matrix - A
        L_pinv = np.linalg.pinv(L)
        Re = np.zeros((N, N))
        for ii in range(N):
            for jj in range(N):
                e = np.zeros((N, 1))
                e[ii, 0], e[jj, 0] = 1, -1
                Re[ii, jj] = (e.T @ L_pinv @ e).item()
        u = we * Re[edge_i, edge_j]
    else:
        raise ValueError(method)
    return edge_i, edge_j, we, u / u.sum()


def sparsify(edge_i, edge_j, we, pe, rng):
    s = max(1, int(q * len(edge_i)))
    A_sparse = np.zeros((N, N))
    for k in rng.choice(len(edge_i), s, replace=True, p=pe):
        w = we[k] / (s * pe[k])
        A_sparse[edge_i[k], edge_j[k]] += w
        A_sparse[edge_j[k], edge_i[k]] += w
    return A_sparse


def order_param_series(theta_t):
    z = np.mean(np.exp(1j * theta_t), axis=0)  # order param full: one complex number per time
    r = np.abs(z)
    return r, z


def pack_omega_series(psi, t):
    return np.gradient(np.unwrap(psi), t)


def oscillator_freq_series(sol, t):
    return np.gradient(np.unwrap(sol, axis=1), t, axis=1)
    # returns the instantaneous frequency of each oscillator at each time point
    # np.gradient calculates the numerical derivative of the phase angle


def lock_drift_series(sol, t, cutoff=freq_lock_cutoff):
    """f_drift, r_lock, r_drift — same definition as interactive_kuramoto.py."""
    psi = np.angle(np.mean(np.exp(1j * sol), axis=0))
    omega_pack = pack_omega_series(psi, t)
    omega_inst = oscillator_freq_series(sol, t)
    locked = np.abs(omega_inst - omega_pack[np.newaxis, :]) <= cutoff
    n_t = sol.shape[1]
    f_drift = 1.0 - locked.mean(axis=0)
    r_lock = np.zeros(n_t)
    r_drift = np.zeros(n_t)
    for k in range(n_t):
        lk, dk = locked[:, k], ~locked[:, k]
        if lk.any():
            r_lock[k] = np.abs(np.sum(np.exp(1j * sol[lk, k])) / N)
        if dk.any():
            r_drift[k] = np.abs(np.sum(np.exp(1j * sol[dk, k])) / N)
    return f_drift, r_lock, r_drift


def trajectory_errors(sol_full, sol_sparse):
    r_full, z_full = order_param_series(sol_full)
    r_sparse, z_sparse = order_param_series(sol_sparse)
    err_r = np.abs(r_full - r_sparse)
    err_z = np.abs(z_full - z_sparse)
    err_psi = np.degrees(np.abs(np.angle(z_full * np.conj(z_sparse))))

    f_drift_f, r_lock_f, r_drift_f = lock_drift_series(sol_full, t_eval)
    f_drift_s, r_lock_s, r_drift_s = lock_drift_series(sol_sparse, t_eval)
    err_f_drift = np.abs(f_drift_f - f_drift_s)
    err_r_lock = np.abs(r_lock_f - r_lock_s)
    err_r_drift = np.abs(r_drift_f - r_drift_s)

    late = t_eval >= t_eval[-1] * (1 - steady_fraction)
    return {
        'r_steady': float(np.mean(err_r[late])),
        'r_mean': float(np.mean(err_r)),
        'psi_steady': float(np.mean(err_psi[late])),
        'psi_mean': float(np.mean(err_psi)),
        'z_mean': float(np.mean(err_z)),
        'f_drift_steady': float(np.mean(err_f_drift[late])),
        'f_drift_mean': float(np.mean(err_f_drift)),
        'r_lock_steady': float(np.mean(err_r_lock[late])),
        'r_lock_mean': float(np.mean(err_r_lock)),
        'r_drift_steady': float(np.mean(err_r_drift[late])),
        'r_drift_mean': float(np.mean(err_r_drift)),
    }


graphs = {
    'fc': A_fc,
    'random_weights': A_random_weights,
}
methods = ('er', 'wts')
METHOD_LABELS = {'er': 'effective resistance', 'wts': 'weight-based'}
GRAPH_LABELS = {'fc': 'FC (w=1)', 'random_weights': 'FC random weights'}

# results[g][m][k_idx] -> list of metric dicts, one per seed
results = {
    g: {m: {ki: [] for ki in range(len(K_values))} for m in methods}
    for g in graphs
}

print(f'seed × K sweep: {n_seeds} seeds, {len(K_values)} K values, N={N}, q={q}')
for gname, A in graphs.items():
    print(f'  {gname}: {len(edges_from_A(A)[0]):,} edges')

prob_cache = {g: {m: edge_probs(A, m) for m in methods} for g, A in graphs.items()}
degree_full_cache = {g: weighted_degree(A) for g, A in graphs.items()}
sparse_edges = {g: {m: {} for m in methods} for g in graphs}
sparse_degrees = {g: {m: {} for m in methods} for g in graphs}

for seed in range(n_seeds):
    for gname, A in graphs.items():
        for method in methods:
            edge_i, edge_j, we, pe = prob_cache[gname][method]
            trial_seed = RUN_SEED + 1000 * list(graphs).index(gname) + 100 * methods.index(method) + seed
            A_sparse = sparsify(edge_i, edge_j, we, pe, np.random.default_rng(trial_seed))
            sparse_edges[gname][method][seed] = edges_from_A(A_sparse)
            sparse_degrees[gname][method][seed] = weighted_degree(A_sparse)

for ki, K in enumerate(K_values):
    for gname, A in graphs.items():
        edges_full = edges_from_A(A)
        sol_full = solve_ivp(
            kuramoto_rhs, t_span, theta_0,
            args=(K, omega, edges_full, degree_full_cache[gname]),
            t_eval=t_eval, **solver_kw,
        ).y

        for method in methods:
            for seed in range(n_seeds):
                sol_sparse = solve_ivp(
                    kuramoto_rhs, t_span, theta_0,
                    args=(K, omega, sparse_edges[gname][method][seed],
                          sparse_degrees[gname][method][seed]),
                    t_eval=t_eval, **solver_kw,
                ).y
                metrics = trajectory_errors(sol_full, sol_sparse)
                metrics['seed'] = seed
                metrics['K'] = K
                results[gname][method][ki].append(metrics)

    if (ki + 1) % 5 == 0 or ki == len(K_values) - 1:
        print(f'  finished K={K:.2f} ({ki + 1}/{len(K_values)})')


def mean_std(records, key):
    vals = [r[key] for r in records]
    return float(np.mean(vals)), float(np.std(vals))


# summary at K ≈ 6.25 (near default coupling)
K_ref_idx = int(np.argmin(np.abs(K_values - 6.25)))
K_ref = K_values[K_ref_idx]

for gname in graphs:
    for method in methods:
        rs_m, rs_s = mean_std(results[gname][method][K_ref_idx], 'r_steady')
        rm_m, rm_s = mean_std(results[gname][method][K_ref_idx], 'r_mean')
        ps_m, ps_s = mean_std(results[gname][method][K_ref_idx], 'psi_steady')
        pm_m, pm_s = mean_std(results[gname][method][K_ref_idx], 'psi_mean')
        zm_m, zm_s = mean_std(results[gname][method][K_ref_idx], 'z_mean')
        fd_m, fd_s = mean_std(results[gname][method][K_ref_idx], 'f_drift_steady')
        rl_m, rl_s = mean_std(results[gname][method][K_ref_idx], 'r_lock_mean')
        rd_m, rd_s = mean_std(results[gname][method][K_ref_idx], 'r_drift_mean')
        print(
            f'{GRAPH_LABELS[gname]:<22} {METHOD_LABELS[method]:<20} '
            f'{rs_m:6.4f}±{rs_s:4.4f}  '
            f'{rm_m:6.4f}±{rm_s:4.4f}  '
            f'{ps_m:6.2f}±{ps_s:4.2f}  '
            f'{pm_m:6.2f}±{pm_s:4.2f}  '
            f'{zm_m:6.4f}±{zm_s:4.4f}  '
            f'{fd_m:6.4f}±{fd_s:4.4f}  '
            f'{rl_m:6.4f}±{rl_s:4.4f}  '
            f'{rd_m:6.4f}±{rd_s:4.4f}'
        )

for gname in graphs:
    wins_ss = sum(
        mean_std(results[gname]['er'][ki], 'r_steady')[0]
        < mean_std(results[gname]['wts'][ki], 'r_steady')[0]
        for ki in range(len(K_values))
    )
    wins_mean = sum(
        mean_std(results[gname]['er'][ki], 'r_mean')[0]
        < mean_std(results[gname]['wts'][ki], 'r_mean')[0]
        for ki in range(len(K_values))
    )
    print(f'{GRAPH_LABELS[gname]}: effective resistance wins steady-state |Δr| at {wins_ss}/{len(K_values)} K values')
    print(f'{GRAPH_LABELS[gname]}: effective resistance wins mean |Δr| at {wins_mean}/{len(K_values)} K values')

print(f'\npeak K on grid {K_values[0]:.2g}–{K_values[-1]:.2g} ({len(K_values)} points, mean over seeds):')
for gname in graphs:
    for method in methods:
        for key in (
            'r_mean', 'r_steady', 'psi_mean', 'psi_steady',
            'f_drift_mean', 'f_drift_steady', 'r_lock_mean', 'r_lock_steady',
            'r_drift_mean', 'r_drift_steady',
        ):
            means = [mean_std(results[gname][method][ki], key)[0] for ki in range(len(K_values))]
            ki = int(np.argmax(means))
            print(
                f'  {GRAPH_LABELS[gname]} / {METHOD_LABELS[method]} / {key}: '
                f'K={K_values[ki]:.2g}  (error={means[ki]:.4g})'
            )

# plots: K vs error
fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
plot_specs = [
    (0, 0, 'r_steady', 'steady-state |Δr|'),
    (0, 1, 'r_mean', 'time-averaged |Δr|'),
    (0, 2, 'psi_steady', 'steady-state |Δψ| (°)'),
    (1, 0, 'psi_mean', 'time-averaged |Δψ| (°)'),
    (1, 1, 'z_mean', 'time-averaged |Δz|'),
    (1, 2, 'f_drift_steady', 'steady-state |Δf_drift|'),
    (2, 0, 'f_drift_mean', 'time-averaged |Δf_drift|'),
    (2, 1, 'r_lock_mean', 'time-averaged |Δr_lock|'),
    (2, 2, 'r_drift_mean', 'time-averaged |Δr_drift|'),
]
colors = {'er': 'C0', 'wts': 'C1'}


def plot_k_curve(ax, key, ylabel):
    for gname in graphs:
        for method in methods:
            means = [mean_std(results[gname][method][ki], key)[0] for ki in range(len(K_values))]
            stds = [mean_std(results[gname][method][ki], key)[1] for ki in range(len(K_values))]
            ls = '-' if gname == 'fc' else '--'
            label = f'{GRAPH_LABELS[gname]} — {METHOD_LABELS[method]}'
            ax.plot(K_values, means, ls=ls, color=colors[method], label=label)
            ax.fill_between(
                K_values, np.array(means) - stds, np.array(means) + stds,
                color=colors[method], alpha=0.12 if gname == 'fc' else 0.06,
            )
    ax.set_ylabel(ylabel)
    ax.set_xlim(K_values[0], K_values[-1])
    ax.legend(fontsize=7, loc='upper left')


for row, col, key, ylabel in plot_specs:
    plot_k_curve(axes[row, col], key, ylabel)
for col in range(3):
    axes[2, col].set_xlabel('coupling K')
fig.suptitle(f'sparsification error vs K ({n_seeds} seeds, q={q})')
plt.tight_layout()
plots_dir = Path('plots')
plots_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plots_dir / 'seed_sweep.png', dpi=120)
print('\nsaved plots/seed_sweep.png')

def seed_curves(gname, method, key):
    """one K-curve per seed."""
    n = len(results[gname][method][0])
    return [
        [results[gname][method][ki][si][key] for ki in range(len(K_values))]
        for si in range(n)
    ]


# per-graph panels: time-averaged r, ψ, and lock/drift errors
by_graph_specs = [
    ('r_mean', 'time-averaged |Δr|'),
    ('psi_mean', 'time-averaged |Δψ| (°)'),
    ('f_drift_mean', 'time-averaged |Δf_drift|'),
    ('r_lock_mean', 'time-averaged |Δr_lock|'),
    ('r_drift_mean', 'time-averaged |Δr_drift|'),
]
fig2, axes2 = plt.subplots(len(by_graph_specs), 2, figsize=(11, 2.5 * len(by_graph_specs)), sharex=True)
for row, (key, ylabel) in enumerate(by_graph_specs):
    for col, gname in enumerate(graphs):
        ax = axes2[row, col]
        for method in methods:
            for curve in seed_curves(gname, method, key):
                ax.plot(K_values, curve, color=colors[method], alpha=0.25, lw=0.8, zorder=1)
            means = [
                mean_std(results[gname][method][ki], key)[0]
                for ki in range(len(K_values))
            ]
            ax.plot(
                K_values, means, label=METHOD_LABELS[method],
                color=colors[method], lw=2.5, zorder=2,
            )
        if row == 0:
            ax.set_title(GRAPH_LABELS[gname])
        ax.set_ylabel(ylabel)
        if row == 0:
            ax.legend()
axes2[-1, 0].set_xlabel('coupling K')
axes2[-1, 1].set_xlabel('coupling K')
fig2.suptitle(f'effective resistance vs weight-based by graph ({n_seeds} seeds, q={q})')
plt.tight_layout()
plt.savefig(plots_dir / 'seed_sweep_by_graph.png', dpi=120)
print('saved plots/seed_sweep_by_graph.png')
