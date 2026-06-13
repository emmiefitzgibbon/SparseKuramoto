"""
Seed sweep: two FC communities (bridged) — ER vs weight-based vs random sparsification.
Run: python bridge_seed_sweep.py
     python bridge_seed_sweep.py --sweep-bridge
     python bridge_seed_sweep.py --quick
     python bridge_seed_sweep.py --n-seeds 30 --bridge-rho 0.01 --bridge-w 0.5
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from kuramoto import edges_from_A, weighted_degree

solver_kw = dict(method='RK45', rtol=1e-6, atol=1e-8)


def kuramoto_rhs(t, theta, K, omega, edges, degree):
    ei, ej, w = edges
    coupling = np.zeros(theta.shape[0])
    np.add.at(coupling, ei, w * np.sin(theta[ej] - theta[ei]))
    np.add.at(coupling, ej, w * np.sin(theta[ei] - theta[ej]))
    return omega + K * coupling / degree

BRIDGE_RHO = 0.001
BRIDGE_W = 0.15
DOMEGA_DEFAULT = 0.0
# interactive_kuramoto bridged defaults: Δω=2 shifts comm B → global r beats (T≈π s)
OSCILLATION_BRIDGE_RHO = 0.001
OSCILLATION_BRIDGE_W = 0.15
OSCILLATION_DOMEGA = 2.0
Q = 0.25
N_DEFAULT = 624
K_DEFAULT = 6.25
OMEGA_MEAN = 5.0
OMEGA_STD = 0.5
T_END = 20.0
STEADY_FRACTION = 0.2
FREQ_LOCK_CUTOFF = 0.1
METHODS = ('er', 'weight_based', 'random')
METHOD_LABELS = {
    'er': 'ER (effective resistance)',
    'weight_based': 'weight based',
    'random': 'random',
}
METHOD_COLORS = {'er': 'C0', 'weight_based': 'C1', 'random': 'C2'}

# (label, ρ, w) — baseline plus weaker/stronger coupling and sparser/denser bridges
BRIDGE_CONFIGS = (
    ('baseline', 0.001, 0.15),
    ('weak w', 0.001, 0.05),
    ('strong w', 0.001, 0.5),
    ('sparse ρ', 0.0001, 0.15),
    ('dense ρ', 0.01, 0.15),
    ('dense+strong', 0.01, 0.5),
)

METRIC_SPECS = (
    ('r_steady', 'steady-state |Δr| (global)'),
    ('a_r_steady', 'steady-state |Δr| (comm A)'),
    ('b_r_steady', 'steady-state |Δr| (comm B)'),
    ('psi_steady', 'steady-state |Δψ| (°)'),
    ('r_mean', 'time-averaged |Δr|'),
)


def fresh_seed():
    return int.from_bytes(os.urandom(8), 'big') % (2**63)


def build_bridged_communities_A(n, rng, bridge_density, bridge_weight):
    n0 = n // 2
    A = np.zeros((n, n))
    if n0 > 1:
        A[:n0, :n0] = 1.0
        np.fill_diagonal(A[:n0, :n0], 0)
    if n - n0 > 1:
        A[n0:, n0:] = 1.0
        np.fill_diagonal(A[n0:, n0:], 0)
    if n0 > 0 and n - n0 > 0:
        ci = np.repeat(np.arange(n0), n - n0)
        cj = np.tile(np.arange(n0, n), n0)
        keep = rng.random(len(ci)) < bridge_density
        A[ci[keep], cj[keep]] = bridge_weight
        A[cj[keep], ci[keep]] = bridge_weight
    return A, n0


def precompute_er(A):
    graph_laplacian = np.diag(A.sum(1)) - A
    graph_laplacian_pinv = np.linalg.pinv(graph_laplacian)
    diag = np.diag(graph_laplacian_pinv)
    effective_resistances = diag[:, None] + diag[None, :] - 2 * graph_laplacian_pinv
    edge_i, edge_j = np.where(np.triu(A, 1))
    we = A[edge_i, edge_j].astype(float)
    Re = we * effective_resistances[edge_i, edge_j]
    pe = Re / Re.sum()
    return edge_i, edge_j, we, pe


def sparsify_stochastic(edge_i, edge_j, we, pe, q, rng):
    s = max(1, int(q * len(edge_i)))
    n = int(np.max(np.concatenate([edge_i, edge_j])) + 1)
    A = np.zeros((n, n))
    for idx in rng.choice(len(edge_i), s, replace=True, p=pe):
        w = we[idx] / (s * pe[idx])
        A[edge_i[idx], edge_j[idx]] += w
        A[edge_j[idx], edge_i[idx]] += w
    return A


def sparsify_graph(A, method, q, rng):
    edge_i, edge_j, we, pe_er = precompute_er(A)
    if method == 'er':
        pe = pe_er
    elif method == 'weight_based':
        pe = we / we.sum()
    elif method == 'random':
        pe = np.full(len(edge_i), 1.0 / len(edge_i))
    else:
        raise ValueError(method)
    return sparsify_stochastic(edge_i, edge_j, we, pe, q, rng)


def order_param_series(theta_t, sl=None):
    sub = theta_t if sl is None else theta_t[sl]
    z = np.mean(np.exp(1j * sub), axis=0)
    return np.abs(z), z


def pack_omega_series(psi, t):
    return np.gradient(np.unwrap(psi), t)


def oscillator_freq_series(sol, t):
    return np.gradient(np.unwrap(sol, axis=1), t, axis=1)


def lock_drift_series(sol, t, n, cutoff=FREQ_LOCK_CUTOFF):
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
            r_lock[k] = np.abs(np.sum(np.exp(1j * sol[lk, k])) / n)
        if dk.any():
            r_drift[k] = np.abs(np.sum(np.exp(1j * sol[dk, k])) / n)
    return f_drift, r_lock, r_drift


def trajectory_errors(sol_full, sol_sparse, t_eval, n0):
    late = t_eval >= t_eval[-1] * (1 - STEADY_FRACTION)
    n = sol_full.shape[0]
    out = {}

    def block_errors(label, sl_f, sl_s):
        r_f, z_f = order_param_series(sol_full, sl_f)
        r_s, z_s = order_param_series(sol_sparse, sl_s)
        err_r = np.abs(r_f - r_s)
        err_psi = np.degrees(np.abs(np.angle(z_f * np.conj(z_s))))
        err_z = np.abs(z_f - z_s)
        prefix = f'{label}_' if label else ''
        out[f'{prefix}r_steady'] = float(np.mean(err_r[late]))
        out[f'{prefix}r_mean'] = float(np.mean(err_r))
        out[f'{prefix}psi_steady'] = float(np.mean(err_psi[late]))
        out[f'{prefix}psi_mean'] = float(np.mean(err_psi))
        out[f'{prefix}z_mean'] = float(np.mean(err_z))

    block_errors('', slice(None), slice(None))
    block_errors('a', slice(0, n0), slice(0, n0))
    block_errors('b', slice(n0, n), slice(n0, n))

    f_drift_f, r_lock_f, r_drift_f = lock_drift_series(sol_full, t_eval, n)
    f_drift_s, r_lock_s, r_drift_s = lock_drift_series(sol_sparse, t_eval, n)
    for key, err in (
        ('f_drift', np.abs(f_drift_f - f_drift_s)),
        ('r_lock', np.abs(r_lock_f - r_lock_s)),
        ('r_drift', np.abs(r_drift_f - r_drift_s)),
    ):
        out[f'{key}_steady'] = float(np.mean(err[late]))
        out[f'{key}_mean'] = float(np.mean(err))
    return out


def setup_trial(run_seed, n, bridge_rho, bridge_w, domega=DOMEGA_DEFAULT):
    rng = np.random.default_rng(run_seed)
    A, n0 = build_bridged_communities_A(n, rng, bridge_rho, bridge_w)
    omega = rng.normal(loc=OMEGA_MEAN, scale=OMEGA_STD, size=n)
    if domega != 0.0:
        omega = omega.copy()
        omega[n0:] += domega
    theta_0 = rng.uniform(0, 2 * np.pi, n)
    return A, n0, omega, theta_0


def param_suffix(bridge_rho, bridge_w, q, k, domega=0.0):
    parts = [f'ρ={bridge_rho:g}  w={bridge_w:g}  q={q:g}  K={k:g}']
    if domega != 0.0:
        parts.append(f'Δω={domega:g}')
    return '  '.join(parts)


def run_trial(A, n0, omega, theta_0, K, q, method, sparsify_seed, t_eval):
    rng = np.random.default_rng(sparsify_seed)
    A_sparse = sparsify_graph(A, method, q, rng)
    edges_full = edges_from_A(A)
    edges_sparse = edges_from_A(A_sparse)
    degree_full = weighted_degree(A)
    degree_sparse = weighted_degree(A_sparse)
    t_span = (0.0, t_eval[-1])
    sol_full = solve_ivp(
        kuramoto_rhs, t_span, theta_0,
        args=(K, omega, edges_full, degree_full),
        t_eval=t_eval, **solver_kw,
    ).y
    sol_sparse = solve_ivp(
        kuramoto_rhs, t_span, theta_0,
        args=(K, omega, edges_sparse, degree_sparse),
        t_eval=t_eval, **solver_kw,
    ).y
    metrics = trajectory_errors(sol_full, sol_sparse, t_eval, n0)
    metrics['n_sparse_edges'] = len(edges_sparse[0])
    metrics['n_bridges_sparse'] = int(np.count_nonzero(A_sparse[:n0, n0:]))
    return metrics


def mean_std(records, key):
    vals = [r[key] for r in records]
    return float(np.mean(vals)), float(np.std(vals))


def annotate_boxplot_means(ax, bp, data, *, fmt_auto=True):
    for i, vals in enumerate(data):
        mu = float(np.mean(vals))
        if fmt_auto:
            fmt = '.1f' if mu > 10 else ('.4f' if mu < 0.01 else '.3f')
        else:
            fmt = '.3f'
        whisker_tops = [
            bp['whiskers'][2 * i].get_ydata()[1],
            bp['whiskers'][2 * i + 1].get_ydata()[1],
        ]
        y = max(whisker_tops)
        ax.text(i + 1, y, f'{mu:{fmt}}', ha='center', va='bottom', fontsize=7)


def load_results_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    methods = tuple(str(m) for m in d['methods'])
    results = d['results'].item()
    return {
        'results': {m: results[m] for m in methods},
        'n': int(d['n']),
        'n_seeds': len(results[methods[0]]),
        'bridge_rho': float(d['bridge_rho']),
        'bridge_w': float(d['bridge_w']),
        'q': float(d['q']),
        'k': float(d['k']),
        'base_seed': int(d['base_seed']),
        'domega': float(d['domega']) if 'domega' in d else 0.0,
    }


def plot_boxplot_results(
    results, plots_dir, stem, n_seeds, bridge_rho, bridge_w, q, k,
    domega=0.0,
):
    fig, axes = plt.subplots(1, len(METRIC_SPECS), figsize=(3.2 * len(METRIC_SPECS), 4.2))
    if len(METRIC_SPECS) == 1:
        axes = [axes]
    for ax, (key, title) in zip(axes, METRIC_SPECS):
        data = [[r[key] for r in results[m]] for m in METHODS]
        bp = ax.boxplot(
            data, tick_labels=[METHOD_LABELS[m].replace(' ', '\n') for m in METHODS],
            patch_artist=True,
        )
        for patch, method in zip(bp['boxes'], METHODS):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.55)
        annotate_boxplot_means(ax, bp, data)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(
        f'bridged FC seed sweep ({n_seeds} seeds)\n'
        f'{param_suffix(bridge_rho, bridge_w, q, k, domega)}',
        fontsize=11,
    )
    plt.tight_layout()
    plot_path = plots_dir / f'{stem}.png'
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def integrate_global_r(A, omega, theta_0, K, t_eval):
    edges = edges_from_A(A)
    degree = weighted_degree(A)
    sol = solve_ivp(
        kuramoto_rhs, (0.0, t_eval[-1]), theta_0,
        args=(K, omega, edges, degree),
        t_eval=t_eval, **solver_kw,
    ).y
    r_g, _ = order_param_series(sol)
    return r_g, sol


def full_order_param_trace(A, n0, omega, theta_0, K, t_eval):
    r_g, sol = integrate_global_r(A, omega, theta_0, K, t_eval)
    r_a, _ = order_param_series(sol, slice(0, n0))
    r_b, _ = order_param_series(sol, slice(n0, sol.shape[0]))
    return r_g, r_a, r_b


def sparse_global_traces(A, omega, theta_0, K, q, t_eval, base_seed):
    """global r(t) on each sparsified graph (seed 0 sparsify seeds)."""
    traces = {}
    for mi, method in enumerate(METHODS):
        sparsify_seed = base_seed + 100 * mi
        rng = np.random.default_rng(sparsify_seed)
        A_sparse = sparsify_graph(A, method, q, rng)
        traces[method], _ = integrate_global_r(A_sparse, omega, theta_0, K, t_eval)
    return traces


def plot_full_r_timeseries(
    t_eval, r_global, r_a, r_b, plots_dir, stem,
    bridge_rho, bridge_w, q, k, seed, domega=0.0,
):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t_eval, r_global, 'k-', lw=1.5, label='global')
    ax.plot(t_eval, r_a, color='C0', lw=1.2, label='comm A')
    ax.plot(t_eval, r_b, color='C1', lw=1.2, label='comm B')
    ax.set_xlim(t_eval[0], t_eval[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('time')
    ax.set_ylabel('r = |⟨e^{iθ}⟩|')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    fig.suptitle(
        f'full-network order parameter  (seed {seed})\n'
        f'{param_suffix(bridge_rho, bridge_w, q, k, domega)}',
        fontsize=10,
    )
    plt.tight_layout()
    path = plots_dir / f'{stem}_r_timeseries.png'
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_r_compare_full_sparse(
    t_eval, r_full, r_sparse, plots_dir, stem,
    bridge_rho, bridge_w, q, k, seed, domega=0.0,
):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t_eval, r_full, 'k-', lw=1.8, label='full')
    for method in METHODS:
        ax.plot(
            t_eval, r_sparse[method], color=METHOD_COLORS[method], lw=1.2,
            ls='--', label=METHOD_LABELS[method],
        )
    ax.set_xlim(t_eval[0], t_eval[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('time')
    ax.set_ylabel('r = |⟨e^{iθ}⟩|  (global)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    fig.suptitle(
        f'global order parameter: full vs sparse  (seed {seed})\n'
        f'{param_suffix(bridge_rho, bridge_w, q, k, domega)}',
        fontsize=10,
    )
    plt.tight_layout()
    path = plots_dir / f'{stem}_r_compare.png'
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def make_r_trace_plots(
    A, n0, omega, theta_0, K, q, t_eval, plots_dir, stem,
    bridge_rho, bridge_w, k, base_seed, domega=0.0, seed=0,
):
    r_g, r_a, r_b = full_order_param_trace(A, n0, omega, theta_0, K, t_eval)
    r_sparse = sparse_global_traces(A, omega, theta_0, K, q, t_eval, base_seed)
    paths = {}
    paths['full'] = plot_full_r_timeseries(
        t_eval, r_g, r_a, r_b, plots_dir, stem,
        bridge_rho, bridge_w, q, k, seed=seed, domega=domega,
    )
    paths['compare'] = plot_r_compare_full_sparse(
        t_eval, r_g, r_sparse, plots_dir, stem,
        bridge_rho, bridge_w, q, k, seed=seed, domega=domega,
    )
    return paths


def config_stem(n, bridge_rho, bridge_w, q, k, domega=0.0):
    stem = (
        f'bridge_seed_sweep_n={n}_rho={bridge_rho:g}_w={bridge_w:g}'
        f'_q={q:g}_k={k:g}'
    )
    if domega != 0.0:
        stem += f'_domega={domega:g}'
    return stem


def print_mean_errors(results, header=''):
    if header:
        print(f'\n{header}')
    for key, title in METRIC_SPECS:
        print(f'  {title}')
        for method in METHODS:
            mu, sig = mean_std(results[method], key)
            fmt = '.4f' if key != 'psi_steady' else '.2f'
            print(f'    {METHOD_LABELS[method]:<28}  {mu:{fmt}} ± {sig:{fmt}}')
        print()


def save_config_results(
    results, plots_dir, data_dir, n, n_seeds, bridge_rho, bridge_w, q, k, base_seed,
    domega=0.0,
):
    stem = config_stem(n, bridge_rho, bridge_w, q, k, domega)
    np.savez_compressed(
        data_dir / f'{stem}.npz',
        base_seed=base_seed,
        n=n,
        n0=n // 2,
        bridge_rho=bridge_rho,
        bridge_w=bridge_w,
        domega=domega,
        q=q,
        k=k,
        methods=np.array(METHODS),
        results={m: results[m] for m in METHODS},
    )
    print(f'saved data/{stem}.npz')
    plot_path = plot_boxplot_results(
        results, plots_dir, stem, n_seeds, bridge_rho, bridge_w, q, k, domega,
    )
    print(f'saved {plot_path}')
    return stem


def run_one_config(
    n, n_seeds, bridge_rho, bridge_w, q, k, base_seed, t_eval, label=None,
    *, domega=DOMEGA_DEFAULT, plot_r_trace=False, plots_dir=None,
):
    tag = f' [{label}]' if label else ''
    print(
        f'\nbridge seed sweep{tag}: N={n}  {param_suffix(bridge_rho, bridge_w, q, k, domega)}  '
        f'seeds={n_seeds}  base_seed={base_seed}',
    )
    results = {m: [] for m in METHODS}
    for si in range(n_seeds):
        run_seed = base_seed + si
        A, n0, omega, theta_0 = setup_trial(run_seed, n, bridge_rho, bridge_w, domega)
        if si == 0:
            n_full = len(edges_from_A(A)[0])
            n_bridges_full = int(np.count_nonzero(A[:n0, n0:]))
            print(f'  full graph: {n_full:,} edges, {n_bridges_full} bridges')
        for mi, method in enumerate(METHODS):
            sparsify_seed = base_seed + 10_000 * si + 100 * mi
            metrics = run_trial(
                A, n0, omega, theta_0, k, q, method, sparsify_seed, t_eval,
            )
            metrics['seed'] = si
            metrics['method'] = method
            results[method].append(metrics)
        if (si + 1) % max(1, n_seeds // 5) == 0 or si == n_seeds - 1:
            print(f'  finished seed {si + 1}/{n_seeds}')
    if plot_r_trace and plots_dir is not None:
        A, n0, omega, theta_0 = setup_trial(base_seed, n, bridge_rho, bridge_w, domega)
        stem = config_stem(n, bridge_rho, bridge_w, q, k, domega)
        for path in make_r_trace_plots(
            A, n0, omega, theta_0, k, q, t_eval, plots_dir, stem,
            bridge_rho, bridge_w, k, base_seed, domega=domega,
        ).values():
            print(f'saved {path}')
    print_mean_errors(results, header='mean errors (± std over seeds):')
    return results


def save_compare_plot(sweep_results, plots_dir, n, n_seeds, q, k):
    """Grouped bars: mean steady |Δr| global per method, one group per bridge config."""
    labels = [cfg['label'] for cfg in sweep_results]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(labels)), 4.5))
    for i, method in enumerate(METHODS):
        means = [mean_std(cfg['results'][method], 'r_steady')[0] for cfg in sweep_results]
        ax.bar(x + (i - 1) * width, means, width, label=METHOD_LABELS[method],
               color=METHOD_COLORS[method], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c['label']}\nρ={c['bridge_rho']:g} w={c['bridge_w']:g}" for c in sweep_results],
        fontsize=9,
    )
    ax.set_ylabel('mean steady-state |Δr| (global)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.suptitle(
        f'bridge param comparison ({n_seeds} seeds, q={q:g}, K={k:g}, N={n})',
        fontsize=11,
    )
    plt.tight_layout()
    path = plots_dir / f'bridge_seed_sweep_compare_n={n}_q={q:g}_k={k:g}.png'
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f'saved {path}')


def print_compare_table(sweep_results):
    print('\n=== bridge config comparison (mean steady |Δr| global) ===')
    header = f"{'config':<16} {'ρ':>8} {'w':>6}"
    for method in METHODS:
        header += f'  {METHOD_LABELS[method]:>12}'
    print(header)
    for cfg in sweep_results:
        row = f"{cfg['label']:<16} {cfg['bridge_rho']:8g} {cfg['bridge_w']:6g}"
        for method in METHODS:
            mu, _ = mean_std(cfg['results'][method], 'r_steady')
            row += f'  {mu:12.4f}'
        print(row)


def parse_args():
    p = argparse.ArgumentParser(description='bridged-community sparsification seed sweep')
    p.add_argument('--n-seeds', type=int, default=500)
    p.add_argument('--n', type=int, default=N_DEFAULT)
    p.add_argument('--bridge-rho', type=float, default=BRIDGE_RHO)
    p.add_argument('--bridge-w', type=float, default=BRIDGE_W)
    p.add_argument(
        '--domega', type=float, default=DOMEGA_DEFAULT,
        help='frequency offset added to comm B (interactive default for beats: 2.0)',
    )
    p.add_argument('--q', type=float, default=Q)
    p.add_argument('--k', type=float, default=K_DEFAULT)
    p.add_argument('--base-seed', type=int, default=None)
    p.add_argument('--quick', action='store_true', help='5 seeds, shorter time grid')
    p.add_argument(
        '--sweep-bridge', action='store_true',
        help=f'run {len(BRIDGE_CONFIGS)} bridge (ρ, w) presets and save comparison plot',
    )
    p.add_argument(
        '--plot-only', type=Path, default=None, metavar='NPZ',
        help='regenerate boxplots (with means) from saved .npz',
    )
    p.add_argument(
        '--plot-only-all', action='store_true',
        help='regenerate boxplots for every data/bridge_seed_sweep*.npz',
    )
    p.add_argument(
        '--oscillation', action='store_true',
        help=(
            f'interactive bridged defaults (ρ={OSCILLATION_BRIDGE_RHO}, '
            f'w={OSCILLATION_BRIDGE_W}, Δω={OSCILLATION_DOMEGA}) + r(t) plot'
        ),
    )
    p.add_argument(
        '--plot-r-trace', action='store_true',
        help='also plot full-network r(t) for seed 0',
    )
    p.add_argument(
        '--r-trace-only', type=Path, default=None, metavar='NPZ',
        help='plot full-network r(t) from .npz params (no simulation)',
    )
    return p.parse_args()


def main():
    args = parse_args()
    n = args.n - (args.n % 2)
    n_seeds = 5 if args.quick else args.n_seeds
    base_seed = args.base_seed if args.base_seed is not None else fresh_seed()
    n_t = 120 if args.quick else 300
    t_eval = np.linspace(0.0, T_END, n_t)

    plots_dir = Path('plots')
    data_dir = Path('data')
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.r_trace_only is not None:
        meta = load_results_npz(args.r_trace_only)
        n = meta['n']
        domega = meta['domega']
        A, n0, omega, theta_0 = setup_trial(
            meta['base_seed'], n, meta['bridge_rho'], meta['bridge_w'], domega,
        )
        stem = Path(args.r_trace_only).stem
        for path in make_r_trace_plots(
            A, n0, omega, theta_0, meta['k'], meta['q'], t_eval, plots_dir, stem,
            meta['bridge_rho'], meta['bridge_w'], meta['k'], meta['base_seed'],
            domega=domega,
        ).values():
            print(f'saved {path}')
        return

    if args.plot_only_all:
        for npz_path in sorted(data_dir.glob('bridge_seed_sweep*.npz')):
            meta = load_results_npz(npz_path)
            stem = npz_path.stem
            path = plot_boxplot_results(
                meta['results'], plots_dir, stem, meta['n_seeds'],
                meta['bridge_rho'], meta['bridge_w'], meta['q'], meta['k'],
                meta['domega'],
            )
            print(f'saved {path}')
        return

    if args.plot_only is not None:
        meta = load_results_npz(args.plot_only)
        stem = Path(args.plot_only).stem
        path = plot_boxplot_results(
            meta['results'], plots_dir, stem, meta['n_seeds'],
            meta['bridge_rho'], meta['bridge_w'], meta['q'], meta['k'],
            meta['domega'],
        )
        print(f'saved {path}')
        return

    if args.oscillation:
        args.bridge_rho = OSCILLATION_BRIDGE_RHO
        args.bridge_w = OSCILLATION_BRIDGE_W
        args.domega = OSCILLATION_DOMEGA
        args.plot_r_trace = True

    if args.sweep_bridge:
        sweep_results = []
        for label, bridge_rho, bridge_w in BRIDGE_CONFIGS:
            results = run_one_config(
                n, n_seeds, bridge_rho, bridge_w, args.q, args.k,
                base_seed, t_eval, label=label, domega=args.domega,
                plot_r_trace=args.plot_r_trace, plots_dir=plots_dir,
            )
            save_config_results(
                results, plots_dir, data_dir, n, n_seeds,
                bridge_rho, bridge_w, args.q, args.k, base_seed, args.domega,
            )
            sweep_results.append({
                'label': label,
                'bridge_rho': bridge_rho,
                'bridge_w': bridge_w,
                'results': results,
            })
        print_compare_table(sweep_results)
        save_compare_plot(sweep_results, plots_dir, n, n_seeds, args.q, args.k)
        return

    results = run_one_config(
        n, n_seeds, args.bridge_rho, args.bridge_w, args.q, args.k, base_seed, t_eval,
        domega=args.domega, plot_r_trace=args.plot_r_trace, plots_dir=plots_dir,
    )
    save_config_results(
        results, plots_dir, data_dir, n, n_seeds,
        args.bridge_rho, args.bridge_w, args.q, args.k, base_seed, args.domega,
    )


if __name__ == '__main__':
    main()
