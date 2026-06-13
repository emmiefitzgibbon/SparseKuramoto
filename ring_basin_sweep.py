"""
Ring basin sweep — Wiley, Strogatz & Girvan (Chaos 2006) methodology.

For each ring k-NN graph (full vs ER-sparse), sample random initial phases,
integrate the paper Kuramoto model (identical ω, K=1, unnormalized coupling),
and classify final states by winding number q. Compare basin statistics:
  - fraction syncing (q=0, r > threshold)
  - discrete-Gaussian spread σ in q
  - paper prediction σ ≈ 0.19√(N/k) − 0.11
  - agreement between full and sparse from the same IC

Run: python ring_basin_sweep.py
     python ring_basin_sweep.py --quick
     python ring_basin_sweep.py --heterogeneous
     python ring_basin_sweep.py --falloff
     python ring_basin_sweep.py --plot-only data/ring_basin_sweep_falloff_n=400.npz
     python ring_basin_sweep.py --append --k 145 160 --N 400 --falloff
     python ring_basin_sweep.py --ring-sparse-only --N 400 --kn 0.1 0.2
     python ring_basin_sweep.py --sparsify-methods er ring_farthest ring_every_other
     python ring_basin_sweep.py --plot-comparison data/ring_basin_sweep_ring_sparse_n=400.npz data/ring_basin_sweep_n=400.npz data/ring_basin_sweep_weight_based_n=400.npz
     python ring_basin_sweep.py --plot-comparison data/ring_basin_sweep_ring_sparse_n=400.npz --compare-methods ring_odd
     python ring_basin_sweep.py --plot-comparison data/ring_basin_sweep_ring_sparse_n=400.npz data/ring_basin_sweep_n=400.npz data/ring_basin_sweep_weight_based_n=400.npz --compare-methods ring_odd er weight_based
     python ring_basin_sweep.py --N 400 --sparsify-methods weight_based --match-k-from data/ring_basin_sweep_ring_sparse_n=400.npz --n-ic 50 --n-sparsify-seeds 1
     python ring_basin_sweep.py --ring-sparse-only --N 400 --append --interpolate-k --n-ic 50
     python ring_basin_sweep.py --N 400 --append --match-k-from data/ring_basin_sweep_ring_sparse_n=400.npz --n-ic 50
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.stats import norm

# --- paper-aligned defaults ---
PAPER_K = 0.5
OMEGA = 5.0
SPARSIFY_Q = 0.25
SYNC_R_THRESH = 0.95
SIGMA_SLOPE = 0.19
SIGMA_INTERCEPT = -0.11
HE_WEIGHT_STD = 1.0
DEFAULT_KN = [k / 80 for k in (3, 5, 8, 10, 12, 15, 18, 21, 25)]
# k/n at N=80: [0.0375, 0.0625, 0.1, 0.125, 0.15, 0.1875, 0.225, 0.2625, 0.3125]
SPARSIFY_METHOD_CHOICES = (
    'er', 'weight_based', 'ring_farthest', 'ring_every_other', 'ring_odd',
)
STOCHASTIC_METHODS = frozenset({'er', 'weight_based'})
SPARSE_COLORS = {
    'er': 'C1', 'weight_based': 'C5', 'ring_farthest': 'C2',
    'ring_every_other': 'C3', 'ring_odd': 'C4',
}
SPARSE_MARKERS = {
    'er': 's', 'weight_based': 'p', 'ring_farthest': '^',
    'ring_every_other': 'v', 'ring_odd': 'D',
}
SPARSE_LABELS = {
    'er': 'sparse (ER)',
    'weight_based': 'weight-based',
    'ring_farthest': 'ring farthest',
    'ring_every_other': 'ring every other',
    'ring_odd': 'ring odd',
}


def k_from_kn(n, kn_values):
    return sorted({max(1, round(kn * n)) for kn in kn_values})


def parse_int_list(tokens):
    """accept space- or comma-separated integers."""
    out = []
    for tok in tokens:
        for part in str(tok).split(','):
            part = part.strip()
            if part:
                out.append(int(float(part)))
    return out


def parse_float_list(tokens):
    """accept space- or comma-separated floats."""
    out = []
    for tok in tokens:
        for part in str(tok).split(','):
            part = part.strip()
            if part:
                out.append(float(part))
    return out


def midpoint_k_values(k_values, n):
    """integer k halfway between consecutive sorted anchor values."""
    ks = sorted(set(int(k) for k in k_values))
    out = []
    for a, b in zip(ks[:-1], ks[1:]):
        mid = max(1, round(0.5 * (a + b)))
        if a < mid < b and mid < n:
            out.append(mid)
    return sorted(out)


def resolve_k_values(n, *, k_args=None, kn_args=None):
    if k_args is not None:
        k_values = sorted({max(1, int(k)) for k in parse_int_list(k_args)})
    else:
        kn = DEFAULT_KN if kn_args is None else parse_float_list(kn_args)
        k_values = k_from_kn(n, kn)
    bad = [k for k in k_values if k >= n]
    if bad:
        raise ValueError(f'k must be < N={n}; got {bad}')
    return k_values


def generate_ring_matrix(
    n, k, heterogeneous=False, falloff=False, rng=None, weight_std=HE_WEIGHT_STD,
):
    offsets = list(range(1, int(k) + 1))
    A = nx.to_numpy_array(nx.circulant_graph(n, offsets), dtype=float)
    if not heterogeneous and not falloff:
        return A
    ei, ej = np.where(np.triu(A, 1))
    if falloff:
        d = np.minimum(np.abs(ei - ej), n - np.abs(ei - ej))
        w = 1.0 / d
    else:
        w = rng.lognormal(0, weight_std, len(ei))
    A = np.zeros((n, n))
    A[ei, ej] = w
    A[ej, ei] = w
    return A


def edges_from_A(A):
    ei, ej = np.where(np.triu(A, 1))
    return ei, ej, A[ei, ej].astype(float)


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


def sparsify_er(edge_i, edge_j, we, pe, q_frac, rng):
    s = max(1, int(q_frac * len(edge_i)))
    n = int(np.max(np.concatenate([edge_i, edge_j])) + 1)
    A = np.zeros((n, n))
    for idx in rng.choice(len(edge_i), s, replace=True, p=pe):
        w = we[idx] / (s * pe[idx])
        A[edge_i[idx], edge_j[idx]] += w
        A[edge_j[idx], edge_i[idx]] += w
    return A


def sparsify_weight(edge_i, edge_j, we, q_frac, rng):
    pe = we / we.sum()
    s = max(1, int(q_frac * len(edge_i)))
    n = int(np.max(np.concatenate([edge_i, edge_j])) + 1)
    A = np.zeros((n, n))
    for idx in rng.choice(len(edge_i), s, replace=True, p=pe):
        w = we[idx] / (s * pe[idx])
        A[edge_i[idx], edge_j[idx]] += w
        A[edge_j[idx], edge_i[idx]] += w
    return A


def ring_farthest_sparsification(A, q_frac):
    # keep the k/2 neighbor shells farthest from the center (distances > k//2)
    n = A.shape[0]
    ei, ej = np.where(np.triu(A, 1))
    d = np.minimum(np.abs(ei - ej), n - np.abs(ei - ej))
    k_max = int(d.max())
    keep = d > k_max // 2
    out = np.zeros((n, n))
    out[ei[keep], ej[keep]] = A[ei[keep], ej[keep]]
    out[ej[keep], ei[keep]] = A[ej[keep], ei[keep]]
    return out


def ring_every_other_sparsification(A, q_frac):
    # keep even-distance shells only (2, 4, 6, ...); no distance-1 edges
    n = A.shape[0]
    ei, ej = np.where(np.triu(A, 1))
    d = np.minimum(np.abs(ei - ej), n - np.abs(ei - ej))
    k_max = int(d.max())
    keep_distances = set(range(2, k_max + 1, 2))
    keep = np.isin(d, list(keep_distances))
    out = np.zeros((n, n))
    out[ei[keep], ej[keep]] = A[ei[keep], ej[keep]]
    out[ej[keep], ei[keep]] = A[ej[keep], ei[keep]]
    return out


def ring_odd_sparsification(A, q_frac):
    # keep odd-distance shells only (1, 3, 5, ...); couples even ↔ odd, not within parity
    n = A.shape[0]
    ei, ej = np.where(np.triu(A, 1))
    d = np.minimum(np.abs(ei - ej), n - np.abs(ei - ej))
    k_max = int(d.max())
    keep_distances = set(range(1, k_max + 1, 2))
    keep = np.isin(d, list(keep_distances))
    out = np.zeros((n, n))
    out[ei[keep], ej[keep]] = A[ei[keep], ej[keep]]
    out[ej[keep], ei[keep]] = A[ej[keep], ei[keep]]
    return out


def resolve_sparsify_methods(args):
    if args.ring_sparse_only and args.sparsify_methods is not None:
        raise ValueError('use --ring-sparse-only or --sparsify-methods, not both')
    if args.ring_sparse_only:
        return ['ring_farthest', 'ring_every_other', 'ring_odd']
    if args.sparsify_methods is not None:
        return list(args.sparsify_methods)
    return ['er']


def sparse_key_prefix(method):
    return 'sparse' if method == 'er' else method


def sparse_q_key(method):
    p = sparse_key_prefix(method)
    return 'q_sparse_all' if p == 'sparse' else f'q_{p}_all'


def sparse_stat_key(method, stat):
    p = sparse_key_prefix(method)
    if p == 'sparse':
        return {
            'f_sync': 'f_sync_sparse',
            'sigma': 'sigma_sparse',
            'match_rate': 'match_rate',
            'mean_abs_dq': 'mean_abs_dq',
        }[stat]
    return f'{stat}_{p}'


def npz_agg_keys(sparsify_methods):
    keys = ['q_full_all', 'f_sync_full', 'sigma_full']
    for method in sparsify_methods:
        keys.extend([
            sparse_q_key(method),
            sparse_stat_key(method, 'f_sync'),
            sparse_stat_key(method, 'sigma'),
            sparse_stat_key(method, 'match_rate'),
            sparse_stat_key(method, 'mean_abs_dq'),
        ])
    return keys


def sparsify_tag(sparsify_methods):
    if sparsify_methods == ['er']:
        return ''
    ring_methods = {'ring_farthest', 'ring_every_other', 'ring_odd'}
    if set(sparsify_methods) <= ring_methods and not STOCHASTIC_METHODS & set(sparsify_methods):
        return '_ring_sparse'
    return '_' + '_'.join(sparsify_methods)


def build_sparse_graphs(A, er_data, sparsify_methods, n_sparsify_seeds, base_seed):
    er_i, er_j, er_w, er_pe = er_data
    wt_i, wt_j = np.where(np.triu(A, 1))
    wt_w = A[wt_i, wt_j].astype(float)
    graphs = {}
    for method in sparsify_methods:
        if method == 'er':
            graphs[method] = []
            for si in range(n_sparsify_seeds):
                sp_rng = np.random.default_rng(base_seed + 1000 + si)
                graphs[method].append(
                    sparsify_er(er_i, er_j, er_w, er_pe, SPARSIFY_Q, sp_rng),
                )
        elif method == 'weight_based':
            graphs[method] = []
            for si in range(n_sparsify_seeds):
                sp_rng = np.random.default_rng(base_seed + 2000 + si)
                graphs[method].append(
                    sparsify_weight(wt_i, wt_j, wt_w, SPARSIFY_Q, sp_rng),
                )
        elif method == 'ring_farthest':
            graphs[method] = [ring_farthest_sparsification(A, SPARSIFY_Q)]
        elif method == 'ring_every_other':
            graphs[method] = [ring_every_other_sparsification(A, SPARSIFY_Q)]
        elif method == 'ring_odd':
            graphs[method] = [ring_odd_sparsification(A, SPARSIFY_Q)]
        else:
            raise ValueError(f'unknown sparsify method: {method}')
    return graphs


def kuramoto_rhs_paper(t, theta, omega, ei, ej, w):
    """θ̇_i = ω + K Σ_j sin(θ_j − θ_i), K absorbed into edge weights."""
    coupling = np.zeros_like(theta)
    np.add.at(coupling, ei, w * np.sin(theta[ej] - theta[ei]))
    np.add.at(coupling, ej, w * np.sin(theta[ei] - theta[ej]))
    return omega + PAPER_K * coupling


def estimate_winding(theta):
    dtheta = np.angle(np.exp(1j * (np.roll(theta, -1) - theta)))
    return int(np.round(np.sum(dtheta) / (2 * np.pi)))


def order_param(theta):
    z = np.mean(np.exp(1j * theta))
    return float(np.abs(z)), z


def classify_final(theta_final, t_final):
    """co-rotating frame; sync only if r high; else report winding q."""
    theta = theta_final - OMEGA * t_final
    r, _ = order_param(theta)
    q = estimate_winding(theta)
    if r >= SYNC_R_THRESH:
        return 0, r
    return q, r


def integrate(theta0, ei, ej, w, t_end, solver_kw):
    sol = solve_ivp(
        kuramoto_rhs_paper, (0.0, t_end), theta0,
        args=(np.full(len(theta0), OMEGA), ei, ej, w),
        **solver_kw,
    )
    return sol.y[:, -1], t_end


def discrete_gaussian_pmf(q_support, sigma):
    pmf = norm.cdf((q_support + 0.5) / sigma) - norm.cdf((q_support - 0.5) / sigma)
    pmf = np.maximum(pmf, 0.0)
    s = pmf.sum()
    return pmf / s if s > 0 else pmf


def fit_sigma_mle(q_samples):
    """MLE for discrete Gaussian on integer q (paper Fig. 2 fit)."""
    q_arr = np.asarray(q_samples, dtype=int)
    if len(q_arr) < 3:
        return float(np.std(q_arr))
    if np.unique(q_arr).size == 1:
        return 0.0

    def neg_log_lik(log_sigma):
        sigma = np.exp(log_sigma)
        if sigma < 0.05:
            return 1e12
        q_lo, q_hi = q_arr.min(), q_arr.max()
        support = np.arange(q_lo, q_hi + 1)
        pmf = discrete_gaussian_pmf(support, sigma)
        idx = q_arr - q_lo
        probs = np.maximum(pmf[idx], 1e-15)
        return -float(np.sum(np.log(probs)))

    res = minimize_scalar(neg_log_lik, bounds=(-2, 3), method='bounded')
    return float(np.exp(res.x))


def empirical_kn_onset(kn, probs, eps=0.0):
    """highest k/N in the sweep where P(|q|) is still nonzero."""
    kn = np.asarray(kn, dtype=float)
    probs = np.asarray(probs, dtype=float)
    hit = kn[probs > eps]
    if hit.size == 0:
        return None
    return float(hit.max())


def empirical_kn_emergence(kn, probs, eps=0.0):
    """lowest k/N in the sweep where P(|q|) is nonzero."""
    kn = np.asarray(kn, dtype=float)
    probs = np.asarray(probs, dtype=float)
    hit = kn[probs > eps]
    if hit.size == 0:
        return None
    return float(hit.min())


def critical_kn_for_attractor(abs_q, kn, probs, eps=0.005):
    """k/N threshold for attractor transition: sync emerges, |q|>0 vanishes."""
    if abs_q == 0:
        return empirical_kn_emergence(kn, probs, eps=eps)
    return empirical_kn_onset(kn, probs, eps=eps)


def paper_sigma(n, k):
    return max(SIGMA_SLOPE * np.sqrt(n / k) + SIGMA_INTERCEPT, 0.01)


def _aggregate_method_records(records):
    keys = (
        'f_sync_sparse', 'sigma_sparse', 'sigma_sample_sparse',
        'match_rate', 'mean_abs_dq',
    )
    out = {}
    for key in keys:
        vals = [r[key] for r in records]
        out[key] = float(np.mean(vals))
        out[f'{key}_std'] = float(np.std(vals))
    out['q_sparse_all'] = np.concatenate([r['q_sparse'] for r in records])
    return out


def run_condition(
    n, k, n_ic, n_sparsify_seeds, base_seed, t_end, solver_kw,
    sparsify_methods,
    heterogeneous=False, falloff=False, weight_std=HE_WEIGHT_STD,
    *, ic_skip=0,
):
    rng_weights = np.random.default_rng(base_seed + 7)
    A = generate_ring_matrix(
        n, k, heterogeneous=heterogeneous, falloff=falloff,
        rng=rng_weights, weight_std=weight_std,
    )
    ei_f, ej_f, w_f = edges_from_A(A)
    er_data = precompute_er(A)
    sparse_graphs = build_sparse_graphs(
        A, er_data, sparsify_methods, n_sparsify_seeds, base_seed,
    )
    rng_master = np.random.default_rng(base_seed)
    for _ in range(ic_skip):
        rng_master.uniform(0, 2 * np.pi, n)

    method_records = {method: [] for method in sparsify_methods}
    for method in sparsify_methods:
        for si, A_s in enumerate(sparse_graphs[method]):
            ei_s, ej_s, w_s = edges_from_A(A_s)
            q_full, q_sparse, r_full, r_sparse = [], [], [], []
            for _ in range(n_ic):
                theta0 = rng_master.uniform(0, 2 * np.pi, n)
                th_f, t_f = integrate(theta0, ei_f, ej_f, w_f, t_end, solver_kw)
                th_s, t_s = integrate(theta0, ei_s, ej_s, w_s, t_end, solver_kw)
                qf, rf = classify_final(th_f, t_f)
                qs, rs = classify_final(th_s, t_s)
                q_full.append(qf)
                q_sparse.append(qs)
                r_full.append(rf)
                r_sparse.append(rs)

            q_full = np.array(q_full)
            q_sparse = np.array(q_sparse)
            r_full = np.array(r_full)
            r_sparse = np.array(r_sparse)
            method_records[method].append(dict(
                sparsify_seed=base_seed + 1000 + si,
                q_full=q_full,
                q_sparse=q_sparse,
                r_full=r_full,
                r_sparse=r_sparse,
                f_sync_full=float(np.mean(r_full >= SYNC_R_THRESH)),
                f_sync_sparse=float(np.mean(r_sparse >= SYNC_R_THRESH)),
                sigma_full=fit_sigma_mle(q_full),
                sigma_sparse=fit_sigma_mle(q_sparse),
                sigma_sample_full=float(np.std(q_full)),
                sigma_sample_sparse=float(np.std(q_sparse)),
                match_rate=float(np.mean(q_full == q_sparse)),
                mean_abs_dq=float(np.mean(np.abs(q_full - q_sparse))),
            ))
    return method_records


def aggregate_records(method_records, sparsify_methods):
    first_records = method_records[sparsify_methods[0]]
    out = {
        'f_sync_full': float(np.mean([r['f_sync_full'] for r in first_records])),
        'sigma_full': float(np.mean([r['sigma_full'] for r in first_records])),
        'q_full_all': np.concatenate([r['q_full'] for r in first_records]),
    }
    for method in sparsify_methods:
        agg = _aggregate_method_records(method_records[method])
        out[sparse_q_key(method)] = agg['q_sparse_all']
        out[sparse_stat_key(method, 'f_sync')] = agg['f_sync_sparse']
        out[sparse_stat_key(method, 'sigma')] = agg['sigma_sparse']
        out[sparse_stat_key(method, 'match_rate')] = agg['match_rate']
        out[sparse_stat_key(method, 'mean_abs_dq')] = agg['mean_abs_dq']
    return out


def merge_agg_with_additional(existing, additional, sparsify_methods):
    """concatenate IC batches and recompute basin stats."""
    q_full = np.concatenate([existing['q_full_all'], additional['q_full_all']])
    out = {
        'q_full_all': q_full,
        'f_sync_full': float(np.mean(q_full == 0)),
        'sigma_full': fit_sigma_mle(q_full),
    }
    for method in sparsify_methods:
        qk = sparse_q_key(method)
        qs_old = existing[qk]
        qs_new = additional[qk]
        qs = np.concatenate([qs_old, qs_new])
        qf = np.concatenate([existing['q_full_all'], additional['q_full_all']])
        out[qk] = qs
        out[sparse_stat_key(method, 'f_sync')] = float(np.mean(qs == 0))
        out[sparse_stat_key(method, 'sigma')] = fit_sigma_mle(qs)
        out[sparse_stat_key(method, 'match_rate')] = float(np.mean(qf == qs))
        out[sparse_stat_key(method, 'mean_abs_dq')] = float(
            np.mean(np.abs(qf - qs)),
        )
    return out


def _file_stem(*, n, heterogeneous=False, falloff=False, sparsify_methods=None):
    if heterogeneous:
        base = 'ring_basin_sweep_het'
    elif falloff:
        base = 'ring_basin_sweep_falloff'
    else:
        base = 'ring_basin_sweep'
    methods = sparsify_methods or ['er']
    return f'{base}{sparsify_tag(methods)}_n={n}'


def _weight_tag(*, heterogeneous=False, falloff=False):
    if heterogeneous:
        return '  het. weights'
    if falloff:
        return '  falloff (1/d)'
    return ''


def _plots_dir(*, n, heterogeneous=False, falloff=False, sparsify_methods=None):
    methods = sparsify_methods or ['er']
    tag = sparsify_tag(methods)
    if heterogeneous:
        return Path(f'plots/heterogeneous_weights_n={n}{tag}')
    if falloff:
        return Path(f'plots/falloff_weights_n={n}{tag}')
    return Path(f'plots/homogeneous_weights_n={n}{tag}')


def n_ic_by_k_from_npz(d, k_values):
    if 'n_ic_per_k' in d:
        return {int(k): int(n) for k, n in zip(d['k_values'], d['n_ic_per_k'])}
    n_ic = int(d['n_ic'])
    return {int(k): n_ic for k in k_values}


def _sample_banner(
    n_ic, n_sparsify_seeds, n_k_values, sparsify_methods, n_ic_by_k=None,
):
    if n_ic_by_k:
        ic_vals = sorted(set(n_ic_by_k.values()))
        if len(ic_vals) > 1:
            return (
                f'n_ic={ic_vals[0]}–{ic_vals[-1]} per k; {n_k_values} k values'
            )
        n_ic = ic_vals[0]
    parts = [f'full: {n_ic} ICs/k']
    for method in sparsify_methods:
        if method not in STOCHASTIC_METHODS:
            continue
        label = SPARSE_LABELS[method]
        if n_sparsify_seeds > 1:
            parts.append(
                f'{label}: {n_ic * n_sparsify_seeds} trials/k '
                f'({n_ic} ICs × {n_sparsify_seeds} seeds)',
            )
        else:
            parts.append(f'{label}: {n_ic} ICs/k')
    ring_methods = [m for m in sparsify_methods if m not in STOCHASTIC_METHODS]
    if ring_methods:
        labels = ', '.join(SPARSE_LABELS[m] for m in ring_methods)
        parts.append(f'{labels}: {n_ic} ICs/k each (deterministic)')
    return ';  '.join(parts) + f';  {n_k_values} k values'


def _sync_criteria_banner():
    return (
        f'sync: r ≥ {SYNC_R_THRESH} in co-rotating frame (θᵢ − Ωt),  '
        f'r = |⟨e^{{iθ}}⟩|'
    )


def _fig_suptitle(
    fig, title_line, n, n_ic, n_sparsify_seeds, n_k_values,
    sparsify_methods,
    *, heterogeneous=False, falloff=False, extra_lines=(), n_ic_by_k=None,
):
    wt = _weight_tag(heterogeneous=heterogeneous, falloff=falloff)
    q_note = (
        f'q_sparsify={SPARSIFY_Q}'
        if any(m in STOCHASTIC_METHODS for m in sparsify_methods)
        else 'ring sparse (deterministic)'
    )
    lines = [
        f'{title_line}  N={n}  K={PAPER_K}  {q_note}{wt}',
        _sample_banner(
            n_ic, n_sparsify_seeds, n_k_values, sparsify_methods, n_ic_by_k=n_ic_by_k,
        ),
        *extra_lines,
    ]
    fig.suptitle('\n'.join(lines), fontsize=9)


def _save_fig(fig, path, tight_top=0.86):
    fig.tight_layout(rect=[0, 0, 1, tight_top])
    fig.savefig(path, dpi=120, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def q_probability_matrix(q_samples, q_support):
    q_arr = np.asarray(q_samples, dtype=int)
    return np.array([np.mean(q_arr == q) for q in q_support])


def _agg_from_npz_arrays(d, k, sparsify_methods):
    p = f'k{k}'
    qf = d[f'{p}_q_full_all']
    out = {
        'q_full_all': qf,
        'sigma_full': fit_sigma_mle(qf),
        'f_sync_full': float(d[f'{p}_f_sync_full']),
    }
    for method in sparsify_methods:
        qk = sparse_q_key(method)
        out[qk] = d[f'{p}_{qk}']
        out[sparse_stat_key(method, 'sigma')] = fit_sigma_mle(d[f'{p}_{qk}'])
        out[sparse_stat_key(method, 'f_sync')] = float(
            d[f'{p}_{sparse_stat_key(method, "f_sync")}'],
        )
        out[sparse_stat_key(method, 'match_rate')] = float(
            d[f'{p}_{sparse_stat_key(method, "match_rate")}'],
        )
        out[sparse_stat_key(method, 'mean_abs_dq')] = float(
            d[f'{p}_{sparse_stat_key(method, "mean_abs_dq")}'],
        )
    return out


def _sparsify_methods_from_npz(d, stem=''):
    if 'sparsify_methods' in d:
        return [str(m) for m in d['sparsify_methods']]
    if '_ring_sparse' in stem or 'ring_farthest' in stem:
        return ['ring_farthest', 'ring_every_other']
    return ['er']


def load_agg_from_npz(npz_path):
    global PAPER_K
    d = np.load(npz_path)
    if 'PAPER_K' in d:
        PAPER_K = float(d['PAPER_K'])
    n = int(d['N'])
    k_values = [int(k) for k in d['k_values']]
    stem = Path(npz_path).stem
    het = bool(d['heterogeneous']) if 'heterogeneous' in d else '_het' in stem
    falloff = bool(d['falloff']) if 'falloff' in d else '_falloff' in stem
    sparsify_methods = _sparsify_methods_from_npz(d, stem)
    agg_by_k = {
        k: _agg_from_npz_arrays(d, k, sparsify_methods) for k in k_values
    }
    n_ic = int(d['n_ic'])
    n_sparsify_seeds = int(d['n_sparsify_seeds'])
    n_ic_by_k = n_ic_by_k_from_npz(d, k_values)
    return (
        n, k_values, agg_by_k, n_ic, n_sparsify_seeds, n_ic_by_k,
        het, falloff, sparsify_methods,
    )


def merge_agg_for_comparison(ring_npz, *extra_npzs):
    """merge ring-sparse sweep with stochastic sparse q arrays at shared k."""
    (
        n, k_ring, agg_ring, n_ic, _, n_ic_by_k,
        het, falloff, ring_methods,
    ) = load_agg_from_npz(ring_npz)
    k_values = set(k_ring)
    sparsify_methods = list(ring_methods)
    extra_methods = []

    for extra_npz in extra_npzs:
        (
            n_x, k_x, agg_x, _, _, _,
            het_x, falloff_x, methods_x,
        ) = load_agg_from_npz(extra_npz)
        if n != n_x:
            raise ValueError(f'N mismatch: {n} vs {n_x} ({extra_npz})')
        if het != het_x or falloff != falloff_x:
            raise ValueError(f'weight mode must match ({extra_npz})')
        stochastic = [m for m in methods_x if m in STOCHASTIC_METHODS]
        if len(stochastic) != 1:
            raise ValueError(
                f'{extra_npz} must contain exactly one of {sorted(STOCHASTIC_METHODS)}',
            )
        method = stochastic[0]
        if method in sparsify_methods:
            raise ValueError(f'duplicate sparsify method {method}')
        missing = sorted(set(k_ring) - set(k_x))
        if missing:
            print(f'note: no {SPARSE_LABELS[method]} data at k={missing}')
        k_values &= set(k_x)
        extra_methods.append((method, agg_x))

    if not k_values:
        raise ValueError('no shared k across comparison files')
    k_values = sorted(k_values)

    agg_by_k = {}
    for k in k_values:
        agg_by_k[k] = dict(agg_ring[k])
        n_ic_k = n_ic_by_k.get(k, n_ic)
        for method, agg_x in extra_methods:
            q = np.asarray(agg_x[k][sparse_q_key(method)])
            if len(q) > n_ic_k:
                q = q[:n_ic_k]
            agg_by_k[k][sparse_q_key(method)] = q
    sparsify_methods.extend(m for m, _ in extra_methods)

    return (
        n, k_values, agg_by_k, n_ic, 1, n_ic_by_k,
        het, falloff, sparsify_methods,
    )


def load_npz_for_append(npz_path):
    """Load full per-k arrays from an existing sweep file."""
    d = np.load(npz_path)
    stem = Path(npz_path).stem
    sparsify_methods = _sparsify_methods_from_npz(d, stem)
    meta = {
        'N': int(d['N']),
        'n_ic': int(d['n_ic']),
        'n_sparsify_seeds': int(d['n_sparsify_seeds']),
        'heterogeneous': bool(d['heterogeneous']) if 'heterogeneous' in d else '_het' in stem,
        'falloff': bool(d['falloff']) if 'falloff' in d else '_falloff' in stem,
        'weight_std': float(d['weight_std']) if 'weight_std' in d else 0.0,
        'sparsify_methods': sparsify_methods,
    }
    k_values = [int(k) for k in d['k_values']]
    agg_keys = npz_agg_keys(sparsify_methods)
    agg_by_k = {}
    for k in k_values:
        p = f'k{k}'
        agg_by_k[k] = {key: d[f'{p}_{key}'] for key in agg_keys}
    meta['n_ic_by_k'] = n_ic_by_k_from_npz(d, k_values)
    return meta, k_values, agg_by_k


def save_agg_to_npz(
    npz_path, n, k_values, agg_by_k, n_ic, n_sparsify_seeds,
    sparsify_methods,
    *, heterogeneous=False, falloff=False, weight_std=0.0, n_ic_by_k=None,
):
    if n_ic_by_k is None:
        n_ic_by_k = {k: n_ic for k in k_values}
    save = dict(
        N=n, k_values=np.array(k_values), PAPER_K=PAPER_K, SPARSIFY_Q=SPARSIFY_Q,
        n_ic=n_ic, n_ic_per_k=np.array([n_ic_by_k[k] for k in k_values], dtype=int),
        n_sparsify_seeds=n_sparsify_seeds,
        heterogeneous=heterogeneous, falloff=falloff, weight_std=weight_std,
        sparsify_methods=np.array(sparsify_methods),
    )
    agg_keys = npz_agg_keys(sparsify_methods)
    for k in k_values:
        pref = f'k{k}'
        for key in agg_keys:
            save[f'{pref}_{key}'] = agg_by_k[k][key]
    np.savez_compressed(npz_path, **save)


def _q_prob_data(n, k_values, agg_by_k, sparsify_methods):
    q_arrays = []
    for k in k_values:
        q_arrays.append(agg_by_k[k]['q_full_all'])
        for method in sparsify_methods:
            q_arrays.append(agg_by_k[k][sparse_q_key(method)])
    q_global = np.concatenate(q_arrays)
    q_lo, q_hi = int(q_global.min()), int(q_global.max())
    q_support = np.arange(q_lo, q_hi + 1)
    kn = np.array([k / n for k in k_values])
    mats = {'q_full_all': np.zeros((len(q_support), len(k_values)))}
    for j, k in enumerate(k_values):
        mats['q_full_all'][:, j] = q_probability_matrix(
            agg_by_k[k]['q_full_all'], q_support,
        )
    for method in sparsify_methods:
        key = sparse_q_key(method)
        mat = np.zeros((len(q_support), len(k_values)))
        for j, k in enumerate(k_values):
            mat[:, j] = q_probability_matrix(agg_by_k[k][key], q_support)
        mats[key] = mat
    return kn, q_support, mats


def _collapse_q_sign(q_support, mat):
    """P(|q|=m) = sum of P(q=+m) and P(q=-m); q=0 unchanged."""
    abs_qs = sorted(set(int(abs(q)) for q in q_support))
    out = np.zeros((len(abs_qs), mat.shape[1]))
    for i, aq in enumerate(abs_qs):
        out[i] = mat[np.abs(q_support) == aq].sum(axis=0)
    return abs_qs, out


def _q_prob_data_abs(n, k_values, agg_by_k, sparsify_methods):
    kn, q_support, mats = _q_prob_data(n, k_values, agg_by_k, sparsify_methods)
    abs_mats = {}
    abs_qs = None
    for key, mat in mats.items():
        abs_qs, abs_mats[key] = _collapse_q_sign(q_support, mat)
    return kn, abs_qs, abs_mats


def plot_q_curves_by_kn(
    n, k_values, agg_by_k, out_dir, n_ic, n_sparsify_seeds,
    sparsify_methods,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
    *, file_stem=None,
):
    kn, abs_qs, mats = _q_prob_data_abs(n, k_values, agg_by_k, sparsify_methods)
    sparse_keys = [sparse_q_key(m) for m in sparsify_methods]
    active = []
    for i, aq in enumerate(abs_qs):
        peak = mats['q_full_all'][i].max()
        peak = max(peak, max(mats[k][i].max() for k in sparse_keys))
        if peak >= 0.005:
            active.append(aq)
    if not active:
        active = list(abs_qs)

    n_panels = len(active)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False)

    for idx, aq in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        qi = int(aq - abs_qs[0])
        ax.plot(kn, 100 * mats['q_full_all'][qi], 'o-', color='C0', label='full', ms=4)
        for method in sparsify_methods:
            key = sparse_q_key(method)
            ax.plot(
                kn, 100 * mats[key][qi], f'{SPARSE_MARKERS[method]}-',
                color=SPARSE_COLORS[method], label=SPARSE_LABELS[method], ms=4,
            )
        ax.set_title(f'|q|={aq}' if aq else 'q=0')
        ax.set_xlabel('k/N')
        ax.set_ylabel('% of trials')
        ymax = 100 * mats['q_full_all'][qi].max() * 1.15 + 1
        ymax = max(ymax, 100 * max(mats[k][qi].max() for k in sparse_keys) * 1.15 + 1)
        ax.set_ylim(0, min(105, ymax))
        if ax.get_ylim()[1] < 5:
            ax.set_ylim(0, 5)
        series = [('q_full_all', 'C0', 'full')]
        for method in sparsify_methods:
            series.append((
                sparse_q_key(method), SPARSE_COLORS[method], SPARSE_LABELS[method],
            ))
        for j, (key, color, tag) in enumerate(series):
            prob = mats[key][qi]
            kc = empirical_kn_onset(kn, prob)
            if kc is not None and kn.min() <= kc <= kn.max():
                ax.axvline(kc, color=color, ls=':', lw=1.4, alpha=0.9, zorder=0)
                ytxt = ax.get_ylim()[1] * (0.92 - 0.14 * j)
                ax.text(
                    kc, ytxt, f'{tag}\n{kc:.3f}', fontsize=5.5, color=color,
                    va='top', ha='center',
                )
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    _fig_suptitle(
        fig, 'P(|q|) vs k/N', n, n_ic, n_sparsify_seeds, len(k_values),
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    path = out_dir / (
        f'{file_stem or _file_stem(n=n, heterogeneous=heterogeneous, falloff=falloff, sparsify_methods=sparsify_methods)}'
        '_q_curves.png'
    )
    _save_fig(fig, path, tight_top=0.88)
    return path


def plot_q_error_from_full(
    n, k_values, agg_by_k, out_dir, n_ic, n_sparsify_seeds,
    sparsify_methods,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
    *, file_stem=None,
):
    """|P_sparse(|q|) − P_full(|q|)| in percentage points for each sparsify method."""
    kn, abs_qs, mats = _q_prob_data_abs(n, k_values, agg_by_k, sparsify_methods)
    sparse_keys = [sparse_q_key(m) for m in sparsify_methods]
    full = mats['q_full_all']
    active = []
    for i, aq in enumerate(abs_qs):
        peak = full[i].max()
        peak = max(peak, max(mats[k][i].max() for k in sparse_keys))
        if peak >= 0.005:
            active.append(aq)
    if not active:
        active = list(abs_qs)

    n_panels = len(active)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False)

    for idx, aq in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        qi = int(aq - abs_qs[0])
        ymax = 0.0
        for method in sparsify_methods:
            key = sparse_q_key(method)
            err = 100 * np.abs(mats[key][qi] - full[qi])
            ymax = max(ymax, err.max())
            ax.plot(
                kn, err, f'{SPARSE_MARKERS[method]}-',
                color=SPARSE_COLORS[method], label=SPARSE_LABELS[method], ms=4,
            )
        ax.set_title(f'|q|={aq}' if aq else 'q=0')
        ax.set_xlabel('k/N')
        ax.set_ylabel('probability gap (%)')
        ax.set_ylim(0, min(105, ymax * 1.15 + 0.5))
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    _fig_suptitle(
        fig, '|P_sparse − P_full| vs k/N', n, n_ic, n_sparsify_seeds, len(k_values),
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    stem = file_stem or _file_stem(
        n=n, heterogeneous=heterogeneous, falloff=falloff,
        sparsify_methods=sparsify_methods,
    )
    path = out_dir / f'{stem}_q_error.png'
    _save_fig(fig, path, tight_top=0.88)
    return path


def _attractor_onset_errors(n, k_values, agg_by_k, sparsify_methods, eps=0.005):
    """|Δ k/N| between sparse and full transition k for each |q|."""
    kn, abs_qs, mats = _q_prob_data_abs(n, k_values, agg_by_k, sparsify_methods)
    full = mats['q_full_all']
    sparse_keys = [sparse_q_key(m) for m in sparsify_methods]
    active = []
    for i, aq in enumerate(abs_qs):
        peak = full[i].max()
        peak = max(peak, max(mats[k][i].max() for k in sparse_keys))
        if peak >= eps:
            active.append(int(aq))
    if not active:
        active = [int(q) for q in abs_qs]

    kn_span = float(kn.max() - kn.min()) if len(kn) > 1 else float(kn.max())
    errors = {m: {} for m in sparsify_methods}
    full_crit = {}
    for aq in active:
        qi = list(abs_qs).index(aq)
        kf = critical_kn_for_attractor(aq, kn, full[qi], eps=eps)
        full_crit[aq] = kf
        for method in sparsify_methods:
            key = sparse_q_key(method)
            ks = critical_kn_for_attractor(aq, kn, mats[key][qi], eps=eps)
            if kf is None and ks is None:
                errors[method][aq] = 0.0
            elif kf is None or ks is None:
                missing = kn.min() if ks is None else kn.max()
                present = ks if kf is None else kf
                errors[method][aq] = min(kn_span, abs(present - missing))
            else:
                errors[method][aq] = abs(ks - kf)
    return active, full_crit, errors


def plot_attractor_onset_error(
    n, k_values, agg_by_k, out_dir, n_ic, n_sparsify_seeds,
    sparsify_methods,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
    *, file_stem=None, eps=0.005,
):
    """|Δ k/N| at attractor transition vs full (sync emerges, |q|>0 vanishes)."""
    active, full_crit, errors = _attractor_onset_errors(
        n, k_values, agg_by_k, sparsify_methods, eps=eps,
    )
    n_q = len(active)
    ncols = min(4, n_q)
    nrows = int(np.ceil(n_q / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.6 * nrows), squeeze=False)

    x = np.arange(len(sparsify_methods))
    for idx, aq in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        vals = [errors[m][aq] for m in sparsify_methods]
        colors = [SPARSE_COLORS[m] for m in sparsify_methods]
        bars = ax.bar(x, vals, color=colors, edgecolor='0.2', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SPARSE_LABELS[m].replace('sparse (ER)', 'ER').replace('ring ', 'r.')
             for m in sparsify_methods],
            rotation=35, ha='right', fontsize=7,
        )
        kf = full_crit[aq]
        title = f'|q|={aq}' if aq else 'q=0 (sync emerges)'
        if kf is not None:
            title += f'\nfull k/N={kf:.3f}'
        ax.set_title(title, fontsize=9)
        ax.set_ylabel('|Δ k/N|')
        ymax = max(vals) if vals else 0.05
        ax.set_ylim(0, ymax * 1.2 + 0.005)
        ax.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars, vals):
            if val > 0.001:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7,
                )

    for idx in range(n_q, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    note = 'sync: min k/N with P>0;  |q|>0: max k/N with P>0'
    _fig_suptitle(
        fig, '|Δ transition k/N| from full', n, n_ic, n_sparsify_seeds, len(k_values),
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
        extra_lines=(note,),
    )
    stem = file_stem or _file_stem(
        n=n, heterogeneous=heterogeneous, falloff=falloff,
        sparsify_methods=sparsify_methods,
    )
    path = out_dir / f'{stem}_onset_error.png'
    _save_fig(fig, path, tight_top=0.88)
    return path


def _mean_onset_error_by_method(n, k_values, agg_by_k, sparsify_methods, eps=0.005):
    """mean |Δ transition k/N| averaged over active |q|."""
    active, _, errors = _attractor_onset_errors(
        n, k_values, agg_by_k, sparsify_methods, eps=eps,
    )
    out = {}
    for method in sparsify_methods:
        vals = [errors[method][aq] for aq in active]
        out[method] = float(np.mean(vals)) if vals else 0.0
    return out


def plot_overall_onset_error(
    n, k_values, agg_by_k, out_dir, n_ic, n_sparsify_seeds,
    sparsify_methods,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
    *, file_stem=None, eps=0.005,
):
    """bar chart of mean |Δ transition k/N| over all |q|."""
    mean_err = _mean_onset_error_by_method(
        n, k_values, agg_by_k, sparsify_methods, eps=eps,
    )
    ranked = sorted(sparsify_methods, key=lambda m: mean_err[m])

    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(ranked)), 3.8))
    labels = [SPARSE_LABELS[m] for m in ranked]
    vals = [mean_err[m] for m in ranked]
    colors = [SPARSE_COLORS[m] for m in ranked]
    bars = ax.bar(labels, vals, color=colors, edgecolor='0.2', linewidth=0.8)
    best = ranked[0]
    bars[0].set_edgecolor('0.1')
    bars[0].set_linewidth(2.0)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9,
        )
    ax.set_ylabel('mean |Δ transition k/N|')
    ax.set_title(f'overall onset error  (best: {SPARSE_LABELS[best]})')
    ax.set_ylim(0, max(vals) * 1.18 + 0.01)
    ax.grid(True, axis='y', alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

    _fig_suptitle(
        fig, 'mean |Δ transition k/N| from full', n, n_ic, n_sparsify_seeds,
        len(k_values), sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
        extra_lines=(
            'sync: min k/N with P>0;  |q|>0: max k/N with P>0',
            'averaged over |q| in sweep',
        ),
    )
    stem = file_stem or _file_stem(
        n=n, heterogeneous=heterogeneous, falloff=falloff,
        sparsify_methods=sparsify_methods,
    )
    path = out_dir / f'{stem}_overall_onset_error.png'
    _save_fig(fig, path, tight_top=0.86)
    return path


def _mean_q_error_by_method(n, k_values, agg_by_k, sparsify_methods):
    """mean |P_sparse(|q|) − P_full(|q|)| in pp, averaged over |q| and k/N."""
    _, abs_qs, mats = _q_prob_data_abs(n, k_values, agg_by_k, sparsify_methods)
    full = mats['q_full_all']
    sparse_keys = [sparse_q_key(m) for m in sparsify_methods]
    active_idx = []
    for i in range(len(abs_qs)):
        peak = full[i].max()
        peak = max(peak, max(mats[k][i].max() for k in sparse_keys))
        if peak >= 0.005:
            active_idx.append(i)
    if not active_idx:
        active_idx = list(range(len(abs_qs)))
    out = {}
    for method in sparsify_methods:
        key = sparse_q_key(method)
        err = 100 * np.abs(mats[key][active_idx] - full[active_idx])
        out[method] = float(np.mean(err))
    return out


def plot_overall_error_bars(
    n, k_values, agg_by_k, out_dir, n_ic, n_sparsify_seeds,
    sparsify_methods,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
    *, file_stem=None,
):
    """bar chart of mean |P_sparse − P_full| (pp) over all |q| and k/N."""
    mean_err = _mean_q_error_by_method(n, k_values, agg_by_k, sparsify_methods)
    ranked = sorted(sparsify_methods, key=lambda m: mean_err[m])

    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(ranked)), 3.8))
    labels = [SPARSE_LABELS[m] for m in ranked]
    vals = [mean_err[m] for m in ranked]
    colors = [SPARSE_COLORS[m] for m in ranked]
    bars = ax.bar(labels, vals, color=colors, edgecolor='0.2', linewidth=0.8)
    best = ranked[0]
    bars[0].set_edgecolor('0.1')
    bars[0].set_linewidth(2.0)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9,
        )
    ax.set_ylabel('mean probability gap (%)')
    ax.set_title(f'overall basin error  (best: {SPARSE_LABELS[best]})')
    ax.set_ylim(0, max(vals) * 1.18 + 0.5)
    ax.grid(True, axis='y', alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

    _fig_suptitle(
        fig, 'mean |P_sparse − P_full|', n, n_ic, n_sparsify_seeds, len(k_values),
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
        extra_lines=('averaged over |q| and k/N in sweep',),
    )
    stem = file_stem or _file_stem(
        n=n, heterogeneous=heterogeneous, falloff=falloff,
        sparsify_methods=sparsify_methods,
    )
    path = out_dir / f'{stem}_overall_error.png'
    _save_fig(fig, path, tight_top=0.86)
    return path


def plot_sigma_vs_kn(
    n, k_values, agg_by_k, out_dir, n_ic, n_sparsify_seeds,
    sparsify_methods,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    kn = [k / n for k in k_values]
    ax.plot(kn, [agg_by_k[k]['sigma_full'] for k in k_values], 'o-', label='full', color='C0')
    for method in sparsify_methods:
        ax.plot(
            kn,
            [agg_by_k[k][sparse_stat_key(method, 'sigma')] for k in k_values],
            f'{SPARSE_MARKERS[method]}-',
            label=SPARSE_LABELS[method],
            color=SPARSE_COLORS[method],
        )
    ax.set_xlabel('k/N')
    ax.set_ylabel('σ  (MLE fit to q distribution)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    _fig_suptitle(
        fig, 'ring basin sweep', n, n_ic, n_sparsify_seeds, len(k_values),
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    path = out_dir / (
        f'{_file_stem(n=n, heterogeneous=heterogeneous, falloff=falloff, sparsify_methods=sparsify_methods)}'
        '_sigma.png'
    )
    _save_fig(fig, path, tight_top=0.86)
    return path


def plot_curves(
    n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
    sparsify_methods,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
    *, file_stem=None,
):
    sync = plot_sync_basin(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    qcurves = plot_q_curves_by_kn(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    qerror = plot_q_error_from_full(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
        file_stem=file_stem,
    )
    overall = plot_overall_error_bars(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
        file_stem=file_stem,
    )
    onset = plot_attractor_onset_error(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
        file_stem=file_stem,
    )
    overall_onset = plot_overall_onset_error(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
        file_stem=file_stem,
    )
    sigma = plot_sigma_vs_kn(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    return sync, qcurves, qerror, overall, onset, overall_onset, sigma


def plot_all(
    n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
    sparsify_methods,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
):
    dist = plot_basin_distributions(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    sync, qcurves, qerror, overall, onset, overall_onset, sigma = plot_curves(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    return dist, sync, qcurves, qerror, overall, onset, overall_onset, sigma


def checkpoint_sweep(
    npz_path, n, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
    sparsify_methods,
    *, heterogeneous=False, falloff=False, weight_std=0.0, n_ic_by_k=None,
):
    """save .npz and refresh plots after each completed k."""
    k_values = sorted(agg_by_k)
    save_agg_to_npz(
        npz_path, n, k_values, agg_by_k, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, weight_std=weight_std,
        n_ic_by_k=n_ic_by_k,
    )
    return plot_all(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )


def plot_basin_distributions(
    n, k_values, agg_by_k, out_dir, n_ic, n_sparsify_seeds,
    sparsify_methods,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
    *, file_stem=None,
):
    q_arrays = []
    for k in k_values:
        q_arrays.append(agg_by_k[k]['q_full_all'])
        for method in sparsify_methods:
            q_arrays.append(agg_by_k[k][sparse_q_key(method)])
    q_global = np.concatenate(q_arrays)
    q_lo, q_hi = int(q_global.min()), int(q_global.max())
    q_bins = np.arange(q_lo - 0.5, q_hi + 1.5, 1)
    support = np.arange(q_lo, q_hi + 1)
    xlim = (q_lo - 0.5, q_hi + 0.5)

    n_k = len(k_values)
    fig, axes = plt.subplots(1, n_k, figsize=(3.5 * n_k, 4), squeeze=False)
    for ax, k in zip(axes[0], k_values):
        d = agg_by_k[k]
        ax.hist(
            d['q_full_all'], bins=q_bins, density=True, alpha=0.55,
            label='full', color='C0',
        )
        for method in sparsify_methods:
            ax.hist(
                d[sparse_q_key(method)], bins=q_bins, density=True, alpha=0.45,
                label=SPARSE_LABELS[method], color=SPARSE_COLORS[method],
            )
        sigma_full = d.get('sigma_full', fit_sigma_mle(d['q_full_all']))
        pmf = discrete_gaussian_pmf(support, max(sigma_full, 0.1))
        ax.plot(support, pmf, ls='-', lw=2, color='C0')
        for method in sparsify_methods:
            qk = sparse_q_key(method)
            sigma = d.get(
                sparse_stat_key(method, 'sigma'),
                fit_sigma_mle(d[qk]),
            )
            pmf = discrete_gaussian_pmf(support, max(sigma, 0.1))
            ax.plot(
                support, pmf, ls='--', lw=2,
                color=SPARSE_COLORS[method],
            )
        ax.set_xlim(xlim)
        ax.set_xlabel('winding number q')
        ax.set_ylabel('probability')
        ax.set_title(f'k={k}  (k/N={k / n:.3f})')
        ax.legend(fontsize=7)

    _fig_suptitle(
        fig, 'ring basin distributions', n, n_ic, n_sparsify_seeds, len(k_values),
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    stem = file_stem or _file_stem(
        n=n, heterogeneous=heterogeneous, falloff=falloff,
        sparsify_methods=sparsify_methods,
    )
    path = out_dir / f'{stem}_distributions.png'
    _save_fig(fig, path, tight_top=0.88)
    return path


def plot_sync_basin(
    n, k_values, agg_by_k, out_dir, n_ic, n_sparsify_seeds,
    sparsify_methods,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    kn = [k / n for k in k_values]
    ax.plot(kn, [100 * agg_by_k[k]['f_sync_full'] for k in k_values], 'o-', label='full', color='C0')
    for method in sparsify_methods:
        ax.plot(
            kn,
            [100 * agg_by_k[k][sparse_stat_key(method, 'f_sync')] for k in k_values],
            f'{SPARSE_MARKERS[method]}-',
            label=SPARSE_LABELS[method],
            color=SPARSE_COLORS[method],
        )
    ax.set_xlabel('k/N')
    ax.set_ylabel('% of trials landing in sync state')
    ax.legend(fontsize=8)

    _fig_suptitle(
        fig, 'ring basin sweep', n, n_ic, n_sparsify_seeds, len(k_values),
        sparsify_methods,
        heterogeneous=heterogeneous, falloff=falloff,
        extra_lines=(_sync_criteria_banner(),), n_ic_by_k=n_ic_by_k,
    )
    path = out_dir / (
        f'{_file_stem(n=n, heterogeneous=heterogeneous, falloff=falloff, sparsify_methods=sparsify_methods)}'
        '_sync.png'
    )
    _save_fig(fig, path, tight_top=0.82)
    return path


def _comparison_plots_dir(n, het, falloff, compare_tag):
    base = _plots_dir(n=n, heterogeneous=het, falloff=falloff, sparsify_methods=['er'])
    return base / f'compare_{compare_tag}'


def run_plot_comparison(
    ring_npz, extra_npzs, *, compare_methods=None, compare_tag=None,
):
    """merge .npz files and write comparison plots (full vs selected sparse methods)."""
    (
        n, k_values, agg_by_k, n_ic, n_sparsify_seeds, n_ic_by_k,
        het, falloff, sparsify_methods,
    ) = merge_agg_for_comparison(ring_npz, *extra_npzs)
    if compare_methods is not None:
        unknown = set(compare_methods) - set(sparsify_methods)
        if unknown:
            raise ValueError(
                f'--compare-methods {sorted(unknown)} not in merged data '
                f'({sparsify_methods})',
            )
        sparsify_methods = list(compare_methods)
    tag = compare_tag or '_'.join(sparsify_methods)
    plots_dir = _comparison_plots_dir(n, het, falloff, tag)
    plots_dir.mkdir(parents=True, exist_ok=True)
    compare_stem = f'ring_basin_sweep_compare_{tag}_n={n}'
    paths = {}
    paths['distributions'] = plot_basin_distributions(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=het, falloff=falloff, n_ic_by_k=n_ic_by_k,
        file_stem=compare_stem,
    )
    paths['q_error'] = plot_q_error_from_full(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=het, falloff=falloff, n_ic_by_k=n_ic_by_k,
        file_stem=compare_stem,
    )
    paths['q_curves'] = plot_q_curves_by_kn(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=het, falloff=falloff, n_ic_by_k=n_ic_by_k,
        file_stem=compare_stem,
    )
    paths['overall'] = plot_overall_error_bars(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=het, falloff=falloff, n_ic_by_k=n_ic_by_k,
        file_stem=compare_stem,
    )
    paths['onset'] = plot_attractor_onset_error(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=het, falloff=falloff, n_ic_by_k=n_ic_by_k,
        file_stem=compare_stem,
    )
    paths['overall_onset'] = plot_overall_onset_error(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        sparsify_methods,
        heterogeneous=het, falloff=falloff, n_ic_by_k=n_ic_by_k,
        file_stem=compare_stem,
    )
    return paths


def main():
    p = argparse.ArgumentParser(description='ring basin statistics: full vs sparse')
    p.add_argument('--N', type=int, default=80, help='ring size (paper uses 40–80; 625 is slow)')
    p.add_argument('--kn', type=str, nargs='+', default=None, metavar='K/N',
                   help='k/N ratios to sweep (k = round(kn * N); comma- or space-separated)')
    p.add_argument('--k', type=str, nargs='+', default=None, metavar='K',
                   help='neighbor shells k directly (alternative to --kn; e.g. --k 145 160)')
    p.add_argument('--n-ic', type=int, default=500, help='random ICs per (k, sparsify seed)')
    p.add_argument('--n-sparsify-seeds', type=int, default=5, help='ER sparsify seeds per k')
    p.add_argument('--base-seed', type=int, default=42)
    p.add_argument('--t-end', type=float, default=60.0)
    p.add_argument('--quick', action='store_true', help='small fast run')
    p.add_argument(
        '--heterogeneous', action='store_true',
        help='lognormal edge weights on ring (same as A_ring_heterogeneous in kuramoto.py)',
    )
    p.add_argument(
        '--falloff', action='store_true',
        help='1/distance edge weights on ring (Wiley et al. Eq. 3 generalization)',
    )
    p.add_argument('--weight-std', type=float, default=HE_WEIGHT_STD,
                   help='lognormal σ for heterogeneous weights')
    p.add_argument('--out-dir', type=Path, default=Path('data'))
    p.add_argument(
        '--plot-only', type=Path, default=None, metavar='NPZ',
        help='regenerate plots from saved .npz (skip simulation)',
    )
    p.add_argument(
        '--append', action='store_true',
        help='merge new k values into existing .npz (skip k already saved)',
    )
    p.add_argument(
        '--interpolate-k', action='store_true',
        help='with --append, add midpoint k between saved anchor values',
    )
    p.add_argument(
        '--match-k-from', type=Path, default=None, metavar='NPZ',
        help='with --append, also run any k listed in another sweep .npz',
    )
    p.add_argument(
        '--target-n-ic', type=int, default=None, metavar='N',
        help='with --append, add ICs until each k has this many (e.g. 150)',
    )
    p.add_argument(
        '--ring-sparse-only', action='store_true',
        help='compare full vs ring farthest + even (every other) + odd only (no ER)',
    )
    p.add_argument(
        '--sparsify-methods', nargs='+', choices=SPARSIFY_METHOD_CHOICES,
        default=None, metavar='METHOD',
        help='sparse graphs to run (default: er only)',
    )
    p.add_argument(
        '--plot-comparison', nargs='+', type=Path, metavar='NPZ',
        help='ring .npz first; optional ER / weight_based .npz files to merge',
    )
    p.add_argument(
        '--compare-methods', nargs='+', choices=SPARSIFY_METHOD_CHOICES,
        default=None, metavar='METHOD',
        help='subset of sparse methods to plot (default: all merged)',
    )
    p.add_argument(
        '--compare-tag', type=str, default=None,
        help='output subfolder name under plots/ (default: joined method names)',
    )
    args = p.parse_args()

    if args.plot_comparison is not None:
        if len(args.plot_comparison) < 1:
            p.error('--plot-comparison needs at least one .npz')
        ring_npz, *extra_npzs = args.plot_comparison
        try:
            paths = run_plot_comparison(
                ring_npz, extra_npzs,
                compare_methods=args.compare_methods,
                compare_tag=args.compare_tag,
            )
        except ValueError as exc:
            p.error(str(exc))
        for path in paths.values():
            print(f'saved {path}')
        return

    if args.heterogeneous and args.falloff:
        p.error('use at most one of --heterogeneous and --falloff')

    if args.k is not None and args.kn is not None:
        p.error('use --k or --kn, not both')

    try:
        sparsify_methods = resolve_sparsify_methods(args)
    except ValueError as exc:
        p.error(str(exc))

    if args.plot_only is not None:
        (
            n, k_values, agg_by_k, n_ic, n_sparsify_seeds, n_ic_by_k,
            het, falloff, sparsify_methods,
        ) = load_agg_from_npz(args.plot_only)
        plots_dir = _plots_dir(
            n=n, heterogeneous=het, falloff=falloff, sparsify_methods=sparsify_methods,
        )
        plots_dir.mkdir(parents=True, exist_ok=True)
        dist_path, sync_path, qcurves_path, qerror_path, overall_path, onset_path, overall_onset_path, sigma_path = plot_all(
            n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
            sparsify_methods,
            heterogeneous=het, falloff=falloff, n_ic_by_k=n_ic_by_k,
        )
        print(f'saved {dist_path}')
        print(f'saved {sync_path}')
        print(f'saved {qcurves_path}')
        print(f'saved {qerror_path}')
        print(f'saved {overall_path}')
        print(f'saved {onset_path}')
        print(f'saved {overall_onset_path}')
        print(f'saved {sigma_path}')
        return

    if args.quick:
        args.N = 60
        args.kn = [3 / 80, 8 / 80, 18 / 80]
        args.k = None
        args.n_ic = 100
        args.n_sparsify_seeds = 2
        args.t_end = 20.0

    n = args.N
    if args.target_n_ic is not None and not args.append:
        p.error('--target-n-ic requires --append')
    if args.interpolate_k and not args.append:
        p.error('--interpolate-k requires --append')
    if (
        (args.interpolate_k or args.match_k_from or args.target_n_ic is not None)
        and args.k is None and args.kn is None
    ):
        k_values = []
    else:
        try:
            k_values = resolve_k_values(n, k_args=args.k, kn_args=args.kn)
        except ValueError as exc:
            p.error(str(exc))
    out_dir = args.out_dir
    het = args.heterogeneous
    falloff = args.falloff
    plots_dir = _plots_dir(
        n=n, heterogeneous=het, falloff=falloff, sparsify_methods=sparsify_methods,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    weight_note = ''
    if het:
        weight_note = f'  het. weights σ={args.weight_std}'
    elif falloff:
        weight_note = '  falloff (1/d)'
    sparse_note = ', '.join(SPARSE_LABELS[m] for m in sparsify_methods)
    q_note = (
        f'q={SPARSIFY_Q}'
        if any(m in STOCHASTIC_METHODS for m in sparsify_methods)
        else 'deterministic ring sparse'
    )
    print(
        f'ring basin sweep  N={n}  K={PAPER_K}  ω={OMEGA}  '
        f'n_ic={args.n_ic}  sparsify_seeds={args.n_sparsify_seeds}  {q_note}  '
        f'sparse methods: {sparse_note}'
        f'{weight_note}',
    )
    stem = _file_stem(
        n=n, heterogeneous=het, falloff=falloff, sparsify_methods=sparsify_methods,
    )
    npz_path = out_dir / f'{stem}.npz'

    agg_by_k = {}
    n_ic_by_k = {}
    existing_k = set()
    k_ic_add = {}
    if args.append and npz_path.exists():
        meta, existing_k_list, agg_by_k = load_npz_for_append(npz_path)
        n_ic_by_k = dict(meta['n_ic_by_k'])
        existing_k = set(existing_k_list)
        if meta['N'] != n:
            p.error(f'--append: existing N={meta["N"]} != requested N={n}')
        if meta['heterogeneous'] != het or meta['falloff'] != falloff:
            p.error('--append: weight mode must match existing file')
        if meta['sparsify_methods'] != sparsify_methods:
            p.error(
                '--append: sparsify methods must match existing file '
                f'({meta["sparsify_methods"]})',
            )
        if meta['n_sparsify_seeds'] != args.n_sparsify_seeds:
            p.error(
                '--append: n_sparsify_seeds must match existing file '
                f'({meta["n_sparsify_seeds"]})',
            )
        if args.n_ic not in n_ic_by_k.values() and args.n_ic != meta['n_ic']:
            print(
                f'note: appending new k with n_ic={args.n_ic} '
                f'(existing k used {sorted(set(n_ic_by_k.values()))})',
            )
        skipped = sorted(k for k in k_values if k in existing_k)
        if skipped:
            print(f'skipping k already in {npz_path.name}: {skipped}')
        k_values = sorted(set(k_values) | existing_k)
        print(f'append mode: {len(existing_k)} k in file, running new k only')
        if args.target_n_ic is not None:
            for k in sorted(existing_k):
                cur = n_ic_by_k.get(k, meta['n_ic'])
                need = args.target_n_ic - cur
                if need > 0:
                    k_ic_add[k] = (cur, need)
            if k_ic_add:
                print(
                    f'target-n-ic={args.target_n_ic}: adding ICs at '
                    f'{len(k_ic_add)} k values',
                )
            else:
                print(f'target-n-ic={args.target_n_ic}: all k already satisfied')
    elif args.append:
        print(f'--append: {npz_path} not found, starting fresh')
    if args.target_n_ic is not None and not npz_path.exists():
        p.error(f'--target-n-ic: {npz_path} not found')

    if args.interpolate_k:
        anchor = sorted(existing_k | set(k_values))
        if len(anchor) < 2:
            p.error('--interpolate-k needs at least 2 anchor k values in file or on CLI')
        mids = midpoint_k_values(anchor, n)
        new_mids = [k for k in mids if k not in existing_k]
        print(f'interpolate-k: midpoints {mids}  (new: {new_mids})')
        k_values = sorted(set(k_values) | existing_k | set(mids))

    if args.match_k_from is not None:
        if not args.match_k_from.exists():
            p.error(f'--match-k-from: {args.match_k_from} not found')
        ref_k = [int(k) for k in np.load(args.match_k_from)['k_values']]
        missing = sorted(k for k in ref_k if k not in existing_k)
        print(f'match-k-from {args.match_k_from.name}: need {missing}')
        k_values = sorted(set(k_values) | existing_k | set(ref_k))

    if not k_values:
        p.error('no k values to run (check --k, --kn, --append, or --interpolate-k)')
    n_edges_ref = n * max(k_values)
    if n_edges_ref > 30_000:
        solver_kw = dict(method='RK45', rtol=1e-5, atol=1e-7)
    else:
        solver_kw = dict(method='RK45', rtol=1e-6, atol=1e-8)

    print(f'k sweep -> k values: {k_values}  (k/N: {[k / n for k in k_values]})')
    k_to_run = [k for k in k_values if k not in existing_k]
    k_ic_run = sorted(k_ic_add)
    weight_std = args.weight_std if het else 0.0
    report_n_ic = args.target_n_ic if args.target_n_ic is not None else args.n_ic
    dist_path = sync_path = qcurves_path = qerror_path = overall_path = onset_path = overall_onset_path = sigma_path = None
    for k in k_to_run:
        print(f'  k={k}  (k/N={k/n:.3f})  edges≈{n*k}  paper σ≈{paper_sigma(n,k):.2f}...')
        records = run_condition(
            n, k, args.n_ic, args.n_sparsify_seeds, args.base_seed + k * 10_000,
            args.t_end, solver_kw, sparsify_methods,
            heterogeneous=het, falloff=falloff, weight_std=args.weight_std,
        )
        agg = aggregate_records(records, sparsify_methods)
        agg_by_k[k] = agg
        n_ic_by_k[k] = args.n_ic
        sparse_bits = []
        for method in sparsify_methods:
            sparse_bits.append(
                f'{SPARSE_LABELS[method]}: sync={agg[sparse_stat_key(method, "f_sync")]:.3f}  '
                f'σ={agg[sparse_stat_key(method, "sigma")]:.2f}  '
                f'match={agg[sparse_stat_key(method, "match_rate")]:.3f}',
            )
        print(
            f'    sync full={agg["f_sync_full"]:.3f}  σ full={agg["sigma_full"]:.2f}  '
            f'paper σ≈{paper_sigma(n, k):.2f}',
        )
        for bit in sparse_bits:
            print(f'    {bit}')
        dist_path, sync_path, qcurves_path, qerror_path, overall_path, onset_path, overall_onset_path, sigma_path = checkpoint_sweep(
            npz_path, n, agg_by_k, plots_dir, report_n_ic, args.n_sparsify_seeds,
            sparsify_methods,
            heterogeneous=het, falloff=falloff, weight_std=weight_std,
            n_ic_by_k=n_ic_by_k,
        )
        print(f'    checkpoint: saved {npz_path.name} + plots')

    for k in k_ic_run:
        ic_skip, n_add = k_ic_add[k]
        print(
            f'  k={k}  (k/N={k/n:.3f})  +{n_add} ICs (skip {ic_skip})  '
            f'edges≈{n*k}...',
        )
        records = run_condition(
            n, k, n_add, args.n_sparsify_seeds, args.base_seed + k * 10_000,
            args.t_end, solver_kw, sparsify_methods,
            heterogeneous=het, falloff=falloff, weight_std=args.weight_std,
            ic_skip=ic_skip,
        )
        agg_new = aggregate_records(records, sparsify_methods)
        agg_by_k[k] = merge_agg_with_additional(agg_by_k[k], agg_new, sparsify_methods)
        n_ic_by_k[k] = ic_skip + n_add
        sparse_bits = []
        agg = agg_by_k[k]
        for method in sparsify_methods:
            sparse_bits.append(
                f'{SPARSE_LABELS[method]}: sync={agg[sparse_stat_key(method, "f_sync")]:.3f}  '
                f'σ={agg[sparse_stat_key(method, "sigma")]:.2f}  '
                f'match={agg[sparse_stat_key(method, "match_rate")]:.3f}',
            )
        print(
            f'    n_ic={n_ic_by_k[k]}  sync full={agg["f_sync_full"]:.3f}  '
            f'σ full={agg["sigma_full"]:.2f}',
        )
        for bit in sparse_bits:
            print(f'    {bit}')
        dist_path, sync_path, qcurves_path, qerror_path, overall_path, onset_path, overall_onset_path, sigma_path = checkpoint_sweep(
            npz_path, n, agg_by_k, plots_dir, report_n_ic, args.n_sparsify_seeds,
            sparsify_methods,
            heterogeneous=het, falloff=falloff, weight_std=weight_std,
            n_ic_by_k=n_ic_by_k,
        )
        print(f'    checkpoint: saved {npz_path.name} + plots')

    k_values = sorted(agg_by_k)
    if not k_to_run and not k_ic_run and k_values:
        dist_path, sync_path, qcurves_path, qerror_path, overall_path, onset_path, overall_onset_path, sigma_path = plot_all(
            n, k_values, agg_by_k, plots_dir, report_n_ic, args.n_sparsify_seeds,
            sparsify_methods,
            heterogeneous=het, falloff=falloff, n_ic_by_k=n_ic_by_k,
        )
        save_agg_to_npz(
            npz_path, n, k_values, agg_by_k, report_n_ic, args.n_sparsify_seeds,
            sparsify_methods,
            heterogeneous=het, falloff=falloff, weight_std=weight_std,
            n_ic_by_k=n_ic_by_k,
        )

    print(f'\nsaved {dist_path}')
    print(f'saved {sync_path}')
    print(f'saved {qcurves_path}')
    print(f'saved {qerror_path}')
    print(f'saved {overall_path}')
    print(f'saved {onset_path}')
    print(f'saved {overall_onset_path}')
    print(f'saved {sigma_path}')
    print(f'saved {npz_path}')


if __name__ == '__main__':
    main()
