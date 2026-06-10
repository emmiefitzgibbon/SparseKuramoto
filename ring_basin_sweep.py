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
LIVE_DIST_MAX_K = 12  # update distribution panel live only when k count stays small


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


def paper_sigma(n, k):
    return max(SIGMA_SLOPE * np.sqrt(n / k) + SIGMA_INTERCEPT, 0.01)


def run_condition(
    n, k, n_ic, n_sparsify_seeds, base_seed, t_end, solver_kw,
    heterogeneous=False, falloff=False, weight_std=HE_WEIGHT_STD,
):
    rng_weights = np.random.default_rng(base_seed + 7)
    A = generate_ring_matrix(
        n, k, heterogeneous=heterogeneous, falloff=falloff,
        rng=rng_weights, weight_std=weight_std,
    )
    ei_f, ej_f, w_f = edges_from_A(A)
    er_i, er_j, er_w, er_pe = precompute_er(A)
    rng_master = np.random.default_rng(base_seed)

    records = []
    for si in range(n_sparsify_seeds):
        sp_rng = np.random.default_rng(base_seed + 1000 + si)
        A_s = sparsify_er(er_i, er_j, er_w, er_pe, SPARSIFY_Q, sp_rng)
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
        records.append(dict(
            sparsify_seed=base_seed + 1000 + si,
            q_full=q_full,
            q_sparse=q_sparse,
            r_full=np.array(r_full),
            r_sparse=np.array(r_sparse),
            f_sync_full=float(np.mean(r_full >= SYNC_R_THRESH)),
            f_sync_sparse=float(np.mean(r_sparse >= SYNC_R_THRESH)),
            sigma_full=fit_sigma_mle(q_full),
            sigma_sparse=fit_sigma_mle(q_sparse),
            sigma_sample_full=float(np.std(q_full)),
            sigma_sample_sparse=float(np.std(q_sparse)),
            match_rate=float(np.mean(q_full == q_sparse)),
            mean_abs_dq=float(np.mean(np.abs(q_full - q_sparse))),
        ))
    return records


def aggregate_records(records):
    keys = (
        'f_sync_full', 'f_sync_sparse', 'sigma_full', 'sigma_sparse',
        'sigma_sample_full', 'sigma_sample_sparse',
        'match_rate', 'mean_abs_dq',
    )
    out = {}
    for key in keys:
        vals = [r[key] for r in records]
        out[key] = float(np.mean(vals))
        out[f'{key}_std'] = float(np.std(vals))
    out['q_full_all'] = np.concatenate([r['q_full'] for r in records])
    out['q_sparse_all'] = np.concatenate([r['q_sparse'] for r in records])
    return out


def _file_stem(*, n, heterogeneous=False, falloff=False):
    if heterogeneous:
        base = 'ring_basin_sweep_het'
    elif falloff:
        base = 'ring_basin_sweep_falloff'
    else:
        base = 'ring_basin_sweep'
    return f'{base}_n={n}'


def _weight_tag(*, heterogeneous=False, falloff=False):
    if heterogeneous:
        return '  het. weights'
    if falloff:
        return '  falloff (1/d)'
    return ''


def _plots_dir(*, n, heterogeneous=False, falloff=False):
    if heterogeneous:
        return Path(f'plots/heterogeneous_weights_n={n}')
    if falloff:
        return Path(f'plots/falloff_weights_n={n}')
    return Path(f'plots/homogeneous_weights_n={n}')


def n_ic_by_k_from_npz(d, k_values):
    if 'n_ic_per_k' in d:
        return {int(k): int(n) for k, n in zip(d['k_values'], d['n_ic_per_k'])}
    n_ic = int(d['n_ic'])
    return {int(k): n_ic for k in k_values}


def _sample_banner(n_ic, n_sparsify_seeds, n_k_values, n_ic_by_k=None):
    if n_ic_by_k:
        ic_vals = sorted(set(n_ic_by_k.values()))
        if len(ic_vals) > 1:
            return (
                f'n_ic={ic_vals[0]}–{ic_vals[-1]} per k  ({n_sparsify_seeds} sparse graphs); '
                f'{n_k_values} k values'
            )
        n_ic = ic_vals[0]
    sparse_trials = n_ic * n_sparsify_seeds
    return (
        f'sparse: {sparse_trials} trials/k ({n_ic} ICs × {n_sparsify_seeds} sparse graphs); '
        f'full: {n_ic} ICs/k (1 full graph); {n_k_values} k values'
    )


def _sync_criteria_banner():
    return (
        f'sync: r ≥ {SYNC_R_THRESH} in co-rotating frame (θᵢ − Ωt),  '
        f'r = |⟨e^{{iθ}}⟩|'
    )


def _fig_suptitle(
    fig, title_line, n, n_ic, n_sparsify_seeds, n_k_values,
    *, heterogeneous=False, falloff=False, extra_lines=(), n_ic_by_k=None,
):
    wt = _weight_tag(heterogeneous=heterogeneous, falloff=falloff)
    lines = [
        f'{title_line}  N={n}  K={PAPER_K}  q_sparsify={SPARSIFY_Q}{wt}',
        _sample_banner(n_ic, n_sparsify_seeds, n_k_values, n_ic_by_k=n_ic_by_k),
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


NPZ_AGG_KEYS = (
    'f_sync_full', 'f_sync_sparse', 'sigma_full', 'sigma_sparse',
    'match_rate', 'mean_abs_dq', 'q_full_all', 'q_sparse_all',
)


def _agg_from_npz_arrays(d, k):
    p = f'k{k}'
    qf = d[f'{p}_q_full_all']
    qs = d[f'{p}_q_sparse_all']
    return {
        'q_full_all': qf,
        'q_sparse_all': qs,
        'sigma_full': fit_sigma_mle(qf),
        'sigma_sparse': fit_sigma_mle(qs),
        'f_sync_full': float(d[f'{p}_f_sync_full']),
        'f_sync_sparse': float(d[f'{p}_f_sync_sparse']),
        'match_rate': float(d[f'{p}_match_rate']),
        'mean_abs_dq': float(d[f'{p}_mean_abs_dq']),
    }


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
    agg_by_k = {k: _agg_from_npz_arrays(d, k) for k in k_values}
    n_ic = int(d['n_ic'])
    n_sparsify_seeds = int(d['n_sparsify_seeds'])
    n_ic_by_k = n_ic_by_k_from_npz(d, k_values)
    return n, k_values, agg_by_k, n_ic, n_sparsify_seeds, n_ic_by_k, het, falloff


def load_npz_for_append(npz_path):
    """Load full per-k arrays from an existing sweep file."""
    d = np.load(npz_path)
    stem = Path(npz_path).stem
    meta = {
        'N': int(d['N']),
        'n_ic': int(d['n_ic']),
        'n_sparsify_seeds': int(d['n_sparsify_seeds']),
        'heterogeneous': bool(d['heterogeneous']) if 'heterogeneous' in d else '_het' in stem,
        'falloff': bool(d['falloff']) if 'falloff' in d else '_falloff' in stem,
        'weight_std': float(d['weight_std']) if 'weight_std' in d else 0.0,
    }
    k_values = [int(k) for k in d['k_values']]
    agg_by_k = {}
    for k in k_values:
        p = f'k{k}'
        agg_by_k[k] = {key: d[f'{p}_{key}'] for key in NPZ_AGG_KEYS}
    meta['n_ic_by_k'] = n_ic_by_k_from_npz(d, k_values)
    return meta, k_values, agg_by_k


def save_agg_to_npz(
    npz_path, n, k_values, agg_by_k, n_ic, n_sparsify_seeds,
    *, heterogeneous=False, falloff=False, weight_std=0.0, n_ic_by_k=None,
):
    if n_ic_by_k is None:
        n_ic_by_k = {k: n_ic for k in k_values}
    save = dict(
        N=n, k_values=np.array(k_values), PAPER_K=PAPER_K, SPARSIFY_Q=SPARSIFY_Q,
        n_ic=n_ic, n_ic_per_k=np.array([n_ic_by_k[k] for k in k_values], dtype=int),
        n_sparsify_seeds=n_sparsify_seeds,
        heterogeneous=heterogeneous, falloff=falloff, weight_std=weight_std,
    )
    for k in k_values:
        pref = f'k{k}'
        for key in NPZ_AGG_KEYS:
            save[f'{pref}_{key}'] = agg_by_k[k][key]
    np.savez_compressed(npz_path, **save)


def _q_prob_data(n, k_values, agg_by_k):
    q_global = np.concatenate([
        np.concatenate([agg_by_k[k]['q_full_all'], agg_by_k[k]['q_sparse_all']])
        for k in k_values
    ])
    q_lo, q_hi = int(q_global.min()), int(q_global.max())
    q_support = np.arange(q_lo, q_hi + 1)
    kn = np.array([k / n for k in k_values])
    mats = {}
    for key in ('q_full_all', 'q_sparse_all'):
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


def _q_prob_data_abs(n, k_values, agg_by_k):
    kn, q_support, mats = _q_prob_data(n, k_values, agg_by_k)
    abs_mats = {}
    abs_qs = None
    for key, mat in mats.items():
        abs_qs, abs_mats[key] = _collapse_q_sign(q_support, mat)
    return kn, abs_qs, abs_mats


def plot_q_curves_by_kn(
    n, k_values, agg_by_k, out_dir, n_ic, n_sparsify_seeds,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
):
    kn, abs_qs, mats = _q_prob_data_abs(n, k_values, agg_by_k)
    active = [
        aq for i, aq in enumerate(abs_qs)
        if max(mats['q_full_all'][i].max(), mats['q_sparse_all'][i].max()) >= 0.005
    ]
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
        ax.plot(kn, 100 * mats['q_sparse_all'][qi], 's-', color='C1', label='sparse', ms=4)
        ax.set_title(f'|q|={aq}' if aq else 'q=0')
        ax.set_xlabel('k/N')
        ax.set_ylabel('% of trials')
        ymax = 100 * max(mats['q_full_all'][qi].max(), mats['q_sparse_all'][qi].max()) * 1.15 + 1
        ax.set_ylim(0, min(105, ymax))
        if ax.get_ylim()[1] < 5:
            ax.set_ylim(0, 5)
        for prob, color, tag in (
            (mats['q_full_all'][qi], 'C0', 'full'),
            (mats['q_sparse_all'][qi], 'C1', 'sparse'),
        ):
            kc = empirical_kn_onset(kn, prob)
            if kc is not None and kn.min() <= kc <= kn.max():
                ax.axvline(kc, color=color, ls=':', lw=1.4, alpha=0.9, zorder=0)
                ytxt = ax.get_ylim()[1] * (0.92 if tag == 'full' else 0.78)
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
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    path = out_dir / f'{_file_stem(n=n, heterogeneous=heterogeneous, falloff=falloff)}_q_curves.png'
    _save_fig(fig, path, tight_top=0.88)
    return path


def plot_sigma_vs_kn(
    n, k_values, agg_by_k, out_dir, n_ic, n_sparsify_seeds,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    kn = [k / n for k in k_values]
    ax.plot(kn, [agg_by_k[k]['sigma_full'] for k in k_values], 'o-', label='full', color='C0')
    ax.plot(kn, [agg_by_k[k]['sigma_sparse'] for k in k_values], 's-', label='sparse', color='C1')
    ax.set_xlabel('k/N')
    ax.set_ylabel('σ  (MLE fit to q distribution)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    _fig_suptitle(
        fig, 'ring basin sweep', n, n_ic, n_sparsify_seeds, len(k_values),
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    path = out_dir / f'{_file_stem(n=n, heterogeneous=heterogeneous, falloff=falloff)}_sigma.png'
    _save_fig(fig, path, tight_top=0.86)
    return path


def plot_curves(
    n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
):
    sync = plot_sync_basin(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    qcurves = plot_q_curves_by_kn(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    sigma = plot_sigma_vs_kn(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    return sync, qcurves, sigma


def plot_all(
    n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
):
    dist = plot_basin_distributions(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    sync, qcurves, sigma = plot_curves(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    return dist, sync, qcurves, sigma


def checkpoint_sweep(
    npz_path, n, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
    *, heterogeneous=False, falloff=False, weight_std=0.0, n_ic_by_k=None,
):
    """save .npz and refresh plots after each completed k."""
    k_values = sorted(agg_by_k)
    save_agg_to_npz(
        npz_path, n, k_values, agg_by_k, n_ic, n_sparsify_seeds,
        heterogeneous=heterogeneous, falloff=falloff, weight_std=weight_std,
        n_ic_by_k=n_ic_by_k,
    )
    dist = None
    if len(k_values) <= LIVE_DIST_MAX_K:
        dist = plot_basin_distributions(
            n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
            heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
        )
    sync, qcurves, sigma = plot_curves(
        n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    return dist, sync, qcurves, sigma


def plot_basin_distributions(
    n, k_values, agg_by_k, out_dir, n_ic, n_sparsify_seeds,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
):
    q_global = np.concatenate([
        np.concatenate([agg_by_k[k]['q_full_all'], agg_by_k[k]['q_sparse_all']])
        for k in k_values
    ])
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
        ax.hist(
            d['q_sparse_all'], bins=q_bins, density=True, alpha=0.55,
            label='sparse', color='C1',
        )
        for sigma, ls, color in [
            (d['sigma_full'], '-', 'C0'),
            (d['sigma_sparse'], '--', 'C1'),
        ]:
            pmf = discrete_gaussian_pmf(support, max(sigma, 0.1))
            ax.plot(support, pmf, ls=ls, lw=2, color=color)
        ax.set_xlim(xlim)
        ax.set_xlabel('winding number q')
        ax.set_ylabel('probability')
        ax.set_title(f'k={k}  (k/N={k / n:.3f})')
        ax.legend(fontsize=7)

    _fig_suptitle(
        fig, 'ring basin distributions', n, n_ic, n_sparsify_seeds, len(k_values),
        heterogeneous=heterogeneous, falloff=falloff, n_ic_by_k=n_ic_by_k,
    )
    path = out_dir / f'{_file_stem(n=n, heterogeneous=heterogeneous, falloff=falloff)}_distributions.png'
    _save_fig(fig, path, tight_top=0.88)
    return path


def plot_sync_basin(
    n, k_values, agg_by_k, out_dir, n_ic, n_sparsify_seeds,
    heterogeneous=False, falloff=False, n_ic_by_k=None,
):
    fig, ax = plt.subplots(figsize=(6, 4))
    kn = [k / n for k in k_values]
    ax.plot(kn, [100 * agg_by_k[k]['f_sync_full'] for k in k_values], 'o-', label='full', color='C0')
    ax.plot(kn, [100 * agg_by_k[k]['f_sync_sparse'] for k in k_values], 's-', label='sparse', color='C1')
    ax.set_xlabel('k/N')
    ax.set_ylabel('% of trials landing in sync state')
    ax.legend(fontsize=8)

    _fig_suptitle(
        fig, 'ring basin sweep', n, n_ic, n_sparsify_seeds, len(k_values),
        heterogeneous=heterogeneous, falloff=falloff,
        extra_lines=(_sync_criteria_banner(),), n_ic_by_k=n_ic_by_k,
    )
    path = out_dir / f'{_file_stem(n=n, heterogeneous=heterogeneous, falloff=falloff)}_sync.png'
    _save_fig(fig, path, tight_top=0.82)
    return path


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
    args = p.parse_args()

    if args.heterogeneous and args.falloff:
        p.error('use at most one of --heterogeneous and --falloff')

    if args.k is not None and args.kn is not None:
        p.error('use --k or --kn, not both')

    if args.plot_only is not None:
        n, k_values, agg_by_k, n_ic, n_sparsify_seeds, n_ic_by_k, het, falloff = (
            load_agg_from_npz(args.plot_only)
        )
        plots_dir = _plots_dir(n=n, heterogeneous=het, falloff=falloff)
        plots_dir.mkdir(parents=True, exist_ok=True)
        dist_path, sync_path, qcurves_path, sigma_path = plot_all(
            n, k_values, agg_by_k, plots_dir, n_ic, n_sparsify_seeds,
            heterogeneous=het, falloff=falloff, n_ic_by_k=n_ic_by_k,
        )
        print(f'saved {dist_path}')
        print(f'saved {sync_path}')
        print(f'saved {qcurves_path}')
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
    try:
        k_values = resolve_k_values(n, k_args=args.k, kn_args=args.kn)
    except ValueError as exc:
        p.error(str(exc))
    out_dir = args.out_dir
    het = args.heterogeneous
    falloff = args.falloff
    plots_dir = _plots_dir(n=n, heterogeneous=het, falloff=falloff)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_edges_ref = n * k_values[-1]
    if n_edges_ref > 30_000:
        solver_kw = dict(method='RK45', rtol=1e-5, atol=1e-7)
    else:
        solver_kw = dict(method='RK45', rtol=1e-6, atol=1e-8)

    weight_note = ''
    if het:
        weight_note = f'  het. weights σ={args.weight_std}'
    elif falloff:
        weight_note = '  falloff (1/d)'
    print(
        f'ring basin sweep  N={n}  K={PAPER_K}  ω={OMEGA}  '
        f'n_ic={args.n_ic}  sparsify_seeds={args.n_sparsify_seeds}  q={SPARSIFY_Q}'
        f'{weight_note}',
    )
    stem = _file_stem(n=n, heterogeneous=het, falloff=falloff)
    npz_path = out_dir / f'{stem}.npz'

    agg_by_k = {}
    n_ic_by_k = {}
    existing_k = set()
    if args.append and npz_path.exists():
        meta, existing_k_list, agg_by_k = load_npz_for_append(npz_path)
        n_ic_by_k = dict(meta['n_ic_by_k'])
        existing_k = set(existing_k_list)
        if meta['N'] != n:
            p.error(f'--append: existing N={meta["N"]} != requested N={n}')
        if meta['heterogeneous'] != het or meta['falloff'] != falloff:
            p.error('--append: weight mode must match existing file')
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
    elif args.append:
        print(f'--append: {npz_path} not found, starting fresh')

    print(f'k sweep -> k values: {k_values}  (k/N: {[k / n for k in k_values]})')
    k_to_run = [k for k in k_values if k not in existing_k]
    weight_std = args.weight_std if het else 0.0
    dist_path = sync_path = qcurves_path = sigma_path = None
    for k in k_to_run:
        print(f'  k={k}  (k/N={k/n:.3f})  edges≈{n*k}  paper σ≈{paper_sigma(n,k):.2f}...')
        records = run_condition(
            n, k, args.n_ic, args.n_sparsify_seeds, args.base_seed + k * 10_000,
            args.t_end, solver_kw,
            heterogeneous=het, falloff=falloff, weight_std=args.weight_std,
        )
        agg = aggregate_records(records)
        agg_by_k[k] = agg
        n_ic_by_k[k] = args.n_ic
        print(
            f'    sync: full={agg["f_sync_full"]:.3f}  sparse={agg["f_sync_sparse"]:.3f}  |  '
            f'σ: full={agg["sigma_full"]:.2f}  sparse={agg["sigma_sparse"]:.2f}  '
            f'paper={paper_sigma(n,k):.2f}  |  '
            f'match={agg["match_rate"]:.3f}  |Δq|={agg["mean_abs_dq"]:.2f}',
        )
        dist_path, sync_path, qcurves_path, sigma_path = checkpoint_sweep(
            npz_path, n, agg_by_k, plots_dir, args.n_ic, args.n_sparsify_seeds,
            heterogeneous=het, falloff=falloff, weight_std=weight_std,
            n_ic_by_k=n_ic_by_k,
        )
        print(f'    checkpoint: saved {npz_path.name}', end='')
        if dist_path is not None:
            print(' + plots')
        else:
            print(' + curve plots (distributions deferred)')

    k_values = sorted(agg_by_k)
    if k_to_run:
        if dist_path is None and k_values:
            dist_path = plot_basin_distributions(
                n, k_values, agg_by_k, plots_dir, args.n_ic, args.n_sparsify_seeds,
                heterogeneous=het, falloff=falloff, n_ic_by_k=n_ic_by_k,
            )
    elif k_values:
        dist_path, sync_path, qcurves_path, sigma_path = plot_all(
            n, k_values, agg_by_k, plots_dir, args.n_ic, args.n_sparsify_seeds,
            heterogeneous=het, falloff=falloff, n_ic_by_k=n_ic_by_k,
        )
        save_agg_to_npz(
            npz_path, n, k_values, agg_by_k, args.n_ic, args.n_sparsify_seeds,
            heterogeneous=het, falloff=falloff, weight_std=weight_std,
            n_ic_by_k=n_ic_by_k,
        )

    print(f'\nsaved {dist_path}')
    print(f'saved {sync_path}')
    print(f'saved {qcurves_path}')
    print(f'saved {sigma_path}')
    print(f'saved {npz_path}')


if __name__ == '__main__':
    main()
