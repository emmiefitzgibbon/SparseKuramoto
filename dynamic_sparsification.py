"""dynamic_sparsification.py

Dynamic sparsification for Kuramoto networks: every `resparse_every` timesteps,
rebuild the sparse coupling network using current phase information.

--- Theory note: can we use ER on a signed effective-adjacency? ---

The "effective influence" of edge (i,j) in the current dynamical state is

    f_ij = w_ij * cos(theta_j - theta_i)

because cos(Δθ) > 0 means the two oscillators are within π/2 of each other
(coupling is constructive toward synchrony), and cos(Δθ) < 0 means they are
frustrated.  But f_ij can be negative.

SIGNED ER IS NOT VALID:  ER sparsification requires the weighted Laplacian
L = D - A to be PSD.  If A has negative entries, L loses PSD and the
pseudoinverse L^+ is no longer a valid metric; the resulting "effective
resistances" R_e = (e_i-e_j)^T L^+ (e_i-e_j) can be negative and are
meaningless as sampling weights.

DOES R_e DEPEND ON WEIGHTS OR JUST TOPOLOGY?  Both.  For a weighted graph,
R_e = (e_i-e_j)^T L(w)^+ (e_i-e_j) where L(w) uses the actual weights.
High-weight edges have LOW R_e (cheap, redundant paths traverse them); a
topological bridge has HIGH R_e even with small weight.  The ER sampling
probability p_e ∝ w_e * R_e is the statistical leverage score of edge e:
it balances structural criticality against weight magnitude.  For a purely
unweighted (binary) graph, R_e is purely topological.

VIABLE APPROACHES TESTED HERE:

  er_static      baseline: ER on original weights, computed once at t=0
  weight_static  baseline: weight-proportional sampling, computed once
  dyn_cos        p ∝ w_ij * max(cos Δθ, 0) + floor * w_ij  (no ER; cheap)
  dyn_er_cos     ER precomputed once on original weights, then modulated by
                 max(cos Δθ, floor) each step — combines static structural
                 importance with dynamic phase alignment; O(E) per step
  dyn_abs_er     ER recomputed each step on the ABSOLUTE effective adjacency
                 |w_ij * cos Δθ| + floor * w_ij  (always PSD); most
                 principled but O(n^3) per resparsification step

In all dynamic methods the edge WEIGHTS in the sparse network use the
original structural w_ij rescaled by 1/(s*p_e), so the sparse network is
an unbiased estimate of the original coupling.  The phase information only
guides WHICH edges to sample (importance sampling), not the output weights.

Run: python dynamic_sparsification.py
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from heterogeneous_sparsification_compare import (
    build_connectome_network,
    build_heterogeneous_network,
    build_celegans_network,
    build_scale_free_network,
    build_uk_power_grid_network,
    build_fully_connected_network,
    build_ring_network,
    build_cintestinalis_network,
    precompute_er,
    sparsify_stochastic,
)

PLOT_DIR = Path('plots/dynamic_sparsification_fast')
solver_kw = dict(method='RK45', rtol=1e-6, atol=1e-8)

FLOOR_FRAC = 0.05   # keeps all edges in sampling support (frustrated get floor * w)
N_SEEDS    = 20
T_MAX      = 20
N_T        = 600
K_COEF     = 0.1   # K = K_COEF * n  (mild supercritical for most networks)

QS = np.array([0.05, 0.1, 0.2, 0.25, 0.3, 0.5])
RESPARSE_EVERY = 5    # default: rebuild network every 5 of 600 time points

METHOD_LABELS = {
    'er_static':     'ER static',
    'weight_static': 'weight static',
    'dyn_cos':       'dynamic cosine',
    'dyn_er_cos':    'dynamic ER×cosine',
    'dyn_abs_er':    'dynamic |cos| ER (recomputed)',
    'oracle_abs_er': 'oracle |cos| ER (full phases)',
    'woodbury_er':   'incremental |cos| ER (Woodbury rank-1)',
}
METHOD_COLORS = {
    'er_static':     'C0',
    'weight_static': 'C4',
    'dyn_cos':       'C1',
    'dyn_er_cos':    'C2',
    'dyn_abs_er':    'C3',
    'oracle_abs_er': 'C5',
    'woodbury_er':   'C6',
}


def fresh_seed():
    return int.from_bytes(os.urandom(8), 'big') % (2**63)


# ---------------------------------------------------------------------------
# core Kuramoto utilities
# ---------------------------------------------------------------------------

def edges_from_A(A):
    ei, ej = np.where(np.triu(A, 1))
    return ei, ej, A[ei, ej].astype(float)


def weighted_degree(A):
    return np.maximum(A.sum(axis=1), 1e-12)


def kuramoto_rhs(t, theta, K, omega, edges, degree):
    ei, ej, w = edges
    coupling = np.zeros(len(theta))
    np.add.at(coupling, ei, w * np.sin(theta[ej] - theta[ei]))
    np.add.at(coupling, ej, w * np.sin(theta[ei] - theta[ej]))
    return omega + K * coupling / degree


def order_param(theta_t):
    """Global Kuramoto order parameter r(t) from (n, T) phase array."""
    return np.abs(np.mean(np.exp(1j * theta_t), axis=0))


# ---------------------------------------------------------------------------
# dynamic sampling probability methods
# ---------------------------------------------------------------------------

def probs_dyn_cos(ei, ej, we, theta, floor_frac=FLOOR_FRAC):
    """
    p_e ∝ w_e * (max(cos(Δθ_e), 0) + floor_frac)

    Edges between in-phase oscillators (|Δθ| < π/2, cos > 0) get proportionally
    higher probability.  Frustrated edges (cos < 0) fall back to floor_frac * w_e.
    Always positive → valid sampling distribution.
    """
    cos_d = np.cos(theta[ej] - theta[ei])
    f = we * (np.maximum(cos_d, 0.0) + floor_frac)
    return f / f.sum()


def probs_dyn_er_cos(pe_er, ei, ej, theta, floor_frac=FLOOR_FRAC):
    """
    p_e ∝ (w_e * R_e)  *  (max(cos(Δθ_e), 0) + floor_frac)

    Multiplies the pre-computed static ER probabilities by the dynamic cosine
    term.  ER structure (which edges are topologically critical) is fixed;
    cosine modulation up-weights currently-constructive edges.  O(E) per step.
    """
    cos_d = np.cos(theta[ej] - theta[ei])
    f = pe_er * (np.maximum(cos_d, 0.0) + floor_frac)
    return f / f.sum()


def probs_dyn_abs_er(A, theta, floor_frac=FLOOR_FRAC):
    """
    Build an effective adjacency A_eff = w_ij * |cos(Δθ)| + floor * w_ij,
    recompute ER on A_eff, return p_e ∝ A_eff_e * R_e(A_eff).

    |cos| is always ≥ 0, so A_eff is a valid weighted graph (PSD Laplacian).
    This is the most principled option but costs O(n^3) per resparsification.

    WHY |cos| AND NOT max(cos, 0)?
    Using abs keeps anti-phase pairs as HEAVY edges in A_eff — the ER on A_eff
    will identify them as low-resistance (easy to traverse), so the algorithm
    preferentially samples them.  Whether that's desirable depends on context;
    we also implement dyn_er_cos with clipping as the alternative.
    """
    n = A.shape[0]
    ei, ej = np.where(np.triu(A, 1))
    we = A[ei, ej].astype(float)
    cos_d = np.cos(theta[ej] - theta[ei])
    w_eff = we * (np.abs(cos_d) + floor_frac)

    A_eff = np.zeros((n, n))
    A_eff[ei, ej] = w_eff
    A_eff[ej, ei] = w_eff

    L = np.diag(A_eff.sum(1)) - A_eff
    Lp = np.linalg.pinv(L)
    d = np.diag(Lp)
    Re = d[ei] + d[ej] - 2 * Lp[ei, ej]
    p = w_eff * np.maximum(Re, 0.0)
    total = p.sum()
    if total == 0:
        p = np.ones(len(ei)) / len(ei)
    else:
        p = p / total
    return ei, ej, we, p


# ---------------------------------------------------------------------------
# Woodbury incremental pseudoinverse
# ---------------------------------------------------------------------------

def _lp_from_w_eff(n, ei, ej, w_eff):
    A_eff = np.zeros((n, n))
    A_eff[ei, ej] = w_eff
    A_eff[ej, ei] = w_eff
    L = np.diag(A_eff.sum(1)) - A_eff
    return np.linalg.pinv(L)


def _woodbury_update(Lp, ei_changed, ej_changed, delta_w):
    """
    Apply sequential rank-1 Woodbury updates to Lp for each changed edge.

    For one edge (i,j) with weight change δw:
        fₑ = eᵢ − eⱼ
        gₑ = Lp @ fₑ           (O(n))
        Rₑ = gₑ · fₑ           (current effective resistance)
        Lp ← Lp − δw/(1+δw·Rₑ) · outer(gₑ, gₑ)   (O(n²))

    Exact (not approximate) as long as the graph stays connected.
    """
    for k in range(len(ei_changed)):
        i_k, j_k = ei_changed[k], ej_changed[k]
        dw = delta_w[k]
        g = Lp[:, i_k] - Lp[:, j_k]   # Lp @ fₑ, reading columns (O(n))
        R_e = g[i_k] - g[j_k]           # gₑ · fₑ
        denom = 1.0 + dw * R_e
        if np.abs(denom) > 1e-10:
            Lp = Lp - (dw / denom) * np.outer(g, g)
    return Lp


def _probs_from_lp(Lp, ei, ej, w_eff):
    d = np.diag(Lp)
    Re = d[ei] + d[ej] - 2 * Lp[ei, ej]
    p = w_eff * np.maximum(Re, 0.0)
    s = p.sum()
    return p / s if s > 0 else np.ones(len(ei)) / len(ei)


def integrate_woodbury_er(A, omega, theta_0, K, t_eval, q, rng, resparse_every,
                           floor_frac=FLOOR_FRAC, rel_thresh=0.02,
                           fallback_frac=0.3, refresh_every=40):
    """
    Like dyn_abs_er but maintains L^+ incrementally using exact rank-1
    Woodbury updates instead of recomputing pinv from scratch each step.

    At each resparsification:
      - Compute δwₑ = new_w_eff_e − old_w_eff_e for all edges.
      - Edges where |δwₑ| / wₑ > rel_thresh get a rank-1 Woodbury update (O(n²) each).
      - If more than fallback_frac of edges need updating, or every refresh_every
        chunks, fall back to a full pinv recompute for numerical stability.

    Speed vs dyn_abs_er: O(r·n²) per step where r = number of changed edges,
    vs O(n³). Benefit grows as the network synchronises and fewer edges cross
    the rel_thresh boundary each step.
    """
    n = A.shape[0]
    ei, ej = np.where(np.triu(A, 1))
    we = A[ei, ej].astype(float)
    E = len(ei)

    cos_d = np.cos(theta_0[ej] - theta_0[ei])
    w_eff = we * (np.abs(cos_d) + floor_frac)
    Lp = _lp_from_w_eff(n, ei, ej, w_eff)

    n_pts = len(t_eval)
    sol_chunks = []
    theta = theta_0.copy()
    t_current = t_eval[0]
    idx = 0
    chunk_count = 0

    while idx < n_pts:
        p = _probs_from_lp(Lp, ei, ej, w_eff)
        A_sparse = sparsify_stochastic(ei, ej, we, p, q, rng, n)

        end_idx = min(idx + resparse_every, n_pts)
        t_chunk = t_eval[idx:end_idx]
        chunk_sol = _integrate_chunk(theta, t_current, t_chunk, K, omega, A_sparse)
        sol_chunks.append(chunk_sol)
        theta = chunk_sol[:, -1]
        t_current = t_chunk[-1]
        idx = end_idx
        chunk_count += 1

        if idx >= n_pts:
            break

        cos_d_new = np.cos(theta[ej] - theta[ei])
        w_eff_new = we * (np.abs(cos_d_new) + floor_frac)
        delta_w = w_eff_new - w_eff

        changed = np.where(np.abs(delta_w) > rel_thresh * we)[0]

        if len(changed) > fallback_frac * E or chunk_count % refresh_every == 0:
            Lp = _lp_from_w_eff(n, ei, ej, w_eff_new)
        else:
            Lp = _woodbury_update(Lp, ei[changed], ej[changed], delta_w[changed])

        w_eff = w_eff_new

    return np.hstack(sol_chunks)


# ---------------------------------------------------------------------------
# integration helpers
# ---------------------------------------------------------------------------

def _integrate_chunk(theta, t_current, t_chunk, K, omega, A_sparse):
    if t_chunk[-1] == t_current:
        # degenerate span (first chunk when resparse_every=1): return IC as column
        return theta.reshape(-1, 1)
    edges = edges_from_A(A_sparse)
    degree = weighted_degree(A_sparse)
    result = solve_ivp(
        kuramoto_rhs, (t_current, t_chunk[-1]), theta,
        args=(K, omega, edges, degree),
        t_eval=t_chunk, **solver_kw,
    )
    if not result.success:
        raise RuntimeError(f'solver failed: {result.message}')
    return result.y


def integrate_full(A, omega, theta_0, K, t_eval):
    edges = edges_from_A(A)
    degree = weighted_degree(A)
    result = solve_ivp(
        kuramoto_rhs, (t_eval[0], t_eval[-1]), theta_0,
        args=(K, omega, edges, degree),
        t_eval=t_eval, **solver_kw,
    )
    return result.y


def integrate_static(A, omega, theta_0, K, t_eval, q, rng, method='er', precomp_er=None):
    """Sparsify once at t=0, integrate on fixed sparse network."""
    n = A.shape[0]
    if method == 'er':
        ei, ej, we, pe = precomp_er
    else:  # weight_static
        ei, ej, we = edges_from_A(A)
        pe = we / we.sum()
    A_sparse = sparsify_stochastic(ei, ej, we, pe, q, rng, n)
    edges = edges_from_A(A_sparse)
    degree = weighted_degree(A_sparse)
    result = solve_ivp(
        kuramoto_rhs, (t_eval[0], t_eval[-1]), theta_0,
        args=(K, omega, edges, degree),
        t_eval=t_eval, **solver_kw,
    )
    return result.y


def integrate_dynamic(A, omega, theta_0, K, t_eval, q, rng, resparse_every,
                      method='dyn_cos', precomp_er=None, floor_frac=FLOOR_FRAC):
    """
    Rebuild sparse network from current phases every resparse_every timesteps.

    Integration is piecewise: each chunk uses the sparse network built from the
    phases at the START of that chunk.  The initial condition (theta) is carried
    over continuously — the solver restarts from the last computed state each chunk.
    """
    n = A.shape[0]
    n_pts = len(t_eval)
    sol_chunks = []
    theta = theta_0.copy()
    t_current = t_eval[0]
    idx = 0

    while idx < n_pts:
        # ---- compute dynamic sampling probabilities ----
        if method == 'dyn_cos':
            ei, ej, we = edges_from_A(A)
            pe = probs_dyn_cos(ei, ej, we, theta, floor_frac)
        elif method == 'dyn_er_cos':
            ei, ej, we, pe_er = precomp_er
            pe = probs_dyn_er_cos(pe_er, ei, ej, theta, floor_frac)
        elif method == 'dyn_abs_er':
            ei, ej, we, pe = probs_dyn_abs_er(A, theta, floor_frac)
        else:
            raise ValueError(method)

        # ---- build sparse network (unbiased w.r.t. original weights) ----
        A_sparse = sparsify_stochastic(ei, ej, we, pe, q, rng, n)

        end_idx = min(idx + resparse_every, n_pts)
        t_chunk = t_eval[idx:end_idx]

        chunk_sol = _integrate_chunk(theta, t_current, t_chunk, K, omega, A_sparse)
        sol_chunks.append(chunk_sol)
        theta = chunk_sol[:, -1]
        t_current = t_chunk[-1]
        idx = end_idx

    return np.hstack(sol_chunks)


def integrate_oracle(A, sol_full, omega, theta_0, K, t_eval, q, rng,
                     resparse_every, floor_frac=FLOOR_FRAC):
    """
    Oracle version of dyn_abs_er: uses the FULL network's phases to decide
    which edges to sample at each resparsification step, but still integrates
    forward on the resulting sparse network.

    This isolates whether the dynamic method's errors come from (a) the
    cosine-ER sampling strategy being wrong in principle, or (b) the phase
    estimates drifting because we're integrating on the reduced graph.  The
    oracle can't drift — it always sees the true phases — so if it's much
    better than dyn_abs_er, phase estimation error is the main bottleneck.
    """
    n = A.shape[0]
    n_pts = len(t_eval)
    sol_chunks = []
    theta = theta_0.copy()
    t_current = t_eval[0]
    idx = 0

    while idx < n_pts:
        theta_oracle = sol_full[:, idx]   # true phases from full network at this moment
        ei, ej, we, pe = probs_dyn_abs_er(A, theta_oracle, floor_frac)
        A_sparse = sparsify_stochastic(ei, ej, we, pe, q, rng, n)

        end_idx = min(idx + resparse_every, n_pts)
        t_chunk = t_eval[idx:end_idx]

        chunk_sol = _integrate_chunk(theta, t_current, t_chunk, K, omega, A_sparse)
        sol_chunks.append(chunk_sol)
        theta = chunk_sol[:, -1]
        t_current = t_chunk[-1]
        idx = end_idx

    return np.hstack(sol_chunks)


# ---------------------------------------------------------------------------
# comparison sweep
# ---------------------------------------------------------------------------

def run_comparison(A, qs, n_seeds, base_seed, resparse_every=RESPARSE_EVERY,
                   t_max=T_MAX, n_t=N_T, include_abs_er=True):
    """
    For every (q, seed), run full + all sparse methods.  Returns:
      err_mean[method]    shape (len(qs), n_seeds)  — mean |Δr| over full trajectory
      err_steady[method]  shape (len(qs), n_seeds)  — mean |Δr| over last 50%
      phase_mean[method]  shape (len(qs), n_seeds)  — mean |Δθᵢ| (°) over full trajectory
      phase_steady[method] shape (len(qs), n_seeds) — mean |Δθᵢ| (°) over last 50%
      t_eval              (n_t,) time points
    """
    n = A.shape[0]
    K = K_COEF * n
    t_eval = np.linspace(0, t_max, n_t)
    late = t_eval >= t_eval[-1] * 0.5

    methods = ['er_static', 'weight_static', 'dyn_cos', 'dyn_er_cos']
    if include_abs_er:
        methods.append('dyn_abs_er')
        methods.append('oracle_abs_er')
        methods.append('woodbury_er')

    err_mean     = {m: np.zeros((len(qs), n_seeds)) for m in methods}
    err_steady   = {m: np.zeros((len(qs), n_seeds)) for m in methods}
    phase_mean   = {m: np.zeros((len(qs), n_seeds)) for m in methods}
    phase_steady = {m: np.zeros((len(qs), n_seeds)) for m in methods}

    # precompute ER once (shared across seeds and q values)
    precomp_er = precompute_er(A)

    for si in range(n_seeds):
        rng = np.random.default_rng(base_seed + 7919 * si)
        omega   = rng.normal(5.0, 0.5, n)
        theta_0 = rng.uniform(0, 2 * np.pi, n)

        sol_full = integrate_full(A, omega, theta_0, K, t_eval)
        r_full   = order_param(sol_full)

        for qi, q in enumerate(qs):
            seed_offset = base_seed + si * 997 + qi * 13

            def run_method(m, method_fn, *args):
                sol = method_fn(*args)
                r = order_param(sol)
                diff = np.abs(r - r_full)
                err_mean[m][qi, si]     = diff.mean()
                err_steady[m][qi, si]   = diff[late].mean()
                pe = np.degrees(_mean_phase_err(sol, sol_full))
                phase_mean[m][qi, si]   = pe.mean()
                phase_steady[m][qi, si] = pe[late].mean()

            run_method('er_static',
                integrate_static, A, omega, theta_0, K, t_eval, q,
                np.random.default_rng(seed_offset),
                'er', precomp_er)

            run_method('weight_static',
                integrate_static, A, omega, theta_0, K, t_eval, q,
                np.random.default_rng(seed_offset + 1),
                'weight', precomp_er)

            run_method('dyn_cos',
                integrate_dynamic, A, omega, theta_0, K, t_eval, q,
                np.random.default_rng(seed_offset + 2), resparse_every,
                'dyn_cos', precomp_er)

            run_method('dyn_er_cos',
                integrate_dynamic, A, omega, theta_0, K, t_eval, q,
                np.random.default_rng(seed_offset + 3), resparse_every,
                'dyn_er_cos', precomp_er)

            if include_abs_er:
                run_method('dyn_abs_er',
                    integrate_dynamic, A, omega, theta_0, K, t_eval, q,
                    np.random.default_rng(seed_offset + 4), resparse_every,
                    'dyn_abs_er', precomp_er)

                run_method('oracle_abs_er',
                    integrate_oracle, A, sol_full, omega, theta_0, K, t_eval, q,
                    np.random.default_rng(seed_offset + 5), resparse_every)

                run_method('woodbury_er',
                    integrate_woodbury_er, A, omega, theta_0, K, t_eval, q,
                    np.random.default_rng(seed_offset + 6), resparse_every)

        print(f'  seed {si + 1}/{n_seeds} done')

    return err_mean, err_steady, phase_mean, phase_steady, t_eval


def resparse_interval_sweep(A, q, resparse_vals, n_seeds, base_seed,
                            include_abs_er=True, t_max=T_MAX, n_t=N_T):
    """
    Fix q, sweep over resparse_every values.
    Returns err_mean, err_steady, phase_mean, phase_steady dicts
    keyed by method with shape (len(resparse_vals), n_seeds).
    """
    n = A.shape[0]
    K = K_COEF * n
    t_eval = np.linspace(0, t_max, n_t)
    precomp_er = precompute_er(A)

    methods = ['dyn_cos', 'dyn_er_cos']
    if include_abs_er:
        methods += ['dyn_abs_er', 'woodbury_er']
    err_mean     = {m: np.zeros((len(resparse_vals), n_seeds)) for m in methods}
    err_steady   = {m: np.zeros((len(resparse_vals), n_seeds)) for m in methods}
    phase_mean   = {m: np.zeros((len(resparse_vals), n_seeds)) for m in methods}
    phase_steady = {m: np.zeros((len(resparse_vals), n_seeds)) for m in methods}
    late = t_eval >= t_eval[-1] * 0.5

    for si in range(n_seeds):
        rng = np.random.default_rng(base_seed + 7919 * si)
        omega   = rng.normal(5.0, 0.5, n)
        theta_0 = rng.uniform(0, 2 * np.pi, n)
        sol_full = integrate_full(A, omega, theta_0, K, t_eval)
        r_full   = order_param(sol_full)

        for ri, re in enumerate(resparse_vals):
            for mi, m in enumerate(methods):
                seed_i = np.random.default_rng(base_seed + si * 997 + ri * 13 + mi)
                if m in ('dyn_cos', 'dyn_er_cos', 'dyn_abs_er'):
                    sol = integrate_dynamic(A, omega, theta_0, K, t_eval, q,
                                            seed_i, re, m, precomp_er)
                else:  # woodbury_er
                    sol = integrate_woodbury_er(A, omega, theta_0, K, t_eval, q,
                                                seed_i, re)
                r = order_param(sol)
                diff = np.abs(r - r_full)
                err_mean[m][ri, si]     = diff.mean()
                err_steady[m][ri, si]   = diff[late].mean()
                pe = np.degrees(_mean_phase_err(sol, sol_full))
                phase_mean[m][ri, si]   = pe.mean()
                phase_steady[m][ri, si] = pe[late].mean()

        print(f'  interval sweep seed {si + 1}/{n_seeds} done')

    return err_mean, err_steady, phase_mean, phase_steady


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_error_vs_q(qs, err_mean, err_steady, methods, n_seeds, plots_dir, stem, note='',
                    phase_mean=None, phase_steady=None):
    has_phase = phase_mean is not None and phase_steady is not None
    nrows = 2 if has_phase else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]   # shape (1, 2)

    ax_r1, ax_r2 = axes[0]
    if has_phase:
        ax_p1, ax_p2 = axes[1]

    for m in methods:
        if m not in err_mean:
            continue
        color = METHOD_COLORS[m]
        label = METHOD_LABELS[m]
        ls = '-' if 'static' in m else '--'

        for ax, data in [(ax_r1, err_mean[m]), (ax_r2, err_steady[m])]:
            mean = data.mean(axis=1)
            moe  = 1.96 * data.std(axis=1) / np.sqrt(n_seeds)
            ax.plot(qs, mean, color=color, ls=ls, marker='o', label=label)
            ax.fill_between(qs, np.maximum(mean - moe, 0), mean + moe,
                            color=color, alpha=0.15)

        if has_phase:
            for ax, data in [(ax_p1, phase_mean[m]), (ax_p2, phase_steady[m])]:
                mean = data.mean(axis=1)
                moe  = 1.96 * data.std(axis=1) / np.sqrt(n_seeds)
                ax.plot(qs, mean, color=color, ls=ls, marker='o', label=label)
                ax.fill_between(qs, np.maximum(mean - moe, 0), mean + moe,
                                color=color, alpha=0.15)

    row_axes = [(ax_r1, ax_r2, 'mean |Δr|')]
    if has_phase:
        row_axes.append((ax_p1, ax_p2, 'mean |Δθᵢ| (°)'))

    for ax_l, ax_r, ylabel in row_axes:
        for ax, suffix in [(ax_l, f'(t = 0–{T_MAX})'), (ax_r, f'(steady state, t = {T_MAX//2}–{T_MAX})')]:
            ax.set_xscale('log')
            ax.set_xticks(qs)
            ax.set_xticklabels([f'{q:g}' for q in qs])
            ax.minorticks_off()
            ax.set_xlabel('fraction of edges preserved (q)')
            ax.set_ylabel(f'{ylabel} {suffix}')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    title = 'dynamic vs static sparsification: order-parameter error'
    if note:
        title += f'\n{note}'
    fig.suptitle(title)
    fig.tight_layout()
    path = plots_dir / f'{stem}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  wrote {path}')


def plot_interval_sweep(resparse_vals, err_mean, err_steady, n_seeds,
                        q, plots_dir, stem, note='',
                        phase_mean=None, phase_steady=None):
    methods = list(err_mean.keys())
    has_phase = phase_mean is not None and phase_steady is not None
    nrows = 2 if has_phase else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    panel_specs = [(axes[0, 0], err_mean,   f'mean |Δr| (t = 0–{T_MAX})'),
                   (axes[0, 1], err_steady, f'mean |Δr| (steady state, t = {T_MAX//2}–{T_MAX})')]
    if has_phase:
        panel_specs += [(axes[1, 0], phase_mean,   f'mean |Δθᵢ| (°) (t = 0–{T_MAX})'),
                        (axes[1, 1], phase_steady, f'mean |Δθᵢ| (°) (steady state, t = {T_MAX//2}–{T_MAX})')]

    for ax, data_dict, ylabel in panel_specs:
        for m in methods:
            d = data_dict[m]
            mean = d.mean(axis=1)
            moe  = 1.96 * d.std(axis=1) / np.sqrt(n_seeds)
            ax.plot(resparse_vals, mean, marker='o',
                    color=METHOD_COLORS[m], label=METHOD_LABELS[m])
            ax.fill_between(resparse_vals, np.maximum(mean - moe, 0), mean + moe,
                            color=METHOD_COLORS[m], alpha=0.15)
        ax.set_xlabel('resparsification interval (timesteps)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.invert_xaxis()   # left = frequent, right = rare (static limit)

    title = f'error vs resparsification frequency (q={q:g})'
    if note:
        title += f'\n{note}'
    fig.suptitle(title)
    fig.tight_layout()
    path = plots_dir / f'{stem}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  wrote {path}')


def _mean_phase_err(sol_sparse, sol_full):
    """Mean over oscillators of |Δθ_i(t)| wrapped to [0, π]."""
    dphi = np.angle(np.exp(1j * (sol_sparse - sol_full)))   # wrapped to (−π, π]
    return np.abs(dphi).mean(axis=0)                         # (T,)


def plot_trajectory(A, K, t_eval, q, resparse_every, precomp_er,
                    base_seed, n_traj_seeds, methods, plots_dir, stem, note=''):
    """
    Average r(t), |Δr(t)|, and mean per-oscillator |Δθ(t)| over n_traj_seeds ICs.
    Plots mean ± 1 std band for each method.
    """
    n = A.shape[0]
    T = len(t_eval)

    # accumulators: sum and sum-of-squares for online mean/std
    r_full_sum  = np.zeros(T);  r_full_sq  = np.zeros(T)
    acc = {m: {'r': np.zeros(T), 'r_sq': np.zeros(T),
               'dr': np.zeros(T), 'dr_sq': np.zeros(T),
               'pe': np.zeros(T), 'pe_sq': np.zeros(T)} for m in methods}

    for si in range(n_traj_seeds):
        rng = np.random.default_rng(base_seed + 7919 * si)
        omega   = rng.normal(5.0, 0.5, n)
        theta_0 = rng.uniform(0, 2 * np.pi, n)
        sol_full = integrate_full(A, omega, theta_0, K, t_eval)
        rf = order_param(sol_full)
        r_full_sum += rf;  r_full_sq += rf ** 2

        for mi, m in enumerate(methods):
            s = np.random.default_rng(base_seed + si * 997 + mi * 13)
            if m == 'er_static':
                sol = integrate_static(A, omega, theta_0, K, t_eval, q, s, 'er', precomp_er)
            elif m == 'weight_static':
                sol = integrate_static(A, omega, theta_0, K, t_eval, q, s, 'weight', precomp_er)
            elif m == 'oracle_abs_er':
                sol = integrate_oracle(A, sol_full, omega, theta_0, K, t_eval, q, s, resparse_every)
            elif m == 'woodbury_er':
                sol = integrate_woodbury_er(A, omega, theta_0, K, t_eval, q, s, resparse_every)
            else:
                sol = integrate_dynamic(A, omega, theta_0, K, t_eval, q, s,
                                        resparse_every, m, precomp_er)
            r   = order_param(sol)
            dr  = np.abs(r - rf)
            pe  = np.degrees(_mean_phase_err(sol, sol_full))
            a = acc[m]
            a['r'] += r;   a['r_sq'] += r ** 2
            a['dr'] += dr; a['dr_sq'] += dr ** 2
            a['pe'] += pe; a['pe_sq'] += pe ** 2

    S = n_traj_seeds

    def ms(s, sq):
        mean = s / S
        std  = np.sqrt(np.maximum(sq / S - mean ** 2, 0))
        return mean, std

    r_full_mean, r_full_std = ms(r_full_sum, r_full_sq)

    fig, (ax_r, ax_dr, ax_phase) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax_r.plot(t_eval, r_full_mean, 'k-', lw=2, label='full')
    ax_r.fill_between(t_eval, r_full_mean - r_full_std, r_full_mean + r_full_std,
                      color='k', alpha=0.1)

    for m in methods:
        color = METHOD_COLORS[m]
        label = METHOD_LABELS[m]
        ls = '-' if 'static' in m else '--'
        a = acc[m]
        rm, rs   = ms(a['r'],  a['r_sq'])
        drm, drs = ms(a['dr'], a['dr_sq'])
        pem, pes = ms(a['pe'], a['pe_sq'])

        ax_r.plot(t_eval, rm, color=color, ls=ls, label=label)
        ax_r.fill_between(t_eval, np.maximum(rm - rs, 0), rm + rs, color=color, alpha=0.12)

        ax_dr.plot(t_eval, drm, color=color, ls=ls, label=label)
        ax_dr.fill_between(t_eval, np.maximum(drm - drs, 0), drm + drs, color=color, alpha=0.12)

        ax_phase.plot(t_eval, pem, color=color, ls=ls, label=label)
        ax_phase.fill_between(t_eval, np.maximum(pem - pes, 0), pem + pes, color=color, alpha=0.12)

    ax_r.set_ylabel('r(t)')
    ax_r.set_ylim(0, 1.05)
    ax_r.legend(fontsize=7)
    ax_r.grid(alpha=0.3)
    ax_dr.set_ylabel('mean |Δr(t)| over ICs')
    ax_dr.legend(fontsize=7)
    ax_dr.grid(alpha=0.3)
    ax_phase.set_ylabel('mean |Δθᵢ(t)| (°)\naveraged over ICs and oscillators')
    ax_phase.set_xlabel('time')
    ax_phase.legend(fontsize=7)
    ax_phase.grid(alpha=0.3)

    title = (f'IC-averaged trajectories (q={q:g}, resparse every {resparse_every} steps, '
             f'{S} ICs)')
    if note:
        title += f'\n{note}'
    fig.suptitle(title)
    fig.tight_layout()
    path = plots_dir / f'{stem}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  wrote {path}')


# ---------------------------------------------------------------------------
# per-network pipeline
# ---------------------------------------------------------------------------

def run_network(name, A, run_seed, note='', include_abs_er=True):
    n = A.shape[0]
    n_edges = int(np.count_nonzero(np.triu(A, 1)))
    print(f'\n=== {name}  (n={n}, {n_edges} edges) ===')

    plots_dir = PLOT_DIR / name
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- main q-sweep ---
    print('running q-sweep...')
    err_mean, err_steady, phase_mean, phase_steady, t_eval = run_comparison(
        A, QS, N_SEEDS, run_seed,
        resparse_every=RESPARSE_EVERY,
        include_abs_er=include_abs_er,
    )
    methods = list(err_mean.keys())
    plot_error_vs_q(QS, err_mean, err_steady, methods, N_SEEDS,
                    plots_dir, 'error_vs_q', note=note,
                    phase_mean=phase_mean, phase_steady=phase_steady)

    # --- resparsification interval sweep at q=0.25 ---
    print('running interval sweep (q=0.25)...')
    resparse_vals = [1, 2, 3, 5, 10, 20, 30, 60, 120, 300, 600]
    iv_mean, iv_steady, iv_phase_mean, iv_phase_steady = resparse_interval_sweep(
        A, q=0.25, resparse_vals=resparse_vals, n_seeds=N_SEEDS,
        base_seed=run_seed + 1, include_abs_er=include_abs_er,
    )
    plot_interval_sweep(resparse_vals, iv_mean, iv_steady, N_SEEDS,
                        q=0.25, plots_dir=plots_dir, stem='interval_sweep', note=note,
                        phase_mean=iv_phase_mean, phase_steady=iv_phase_steady)

    # --- IC-averaged trajectory at q=0.25 ---
    print('plotting IC-averaged trajectories (q=0.25)...')
    precomp_er  = precompute_er(A)
    K_demo      = K_COEF * n
    t_eval_demo = np.linspace(0, T_MAX, N_T)
    plot_trajectory(A, K_demo, t_eval_demo,
                    q=0.25, resparse_every=RESPARSE_EVERY,
                    precomp_er=precomp_er, base_seed=run_seed + 3,
                    n_traj_seeds=20,
                    methods=methods, plots_dir=plots_dir,
                    stem='trajectory_q0.25', note=note)

    # --- print summary table ---
    print(f'\n  steady-state mean |Δr| at q=0.25:')
    qi_print = np.argmin(np.abs(QS - 0.25))
    for m in methods:
        v = err_steady[m][qi_print].mean()
        print(f'    {METHOD_LABELS[m]:42s}: {v:.4f}')

    print(f'\n  wrote plots to {plots_dir}/')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    run_seed = fresh_seed()
    print(f'run seed: {run_seed}')
    rng = np.random.default_rng(run_seed)

    # connectome_66: small n → can run dyn_abs_er (O(n^3) per step) affordably
    A_conn, _, _ = build_connectome_network(rng)
    run_network('connectome_66', A_conn, run_seed,
                note='human structural connectome, 66 cortical regions',
                include_abs_er=True)

    A_cp, _, _ = build_heterogeneous_network(rng)
    run_network('core_periphery', A_cp, run_seed + 1,
                note='synthetic core-periphery (dense core + tree-like periphery)',
                include_abs_er=True)

    A_ce, _, _ = build_celegans_network(rng)
    run_network('celegans', A_ce, run_seed + 2,
                note='C. elegans neural connectome, 297 neurons',
                include_abs_er=True)

    A_sf, _, _ = build_scale_free_network(rng)
    run_network('scale_free_hub', A_sf, run_seed + 3,
                note='scale-free network (preferential attachment, n=150)',
                include_abs_er=True)

    A_uk, _, _ = build_uk_power_grid_network(rng)
    run_network('uk_power_grid', A_uk, run_seed + 4,
                note='UK power grid',
                include_abs_er=True)

    A_fc, _, _ = build_fully_connected_network(rng)
    run_network('fully_connected', A_fc, run_seed + 5,
                note='fully connected (n=66)',
                include_abs_er=True)

    A_ring, _, _ = build_ring_network(rng)
    run_network('ring', A_ring, run_seed + 6,
                note='ring network (n=66, k=5)',
                include_abs_er=True)

    A_ci, _, _ = build_cintestinalis_network(rng)
    run_network('cintestinalis', A_ci, run_seed + 7,
                note='C. intestinalis neural connectome',
                include_abs_er=True)


if __name__ == '__main__':
    main()
