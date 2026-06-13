"""
Adaptive ER sparsification with L(θ) = L₊ − L₋ split.

|Δθ| < π/2  → L₊ (cos > 0): ER-sparsify with valid PSD Laplacian walks.
|Δθ| ≥ π/2  → L₋ (frustrated): keep every edge at w = +1 (sin handles repulsion).

Run, then animate:
    python adaptive_sparsify_kuramoto.py
    python animate_kuramoto.py adaptive
"""
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def fresh_seed():
    return int.from_bytes(os.urandom(8), 'big') % (2**63)


RUN_SEED = fresh_seed()
print(f'run seed: {RUN_SEED}')

# problem setup — FC homogeneous weights
N = 64
# k/N ≈ 1–2 → supercritical (K/(N−1) well above K_c ≈ 0.8 for ω ~ N(5, 0.5))
k_over_n = 1.5
K = k_over_n * N
q = 0.25
A_name = 'A_fc'

t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# resparsify every this many t_eval points (tunable)
resparsify_every = 10

bool_plots = True
solver_kw = dict(method='DOP853', rtol=1e-9, atol=1e-11)

rng = np.random.default_rng(RUN_SEED)
A = np.ones((N, N), dtype=float) - np.eye(N)
omega = rng.normal(loc=5.0, scale=0.5, size=N)
theta_0 = rng.uniform(0, 2 * np.pi, N)


def edges_from_A(A_mat):
    ei, ej = np.where(np.triu(A_mat, 1))
    return ei, ej, A_mat[ei, ej].astype(float)


def weighted_degree(A_mat):
    return np.maximum(A_mat.sum(axis=1), 1e-12)


def kuramoto_rhs(t, theta, K_val, omega_vec, edges, degree):
    ei, ej, w = edges
    coupling = np.zeros(N)
    np.add.at(coupling, ei, w * np.sin(theta[ej] - theta[ei]))
    np.add.at(coupling, ej, w * np.sin(theta[ei] - theta[ej]))
    return omega_vec + K_val * coupling / degree


def top_weight_edges(A_mat, max_edges):
    ei, ej, w = edges_from_A(A_mat)
    if len(ei) <= max_edges:
        return ei, ej, w
    idx = np.argpartition(w, -max_edges)[-max_edges:]
    return ei[idx], ej[idx], w[idx]


def split_w_plus_minus(theta):
    """W = W₊ − W₋ with W₊, W₋ ≥ 0 entrywise; cos > 0 ↔ |Δθ| < π/2 (pure cosine, no K/N)."""
    delta = theta[None, :] - theta[:, None]
    cos_d = np.cos(delta)
    np.fill_diagonal(cos_d, 0.0)
    W_plus = np.where(cos_d > 0, cos_d, 0.0)
    W_minus = np.where(cos_d < 0, -cos_d, 0.0)
    return W_plus, W_minus


def effective_resistances_from_laplacian(A_laplacian):
    graph_laplacian = np.diag(A_laplacian.sum(1)) - A_laplacian
    graph_laplacian_pinv = np.linalg.pinv(graph_laplacian)
    diag = np.diag(graph_laplacian_pinv)
    return diag[:, None] + diag[None, :] - 2 * graph_laplacian_pinv


def er_sparsify_positive(W_plus, q_frac, rng):
    """ER on L₊; leverage p_e ∝ cos·R_eff; sin weights cos/(s·p_e), normalized by degree_full."""
    n = W_plus.shape[0]
    edge_i, edge_j = np.where(np.triu(W_plus, 1))
    if len(edge_i) == 0:
        return np.zeros((n, n))

    effective_resistances = effective_resistances_from_laplacian(W_plus)
    we = W_plus[edge_i, edge_j]
    Re = we * effective_resistances[edge_i, edge_j]
    pe = Re / Re.sum()
    s = max(1, int(q_frac * len(edge_i)))
    sparsified = np.zeros((n, n))
    for idx in rng.choice(len(edge_i), s, replace=True, p=pe):
        w = we[idx] / (s * pe[idx])
        sparsified[edge_i[idx], edge_j[idx]] += w
        sparsified[edge_j[idx], edge_i[idx]] += w
    return sparsified


def frustrate_edges_exact(W_minus):
    """All L₋ edges kept at base coupling w = +1 (not sparsified, not sign-flipped)."""
    n = W_minus.shape[0]
    A_frust = np.zeros((n, n))
    edge_i, edge_j = np.where(np.triu(W_minus, 1))
    A_frust[edge_i, edge_j] = 1.0
    A_frust[edge_j, edge_i] = 1.0
    return A_frust


def split_sparsify(theta, q_frac, rng):
    W_plus, W_minus = split_w_plus_minus(theta)
    A_plus = er_sparsify_positive(W_plus, q_frac, rng)
    A_minus = frustrate_edges_exact(W_minus)
    return A_plus + A_minus


def integrate_adaptive(theta_init, t_eval_pts, resparsify_step, q_frac, seed, degree_ref):
    sparsify_rng = np.random.default_rng(seed)
    n_pts = len(t_eval_pts)
    sol_cols = []
    snapshots = []
    theta = theta_init.copy()
    chunk_start = 0

    while chunk_start < n_pts:
        A_sparse = split_sparsify(theta, q_frac, sparsify_rng)
        edges = edges_from_A(A_sparse)
        snapshots.append({'t_idx': chunk_start, 'A': A_sparse.copy()})

        chunk_end = min(chunk_start + resparsify_step, n_pts)
        t_chunk = t_eval_pts[chunk_start:chunk_end]
        t_lo = float(t_chunk[0])
        t_hi = float(t_chunk[-1])
        result = solve_ivp(
            kuramoto_rhs, (t_lo, t_hi), theta,
            args=(K, omega, edges, degree_ref),
            t_eval=t_chunk, **solver_kw,
        )
        if not result.success:
            raise RuntimeError(f'integration failed at t={t_lo}: {result.message}')
        sol_cols.append(result.y)
        theta = result.y[:, -1]
        chunk_start = chunk_end
        if chunk_start % (resparsify_step * 10) == 0 or chunk_start >= n_pts:
            print(f'  adaptive: {chunk_start}/{n_pts} t_eval points')

    return np.hstack(sol_cols), snapshots


def order_param_series(theta_t):
    z = np.mean(np.exp(1j * theta_t), axis=0)
    return np.abs(z), z


edges_full = edges_from_A(A)
degree_full = weighted_degree(A)

print('running full FC reference...')
solution_full = solve_ivp(
    kuramoto_rhs, t_span, theta_0, args=(K, omega, edges_full, degree_full),
    t_eval=t_eval, **solver_kw,
)
sol_full = solution_full.y

print(f'running adaptive sparse (refresh every {resparsify_every} t_eval points)...')
sol_adaptive, A_snapshots = integrate_adaptive(
    theta_0, t_eval, resparsify_every, q, RUN_SEED + 1, degree_full,
)

# names animate_kuramoto.py expects in adaptive mode
sol_ER = sol_adaptive
A_sparse_ER = A_snapshots[-1]['A']
random_weights_std = None
edge_density = None
ring_k = None

r_full, z = order_param_series(sol_full)
r_sparse, z_sparse = order_param_series(sol_ER)
err_r_ER = np.abs(r_full - r_sparse)
err_psi_ER = np.degrees(np.abs(np.angle(z * np.conj(z_sparse))))

if __name__ == '__main__' and bool_plots:
    fig, (ax_r, ax_dr, ax_dpsi) = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax_r.plot(t_eval, r_full, label='full FC', color='C0')
    ax_r.plot(t_eval, r_sparse, label='adaptive sparse', color='C1', linestyle='--')
    ax_r.set_ylabel('r = |⟨e^{iθ}⟩|')
    ax_r.set_ylim(0, 1.05)
    ax_r.legend()
    ax_r.set_title(f'adaptive ER (refresh every {resparsify_every} steps)')

    ax_dr.plot(t_eval, err_r_ER, color='C1')
    ax_dr.set_ylabel('|Δr|')

    ax_dpsi.plot(t_eval, err_psi_ER, color='C1')
    ax_dpsi.set_ylabel('|Δψ| (°)')
    ax_dpsi.set_xlabel('time (s)')

    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f'adaptive_kuramoto_n={N}_q={q}_refresh={resparsify_every}.png'
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f'saved {out}')
    plt.show()
