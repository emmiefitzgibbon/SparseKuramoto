"""
Interactive Kuramoto comparison (full vs ER-sparsified graph).
Run: python interactive_kuramoto.py

Physics sliders auto-re-run ~0.7s after you stop dragging (ODE solve is slow).
Coupling is K times mean neighbor sin term (normalized by node degree, not N).
Time slider scrubs frames; Play animates. N slider (25–900) rebuilds the graph.
Re-runs reset to t=0. Square lattice snaps N to a perfect square.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from scipy.integrate import solve_ivp
import networkx as nx

# --- defaults (match kuramoto.py) ---
N_MIN, N_MAX = 25, 900
N = 625
L = int(np.sqrt(N))
R = 0.35
rows = cols = ring_idx = None
RING_R = L / 2
t_end_default = 5.0
t_end_max = 60.0
n_t_eval = 150
n_t_eval_max = 800
steady_fraction = 0.2
freq_lock_cutoff = 0.1  # |dθ/dt − Ω_pack| below this ⇒ in the pack
anim_frames_base = 100
sim_dt_per_frame = t_end_default / anim_frames_base  # sim seconds per animation frame
play_interval_ms = 50
anim_max_edges = 1500
debounce_ms = 700


def fresh_seed():
    return int.from_bytes(os.urandom(8), 'big') % (2**63)


# paste kuramoto.py's printed run seed here to reproduce the same problem
RUN_SEED = fresh_seed()
K_MIN, K_MAX = 0.1, 100.0
LOG_K_MIN, LOG_K_MAX = np.log10(K_MIN), np.log10(K_MAX)
solver_kw = dict(method='RK45', rtol=1e-6, atol=1e-8)
DATA_DIR = Path(__file__).resolve().parent / 'data'
DEFAULT_TOPOLOGY = 'FC random weights'
random_weights_std = 1.0
KURAMOTO_EDGE_DENSITY = 0.75  # match kuramoto.py rng consumption
TOPOLOGY_NAMES = (
    'FC (w=1)',
    'square lattice',
    'FC random weights',
    'random edges',
    'random edge weights',
    'ring (k-NN)',
    'ring falloff (1/d)',
    'ring het. weights',
)
RING_TOPOLOGIES = ('ring (k-NN)', 'ring falloff (1/d)', 'ring het. weights')
RING_K_DEFAULT = 25  # fast default; k/N≈0.04 — raise toward ~212 (k_c) to study sync transition
RING_K_MIN = 1
ring_k = RING_K_DEFAULT
coupling_state = {'normalize': True}  # /degree by default; ring locks to unnormalized + K=1
RING_PAPER_K = 1.0
saved_before_ring = {'log_k': None}


def weighted_degree(A):
    return np.maximum(A.sum(axis=1), 1e-12)


def ring_k_max():
    return max(RING_K_MIN, N // 2)


def sync_n_geometry():
    """refresh layout globals after N changes."""
    global L, rows, cols, ring_idx, RING_R
    L = int(np.sqrt(N))
    rows = np.arange(N) // L
    cols = np.arange(N) % L
    ring_idx = np.arange(N)
    RING_R = L / 2


def set_n(n):
    global N
    N = int(np.clip(n, N_MIN, N_MAX))
    sync_n_geometry()


def snap_n_for_topology(n, topo):
    n = int(np.clip(n, N_MIN, N_MAX))
    if topo == 'square lattice':
        side = int(round(np.sqrt(n)))
        side = max(int(np.sqrt(N_MIN)), min(side, int(np.sqrt(N_MAX))))
        return side * side
    return n


def generate_ring_matrix(n, k):
    offsets = list(range(1, int(k) + 1))
    G = nx.circulant_graph(n, offsets)
    return nx.to_numpy_array(G, dtype=float)


def build_ring_A(topology_name, k_ring, rng=None):
    base = generate_ring_matrix(N, k_ring)
    if topology_name == 'ring (k-NN)':
        return base
    ei, ej = np.where(np.triu(base, 1))
    if topology_name == 'ring falloff (1/d)':
        d = np.minimum(np.abs(ei - ej), N - np.abs(ei - ej))
        w = 1.0 / d
    else:
        rng = rng or np.random.default_rng()
        w = rng.lognormal(0, random_weights_std, len(ei))
    A = np.zeros((N, N))
    A[ei, ej] = w
    A[ej, ei] = w
    return A


def setup_from_run_seed(run_seed, topology_name, k_ring=RING_K_DEFAULT):
    """build only the selected graph; mirror kuramoto.py rng for non-ring topologies."""
    rng = np.random.default_rng(run_seed)
    if topology_name in RING_TOPOLOGIES:
        weight_rng = (
            np.random.default_rng(run_seed + 2)
            if topology_name == 'ring het. weights' else None
        )
        A = build_ring_A(topology_name, k_ring, weight_rng)
        omega = np.full(N, 5.0)
        theta_0 = rng.uniform(0, 2 * np.pi, N)
        return A, omega, theta_0

    i, j = np.triu_indices(N, k=1)
    exists = rng.random(len(i)) < KURAMOTO_EDGE_DENSITY
    weights = rng.lognormal(0, random_weights_std, len(i))

    A_random_edges = np.zeros((N, N))
    A_random_edges[i[exists], j[exists]] = 1
    A_random_edges += A_random_edges.T

    A_random_edges_weights = np.zeros((N, N))
    A_random_edges_weights[i[exists], j[exists]] = weights[exists]
    A_random_edges_weights += A_random_edges_weights.T

    W = rng.lognormal(0, random_weights_std, (N, N))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    A_random_weights = (np.ones((N, N)) - np.eye(N)) * W

    grid = nx.grid_2d_graph(L, L)
    grid = nx.relabel_nodes(grid, {(r, c): r * L + c for r, c in grid.nodes()})
    A_square = nx.to_numpy_array(grid, dtype=float)
    if topology_name == 'square lattice' and L * L != N:
        raise ValueError(f'square lattice requires perfect-square N, got {N}')
    A_fc = np.ones((N, N)) - np.eye(N)

    graphs = {
        'FC (w=1)': A_fc,
        'square lattice': A_square,
        'FC random weights': A_random_weights,
        'random edges': A_random_edges,
        'random edge weights': A_random_edges_weights,
    }
    omega = rng.normal(loc=5.0, scale=0.5, size=N)
    theta_0 = rng.uniform(0, 2 * np.pi, N)
    return graphs[topology_name], omega, theta_0


resample_count = 0
topology_name = DEFAULT_TOPOLOGY
print(f'run seed: {RUN_SEED}')


def problem_run_seed():
    return RUN_SEED + resample_count


sync_n_geometry()


def use_ring_layout():
    return topology_name in RING_TOPOLOGIES


phase_cmap = plt.cm.twilight


def phase_facecolors(theta):
    return phase_cmap((theta / (2 * np.pi)) % 1.0)


def estimate_winding(theta):
    dtheta = np.angle(np.exp(1j * (np.roll(theta, -1) - theta)))
    return int(np.round(np.sum(dtheta) / (2 * np.pi)))


def phase_profile_deg(theta):
    unwrapped = np.unwrap(theta)
    return np.degrees(unwrapped - unwrapped[0])


def nodes_xy(indices):
    idx = np.asarray(indices)
    if use_ring_layout():
        ang = 2 * np.pi * idx / N - np.pi / 2
        return RING_R * np.cos(ang), RING_R * np.sin(ang)
    return (idx % L).astype(float), -(idx // L).astype(float)


def panel_limits():
    if use_ring_layout():
        pad = R + 0.5
        return (-RING_R - pad, RING_R + pad), (-RING_R - pad, RING_R + pad)
    return (-0.6, L - 0.4), (-L + 0.4, 0.6)


def edges_from_A(A):
    ei, ej = np.where(np.triu(A, 1))
    return ei, ej, A[ei, ej].astype(float)


def mean_degree(A):
    return float(A.astype(bool).sum(axis=1).mean())


def k_from_slider(log_k):
    return float(10.0 ** log_k)


def format_k(k):
    return f'{k:.3g}'


def format_ring_k(k):
    return f'{int(k)}  (k/N={k / N:.3f})'


def set_ring_k_slider_text(k=None):
    slider_ring_k.valtext.set_text(format_ring_k(ring_k if k is None else k))


def kuramoto_rhs(t, theta, K, omega, edges, degree, normalize):
    ei, ej, w = edges
    coupling = np.zeros(N)
    np.add.at(coupling, ei, w * np.sin(theta[ej] - theta[ei]))
    np.add.at(coupling, ej, w * np.sin(theta[ei] - theta[ej]))
    if normalize:
        return omega + K * coupling / degree
    return omega + K * coupling


def normalize_by_degree():
    return coupling_state['normalize']


def top_weight_edges(A, max_edges):
    ei, ej, w = edges_from_A(A)
    if len(ei) <= max_edges:
        return ei, ej, w
    idx = np.argpartition(w, -max_edges)[-max_edges:]
    return ei[idx], ej[idx], w[idx]


def precompute_er(A):
    """effective-resistance sampling probs — depends on A only, not q."""
    graph_laplacian = np.diag(A.sum(1)) - A
    graph_laplacian_pinv = np.linalg.pinv(graph_laplacian)
    diag = np.diag(graph_laplacian_pinv)
    effective_resistances = diag[:, None] + diag[None, :] - 2 * graph_laplacian_pinv
    edge_i, edge_j = np.where(np.triu(A, 1))
    we = A[edge_i, edge_j].astype(float)
    Re = we * effective_resistances[edge_i, edge_j]
    pe = Re / Re.sum()
    return edge_i, edge_j, we, pe


def a_cache_key(A):
    ei, ej, w = edges_from_A(A)
    h = hashlib.sha256()
    h.update(np.array([A.shape[0]], dtype=np.int64).tobytes())
    h.update(ei.astype(np.int32).tobytes())
    h.update(ej.astype(np.int32).tobytes())
    h.update(w.astype(np.float64).tobytes())
    return h.hexdigest()[:16]


def er_cache_path(A):
    return DATA_DIR / f'er_probs_N{A.shape[0]}_{a_cache_key(A)}.npz'


def load_or_precompute_er(A):
    path = er_cache_path(A)
    key = a_cache_key(A)
    if path.exists():
        cached = np.load(path)
        if cached['cache_key'] == key:
            print(f'loaded ER cache: {path.name}')
            return cached['edge_i'], cached['edge_j'], cached['we'], cached['pe']
        print(f'stale ER cache ({path.name}), recomputing...')
    else:
        print(f'computing ER probs for this graph ({path.name})...')
    edge_i, edge_j, we, pe = precompute_er(A)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, edge_i=edge_i, edge_j=edge_j, we=we, pe=pe, cache_key=key,
    )
    print(f'saved ER cache: {path.name}')
    return edge_i, edge_j, we, pe


def load_problem(topology_name):
    global A, degree_full, omega_unit, theta_0, sparsify_seed
    global er_edge_i, er_edge_j, er_we, er_pe
    run_seed = problem_run_seed()
    A, omega, theta_0 = setup_from_run_seed(run_seed, topology_name, ring_k)
    omega_unit = (omega - 5.0) / 0.5
    sparsify_seed = run_seed + 1
    degree_full = weighted_degree(A)
    er_edge_i, er_edge_j, er_we, er_pe = load_or_precompute_er(A)
    print(f'problem run seed: {run_seed}  sparsify seed: {sparsify_seed}')


def update_ring_graph():
    """ring k changed — rebuild A and ER probs only; keep ω and θ₀."""
    global A, degree_full, er_edge_i, er_edge_j, er_we, er_pe
    weight_rng = (
        np.random.default_rng(problem_run_seed() + 2)
        if topology_name == 'ring het. weights' else None
    )
    A = build_ring_A(topology_name, ring_k, weight_rng)
    degree_full = weighted_degree(A)
    er_edge_i, er_edge_j, er_we, er_pe = load_or_precompute_er(A)


load_problem(topology_name)


def sparsify_er(edge_i, edge_j, we, pe, q, seed=0):
    rng = np.random.default_rng(seed)
    s = max(1, int(q * len(edge_i)))
    sparsified_matrix = np.zeros((N, N))
    for k in rng.choice(len(edge_i), s, replace=True, p=pe):
        w = we[k] / (s * pe[k])
        sparsified_matrix[edge_i[k], edge_j[k]] += w
        sparsified_matrix[edge_j[k], edge_i[k]] += w
    return sparsified_matrix


def aligned_edge_lists(A_full, A_sparse, max_edges):
    ei_s, ej_s, _ = top_weight_edges(A_sparse, max_edges)
    ei_f, ej_f, _ = top_weight_edges(A_full, max_edges)
    sparse_pairs = set(zip(ei_s.tolist(), ej_s.tolist()))
    full_only = [(i, j) for i, j in zip(ei_f, ej_f) if (i, j) not in sparse_pairs]
    n_extra = max(0, max_edges - len(ei_s))
    full_only = full_only[:n_extra]
    ei_fo = np.fromiter((p[0] for p in full_only), dtype=int, count=len(full_only))
    ej_fo = np.fromiter((p[1] for p in full_only), dtype=int, count=len(full_only))
    return (ei_s, ej_s), (np.concatenate([ei_s, ei_fo]), np.concatenate([ej_s, ej_fo]))


def edge_segments(ei, ej):
    x1, y1 = nodes_xy(ei)
    x2, y2 = nodes_xy(ej)
    dx, dy = x2 - x1, y2 - y1
    d = np.maximum(np.hypot(dx, dy), 1e-12)
    p1 = np.column_stack([x1 + R * dx / d, y1 + R * dy / d])
    p2 = np.column_stack([x2 - R * dx / d, y2 - R * dy / d])
    return np.stack([p1, p2], axis=1)


def ball_offsets(sol, k, omega_lock=0.0, t=0.0):
    theta = sol[:, k] - omega_lock * t
    cx, cy = nodes_xy(np.arange(N))
    return np.stack((cx + R * np.cos(theta), cy + R * np.sin(theta)), axis=-1)


def unit_circle_offsets(sol, k, omega_lock=0.0, t=0.0):
    theta = sol[:, k] - omega_lock * t
    return np.stack((np.cos(theta), np.sin(theta)), axis=-1)


def edge_sin(sol, ei, ej):
    return np.sin(sol[ej, :] - sol[ei, :])


def steady_group_omega(psi, t, fraction=steady_fraction):
    """mean rotation rate of order-parameter phase over the last fraction of the run."""
    late = t >= t[-1] * (1 - fraction)
    if np.sum(late) < 2:
        return 0.0
    psi_u = np.unwrap(psi[late])
    return float(np.polyfit(t[late], psi_u, 1)[0])


def oscillator_freq_series(sol, t):
    """instantaneous rotation rate dθ/dt for each oscillator."""
    return np.gradient(np.unwrap(sol, axis=1), t, axis=1)


def pack_omega_series(sol, t):
    """bulk rotation rate Ω(t) — median dθ/dt (stable when r is low)."""
    return np.median(oscillator_freq_series(sol, t), axis=0)


def lock_drift_mask(sol, t, cutoff=freq_lock_cutoff):
    """locked if |dθ_i/dt − Ω_pack(t)| ≤ cutoff."""
    omega_inst = oscillator_freq_series(sol, t)
    omega_pack = pack_omega_series(sol, t)
    return np.abs(omega_inst - omega_pack[np.newaxis, :]) <= cutoff


def lock_heatmap_axes(omega_vals):
    """y extent for lock heatmap; use index when all ω are equal (ring mode)."""
    span = float(omega_vals.max() - omega_vals.min())
    if span < 1e-9:
        return 0.0, float(len(omega_vals) - 1), 'oscillator index (sorted by ω)'
    pad = max(0.05 * span, 0.02)
    return float(omega_vals.min()) - pad, float(omega_vals.max()) + pad, 'natural frequency ω'


def lock_drift_series(sol, t, cutoff=freq_lock_cutoff):
    """aggregate f_lock/f_drift and r_lock/r_drift from lock_drift_mask."""
    locked = lock_drift_mask(sol, t, cutoff)
    n_t = sol.shape[1]
    f_lock = locked.mean(axis=0)
    f_drift = 1.0 - f_lock
    r_lock = np.zeros(n_t)
    r_drift = np.zeros(n_t)
    for k in range(n_t):
        lk, dk = locked[:, k], ~locked[:, k]
        if lk.any():
            r_lock[k] = np.abs(np.sum(np.exp(1j * sol[lk, k])) / N)
        if dk.any():
            r_drift[k] = np.abs(np.sum(np.exp(1j * sol[dk, k])) / N)
    return f_lock, f_drift, r_lock, r_drift, locked


def lock_status_codes(locked_f, locked_s):
    """0=both drift, 1=both lock, 2=full lock/sparse drift, 3=full drift/sparse lock."""
    return np.select(
        [locked_f & locked_s, ~locked_f & ~locked_s, locked_f & ~locked_s, ~locked_f & locked_s],
        [1, 0, 2, 3],
        default=0,
    ).astype(np.float32)


def simulation_settings(t_end, n_edges):
    """interactive prioritizes speed; large graphs get even fewer time points."""
    n_eval = min(n_t_eval_max, max(n_t_eval, int(n_t_eval * t_end / t_end_default)))
    if n_edges > 50_000:
        return min(n_eval, 100), dict(method='RK45', rtol=1e-5, atol=1e-7)
    if n_edges > 10_000:
        return min(n_eval, 120), dict(method='RK45', rtol=1e-6, atol=1e-8)
    return n_eval, solver_kw


def effective_K(fallback):
    if 'slider_k' in globals():
        return k_from_slider(slider_k.val)
    return fallback


def effective_omega_std(fallback=0.0):
    if use_ring_layout():
        return 0.0
    if 'slider_os' in globals():
        return slider_os.val
    return fallback


def run_simulation(K, omega_mean, omega_std, q, t_end):
    K = effective_K(K)
    omega_std = effective_omega_std(omega_std) if use_ring_layout() else omega_std
    omega = omega_mean + omega_std * omega_unit
    t_span = (0.0, t_end)
    n_edges = len(er_edge_i)
    n_eval, integrate_kw = simulation_settings(t_end, n_edges)
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    A_sparse = sparsify_er(er_edge_i, er_edge_j, er_we, er_pe, q, seed=sparsify_seed)
    edges_full = edges_from_A(A)
    edges_er = edges_from_A(A_sparse)
    degree_er = weighted_degree(A_sparse)
    norm = normalize_by_degree()
    sol_full = solve_ivp(
        kuramoto_rhs, t_span, theta_0, args=(K, omega, edges_full, degree_full, norm),
        t_eval=t_eval, **integrate_kw,
    ).y
    sol_er = solve_ivp(
        kuramoto_rhs, t_span, theta_0, args=(K, omega, edges_er, degree_er, norm),
        t_eval=t_eval, **integrate_kw,
    ).y
    return omega, A_sparse, t_eval, sol_full, sol_er


def n_anim_frames(t_end):
    return max(anim_frames_base, int(round(t_end / sim_dt_per_frame)))


def subsample_trajectory(t_eval, sol_full, sol_er, t_end, omega_vals):
    n_frames = n_anim_frames(t_end)
    frame_idx = np.linspace(0, sol_full.shape[1] - 1, n_frames, dtype=int)
    t_anim = t_eval[frame_idx]
    sol_f = sol_full[:, frame_idx]
    sol_s = sol_er[:, frame_idx]
    z = np.mean(np.exp(1j * sol_f), axis=0)
    z_sparse = np.mean(np.exp(1j * sol_s), axis=0)
    diff = np.degrees(np.angle(np.exp(1j * (sol_f - sol_s))))
    r_full = np.abs(z)
    r_sparse = np.abs(z_sparse)
    err_r = np.abs(r_full - r_sparse)
    err_angle = np.degrees(np.abs(np.angle(z * np.conj(z_sparse))))
    psi_full = np.angle(z)
    psi_sparse = np.angle(z_sparse)
    omega_lock_full = steady_group_omega(psi_full, t_anim)
    omega_lock_sparse = steady_group_omega(psi_sparse, t_anim)
    # lock/drift uses dθ/dt → needs unique times; compute on full ODE grid first
    f_lock_f, f_drift_f, r_lock_f, r_drift_f, locked_f = lock_drift_series(sol_full, t_eval)
    f_lock_s, f_drift_s, r_lock_s, r_drift_s, locked_s = lock_drift_series(sol_er, t_eval)
    f_lock_f, f_drift_f = f_lock_f[frame_idx], f_drift_f[frame_idx]
    f_lock_s, f_drift_s = f_lock_s[frame_idx], f_drift_s[frame_idx]
    r_lock_f, r_drift_f = r_lock_f[frame_idx], r_drift_f[frame_idx]
    r_lock_s, r_drift_s = r_lock_s[frame_idx], r_drift_s[frame_idx]
    locked_f, locked_s = locked_f[:, frame_idx], locked_s[:, frame_idx]
    lock_codes = lock_status_codes(locked_f, locked_s)
    omega_sort = np.argsort(omega_vals)
    lock_y_lo, lock_y_hi, lock_y_label = lock_heatmap_axes(omega_vals)
    return dict(
        t_anim=t_anim, sol_f=sol_f, sol_s=sol_s, z=z, z_sparse=z_sparse,
        diff=diff, err_r=err_r, err_angle=err_angle,
        psi_full=psi_full, psi_sparse=psi_sparse,
        omega_lock_full=omega_lock_full, omega_lock_sparse=omega_lock_sparse,
        f_lock_f=f_lock_f, f_drift_f=f_drift_f, r_lock_f=r_lock_f, r_drift_f=r_drift_f,
        f_lock_s=f_lock_s, f_drift_s=f_drift_s, r_lock_s=r_lock_s, r_drift_s=r_drift_s,
        locked_f=locked_f, locked_s=locked_s, lock_codes=lock_codes, omega_sort=omega_sort,
        lock_y_lo=lock_y_lo, lock_y_hi=lock_y_hi, lock_y_label=lock_y_label,
    )


# initial run
K0 = 0.01 * N
omega_mean0 = 5.0
omega_std0 = 0.5
q0 = 0.25
t_end0 = t_end_default
omega, A_sparse, t_eval, sol_full, sol_er = run_simulation(
    K0, omega_mean0, omega_std0, q0, t_end0,
)
data = subsample_trajectory(t_eval, sol_full, sol_er, t_end0, omega)
ball_colors = plt.cm.viridis(plt.Normalize(omega.min(), omega.max())(omega))

(ei_s, ej_s), (ei_f, ej_f) = aligned_edge_lists(A, A_sparse, anim_max_edges)
ball_f = ball_offsets(data['sol_f'], 0)
ball_s = ball_offsets(data['sol_s'], 0)

fig = plt.figure(figsize=(11, 6.5))
fig.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.42, hspace=0.48, wspace=0.28)
gs = fig.add_gridspec(2, 3, height_ratios=[1.7, 0.75], width_ratios=[1, 1, 1.4])
ax_f = fig.add_subplot(gs[0, 0])
ax_s = fig.add_subplot(gs[0, 1])
ax_d = fig.add_subplot(gs[0, 2])
ax_zf = fig.add_subplot(gs[1, 0])
ax_zs = fig.add_subplot(gs[1, 1])
ax_rd = fig.add_subplot(gs[1, 2])

panels = []
panel_circles = []
edge_count_texts = []
_xlim, _ylim = panel_limits()
for ax, A_mat, ei, ej, _, sol_sub, title in [
    (ax_f, A, ei_f, ej_f, ball_f, data['sol_f'], f'full — {topology_name}'),
    (ax_s, A_sparse, ei_s, ej_s, ball_s, data['sol_s'], 'sparse (ER)'),
]:
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    ax.set_xlim(*_xlim)
    ax.set_ylim(*_ylim)
    n_total = len(edges_from_A(A_mat)[0])
    txt = ax.text(
        0.5, -0.04, f'{n_total:,} edges ({len(ei):,} shown)',
        transform=ax.transAxes, ha='center', va='top', fontsize=9,
    )
    edge_count_texts.append(txt)
    circles = []
    for i in range(N):
        x, y = nodes_xy(i)
        circ = plt.Circle((x, y), R, fill=False, ec='k', lw=0.8)
        ax.add_patch(circ)
        circles.append(circ)
    panel_circles.append(circles)
    lc = LineCollection(
        edge_segments(ei, ej), linewidths=0.1, cmap='coolwarm', clim=(-1, 1), zorder=1,
    )
    ax.add_collection(lc)
    sc = ax.scatter(
        ball_f[:, 0], ball_f[:, 1], s=18, c=ball_colors, zorder=3, edgecolors='none',
    )
    panels.append((lc, sc, edge_sin(sol_sub, ei, ej), ei, ej))

op_panels = []
for ax, z_t, sol_sub, title in [
    (ax_zf, data['z'], data['sol_f'], 'full order param'),
    (ax_zs, data['z_sparse'], data['sol_s'], 'sparse order param'),
]:
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec='gray', lw=1, zorder=1))
    phase_pos = unit_circle_offsets(sol_sub, 0)
    sc = ax.scatter(
        phase_pos[:, 0], phase_pos[:, 1],
        s=10, c=ball_colors, alpha=0.75, zorder=2, edgecolors='none',
    )
    ln, = ax.plot([0, z_t[0].real], [0, z_t[0].imag], 'k-', lw=2, zorder=3)
    dot, = ax.plot(z_t[0].real, z_t[0].imag, 'o', color='crimson', ms=6, zorder=4)
    op_panels.append((ln, dot, sc, z_t))

ERR_R_YMAX_FIXED = 1.05
ERR_PSI_YMAX_FIXED = 180.0

ax_rd.set_title('order param error (full − sparse)')
ax_rd.set_xlim(0, data['t_anim'][-1])
ax_rd.set_xlabel('time (s)')
ax_rd.set_ylabel('|Δr|', color='crimson')
ax_rd.tick_params(axis='y', labelcolor='crimson')
bg_err_r, = ax_rd.plot(data['t_anim'], data['err_r'], color='crimson', alpha=0.3)
ax_rd_twin = ax_rd.twinx()
ax_rd_twin.set_ylabel('|Δψ| (°)', color='royalblue', labelpad=16)
ax_rd_twin.tick_params(axis='y', labelcolor='royalblue', pad=2)
bg_err_angle, = ax_rd_twin.plot(data['t_anim'], data['err_angle'], color='royalblue', alpha=0.3)
op_err_r, = ax_rd.plot([], [], color='crimson', lw=2)
op_err_angle, = ax_rd_twin.plot([], [], color='royalblue', lw=2)
ax_rd.legend([bg_err_r, bg_err_angle], ['|Δr|', '|Δψ|'], loc='upper left', fontsize=8)

ax_lock = fig.add_subplot(gs[1, 2])
ax_lock.set_title('f_drift & r (solid=full, dashed=sparse)', fontsize=9, pad=6)
ax_lock.set_xlim(0, data['t_anim'][-1])
ax_lock.set_xlabel('time (s)')
ax_lock.set_ylabel('f_drift', color='darkorange')
ax_lock.set_ylim(0, 1.05)
ax_lock.tick_params(axis='y', labelcolor='darkorange')
lock_style = {'full': '-', 'sparse': '--'}
bg_f_drift_f, = ax_lock.plot(
    data['t_anim'], data['f_drift_f'], color='darkorange', ls=lock_style['full'], alpha=0.3,
)
bg_f_drift_s, = ax_lock.plot(
    data['t_anim'], data['f_drift_s'], color='darkorange', ls=lock_style['sparse'], alpha=0.3,
)
ax_lock_twin = ax_lock.twinx()
ax_lock_twin.set_ylabel('r contribution', color='crimson', labelpad=16)
ax_lock_twin.set_ylim(0, 1.05)
ax_lock_twin.tick_params(axis='y', labelcolor='crimson', pad=2)
bg_r_lock_f, = ax_lock_twin.plot(
    data['t_anim'], data['r_lock_f'], color='crimson', ls=lock_style['full'], alpha=0.3,
)
bg_r_drift_f, = ax_lock_twin.plot(
    data['t_anim'], data['r_drift_f'], color='royalblue', ls=lock_style['full'], alpha=0.3,
)
bg_r_lock_s, = ax_lock_twin.plot(
    data['t_anim'], data['r_lock_s'], color='crimson', ls=lock_style['sparse'], alpha=0.3,
)
bg_r_drift_s, = ax_lock_twin.plot(
    data['t_anim'], data['r_drift_s'], color='royalblue', ls=lock_style['sparse'], alpha=0.3,
)
op_f_drift_f, = ax_lock.plot([], [], color='darkorange', ls=lock_style['full'], lw=2)
op_f_drift_s, = ax_lock.plot([], [], color='darkorange', ls=lock_style['sparse'], lw=2)
op_r_lock_f, = ax_lock_twin.plot([], [], color='crimson', ls=lock_style['full'], lw=2)
op_r_drift_f, = ax_lock_twin.plot([], [], color='royalblue', ls=lock_style['full'], lw=2)
op_r_lock_s, = ax_lock_twin.plot([], [], color='crimson', ls=lock_style['sparse'], lw=2)
op_r_drift_s, = ax_lock_twin.plot([], [], color='royalblue', ls=lock_style['sparse'], lw=2)
ax_lock.legend(
    [bg_f_drift_f, bg_r_lock_f, bg_r_drift_f],
    ['f_drift', 'r_lock', 'r_drift'],
    loc='upper left', fontsize=8,
)
vline_lock = ax_lock.axvline(data['t_anim'][0], color='crimson', lw=2)
ax_lock.set_visible(False)
ax_lock_twin.set_visible(False)

ax_d.set_title('phase diff per oscillator (full − sparse)')
phase_diff_lc = LineCollection(
    np.dstack([np.broadcast_to(data['t_anim'], (N, len(data['t_anim']))), data['diff']]),
    colors=ball_colors, alpha=0.5, linewidths=0.4,
)
ax_d.add_collection(phase_diff_lc)
lock_cmap = ListedColormap(['#d0d0d0', '#2ca02c', '#ff7f0e', '#1f77b4'])
lock_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], lock_cmap.N)
lock_im = ax_d.imshow(
    data['lock_codes'][data['omega_sort']],
    aspect='auto', origin='lower', interpolation='nearest',
    extent=[data['t_anim'][0], data['t_anim'][-1], data['lock_y_lo'], data['lock_y_hi']],
    cmap=lock_cmap, norm=lock_norm, visible=False,
)
lock_legend = ax_d.legend(
    handles=[
        plt.Line2D([0], [0], color='#d0d0d0', lw=6, label='both drift'),
        plt.Line2D([0], [0], color='#2ca02c', lw=6, label='both lock'),
        plt.Line2D([0], [0], color='#ff7f0e', lw=6, label='full lock, sparse drift'),
        plt.Line2D([0], [0], color='#1f77b4', lw=6, label='full drift, sparse lock'),
    ],
    loc='upper right', fontsize=7, framealpha=0.85,
)
lock_legend.set_visible(False)
ax_d.set_xlim(0, data['t_anim'][-1])
ax_d.set_xlabel('')
ax_d.set_ylabel('Δθ (°)')
ax_d.set_ylim(-180, 180)
vline = ax_d.axvline(data['t_anim'][0], color='crimson', lw=2)
twist_prof_f, = ax_d.plot([], [], color='C0', lw=1.2, visible=False)
twist_prof_s, = ax_d.plot([], [], color='C1', lw=1.2, ls='--', visible=False)
twist_hlines = []


def clear_twist_hlines():
    global twist_hlines
    for ln in twist_hlines:
        ln.remove()
    twist_hlines = []


def set_twist_hlines(ylo, yhi):
    clear_twist_hlines()
    for m in range(int(np.floor(ylo / 360)), int(np.ceil(yhi / 360)) + 1):
        y = m * 360
        if ylo <= y <= yhi:
            twist_hlines.append(
                ax_d.axhline(y, color='gray', lw=0.6, alpha=0.45, zorder=0),
            )
err_txt = ax_d.text(0.02, 0.95, '', transform=ax_d.transAxes, va='top', fontsize=9)
status_txt = fig.text(0.5, 0.965, '', ha='center', va='top', fontsize=9)
ring_viz = {'mode': 'balls'}

ax_k = fig.add_axes([0.08, 0.26, 0.35, 0.018])
ax_om = fig.add_axes([0.08, 0.225, 0.35, 0.018])
ax_os = fig.add_axes([0.08, 0.195, 0.35, 0.018])
ax_q = fig.add_axes([0.55, 0.26, 0.35, 0.018])
ax_tend = fig.add_axes([0.55, 0.225, 0.35, 0.018])
ax_time = fig.add_axes([0.55, 0.195, 0.35, 0.018])
ax_n = fig.add_axes([0.08, 0.165, 0.35, 0.018])
ax_topo = fig.add_axes([0.06, 0.02, 0.40, 0.11])
ax_topo.set_title('full graph A', fontsize=8, loc='left', pad=1)
ax_ring_k = fig.add_axes([0.08, 0.135, 0.35, 0.018])
ax_coupling = fig.add_axes([0.50, 0.02, 0.17, 0.045])
ax_coupling.set_title('coupling', fontsize=8, loc='left', pad=1)
ax_ring_viz = fig.add_axes([0.50, 0.075, 0.17, 0.055])
ax_ring_viz.set_title('ring view', fontsize=8, loc='left', pad=1)
ax_play = fig.add_axes([0.75, 0.02, 0.09, 0.03])
ax_randomize = fig.add_axes([0.85, 0.02, 0.12, 0.03])

slider_k = Slider(
    ax_k, 'K', LOG_K_MIN, LOG_K_MAX, valinit=np.log10(K0), valstep=0.025,
)
slider_k.valtext.set_text(format_k(K0))
slider_om = Slider(ax_om, 'ω mean', 0.0, 7.0, valinit=omega_mean0, valstep=0.05)
slider_os = Slider(ax_os, 'ω std', 0.05, 2.0, valinit=omega_std0, valstep=0.05)

slider_q = Slider(ax_q, 'q (ER)', 0.05, 1.0, valinit=q0, valstep=0.01)
slider_tend = Slider(ax_tend, 't end (s)', 1.0, t_end_max, valinit=t_end0, valstep=0.5)
slider_time = Slider(ax_time, 'time (s)', 0.0, t_end0, valinit=0.0)
slider_n = Slider(ax_n, 'N (oscillators)', N_MIN, N_MAX, valinit=N, valstep=1)
slider_ring_k = Slider(
    ax_ring_k, 'ring k (k-NN)', RING_K_MIN, ring_k_max(),
    valinit=min(RING_K_DEFAULT, ring_k_max()), valstep=1,
)
set_ring_k_slider_text(min(RING_K_DEFAULT, ring_k_max()))
radio_ring_viz = RadioButtons(ax_ring_viz, ['balls (ω)', 'twists (θ)'], active=0)
for label in radio_ring_viz.labels:
    label.set_fontsize(8)
radio_coupling = RadioButtons(ax_coupling, ['unnormalized', '/degree'], active=1)
for label in radio_coupling.labels:
    label.set_fontsize(8)
coupling_syncing = False
btn_play = Button(ax_play, 'Play')
btn_randomize = Button(ax_randomize, 'Randomize')
ax_panel = fig.add_axes([0.75, 0.058, 0.22, 0.058])
ax_checks = fig.add_axes([0.75, 0.122, 0.22, 0.06])
check_opts = CheckButtons(ax_checks, ['rotating frame', 'autoscale error y'], [False, False])
radio_panel = RadioButtons(ax_panel, ['order error', 'lock / drift'], active=0)
for label in radio_panel.labels:
    label.set_fontsize(8)
bottom_panel = {'mode': 'error'}
radio_topo = RadioButtons(
    ax_topo, TOPOLOGY_NAMES,
    active=TOPOLOGY_NAMES.index(DEFAULT_TOPOLOGY),
)
for label in radio_topo.labels:
    label.set_fontsize(8)

playing = {'on': False}
rotating = {'on': False}
err_autoscale = {'on': False}
sim_state = {'busy': False, 'pending': False}
ring_k_dirty = False
n_dirty = False
panels_need_rebuild = False
play_timer = fig.canvas.new_timer(interval=play_interval_ms)
debounce_timer = fig.canvas.new_timer(interval=debounce_ms)
physics_sliders = (slider_om, slider_os, slider_q, slider_tend)


def set_play_timer_interval():
    play_timer.interval = play_interval_ms


def _radio_markers(radio):
    # mpl <3.10: one circle patch per label; mpl 3.10+: single PathCollection
    if hasattr(radio, 'circles'):
        return radio.circles
    return [radio._buttons]


def prune_radio_circles(radio):
    """matplotlib can leave stray circles after set_active — keep one per label."""
    if not hasattr(radio, 'circles'):
        return
    n = len(radio.labels)
    while len(radio.circles) > n:
        radio.circles.pop(0).remove()


def set_radio_artists_visible(radio, visible):
    for artist in (*radio.labels, *_radio_markers(radio)):
        artist.set_visible(visible)


def set_radio_active(radio, index):
    if radio.active == index:
        prune_radio_circles(radio)
        return
    radio.set_active(index)
    prune_radio_circles(radio)


def frame_from_time(t):
    return int(np.clip(np.searchsorted(data['t_anim'], t), 0, len(data['t_anim']) - 1))


def update_panel_layout():
    xlim, ylim = panel_limits()
    for ax in (ax_f, ax_s):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    for circles in panel_circles:
        for i, circ in enumerate(circles):
            x, y = nodes_xy(i)
            circ.center = (x, y)


def rebuild_network_panels(ei_f, ej_f, ei_s, ej_s, sol_f, sol_s, omega_vals, A_full, A_sparse):
    """recreate node/edge artists when N changes."""
    global panels, panel_circles, op_panels, ball_colors, ring_idx

    ball_colors = plt.cm.viridis(
        plt.Normalize(omega_vals.min(), omega_vals.max())(omega_vals),
    )
    ring_idx = np.arange(N)
    ball_f = ball_offsets(sol_f, 0)
    ball_s = ball_offsets(sol_s, 0)
    xlim, ylim = panel_limits()

    for ax, circles, panel in zip((ax_f, ax_s), panel_circles, panels):
        lc, sc, _, _, _ = panel
        lc.remove()
        sc.remove()
        for circ in circles:
            circ.remove()

    panels.clear()
    panel_circles.clear()
    configs = [
        (ax_f, A_full, ei_f, ej_f, ball_f, sol_f, f'full — {topology_name}'),
        (ax_s, A_sparse, ei_s, ej_s, ball_s, sol_s, 'sparse (ER)'),
    ]
    for ax, A_mat, ei, ej, ball, sol_sub, title in configs:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        circles = []
        for i in range(N):
            x, y = nodes_xy(i)
            circ = plt.Circle((x, y), R, fill=False, ec='k', lw=0.8)
            ax.add_patch(circ)
            circles.append(circ)
        panel_circles.append(circles)
        lc = LineCollection(
            edge_segments(ei, ej), linewidths=0.1, cmap='coolwarm', clim=(-1, 1), zorder=1,
        )
        ax.add_collection(lc)
        sc = ax.scatter(
            ball[:, 0], ball[:, 1], s=18, c=ball_colors, zorder=3, edgecolors='none',
        )
        panels.append((lc, sc, edge_sin(sol_sub, ei, ej), ei, ej))
        edge_count_texts[len(panels) - 1].set_text(
            f'{len(edges_from_A(A_mat)[0]):,} edges ({len(ei):,} shown)'
        )

    new_op = []
    for ax, (ln, dot, sc, _), (z_t, sol_sub) in zip(
        (ax_zf, ax_zs),
        op_panels,
        [(data['z'], sol_f), (data['z_sparse'], sol_s)],
    ):
        sc.remove()
        phase_pos = unit_circle_offsets(sol_sub, 0)
        sc = ax.scatter(
            phase_pos[:, 0], phase_pos[:, 1],
            s=10, c=ball_colors, alpha=0.75, zorder=2, edgecolors='none',
        )
        new_op.append((ln, dot, sc, z_t))
    op_panels[:] = new_op

    phase_diff_lc.set_segments(
        np.dstack([np.broadcast_to(data['t_anim'], (N, len(data['t_anim']))), data['diff']]),
    )
    phase_diff_lc.set_colors(ball_colors)
    lock_im.set_data(data['lock_codes'][data['omega_sort']])
    lock_im.set_extent([
        data['t_anim'][0], data['t_anim'][-1],
        data['lock_y_lo'], data['lock_y_hi'],
    ])
    ax_d.set_ylabel(data['lock_y_label'])
    set_bottom_panel(bottom_panel['mode'])


def update_ring_k_slider_limits():
    global ring_k
    kmax = ring_k_max()
    slider_ring_k.valmax = kmax
    slider_ring_k.ax.set_xlim(RING_K_MIN, kmax)
    ring_k = max(RING_K_MIN, min(ring_k, kmax))
    slider_ring_k.eventson = False
    slider_ring_k.set_val(ring_k)
    slider_ring_k.eventson = True
    set_ring_k_slider_text(ring_k)


def apply_n_change():
    """rebuild graph and panels for a new N; caller runs apply_simulation next."""
    global panels_need_rebuild
    target = snap_n_for_topology(int(slider_n.val), topology_name)
    if target == N:
        return False
    set_n(target)
    slider_n.eventson = False
    slider_n.set_val(N)
    slider_n.valtext.set_text(str(N))
    slider_n.eventson = True
    update_ring_k_slider_limits()
    panels_need_rebuild = True
    status_txt.set_text(f'N={N} — loading graph...')
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    load_problem(topology_name)
    return True


def on_n_slider(val):
    snapped = snap_n_for_topology(int(val), topology_name)
    if snapped != int(val):
        slider_n.eventson = False
        slider_n.set_val(snapped)
        slider_n.eventson = True
    slider_n.valtext.set_text(str(snapped))
    global n_dirty
    n_dirty = True
    debounce_timer.stop()
    status_txt.set_text(f'N={snapped} → rebuild in ~0.7s')
    fig.canvas.draw_idle()
    debounce_timer.start()


def show_twist_viz():
    return use_ring_layout() and ring_viz['mode'] == 'twists'


def sync_ring_paper_defaults():
    """ring modes: unnormalized coupling, ω std=0; K defaults to 1 on enter only."""
    global coupling_syncing
    if use_ring_layout():
        if saved_before_ring['log_k'] is None:
            saved_before_ring['log_k'] = slider_k.val
            slider_k.eventson = False
            slider_k.set_val(np.log10(RING_PAPER_K))
            slider_k.valtext.set_text(format_k(RING_PAPER_K))
            slider_k.eventson = True
        ax_k.set_alpha(1.0)
        slider_os.eventson = False
        slider_os.set_val(0.0)
        slider_os.eventson = True
        ax_os.set_alpha(0.35)
        slider_os.valtext.set_text('0 (paper)')
        coupling_state['normalize'] = False
        if radio_coupling.active != 0:
            coupling_syncing = True
            set_radio_active(radio_coupling, 0)
            coupling_syncing = False
        ax_coupling.set_alpha(0.35)
    else:
        ax_k.set_alpha(1.0)
        ax_os.set_alpha(1.0)
        ax_coupling.set_alpha(1.0)
        if saved_before_ring['log_k'] is not None:
            slider_k.eventson = False
            slider_k.set_val(saved_before_ring['log_k'])
            slider_k.valtext.set_text(format_k(k_from_slider(saved_before_ring['log_k'])))
            slider_k.eventson = True
            saved_before_ring['log_k'] = None
        elif slider_os.val < 0.05:
            slider_os.eventson = False
            slider_os.set_val(0.5)
            slider_os.eventson = True
        coupling_state['normalize'] = True
        if radio_coupling.active != 1:
            coupling_syncing = True
            set_radio_active(radio_coupling, 1)
            coupling_syncing = False


def sync_omega_std_slider():
    if not use_ring_layout() and slider_os.val < 0.05:
        slider_os.eventson = False
        slider_os.set_val(0.5)
        slider_os.eventson = True


def set_ring_controls_visible():
    show = use_ring_layout()
    ax_ring_k.set_visible(show)
    ax_ring_viz.set_visible(show)
    set_radio_artists_visible(radio_ring_viz, show)


def sync_coupling_default():
    sync_ring_paper_defaults()


def coupling_label():
    return '/degree' if coupling_state['normalize'] else 'unnormalized'


def set_ring_viz_panel(show_twists):
    ring_viz['mode'] = 'twists' if show_twists else 'balls'
    show_tw = show_twist_viz()
    phase_diff_lc.set_visible(not show_tw and bottom_panel['mode'] == 'error')
    twist_prof_f.set_visible(show_tw)
    twist_prof_s.set_visible(show_tw)
    lock_im.set_visible(not show_tw and bottom_panel['mode'] != 'error')
    lock_legend.set_visible(not show_tw and bottom_panel['mode'] != 'error')
    if show_tw:
        phase_diff_lc.set_visible(False)
        vline.set_visible(False)
        ax_d.set_title('phase around ring (unwrapped θ vs index)')
        ax_d.set_xlabel('oscillator index')
        ax_d.set_ylabel('θ (°)')
        ax_d.set_xlim(0, N - 1)
    else:
        clear_twist_hlines()
        set_bottom_panel(bottom_panel['mode'])


def set_err_plot_ylim():
    if err_autoscale['on']:
        r_max = max(data['err_r'].max() * 1.05, 0.01)
        psi_max = max(data['err_angle'].max() * 1.05, 1.0)
        ax_rd.set_ylim(0, r_max)
        ax_rd.set_yticks(np.linspace(0, r_max, 5))
        ax_rd_twin.set_ylim(0, psi_max)
        ax_rd_twin.set_yticks(np.linspace(0, psi_max, 5))
    else:
        ax_rd.set_ylim(0, ERR_R_YMAX_FIXED)
        ax_rd.set_yticks([0, 0.5, 1.0])
        ax_rd_twin.set_ylim(0, ERR_PSI_YMAX_FIXED)
        ax_rd_twin.set_yticks([0, 90, 180])


set_err_plot_ylim()


def set_bottom_panel(mode):
    bottom_panel['mode'] = mode
    if show_twist_viz():
        return
    clear_twist_hlines()
    twist_prof_f.set_visible(False)
    twist_prof_s.set_visible(False)
    vline.set_visible(True)
    show_error = mode == 'error'
    ax_rd.set_visible(show_error)
    ax_rd_twin.set_visible(show_error)
    ax_lock.set_visible(not show_error)
    ax_lock_twin.set_visible(not show_error)
    phase_diff_lc.set_visible(show_error)
    lock_im.set_visible(not show_error)
    lock_legend.set_visible(not show_error)
    ax_d.set_xlim(0, data['t_anim'][-1])
    ax_d.set_xlabel('time (s)')
    if show_error:
        ax_d.set_title('phase diff per oscillator (full − sparse)')
        ax_d.set_ylabel('Δθ (°)')
        ax_d.set_ylim(-180, 180)
    else:
        ax_d.set_title(
            f'lock status mismatch (full vs sparse)  '
            f'locked if |dθ/dt − median(dθ/dt)| ≤ {freq_lock_cutoff}',
        )
        ax_d.set_ylabel(data['lock_y_label'])
        ax_d.set_ylim(data['lock_y_lo'], data['lock_y_hi'])


def draw_frame(k):
    t_k = data['t_anim'][k]
    om_f = data['omega_lock_full'] if rotating['on'] else 0.0
    om_s = data['omega_lock_sparse'] if rotating['on'] else 0.0
    show_tw = show_twist_viz()
    q_notes = []
    for i, (lc, sc, s_all, _, _) in enumerate(panels):
        sol_sub = data['sol_f'] if i == 0 else data['sol_s']
        om = om_f if i == 0 else om_s
        theta_k = sol_sub[:, k] - om * t_k
        balls_k = ball_offsets(sol_sub, k, om, t_k)
        s = s_all[:, k]
        lc.set_array(s)
        lc.set_linewidths(0.1 + 1.9 * np.abs(s))
        sc.set_offsets(balls_k)
        if show_tw:
            colors = phase_facecolors(theta_k)
            sc.set_facecolors(colors)
            for j, circ in enumerate(panel_circles[i]):
                circ.set_edgecolor(colors[j])
            q_notes.append(estimate_winding(theta_k))
        else:
            sc.set_facecolors(ball_colors)
            for circ in panel_circles[i]:
                circ.set_edgecolor('k')
    if show_tw:
        theta_f = data['sol_f'][:, k] - om_f * t_k
        theta_s = data['sol_s'][:, k] - om_s * t_k
        prof_f = phase_profile_deg(theta_f)
        prof_s = phase_profile_deg(theta_s)
        twist_prof_f.set_data(ring_idx, prof_f)
        twist_prof_s.set_data(ring_idx, prof_s)
        ylo = min(prof_f.min(), prof_s.min()) - 15
        yhi = max(prof_f.max(), prof_s.max()) + 15
        ax_d.set_ylim(ylo, yhi)
        set_twist_hlines(ylo, yhi)
        vline.set_visible(False)
        ax_d.set_xlim(0, N - 1)
        ax_d.set_xlabel('oscillator index')
        ax_d.set_title(
            f'phase around ring  q̂ full={q_notes[0]}  sparse={q_notes[1]}',
        )
        ax_f.set_title(
            f'full — ring (k-NN)  k={int(ring_k)} (k/N={ring_k / N:.3f})  q̂={q_notes[0]}',
        )
        ax_s.set_title(f'sparse (ER)  q̂={q_notes[1]}')
    elif use_ring_layout():
        ring_note = f'  k={int(ring_k)} (k/N={ring_k / N:.3f})'
        ax_f.set_title(f'full — {topology_name}{ring_note}')
        ax_s.set_title('sparse (ER)')
    for i, (ln, dot, sc, z_t) in enumerate(op_panels):
        sol_sub = data['sol_f'] if i == 0 else data['sol_s']
        om = om_f if i == 0 else om_s
        phase_pos = unit_circle_offsets(sol_sub, k, om, t_k)
        z_disp = z_t[k] * np.exp(-1j * om * t_k) if rotating['on'] else z_t[k]
        ln.set_data([0, z_disp.real], [0, z_disp.imag])
        dot.set_data([z_disp.real], [z_disp.imag])
        sc.set_offsets(phase_pos)
    op_err_r.set_data(data['t_anim'][: k + 1], data['err_r'][: k + 1])
    op_err_angle.set_data(data['t_anim'][: k + 1], data['err_angle'][: k + 1])
    sl = slice(None, k + 1)
    op_f_drift_f.set_data(data['t_anim'][sl], data['f_drift_f'][sl])
    op_f_drift_s.set_data(data['t_anim'][sl], data['f_drift_s'][sl])
    op_r_lock_f.set_data(data['t_anim'][sl], data['r_lock_f'][sl])
    op_r_drift_f.set_data(data['t_anim'][sl], data['r_drift_f'][sl])
    op_r_lock_s.set_data(data['t_anim'][sl], data['r_lock_s'][sl])
    op_r_drift_s.set_data(data['t_anim'][sl], data['r_drift_s'][sl])
    vline.set_xdata([data['t_anim'][k], data['t_anim'][k]])
    vline_lock.set_xdata([data['t_anim'][k], data['t_anim'][k]])
    if not show_tw:
        vline.set_visible(bottom_panel['mode'] == 'error')
    if show_tw:
        err_txt.set_text(
            f'q̂ full={q_notes[0]}  sparse={q_notes[1]}  |  '
            f'|Δr| = {data["err_r"][k]:.3f}  |  |Δψ| = {data["err_angle"][k]:.1f}°'
        )
    elif bottom_panel['mode'] == 'error':
        err_txt.set_text(
            f'mean |Δθ| = {np.mean(np.abs(data["diff"][:, k])):.1f}°  '
            f'|  |Δr| = {data["err_r"][k]:.3f}  '
            f'|  |Δψ| = {data["err_angle"][k]:.1f}°'
        )
    else:
        lf, ls = data['locked_f'][:, k], data['locked_s'][:, k]
        n_mis = int(np.sum(lf != ls))
        n_f_only = int(np.sum(lf & ~ls))
        n_s_only = int(np.sum(~lf & ls))
        err_txt.set_text(
            f'lock mismatch: {n_mis}/{N}  '
            f'(full-only {n_f_only}, sparse-only {n_s_only})  |  '
            f'f_drift full={data["f_drift_f"][k]:.2f} sparse={data["f_drift_s"][k]:.2f}'
        )
    fig.canvas.draw_idle()


def on_time_change(val):
    draw_frame(frame_from_time(val))


def current_params():
    return (
        k_from_slider(slider_k.val), slider_om.val, slider_os.val,
        slider_q.val, slider_tend.val,
    )


def on_k_slider(log_k):
    slider_k.valtext.set_text(format_k(k_from_slider(log_k)))
    schedule_simulation(log_k)


def schedule_simulation(_val=None):
    debounce_timer.stop()
    status_txt.set_text('paused dragging → re-run in ~0.7s (resets to t=0)')
    fig.canvas.draw_idle()
    debounce_timer.start()


def apply_simulation(_event=None, reset_time=True):
    if sim_state['busy']:
        sim_state['pending'] = True
        return
    sim_state['busy'] = True
    debounce_timer.stop()
    K, omega_mean, omega_std, q, t_end = current_params()
    status_txt.set_text('running simulation...')
    fig.canvas.draw_idle()
    try:
        omega_new, A_sparse_new, t_eval_new, sol_full, sol_er = run_simulation(
            K, omega_mean, omega_std, q, t_end,
        )
        global data, ball_colors, panels_need_rebuild
        data = subsample_trajectory(t_eval_new, sol_full, sol_er, t_end, omega_new)
        ball_colors = plt.cm.viridis(
            plt.Normalize(omega_new.min(), omega_new.max())(omega_new),
        )

        (ei_s, ej_s), (ei_f, ej_f) = aligned_edge_lists(A, A_sparse_new, anim_max_edges)
        rebuilt_panels = panels_need_rebuild
        if panels_need_rebuild:
            rebuild_network_panels(
                ei_f, ej_f, ei_s, ej_s,
                data['sol_f'], data['sol_s'], omega_new, A, A_sparse_new,
            )
            panels_need_rebuild = False
        else:
            new_panels = []
            for i, (lc, sc, _, _, _) in enumerate(panels):
                cfg = [
                    (A, ei_f, ej_f, data['sol_f']),
                    (A_sparse_new, ei_s, ej_s, data['sol_s']),
                ][i]
                A_mat, ei, ej, sol_sub = cfg
                lc.set_segments(edge_segments(ei, ej))
                s_all = edge_sin(sol_sub, ei, ej)
                lc.set_array(s_all[:, 0])
                sc.set_facecolors(ball_colors)
                new_panels.append((lc, sc, s_all, ei, ej))
                edge_count_texts[i].set_text(
                    f'{len(edges_from_A(A_mat)[0]):,} edges ({len(ei):,} shown)'
                )
            panels[:] = new_panels
            update_panel_layout()

            new_op = []
            for idx, (ln, dot, sc, _) in enumerate(op_panels):
                z_t = data['z'] if idx == 0 else data['z_sparse']
                sc.set_facecolors(ball_colors)
                new_op.append((ln, dot, sc, z_t))
            op_panels[:] = new_op

        bg_err_r.set_data(data['t_anim'], data['err_r'])
        bg_err_angle.set_data(data['t_anim'], data['err_angle'])
        ax_rd.set_xlim(0, data['t_anim'][-1])
        set_err_plot_ylim()
        lock_bg = [
            (bg_f_drift_f, 'f_drift_f'), (bg_f_drift_s, 'f_drift_s'),
            (bg_r_lock_f, 'r_lock_f'), (bg_r_drift_f, 'r_drift_f'),
            (bg_r_lock_s, 'r_lock_s'), (bg_r_drift_s, 'r_drift_s'),
        ]
        for ln, key in lock_bg:
            ln.set_data(data['t_anim'], data[key])
        ax_lock.set_xlim(0, data['t_anim'][-1])
        if not rebuilt_panels:
            phase_diff_lc.set_segments(
                np.dstack([np.broadcast_to(data['t_anim'], (N, len(data['t_anim']))), data['diff']])
            )
            phase_diff_lc.set_colors(ball_colors)
            lock_im.set_data(data['lock_codes'][data['omega_sort']])
            lock_im.set_extent([
                data['t_anim'][0], data['t_anim'][-1],
                data['lock_y_lo'], data['lock_y_hi'],
            ])
        ax_d.set_xlim(0, data['t_anim'][-1])
        set_bottom_panel(bottom_panel['mode'])
        slider_time.valmax = t_end
        slider_time.ax.set_xlim(0.0, t_end)
        omega_std_eff = effective_omega_std(omega_std)
        ring_note = f'  k={int(ring_k)} (k/N={ring_k / N:.3f})' if use_ring_layout() else ''
        status_txt.set_text(
            f'N={N}  {topology_name}{ring_note}  coupling={coupling_label()}  '
            f'avg deg={mean_degree(A):.1f}  |  '
            f'K={format_k(effective_K(K))}  ω={omega_mean:.2f}±{omega_std_eff:.2f}  '
            f'q={q:.2f}  sparsity={1 - q:.2f}  '
            f't∈[0,{t_end:.1f}]  ({len(edges_from_A(A_sparse_new)[0]):,} sparse edges)'
        )
        sync_ring_paper_defaults()
        set_ring_controls_visible()
        set_ring_viz_panel(radio_ring_viz.value_selected == 'twists (θ)')
        if reset_time:
            slider_time.eventson = False
            slider_time.set_val(0.0)
            slider_time.eventson = True
            draw_frame(0)
        else:
            t = min(slider_time.val, data['t_anim'][-1])
            slider_time.eventson = False
            slider_time.set_val(t)
            slider_time.eventson = True
            draw_frame(frame_from_time(t))
    finally:
        sim_state['busy'] = False
        if sim_state['pending']:
            sim_state['pending'] = False
            apply_simulation(reset_time=True)


def tick_play(_event=None):
    if not playing['on']:
        return
    t = slider_time.val + sim_dt_per_frame
    if t >= data['t_anim'][-1]:
        t = data['t_anim'][0]
    slider_time.set_val(t)


def on_check(_label):
    rotating['on'] = check_opts.get_status()[0]
    err_autoscale['on'] = check_opts.get_status()[1]
    set_err_plot_ylim()
    draw_frame(frame_from_time(slider_time.val))


def on_panel_toggle(_label):
    set_bottom_panel(
        'error' if radio_panel.value_selected == 'order error' else 'lock / drift',
    )
    draw_frame(frame_from_time(slider_time.val))


def toggle_play(_event):
    playing['on'] = not playing['on']
    if playing['on']:
        set_play_timer_interval()
    btn_play.label.set_text('Pause' if playing['on'] else 'Play')


topo_resample_timer = fig.canvas.new_timer(interval=1)


def deferred_topo_resample(_event=None):
    topo_resample_timer.stop()
    resample_topology(radio_topo.value_selected)


topo_resample_timer.add_callback(deferred_topo_resample)


def resample_topology(label):
    global topology_name, resample_count, panels_need_rebuild
    topology_name = label
    snapped = snap_n_for_topology(N, label)
    if snapped != N:
        set_n(snapped)
        slider_n.eventson = False
        slider_n.set_val(N)
        slider_n.valtext.set_text(str(N))
        slider_n.eventson = True
        update_ring_k_slider_limits()
        panels_need_rebuild = True
    resample_count += 1
    status_txt.set_text(f'loading {label}...')
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    sync_ring_paper_defaults()
    set_ring_controls_visible()
    load_problem(label)
    status_txt.set_text(f'{label} loaded — running simulation...')
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    apply_simulation(reset_time=True)


def on_ring_viz_toggle(_label):
    set_ring_viz_panel(radio_ring_viz.value_selected == 'twists (θ)')
    draw_frame(frame_from_time(slider_time.val))


def on_coupling_toggle(_label):
    if coupling_syncing:
        return
    if use_ring_layout():
        sync_ring_paper_defaults()
        return
    coupling_state['normalize'] = radio_coupling.value_selected == '/degree'
    schedule_simulation()


def on_ring_k_change(val):
    global ring_k, ring_k_dirty
    ring_k = int(val)
    set_ring_k_slider_text(ring_k)
    if use_ring_layout():
        ring_k_dirty = True
        schedule_simulation()


def randomize(_event=None):
    global RUN_SEED, resample_count
    RUN_SEED = fresh_seed()
    resample_count = 0
    print(f'run seed: {RUN_SEED}')
    status_txt.set_text('randomizing...')
    fig.canvas.draw_idle()
    load_problem(topology_name)
    apply_simulation(reset_time=True)


def on_topo_release(event):
    if event.inaxes is not ax_topo or event.button != 1:
        return
    topo_resample_timer.stop()
    topo_resample_timer.start()


def on_debounce(_event=None):
    global ring_k_dirty, n_dirty
    debounce_timer.stop()
    if n_dirty:
        n_dirty = False
        if apply_n_change():
            apply_simulation(reset_time=True)
        return
    if ring_k_dirty and use_ring_layout():
        ring_k_dirty = False
        status_txt.set_text('updating ring graph...')
        fig.canvas.draw_idle()
        update_ring_graph()
    apply_simulation(reset_time=True)


slider_time.on_changed(on_time_change)
slider_k.on_changed(on_k_slider)
slider_n.on_changed(on_n_slider)
slider_ring_k.on_changed(on_ring_k_change)
for s in physics_sliders:
    s.on_changed(schedule_simulation)
debounce_timer.add_callback(on_debounce)
fig.canvas.mpl_connect('button_release_event', on_topo_release)
btn_play.on_clicked(toggle_play)
btn_randomize.on_clicked(randomize)
check_opts.on_clicked(on_check)
radio_panel.on_clicked(on_panel_toggle)
radio_ring_viz.on_clicked(on_ring_viz_toggle)
radio_coupling.on_clicked(on_coupling_toggle)
play_timer.add_callback(tick_play)
play_timer.start()

sync_ring_paper_defaults()
prune_radio_circles(radio_coupling)
set_ring_controls_visible()
status_txt.set_text('physics sliders auto-update after you pause; time scrubs live')
draw_frame(0)
plt.show()
