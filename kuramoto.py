import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx


def fresh_seed():
    return int.from_bytes(os.urandom(8), 'big') % (2**63)

RUN_SEED = fresh_seed()
print(f'run seed: {RUN_SEED}')

N = 625
k_over_n = 0.01
K = k_over_n * N
q = 0.25


bool_plots = True

# time span
t_span = (0, 20) 
t_eval = np.linspace(0, 20, 1000)

#building different adjacency matricies:

#for the adjacency matrix: 
random_weights_std = 1.0

A_fc = np.ones((N, N)) - np.eye(N)

grid = nx.grid_2d_graph(int(np.sqrt(N)), int(np.sqrt(N)))
grid = nx.relabel_nodes(grid, {(r, c): r * int(np.sqrt(N)) + c for r in range(int(np.sqrt(N))) for c in range(int(np.sqrt(N)))})
A_square = nx.to_numpy_array(grid, dtype=int)


rng = np.random.default_rng(RUN_SEED) 
edge_density = 0.75  # fraction of possible undirected edges to keep
i, j = np.triu_indices(N, k=1)
exists = rng.random(len(i)) < edge_density #indicies of existing edges
weights = rng.lognormal(0, random_weights_std, len(i))

A_random_edges_unsymmetrized = np.zeros((N, N))
A_random_edges_unsymmetrized[i[exists], j[exists]] = 1
A_random_edges = A_random_edges_unsymmetrized + A_random_edges_unsymmetrized.T #symmetrize

A_random_edges_weights_unsymmetrized = np.zeros((N, N))
A_random_edges_weights_unsymmetrized[i[exists], j[exists]] = weights[exists]
A_random_edges_weights = A_random_edges_weights_unsymmetrized + A_random_edges_weights_unsymmetrized.T

W = rng.lognormal(0, random_weights_std, (N, N)) # mean, std, shape, lognormal is exp(normal_dist) so always positive, left heavy
W = (W + W.T) / 2; np.fill_diagonal(W, 0) # symmetrize and set diagonal to 0
A_random_weights = (np.ones((N, N)) - np.eye(N)) * W   # FC conncetivity, random strengths

def generate_ring_matrix(N, k):
    # Create the offsets for k nearest neighbors in both directions
    offsets = list(range(1, k + 1))


    # Generate the circulant graph for the ring
    G = nx.circulant_graph(N, offsets)
    
    # Get the adjacency matrix as a dense NumPy array
    adj_matrix = nx.to_numpy_array(G, dtype=int) 
    
    return adj_matrix

# k = 212.5 is k_c for sync (Wiley et al.); use smaller k for faster runs
ring_k = 10
A_ring = generate_ring_matrix(N, ring_k)


ei_ring, ej_ring = np.where(np.triu(A_ring, 1))
A_ring_heterogeneous = np.zeros((N, N))
w_ring = rng.lognormal(0, random_weights_std, len(ei_ring))
A_ring_heterogeneous[ei_ring, ej_ring] = w_ring
A_ring_heterogeneous[ej_ring, ei_ring] = w_ring


d = np.minimum(np.abs(ei_ring - ej_ring), N - np.abs(ei_ring - ej_ring))
w_ring_falloff = 1.0 / d
A_ring_falloff = np.zeros((N, N))
A_ring_falloff[ei_ring, ej_ring] = w_ring_falloff
A_ring_falloff[ej_ring, ei_ring] = w_ring_falloff


A = A_ring_falloff  # A_fc, A_square, A_random_weights, A_random_edges, A_random_edges_weights



A_matrices_names = {
    'A_fc': A_fc,
    'A_square': A_square,
    'A_random_weights': A_random_weights,
    'A_random_edges': A_random_edges,
    'A_random_edges_weights': A_random_edges_weights,
    'A_ring': A_ring,
    'A_ring_heterogeneous': A_ring_heterogeneous,
    'A_ring_falloff': A_ring_falloff,
}

A_name = next(name for name, mat in A_matrices_names.items() if mat is A)



def effective_resistance_sparsification(A, q, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    N = A.shape[0]
    # L = D - A
    degree_matrix = np.diag(A.sum(1)) 
    graph_laplacian = degree_matrix - A
    graph_laplacian_psuedoinverse = np.linalg.pinv(graph_laplacian)
    effective_resistances = np.zeros((N,N))
    for i in range(N): #double for loop for now for readability, but there are more efficient ways 
        for j in range(N):
            e = np.zeros((N, 1))
            e[i, 0], e[j, 0] = 1, -1 # ei - ej
            effective_resistances[i,j] = (e.T @ graph_laplacian_psuedoinverse @ e).item() #item so its a number not a 1x1 matrix
    edge_i, edge_j = np.where(np.triu(A, 1))  # #np.triu(A, 1) keeps the upper triangle of A, excluding the diagonal
    we = A[edge_i, edge_j].astype(float) #we is the weight of the edge
    Re = we * effective_resistances[edge_i, edge_j]
    pe = Re / Re.sum() # the prob of sampling edge is the effective resistance * weight of that edge divided by the sum of the effective resistances * weights of all edges
    s = max(1, int(q * len(edge_i))) # number of edges to sampple is q times the number of edges in the original graph
    sparsified_matrix = np.zeros((N, N)) 
    for k in rng.choice(len(edge_i), s, replace=True, p=pe):
        w = we[k] / (s * pe[k]) #new weight is the old weight divided by the probability of sampling the edge and the number of edges to sample
        sparsified_matrix[edge_i[k], edge_j[k]] += w #add the new weight to the edge
        sparsified_matrix[edge_j[k], edge_i[k]] += w #undirected
    return sparsified_matrix


def weight_based_sparsification(A, q, rng=None): # only using edge weights, not effective resistance
    if rng is None:
        rng = np.random.default_rng()
    N = A.shape[0]
    edge_i, edge_j = np.where(np.triu(A, 1))
    we = A[edge_i, edge_j].astype(float)
    pe = we / we.sum()
    s = max(1, int(q * len(edge_i)))
    sparsified_matrix = np.zeros((N, N))
    for k in rng.choice(len(edge_i), s, replace=True, p=pe):
        w = we[k] / (s * pe[k])
        sparsified_matrix[edge_i[k], edge_j[k]] += w
        sparsified_matrix[edge_j[k], edge_i[k]] += w
    return sparsified_matrix


def ring_farthest_sparsification(A, q, rng=None):
    # keep the k/2 neighbor shells farthest from the center (distances > k//2)
    N = A.shape[0]
    ei, ej = np.where(np.triu(A, 1))
    d = np.minimum(np.abs(ei - ej), N - np.abs(ei - ej))
    k_max = int(d.max())
    keep = d > k_max // 2
    sparsified_matrix = np.zeros((N, N))
    sparsified_matrix[ei[keep], ej[keep]] = A[ei[keep], ej[keep]]
    sparsified_matrix[ej[keep], ei[keep]] = A[ej[keep], ei[keep]]
    return sparsified_matrix

def ring_every_other_sparsification(A, q, rng=None):
    # keep even-distance shells only (2, 4, 6, ...); no distance-1 edges
    N = A.shape[0]
    ei, ej = np.where(np.triu(A, 1))
    d = np.minimum(np.abs(ei - ej), N - np.abs(ei - ej))
    k_max = int(d.max())
    keep_distances = set(range(2, k_max + 1, 2))
    keep = np.isin(d, list(keep_distances))
    sparsified_matrix = np.zeros((N, N))
    sparsified_matrix[ei[keep], ej[keep]] = A[ei[keep], ej[keep]]
    sparsified_matrix[ej[keep], ei[keep]] = A[ej[keep], ei[keep]]
    return sparsified_matrix


def ring_odd_sparsification(A, q, rng=None):
    # keep odd-distance shells only (1, 3, 5, ...); couples even ↔ odd, not within parity
    N = A.shape[0]
    ei, ej = np.where(np.triu(A, 1))
    d = np.minimum(np.abs(ei - ej), N - np.abs(ei - ej))
    k_max = int(d.max())
    keep_distances = set(range(1, k_max + 1, 2))
    keep = np.isin(d, list(keep_distances))
    sparsified_matrix = np.zeros((N, N))
    sparsified_matrix[ei[keep], ej[keep]] = A[ei[keep], ej[keep]]
    sparsified_matrix[ej[keep], ei[keep]] = A[ej[keep], ei[keep]]
    return sparsified_matrix


# identical ω for ring (Wiley et al. twist analysis); spread ω otherwise
if A_name in ('A_ring', 'A_ring_heterogeneous'):
    omega = np.full(N, 5.0)
else:
    omega = rng.normal(loc=5.0, scale=0.5, size=N)

sparsify_rng = np.random.default_rng(RUN_SEED + 1)
A_sparse_ER= effective_resistance_sparsification(A, q, sparsify_rng)
A_sparse_weight_based = weight_based_sparsification(A, q, sparsify_rng)

theta_0 = rng.uniform(0, 2*np.pi, N)           # Initial phases

def edges_from_A(A): # adjacency matrix to edges (i, j, weight), used in kuramoto_rhs instead of matmuls for speed
    """upper-triangle (i, j, weight) for undirected A."""
    ei, ej = np.where(np.triu(A, 1))
    return ei, ej, A[ei, ej].astype(float) # shape (E,), (E,), (E,) where E is the number of edges
    # for each edge that exists, return the indices of the nodes it connects and the weight of the edge


def weighted_degree(A):
    return np.maximum(A.sum(axis=1), 1e-12)


def kuramoto_rhs(t, theta, K, omega, edges, degree):
    """
    theta: (N,) phases; omega: (N,) natural frequencies.
    edges: (ei, ej, w) from edges_from_A)
    degree: (N,) weighted degree per node — normalizes coupling locally.
    """
    ei, ej, w = edges
    coupling = np.zeros(N)
    np.add.at(coupling, ei, w * np.sin(theta[ej] - theta[ei]))
    np.add.at(coupling, ej, w * np.sin(theta[ei] - theta[ej]))
    return omega + K * coupling / degree


def top_weight_edges(A, max_edges):
    """keep strongest edges so matplotlib stays responsive."""
    ei, ej, w = edges_from_A(A)
    if len(ei) <= max_edges:
        return ei, ej, w
    idx = np.argpartition(w, -max_edges)[-max_edges:]
    return ei[idx], ej[idx], w[idx]


# DOP853: higher-order RK, better for long phase integrations than RK45
solver_kw = dict(method='DOP853', rtol=1e-9, atol=1e-11)

edges_full = edges_from_A(A)
edges_ER = edges_from_A(A_sparse_ER)
edges_weight_based = edges_from_A(A_sparse_weight_based)
degree_full = weighted_degree(A)
degree_ER = weighted_degree(A_sparse_ER)
degree_weight_based= weighted_degree(A_sparse_weight_based)

def order_param_series(theta_t):
    """r(t) = |⟨e^{iθ}⟩|; z(t) is the complex order parameter (axis 0 = oscillators)."""
    z = np.mean(np.exp(1j * theta_t), axis=0)
    return np.abs(z), z


solution_full = solve_ivp(
    kuramoto_rhs, t_span, theta_0, args=(K, omega, edges_full, degree_full),
    t_eval=t_eval, **solver_kw,
)

solution_ER = solve_ivp(
    kuramoto_rhs, t_span, theta_0, args=(K, omega, edges_ER, degree_ER),
    t_eval=t_eval, **solver_kw,
)

solution_weight_based = solve_ivp(
    kuramoto_rhs, t_span, theta_0, args=(K, omega, edges_weight_based, degree_weight_based),
    t_eval=t_eval, **solver_kw,
)

sol_full = solution_full.y
sol_ER = solution_ER.y
sol_weight_based = solution_weight_based.y

r_full, z = order_param_series(sol_full)
r_sparse, z_sparse = order_param_series(sol_ER)
r_sparse_weight_based, z_sparse_weight_based = order_param_series(sol_weight_based)

err_r_ER = np.abs(r_full - r_sparse)
err_psi_ER = np.degrees(np.abs(np.angle(z * np.conj(z_sparse))))
err_r_weight = np.abs(r_full - r_sparse_weight_based)
err_psi_weight = np.degrees(np.abs(np.angle(z * np.conj(z_sparse_weight_based))))

if __name__ == "__main__" and bool_plots:
    plot_style = {
        'full': dict(color='C0', linestyle='-'),
        'ER': dict(color='C1', linestyle='--'),
        'weight': dict(color='C2', linestyle=':'),
    }
    fig, (ax_r, ax_dr, ax_dpsi) = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    ax_r.plot(t_eval, r_full, label='full', **plot_style['full'])
    ax_r.plot(t_eval, r_sparse, label='sparse ER', **plot_style['ER'])
    ax_r.plot(t_eval, r_sparse_weight_based, label='sparse weight', **plot_style['weight'])
    ax_r.set_ylabel('r = |⟨e^{iθ}⟩|')
    ax_r.set_ylim(0, 1.05)
    ax_r.legend()
    ax_r.set_title('order parameter (sync strength, not Re/Im of z)')

    ax_dr.plot(t_eval, err_r_ER, label='ER', **plot_style['ER'])
    ax_dr.plot(t_eval, err_r_weight, label='weight', **plot_style['weight'])
    ax_dr.set_ylabel('|Δr|')
    ax_dr.legend(title='sparsify')

    ax_dpsi.plot(t_eval, err_psi_ER, label='ER', **plot_style['ER'])
    ax_dpsi.plot(t_eval, err_psi_weight, label='weight', **plot_style['weight'])
    ax_dpsi.set_ylabel('|Δψ| (°)')
    ax_dpsi.set_xlabel('time (s)')
    ax_dpsi.legend(title='sparsify')

    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / 'kuramoto.png', dpi=120)
    fig.tight_layout()
    plt.show()






