"""ER vs weight-based vs random sparsification on a heterogeneous core-periphery
Kuramoto network.

Topology rationale (from Mercier, Scarpino & Moore 2022, "Effective Resistance
for Pandemics"): effective-resistance (ER) sparsification beats weight-based
and random edge-sampling specifically on networks that mix
  - a few dense, well-connected "core" communities (many redundant paths, so
    individual edges have low effective resistance even if high weight), with
  - several sparse, tree-like "periphery" communities (every edge is close to
    its only path, so weR_e ~ 1 regardless of weight), connected to the core
  - via single, low-weight "bridge" edges that are nonetheless the only route
    in or out (high effective resistance "weak ties").

Weight-based / random sampling tends to spend its edge budget on the
high-weight core and miss the low-weight tree/bridge edges, disconnecting
periphery nodes. ER recognizes that the bridge and tree edges are structurally
critical and keeps them.

We build this network, then for a range of preserved-edge fractions q:
  1) measure connectivity loss (fraction of nodes outside the largest
     connected component) -- the paper's Figure 5 analog, and
  2) run Kuramoto dynamics (full vs each sparsifier) and compare the global
     and per-community order parameter r(t).

Run: python heterogeneous_sparsification_compare.py
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.csgraph import connected_components

from network_plotter import compute_layout, plot_network

solver_kw = dict(method='RK45', rtol=1e-6, atol=1e-8)


def edges_from_A(A):
    ei, ej = np.where(np.triu(A, 1))
    return ei, ej, A[ei, ej].astype(float)


def weighted_degree(A):
    return np.maximum(A.sum(axis=1), 1e-12)


def kuramoto_rhs(t, theta, K, omega, edges, degree):
    ei, ej, w = edges
    coupling = np.zeros(theta.shape[0])
    np.add.at(coupling, ei, w * np.sin(theta[ej] - theta[ei]))
    np.add.at(coupling, ej, w * np.sin(theta[ei] - theta[ej]))
    return omega + K * coupling / degree


def order_param_series(theta_t):
    z = np.mean(np.exp(1j * theta_t), axis=0)
    return np.abs(z), z

PLOT_DIR = Path('plots/heterogeneous_sparsification')

METHODS = ('er', 'weight_based', 'random')
METHOD_LABELS = {
    'er': 'ER (effective resistance)',
    'weight_based': 'weight based',
    'random': 'random',
}
METHOD_COLORS = {'er': 'C0', 'weight_based': 'C1', 'random': 'C2'}


def fresh_seed():
    return int.from_bytes(os.urandom(8), 'big') % (2**63)


# ---------------------------------------------------------------------------
# network construction
# ---------------------------------------------------------------------------
def build_heterogeneous_network(
    rng,
    core_sizes=(50, 50),
    periph_sizes=(25, 25, 25, 25, 25),
    core_p=0.8,
    periph_extra_edge_frac=0.15,
    core_weight_mu=1.5,
    periph_weight_mu=-1.0,
    bridge_weight_mu=-1.5,
    weight_sigma=0.5,
    n_core_core_bridges=1,
):
    """Core-periphery network with heterogeneous log-normal edge weights.

    - each "core" community is an Erdos-Renyi-like dense block (prob core_p)
    - each "periphery" community is a random spanning tree plus a few extra
      edges (prob periph_extra_edge_frac among non-tree pairs)
    - each periphery community attaches to a random core community via a
      single low-weight bridge edge (the critical "weak tie")
    - a few extra core-core bridges connect the cores to each other

    Returns
    -------
    A : (N, N) symmetric weighted adjacency matrix
    community : (N,) int array, community index (cores first, then peripheries)
    kind : list of 'core'/'periph' per community index
    """
    sizes = list(core_sizes) + list(periph_sizes)
    kinds = ['core'] * len(core_sizes) + ['periph'] * len(periph_sizes)
    n = sum(sizes)
    offsets = np.cumsum([0] + sizes[:-1])
    community = np.concatenate([np.full(s, ci) for ci, s in enumerate(sizes)])

    A = np.zeros((n, n))

    def lognorm_w(mu, size=None):
        return rng.lognormal(mu, weight_sigma, size)

    # --- intra-community structure ---
    for ci, (off, size, kind) in enumerate(zip(offsets, sizes, kinds)):
        nodes = np.arange(off, off + size)
        if kind == 'core':
            i, j = np.triu_indices(size, k=1)
            exists = rng.random(len(i)) < core_p
            i, j = i[exists], j[exists]
            w = lognorm_w(core_weight_mu, len(i))
        else:  # periph: random spanning tree + sparse extra edges
            perm = rng.permutation(size)
            tree_i, tree_j = [], []
            for k in range(1, size):
                child = perm[k]
                parent = perm[rng.integers(0, k)]
                tree_i.append(min(child, parent))
                tree_j.append(max(child, parent))
            i_full, j_full = np.triu_indices(size, k=1)
            tree_set = set(zip(tree_i, tree_j))
            extra_mask = np.array([(a, b) not in tree_set for a, b in zip(i_full, j_full)])
            extra_i, extra_j = i_full[extra_mask], j_full[extra_mask]
            keep_extra = rng.random(len(extra_i)) < periph_extra_edge_frac
            i = np.concatenate([np.array(tree_i, dtype=int), extra_i[keep_extra]])
            j = np.concatenate([np.array(tree_j, dtype=int), extra_j[keep_extra]])
            w = lognorm_w(periph_weight_mu, len(i))
        A[nodes[i], nodes[j]] = w
        A[nodes[j], nodes[i]] = w

    # --- periphery -> random core bridges (one critical weak tie each) ---
    core_community_ids = [ci for ci, k in enumerate(kinds) if k == 'core']
    for ci, (off, size, kind) in enumerate(zip(offsets, sizes, kinds)):
        if kind != 'periph':
            continue
        target_ci = rng.choice(core_community_ids)
        t_off, t_size = offsets[target_ci], sizes[target_ci]
        a = off + rng.integers(0, size)
        b = t_off + rng.integers(0, t_size)
        w = lognorm_w(bridge_weight_mu)
        A[a, b] += w
        A[b, a] += w

    # --- a few core-core bridges ---
    if len(core_community_ids) > 1:
        for _ in range(n_core_core_bridges):
            c1, c2 = rng.choice(core_community_ids, size=2, replace=False)
            a = offsets[c1] + rng.integers(0, sizes[c1])
            b = offsets[c2] + rng.integers(0, sizes[c2])
            w = lognorm_w(bridge_weight_mu)
            A[a, b] += w
            A[b, a] += w

    return A, community, kinds


def build_scale_free_network(
    rng,
    n=150,
    m=2,
    hub_frac=0.15,
    hub_weight_mu=1.5,
    leaf_weight_mu=-1.0,
    weight_sigma=0.5,
):
    """Barabasi-Albert preferential-attachment network with heterogeneous weights.

    Unlike the hand-built core-periphery network, this has no explicit community
    blocks -- it's a more 'organic' test of the same idea. High-degree hub nodes
    end up with many redundant high-weight edges (low per-edge effective
    resistance despite high weight); low-degree leaf nodes hang off the network
    by a handful of low-weight edges that are close to their only path to the
    rest of the graph (weR_e ~ 1, high effective resistance). Weight-based /
    random sampling spends its budget on the hub edges and prunes leaves away;
    ER should preserve the leaves' lifelines.

    Returns
    -------
    A : (n, n) symmetric weighted adjacency matrix
    community : (n,) int array, 0 = 'hub' community, 1 = 'leaf' community
    kinds : ['core', 'periph'] (so existing core/periph analysis code applies)
    """
    edges_i, edges_j = [], []
    degree = np.zeros(n, dtype=int)
    for i in range(m):
        for j in range(i + 1, m):
            edges_i.append(i)
            edges_j.append(j)
            degree[i] += 1
            degree[j] += 1
    for new in range(m, n):
        existing = np.arange(new)
        p = degree[:new].astype(float) + 1.0
        p = p / p.sum()
        n_targets = min(m, new)
        targets = rng.choice(existing, size=n_targets, replace=False, p=p)
        for t in targets:
            edges_i.append(new)
            edges_j.append(int(t))
            degree[new] += 1
            degree[t] += 1

    edges_i = np.array(edges_i)
    edges_j = np.array(edges_j)
    hub_thresh = np.quantile(degree, 1 - hub_frac)
    hub_mask = degree >= hub_thresh

    w = np.empty(len(edges_i))
    for idx, (a, b) in enumerate(zip(edges_i, edges_j)):
        mu = hub_weight_mu if (hub_mask[a] or hub_mask[b]) else leaf_weight_mu
        w[idx] = rng.lognormal(mu, weight_sigma)

    A = np.zeros((n, n))
    A[edges_i, edges_j] = w
    A[edges_j, edges_i] = w

    community = (~hub_mask).astype(int)  # 0 = hub ('core'), 1 = leaf ('periph')
    kinds = ['core', 'periph']
    return A, community, kinds


def build_connectome_network(rng, weights_path='data/connectome_66/weights.txt', hub_frac=0.15):
    """Real human structural brain connectome (Hagmann et al. DSI tractography,
    66 cortical regions, from The Virtual Brain demo dataset) -- the same kind
    of network used in Kuramoto whole-brain models (e.g. Cabral et al. 2011).

    The raw matrix is asymmetric with nonzero diagonal (an artifact of the
    tractography pipeline); we symmetrize and zero the diagonal to get a
    weighted undirected graph. Regions are split into 'hub' (core) vs
    'non-hub' (periphery) by weighted degree, exactly as for the synthetic
    scale-free network -- real connectomes are known to have a small-world,
    hub-dominated 'rich club' structure, so the same ER-favors-low-degree-
    high-resistance-edges argument should apply.

    `rng` is accepted for interface consistency but unused (the topology is
    fixed real data).
    """
    W = np.loadtxt(weights_path)
    A = (W + W.T) / 2.0
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]

    degree = A.sum(axis=1)
    hub_thresh = np.quantile(degree, 1 - hub_frac)
    hub_mask = degree >= hub_thresh

    community = (~hub_mask).astype(int)  # 0 = hub ('core'), 1 = non-hub ('periph')
    kinds = ['core', 'periph']
    return A, community, kinds


def build_uk_power_grid_network(rng, weights_path='data/uk_power_grid/adjacency.npy', hub_frac=0.15):
    """Real UK high-voltage power grid network (631 substations, 758 transmission
    lines), from the NeuralABM Kuramoto power-grid dataset (ThGaskin/NeuralABM).
    Power-grid synchronization is one of the original and most studied
    applications of the Kuramoto model (generator/bus phase angles), and grids
    are known to have a few heavily-meshed hub substations plus long, tree-like
    radial feeders to peripheral substations connected by single critical lines
    -- exactly the structure where ER sparsification should help. As with the
    connectome, nodes are split into 'hub' (core, top hub_frac by weighted
    degree) vs 'non-hub' (periphery).

    `rng` is accepted for interface consistency but unused (fixed real data).
    """
    A = np.load(weights_path)
    n = A.shape[0]

    degree = A.sum(axis=1)
    hub_thresh = np.quantile(degree, 1 - hub_frac)
    hub_mask = degree >= hub_thresh

    community = (~hub_mask).astype(int)  # 0 = hub ('core'), 1 = non-hub ('periph')
    kinds = ['core', 'periph']
    return A, community, kinds


def build_celegans_network(rng, weights_path='data/celegans_adjacency.npy', hub_frac=0.15):
    """Real C. elegans neural connectome (297 neurons, Watts & Strogatz 1998
    compilation of White et al. synapse-count data). Synchronization on this
    exact network is widely studied (it's one of the classic small-world test
    networks, and has been used directly in Kuramoto/oscillator
    synchronization papers). Edge weight = total synapse count between a
    neuron pair (summed over both directions, since our Kuramoto model is
    undirected). As with the other real networks, nodes are split into 'hub'
    (core, top hub_frac by weighted degree) vs 'non-hub' (periphery).

    `rng` is accepted for interface consistency but unused (fixed real data).
    """
    A = np.load(weights_path)
    n = A.shape[0]

    degree = A.sum(axis=1)
    hub_thresh = np.quantile(degree, 1 - hub_frac)
    hub_mask = degree >= hub_thresh

    community = (~hub_mask).astype(int)  # 0 = hub ('core'), 1 = non-hub ('periph')
    kinds = ['core', 'periph']
    return A, community, kinds


def build_fully_connected_network(rng, n=66, weight=1.0):
    """Complete graph with homogeneous edge weights -- a null-model baseline.
    By symmetry every edge has the same weight and the same effective
    resistance, so ER, weight-based, and random sparsification should all
    behave identically here. `rng` is unused (fixed topology); n=66 matches
    connectome_66 for size comparability. The 50/50 'core'/'periph' community
    split is arbitrary (the network itself has no structure) -- it exists only
    so the existing per-community plotting code has something to plot.
    """
    A = weight * (np.ones((n, n)) - np.eye(n))
    community = np.zeros(n, dtype=int)
    community[n // 2:] = 1
    kinds = ['core', 'periph']
    return A, community, kinds


def build_ring_network(rng, n=66, k=2, weight=1.0):
    """Ring lattice with homogeneous edge weights: each node connects to its k
    nearest neighbors on each side (degree 2k), as in the Watts-Strogatz
    small-world base graph. A plain 1-nearest-neighbor cycle (k=1) is a pure
    tree-like baseline where w_e*R_e is exactly uniform across edges (ER =
    random by symmetry); k>1 adds redundant short-range cycles, so ER can
    differ from random/weight-based even though all weights are equal.
    `rng` is unused (fixed topology); n=66 matches connectome_66 for size
    comparability. The 50/50 'core'/'periph' community split is arbitrary, as
    for the FC network.
    """
    A = np.zeros((n, n))
    for i in range(n):
        for d in range(1, k + 1):
            j = (i + d) % n
            A[i, j] = weight
            A[j, i] = weight
    community = np.zeros(n, dtype=int)
    community[n // 2:] = 1
    kinds = ['core', 'periph']
    return A, community, kinds


# ---------------------------------------------------------------------------
# sparsifiers
# ---------------------------------------------------------------------------
def precompute_er(A):
    graph_laplacian = np.diag(A.sum(1)) - A
    graph_laplacian_pinv = np.linalg.pinv(graph_laplacian)
    diag = np.diag(graph_laplacian_pinv)
    eff_res = diag[:, None] + diag[None, :] - 2 * graph_laplacian_pinv
    edge_i, edge_j = np.where(np.triu(A, 1))
    we = A[edge_i, edge_j].astype(float)
    Re = we * eff_res[edge_i, edge_j]
    pe = Re / Re.sum()
    return edge_i, edge_j, we, pe


def sparsify_stochastic(edge_i, edge_j, we, pe, q, rng, n):
    s = max(1, int(round(q * len(edge_i))))
    A = np.zeros((n, n))
    for idx in rng.choice(len(edge_i), s, replace=True, p=pe):
        w = we[idx] / (s * pe[idx])
        A[edge_i[idx], edge_j[idx]] += w
        A[edge_j[idx], edge_i[idx]] += w
    return A


def sparsify_graph(A, method, q, rng, precomputed_er=None):
    if precomputed_er is None:
        edge_i, edge_j, we, pe_er = precompute_er(A)
    else:
        edge_i, edge_j, we, pe_er = precomputed_er
    if method == 'er':
        pe = pe_er
    elif method == 'weight_based':
        pe = we / we.sum()
    elif method == 'random':
        pe = np.full(len(edge_i), 1.0 / len(edge_i))
    else:
        raise ValueError(method)
    return sparsify_stochastic(edge_i, edge_j, we, pe, q, rng, A.shape[0])


# ---------------------------------------------------------------------------
# connectivity sweep (paper Fig 5 analog)
# ---------------------------------------------------------------------------
def frac_disconnected(A):
    n = A.shape[0]
    n_comp, labels = connected_components(A, directed=False)
    if n_comp <= 1:
        return 0.0
    sizes = np.bincount(labels)
    largest = sizes.max()
    return (n - largest) / n


def connectivity_sweep(A, qs, n_seeds, base_seed):
    precomputed_er = precompute_er(A)
    out = {m: np.zeros((len(qs), n_seeds)) for m in METHODS}
    for qi, q in enumerate(qs):
        for si in range(n_seeds):
            rng = np.random.default_rng(base_seed + 1000 * qi + si)
            for m in METHODS:
                A_sparse = sparsify_graph(A, m, q, rng, precomputed_er)
                out[m][qi, si] = frac_disconnected(A_sparse)
    return out


def plot_connectivity_sweep(qs, out, plots_dir, stem):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for m in METHODS:
        mean = out[m].mean(axis=1)
        n_seeds = out[m].shape[1]
        moe = 1.96 * out[m].std(axis=1) / np.sqrt(n_seeds)
        ax.plot(qs, mean, label=METHOD_LABELS[m], color=METHOD_COLORS[m], marker='o')
        ax.fill_between(qs, mean - moe, mean + moe, color=METHOD_COLORS[m], alpha=0.2)
    ax.set_xscale('log')
    ax.set_xticks(qs)
    ax.set_xticklabels([f'{q:g}' for q in qs])
    ax.minorticks_off()
    ax.set_xlabel('fraction of edges preserved (q)')
    ax.set_ylabel('fraction of nodes disconnected\nfrom largest component')
    ax.set_title('connectivity loss vs sparsification')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = plots_dir / f'{stem}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Kuramoto dynamics comparison
# ---------------------------------------------------------------------------
def integrate(A, omega, theta_0, K, t_eval):
    edges = edges_from_A(A)
    degree = weighted_degree(A)
    sol = solve_ivp(
        kuramoto_rhs, (0.0, t_eval[-1]), theta_0,
        args=(K, omega, edges, degree),
        t_eval=t_eval, **solver_kw,
    ).y
    return sol


def community_order_params(sol, community, kinds, sizes_by_kind):
    """global r(t) plus r(t) restricted to all 'core' nodes and all 'periph' nodes."""
    r_global, z_global = order_param_series(sol)
    core_mask = np.isin(community, [ci for ci, k in enumerate(kinds) if k == 'core'])
    periph_mask = ~core_mask
    r_core, z_core = order_param_series(sol[core_mask])
    r_periph, z_periph = order_param_series(sol[periph_mask])
    return dict(
        r_global=r_global, z_global=z_global,
        r_core=r_core, z_core=z_core,
        r_periph=r_periph, z_periph=z_periph,
    )


def dynamics_errors(sol_full, sol_sparse, community, kinds, t_eval, steady_frac=0.5):
    full_m = community_order_params(sol_full, community, kinds, None)
    sparse_m = community_order_params(sol_sparse, community, kinds, None)
    late = t_eval >= t_eval[-1] * (1 - steady_frac)
    out = {}
    for key, label in [('r_global', 'global'), ('r_core', 'core'), ('r_periph', 'periph')]:
        err = np.abs(full_m[key] - sparse_m[key])
        out[f'{label}_mean'] = float(np.mean(err))
        out[f'{label}_steady'] = float(np.mean(err[late]))
    return out


def dynamics_sweep(A, community, kinds, qs, n_seeds, base_seed, K_coef=0.1, t_max=20, n_t=600):
    """Sweep over q and seeds, returning (1) scalar error metrics per (method, q, seed)
    and (2) running mean/std of the r(t) trajectories themselves (full + each sparsifier,
    per q), averaged over seeds -- the inputs for a Fig.-2-style averaged comparison."""
    n = A.shape[0]
    t_eval = np.linspace(0, t_max, n_t)
    precomputed_er = precompute_er(A)
    K = K_coef * n
    metrics = ['global_mean', 'global_steady', 'core_mean', 'core_steady',
               'periph_mean', 'periph_steady']
    keys = ['r_global', 'r_core', 'r_periph']
    out = {m: {met: np.zeros((len(qs), n_seeds)) for met in metrics} for m in METHODS}

    n_t_ = len(t_eval)
    full_sum = {k: np.zeros(n_t_) for k in keys}
    full_sumsq = {k: np.zeros(n_t_) for k in keys}
    sparse_sum = {m: {q: {k: np.zeros(n_t_) for k in keys} for q in qs} for m in METHODS}
    sparse_sumsq = {m: {q: {k: np.zeros(n_t_) for k in keys} for q in qs} for m in METHODS}

    for si in range(n_seeds):
        rng = np.random.default_rng(base_seed + 7919 * si)
        omega = rng.normal(5.0, 0.5, n)
        theta_0 = rng.uniform(0, 2 * np.pi, n)
        sol_full = integrate(A, omega, theta_0, K, t_eval)
        full_m = community_order_params(sol_full, community, kinds, None)
        for k in keys:
            full_sum[k] += full_m[k]
            full_sumsq[k] += full_m[k] ** 2
        for qi, q in enumerate(qs):
            for m in METHODS:
                sp_rng = np.random.default_rng(
                    base_seed + 1_000_000 * qi + 7919 * si + hash(m) % 1000)
                A_sparse = sparsify_graph(A, m, q, sp_rng, precomputed_er)
                sol_sparse = integrate(A_sparse, omega, theta_0, K, t_eval)
                sparse_m = community_order_params(sol_sparse, community, kinds, None)
                errs = dynamics_errors(sol_full, sol_sparse, community, kinds, t_eval)
                for met in metrics:
                    out[m][met][qi, si] = errs[met]
                for k in keys:
                    sparse_sum[m][q][k] += sparse_m[k]
                    sparse_sumsq[m][q][k] += sparse_m[k] ** 2

    def mean_std(sum_, sumsq):
        mean = sum_ / n_seeds
        var = np.maximum(sumsq / n_seeds - mean ** 2, 0)
        return mean, np.sqrt(var)

    traj_stats = dict(
        t_eval=t_eval, n_seeds=n_seeds,
        full_mean={}, full_std={}, sparse_mean={m: {} for m in METHODS},
        sparse_std={m: {} for m in METHODS},
    )
    for k in keys:
        traj_stats['full_mean'][k], traj_stats['full_std'][k] = mean_std(full_sum[k], full_sumsq[k])
    for m in METHODS:
        for q in qs:
            traj_stats['sparse_mean'][m][q] = {}
            traj_stats['sparse_std'][m][q] = {}
            for k in keys:
                mean, std = mean_std(sparse_sum[m][q][k], sparse_sumsq[m][q][k])
                traj_stats['sparse_mean'][m][q][k] = mean
                traj_stats['sparse_std'][m][q][k] = std

    # difference of IC-averaged curves (as opposed to mean of per-IC |differences|
    # in `out`): |<r_sparse>_ICs - <r_full>_ICs|, averaged over the steady-state window.
    late = t_eval >= t_eval[-1] * 0.5
    diff_metrics = ['global_steady', 'core_steady', 'periph_steady']
    diff_out = {m: {met: np.zeros(len(qs)) for met in diff_metrics} for m in METHODS}
    for m in METHODS:
        for qi, q in enumerate(qs):
            for met, key in zip(diff_metrics, keys):
                fm = traj_stats['full_mean'][key]
                sm = traj_stats['sparse_mean'][m][q][key]
                diff_out[m][met][qi] = np.mean(np.abs(sm[late] - fm[late]))

    return out, traj_stats, diff_out


def plot_dynamics_sweep(qs, out, diff_out, plots_dir, stem, n_seeds):
    """Top row: mean (over ICs) of per-IC |r_full(t) - r_sparse(t)|, steady-state avg.
    Bottom row: |<r_full(t)>_ICs - <r_sparse(t)>_ICs|, steady-state avg -- i.e. the
    difference between the IC-averaged curves themselves. The bottom row can be much
    smaller than the top when per-IC errors partially cancel across ICs."""
    panels = [('global_steady', 'global'), ('core_steady', 'core'), ('periph_steady', 'periphery')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=False)
    for ax, (met, title) in zip(axes[0], panels):
        for m in METHODS:
            mean = out[m][met].mean(axis=1)
            moe = 1.96 * out[m][met].std(axis=1) / np.sqrt(n_seeds)
            ax.plot(qs, mean, label=METHOD_LABELS[m], color=METHOD_COLORS[m], marker='o')
            ax.fill_between(qs, np.maximum(mean - moe, 0), mean + moe,
                             color=METHOD_COLORS[m], alpha=0.2)
        ax.set_xscale('log')
        ax.set_xticks(qs)
        ax.set_xticklabels([f'{q:g}' for q in qs])
        ax.minorticks_off()
        ax.set_title(title)
        ax.grid(alpha=0.3)
    for ax, (met, title) in zip(axes[1], panels):
        for m in METHODS:
            ax.plot(qs, diff_out[m][met], label=METHOD_LABELS[m], color=METHOD_COLORS[m], marker='o')
        ax.set_xscale('log')
        ax.set_xticks(qs)
        ax.set_xticklabels([f'{q:g}' for q in qs])
        ax.minorticks_off()
        ax.set_xlabel('fraction of edges preserved (q)')
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel('mean of per-IC |Δr|\n(steady state)')
    axes[1, 0].set_ylabel('|<r_full> - <r_sparse>|\n(steady state)')
    axes[0, 0].legend()
    fig.suptitle(f'Kuramoto dynamics error vs sparsification level (over {n_seeds} ICs)')
    fig.tight_layout()
    path = plots_dir / f'{stem}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_dynamics_comparison_grid(traj_stats, q_cols, plots_dir, stem):
    t_eval = traj_stats['t_eval']
    n_seeds = traj_stats['n_seeds']
    keys = ['r_global', 'r_core', 'r_periph']
    row_titles = ['global', 'core', 'periphery']
    fig, axes = plt.subplots(3, len(q_cols), figsize=(4 * len(q_cols), 9),
                              sharex=True, sharey='row')
    for col, q in enumerate(q_cols):
        for row, key in enumerate(keys):
            ax = axes[row, col]
            fm, fs = traj_stats['full_mean'][key], traj_stats['full_std'][key]
            ax.plot(t_eval, fm, 'k-', label='full', lw=2)
            ax.fill_between(t_eval, fm - fs, fm + fs, color='k', alpha=0.15)
            for m in METHODS:
                sm = traj_stats['sparse_mean'][m][q][key]
                ss = traj_stats['sparse_std'][m][q][key]
                ax.plot(t_eval, sm, color=METHOD_COLORS[m], linestyle='--', label=METHOD_LABELS[m])
                ax.fill_between(t_eval, sm - ss, sm + ss, color=METHOD_COLORS[m], alpha=0.15)
        axes[0, col].set_title(f'q={q:g}')

    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(f'{title}\nr')
        axes[row, 0].set_ylim(0, 1.05)
        for col in range(len(q_cols)):
            axes[row, col].grid(alpha=0.3)
    for col in range(len(q_cols)):
        axes[-1, col].set_xlabel('time')
    axes[0, 0].legend()
    fig.suptitle(f'Kuramoto dynamics, mean ± std over {n_seeds} ICs')
    fig.tight_layout()
    path = plots_dir / f'{stem}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_network_comparison(A, community, q_demo, base_seed, plots_dir, stem):
    """Force-directed layout of the full network and each sparsified network at q_demo,
    all sharing the same node positions (computed once on the full network) and the
    same node color scale (community index) so they're directly comparable."""
    pos = compute_layout(community.astype(float), A, iterations=200, seed=base_seed)
    precomputed_er = precompute_er(A)
    titles = ['original'] + [METHOD_LABELS[m] for m in METHODS]
    matrices = [A]
    for m in METHODS:
        sp_rng = np.random.default_rng(base_seed + hash(m) % 10_000)
        matrices.append(sparsify_graph(A, m, q_demo, sp_rng, precomputed_er))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    color_norm = (float(community.min()), float(community.max()))
    for ax, title, mat in zip(axes, titles, matrices):
        plot_network(community.astype(float), mat, ax=ax, initial_pos=pos, iterations=0,
                      add_colorbar=(ax is axes[-1]), color_norm=color_norm, node_radius=15)
        ax.set_title(title)
    fig.suptitle(f'network structure, q={q_demo:g} preserved edges (color = community)')
    fig.tight_layout()
    path = plots_dir / f'{stem}.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_pipeline(network_name, A, community, kinds, run_seed):
    n = A.shape[0]
    n_edges = int(np.count_nonzero(np.triu(A, 1)))
    print(f'\n=== network: {network_name} (n={n} nodes, {n_edges} edges, '
          f'{sum(1 for k in kinds if k == "core")} core / '
          f'{sum(1 for k in kinds if k == "periph")} periph communities) ===')

    plots_dir = PLOT_DIR / network_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- network structure visualization ---
    plot_network_comparison(A, community, q_demo=0.25, base_seed=run_seed,
                             plots_dir=plots_dir, stem='network_structure')

    # --- connectivity sweep ---
    qs = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    conn = connectivity_sweep(A, qs, n_seeds=100, base_seed=run_seed)
    print(f'{"q":>6s}  ' + '  '.join(f'{m:>14s}' for m in METHODS))
    for qi, q in enumerate(qs):
        print(f'{q:6.3f}  ' + '  '.join(f'{conn[m][qi].mean():14.3f}' for m in METHODS))
    plot_connectivity_sweep(qs, conn, plots_dir, 'connectivity_sweep')

    # --- dynamics error sweep + averaged trajectory comparison (over many ICs) ---
    dyn_qs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    n_seeds_dyn = 200
    dyn, traj_stats, diff_out = dynamics_sweep(A, community, kinds, dyn_qs, n_seeds=n_seeds_dyn,
                                                base_seed=run_seed)
    for met, label in [('global_steady', 'global'), ('core_steady', 'core'),
                        ('periph_steady', 'periph')]:
        print(f'\nsteady-state |Δr| ({label}):')
        print(f'{"q":>6s}  ' + '  '.join(f'{m:>14s}' for m in METHODS))
        for qi, q in enumerate(dyn_qs):
            print(f'{q:6.3f}  ' + '  '.join(f'{dyn[m][met][qi].mean():14.4f}' for m in METHODS))
        print(f'  (|<r_full>-<r_sparse>| steady-state, {label}):')
        print(f'{"q":>6s}  ' + '  '.join(f'{m:>14s}' for m in METHODS))
        for qi, q in enumerate(dyn_qs):
            print(f'{q:6.3f}  ' + '  '.join(f'{diff_out[m][met][qi]:14.4f}' for m in METHODS))
    plot_dynamics_sweep(dyn_qs, dyn, diff_out, plots_dir, 'dynamics_error_sweep', n_seeds_dyn)

    q_cols = [q for q in dyn_qs if q >= 0.1]
    plot_dynamics_comparison_grid(traj_stats, q_cols, plots_dir, 'dynamics_comparison')

    print(f'\nwrote plots to {plots_dir}/')


def main():
    run_seed = fresh_seed()
    print(f'run seed: {run_seed}')
    rng = np.random.default_rng(run_seed)

    networks = {
        'core_periphery': build_heterogeneous_network(rng),
        'scale_free_hub': build_scale_free_network(rng),
        'connectome_66': build_connectome_network(rng),
        'uk_power_grid': build_uk_power_grid_network(rng),
        'celegans': build_celegans_network(rng),
        'fully_connected': build_fully_connected_network(rng),
        'ring': build_ring_network(rng),
    }
    for network_name, (A, community, kinds) in networks.items():
        run_pipeline(network_name, A, community, kinds, run_seed)


if __name__ == '__main__':
    main()
