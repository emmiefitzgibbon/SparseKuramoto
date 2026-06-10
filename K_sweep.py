from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from kuramoto import (
    N, q, RUN_SEED, A_name, kuramoto_rhs, solver_kw,
    A_fc, A_square, edge_density, random_weights_std,
    effective_resistance_sparsification, weight_based_sparsification,
    edges_from_A, weighted_degree,
)

# sweep settings
num_trials = 10
K_values = np.linspace(0, 10, 50)
t_span = (0, 20)
t_eval = np.linspace(*t_span, 500)
steady_fraction = 0.2   # average r over the last 20% of the run
r_threshold = 0.5       # r level used to estimate K_c


def build_A(rng):
    """build the full adjacency matrix for the current A_name."""
    if A_name == "A_fc":
        return A_fc.copy()
    if A_name == "A_square":
        return A_square.copy()

    i, j = np.triu_indices(N, k=1)
    if A_name == "A_random_edges":
        exists = rng.random(len(i)) < edge_density
        A = np.zeros((N, N))
        A[i[exists], j[exists]] = 1
        return A + A.T

    weights = rng.lognormal(0, random_weights_std, len(i))
    if A_name == "A_random_edges_weights":
        exists = rng.random(len(i)) < edge_density
        A = np.zeros((N, N))
        A[i[exists], j[exists]] = weights[exists]
        return A + A.T

    if A_name == "A_random_weights":
        W = rng.lognormal(0, random_weights_std, (N, N))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        return (np.ones((N, N)) - np.eye(N)) * W

    raise ValueError(f"unknown A_name: {A_name}")


def setup_trial(seed):
    """sample graph, frequencies, initial phases, and sparsified graphs for one trial."""
    rng = np.random.default_rng(seed)
    A = build_A(rng)
    omega = rng.normal(loc=5.0, scale=0.5, size=N)
    theta_0 = rng.uniform(0, 2 * np.pi, N)
    A_sparse_ER = effective_resistance_sparsification(A, q, rng)
    A_sparse_weight_based = weight_based_sparsification(A, q, rng)
    return {
        "omega": omega,
        "theta_0": theta_0,
        "edges_full": edges_from_A(A),
        "edges_ER": edges_from_A(A_sparse_ER),
        "edges_weight_based": edges_from_A(A_sparse_weight_based),
        "degree_full": weighted_degree(A),
        "degree_ER": weighted_degree(A_sparse_ER),
        "degree_weight_based": weighted_degree(A_sparse_weight_based),
    }


def order_parameter_trajectory(theta_t):
    z = np.mean(np.exp(1j * theta_t), axis=0)
    return np.abs(z)


def steady_state_r(sol, t_eval, fraction=steady_fraction):
    late = t_eval >= t_eval[-1] * (1 - fraction)
    return float(np.mean(order_parameter_trajectory(sol[:, late])))


def estimate_kc(K_values, r_values, threshold=r_threshold):
    """linear interpolation where r crosses threshold; nan if it never does."""
    r = np.asarray(r_values)
    K = np.asarray(K_values)
    above = r >= threshold
    if not np.any(above):
        return np.nan
    idx = int(np.argmax(above))
    if idx == 0:
        return float(K[0])
    K0, K1 = K[idx - 1], K[idx]
    r0, r1 = r[idx - 1], r[idx]
    if r1 == r0:
        return float(K1)
    return float(K0 + (threshold - r0) * (K1 - K0) / (r1 - r0))


r_full_trials = np.zeros((num_trials, len(K_values)))
r_ER_trials = np.zeros((num_trials, len(K_values)))
r_random_trials = np.zeros((num_trials, len(K_values)))

print(
    f"K sweep: {len(K_values)} values, {num_trials} trials, "
    f"{A_name}, N={N}, q={q}"
)
for trial in range(num_trials):
    seed = RUN_SEED + trial
    trial_data = setup_trial(seed)
    for idx, K in enumerate(K_values):
        sol_full = solve_ivp(
            kuramoto_rhs, t_span, trial_data["theta_0"],
            args=(K, trial_data["omega"], trial_data["edges_full"], trial_data["degree_full"]),
            t_eval=t_eval, **solver_kw,
        )
        sol_ER = solve_ivp(
            kuramoto_rhs, t_span, trial_data["theta_0"],
            args=(K, trial_data["omega"], trial_data["edges_ER"], trial_data["degree_ER"]),
            t_eval=t_eval, **solver_kw,
        )
        sol_random = solve_ivp(
            kuramoto_rhs, t_span, trial_data["theta_0"],
            args=(K, trial_data["omega"], trial_data["edges_weight_based"], trial_data["degree_weight_based"]),
            t_eval=t_eval, **solver_kw,
        )

        r_full_trials[trial, idx] = steady_state_r(sol_full.y, t_eval)
        r_ER_trials[trial, idx] = steady_state_r(sol_ER.y, t_eval)
        r_random_trials[trial, idx] = steady_state_r(sol_random.y, t_eval)

    print(f"  finished trial {trial + 1}/{num_trials} (seed={seed})")

r_full = r_full_trials.mean(axis=0)
r_ER = r_ER_trials.mean(axis=0)
r_random = r_random_trials.mean(axis=0)
r_full_std = r_full_trials.std(axis=0)
r_ER_std = r_ER_trials.std(axis=0)
r_random_std = r_random_trials.std(axis=0)

Kc_full = estimate_kc(K_values, r_full)
Kc_ER = estimate_kc(K_values, r_ER)
Kc_weight_based = estimate_kc(K_values, r_random)

print("\ncritical K estimates (r crosses {:.2f}, from mean r curve):".format(r_threshold))
print(f"  full:                K_c = {Kc_full:.3f}")
print(f"  ER sparse:           K_c = {Kc_ER:.3f}")
print(f"  weight based sparse: K_c = {Kc_weight_based:.3f}")
print(f"  |K_c full - ER| = {abs(Kc_full - Kc_ER):.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
for r_vals, r_std, style, label, color in [
    (r_full, r_full_std, "k-", "full", "k"),
    (r_ER, r_ER_std, "--", "ER sparsified", "C0"),
    (r_random, r_random_std, ":", "weight_based sparsified", "C1"),
]:
    ax.plot(K_values, r_vals, style, lw=2, label=label, color=color)
    ax.fill_between(
        K_values, r_vals - r_std, r_vals + r_std,
        color=color, alpha=0.15,
    )
ax.axhline(r_threshold, color="gray", lw=0.8, ls="--")
kc_entries = [
    (Kc_full, "full", "k"),
    (Kc_ER, "ER", "C0"),
    (Kc_weight_based, "weight_based", "C1"),
]
line_height = 0.035
kc_line = 0
for Kc, name, color in kc_entries:
    if not np.isnan(Kc):
        ax.axvline(Kc, color=color, lw=0.8, ls=":", alpha=0.7)
        ax.text(
            0.02, 0.02 + kc_line * line_height,
            f"{name}: K_c={Kc:.2f}",
            transform=ax.transAxes,
            fontsize=8, color=color, va="bottom", ha="left",
        )
        kc_line += 1
ax.set_xlabel("coupling K")
ax.set_ylabel("steady-state order parameter r")
ax.set_title(
    f"K sweep ({A_name.removeprefix('A_')}, N={N}, q={q}, {num_trials} trials)"
)
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout()
plots_dir = Path('plots')
plots_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plots_dir / 'K_sweep.png', dpi=120)
plt.show()
