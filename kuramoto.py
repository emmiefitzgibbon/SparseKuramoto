import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

N = 64
K = 5
q = 0.25
random_weights_std = 1.0

bool_plots = False

#critical value: 0.10 for fc graph, 0.5 std natural freq, 25 N
A_fc = np.ones((N, N)) - np.eye(N)


grid = nx.grid_2d_graph(int(np.sqrt(N)), int(np.sqrt(N)))
A_square = nx.to_numpy_array(grid, dtype=int)

rng = np.random.default_rng(0)
edge_density = 0.1  # fraction of possible undirected edges to keep
i, j = np.triu_indices(N, k=1)
exists = rng.random(len(i)) < edge_density
weights = rng.lognormal(0, random_weights_std, len(i))

A_random_edges_unsymmetrized = np.zeros((N, N))
A_random_edges_unsymmetrized[i[exists], j[exists]] = 1
A_random_edges = A_random_edges_unsymmetrized + A_random_edges_unsymmetrized.T #symmetrize

A_random_edges_weights_unsymmetrized = np.zeros((N, N))
A_random_edges_weights_unsymmetrized[i[exists], j[exists]] = weights[exists]
A_random_edges_weights = A_random_edges_weights_unsymmetrized + A_random_edges_weights_unsymmetrized.T

W = rng.lognormal(0, random_weights_std, (N, N)) # mean, std, shape, lognormal is exp(normal) so always positive
W = (W + W.T) / 2; np.fill_diagonal(W, 0) # symmetrize and set diagonal to 0
A_random_weights = (np.ones((N, N)) - np.eye(N)) * W   # FC topology, random strengths

A = A_random_weights # A_fc, A_square, A_random_weights, A_random_edges, A_random_edges_weights

def effective_resistance_sparsification(A, q):
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
    pe = Re / Re.sum()
    s = max(1, int(q * len(edge_i)))
    sparsified_matrix = np.zeros((N, N))
    for k in np.random.choice(len(edge_i), s, replace=True, p=pe):
        w = we[k] / (s * pe[k]) #new weight is the old weight divided by the probability of sampling the edge and the number of edges to sample
        sparsified_matrix[edge_i[k], edge_j[k]] += w #add the new weight to the edge
        sparsified_matrix[edge_j[k], edge_i[k]] += w #undirected
    return sparsified_matrix


def effective_resistance_sparsification_determinstic(A, q):
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
    pe = Re / Re.sum()
    s = max(1, int(q * len(edge_i)))
    sparsified_matrix = np.zeros((N, N))
    


#loc is the mean, scale is the standard deviation
omega = np.random.normal(loc=5.0, scale=0.5, size=N) # Intrinsic frequencies

A_sparse = effective_resistance_sparsification(A, q)

theta_0 = np.random.uniform(0, 2*np.pi, N)           # Initial phases

# Define the time span
t_span = (0, 20)
t_eval = np.linspace(0, 20, 1000)

def kuramoto_rhs(t, theta, K, N, omega, A):
    """ 
    theta: (N,) array of oscillator phases.
    omega: (N,) array of natural frequencies.
    A: adjacency matrix of the network
    """

    #Create a 2D matrix of phase differences (N x N)
    phase_differences = theta[np.newaxis, :] - theta[:, np.newaxis]  # sin(theta_j - theta_i)

    #Evaluate the sine of all differences simultaneously
    sin_diffs = np.sin(phase_differences)
    #Sum adjacent elements in the matrix
    coupling_sum = np.sum(A * sin_diffs, axis=1)   

    return omega + K/N*coupling_sum


# DOP853: higher-order RK, better for long phase integrations than RK45
solver_kw = dict(method='DOP853', rtol=1e-9, atol=1e-11)
solution_square = solve_ivp(
    kuramoto_rhs, t_span, theta_0, args=(K, N, omega, A),
    t_eval=t_eval, **solver_kw,
)

solution_er = solve_ivp(
    kuramoto_rhs, t_span, theta_0, args=(K, N, omega, A_sparse),
    t_eval=t_eval, **solver_kw,
)

sol_full = solution_square.y
sol_er = solution_er.y

z = np.mean(np.exp(1j * sol_full), axis=0)   # one complex number per time
r_full = np.abs(z)                            # sync strength over time
r_full_real = z.real
r_full_imag = z.imag

z_sparse = np.mean(np.exp(1j * sol_er), axis=0)
r_sparse = np.abs(z_sparse)
r_sparse_real = z_sparse.real
r_sparse_imag = z_sparse.imag

if bool_plots:
    plt.plot(t_eval, r_full_real, label='full_real')
    plt.plot(t_eval, r_full_imag, label='full_imag')
    plt.plot(t_eval, r_sparse_real, label='sparse_real', linestyle='--')
    plt.plot(t_eval, r_sparse_imag, label='sparse_imag', linestyle='--')
    plt.legend()
    plt.show()

    order_param_error_real = np.abs(r_full_real - r_sparse_real)
    order_param_error_imag = np.abs(r_full_imag - r_sparse_imag)    
    plt.plot(t_eval, order_param_error_real, label='order_param_error_real')
    plt.plot(t_eval, order_param_error_imag, label='order_param_error_imag')
    plt.legend()
    plt.title('sparsification order param error')
    plt.show()









# Animation: 

# side-by-side grids + order-parameter circles + phase-diff tracker
L = int(np.sqrt(N)); R = 0.35
norm = plt.Normalize(omega.min(), omega.max())
pos = lambda i: (divmod(i, L)[1], -divmod(i, L)[0])
diff = np.degrees(np.angle(np.exp(1j * (sol_full - sol_er))))  # wrap to (-180, 180]
fig, axes = plt.subplots(2, 3, figsize=(13, 8), gridspec_kw={'height_ratios': [2.2, 0.9], 'width_ratios': [1, 1, 1.4]})
ax_f, ax_s, ax_d = axes[0]
ax_zf, ax_zs, ax_rd = axes[1]
grids, z_vecs, op_lines, op_dots = [], [z, z_sparse], [], []
for ax, A, sol, title in [(ax_f, A, sol_full, 'full'), (ax_s, A_sparse, sol_er, 'sparse')]:
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title)
    ax.set_xlim(-0.6, L - 0.4); ax.set_ylim(-L + 0.4, 0.6)
    lines, balls = [], []
    for i, j in [(i, j) for i in range(N) for j in range(i + 1, N) if A[i, j]]:
        x1, y1 = pos(i); x2, y2 = pos(j); dx, dy = x2 - x1, y2 - y1; d = np.hypot(dx, dy)
        ln, = ax.plot([x1 + R * dx / d, x2 - R * dx / d], [y1 + R * dy / d, y2 - R * dy / d], lw=1, zorder=1)
        lines.append((ln, i, j))
    for i in range(N):
        r, c = divmod(i, L); th0 = sol[i, 0]
        ax.add_patch(plt.Circle((c, -r), R, fill=False, ec='k', lw=1))
        b = plt.Circle((c + R * np.cos(th0), -r + R * np.sin(th0)), 0.08, color=plt.cm.viridis(norm(omega[i])), zorder=3)
        ax.add_patch(b); balls.append(b)
    grids.append((lines, balls, sol))
for ax, z_t, title in [(ax_zf, z, 'full order param'), (ax_zs, z_sparse, 'sparse order param')]:
    ax.set_aspect('equal'); ax.set_title(title)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Re(z)'); ax.set_ylabel('Im(z)')
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec='gray', lw=1))
    ln, = ax.plot([0, z_t[0].real], [0, z_t[0].imag], 'k-', lw=2)
    dot, = ax.plot(z_t[0].real, z_t[0].imag, 'o', color='crimson', ms=6)
    op_lines.append(ln); op_dots.append(dot)
ax_rd.set_title('|Δz| (full − sparse)')
ax_rd.set_xlim(0, t_eval[-1]); ax_rd.set_ylim(0, 1.05)
ax_rd.set_xlabel('time (s)'); ax_rd.set_ylabel('|Δz|')
ax_rd.plot(t_eval, np.abs(z - z_sparse), alpha=0.5)
op_err, = ax_rd.plot([], [], 'k-', lw=2)
ax_d.set_title('phase diff (full − sparse, wrapped)')
for i in range(N): ax_d.plot(t_eval, diff[i], alpha=0.4, lw=0.8)
ax_d.set_xlabel('time (s)'); ax_d.set_ylabel('Δθ (°)'); ax_d.set_ylim(-180, 180)
vline = ax_d.axvline(t_eval[0], color='k', lw=2)
err_txt = ax_d.text(0.02, 0.95, '', transform=ax_d.transAxes, va='top')

def update(k):
    artists = []
    for lines, balls, sol in grids:
        for ln, i, j in lines:
            s = np.sin(sol[j, k] - sol[i, k])
            ln.set_linewidth(0.5 + 3 * abs(s)); ln.set_color(plt.cm.coolwarm(0.5 + 0.5 * s)); artists.append(ln)
        for i, b in enumerate(balls):
            r, c = divmod(i, L); th = sol[i, k]
            b.center = (c + R * np.cos(th), -r + R * np.sin(th)); artists.append(b)
    for ln, dot, z_t in zip(op_lines, op_dots, z_vecs):
        ln.set_data([0, z_t[k].real], [0, z_t[k].imag]); artists.append(ln)
        dot.set_data([z_t[k].real], [z_t[k].imag]); artists.append(dot)
    op_err.set_data(t_eval[:k+1], np.abs(z[:k+1] - z_sparse[:k+1])); artists.append(op_err)
    vline.set_xdata([t_eval[k], t_eval[k]])
    err_txt.set_text(f'mean |Δθ| = {np.mean(np.abs(diff[:, k])):.1f}°  |  |Δz| = {np.abs(z[k]-z_sparse[k]):.3f}')
    return artists + [vline, err_txt]

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=50)
plt.tight_layout(); plt.show()

