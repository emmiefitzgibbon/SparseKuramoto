"""
sandbox — zoomed arc view: full vs ER vs weight-based sparsified ring graphs.

Run: python plot_er_zoom.py
     python plot_er_zoom.py --N 80 --k 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ring_basin_sweep import (  # noqa: E402
    HE_WEIGHT_STD,
    SPARSIFY_Q,
    generate_ring_matrix,
    precompute_er,
    sparsify_er,
)

WEIGHT_MODES = (
    ('homogeneous', dict(heterogeneous=False, falloff=False)),
    ('heterogeneous', dict(heterogeneous=True, falloff=False)),
    ('falloff', dict(heterogeneous=False, falloff=True)),
)


def arc_window(n, n_arc, center=None):
    center = n // 2 if center is None else int(center)
    half = n_arc // 2
    return [(center - half + i) % n for i in range(n_arc)]


def induced_subgraph(A, nodes):
    n_local = len(nodes)
    sub = np.zeros((n_local, n_local))
    for a in range(n_local):
        for b in range(a + 1, n_local):
            w = A[nodes[a], nodes[b]]
            if w:
                sub[a, b] = w
                sub[b, a] = w
    return sub


def arc_xy(n_local, arc_deg=155.0, radius=1.15):
    span = np.deg2rad(arc_deg)
    angles = np.linspace(-span / 2, span / 2, n_local)
    return radius * np.cos(angles), radius * np.sin(angles)


LW_MIN, LW_MAX = 0.5, 4.0
CMAP = plt.cm.plasma
ER_COLOR = '#e67e22'
WT_COLOR = '#27ae60'


def weight_range_row(matrices, nodes):
    """per-row scale from all panels in the row."""
    weights = []
    for A in matrices:
        sub = induced_subgraph(A, nodes)
        w = sub[np.triu(sub, 1) > 0]
        if w.size:
            weights.append(w)
    if not weights:
        return 1.0, 1.0
    all_w = np.concatenate(weights)
    return float(all_w.min()), float(all_w.max())


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


def norm_weight(w, w_lo, w_hi):
    if w_hi <= w_lo:
        return 1.0
    return (w - w_lo) / (w_hi - w_lo)


def edge_segments(xy, A, w_lo, w_hi, w_floor=0.0):
    """return segments sorted light→heavy so thick edges draw on top."""
    n = A.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            w = A[i, j]
            if w <= w_floor:
                continue
            edges.append((
                w,
                [(xy[0][i], xy[1][i]), (xy[0][j], xy[1][j])],
            ))
    edges.sort(key=lambda item: item[0])
    segs, widths, colors = [], [], []
    for w, seg in edges:
        t = norm_weight(w, w_lo, w_hi)
        segs.append(seg)
        widths.append(LW_MIN + t * (LW_MAX - LW_MIN))
        colors.append(CMAP(0.2 + 0.75 * t))
    return segs, widths, colors


def draw_panel(ax, A, nodes, *, title, w_lo, w_hi, arc_deg=155.0):
    sub = induced_subgraph(A, nodes)
    x, y = arc_xy(len(nodes), arc_deg=arc_deg)
    segs, widths, colors = edge_segments((x, y), sub, w_lo, w_hi)
    if segs:
        for seg, lw, color in zip(segs, widths, colors):
            ax.plot(
                [seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
                color=color, linewidth=lw, alpha=0.9,
                solid_capstyle='round', zorder=1,
            )
    ax.scatter(x, y, s=90, c='white', edgecolors='k', linewidths=1.0, zorder=3)
    n_edges = int(np.count_nonzero(np.triu(sub, 1)))
    ax.set_title(f'{title}\n{len(nodes)} nodes, {n_edges} edges', fontsize=9)
    pad = 0.22
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + 0.12)
    ax.set_aspect('equal')
    ax.axis('off')


def add_row_weight_scale(ax, w_lo, w_hi):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.05, 0.97, 'weight →\ncolor & thickness', fontsize=7, va='top', ha='left')
    if w_hi <= w_lo * (1 + 1e-9):
        samples = [(w_hi, 'uniform')]
    else:
        w_mid = 0.5 * (w_lo + w_hi)
        samples = [(w_lo, f'{w_lo:.2g}'), (w_mid, f'{w_mid:.2g}'), (w_hi, f'{w_hi:.2g}')]
    y_slots = np.linspace(0.72, 0.28, len(samples))
    x0, x1 = 0.08, 0.55
    for (w, label), y in zip(samples, y_slots):
        t = norm_weight(w, w_lo, w_hi)
        lw = LW_MIN + t * (LW_MAX - LW_MIN)
        ax.plot([x0, x1], [y, y], color=CMAP(0.2 + 0.75 * t), linewidth=lw, alpha=0.9,
                solid_capstyle='round', clip_on=False, transform=ax.transAxes)
        ax.text(x1 + 0.05, y, label, fontsize=7, va='center', transform=ax.transAxes)


def ring_distance(i, j, n):
    return int(np.minimum(np.abs(i - j), n - np.abs(i - j)))


def sparse_edge_stats(A, n):
    ei, ej = np.where(np.triu(A, 1))
    w = A[ei, ej].astype(float)
    d = np.array([ring_distance(ei[t], ej[t], n) for t in range(len(ei))])
    return d, w


def distance_fraction(d, k):
    if d.size == 0:
        return np.zeros(k)
    counts = np.bincount(d, minlength=k + 1)[1:k + 1]
    return counts / counts.sum()


def weight_density(w, bins):
    if w.size == 0:
        return np.zeros(len(bins) - 1)
    hist, _ = np.histogram(w, bins=bins, density=True)
    return hist


def collect_sparse_stats(n, k, base_seed, weight_std, n_seeds):
    """average distance/weight distributions over sparsify seeds."""
    dist_vals = np.arange(1, k + 1)
    full_d_frac = np.ones(k) / k
    out = {}

    for label, mode in WEIGHT_MODES:
        rng_w = np.random.default_rng(base_seed + 7)
        A_full = generate_ring_matrix(
            n, k, rng=rng_w, weight_std=weight_std, **mode,
        )
        er_i, er_j, er_w, er_pe = precompute_er(A_full)

        er_d, wt_d, er_ws, wt_ws = [], [], [], []
        for s in range(n_seeds):
            rng_er = np.random.default_rng(base_seed + 1000 + s)
            rng_wt = np.random.default_rng(base_seed + 2000 + s)
            A_er = sparsify_er(er_i, er_j, er_w, er_pe, SPARSIFY_Q, rng_er)
            A_wt = sparsify_weight(er_i, er_j, er_w, SPARSIFY_Q, rng_wt)
            d_er, w_er = sparse_edge_stats(A_er, n)
            d_wt, w_wt = sparse_edge_stats(A_wt, n)
            er_d.append(distance_fraction(d_er, k))
            wt_d.append(distance_fraction(d_wt, k))
            er_ws.append(w_er)
            wt_ws.append(w_wt)

        all_w = np.concatenate([er_w, *er_ws, *wt_ws])
        w_min, w_max = float(all_w.min()), float(all_w.max())
        if w_max <= w_min * (1 + 1e-9):
            w_max = w_min + 1.0
        weight_bins = np.linspace(w_min, w_max, 21)
        bin_centers = 0.5 * (weight_bins[:-1] + weight_bins[1:])
        full_w_density = weight_density(er_w, weight_bins)
        er_wd = [weight_density(w, weight_bins) for w in er_ws]
        wt_wd = [weight_density(w, weight_bins) for w in wt_ws]

        er_d_arr, wt_d_arr = np.array(er_d), np.array(wt_d)
        er_w_arr, wt_w_arr = np.array(er_wd), np.array(wt_wd)
        out[label] = dict(
            dist_vals=dist_vals,
            full_d_frac=full_d_frac,
            full_w_density=full_w_density,
            bin_centers=bin_centers,
            er_d_mean=er_d_arr.mean(0), er_d_std=er_d_arr.std(0),
            wt_d_mean=wt_d_arr.mean(0), wt_d_std=wt_d_arr.std(0),
            er_w_mean=er_w_arr.mean(0), er_w_std=er_w_arr.std(0),
            wt_w_mean=wt_w_arr.mean(0), wt_w_std=wt_w_arr.std(0),
        )
    return out


def _shade_band(ax, x, mean, std, *, color, label, zorder, linestyle='-'):
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0, zorder=zorder)
    ax.plot(x, mean, color=color, lw=2, linestyle=linestyle, label=label, zorder=zorder + 1)


def plot_sparse_stats(
    n=400,
    k=40,
    base_seed=42,
    weight_std=HE_WEIGHT_STD,
    n_seeds=10,
    out_dir=None,
):
    out_dir = ROOT / 'plots' if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = collect_sparse_stats(n, k, base_seed, weight_std, n_seeds)

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), squeeze=False)
    fig.suptitle(
        f'sparsification edge statistics (mean ± std over {n_seeds} seeds, q={SPARSIFY_Q})\n'
        f'N={n}  k={k}  k/N={k / n:.3f}',
        fontsize=12,
        y=0.98,
    )

    for row, (label, s) in enumerate(stats.items()):
        ax_d, ax_w = axes[row, 0], axes[row, 1]
        x = s['dist_vals']

        ax_d.plot(x, s['full_d_frac'], 'k--', lw=1.5, alpha=0.75, label='full (uniform)', zorder=1)
        _shade_band(ax_d, x, s['wt_d_mean'], s['wt_d_std'], color=WT_COLOR, label='weight sparse', zorder=2)
        _shade_band(ax_d, x, s['er_d_mean'], s['er_d_std'], color=ER_COLOR, label='ER sparse',
                    zorder=5, linestyle='--')
        ax_d.set_xlim(0.5, k + 0.5)
        ax_d.set_xticks(x[::max(1, k // 10)])
        ax_d.set_ylim(0, None)
        ax_d.set_xlabel('ring distance d')
        ax_d.set_ylabel('fraction of kept edges')
        ax_d.set_title(f'{label}: connection type')
        ax_d.grid(True, alpha=0.25)
        ax_d.legend(fontsize=8)

        bx = s['bin_centers']
        ax_w.plot(bx, s['full_w_density'], 'k--', lw=1.5, alpha=0.75, label='full graph', zorder=1)
        _shade_band(ax_w, bx, s['wt_w_mean'], s['wt_w_std'], color=WT_COLOR, label='weight sparse', zorder=2)
        _shade_band(ax_w, bx, s['er_w_mean'], s['er_w_std'], color=ER_COLOR, label='ER sparse',
                    zorder=5, linestyle='--')
        ax_w.set_xlabel('edge weight')
        ax_w.set_ylabel('density')
        ax_w.set_title(f'{label}: edge weight')
        ax_w.grid(True, alpha=0.25)
        ax_w.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / f'er_zoom_stats_n={n}_k={k}_seeds{n_seeds}.png'
    fig.savefig(path, dpi=140, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print(f'saved {path}')
    return path


def build_graphs(n, k, base_seed, weight_std):
    graphs = {}
    for label, mode in WEIGHT_MODES:
        rng_w = np.random.default_rng(base_seed + 7)
        A_full = generate_ring_matrix(
            n, k, rng=rng_w, weight_std=weight_std, **mode,
        )
        er_i, er_j, er_w, er_pe = precompute_er(A_full)
        rng_er = np.random.default_rng(base_seed + 1000)
        rng_wt = np.random.default_rng(base_seed + 2000)
        A_er = sparsify_er(er_i, er_j, er_w, er_pe, SPARSIFY_Q, rng_er)
        A_wt = sparsify_weight(er_i, er_j, er_w, SPARSIFY_Q, rng_wt)
        graphs[label] = (A_full, A_er, A_wt)
    return graphs


def plot_er_zoom(
    n=400,
    k=40,
    n_arc=15,
    center=None,
    base_seed=42,
    weight_std=HE_WEIGHT_STD,
    out_dir=None,
):
    out_dir = ROOT / 'plots' if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = arc_window(n, n_arc, center=center)
    graphs = build_graphs(n, k, base_seed, weight_std)

    n_modes = len(WEIGHT_MODES)
    fig, axes = plt.subplots(
        n_modes, 4, figsize=(13.5, 3.6 * n_modes), squeeze=False,
        gridspec_kw={'width_ratios': [0.5, 1, 1, 1]},
    )
    fig.suptitle(
        f'full vs ER vs weight-based sparsified ring (q={SPARSIFY_Q})  '
        f'N={n}  k={k}  k/N={k / n:.3f}  arc nodes {nodes[0]}…{nodes[-1]}  '
        f'(per-row weight scale)',
        fontsize=11,
        y=0.995,
    )
    col_titles = ('', 'full', 'ER sparse', 'weight sparse')
    for col, title in enumerate(col_titles[1:], start=1):
        axes[0, col].annotate(
            title, xy=(0.5, 1.12), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
        )

    for row, (label, (A_full, A_er, A_wt)) in enumerate(graphs.items()):
        w_lo, w_hi = weight_range_row((A_full, A_er, A_wt), nodes)
        add_row_weight_scale(axes[row, 0], w_lo, w_hi)
        draw_panel(axes[row, 1], A_full, nodes, title=f'{label} — full', w_lo=w_lo, w_hi=w_hi)
        draw_panel(axes[row, 2], A_er, nodes, title=f'{label} — ER sparse', w_lo=w_lo, w_hi=w_hi)
        draw_panel(axes[row, 3], A_wt, nodes, title=f'{label} — weight sparse', w_lo=w_lo, w_hi=w_hi)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f'er_zoom_n={n}_k={k}_arc{n_arc}.png'
    fig.savefig(path, dpi=140, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print(f'saved {path}')
    return path


def main():
    p = argparse.ArgumentParser(description='zoomed full vs ER ring arc plots')
    p.add_argument('--N', type=int, default=400)
    p.add_argument('--k', type=int, default=None, help='neighbor shells (default: round(0.1*N))')
    p.add_argument('--n-arc', type=int, default=15, help='nodes shown along the arc')
    p.add_argument('--center', type=int, default=None, help='ring index at arc center')
    p.add_argument('--base-seed', type=int, default=42)
    p.add_argument('--weight-std', type=float, default=HE_WEIGHT_STD)
    p.add_argument('--out-dir', type=Path, default=None)
    p.add_argument('--n-seeds', type=int, default=10, help='seeds for stats averaging')
    p.add_argument('--stats-only', action='store_true', help='only generate stats figure')
    p.add_argument('--graphs-only', action='store_true', help='only generate arc graph figure')
    args = p.parse_args()
    k = round(0.1 * args.N) if args.k is None else args.k
    kw = dict(
        n=args.N, k=k, base_seed=args.base_seed,
        weight_std=args.weight_std, out_dir=args.out_dir,
    )
    if not args.stats_only:
        plot_er_zoom(n_arc=args.n_arc, center=args.center, **kw)
    if not args.graphs_only:
        plot_sparse_stats(n_seeds=args.n_seeds, **kw)


if __name__ == '__main__':
    main()
