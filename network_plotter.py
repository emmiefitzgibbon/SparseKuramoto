"""force-directed network plot with optional alignment bias (from Mikaberidze et al.)."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

__all__ = ["compute_layout", "plot_network"]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _graph_arrays(ww, A):
    ww_np = _to_numpy(ww)
    A_np = _to_numpy(A)
    n = A_np.shape[0]
    if A_np.shape[1] != n:
        raise ValueError("adjacency matrix must be square")
    rows, cols = np.triu_indices(n, 1)
    mask = A_np[rows, cols] > 1e-12
    edges = np.vstack((rows[mask], cols[mask])).T
    weights = A_np[edges[:, 0], edges[:, 1]] if edges.size else np.array([])
    return ww_np, A_np, n, edges, weights


def _fruchterman_reingold_step(
    pos, n, edges, weights, k, temperature,
    alignment, lambda_align,
):
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(delta, axis=2) + 1e-9
    repulsion = (k**2 / dist**2)[..., None] * delta
    disp = np.sum(repulsion, axis=1)
    if edges.size:
        diff_e = pos[edges[:, 0]] - pos[edges[:, 1]]
        dist_e = np.linalg.norm(diff_e, axis=1, keepdims=True) + 1e-9
        attr = (weights[:, None] * (dist_e[:, 0] ** 2 / k)[:, None]) * diff_e / dist_e
        np.add.at(disp, edges[:, 0], -attr)
        np.add.at(disp, edges[:, 1], attr)
    if alignment == "vertical":
        disp[:, 0] += -2.0 * lambda_align * pos[:, 0]
    elif alignment == "horizontal":
        disp[:, 1] += -2.0 * lambda_align * pos[:, 1]
    length = np.linalg.norm(disp, axis=1) + 1e-9
    return disp / length[:, None] * np.minimum(length, temperature)[:, None]


def compute_layout(
    ww,
    A,
    *,
    alignment: Optional[Literal["vertical", "horizontal"]] = None,
    alignment_strength: float = 0.1,
    initial_pos: Optional[np.ndarray] = None,
    iterations: int = 150,
    seed: Optional[int] = None,
) -> np.ndarray:
    """run force simulation and return node positions (n, 2)."""
    rng = np.random.default_rng(seed)
    _, _, n, edges, weights = _graph_arrays(ww, A)
    k = np.sqrt(1.0 / n)
    if initial_pos is None:
        pos = rng.standard_normal((n, 2)) * 0.01
    else:
        pos = np.asarray(initial_pos, dtype=float)
        if pos.shape != (n, 2):
            raise ValueError("initial_pos must have shape (n, 2)")
    t0 = k
    for t in range(iterations):
        temperature = t0 * (1 - t / iterations)
        pos += _fruchterman_reingold_step(
            pos, n, edges, weights, k, temperature, alignment, alignment_strength,
        )
    return pos


def _edge_linewidths(weights, edge_thickness, max_weight, *, uniform=False, min_lw=0.2, power=1.0):
    if not weights.size:
        return np.array([])
    if uniform:
        return np.full(len(weights), edge_thickness)
    rel = weights / max_weight
    return np.maximum(edge_thickness * rel**power, min_lw)


def plot_network(
    ww,
    A,
    *,
    node_radius: Optional[float] = 20,
    edge_thickness: float = 2.0,
    edge_alpha: float = 1.0,
    uniform_edges: bool = False,
    min_linewidth: float = 0.2,
    edge_width_power: float = 1.0,
    alignment: Optional[Literal["vertical", "horizontal"]] = None,
    alignment_strength: float = 0.1,
    initial_pos: Optional[np.ndarray] = None,
    iterations: int = 150,
    ax: Optional[plt.Axes] = None,
    seed: Optional[int] = None,
    add_colorbar: bool = True,
    color_norm: Optional[Tuple[float, float]] = None,
):
    """visualize a weighted network; node color = ww, edge width = weight."""
    ww_np, A_np, n, edges, weights = _graph_arrays(ww, A)
    if initial_pos is None and iterations > 0:
        pos = compute_layout(
            ww_np, A_np,
            alignment=alignment,
            alignment_strength=alignment_strength,
            iterations=iterations,
            seed=seed,
        )
    elif initial_pos is not None:
        pos = np.asarray(initial_pos, dtype=float)
        if pos.shape != (n, 2):
            raise ValueError("initial_pos must have shape (n, 2)")
    else:
        raise ValueError("provide initial_pos when iterations=0")

    if ax is None:
        ax = plt.gca()
    fig = ax.figure
    cmap = plt.cm.RdBu
    if color_norm is None:
        norm = plt.Normalize(vmin=ww_np.min(), vmax=ww_np.max())
    else:
        norm = plt.Normalize(vmin=color_norm[0], vmax=color_norm[1])
    node_colors = cmap(norm(ww_np))

    ax.cla()
    x_coords, y_coords = pos[:, 0], pos[:, 1]
    max_weight = weights.max() if weights.size else 1.0
    line_widths = _edge_linewidths(
        weights, edge_thickness, max_weight,
        uniform=uniform_edges, min_lw=min_linewidth, power=edge_width_power,
    )
    if edges.size:
        segs = np.stack(
            [np.column_stack([x_coords[edges[:, 0]], y_coords[edges[:, 0]]]),
             np.column_stack([x_coords[edges[:, 1]], y_coords[edges[:, 1]]])],
            axis=1,
        )
        ax.add_collection(LineCollection(
            segs, linewidths=line_widths, colors="black",
            alpha=edge_alpha, zorder=1,
        ))
    node_size = node_radius if node_radius is not None else max(4.0, 900.0 / n)
    ax.scatter(
        x_coords, y_coords, c=node_colors, s=node_size,
        edgecolors="k", linewidths=0.3, zorder=3,
    )
    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.75, label=r"$\omega_i$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
