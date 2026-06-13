"""
Sweep natural-frequency-based sparsification schemes; goal is to beat random sampling.

    python freq_sparsify_sweep.py
    python freq_sparsify_sweep.py plots/best_sparsification_logits.pt --n-sparsify-seeds 3
    python freq_sparsify_sweep.py --test-n-ics 100 --top-n 25
    python freq_sparsify_sweep.py --fair-only
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_learned_sparsification import (
    TEST_IC_SEED_OFFSET,
    TEST_N_ICS,
    count_edges,
    load_trial,
    make_test_theta0,
    sparsify_random,
    sparsify_stochastic,
    trajectory_metrics,
)
from optimized_sparsification import N, K, Q, fc_adj, integrate_torch


@dataclass(frozen=True)
class FreqMethod:
    name: str
    mode: str  # "stochastic" or "topk"
    score_fn: object  # callable(d: ndarray) -> ndarray of nonnegative scores


def pairwise_freq_gap(omega, edge_i, edge_j):
    return np.abs(omega[edge_i] - omega[edge_j]).astype(float)


def learned_gap_median(omega, a_learned, edge_i, edge_j):
    d = pairwise_freq_gap(omega, edge_i, edge_j)
    mask = a_learned[edge_i, edge_j] > 0
    if not np.any(mask):
        return float(np.median(d))
    return float(np.median(d[mask]))


def normalize_scores(score):
    score = np.maximum(np.asarray(score, dtype=float), 0.0)
    total = score.sum()
    if total <= 0:
        return np.full_like(score, 1.0 / len(score))
    return score / total


def sparsify_topk_random(edge_i, edge_j, we, q_frac, rng):
    """uniform random edge set, fixed size q (no duplicate resampling)."""
    s = max(1, int(q_frac * len(edge_i)))
    idx = rng.choice(len(edge_i), s, replace=False)
    n = int(np.max(np.concatenate([edge_i, edge_j])) + 1)
    A = np.zeros((n, n))
    A[edge_i[idx], edge_j[idx]] = we[idx]
    A[edge_j[idx], edge_i[idx]] = we[idx]
    return A


def sparsify_topk(edge_i, edge_j, we, score, q_frac):
    """deterministic: keep top-q edges by score (no duplicate resampling)."""
    s = max(1, int(q_frac * len(edge_i)))
    idx = np.argpartition(score, -s)[-s:]
    n = int(np.max(np.concatenate([edge_i, edge_j])) + 1)
    A = np.zeros((n, n))
    A[edge_i[idx], edge_j[idx]] = we[idx]
    A[edge_j[idx], edge_i[idx]] = we[idx]
    return A


def build_method_catalog(d, d_learned_peak):
    """return list of (name, mode, score_fn) tuples."""
    methods: list[FreqMethod] = []
    d_std = max(float(np.std(d)), 1e-12)

    def add(name, mode, fn):
        methods.append(FreqMethod(name, mode, fn))

    add("random", "stochastic", lambda _d: np.ones_like(_d))
    add("topk_random", "topk_random", lambda _d: np.ones_like(_d))

    # same rule as compare_learned_sparsification.sparsify_freq (purple bar in compare plot)
    def freq_compare_score(_d, d_std=d_std):
        peak = float(np.median(_d))
        width = max(2.0 * d_std, 1e-12)
        return np.exp(-0.5 * ((_d - peak) / width) ** 2)

    add("freq_based", "stochastic", freq_compare_score)
    add("topk_freq_based", "topk", freq_compare_score)

    for q_peak in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
        peak = float(np.quantile(d, q_peak))
        for wmult in (0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
            width = wmult * d_std

            def gauss(_d, peak=peak, width=width):
                return np.exp(-0.5 * ((_d - peak) / width) ** 2)

            add(f"gauss_q{q_peak:.2f}_w{wmult:g}std", "stochastic", gauss)
            add(f"topk_gauss_q{q_peak:.2f}_w{wmult:g}std", "topk", gauss)

    for peak, tag in ((d_learned_peak, "learned"), (float(np.median(d)), "med")):
        for wmult in (1.5, 2.0, 2.5, 3.0):
            width = wmult * d_std

            def gauss(_d, peak=peak, width=width):
                return np.exp(-0.5 * ((_d - peak) / width) ** 2)

            add(f"gauss_{tag}_w{wmult:g}std", "stochastic", gauss)

    for scale in (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5):
        add(f"exp(-d/{scale})", "stochastic", lambda _d, s=scale: np.exp(-_d / s))
        add(f"d*exp(-d/{scale})", "stochastic", lambda _d, s=scale: _d * np.exp(-_d / s))
        add(f"topk_d*exp(-d/{scale})", "topk", lambda _d, s=scale: _d * np.exp(-_d / s))

    for c in (0.05, 0.1, 0.2, 0.3, 0.5):
        add(f"1/(d+{c})", "stochastic", lambda _d, c=c: 1.0 / (_d + c))

    for qlo, qhi in ((0.2, 0.6), (0.25, 0.75), (0.3, 0.7), (0.35, 0.65), (0.4, 0.8)):
        lo, hi = float(np.quantile(d, qlo)), float(np.quantile(d, qhi))
        add(
            f"band_q{qlo}-{qhi}",
            "stochastic",
            lambda _d, lo=lo, hi=hi: ((_d >= lo) & (_d <= hi)).astype(float) + 1e-12,
        )
        add(
            f"topk_band_q{qlo}-{qhi}",
            "topk",
            lambda _d, lo=lo, hi=hi: ((_d >= lo) & (_d <= hi)).astype(float) + 1e-12,
        )

    for scale in (0.85, 1.0):
        peak = float(np.median(d))
        width = 2.0 * d_std

        def mix(_d, peak=peak, width=width, scale=scale):
            bump = np.exp(-0.5 * ((_d - peak) / width) ** 2)
            return bump + 0.5 * _d * np.exp(-_d / scale)

        add(f"mix_gauss+d*exp/{scale}", "stochastic", mix)

    return methods


def sparsify_method(method, edge_i, edge_j, we, d, q_frac, rng):
    if method.name == "random":
        return sparsify_random(edge_i, edge_j, we, q_frac, rng)
    if method.mode == "topk_random":
        return sparsify_topk_random(edge_i, edge_j, we, q_frac, rng)
    score = normalize_scores(method.score_fn(d))
    if method.mode == "topk":
        return sparsify_topk(edge_i, edge_j, we, score, q_frac)
    return sparsify_stochastic(edge_i, edge_j, we, score, q_frac, rng)


FAIR_COMPARE_OFFSETS = {
    "random": 3000,
    "topk_random": 3000,
    "freq_based": 4000,
    "topk_freq_based": 4000,
}

FAIR_COMPARE_NAMES = (
    "random",
    "topk_random",
    "freq_based",
    "topk_freq_based",
)


def print_fair_comparison(rows):
    """same edge budget mechanics: stochastic vs topk, uniform vs gauss scores."""
    by_name = {r["name"]: r for r in rows}
    missing = [n for n in FAIR_COMPARE_NAMES if n not in by_name]
    if missing:
        print(f"\n(fair comparison skipped — missing {missing})")
        return None

    fair_rows = [by_name[n] for n in FAIR_COMPARE_NAMES]
    topk_rand_test = by_name["topk_random"]["test_circ"]

    print("\nfair baseline comparison (compare-plot freq_based vs topk_random, stochastic vs topk):")
    print(f"{'method':<32} {'mode':<12} {'train':>8} {'test':>8} {'edges':>6}  vs topk_rand")
    print("-" * 82)
    for row in fair_rows:
        delta = row["test_circ"] - topk_rand_test
        tag = "BETTER" if delta < -1e-4 else ("worse" if delta > 1e-4 else "tie")
        print(
            f"{row['name']:<32} {row['mode']:<12} "
            f"{row['train_circ']:8.4f} {row['test_circ']:8.4f} "
            f"{row['n_edges_mean']:6.0f}  {delta:+.4f} ({tag})"
        )
    return fair_rows


def _save_fair_comparison_plot(fair_rows, trial, path):
    names = [r["name"] for r in fair_rows]
    fig, (ax_test, ax_train) = plt.subplots(1, 2, figsize=(10, 3.5))
    x = np.arange(len(names))
    colors = ["C0", "C1", "C4", "C5"][: len(names)]
    if len(names) > len(colors):
        colors = [f"C{i % 10}" for i in range(len(names))]

    ax_test.bar(x, [r["test_circ"] for r in fair_rows], color=colors, edgecolor="0.2", linewidth=0.4)
    ax_test.set_xticks(x)
    ax_test.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax_test.set_ylabel("test circ loss")
    ax_test.set_title("out-of-training ICs")

    ax_train.bar(x, [r["train_circ"] for r in fair_rows], color=colors, edgecolor="0.2", linewidth=0.4)
    ax_train.set_xticks(x)
    ax_train.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax_train.set_ylabel("train circ loss")
    ax_train.set_title(f"training ICs ({trial['n_ics']})")

    fig.suptitle(
        f"fair sparsify comparison | N={N} q={Q} | seed {trial['seed']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_method(
    method, edge_i, edge_j, we, d, q_frac, omega, t,
    seed, offset, n_seeds, theta_batches, targets,
):
    edge_counts = []
    metrics_by_split = {label: [] for label, _ in theta_batches}
    omega_t = torch.tensor(omega, dtype=torch.float64)
    t_t = torch.tensor(t, dtype=torch.float64)
    a_full_t = torch.tensor(fc_adj(N), dtype=torch.float64)

    for si in range(n_seeds):
        rng = np.random.default_rng(seed + offset + si)
        A = sparsify_method(method, edge_i, edge_j, we, d, q_frac, rng)
        edge_counts.append(count_edges(A))
        a_t = torch.tensor(A, dtype=torch.float64)
        for label, theta_t in theta_batches:
            with torch.no_grad():
                pred = integrate_torch(theta_t, omega_t, a_t, K, t_t).numpy()
            metrics_by_split[label].append(trajectory_metrics(pred, targets[label]))

    row = {
        "name": method.name,
        "mode": method.mode,
        "n_edges_mean": float(np.mean(edge_counts)),
        "n_edges_std": float(np.std(edge_counts)) if n_seeds > 1 else 0.0,
    }
    for label, ms in metrics_by_split.items():
        row[f"{label}_circ"] = float(np.mean([m["circ_loss_mean"] for m in ms]))
        row[f"{label}_circ_std"] = float(np.std([m["circ_loss_mean"] for m in ms])) if n_seeds > 1 else 0.0
        row[f"{label}_dr"] = float(np.mean([m["dr_mean"] for m in ms]))
        row[f"{label}_dpsi"] = float(np.mean([m["dpsi_mean"] for m in ms]))
    return row


def precompute_targets(theta_batches, omega, t):
    omega_t = torch.tensor(omega, dtype=torch.float64)
    t_t = torch.tensor(t, dtype=torch.float64)
    a_full_t = torch.tensor(fc_adj(N), dtype=torch.float64)
    targets = {}
    for label, theta_t in theta_batches:
        with torch.no_grad():
            targets[label] = integrate_torch(theta_t, omega_t, a_full_t, K, t_t).numpy()
    return targets


def run_fair_only(
    ckpt_path=Path("plots/best_sparsification_logits.pt"),
    *,
    n_sparsify_seeds=10,
    test_n_ics=TEST_N_ICS,
):
    trial = load_trial(ckpt_path)
    edge_i, edge_j, we, _ = trial["er_data"]
    d = pairwise_freq_gap(trial["omega"], edge_i, edge_j)
    d_learned = learned_gap_median(trial["omega"], trial["a_learned"], edge_i, edge_j)
    catalog = {m.name: m for m in build_method_catalog(d, d_learned)}
    methods = [catalog[n] for n in FAIR_COMPARE_NAMES]

    test_theta0 = make_test_theta0(trial["seed"], test_n_ics)
    theta_batches = [
        ("train", torch.tensor(trial["theta0_batch"], dtype=torch.float64)),
        ("test", torch.tensor(test_theta0, dtype=torch.float64)),
    ]
    targets = precompute_targets(theta_batches, trial["omega"], trial["t"])

    print(f"fair-only comparison | seed {trial['seed']} | sparsify seeds {n_sparsify_seeds}")
    rows = []
    for mi, method in enumerate(methods):
        offset = FAIR_COMPARE_OFFSETS.get(method.name, 50_000 + 17 * mi)
        rows.append(evaluate_method(
            method, edge_i, edge_j, we, d, trial["q"], trial["omega"], trial["t"],
            trial["seed"], offset=offset, n_seeds=n_sparsify_seeds,
            theta_batches=theta_batches, targets=targets,
        ))
    fair_rows = print_fair_comparison(rows)
    if fair_rows is not None:
        out = Path("plots") / f"freq_sparsify_fair_baseline_n={N}_q={Q}.png"
        out.parent.mkdir(exist_ok=True)
        _save_fair_comparison_plot(fair_rows, trial, out)
        print(f"saved {out}")
    return rows


def run_sweep(
    ckpt_path=Path("plots/best_sparsification_logits.pt"),
    *,
    n_sparsify_seeds=10,
    test_n_ics=TEST_N_ICS,
    top_n=20,
    save_plot=True,
):
    trial = load_trial(ckpt_path)

    edge_i, edge_j, we, _ = trial["er_data"]
    d = pairwise_freq_gap(trial["omega"], edge_i, edge_j)
    d_learned = learned_gap_median(trial["omega"], trial["a_learned"], edge_i, edge_j)
    q_frac = trial["q"]
    seed = trial["seed"]

    methods = build_method_catalog(d, d_learned)
    test_theta0 = make_test_theta0(seed, test_n_ics)
    theta_batches = [
        ("train", torch.tensor(trial["theta0_batch"], dtype=torch.float64)),
        ("test", torch.tensor(test_theta0, dtype=torch.float64)),
    ]
    targets = precompute_targets(theta_batches, trial["omega"], trial["t"])

    print(f"checkpoint: {ckpt_path}")
    print(
        f"seed {seed} | N={N} K={K:.2g} q={Q} | "
        f"train ICs {trial['n_ics']} | test ICs {test_n_ics} | "
        f"methods {len(methods)} | sparsify seeds {n_sparsify_seeds}"
    )
    print(f"|Δω| median={np.median(d):.3f} learned-edge median={d_learned:.3f}\n")

    rows = []
    for mi, method in enumerate(methods):
        row = evaluate_method(
            method, edge_i, edge_j, we, d, q_frac, trial["omega"], trial["t"],
            seed, offset=10_000 + 17 * mi, n_seeds=n_sparsify_seeds,
            theta_batches=theta_batches, targets=targets,
        )
        rows.append(row)
        if (mi + 1) % 25 == 0 or mi + 1 == len(methods):
            print(f"  evaluated {mi + 1}/{len(methods)} ...")

    random_test = next(r["test_circ"] for r in rows if r["name"] == "random")
    random_train = next(r["train_circ"] for r in rows if r["name"] == "random")
    for row in rows:
        row["beats_random_test"] = row["test_circ"] < random_test
        row["beats_random_train"] = row["train_circ"] < random_train
        row["delta_test_vs_random"] = row["test_circ"] - random_test

    rows_by_test = sorted(rows, key=lambda r: r["test_circ"])
    n_beat = sum(r["beats_random_test"] for r in rows if r["name"] != "random")

    print(f"\nrandom baseline  train={random_train:.4f}  test={random_test:.4f}")
    print(f"{n_beat}/{len(methods)-1} freq methods beat random on test circ loss\n")

    print(f"top {top_n} on test circ loss:")
    print(f"{'rank':>4} {'method':<32} {'mode':<10} {'train':>8} {'test':>8} {'Δtest':>8} {'edges':>6} beat")
    print("-" * 88)
    for rank, row in enumerate(rows_by_test[:top_n], 1):
        beat = "yes" if row["beats_random_test"] else "no"
        print(
            f"{rank:4d} {row['name']:<32} {row['mode']:<10} "
            f"{row['train_circ']:8.4f} {row['test_circ']:8.4f} "
            f"{row['delta_test_vs_random']:+8.4f} {row['n_edges_mean']:6.0f} {beat:>4}"
        )

    fair_rows = print_fair_comparison(rows)

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    if fair_rows is not None:
        fair_plot = out_dir / f"freq_sparsify_fair_baseline_n={N}_q={Q}.png"
        _save_fair_comparison_plot(fair_rows, trial, fair_plot)
        print(f"saved {fair_plot}")

    npz_path = out_dir / f"freq_sparsify_sweep_n={N}_q={Q}.npz"
    np.savez(
        npz_path,
        rows=np.array(rows, dtype=object),
        random_test=random_test,
        random_train=random_train,
        seed=seed,
        d_median=np.median(d),
        d_learned_median=d_learned,
        n_methods=len(methods),
    )
    print(f"\nsaved {npz_path}")

    if save_plot:
        plot_path = out_dir / f"freq_sparsify_sweep_n={N}_q={Q}.png"
        _save_summary_plot(rows_by_test, random_test, trial, n_beat, plot_path, top_n=min(top_n, 15))
        print(f"saved {plot_path}")

    best = rows_by_test[0]
    if best["name"] != "random" and best["beats_random_test"]:
        print(f"\nbest: {best['name']} ({best['mode']}) test={best['test_circ']:.4f}")
    elif best["name"] == "random":
        print("\nbest on test is still random — try more seeds or checkpoint.")
    return rows_by_test, random_test


def _save_summary_plot(rows_sorted, random_test, trial, n_beat, path, top_n=15):
    # show top_n freq methods plus random reference line
    show = [r for r in rows_sorted if r["name"] != "random"][:top_n]
    if not any(r["name"] == "random" for r in show):
        random_row = next(r for r in rows_sorted if r["name"] == "random")
        show = [random_row] + show[: top_n - 1]

    names = [r["name"] for r in show]
    test_vals = [r["test_circ"] for r in show]
    train_vals = [r["train_circ"] for r in show]
    colors = ["C3" if r["beats_random_test"] else "0.75" for r in show]

    fig, (ax_test, ax_train) = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * len(names))), sharey=True)
    y = np.arange(len(names))

    ax_test.barh(y, test_vals, color=colors, edgecolor="0.2", linewidth=0.4)
    ax_test.axvline(random_test, color="C0", ls="--", lw=1.2, label="random test")
    ax_test.set_yticks(y)
    ax_test.set_yticklabels(names, fontsize=7)
    ax_test.set_xlabel("test circ loss")
    ax_test.invert_yaxis()
    ax_test.legend(fontsize=8)

    ax_train.barh(y, train_vals, color=colors, edgecolor="0.2", linewidth=0.4)
    random_train = next(r["train_circ"] for r in rows_sorted if r["name"] == "random")
    ax_train.axvline(random_train, color="C0", ls="--", lw=1.2, label="random train")
    ax_train.set_xlabel("train circ loss")
    ax_train.legend(fontsize=8)

    fig.suptitle(
        f"freq sparsify sweep | N={N} q={Q} | seed {trial['seed']} | "
        f"{n_beat} methods beat random (test)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", nargs="?", default="plots/best_sparsification_logits.pt", type=Path,
    )
    parser.add_argument("--n-sparsify-seeds", type=int, default=10)
    parser.add_argument("--test-n-ics", type=int, default=TEST_N_ICS)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--fair-only", action="store_true",
        help="only run fair baseline (random, topk_random, freq_based, topk_freq_based)",
    )
    args = parser.parse_args()
    if args.fair_only:
        run_fair_only(
            args.checkpoint,
            n_sparsify_seeds=args.n_sparsify_seeds,
            test_n_ics=args.test_n_ics,
        )
        return
    run_sweep(
        args.checkpoint,
        n_sparsify_seeds=args.n_sparsify_seeds,
        test_n_ics=args.test_n_ics,
        top_n=args.top_n,
        save_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
