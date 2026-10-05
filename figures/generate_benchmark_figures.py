"""TVL-D vs. six baselines (xLSTMAD, KAN-AD, ALoRa, CANDI, TopoGDN, SensitiveHUE) across 
4 datasets (SMD, MSL, SMAP, PSM). Built entirely from two small, already-computed CSVs.

  - Critical Difference diagram (PR-AUC): ranks the 7 methods per dataset, tests
    whether average-rank differences are statistically real (Friedman + Nemenyi,
    Demsar 2006), draws the standard diagram.
  - Heatmap, all six reported metrics at once (PR-AUC, AUC-ROC, AUC-PR_T, F1,
    precision, recall) for all 4 datasets x 7 methods in one grid.
  - Line plot: PR-AUC and F1 across the four datasets, one line per method.
 
Inputs (in data/, both already bundled with this package):
  - seven_method_combined.csv: one row per (dataset, method) with pr_auc, auc_roc,
    auc_pr_t, f1, p, r

To rebuild these from scratch instead of using the bundled copies: seven_method_combined.csv
is each method's own per-dataset-averaged results merged into one table (one row per
(dataset, method) pair, 4 datasets x 7 methods = 28 rows);

Usage:
    python generate_benchmark_figures.py
Requires scikit-posthocs in addition to this package's base requirements
(pip install scikit-posthocs).
Writes: cd_diagram_per_dataset.png, heatmap_all_metrics.png,
        line_plot_headline_metrics.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

try:
    import scikit_posthocs as sp
except ImportError:
    print("Missing dependency: pip install scikit-posthocs")
    sys.exit(1)

HERE = Path(__file__).resolve().parent
COMBINED_CSV = HERE / "data" / "seven_method_combined.csv"

METHOD_ORDER = ["SensitiveHUE", "TopoGDN", "KAN-AD", "xLSTMAD", "ALoRa", "CANDI", "TVL-D"]
METHOD_YEAR = {"SensitiveHUE": "2024", "TopoGDN": "2024", "KAN-AD": "2024",
               "xLSTMAD": "2025", "ALoRa": "2026", "CANDI": "2026", "TVL-D": "ours"}
COLORS = {
    "SensitiveHUE": "#e87ba4", "TopoGDN": "#008300", "KAN-AD": "#eda100",
    "xLSTMAD": "#4a3aa7", "ALoRa": "#eb6834", "CANDI": "#1baf7a", "TVL-D": "#2a78d6",
}
DATASETS = ["SMD", "MSL", "SMAP", "PSM"]

# Nemenyi q_alpha table, alpha = 0.05 (Demsar 2006, Table 5(b))
Q_ALPHA_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164}


def critical_difference(k, N, q_table=Q_ALPHA_005):
    return q_table[k] * np.sqrt(k * (k + 1) / (6.0 * N))


def form_cliques(vals, cd):
    """Maximal groups of consecutive (by rank) methods whose rank range is < cd."""
    n = len(vals)
    cliques = []
    for i in range(n):
        j = i
        while j + 1 < n and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i:
            cliques.append((i, j))
    return [c for c in cliques if not any(c != c2 and c[0] >= c2[0] and c[1] <= c2[1] for c2 in cliques)]


def plot_cd_diagram(avg_ranks: dict, cd: float, k: int, title: str, subtitle: str, out_path: Path):
    names = sorted(avg_ranks, key=lambda m: avg_ranks[m])
    vals = [avg_ranks[n] for n in names]
    n = len(names)
    cliques = form_cliques(vals, cd)

    lo, hi = 1, k
    half = (n + 1) // 2
    left_names, right_names = names[:half], names[half:]
    max_rows = max(len(left_names), len(right_names))

    row_h, stem0, bar_h = 0.34, 0.5, 0.20
    axis_y = bar_h * (len(cliques) + 1)
    top_y = axis_y + 0.65
    bottom_y = -(stem0 + max_rows * row_h + 0.25)

    fig, ax = plt.subplots(figsize=(9.2, (top_y - bottom_y) * 1.15 + 0.6))
    ax.set_xlim(lo - 0.65, hi + 0.65)
    ax.set_ylim(bottom_y, top_y)

    ax.plot([lo, hi], [axis_y, axis_y], color="#333333", lw=1.3, zorder=3)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [axis_y - 0.045, axis_y + 0.045], color="#333333", lw=1.2, zorder=3)
        ax.text(r, axis_y + 0.13, str(r), ha="center", va="bottom", fontsize=9.5, color="#333333")

    for i, (a, b) in enumerate(cliques):
        y = axis_y - 0.10 - i * bar_h
        ax.plot([vals[a], vals[b]], [y, y], color="#3F7D5C", lw=3.4, solid_capstyle="round", zorder=2)

    for idx, name in enumerate(left_names):
        r = avg_ranks[name]
        y = -(stem0 + idx * row_h)
        ax.plot([r, r], [0, y], color="#888888", lw=0.8, zorder=1)
        ax.plot([r, lo - 0.55], [y, y], color="#888888", lw=0.8, zorder=1)
        ax.text(lo - 0.6, y, f"{name}  ({r:.2f})", ha="right", va="center", fontsize=9.5, color="#111111")

    for idx, name in enumerate(right_names):
        r = avg_ranks[name]
        y = -(stem0 + idx * row_h)
        ax.plot([r, r], [0, y], color="#888888", lw=0.8, zorder=1)
        ax.plot([r, hi + 0.55], [y, y], color="#888888", lw=0.8, zorder=1)
        ax.text(hi + 0.6, y, f"({r:.2f})  {name}", ha="left", va="center", fontsize=9.5, color="#111111")

    ax.axis("off")
    ax.set_title(title, fontsize=12, color="#111111", pad=14, loc="center")
    ax.text((lo + hi) / 2, top_y - 0.02, subtitle, ha="center", va="top", fontsize=9.5, color="#555555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor="white")
    print("saved", out_path)


def part1_cd_diagram_pr_auc(df, methods, k, N):
    pivot = df.pivot(index="dataset", columns="method", values="pr_auc")
    ranks = pivot.apply(lambda row: rankdata(-row.values), axis=1, result_type="expand")
    ranks.columns = methods
    avg_rank = ranks.mean(axis=0).sort_values()

    stat, p_friedman = friedmanchisquare(*[pivot[m].values for m in methods])
    print(f"Friedman test (PR-AUC): chi2 = {stat:.4f}, p = {p_friedman:.4f}")

    CD = critical_difference(k, N)
    print(f"k={k} methods, N={N} datasets -> CD = {CD:.4f}")

    plot_cd_diagram(
        avg_rank.to_dict(), cd=CD, k=k,
        title="Critical Difference diagram - PR-AUC ranked per dataset (SMD, MSL, SMAP, PSM)",
        subtitle=f"CD = {CD:.2f}  (Nemenyi, alpha=0.05, N={N} datasets)  --  lower average rank = better",
        out_path=HERE / "cd_diagram_per_dataset.png",
    )
    return CD


def part2_heatmap(df):
    METRICS = [("pr_auc", "PR-AUC"), ("auc_roc", "ROC"), ("auc_pr_t", "PR$_T$"),
               ("f1_raw", "F1"), ("p_raw", "Prec"), ("r_raw", "Rec")]

    n_rows, n_metrics, n_ds = len(METHOD_ORDER), len(METRICS), len(DATASETS)
    n_cols = n_ds * n_metrics
    raw = np.zeros((n_rows, n_cols))
    for j, ds in enumerate(DATASETS):
        sub = df[df.dataset == ds].set_index("method")
        for k_, (col, _) in enumerate(METRICS):
            for i, m in enumerate(METHOD_ORDER):
                raw[i, j * n_metrics + k_] = sub.loc[m, col]

    norm = (raw - raw.min(axis=0, keepdims=True)) / (raw.max(axis=0, keepdims=True) - raw.min(axis=0, keepdims=True))

    cmap = LinearSegmentedColormap.from_list("tvld_seq", ["#F7F5EE", "#BFD8CF", "#2F6F62", "#1B2A3B"])

    fig, ax = plt.subplots(figsize=(15.5, 6.0))
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            col_vals = raw[:, j]
            is_best = np.isclose(raw[i, j], col_vals.max())
            txt_color = "white" if norm[i, j] > 0.55 else "#1B2A3B"
            weight = "bold" if is_best else "normal"
            ax.text(j, i, f"{raw[i, j]:.3f}", ha="center", va="center",
                    fontsize=7.3, color=txt_color, fontweight=weight, zorder=4)
            if is_best:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor="#D99A2B", linewidth=1.8, zorder=5))

    for j in range(1, n_ds):
        ax.axvline(j * n_metrics - 0.5, color="white", linewidth=2.4, zorder=3)
    ax.axhline(n_rows - 1.5, color="white", linewidth=2.6, zorder=3)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([lbl for _ in DATASETS for _, lbl in METRICS], fontsize=8, color="#333333")
    ax.tick_params(axis="x", length=0)

    for j, ds in enumerate(DATASETS):
        x0 = j * n_metrics - 0.5
        ax.text(x0 + n_metrics / 2, -1.05, ds, ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color="#1B2A3B")

    row_labels = [f"{m}  ({METHOD_YEAR[m]})" for m in METHOD_ORDER]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10, color="#333333")
    for i, m in enumerate(METHOD_ORDER):
        if m == "TVL-D":
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_color("#1B2A3B")
    ax.tick_params(axis="y", length=0)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -1.4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Normalized score within column (0 = worst of 7, 1 = best of 7)", fontsize=8.5, color="#333333")
    cbar.ax.tick_params(labelsize=7.5, length=0)

    ax.set_title("TVL-D vs. six baselines across all reported metrics\n"
                 "(baselines ordered 2024–2026; gold outline = best in column)",
                 fontsize=12.5, color="#111111", pad=28)

    fig.tight_layout()
    out_path = HERE / "heatmap_all_metrics.png"
    fig.savefig(out_path, dpi=170, facecolor="white", bbox_inches="tight")
    print("saved", out_path)


def part3_line_plot(df):
    PANELS = [("pr_auc", "PR-AUC"), ("f1_raw", "F1 (best threshold)")]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x = np.arange(len(DATASETS))

    for ax, (col, title) in zip(axes, PANELS):
        for m in METHOD_ORDER:
            sub = df[df.method == m].set_index("dataset")
            y = [sub.loc[ds, col] for ds in DATASETS]
            is_tvld = (m == "TVL-D")
            ax.plot(x, y, color=COLORS[m],
                    linewidth=3.2 if is_tvld else 1.6,
                    marker="o", markersize=8 if is_tvld else 5.5,
                    markeredgecolor="white" if is_tvld else "none",
                    markeredgewidth=1.3 if is_tvld else 0,
                    alpha=1.0 if is_tvld else 0.85,
                    zorder=5 if is_tvld else 3,
                    label=f"{m} ({METHOD_YEAR[m]})")
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, fontsize=10.5)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=12.5, color="#111111")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c3c2b7")
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e")
        ax.grid(axis="y", color="#e1e0d9", lw=0.7, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Score", fontsize=10.5, color="#52514e")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=9.5)

    fig.suptitle("TVL-D vs. six baselines across datasets, ordered 2024–2026 (TVL-D emphasized)",
                 fontsize=13.5, color="#111111", y=1.03)
    fig.tight_layout()
    out_path = HERE / "line_plot_headline_metrics.png"
    fig.savefig(out_path, dpi=170, facecolor="white", bbox_inches="tight")
    print("saved", out_path)


def main():
    if not COMBINED_CSV.exists():
        print(f"Missing {COMBINED_CSV} - see this script's docstring for the expected format.")
        sys.exit(1)
    df = pd.read_csv(COMBINED_CSV)
    expected = {"TVL-D", "xLSTMAD", "KAN-AD", "ALoRa", "CANDI", "TopoGDN", "SensitiveHUE"}
    if set(df.method.unique()) != expected:
        print(f"Warning: expected methods {expected}, got {set(df.method.unique())}")
    methods = list(df.pivot(index="dataset", columns="method", values="pr_auc").columns)
    k, N = len(methods), df.dataset.nunique()
    print(f"{k} methods x {N} datasets")

    part1_cd_diagram_pr_auc(df, methods, k, N)
    part2_heatmap(df)
    part3_line_plot(df)


if __name__ == "__main__":
    main()
