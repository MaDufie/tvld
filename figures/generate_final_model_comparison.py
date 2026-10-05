"""Figure: TVL-D (pre-smoothing) vs. TVL-D (final, smoothed), side by side across
all 4 datasets (SMD, SMAP, MSL, PSM) and all 6 reported metrics - shows what the
smoothing step actually buys, since everything else (weights, severities) is
identical between the two bars.

Input: data/final_model_combined_summary.csv (8 rows: 4 datasets x 2 TVL-D variants).
This is a small, already-computed summary table -- regenerate it  by running
../main_benchmark/run_smd_eval.py, run_smap_msl_eval.py, run_psm_eval.py and
combining their per-method-averaged output rows into the same 8-row shape (columns:
dataset, method, pr_auc, auc_roc, auc_pr_t, f1, precision, recall; method is either
"TVL-D" or "TVL-D (pre-smoothing)") if you want to rebuild this from a fresh run.

Usage:
    python generate_final_model_comparison.py
Writes: final_model_comparison.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE / "data" / "final_model_combined_summary.csv"
OUT_PNG = HERE / "final_model_comparison.png"

DATASETS = ["SMD", "SMAP", "MSL", "PSM"]
METHODS_ORDER = ["TVL-D (pre-smoothing)", "TVL-D"]
COLORS = {"TVL-D (pre-smoothing)": "#E2A03F", "TVL-D": "#3F7D5C"}
METRIC_TITLES = [
    ("pr_auc", "PR-AUC"), ("auc_roc", "AUC-ROC"), ("auc_pr_t", "AUC-PR_T (range)"),
    ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"),
]


def main():
    if not DATA_CSV.exists():
        print(f"Missing {DATA_CSV} - see this script's docstring for how to rebuild it.")
        sys.exit(1)
    combined = pd.read_csv(DATA_CSV)
    missing = set(DATASETS) - set(combined.dataset.unique())
    if missing:
        print(f"Warning: datasets missing from {DATA_CSV}: {missing}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    width = 0.35
    for ax, (col, title) in zip(axes, METRIC_TITLES):
        x = np.arange(len(DATASETS))
        for j, method in enumerate(METHODS_ORDER):
            vals = [combined[(combined.dataset == ds) & (combined.method == method)][col].values[0]
                    for ds in DATASETS]
            ax.bar(x + (j - 0.5) * width, vals, width, label=method, color=COLORS[method])
        ax.set_xticks(x); ax.set_xticklabels(DATASETS, fontsize=13)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2, fontsize=12)
    fig.tight_layout(pad=1.4, h_pad=2.4, w_pad=1.8)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
