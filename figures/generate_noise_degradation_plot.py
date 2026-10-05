"""Figure: degradation vs. noise level -- how each of the 7 methods' PR-AUC and F1
hold up as Gaussian test-set noise increases (10/20/30/40/50%), on SMD.

Usage:
    python generate_noise_degradation_plot.py path/to/combined_noise_results.csv
    (or put it at ./data/combined_noise_results.csv and run with no argument)
Writes: noise_degradation.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "data" / "combined_noise_results.csv"

METHOD_ORDER = ["SensitiveHUE", "TopoGDN", "KAN-AD", "xLSTMAD", "ALoRa", "CANDI", "TVL-D"]
METHOD_YEAR = {"SensitiveHUE": "2024", "TopoGDN": "2024", "KAN-AD": "2024",
               "xLSTMAD": "2025", "ALoRa": "2026", "CANDI": "2026", "TVL-D": "ours"}
COLORS = {
    "SensitiveHUE": "#e87ba4", "TopoGDN": "#008300", "KAN-AD": "#eda100",
    "xLSTMAD": "#4a3aa7", "ALoRa": "#eb6834", "CANDI": "#1baf7a", "TVL-D": "#2a78d6",
}
NOISE_LEVELS = [10, 20, 30, 40, 50]


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        print(f"Missing {csv_path}. This figure needs the noise study's combined results -- "
              f"see this script's docstring for the expected schema and how to build it.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required_cols = {"method", "machine", "noise_pct"}
    if not required_cols.issubset(df.columns):
        print(f"{csv_path} is missing required columns {required_cols - set(df.columns)}.")
        sys.exit(1)

    present_methods = [m for m in METHOD_ORDER if m in df.method.unique()]
    if len(present_methods) < len(METHOD_ORDER):
        missing = set(METHOD_ORDER) - set(present_methods)
        print(f"Note: methods not found in the data (plotting the rest anyway): {missing}")

    # mean across machines, per (method, noise_pct)
    PANELS = [("pr_auc", "PR-AUC"), ("f1_raw", "F1 (best threshold)")]
    panels_available = [(col, title) for col, title in PANELS if col in df.columns]
    if not panels_available:
        print(f"Neither pr_auc nor f1_raw found in {csv_path}'s columns: {list(df.columns)}")
        sys.exit(1)

    fig, axes = plt.subplots(1, len(panels_available), figsize=(6.25 * len(panels_available), 5.2))
    if len(panels_available) == 1:
        axes = [axes]
    x = np.arange(len(NOISE_LEVELS))

    for ax, (col, title) in zip(axes, panels_available):
        for m in present_methods:
            sub = df[df.method == m].groupby("noise_pct")[col].mean()
            y = [sub.get(n, np.nan) for n in NOISE_LEVELS]
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
        ax.set_xticklabels([f"{n}%" for n in NOISE_LEVELS], fontsize=10.5)
        ax.set_xlabel("test-set noise level", fontsize=10.5, color="#52514e")
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

    fig.suptitle("Degradation under test-set noise -- SMD, ordered 2024–2026 (TVL-D emphasized)",
                 fontsize=13.5, color="#111111", y=1.03)
    fig.tight_layout()
    out_path = HERE / "noise_degradation.png"
    fig.savefig(out_path, dpi=170, facecolor="white", bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
