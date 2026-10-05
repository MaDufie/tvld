"""Figure: TVL-D's detection score alongside the per-channel severity heatmap it's
built from, around one real SMD labeled anomaly segment - the concrete illustration
of how Shape+Level+Structure severity (summed per timestep = detection score, ranked
per timestep = localization) relate to each other on actual data.

This one  recomputes the severity maps for one machine using
../tvld_final.py's  pipeline.

Usage:
    python generate_localization_demo.py
Expects SMD at ../main_benchmark/data/ (reuses that copy) or set SMD_DATA_DIR.
Also needs that machine's interpretation_label/ file (ground-truth root-cause
channels), which most SMD mirrors do include alongside train/test/test_label/.
Writes: final_model_localization_demo.png
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for tvld_final.py

from tvld_final import pick_window, shape_severity, level_severity, structure_severity, fuse, W_SHAPE, W_LEVEL, W_STRUCT  # noqa: E402

SMD_DATA_DIR = Path(os.environ.get("SMD_DATA_DIR", str(HERE.parent / "main_benchmark" / "data")))
OUT_PNG = HERE / "final_model_localization_demo.png"


def load_smd_machine(mid, data_dir):
    train = pd.read_csv(data_dir / "train" / f"{mid}.txt", header=None).to_numpy(dtype=float)
    test = pd.read_csv(data_dir / "test" / f"{mid}.txt", header=None).to_numpy(dtype=float)
    label = pd.read_csv(data_dir / "test_label" / f"{mid}.txt", header=None).to_numpy(dtype=int).ravel()
    return train, test, label


def load_smd_interpretation(mid, data_dir):
    """SMD's interpretation_label file: 'start-end:dim1,dim2,...' -- a 0-indexed,
    end-exclusive span (Python slice convention) with 1-indexed dimensions."""
    interp = []
    path = data_dir / "interpretation_label" / f"{mid}.txt"
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                span, dims = line.split(":")
                start, end = (int(x) for x in span.split("-"))
                dim_list = [int(d) - 1 for d in dims.split(",")]
                interp.append({"start": start, "end": end, "dims": dim_list})
    return interp


def hitrate_at_p(the_map, interpretation, p=1.0):
    if not interpretation:
        return None
    hits = []
    for seg in interpretation:
        start, end = seg["start"], seg["end"]
        true_dims = set(seg["dims"])
        K = len(true_dims)
        n_pick = max(1, round(K * p))
        window_scores = the_map[start:end, :].mean(axis=0)
        ranked = np.argsort(-window_scores)[:n_pick]
        overlap = len(set(ranked.tolist()) & true_dims)
        hits.append(overlap / K)
    return float(np.mean(hits))


def main():
    machines = sorted(p.stem for p in (SMD_DATA_DIR / "train").glob("machine-*.txt"))
    if not machines:
        print(f"No machine-*.txt under {SMD_DATA_DIR}/train -- check SMD_DATA_DIR.")
        sys.exit(1)
    if not (SMD_DATA_DIR / "interpretation_label").exists():
        print(f"No interpretation_label/ under {SMD_DATA_DIR} -- this figure needs root-cause "
              f"channel labels, which most SMD mirrors include but not all. Can't continue.")
        sys.exit(1)

    demo_mid, demo_seg = None, None
    for mid in machines:
        for seg in load_smd_interpretation(mid, SMD_DATA_DIR):
            key = (len(seg["dims"]), -(seg["end"] - seg["start"]))
            if demo_seg is None or key < (len(demo_seg["dims"]), -(demo_seg["end"] - demo_seg["start"])):
                demo_mid, demo_seg = mid, seg

    if demo_seg is None:
        print("No interpretation labels found on any machine -- can't build the demo plot.")
        sys.exit(1)

    print(f"Using {demo_mid}, segment [{demo_seg['start']}, {demo_seg['end']}), "
          f"{len(demo_seg['dims'])} true root-cause channel(s)")

    train, test, label = load_smd_machine(demo_mid, SMD_DATA_DIR)
    m = pick_window(train.shape[0], test.shape[0])
    shape_sev = shape_severity(train, test, m)
    level_sev = level_severity(train, test)
    struct_sev = structure_severity(train, test)
    weighted_map = W_SHAPE * shape_sev + W_LEVEL * level_sev + W_STRUCT * struct_sev
    _, smoothed_score = fuse(shape_sev, level_sev, struct_sev, primary_cols=None)

    seg = demo_seg
    pad = 200
    lo, hi = max(0, seg["start"] - pad), min(len(label), seg["end"] + pad)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2]})

    axes[0].plot(range(lo, hi), smoothed_score[lo:hi], color="#3F7D5C", lw=1.2, label="TVL-D detection score")
    axes[0].axvspan(seg["start"], seg["end"], color="gold", alpha=0.3, label="true anomaly segment")
    axes[0].set_ylabel("detection score")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title(f"{demo_mid}: TVL-D detection score around segment [{seg['start']}, {seg['end']})")

    window = weighted_map[lo:hi, :].T
    im = axes[1].imshow(window, aspect="auto", cmap="magma", extent=[lo, hi, weighted_map.shape[1], 0])
    for d in seg["dims"]:
        axes[1].axhline(d + 0.5, color="cyan", lw=1, ls="--")
    axes[1].axvspan(seg["start"], seg["end"], color="white", alpha=0.08)
    axes[1].set_ylabel("variable index")
    axes[1].set_xlabel("timestep")
    axes[1].set_title("severity heatmap (time x variable) -- cyan dashed lines = true root-cause channels")
    fig.colorbar(im, ax=axes[1], label="severity", fraction=0.02)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"saved {OUT_PNG}")

    K = len(seg["dims"])
    seg_hitrate = hitrate_at_p(weighted_map, [seg], p=1.0)
    print(f"This segment: {K} true root-cause channel(s) {sorted(d + 1 for d in seg['dims'])} (1-indexed), "
          f"HitRate@1.0 = {seg_hitrate:.2f}")


if __name__ == "__main__":
    main()
