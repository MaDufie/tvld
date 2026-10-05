"""TVL-D localization evaluation on SMD.

Detection (run_smd_eval.py) answers "is something wrong right now."
Localization answers "which of the D channels is responsible"  and it
comes from the SAME per-channel severity tensor the detection score is
built from, at no extra modeling cost: Shape/Level/Structure severity are
each computed per-channel *before* being summed into one number, so ranking
channels by their mean severity inside a flagged window is the entire
mechanism.

SMD is the only one of the three datasets with ground-truth root-cause channel labels
(interpretation_label/). SMAP/MSL's channels are already reduced to one
real telemetry column plus one-hot context flags (no multi-channel
ambiguity left to localize within a channel), and PSM ships no root-cause
channel labels at all so there is nothing to evaluate against for
either.

This is also the full-dataset counterpart to ../figures/generate_localization_demo.py,
which only computes HitRate@1.0 for one illustrative segment to make a
legible plot. This script is the actual evaluation: every SMD machine that
has interpretation labels, macro-averaged.

Usage:
    python run_localization_eval.py
Expects ./data/{train,test,test_label,interpretation_label}/machine-*.txt
(standard SMD layout)
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for tvld_final.py

from tvld_final import pick_window, shape_severity, level_severity, structure_severity, W_SHAPE, W_LEVEL, W_STRUCT  # noqa: E402

SMD_DATA_DIR = Path(os.environ.get("SMD_DATA_DIR", str(HERE / "data")))
CKPT_DIR = HERE / "checkpoints_smd"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = HERE / "smd_localization_results.csv"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [LOC] {msg}", flush=True)


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


def severities(mid, train, test):
    """Same cache schema/location as run_smd_eval.py's severities() - if that
    script already ran for this machine, this reuses its checkpoint instead
    of recomputing the expensive matrix-profile pass."""
    ckpt_path = CKPT_DIR / f"{mid}.pkl"
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ck = pickle.load(f)
        return ck["shape_sev"], ck["level_sev"], ck["struct_sev"]
    m = pick_window(train.shape[0], test.shape[0])
    shape_sev = shape_severity(train, test, m)
    level_sev = level_severity(train, test)
    struct_sev = structure_severity(train, test)
    with open(ckpt_path, "wb") as f:
        pickle.dump({"shape_sev": shape_sev, "level_sev": level_sev, "struct_sev": struct_sev}, f)
    return shape_sev, level_sev, struct_sev


def hitrate_at_p(the_map, interpretation, p=1.0):
    """HitRate@p (InterFusion-style): for each true anomaly segment with K
    root-cause dims, take the top round(K*p) dims ranked by mean map value
    over that segment's time window, and measure overlap with the true dims.
    K comes from the ground-truth interpretation label, not a free parameter
    of the method -- at deployment time (no root-cause labels available),
    TVL-D can rank channels by severity but has no label-derived K to pick."""
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
    log(f"Found {len(machines)} SMD machines in {SMD_DATA_DIR}")
    if not machines:
        log(f"No machine-*.txt files under {SMD_DATA_DIR}/train - check SMD_DATA_DIR / the data/ layout.")
        return
    if not (SMD_DATA_DIR / "interpretation_label").exists():
        log(f"No interpretation_label/ under {SMD_DATA_DIR} - this evaluation needs root-cause channel "
            f"labels, which most SMD mirrors include but not all. Can't continue.")
        return

    rows = []
    t_start = time.time()
    for i, mid in enumerate(machines):
        interp = load_smd_interpretation(mid, SMD_DATA_DIR)
        if not interp:
            log(f"[{i+1}/{len(machines)}] {mid}: no interpretation labels - skipped")
            continue

        train, test, label = load_smd_machine(mid, SMD_DATA_DIR)
        shape_sev, level_sev, struct_sev = severities(mid, train, test)
        weighted_map = W_SHAPE * shape_sev + W_LEVEL * level_sev + W_STRUCT * struct_sev

        hr1 = hitrate_at_p(weighted_map, interp, p=1.0)
        rows.append({"machine": mid, "n_segments": len(interp), "hitrate_at_1": hr1})
        log(f"[{i+1}/{len(machines)}] {mid}: {len(interp)} segment(s), HitRate@1.0={hr1:.4f} "
            f"-- {(time.time()-t_start)/60:.1f} min elapsed")
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        log("No machines had usable interpretation labels.")
        return
    log(f"ALL DONE. Wrote {OUT_CSV}. Total time: {(time.time()-t_start)/60:.1f} min")
    print(f"Localization evaluated on {len(df)}/{len(machines)} SMD machines "
          f"({int(df.n_segments.sum())} labeled anomaly segments total)")
    print(f"Mean HitRate@1.0, macro-averaged across machines: {df.hitrate_at_1.mean():.4f}")
    print(df.sort_values("hitrate_at_1").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
