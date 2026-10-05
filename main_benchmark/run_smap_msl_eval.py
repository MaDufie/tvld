"""TVL-D evaluation on SMAP + MSL (NASA spacecraft telemetry, Hundman et al. 2018) 

SMAP/MSL channels are one real telemetry column plus D-1 sparse one-hot command/event 
flags. Shape and Level are computed on the telemetry column ONLY (train[:, :1] / test[:, :1]); 
Structure is fit on the FULL feature vector (so it still sees command-state context)
but only its telemetry column is kept..

SMAP and MSL are two separate spacecraft bundled in one release (55 and 27
channels respectively) - reported separately in the output.

Usage:
    python run_smap_msl_eval.py
Expects ./data/{train,test}/<chan_id>.npy and ./labeled_anomalies.csv next to
this script (the standard NASA SMAP/MSL release layout). 
"""
import ast
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for tvld_final.py
sys.path.insert(0, str(HERE))         # for metrics.py

from tvld_final import (  # noqa: E402
    pick_window, shape_severity, level_severity, structure_severity,
    W_SHAPE, W_LEVEL, W_STRUCT, SMOOTH_WINDOW,
)
from metrics import full_metrics  # noqa: E402

SMAP_MSL_DATA_DIR = Path(os.environ.get("SMAP_MSL_DATA_DIR", str(HERE / "data")))
SMAP_MSL_LABELS = Path(os.environ.get("SMAP_MSL_LABELS", str(HERE / "labeled_anomalies.csv")))
CKPT_DIR = HERE / "checkpoints_smap_msl"
OUT_CSV = HERE / "smap_msl_per_channel.csv"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [SMAP/MSL] {msg}", flush=True)


def load_smap_msl_meta(csv_path):
    df = pd.read_csv(csv_path)
    meta = {}
    for _, row in df.iterrows():
        cid = row["chan_id"]
        windows = ast.literal_eval(row["anomaly_sequences"])
        if cid not in meta:
            meta[cid] = {"spacecraft": row["spacecraft"], "windows": list(windows)}
        else:
            meta[cid]["windows"].extend(windows)  # duplicate chan_id (e.g. P-2): union windows
    return meta


def load_smap_msl_channel(cid, meta, data_dir):
    train = np.load(data_dir / "train" / f"{cid}.npy").astype(float)
    test = np.load(data_dir / "test" / f"{cid}.npy").astype(float)
    label = np.zeros(test.shape[0], dtype=int)
    for s, e in meta[cid]["windows"]:
        label[s:e] = 1
    return train, test, label


def severities(cid, train, test):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{cid}.pkl"
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            return pickle.load(f)
    m = pick_window(train.shape[0], test.shape[0])
    shape_sev_full = shape_severity(train[:, :1], test[:, :1], m)
    level_sev_full = level_severity(train[:, :1], test[:, :1])
    struct_sev_full = structure_severity(train, test)  # fit on the FULL feature vector
    sev = {"shape_col0": shape_sev_full[:, 0], "level_col0": level_sev_full[:, 0],
           "struct_col0": struct_sev_full[:, 0]}
    with open(ckpt_path, "wb") as f:
        pickle.dump(sev, f)
    return sev


def main():
    if not SMAP_MSL_LABELS.exists():
        log(f"Labels file not found: {SMAP_MSL_LABELS} -- check SMAP_MSL_LABELS.")
        return
    meta = load_smap_msl_meta(SMAP_MSL_LABELS)
    channel_ids = sorted(meta.keys())
    log(f"{len(channel_ids)} SMAP/MSL channels: "
        f"{sum(1 for c in channel_ids if meta[c]['spacecraft']=='SMAP')} SMAP, "
        f"{sum(1 for c in channel_ids if meta[c]['spacecraft']=='MSL')} MSL")

    rows, skipped = [], []
    t_start = time.time()
    for i, cid in enumerate(channel_ids):
        train, test, label = load_smap_msl_channel(cid, meta, SMAP_MSL_DATA_DIR)
        if label.sum() == 0:
            skipped.append(cid)
            continue
        sev = severities(cid, train, test)

        weighted = W_SHAPE * sev["shape_col0"] + W_LEVEL * sev["level_col0"] + W_STRUCT * sev["struct_col0"]
        v2 = weighted  # already single-column (telemetry-only)
        v3 = uniform_filter1d(v2, size=SMOOTH_WINDOW, mode="nearest")

        for method, scores in [("TVL-D (pre-smoothing)", v2), ("TVL-D", v3)]:
            met = full_metrics(scores, label)
            if met is None:
                continue
            met.update({"chan_id": cid, "spacecraft": meta[cid]["spacecraft"], "method": method,
                        "D": train.shape[1], "test_len": test.shape[0]})
            rows.append(met)

        if (i + 1) % 20 == 0 or (i + 1) == len(channel_ids):
            log(f"[{i+1}/{len(channel_ids)}] {cid} done -- {(time.time()-t_start)/60:.1f} min elapsed")

    if skipped:
        log(f"Skipped (no labeled anomaly in test set): {skipped}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    cols = ["pr_auc", "auc_roc", "auc_pr_t", "f1", "precision", "recall"]
    spacecraft_summary = df.groupby(["method", "spacecraft"])[cols].mean().reset_index()
    for sc in ["SMAP", "MSL"]:
        sub = spacecraft_summary[spacecraft_summary.spacecraft == sc].drop(columns="spacecraft").reset_index(drop=True)
        sub.insert(1, "dataset", sc)
        print(f"\n{sc}:")
        print(sub.round(4).to_string(index=False))

    log(f"ALL DONE. Wrote {OUT_CSV}. Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
