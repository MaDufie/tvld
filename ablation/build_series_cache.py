"""Build series_cache.pkl -- the input the leave-one-component-out ablation
(ablation_table_full.py) needs: every evaluation series (SMD machine,
SMAP/MSL channel, PSM) reduced to its per-timestep Shape/Level/Structure
severity SCALARS plus its label, using the exact same per-dataset column
aggregation the final model uses (sum over every scored column for SMD/PSM,
telemetry-column-only for SMAP/MSL..

This reuses whichever severity checkpoints the three main_benchmark/ scripts
already produced (../main_benchmark/checkpoints_smd/, checkpoints_smap_msl/,
checkpoint_psm.pkl) if they exist, so running this AFTER main_benchmark/'s
three eval scripts is instant. If a checkpoint is missing, this computes it
itself (calling the same tvld_final.py functions those scripts do), so this
script also works standalone - it doesn't require main_benchmark/ to have
been run first, just the same three datasets to be present somewhere.

Usage:
    python build_series_cache.py
Expects SMD/SMAP-MSL/PSM data either already cached as checkpoints under
../main_benchmark/ (fastest), or available at the same locations
main_benchmark/'s own scripts expect (SMD_DATA_DIR, SMAP_MSL_DATA_DIR,
SMAP_MSL_LABELS, PSM_DATA_DIR env vars, defaulting to ../main_benchmark/data
and ../main_benchmark/labeled_anomalies.csv -- the same one shared copy of
each dataset every script in this package uses).
Writes: series_cache.pkl
"""
import ast
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN_BENCH = HERE.parent / "main_benchmark"
sys.path.insert(0, str(HERE.parent))  # for tvld_final.py
sys.path.insert(0, str(MAIN_BENCH))   # for metrics.py 

from tvld_final import pick_window, shape_severity, level_severity, structure_severity  # noqa: E402

SMD_DATA_DIR = Path(os.environ.get("SMD_DATA_DIR", str(MAIN_BENCH / "data")))
SMAP_MSL_DATA_DIR = Path(os.environ.get("SMAP_MSL_DATA_DIR", str(MAIN_BENCH / "data")))
SMAP_MSL_LABELS = Path(os.environ.get("SMAP_MSL_LABELS", str(MAIN_BENCH / "labeled_anomalies.csv")))
PSM_DATA_DIR = Path(os.environ.get("PSM_DATA_DIR", str(MAIN_BENCH / "data")))

SMD_CKPT_DIR = MAIN_BENCH / "checkpoints_smd"
SMAP_MSL_CKPT_DIR = MAIN_BENCH / "checkpoints_smap_msl"
PSM_CKPT_PATH = MAIN_BENCH / "checkpoint_psm.pkl"

OUT_PKL = HERE / "series_cache.pkl"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [CACHE] {msg}", flush=True)


# ---------------------------------------------------------------- SMD --
def load_smd_machine(mid, data_dir):
    train = pd.read_csv(data_dir / "train" / f"{mid}.txt", header=None).to_numpy(dtype=float)
    test = pd.read_csv(data_dir / "test" / f"{mid}.txt", header=None).to_numpy(dtype=float)
    label = pd.read_csv(data_dir / "test_label" / f"{mid}.txt", header=None).to_numpy(dtype=int).ravel()
    return train, test, label


def smd_series():
    out = []
    machines = sorted(p.stem for p in (SMD_DATA_DIR / "train").glob("machine-*.txt"))
    for mid in machines:
        ckpt_path = SMD_CKPT_DIR / f"{mid}.pkl"
        _, _, label = load_smd_machine(mid, SMD_DATA_DIR)
        if ckpt_path.exists():
            with open(ckpt_path, "rb") as f:
                ck = pickle.load(f)
            shape_sev, level_sev, struct_sev = ck["shape_sev"], ck["level_sev"], ck["struct_sev"]
        else:
            train, test, _ = load_smd_machine(mid, SMD_DATA_DIR)
            m = pick_window(train.shape[0], test.shape[0])
            shape_sev = shape_severity(train, test, m)
            level_sev = level_severity(train, test)
            struct_sev = structure_severity(train, test)
        out.append({"dataset": "SMD", "id": mid, "label": label,
                     "shape": shape_sev.sum(axis=1), "level": level_sev.sum(axis=1),
                     "struct": struct_sev.sum(axis=1)})
    log(f"SMD: {len(out)} series loaded")
    return out


# ---------------------------------------------------------------- SMAP/MSL --
def load_smap_msl_meta(csv_path):
    df = pd.read_csv(csv_path)
    meta = {}
    for _, row in df.iterrows():
        cid = row["chan_id"]
        windows = ast.literal_eval(row["anomaly_sequences"])
        if cid not in meta:
            meta[cid] = {"spacecraft": row["spacecraft"], "windows": list(windows)}
        else:
            meta[cid]["windows"].extend(windows)
    return meta


def smap_msl_series():
    out = []
    if not SMAP_MSL_LABELS.exists():
        log(f"SMAP/MSL labels not found at {SMAP_MSL_LABELS} -- skipping SMAP/MSL.")
        return out
    meta = load_smap_msl_meta(SMAP_MSL_LABELS)
    for cid in sorted(meta.keys()):
        ckpt_path = SMAP_MSL_CKPT_DIR / f"{cid}.pkl"
        train_path = SMAP_MSL_DATA_DIR / "train" / f"{cid}.npy"
        test_path = SMAP_MSL_DATA_DIR / "test" / f"{cid}.npy"
        if ckpt_path.exists():
            with open(ckpt_path, "rb") as f:
                sev = pickle.load(f)
            n_points = len(sev["shape_col0"])
        elif train_path.exists() and test_path.exists():
            train = np.load(train_path).astype(float)
            test = np.load(test_path).astype(float)
            m = pick_window(train.shape[0], test.shape[0])
            shape_sev_full = shape_severity(train[:, :1], test[:, :1], m)
            level_sev_full = level_severity(train[:, :1], test[:, :1])
            struct_sev_full = structure_severity(train, test)
            sev = {"shape_col0": shape_sev_full[:, 0], "level_col0": level_sev_full[:, 0],
                   "struct_col0": struct_sev_full[:, 0]}
            n_points = test.shape[0]
        else:
            continue
        label = np.zeros(n_points, dtype=int)
        for s, e in meta[cid]["windows"]:
            label[s:e] = 1
        if label.sum() == 0:
            continue
        out.append({"dataset": "SMAP/MSL", "id": cid, "label": label,
                     "shape": sev["shape_col0"], "level": sev["level_col0"], "struct": sev["struct_col0"]})
    log(f"SMAP/MSL: {len(out)} series loaded")
    return out


# ---------------------------------------------------------------- PSM --
def load_psm(data_dir):
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    label_df = pd.read_csv(data_dir / "test_label.csv")
    feat_cols = [c for c in train_df.columns if c != "timestamp_(min)"]
    train_df[feat_cols] = train_df[feat_cols].interpolate(method="linear", limit_direction="both")
    train = train_df[feat_cols].to_numpy(dtype=float)
    test = test_df[feat_cols].to_numpy(dtype=float)
    label = label_df["label"].to_numpy(dtype=int)
    return train, test, label


def psm_series():
    out = []
    if not (PSM_DATA_DIR / "train.csv").exists():
        log(f"PSM train.csv not found at {PSM_DATA_DIR} -- skipping PSM.")
        return out
    if PSM_CKPT_PATH.exists():
        with open(PSM_CKPT_PATH, "rb") as f:
            ck = pickle.load(f)
        shape_sev, level_sev, struct_sev = ck["shape_sev"], ck["level_sev"], ck["struct_sev"]
        _, _, label = load_psm(PSM_DATA_DIR)
    else:
        train, test, label = load_psm(PSM_DATA_DIR)
        m = pick_window(train.shape[0], test.shape[0])
        shape_sev = shape_severity(train, test, m)
        level_sev = level_severity(train, test)
        struct_sev = structure_severity(train, test)
    out.append({"dataset": "PSM", "id": "PSM", "label": label,
                "shape": shape_sev.sum(axis=1), "level": level_sev.sum(axis=1), "struct": struct_sev.sum(axis=1)})
    log("PSM: 1 series loaded")
    return out


def main():
    series = smd_series() + smap_msl_series() + psm_series()
    with open(OUT_PKL, "wb") as f:
        pickle.dump(series, f)
    log(f"TOTAL: {len(series)} series cached -> {OUT_PKL}")


if __name__ == "__main__":
    main()
