"""Run xLSTMAD (official repo, vanilla/CPU backend) on SMD, SMAP, MSL, and PSM using the
exact same fixed hyperparameters everywhere, and score it with the exact same metrics suite
(full_metrics from tvld_final.py) used throughout this project for TVL-D, so the numbers are
directly comparable.

Usage (run from anywhere, or from this directory):
    python run_all_xlstmad.py            # runs SMD + SMAP/MSL + PSM
    python run_all_xlstmad.py smd        # just SMD
    python run_all_xlstmad.py smap_msl   # just SMAP/MSL
    python run_all_xlstmad.py psm        # just PSM

Expects the same ./data/ layouts as main_benchmark/run_smd_eval.py /
run_smap_msl_eval.py / run_psm_eval.py (SMD: ./data/{train,test,test_label}/
machine-*.txt; SMAP/MSL: ./data/{train,test}/<chan_id>.npy +
./labeled_anomalies.csv; PSM: ./data/{train.csv,test.csv,test_label.csv}) --
set the SMD_DATA_DIR / SMAP_MSL_DATA_DIR / SMAP_MSL_LABELS / PSM_DATA_DIR env
vars to point elsewhere instead, e.g. to a data/ folder shared with the other
main_benchmark scripts one level up.
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))  # for tvld_final.py
sys.path.insert(0, str(HERE))          # for xlstmad_runner.py

from tvld_final import (full_metrics, load_smd_machine, load_smap_msl_meta,  # noqa: E402
                         load_smap_msl_channel)
from xlstmad_runner import run_xlstmad  # noqa: E402

BASE = HERE
CKPT_DIR = BASE / "checkpoints_xlstmad"
CKPT_DIR.mkdir(exist_ok=True)

SMD_DATA_DIR = Path(os.environ.get("SMD_DATA_DIR", str(HERE.parent.parent / "data")))
SMAP_MSL_DATA_DIR = Path(os.environ.get("SMAP_MSL_DATA_DIR", str(HERE.parent.parent / "data")))
SMAP_MSL_LABELS = Path(os.environ.get("SMAP_MSL_LABELS", str(HERE.parent.parent / "data" / "labeled_anomalies.csv")))
PSM_DATA_DIR = Path(os.environ.get("PSM_DATA_DIR", str(HERE.parent.parent / "data")))

METRIC_COLS = ["pr_auc", "auc_roc", "auc_pr_t", "f1_raw", "p_raw", "r_raw", "f1_pa", "p_pa", "r_pa"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [xLSTMAD] {msg}", flush=True)


def cached_run(cache_path, train, test, label):
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        return d["scores"], d["label"], d["meta"]
    scores, aligned_label, meta = run_xlstmad(train, test, label)
    with open(cache_path, "wb") as f:
        pickle.dump({"scores": scores, "label": aligned_label, "meta": meta}, f)
    return scores, aligned_label, meta


# ============================================================== SMD ==
def run_smd():
    out_dir = CKPT_DIR / "smd"
    out_dir.mkdir(exist_ok=True)
    machines = sorted(p.stem for p in (SMD_DATA_DIR / "train").glob("machine-*.txt"))
    log(f"SMD: {len(machines)} machines")
    rows = []
    t0 = time.time()
    for i, mid in enumerate(machines):
        cache_path = out_dir / f"{mid}.pkl"
        train, test, label = load_smd_machine(mid, SMD_DATA_DIR)
        scores, aligned_label, meta = cached_run(cache_path, train, test, label)
        if scores is None:
            log(f"  [{i+1}/{len(machines)}] {mid} SKIPPED ({meta.get('skipped')})")
            continue
        met = full_metrics(scores, aligned_label)
        if met is None:
            log(f"  [{i+1}/{len(machines)}] {mid} SKIPPED (no positive/negative labels)")
            continue
        met.update({"id": mid, "method": "xLSTMAD", "elapsed": meta["elapsed"], "epochs": meta["epochs_trained"]})
        rows.append(met)
        log(f"  [{i+1}/{len(machines)}] {mid} done in {meta['elapsed']:.1f}s "
            f"(epochs={meta['epochs_trained']}) pr_auc={met['pr_auc']:.4f} auc_roc={met['auc_roc']:.4f} "
            f"-- total elapsed {(time.time()-t0)/60:.1f} min")
    df = pd.DataFrame(rows)
    df.to_csv(BASE / "xlstmad_smd_per_machine.csv", index=False)
    log(f"SMD done: {len(df)} machines, total {(time.time()-t0)/60:.1f} min")
    return df


# ======================================================== SMAP / MSL ==
def run_smap_msl():
    out_dir = CKPT_DIR / "smap_msl"
    out_dir.mkdir(exist_ok=True)
    meta_all = load_smap_msl_meta(SMAP_MSL_LABELS)
    channel_ids = sorted(meta_all.keys())
    log(f"SMAP/MSL: {len(channel_ids)} channels")
    rows = []
    skipped = []
    t0 = time.time()
    for i, cid in enumerate(channel_ids):
        train, test, label = load_smap_msl_channel(cid, meta_all, SMAP_MSL_DATA_DIR)
        if label.sum() == 0:
            skipped.append(cid)
            continue
        cache_path = out_dir / f"{cid}.pkl"
        scores, aligned_label, meta = cached_run(cache_path, train, test, label)
        if scores is None:
            log(f"  [{i+1}/{len(channel_ids)}] {cid} SKIPPED ({meta.get('skipped')})")
            continue
        met = full_metrics(scores, aligned_label)
        if met is None:
            continue
        met.update({"id": cid, "spacecraft": meta_all[cid]["spacecraft"], "method": "xLSTMAD",
                    "elapsed": meta["elapsed"], "epochs": meta["epochs_trained"]})
        rows.append(met)
        if (i + 1) % 10 == 0 or (i + 1) == len(channel_ids):
            log(f"  [{i+1}/{len(channel_ids)}] {cid} done -- total elapsed {(time.time()-t0)/60:.1f} min")
    if skipped:
        log(f"Skipped (no labeled anomaly in test set): {skipped}")
    df = pd.DataFrame(rows)
    df.to_csv(BASE / "xlstmad_smap_msl_per_channel.csv", index=False)
    log(f"SMAP/MSL done: {len(df)} channels, total {(time.time()-t0)/60:.1f} min")
    return df


# ============================================================== PSM ==
def run_psm():
    out_dir = CKPT_DIR / "psm"
    out_dir.mkdir(exist_ok=True)
    train_df = pd.read_csv(PSM_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PSM_DATA_DIR / "test.csv")
    label_df = pd.read_csv(PSM_DATA_DIR / "test_label.csv")
    feat_cols = [c for c in train_df.columns if c != "timestamp_(min)"]

    n_nan_train = train_df[feat_cols].isna().sum().sum()
    train_df[feat_cols] = train_df[feat_cols].interpolate(method="linear", limit_direction="both")
    n_nan_test = test_df[feat_cols].isna().sum().sum()
    test_df[feat_cols] = test_df[feat_cols].interpolate(method="linear", limit_direction="both")
    log(f"PSM train NaNs interpolated: {n_nan_train}, test NaNs interpolated: {n_nan_test}")

    train = train_df[feat_cols].to_numpy(dtype=float)
    test = test_df[feat_cols].to_numpy(dtype=float)
    label = label_df["label"].to_numpy(dtype=int)
    log(f"PSM: train {train.shape}, test {test.shape}, anomaly frac {label.mean():.4f}")

    cache_path = out_dir / "psm.pkl"
    t0 = time.time()
    scores, aligned_label, meta = cached_run(cache_path, train, test, label)
    met = full_metrics(scores, aligned_label)
    met.update({"id": "PSM", "method": "xLSTMAD", "elapsed": meta["elapsed"], "epochs": meta["epochs_trained"]})
    log(f"PSM done in {(time.time()-t0)/60:.1f} min: pr_auc={met['pr_auc']:.4f} auc_roc={met['auc_roc']:.4f}")
    df = pd.DataFrame([met])
    df.to_csv(BASE / "xlstmad_psm_results.csv", index=False)
    return df


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("all", "smd"):
        run_smd()
    if which in ("all", "smap_msl"):
        run_smap_msl()
    if which in ("all", "psm"):
        run_psm()
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")
