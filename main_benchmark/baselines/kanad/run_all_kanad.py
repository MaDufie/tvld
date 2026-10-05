"""Run KAN-AD (official repo's model, reimplemented training loop for direct numpy
input) on SMD, SMAP, MSL, and PSM, scored with the exact same full_metrics suite used
throughout this project.

Usage (from inside this directory, or anywhere -- paths are not cwd-dependent):
    python run_all_kanad.py            # runs smap_msl, psm, then smd
    python run_all_kanad.py smd        # just one dataset (smd / smap_msl / psm)
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# This file lives at main_benchmark/baselines/kanad/, i.e. 3 levels under the
# package root.
PACKAGE_ROOT = HERE.parent.parent.parent
MAIN_BENCHMARK_DIR = HERE.parent.parent  # main_benchmark/

sys.path.insert(0, str(PACKAGE_ROOT))  # for tvld_final.py
sys.path.insert(0, str(HERE))          # for kanad_runner.py

from tvld_final import (full_metrics, load_smd_machine, load_smap_msl_meta,  # noqa: E402
                         load_smap_msl_channel)
from kanad_runner import run_kanad  # noqa: E402

BASE = HERE
CKPT_DIR = BASE / "checkpoints_kanad"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Shared with the TVL-D main_benchmark scripts -- point these at the SAME dataset
# copies (main_benchmark/data/, main_benchmark/labeled_anomalies.csv) by default so
# you only need one copy of each dataset for the whole package. Override any of
# them individually if your data lives elsewhere.
SMD_DATA_DIR = Path(os.environ.get("SMD_DATA_DIR", str(MAIN_BENCHMARK_DIR / "data")))
SMAP_MSL_DATA_DIR = Path(os.environ.get("SMAP_MSL_DATA_DIR", str(MAIN_BENCHMARK_DIR / "data")))
SMAP_MSL_LABELS = Path(os.environ.get("SMAP_MSL_LABELS", str(MAIN_BENCHMARK_DIR / "labeled_anomalies.csv")))
PSM_DATA_DIR = Path(os.environ.get("PSM_DATA_DIR", str(MAIN_BENCHMARK_DIR / "data")))

RUN_KWARGS = dict(batch_size=2048, max_epochs=8, patience=3)

METRIC_COLS = ["pr_auc", "auc_roc", "auc_pr_t", "f1_raw", "p_raw", "r_raw", "f1_pa", "p_pa", "r_pa"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cached_channel_run(cache_path, train_1d, test_1d, label):
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        return d["scores"], d["label"], d["meta"]
    scores, aligned_label, meta = run_kanad(train_1d, test_1d, label, **RUN_KWARGS)
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
    for mi, mid in enumerate(machines):
        m_dir = out_dir / mid
        m_dir.mkdir(exist_ok=True)
        train, test, label = load_smd_machine(mid, SMD_DATA_DIR)
        D = train.shape[1]
        per_dim_scores = []
        aligned_label = None
        t_machine = time.time()
        for d in range(D):
            cache_path = m_dir / f"dim{d:02d}.pkl"
            scores, al, meta = cached_channel_run(cache_path, train[:, d], test[:, d], label)
            if scores is None:
                continue
            per_dim_scores.append(scores)
            aligned_label = al
        if not per_dim_scores:
            log(f"  [{mi+1}/{len(machines)}] {mid} SKIPPED (no usable channels)")
            continue
        agg = np.max(np.stack(per_dim_scores, axis=0), axis=0)
        met = full_metrics(agg, aligned_label)
        if met is None:
            log(f"  [{mi+1}/{len(machines)}] {mid} SKIPPED (degenerate labels)")
            continue
        met.update({"id": mid, "method": "KAN-AD", "D": D, "elapsed": time.time() - t_machine})
        rows.append(met)
        log(f"  [{mi+1}/{len(machines)}] {mid} done ({D} channels) in {time.time()-t_machine:.1f}s "
            f"pr_auc={met['pr_auc']:.4f} auc_roc={met['auc_roc']:.4f} -- "
            f"total elapsed {(time.time()-t0)/60:.1f} min")
    df = pd.DataFrame(rows)
    df.to_csv(BASE / "kanad_smd_per_machine.csv", index=False)
    log(f"SMD done: {len(df)} machines, total {(time.time()-t0)/60:.1f} min")
    return df


# ======================================================== SMAP / MSL ==
def run_smap_msl():
    out_dir = CKPT_DIR / "smap_msl"
    out_dir.mkdir(exist_ok=True)
    meta_all = load_smap_msl_meta(SMAP_MSL_LABELS)
    channel_ids = sorted(meta_all.keys())
    log(f"SMAP/MSL: {len(channel_ids)} channels (column 0 only, matching TVL-D's convention)")
    rows = []
    skipped = []
    t0 = time.time()
    for i, cid in enumerate(channel_ids):
        train, test, label = load_smap_msl_channel(cid, meta_all, SMAP_MSL_DATA_DIR)
        if label.sum() == 0:
            skipped.append(cid)
            continue
        cache_path = out_dir / f"{cid}.pkl"
        scores, aligned_label, meta = cached_channel_run(cache_path, train[:, 0], test[:, 0], label)
        if scores is None:
            log(f"  [{i+1}/{len(channel_ids)}] {cid} SKIPPED ({meta.get('skipped')})")
            continue
        met = full_metrics(scores, aligned_label)
        if met is None:
            continue
        met.update({"id": cid, "spacecraft": meta_all[cid]["spacecraft"], "method": "KAN-AD",
                    "elapsed": meta["elapsed"]})
        rows.append(met)
        if (i + 1) % 10 == 0 or (i + 1) == len(channel_ids):
            log(f"  [{i+1}/{len(channel_ids)}] {cid} done -- total elapsed {(time.time()-t0)/60:.1f} min")
    if skipped:
        log(f"Skipped (no labeled anomaly in test set): {skipped}")
    df = pd.DataFrame(rows)
    df.to_csv(BASE / "kanad_smap_msl_per_channel.csv", index=False)
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
    D = train.shape[1]
    log(f"PSM: train {train.shape}, test {test.shape}, anomaly frac {label.mean():.4f}")

    t0 = time.time()
    per_dim_scores = []
    aligned_label = None
    for d in range(D):
        cache_path = out_dir / f"dim{d:02d}.pkl"
        scores, al, meta = cached_channel_run(cache_path, train[:, d], test[:, d], label)
        if scores is None:
            continue
        per_dim_scores.append(scores)
        aligned_label = al
        log(f"  dim {d+1}/{D} done in {meta['elapsed']:.1f}s -- total elapsed {(time.time()-t0)/60:.1f} min")

    agg = np.max(np.stack(per_dim_scores, axis=0), axis=0)
    met = full_metrics(agg, aligned_label)
    met.update({"id": "PSM", "method": "KAN-AD", "D": D, "elapsed": time.time() - t0})
    log(f"PSM done in {(time.time()-t0)/60:.1f} min: pr_auc={met['pr_auc']:.4f} auc_roc={met['auc_roc']:.4f}")
    df = pd.DataFrame([met])
    df.to_csv(BASE / "kanad_psm_results.csv", index=False)
    return df


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("all", "smap_msl"):
        run_smap_msl()
    if which in ("all", "psm"):
        run_psm()
    if which in ("all", "smd"):
        run_smd()
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")
