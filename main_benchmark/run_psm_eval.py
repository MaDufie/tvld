"""TVL-D evaluation on PSM (Pooled Server Metrics, Abdulaal et al., KDD 2021)

PSM's 25 columns are all genuine, correlated continuous telemetry,
so every column is scored (primary_cols=None). This is a single pooled
train/test stream, so there's one result row per method rather than a macro-average 
over many series. PSM's training split has some missing values (concentrated in a 
handful of columns), linearly interpolated before computing any severity signal.

Usage:
    python run_psm_eval.py
Expects ./data/{train.csv,test.csv,test_label.csv} next to this script
(the standard PSM release layout, with a "timestamp_(min)" column plus
feature columns, and test_label.csv having a "label" column).
"""
import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for tvld_final.py
sys.path.insert(0, str(HERE))         # for metrics.py

from tvld_final import pick_window, shape_severity, level_severity, structure_severity, fuse  # noqa: E402
from metrics import full_metrics  # noqa: E402

PSM_DATA_DIR = Path(os.environ.get("PSM_DATA_DIR", str(HERE / "data")))
CKPT_PATH = HERE / "checkpoint_psm.pkl"
OUT_CSV = HERE / "psm_results.csv"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [PSM] {msg}", flush=True)


def load_psm(data_dir):
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    label_df = pd.read_csv(data_dir / "test_label.csv")
    feat_cols = [c for c in train_df.columns if c != "timestamp_(min)"]

    n_nan_before = train_df[feat_cols].isna().sum().sum()
    train_df[feat_cols] = train_df[feat_cols].interpolate(method="linear", limit_direction="both")
    log(f"train NaNs: {n_nan_before} -> {train_df[feat_cols].isna().sum().sum()} after linear interpolation")

    train = train_df[feat_cols].to_numpy(dtype=float)
    test = test_df[feat_cols].to_numpy(dtype=float)
    label = label_df["label"].to_numpy(dtype=int)
    return train, test, label


def severities(train, test):
    if CKPT_PATH.exists():
        with open(CKPT_PATH, "rb") as f:
            ck = pickle.load(f)
        return ck["shape_sev"], ck["level_sev"], ck["struct_sev"]
    m = pick_window(train.shape[0], test.shape[0])
    t0 = time.time()
    shape_sev = shape_severity(train, test, m)
    log(f"Shape done ({time.time()-t0:.1f}s)")
    level_sev = level_severity(train, test)
    struct_sev = structure_severity(train, test)
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CKPT_PATH, "wb") as f:
        pickle.dump({"shape_sev": shape_sev, "level_sev": level_sev, "struct_sev": struct_sev}, f)
    return shape_sev, level_sev, struct_sev


def main():
    if not (PSM_DATA_DIR / "train.csv").exists():
        log(f"train.csv not found under {PSM_DATA_DIR} -- check PSM_DATA_DIR / the data/ layout.")
        return
    train, test, label = load_psm(PSM_DATA_DIR)
    log(f"train {train.shape}, test {test.shape}, anomaly frac {label.mean():.4f}")

    shape_sev, level_sev, struct_sev = severities(train, test)
    v2, v3 = fuse(shape_sev, level_sev, struct_sev, primary_cols=None)

    rows = []
    for method, scores in [("TVL-D (pre-smoothing)", v2), ("TVL-D", v3)]:
        met = full_metrics(scores, label)
        met.update({"method": method, "D": train.shape[1], "test_len": test.shape[0]})
        rows.append(met)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    cols = ["method", "pr_auc", "auc_roc", "auc_pr_t", "f1", "precision", "recall"]
    print(df[cols].round(4).to_string(index=False))
    log(f"ALL DONE. Wrote {OUT_CSV}.")


if __name__ == "__main__":
    main()
