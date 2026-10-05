"""TVL-D evaluation on SMD (28 machines).

Reuses the final-model severity/fusion pipeline in ../tvld_final.py
unmodified: same fixed weights (W_SHAPE=0.15, W_LEVEL=0.75, W_STRUCT=1.0),
same smoothing window (35), zero retuning. Every column is scored
(primary_cols=None) since all 38 SMD columns are genuine, correlated sensor
telemetry.

Usage:
    python run_smd_eval.py
Expects ./data/{train,test,test_label}/machine-*.txt (standard SMD layout)
next to this script.
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

SMD_DATA_DIR = Path(os.environ.get("SMD_DATA_DIR", str(HERE / "data")))
CKPT_DIR = HERE / "checkpoints_smd"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = HERE / "smd_results.csv"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [SMD] {msg}", flush=True)


def load_smd_machine(mid, data_dir):
    train = pd.read_csv(data_dir / "train" / f"{mid}.txt", header=None).to_numpy(dtype=float)
    test = pd.read_csv(data_dir / "test" / f"{mid}.txt", header=None).to_numpy(dtype=float)
    label = pd.read_csv(data_dir / "test_label" / f"{mid}.txt", header=None).to_numpy(dtype=int).ravel()
    return train, test, label


def severities(mid, train, test):
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


def main():
    machines = sorted(p.stem for p in (SMD_DATA_DIR / "train").glob("machine-*.txt"))
    log(f"Found {len(machines)} SMD machines in {SMD_DATA_DIR}")
    if not machines:
        log(f"No machine-*.txt files under {SMD_DATA_DIR}/train -- check SMD_DATA_DIR / the data/ layout.")
        return

    rows = []
    t_start = time.time()
    for i, mid in enumerate(machines):
        train, test, label = load_smd_machine(mid, SMD_DATA_DIR)
        shape_sev, level_sev, struct_sev = severities(mid, train, test)
        v2, v3 = fuse(shape_sev, level_sev, struct_sev, primary_cols=None)

        for method, scores in [("TVL-D (pre-smoothing)", v2), ("TVL-D", v3)]:
            met = full_metrics(scores, label)
            if met is None:
                continue
            met.update({"machine": mid, "method": method, "D": train.shape[1], "test_len": test.shape[0]})
            rows.append(met)

        log(f"[{i+1}/{len(machines)}] {mid} done -- {(time.time()-t_start)/60:.1f} min elapsed")
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    df = pd.DataFrame(rows)
    cols = ["pr_auc", "auc_roc", "auc_pr_t", "f1", "precision", "recall"]
    summary = df.groupby("method")[cols].mean().reset_index()
    summary.insert(1, "dataset", "SMD")
    print(summary.round(4).to_string(index=False))
    log(f"ALL DONE. Wrote {OUT_CSV}. Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
