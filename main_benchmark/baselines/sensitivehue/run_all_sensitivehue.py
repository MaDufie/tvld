"""Orchestrate SensitiveHUE (official vendored Transformer+RevIN+MTS-NLL
model, our own outer train/eval loop.

SensitiveHUE's self-attention is over the TIME axis within each window (each
timestep is a token, its feature vector spans all channels), so like every
other baseline in this project it is run per-entity on the FULL local
feature array, not per-channel:
  - SMD: 28 machines, each on its full 38-channel array.
  - PSM: single dataset, full 25-channel array.
  - SMAP/MSL: each channel on its FULL local feature array (telemetry +
    one-hot command columns).

Each entity trains+scores in its OWN subprocess (run_one_sensitivehue.py),
the same isolation pattern used for every prior baseline.

Usage (run from anywhere, or from this directory):
    python run_all_sensitivehue.py            # runs SMD + SMAP/MSL + PSM
    python run_all_sensitivehue.py smd        # just SMD
    python run_all_sensitivehue.py smap_msl   # just SMAP/MSL
    python run_all_sensitivehue.py psm        # just PSM
"""
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))  # for tvld_final.py

from tvld_final import full_metrics, load_smap_msl_meta  # noqa: E402

BASE = HERE
CKPT_DIR = BASE / "checkpoints_sensitivehue"
CKPT_DIR.mkdir(exist_ok=True)

SMD_DATA_DIR = Path(os.environ.get("SMD_DATA_DIR", str(HERE.parent.parent / "data")))
SMAP_MSL_LABELS = Path(os.environ.get("SMAP_MSL_LABELS", str(HERE.parent.parent / "data" / "labeled_anomalies.csv")))

RUNNER = HERE / "run_one_sensitivehue.py"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one(dataset, entity_id, cache_path, timeout=1800, retries=3):
    if cache_path.exists():
        return True
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                ["python3", "-u", str(RUNNER), dataset, entity_id, str(cache_path)],
                timeout=timeout, capture_output=True, text=True,
            )
            if result.returncode == 0 and cache_path.exists():
                return True
            log(f"  SUBPROCESS FAILED ({dataset}/{entity_id}) attempt {attempt}/{retries} "
                f"rc={result.returncode}: "
                f"{result.stderr.strip().splitlines()[-3:] if result.stderr else 'no stderr'}")
        except subprocess.TimeoutExpired:
            log(f"  SUBPROCESS TIMEOUT ({dataset}/{entity_id}) attempt {attempt}/{retries} after {timeout}s")
    return False


def load_cached(cache_path):
    with open(cache_path, "rb") as f:
        d = pickle.load(f)
    return d["scores"], d["label"], d["meta"]


# ============================================================== SMD ==
def run_smd():
    out_dir = CKPT_DIR / "smd"
    out_dir.mkdir(exist_ok=True)
    machines = sorted(p.stem for p in (SMD_DATA_DIR / "train").glob("machine-*.txt"))
    log(f"SMD: {len(machines)} machines")
    rows = []
    t0 = time.time()
    failed = []
    for i, mid in enumerate(machines):
        cache_path = out_dir / f"{mid}.pkl"
        ok = run_one("smd", mid, cache_path)
        if not ok:
            failed.append(mid)
            continue
        scores, aligned_label, meta = load_cached(cache_path)
        met = full_metrics(scores, aligned_label)
        if met is None:
            log(f"  [{i+1}/{len(machines)}] {mid} SKIPPED (degenerate labels)")
            continue
        met.update({"id": mid, "method": "SensitiveHUE", "elapsed": meta["elapsed"]})
        rows.append(met)
        log(f"  [{i+1}/{len(machines)}] {mid} done in {meta['elapsed']:.1f}s "
            f"pr_auc={met['pr_auc']:.4f} auc_roc={met['auc_roc']:.4f} -- "
            f"total elapsed {(time.time()-t0)/60:.1f} min")
    df = pd.DataFrame(rows)
    df.to_csv(BASE / "sensitivehue_smd_per_machine.csv", index=False)
    log(f"SMD done: {len(df)} machines, total {(time.time()-t0)/60:.1f} min"
        + (f" -- FAILED (will retry next pass): {failed}" if failed else ""))
    return df, failed


# ======================================================== SMAP / MSL ==
def run_smap_msl():
    out_dir = CKPT_DIR / "smap_msl"
    out_dir.mkdir(exist_ok=True)
    meta_all = load_smap_msl_meta(SMAP_MSL_LABELS)
    channel_ids = sorted(meta_all.keys())
    log(f"SMAP/MSL: {len(channel_ids)} channels (full multivariate array per channel)")
    rows = []
    skipped = []
    failed = []
    t0 = time.time()
    for i, cid in enumerate(channel_ids):
        sc = meta_all[cid]["spacecraft"]
        dataset_key = "smap" if sc == "SMAP" else "msl"
        cache_path = out_dir / f"{cid}.pkl"
        ok = run_one(dataset_key, cid, cache_path)
        if not ok:
            failed.append(cid)
            continue
        scores, aligned_label, meta = load_cached(cache_path)
        met = full_metrics(scores, aligned_label)
        if met is None:
            log(f"  [{i+1}/{len(channel_ids)}] {cid} SKIPPED (degenerate labels)")
            skipped.append(cid)
            continue
        met.update({"id": cid, "spacecraft": sc, "method": "SensitiveHUE", "elapsed": meta["elapsed"]})
        rows.append(met)
        if (i + 1) % 10 == 0 or (i + 1) == len(channel_ids):
            log(f"  [{i+1}/{len(channel_ids)}] {cid} done -- total elapsed {(time.time()-t0)/60:.1f} min")
    if skipped:
        log(f"Skipped: {skipped}")
    df = pd.DataFrame(rows)
    df.to_csv(BASE / "sensitivehue_smap_msl_per_channel.csv", index=False)
    log(f"SMAP/MSL done: {len(df)} channels, total {(time.time()-t0)/60:.1f} min"
        + (f" -- FAILED (will retry next pass): {failed}" if failed else ""))
    return df, failed


# ============================================================== PSM ==
def run_psm():
    out_dir = CKPT_DIR / "psm"
    out_dir.mkdir(exist_ok=True)
    cache_path = out_dir / "psm.pkl"
    t0 = time.time()
    ok = run_one("psm", "PSM", cache_path, timeout=2400, retries=5)
    if not ok:
        log("PSM FAILED (subprocess did not complete after retries)")
        return None, True
    scores, aligned_label, meta = load_cached(cache_path)
    met = full_metrics(scores, aligned_label)
    met.update({"id": "PSM", "method": "SensitiveHUE", "elapsed": meta["elapsed"]})
    log(f"PSM done in {(time.time()-t0)/60:.1f} min: pr_auc={met['pr_auc']:.4f} auc_roc={met['auc_roc']:.4f}")
    df = pd.DataFrame([met])
    df.to_csv(BASE / "sensitivehue_psm_results.csv", index=False)
    return df, False


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    any_failed = False
    if which in ("all", "smd"):
        _, failed = run_smd()
        any_failed = any_failed or bool(failed)
    if which in ("all", "psm"):
        _, psm_failed = run_psm()
        any_failed = any_failed or psm_failed
    if which in ("all", "smap_msl"):
        _, failed = run_smap_msl()
        any_failed = any_failed or bool(failed)
    if any_failed:
        log(f"INCOMPLETE after {(time.time()-t0)/60:.1f} min -- some entities still missing, "
            f"supervisor will relaunch to retry them.")
        sys.exit(1)
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")
