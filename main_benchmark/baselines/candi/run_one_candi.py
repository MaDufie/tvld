"""Train+score CANDI on ONE entity (one SMD machine, one SMAP/MSL channel, or
PSM) in its own fresh process, then exit.

Usage: python3 run_one_candi.py <dataset> <entity_id_or__> <cache_path>
  dataset in {smd, smap, msl, psm}
"""
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))  # for tvld_final.py
sys.path.insert(0, str(HERE))          # for candi_runner.py 

from tvld_final import load_smd_machine, load_smap_msl_meta, load_smap_msl_channel  # noqa: E402
from candi_runner import run_candi  # noqa: E402
import pandas as pd

SMD_DATA_DIR = Path(os.environ.get("SMD_DATA_DIR", str(HERE.parent.parent / "data")))
SMAP_MSL_DATA_DIR = Path(os.environ.get("SMAP_MSL_DATA_DIR", str(HERE.parent.parent / "data")))
SMAP_MSL_LABELS = Path(os.environ.get("SMAP_MSL_LABELS", str(HERE.parent.parent / "data" / "labeled_anomalies.csv")))
PSM_DATA_DIR = Path(os.environ.get("PSM_DATA_DIR", str(HERE.parent.parent / "data")))


COMMON_KWARGS = dict(win_size=50, batch_size=256, latent_dim=64, lr=1e-3,
                      ttlr=0.01, th_pctl=0.95, val_frac=0.2, gating_init=0.5,
                      tta_steps=1, patience=5)
SMD_KWARGS = dict(epochs=30, **COMMON_KWARGS)
MSL_KWARGS = dict(epochs=30, **COMMON_KWARGS)
SMAP_KWARGS = dict(epochs=30, **COMMON_KWARGS)
PSM_KWARGS = dict(epochs=15, **COMMON_KWARGS)

ENTITY_OVERRIDES = {
    ("msl", "P-10"): dict(batch_size=64),
}


def main():
    dataset, entity_id, cache_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset == "smd":
        train, test, label = load_smd_machine(entity_id, SMD_DATA_DIR)
        kwargs, tag = SMD_KWARGS, f"SMD_{entity_id}"
    elif dataset in ("smap", "msl"):
        meta_all = load_smap_msl_meta(SMAP_MSL_LABELS)
        train, test, label = load_smap_msl_channel(entity_id, meta_all, SMAP_MSL_DATA_DIR)
        kwargs = SMAP_KWARGS if dataset == "smap" else MSL_KWARGS
        tag = f"{dataset.upper()}_{entity_id}"
    elif dataset == "psm":
        train_df = pd.read_csv(PSM_DATA_DIR / "train.csv")
        test_df = pd.read_csv(PSM_DATA_DIR / "test.csv")
        label_df = pd.read_csv(PSM_DATA_DIR / "test_label.csv")
        feat_cols = [c for c in train_df.columns if c != "timestamp_(min)"]
        train_df[feat_cols] = train_df[feat_cols].interpolate(method="linear", limit_direction="both")
        test_df[feat_cols] = test_df[feat_cols].interpolate(method="linear", limit_direction="both")
        train = train_df[feat_cols].to_numpy(dtype=float)
        test = test_df[feat_cols].to_numpy(dtype=float)
        label = label_df["label"].to_numpy(dtype=int)
        kwargs, tag = PSM_KWARGS, "PSM"
    else:
        raise ValueError(f"unknown dataset {dataset}")

    override = ENTITY_OVERRIDES.get((dataset, entity_id))
    if override:
        kwargs = {**kwargs, **override}
        print(f"[run_one_candi] applying override for {dataset}/{entity_id}: {override}", flush=True)

    t0 = time.time()
    scores, aligned_label, meta = run_candi(train, test, label, tag=tag, verbose=True, **kwargs)
    with open(cache_path, "wb") as f:
        pickle.dump({"scores": scores, "label": aligned_label, "meta": meta}, f)
    print(f"[run_one_candi] {dataset}/{entity_id} cached to {cache_path} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
