"""Train+score TopoGDN on ONE entity (one SMD machine, one SMAP/MSL channel,
or PSM) in its own fresh process, then exit.

Usage: python3 run_one_topogdn.py <dataset> <entity_id_or__> <cache_path>
  dataset in {smd, smap, msl, psm}
"""
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent.parent.parent  
TOPOGDN_SRC = PACKAGE_ROOT / "baselines_src" / "TopoGDN"

try:
    import torch_scatter  # noqa: F401  
except ImportError:
    sys.path.insert(0, str(TOPOGDN_SRC / "torch_scatter_shim")) 

sys.path.insert(0, str(PACKAGE_ROOT))  # for tvld_final.py
sys.path.insert(0, str(HERE))          # for topogdn_runner.py

from tvld_final import load_smd_machine, load_smap_msl_meta, load_smap_msl_channel
from topogdn_runner import run_topogdn
import pandas as pd

SMD_DATA_DIR = Path(os.environ.get("SMD_DATA_DIR", str(HERE / "data")))
SMAP_MSL_DATA_DIR = Path(os.environ.get("SMAP_MSL_DATA_DIR", str(HERE / "data")))
SMAP_MSL_LABELS = Path(os.environ.get("SMAP_MSL_LABELS", str(HERE / "labeled_anomalies.csv")))
PSM_DATA_DIR = Path(os.environ.get("PSM_DATA_DIR", str(HERE / "data")))


COMMON_KWARGS = dict(win_size=100, stride=10, batch=32, dim=128, out_layer_num=1,
                      out_layer_inter_dim=64, topk=15, agg_topk=1, val_ratio=0.02,
                      use_topo=True, decay=0, seed=5, early_stop_win=15)


SMD_KWARGS = dict(epoch=25, **COMMON_KWARGS)
MSL_KWARGS = dict(epoch=25, **COMMON_KWARGS)
SMAP_KWARGS = dict(epoch=25, **COMMON_KWARGS)
PSM_KWARGS = dict(epoch=15, **COMMON_KWARGS)


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

    t0 = time.time()
    scores, aligned_label, meta = run_topogdn(train, test, label, tag=tag, verbose=True, **kwargs)
    with open(cache_path, "wb") as f:
        pickle.dump({"scores": scores, "label": aligned_label, "meta": meta}, f)
    print(f"[run_one_topogdn] {dataset}/{entity_id} cached to {cache_path} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
