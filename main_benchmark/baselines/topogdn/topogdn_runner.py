"""Train+score TopoGDN on ONE entity's full (T, D) numpy arrays, reusing the
official vendored model/training code vendored into ../../../baselines_src/TopoGDN
(models.GDN.GDN, datasets.TimeDataset.TimeDataset, train.train, test.test,
evaluate.get_full_err_scores) rather than reimplementing the architecture.

WHAT TopoGDN IS
----------------
TopoGDN extends GDN (Deng & Hooi, AAAI 2021, "Graph Neural Network-Based
Anomaly Detection in Multivariate Time Series") -- a forecasting-based
detector -- with an optional topological-pooling layer borrowed from
"Topological Graph Neural Networks" (Horn et al., ICLR 2022). Per-sensor
windows are embedded via a learned per-node embedding, a top-k
cosine-similarity sparsified attention graph is built over the sensors each
forward pass (GDN's own graph-structure-learning mechanism), a custom
graph-attention layer (GraphLayer, MessagePassing-based) aggregates
neighbor information, and -- when enabled -- a TopologyLayer computes
persistent homology (0-dimensional, vertex + edge-max filtration, 8 learned
filtration functions) over the just-learned sparse graph and folds a small
per-node topological summary vector back into the node representations
before the final forecasting head. The anomaly score is the per-sensor
forecasting error, robust-normalized (median/IQR) and smoothed, aggregated
across sensors by taking the (by default) single highest-scoring sensor at
each timestep -- GDN's own original "top-1" deviation scoring.
"""
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent.parent.parent
TOPOGDN_SRC = PACKAGE_ROOT / "baselines_src" / "TopoGDN"

try:
    import torch_scatter  # noqa: F401  
except ImportError:
    sys.path.insert(0, str(TOPOGDN_SRC / "torch_scatter_shim")) 
sys.path.insert(0, str(TOPOGDN_SRC))

from util.env import set_device, get_device
from datasets.TimeDataset import TimeDataset
from models.GDN import GDN
from train import train as vendor_train
from test import test as vendor_test
from evaluate import get_full_err_scores


def _build_fc_edge_index(num_nodes):
    """Fully-connected candidate graph over `num_nodes` sensors"""
    src, dst = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def _to_indata(arr, labels):
    """Replicates util.preprocess.construct_data()'s output shape: a list of
    D per-channel value lists, with the label list appended last."""
    D = arr.shape[1]
    res = [arr[:, i].tolist() for i in range(D)]
    res.append(list(labels))
    return res


def run_topogdn(train_arr, test_arr, label, win_size=100, stride=10, batch=16,
                 epoch=150, dim=128, out_layer_num=1, out_layer_inter_dim=64,
                 topk=15, agg_topk=1, val_ratio=0.02, use_topo=True, decay=0,
                 seed=5, early_stop_win=15, tag="run", verbose=False,
                 scratch_dir_base=None):
    t0 = time.time()

    np.random.seed(seed)
    torch.manual_seed(seed)

    set_device(torch.device("cpu"))
    device = get_device()

    D = train_arr.shape[1]

    mu = train_arr.mean(axis=0, keepdims=True)
    sigma = train_arr.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    train_norm = (train_arr - mu) / sigma
    test_norm = (test_arr - mu) / sigma

    fc_edge_index = _build_fc_edge_index(D)

    train_indata = _to_indata(train_norm, [0] * len(train_norm))
    test_indata = _to_indata(test_norm, label.tolist())

    cfg = {"slide_win": win_size, "slide_stride": stride}
    train_dataset = TimeDataset(train_indata, fc_edge_index, mode="train", config=cfg)
    test_dataset = TimeDataset(test_indata, fc_edge_index, mode="test", config=cfg)

   
    from torch.utils.data import DataLoader, Subset
    dataset_len = len(train_dataset)
    if dataset_len < 2:
        train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        val_dataloader = None
    else:
        val_use_len = max(1, int(dataset_len * val_ratio))
        val_use_len = min(val_use_len, dataset_len - 1)
        train_use_len = dataset_len - val_use_len
        import random as _random
        _random.seed(seed)
        val_start_index = _random.randrange(train_use_len)
        indices = torch.arange(dataset_len)
        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
        val_sub_indices = indices[val_start_index:val_start_index + val_use_len]

        train_subset = Subset(train_dataset, train_sub_indices)
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    model = GDN([fc_edge_index], D, dim=dim, input_dim=win_size,
                out_layer_num=out_layer_num, out_layer_inter_dim=out_layer_inter_dim,
                topk=topk, use_topo=use_topo).to(device)


    scratch_base = Path(scratch_dir_base) if scratch_dir_base else (HERE / "scratch")
    scratch_base.mkdir(parents=True, exist_ok=True)
    scratch_dir = tempfile.mkdtemp(prefix=f"topogdn_{tag}_", dir=str(scratch_base))
    save_path = os.path.join(scratch_dir, "best.pt")
    prev_cwd = os.getcwd()
    os.chdir(scratch_dir)
    try:
        train_config = {"seed": seed, "decay": decay, "epoch": epoch}
        vendor_train(model, save_path, config=train_config,
                     train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                     feature_map=[], test_dataloader=None, test_dataset=None,
                     train_dataset=train_dataset, dataset_name=tag)
    finally:
        os.chdir(prev_cwd)

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    _, test_result = vendor_test(model, test_dataloader)
    if val_dataloader is not None:
        _, val_result = vendor_test(model, val_dataloader)
    else:
        val_result = test_result

    test_scores, _normal_scores = get_full_err_scores(test_result, val_result)
    test_scores = np.nan_to_num(test_scores, nan=0.0, posinf=0.0, neginf=0.0)

    total_features = test_scores.shape[0]
    k = min(agg_topk, total_features)
    topk_indices = np.argpartition(test_scores, range(total_features - k, total_features), axis=0)[-k:]
    agg_scores = np.sum(np.take_along_axis(test_scores, topk_indices, axis=0), axis=0)

    aligned_label = np.asarray(label)[win_size:]
    n = min(len(agg_scores), len(aligned_label))
    agg_scores = agg_scores[:n]
    aligned_label = aligned_label[:n]

    meta = {
        "elapsed": time.time() - t0,
        "D": D,
        "n_train_windows": len(train_dataset),
        "n_test_windows": len(test_dataset),
        "win_size": win_size,
        "use_topo": use_topo,
    }
    if verbose:
        print(f"[topogdn_runner] {tag}: D={D} elapsed={meta['elapsed']:.1f}s "
              f"n_train_win={meta['n_train_windows']} n_test_win={meta['n_test_windows']}", flush=True)

    return agg_scores, aligned_label, meta
