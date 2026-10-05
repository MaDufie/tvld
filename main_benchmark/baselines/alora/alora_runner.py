"""ALoRa (ALoRaT) train+eval runner for a SINGLE multivariate entity 
(machine / channel / dataset), replicating the official
repo's model + training objective + scoring formula (model/ALoRa.py,
solver.py) on our own arrays.

ALoRa ("Low Rank Transformer for Multivariate Time Series Anomaly Detection
and Localization", ICLR 2026) is a genuinely MULTIVARIATE transformer:
  - Embedding: instead of embedding each channel independently, it embeds
    PAIRS of channels (w1*smooth(x_i) + w2*smooth(x_j)) for the top-K most
    correlated pairs (by train-only Spearman rho), capped at top_k_limit=512
    dims. If C(D,2) <= 512 it just uses all pairs.
  - Encoder: 3-layer transformer whose self-attention is over TIME (a T x T
    matrix per window, T=win_size), with an explicit low-rank regularizer
    (TNN+Geman penalty on the attention matrix's singular values, skipping
    the top r=1) added to the reconstruction loss during training. This
    encourages each layer's temporal self-attention to look "rank-1-ish" on
    normal data.
  - Scoring: anomaly score = (last-timestep reconstruction MSE) x (count of
    singular values of the last layer's attention matrix above a threshold,
    i.e. its numerical "rank"). An anomalous window is expected to BOTH
    reconstruct poorly AND break the learned low-rank attention structure.

"""
import copy
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent

PACKAGE_ROOT = HERE.parent.parent.parent
ALORA_SRC = PACKAGE_ROOT / "baselines_src" / "ALoRa"
sys.path.insert(0, str(ALORA_SRC))

from model.ALoRa import ALoRaT  # noqa: E402

WIN_SIZE = 20
D_MODEL_REQUEST = 520  
TOP_K_LIMIT = 512
BATCH_SIZE = 128
MAX_EPOCHS = 4
LR = 1e-4
PATIENCE = 3
DELTA = 0.0
VAL_FRAC = 0.2
LAMBDA_REG = 10.0
RANK_R = 1
RANK_GAMMA = 1.0
PAIRS_CACHE_DIR = HERE / "pairs_cache_alora"


def _spearman_top_pairs(train_raw, top_k=TOP_K_LIMIT):
    """Top-K feature pairs by |Spearman rho|, TRAIN-only -- mirrors
    data_factory/Preprocess/Preprocess.py's compute_spearman_matrix_from_array
    + top_k_pairs_by_abs_rho exactly."""
    Xr = np.apply_along_axis(rankdata, 0, train_raw)
    std = Xr.std(axis=0, ddof=1)
    std[std < 1e-12] = 1.0
    Xr = (Xr - Xr.mean(axis=0)) / std
    R = (Xr.T @ Xr) / (Xr.shape[0] - 1)
    C = R.shape[0]
    iu, ju = np.triu_indices(C, k=1)
    rho = R[iu, ju]
    order = np.argsort(-np.abs(rho))
    K = min(top_k, len(rho))
    pairs_idx = np.stack([iu[order][:K], ju[order][:K]], axis=1).astype(np.int64)
    return pairs_idx


def _low_rank_loss(SA_avg, r=RANK_R, gamma=RANK_GAMMA):
    """TNN + Geman penalty, skipping the top r singular values -- matches
    solver.py's calculate_low_rank_loss(reg_type='tnn_geman', r=1, gamma=1.0)."""
    sv = th.linalg.svdvals(SA_avg)
    sv_trunc = sv[:, r:]
    return th.sum(sv_trunc / (sv_trunc + gamma))


def _make_windows(x, win_size):
    """Vectorized windowing via a strided view + a SINGLE contiguous copy.
    `x` must already be float32"""
    N, D = x.shape
    n = N - win_size + 1
    if n <= 0:
        return None, 0
    w = np.lib.stride_tricks.sliding_window_view(x, window_shape=(win_size, D))
    return np.ascontiguousarray(w[:, 0, :, :]), n


def run_alora(train, test, label, win_size=WIN_SIZE, batch_size=BATCH_SIZE,
              max_epochs=MAX_EPOCHS, lr=LR, patience=PATIENCE, val_frac=VAL_FRAC,
              lambda_reg=LAMBDA_REG, rank_threshold=0.01, d_model_request=D_MODEL_REQUEST,
              top_k_limit=TOP_K_LIMIT, e_layers=3, n_heads=8, tag="run", verbose=False):
    t0 = time.time()
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    label = np.asarray(label, dtype=int).ravel()
    N, D = train.shape

    split_idx = int(N * val_frac)
    if split_idx <= win_size or (N - split_idx) <= win_size or (test.shape[0]) <= win_size:
        return None, None, {"skipped": "series too short to split/window", "elapsed": 0.0}

    tr_raw, va_raw = train[:-split_idx], train[-split_idx:]
    total_pairs = D * (D - 1) // 2
    dataset_tag, precomputed_dir = None, None
    if total_pairs > top_k_limit:
        pairs_idx = _spearman_top_pairs(tr_raw, top_k=top_k_limit)
        PAIRS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dataset_tag = tag
        np.save(str(PAIRS_CACHE_DIR / f"pairs_idx_{dataset_tag}.npy"), pairs_idx)
        precomputed_dir = str(PAIRS_CACHE_DIR)

    mu, sd = tr_raw.mean(axis=0), tr_raw.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    tr = ((tr_raw - mu) / sd).astype(np.float32)
    va = ((va_raw - mu) / sd).astype(np.float32)
    te = ((test - mu) / sd).astype(np.float32)
    del tr_raw, va_raw, train, test

    Xtr, n_tr = _make_windows(tr, win_size)
    Xva, n_va = _make_windows(va, win_size)
    Xte, n_te = _make_windows(te, win_size)
    del tr, va, te
    if Xtr is None or Xva is None or Xte is None:
        return None, None, {"skipped": "series too short for window", "elapsed": 0.0}

    train_loader = DataLoader(TensorDataset(th.from_numpy(Xtr)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(th.from_numpy(Xva)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(th.from_numpy(Xte)), batch_size=batch_size, shuffle=False)

    model_kwargs = dict(win_size=win_size, d_model=d_model_request, enc_in=D, c_out=D,
                         e_layers=e_layers, n_heads=n_heads, top_k_limit=top_k_limit,
                         token_kernel_size=3)
    if dataset_tag is not None:
        model_kwargs.update(dataset=dataset_tag, precomputed_dir=precomputed_dir)

    model = ALoRaT(**model_kwargs)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = np.inf
    best_state = None
    counter = 0
    epochs_run = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for bi, (xb,) in enumerate(train_loader):
            optimizer.zero_grad()
            out, SA, _ = model(xb)
            rec_loss = criterion(out, xb)
            reg = 0.0
            for u in range(len(SA)):
                SA_avg = SA[u].mean(dim=1)
                reg = reg + _low_rank_loss(SA_avg)
            reg = reg / len(SA)
            loss = rec_loss + lambda_reg * reg
            loss.backward()
            optimizer.step()
            if bi % 50 == 0:
                del out, SA, rec_loss, reg, loss
                gc.collect()

        gc.collect()
        model.eval()
        val_tot, val_n = 0.0, 0
        with th.no_grad():
            for (xb,) in val_loader:
                out, _, _ = model(xb)
                val_tot += criterion(out, xb).item()
                val_n += 1
        val_loss = val_tot / max(val_n, 1)
        epochs_run = epoch
        if verbose:
            print(f"    epoch {epoch}: val_loss={val_loss:.6f}")

        if val_loss < best_val - DELTA:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    rec_scores, rank_scores = [], []
    with th.no_grad():
        for (xb,) in test_loader:
            out, SA, _ = model(xb)
            last_in = xb[:, -1, :]
            last_out = out[:, -1, :]
            rec = ((last_in - last_out) ** 2).mean(dim=-1).numpy()
            rec_scores.append(rec)
            SA_last = SA[-1].mean(dim=1)
            sv = th.linalg.svdvals(SA_last).numpy()
            rank = (sv > rank_threshold).sum(axis=1)
            rank_scores.append(rank)
    rec_scores = np.concatenate(rec_scores)
    rank_scores = np.concatenate(rank_scores).astype(float)
    scores = rec_scores * rank_scores
    scores[np.isnan(scores)] = 0.0
    aligned_label = label[win_size - 1:]

    elapsed = time.time() - t0
    meta = {"epochs_trained": epochs_run, "elapsed": elapsed, "D": D,
            "d_eff": model.d_model, "n_train_windows": n_tr, "n_test_windows": len(scores),
            "rank_threshold": rank_threshold, "win_size": win_size}
    if verbose:
        print(f"  done in {elapsed:.1f}s, D={D}, d_eff={model.d_model}, epochs={epochs_run}, "
              f"train_windows={n_tr}, test_windows={len(scores)}")
    return scores, aligned_label, meta
