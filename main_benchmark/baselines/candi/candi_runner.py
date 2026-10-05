"""CANDI train+eval runner for a SINGLE multivariate entity
(machine / channel / dataset), reusing the official model/training/TTA
classes from the authors' own TSB-AD-embedded implementation
(CANDI/TSB-AD/TSB_AD/models/CANDI.py).

CANDI ("Curated Test-Time Adaptation for Multivariate Time-Series Anomaly
Detection Under Distribution Shift", AAAI 2026) is NOT a new detector
architecture -- it is a test-time-adaptation (TTA) wrapper around a simple
frozen reconstruction autoencoder (a flatten+MLP encoder/decoder):
  - Base model: MLP autoencoder over a flattened (win_size * n_channels)
    window. Anomaly score = mean squared reconstruction error of a window.
  - After ordinary offline training (this base model is trained exactly
    like any other reconstruction AE), a threshold tau is set to the 95th
    percentile of TRAIN reconstruction error.
  - At TEST time, CANDI streams through the test set batch by batch. For
    each batch, it uses Mahalanobis distance (in the base model's learned
    latent space, with covariance estimated from a curated VALIDATION
    reference set) to decide whether each window "looks like" a past
    false-positive-prone normal window (top-5th-percentile-scoring val
    windows = the "hard" reference set) or a typical/moderate normal window
    (Q1-Q3-scoring val windows = the "moderate" reference set) -- combined
    with whether the window's OWN current reconstruction score is above/
    below tau. Windows that match either reference set accumulate into a
    buffer; once >=16 such windows have been seen, a few (tta_steps) SGD
    steps update a small residual adapter called SANA (Spatiotemporally-
    Aware Normality Adaptation: per-channel dilated-TCN embedding -> a
    cross-channel Transformer encoder -> per-channel linear projection ->
    elementwise *tanh(gating)) while the base MLP's own weights stay
    completely frozen. Two SANA instances are learned: sana_in (added to
    the raw input before encoding) and sana_out (subtracted from the
    reconstructed output). This lets the score account for a session's own
    idiosyncratic normal patterns (distribution shift) without ever seeing
    a test label -- it is genuinely online/causal: a batch's reported score
    reflects adaptation from only *prior* batches, never itself or future
    ones.
"""
import copy
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent.parent.parent
CANDI_TSBAD_DIR = PACKAGE_ROOT / "baselines_src" / "CANDI" / "TSB-AD"
sys.path.insert(0, str(CANDI_TSBAD_DIR))
from TSB_AD.models.CANDI import MLP, MLP_Tester  # noqa: E402

WIN_SIZE = 50
BATCH_SIZE = 256
EPOCHS = 30
LATENT_DIM = 64
LR = 1e-3
TTLR = 0.01
TH_PCTL = 0.95
VAL_FRAC = 0.2
GATING_INIT = 0.5
TTA_STEPS = 5
PATIENCE = 5
DELTA = 1e-4


def _make_windows(x, win_size):
    """Vectorized windowing via a strided view + a SINGLE contiguous copy
    (float32 throughout) """
    N, D = x.shape
    n = N - win_size + 1
    if n <= 0:
        return None, 0
    w = np.lib.stride_tricks.sliding_window_view(x, window_shape=(win_size, D))
    return np.ascontiguousarray(w[:, 0, :, :]), n


def run_candi(train, test, label, win_size=WIN_SIZE, batch_size=BATCH_SIZE,
              epochs=EPOCHS, latent_dim=LATENT_DIM, lr=LR, ttlr=TTLR,
              th_pctl=TH_PCTL, val_frac=VAL_FRAC, gating_init=GATING_INIT,
              tta_steps=TTA_STEPS, patience=PATIENCE, delta=DELTA,
              tag="run", verbose=False):
    t0 = time.time()
    device = torch.device("cpu")

    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    label = np.asarray(label, dtype=int).ravel()
    N, D = train.shape

    split_idx = int(N * val_frac)
    if split_idx <= win_size or (N - split_idx) <= win_size or test.shape[0] <= win_size:
        return None, None, {"skipped": "series too short to split/window", "elapsed": 0.0}

    tr_raw, va_raw = train[:-split_idx], train[-split_idx:]
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

    Xtr_t = torch.from_numpy(Xtr)
    Xva_t = torch.from_numpy(Xva)
    Xte_t = torch.from_numpy(Xte)

    train_loader = DataLoader(TensorDataset(Xtr_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(Xte_t), batch_size=batch_size, shuffle=False)

    model = MLP(seq_len=win_size, num_channels=D, latent_space_size=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = np.inf
    best_state = None
    counter = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for bi, (xb,) in enumerate(train_loader):
            optimizer.zero_grad()
            xhat = model(xb)
            loss = F.mse_loss(xhat, xb)
            loss.backward()
            optimizer.step()
            if bi % 50 == 0:
                del xhat, loss
                gc.collect()
        gc.collect()

        model.eval()
        val_tot, val_n = 0.0, 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xhat = model(xb)
                val_tot += F.mse_loss(xhat, xb).item()
                val_n += 1
        val_loss = val_tot / max(val_n, 1)
        epochs_run = epoch
        if verbose:
            print(f"    epoch {epoch}: val_loss={val_loss:.6f}", flush=True)

        if val_loss < best_val - delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    del best_state
    gc.collect()

    model.eval()
    train_scores = model.get_anomaly_scores(Xtr_t)
    tau = float(np.quantile(train_scores, th_pctl))
    if verbose:
        print(f"    tau={tau:.6f}", flush=True)

    tester = MLP_Tester(
        model=model, train_loader=train_loader, valid_loader=val_loader,
        test_loader=test_loader, lr=ttlr, device=device,
        gating_init=gating_init, tta_steps=tta_steps,
    )
    anoscs = tester.online(test_loader, tau)
    scores = np.asarray(anoscs, dtype=float).reshape(-1)
    scores[np.isnan(scores)] = 0.0

    aligned_label = label[win_size - 1:]
    if len(aligned_label) != len(scores):
        m = min(len(aligned_label), len(scores))
        aligned_label = aligned_label[:m]
        scores = scores[:m]

    elapsed = time.time() - t0
    meta = {"epochs_trained": epochs_run, "elapsed": elapsed, "D": D,
            "n_train_windows": n_tr, "n_test_windows": len(scores),
            "win_size": win_size, "tau": tau}
    if verbose:
        print(f"  done in {elapsed:.1f}s, D={D}, epochs={epochs_run}, "
              f"train_windows={n_tr}, test_windows={len(scores)}", flush=True)
    return scores, aligned_label, meta
