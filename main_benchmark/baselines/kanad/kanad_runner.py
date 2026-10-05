"""KAN-AD train+eval runner for a SINGLE univariate channel, replicating
EasyTSAD's exact preprocessing/windowing/training conventions (TSData.differential,
TSData.z_score_norm, UTSOneByOneDataset, EarlyStoppingTorch) without going through the
full EasyTSAD Controller/file-based dataset machinery - so we can feed it our own
SMD/SMAP/MSL/PSM arrays directly.

KAN-AD is a UNIVARIATE method (one series in, one series out). To apply it to our
multivariate datasets we run it independently per feature channel and the caller
aggregates (see run_all_kanad.py for the aggregation policy).

Note: KAN-AD/kanad/kanad.py imports `EasyTSAD` at module level (for the
`BaseMethod`/`TSData` types referenced in its unused-here `KANAD` training-
loop class); only `KANADModel` (the actual network) is used below, but the
import still requires `pip install EasyTSAD` -- see ../../../baselines_src/KAN-AD/PACKAGE_NOTES.md.
"""
import copy
import time
from pathlib import Path

import numpy as np
import torch as th
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import sys
HERE = Path(__file__).resolve().parent
KANAD_SRC_DIR = HERE.parent.parent.parent / "baselines_src" / "KAN-AD"
sys.path.insert(0, str(KANAD_SRC_DIR))
from kanad.kanad import KANADModel  # noqa: E402

WINDOW = 96
ORDER = 2
BATCH_SIZE = 1024
MAX_EPOCHS = 100
LR = 0.01
DIFF_ORDER = 1
PATIENCE = 3
DELTA = 1e-4
VAL_FRAC = 0.2
HORIZON = 1


def _differential(x, p=1):
    for _ in range(p):
        r = np.pad(x, (0, 1), "edge") - np.pad(x, (1, 0), "edge")
        x = r[:-1]
    return x


def run_kanad(train_1d, test_1d, label_1d, window=WINDOW, order=ORDER,
              batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, lr=LR,
              diff_order=DIFF_ORDER, patience=PATIENCE, val_frac=VAL_FRAC,
              horizon=HORIZON, verbose=False):
    t0 = time.time()
    train_1d = np.asarray(train_1d, dtype=np.float64).ravel()
    test_1d = np.asarray(test_1d, dtype=np.float64).ravel()
    label_1d = np.asarray(label_1d, dtype=int).ravel()

    n = len(train_1d)
    split_idx = int(n * val_frac)
    if split_idx <= 0 or split_idx >= n:
        return None, None, {"skipped": "series too short to split", "elapsed": 0.0}
    tr, va = train_1d[:-split_idx], train_1d[-split_idx:]
    te = test_1d.copy()

    if diff_order > 0:
        tr = _differential(tr, diff_order)
        va = _differential(va, diff_order)
        te = _differential(te, diff_order)

    mu, sigma = tr.mean(), tr.std()
    sigma = sigma if sigma > 1e-8 else 1.0
    tr = (tr - mu) / sigma
    va = (va - mu) / sigma
    te = (te - mu) / sigma

    def make_windows(x):
        m = len(x)
        sample_num = max(m - window - horizon + 1, 0)
        if sample_num <= 0:
            return None, None
        X = np.zeros((sample_num, window), dtype=np.float32)
        Y = np.zeros((sample_num, 1), dtype=np.float32)
        for i in range(sample_num):
            X[i, :] = x[i:i + window]
            Y[i, 0] = x[i + window + horizon - 1]
        return X, Y

    Xtr, Ytr = make_windows(tr)
    Xva, Yva = make_windows(va)
    Xte, Yte = make_windows(te)
    if Xtr is None or Xva is None or Xte is None:
        return None, None, {"skipped": "series too short for window", "elapsed": 0.0}

    train_loader = DataLoader(TensorDataset(th.from_numpy(Xtr), th.from_numpy(Ytr)),
                               batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(th.from_numpy(Xva), th.from_numpy(Yva)),
                               batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(th.from_numpy(Xte), th.from_numpy(Yte)),
                              batch_size=batch_size, shuffle=False)

    model = KANADModel(window=window, order=order)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    best_state = None
    counter = 0
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_tot, val_n = 0.0, 0
        with th.no_grad():
            for xb, yb in valid_loader:
                out = model(xb)
                val_tot += loss_fn(out, yb).item()
                val_n += 1
        val_loss = val_tot / max(val_n, 1)
        scheduler.step()
        epochs_run = epoch

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
    scores = []
    with th.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            scores.append((out - yb).abs().squeeze(-1).numpy())
    scores = np.concatenate(scores)
    scores[np.isnan(scores)] = 1000.0
    aligned_label = label_1d[window + horizon - 1:]

    elapsed = time.time() - t0
    meta = {"epochs_trained": epochs_run, "elapsed": elapsed,
            "n_train_windows": len(Xtr), "n_test_windows": len(scores)}
    if verbose:
        print(f"  done in {elapsed:.1f}s, epochs={epochs_run}, "
              f"train_windows={meta['n_train_windows']}, test_windows={meta['n_test_windows']}")
    return scores, aligned_label, meta
