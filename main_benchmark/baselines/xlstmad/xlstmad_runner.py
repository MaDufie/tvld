"""Generic per-entity xLSTMAD train+eval runner, reused across SMD / SMAP / MSL / PSM
so the exact same fixed hyperparameters and evaluation convention are applied everywhere
"""
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
XLSTMAD_SRC_DIR = HERE.parent.parent.parent / "baselines_src" / "xlstmad"
sys.path.insert(0, str(XLSTMAD_SRC_DIR))
from dataset import SlidingWindowDataset  # noqa: E402
from xlstmad import xLSTMAD  # noqa: E402

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)

WINDOW_SIZE = 20
EMBEDDING_DIM = 40
BATCH_SIZE = 256
MAX_EPOCHS = 3 
PATIENCE = 2
VAL_FRAC = 0.2
LR = 0.001
SEED = 0


def run_xlstmad(train, test, label, window_size=WINDOW_SIZE, embedding_dim=EMBEDDING_DIM,
                 batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                 val_frac=VAL_FRAC, verbose=False):
    """train: (T_train, D) array assumed (mostly) normal. test: (T_test, D). label: (T_test,) 0/1.
    Returns (scores, aligned_label, meta) where scores/aligned_label both have length
    T_test - window_size + 1 (the first window_size-1 test points have no window-ending
    score and are dropped from evaluation -- same convention used across the deep-AD
    literature for this exact windowing scheme)."""
    t0 = time.time()
    L.seed_everything(SEED, verbose=False)

    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    label = np.asarray(label, dtype=int)

    mu = train.mean(axis=0, keepdims=True)
    sigma = train.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    train_n = (train - mu) / sigma
    test_n = (test - mu) / sigma

    n = len(train_n)
    split = int(n * (1 - val_frac))
    tr_x, va_x = train_n[:split], train_n[split:]
    tr_y = np.zeros(len(tr_x))
    va_y = np.zeros(len(va_x))

    if len(tr_x) <= window_size or len(va_x) <= window_size or len(test_n) <= window_size:
        return None, None, {"skipped": "series too short for window_size", "elapsed": 0.0}

    train_loader = DataLoader(SlidingWindowDataset(tr_x, tr_y, window_size=window_size),
                               batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SlidingWindowDataset(va_x, va_y, window_size=window_size),
                             batch_size=4 * batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(SlidingWindowDataset(test_n, label, window_size=window_size),
                              batch_size=4 * batch_size, shuffle=False, num_workers=0)

    model = xLSTMAD(embedding_dim=embedding_dim, features_no=train_n.shape[1],
                     window_size=window_size, lr=LR, slstm_backend="vanilla")

    ckpt_dir = tempfile.mkdtemp(prefix="xlstmad_ckpt_")
    checkpoint_cb = ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", save_top_k=1,
                                     save_last=False, mode="min")
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience, mode="min", min_delta=1e-4),
                   checkpoint_cb],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpoint_cb.best_model_path
    if best_path:
        model = xLSTMAD.load_from_checkpoint(best_path)
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in test_loader:
            x_hat = model(x)
            rec_error = torch.mean((x - x_hat) ** 2, dim=(1, 2))
            scores.append(rec_error.numpy())
    scores = np.concatenate(scores)
    aligned_label = label[window_size - 1:]

    elapsed = time.time() - t0
    meta = {"epochs_trained": trainer.current_epoch, "elapsed": elapsed,
            "n_train_windows": len(train_loader.dataset), "n_test_windows": len(scores)}
    if verbose:
        print(f"  done in {elapsed:.1f}s, epochs={trainer.current_epoch}, "
              f"train_windows={meta['n_train_windows']}, test_windows={meta['n_test_windows']}")
    return scores, aligned_label, meta
