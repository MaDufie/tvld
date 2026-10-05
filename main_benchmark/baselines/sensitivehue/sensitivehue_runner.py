"""Train+score SensitiveHUE on ONE entity's (T, D) numpy arrays, reusing the
official vendored model/training code from ../../../baselines_src/SensitiveHUE
directly (sensitive_hue.SensitiveHUE, sensitive_hue.Trainer, main.get_data_loaders).

WHAT SensitiveHUE IS
---------------------
SensitiveHUE ("Under Review" at the time this repo was cloned) is a
Transformer-based reconstruction detector built around two ideas, both
literally named in the repo's README:

1. **Statistical Feature Removal (SFR).** Implemented via RevIN
   (Reversible Instance Normalization, `affine=False`): each input window is
   normalized by its OWN per-window mean/std before encoding, and the
   reconstruction is denormalized back at the output. This strips each
   window's absolute-level statistics ("statistical features") before the
   Transformer ever sees it, forcing it to model relative/dependency
   structure across channels and timesteps rather than trivially memorizing
   per-window magnitude.
2. **MTS-NLL loss (heterogeneous uncertainty estimation).** The model has
   TWO output heads sharing one Transformer encoder trunk: a reconstruction
   head (`rec`) and a per-channel, per-timestep log-inverse-variance head
   (`log_var_recip`, i.e. log(1/sigma^2)). The training loss is a
   heteroscedastic Gaussian negative log-likelihood
   (`rec_error * exp(log_var_recip) - log_var_recip`, algebraically
   `error^2/sigma^2 + log(sigma^2)`) -- the model learns to predict its OWN
   aleatoric uncertainty per channel/timestep, so a large reconstruction
   error in a channel the model has learned is inherently noisy is weighted
   down relative to the same error in a channel the model is usually very
   confident about. On top of the plain NLL, the loss additionally reweights
   each channel by `var / mean(var)^alpha` (`alpha` a per-dataset
   hyperparameter) -- the "heterogeneous" part -- so channels with very
   different natural variance scales contribute comparably to the loss
   rather than the loss being dominated by inherently high-variance
   channels.
3. **Architecture.** A single-block flow: RevIN norm -> linear input
   projection + learned positional embedding -> N standard
   pre-LN-style Transformer encoder layers (multi-head self-attention over
   the TIME axis, i.e. each window is treated as a sequence of `step_num_in`
   tokens, one per timestep, each token being the full D-channel feature
   vector at that timestep) -> two linear heads (reconstruction, log
   inverse-variance) -> RevIN denorm on the reconstruction head only.
4. **Scoring.** At test time, the per-channel, per-timestep NLL-form
   anomaly score (`mse * exp(log_var_recip) - log_var_recip`) is computed
   for every window, but -- notably -- only the score at ONE representative
   position within each window is kept (`select_pos`: 'mid', the window's
   middle timestep, for every dataset this repo ships a config for). Since
   test-time windows slide with stride 1, this produces exactly one score
   per original timestep (offset by `step_num_in // 2` at each end), with no
   overlap-averaging needed -- effectively a centered, non-causal windowed
   scoring convention (the model sees `step_num_in // 2` steps of future
   context on both sides of the scored point within its own window).
   Per-channel scores are then median/IQR-normalized and aggregated by
   taking the max across channels.

"""
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

HERE = Path(__file__).resolve().parent
SENSITIVEHUE_SRC_DIR = HERE.parent.parent.parent / "baselines_src" / "SensitiveHUE"
sys.path.insert(0, str(SENSITIVEHUE_SRC_DIR))

import sensitive_hue
from scipy.stats import iqr as _iqr

SCRATCH_DIR = HERE / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)


def _floor_val_split(dataset, val_ratio, random_val, seed):
    """DISCLOSED fix (deviation 8): floor val_use_len at 1 whenever there
    are >=2 windows; skip validation (return None) for <2 windows."""
    from torch.utils.data import Subset
    dataset_len = len(dataset)
    if dataset_len < 2:
        return dataset, None
    val_use_len = max(1, int(dataset_len * val_ratio))
    val_use_len = min(val_use_len, dataset_len - 1)
    g = torch.Generator().manual_seed(seed)
    if random_val:
        val_indices = torch.randperm(dataset_len, generator=g)[:val_use_len].numpy().tolist()
    else:
        start = torch.randint(0, dataset_len - val_use_len, (1,), generator=g).item()
        val_indices = list(range(start, start + val_use_len))
    train_indices = list(set(range(dataset_len)) - set(val_indices))
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def run_sensitivehue(train_arr, test_arr, label, step_num_in=48, stride=1, batch_size=64,
                      dim_model=128, dim_hidden_fc=256, head_num=4, encode_layer_num=2,
                      val_ratio=0.2, random_val=True, alpha=1.0, select_pos="mid",
                      max_epoch=30, lr=0.001, seed=5, tag="run", verbose=False):
    import time
    t0 = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    D = train_arr.shape[1]

    scratch_dir = tempfile.mkdtemp(prefix=f"shue_{tag}_", dir=str(SCRATCH_DIR))
    train_path = os.path.join(scratch_dir, "train.npy")
    test_path = os.path.join(scratch_dir, "test.npz")
    np.save(train_path, train_arr.astype(np.float64))
    np.savez(test_path, x=test_arr.astype(np.float64), y=np.asarray(label, dtype=np.float64))

    data_args = Namespace(
        data_dir=scratch_dir, step_num_in=step_num_in, batch_size=batch_size,
        val_ratio=val_ratio, random_val=random_val, f_in=D, alpha=alpha,
        select_pos=select_pos, ignore_dims=None,
    )
    global_args = Namespace(stride=stride, head_num=head_num, max_epoch=max_epoch, lr=lr)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    from base.dataset import ADataset
    train_dataset = ADataset(os.path.join(scratch_dir, "train.npy"), step_num_in, stride)
    train_dataset.fit_transform(scaler.fit_transform)
    train_subset, val_subset = _floor_val_split(train_dataset, val_ratio, random_val, seed)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    if val_subset is not None:
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)

    test_dataset = ADataset(os.path.join(scratch_dir, "test.npz"), step_num_in, 1, keys=("x", "y"))
    test_dataset.fit_transform(scaler.transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = sensitive_hue.SensitiveHUE(
        step_num_in, D, dim_model, head_num, dim_hidden_fc, encode_layer_num, 0.1
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    model_save_dir = os.path.join(scratch_dir, "model_states")
    trainer = sensitive_hue.Trainer(
        model, optimizer, alpha, max_epoch, model_save_dir, scheduler, use_prob=True
    )

    trainer.train(train_loader, val_loader)

    errors, labels_out = trainer._get_anomaly_score(test_loader, load_state=True, select_pos=select_pos)

    median, iqr_ = np.median(errors, axis=0), _iqr(errors, axis=0)
    errors_norm = (errors - median) / (iqr_ + 1e-9)
    scores = np.nan_to_num(errors_norm, nan=0.0, posinf=0.0, neginf=0.0).max(axis=1)

    aligned_label = np.asarray(labels_out)
    n = min(len(scores), len(aligned_label))
    scores = scores[:n]
    aligned_label = aligned_label[:n]

    meta = {
        "elapsed": time.time() - t0,
        "D": D,
        "n_train_windows": len(train_dataset),
        "n_test_windows": len(test_dataset),
        "step_num_in": step_num_in,
        "model_save_dir": model_save_dir,
    }
    if verbose:
        print(f"[sensitivehue_runner] {tag}: D={D} elapsed={meta['elapsed']:.1f}s "
              f"n_train_win={meta['n_train_windows']} n_test_win={meta['n_test_windows']}", flush=True)

    return scores, aligned_label, meta
