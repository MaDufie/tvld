# ALoRa baseline -- vendoring notes

## Original repo

**ALoRa** ("Low Rank Transformer for Multivariate Time Series Anomaly
Detection and Localization", ICLR 2026, Shimillas et al.):
https://github.com/CharisShimillas/ALoRa.git

Vendored here (`baselines_src/ALoRa/`) from a full clone of that repo, minus
`.git/`, its bundled datasets, its pretrained checkpoints, and other cached
run output -- see **What was excluded** below. The vendored tree is ~1 MB
(the original clone was ~229 MB). Nothing here was modified from the
original source.

This package does **not** run ALoRa's own `main.py`/`solver.py` training
loop. It reimplements ALoRa's train+eval loop (`alora_runner.py`, next to
`run_one_alora.py` in `main_benchmark/baselines/alora/`) directly against
`model/ALoRa.py` (the official `ALoRaT` model class, imported unmodified),
so it can take plain numpy arrays as input and plug into this project's
`full_metrics` scoring, the same way the other baselines in this study were
run. See that file's module docstring for the three disclosed, intentional
deviations from the official `solver.py` (validation-split early stopping
instead of test-set early stopping; per-dataset `rank_threshold`; and
per-entity Spearman top-pairs computed by this project rather than reusing
the repo's own precomputed `pairs_idx_*.npy`, which only cover the repo's
own bundled dataset/channel-count combinations).

## What was excluded, and why

- **`.git/`** (98 MB) -- version history, not needed to run the code.
- **`Datasets/`** (30 MB) -- the official repo bundles its own copies of SMD
  and HAI (`train.csv`/`test.csv`/`test_label.csv`). Confirmed to be plain
  bundled data, not source code. This package sources data separately (see
  the top-level README's "Get the data from..." sections for SMD/SMAP-MSL/
  PSM) so every method shares one copy; ALoRa's own bundled copies would
  just be redundant, differently-formatted duplicates.
- **`checkpoints/`** (46 MB) -- pretrained `.pth` weights for SMD and HAI
  (`checkpoints/SMD/SMD_checkpoint.pth`, `checkpoints/HAI/HAI_checkpoint.pth`)
  plus one precomputed `.npy` matrix. Confirmed to be cached pretrained
  weights, not source. Not needed: this package's runner
  (`alora_runner.py`) always trains ALoRa from scratch per entity (see
  **Caveats** below) rather than loading a pretrained checkpoint.
- **`Localization/PATHS/`** (55 MB, the bulk of the `Localization/` subtree)
  -- cached intermediate matrices and a second copy of the SMD checkpoint
  (`C_matrix.npy`, `RecLoc_matrix.npy`, `AS_matrix_normalized.npy`,
  `filter_weight_matrix_SMD.npy`, `B_matrix.pt`, `SMD_checkpoint.pth`,
  `output.csv`). Confirmed to be cached run output, not source. **The actual
  localization source code (`Localization/ALoRa_Loc.py`,
  `Localization/Loc_Metrics.py`, `Localization/Loc_How_to_run.sh`) IS
  included** -- only its `PATHS/` cache subfolder was excluded. This
  package's own runners (`run_*_alora*.py`) do not use ALoRa's localization
  path at all (TVL-D's own localization story is evaluated separately via
  `main_benchmark/run_localization_eval.py`), so `Localization/` is vendored
  for completeness/reference only and is not imported by anything in this
  package.
- **`Results/`** (24 KB) -- the official repo's own `performance.txt` run
  logs for SMD/MSL/PSM/SWAT/HAI, i.e. cached output, not source.
- **`temp_gt.txt` / `temp_preds.txt`** (32 KB total) -- leftover debug
  scratch files from a prior run of the official repo (raw 0/1 label and
  prediction dumps), not source.
- **`__pycache__/`** (all levels) and `*.pyc` -- compiled bytecode caches.

**Everything else is genuine source** and is included as-is: `model/`
(`ALoRa.py`, `MHA.py`, `LMTSembed.py` -- the actual `ALoRaT` architecture),
`data_factory/` (`data_loader.py`, `MSDS.py`, and `Preprocess/` -- the latter
includes small (<50 KB each) precomputed Spearman-correlation `.npy` files
for the repo's own bundled datasets, which are themselves committed source
artifacts of the official repo's `Preprocess.py` script, not bloat), `utils/`
(`utils.py`, `logger.py`, `affiliation/`, `evaluation/`), `solver.py`,
`main.py`, `How_to_run.sh`, `requirements.txt`, `LICENSE.txt`, `README.md`,
and `images/` (the one architecture diagram the README references).

## Extra dependencies beyond this package's `requirements.txt`

This package's base `requirements.txt` (numpy/pandas/scipy/scikit-learn/
stumpy) does **not** include PyTorch. ALoRa is the only method in this
package that needs it:

```bash
pip install torch   # any reasonably recent CPU or CUDA build works --
                     # this was verified against torch==2.14.0 (CPU) here;
                     # the official repo itself was developed against the
                     # much older torch==1.10.2+cu102
```

Nothing else is needed: `alora_runner.py` only imports `model/ALoRa.py`
(which itself only needs `torch`/`numpy`), plus `scipy.stats.rankdata`
(already in the base `requirements.txt`). The official repo's own
`requirements.txt` (`tensorflow`, `plotly`, `seaborn`, `torchviz`,
`Pillow`, `xlsx2csv`, pinned `numpy`/`pandas`/`scipy`/`scikit-learn`) is
only needed if you run the official `main.py`/`solver.py` pipeline directly
(e.g. `utils/logger.py` needs `tensorflow`, `solver.py` needs `torchviz`/
`seaborn`/`plotly`) -- this package's own runner scripts never import
`solver.py`, `main.py`, `utils/logger.py`, or `data_factory/data_loader.py`,
so none of that is required here.

## How to run

### `main_benchmark/` (clean data, SMD + SMAP/MSL + PSM)

```bash
cd main_benchmark
python baselines/alora/run_all_alora.py              # all three datasets
# or just one:
python baselines/alora/run_all_alora.py smd
python baselines/alora/run_all_alora.py smap_msl
python baselines/alora/run_all_alora.py psm
```

Expects the same data layout/env vars as the TVL-D scripts next to it:
`SMD_DATA_DIR`, `SMAP_MSL_DATA_DIR`, `SMAP_MSL_LABELS`, `PSM_DATA_DIR` (all
default to `main_benchmark/data/` -- or `labeled_anomalies.csv` next to it
for SMAP/MSL -- so one copy of each dataset serves every method in this
package, TVL-D included). Writes `alora_smd_per_machine.csv`,
`alora_smap_msl_per_channel.csv`, `alora_psm_results.csv` next to
`run_all_alora.py`, and caches each trained entity's raw scores under
`baselines/alora/checkpoints/<dataset>/<entity>.pkl` (resumable -- an
interrupted run skips any entity whose cache file already exists).

Each entity (SMD machine / SMAP or MSL channel / PSM) is trained in its own
subprocess, not in the orchestrator's process -- this was forced by a real
OOM in the original study (see `run_all_alora.py`'s docstring) and is
preserved here even though a single run on a well-provisioned server may
never hit it.

### `noise_robustness/` (SMD only, 5 noise levels)

```bash
cd noise_robustness
python baselines/alora/run_all_alora_noise.py
```

Expects the same SMD data as `run_tvld_noise.py` (`TVLD_DATA_DIR` env var,
default `noise_robustness/data/`). Writes `baselines/alora/alora_noise_results.csv`
with the schema `method, machine, noise_pct, <full_metrics() columns>` --
identical to `run_tvld_noise.py`'s `tvld_noise_results.csv` schema, so the
two (and every other baseline's noise-results CSV) concatenate directly into
one combined table for `figures/generate_noise_degradation_plot.py`. Caches
each (machine, noise_pct) combo under `baselines/alora/checkpoints_alora/`
(resumable, same convention as `run_tvld_noise.py`'s own `checkpoints/`).

ALoRa **retrains from scratch for every noise level** (28 machines x 5
levels = 140 full training runs) since no pretrained-weight reuse is
available or desired here -- this is the slowest baseline in the
noise-robustness ablation by a wide margin; see **Caveats** below.


## Caveats

- **No pretrained weights are included or used.** `checkpoints/` (the
  official repo's SMD/HAI `.pth` files) was excluded as bloat/cached output
  (see above); this package's runner always trains ALoRa from scratch, per
  entity, per run (and per noise level, in `noise_robustness/`). This
  matches how every other baseline in this study is run (no leakage from a
  pretrained checkpoint) but means a full run is comparatively slow.
- **CPU works but GPU is strongly recommended for a full run.** The vendored
  model itself doesn't force a device; this package's `alora_runner.py`
  runs on whatever device PyTorch defaults to (CPU here, unless you move the
  model/tensors yourself) -- it was never modified to add explicit
  `.cuda()` calls. The official repo's own `README.md` environment section
  assumes 2x Tesla V100 GPUs; the smoke tests above confirm CPU-only
  execution works correctly, just slowly, especially for PSM (win_size=100)
  and the 28-machine x 5-noise-level noise-robustness sweep (140 full
  retrains).
- **SMAP is not part of the official repo's own dataset list or precomputed
  pairs** (only SMD, HAI, SWaT, MSL, PSM, MSDS are). This package still
  evaluates SMAP (disclosed as a deviation in `alora_runner.py`'s
  docstring: a controlled sensitivity check was used to pick its
  `rank_threshold` rather than guessing) -- if you intend to cite ALoRa's
  SMAP numbers, note this is this project's own extension, not an
  official-repo result.
- **A per-entity subprocess is launched for every machine/channel** (SMD,
  SMAP, MSL) or just once (PSM); a 1800-3600s timeout with 3-5 retries is
  applied in case of an OOM-kill or transient failure, matching the
  original cloud study's safety margin. On a memory-constrained host,
  running `main_benchmark/baselines/alora/run_all_alora.py smap_msl` (81
  channels back-to-back subprocesses) is the most failure-prone leg; rerun
  it (it resumes via its per-channel `.pkl` cache) if any channel's
  subprocess fails.
- **No special import shims were needed.** `model/ALoRa.py`, `MHA.py`, and
  `LMTSembed.py` have no dependency on `data_factory/`, `utils/`, or
  `solver.py` -- they only need `torch`/`numpy`/`itertools`/`math`, so
  vendoring just `model/` would in fact have been sufficient for this
  package's own runners to work; the rest of the tree (`data_factory/`,
  `utils/`, `solver.py`, `main.py`) is included for completeness/reference
  to the official repo, not because this package's scripts import it.
