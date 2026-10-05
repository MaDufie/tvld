# CANDI baseline -- vendoring notes

**Original repo:** https://github.com/kimanki/CANDI.git (AAAI 2026: "CANDI:
Curated Test-Time Adaptation for Multivariate Time-Series Anomaly Detection
Under Distribution Shift", Kim et al.)

Vendored here (`baselines_src/CANDI/`) from a clone of that repo, **minus**:

- **`.git/`** -- version-control metadata, not needed to run the code.
- **`results/`** (91M) -- the original authors' own cached experiment
  outputs. Not source code; this package documents dataset sourcing and
  produces its own results separately (see `main_benchmark/`,
  `noise_robustness/`).
- **`data/`** (35M) -- bundled/preprocessed datasets (SWaT, a preprocessed
  SMD copy). Not needed: this package's own `main_benchmark/` scripts
  source SMD/SMAP/MSL/PSM independently (see the top-level `README.md`) so
  one copy of each dataset serves every baseline, this one included.
- `wandb` -- a broken symlink in the original clone pointing outside the
  repo (`/mnt/workspace/gusrl/CANDI/wandb/`); not a real asset.
- `__pycache__/` and `*.pyc` everywhere (including inside the vendored
  `TSB-AD/` subtree).

Kept: `README.md`, `LICENSE`, `.gitignore`, `config.py`, `main.py`,
`predictor.py`, `threshold.py`, `trainer.py`, `models/`, `scripts/`,
`figures/` (two small motivation/overview PNGs, not result figures),
`datasets/`, `layers/`, `tta/`, `utils/` -- the repo's own standalone
implementation, kept for reference/completeness even though our runner
(below) does not import from it directly.

Net effect: **212M -> ~1.3M** vendored.

## TSB-AD/ decision: REQUIRED, included

`TSB-AD/` is **not** bloat -- it is a hard runtime dependency. CANDI's own
top-level `models/`/`trainer.py`/`scripts/` code never imports anything
from `TSB_AD`, but **this project's actual runner** (`candi_runner.py`)
does not use that top-level implementation at all. Instead it imports the
model classes straight from the authors' own TSB-AD-embedded
implementation:

```python
from TSB_AD.models.CANDI import MLP, MLP_Tester
```

(`TSB-AD/TSB_AD/models/CANDI.py`, which in turn imports
`TSB_AD/models/base.py`, `TSB_AD/utils/torch_utility.py`,
`TSB_AD/utils/dataset.py`, and `TSB_AD/layers/{Transformer_EncDec,
SelfAttention_Family}.py`). Without `TSB-AD/` on `sys.path`, `candi_runner.py`
fails to import at all -- confirmed by actually importing it against this
vendored copy (`import candi_runner` succeeds end-to-end, pulling in the
real `MLP`/`MLP_Tester` classes).

Only the `TSB_AD/` Python package plus `README.md`/`LICENSE`/
`requirements.txt`/`setup.py` from the TSB-AD submodule were vendored
(900K). TSB-AD's own `docs/` (5.0M, a static HTML doc site) and `assets/`
(3.4M, doc-site images) were excluded as pure documentation bloat unrelated
to running CANDI; `benchmark_exp/` (820K, TSB-AD's own general multi-method
benchmark-sweep scripts plus 340K of its own cached `benchmark_eval_results/`)
was excluded too since nothing in this baseline's runner imports from it.

## Extra pip dependencies

Beyond this package's base `requirements.txt` (numpy/pandas/scipy/
scikit-learn/stumpy), CANDI needs:

- `torch` (>=1.x; TSB-AD's own `requirements.txt` pins `torch==2.3.0`, but
  any reasonably recent CPU or CUDA build works 
- `tqdm`
- `einops`

(TSB-AD's `requirements.txt`, vendored at `TSB-AD/requirements.txt`, also
lists `torchinfo`, `h5py`, `arch`, `hurst`, `tslearn`, `cython`, `networkx`,
`transformers`, `matplotlib` -- those are for TSB-AD's *other* 30+ bundled
detectors and its own benchmark/plotting scripts, not needed to import or
run `TSB_AD.models.CANDI` alone.)

## How to run

**Clean-data (main benchmark), SMD + SMAP/MSL + PSM:**

```bash
cd main_benchmark/baselines/candi
python run_all_candi.py              # all three datasets
python run_all_candi.py smd          # just SMD
python run_all_candi.py smap_msl     # just SMAP/MSL
python run_all_candi.py psm          # just PSM
```

Expects the same `./data/` layouts as `main_benchmark/run_smd_eval.py` /
`run_smap_msl_eval.py` / `run_psm_eval.py`, two levels up
(`main_benchmark/data/`) by default -- override with the same env vars
those scripts use: `SMD_DATA_DIR`, `SMAP_MSL_DATA_DIR`, `SMAP_MSL_LABELS`,
`PSM_DATA_DIR`. Writes `candi_smd_per_machine.csv`,
`candi_smap_msl_per_channel.csv`, `candi_psm_results.csv` next to the
script, plus per-entity checkpoints under `checkpoints_candi/`.

**Noise-robustness ablation (SMD only, 10/20/30/40/50% noise):**

```bash
cd noise_robustness/baselines/candi
python run_all_candi_noise.py
```

Expects SMD at `../../data` (shared with `noise_robustness/run_tvld_noise.py`)
or `TVLD_DATA_DIR` pointed elsewhere (same env var `common_noise.py` already
uses). Writes `candi_noise_results.csv` (schema: `method, machine,
noise_pct, ...`, matching `run_tvld_noise.py`) plus per-combo checkpoints
under `checkpoints/`.


## Caveats

- **No separately-sourced pretrained weights needed.** CANDI is
  test-time-adaptation (TTA) on top of a reconstruction model, but that
  base model is trained from scratch, inside this project's own runner,
  for every entity (`candi_runner.run_candi()`: ordinary offline
  AdamW training of the `MLP` autoencoder, each run tagged/cached
  independently) -- "pretrained" in the paper's own terminology just means
  "trained offline before the online TTA/SANA phase begins," not a
  checkpoint shipped separately. No `.pt`/`.pth`/`.ckpt` file exists
  anywhere in the original `CANDI/` clone outside its own `results/`/`data/`
  (which were excluded here as bloat, and contained no model weights
  either). This baseline is therefore fully self-contained and
  reproducible from source + data alone -- not a reproducibility blocker.