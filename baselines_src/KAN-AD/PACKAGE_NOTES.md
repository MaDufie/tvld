# KAN-AD (baseline)

Official repo: https://github.com/CSTCloudOps/KAN-AD.git (ICML 2025, "KAN-AD:
Time Series Anomaly Detection with Kolmogorov-Arnold Networks"). Vendored
here as-is, minus `.git/`, `__pycache__/`, and any data/checkpoints folders
(the upstream repo ships none of those committed -- its own `data/`,
`checkpoints/`, `results/` only appear after you run `prepare_env.sh` /
`run_exp.py`, so there was nothing extra to strip; this copy is the clean
~236K of `kanad/`, `pyproject.toml`, `run_exp.py`, `prepare_env.sh`,
`README.md`, `LICENSE`, `.gitignore`, `.python-version`, `uv.lock`).

The real method implementation is `kanad/kanad.py`: `KANADModel` (the actual
~4491-parameter KAN network -- a windowed Conv1d stack over periodic cosine
features, per the paper) plus `KANAD` (an `EasyTSAD`-framework
`BaseMethod` wrapper around it that this package does **not** use).

## How this package actually runs it

This package does not call `run_exp.py` / the `EasyTSAD.Controller` pipeline
upstream's own quick-start uses. Instead, `main_benchmark/baselines/kanad/kanad_runner.py`
reimplements `EasyTSAD`'s exact preprocessing/windowing/training conventions
(`TSData.differential` -> `TSData.z_score_norm` -> `UTSOneByOneDataset` ->
`EarlyStoppingTorch`) directly against plain numpy arrays, importing only
`KANADModel` from the vendored source. This lets the same `run_kanad(train_1d,
test_1d, label_1d, ...)` function be fed SMD/SMAP/MSL/PSM arrays straight from
`tvld_final.py`'s own loaders, with no file-based dataset layer in between.
Model code and hyperparameters (`window=96, order=2, batch_size=1024,
max_epochs=100, lr=0.01, diff_order=1, patience=3`, overridden per-run to
`batch_size=2048, max_epochs=8, patience=3` by `run_all_kanad.py`) are taken
unmodified from the upstream repo / its `EasyTSAD` defaults.

KAN-AD is **univariate**: one model is trained per feature channel, and
multivariate datasets are scored by taking the elementwise **max** anomaly
score across channels per timestep (SMD, PSM: every channel; SMAP/MSL: column
0 only, the one real telemetry channel, matching TVL-D's own disclosed
convention for that dataset elsewhere in this package).

## Extra dependencies

Beyond this package's base `requirements.txt` (numpy/pandas/scipy/scikit-learn/stumpy),
KAN-AD needs:

```
torch>=2.7.0
torchinfo>=1.8.0
EasyTSAD>=0.3.0.2
tqdm
```

(`torch`/`torchinfo`/`tqdm` from `kanad_runner.py`'s own imports; `EasyTSAD`
is pulled in only because `kanad/kanad.py` imports `EasyTSAD.DataFactory` /
`EasyTSAD.Exptools` / `EasyTSAD.Methods` at **module level** for its `KANAD`
wrapper class -- a class `kanad_runner.py` never instantiates. You still need
`pip install EasyTSAD` for the import to succeed, even though none of its
training/data machinery actually runs here. `pyproject.toml`/`uv.lock` in
this vendored copy pin the exact versions upstream validated against,
if you want to reproduce the original environment via `uv sync` instead.)

## Running it

### Main benchmark (clean data, SMD + SMAP/MSL + PSM)

```bash
cd main_benchmark/baselines/kanad
python run_all_kanad.py            # all three datasets: smap_msl, then psm, then smd
python run_all_kanad.py smd        # or just one: smd / smap_msl / psm
```

Reuses the exact same `SMD_DATA_DIR`, `SMAP_MSL_DATA_DIR`, `SMAP_MSL_LABELS`,
`PSM_DATA_DIR` env vars (and the same `main_benchmark/data/` /
`main_benchmark/labeled_anomalies.csv` defaults) as
`main_benchmark/run_smd_eval.py` / `run_smap_msl_eval.py` / `run_psm_eval.py`,
so one copy of each dataset serves TVL-D and every baseline in this package --
no per-method data copies needed. Per-channel scores are cached to
`checkpoints_kanad/{smd,smap_msl,psm}/...` next to the script (resumable).
Writes `kanad_smd_per_machine.csv`, `kanad_smap_msl_per_channel.csv`,
`kanad_psm_results.csv` next to the script.

### Noise robustness (SMD only, 10/20/30/40/50% test-set Gaussian noise)

```bash
cd noise_robustness/baselines/kanad
python run_all_kanad_noise.py
```

KAN-AD caches clean-data **scores**, not trained weights, so the
noise-robustness run is a full per-channel **retrain** at every (machine,
noise_pct) combination -- train is always clean, only the test split is
noise-injected, via the same `common_noise.py` (`TVLD_DATA_DIR`-overridable,
same `zlib.crc32`-seeded noise as every other method in this study) that
`noise_robustness/run_tvld_noise.py` uses, so results are apples-to-apples
comparable. Resumable via per-(machine, noise_pct, dim) pickle caching under
`checkpoints/`. Writes `kanad_noise_results.csv` with the same
`method, machine, noise_pct, <full_metrics() columns>` schema as
`run_tvld_noise.py`'s `tvld_noise_results.csv` (both call `tvld_final.full_metrics()`),
so the two CSVs concatenate directly.

This noise runner imports `kanad_runner.py` from
`main_benchmark/baselines/kanad/` rather than vendoring a second copy of it --
the two runner directories share that one file.


## Caveats

- KAN-AD is lightweight (~4491 parameters per the paper) but trains **one
  model per feature channel** (and per noise level, for the noise ablation),
  so wall-clock cost scales with total channel-count x noise-levels, not
  with the model's own small size.
- `kanad/kanad.py`'s `KANAD` class (the actual `EasyTSAD.Methods.BaseMethod`
  subclass upstream's own `run_exp.py` instantiates) is never used by this
  package's runners -- only its `KANADModel` network definition is imported.
  If you want to reproduce upstream's own benchmark numbers via
  `EasyTSAD.Controller` + `run_exp.py` directly instead of this package's
  `kanad_runner.py`, see `prepare_env.sh` (clones
  `https://github.com/CSTCloudOps/datasets.git`, `uv sync`) and `run_exp.py`
  in this vendored copy.
- No GPU is used by this package's runner (`kanad_runner.py` runs on CPU by
  default, same as every other baseline and TVL-D itself in this package);
  `kanad/kanad.py`'s own `KANAD` wrapper (unused here) does request CUDA when
  available.
