# TVL-D
This reproduces every quantitative claim in the paper: TVL-D itself, all six baselines it's compared against, the leave-one-component-out ablation, and every figure.

TVL-D is a statistical ensemble for multivariate time-series anomaly detection and localization. It does not use gradient-based model optimization or require a pretrained neural-network checkpoint. Statistical reference quantities are estimated from each dataset's provided normal training split. The core method is `tvld_final.py`'s severity/fusion pipeline with fixed fusion weights (`W_SHAPE=0.15`, `W_LEVEL=0.75`, `W_STRUCT=1.0`) and smoothing window 35, applied unchanged across the evaluation datasets. Every TVL-D evaluation script imports that implementation without modifying the method.

The six baselines (xLSTMAD, ALoRa, KAN-AD, CANDI, TopoGDN, SensitiveHUE) are ****cloned from their official repos**** (source code only - see `baselines_src/`) plus this project's own driver scripts that train/run them and score them with the exact same metrics code TVL-D uses, so every method's numbers are produced the same way. Unlike TVL-D, all six depend on learned or pretrained model representations and need PyTorch (and TopoGDN additionally needs `torch-geometric` and a C++ compiler); see each baseline's `PACKAGE_NOTES.md` for exact dependencies.

## Package structure
This package has four independent parts:

- **`main_benchmark/`** - the evaluation on four benchmarks from three data sources: SMD, SMAP/MSL, and PSM. `main_benchmark/baselines/<method>/` holds the six baselines' equivalent runners.

- **`ablation/`** - the leave-one-component-out ablation: starting from the full fixed TVL-D configuration, zero out Shape, Level, or Structure (or disable smoothing) one at a time and re-measure every metric, to see how much each component contributes.

- **`baselines_src/`** - the six baselines' own source code, cloned from their official GitHub repos.

- **`figures/`** - the code that actually draws every figure in the paper: pre-smoothing-vs-final comparison, the per-channel localization demo, the 7-method heatmap, critical difference diagram, and the headline line plot.

## Files
```

tvld/
  tvld_final.py                        -- the TVL-D algorithm
  requirements.txt                     -- base deps (numpy/pandas/scipy/scikit-learn/stumpy); each
                                           baseline needs extras, see baselines_src/<name>/PACKAGE_NOTES.md
  main_benchmark/
    metrics.py                         -- pr_auc, auc_roc, auc_pr_t, f1/precision/recall
    run_smd_eval.py                    -- TVL-D, SMD, 28 machines
    run_smap_msl_eval.py               -- TVL-D, SMAP + MSL, 81 channels
    run_psm_eval.py                    -- TVL-D, PSM, 1 pooled stream
    run_localization_eval.py           -- TVL-D, SMD only, HitRate@1.0 macro-averaged across machines
    baselines/
      xlstmad/        -- run_all_xlstmad.py, xlstmad_runner.py
      alora/           -- run_all_alora.py, run_one_alora.py, alora_runner.py
      kanad/           -- run_all_kanad.py, kanad_runner.py
      candi/           -- run_all_candi.py, run_one_candi.py, candi_runner.py
      topogdn/         -- run_all_topogdn.py, run_one_topogdn.py, topogdn_runner.py
      sensitivehue/    -- run_all_sensitivehue.py, run_one_sensitivehue.py, sensitivehue_runner.py
  ablation/
    build_series_cache.py              -- reduces every series to per-timestep Shape/Level/Structure scalars
    ablation_table_full.py             -- the leave-one-component-out sweep
  baselines_src/
    xlstmad/           PACKAGE_NOTES.md -- https://github.com/Nyderx/xlstmad.git
    ALoRa/              PACKAGE_NOTES.md -- https://github.com/CharisShimillas/ALoRa.git
    KAN-AD/             PACKAGE_NOTES.md -- https://github.com/CSTCloudOps/KAN-AD.git
    CANDI/              PACKAGE_NOTES.md -- https://github.com/kimanki/CANDI.git
    TopoGDN/            PACKAGE_NOTES.md -- https://github.com/ljj-cyber/TopoGDN.git
    SensitiveHUE/       PACKAGE_NOTES.md -- https://github.com/yuesuoqingqiu/SensitiveHUE.git
  figures/
    data/                              -- small pre-computed CSVs the comparison figures read (no re-run needed)
      final_model_combined_summary.csv
      seven_method_combined.csv
    generate_final_model_comparison.py -- pre-smoothing vs. final TVL-D, all datasets x 6 metrics
    generate_localization_demo.py      -- per-channel severity heatmap around one real SMD anomaly
    generate_benchmark_figures.py      -- 7-method heatmap + CD diagram + line plot
```

`data/` each dataset is sourced separately below. Every script in this package (TVL-D's and every baseline's) reads the SAME `main_benchmark/data/` folder via the same env vars, so you only need one copy of each dataset for the whole package, not one per method.

## 1. Install dependencies
```bash

python3 -m venv venv && source venv/bin/activate   # optional but recommended

pip install -r requirements.txt

```

This covers TVL-D itself (`numpy`, `pandas`, `scipy`, `scikit-learn`, `stumpy`). ****Each baseline needs more**** -- at minimum `torch` for all six, plus method-specific extras (`xlstm`+`lightning` for xLSTMAD, `EasyTSAD` for KAN-AD, `einops` for CANDI, `torch-geometric` + a C++ build step for TopoGDN, `PyYAML` for SensitiveHUE). See `baselines_src/<name>/PACKAGE_NOTES.md` for the exact list and install commands per method -- install only the ones for the baselines you actually plan to run.

## 2. `main_benchmark/`
Run each dataset's script from inside `main_benchmark/`, with that dataset's data under a `data/` folder next to the script (or point the env var below elsewhere).

### SMD
```bash

cd main_benchmark

python run_smd_eval.py        # expects ./data/{train,test,test_label}/machine-*.txt

# or: SMD_DATA_DIR=/path/to/ServerMachineDataset python run_smd_eval.py

```

Get the data from the OmniAnomaly release: https://github.com/NetManAIOps/OmniAnomaly (the `ServerMachineDataset/` folder has the `train/`, `test/`, `test_label/` layout this expects; add `interpretation_label/` too if you also want to run the localization evaluation below).

### SMAP/MSL
```bash

python run_smap_msl_eval.py    # expects ./data/{train,test}/<chan_id>.npy + ./labeled_anomalies.csv

# or: SMAP_MSL_DATA_DIR=/path/... SMAP_MSL_LABELS=/path/to/labeled_anomalies.csv python run_smap_msl_eval.py

```

Get the data from the original NASA telemanom release: https://github.com/khundman/telemanom -- it provides the per-channel `train/`/`test/` `.npy` files and `labeled_anomalies.csv` this expects.

### PSM
```bash
python run_psm_eval.py        # expects ./data/{train.csv,test.csv,test_label.csv}
# or: PSM_DATA_DIR=/path/to/psm python run_psm_eval.py
```

Get the data from the original authors' own release: Abdulaal et al., KDD 2021 is an eBay paper, and eBay's RANSynCoders repo is where PSM was first published: https://github.com/eBay/RANSynCoders (the `data/` folder has `train.csv`/`test.csv`/`test_label.csv` in exactly the layout this script expects -- `timestamp_(min)` plus 25 `feature_0`..`feature_24` columns in train/test, `timestamp_(min)`/`label` in test_label).

### What TVL-D's own scripts do / write
Each script computes Shape/Level/Structure severity maps once per machine/channel/stream (expensive -- matrix-profile computation via `stumpy` dominates), caches them to a `checkpoints_*` folder or file next to itself, fuses them with the fixed weights, and scores both the pre-smoothing and final (smoothed) variant. Output:

- `run_smd_eval.py` -> `smd_results.csv` (one row per machine x method)
- `run_smap_msl_eval.py` -> `smap_msl_per_channel.csv` (one row per channel x method; printed summary breaks out SMAP and MSL separately, never pooled, matching the paper)
- `run_psm_eval.py` -> `psm_results.csv` (one row per method -- PSM is a single pooled stream, not multiple machines/channels)

All three report six metrics: `pr_auc`, `auc_roc`, `auc_pr_t` (range-based PR-AUC), and `f1`/`precision`/ `recall` at the single best-F1 threshold.

### Timing (TVL-D)
Per-unit cost is dominated by `shape_severity`'s matrix-profile computation, called once per scored variable.

### Localization (SMD only)
```bash
python run_localization_eval.py           # expects the same ./data/ as run_smd_eval.py,
                                          # plus ./data/interpretation_label/machine-*.txt
```

TVL-D's localization result (HitRate@1.0, the fraction of a labeled anomaly segment's true root-cause channels that land in the top-K channels ranked by mean severity over that segment's window, K taken from the ground-truth label) comes from the exact same per-channel Shape/Level/Structure severity tensor `run_smd_eval.py` sums into the detection score -- no separate model, no separate tuning. This is SMD-only: SMAP/MSL's channels are already one real telemetry column plus one-hot context flags (nothing left to localize within a channel), and PSM ships no root-cause channel labels at all. If `run_smd_eval.py` has already been run, this reuses its cached `checkpoints_smd/<machine>.pkl` severities instead of recomputing the matrix-profile pass. Writes `smd_localization_results.csv` (one row per machine with labels: `machine, n_segments, hitrate_at_1`) and prints the macro-averaged HitRate@1.0 across machines. (`figures/generate_localization_demo.py` is the companion ***illustration***: one real segment plotted as a heatmap, not this full evaluation.)

### Baselines (`main_benchmark/baselines/`)

Each of the six baselines has its own subdirectory with a `run_all_<method>.py` (mirrors `run_smd_eval.py` etc. -- same env vars, same shared `data/` folder, same `full_metrics()` scoring) plus any helper files it needs (`<method>_runner.py`, and for the three slowest/heaviest baselines a `run_one_<method>.py` that trains one entity in its own subprocess, same OOM-safety pattern this project used throughout):
```bash
cd main_benchmark/baselines/xlstmad    && python run_all_xlstmad.py
cd main_benchmark/baselines/alora      && python run_all_alora.py
cd main_benchmark/baselines/kanad      && python run_all_kanad.py

cd main_benchmark/baselines/candi      && python run_all_candi.py
cd main_benchmark/baselines/topogdn    && python run_all_topogdn.py
cd main_benchmark/baselines/sensitivehue && python run_all_sensitivehue.py
```

Each accepts an optional dataset argument (`smd` / `smap_msl` / `psm`) to run just one instead of all three. Each writes its own `<method>_smd_per_machine.csv` / `<method>_smap_msl_per_channel.csv` / `<method>_psm_results.csv` next to the script, with the same metric columns as TVL-D's, so all seven methods' CSVs line up for a cross-method table.

Summary:

| Method | Extra deps |
|---|---|
| xLSTMAD | `xlstm`, `lightning`, `torch` |
| ALoRa | `torch` |
| KAN-AD | `torch`, `torchinfo`, `EasyTSAD`, `tqdm` |
| CANDI | `torch`, `tqdm`, `einops` |
| TopoGDN | `torch`, `torch-geometric` (+ C++ build) |
| SensitiveHUE | `torch`, `PyYAML` |

## 3. `ablation/` -- leave-one-component-out
```bash
cd ablation
python build_series_cache.py    # builds series_cache.pkl -- reuses main_benchmark/'s checkpoints if
                                 # they already exist (instant), else computes them itself (slow, same
                                 # cost as running main_benchmark/'s three TVL-D scripts once)
python ablation_table_full.py   # -> ablation_results.json + printed table
```

Starting from the full fixed TVL-D configuration, this zeroes out one weight at a time (Shape, Level, or Structure) or disables smoothing (window=1), and re-measures all six metrics, macro-averaged per data collection (SMD / SMAP-MSL / PSM) and then across the three. `build_series_cache.py` reduces every SMD machine / SMAP-MSL channel / PSM stream to per-timestep Shape/Level/Structure severity ***scalars*** (already column-aggregated the same way each dataset's own main-benchmark script does: sum over every column for SMD/PSM, telemetry-column-only for SMAP/MSL).

## 4. `figures/`
Run any of them from inside `figures/`
```bash
cd figures
pip install -r ../requirements.txt
```

### `generate_final_model_comparison.py` -- pre-smoothing vs. final TVL-D

The bar chart showing what the smoothing step buys: TVL-D's score before vs. after the final smoothing pass, across all 4 datasets (SMD, SMAP, MSL, PSM) and all 6 reported metrics, with everything else (weights, severities) held identical between the two bars. Reads the small, already- computed `data/final_model_combined_summary.csv` (8 rows: 4 datasets x 2 TVL-D variants) -- no re-run needed. To rebuild that CSV from scratch, run `main_benchmark/run_smd_eval.py`/`run_smap_msl_eval.py`/ `run_psm_eval.py` and combine their per-method-averaged rows into the same 8-row shape (see the script's docstring for the exact columns).
```bash
python generate_final_model_comparison.py   # -> final_model_comparison.png
```

### `generate_localization_demo.py` -- severity heatmap on a real anomaly

The concrete illustration of localization: TVL-D's detection score plotted above a per-channel severity heatmap, around the one real SMD labeled segment with the fewest true root-cause channels (the legible case, since most SMD segments implicate many channels at once). This is the one figure that recomputes severities for one machine via `../tvld_final.py`'s pipeline, rather than reading a pre-computed summary.
```bash
# expects SMD at ../main_benchmark/data/ (reuses that copy) or set SMD_DATA_DIR
python generate_localization_demo.py        # -> final_model_localization_demo.png
```

Needs that machine's `interpretation_label/` file (ground-truth root-cause channels) alongside the usual `train/test/test_label/` -- most SMD mirrors include it, but confirm yours does before running this one. This is a single-segment illustration, not an evaluation; for the macro-averaged HitRate@1.0 result across all SMD machines, run `main_benchmark/run_localization_eval.py` instead.

### `generate_benchmark_figures.py` -- the cross-method comparison figures
```bash
python generate_benchmark_figures.py
# -> cd_diagram_per_dataset.png, heatmap_all_metrics.png,
#    line_plot_headline_metrics.png
```

## Running everything, in order

1. Get the three datasets under `main_benchmark/data/` (SMD,SMAP/MSL, PSM, plus SMD's `interpretation_label/` if you want localization too). `ablation/` reuses this same copy via its own env var -- you do not need a separate copy.
2. `pip install -r requirements.txt`, then each baseline's extra deps per its `PACKAGE_NOTES.md` (or just the ones for the baselines you plan to run).
3. Run TVL-D's three `main_benchmark/` scripts, then `run_localization_eval.py`.
4. Run each baseline's `main_benchmark/baselines/<name>/run_all_<name>.py`
5. Run `ablation/build_series_cache.py` then `ablation_table_full.py` (fast, reuses step 3's checkpoints).
6. Run the three `figures/` scripts (the bundled summary CSVs they read can be rebuilt from steps 3-4's output per each script's own docstring).
