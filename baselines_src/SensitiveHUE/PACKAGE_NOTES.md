# SensitiveHUE -- vendoring notes

Original repo: https://github.com/yuesuoqingqiu/SensitiveHUE.git.

## What was excluded from the vendored copy

Copied everything except:

- **`.git/`** -- version-control history, not needed to run the code.
- **`data/`** (the repo's bundled `data/SWaT/`, ~22M) -- a pre-processed
  sample dataset for a dataset this project doesn't evaluate on (SWaT).
  This project's own datasets (SMD, SMAP, MSL, PSM) are sourced separately,
  same as for every other method in this package -- see the top-level
  README's "Get the data from..." links per dataset.
- **`sensitive_hue/model_states/`** -- the repo's own bundled SWaT
  experiment log/checkpoint folder, irrelevant to this project's runs.
- **`__pycache__/`** (four of them, one per subpackage) -- compiled bytecode,
  regenerated automatically on first import.

Nothing in `base/`, `config/`, `sensitive_hue/`, `rec_strategy/`, `utils/`,
`main.py`, `main_rec_strategy.py`, `README.md`, `requirements.txt`, or
`figures/` (the paper's own framework/results diagrams, kept for reference)
was modified. Net size: 35M -> 2.9M.

## Extra pip dependencies

SensitiveHUE needs `torch` (it is a PyTorch Transformer model) on top of this
package's base `requirements.txt` (numpy/pandas/scipy/scikit-learn/stumpy).
The repo's own `requirements.txt` (vendored here unmodified) lists:

```
numpy
scikit-learn
torch
PyYAML
scipy
```

All but `torch` and `PyYAML` are already in this package's base
requirements. Install both before running either SensitiveHUE script:

```bash
pip install torch PyYAML
```

(CPU-only `torch` is sufficient -- nothing in this port uses a GPU.)

## How to run

### 1. Main-benchmark (clean-data) evaluation -- SMD, SMAP, MSL, PSM

```bash
cd main_benchmark/baselines/sensitivehue
python run_all_sensitivehue.py            # all three datasets
# or just one:
python run_all_sensitivehue.py smd
python run_all_sensitivehue.py smap_msl
python run_all_sensitivehue.py psm
```

Expects the same `../../data/` layout as `main_benchmark/run_smd_eval.py` /
`run_smap_msl_eval.py` / `run_psm_eval.py` (one shared data folder for every
method in the package). Override with the same env vars those scripts use:
`SMD_DATA_DIR`, `SMAP_MSL_DATA_DIR`, `SMAP_MSL_LABELS`, `PSM_DATA_DIR`.

Each entity (28 SMD machines, 81 SMAP/MSL channels, 1 PSM stream) trains in
its own subprocess via `run_one_sensitivehue.py`, cached immediately to
`checkpoints_sensitivehue/<dataset>/<id>.pkl` -- interrupting and re-running
`run_all_sensitivehue.py` resumes rather than restarts. Trained per-entity
model weights themselves land under a fresh `./scratch/shue_<tag>_*/model_states/`
directory, created on demand (see "Caveat" below) -- **this is also where
`run_sensitivehue_noise.py` (step 2) looks for SMD's trained weights**, so
run the SMD portion of this step before the noise-robustness step.

Output: `sensitivehue_smd_per_machine.csv`, `sensitivehue_smap_msl_per_channel.csv`,
`sensitivehue_psm_results.csv`, all written next to the scripts.

### 2. Noise-robustness ablation -- SMD only

Requires step 1's SMD run to have completed first (inference-only -- it
reloads the trained weights step 1 produced rather than retraining):

```bash
cd main_benchmark/baselines/sensitivehue && python run_all_sensitivehue.py smd   # if not already done
cd ../../../noise_robustness/baselines/sensitivehue
python run_sensitivehue_noise.py
```

Set `SENSITIVEHUE_WEIGHTS_DIR` if you trained SMD's weights into a
non-default scratch location. Same noise levels (10/20/30/40/50%), same
deterministic per-(machine, noise_pct) seeding, and the same
`method, machine, noise_pct, ...` CSV schema as `noise_robustness/run_tvld_noise.py`,
written to `sensitivehue_noise_results.csv` next to the script, resumable
via `checkpoints/<machine>_noise<pct>.pkl`.


