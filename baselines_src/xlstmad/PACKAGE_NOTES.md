# xLSTMAD baseline

## Source

Vendored from the official repo: https://github.com/Nyderx/xlstmad.git
(commit on `main` as cloned; no tag pinned). Copied here minus `.git/`
(history, ~1.4MB), `data/` (`sample_dataset.csv`, ~5.8MB -- the repo's own
toy demo dataset, unrelated to SMD/SMAP/MSL/PSM), and `__pycache__/`
(compiled bytecode). Nothing else in the upstream repo was excluded -- the
full clone is only 7.2MB and everything remaining (`xlstmad.py`,
`dataset.py`, `bench.py`, `example.py`, `requirements.txt`, `LICENSE`,
`ReadMe.md`, `.gitignore`) is source/docs, not data or build output. No
pretrained weights or checkpoints ship with the upstream repo at all --
xLSTMAD always trains from scratch (see Caveats).

## Extra dependencies

Beyond this package's base `requirements.txt` (numpy/pandas/scipy/
scikit-learn/stumpy), xLSTMAD additionally needs:

```
xlstm==2.0.5
lightning==2.6.1
torch            # + torchvision per https://pytorch.org/get-started/locally/ for your platform/CUDA version
torchmetrics     # pulled in by xlstm/lightning; used directly by xlstmad.py (BinaryAUROC, MeanSquaredError)
```

(`xlstm` and `lightning` pins are copied verbatim from the upstream repo's
own `requirements.txt`, in `baselines_src/xlstmad/requirements.txt`.) Install
torch/torchvision first, per the upstream README, since the right wheel
depends on your CUDA toolkit (or CPU-only); then `pip install xlstm lightning`.

## Running

### Main benchmark (clean data) -- SMD, SMAP/MSL, PSM

```bash
cd main_benchmark/baselines/xlstmad
python run_all_xlstmad.py            # all three datasets
python run_all_xlstmad.py smd        # just SMD
python run_all_xlstmad.py smap_msl   # just SMAP/MSL
python run_all_xlstmad.py psm        # just PSM
```

Reuses the SAME `data/` folder and the SAME env vars
(`SMD_DATA_DIR`, `SMAP_MSL_DATA_DIR`, `SMAP_MSL_LABELS`, `PSM_DATA_DIR`) as
`main_benchmark/run_smd_eval.py` / `run_smap_msl_eval.py` / `run_psm_eval.py`
-- defaults to `main_benchmark/data/`, so one copy of each dataset serves
every method in the package, TVL-D included. Per-entity raw scores are
cached under `checkpoints_xlstmad/{smd,smap_msl,psm}/`, so an interrupted
run resumes without retraining. Writes `xlstmad_smd_per_machine.csv`,
`xlstmad_smap_msl_per_channel.csv`, `xlstmad_psm_results.csv` (same
`full_metrics` columns as every other method, plus `elapsed`/`epochs`).

### Noise robustness (SMD only, 5 noise levels)

```bash
cd noise_robustness/baselines/xlstmad
python run_all_xlstmad_noise.py
```

Reuses the same SMD data and `TVLD_DATA_DIR` env var as
`noise_robustness/run_tvld_noise.py` (via the shared `common_noise.py`).
**Full retrain per (machine, noise_pct) combo** -- unlike TVL-D or any
baseline that only needs cached severities/scores, xLSTMAD's clean-data run
only cached its output *scores*, not trained weights, so robustness testing
here means training 28 machines x 5 noise levels = 140 models from scratch.
Checkpointed per-combo under `checkpoints/`, resumable; writes
`xlstmad_noise_results.csv` with the `method, machine, noise_pct, <metrics>`
schema `run_tvld_noise.py` uses, so it concatenates directly with that file
and every other baseline's noise CSV.


## Caveats

- **Trains from scratch every time** -- xLSTMAD ships no pretrained
  weights, and this package's clean-data runner only persists output
  scores (not model weights) to its cache, so the noise-robustness variant
  cannot reuse a clean-data checkpoint and must retrain per noise level.
- **GPU strongly recommended but not required.** The model's sLSTM cell has
  two backends: `cuda` (fast, requires NVIDIA Compute Capability >= 8.0 and
  a matching CUDA toolkit) and `vanilla` (CPU-compatible, slower). Both
  runners here (`xlstmad_runner.py`) hardcode `slstm_backend="vanilla"` so
  the whole package stays runnable CPU-only like TVL-D and the other
  baselines, at a real speed cost versus the `cuda` backend the upstream
  repo recommends by default.
- **PyTorch install is platform-specific** and not pinned in
  `requirements.txt` -- install it yourself per
  https://pytorch.org/get-started/locally/ before `pip install -r
  baselines_src/xlstmad/requirements.txt`.
- Determinism: `run_xlstmad()` calls `L.seed_everything(SEED=0)` per entity,
  but full bit-for-bit reproducibility across machines/PyTorch builds is not
  guaranteed by upstream (standard caveat for any PyTorch/cuDNN training
  run, independent of this port).
