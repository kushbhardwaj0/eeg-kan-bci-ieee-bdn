# Re-running the experiments (Phase 6)

Everything under `results/legacy/` is invalid (see `results/legacy/README.md`). This file is the
exact procedure for producing numbers that can go in the paper. Runners must be invoked from
`codebase/` with the pinned environment.

## 0. Environment

```bash
conda activate eeg-kan-env            # or your venv with codebase/requirements.txt
cd codebase
python -m pytest -m "not slow" -q     # must be green before any run
python -m pytest -m slow -q           # once per machine: downloads 2 BNCI + 2 Physionet subjects
```

On Apple Silicon make sure the Python itself is arm64 (`python -c "import platform; print(platform.machine())"`
→ `arm64`). An x86 conda under Rosetta caps torch at 2.2 and runs ~8x slower.

## 1. Clean state

* Delete any old global MOABB store so nobody can be served stale scores by accident:
  `rm -rf ~/mne_data/results/MotorImagery/CrossSubjectEvaluation/results_*.hdf5*`
  (The runners never read or write there any more — each run keeps its own store under its
  `--run-dir` — but older checkouts did.)
* `git status` must be clean. Note the commit SHA; it is recorded in every `run_info.json`.

## 2. The runs

Every command writes `scores.csv`, `run_info.json`, one MOABB store per seed (`seed{N}/…hdf5`)
and the saved fold models into `--run-dir`. Seeds are fixed in the configs (`0 1 2 3 4`).

### 2.1 Smoke (every machine, ~1 min)

```bash
python run_experiments.py --subjects 1 2 --seeds 0 --max-epochs 2 --run-dir results/smoke
python export_results.py --run-dir results/smoke
```
Check: `scores.csv` has sessions `0train` and `1test`, a `seed` column, 4 classes; the LaTeX
snippet compiles; `git status` still clean.

### 2.2 BNCI2014_001 — main table (7 neural fits + CSP+LDA, 5 seeds)

```bash
python run_experiments.py --config configs/bnci2014_001.yaml --run-dir results/bnci_full
python export_results.py --run-dir results/bnci_full
```
Models: `cnn_kan` (G=3), `cnn_kan_G5`, `cnn_kan_G8`, `cnn_linear`, `cnn_mlp`, `eegnet`, `csp_lda`.
The main grid size is fitted once (the sweep no longer duplicates it).

### 2.3 PhysionetMI — 20-subject scaling (2 models, 5 seeds)

```bash
python run_physionet_scaling.py --config configs/physionet.yaml --run-dir results/physionet_20
python export_results.py --run-dir results/physionet_20 --label PhysionetMI
```
Events are `left_hand, right_hand, feet, hands`; `rest` is excluded (the legacy run included
it and collapsed to a constant predictor).

### 2.4 Combined Pareto figure (both datasets in one figure)

```bash
python -m src.utils.pareto --summary results/bnci_full/phase_3_kan_benchmark_results.json \
    results/physionet_20/phase_3_kan_benchmark_results.json --labels BNCI2014-001 PhysionetMI \
    --output-dir results/figures --results-dir results
```

### 2.5 Interpretability reference fold (PR 5)

Train one extra CNN_KAN fold on CPU so the saved model unpickles on any machine:

```bash
python run_benchmark.py --model cnn_kan --seeds 0 --device cpu --run-dir results/bnci_cpu_reference
python -m src.utils.visualize --run-dir results/bnci_cpu_reference --output-dir results/figures
```

`--run-dir` picks the first CNN_KAN fold model (lowest seed, then held-out subject); use
`--model-path <run-dir>/seed0/Models_CrossSubject/BNCI2014-001/<subject>/cnn_kan/fitted_model_<fold>.pkl`
to choose one. Channel names come from the run's `run_info.json`; the spatial patterns use the
fold's held-out subject's trials (`--subjects` to override, `--no-patterns` for weights only).
The model is loaded onto the CPU, so MPS- or CUDA-trained folds work too.

## 3. Cost (measured on an M4 MacBook, MPS, fit time only)

| dataset | per fit (40/30 epochs × all folds) | per seed | 5 seeds |
|---|---|---|---|
| BNCI2014_001, one neural model | ~19 min (18–21; early stopping may cut it) | 7 fits ≈ 2.2 h | **≈ 11 h** |
| BNCI2014_001, CSP+LDA | < 1 min | | |
| PhysionetMI (20 subj), cnn_kan + eegnet | — | ≈ 80 min (extrapolated from 2 subjects) | **≈ 7 h** |

CPU is ~8x slower than MPS; a CUDA box is faster. The validation split removes 20% of the
training batches relative to these numbers. Everything else (loading, epoching, predicting)
is under 2% of the total.

## 4. Splitting the work across machines

Split by **config**, never by seed: all five seeds of one config must run on the same device
type (a CPU run and an MPS run with the same seed draw different dropout masks, so they are
different seeds, not the same experiment). Suggested split:

| machine | runs |
|---|---|
| A | `run_experiments.py --mode ablations` (cnn_kan, cnn_linear, cnn_mlp, eegnet, csp_lda) |
| B | `run_experiments.py --mode sweep` (cnn_kan_G3, G5, G8) |
| C | `run_physionet_scaling.py` |

`--mode` keeps the same `run_dir` layout, so results from different machines are merged by
concatenating the `scores.csv` files (keep the `seed` and `model` columns) and exporting the
merged directory. Record which machine ran what; `run_info.json` already holds the device,
platform, package versions and git SHA.

## 5. Reproducibility notes

* Same seed, same device type → bit-for-bit identical (verified on MPS and CPU).
* Same seed, different device type → different model. Report the device with any number.
* `overwrite: true` in the configs and the per-seed `hdf5_path` mean a code change can never
  be served cached scores. `--no-overwrite` exists only for resuming an interrupted run of the
  *same* code.
* Do not edit `configs/*.yaml` between seeds of one run.

## 6. Handing results to the paper

`export_results.py` writes, per run directory:
* `phase_3_kan_benchmark_results.json` — per-model mean over subjects, subject std, seed std,
  n, params, MACs (KAN layers counted), and Holm-corrected Wilcoxon tests against `cnn_kan`;
* `phase_3_table_snippet.tex` — booktabs table, escaped, compiles standalone;
* `figures/pareto_*.{png,pdf}` and `pareto_efficiency_summary.json` — frontier from the data.

Cite only these. The legacy numbers must not appear in the manuscript.
