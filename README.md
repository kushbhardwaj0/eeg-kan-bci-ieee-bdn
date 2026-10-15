# Interpretable Neural Decoding: Benchmarking Kolmogorov-Arnold Networks for Cross-Subject Motor Imagery Generalization

**Authors:** Kush Bhardwaj, Abhay Kashyap, Esha Himagirish, Owen Park, Aniketh Ganta  
**Affiliation:** University of Illinois Urbana-Champaign (NeuroTech@UIUC)  
**Repository:** https://github.com/kushbhardwaj0/eeg-kan-bci-ieee-bdn  

---

## Overview

This repository contains the empirical benchmark, model implementations, and evaluation framework. 

The project investigates Kolmogorov-Arnold Networks (KANs) for non-invasive electroencephalography (EEG) motor imagery decoding under zero-calibration leave-one-subject-out (LOSO) cross-subject transfer. While standard deep architectures rely on opaque Multi-Layer Perceptron (MLP) classification heads that obscure neurophysiological decision dynamics, KANs parameterize network edges with learnable univariate B-splines. 

This work evaluates whether KANs deliver genuine performance improvements over capacity-matched MLPs, investigates spline grid sensitivity in low signal-to-noise ratio (SNR) electrophysiology, and demonstrates intrinsic neuroscientific interpretability without post-hoc attribution tools.

---

## Key Empirical Findings

1. **Statistical Parity Under Strict Capacity Matching:** Under exact parameter matching on BNCI2014-001 (4 classes, 9 subjects, 5,184 trials across 5 random seeds; 45 models per architecture), CNN-KAN (G = 3, 2,784 parameters) achieves 42.37 +/- 15.74% (seed std +/- 1.24%), matching a capacity-matched CNN-MLP (2,780 parameters, 42.20 +/- 14.45%, p = 1.000) and EEGNet (3,444 parameters, 44.55 +/- 16.75%, p = 1.000). KANs achieve complete parity rather than inflated accuracy gains.
2. **Spline Grid Collapse and Inductive Regularization:** Unconstrained spline flexibility induces severe noise overfitting in low-SNR EEG. Scaling grid resolution to G = 8 causes decoding accuracy to collapse to 30.29 +/- 5.66% (p = 0.023, Cohen's d = 1.086). Compact grids (G = 3) act as an essential regularizing inductive bias.
3. **Intrinsic Neuroscientific Interpretability:** Learned edge splines autonomously discover quadratic energy scaling (phi(x) proportional to x^2, recovering classical Filter Bank Common Spatial Pattern principles) and step-like event-related desynchronization (ERD) gating thresholds. Backward spatial filters transformed into Haufe forward patterns confirm physiological somatotopy over C3, C4, and vertex Cz.
4. **Computational Efficiency on Embedded Hardware:** Spatial-temporal convolutional backbone operations account for over 99.97% of total computation (11.81 M MACs). The two-layer KAN head requires only 3,752 MACs (<0.03%), introducing negligible latency overhead for real-time edge neural interfaces.
5. **Cross-Dataset Stability:** On the 20-subject uncalibrated PhysioNet motor imagery transfer benchmark, CNN-KAN (26.89%) matches CNN-MLP (26.22%), demonstrating generalization across heterogeneous recording cohorts.

---

## Benchmark Results

### BNCI2014-001 Cross-Subject (LOSO) Evaluation

Evaluated across 5 random initialization seeds (45 trained models per architecture). Significance assessed via two-tailed paired Wilcoxon signed-rank tests with Holm-Bonferroni correction (m = 6) relative to CNN-KAN (G = 3).

| Model | Classification Head | Grid (G) | Parameters | Accuracy (Mean +/- Subj Std, Seed Std) | Adjusted p-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **cnn_kan** | B-spline KAN (G = 3) | 3 | 2,784 | 42.37 +/- 15.74 (seed: +/- 1.24%) | Reference |
| **cnn_kan_G5** | B-spline KAN (G = 5) | 5 | 3,104 | 40.74 +/- 13.47 | 1.000 |
| **cnn_kan_G8** | B-spline KAN (G = 8) | 8 | 3,584 | 30.29 +/- 5.66 (d = 1.086) | 0.023 |
| **cnn_linear** | Dense linear | N/A | 1,556 | 40.95 +/- 12.48 | 1.000 |
| **cnn_mlp** | MLP (param-matched) | N/A | 2,780 | 42.20 +/- 14.45 | 1.000 |
| **csp_lda** | Classical CSP + LDA | N/A | N/A | 38.93 +/- 12.51 | 1.000 |
| **eegnet** | Dense classifier | N/A | 3,444 | 44.55 +/- 16.75 | 1.000 |

---

## Repository Structure

```
.
├── README.md
├── codebase/
│   ├── configs/
│   │   ├── bnci2014_001.yaml         # BNCI2014-001 preprocessing and training config
│   │   └── physionet.yaml            # PhysioNet 20-subject scaling config
│   ├── src/
│   │   ├── data/                     # Epoching, FIR filtering, pre-fetching
│   │   │   ├── loaders.py
│   │   │   └── prefetch.py
│   │   ├── models/                   # Neural network architectures
│   │   │   ├── backbone.py           # Spatial-temporal 2D convolutional backbone
│   │   │   ├── kan_head.py           # B-spline KAN classification layers
│   │   │   ├── cnn_kan.py            # Unified CNN-KAN model
│   │   │   ├── ablations.py          # Network ablation variants
│   │   │   └── skorch_wrapper.py     # Skorch adapter for MOABB integration
│   │   ├── baselines/                # Reference models (CSP+LDA, EEGNet, linear)
│   │   │   └── classifiers.py
│   │   └── utils/                    # Profiling, statistics, and visualization
│   │       ├── metrics.py            # Parameter and MACs calculations
│   │       ├── stats.py              # Wilcoxon signed-rank tests with Holm-Bonferroni
│   │       ├── visualize.py          # B-spline activation and topomap rendering
│   │       ├── pareto.py             # Pareto frontier plot generation
│   │       └── seed.py               # Deterministic seed management
│   ├── tests/                        # Comprehensive test suite
│   ├── results/                      # Saved benchmark scores, summaries, and figures
│   │   ├── figures/                  # Pareto and activation plots
│   │   ├── phase_3_kan_benchmark_results.csv
│   │   ├── phase_3_kan_benchmark_results.json
│   │   ├── phase_4_physionet_scaling_results.csv
│   │   └── pareto_efficiency_summary.json
│   ├── run_benchmark.py              # Single-model evaluation runner
│   ├── run_experiments.py            # Experiment matrix entrypoint
│   ├── run_physionet_scaling.py      # Multi-subject scaling runner
│   ├── export_results.py             # Statistical aggregation and figure generation
│   ├── environment.yml               # Conda environment specification
│   ├── requirements.txt              # Pinned pip dependencies
│   └── pytest.ini                    # Test runner configuration
```

---

## Installation and Environment Setup

### Prerequisites
- Python 3.10+ (tested with Python 3.12)
- PyTorch 2.0+ (CUDA or Apple Silicon MPS support)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/kushbhardwaj0/eeg-kan-bci-ieee-bdn.git
   cd eeg-kan-bci-ieee-bdn
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.12 -m venv env
   source env/bin/activate
   pip install -r codebase/requirements.txt
   ```
   Or via conda:
   ```bash
   cd codebase && conda env create -f environment.yml && conda activate eeg-kan-env
   ```

3. Run the test suite:
   ```bash
   cd codebase && python -m pytest -m "not slow" -q
   ```

---

## Benchmark Reproduction

All commands are executed from the `codebase/` directory inside the verified environment.

### 1. Pre-fetch Datasets
```bash
python src/data/prefetch.py --dataset bnci2014_001
python src/data/prefetch.py --dataset physionet
```

### 2. Fast Verification Run (Smoke Test)
Executes a quick 2-subject, 1-seed run to verify the entire pipeline:
```bash
python run_experiments.py --subjects 1 2 --seeds 0 --max-epochs 2 --run-dir results/smoke
python export_results.py --run-dir results/smoke
```

### 3. Full BNCI2014-001 Benchmark (5 Seeds, Multi-Model Matrix)
```bash
python run_experiments.py --config configs/bnci2014_001.yaml --run-dir results/bnci_full
```

To run an individual model (e.g., CNN-KAN):
```bash
python run_benchmark.py --model cnn_kan --dataset bnci2014_001 --seeds 0 1 2 3 4 --run-dir results/cnn_kan
```

### 4. PhysioNet MI Cross-Subject Scaling (20 Subjects)
```bash
python run_physionet_scaling.py --config configs/physionet.yaml --run-dir results/physionet_20
```

### 5. Export Statistics, LaTeX Tables, and Pareto Figures
```bash
python export_results.py --run-dir results/bnci_full
```

---

## License

This project is licensed under the MIT License.
