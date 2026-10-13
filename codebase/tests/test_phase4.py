"""
Unit Test Suite for Phase 4: Physionet Scaling Benchmark, Pareto Efficiency Profiler, and Interpretability Suite.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_kan import CNN_KAN
from src.utils.pareto import compute_pareto_frontier, generate_pareto_plots
from src.utils import visualize as viz
from src.utils.visualize import plot_bspline_activations, plot_spatial_topomaps

# BNCI2014_001 channel order as MOABB returns it (channel 22 is POz, not Oz: audit item 13).
BNCI_CH_NAMES = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                 "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]


def test_pareto_frontier_computation():
    """
    Verify Pareto frontier index computation on known cost/yield point sets.
    """
    # Costs (e.g. parameter counts): [100, 200, 300, 400]
    # Yields (e.g. accuracy):         [50.0, 60.0, 55.0, 70.0]
    costs = np.array([100, 200, 300, 400])
    yields = np.array([50.0, 60.0, 55.0, 70.0])

    pareto_indices = compute_pareto_frontier(costs, yields)
    
    # Points 0 (100, 50), 1 (200, 60), 3 (400, 70) are Pareto optimal.
    # Point 2 (300, 55) is dominated by Point 1 (200, 60).
    assert 0 in pareto_indices
    assert 1 in pareto_indices
    assert 3 in pareto_indices
    assert 2 not in pareto_indices


def test_pareto_figures_generation(tmp_path):
    """
    Verify generate_pareto_plots creates publication figures and summary JSON.
    """
    fixture = {"summary": {
        "cnn_kan":    {"name": "cnn_kan",    "params": 2976, "macs_mmac": 11.9, "mean_acc": 0.45, "std_acc": 0.10, "n_subjects": 9},
        "cnn_linear": {"name": "cnn_linear", "params": 1556, "macs_mmac": 11.8, "mean_acc": 0.41, "std_acc": 0.12, "n_subjects": 9},
        "eegnet":     {"name": "eegnet",     "params": 3444, "macs_mmac": 9.6,  "mean_acc": 0.44, "std_acc": 0.15, "n_subjects": 9},
        "csp_lda":    {"name": "csp_lda",    "params": None, "macs_mmac": None, "mean_acc": 0.40, "std_acc": 0.13, "n_subjects": 9},
    }}
    summary = generate_pareto_plots({"BNCI2014-001": fixture}, output_dir=tmp_path, results_dir=tmp_path)

    assert summary["pareto_models"]["BNCI2014-001"] == ["cnn_kan", "cnn_linear"]  # eegnet dominated; csp_lda has no params -> excluded
    assert summary["profiles"]["BNCI2014-001"]["csp_lda"]["params"] is None
    assert "cnn_kan" in summary["dominance"]["BNCI2014-001"]
    assert summary["profiles"]["BNCI2014-001"]["cnn_kan"]["mean_acc"] == 45.0  # points come from the data
    assert all(Path(f).exists() for f in summary["figures"])  # outside the repo paths stay absolute
    with pytest.raises(ValueError):
        generate_pareto_plots(output_dir=tmp_path, results_dir=tmp_path)  # no built-in numbers
    assert (tmp_path / "pareto_efficiency_summary.json").exists()
    assert (tmp_path / "pareto_accuracy_vs_parameters.png").exists()
    assert (tmp_path / "pareto_accuracy_vs_parameters.pdf").exists()
    assert (tmp_path / "pareto_accuracy_vs_macs.png").exists()
    assert (tmp_path / "pareto_accuracy_vs_macs.pdf").exists()


@pytest.fixture
def trained_fold(tmp_path):
    """A CNN_KAN fold model trained for 2 epochs on random data, saved the way MOABB saves it."""
    import json
    import pickle
    from src.models.skorch_wrapper import SkorchEEGClassifier

    run_dir = tmp_path / "run"
    fold_dir = run_dir / "seed0" / "Models_CrossSubject" / "BNCI2014-001" / "3" / "cnn_kan"
    fold_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(json.dumps({
        "dataset": "BNCI2014-001", "subjects": [1, 2, 3],
        "shape": {"n_chans": 22, "n_times": 250, "n_classes": 4, "ch_names": BNCI_CH_NAMES, "sfreq": 250.0},
    }))
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 22, 250)).astype("float32")
    y = np.repeat(np.arange(4), 16)
    est = SkorchEEGClassifier(module=CNN_KAN, module_kwargs={"channels": 22, "num_classes": 4},
                              batch_size=16, max_epochs=2, patience=0, device="cpu", random_state=0).fit(X, y)
    path = fold_dir / "fitted_model_0.pkl"
    with open(path, "wb") as f:
        pickle.dump(est, f)
    return {"run_dir": run_dir, "path": path, "X": X}


def _capture_titles(monkeypatch):
    titles = []
    original = viz.plt.suptitle
    monkeypatch.setattr(viz.plt, "suptitle", lambda t, *a, **k: (titles.append(t), original(t, *a, **k))[1])
    return titles


def test_load_fold_model_returns_cpu_eval_module(trained_fold):
    est, module = viz.load_fold_model(trained_fold["path"])
    assert isinstance(module, CNN_KAN) and not module.training
    assert all(p.device.type == "cpu" for p in module.parameters())
    assert est.classes_.tolist() == [0, 1, 2, 3]


def test_fold_model_discovery(trained_fold):
    folds = viz.find_fold_models(trained_fold["run_dir"])
    assert folds == [trained_fold["path"]]
    assert viz.held_out_subject(folds[0]) == 3
    assert viz.find_run_info(folds[0])["shape"]["ch_names"][-1] == "POz"


def test_haufe_patterns_follow_the_formula(trained_fold):
    """a_j = Sigma_g w_j / (w_j^T Sigma_g w_j) on the temporally filtered signal of filter j's group."""
    _, module = viz.load_fold_model(trained_fold["path"])
    X = trained_fold["X"]
    W = viz.spatial_filter_weights(module)
    A = viz.compute_spatial_patterns(module, X)
    assert A.shape == W.shape == (16, 22)

    with torch.no_grad():
        h = module.backbone.bn_temporal(module.backbone.conv_temporal(torch.from_numpy(X)[:, None])).numpy()
    j = 5
    g = j // 2  # D = 2
    cov = np.cov(h[:, g].transpose(1, 0, 2).reshape(22, -1))
    expected = cov @ W[j] / (W[j] @ cov @ W[j])
    np.testing.assert_allclose(A[j], expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.einsum("jc,jc->j", W, A), np.ones(16), rtol=1e-4)  # w^T a = 1


def test_spatial_topomaps_use_given_channels_and_patterns(trained_fold, tmp_path, monkeypatch):
    _, module = viz.load_fold_model(trained_fold["path"])
    titles = _capture_titles(monkeypatch)
    png = plot_spatial_topomaps(module, BNCI_CH_NAMES, output_dir=tmp_path, X=trained_fold["X"])
    assert png.exists() and png.stat().st_size > 0
    assert titles[-1].startswith("Learned") and "Spatial Patterns" in titles[-1]
    with pytest.raises(ValueError):
        plot_spatial_topomaps(module, BNCI_CH_NAMES[:-1], output_dir=tmp_path)


def test_untrained_model_is_not_titled_learned(tmp_path, monkeypatch):
    model = CNN_KAN(channels=22, samples=1001, num_classes=4, grid_size=3).eval()
    titles = _capture_titles(monkeypatch)
    assert plot_bspline_activations(model, output_dir=tmp_path, trained=False).exists()
    assert plot_spatial_topomaps(model, BNCI_CH_NAMES, output_dir=tmp_path, trained=False).exists()
    assert all(t.startswith("Untrained") and "Learned" not in t for t in titles)


def test_visualize_cli_on_saved_fold(trained_fold, tmp_path, monkeypatch):
    """--model-path renders both figures on CPU with channel names from run_info (channel 22 = POz)."""
    titles = _capture_titles(monkeypatch)
    out = viz.main(["--model-path", str(trained_fold["path"]), "--no-patterns", "--output-dir", str(tmp_path)])
    assert out["channels"][21] == "POz" and len(out["channels"]) == 22
    assert out["splines"].exists() and out["topomaps"].exists()
    assert len(titles) == 2 and all(t.startswith("Learned") for t in titles)

    out = viz.main(["--run-dir", str(trained_fold["run_dir"]), "--no-patterns", "--output-dir", str(tmp_path / "b")])
    assert out["model_path"] == str(trained_fold["path"])
    with pytest.raises(SystemExit):
        viz.main([])  # a trained model or run directory is required


@pytest.mark.slow
def test_visualize_patterns_from_real_trials(trained_fold, tmp_path):
    """Spatial patterns from real BNCI trials (downloads subject 1 on first run)."""
    pytest.importorskip("moabb")
    out = viz.run_interpretability_suite(model_path=trained_fold["path"], output_dir=tmp_path, subjects=[1])
    assert out["channels"][21] == "POz" and out["topomaps"].exists()


@pytest.mark.slow
def test_physionet_scaling_benchmark_runner(tmp_path):
    """
    Fast verification test of run_physionet_scaling_benchmark on 2 subjects.
    """
    from run_physionet_scaling import run_physionet_scaling_benchmark

    df = run_physionet_scaling_benchmark(num_subjects=2, seed=42, output_dir=tmp_path, overwrite=True)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "model" in df.columns or "pipeline" in df.columns
    assert "score" in df.columns
    assert (tmp_path / "phase_4_physionet_scaling_results.csv").exists()
    assert (tmp_path / "phase_4_physionet_scaling_results.json").exists()
