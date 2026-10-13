"""
Unit Test Suite for Phase 3: Ablation Models, Statistical Significance Suite, and Exporters.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.ablations import CNN_Linear, CNN_MLP
from src.models.skorch_wrapper import SkorchEEGClassifier
from src.utils.stats import (
    compute_wilcoxon_test,
    compute_cohens_d,
    compute_rank_biserial,
    run_statistical_suite,
)
from run_experiments import format_latex_results_table


def test_ablation_models_forward_and_shapes():
    """
    Verify CNN_Linear and CNN_MLP output shapes (Batch, num_classes) for 3D and 4D inputs.
    """
    batch_size = 16
    channels = 22
    samples = 1001
    num_classes = 4

    x4d = torch.randn(batch_size, 1, channels, samples)
    x3d = torch.randn(batch_size, channels, samples)

    model_linear = CNN_Linear(channels=channels, samples=samples, num_classes=num_classes)
    model_mlp = CNN_MLP(channels=channels, samples=samples, num_classes=num_classes)

    model_linear.eval()
    model_mlp.eval()

    out_lin_4d = model_linear(x4d)
    out_lin_3d = model_linear(x3d)
    assert out_lin_4d.shape == (batch_size, num_classes)
    assert out_lin_3d.shape == (batch_size, num_classes)
    assert not torch.isnan(out_lin_4d).any()

    out_mlp_4d = model_mlp(x4d)
    out_mlp_3d = model_mlp(x3d)
    assert out_mlp_4d.shape == (batch_size, num_classes)
    assert out_mlp_3d.shape == (batch_size, num_classes)
    assert not torch.isnan(out_mlp_4d).any()


def test_ablation_parameter_counts():
    """
    Assert trainable parameter counts for CNN_Linear and CNN_MLP are strictly <= 3,000.
    """
    lin_model = CNN_Linear(channels=22, samples=1001, num_classes=4)
    mlp_model = CNN_MLP(channels=22, samples=1001, num_classes=4)

    lin_info = lin_model.get_parameter_count()
    mlp_info = mlp_model.get_parameter_count()

    print(f"\n[CNN_Linear PROFILE] Params: {lin_info['total_trainable_params']:,}")
    print(f"[CNN_MLP PROFILE] Params: {mlp_info['total_trainable_params']:,}")

    assert lin_info["budget_valid"]
    assert lin_info["total_trainable_params"] <= 3000

    assert mlp_info["budget_valid"]
    assert mlp_info["total_trainable_params"] <= 3000


def test_stats_wilcoxon_cohens_d_rank_biserial():
    """
    Verify Wilcoxon signed-rank test, Cohen's d, and rank-biserial correlation functions.
    """
    scores_a = np.array([0.55, 0.60, 0.58, 0.62, 0.50, 0.70, 0.65, 0.58, 0.61])
    scores_b = np.array([0.45, 0.48, 0.51, 0.50, 0.42, 0.58, 0.52, 0.49, 0.50])

    res = compute_wilcoxon_test(scores_a, scores_b)
    assert "statistic" in res
    assert "p_value" in res
    assert res["p_value"] < 0.05, "Target model is consistently superior, p-value should be < 0.05"

    d = compute_cohens_d(scores_a, scores_b)
    assert d > 0.0, "Cohen's d should be positive"

    r = compute_rank_biserial(scores_a, scores_b)
    assert r > 0.0, "Rank biserial correlation should be positive"


def test_run_statistical_suite():
    """
    Verify statistical suite runner with multi-model subject results DataFrame.
    """
    data = []
    subjects = [f"S{i+1:02d}" for i in range(9)]

    for s in subjects:
        data.append({"model": "cnn_kan", "subject": s, "score": np.random.uniform(0.55, 0.65)})
        data.append({"model": "eegnet", "subject": s, "score": np.random.uniform(0.40, 0.50)})
        data.append({"model": "csp_lda", "subject": s, "score": np.random.uniform(0.35, 0.45)})

    df = pd.DataFrame(data)
    stats_out = run_statistical_suite(df, target_model="cnn_kan")

    assert "eegnet" in stats_out
    assert "csp_lda" in stats_out
    assert stats_out["eegnet"]["target_model"] == "cnn_kan"
    assert "p_value" in stats_out["eegnet"]


def test_latex_exporter_format():
    """
    Verify format_latex_results_table builds valid LaTeX string snippet.
    """
    sample_summary = {
        "cnn_kan": {
            "name": "CNN_KAN",
            "head_type": "B-Spline KAN",
            "grid_size": 3,
            "params": 2944,
            "mean_acc": 0.523,
            "std_acc": 0.12,
            "p_value": 0.021,
        },
        "eegnet": {
            "name": "EEGNet-8,2",
            "head_type": "Dense Linear",
            "grid_size": "N/A",
            "params": 6516,
            "mean_acc": 0.454,
            "std_acc": 0.144,
        },
    }

    latex_code = format_latex_results_table(sample_summary)
    assert r"\begin{table}" in latex_code
    assert r"\end{table}" in latex_code
    assert r"CNN\_KAN" in latex_code  # underscores are escaped (audit item 15)
    assert "EEGNet-8,2" in latex_code


def test_skorch_integration_with_ablations():
    """
    Verify CNN_Linear and CNN_MLP fit and predict with SkorchEEGClassifier.
    """
    num_samples = 32
    channels = 22
    samples = 1001
    num_classes = 4

    X_train = np.random.randn(num_samples, channels, samples).astype(np.float32)
    y_train = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int64)

    for ModuleClass in [CNN_Linear, CNN_MLP]:
        clf = SkorchEEGClassifier(
            module=ModuleClass,
            module_kwargs={
                "channels": channels,
                "samples": samples,
                "num_classes": num_classes,
            },
            lr=1e-3,
            batch_size=16,
            max_epochs=2,
            patience=2,
            verbose=0,
        )

        clf.fit(X_train, y_train)
        preds = clf.predict(X_train)
        probs = clf.predict_proba(X_train)

        assert preds.shape == (num_samples,)
        assert probs.shape == (num_samples, num_classes)


def test_skorch_save_and_load_params(tmp_path):
    """Verify SkorchEEGClassifier parameter saving and loading via state_dict."""
    from src.models.cnn_kan import CNN_KAN
    X = np.random.randn(8, 22, 100).astype(np.float32)
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)

    clf1 = SkorchEEGClassifier(
        module=CNN_KAN,
        module_kwargs={"channels": 22},
        max_epochs=1,
        batch_size=4,
        verbose=0,
    )
    clf1.fit(X, y)
    pred1 = clf1.predict(X)

    weight_path = tmp_path / "model_params.pt"
    clf1.save_params(str(weight_path))
    assert weight_path.exists()

    clf2 = SkorchEEGClassifier(
        module=CNN_KAN,
        module_kwargs={"channels": 22},
        verbose=0,
    )
    clf2.load_params(str(weight_path))
    pred2 = clf2.predict(X)

    assert pred2.shape == (8,)
    assert np.array_equal(pred1, pred2)


def test_session_filtering_logic():
    """Verify session filtering logic retains only '1test' rows."""
    raw_df = pd.DataFrame({
        "pipeline": ["cnn_kan", "cnn_kan", "csp_lda", "csp_lda"],
        "session": ["0train", "1test", "0train", "1test"],
        "subject": [1, 1, 1, 1],
        "score": [0.85, 0.45, 0.75, 0.40],
    })

    filtered_df = raw_df[raw_df["session"] == "1test"] if "1test" in raw_df["session"].values else raw_df
    assert len(filtered_df) == 2
    assert set(filtered_df["session"].unique()) == {"1test"}
    assert list(filtered_df["score"]) == [0.45, 0.40]
