"""
Phase 1 Integration and Unit Test Suite
"""

import os
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Add codebase root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.baselines.classifiers import CSP_LDA, EEGNet, SVM_Bandpower
from src.models.skorch_wrapper import SkorchEEGClassifier
from src.utils.metrics import compute_classification_metrics, get_model_flops, get_parameter_count


def test_baseline_eegnet_forward():
    """Verify shape contract and forward pass for standard EEGNet model."""
    set_seed(42)
    model = EEGNet(channels=22, samples=750, num_classes=4)
    dummy_x = torch.randn(8, 1, 22, 750)
    out = model(dummy_x)

    assert out.shape == (8, 4), f"Expected shape (8, 4), got {out.shape}"
    assert not torch.isnan(out).any(), "EEGNet forward pass produced NaN values!"
    print("[PASS] EEGNet forward pass verified.")


def test_eegnet_rejects_wrong_input_shape():
    """A model built for 22x1001 must refuse 64x481 input instead of failing deep in the classifier."""
    model = EEGNet(channels=22, samples=1001, num_classes=4)
    with pytest.raises(ValueError, match=r"22, 1001"):
        model(torch.randn(2, 1, 64, 481))
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 22, 750))
    assert model(torch.randn(2, 22, 1001)).shape == (2, 4)  # 3D input still accepted


def test_eegnet_construction_leaves_batchnorm_untouched():
    """The flatten-size probe must not update BatchNorm running statistics or the train/eval mode."""
    model = EEGNet(channels=22, samples=1001, num_classes=4)
    assert model.training
    for bn in (model.bn1, model.bn2, model.bn3):
        assert int(bn.num_batches_tracked) == 0
        assert torch.equal(bn.running_mean, torch.zeros_like(bn.running_mean))
        assert torch.equal(bn.running_var, torch.ones_like(bn.running_var))


def test_baseline_csp_lda():
    """Verify CSP+LDA pipeline fit and predict on dummy numpy EEG tensor."""
    set_seed(42)
    model = CSP_LDA(n_components=4)
    dummy_x = np.random.randn(20, 1, 22, 750).astype(np.float32)
    dummy_y = np.random.randint(0, 4, size=20)

    model.fit(dummy_x, dummy_y)
    preds = model.predict(dummy_x)
    probs = model.predict_proba(dummy_x)

    assert preds.shape == (20,), f"Expected preds shape (20,), got {preds.shape}"
    assert probs.shape == (20, 4), f"Expected probs shape (20, 4), got {probs.shape}"
    print("[PASS] CSP+LDA baseline fit and predict verified.")


def test_baseline_svm_bandpower():
    """Verify SVM bandpower baseline fit and predict on dummy EEG tensor."""
    set_seed(42)
    model = SVM_Bandpower(kernel="rbf", C=1.0)
    dummy_x = np.random.randn(20, 1, 22, 750).astype(np.float32)
    dummy_y = np.random.randint(0, 4, size=20)

    model.fit(dummy_x, dummy_y)
    preds = model.predict(dummy_x)
    probs = model.predict_proba(dummy_x)

    assert preds.shape == (20,), f"Expected preds shape (20,), got {preds.shape}"
    assert probs.shape == (20, 4), f"Expected probs shape (20, 4), got {probs.shape}"
    print("[PASS] SVM Bandpower baseline fit and predict verified.")


def test_skorch_wrapper():
    """Verify SkorchEEGClassifier adapter compatibility with PyTorch module."""
    try:
        import skorch
    except ImportError:
        print("[SKIP] skorch is not installed in the environment.")
        return

    set_seed(42)
    model_kwargs = {"channels": 22, "samples": 750, "num_classes": 4}
    clf = SkorchEEGClassifier(
        module=EEGNet,
        module_kwargs=model_kwargs,
        max_epochs=2,
        batch_size=8,
        verbose=0,
    )

    dummy_x = np.random.randn(16, 1, 22, 750).astype(np.float32)
    dummy_y = np.random.randint(0, 4, size=16)

    clf.fit(dummy_x, dummy_y)
    preds = clf.predict(dummy_x)
    probs = clf.predict_proba(dummy_x)

    assert preds.shape == (16,)
    assert probs.shape == (16, 4)
    print("[PASS] SkorchEEGClassifier adapter successfully verified.")


def test_metrics_and_profiling():
    """Verify parameter counting, FLOPs profiling, and classification metrics."""
    model = EEGNet(channels=22, samples=1001, num_classes=4)
    params = get_parameter_count(model)
    assert params["trainable_params"] == 3444, f"EEGNet parameters expected exactly 3,444, got {params['trainable_params']}"

    flops = get_model_flops(model, input_size=(1, 22, 1001))
    assert "macs" in flops
    assert np.isfinite(flops["macs_value"]) and flops["macs_value"] > 0

    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_prob = np.eye(4)[y_true]

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    print("[PASS] Metrics calculation and parameter/FLOPs profiling verified.")


def test_sklearn_is_classifier_compliance():
    """Verify all custom estimators pass sklearn.base.is_classifier check."""
    from sklearn.base import is_classifier
    csp = CSP_LDA()
    svm = SVM_Bandpower()
    skorch = SkorchEEGClassifier(module=EEGNet, module_kwargs={"channels": 22})

    assert is_classifier(csp), "CSP_LDA must pass is_classifier check"
    assert is_classifier(svm), "SVM_Bandpower must pass is_classifier check"
    assert is_classifier(skorch), "SkorchEEGClassifier must pass is_classifier check"


def test_classes_attribute_populated_on_fit():
    """Verify classes_ attribute is populated after fit across estimators."""
    np.random.seed(42)
    X = np.random.randn(12, 22, 100).astype(np.float32)
    y = np.array([0, 1, 2, 3] * 3, dtype=np.int64)

    csp = CSP_LDA(n_components=2)
    csp.fit(X, y)
    assert hasattr(csp, "classes_")
    assert np.array_equal(csp.classes_, [0, 1, 2, 3])

    svm = SVM_Bandpower()
    svm.fit(X, y)
    assert hasattr(svm, "classes_")
    assert np.array_equal(svm.classes_, [0, 1, 2, 3])


if __name__ == "__main__":
    print("=== Phase 1 Unit Test Suite Verification ===")
    test_baseline_eegnet_forward()
    test_baseline_csp_lda()
    test_baseline_svm_bandpower()
    test_skorch_wrapper()
    test_metrics_and_profiling()
    test_sklearn_is_classifier_compliance()
    test_classes_attribute_populated_on_fit()
    print("\n[ALL PHASE 1 UNIT TESTS COMPLETED SUCCESSFULLY!]")
