"""
Unit Test Suite for Phase 2: CNN-KAN Architecture, Backbone, Spline KAN Head, and Parameter Budget
"""

import numpy as np
import pytest
import torch
from src.models.backbone import SpatialTemporalBackbone
from src.models.kan_head import KANHead, KANLinear
from src.models.cnn_kan import CNN_KAN
from src.models.skorch_wrapper import SkorchEEGClassifier


def test_backbone_forward_and_shapes():
    """
    Verify SpatialTemporalBackbone output vector shape is (Batch, d_bottleneck) for 3D and 4D inputs.
    """
    batch_size = 16
    channels = 22
    samples = 1001
    d_bottleneck = 16

    backbone = SpatialTemporalBackbone(
        channels=channels, samples=samples, d_bottleneck=d_bottleneck
    )
    backbone.eval()

    # Test 4D input: (Batch, 1, Channels, Time)
    x4d = torch.randn(batch_size, 1, channels, samples)
    z4d = backbone(x4d)
    assert z4d.shape == (batch_size, d_bottleneck)

    # Test 3D input: (Batch, Channels, Time)
    x3d = torch.randn(batch_size, channels, samples)
    z3d = backbone(x3d)
    assert z3d.shape == (batch_size, d_bottleneck)

    # Test arbitrary sequence length T=750
    x_var = torch.randn(batch_size, 1, channels, 750)
    z_var = backbone(x_var)
    assert z_var.shape == (batch_size, d_bottleneck)


def test_kan_linear_spline_computation():
    """
    Verify KANLinear forward evaluation and dynamic knot grid updates.
    """
    batch_size = 32
    in_features = 16
    out_features = 10

    kan_layer = KANLinear(
        in_features=in_features, out_features=out_features, grid_size=3, spline_order=3
    )
    x = torch.randn(batch_size, in_features)
    y = kan_layer(x)

    assert y.shape == (batch_size, out_features)
    assert not torch.isnan(y).any()

    # Test knot grid range update routine
    kan_layer.update_grid_from_samples(x)
    assert kan_layer.grid.shape == (in_features, 3 + 2 * 3 + 1)


def test_kan_head_forward_and_backward():
    """
    Verify KANHead forward pass and backward gradient flow produces non-NaN gradients.
    """
    batch_size = 16
    in_features = 16
    hidden_dim = 10
    num_classes = 4

    head = KANHead(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        grid_size=3,
        spline_order=3,
    )
    z = torch.randn(batch_size, in_features, requires_grad=True)
    logits = head(z)

    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()

    # Compute dummy loss and backward pass
    loss = logits.sum()
    loss.backward()

    for name, param in head.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient detected in {name}"


def test_cnn_kan_parameter_budget():
    """
    Assert total trainable parameters of CNN_KAN are strictly <= 3,000 parameters.
    """
    model = CNN_KAN(
        channels=22,
        samples=1001,
        num_classes=4,
        F1=8,
        D=2,
        d_bottleneck=16,
        kan_hidden_dim=8,
        grid_size=3,
        spline_order=3,
    )
    params_info = model.get_parameter_count()

    total_params = params_info["total_trainable_params"]
    print(f"\n[CNN_KAN PROFILE] Backbone Params: {params_info['backbone_params']:,}")
    print(f"[CNN_KAN PROFILE] KAN Head Params: {params_info['kan_head_params']:,}")
    print(f"[CNN_KAN PROFILE] Total Trainable Params: {total_params:,}")

    assert params_info["budget_valid"], f"Parameter budget exceeded: {total_params} > 3000"
    assert total_params <= 3000


def test_skorch_integration_with_cnn_kan():
    """
    Verify CNN_KAN fits and predicts inside SkorchEEGClassifier.
    """
    num_samples = 64
    channels = 22
    samples = 1001
    num_classes = 4

    X_train = np.random.randn(num_samples, channels, samples).astype(np.float32)
    y_train = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int64)

    clf = SkorchEEGClassifier(
        module=CNN_KAN,
        module_kwargs={
            "channels": channels,
            "samples": samples,
            "num_classes": num_classes,
            "d_bottleneck": 16,
            "kan_hidden_dim": 8,
            "grid_size": 3,
            "spline_order": 3,
        },
        lr=1e-3,
        batch_size=32,
        max_epochs=2,
        patience=2,
        verbose=0,
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_train)
    probs = clf.predict_proba(X_train)

    assert preds.shape == (num_samples,)
    assert probs.shape == (num_samples, num_classes)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_moabb_cache_device_decoupling():
    """Verify device parameter is deferred and not baked into estimator get_params."""
    clf = SkorchEEGClassifier(module=CNN_KAN, module_kwargs={"channels": 22}, device=None)
    params = clf.get_params()
    assert params["device"] is None, "Default device parameter must remain None in get_params"


def test_samples_stripping_for_global_pooling_models():
    """Verify 'samples' key is removed from module_kwargs for global-pooling models in _build_net."""
    clf = SkorchEEGClassifier(
        module=CNN_KAN,
        module_kwargs={"channels": 22, "samples": 1000},
        max_epochs=1,
        verbose=0,
    )
    clf._build_net()
    assert hasattr(clf.net_, "module__channels")
    assert not hasattr(clf.net_, "module__samples")


def test_kan_grid_covers_inputs_after_update():
    """After a grid update, inputs far outside the initial [-1.5, 1.5] grid lie inside the knot range."""
    kan_head = KANHead(in_features=16, hidden_dim=10, num_classes=4, grid_range=(-1.5, 1.5))

    torch.manual_seed(42)
    z = torch.randn(64, 16) * 2.5
    k = kan_head.kan1.spline_order
    inside = lambda g: ((z >= g[:, k]) & (z <= g[:, -k - 1])).float().mean().item()
    assert inside(kan_head.kan1.grid) < 0.85  # most of this input spills off the default grid

    kan_head.update_grid(z)
    assert inside(kan_head.kan1.grid) == 1.0

    logits = kan_head.train()(z)
    assert logits.shape == (64, 4)
    assert torch.isfinite(logits).all()
