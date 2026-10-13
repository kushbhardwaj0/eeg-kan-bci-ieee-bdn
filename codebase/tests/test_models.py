"""
Numerical model checks (plan1.md PR 2): properties that shape-only tests cannot catch.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbone import SpatialTemporalBackbone


# --------------------------------------------------------------------------- backbone
@pytest.mark.parametrize("d_bottleneck", [8, 16, 32])
def test_bottleneck_full_rank(d_bottleneck):
    """Audit item 4: the d-dim bottleneck must carry d independent features (it used to be rank 8)."""
    torch.manual_seed(0)
    backbone = SpatialTemporalBackbone(channels=22, samples=1001, d_bottleneck=d_bottleneck).eval()
    with torch.no_grad():
        z = backbone(torch.randn(256, 1, 22, 1001))
    assert z.shape == (256, d_bottleneck)
    assert torch.linalg.matrix_rank(torch.cov(z.T)).item() == d_bottleneck


def test_backbone_has_no_low_rank_projection():
    """No Linear layer after pooling: the pointwise conv itself produces the d_bottleneck channels."""
    backbone = SpatialTemporalBackbone(channels=22, samples=1001, d_bottleneck=16)
    assert not any(isinstance(m, nn.Linear) for m in backbone.modules())
    assert backbone.conv_pointwise.out_channels == 16
    assert backbone.bn_bottleneck.num_features == 16


def test_backbone_parameter_count_default():
    """22 channels, F1=8, D=2, d=16: 1,488 parameters (same total as before the rank fix)."""
    backbone = SpatialTemporalBackbone(channels=22, samples=1001, d_bottleneck=16)
    assert sum(p.numel() for p in backbone.parameters() if p.requires_grad) == 1488


# --------------------------------------------------------------------------- KAN head
from src.models.kan_head import KANGridUpdate, KANHead, KANLinear, b_splines  # noqa: E402


def _spline_output(layer: KANLinear, x: torch.Tensor) -> torch.Tensor:
    return layer(x) - torch.nn.functional.linear(torch.nn.functional.silu(x), layer.base_weight)


@pytest.mark.parametrize("grid_size", [3, 5, 8])
def test_bspline_partition_of_unity(grid_size):
    """Cubic B-spline bases sum to 1 everywhere inside the grid range."""
    layer = KANLinear(4, 2, grid_size=grid_size, grid_range=(-1.5, 1.5))
    x = torch.linspace(-1.5, 1.5 - 1e-4, 301).unsqueeze(1).repeat(1, 4)
    bases = b_splines(x, layer.grid, k=layer.spline_order)
    assert bases.shape == (301, 4, grid_size + layer.spline_order)
    torch.testing.assert_close(bases.sum(-1), torch.ones(301, 4), atol=1e-5, rtol=0)


def test_kan_linear_has_no_duplicate_base_scale():
    """Audit item 27: base_weight already is the per-edge base scale; 16->8 at G=3 has 128+768+128 params."""
    layer = KANLinear(16, 8, grid_size=3)
    assert not hasattr(layer, "scale_base_param")
    assert sum(p.numel() for p in layer.parameters()) == 16 * 8 * (1 + 6 + 1)


def test_kan_linear_init_scales_are_used():
    torch.manual_seed(0)
    a = KANLinear(16, 8, scale_base=1.0, scale_spline=1.0)
    torch.manual_seed(0)
    b = KANLinear(16, 8, scale_base=0.5, scale_spline=2.0)
    torch.testing.assert_close(b.base_weight, 0.5 * a.base_weight)
    torch.testing.assert_close(b.scale_spline_param, torch.full((8, 16), 2.0))


def test_grid_refit_preserves_function_when_range_unchanged():
    """Refitting onto the same knot positions must reproduce the layer exactly (least squares is exact)."""
    torch.manual_seed(0)
    layer = KANLinear(4, 3, grid_size=3, grid_range=(-1.5, 1.5))
    with torch.no_grad():
        layer.spline_weight.normal_(0, 0.5)
    x = torch.rand(256, 4) * 3 - 1.5
    x[0], x[1] = -1.5, 1.5
    grid_before, y_before = layer.grid.clone(), layer(x)
    layer.update_grid_from_samples(x, margin=0.0)
    torch.testing.assert_close(layer.grid, grid_before, atol=1e-5, rtol=0)
    assert (layer(x) - y_before).abs().max().item() < 1e-3


def test_grid_refit_tracks_function_when_range_moves():
    """
    With a new range the old spline is generally not representable on the new knots, so the refit is
    approximate; it must stay close and be far better than moving the knots without refitting.
    """
    torch.manual_seed(0)
    layer = KANLinear(16, 8, grid_size=3, grid_range=(-1.5, 1.5))
    with torch.no_grad():
        layer.spline_weight.normal_(0, 0.5)
    x = torch.randn(512, 16) * 0.5
    ref = _spline_output(layer, x)

    moved_only = KANLinear(16, 8, grid_size=3)
    moved_only.load_state_dict(layer.state_dict())
    layer.update_grid_from_samples(x)
    with torch.no_grad():
        moved_only.grid.copy_(layer.grid)  # the old behaviour: new knots, old coefficients

    rel = lambda y: ((y - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    assert rel(_spline_output(layer, x)) < 0.1
    assert rel(_spline_output(layer, x)) < 0.25 * rel(_spline_output(moved_only, x))
    core_lo, core_hi = layer.grid[:, layer.spline_order], layer.grid[:, -layer.spline_order - 1]
    assert ((x >= core_lo) & (x <= core_hi)).all(), "every sample must lie inside the refit grid"


def test_kan_head_update_grid_leaves_batchnorm_stats():
    head = KANHead(in_features=16, hidden_dim=8, num_classes=4).train()
    stats = head.bn1.running_mean.clone(), int(head.bn1.num_batches_tracked)
    head.update_grid(torch.randn(128, 16) * 2)
    assert head.training
    assert torch.equal(head.bn1.running_mean, stats[0]) and int(head.bn1.num_batches_tracked) == stats[1]
    assert not hasattr(head, "in_bn")


def _fit_cnn_kan(kan_grid_updates):
    np = pytest.importorskip("numpy")
    from src.models.cnn_kan import CNN_KAN
    from src.models.skorch_wrapper import SkorchEEGClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 1, 22, 250)).astype("float32")
    y = np.repeat(np.arange(4), 16)
    est = SkorchEEGClassifier(module=CNN_KAN, module_kwargs={"channels": 22, "num_classes": 4},
                              batch_size=16, max_epochs=2, patience=0, device="cpu",
                              random_state=0, kan_grid_updates=kan_grid_updates)
    return est.fit(X, y)


def test_kan_grid_update_callback_runs_during_fit():
    initial = KANLinear(16, 8, grid_size=3).grid
    est = _fit_cnn_kan(kan_grid_updates=5)
    assert any(isinstance(cb, KANGridUpdate) for cb in est.net_.callbacks)
    grid = est.net_.module_.kan_head.kan1.grid.cpu()
    assert not torch.allclose(grid, initial), "grid should have been refit to the bottleneck range"

    off = _fit_cnn_kan(kan_grid_updates=0)
    assert not any(isinstance(cb, KANGridUpdate) for cb in off.net_.callbacks)
    torch.testing.assert_close(off.net_.module_.kan_head.kan1.grid.cpu(), initial)


# --------------------------------------------------------------------------- matched MLP ablation
from src.models.ablations import CNN_Linear, CNN_MLP, match_mlp_hidden_dim, mlp_head_param_count  # noqa: E402
from src.models.cnn_kan import CNN_KAN  # noqa: E402


def test_mlp_head_param_count_formula():
    for h in (8, 56):
        model = CNN_MLP(channels=22, num_classes=4, d_bottleneck=16, mlp_hidden_dim=h)
        assert model.get_parameter_count()["head_params"] == mlp_head_param_count(h, 16, 4)


@pytest.mark.parametrize("grid_size", [3, 5, 8])
@pytest.mark.parametrize("n_chans,n_classes", [(22, 4), (64, 4)])
def test_matched_mlp_within_5_percent_of_kan_head(grid_size, n_chans, n_classes):
    kan = CNN_KAN(channels=n_chans, num_classes=n_classes, grid_size=grid_size).get_parameter_count()
    h = match_mlp_hidden_dim(kan["kan_head_params"], d_bottleneck=16, num_classes=n_classes)
    mlp = CNN_MLP(channels=n_chans, num_classes=n_classes, mlp_hidden_dim=h).get_parameter_count()
    assert abs(mlp["head_params"] - kan["kan_head_params"]) <= 0.05 * kan["kan_head_params"]
    assert mlp["backbone_params"] == kan["backbone_params"]


def test_build_pipeline_gives_cnn_mlp_the_matched_width():
    from src.utils import experiment as ex

    est = ex.build_pipeline("cnn_mlp", 22, 1001, 4)
    kan_head = CNN_KAN(channels=22, num_classes=4).get_parameter_count()["kan_head_params"]
    mlp = CNN_MLP(**{k: v for k, v in est.module_kwargs.items()}).get_parameter_count()
    assert abs(mlp["head_params"] - kan_head) <= 0.05 * kan_head
    assert ex.build_pipeline("cnn_mlp", 22, 1001, 4, mlp_hidden_dim=8).module_kwargs["mlp_hidden_dim"] == 8
    # the matched width follows the grid size used for the KAN comparison
    assert ex.build_pipeline("cnn_mlp", 22, 1001, 4, grid_size=8).module_kwargs["mlp_hidden_dim"] > est.module_kwargs["mlp_hidden_dim"]


def test_parameter_budget_is_a_parameter():
    for model in (CNN_KAN(), CNN_Linear(), CNN_MLP()):
        info = model.get_parameter_count(budget=100)
        assert info["budget"] == 100 and info["budget_valid"] is False
        assert model.get_parameter_count()["budget"] == 3000
