"""
B-Spline Kolmogorov-Arnold Network (KAN) Classification Head
Implements vectorized B-spline activation layers for non-linear, parameter-efficient classification.
"""

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from skorch.callbacks import Callback
except ImportError:  # pragma: no cover
    Callback = object


def b_splines(x: torch.Tensor, grid: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Compute B-spline basis function evaluations for input tensor x over grid.

    Args:
        x: Input tensor of shape (Batch, in_features)
        grid: Knot grid tensor of shape (in_features, grid_size + 2*k + 1)
        k: Spline order (default: 3 for cubic B-splines)

    Returns:
        B-spline basis evaluations of shape (Batch, in_features, grid_size + k)
    """
    x = x.unsqueeze(-1)  # (Batch, in_features, 1)
    bases = (x >= grid[:, :-1]) & (x < grid[:, 1:])  # (Batch, in_features, total_knots-1)
    bases = bases.to(x.dtype)

    for p in range(1, k + 1):
        left_num = x - grid[:, : -(p + 1)]
        left_denom = grid[:, p:-1] - grid[:, : -(p + 1)]
        right_num = grid[:, p + 1 :] - x
        right_denom = grid[:, p + 1 :] - grid[:, 1:-p]

        # Avoid divide-by-zero
        left_denom = torch.where(left_denom == 0, torch.ones_like(left_denom), left_denom)
        right_denom = torch.where(right_denom == 0, torch.ones_like(right_denom), right_denom)

        term1 = (left_num / left_denom) * bases[:, :, :-1]
        term2 = (right_num / right_denom) * bases[:, :, 1:]
        bases = term1 + term2

    return bases


class KANLinear(nn.Module):
    """
    1D Kolmogorov-Arnold Network (KAN) Layer combining base activation (SiLU)
    with B-spline non-linear edge functions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 3,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple = (-1.5, 1.5),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = k = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        # Compute knot vector grid [-1.5, 1.5] extended by k knots on left and right
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - k * h,
            grid_range[1] + k * h,
            grid_size + 2 * k + 1,
            dtype=torch.float32,
        ).repeat(in_features, 1)
        self.register_buffer("grid", grid)

        # Base linear weight matrix: (out_features, in_features)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # Spline coefficient tensor: (out_features, in_features, grid_size + k)
        num_spline_coeffs = grid_size + k
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, num_spline_coeffs)
        )

        # Per-edge spline scale (the base path's per-edge scale is base_weight itself).
        self.scale_spline_param = nn.Parameter(torch.ones(out_features, in_features))

        self.reset_parameters(scale_noise)

    def reset_parameters(self, scale_noise: float = 0.1):
        """``scale_base`` / ``scale_spline`` set the initial magnitude of the base and spline paths."""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.base_weight.mul_(self.scale_base)
            self.scale_spline_param.fill_(self.scale_spline)
            noise = (
                torch.rand(self.out_features, self.in_features, self.grid_size + self.spline_order)
                - 0.5
            ) * scale_noise
            self.spline_weight.copy_(noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward evaluation: y = Base(SiLU(x)) + Splines(x)
        """
        # Base linear path
        base_output = F.linear(F.silu(x), self.base_weight)

        # Spline non-linear path
        spline_basis = b_splines(x, self.grid, k=self.spline_order)  # (Batch, in_features, grid_size + k)

        # Weighted sum across spline coefficients
        # spline_weight: (out, in, coeffs), spline_basis: (Batch, in, coeffs)
        spline_output = torch.einsum("bic,oic->bo", spline_basis, self.spline_weight * self.scale_spline_param.unsqueeze(-1))

        return base_output + spline_output

    def update_grid_from_samples(self, x: torch.Tensor, margin: float = 0.01):
        """
        Move each input's knot grid to cover the observed range of ``x`` and refit the spline
        coefficients by least squares so the layer computes (approximately) the same function on
        those samples. Without the refit, moving the knots would silently change the learned
        function. The fit is exact when the range is unchanged; otherwise it is the best cubic
        spline on the new knots, evaluated at the samples (needs ``len(x) >= grid_size + k``).
        """
        with torch.no_grad():
            x = x.detach().to(self.grid.dtype)
            k = self.spline_order

            # Spline-path output per edge (unscaled; scale_spline_param is kept as is).
            old_bases = b_splines(x, self.grid, k=k)                                  # (B, in, c)
            target = torch.einsum("bic,oic->ibo", old_bases, self.spline_weight)     # (in, B, out)

            x_min = x.min(dim=0).values - margin
            x_max = x.max(dim=0).values + margin
            h = (x_max - x_min) / self.grid_size                                     # (in,)
            steps = torch.arange(-k, self.grid_size + k + 1, device=x.device, dtype=x.dtype)
            new_grid = x_min.unsqueeze(1) + steps.unsqueeze(0) * h.unsqueeze(1)      # (in, G + 2k + 1)

            new_bases = b_splines(x, new_grid, k=k).permute(1, 0, 2)                # (in, B, c)
            # Least squares on CPU (MPS has no lstsq); gelsd copes with rank-deficient bases.
            sol = torch.linalg.lstsq(
                new_bases.cpu().double(), target.cpu().double(), driver="gelsd"
            ).solution                                                               # (in, c, out)
            coeffs = sol.to(device=self.spline_weight.device, dtype=self.spline_weight.dtype)

            self.grid.copy_(new_grid)
            self.spline_weight.copy_(coeffs.permute(2, 0, 1))


class KANHead(nn.Module):
    """
    2-Layer B-Spline Kolmogorov-Arnold Network Classification Head.
    Maps bottleneck vectors z (Batch, in_features) to logits (Batch, num_classes).
    """

    def __init__(
        self,
        in_features: int = 16,
        hidden_dim: int = 10,
        num_classes: int = 4,
        grid_size: int = 3,
        spline_order: int = 3,
        grid_range: tuple = (-1.5, 1.5),
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Inputs arrive batch-normalised (CNN_KAN's backbone ends in BatchNorm1d); the knot grids
        # are then fitted to the actual input range during training by ``KANGridUpdate``.

        # Layer 1: KANLinear (in_features -> hidden_dim)
        self.kan1 = KANLinear(
            in_features=in_features,
            out_features=hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Layer 2: KANLinear (hidden_dim -> num_classes)
        self.kan2 = KANLinear(
            in_features=hidden_dim,
            out_features=num_classes,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.kan1(z)
        h = h if (self.training and h.size(0) == 1) else self.bn1(h)
        logits = self.kan2(h)
        return logits

    def update_grid(self, z: torch.Tensor, margin: float = 0.01) -> None:
        """
        Refit both layers' grids to a batch of head inputs ``z``. The second layer is fitted to the
        first layer's (updated) outputs. BatchNorm runs in eval mode so its statistics are untouched.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                self.kan1.update_grid_from_samples(z, margin=margin)
                self.kan2.update_grid_from_samples(self.bn1(self.kan1(z)), margin=margin)
        finally:
            self.train(was_training)


class KANGridUpdate(Callback):
    """
    skorch callback: after each of the first ``n_updates`` epochs, refit every KAN head's knot grids
    to the current bottleneck distribution of (up to ``max_samples``) training trials.

    The module must expose ``backbone`` and ``kan_head`` (``CNN_KAN``). Runs after the epoch's
    scoring and early-stopping callbacks, so the checkpoint EarlyStopping keeps is the state that
    produced that epoch's validation loss; the refit grid is a buffer, so ``load_best`` restores it
    together with the weights.
    """

    def __init__(self, n_updates: int = 5, max_samples: int = 1024, margin: float = 0.01):
        self.n_updates = n_updates
        self.max_samples = max_samples
        self.margin = margin

    def _training_inputs(self, net, dataset) -> Optional[torch.Tensor]:
        chunks, n = [], 0
        for batch in net.get_iterator(dataset, training=False):
            X = batch[0] if isinstance(batch, (tuple, list)) else batch
            X = torch.as_tensor(X)
            chunks.append(X)
            n += X.shape[0]
            if n >= self.max_samples:
                break
        if not chunks:
            return None
        return torch.cat(chunks)[: self.max_samples]

    def on_epoch_end(self, net, dataset_train=None, **kwargs):
        epoch = len(net.history)
        module = net.module_
        if epoch > self.n_updates or dataset_train is None:
            return
        if not (hasattr(module, "backbone") and hasattr(module, "kan_head")):
            return
        X = self._training_inputs(net, dataset_train)
        if X is None:
            return
        n_coeffs = module.kan_head.kan1.grid_size + module.kan_head.kan1.spline_order
        if X.shape[0] < n_coeffs:
            return  # too few trials to determine the spline coefficients
        was_training = module.training
        module.eval()
        try:
            with torch.no_grad():
                z = module.backbone(X.to(net.device))
            module.kan_head.update_grid(z, margin=self.margin)
        finally:
            module.train(was_training)
