"""
Spatial-Temporal CNN Feature Extractor for EEG Motor Imagery Signals
Compresses raw EEG trials (Batch, 1, Channels, Time) into low-dimensional bottleneck vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    """
    Conv2d layer with max-norm weight constraint along spatial filters (dim 0).
    """

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                self.weight.copy_(
                    torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm)
                )
        return super().forward(x)


class SpatialTemporalBackbone(nn.Module):
    """
    Compact Spatial-Temporal CNN feature extractor for EEG signals.
    Emits low-dimensional bottleneck feature vectors z of shape (Batch, d_bottleneck).
    """

    def __init__(
        self,
        channels: int = 22,
        samples: int = 1001,
        F1: int = 8,
        D: int = 2,
        d_bottleneck: int = 16,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.F1 = F1
        self.D = D
        self.d_bottleneck = d_bottleneck
        F2 = F1 * D  # 16 channels

        # Block 1: Temporal 1D Convolution (Bandpass 8-30 Hz Mu/Beta dynamics)
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, 64),
            padding="same",
            bias=False,
        )
        self.bn_temporal = nn.BatchNorm2d(F1)

        # Block 2: Spatial Depthwise Convolution (Virtual scalp topomaps)
        self.conv_spatial = Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F2,
            kernel_size=(channels, 1),
            groups=F1,
            bias=False,
            max_norm=1.0,
        )
        self.bn_spatial = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 3: Depthwise-separable convolution whose pointwise step projects straight to the
        # d_bottleneck channels, so the bottleneck has d_bottleneck independent features. (An earlier
        # F2 -> F1 pointwise conv followed by an F1 -> d_bottleneck Linear capped its rank at F1.)
        self.conv_separable = nn.Conv2d(
            in_channels=F2,
            out_channels=F2,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F2,
            bias=False,
        )
        self.conv_pointwise = nn.Conv2d(
            in_channels=F2,
            out_channels=d_bottleneck,
            kernel_size=1,
            bias=False,
        )
        self.bn_separable = nn.BatchNorm2d(d_bottleneck)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Global average pooling (in forward) followed by normalisation of the bottleneck vector
        self.bn_bottleneck = nn.BatchNorm1d(d_bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting raw trial tensor to bottleneck feature vector z.
        Accepts 3D (Batch, Channels, Time) or 4D (Batch, 1, Channels, Time) inputs.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.conv_temporal(x)
        x = self.bn_temporal(x)

        x = self.conv_spatial(x)
        x = self.bn_spatial(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv_separable(x)
        x = self.conv_pointwise(x)
        x = self.bn_separable(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Global spatial-temporal mean pooling (Batch, d_bottleneck)
        z = x.mean(dim=(-2, -1))
        z = self.bn_bottleneck(z)
        return z
