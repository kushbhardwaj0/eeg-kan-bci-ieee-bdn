"""
Ablation Classifier Heads for Isolating B-Spline KAN Activation Gains.
Implements CNN_Linear (single dense linear head) and CNN_MLP (2-layer MLP head)
using the same Spatial-Temporal Backbone feature extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.backbone import SpatialTemporalBackbone

DEFAULT_PARAM_BUDGET = 3000


def mlp_head_param_count(hidden_dim: int, d_bottleneck: int, num_classes: int) -> int:
    """CNN_MLP head size: fc1 (d*h + h) + BatchNorm1d (2h) + fc2 (h*C + C)."""
    return hidden_dim * (d_bottleneck + 3 + num_classes) + num_classes


def match_mlp_hidden_dim(target_params: int, d_bottleneck: int = 16, num_classes: int = 4) -> int:
    """
    Hidden width whose CNN_MLP head has the parameter count closest to ``target_params``
    (e.g. the KAN head's), so the MLP ablation isolates the activation type, not head capacity.
    """
    per_unit = d_bottleneck + 3 + num_classes
    guess = max(1, round((target_params - num_classes) / per_unit))
    candidates = [h for h in (guess - 1, guess, guess + 1) if h >= 1]
    return min(candidates, key=lambda h: abs(mlp_head_param_count(h, d_bottleneck, num_classes) - target_params))


class CNN_Linear(nn.Module):
    """
    CNN Backbone paired with a single Dense Linear classification head (16 -> 4).
    Used as an ablation baseline to evaluate spatial-temporal feature quality without non-linear heads.
    """

    def __init__(
        self,
        channels: int = 22,
        samples: int = 1001,
        num_classes: int = 4,
        F1: int = 8,
        D: int = 2,
        d_bottleneck: int = 16,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.num_classes = num_classes

        # Spatial-Temporal Feature Extractor Backbone
        self.backbone = SpatialTemporalBackbone(
            channels=channels,
            samples=samples,
            F1=F1,
            D=D,
            d_bottleneck=d_bottleneck,
            dropout_rate=dropout_rate,
        )

        # Single Dense Linear Head
        self.classifier = nn.Linear(d_bottleneck, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (Batch, 1, Channels, Time) -> Bottleneck z -> Logits (Batch, num_classes)
        """
        z = self.backbone(x)
        logits = self.classifier(z)
        return logits

    def get_parameter_count(self, budget: int = DEFAULT_PARAM_BUDGET) -> dict:
        """
        Compute parameter breakdown for Backbone, Head, and Total Trainable Parameters.
        """
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        total_params = backbone_params + head_params
        return {
            "backbone_params": backbone_params,
            "head_params": head_params,
            "total_trainable_params": total_params,
            "budget": budget,
            "budget_valid": total_params <= budget,
        }


class CNN_MLP(nn.Module):
    """
    CNN Backbone paired with a 2-Layer Multi-Layer Perceptron (MLP) classification head
    (d_bottleneck -> mlp_hidden_dim -> SiLU -> BatchNorm -> num_classes).
    Used to isolate B-spline non-linear edge activation gains from head layer depth/capacity; the
    runners set ``mlp_hidden_dim`` with ``match_mlp_hidden_dim`` so the head matches the KAN head's size.
    """

    def __init__(
        self,
        channels: int = 22,
        samples: int = 1001,
        num_classes: int = 4,
        F1: int = 8,
        D: int = 2,
        d_bottleneck: int = 16,
        mlp_hidden_dim: int = 8,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.num_classes = num_classes

        # Spatial-Temporal Feature Extractor Backbone
        self.backbone = SpatialTemporalBackbone(
            channels=channels,
            samples=samples,
            F1=F1,
            D=D,
            d_bottleneck=d_bottleneck,
            dropout_rate=dropout_rate,
        )

        # 2-Layer MLP Head (same depth as KANHead; width set by the caller to match its parameter count)
        self.fc1 = nn.Linear(d_bottleneck, mlp_hidden_dim)
        self.bn1 = nn.BatchNorm1d(mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (Batch, 1, Channels, Time) -> Bottleneck z -> MLP -> Logits (Batch, num_classes)
        """
        z = self.backbone(x)
        h = F.silu(self.fc1(z))
        h = self.bn1(h)
        logits = self.fc2(h)
        return logits

    def get_parameter_count(self, budget: int = DEFAULT_PARAM_BUDGET) -> dict:
        """
        Compute parameter breakdown for Backbone, Head, and Total Trainable Parameters.
        """
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_params = sum(
            p.numel() for p in list(self.fc1.parameters()) + list(self.bn1.parameters()) + list(self.fc2.parameters())
            if p.requires_grad
        )
        total_params = backbone_params + head_params
        return {
            "backbone_params": backbone_params,
            "head_params": head_params,
            "total_trainable_params": total_params,
            "budget": budget,
            "budget_valid": total_params <= budget,
        }
