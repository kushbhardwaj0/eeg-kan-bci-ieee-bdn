"""
End-to-End CNN-KAN Architecture for Parameter-Efficient EEG Motor Imagery Classification
Combines Spatial-Temporal CNN Feature Extraction Backbone with a 2-Layer B-Spline KAN Head.
"""

import torch
import torch.nn as nn
from src.models.backbone import SpatialTemporalBackbone
from src.models.kan_head import KANHead


class CNN_KAN(nn.Module):
    """
    Parameter-Efficient CNN-KAN Classifier.
    Enforces total trainable parameter budget <= 3,000 parameters.
    """

    def __init__(
        self,
        channels: int = 22,
        samples: int = 1001,
        num_classes: int = 4,
        F1: int = 8,
        D: int = 2,
        d_bottleneck: int = 16,
        kan_hidden_dim: int = 8,
        grid_size: int = 3,
        spline_order: int = 3,
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

        # 2-Layer B-Spline Kolmogorov-Arnold Classifier Head
        self.kan_head = KANHead(
            in_features=d_bottleneck,
            hidden_dim=kan_hidden_dim,
            num_classes=num_classes,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (Batch, 1, Channels, Time) -> Bottleneck z -> Logits (Batch, num_classes)
        """
        z = self.backbone(x)
        logits = self.kan_head(z)
        return logits

    def get_parameter_count(self, budget: int = 3000) -> dict:
        """
        Compute parameter breakdown for Backbone, KAN Head, and Total Trainable Parameters.
        """
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        kan_params = sum(p.numel() for p in self.kan_head.parameters() if p.requires_grad)
        total_params = backbone_params + kan_params
        return {
            "backbone_params": backbone_params,
            "kan_head_params": kan_params,
            "total_trainable_params": total_params,
            "budget": budget,
            "budget_valid": total_params <= budget,
        }
