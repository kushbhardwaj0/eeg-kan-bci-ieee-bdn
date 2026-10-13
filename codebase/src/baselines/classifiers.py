"""
Baseline EEG Classifier Architectures (CSP+LDA, EEGNet, SVM)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

try:
    import mne
    from mne.decoding import CSP
except ImportError:
    CSP = None


# =====================================================================
# Baseline 1: CSP + LDA Pipeline
# =====================================================================
class CSP_LDA(ClassifierMixin, BaseEstimator):
    """
    Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA) Pipeline.
    """

    def __init__(self, n_components: int = 6, log: bool = True):
        self.n_components = n_components
        self.log = log
        self.pipeline_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if CSP is None:
            raise ImportError("MNE is required for CSP. Please run: pip install mne")

        # Ensure input shape (B, Channels, Time) and float64 for MNE CSP
        if X.ndim == 4:
            X = X.squeeze(1)
        X = np.asarray(X, dtype=np.float64)

        self.classes_ = np.unique(y)
        self.csp_ = CSP(n_components=self.n_components, log=self.log)
        self.lda_ = LinearDiscriminantAnalysis()
        self.pipeline_ = Pipeline([("csp", self.csp_), ("lda", self.lda_)])

        self.pipeline_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 4:
            X = X.squeeze(1)
        X = np.asarray(X, dtype=np.float64)
        return self.pipeline_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 4:
            X = X.squeeze(1)
        X = np.asarray(X, dtype=np.float64)
        return self.pipeline_.predict_proba(X)


# =====================================================================
# Baseline 2: Standard EEGNet-8,2 Architecture (Lawhern et al. 2018)
# =====================================================================
class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with max norm weight constraint on spatial filters."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.data = torch.renorm(
                    self.weight.data, p=2, dim=0, maxnorm=self.max_norm
                )
        return super().forward(x)


class EEGNet(nn.Module):
    """
    EEGNet-8,2 Architecture for Motor Imagery EEG Decoding.
    Reference: Lawhern et al. (2018), Journal of Neural Engineering.
    """

    def __init__(
        self,
        channels: int = 22,
        samples: int = 750,
        num_classes: int = 4,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.num_classes = num_classes

        # Block 1: 1D Temporal Convolution + Depthwise Spatial Convolution
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise_conv = Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(channels, 1),
            groups=F1,
            bias=False,
            max_norm=1.0,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2: Depthwise-Separable Convolution (Spatial-Temporal Pointwise Combination)
        self.spatial_conv = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate flattened linear head input features dynamically
        self.flatten_features = self._get_flatten_size(channels, samples)
        self.classifier = nn.Linear(self.flatten_features, num_classes)

    def _get_flatten_size(self, C: int, T: int) -> int:
        # Probe in eval mode so the dummy pass does not update BatchNorm running statistics.
        was_training = self.training
        self.eval()
        try:
            return self._probe_flatten_size(C, T)
        finally:
            self.train(was_training)

    def _probe_flatten_size(self, C: int, T: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            x = self.conv1(dummy)
            x = self.bn1(x)
            x = self.depthwise_conv(x)
            x = self.bn2(x)
            x = F.elu(x)
            x = self.pooling1(x)
            x = self.dropout1(x)

            x = self.spatial_conv(x)
            x = self.pointwise_conv(x)
            x = self.bn3(x)
            x = F.elu(x)
            x = self.pooling2(x)
            x = self.dropout2(x)
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Support 3D input tensors: (Batch, Channels, Time) -> (Batch, 1, Channels, Time)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1:] != (1, self.channels, self.samples):
            raise ValueError(
                f"EEGNet was built for input (batch, 1, {self.channels}, {self.samples}) "
                f"or (batch, {self.channels}, {self.samples}); got {tuple(x.shape)}. "
                "Rebuild the model with the dataset's channel and sample counts."
            )
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        x = self.spatial_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


# =====================================================================
# Baseline 3: SVM (Bandpower Features)
# =====================================================================
class SVM_Bandpower(ClassifierMixin, BaseEstimator):
    """
    Support Vector Machine (SVM) on Log-Variance Bandpower Features.
    """

    def __init__(self, kernel: str = "rbf", C: float = 1.0):
        self.kernel = kernel
        self.C = C
        self.svc_ = None

    def _extract_bandpower(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 4:
            X = X.squeeze(1)
        # Compute log-variance across time dimension per electrode channel
        var = np.var(X, axis=-1)
        log_var = np.log(var + 1e-6)
        return log_var

    def fit(self, X: np.ndarray, y: np.ndarray):
        features = self._extract_bandpower(X)
        self.classes_ = np.unique(y)
        self.svc_ = SVC(kernel=self.kernel, C=self.C)
        self.svc_.fit(features, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        features = self._extract_bandpower(X)
        return self.svc_.predict(features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        features = self._extract_bandpower(X)
        if hasattr(self.svc_, "predict_proba"):
            return self.svc_.predict_proba(features)
        # Fallback Decision function to pseudo-probabilities
        df = self.svc_.decision_function(features)
        exp_df = np.exp(df - np.max(df, axis=-1, keepdims=True))
        return exp_df / np.sum(exp_df, axis=-1, keepdims=True)
