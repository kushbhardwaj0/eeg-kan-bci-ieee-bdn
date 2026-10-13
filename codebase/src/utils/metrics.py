"""
Analytics, Profiling (Params/MACs), and Evaluation Metrics
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

try:
    from torchinfo import summary as torchinfo_summary
except ImportError:
    torchinfo_summary = None

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None

try:
    from src.models.kan_head import KANLinear
except Exception:  # pragma: no cover
    KANLinear = None


def get_parameter_count(model: nn.Module) -> Dict[str, int]:
    """Total and trainable parameter counts of a PyTorch module."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total_params, "trainable_params": trainable_params}


# --------------------------------------------------------------------------- KAN MACs
def kan_linear_macs(layer: nn.Module, batch: int = 1) -> int:
    """
    Multiply-accumulates of one KANLinear forward for ``batch`` inputs.

    base path     : in * out                      (F.linear on SiLU(x))
    spline path   : in * out * (grid_size + k)    (einsum over basis coefficients)
    basis eval    : in * (grid_size + 2k) * k * 4 (Cox-de Boor recursion, ~4 ops per update)
    ptflops counts neither custom module (audit item 14), so this hook supplies it.
    """
    n_in, n_out = int(layer.in_features), int(layer.out_features)
    g, k = int(layer.grid_size), int(layer.spline_order)
    base = n_in * n_out
    spline = n_in * n_out * (g + k)
    basis = n_in * (g + 2 * k) * k * 4
    return int(batch * (base + spline + basis))


def _kan_linear_hook(module: nn.Module, inputs, output):
    batch = int(inputs[0].shape[0]) if inputs and hasattr(inputs[0], "shape") else 1
    module.__flops__ += kan_linear_macs(module, batch)


def _custom_hooks() -> dict:
    return {KANLinear: _kan_linear_hook} if KANLinear is not None else {}


def _macs(model: nn.Module, input_size: Tuple[int, ...]) -> float:
    """Numeric MACs for one forward pass of ``input_size`` (no batch dim). NaN on failure."""
    if get_model_complexity_info is None:
        return float("nan")
    was_training = model.training
    model.eval()
    try:
        macs, _ = get_model_complexity_info(
            model, tuple(input_size), as_strings=False, print_per_layer_stat=False,
            verbose=False, custom_modules_hooks=_custom_hooks(),
        )
        return float(macs)
    except Exception:
        return float("nan")
    finally:
        model.train(was_training)


def _fmt_macs(value: float) -> str:
    if value is None or np.isnan(value):
        return "N/A"
    if value >= 1e9:
        return f"{value / 1e9:.2f} GMac"
    if value >= 1e6:
        return f"{value / 1e6:.2f} MMac"
    if value >= 1e3:
        return f"{value / 1e3:.2f} KMac"
    return f"{value:.0f} Mac"


def get_model_flops(
    model: nn.Module, input_size: Tuple[int, ...] = (1, 22, 750)
) -> Dict[str, Union[float, str]]:
    """
    MACs and parameter count for a PyTorch module.

    Returns ``macs`` / ``params`` as strings (legacy keys) plus numeric ``macs_value``,
    ``macs_mmac`` and ``params_value``. Failures give NaN, never 0.
    """
    total = _macs(model, input_size)
    params = get_parameter_count(model)["trainable_params"]
    return {
        "macs": _fmt_macs(total),
        "params": f"{params:,}",
        "macs_value": total,
        "macs_mmac": total / 1e6 if not np.isnan(total) else float("nan"),
        "params_value": params,
    }


def profile_model(model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, Union[float, int]]:
    """
    Whole-model MACs plus a backbone / head split when the model exposes ``backbone`` and a
    head (``kan_head`` or ``classifier``). The head is profiled on the bottleneck vector.
    """
    out = get_model_flops(model, input_size)
    prof: Dict[str, Union[float, int]] = {
        "params": int(out["params_value"]),
        "macs": float(out["macs_value"]),
        "macs_mmac": float(out["macs_mmac"]),
        "backbone_macs": float("nan"),
        "head_macs": float("nan"),
    }
    backbone = getattr(model, "backbone", None)
    head = getattr(model, "kan_head", None) or getattr(model, "classifier", None)
    if isinstance(backbone, nn.Module):
        prof["backbone_macs"] = _macs(backbone, input_size)
        d = getattr(backbone, "d_bottleneck", None)
        if isinstance(head, nn.Module) and d is not None:
            prof["head_macs"] = _macs(head, (int(d),))
        elif not np.isnan(prof["backbone_macs"]) and not np.isnan(prof["macs"]):
            prof["head_macs"] = max(prof["macs"] - prof["backbone_macs"], 0.0)  # e.g. CNN_MLP (fc1/fc2)
    return prof


# --------------------------------------------------------------------------- metrics
def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Accuracy, Cohen's kappa and (if probabilities are given) ROC-AUC; AUC is NaN on failure."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    if y_prob is not None:
        try:
            y_prob = np.asarray(y_prob)
            if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            elif y_prob.ndim == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = float("nan")
    return metrics
