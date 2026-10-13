"""
Neuroscientific Interpretability Suite (Phase 4.3).

Renders, for a *trained* CNN_KAN fold model saved by MOABB (``save_model=True``):
  * the learned KAN edge functions phi(x) = base + spline over each edge's knot range;
  * scalp topomaps of the backbone's depthwise spatial filters: the raw weights and, when trial
    data is available, the corresponding spatial patterns (Haufe et al. 2014), which are what can
    be read neurophysiologically. Raw filter weights are not activation maps.

Channel names come from the run (``run_info.json``, written by the runners from the data) or from
the dataset itself; nothing is hardcoded.

CLI::

    python -m src.utils.visualize --model-path results/bnci_full/seed0/Models_CrossSubject/BNCI2014-001/1/cnn_kan/fitted_model_0.pkl
    python -m src.utils.visualize --run-dir results/bnci_full            # first cnn_kan fold found
"""

import argparse
import contextlib
import json
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# pyrefly: ignore [missing-import]
import mne
# pyrefly: ignore [missing-import]
from mne.channels import make_standard_montage
# pyrefly: ignore [missing-import]
from mne.viz import plot_topomap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MONTAGE = "colin27_1020"  # replaces the deprecated "standard_1020" in MNE >= 1.10


def _get_standard_montage():
    try:
        return make_standard_montage(MONTAGE)
    except ValueError:
        return make_standard_montage("standard_1020")


# --------------------------------------------------------------------------- loading
@contextlib.contextmanager
def _map_unavailable_devices_to_cpu():
    """skorch remaps CUDA->CPU when unpickling on a machine without CUDA, but not MPS; do both."""
    import skorch.net
    import skorch.utils

    original = skorch.net.get_map_location

    def get_map_location(target_device, fallback_device="cpu"):
        device = torch.device(target_device or fallback_device)
        if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return torch.device("cpu")
        return original(target_device, fallback_device)

    skorch.net.get_map_location = get_map_location
    try:
        yield
    finally:
        skorch.net.get_map_location = original


def load_fold_model(model_path) -> Tuple[object, torch.nn.Module]:
    """
    Unpickle a fold model saved by MOABB and return ``(estimator, module)`` with the PyTorch
    module on CPU in eval mode. Accepts the bare estimator or a Pipeline ending in one.
    """
    with open(model_path, "rb") as f, _map_unavailable_devices_to_cpu():
        obj = pickle.load(f)
    est = obj.steps[-1][1] if hasattr(obj, "steps") else obj
    net = getattr(est, "net_", None)
    module = getattr(net, "module_", None)
    if module is None:
        raise ValueError(f"{model_path} does not hold a fitted PyTorch model (got {type(est).__name__})")
    module = module.cpu().eval()
    return est, module


def find_fold_models(run_dir, model_name: str = "cnn_kan") -> List[Path]:
    """Fold models of one pipeline under a run directory, ordered by seed, subject and fold."""
    paths = Path(run_dir).glob(f"seed*/Models_CrossSubject/*/*/{model_name}/fitted_model_*.pkl")

    def key(p: Path):
        seed = int(p.parts[-6].replace("seed", "")) if p.parts[-6].startswith("seed") else 0
        subject = int(p.parts[-3]) if p.parts[-3].isdigit() else 0
        return seed, subject, p.name

    return sorted(paths, key=key)


def find_run_info(path) -> Optional[dict]:
    """``run_info.json`` of the run a fold model belongs to (searched upwards from ``path``)."""
    for parent in [Path(path)] + list(Path(path).parents):
        candidate = parent / "run_info.json" if parent.is_dir() else parent.parent / "run_info.json"
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None


def held_out_subject(model_path) -> Optional[int]:
    """MOABB stores cross-subject fold models under ``<dataset>/<held-out subject>/<pipeline>/``."""
    part = Path(model_path).parts[-3]
    return int(part) if part.isdigit() else None


def load_trials(dataset_name: str, subjects: Sequence[int], cfg: Optional[dict] = None):
    """Epoched trials and channel names with the runners' paradigm (needs the MOABB data)."""
    from src.utils.experiment import make_dataset, make_paradigm

    dcfg = (cfg or {}).get("dataset", {})
    dataset = make_dataset(dataset_name, subjects=list(subjects))
    paradigm = make_paradigm(
        dataset_name,
        fmin=dcfg.get("bandpass", {}).get("low", 8.0),
        fmax=dcfg.get("bandpass", {}).get("high", 30.0),
        tmin=dcfg.get("tmin", 0.0),
        tmax=dcfg.get("tmax"),
        resample=dcfg.get("resample"),
        events=dcfg.get("events"),
    )
    epochs, _, _ = paradigm.get_data(dataset=dataset, subjects=list(subjects), return_epochs=True)
    return epochs.get_data(copy=False).astype(np.float32), list(epochs.ch_names), float(epochs.info["sfreq"])


# --------------------------------------------------------------------------- KAN edge functions
def plot_bspline_activations(
    model,
    num_points: int = 200,
    output_dir: Path = FIGURES_DIR,
    trained: bool = True,
    grid_range: Optional[tuple] = None,
) -> Path:
    """
    Plot phi(x) = base(x) + spline(x) for a 4x4 block of first-layer KAN edges, each over its own
    knot range (the grids are refit to the data during training). ``grid_range`` forces one range.
    """
    from src.models.kan_head import b_splines

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    kan_layer = model.kan_head.kan1
    k = kan_layer.spline_order
    fig, axes = plt.subplots(4, 4, figsize=(12, 10), dpi=300)
    axes = axes.flatten()

    with torch.no_grad():
        grid = kan_layer.grid.cpu()
        spline_weight = kan_layer.spline_weight.cpu()
        scale_spline = kan_layer.scale_spline_param.cpu()
        base_weight = kan_layer.base_weight.cpu()  # includes the per-edge base scale

        idx = 0
        for out_idx in range(min(4, kan_layer.out_features)):
            for in_idx in range(min(4, kan_layer.in_features)):
                ax = axes[idx]
                lo, hi = grid_range if grid_range else (float(grid[in_idx, k]), float(grid[in_idx, -k - 1]))
                x = torch.linspace(lo, hi, num_points)
                x_inp = x.unsqueeze(1).repeat(1, kan_layer.in_features)
                bases = b_splines(x_inp, grid, k=k)

                y_base = base_weight[out_idx, in_idx].item() * torch.nn.functional.silu(x).numpy()
                w_s = spline_weight[out_idx, in_idx] * scale_spline[out_idx, in_idx]
                y_spline = torch.einsum("pc,c->p", bases[:, in_idx, :], w_s).numpy()

                ax.plot(x.numpy(), y_base + y_spline, color="#1f77b4", linewidth=2.0, label=r"$\phi(x)$ Total")
                ax.plot(x.numpy(), y_base, color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.7, label="Base SiLU")
                ax.plot(x.numpy(), y_spline, color="#2ca02c", linestyle=":", linewidth=1.2, alpha=0.8, label="B-Spline")
                ax.set_title(f"Edge $z_{{{in_idx+1}}} \\to h_{{{out_idx+1}}}$", fontsize=9, fontweight="bold")
                ax.set_xlim(lo, hi)
                ax.tick_params(axis="both", which="major", labelsize=7)
                for knot in grid[in_idx].numpy():
                    if lo <= knot <= hi:
                        ax.axvline(knot, color="gray", linestyle=":", alpha=0.3)
                if idx == 0:
                    ax.legend(fontsize=6.5, loc="upper left")
                idx += 1

    prefix = "Learned" if trained else "Untrained (random initialisation)"
    title = f"{prefix} B-Spline Edge Functions $\\phi(x)$ of the First KAN Layer"
    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / "kan_spline_activations.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "kan_spline_activations.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] B-spline activation curves figure saved to: {png_path}")
    return png_path


# --------------------------------------------------------------------------- spatial filters
def spatial_filter_weights(model) -> np.ndarray:
    """Depthwise spatial filters of the backbone, shape (F1*D, channels)."""
    return model.backbone.conv_spatial.weight.detach().cpu().numpy()[:, 0, :, 0]


def compute_spatial_patterns(model, X: np.ndarray, max_trials: int = 256) -> np.ndarray:
    """
    Haufe et al. (2014) activation patterns of the depthwise spatial filters.

    Filter j acts on the output of temporal filter g = j // D (after its BatchNorm), so its pattern
    is a_j = Sigma_g w_j / (w_j^T Sigma_g w_j), with Sigma_g the channel covariance of that
    temporally filtered signal. Returns shape (F1*D, channels).
    """
    backbone = model.backbone
    W = spatial_filter_weights(model)
    D = W.shape[0] // backbone.conv_temporal.out_channels
    X = np.asarray(X, dtype=np.float32)[:max_trials]
    x = torch.from_numpy(X if X.ndim == 4 else X[:, None])
    with torch.no_grad():
        h = backbone.bn_temporal(backbone.conv_temporal(x.cpu())).numpy()  # (N, F1, C, T)

    patterns = np.empty_like(W)
    for g in range(h.shape[1]):
        sig = h[:, g].transpose(1, 0, 2).reshape(h.shape[2], -1)  # (C, N*T)
        cov = np.cov(sig)
        for j in range(g * D, (g + 1) * D):
            w = W[j]
            var = float(w @ cov @ w)
            patterns[j] = cov @ w / var if var > 0 else np.nan
    return patterns


def _eeg_info(channel_names: Sequence[str], sfreq: float):
    info = mne.create_info(ch_names=list(channel_names), sfreq=sfreq, ch_types="eeg")
    info.set_montage(_get_standard_montage(), match_case=False, on_missing="warn")
    return info


def plot_spatial_topomaps(
    model,
    channel_names: Sequence[str],
    output_dir: Path = FIGURES_DIR,
    X: Optional[np.ndarray] = None,
    sfreq: float = 250.0,
    trained: bool = True,
) -> Path:
    """
    Scalp maps of every depthwise spatial filter: raw weights, plus Haufe spatial patterns when
    trial data ``X`` (N, C, T) is given. ``channel_names`` must come from the data.
    """
    W = spatial_filter_weights(model)
    if W.shape[1] != len(channel_names):
        raise ValueError(f"model has {W.shape[1]} input channels but {len(channel_names)} channel names were given")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info = _eeg_info(channel_names, sfreq)

    rows = [("Filter weights", W)]
    if X is not None:
        rows.append(("Spatial patterns (Haufe)", compute_spatial_patterns(model, X)))

    n = W.shape[0]
    cols = min(8, n)
    per_row = int(np.ceil(n / cols))
    fig, axes = plt.subplots(per_row * len(rows), cols, figsize=(2 * cols, 2.1 * per_row * len(rows)), dpi=300, squeeze=False)
    for r, (label, maps) in enumerate(rows):
        lim = np.nanmax(np.abs(maps))
        for i in range(per_row * cols):
            ax = axes[r * per_row + i // cols, i % cols]
            if i >= n:
                ax.axis("off")
                continue
            plot_topomap(maps[i], info, axes=ax, show=False, cmap="RdBu_r", vlim=(-lim, lim), sensors=True, contours=4)
            ax.set_title(f"{label.split(' (')[0]} {i + 1}", fontsize=7, fontweight="bold")

    prefix = "Learned" if trained else "Untrained (random initialisation)"
    kinds = " and Spatial Patterns" if X is not None else ""
    title = f"{prefix} Depthwise Spatial Filters{kinds} ({n} filters, {len(channel_names)} channels)"
    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    png_path = output_dir / "spatial_scalp_topomaps.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "spatial_scalp_topomaps.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Spatial scalp topomaps figure saved to: {png_path}")
    return png_path


# --------------------------------------------------------------------------- entry points
def run_interpretability_suite(
    model_path: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    dataset: Optional[str] = None,
    subjects: Optional[Sequence[int]] = None,
    patterns: bool = True,
) -> dict:
    """
    Render both figures for one trained fold model (``model_path``, or the first CNN_KAN fold under
    ``run_dir``). Channel names come from the run's ``run_info.json`` or the dataset; spatial
    patterns use the held-out subject's trials (or ``subjects``) and need the MOABB data.
    """
    if model_path is None:
        if run_dir is None:
            raise ValueError("give a trained fold model (--model-path) or a run directory (--run-dir)")
        folds = find_fold_models(run_dir)
        if not folds:
            raise FileNotFoundError(f"no cnn_kan fold models under {run_dir} (was the run made with save_model: true?)")
        model_path = folds[0]
    model_path = Path(model_path)
    _, module = load_fold_model(model_path)
    if not hasattr(module, "kan_head"):
        raise ValueError(f"{model_path} is a {type(module).__name__}, not a CNN_KAN model")

    info = find_run_info(model_path) or {}
    dataset = dataset or info.get("dataset") or "bnci2014_001"
    dataset_key = "physionet" if "physionet" in dataset.lower() else dataset.lower().replace("-", "_")
    shape = info.get("shape") or {}
    ch_names, sfreq, X = shape.get("ch_names"), float(shape.get("sfreq", 250.0)), None

    if patterns or not ch_names:
        subs = list(subjects) if subjects else [held_out_subject(model_path) or (info.get("subjects") or [1])[0]]
        X, data_ch_names, sfreq = load_trials(dataset_key, subs, info.get("config"))
        ch_names = ch_names or data_ch_names
        if list(ch_names) != list(data_ch_names):
            raise ValueError("run_info channel names do not match the data")

    output_dir = Path(output_dir) if output_dir is not None else Path(model_path).parent / "figures"
    print(f"[INTERPRETABILITY SUITE] model={model_path}")
    spline_png = plot_bspline_activations(module, output_dir=output_dir, trained=True)
    topo_png = plot_spatial_topomaps(module, ch_names, output_dir=output_dir, X=X if patterns else None,
                                     sfreq=sfreq, trained=True)
    return {"model_path": str(model_path), "splines": spline_png, "topomaps": topo_png, "channels": list(ch_names)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="KAN edge functions and spatial-filter topomaps of a trained fold model")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-path", type=Path, help="fitted_model_<fold>.pkl saved by MOABB")
    src.add_argument("--run-dir", type=Path, help="run directory; uses the first cnn_kan fold model in it")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <model dir>/figures")
    parser.add_argument("--dataset", type=str, default=None, help="default: from run_info.json")
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="trials for the spatial patterns (default: the fold's held-out subject)")
    parser.add_argument("--no-patterns", action="store_true", help="raw filter weights only (no data needed)")
    args = parser.parse_args(argv)
    return run_interpretability_suite(model_path=args.model_path, run_dir=args.run_dir, output_dir=args.output_dir,
                                      dataset=args.dataset, subjects=args.subjects, patterns=not args.no_patterns)


if __name__ == "__main__":
    main()
