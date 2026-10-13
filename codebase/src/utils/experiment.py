"""
Shared evaluation protocol for all benchmark runners (issue #3, PR 3).

One place for: dataset/paradigm construction with explicit events, input-shape inference,
pipeline construction by name, and a seed-looped MOABB cross-subject evaluation that
writes ``scores.csv`` + ``run_info.json`` into a run directory.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.models.ablations import CNN_Linear, CNN_MLP, match_mlp_hidden_dim
from src.models.cnn_kan import CNN_KAN
from src.models.skorch_wrapper import SkorchEEGClassifier
from src.baselines.classifiers import CSP_LDA, EEGNet, SVM_Bandpower
from src.utils.seed import set_seed

try:  # MOABB is a hard dependency of the runners, optional for unit tests of pure helpers
    from moabb.datasets import BNCI2014_001, PhysionetMI
    from moabb.evaluations import CrossSubjectEvaluation
    from moabb.paradigms import MotorImagery
except ImportError:  # pragma: no cover
    BNCI2014_001 = PhysionetMI = CrossSubjectEvaluation = MotorImagery = None


def _moabb_marks_pipelines_fitted() -> bool:
    """MOABB >= 1.7 calls ``_ensure_fitted`` on every fold model, so sklearn's fitted-checks pass."""
    try:
        from moabb.evaluations.utils import _ensure_fitted  # noqa: F401
    except ImportError:
        return False
    return True


def _install_legacy_pipeline_fitted_shim() -> bool:
    """
    Compatibility shim for MOABB releases older than the pinned 1.7.2 combined with sklearn >= 1.8,
    where ``Pipeline.predict`` fails sklearn's fitted-check. Installed only when MOABB lacks its own
    fix; with the pinned versions nothing is patched and sklearn's checks stay intact (audit item 24).
    """
    if CrossSubjectEvaluation is None or _moabb_marks_pipelines_fitted():
        return False
    from sklearn.pipeline import Pipeline

    warnings.warn(
        "Installed MOABB predates moabb.evaluations.utils._ensure_fitted; patching "
        "sklearn Pipeline.__sklearn_is_fitted__ for compatibility. Install the pinned "
        "requirements (moabb 1.7.2) to avoid this.",
        RuntimeWarning,
    )
    Pipeline.__sklearn_is_fitted__ = lambda self: True
    return True


LEGACY_PIPELINE_SHIM = _install_legacy_pipeline_fitted_shim()


# --------------------------------------------------------------------------- datasets
DATASET_ALIASES = {
    "bnci2014_001": "bnci2014_001",
    "bnci2014-001": "bnci2014_001",
    "bnci": "bnci2014_001",
    "physionet": "physionet",
    "physionetmi": "physionet",
}

# Explicit class lists. Leaving ``events=None`` lets MOABB pick every event in the dataset,
# which on PhysionetMI silently adds ``rest`` as a 5th class (audit item 7).
EVENTS: Dict[str, List[str]] = {
    "bnci2014_001": ["left_hand", "right_hand", "feet", "tongue"],
    "physionet": ["left_hand", "right_hand", "feet", "hands"],
}

NEURAL_MODELS = ("eegnet", "cnn_kan", "cnn_linear", "cnn_mlp")
CLASSICAL_MODELS = ("csp_lda", "svm")


def canonical_dataset_name(name: str) -> str:
    key = name.lower().replace(" ", "")
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset '{name}'. Known: {sorted(set(DATASET_ALIASES.values()))}")
    return DATASET_ALIASES[key]


def make_dataset(name: str, subjects: Optional[Iterable[int]] = None):
    """Instantiate the MOABB dataset, optionally restricted to a subject subset."""
    if BNCI2014_001 is None:
        raise ImportError("MOABB is required: pip install moabb")
    key = canonical_dataset_name(name)
    dataset = BNCI2014_001() if key == "bnci2014_001" else PhysionetMI()
    if subjects is not None:
        subjects = [int(s) for s in subjects]
        unknown = sorted(set(subjects) - set(dataset.subject_list))
        if unknown:
            raise ValueError(f"Subjects {unknown} not in {dataset.code}")
        dataset.subject_list = subjects
    return dataset


def make_paradigm(
    name: str,
    fmin: float = 8.0,
    fmax: float = 30.0,
    tmin: float = 0.0,
    tmax: Optional[float] = None,
    resample: Optional[float] = None,
    events: Optional[List[str]] = None,
):
    """MotorImagery paradigm with an explicit event list (never ``rest``)."""
    if MotorImagery is None:
        raise ImportError("MOABB is required: pip install moabb")
    key = canonical_dataset_name(name)
    events = list(events) if events is not None else list(EVENTS[key])
    return MotorImagery(
        events=events,
        n_classes=len(events),
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        resample=resample,
    )


def infer_shape(paradigm, dataset, subject: Optional[int] = None) -> Dict[str, object]:
    """
    Read (n_chans, n_times, n_classes, ch_names, sfreq) from one subject's epochs so that
    nothing about the input shape is hardcoded (audit items 1, 20).
    """
    subject = dataset.subject_list[0] if subject is None else subject
    epochs, labels, _ = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)
    # Class count comes from the paradigm's explicit event list, not from whichever labels the
    # probe subject happens to have (a partial recording must not shrink every model's output).
    n_classes = len(paradigm.events) if getattr(paradigm, "events", None) else int(len(np.unique(labels)))
    return {
        "n_chans": int(len(epochs.ch_names)),
        "n_times": int(len(epochs.times)),
        "n_classes": int(n_classes),
        "ch_names": list(epochs.ch_names),
        "sfreq": float(epochs.info["sfreq"]),
        "classes": sorted(str(c) for c in np.unique(labels)),
    }


# --------------------------------------------------------------------------- pipelines
def parse_model_name(name: str) -> Tuple[str, Optional[int]]:
    """``cnn_kan_G5`` -> (``cnn_kan``, 5); ``eegnet`` -> (``eegnet``, None)."""
    base = name.lower()
    grid = None
    if "_g" in base:
        head, _, tail = base.rpartition("_g")
        if tail.isdigit():
            base, grid = head, int(tail)
    return base, grid


def build_pipeline(
    name: str,
    n_chans: int,
    n_times: int,
    n_classes: int,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 40,
    patience: int = 10,
    val_split: float = 0.2,
    device: Optional[str] = None,
    random_state: int = 42,
    weight_decay: float = 1e-4,
    grid_size: int = 3,
    spline_order: int = 3,
    d_bottleneck: int = 16,
    csp_components: int = 6,
    verbose: int = 0,
    kan_grid_updates: int = 5,
    mlp_hidden_dim: Optional[int] = None,
):
    """
    Build one estimator by name. Grid size may be encoded in the name (``cnn_kan_G5``).
    ``cnn_mlp`` gets the hidden width that matches the CNN_KAN head's parameter count at
    ``grid_size`` unless ``mlp_hidden_dim`` is given.
    """
    base, grid_from_name = parse_model_name(name)
    if grid_from_name is not None:
        grid_size = grid_from_name

    if base == "csp_lda":
        return CSP_LDA(n_components=csp_components)
    if base == "svm":
        return SVM_Bandpower(kernel="rbf", C=1.0)

    if base == "eegnet":
        module, module_kwargs = EEGNet, {"channels": n_chans, "samples": n_times, "num_classes": n_classes}
    elif base in ("cnn_kan", "kan"):
        module = CNN_KAN
        module_kwargs = {"channels": n_chans, "num_classes": n_classes, "d_bottleneck": d_bottleneck,
                         "grid_size": grid_size, "spline_order": spline_order}
    elif base == "cnn_linear":
        module, module_kwargs = CNN_Linear, {"channels": n_chans, "num_classes": n_classes, "d_bottleneck": d_bottleneck}
    elif base == "cnn_mlp":
        if mlp_hidden_dim is None:
            mlp_hidden_dim = matched_mlp_hidden_dim(n_chans, n_classes, d_bottleneck, grid_size, spline_order)
        module = CNN_MLP
        module_kwargs = {"channels": n_chans, "num_classes": n_classes, "d_bottleneck": d_bottleneck,
                         "mlp_hidden_dim": int(mlp_hidden_dim)}
    else:
        raise ValueError(f"Unsupported model name: {name}")

    return SkorchEEGClassifier(
        module=module,
        module_kwargs=module_kwargs,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        val_split=val_split,
        device=device,
        random_state=random_state,
        weight_decay=weight_decay,
        verbose=verbose,
        kan_grid_updates=kan_grid_updates,
    )


def matched_mlp_hidden_dim(n_chans: int, n_classes: int, d_bottleneck: int = 16,
                           grid_size: int = 3, spline_order: int = 3) -> int:
    """MLP hidden width whose head matches the CNN_KAN head (same bottleneck, grid and classes)."""
    kan = CNN_KAN(channels=n_chans, num_classes=n_classes, d_bottleneck=d_bottleneck,
                  grid_size=grid_size, spline_order=spline_order)
    return match_mlp_hidden_dim(kan.get_parameter_count()["kan_head_params"], d_bottleneck, n_classes)


def hyperparameters_from_config(cfg: dict, max_epochs: Optional[int] = None, device: Optional[str] = None) -> dict:
    """Every training knob the YAML exposes, in build_pipeline's vocabulary (nothing silently dropped)."""
    ecfg, kcfg = cfg.get("experiment", {}), cfg.get("kan", {})
    return dict(
        lr=ecfg.get("learning_rate", 1e-3),
        batch_size=ecfg.get("batch_size", 64),
        max_epochs=max_epochs or ecfg.get("max_epochs", 40),
        patience=ecfg.get("patience", 10),
        val_split=ecfg.get("val_split", 0.2),
        weight_decay=ecfg.get("weight_decay", 1e-4),
        device=device or ecfg.get("device"),
        grid_size=kcfg.get("grid_size", 3),
        spline_order=kcfg.get("spline_order", 3),
        d_bottleneck=kcfg.get("d_bottleneck", 16),
        kan_grid_updates=kcfg.get("grid_updates", 5),
    )


def resolve_overwrite(cfg: dict, cli_no_overwrite: Optional[bool] = None) -> bool:
    """CLI flag wins; otherwise the config's ``experiment.overwrite`` (default True)."""
    if cli_no_overwrite:
        return False
    return bool(cfg.get("experiment", {}).get("overwrite", True))


def build_pipelines(names: Iterable[str], shape: Dict[str, object], **hp) -> Dict[str, object]:
    """Name -> estimator for every (deduplicated) model name."""
    pipelines = {}
    for name in names:
        if name in pipelines:
            continue
        pipelines[name] = build_pipeline(name, shape["n_chans"], shape["n_times"], shape["n_classes"], **hp)
    return pipelines


# --------------------------------------------------------------------------- config
def load_config(path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("dataset", {})
    cfg.setdefault("experiment", {})
    cfg.setdefault("kan", {})
    return cfg


def _git_sha(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _package_versions() -> Dict[str, str]:
    versions = {"python": platform.python_version()}
    for mod in ("torch", "moabb", "skorch", "sklearn", "mne", "numpy", "pandas"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return versions


def write_run_info(run_dir: Path, **fields) -> Path:
    """``run_info.json``: everything needed to reproduce or audit the run."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(Path(__file__).resolve().parents[2]),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": _package_versions(),
        "argv": sys.argv,
    }
    info.update(fields)
    path = run_dir / "run_info.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    return path


# --------------------------------------------------------------------------- evaluation
def run_cross_subject(
    dataset,
    paradigm,
    pipelines: Dict[str, object],
    seeds: Iterable[int],
    run_dir,
    overwrite: bool = True,
    save_model: bool = True,
    n_jobs: int = 1,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Leave-one-subject-out evaluation of every pipeline for every seed.

    * one MOABB results store per seed under ``run_dir/seed{N}`` (never the global
      ``~/mne_data`` cache; ``overwrite=True`` by default so a code change can never be served
      stale scores — audit item 6);
    * both sessions of the held-out subject are kept (audit item 9);
    * a ``seed`` column and the dataset ``code`` string are recorded (audit items 10, 20);
    * fold models are saved by MOABB when ``save_model`` is on.
    """
    if CrossSubjectEvaluation is None:
        raise ImportError("MOABB is required: pip install moabb")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for seed in [int(s) for s in seeds]:
        set_seed(seed)
        seeded = {}
        for name, est in pipelines.items():
            if "random_state" in est.get_params():
                est = est.set_params(random_state=seed)
            seeded[name] = est

        evaluation = CrossSubjectEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            overwrite=overwrite,
            suffix=f"{suffix}seed{seed}",
            hdf5_path=str(run_dir / f"seed{seed}"),
            save_model=save_model,
            random_state=seed,
            n_jobs=n_jobs,
        )
        df = evaluation.process(seeded)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["seed"] = seed
        df["dataset"] = dataset.code
        df["model"] = df["pipeline"]
        frames.append(df)
        # Flush after every seed so an interrupted run keeps the seeds that finished.
        pd.concat(frames, ignore_index=True).to_csv(run_dir / "scores.csv", index=False)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_scores(scores: pd.DataFrame, test_only: bool = False) -> pd.DataFrame:
    """
    One row per model: mean/std over subjects of the per-subject mean (sessions and seeds
    averaged first), plus the spread across seeds.
    """
    if scores.empty:
        return pd.DataFrame()
    df = scores if "seed" in scores.columns else scores.assign(seed=0)
    if test_only and "session" in df.columns and df["session"].astype(str).str.contains("test").any():
        df = df[df["session"].astype(str).str.contains("test")]
    per_subject = df.groupby(["model", "subject"])["score"].mean().reset_index()
    per_seed = df.groupby(["model", "seed"])["score"].mean().reset_index()
    out = per_subject.groupby("model")["score"].agg(mean_acc="mean", std_subjects="std", n_subjects="count")
    out["std_seeds"] = per_seed.groupby("model")["score"].std()
    out["n_seeds"] = per_seed.groupby("model")["score"].count()
    return out.reset_index()
