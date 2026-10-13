"""
Skorch Adapter Bridging PyTorch nn.Module to Scikit-learn / MOABB BaseEstimator
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

try:
    from skorch import NeuralNetClassifier
    from skorch.callbacks import EarlyStopping
    from skorch.dataset import ValidSplit
except ImportError:
    NeuralNetClassifier = object
    EarlyStopping = None
    ValidSplit = None

try:
    from src.models.kan_head import KANGridUpdate
except ImportError:  # pragma: no cover
    KANGridUpdate = None


def get_default_device() -> str:
    """
    Return available PyTorch acceleration device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


TIME_POOLED_MODULES = ("CNN_KAN", "CNN_Linear", "CNN_MLP")
KAN_MODULES = ("CNN_KAN",)


class SkorchEEGClassifier(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn compliant estimator wrapper for PyTorch EEG models using skorch.
    Enables direct compatibility with MOABB CrossSubjectEvaluation.

    Evaluation protocol defaults (issue #3, PR 3):
      * ``train_split="auto"``: a stratified ``val_split`` fraction of the *training* trials is
        held out for early stopping (the LOSO test subject is never touched). On datasets too
        small to stratify it falls back to no validation split.
      * ``EarlyStopping`` monitors ``valid_loss`` (or ``train_loss`` without a split) and
        restores the best epoch (``load_best=True``).
      * The last training batch is dropped when it would otherwise be a single trial, which
        crashes BatchNorm in training mode.
      * For KAN models, ``KANGridUpdate`` refits the spline knot grids to the bottleneck
        distribution after each of the first ``kan_grid_updates`` epochs (0 disables it).
      * ``device`` is resolved at fit time, not at construction, so ``get_params()`` (and
        therefore MOABB's cache key) does not embed the local hardware.
    """

    def __init__(
        self,
        module: type,
        module_kwargs: dict = None,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 100,
        device: str = None,
        weight_decay: float = 1e-4,
        patience: int = 15,
        verbose: int = 0,
        train_split="auto",
        val_split: float = 0.2,
        random_state: int = 42,
        kan_grid_updates: int = 5,
    ):
        # sklearn contract: store constructor arguments exactly as given (clone() relies on it).
        self.module = module
        self.module_kwargs = module_kwargs
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = device
        self.weight_decay = weight_decay
        self.patience = patience
        self.verbose = verbose
        self.train_split = train_split
        self.val_split = val_split
        self.random_state = random_state
        self.kan_grid_updates = kan_grid_updates

    # ------------------------------------------------------------------ helpers
    def _resolved_device(self) -> str:
        if self.device and self.device != "auto":
            return self.device
        return get_default_device()

    def _module_kwargs(self) -> dict:
        kwargs = dict(self.module_kwargs or {})
        if getattr(self.module, "__name__", "") in TIME_POOLED_MODULES:
            kwargs.pop("samples", None)
        return kwargs

    def _resolve_train_split(self, n_samples: int = None, n_classes: int = None, min_class_count: int = None):
        """
        Decide the validation split for this fit.

        Returns a skorch-compatible ``train_split`` (``None`` or a ``ValidSplit``/callable).
        """
        if self.train_split is None or self.train_split == "none":
            return None
        if self.train_split != "auto":
            return self.train_split  # user-supplied ValidSplit / callable
        if ValidSplit is None or not self.val_split or self.val_split <= 0:
            return None
        if n_samples is not None and n_classes is not None:
            n_valid = int(np.ceil(n_samples * self.val_split))  # sklearn's ShuffleSplit rounds up
            # StratifiedShuffleSplit needs at least one trial per class on both sides, i.e.
            # every class present at least twice and a validation set no smaller than n_classes.
            too_small = n_valid < n_classes or (n_samples - n_valid) < n_classes
            if min_class_count is not None and min_class_count < 2:
                too_small = True
            if too_small:
                warnings.warn(
                    f"SkorchEEGClassifier: {n_samples} trials / {n_classes} classes is too small "
                    f"for a stratified {self.val_split:.0%} validation split; training without one.",
                    RuntimeWarning,
                )
                return None
        return ValidSplit(self.val_split, stratified=True, random_state=self.random_state)

    def _build_net(self, n_samples: int = None, n_classes: int = None, min_class_count: int = None):
        train_split = self._resolve_train_split(n_samples, n_classes, min_class_count)

        callbacks = []
        if EarlyStopping is not None and self.patience > 0:
            monitor_metric = "valid_loss" if train_split is not None else "train_loss"
            callbacks.append(
                EarlyStopping(patience=self.patience, monitor=monitor_metric, load_best=True)
            )

        # After early stopping, so EarlyStopping checkpoints the state that produced valid_loss.
        if KANGridUpdate is not None and self.kan_grid_updates and getattr(self.module, "__name__", "") in KAN_MODULES:
            callbacks.append(KANGridUpdate(n_updates=self.kan_grid_updates))

        # Drop the last batch only when it would be a lone trial (BatchNorm crash); a dataset
        # smaller than one batch must keep its single partial batch.
        if n_samples is not None:
            frac = self.val_split
            if ValidSplit is not None and isinstance(train_split, ValidSplit) and isinstance(train_split.cv, (int, float)) and not isinstance(train_split.cv, bool):
                frac = train_split.cv if isinstance(train_split.cv, float) else 1.0 / train_split.cv  # k-fold: 1/k held out
            elif train_split is not None and not isinstance(train_split, ValidSplit):
                frac = None            # arbitrary callable: size unknown
            if train_split is None:
                n_train = n_samples
            elif frac is None:
                n_train = None
            else:
                n_train = n_samples - int(np.ceil(n_samples * frac))
            drop_last = n_train is not None and n_train > self.batch_size and (n_train % self.batch_size) == 1
        else:
            drop_last = False

        skorch_kwargs = {f"module__{k}": v for k, v in self._module_kwargs().items()}
        skorch_kwargs.update({
            "module": self.module,
            "criterion": nn.CrossEntropyLoss,
            "optimizer": torch.optim.AdamW,
            "optimizer__lr": self.lr,
            "optimizer__weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "device": self._resolved_device(),
            "verbose": self.verbose,
            "callbacks": callbacks,
            "train_split": train_split,
            "iterator_train__shuffle": True,
            "iterator_train__drop_last": drop_last,
        })

        self.net_ = NeuralNetClassifier(**skorch_kwargs)
        return self.net_

    # ------------------------------------------------------------------ sklearn API
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit PyTorch module on input trials X and labels y.
        """
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        y_arr = np.ascontiguousarray(y, dtype=np.int64)

        if not np.isfinite(X_arr).all():
            raise ValueError("Input X contains NaN or inf.")

        x_std = float(X_arr.std())
        if x_std < 1e-2:
            warnings.warn(
                f"Input std is {x_std:.2e}. EEG may be incorrectly scaled.",
                RuntimeWarning,
            )

        self.classes_, counts = np.unique(y_arr, return_counts=True)
        torch.manual_seed(self.random_state)
        self._build_net(n_samples=len(y_arr), n_classes=len(self.classes_), min_class_count=int(counts.min()))

        self.net_.fit(X_arr, y_arr)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X.
        """
        check_is_fitted(self, "net_")
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        return self.net_.predict(X_arr)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probability distribution for X.
        """
        check_is_fitted(self, "net_")
        X_arr = np.ascontiguousarray(X, dtype=np.float32)
        return self.net_.predict_proba(X_arr)

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "net_", None) is not None and hasattr(self.net_, "module_")

    # ------------------------------------------------------------------ persistence
    def save_params(self, fpath: str):
        """
        Save underlying PyTorch state_dict parameters safely to file path.
        """
        net = getattr(self, "net_", None)
        if net is None:
            raise RuntimeError(
                "Cannot save parameters before the model has been fitted."
            )

        if hasattr(net, "save_params"):
            net.save_params(f_params=fpath)
        elif hasattr(net, "module_"):
            torch.save(net.module_.state_dict(), fpath)

    def load_params(self, fpath: str):
        """
        Load underlying PyTorch state_dict parameters safely from file path.
        """
        net = getattr(self, "net_", None)
        if net is None:
            self._build_net()
            self.net_.initialize()

        if hasattr(self.net_, "load_params"):
            self.net_.load_params(f_params=fpath)
        else:
            state_dict = torch.load(
                fpath,
                map_location=self.net_.device,
            )
            self.net_.module_.load_state_dict(state_dict)

        return self
