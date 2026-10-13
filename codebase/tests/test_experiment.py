"""
Evaluation-protocol tests (issue #3, PR 3): explicit events, shape inference, pipeline
construction, sklearn contract of the skorch wrapper, validation split, and an end-to-end
2-subject smoke run that checks the scores schema and that nothing touches the global cache.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_kan import CNN_KAN
from src.models.skorch_wrapper import SkorchEEGClassifier
from src.utils import experiment as ex


# --------------------------------------------------------------------------- pure helpers
def test_canonical_dataset_names():
    assert ex.canonical_dataset_name("BNCI2014_001") == "bnci2014_001"
    assert ex.canonical_dataset_name("PhysionetMI") == "physionet"
    with pytest.raises(ValueError):
        ex.canonical_dataset_name("nope")


def test_parse_model_name():
    assert ex.parse_model_name("cnn_kan_G5") == ("cnn_kan", 5)
    assert ex.parse_model_name("cnn_kan") == ("cnn_kan", None)
    assert ex.parse_model_name("EEGNet") == ("eegnet", None)


def test_events_never_include_rest():
    for name, events in ex.EVENTS.items():
        assert "rest" not in events, name
        assert len(events) == 4, name


def test_make_paradigm_uses_explicit_events():
    par = ex.make_paradigm("physionet")
    assert list(par.events) == ex.EVENTS["physionet"]
    assert par.n_classes == 4
    par = ex.make_paradigm("bnci2014_001", tmax=3.0)
    assert par.tmax == 3.0 and par.tmin == 0.0


def test_build_pipeline_every_name():
    shape = {"n_chans": 64, "n_times": 481, "n_classes": 4}
    names = ["csp_lda", "svm", "eegnet", "cnn_kan", "cnn_kan_G8", "cnn_linear", "cnn_mlp"]
    pipes = ex.build_pipelines(names, shape, max_epochs=1)
    assert set(pipes) == set(names)
    kan = pipes["cnn_kan_G8"]
    assert kan.module_kwargs["grid_size"] == 8 and kan.module_kwargs["channels"] == 64
    assert "samples" not in kan.module_kwargs  # time-pooled: keeps the MOABB cache key stable
    assert pipes["eegnet"].module_kwargs["samples"] == 481
    assert pipes["cnn_mlp"].module_kwargs["mlp_hidden_dim"] == 56  # parameter-matched to the G=3 KAN head
    with pytest.raises(ValueError):
        ex.build_pipeline("transformer", 22, 1001, 4)


def test_load_config_has_protocol_fields():
    cfg = ex.load_config(PROJECT_ROOT / "configs" / "bnci2014_001.yaml")
    assert cfg["dataset"]["events"] == ex.EVENTS["bnci2014_001"]
    assert cfg["experiment"]["seeds"] == [0, 1, 2, 3, 4]
    assert 0 < cfg["experiment"]["val_split"] < 1
    assert cfg["kan"]["d_bottleneck"] == CNN_KAN().backbone.d_bottleneck
    cfg = ex.load_config(PROJECT_ROOT / "configs" / "physionet.yaml")
    assert cfg["dataset"]["events"] == ex.EVENTS["physionet"]


# --------------------------------------------------------------------------- wrapper contract
def test_wrapper_clone_with_default_kwargs():
    from sklearn.base import clone, is_classifier
    for kwargs in (None, {}, {"channels": 22}):
        est = SkorchEEGClassifier(module=CNN_KAN, module_kwargs=kwargs)
        cloned = clone(est)  # audit item 17
        assert cloned.get_params() == est.get_params()
        assert is_classifier(est)
    assert SkorchEEGClassifier(module=CNN_KAN).get_params()["device"] is None


def test_wrapper_does_not_crash_on_lone_last_batch():
    """17 trials at batch 8 used to leave a 1-trial batch -> BatchNorm ValueError (audit item 16)."""
    X = np.random.randn(17, 1, 22, 200).astype(np.float32)
    y = np.array([0, 1, 2, 3] * 4 + [0])
    est = SkorchEEGClassifier(module=CNN_KAN, module_kwargs={"channels": 22, "num_classes": 4},
                              batch_size=8, max_epochs=1, patience=0, device="cpu", train_split=None)
    est.fit(X, y)
    assert est.predict(X).shape == (17,)
    assert list(est.classes_) == [0, 1, 2, 3]


def test_wrapper_validation_split_and_fallback():
    big = SkorchEEGClassifier(module=CNN_KAN, random_state=3, val_split=0.2)
    split = big._resolve_train_split(n_samples=200, n_classes=4)
    assert split is not None and split.stratified and split.random_state == 3
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert big._resolve_train_split(n_samples=12, n_classes=4) is None      # 3 valid < 4 classes
        assert big._resolve_train_split(n_samples=200, n_classes=4, min_class_count=1) is None
        assert any("too small" in str(x.message) for x in w)
    assert SkorchEEGClassifier(module=CNN_KAN, train_split=None)._resolve_train_split(200, 4) is None


def test_wrapper_drop_last_follows_custom_split_fraction():
    from skorch.dataset import ValidSplit
    # 713 trials, k=10 -> first fold = ceil(71.3) = 72 held out -> 641 training trials -> lone last batch at 64 -> drop
    est = SkorchEEGClassifier(module=CNN_KAN, train_split=ValidSplit(10), batch_size=64)
    net = est._build_net(n_samples=713, n_classes=4)
    assert net.iterator_train__drop_last is True
    est = SkorchEEGClassifier(module=CNN_KAN, train_split=ValidSplit(0.5, stratified=True), val_split=0.2, batch_size=8)
    net = est._build_net(n_samples=130, n_classes=4)   # 65 train -> 65 % 8 == 1 -> drop
    assert net.iterator_train__drop_last is True


def test_run_cross_subject_flushes_scores_each_seed(tmp_path, monkeypatch):
    """scores.csv must exist after the first seed even if a later seed crashes."""
    import pandas as pd
    calls = {"n": 0}

    class FakeEval:
        def __init__(self, **kw): pass
        def process(self, pipelines):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return pd.DataFrame({"score": [0.5], "subject": [1], "session": ["1test"], "pipeline": ["m"]})

    class FakeDS:
        code = "X"
    from sklearn.base import BaseEstimator
    monkeypatch.setattr(ex, "CrossSubjectEvaluation", FakeEval)
    with pytest.raises(RuntimeError):
        ex.run_cross_subject(FakeDS(), None, {"m": BaseEstimator()}, seeds=[0, 1], run_dir=tmp_path)
    df = pd.read_csv(tmp_path / "scores.csv")
    assert list(df["seed"]) == [0] and list(df["model"]) == ["m"]


def test_wrapper_early_stopping_restores_best_epoch():
    """With a validation split the callback watches valid_loss and keeps the best weights."""
    X = np.random.randn(80, 1, 22, 200).astype(np.float32)
    y = np.tile(np.arange(4), 20)
    est = SkorchEEGClassifier(module=CNN_KAN, module_kwargs={"channels": 22, "num_classes": 4},
                              batch_size=16, max_epochs=3, patience=1, device="cpu", random_state=0)
    est.fit(X, y)
    cb = dict(est.net_.callbacks_)["EarlyStopping"]
    assert cb.monitor == "valid_loss" and cb.load_best is True
    assert "valid_loss" in est.net_.history[-1]


def test_wrapper_seed_reproducible_on_cpu():
    X = np.random.RandomState(0).randn(48, 1, 22, 200).astype(np.float32)
    y = np.tile(np.arange(4), 12)
    out = []
    for _ in range(2):
        est = SkorchEEGClassifier(module=CNN_KAN, module_kwargs={"channels": 22, "num_classes": 4},
                                  batch_size=16, max_epochs=2, patience=0, device="cpu", random_state=11)
        out.append(est.fit(X, y).predict_proba(X))
    np.testing.assert_allclose(out[0], out[1], atol=1e-6)


# --------------------------------------------------------------------------- sklearn fitted-checks
def test_importing_runners_keeps_sklearn_fitted_checks():
    """Audit item 24: no global Pipeline monkeypatch with the pinned MOABB (which marks fold models itself)."""
    pytest.importorskip("moabb")
    from sklearn.exceptions import NotFittedError
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.validation import check_is_fitted
    import run_benchmark, run_experiments, run_physionet_scaling  # noqa: F401  (import side effects)

    if not ex._moabb_marks_pipelines_fitted():
        pytest.skip("Installed MOABB predates _ensure_fitted; shim active as intended.")
    assert ex._moabb_marks_pipelines_fitted()
    assert ex.LEGACY_PIPELINE_SHIM is False
    with pytest.raises(NotFittedError):
        check_is_fitted(Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())]))


def test_legacy_pipeline_shim_only_for_old_moabb(monkeypatch):
    from sklearn.pipeline import Pipeline
    original = Pipeline.__dict__.get("__sklearn_is_fitted__")
    try:
        monkeypatch.setattr(ex, "_moabb_marks_pipelines_fitted", lambda: True)
        assert ex._install_legacy_pipeline_fitted_shim() is False
        assert Pipeline.__dict__.get("__sklearn_is_fitted__") is original

        if ex.CrossSubjectEvaluation is None:
            pytest.skip("moabb not installed")
        monkeypatch.setattr(ex, "_moabb_marks_pipelines_fitted", lambda: False)
        with pytest.warns(RuntimeWarning, match="moabb 1.7.2"):
            assert ex._install_legacy_pipeline_fitted_shim() is True
        assert Pipeline([("x", None)]).__sklearn_is_fitted__() is True
    finally:
        if original is None:
            if "__sklearn_is_fitted__" in Pipeline.__dict__:
                delattr(Pipeline, "__sklearn_is_fitted__")
        else:
            Pipeline.__sklearn_is_fitted__ = original


# --------------------------------------------------------------------------- end to end
@pytest.mark.slow
def test_cross_subject_smoke_bnci(tmp_path):
    """
    2 subjects, 1 seed, 2 epochs, csp_lda + cnn_kan. Checks the scores schema (seed column,
    both sessions, 4 classes, dataset code) and that MOABB's store stays inside tmp_path.
    Downloads BNCI2014_001 subjects 1-2 on first run.
    """
    pytest.importorskip("moabb")
    from src.utils.seed import set_seed

    dataset = ex.make_dataset("bnci2014_001", subjects=[1, 2])
    paradigm = ex.make_paradigm("bnci2014_001")
    shape = ex.infer_shape(paradigm, dataset)
    assert shape["n_chans"] == 22 and shape["n_times"] == 1001 and shape["n_classes"] == 4
    assert shape["ch_names"][-1] == "POz" and len(shape["ch_names"]) == 22  # audit item 13

    pipes = ex.build_pipelines(["csp_lda", "cnn_kan"], shape, max_epochs=2, patience=0, device="cpu")
    scores = ex.run_cross_subject(dataset, paradigm, pipes, seeds=[0], run_dir=tmp_path, suffix="test_")

    assert {"seed", "session", "subject", "score", "model", "dataset"} <= set(scores.columns)
    assert set(scores["seed"]) == {0}
    assert set(scores["session"].astype(str)) == {"0train", "1test"}  # audit item 9
    assert set(scores["dataset"]) == {dataset.code}
    assert set(scores["model"]) == {"csp_lda", "cnn_kan"}
    if "n_classes" in scores.columns:
        assert int(scores["n_classes"].iloc[0]) == 4
    assert (tmp_path / "scores.csv").exists()
    assert list((tmp_path / "seed0").rglob("*.hdf5")), "MOABB store must live under run_dir"
    home_store = Path(os.path.expanduser("~/mne_data/results/MotorImagery/CrossSubjectEvaluation"))
    assert not (home_store / "results_test_seed0.hdf5").exists(), "global cache must not be touched"

    info = ex.write_run_info(tmp_path, seeds=[0], shape=shape)
    data = json.loads(Path(info).read_text())
    assert data["shape"]["ch_names"][-1] == "POz" and "versions" in data and "git_sha" in data

    summary = ex.summarize_scores(scores)
    assert set(summary["model"]) == {"csp_lda", "cnn_kan"} and (summary["n_subjects"] == 2).all()


def test_sweep_mode_names_main_grid_cnn_kan():
    from run_experiments import model_names_for_mode
    cfg = {"experiment": {"models": ["cnn_kan", "eegnet"], "grid_sweep": [3, 5, 8]}, "kan": {"grid_size": 3}, "dataset": {}}
    assert model_names_for_mode(cfg, "sweep") == ["cnn_kan", "cnn_kan_G5", "cnn_kan_G8"]
    assert model_names_for_mode(cfg, "ablations") == ["cnn_kan", "eegnet"]
    full = model_names_for_mode(cfg, "full")
    assert full == ["cnn_kan", "eegnet", "cnn_kan_G5", "cnn_kan_G8"] and len(set(full)) == len(full)
