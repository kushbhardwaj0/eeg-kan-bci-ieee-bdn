"""
Statistics and reporting tests (issue #3, PR 4): subject-level aggregation, Holm correction,
effect-size edge cases, KAN MACs, escaped LaTeX that compiles, Pareto plots from data only,
and the single export entry point on a synthetic run directory.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import stats as st
from src.utils.metrics import compute_classification_metrics, get_model_flops, kan_linear_macs, profile_model
from src.models.cnn_kan import CNN_KAN
from src.models.ablations import CNN_Linear
from src.models.kan_head import KANLinear
import export_results as er


# --------------------------------------------------------------------------- stats
def test_holm_correction_matches_hand_computation():
    # p sorted: 0.01, 0.02, 0.03, 0.04 ; m=4 -> 0.04, 0.06, 0.06, 0.06 (step-down with monotone max)
    adj = st.holm_correction([0.04, 0.01, 0.03, 0.02])
    np.testing.assert_allclose(adj, [0.06, 0.04, 0.06, 0.06])
    # capped at 1, NaN passthrough, single p unchanged
    assert st.holm_correction([0.6, 0.7]) == [1.0, 1.0]
    out = st.holm_correction([0.02, float("nan")])
    assert out[0] == 0.02 and np.isnan(out[1])


def test_aggregate_per_subject_averages_sessions_and_seeds():
    df = pd.DataFrame({
        "model": ["m"] * 4, "subject": [1, 1, 1, 1],
        "session": ["0train", "1test", "0train", "1test"], "seed": [0, 0, 1, 1],
        "score": [0.4, 0.6, 0.5, 0.7],
    })
    agg = st.aggregate_per_subject(df)
    assert len(agg) == 1 and agg["score"].iloc[0] == pytest.approx(0.55)
    spread = st.seed_spread(df)
    assert spread["m"]["n_seeds"] == 2 and spread["m"]["std"] == pytest.approx(np.std([0.5, 0.6], ddof=1))


def test_suite_uses_subject_level_scores_and_holm():
    rows = []
    rng = np.random.RandomState(0)
    for subj in range(9):
        for seed in (0, 1):
            for sess in ("0train", "1test"):
                rows.append({"model": "cnn_kan", "subject": subj, "seed": seed, "session": sess, "score": 0.6 + rng.normal(0, 0.02)})
                rows.append({"model": "eegnet", "subject": subj, "seed": seed, "session": sess, "score": 0.5 + rng.normal(0, 0.02)})
                rows.append({"model": "csp_lda", "subject": subj, "seed": seed, "session": sess, "score": 0.4 + rng.normal(0, 0.02)})
    out = st.run_statistical_suite(pd.DataFrame(rows), target_model="cnn_kan")
    assert set(out) == {"eegnet", "csp_lda"}
    for r in out.values():
        assert r["num_subjects"] == 9 and r["n_comparisons"] == 2 and r["correction"] == "holm"
        assert r["p_adj"] >= r["p_value"]
        assert not np.isnan(r["target_seed_std"])
    with pytest.raises(ValueError):
        st.run_statistical_suite(pd.DataFrame(rows), target_model="transformer")


def test_cohens_d_edge_cases():
    assert np.isinf(st.compute_cohens_d([0.6, 0.6, 0.6], [0.5, 0.5, 0.5]))
    assert np.isnan(st.compute_cohens_d([0.5, 0.5], [0.5, 0.5]))
    assert np.isnan(st.compute_cohens_d([0.5], [0.4]))


# --------------------------------------------------------------------------- metrics
def test_kan_macs_are_counted():
    head_only = get_model_flops(CNN_KAN().kan_head, (16,))
    assert head_only["macs_value"] >= kan_linear_macs(KANLinear(16, 8)) + kan_linear_macs(KANLinear(8, 4))
    kan = profile_model(CNN_KAN(), (1, 22, 1001))
    lin = profile_model(CNN_Linear(), (1, 22, 1001))
    assert kan["macs"] > lin["macs"]                       # heads no longer identical (audit item 14)
    assert kan["head_macs"] > lin["head_macs"] > 0
    assert kan["backbone_macs"] == pytest.approx(lin["backbone_macs"], rel=0.05)
    assert kan["macs"] == pytest.approx(kan["backbone_macs"] + kan["head_macs"], rel=0.05)


def test_metrics_auc_failure_is_nan_not_zero():
    y = np.array([0, 1, 2, 3]); bad_prob = np.ones((4, 3))
    m = compute_classification_metrics(y, y, bad_prob)
    assert np.isnan(m["roc_auc"])
    assert np.isnan(get_model_flops(torch.nn.Linear(2, 2), (1, 2, 3))["macs_value"])


# --------------------------------------------------------------------------- latex
def test_latex_escape_and_table():
    assert er.latex_escape("a_b & 10% #1") == r"a\_b \& 10\% \#1"
    tex = er.format_latex_results_table({
        "cnn_kan": {"name": "cnn_kan", "head_type": "B-spline KAN (G=3)", "grid_size": 3, "params": 2976,
                    "mean_acc": 0.45, "std_acc": 0.1, "p_adj": 0.0004},
        "csp_lda": {"name": "csp_lda", "head_type": "LDA", "grid_size": "N/A", "params": None,
                    "mean_acc": 0.40, "std_acc": 0.1, "p_value": 0.2},
    })
    assert r"\toprule" in tex and r"cnn\_kan" in tex and "--" in tex and "$<$0.001" in tex and "0.200" in tex
    assert "&" not in tex.split(r"\caption")[1].split("\n")[0]  # caption is escaped


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_latex_table_compiles(tmp_path):
    tex = er.format_latex_results_table({"cnn_kan_G5": {"name": "cnn_kan_G5", "head_type": "B-spline KAN (G=5)",
                                                        "grid_size": 5, "params": 3296, "mean_acc": 0.5, "std_acc": 0.1,
                                                        "p_adj": 0.03}})
    (tmp_path / "t.tex").write_text("\\documentclass{article}\\usepackage{booktabs}\\begin{document}" + tex + "\\end{document}")
    res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-800:]
    assert (tmp_path / "t.pdf").exists()


# --------------------------------------------------------------------------- export
def _synthetic_run(run_dir: Path) -> None:
    rng = np.random.RandomState(1)
    rows = []
    for model, base in (("cnn_kan", 0.46), ("cnn_linear", 0.42), ("eegnet", 0.44), ("csp_lda", 0.40)):
        for subj in (1, 2, 3):
            for seed in (0, 1):
                for sess in ("0train", "1test"):
                    rows.append({"score": base + rng.normal(0, 0.03), "subject": subj, "session": sess, "seed": seed,
                                 "model": model, "pipeline": model, "dataset": "BNCI2014-001",
                                 "channels": 22, "samples": 1001, "n_classes": 4})
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / "scores.csv", index=False)
    (run_dir / "run_info.json").write_text(json.dumps({
        "dataset": "BNCI2014-001",
        "shape": {"n_chans": 22, "n_times": 1001, "n_classes": 4},
        "hyperparameters": {"grid_size": 3, "spline_order": 3, "d_bottleneck": 16, "max_epochs": 1},
    }))


def test_export_run_end_to_end(tmp_path):
    _synthetic_run(tmp_path)
    result = er.export_run(tmp_path, target_model="cnn_kan")
    s = result["summary"]
    assert result["target_model"] == "cnn_kan" and set(s) == {"cnn_kan", "cnn_linear", "eegnet", "csp_lda"}
    assert s["cnn_kan"]["params"] == 2784 and s["eegnet"]["params"] == 3444 and s["csp_lda"]["params"] is None
    assert s["cnn_kan"]["macs_mmac"] > s["cnn_linear"]["macs_mmac"]
    assert s["cnn_kan"]["n_subjects"] == 3 and s["cnn_kan"]["n_seeds"] == 2 and s["cnn_kan"]["std_seeds"] is not None
    assert s["eegnet"]["p_adj"] is not None and s["eegnet"]["p_adj"] >= s["eegnet"]["p_value"]
    assert (tmp_path / "phase_3_kan_benchmark_results.json").exists()
    assert (tmp_path / "phase_3_table_snippet.tex").exists()
    assert (tmp_path / "figures" / "pareto_accuracy_vs_parameters.png").exists()
    pareto = json.loads((tmp_path / "pareto_efficiency_summary.json").read_text())
    label = "BNCI2014-001"
    assert pareto["profiles"][label]["cnn_kan"]["mean_acc"] == pytest.approx(s["cnn_kan"]["mean_acc"] * 100)
    assert pareto["profiles"][label]["cnn_kan"]["n"] == 3
    # idempotent: a second export reproduces the same JSON
    again = er.export_run(tmp_path, target_model="cnn_kan", pareto=False)
    assert again["summary"]["cnn_kan"]["mean_acc"] == s["cnn_kan"]["mean_acc"]


def test_export_run_falls_back_to_a_kan_variant(tmp_path):
    _synthetic_run(tmp_path)
    df = pd.read_csv(tmp_path / "scores.csv")
    df.loc[df["model"] == "cnn_kan", ["model", "pipeline"]] = "cnn_kan_G5"
    df.to_csv(tmp_path / "scores.csv", index=False)
    result = er.export_run(tmp_path, target_model="cnn_kan", pareto=False)
    assert result["target_model"] == "cnn_kan_G5" and result["summary"]["cnn_kan_G5"]["grid_size"] == 5


def test_export_run_requires_scores(tmp_path):
    with pytest.raises(FileNotFoundError):
        er.export_run(tmp_path)


def test_export_run_without_seed_column_and_json_is_strict(tmp_path):
    _synthetic_run(tmp_path)
    df = pd.read_csv(tmp_path / "scores.csv").drop(columns=["seed"])
    df = df[df["model"] != "csp_lda"]
    df.to_csv(tmp_path / "scores.csv", index=False)
    result = er.export_run(tmp_path, pareto=True)
    assert result["seeds"] is None and result["summary"]["cnn_kan"]["n_seeds"] == 1
    assert result["summary"]["cnn_kan"]["std_seeds"] is None
    # strict parsers must accept the files (no bare NaN / Infinity tokens)
    for name in ("phase_3_kan_benchmark_results.json", "pareto_efficiency_summary.json"):
        text = (tmp_path / name).read_text()
        assert "NaN" not in text and "Infinity" not in text
        json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))

