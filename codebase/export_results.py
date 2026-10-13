"""
Single reporting entry point (issue #3, PR 4).

Reads ``<run_dir>/scores.csv`` + ``run_info.json`` written by the runners, and produces:
  * ``<run_dir>/phase_3_kan_benchmark_results.json`` — per-model summary (subject-level mean,
    subject std, seed std, params, MACs with KAN layers counted, backbone/head split) and the
    Holm-corrected Wilcoxon comparisons against the target model;
  * ``<run_dir>/phase_3_table_snippet.tex`` — escaped, booktabs table;
  * ``<run_dir>/figures/pareto_*.{png,pdf}`` + ``pareto_efficiency_summary.json`` — from the
    summary, nothing hardcoded.
Parameter and MAC counts are measured on freshly built modules at the run's input shape.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent  # codebase/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.stats import run_statistical_suite, seed_spread
from src.utils.metrics import profile_model
from src.utils.pareto import generate_pareto_plots, sanitize_for_json
from src.utils.experiment import build_pipeline, parse_model_name, summarize_scores


# --------------------------------------------------------------------------- LaTeX
_LATEX_SPECIALS = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                   "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def latex_escape(text: Any) -> str:
    """Escape every character LaTeX treats specially in text mode."""
    out = []
    for ch in str(text):
        out.append(_LATEX_SPECIALS.get(ch, ch))
    return "".join(out)


def _fmt_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def format_latex_results_table(summary_dict: Dict[str, Any], caption: Optional[str] = None,
                               label: str = "tab:loso_benchmark_results") -> str:
    """
    Booktabs table from an ``export_results`` summary. Names and head types are escaped;
    accuracies are per-subject mean $\\pm$ std across subjects; ``p`` is Holm-adjusted when
    available (``p_adj``), else the raw Wilcoxon p.
    """
    caption = caption or "Leave-one-subject-out cross-subject accuracy, parameter count and Holm-corrected Wilcoxon test against the main model."
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Head} & \textbf{$G$} & \textbf{Params} & \textbf{Acc.\ (\%)} & \textbf{$p$} \\",
        r"\midrule",
    ]
    for key, res in summary_dict.items():
        name = latex_escape(res.get("name", key))
        head = latex_escape(res.get("head_type", "N/A"))
        grid = latex_escape(res.get("grid_size", "N/A"))
        p = res.get("params")
        params = f"{int(p):,}" if isinstance(p, (int, float)) and p is not None and not np.isnan(p) else "--"
        acc = f"{float(res.get('mean_acc', 0.0)) * 100:.2f} $\\pm$ {float(res.get('std_acc', 0.0) or 0.0) * 100:.2f}"
        p_val = res.get("p_adj", res.get("p_value"))
        lines.append(f"{name} & {head} & {grid} & {params} & {acc} & {_fmt_p(p_val)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- summary
def head_type_for(name: str, grid_size_main: int = 3) -> Dict[str, Any]:
    base, grid = parse_model_name(name)
    if base in ("cnn_kan", "kan"):
        g = grid if grid is not None else grid_size_main
        return {"head_type": f"B-spline KAN (G={g})", "grid_size": g}
    return {
        "cnn_linear": {"head_type": "Dense linear", "grid_size": "N/A"},
        "cnn_mlp": {"head_type": "MLP (param-matched)", "grid_size": "N/A"},
        "eegnet": {"head_type": "Dense classifier", "grid_size": "N/A"},
        "csp_lda": {"head_type": "LDA", "grid_size": "N/A"},
        "svm": {"head_type": "RBF SVM", "grid_size": "N/A"},
    }.get(base, {"head_type": base, "grid_size": "N/A"})


def profile_for(name: str, shape: Dict[str, Any], hp: Dict[str, Any]) -> Dict[str, Any]:
    """Params / MACs measured on a fresh module (CPU); classical models have no MACs."""
    est = build_pipeline(name, shape["n_chans"], shape["n_times"], shape["n_classes"],
                         **{**{k: v for k, v in hp.items() if k != "device"}, "device": "cpu"})
    if getattr(est, "module", None) is None:
        return {"params": None, "macs_mmac": None, "backbone_macs": None, "head_macs": None}
    net = est._build_net()
    net.initialize()
    prof = profile_model(net.module_, (1, shape["n_chans"], shape["n_times"]))
    return {"params": int(prof["params"]), "macs_mmac": _nan_to_none(prof["macs_mmac"]),
            "backbone_macs": _nan_to_none(prof["backbone_macs"]), "head_macs": _nan_to_none(prof["head_macs"])}


def _nan_to_none(x):
    return None if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)) else float(x)


def pick_target(models, target: Optional[str]) -> Optional[str]:
    """Explicit target if present; else ``cnn_kan``; else the first cnn_kan_G* variant."""
    names = [str(m) for m in models]
    for cand in ([target] if target else []) + ["cnn_kan"]:
        if cand in names:
            return cand
    kan = sorted(n for n in names if parse_model_name(n)[0] in ("cnn_kan", "kan"))
    return kan[0] if kan else None


def build_summary(scores: pd.DataFrame, shape: Dict[str, Any], hp: Dict[str, Any],
                  target_model: Optional[str] = "cnn_kan", grid_size_main: int = 3) -> Dict[str, Any]:
    table = summarize_scores(scores).set_index("model")
    spread = seed_spread(scores)
    target = pick_target(table.index, target_model)
    stats_report = run_statistical_suite(scores, target_model=target) if target else {}

    summary: Dict[str, Any] = {}
    for model, row in table.iterrows():
        info = stats_report.get(model, {})
        summary[model] = {
            "name": model,
            **head_type_for(model, grid_size_main),
            **profile_for(model, shape, hp),
            "mean_acc": float(row["mean_acc"]),
            "std_acc": _nan_to_none(row["std_subjects"]) or 0.0,
            "std_seeds": _nan_to_none(spread.get(model, {}).get("std")),
            "n_subjects": int(row["n_subjects"]),
            "n_seeds": int(spread.get(model, {}).get("n_seeds", 1)),
            "wilcoxon_w": info.get("w_statistic"),
            "p_value": info.get("p_value"),
            "p_adj": info.get("p_adj"),
            "cohens_d": _nan_to_none(info.get("cohens_d")),  # inf/nan -> null in JSON
            "rank_biserial": info.get("rank_biserial"),
            "is_target": model == target,
        }
    return {"target_model": target, "summary": summary, "statistical_significance": stats_report}


# --------------------------------------------------------------------------- entry point
def export_run(run_dir: Path, target_model: Optional[str] = "cnn_kan", pareto: bool = True,
               panel_label: Optional[str] = None) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    scores_path = run_dir / "scores.csv"
    info_path = run_dir / "run_info.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} not found; run a runner with --run-dir {run_dir} first")
    scores = pd.read_csv(scores_path)
    if "model" not in scores.columns:
        if "pipeline" not in scores.columns:
            raise ValueError("scores.csv needs a 'model' or 'pipeline' column")
        scores["model"] = scores["pipeline"]
    if "subject" not in scores.columns or "score" not in scores.columns:
        raise ValueError("scores.csv needs 'subject' and 'score' columns")

    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    shape = info.get("shape") or {}
    if not {"n_chans", "n_times", "n_classes"} <= set(shape):
        shape = {"n_chans": int(scores["channels"].iloc[0]), "n_times": int(scores["samples"].iloc[0]),
                 "n_classes": int(scores.get("n_classes", pd.Series([4])).iloc[0])}
    hp = info.get("hyperparameters") or {}
    grid_main = int(hp.get("grid_size", 3))

    result = build_summary(scores, shape, hp, target_model=target_model, grid_size_main=grid_main)
    result["run_dir"] = str(run_dir)
    result["dataset"] = info.get("dataset") or (str(scores["dataset"].iloc[0]) if "dataset" in scores else None)
    result["seeds"] = sorted(int(s) for s in scores["seed"].unique()) if "seed" in scores else None

    json_path = run_dir / "phase_3_kan_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(sanitize_for_json(result), f, indent=2, allow_nan=False)
    print(f"[SAVED] summary JSON -> {json_path}")

    tex_path = run_dir / "phase_3_table_snippet.tex"
    tex_path.write_text(format_latex_results_table(result["summary"]))
    print(f"[SAVED] LaTeX table -> {tex_path}")

    if pareto:
        label = panel_label or result["dataset"] or run_dir.name
        generate_pareto_plots({label: result}, output_dir=run_dir / "figures", results_dir=run_dir)

    _print_table(result["summary"], result["target_model"])
    return result


def _print_table(summary: Dict[str, Any], target: Optional[str]) -> None:
    print("\n" + "=" * 92)
    print(f"{'Model':<14} | {'Head':<20} | {'Params':>7} | {'MMac':>6} | {'Acc (%) mean±std_subj':<22} | {'std_seed':>8} | {'p_adj':>7}")
    print("-" * 92)
    for m, r in summary.items():
        params = f"{r['params']:,}" if r.get("params") is not None else "--"
        macs = f"{r['macs_mmac']:.2f}" if r.get("macs_mmac") is not None else "--"
        acc = f"{r['mean_acc'] * 100:.2f} ± {(r['std_acc'] or 0) * 100:.2f}"
        sd = f"{r['std_seeds'] * 100:.2f}" if r.get("std_seeds") is not None else "--"
        p = _fmt_p(r.get("p_adj")).replace("$", "")
        flag = " *" if m == target else ""
        print(f"{m + flag:<14} | {r['head_type']:<20} | {params:>7} | {macs:>6} | {acc:<22} | {sd:>8} | {p:>7}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export summary JSON, LaTeX table and Pareto plots from a run directory")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "results" / "phase3",
                        help="Directory holding scores.csv and run_info.json (default: results/phase3)")
    parser.add_argument("--target", type=str, default="cnn_kan", help="Model the Wilcoxon tests compare against")
    parser.add_argument("--label", type=str, default=None, help="Panel label for the Pareto figure (default: dataset)")
    parser.add_argument("--no-pareto", action="store_true")
    args = parser.parse_args()
    export_run(args.run_dir, target_model=args.target, pareto=not args.no_pareto, panel_label=args.label)
