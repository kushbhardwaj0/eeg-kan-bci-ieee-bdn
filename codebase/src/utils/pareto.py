"""
Pareto Frontier & Efficiency Profiling Module.

Every point comes from a results summary (``export_results.py`` JSON); only the colour and
marker style map lives in code. The frontier, the error-bar ``n`` and the summary text are
all derived from the data.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Style only. Keys are base model names (see experiment.parse_model_name); anything else
# falls back to the ``_default`` entry.
STYLE = {
    "cnn_kan":    {"color": "#1f77b4", "marker": "o", "is_kan": True},
    "cnn_linear": {"color": "#ff7f0e", "marker": "v", "is_kan": False},
    "cnn_mlp":    {"color": "#2ca02c", "marker": "D", "is_kan": False},
    "csp_lda":    {"color": "#7f7f7f", "marker": "X", "is_kan": False},
    "svm":        {"color": "#9467bd", "marker": "*", "is_kan": False},
    "eegnet":     {"color": "#d62728", "marker": "P", "is_kan": False},
    "_default":   {"color": "#17becf", "marker": "s", "is_kan": False},
}
KAN_SHADES = ["#1f77b4", "#3897d3", "#5bb4e5", "#8fd0f0"]


def sanitize_for_json(obj):
    """Recursively turn NaN/inf (invalid JSON) into None and numpy scalars into Python ones."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


def compute_pareto_frontier(costs: np.ndarray, yields: np.ndarray) -> List[int]:
    """Indices of Pareto-efficient points (minimise cost, maximise yield)."""
    costs = np.asarray(costs, dtype=float)
    yields = np.asarray(yields, dtype=float)
    n = costs.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and (costs[j] <= costs[i] and yields[j] >= yields[i]) and (
                costs[j] < costs[i] or yields[j] > yields[i]
            ):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0].tolist()


def _base_name(model: str) -> str:
    base = model.lower()
    if "_g" in base and base.rpartition("_g")[2].isdigit():
        base = base.rpartition("_g")[0]
    return base


def profiles_from_summary(summary: Union[Dict[str, Any], str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    ``export_results.py`` JSON (or its ``summary`` dict) -> plot profiles.
    Accuracy is kept in percent; ``n`` is the real subject count behind each mean.
    """
    if not isinstance(summary, dict):
        with open(summary, "r") as f:
            summary = json.load(f)
    summary = summary.get("summary", summary)
    profiles: Dict[str, Dict[str, Any]] = {}
    kan_i = 0
    for model, res in summary.items():
        params = res.get("params")  # None (classical model) -> NaN: excluded from the parameter panel
        base = _base_name(model)
        style = dict(STYLE.get(base, STYLE["_default"]))
        if style["is_kan"]:
            style["color"] = KAN_SHADES[min(kan_i, len(KAN_SHADES) - 1)]
            kan_i += 1
        acc = float(res.get("mean_acc", np.nan))
        std = float(res.get("std_acc", np.nan) or 0.0)
        macs = res.get("macs_mmac")
        profiles[res.get("name", model)] = {
            "params": float(params) if params is not None else float("nan"),
            "macs_mmac": float(macs) if macs is not None else float("nan"),
            "mean_acc": acc * 100 if acc <= 1.0 else acc,
            "std_acc": std * 100 if acc <= 1.0 else std,
            "n": int(res.get("n_subjects", 1) or 1),
            **style,
        }
    return profiles


def _plot_panel(ax, profiles: Dict[str, Dict[str, Any]], x_key: str, title: str, xlabel: str,
                log_x: bool) -> List[str]:
    names = list(profiles)
    xs = np.array([profiles[m][x_key] for m in names], dtype=float)
    ys = np.array([profiles[m]["mean_acc"] for m in names], dtype=float)
    es = np.array([profiles[m]["std_acc"] / np.sqrt(profiles[m]["n"]) for m in names], dtype=float)
    ok = ~np.isnan(xs) & ~np.isnan(ys)
    for i, name in enumerate(names):
        if not ok[i]:
            continue
        p = profiles[name]
        ax.errorbar(xs[i], ys[i], yerr=es[i], fmt=p["marker"], color=p["color"], ecolor=p["color"],
                    elinewidth=1.5, capsize=4, markersize=9, label=name, alpha=0.9)
        display_name = {
            "cnn_kan": "CNN-KAN",
            "cnn_mlp": "CNN-MLP",
            "cnn_linear": "CNN-Linear",
            "cnn_kan_G5": "CNN-KAN (G=5)",
            "cnn_kan_G8": "CNN-KAN (G=8)",
            "eegnet": "EEGNet",
            "csp_lda": "CSP+LDA",
        }.get(name, name)
        txt = (f"{display_name}\n({xs[i]:,.0f} p, {ys[i]:.1f}%)" if x_key == "params"
               else f"{display_name}\n({xs[i]:.2f} MMac, {ys[i]:.1f}%)")

        offset_rules = {
            "params": {
                "cnn_mlp": (-26, 22, "right", "bottom"),
                "cnn_kan": (26, 18, "left", "bottom"),
                "cnn_kan_G5": (22, -22, "left", "top"),
                "cnn_kan_G8": (20, 2, "left", "center"),
                "cnn_linear": (-18, 16, "right", "bottom"),
                "eegnet": (22, 16, "left", "bottom"),
            },
            "macs_mmac": {
                "cnn_mlp": (-26, 22, "right", "bottom"),
                "cnn_kan": (26, 18, "left", "bottom"),
                "cnn_kan_G5": (22, -22, "left", "top"),
                "cnn_kan_G8": (20, 2, "left", "center"),
                "cnn_linear": (-18, 16, "right", "bottom"),
                "eegnet": (22, 16, "left", "bottom"),
            },
        }
        dx, dy, ha, va = offset_rules.get(x_key, {}).get(name, (12, 10, "left", "bottom"))

        ax.annotate(
            txt,
            (xs[i], ys[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold" if p["is_kan"] else "normal",
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=p["color"], lw=0.9, alpha=0.9),
            arrowprops=dict(
                arrowstyle="-|>",
                color=p["color"],
                lw=0.8,
                shrinkA=2,
                shrinkB=4,
                mutation_scale=8,
                alpha=0.8,
            ),
        )
    idx_ok = np.where(ok)[0]
    frontier_local = compute_pareto_frontier(xs[ok], ys[ok]) if ok.any() else []
    frontier = [int(idx_ok[i]) for i in frontier_local]
    if frontier:
        order = sorted(frontier, key=lambda i: xs[i])
        ax.plot(xs[order], ys[order], color="#1f77b4", linestyle="--", linewidth=2, alpha=0.7,
                label="Pareto frontier")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    ax.set_ylabel("LOSO accuracy (%)", fontsize=10, fontweight="bold")
    if log_x and ok.any() and np.all(xs[ok] > 0):
        ax.set_xscale("log")
    if ok.any():
        pad = max(2.0, 0.15 * (np.nanmax(ys) - np.nanmin(ys)))
        emax = np.nanmax(np.where(np.isnan(es), 0, es))
        ax.set_ylim(np.nanmin(ys) - pad - emax, np.nanmax(ys) + pad + emax)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    return [names[i] for i in frontier]


def _dominance_text(profiles: Dict[str, Dict[str, Any]], frontier: List[str]) -> str:
    kan = [m for m in profiles if profiles[m]["is_kan"]]
    if not kan:
        return "no KAN variant in this panel"
    on = [m for m in kan if m in frontier]
    if on:
        return f"{', '.join(on)} on the Pareto frontier; frontier = {', '.join(frontier)}"
    return f"no KAN variant on the Pareto frontier; frontier = {', '.join(frontier)}"


def generate_pareto_plots(
    summaries: Optional[Dict[str, Union[Dict[str, Any], str, Path]]] = None,
    output_dir: Path = FIGURES_DIR,
    results_dir: Path = RESULTS_DIR,
    summary_json: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    ``summaries``: {panel label: export_results summary (dict or JSON path)}, one panel per
    dataset. ``summary_json`` is a shortcut for a single panel. Writes
    ``pareto_accuracy_vs_parameters.{png,pdf}`` (all panels) and
    ``pareto_accuracy_vs_macs.{png,pdf}`` (first panel) plus ``pareto_efficiency_summary.json``.
    """
    if summaries is None:
        if summary_json is None:
            raise ValueError("generate_pareto_plots needs `summaries` or `summary_json`; "
                             "there are no built-in numbers.")
        summaries = {"results": summary_json}
    output_dir = Path(output_dir)
    results_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    panels = {label: profiles_from_summary(s) for label, s in summaries.items()}
    labels = list(panels)

    # Figure 1: accuracy vs trainable parameters, one panel per dataset
    fig, axes = plt.subplots(1, len(labels), figsize=(7.5 * len(labels), 6), dpi=200, squeeze=False)
    frontiers: Dict[str, List[str]] = {}
    for ax, label in zip(axes[0], labels):
        frontiers[label] = _plot_panel(ax, panels[label], "params", f"{label}: accuracy vs parameters",
                                       "Trainable parameters", log_x=True)
    if len(labels) > 1:
        plt.suptitle("Pareto efficiency: cross-subject accuracy vs parameter budget", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()
    fig_params_png = output_dir / "pareto_accuracy_vs_parameters.png"
    fig_params_pdf = output_dir / "pareto_accuracy_vs_parameters.pdf"
    fig.savefig(fig_params_png, dpi=200, bbox_inches="tight")
    fig.savefig(fig_params_pdf, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: accuracy vs MACs for the first panel
    first = labels[0]
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=200)
    macs_frontier = _plot_panel(ax, panels[first], "macs_mmac", f"{first}: accuracy vs MACs",
                                "Multiply-accumulate operations (MMac)", log_x=False)
    plt.tight_layout()
    fig_macs_png = output_dir / "pareto_accuracy_vs_macs.png"
    fig_macs_pdf = output_dir / "pareto_accuracy_vs_macs.pdf"
    fig.savefig(fig_macs_png, dpi=200, bbox_inches="tight")
    fig.savefig(fig_macs_pdf, bbox_inches="tight")
    plt.close(fig)

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            return str(p)

    summary = {
        "pareto_models": frontiers,
        "macs_pareto_models": {first: macs_frontier},
        "dominance": {label: _dominance_text(panels[label], frontiers[label]) for label in labels},
        "profiles": panels,
        "figures": [rel(fig_params_png), rel(fig_params_pdf), rel(fig_macs_png), rel(fig_macs_pdf)],
    }
    summary = sanitize_for_json(summary)  # what is returned is exactly what is written
    summary_path = results_dir / "pareto_efficiency_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(f"[SAVED] Pareto figures -> {output_dir}; summary -> {summary_path}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pareto plots from export_results summaries")
    parser.add_argument("--summary", type=Path, nargs="+", required=True,
                        help="One or more export_results JSON files (one panel each)")
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    labels = args.labels or [p.stem for p in args.summary]
    generate_pareto_plots(dict(zip(labels, args.summary)), output_dir=args.output_dir, results_dir=args.results_dir)
