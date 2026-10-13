"""
Phase 3 Experiment Orchestrator: Spline Grid Sweep, Model Ablations, Statistical Tests, and Table Export.

Protocol (issue #3, PR 3): explicit events, shape inferred from data, stratified validation split
with early stopping, N seeds, both held-out sessions scored, per-seed MOABB store under the run
directory (never the global cache). Output: ``<run_dir>/scores.csv`` + ``run_info.json`` + the
summary JSON / LaTeX snippet.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.experiment import (
    build_pipelines,
    infer_shape,
    load_config,
    make_dataset,
    make_paradigm,
    hyperparameters_from_config,
    resolve_overwrite,
    run_cross_subject,
    write_run_info,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bnci2014_001.yaml"


def format_latex_results_table(summary_dict, **kwargs):  # re-exported for backward compatibility
    from export_results import format_latex_results_table as _fmt
    return _fmt(summary_dict, **kwargs)


# --------------------------------------------------------------------------- setup
def model_names_for_mode(cfg: dict, mode: str) -> List[str]:
    """
    ``full``  : config models + the grid sweep (main grid size is not fitted twice).
    ``sweep`` : ``cnn_kan`` (main grid) plus the other grid sizes in ``experiment.grid_sweep``.
    ``ablations`` : config models only.
    """
    ecfg, kcfg = cfg["experiment"], cfg["kan"]
    base_models = list(ecfg.get("models", ["cnn_kan", "cnn_linear", "cnn_mlp", "eegnet", "csp_lda"]))
    main_grid = int(kcfg.get("grid_size", 3))
    sweep = [f"cnn_kan_G{g}" for g in ecfg.get("grid_sweep", []) if int(g) != main_grid]
    if mode == "sweep":
        return ["cnn_kan"] + sweep  # same name as ablations/full so merged runs never hold the config twice
    if mode == "ablations":
        return base_models
    return base_models + sweep


# --------------------------------------------------------------------------- main entry
def run_full_phase3_pipeline(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Optional[Path] = None,
    seeds: Optional[Iterable[int]] = None,
    subjects: Optional[Iterable[int]] = None,
    max_epochs: Optional[int] = None,
    device: Optional[str] = None,
    mode: str = "full",
    overwrite: Optional[bool] = None,
    target_model: str = "cnn_kan",
) -> pd.DataFrame:
    """
    Run the whole Phase 3 suite (sweep + ablations + baselines) and export summary artifacts.
    Returns the fold-level scores (all seeds, both sessions).
    """
    cfg = load_config(config_path)
    dcfg, ecfg = cfg["dataset"], cfg["experiment"]

    run_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "results" / "phase3"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(dcfg.get("name", "bnci2014_001"), subjects=subjects or dcfg.get("subjects"))
    paradigm = make_paradigm(
        dcfg.get("name", "bnci2014_001"),
        fmin=dcfg.get("bandpass", {}).get("low", 8.0),
        fmax=dcfg.get("bandpass", {}).get("high", 30.0),
        tmin=dcfg.get("tmin", 0.0),
        tmax=dcfg.get("tmax"),
        resample=dcfg.get("resample"),
        events=dcfg.get("events"),
    )
    shape = infer_shape(paradigm, dataset)

    seeds = [int(s) for s in (seeds if seeds is not None else ecfg.get("seeds", [42]))]
    overwrite = resolve_overwrite(cfg) if overwrite is None else overwrite
    hp = hyperparameters_from_config(cfg, max_epochs=max_epochs, device=device)
    names = model_names_for_mode(cfg, mode)
    pipelines = build_pipelines(names, shape, **hp)

    write_run_info(run_dir, dataset=dataset.code, subjects=list(dataset.subject_list), shape=shape,
                   seeds=seeds, hyperparameters=hp, models=names, mode=mode, config=cfg)

    print(f"[PHASE 3] {dataset.code}: {len(dataset.subject_list)} subjects, "
          f"{shape['n_chans']} ch x {shape['n_times']} samples, classes={shape['classes']}")
    print(f"[PHASE 3] models={names} seeds={seeds} max_epochs={hp['max_epochs']} device={hp['device'] or 'auto'}")

    scores = run_cross_subject(dataset, paradigm, pipelines, seeds, run_dir,
                               overwrite=overwrite, save_model=ecfg.get("save_model", True), suffix="phase3_")
    if scores.empty:
        print("[PHASE 3] No scores produced.")
        return scores
    print(f"[SAVED] fold scores -> {run_dir / 'scores.csv'}")

    export_phase3_summary(run_dir, target_model=target_model)
    return scores


def export_phase3_summary(run_dir, target_model: str = "cnn_kan"):
    """Summary JSON + LaTeX + Pareto figures from ``run_dir/scores.csv`` (see export_results.py)."""
    from export_results import export_run
    return export_run(run_dir, target_model=target_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 Experiment Orchestrator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", type=str, default="full", choices=["sweep", "ablations", "full"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds (default: config)")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subject subset for smoke runs")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda | mps | cpu | auto")
    parser.add_argument("--run-dir", type=Path, default=None, help="Output directory (default: results/phase3)")
    parser.add_argument("--no-overwrite", action="store_true", help="Serve MOABB cached scores if present (not recommended)")
    args = parser.parse_args()

    run_full_phase3_pipeline(
        config_path=args.config,
        output_dir=args.run_dir,
        seeds=args.seeds,
        subjects=args.subjects,
        max_epochs=args.max_epochs,
        device=args.device,
        mode=args.mode,
        overwrite=False if args.no_overwrite else None,
    )
