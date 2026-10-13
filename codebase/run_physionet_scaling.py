"""
PhysionetMI Scaling Benchmark Runner (Phase 4).
Evaluates CNN_KAN vs EEGNet on the 64-channel PhysionetMI dataset under CrossSubjectEvaluation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.pareto import sanitize_for_json
from src.utils.experiment import (
    build_pipelines,
    infer_shape,
    load_config,
    make_dataset,
    make_paradigm,
    hyperparameters_from_config,
    resolve_overwrite,
    run_cross_subject,
    summarize_scores,
    write_run_info,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "physionet.yaml"


def _resolved_device(pipelines) -> Optional[str]:
    first = next(iter(pipelines.values()), None)
    return first._resolved_device() if hasattr(first, "_resolved_device") else None


def run_physionet_scaling_benchmark(
    num_subjects: int = 20,
    seed: int = 42,
    output_dir: Path = None,
    overwrite: Optional[bool] = None,
    seeds: Optional[Iterable[int]] = None,
    subjects: Optional[Iterable[int]] = None,
    max_epochs: Optional[int] = None,
    device: Optional[str] = None,
    config_path: Path = DEFAULT_CONFIG,
    models: Optional[Iterable[str]] = None,
    hdf5_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Leave-one-subject-out benchmark on a subset of PhysionetMI (64 channels).

    ``seeds`` overrides the single ``seed`` (kept for backward compatibility). MOABB's result
    store lives under ``output_dir`` (or ``hdf5_path``), never in the global ``~/mne_data``.
    """
    cfg = load_config(config_path)
    dcfg, ecfg = cfg["dataset"], cfg["experiment"]
    overwrite = resolve_overwrite(cfg) if overwrite is None else overwrite

    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "results" / "physionet"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(hdf5_path) if hdf5_path is not None else output_dir

    if not subjects:
        subjects = dcfg.get("subjects") or list(range(1, int(num_subjects) + 1))
    dataset = make_dataset("physionet", subjects=subjects)
    paradigm = make_paradigm(
        "physionet",
        fmin=dcfg.get("bandpass", {}).get("low", 8.0),
        fmax=dcfg.get("bandpass", {}).get("high", 30.0),
        tmin=dcfg.get("tmin", 0.0),
        tmax=dcfg.get("tmax"),
        resample=dcfg.get("resample"),
        events=dcfg.get("events"),
    )
    shape = infer_shape(paradigm, dataset)

    seeds = list(seeds) if seeds is not None else [int(seed)]
    hp = hyperparameters_from_config(cfg, max_epochs=max_epochs, device=device)
    model_names = list(models) if models is not None else list(ecfg.get("models", ["cnn_kan", "eegnet"]))
    if not model_names:
        raise ValueError("no models to run (config experiment.models / --models is empty)")
    pipelines = build_pipelines(model_names, shape, **hp)

    write_run_info(
        run_dir,
        dataset=dataset.code,
        subjects=list(dataset.subject_list),
        shape=shape,
        seeds=seeds,
        hyperparameters=hp,
        models=model_names,
        config=cfg,
        device_resolved=_resolved_device(pipelines),
    )

    print(f"[PHYSIONET SCALING] {len(dataset.subject_list)} subjects, {shape['n_chans']} ch x {shape['n_times']} samples, "
          f"classes={shape['classes']}, seeds={seeds}, models={model_names}")
    scores = run_cross_subject(
        dataset, paradigm, pipelines, seeds, run_dir,
        overwrite=overwrite, save_model=ecfg.get("save_model", True), suffix="physionet_",
    )

    csv_path = output_dir / "phase_4_physionet_scaling_results.csv"
    scores.to_csv(csv_path, index=False)
    print(f"[SAVED] Physionet scaling CSV saved to: {csv_path}")

    summary = summarize_scores(scores)
    json_path = output_dir / "phase_4_physionet_scaling_results.json"
    with open(json_path, "w") as f:
        json.dump(sanitize_for_json(summary.set_index("model").to_dict(orient="index")) if not summary.empty else {}, f, indent=2, allow_nan=False)
    print(f"[SAVED] Physionet scaling JSON saved to: {json_path}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PhysionetMI multi-subject scaling benchmark")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="Explicit subject ids (e.g. --subjects 1 2). Default: first --num-subjects.")
    parser.add_argument("--num-subjects", type=int, default=None, help="Size of the subject subset (default: config)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to run (default: config)")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda | mps | cpu | auto")
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--run-dir", type=Path, default=None, help="Output directory (default: results/physionet)")
    parser.add_argument("--no-overwrite", action="store_true", help="Serve MOABB cached scores if present (not recommended)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_physionet_scaling_benchmark(
        num_subjects=args.num_subjects or cfg["dataset"].get("subset_subjects", 20),
        seeds=args.seeds or cfg["experiment"].get("seeds", [42]),
        subjects=args.subjects,
        output_dir=args.run_dir,
        overwrite=False if args.no_overwrite else None,
        max_epochs=args.max_epochs,
        device=args.device,
        config_path=args.config,
        models=args.models,
    )
