"""
Central MOABB Benchmark Entrypoint for EEG Motor Imagery Decoders

Runs one model on one dataset under leave-one-subject-out evaluation with the shared protocol
in ``src/utils/experiment.py`` (explicit events, inferred input shape, validation split,
N seeds, per-run MOABB store).
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import get_model_flops, get_parameter_count
from src.utils.experiment import (
    NEURAL_MODELS,
    build_pipeline,
    canonical_dataset_name,
    infer_shape,
    load_config,
    make_dataset,
    make_paradigm,
    parse_model_name,
    hyperparameters_from_config,
    resolve_overwrite,
    run_cross_subject,
    summarize_scores,
    write_run_info,
)

try:
    # pyrefly: ignore [missing-import]
    import wandb
except ImportError:
    wandb = None

CONFIGS = {
    "bnci2014_001": PROJECT_ROOT / "configs" / "bnci2014_001.yaml",
    "physionet": PROJECT_ROOT / "configs" / "physionet.yaml",
}


def run_benchmark(
    model_name: str = "cnn_kan",
    dataset_name: str = "bnci2014_001",
    seed: int = 42,
    use_wandb: bool = False,
    seeds: Optional[Iterable[int]] = None,
    subjects: Optional[Iterable[int]] = None,
    max_epochs: Optional[int] = None,
    device: Optional[str] = None,
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    overwrite: Optional[bool] = None,
) -> pd.DataFrame:
    dataset_key = canonical_dataset_name(dataset_name)
    cfg = load_config(config_path or CONFIGS[dataset_key])
    dcfg, ecfg = cfg["dataset"], cfg["experiment"]
    overwrite = resolve_overwrite(cfg) if overwrite is None else overwrite

    print("=" * 65)
    print(f"  EEG-KAN Benchmark Execution: Model={model_name.upper()} | Dataset={dataset_key.upper()}")
    print("=" * 65)

    if subjects is None:
        subjects = dcfg.get("subjects")
        if subjects is None and dataset_key == "physionet":
            subjects = list(range(1, int(dcfg.get("subset_subjects", 20)) + 1))
    dataset = make_dataset(dataset_key, subjects=subjects)
    paradigm = make_paradigm(
        dataset_key,
        fmin=dcfg.get("bandpass", {}).get("low", 8.0),
        fmax=dcfg.get("bandpass", {}).get("high", 30.0),
        tmin=dcfg.get("tmin", 0.0),
        tmax=dcfg.get("tmax"),
        resample=dcfg.get("resample"),
        events=dcfg.get("events"),
    )
    shape = infer_shape(paradigm, dataset)
    seeds = [int(s) for s in (seeds if seeds is not None else [seed])]
    hp = hyperparameters_from_config(cfg, max_epochs=max_epochs, device=device)
    estimator = build_pipeline(model_name, shape["n_chans"], shape["n_times"], shape["n_classes"], **hp)

    run_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "results" / f"{model_name}_{dataset_key}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if use_wandb and wandb is not None:
        wandb.init(project="eeg-kan-decoding", name=f"{model_name}_{dataset_key}",
                   config={"model": model_name, "dataset": dataset_key, "seeds": seeds, **hp})

    # Profile neural models on the real input shape (no 22x1001 hardcoding), on a throwaway
    # copy so the estimator handed to MOABB stays unfitted.
    profile = {}
    if parse_model_name(model_name)[0] in NEURAL_MODELS + ("kan",):
        probe = build_pipeline(model_name, shape["n_chans"], shape["n_times"], shape["n_classes"], **{**hp, "device": "cpu"})
        net = probe._build_net()
        net.initialize()
        params = get_parameter_count(net.module_)
        flops = get_model_flops(net.module_, input_size=(1, shape["n_chans"], shape["n_times"]))
        profile = {"trainable_params": params["trainable_params"], "macs": flops["macs"]}
        print(f"[MODEL PROFILE] Trainable Parameters: {params['trainable_params']:,}")
        print(f"[MODEL PROFILE] Complexity MACs: {flops['macs']}")
        if use_wandb and wandb is not None:
            wandb.log(profile)

    write_run_info(run_dir, dataset=dataset.code, subjects=list(dataset.subject_list), shape=shape,
                   seeds=seeds, hyperparameters=hp, models=[model_name], profile=profile, config=cfg)

    print(f"[MOABB] Running CrossSubjectEvaluation for pipeline '{model_name}' over seeds {seeds}...")
    results_df = run_cross_subject(dataset, paradigm, {model_name: estimator}, seeds, run_dir,
                                   overwrite=overwrite, save_model=ecfg.get("save_model", True), suffix="benchmark_")

    summary = summarize_scores(results_df)
    if not summary.empty:
        row = summary.iloc[0]
        mean_acc, std_acc = float(row["mean_acc"]), float(row["std_subjects"]) if pd.notna(row["std_subjects"]) else 0.0
    else:
        mean_acc, std_acc = 0.0, 0.0

    print("\n" + "=" * 65)
    print(f"  FINAL BENCHMARK RESULT: {model_name.upper()}  (per-subject mean over both sessions and {len(seeds)} seed(s))")
    print(f"  Mean Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    print("=" * 65 + "\n")
    print(f"[SAVED] Detailed fold results written to: {run_dir / 'scores.csv'}")

    if use_wandb and wandb is not None:
        wandb.log({"mean_accuracy": mean_acc, "std_accuracy": std_acc})
        wandb.finish()

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MOABB LOSO Benchmark for EEG decoders.")
    parser.add_argument("--model", type=str, default="cnn_kan",
                        help="csp_lda | svm | eegnet | cnn_kan | cnn_kan_G5 | cnn_linear | cnn_mlp")
    parser.add_argument("--dataset", type=str, default="bnci2014_001", choices=["bnci2014_001", "physionet"])
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/<dataset>.yaml)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds (default: --seed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subject subset for smoke runs")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda | mps | cpu | auto")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        dataset_name=args.dataset,
        seed=args.seed,
        use_wandb=args.wandb,
        seeds=args.seeds,
        subjects=args.subjects,
        max_epochs=args.max_epochs,
        device=args.device,
        config_path=args.config,
        output_dir=args.run_dir,
        overwrite=False if args.no_overwrite else None,
    )
