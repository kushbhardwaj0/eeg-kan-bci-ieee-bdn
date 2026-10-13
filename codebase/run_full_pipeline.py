"""
End-to-End Benchmark Orchestrator executing the complete long run according to run_instructions.md & RERUN.md.

Steps:
1. Full BNCI2014_001 Multi-Seed LOSO Benchmark (7 models x 5 seeds x 9 subjects)
2. PhysionetMI Scaling LOSO Benchmark (20 subjects x 2 models x 5 seeds)
3. Results Export, Holm/FDR Statistical Corrections, LaTeX Tables, Combined Pareto Curves
4. Neuroscientific Interpretability Suite on the first trained CNN_KAN fold model of step 1
   (learned B-spline edge functions, spatial filter weights and Haufe spatial patterns)
"""

import sys
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_experiments import run_full_phase3_pipeline
from run_physionet_scaling import run_physionet_scaling_benchmark
from export_results import export_run
from src.utils.visualize import run_interpretability_suite
from src.utils.pareto import generate_pareto_plots


def main():
    start_time = time.time()
    results_dir = PROJECT_ROOT / "results"
    phase3_dir = results_dir / "phase3"
    physionet_dir = results_dir / "physionet"
    figures_dir = results_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    phase3_dir.mkdir(parents=True, exist_ok=True)
    physionet_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING FULL END-TO-END LONG BENCHMARK EXECUTION")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 80)

    # ---------------------------------------------------------
    # STEP 1: Full BNCI2014_001 Multi-Seed LOSO Benchmark
    # ---------------------------------------------------------
    print("\n[STEP 1/4] Running Full BNCI2014_001 Multi-Seed Benchmark (run_experiments.py)...")
    phase3_scores = run_full_phase3_pipeline(
        config_path=PROJECT_ROOT / "configs" / "bnci2014_001.yaml",
        output_dir=phase3_dir,
        mode="full",
    )

    # Mirror artifacts to codebase/results/ for run_instructions.md path compatibility
    if (phase3_dir / "scores.csv").exists():
        shutil.copy2(phase3_dir / "scores.csv", results_dir / "phase_3_kan_benchmark_results.csv")
    if (phase3_dir / "phase_3_kan_benchmark_results.json").exists():
        shutil.copy2(phase3_dir / "phase_3_kan_benchmark_results.json", results_dir / "phase_3_kan_benchmark_results.json")
    if (phase3_dir / "phase_3_table_snippet.tex").exists():
        shutil.copy2(phase3_dir / "phase_3_table_snippet.tex", results_dir / "phase_3_table_snippet.tex")

    print("[STEP 1/4 COMPLETED] BNCI2014_001 benchmark finished.")

    # ---------------------------------------------------------
    # STEP 2: PhysionetMI 20-Subject Transfer Scaling Benchmark
    # ---------------------------------------------------------
    print("\n[STEP 2/4] Running PhysionetMI 20-Subject Scaling Benchmark (run_physionet_scaling.py)...")
    physionet_scores = run_physionet_scaling_benchmark(
        config_path=PROJECT_ROOT / "configs" / "physionet.yaml",
        output_dir=physionet_dir,
        num_subjects=20,
    )

    if (physionet_dir / "phase_4_physionet_scaling_results.csv").exists():
        shutil.copy2(physionet_dir / "phase_4_physionet_scaling_results.csv", results_dir / "phase_4_physionet_scaling_results.csv")
    if (physionet_dir / "phase_4_physionet_scaling_results.json").exists():
        shutil.copy2(physionet_dir / "phase_4_physionet_scaling_results.json", results_dir / "phase_4_physionet_scaling_results.json")

    print("[STEP 2/4 COMPLETED] PhysionetMI scaling benchmark finished.")

    # ---------------------------------------------------------
    # STEP 3: Results Export, Stats, Tables & Combined Pareto Curves
    # ---------------------------------------------------------
    print("\n[STEP 3/4] Exporting Summaries, Statistical Tests, and Combined Pareto Curves...")
    phase3_res = export_run(phase3_dir, target_model="cnn_kan", pareto=True, panel_label="BNCI2014-001")
    
    # Export Physionet summary and combined Pareto plots
    if (physionet_dir / "scores.csv").exists():
        try:
            phys_res = export_run(physionet_dir, target_model="cnn_kan", pareto=False, panel_label="PhysionetMI")
            generate_pareto_plots(
                {"BNCI2014-001": phase3_res, "PhysionetMI": phys_res},
                output_dir=figures_dir,
                results_dir=results_dir,
            )
        except Exception as e:
            print(f"[WARNING] Combined Pareto generation encountered: {e}")

    # Mirror pareto figures to results/figures
    if (phase3_dir / "figures").exists():
        for fig_file in (phase3_dir / "figures").glob("*.*"):
            shutil.copy2(fig_file, figures_dir / fig_file.name)
    if (phase3_dir / "pareto_efficiency_summary.json").exists():
        shutil.copy2(phase3_dir / "pareto_efficiency_summary.json", results_dir / "pareto_efficiency_summary.json")

    print("[STEP 3/4 COMPLETED] Results exported.")

    # ---------------------------------------------------------
    # STEP 4: Neuroscientific Interpretability Visualizations
    # ---------------------------------------------------------
    print("\n[STEP 4/4] Generating Neuroscientific Interpretability Visualizations...")
    run_interpretability_suite(run_dir=phase3_dir, output_dir=figures_dir)
    print("[STEP 4/4 COMPLETED] Interpretability figures generated.")

    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("\n" + "=" * 80)
    print(f"BENCHMARK PIPELINE EXECUTION COMPLETED IN {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
