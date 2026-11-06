#!/usr/bin/env python3
"""Main execution CLI for Member 3 experiments.

Runs Experiments A, B, C, D (optional), and E across participant cohorts.
Computes evaluation metrics and generates publication figures.
All outputs are stored in Member_3/results/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import Member3DataLoader
from src.experiment_runner import ExperimentRunner
from src.llm_provider import create_llm_provider
from src.metrics import MetricsCalculator
from src.visualization import Visualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExperimentCLI")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Member 3: Persona Conformity & Revision Gate Experiments")
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml", help="Path to config file")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to evaluate: test, validation, or train")
    parser.add_argument("--participants", type=int, default=None, help="Number of participants to evaluate")
    parser.add_argument("--provider", type=str, default=None, help="LLM Provider: mock, transformers, hf_inference, ollama")
    parser.add_argument("--pilot", action="store_true", help="Run 5-participant pilot mode")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = (base_dir / args.config).resolve()
    config = load_yaml_config(config_path)

    # CLI Overrides
    split = args.split or config.get("data", {}).get("split", "test")
    provider_name = args.provider or config.get("model", {}).get("provider", "mock")
    config["model"]["provider"] = provider_name

    if args.pilot:
        n_participants = config.get("execution", {}).get("pilot_participants", 5)
    elif args.participants is not None:
        n_participants = args.participants
    else:
        n_participants = config.get("execution", {}).get("full_participants", 60)

    results_dir = (base_dir / (args.output_dir or config.get("output", {}).get("results_dir", "results"))).resolve()
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    agent_outputs_dir = results_dir / "agent_outputs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    agent_outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MEMBER 3: EXPERIMENT EXECUTION ENGINE")
    print(f"Split: {split} | Participants: {n_participants} | Provider: {provider_name}")
    print(f"Results Directory: {results_dir}")
    print("=" * 80)

    # 1. Initialize Data Loader
    processed_dir = (base_dir / config.get("data", {}).get("processed_dir", "../persona-psycophancy-main/data/processed")).resolve()
    manifest_path = (base_dir / config.get("data", {}).get("manifest_path", "../persona-psycophancy-main/data/preprocessing_manifest.json")).resolve()

    loader = Member3DataLoader(processed_dir=processed_dir, manifest_path=manifest_path)

    # 2. Initialize LLM Provider
    llm = create_llm_provider(config)

    # 3. Initialize Runner and Execute Conditions
    runner = ExperimentRunner(data_loader=loader, llm_provider=llm, config=config)

    exec_cfg = config.get("execution", {})
    runs = runner.run_all(
        split=split,
        max_participants=n_participants,
        enable_a=exec_cfg.get("enable_experiment_a", True),
        enable_b=exec_cfg.get("enable_experiment_b", True),
        enable_c=exec_cfg.get("enable_experiment_c", True),
        enable_d=exec_cfg.get("enable_experiment_d", False),
        enable_e=exec_cfg.get("enable_experiment_e", True),
    )

    # 4. Export Raw Runs
    jsonl_path, csv_path = runner.export_results(output_dir=results_dir)
    print(f"\n[Artifact] Saved run records to: {jsonl_path}")
    print(f"[Artifact] Saved tabular data to: {csv_path}")

    # 5. Calculate Metrics
    df_runs = pd.read_csv(csv_path)

    metrics_dict: Dict[str, Any] = {}
    metrics_summary_rows = []

    for cond in ["A", "B", "C"]:
        if (df_runs["condition"] == cond).any():
            m = MetricsCalculator.calculate_condition_metrics(df_runs, cond)
            metrics_dict[cond] = m
            row = asdict(m)
            metrics_summary_rows.append(row)

    # Experiment E (Evidence)
    evidence_metrics = MetricsCalculator.calculate_evidence_metrics(df_runs)
    if evidence_metrics:
        with open(metrics_dir / "evidence_metrics.json", "w", encoding="utf-8") as f:
            json.dump(asdict(evidence_metrics), f, indent=2)

    # Statistical Comparison (B vs C)
    comparison = MetricsCalculator.compare_b_vs_c(df_runs)
    with open(metrics_dir / "statistical_comparison_b_vs_c.json", "w", encoding="utf-8") as f:
        json.dump(asdict(comparison), f, indent=2)

    # Export Metrics CSV & JSON
    df_metrics = pd.DataFrame(metrics_summary_rows)
    df_metrics.to_csv(metrics_dir / "condition_metrics_table.csv", index=False)
    with open(metrics_dir / "condition_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary_rows, f, indent=2)

    print(f"[Artifact] Saved metrics to: {metrics_dir / 'condition_metrics_table.csv'}")

    # 6. Generate Figures
    saved_plots = Visualizer.generate_all_plots(metrics_dict, comparison, evidence_metrics, figures_dir)
    print(f"[Artifact] Generated {len(saved_plots)} publication figures in: {figures_dir}")

    # 7. Print Console Executive Summary
    print("\n" + "=" * 80)
    print("EXPERIMENTAL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Condition':<25} | {'Instances':<10} | {'Change Rate':<12} | {'Switch Rate':<12} | {'Wave4 Exact Match':<18}")
    print("-" * 80)
    for row in metrics_summary_rows:
        cond_name = {
            "A": "A: Independent Baseline",
            "B": "B: Unrestricted Peer",
            "C": "C: Revision Gate (Ours)",
        }.get(row["condition"], row["condition"])
        print(
            f"{cond_name:<25} | "
            f"{row['valid_parses']:<10} | "
            f"{row['answer_change_rate']:<12.1%} | "
            f"{row['minority_to_majority_switch_rate']:<12.1%} | "
            f"{row['wave4_exact_match_rate']:<18.1%}"
        )

    print("-" * 80)
    print("STATISTICAL COMPARISON (Condition B vs Condition C):")
    print(f"  Minority Switched in B (Unrestricted): {comparison.switches_in_a} / {comparison.paired_trials} ({comparison.switch_rate_a:.1%})")
    print(f"  Minority Switched in C (Gated):        {comparison.switches_in_b} / {comparison.paired_trials} ({comparison.switch_rate_b:.1%})")
    print(f"  Absolute Conformity Reduction:         {comparison.absolute_reduction * 100:.1f}%")
    print(f"  Relative Conformity Reduction:         {comparison.relative_reduction_pct:.1f}%")
    print(f"  Hypothesis Test:                       {comparison.test_name} (p = {comparison.p_value:.4f})")

    if evidence_metrics and evidence_metrics.initial_non_evidence_choices > 0:
        print("\nEVIDENCE-POSITIVE TEST (Condition E):")
        print(f"  Initial Non-Evidence Choices:          {evidence_metrics.initial_non_evidence_choices}")
        print(f"  Evidence-Consistent Revisions:         {evidence_metrics.evidence_consistent_revisions} ({evidence_metrics.evidence_consistent_revision_rate:.1%})")
        print(f"  Evidence Rejections:                   {evidence_metrics.evidence_rejections} ({evidence_metrics.evidence_rejection_rate:.1%})")
        print("  -> Confirms gate is NOT stubborn: successfully revises when valid proof is supplied.")

    print("=" * 80)
    print("ALL RUNS AND EVALUATIONS COMPLETED SUCCESSFULLY.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

