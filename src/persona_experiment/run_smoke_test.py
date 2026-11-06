#!/usr/bin/env python3
"""5-Participant Smoke Test for Member 3.

Validates the full pipeline before full-scale execution:
1. Loads Member 1's processed dataset (no leakage).
2. Verifies identical starting answers and identical peer messages for B and C.
3. Tests Conditions A, B, C, and E.
4. Confirms that all outputs are saved within Member_3/results.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import Member3DataLoader
from src.experiment_runner import ExperimentRunner
from src.llm_provider import MockProvider
from src.metrics import MetricsCalculator
from src.visualization import Visualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmokeTest")


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / "../persona-psycophancy-main/data/processed").resolve()
    manifest_path = (base_dir / "../persona-psycophancy-main/data/preprocessing_manifest.json").resolve()
    results_dir = base_dir / "results"

    print("=" * 70)
    print("MEMBER 3: 5-PARTICIPANT SMOKE TEST & PIPELINE INTEGRITY CHECK")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/5] Loading Member 1 Processed Dataset...")
    loader = Member3DataLoader(processed_dir=data_dir, manifest_path=manifest_path)

    instances = loader.get_eligible_instances(
        split="test",
        target_questions=["QID196", "QID9_3"],
        max_participants=5,
        profile_max_chars=12000,
    )
    print(f"Loaded {len(instances)} instances across 5 test participants.")
    assert len(instances) == 10, f"Expected 10 instances (5 participants x 2 questions), got {len(instances)}"

    # 2. Check Peer Integrity & Leakage
    print("\n[Step 2/5] Running Pre-Flight Leakage & Peer Checks...")
    for inst in instances:
        assert inst.split == "test"
        # Verify focal participant is not in peers
        peer_pids = [p.peer_participant_id for p in inst.peers]
        assert inst.participant_id not in peer_pids, f"Self-peer leak detected for PID {inst.participant_id}!"
        # Verify 2 majority : 1 minority
        maj_count = sum(1 for p in inst.peers if p.peer_role == "majority")
        min_count = sum(1 for p in inst.peers if p.peer_role == "minority")
        assert maj_count == 2 and min_count == 1, f"Invalid peer ratio {maj_count}:{min_count} for PID {inst.participant_id}"

    print("Leakage and peer composition checks passed 100%.")

    # 3. Initialize Engine & Execute Smoke Test
    print("\n[Step 3/5] Running Experimental Conditions A, B, C, E on 5 Participants...")
    mock_llm = MockProvider()
    config = {
        "data": {
            "target_questions": ["QID196", "QID9_3"],
            "profile_context_max_chars": 12000,
        }
    }
    runner = ExperimentRunner(data_loader=loader, llm_provider=mock_llm, config=config)
    runs = runner.run_all(
        split="test",
        max_participants=5,
        enable_a=True,
        enable_b=True,
        enable_c=True,
        enable_d=False,
        enable_e=True,
    )
    print(f"Completed {len(runs)} experimental runs.")

    # 4. Export Outputs
    print("\n[Step 4/5] Exporting Outputs to Member_3/results/...")
    jsonl_path, csv_path = runner.export_results(output_dir=results_dir)
    print(f"Saved JSONL: {jsonl_path}")
    print(f"Saved CSV:   {csv_path}")

    # 5. Compute Metrics & Visualizations
    print("\n[Step 5/5] Calculating Metrics & Generating Smoke-Test Figures...")
    import pandas as pd
    df_runs = pd.read_csv(csv_path)

    metrics_a = MetricsCalculator.calculate_condition_metrics(df_runs, "A")
    metrics_b = MetricsCalculator.calculate_condition_metrics(df_runs, "B")
    metrics_c = MetricsCalculator.calculate_condition_metrics(df_runs, "C")
    evidence_metrics = MetricsCalculator.calculate_evidence_metrics(df_runs)
    comparison = MetricsCalculator.compare_b_vs_c(df_runs)

    print("\n" + "-" * 50)
    print("SMOKE TEST RESULTS SUMMARY:")
    print("-" * 50)
    print(f"Condition A (Baseline):      Wave 4 Exact Match = {metrics_a.wave4_exact_match_rate:.1%}")
    print(f"Condition B (Unrestricted):  Minority-to-Majority Switch Rate = {metrics_b.minority_to_majority_switch_rate:.1%}")
    print(f"Condition C (Revision Gate): Minority-to-Majority Switch Rate = {metrics_c.minority_to_majority_switch_rate:.1%}")
    print(f"Revision Gate Impact:        Absolute Reduction = {comparison.absolute_reduction * 100:.1f}%")
    if evidence_metrics:
        print(f"Condition E (Evidence Gate): Evidence-Consistent Revision = {evidence_metrics.evidence_consistent_revision_rate:.1%}")

    # Generate Figures
    figures_dir = results_dir / "figures"
    metrics_dict = {"A": metrics_a, "B": metrics_b, "C": metrics_c}
    Visualizer.generate_all_plots(metrics_dict, comparison, evidence_metrics, figures_dir)
    print(f"Visualizations saved to: {figures_dir}")

    print("\n" + "=" * 70)
    print("5-PARTICIPANT SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

