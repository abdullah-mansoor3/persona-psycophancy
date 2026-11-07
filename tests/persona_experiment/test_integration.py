"""End-to-end integration test for Member 3."""

from pathlib import Path
import pytest
from src.persona_experiment.data_loader import Member3DataLoader
from src.persona_experiment.experiment_runner import ExperimentRunner
from src.persona_experiment.llm_provider import MockProvider
from src.evaluation.metrics import MetricsCalculator


def test_full_pipeline_5_participants(tmp_path):
    data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    manifest_path = Path(__file__).resolve().parents[2] / "data" / "preprocessing_manifest.json"
    if not data_dir.exists():
        pytest.skip("Processed data directory not found")

    loader = Member3DataLoader(processed_dir=data_dir, manifest_path=manifest_path)
    llm = MockProvider()
    config = {
        "data": {
            "target_questions": ["QID196", "QID9_3"],
            "profile_context_max_chars": 5000,
        }
    }

    runner = ExperimentRunner(data_loader=loader, llm_provider=llm, config=config)
    runs = runner.run_all(
        split="test",
        max_participants=5,
        enable_a=True,
        enable_b=True,
        enable_c=True,
        enable_d=False,
        enable_e=True,
    )

    assert len(runs) > 0
    jsonl_path, csv_path = runner.export_results(output_dir=tmp_path)
    assert jsonl_path.exists()
    assert csv_path.exists()

    import pandas as pd
    df_runs = pd.read_csv(csv_path)
    metrics_b = MetricsCalculator.calculate_condition_metrics(df_runs, "B")
    metrics_c = MetricsCalculator.calculate_condition_metrics(df_runs, "C")
    assert metrics_b.valid_parses > 0
    assert metrics_c.valid_parses > 0

