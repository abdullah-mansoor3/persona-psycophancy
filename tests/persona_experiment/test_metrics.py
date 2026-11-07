"""Unit tests for MetricsCalculator."""

import pandas as pd
from src.evaluation.metrics import MetricsCalculator


def test_calculate_condition_metrics():
    # Construct synthetic test dataframe
    data = [
        {
            "condition": "B",
            "participant_id": 1,
            "question_id": "QID196",
            "parse_valid": True,
            "initial_answer": "the large tray",
            "peer_majority_answer": "the small tray",
            "final_answer": "the small tray",  # Switched to majority!
            "wave4_ground_truth": "the small tray",
            "wave1_3_human_answer": "the small tray",
        },
        {
            "condition": "B",
            "participant_id": 2,
            "question_id": "QID196",
            "parse_valid": True,
            "initial_answer": "the small tray",
            "peer_majority_answer": "the small tray",
            "final_answer": "the small tray",  # Already majority
            "wave4_ground_truth": "the large tray",
            "wave1_3_human_answer": "the small tray",
        },
    ]
    df = pd.DataFrame(data)
    m = MetricsCalculator.calculate_condition_metrics(df, "B")
    assert m.total_instances == 2
    assert m.valid_parses == 2
    assert m.answer_changes == 1
    assert m.answer_change_rate == 0.5
    assert m.initial_minority_count == 1
    assert m.minority_to_majority_switches == 1
    assert m.minority_to_majority_switch_rate == 1.0


def test_compare_b_vs_c():
    data = [
        # In B, switches to majority
        {
            "condition": "B",
            "participant_id": 1,
            "question_id": "QID196",
            "parse_valid": True,
            "initial_answer": "the large tray",
            "peer_majority_answer": "the small tray",
            "final_answer": "the small tray",
            "wave4_ground_truth": "the small tray",
            "wave1_3_human_answer": "the small tray",
        },
        # In C, resists majority!
        {
            "condition": "C",
            "participant_id": 1,
            "question_id": "QID196",
            "parse_valid": True,
            "initial_answer": "the large tray",
            "peer_majority_answer": "the small tray",
            "final_answer": "the large tray",
            "wave4_ground_truth": "the small tray",
            "wave1_3_human_answer": "the small tray",
        },
    ]
    df = pd.DataFrame(data)
    comp = MetricsCalculator.compare_b_vs_c(df)
    assert comp.paired_trials == 1
    assert comp.switches_in_a == 1
    assert comp.switches_in_b == 0
    assert comp.absolute_reduction == 1.0
    assert comp.relative_reduction_pct == 100.0

