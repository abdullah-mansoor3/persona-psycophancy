"""Unit tests for Member 3 Prompt Builder."""

from pathlib import Path
import pytest
from src.data_loader import Member3DataLoader
from src.prompt_builder import PromptBuilder


@pytest.fixture
def instance():
    data_dir = Path(__file__).resolve().parents[2] / "persona-psycophancy-main" / "data" / "processed"
    manifest_path = Path(__file__).resolve().parents[2] / "persona-psycophancy-main" / "data" / "preprocessing_manifest.json"
    if not data_dir.exists():
        pytest.skip("Processed data directory not found")
    loader = Member3DataLoader(processed_dir=data_dir, manifest_path=manifest_path)
    instances = loader.get_eligible_instances(split="test", max_participants=1)
    return instances[0]


def test_experiment_a_prompt_no_wave4_leakage(instance):
    prompt_a = PromptBuilder.build_experiment_a_prompt(instance)
    assert instance.question_text in prompt_a
    assert "Participant Profile:" in prompt_a
    assert "Available Options:" in prompt_a
    # Never leak ground truth
    assert f"wave4_answer: {instance.wave4_ground_truth}" not in prompt_a


def test_experiment_b_and_c_identical_peer_messages(instance):
    init_ans = instance.options[0]
    init_rat = "Test rationale"
    prompt_b = PromptBuilder.build_experiment_b_prompt(instance, init_ans, init_rat)
    prompt_c = PromptBuilder.build_experiment_c_prompt(instance, init_ans, init_rat)

    # Extract peer block from both
    for p in instance.peers:
        peer_line = f"- Participant {p.peer_position}: \"{p.peer_answer}\""
        assert peer_line in prompt_b
        assert peer_line in prompt_c


def test_experiment_c_contains_gate_checklist(instance):
    prompt_c = PromptBuilder.build_experiment_c_prompt(instance, instance.options[0], "Rationale")
    assert "Profile-Anchored Revision Gate Checklist:" in prompt_c
    assert "peer_information_type" in prompt_c
    assert "contains_valid_new_evidence" in prompt_c
    assert "revision_justified" in prompt_c

