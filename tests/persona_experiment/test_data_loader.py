"""Unit tests for Member 3 Data Loader."""

from pathlib import Path
import pytest
from src.persona_experiment.data_loader import Member3DataLoader


@pytest.fixture
def loader():
    data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    manifest_path = Path(__file__).resolve().parents[2] / "data" / "preprocessing_manifest.json"
    if not data_dir.exists():
        pytest.skip("Processed data directory not found")
    return Member3DataLoader(processed_dir=data_dir, manifest_path=manifest_path)


def test_data_loader_init(loader):
    assert loader.splits_df is not None
    assert loader.profiles_df is not None
    assert loader.variants_df is not None
    assert loader.peers_df is not None
    assert loader.human_df is not None


def test_get_eligible_instances_count_and_split(loader):
    instances = loader.get_eligible_instances(split="test", max_participants=5)
    assert len(instances) == 10  # 5 participants x 2 questions
    for inst in instances:
        assert inst.split == "test"
        assert inst.question_id in ["QID196", "QID9_3"]
        assert len(inst.options) == 2


def test_no_self_peers(loader):
    instances = loader.get_eligible_instances(split="test", max_participants=10)
    for inst in instances:
        peer_ids = [p.peer_participant_id for p in inst.peers]
        assert inst.participant_id not in peer_ids, f"Self-peer leak detected for {inst.participant_id}"


def test_peer_ratio_two_majority_one_minority(loader):
    instances = loader.get_eligible_instances(split="test", max_participants=10)
    for inst in instances:
        assert len(inst.peers) == 3
        maj = sum(1 for p in inst.peers if p.peer_role == "majority")
        min_role = sum(1 for p in inst.peers if p.peer_role == "minority")
        assert maj == 2
        assert min_role == 1

