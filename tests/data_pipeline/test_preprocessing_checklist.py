#!/usr/bin/env python3
"""Validation checklist for the processed dataset (GPT-plan checklist, adapted to the audit).

Run from the repository root AFTER building the processed files:
    python src/data_pipeline/build_processed_dataset.py
    python -m pytest tests/data_pipeline/test_preprocessing_checklist.py -v

Every test corresponds to an integrity item from the shared preprocessing design.
The tests read only data/processed/, data/raw/ label CSVs and the manifest; they never
mutate files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
MANIFEST = REPO_ROOT / "data" / "preprocessing_manifest.json"
RAW = REPO_ROOT / "data" / "raw"

TARGETS = ["QID196", "QID9_3"]
N_PARTICIPANTS = 2058


def _require_processed() -> None:
    if PROCESSED.exists() and any(PROCESSED.glob("*.csv")):
        return
    pytest.skip("no processed files; run build_processed_dataset.py first")


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip("no manifest; run build_processed_dataset.py first")
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def splits() -> pd.DataFrame:
    _require_processed()
    return pd.read_csv(PROCESSED / "participant_splits.csv")


@pytest.fixture(scope="module")
def human() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "human_answers.csv")


@pytest.fixture(scope="module")
def peers() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "peer_responses.csv")


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "xgboost_features.csv", low_memory=False)


def test_01_all_participants_unique_and_accounted(splits: pd.DataFrame):
    assert splits["participant_id"].is_unique and len(splits) == N_PARTICIPANTS
    assert set(splits["split"]) == {"train", "validation", "test"}


def test_02_split_partitions_disjoint(splits: pd.DataFrame):
    train = set(splits.loc[splits["split"] == "train", "participant_id"])
    val = set(splits.loc[splits["split"] == "validation", "participant_id"])
    test = set(splits.loc[splits["split"] == "test", "participant_id"])
    assert not (train & val) and not (train & test) and not (val & test)


def test_03_both_targets_complete_all_2058(human: pd.DataFrame):
    for qid in TARGETS:
        sub = human[human["question_id"] == qid]
        assert len(sub) == N_PARTICIPANTS and sub["participant_id"].is_unique
        assert sub["wave1_3_answer"].notna().all() and sub["wave4_answer"].notna().all()


def test_04_answer_labels_match_official_options(human: pd.DataFrame, manifest: dict):
    maps = manifest["answer_code_maps"]
    for qid in TARGETS:
        sub = human[human["question_id"] == qid]
        official = list(maps[qid])
        assert set(sub["wave4_answer"]) <= set(official)
        assert set(sub["wave1_3_answer"]) <= set(official)


def test_05_qid9_3_is_correct_subitem_and_wording(manifest: dict):
    md = manifest["target_metadata"]["QID9_3"]
    assert md["options"] == ["Yes, I would purchase the product",
                             "No, I would not purchase the product"]
    assert md["distinct_texts_in_wave4"] == 434  # per-participant randomized passage
    assert md["wording_identical_wave1_3_vs_wave4"] == N_PARTICIPANTS


def test_06_persona_profiles_contain_no_target_answers(manifest: dict, splits: pd.DataFrame):
    # build script ran the phrase-level scan on all 2,058 profiles; manifest records 0 hits
    scan = manifest["persona"]["leakage_keyword_scan"]
    assert scan["hits"] == 0
    # and the profile columns in the manifest must not include any Wave 4 target column
    assert set(TARGETS).isdisjoint(manifest["xgboost_features"]["profile_columns"])


def test_07_xgboost_features_have_no_wave4_information(features: pd.DataFrame, manifest: dict):
    # every feature column must exist in the wave1_3 CSV (values are earlier-wave data),
    # and the wave-4-only answer columns must never appear as features
    w13_cols = set(pd.read_csv(RAW / "wave1_3_response_label.csv", nrows=0).columns)
    w4_cols = set(pd.read_csv(RAW / "wave4_response_label.csv", nrows=0).columns) - {"pid"}
    feature_cols = [c for c in features.columns
                    if c not in ("participant_id", "split", "split_seed", "split_version")]
    assert set(feature_cols) <= w13_cols
    # consistency of the manifest bookkeeping: profile + expanded = all wave1_3 answer cols
    assert len(feature_cols) == len(w13_cols) - 1  # wave1_3 = pid + 760 answer columns


def test_08_profile_features_match_persona_information(manifest: dict):
    finfo = manifest["xgboost_features"]
    assert finfo["profile_column_count"] == len(finfo["profile_columns"])
    assert finfo["expanded_extra_column_count"] == len(finfo["expanded_extra_columns"])
    assert finfo["profile_column_count"] + finfo["expanded_extra_column_count"] == 760


def test_09_branch_missingness_preserved_in_features(features: pd.DataFrame, manifest: dict):
    # earlier branch answers (e.g. QID163/QID164 arms) must keep NaN = not-assigned
    branch_cols = [c for c in ("QID163", "QID164_TEXT") if c in features.columns]
    if not branch_cols:
        pytest.skip("no branch columns present (unexpected)")
    for c in branch_cols:
        # real branch columns contain NaN for not-assigned arms AND text for assigned ones
        assert features[c].isna().any(), f"{c}: not-assigned NaNs were imputed"
        assert features[c].notna().any(), f"{c}: no answers retained"


def test_10_peer_groups_exact_2_majority_1_minority(peers: pd.DataFrame):
    g = peers.groupby(["focal_participant_id", "question_id"])
    assert (g.size() == 3).all()
    roles = g["peer_role"].value_counts().unstack(fill_value=0)
    assert (roles["majority"] == 2).all() and (roles["minority"] == 1).all()
    assert peers.groupby(["focal_participant_id", "question_id"])["peer_participant_id"].nunique().eq(3).all()


def test_11_all_peers_in_train_pool_and_not_self(peers: pd.DataFrame, splits: pd.DataFrame):
    train_pids = set(splits.loc[splits["split"] == "train", "participant_id"])
    assert peers["peer_participant_id"].isin(train_pids).all()
    assert (peers["peer_participant_id"] != peers["focal_participant_id"]).all()
    focal = set(peers["focal_participant_id"])
    assert focal <= set(splits.loc[splits["split"] == "test", "participant_id"])


def test_12_same_peer_group_for_conditions_b_and_c(peers: pd.DataFrame, manifest: dict):
    # design: one group per (focal, question); conditions B and C receive the same messages
    assert (peers["reused_for_conditions"] == "B,C").all()
    assert manifest["peer_design"]["note"].startswith("Same peer group")
    # no duplicate group rows that would double-count peers
    assert peers.duplicated(["focal_participant_id", "question_id", "peer_participant_id"]).sum() == 0


def test_13_targets_file_consistent(human: pd.DataFrame, splits: pd.DataFrame):
    split_map = splits.set_index("participant_id")["split"]
    assert (human.set_index("participant_id")["split"] == split_map.reindex(
        human["participant_id"]).values).all()
