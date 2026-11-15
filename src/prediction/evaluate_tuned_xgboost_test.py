#!/usr/bin/env python3
"""One-time held-out test assessment for the frozen profile-only tuning run.

This script does not fit or select any model settings. Do not use its output to
choose another configuration; that would turn the test split into validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "preprocessing_manifest.json"
TUNING = ROOT / "outputs" / "xgboost" / "tuning_profile"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    metadata = manifest["xgboost_features"]
    role_map: dict[str, str] = {}
    for group in ("roles_profile", "roles_expanded_extra"):
        for role, columns in metadata[group].items():
            role_map.update({column: role for column in columns})
    features = pd.read_csv(DATA / "xgboost_features.csv", low_memory=False)
    targets = pd.read_csv(DATA / "xgboost_targets.csv", low_memory=False)
    result = {
        "purpose": "Final held-out test assessment of frozen training-CV selections; no further tuning",
        "test_used_for_tuning": False,
        "split_version": manifest["split_version"],
        "questions": {},
    }

    for qid in manifest["targets"]:
        bundle = joblib.load(TUNING / "models" / f"xgb_{qid.lower()}_profile_tuned.joblib")
        columns = bundle["feature_columns"]
        block_key = "QID196_model" if qid == "QID196" else "QID9_3_model_strict"
        excluded = set(metadata["model_feature_exclusions"][block_key])
        meta_cols = {"participant_id", "split", "split_seed", "split_version"}
        expected = [c for c in features.columns if c not in meta_cols and c not in excluded
                    and role_map.get(c) != "free_text"]
        if columns != expected:
            raise ValueError(f"Feature schema changed since tuning for {qid}")
        data = targets.loc[targets.question_id == qid].merge(
            features[["participant_id", "split", *columns]],
            on=["participant_id", "split"], how="inner", validate="one_to_one",
        )
        test = data.loc[data.split == "test"].copy()
        y_test = test.wave4_answer_code.astype(int).to_numpy() - 1
        numeric = [c for c in columns if role_map[c] == "numeric"]
        categorical = [c for c in columns if role_map[c] != "numeric"]
        for col in numeric:
            test[col] = pd.to_numeric(test[col], errors="coerce")
        for col in categorical:
            test[col] = test[col].where(test[col].notna(), "__NOT_ASSIGNED__").astype(str)
        prediction = bundle["pipeline"].predict(test[columns])
        prior = test.wave1_3_answer_code.astype(int).to_numpy() - 1
        train_labels = data.loc[data.split == "train", "wave4_answer_code"].astype(int) - 1
        majority = np.full(len(test), int(train_labels.value_counts().idxmax()))
        result["questions"][qid] = {
            "n_test": int(len(test)),
            "tuned_profile_xgboost": metrics(y_test, prediction),
            "prior_answer_persistence": metrics(y_test, prior),
            "majority_class": metrics(y_test, majority),
            "selected_parameters": bundle["selected_parameters"],
        }

    output = TUNING / "test_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved frozen-model test results to {output}")
    for qid, values in result["questions"].items():
        print(qid, "test accuracy=", round(values["tuned_profile_xgboost"]["accuracy"], 3),
              "persistence=", round(values["prior_answer_persistence"]["accuracy"], 3))


if __name__ == "__main__":
    main()
