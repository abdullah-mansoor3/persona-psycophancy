#!/usr/bin/env python3
"""Compare a few model families for the two Wave-4 targets.

Search and model-family selection use training CV and validation only. Results
with the previous same-question answer are labeled separately because the
primary project model is profile-only. No test rows are loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "preprocessing_manifest.json"
OUT = ROOT / "outputs" / "xgboost" / "alternative_models"
SEED = 42


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    metadata = manifest["xgboost_features"]
    role_map = {}
    for group in ("roles_profile", "roles_expanded_extra"):
        for role, columns in metadata[group].items():
            role_map.update({column: role for column in columns})
    features = pd.read_csv(DATA / "xgboost_features.csv", low_memory=False)
    targets = pd.read_csv(DATA / "xgboost_targets.csv", low_memory=False)
    features = features.loc[features.split.isin(["train", "validation"])]
    targets = targets.loc[targets.split.isin(["train", "validation"])]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    results = {
        "selection": "four-fold stratified CV on train; validation used for model comparison",
        "test_split_used": False,
        "prior_answer_variant_is_separate_from_profile_only": True,
        "seed": SEED,
        "scikit_learn": sklearn.__version__,
        "questions": {},
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)

    for qid in manifest["targets"]:
        block_key = "QID196_model" if qid == "QID196" else "QID9_3_model_strict"
        excluded = set(metadata["model_feature_exclusions"][block_key])
        meta_cols = {"participant_id", "split", "split_seed", "split_version"}
        profile_cols = [c for c in features.columns if c not in meta_cols
                        and c not in excluded and role_map.get(c) != "free_text"]
        data = targets.loc[targets.question_id == qid].merge(
            features[["participant_id", "split", *profile_cols]],
            on=["participant_id", "split"], how="inner", validate="one_to_one",
        )
        data["y"] = data.wave4_answer_code.astype(int) - 1
        data["prior_target_answer"] = data.wave1_3_answer.astype(str).fillna("__NOT_ASSIGNED__")
        numeric = [c for c in profile_cols if role_map[c] == "numeric"]
        categorical = [c for c in profile_cols if role_map[c] != "numeric"]
        for col in numeric:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        for col in categorical:
            data[col] = data[col].where(data[col].notna(), "__NOT_ASSIGNED__").astype(str)
        train = data.loc[data.split == "train"]
        valid = data.loc[data.split == "validation"]
        y_train, y_valid = train.y.to_numpy(), valid.y.to_numpy()

        for variant, cols in (
            ("profile_only", profile_cols),
            ("profile_plus_prior_answer", [*profile_cols, "prior_target_answer"]),
        ):
            categorical_cols = categorical + (["prior_target_answer"] if variant.endswith("prior_answer") else [])
            prep = ColumnTransformer([
                ("numeric", SimpleImputer(strategy="median"), numeric),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="__NOT_ASSIGNED__")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), categorical_cols),
            ], remainder="drop")
            candidates = [
                ("logistic_regression", LogisticRegression(
                    solver="liblinear", max_iter=2000, random_state=SEED),
                 {"classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0],
                  "classifier__class_weight": [None, "balanced"]}),
                ("extra_trees", ExtraTreesClassifier(
                    n_estimators=100, random_state=SEED, n_jobs=4),
                 {"classifier__max_features": ["sqrt"],
                  "classifier__min_samples_leaf": [2, 10],
                  "classifier__class_weight": [None, "balanced"]}),
            ]
            for name, classifier, grid in candidates:
                pipe = Pipeline([("preprocessor", prep), ("classifier", classifier)])
                search = GridSearchCV(
                    pipe, grid, scoring="accuracy", cv=cv, n_jobs=1,
                    refit=True, return_train_score=False,
                )
                print(f"CV search: {qid} / {variant} / {name}", flush=True)
                search.fit(train[cols], y_train)
                val_pred = search.best_estimator_.predict(valid[cols])
                key = f"{qid}/{variant}/{name}"
                results["questions"][key] = {
                    "n_train": int(len(train)), "n_validation": int(len(valid)),
                    "cv_accuracy": float(search.best_score_),
                    "cv_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                    "best_parameters": search.best_params_,
                    "validation": metrics(y_valid, val_pred),
                }
                joblib.dump({"pipeline": search.best_estimator_, "feature_columns": cols,
                             "best_parameters": search.best_params_},
                            OUT / "models" / f"{qid.lower()}_{variant}_{name}.joblib")
                print(key, "CV=", round(search.best_score_, 3),
                      "validation=", round(results["questions"][key]["validation"]["accuracy"], 3),
                      flush=True)

    path = OUT / "validation_metrics.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved alternative model comparison to {path}")


if __name__ == "__main__":
    main()
