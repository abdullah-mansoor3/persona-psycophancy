#!/usr/bin/env python3
"""Train a secondary XGBoost variant that may use the earlier target answer.

This is a Wave-4 forecasting model, distinct from the primary profile-only model.
Hyperparameters are selected by cross-validation inside the training split; the
validation split is used once for assessment. The test split is not loaded here.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "preprocessing_manifest.json"
OUT = ROOT / "outputs" / "xgboost" / "prior_answer"
SEED = 42


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob[:, 1])),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["option_1", "option_2"],
            output_dict=True, zero_division=0,
        ),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    feature_data = pd.read_csv(DATA / "xgboost_features.csv", low_memory=False)
    target_data = pd.read_csv(DATA / "xgboost_targets.csv", low_memory=False)
    feature_data = feature_data.loc[feature_data.split.isin(["train", "validation"])].copy()
    target_data = target_data.loc[target_data.split.isin(["train", "validation"])].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)

    meta = manifest["xgboost_features"]
    role_map = {}
    for section in ("roles_profile", "roles_expanded_extra"):
        for role, columns in meta[section].items():
            role_map.update({column: role for column in columns})

    output = {
        "model_variant": "profile features plus prior answer to the same question",
        "selection": "4-fold stratified CV on train, scoring=accuracy",
        "test_split_used": False,
        "seed": SEED,
        "split_version": manifest["split_version"],
        "package_versions": {
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "questions": {},
    }

    for question_id in manifest["targets"]:
        blocklist_key = "QID196_model" if question_id == "QID196" else "QID9_3_model_strict"
        blocklist = set(meta["model_feature_exclusions"][blocklist_key])
        excluded_meta = {"participant_id", "split", "split_seed", "split_version"}
        feature_cols = [
            c for c in feature_data.columns
            if c not in excluded_meta and c not in blocklist and role_map.get(c) != "free_text"
        ]
        role = {c: role_map[c] for c in feature_cols}
        data = target_data.loc[target_data.question_id == question_id].merge(
            feature_data[["participant_id", "split", *feature_cols]],
            on=["participant_id", "split"], how="inner", validate="one_to_one",
        )
        if len(data) != 2058:
            raise ValueError(f"Unexpected joined row count for {question_id}: {len(data)}")

        # Earlier response is available at prediction time and is not the Wave-4 label.
        data["prior_target_answer"] = data["wave1_3_answer"].astype(str)
        data["y"] = data["wave4_answer_code"].astype(int) - 1
        numeric = [c for c in feature_cols if role[c] == "numeric"]
        categorical = [c for c in feature_cols if role[c] != "numeric"] + ["prior_target_answer"]
        for col in numeric:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        for col in categorical:
            data[col] = data[col].where(data[col].notna(), "__NOT_ASSIGNED__").astype(str)

        train = data.loc[data.split == "train"]
        valid = data.loc[data.split == "validation"]
        x_cols = [*feature_cols, "prior_target_answer"]
        preprocessor = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__NOT_ASSIGNED__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ], remainder="drop")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", random_state=SEED,
                n_jobs=4, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=3, reg_lambda=5.0,
            )),
        ])
        grid = {
            "classifier__n_estimators": [100, 300],
            "classifier__max_depth": [2, 4],
            "classifier__learning_rate": [0.03, 0.1],
        }
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
        search = GridSearchCV(
            pipeline, grid, scoring="accuracy", cv=cv, n_jobs=1,
            refit=True, return_train_score=False,
        )
        search.fit(train[x_cols], train["y"].to_numpy())
        val_prob = search.best_estimator_.predict_proba(valid[x_cols])
        val_pred = val_prob.argmax(axis=1)
        y_valid = valid["y"].to_numpy()

        # Comparators on the same validation participants.
        y_train = train["y"].to_numpy()
        majority = int(pd.Series(y_train).value_counts().idxmax())
        prior = valid["wave1_3_answer_code"].astype(int).to_numpy() - 1
        val_result = evaluate(y_valid, val_pred, val_prob)
        val_result["majority_baseline_accuracy"] = float(accuracy_score(
            y_valid, np.full(len(y_valid), majority)
        ))
        val_result["persistence_baseline_accuracy"] = float(accuracy_score(y_valid, prior))

        name = question_id.lower()
        joblib.dump({
            "pipeline": search.best_estimator_,
            "feature_columns": x_cols,
            "class_labels": manifest["target_metadata"][question_id]["options"],
            "class_code_map": manifest["answer_code_maps"][question_id],
            "selected_params": search.best_params_,
        }, OUT / "models" / f"xgb_{name}_with_prior.joblib")
        output["questions"][question_id] = {
            "n_training_participants": int(len(train)),
            "n_validation_participants": int(len(valid)),
            "profile_feature_count": len(feature_cols),
            "prior_answer_included": True,
            "cv_best_accuracy": float(search.best_score_),
            "cv_best_parameters": search.best_params_,
            "validation": val_result,
            "cv_results": [
                {"parameters": params, "mean_accuracy": float(mean_score),
                 "std_accuracy": float(std_score)}
                for params, mean_score, std_score in zip(
                    search.cv_results_["params"], search.cv_results_["mean_test_score"],
                    search.cv_results_["std_test_score"],
                )
            ],
        }

    (OUT / "validation_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved prior-answer-assisted validation results under {OUT}")
    for question_id, result in output["questions"].items():
        print(question_id, "CV accuracy=", round(result["cv_best_accuracy"], 3),
              "validation accuracy=", round(result["validation"]["accuracy"], 3),
              "persistence=", round(result["validation"]["persistence_baseline_accuracy"], 3))


if __name__ == "__main__":
    main()
