#!/usr/bin/env python3
"""Tune the profile-only XGBoost models using training-fold CV.

Validation is used for an external check after selection. The test split is
excluded from search and is not read by this script. Tuned artifacts are
written to a separate directory so the existing prediction files stay intact.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "preprocessing_manifest.json"
OUT = ROOT / "outputs" / "xgboost" / "tuning_profile"
SEED = 42
N_ITER = 32


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    features = pd.read_csv(DATA / "xgboost_features.csv", low_memory=False)
    targets = pd.read_csv(DATA / "xgboost_targets.csv", low_memory=False)
    # Avoid even loading test labels into the tuning dataframe.
    features = features.loc[features.split.isin(["train", "validation"])]
    targets = targets.loc[targets.split.isin(["train", "validation"])]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)

    role_map: dict[str, str] = {}
    metadata = manifest["xgboost_features"]
    for group in ("roles_profile", "roles_expanded_extra"):
        for role, columns in metadata[group].items():
            role_map.update({column: role for column in columns})

    report = {
        "model_variant": "profile-only; same-question prior answer excluded",
        "selection": f"{N_ITER} randomized configurations, 4-fold stratified CV on training split, accuracy scoring",
        "test_split_used": False,
        "seed": SEED,
        "split_version": manifest["split_version"],
        "package_versions": {
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "questions": {},
    }

    for qid in manifest["targets"]:
        block_key = "QID196_model" if qid == "QID196" else "QID9_3_model_strict"
        exclusions = set(metadata["model_feature_exclusions"][block_key])
        meta_cols = {"participant_id", "split", "split_seed", "split_version"}
        columns = [
            col for col in features.columns
            if col not in meta_cols and col not in exclusions
            and role_map.get(col) != "free_text"
        ]
        unknown = [col for col in columns if col not in role_map]
        if unknown:
            raise ValueError(f"Unclassified features for {qid}: {unknown[:10]}")

        data = targets.loc[targets.question_id == qid].merge(
            features[["participant_id", "split", *columns]],
            on=["participant_id", "split"], how="inner", validate="one_to_one",
        )
        if len(data) != len(targets.loc[targets.question_id == qid]):
            raise ValueError(f"Feature/target join lost rows for {qid}")
        data["y"] = data.wave4_answer_code.astype(int) - 1
        numeric = [col for col in columns if role_map[col] == "numeric"]
        categorical = [col for col in columns if role_map[col] != "numeric"]
        for col in numeric:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        for col in categorical:
            data[col] = data[col].where(data[col].notna(), "__NOT_ASSIGNED__").astype(str)

        train = data.loc[data.split == "train"].copy()
        valid = data.loc[data.split == "validation"].copy()
        x_train, y_train = train[columns], train.y.to_numpy()
        x_valid, y_valid = valid[columns], valid.y.to_numpy()
        prep = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__NOT_ASSIGNED__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ], remainder="drop")
        pipeline = Pipeline([
            ("preprocessor", prep),
            ("classifier", XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                random_state=SEED, n_jobs=4,
            )),
        ])
        search_space = {
            "classifier__n_estimators": [100, 200, 300, 500],
            "classifier__max_depth": [1, 2, 3, 4, 5],
            "classifier__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
            "classifier__min_child_weight": [1, 3, 5, 8, 12],
            "classifier__subsample": [0.65, 0.8, 0.95, 1.0],
            "classifier__colsample_bytree": [0.65, 0.8, 0.95, 1.0],
            "classifier__gamma": [0.0, 0.1, 0.3, 0.5],
            "classifier__reg_lambda": [1.0, 3.0, 5.0, 10.0, 20.0],
        }
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
        search = RandomizedSearchCV(
            pipeline, search_space, n_iter=N_ITER, scoring="accuracy", cv=cv,
            random_state=SEED, n_jobs=1, refit=True, return_train_score=False,
            verbose=1,
        )
        print(f"Searching {qid} using training data only...", flush=True)
        search.fit(x_train, y_train)
        best = search.best_estimator_
        val_pred = best.predict(x_valid)
        prior_pred = valid.wave1_3_answer_code.astype(int).to_numpy() - 1
        majority_pred = np.full(len(valid), int(pd.Series(y_train).value_counts().idxmax()))

        joblib.dump({
            "pipeline": best,
            "feature_columns": columns,
            "class_labels": manifest["target_metadata"][qid]["options"],
            "selected_parameters": search.best_params_,
        }, OUT / "models" / f"xgb_{qid.lower()}_profile_tuned.joblib")
        report["questions"][qid] = {
            "n_train": int(len(train)), "n_validation": int(len(valid)),
            "feature_count": len(columns),
            "cv_best_accuracy": float(search.best_score_),
            "cv_best_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "selected_parameters": search.best_params_,
            "validation": {
                "tuned_profile_xgboost": metric_dict(y_valid, val_pred),
                "prior_answer_persistence": metric_dict(y_valid, prior_pred),
                "majority_class": metric_dict(y_valid, majority_pred),
            },
            "search_results": [
                {"parameters": params, "mean_accuracy": float(mean),
                 "std_accuracy": float(std)}
                for params, mean, std in zip(
                    search.cv_results_["params"], search.cv_results_["mean_test_score"],
                    search.cv_results_["std_test_score"],
                )
            ],
        }
        print(
            qid, "CV accuracy=", round(search.best_score_, 3),
            "validation accuracy=", round(report["questions"][qid]["validation"]["tuned_profile_xgboost"]["accuracy"], 3),
            "persistence=", round(report["questions"][qid]["validation"]["prior_answer_persistence"]["accuracy"], 3),
            flush=True,
        )

    (OUT / "validation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved tuning results under {OUT}")


if __name__ == "__main__":
    main()
