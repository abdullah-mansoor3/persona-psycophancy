#!/usr/bin/env python3
"""Try stronger XGBoost variants without using the held-out test split.

Adds the earlier target answer for both questions and, for randomized QID9_3,
features parsed from the exact product passage. Hyperparameters and decision
thresholds are selected from training-only cross-validation. Validation is an
external check; the test split is intentionally never loaded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "preprocessing_manifest.json"
OUT = ROOT / "outputs" / "xgboost" / "optimized"
SEED = 42


def add_passage_fields(text: str) -> tuple[str, str, float]:
    category = re.search(r"product category:\s*(.*?)\.\s*Suppose", text, re.I)
    product = re.search(
        r"following product in that category:\s*(.*?)\.\s*The product is priced at:", text, re.I
    )
    price = re.search(r"priced at:\s*\$([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if not (category and product and price):
        return "__UNKNOWN__", "__UNKNOWN__", np.nan
    return category.group(1).strip(), product.group(1).strip(), float(price.group(1))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    feature_data = pd.read_csv(DATA / "xgboost_features.csv", low_memory=False)
    target_data = pd.read_csv(DATA / "xgboost_targets.csv", low_memory=False)
    variants = pd.read_csv(DATA / "target_question_variants.csv", low_memory=False)
    feature_data = feature_data.loc[feature_data.split.isin(["train", "validation"])].copy()
    target_data = target_data.loc[target_data.split.isin(["train", "validation"])].copy()
    variants = variants.loc[variants.split.isin(["train", "validation"])].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)

    meta = manifest["xgboost_features"]
    roles: dict[str, str] = {}
    for section in ("roles_profile", "roles_expanded_extra"):
        for role, columns in meta[section].items():
            roles.update({column: role for column in columns})

    result = {
        "test_split_used": False,
        "selection": "4-fold stratified CV on train; accuracy scoring; threshold tuned on train OOF predictions",
        "threshold_grid": "0.25 through 0.75 in 0.01 steps",
        "seed": SEED,
        "split_version": manifest["split_version"],
        "package_versions": {"pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                             "xgboost": xgboost.__version__},
        "questions": {},
    }

    for qid in manifest["targets"]:
        block_key = "QID196_model" if qid == "QID196" else "QID9_3_model_strict"
        block = set(meta["model_feature_exclusions"][block_key])
        metadata = {"participant_id", "split", "split_seed", "split_version"}
        profile_cols = [c for c in feature_data.columns
                        if c not in metadata and c not in block and roles.get(c) != "free_text"]
        data = target_data.loc[target_data.question_id == qid].merge(
            feature_data[["participant_id", "split", *profile_cols]],
            on=["participant_id", "split"], how="inner", validate="one_to_one",
        )
        if qid == "QID9_3":
            wording = variants.loc[variants.question_id == qid,
                                   ["participant_id", "split", "wave4_question_text"]]
            data = data.merge(wording, on=["participant_id", "split"],
                              how="inner", validate="one_to_one")
            fields = data["wave4_question_text"].map(add_passage_fields)
            data[["product_category", "product_title", "product_price"]] = pd.DataFrame(
                fields.tolist(), index=data.index
            )
        data = data.copy()
        data["prior_target_answer"] = data["wave1_3_answer"].astype(str)
        data["y"] = data["wave4_answer_code"].astype(int) - 1

        numeric = [c for c in profile_cols if roles[c] == "numeric"]
        categorical = [c for c in profile_cols if roles[c] != "numeric"] + ["prior_target_answer"]
        transformers = [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__NOT_ASSIGNED__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
        if qid == "QID9_3":
            categorical += ["product_category"]
            numeric += ["product_price"]
            transformers = [
                ("numeric", SimpleImputer(strategy="median"), numeric),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="__NOT_ASSIGNED__")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), categorical),
                ("product_text", TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, max_features=2500, stop_words="english"
                ), "product_title"),
            ]
        for col in numeric:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        for col in categorical:
            data[col] = data[col].where(data[col].notna(), "__NOT_ASSIGNED__").astype(str)
        train = data.loc[data.split == "train"].copy()
        valid = data.loc[data.split == "validation"].copy()
        x_cols = [*profile_cols, "prior_target_answer"]
        if qid == "QID9_3":
            x_cols += ["product_category", "product_title", "product_price"]
        prep = ColumnTransformer(transformers, remainder="drop")
        pipeline = Pipeline([
            ("preprocessor", prep),
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
        # 8 candidates x 4 folds; bounded search on training data only.
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
        print(f"Tuning {qid} on training folds", flush=True)
        x_train = train[x_cols]
        y_train = train.y.to_numpy()
        candidate_results = []
        best_params = None
        best_cv_accuracy = -1.0
        for params in ParameterGrid(grid):
            fold_scores = []
            for fit_idx, score_idx in cv.split(x_train, y_train):
                candidate = clone(pipeline).set_params(**params)
                candidate.fit(x_train.iloc[fit_idx], y_train[fit_idx])
                fold_scores.append(float(accuracy_score(
                    y_train[score_idx], candidate.predict(x_train.iloc[score_idx])
                )))
            mean_score = float(np.mean(fold_scores))
            candidate_results.append({"parameters": params, "fold_accuracy": fold_scores,
                                      "mean_accuracy": mean_score})
            if mean_score > best_cv_accuracy:
                best_cv_accuracy, best_params = mean_score, params
        assert best_params is not None

        best_model = clone(pipeline).set_params(**best_params)
        best_model.fit(x_train, y_train)

        # Obtain out-of-fold probabilities from the selected configuration; use
        # only these training rows to choose a decision threshold.
        oof_prob = np.zeros((len(train), 2), dtype=float)
        for fit_idx, score_idx in cv.split(x_train, y_train):
            oof_model = clone(pipeline).set_params(**best_params)
            oof_model.fit(x_train.iloc[fit_idx], y_train[fit_idx])
            oof_prob[score_idx] = oof_model.predict_proba(x_train.iloc[score_idx])
        thresholds = np.round(np.arange(0.25, 0.751, 0.01), 2)
        threshold_scores = [accuracy_score(train.y.to_numpy(), (oof_prob[:, 1] >= t).astype(int))
                            for t in thresholds]
        threshold = float(thresholds[int(np.argmax(threshold_scores))])

        val_prob = best_model.predict_proba(valid[x_cols])
        val_pred = (val_prob[:, 1] >= threshold).astype(int)
        val_y = valid.y.to_numpy()
        prior_pred = valid.wave1_3_answer_code.astype(int).to_numpy() - 1
        majority_pred = np.full(len(valid), int(train.y.value_counts().idxmax()))
        final_pipe = best_model
        model_path = OUT / "models" / f"xgb_{qid.lower()}_optimized.joblib"
        joblib.dump({
            "pipeline": final_pipe, "feature_columns": x_cols, "threshold": threshold,
            "class_labels": manifest["target_metadata"][qid]["options"],
            "selected_params": best_params,
        }, model_path)
        result["questions"][qid] = {
            "n_train": int(len(train)), "n_validation": int(len(valid)),
            "profile_features": len(profile_cols),
            "engineered_question_fields": ["product_category", "product_title", "product_price"]
                if qid == "QID9_3" else [],
            "cv_accuracy": best_cv_accuracy,
            "cv_oof_accuracy_at_selected_threshold": float(max(threshold_scores)),
            "selected_threshold_for_option_2": threshold,
            "selected_parameters": best_params,
            "candidate_results": candidate_results,
            "validation": {
                "optimized_xgboost": metrics(val_y, val_pred),
                "prior_answer_persistence": metrics(val_y, prior_pred),
                "majority_class": metrics(val_y, majority_pred),
            },
            "unoptimized_argmax_validation_accuracy": float(accuracy_score(val_y, val_prob.argmax(1))),
            "parsed_question_context_coverage": (
                float(data.product_price.notna().mean()) if qid == "QID9_3" else None
            ),
        }

    (OUT / "validation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved optimized validation results under {OUT}")
    for qid, qresult in result["questions"].items():
        print(qid, "CV=", round(qresult["cv_accuracy"], 3),
              "thresholded validation=", round(qresult["validation"]["optimized_xgboost"]["accuracy"], 3),
              "persistence=", round(qresult["validation"]["prior_answer_persistence"]["accuracy"], 3),
              "threshold=", qresult["selected_threshold_for_option_2"])


if __name__ == "__main__":
    main()
