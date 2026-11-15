#!/usr/bin/env python3
"""Train a change detector that only overrides the prior answer when warranted.

This is a separate prior-answer-assisted forecasting variant. Candidate models
and the override threshold are selected from training-only OOF predictions;
validation is a held-out check. Test rows are not loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "preprocessing_manifest.json"
OUT = ROOT / "outputs" / "xgboost" / "answer_change_gate"
SEED = 42
FOLDS = 4
THRESHOLDS = np.arange(0.05, 0.951, 0.01)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def gated_prediction(prior: np.ndarray, p_change: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(p_change >= threshold, 1 - prior, prior)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    meta = manifest["xgboost_features"]
    role_map = {}
    for group in ("roles_profile", "roles_expanded_extra"):
        for role, columns in meta[group].items():
            role_map.update({column: role for column in columns})
    features = pd.read_csv(DATA / "xgboost_features.csv", low_memory=False)
    targets = pd.read_csv(DATA / "xgboost_targets.csv", low_memory=False)
    # Test participants are deliberately removed before any merges or fitting.
    features = features.loc[features.split.isin(["train", "validation"])]
    targets = targets.loc[targets.split.isin(["train", "validation"])]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    report = {
        "method": "predict whether the Wave-4 answer changes; flip prior answer only above a training-OOF threshold",
        "test_split_used": False,
        "selection": "maximize answer accuracy on training-only out-of-fold predictions",
        "seed": SEED,
        "folds": FOLDS,
        "scikit_learn": sklearn.__version__,
        "questions": {},
    }
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    for qid in manifest["targets"]:
        block_key = "QID196_model" if qid == "QID196" else "QID9_3_model_strict"
        excluded = set(meta["model_feature_exclusions"][block_key])
        meta_cols = {"participant_id", "split", "split_seed", "split_version"}
        profile_cols = [c for c in features.columns if c not in meta_cols
                        and c not in excluded and role_map.get(c) != "free_text"]
        data = targets.loc[targets.question_id == qid].merge(
            features[["participant_id", "split", *profile_cols]],
            on=["participant_id", "split"], how="inner", validate="one_to_one",
        ).copy()
        data["y"] = data.wave4_answer_code.astype(int).to_numpy() - 1
        data["prior"] = data.wave1_3_answer_code.astype(int).to_numpy() - 1
        data["prior_target_answer"] = data.wave1_3_answer.astype(str).fillna("__NOT_ASSIGNED__")
        data["changed"] = (data.y != data.prior).astype(int)
        numeric = [c for c in profile_cols if role_map[c] == "numeric"]
        categorical = [c for c in profile_cols if role_map[c] != "numeric"] + ["prior_target_answer"]
        for col in numeric:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        for col in categorical:
            data[col] = data[col].where(data[col].notna(), "__NOT_ASSIGNED__").astype(str)

        train = data.loc[data.split == "train"].copy()
        valid = data.loc[data.split == "validation"].copy()
        x_cols = [*profile_cols, "prior_target_answer"]
        y_change = train.changed.to_numpy()
        if set(np.unique(y_change)) != {0, 1}:
            raise ValueError(f"Both change classes are required for {qid}")
        prep = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__NOT_ASSIGNED__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ], remainder="drop")
        configs = []
        for c in (0.01, 0.1, 1.0, 10.0):
            configs.append((f"logistic_C{c}", LogisticRegression(
                C=c, solver="liblinear", max_iter=2000, random_state=SEED)))
        for leaf in (2, 10):
            for weight in (None, "balanced"):
                configs.append((f"extra_leaf{leaf}_weight{weight}", ExtraTreesClassifier(
                    n_estimators=100, max_features="sqrt", min_samples_leaf=leaf,
                    class_weight=weight, random_state=SEED, n_jobs=4)))
        for depth, estimators, rate in ((2, 100, 0.03), (2, 300, 0.03), (3, 100, 0.05), (3, 300, 0.05)):
            configs.append((f"xgb_d{depth}_n{estimators}_lr{rate}", XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", max_depth=depth,
                n_estimators=estimators, learning_rate=rate, min_child_weight=3,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                random_state=SEED, n_jobs=4)))

        best = None
        all_candidates = []
        train_prior = train.prior.to_numpy()
        train_y = train.y.to_numpy()
        base_oof_acc = float(accuracy_score(train_y, train_prior))
        for name, classifier in configs:
            pipe = Pipeline([("preprocessor", prep), ("classifier", classifier)])
            oof = cross_val_predict(
                pipe, train[x_cols], y_change, cv=cv, method="predict_proba", n_jobs=1,
            )[:, 1]
            scored = []
            for threshold in THRESHOLDS:
                pred = gated_prediction(train_prior, oof, float(threshold))
                scored.append(float(accuracy_score(train_y, pred)))
            index = int(np.argmax(scored))
            candidate = {
                "name": name, "classifier": classifier,
                "cv_accuracy": scored[index], "threshold": float(THRESHOLDS[index]),
                "oof_change_accuracy": float(accuracy_score(y_change, (oof >= THRESHOLDS[index]).astype(int))),
            }
            all_candidates.append({k: v for k, v in candidate.items() if k != "classifier"})
            if best is None or (candidate["cv_accuracy"], candidate["threshold"]) > (best["cv_accuracy"], best["threshold"]):
                best = candidate
        assert best is not None

        final = Pipeline([("preprocessor", prep), ("classifier", clone(best["classifier"]))])
        final.fit(train[x_cols], y_change)
        valid_p_change = final.predict_proba(valid[x_cols])[:, 1]
        valid_prior = valid.prior.to_numpy()
        valid_y = valid.y.to_numpy()
        gated = gated_prediction(valid_prior, valid_p_change, best["threshold"])
        overrides = valid_p_change >= best["threshold"]
        persistence_accuracy = float(accuracy_score(valid_y, valid_prior))
        results = {
            "n_train": int(len(train)), "n_validation": int(len(valid)),
            "training_change_rate": float(train.changed.mean()),
            "validation_change_rate": float(valid.changed.mean()),
            "persistence_cv_oof_accuracy": base_oof_acc,
            "selected_model": best["name"],
            "selected_threshold": best["threshold"],
            "cv_oof_accuracy": best["cv_accuracy"],
            "candidate_results": all_candidates,
            "validation": {
                "persistence": metrics(valid_y, valid_prior),
                "change_gate": metrics(valid_y, gated),
                "override_count": int(overrides.sum()),
                "override_accuracy": (float(accuracy_score(valid_y[overrides], gated[overrides]))
                                      if overrides.any() else None),
            },
        }
        report["questions"][qid] = results
        joblib.dump({"pipeline": final, "feature_columns": x_cols,
                     "threshold": best["threshold"], "selected_model": best["name"]},
                    OUT / "models" / f"{qid.lower()}_change_gate.joblib")
        print(qid, "OOF persistence=", round(base_oof_acc, 3),
              "OOF gated=", round(best["cv_accuracy"], 3),
              "validation persistence=", round(persistence_accuracy, 3),
              "validation gated=", round(results["validation"]["change_gate"]["accuracy"], 3),
              "model=", best["name"], "threshold=", best["threshold"], flush=True)

    path = OUT / "validation_metrics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved change-gate results under {OUT}")


if __name__ == "__main__":
    main()
