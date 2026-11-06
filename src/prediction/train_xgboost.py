#!/usr/bin/env python3
"""Train leakage-guarded, one-question-at-a-time XGBoost baselines."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import joblib
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "outputs" / "xgboost" / ".matplotlib_cache"),
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "preprocessing_manifest.json"
OUT = ROOT / "outputs" / "xgboost"
SEED = 42
SPLITS = ("train", "validation", "test")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def score(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    labels = [0, 1]
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=["option_1", "option_2"],
            output_dict=True, zero_division=0,
        ),
    }
    if y_prob is not None:
        result["log_loss"] = float(log_loss(y_true, y_prob, labels=labels))
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
    return result


def baseline_predictions(y_train: np.ndarray, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    majority = int(pd.Series(y_train).value_counts().idxmax())
    return {
        "majority": np.full(len(rows), majority, dtype=int),
        "persistence": rows["wave1_3_answer_code"].astype(int).to_numpy() - 1,
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    features = pd.read_csv(DATA / "xgboost_features.csv", low_memory=False)
    targets = pd.read_csv(DATA / "xgboost_targets.csv", low_memory=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ("models", "predictions", "metrics", "figures"):
        (OUT / folder).mkdir(exist_ok=True)

    role_map: dict[str, str] = {}
    meta = manifest["xgboost_features"]
    for role_group in ("roles_profile", "roles_expanded_extra"):
        for role, columns in meta[role_group].items():
            role_map.update({column: role for column in columns})

    question_order = manifest["targets"]
    result: dict = {
        "seed": SEED,
        "selection_metric": "validation macro_f1",
        "questions": {},
        "package_versions": {
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
        },
    }
    prediction_frames: dict[str, list[pd.DataFrame]] = {s: [] for s in SPLITS if s != "train"}
    gate_frames: list[pd.DataFrame] = []
    reports: list[str] = []

    for question_id in question_order:
        blocklist_key = "QID196_model" if question_id == "QID196" else "QID9_3_model_strict"
        blocklist = set(meta["model_feature_exclusions"][blocklist_key])
        metadata_columns = {"participant_id", "split", "split_seed", "split_version"}
        included = [
            column for column in features.columns
            if column not in metadata_columns
            and column not in blocklist
            and role_map.get(column) != "free_text"
        ]
        if not included:
            raise ValueError(f"No approved predictors remain for {question_id}")
        feature_roles = {col: role_map.get(col, "unknown") for col in included}
        unknown = [c for c, r in feature_roles.items() if r == "unknown"]
        if unknown:
            raise ValueError(f"Unclassified predictors for {question_id}: {unknown[:10]}")

        data = targets.loc[targets.question_id == question_id].merge(
            features[["participant_id", *included, "split"]],
            on=["participant_id", "split"], how="inner", validate="one_to_one",
        )
        if len(data) != len(targets.loc[targets.question_id == question_id]):
            raise ValueError(f"Feature/target join lost rows for {question_id}")
        classes = manifest["target_metadata"][question_id]["options"]
        code_map = manifest["answer_code_maps"][question_id]
        if [code_map[c] for c in classes] != [1, 2]:
            raise ValueError(f"Unexpected class code order for {question_id}: {code_map}")
        numeric = [c for c in included if feature_roles[c] == "numeric"]
        categorical = [c for c in included if feature_roles[c] != "numeric"]
        # Survey exports can mix numbers and strings in one categorical column.
        # Normalize to strings and retain missingness as an explicit category.
        for column in numeric:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        for column in categorical:
            data[column] = data[column].where(data[column].notna(), "__NOT_ASSIGNED__").astype(str)
        data = data.assign(y=data["wave4_answer_code"].astype(int) - 1)
        # Categorical NaNs stay distinct from valid answer categories. For branch
        # questions this captures the documented not-assigned state.
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", SimpleImputer(strategy="median"), numeric),
                (
                    "categorical",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__NOT_ASSIGNED__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    categorical,
                ),
            ],
            remainder="drop",
        )
        train = data.loc[data.split == "train"].copy()
        valid = data.loc[data.split == "validation"].copy()
        test = data.loc[data.split == "test"].copy()
        if any(len(part) == 0 for part in (train, valid, test)):
            raise ValueError(f"A split is empty for {question_id}")
        if set(train.y.unique()) != {0, 1}:
            raise ValueError(f"Training split does not contain both classes for {question_id}")

        x_train = preprocessor.fit_transform(train[included])
        x_valid = preprocessor.transform(valid[included])
        x_test = preprocessor.transform(test[included])
        y_train = train.y.to_numpy()
        y_valid = valid.y.to_numpy()
        y_test = test.y.to_numpy()

        baselines: dict[str, dict] = {}
        for split_name, split_rows, y_true in (
            ("validation", valid, y_valid), ("test", test, y_test)
        ):
            for baseline_name, preds in baseline_predictions(y_train, split_rows).items():
                baselines.setdefault(split_name, {})[baseline_name] = score(y_true, preds)

        candidates = []
        best_model = None
        best_score = -1.0
        best_params = None
        for estimators in (200, 300):
            for depth in (3, 5):
                params = {
                    "n_estimators": estimators,
                    "max_depth": depth,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "random_state": SEED,
                    "n_jobs": -1,
                }
                model = XGBClassifier(**params)
                model.fit(x_train, y_train)
                valid_prob = model.predict_proba(x_valid)
                valid_pred = valid_prob.argmax(axis=1)
                valid_metrics = score(y_valid, valid_pred, valid_prob)
                candidates.append({"parameters": params, "validation": valid_metrics})
                if valid_metrics["macro_f1"] > best_score:
                    best_score, best_params, best_model = valid_metrics["macro_f1"], params, model

        assert best_model is not None and best_params is not None
        model_path = OUT / "models" / f"xgb_{question_id.lower()}.json"
        best_model.save_model(model_path)
        joblib.dump({"preprocessor": preprocessor, "model": best_model,
                    "feature_columns": included, "class_labels": classes,
                    "class_code_map": code_map},
                   OUT / "models" / f"pipeline_{question_id.lower()}.joblib")

        question_result = {
            "n_features_before_encoding": len(included),
            "n_features_after_encoding": int(x_train.shape[1]),
            "class_labels_in_probability_order": classes,
            "class_counts_by_split": {
                split_name: part["wave4_answer"].value_counts().to_dict()
                for split_name, part in (("train", train), ("validation", valid), ("test", test))
            },
            "baselines": baselines,
            "candidate_runs": candidates,
            "selected_parameters": best_params,
        }
        reports.append(f"## {question_id}\n\nSelected parameters: `{best_params}`\n\n")
        test_prob = best_model.predict_proba(x_test)
        for split_name, rows, probs, y_true in (
            ("validation", valid, best_model.predict_proba(x_valid), y_valid),
            ("test", test, test_prob, y_test),
        ):
            pred = probs.argmax(axis=1)
            model_metrics = score(y_true, pred, probs)
            question_result.setdefault("model_metrics", {})[split_name] = model_metrics
            out = pd.DataFrame({
                "participant_id": rows.participant_id.to_numpy(),
                "question_id": question_id,
                "split": split_name,
                "true_label": rows.wave4_answer.to_numpy(),
                "predicted_label": [classes[int(x)] for x in pred],
                "model_id": f"xgb-{question_id.lower()}-seed{SEED}",
            })
            for idx, label in enumerate(classes):
                out[f"probability_{slug(label)}"] = probs[:, idx]
            prediction_frames[split_name].append(out)
            if split_name == "test":
                gate_frames.append(pd.DataFrame({
                    "participant_id": rows.participant_id.to_numpy(),
                    "question_id": question_id,
                    "split": "test",
                    "classes_json": json.dumps(classes),
                    "probabilities_json": [
                        json.dumps({label: float(probs[i, idx]) for idx, label in enumerate(classes)})
                        for i in range(len(rows))
                    ],
                    "model_id": f"xgb-{question_id.lower()}-seed{SEED}",
                }))
                cm = np.asarray(model_metrics["confusion_matrix"])
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.imshow(cm, cmap="Blues")
                ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=classes,
                       yticklabels=classes, xlabel="Predicted", ylabel="True",
                       title=f"{question_id} test confusion matrix")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
                fig.tight_layout()
                fig.savefig(OUT / "figures" / f"{question_id.lower()}_confusion_matrix.png", dpi=160)
                plt.close(fig)
        result["questions"][question_id] = question_result

    for split_name, frames in prediction_frames.items():
        pd.concat(frames, ignore_index=True).to_csv(
            OUT / "predictions" / f"{split_name}_predictions.csv", index=False
        )
    pd.concat(gate_frames, ignore_index=True).to_csv(
        OUT / "predictions" / "xgboost_gate_probabilities.csv", index=False
    )
    for split_name in ("validation", "test"):
        split_results = {
            qid: {
                "n_features_before_encoding": item["n_features_before_encoding"],
                "n_features_after_encoding": item["n_features_after_encoding"],
                "class_labels_in_probability_order": item["class_labels_in_probability_order"],
                "class_counts": item["class_counts_by_split"][split_name],
                "selected_parameters": item["selected_parameters"],
                "model": item["model_metrics"][split_name],
                "baselines": item["baselines"][split_name],
            }
            for qid, item in result["questions"].items()
        }
        (OUT / "metrics" / f"{split_name}_metrics.json").write_text(
            json.dumps({"split": split_name, "seed": SEED,
                        "split_version": manifest["split_version"],
                        "package_versions": result["package_versions"],
                        "selection_metric": result["selection_metric"],
                        "questions": split_results}, indent=2), encoding="utf-8"
        )
    with (OUT / "metrics" / "classification_reports.txt").open("w", encoding="utf-8") as f:
        for qid, qresult in result["questions"].items():
            f.write(f"{qid}\n")
            f.write(json.dumps(qresult["model_metrics"], indent=2))
            f.write("\n\n")
    card = [
        "# XGBoost model card", "",
        "Target: predict each participant's Wave 4 response using earlier-wave features.",
        "Splits are participant-disjoint (seed 42). Preprocessing is fit on training participants only.",
        "Free text and each question's documented target-family exclusions are removed.",
        "Validation macro-F1 selected among four fixed configurations; test metrics are held-out.",
        "Categorical missing values are encoded as `__NOT_ASSIGNED__`; branch-question NaNs therefore do not become answer labels.",
        "The model is a supporting baseline. Probabilities are not calibrated confidence estimates.", "",
        *reports,
    ]
    (OUT / "model_card.md").write_text("\n".join(card), encoding="utf-8")
    print(f"Wrote XGBoost artifacts to {OUT}")
    for qid, qresult in result["questions"].items():
        print(qid, "features=", qresult["n_features_before_encoding"],
              "selected=", qresult["selected_parameters"])
        print("  test=", qresult["model_metrics"]["test"])


if __name__ == "__main__":
    main()
