# Member 2 — XGBoost Prediction Baseline

## Ownership
Own `src/prediction/`, `tests/prediction/`, model artifacts, metrics, and prediction documentation. Do not edit Member 1 or Member 3 internals.

## Objective
Optional baseline: predict a participant's likely Wave 4 answer to a target question from permitted earlier non-target responses. This is not a sycophancy classifier and is not required for core A/B/C experiments.

## Inputs
Use Member 1's verified files, participant splits, item mapping, and feature eligibility. Never use the target answer or Wave 4 target label as a feature. Fit preprocessing on training data only.

## Method
- One model per target question unless data supports a justified alternative.
- Use classification only for categorical targets; confirm type first.
- Compare against majority-class baseline.
- Suggested configurations: (200 trees, depth 3, learning rate .05) and (300 trees, depth 5, learning rate .05); subsample and colsample_bytree .8, reg_lambda 1.
- Select on validation macro-F1/log-loss. Evaluate test once.

## Recommended accuracy improvement

The current model is weaker than the persistence baseline because it excludes the participant's own earlier answer to the target question. Member 2 should train two clearly separated tracks:

1. **Strict profile-only track:** keep the current exclusions. This is the fair comparison with the persona profile.
2. **Persistence-aware track:** add the participant's earlier target answer as an explicit historical feature. This is not Wave 4 leakage, but it is not an apples-to-apples profile-only comparison. Label it clearly and compare it against the persistence baseline.

For both tracks, select settings on validation only. A wider fixed grid may include tree counts 100/200/300/500, depths 2/3/4/5, learning rates 0.03/0.05/0.10, and optionally a class-weighted variant. Select the final model using the paper's primary metric, preferably accuracy if the goal is answer prediction, while still reporting macro-F1, balanced accuracy, log-loss, and ROC-AUC. Do not tune on the test set.

## Metrics
Accuracy, macro-F1, log-loss, confusion matrix, majority baseline, class balance.

## Optional handoff to Member 3
Export class probabilities per participant/question/option. They may be supplied as a soft anchor, never as compulsory ground truth. Match option IDs to `questions.csv`.

The handoff must include one row per test participant and target question with:

- `participant_id`, `question_id`, `split`, and `model_id`;
- the exact option order used in the probabilities;
- one probability per option, summing to approximately 1;
- feature track (`profile_only` or `persistence_aware`);
- seed, package versions, selected parameters, and validation selection metric;
- a leakage audit confirming that no Wave 4 value entered the features.

Member 3 should use the same probability file for every D run and compare D directly with C on the same participants, questions, initial answers, and peer groups.

## Actual repository status

- Leakage audit, models, and validation/test metrics are present under `outputs/xgboost/`.
- Test accuracy is 58.25% for QID196 and 53.07% for QID9_3. Persistence is higher on both targets: 73.14% and 82.52% respectively.
- The probability handoff file `outputs/xgboost/predictions/xgboost_gate_probabilities.csv` is missing from this checkout.
- Condition D was therefore not executed. Do not claim that an XGBoost soft anchor improved, preserved, or destabilized the persona experiment.

## Done when
- [x] Feature/target mapping documented.
- [x] Participant split and no-leakage checks pass.
- [x] Preprocessing fit only on training data.
- [x] Majority baseline and model metrics exported.
- [x] Model/config/seed/version recorded.
- [ ] Probability file validates; this is incomplete in the available checkout.

The current checkout cannot complete this retraining step because the runtime used for this session does not have the declared `xgboost` package installed. No new accuracy claim has been added.
