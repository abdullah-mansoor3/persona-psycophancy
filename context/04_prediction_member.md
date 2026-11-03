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

## Metrics
Accuracy, macro-F1, log-loss, confusion matrix, majority baseline, class balance.

## Optional handoff to Member 3
Export class probabilities per participant/question/option. They may be supplied as a soft anchor, never as compulsory ground truth. Match option IDs to `questions.csv`.

## Done when
- [ ] Feature/target mapping documented.
- [ ] Participant split and no-leakage checks pass.
- [ ] Preprocessing fit only on training data.
- [ ] Majority baseline and model metrics exported.
- [ ] Model/config/seed/version recorded.
- [ ] Probability file validates, if integration is feasible.
