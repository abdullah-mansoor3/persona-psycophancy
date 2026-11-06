# XGBoost model card

Target: predict each participant's Wave 4 response using earlier-wave features.
Splits are participant-disjoint (seed 42). Preprocessing is fit on training participants only.
Free text and each question's documented target-family exclusions are removed.
Validation macro-F1 selected among four fixed configurations; test metrics are held-out.
Categorical missing values are encoded as `__NOT_ASSIGNED__`; branch-question NaNs therefore do not become answer labels.
The model is a supporting baseline. Probabilities are not calibrated confidence estimates.

## QID196

Selected parameters: `{'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.0, 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1}`


## QID9_3

Selected parameters: `{'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 1.0, 'objective': 'binary:logistic', 'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1}`

