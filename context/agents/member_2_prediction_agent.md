# Member 2 AI-Agent Context — XGBoost

Read `shared_agent_context.md`.

Own only `src/prediction/**`, `tests/prediction/**`, and prediction outputs/docs. Do not edit data or persona modules.

Inspect Member 1's actual exports and shared contracts. Confirm target type, option mapping, feature eligibility, missingness, and split IDs. Never assume categorical target.

Fit preprocessing on training data only. Exclude target/Wave 4 labels from features. Respect participant splits. Compare majority baseline. Select on validation; test once. Report accuracy, macro-F1, log-loss, confusion matrix, and class balance. Export option probabilities only if mapping is stable.

XGBoost is optional and must not block core A/B/C experiments.
