# Progress Tracker

Update at each work session. Never fill unmeasured results with estimates.

| Workstream | Owner | Status | Evidence/output |
|---|---|---|---|
| Dataset audit | M1 | DONE | `notebooks/data-audit.ipynb` (executed, all outputs) |
| Question mapping/selection | M1 | DONE | QID196 + QID9_3 verified; `data/processed/target_questions.csv` |
| Splits/profiles/peer pool | M1 | DONE | `data/processed/*` + `tests/data_pipeline/test_preprocessing_checklist.py` (13/13 pass) |
| XGBoost baseline | M2 | IN PROGRESS | `data/processed/xgboost_features.csv`, `xgboost_targets.csv` ready |
| Model smoke test | M3 | IN PROGRESS | use `persona_profiles.csv`, `target_question_variants.csv`, `peer_responses.csv` |
| Conditions A/B/C | M3 | NOT STARTED | |
| Evidence-positive pilot | M3 | NOT STARTED | |
| Integrated evaluation | M3 | NOT STARTED | |
| Documentation | All | NOT STARTED | |

Allowed status: NOT STARTED / IN PROGRESS / BLOCKED / DONE / DROPPED.

## Verified dataset facts
- Dataset/config/version: LLM-Digital-Twin/Twin-2K-500, config `wave_split` (7 parquet chunks), revision `main`
- Actual participant count: 2,058 unique pids; identical across parquet chunks and both label CSVs
- Actual fields and types: `pid` + persona text/JSON + wave4 question JSON; answer payloads vary by question type (MC `SelectedByPosition`/`SelectedText`, Matrix row lists, Slider `Values`, TE `Text`, DB none)
- Selected question IDs and exact wording:
  - QID196 (constant wording, 2,058/2,058): "…two trays of black and white marbles…" options `["the small tray","the large tray"]`
  - QID9_3 (**per-participant randomized passage**, 434 distinct texts, identical across waves): "Please consider the following product category: … priced at $… Would you or would you not purchase this product?" options `["Yes, I would purchase the product","No, I would not purchase the product"]`
- Wave 4 mappings: `wave4_response_label.csv` values equal Wave 4 JSON `SelectedText` 100% for both targets (2,058 × 2); `wave4_Q_wave1_3_A` holds the earlier answers used by the peer pool
- Options/answer types: see two targets above (binary MC); code maps 1/2 (option position) in the manifest
- Missingness and eligible counts: both targets complete in both waves (2,058 each, 0 missing); persona profiles: 156–158 answered blocks, 0 missing answers; earlier wave1_3 CSV: 760 answer columns
- Peer reference pool size: train = 1,440 participants; majority answers `the small tray` (QID196) and `No, I would not purchase the product` (QID9_3); 309 test × 2 questions × 3 peers = 1,854 peer rows
- Split seed and counts: seed 42, stratified on joint Wave 4 target labels, 1,440 / 309 / 309
- Known caveats: QID9_3 wording randomized per participant (use `target_question_variants.csv`); earlier-answer wording of non-target experiment items differs instrument-wise from Wave 4 (match by QID, never by text); persona texts contain single words like "marble"/"grocery" from unrelated word-association tasks — phrase-level leakage scan shows 0 target leakage

## Runtime
- Environment/Python:
- GPU:
- Model ID/revision:
- Load success:
- Average generation time:
- Estimated run time:

## Experiment
- Prompt version/decoding:
- Peer ratio/count:
- Actual focal N and item instances:
- Failed generations/exclusions:
- A/B/C/D metrics:
- Evidence-positive result:
- Statistical tests, if any:

## Decision log
| Date | Decision | Reason | Affected files |
|---|---|---|---|
| 2026-11-05 | Audit executed; core targets = QID196 + QID9_3 | Universal, complete, stable options, 100% JSON/CSV agreement, wording identical across waves, 0 profile leakage | notebooks/data-audit.ipynb, data/processed/* |
| 2026-11-05 | Added `target_question_variants.csv` | QID9_3 passage randomized per participant (434 variants); persona prompt needs the participant's exact product/price | context/data_dictionary.md |
| 2026-11-05 | Feature CSV kept label-level (strings) | Encoding must be fit on the training fold only; profile-vs-expanded column lists live in the manifest | xgboost_features.csv, data/preprocessing_manifest.json |

## Risks
- [x] Official schema and mappings verified.
- [x] Suitable repeated target items confirmed.
- [ ] GPU/runtime confirmed.
- [ ] Peer pool sufficient.
- [ ] Evidence-positive stimulus verified.
- [ ] Final run fits Day 4 deadline.
