# Progress Tracker

Update at each work session. Never fill unmeasured results with estimates.

| Workstream | Owner | Status | Evidence/output |
|---|---|---|---|
| Dataset audit | M1 | DONE | `notebooks/data-audit.ipynb` (executed, all outputs) |
| Question mapping/selection | M1 | DONE | QID196 + QID9_3 verified; `data/processed/target_questions.csv` |
| Splits/profiles/peer pool | M1 | DONE | `data/processed/*` + `tests/data_pipeline/test_preprocessing_checklist.py` (13/13 pass) |
| XGBoost baseline | M2 | DONE | `outputs/xgboost/metrics/`, audit files, models; probability handoff file is missing |
| Model smoke test | M3 | DONE | executed Colab notebook; real Qwen model loaded on Tesla T4 |
| Conditions A/B/C | M3 | DONE | 60 participants x 2 targets; archive and metrics in `Member_3_Real_Qwen_Results.zip` |
| Evidence-positive pilot | M3 | DONE | 60 QID196 trials; no initially non-evidence choices, so the revision test is inconclusive |
| Integrated evaluation | M3 | IN PROGRESS | D was not run; 420 archived A/B/C/E generations |
| Documentation | All | IN PROGRESS | `REPORT.md`, `context/10_empirical_results_audit.md` |

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
- Environment/Python: Google Colab notebook; exact package environment is not captured in the archive
- GPU: Tesla T4, 15,360 MiB shown by notebook
- Model ID/revision: `Qwen/Qwen2.5-1.5B-Instruct`; exact revision not recorded in run rows
- Load success: yes; 420 archived rows have valid parses
- Average generation time: A 4.59s, B 5.24s, C 7.55s, E 9.05s per archived row
- Estimated run time: timestamps span about 44 minutes for the full archived run

## Experiment
- Prompt version/decoding: prompt templates in `src/persona_experiment/prompt_builder.py`; full prompts and complete decoding metadata were not archived
- Peer ratio/count: 2 majority : 1 minority; 3 peers; same peer group for B/C
- Actual focal N and item instances: 60 participants; A/B/C = 120 each; E = 60 QID196; D = 0
- Failed generations/exclusions: 0 parse failures in the archive; only the first 60 sorted test participants were run
- A/B/C/D metrics: B changed 25/120, all majority-to-minority on QID9_3; C changed 0/120; D not run
- Evidence-positive result: 0 eligible revisions from 60 trials; inconclusive for evidence acceptance
- Statistical tests, if any: stored B/C comparison had 0 eligible minority pairs and `p_value = 1.0`; no significance claim is supported

## Decision log
| Date | Decision | Reason | Affected files |
|---|---|---|---|
| 2026-11-05 | Audit executed; core targets = QID196 + QID9_3 | Universal, complete, stable options, 100% JSON/CSV agreement, wording identical across waves, 0 profile leakage | notebooks/data-audit.ipynb, data/processed/* |
| 2026-11-05 | Added `target_question_variants.csv` | QID9_3 passage randomized per participant (434 variants); persona prompt needs the participant's exact product/price | context/data_dictionary.md |
| 2026-11-05 | Feature CSV kept label-level (strings) | Encoding must be fit on the training fold only; profile-vs-expanded column lists live in the manifest | xgboost_features.csv, data/preprocessing_manifest.json |

## Risks
- [x] Official schema and mappings verified.
- [x] Suitable repeated target items confirmed.
- [x] GPU/runtime confirmed for the archived Colab run; exact local package environment is unavailable.
- [x] Peer pool sufficient.
- [x] Evidence-positive stimulus verified for QID196, but the test had no eligible revision.
- [x] Archived run completed within the recorded Colab session.

## Evidence-backed caveats

- The run did not test initial minority-to-majority movement: all A/B/C initial answers matched the peer majority.
- Condition D is incomplete in the available checkout. The XGBoost models and metrics exist, but `outputs/xgboost/predictions/xgboost_gate_probabilities.csv` is missing and no D rows are archived.
- The raw archive does not contain full prompts or complete decoding/model-revision metadata.
- See `context/10_empirical_results_audit.md` for the full conflict register and source mapping.
