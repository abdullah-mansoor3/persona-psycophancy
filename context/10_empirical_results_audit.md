# Empirical Results Audit

## Purpose

This file records what is supported by the experiment artifacts. It takes precedence over earlier planning notes and over claims in older reports when they conflict. The attached literature review is used as background for the research motivation and terminology; it is not evidence that a repository experiment was run.

The repository studies a narrow form of LLM sycophancy-related behaviour in synthetic survey personas. It does **not** measure psychopathy, diagnose people, or test a psychopathy construct.

## Evidence hierarchy

The audit used the following sources, in this order:

1. Raw run records in [`Member_3_Real_Qwen_Results.zip`](../Member_3_Real_Qwen_Results.zip), especially `results/agent_outputs/agent_runs.csv` and `results/agent_outputs/agent_runs.jsonl`.
2. The stored evaluator outputs in [`results/metrics/`](../results/metrics/) and the implementation in [`src/evaluation/metrics.py`](../src/evaluation/metrics.py).
3. Processed data, the preprocessing manifest, the executed data-audit notebook, model artifacts, and tests.
4. Existing README/context/report text, treated as claims to verify rather than as ground truth.

## Dataset and preprocessing facts

These facts are supported by [`data/preprocessing_manifest.json`](../data/preprocessing_manifest.json), the processed CSVs, [`src/data_pipeline/build_processed_dataset.py`](../src/data_pipeline/build_processed_dataset.py), and [`notebooks/data-audit.ipynb`](../notebooks/data-audit.ipynb):

- Twin-2K-500 `wave_split` contains 2,058 unique participants.
- The participant-disjoint split is 1,440 train, 309 validation, and 309 test participants, using seed 42 and joint stratification on the two target labels.
- QID196 is a constant-wording two-option marble-tray item. QID9_3 is a two-option purchase item with 434 participant-specific passages; the participant-specific wording is carried in `target_question_variants.csv`.
- Both targets are complete in Wave 1–3 and Wave 4 for all 2,058 participants. Wave 4 labels were cross-checked against the source JSON.
- Persona profiles exclude the selected target questions and target answer payloads under the phrase-level scan recorded in the manifest.
- Peer groups contain three train-only peers per test participant-question instance: two training-pool majority responses and one minority response. The same group is reused by B and C.
- The processed-data checklist passes 13/13 tests.

## What was actually run

The archive contains 420 valid Qwen records:

| Condition | Participants | Question instances | Valid parses | Actual run |
|---|---:|---:|---:|---|
| A independent | 60 | 120 | 120 | Yes |
| B unrestricted peer interaction | 60 | 120 | 120 | Yes |
| C profile-anchored gate | 60 | 120 | 120 | Yes |
| D gate + XGBoost | 0 | 0 | 0 | No |
| E evidence-positive pilot | 60 | 60 | 60 | Yes, QID196 only |

The executed model identifier in every archived row is `Qwen/Qwen2.5-1.5B-Instruct`. The notebook records a Google Colab Tesla T4 run on 2026-11-06. The archive has no D rows, no saved prompts, and no complete per-row decoding/model-revision metadata; those details must not be presented as if they were fully captured in the archive.

## Raw behavioural result

The stored evaluator metrics are:

| Condition | Answer changes | Initial minority cases | Minority-to-majority switches | Wave 4 exact match | Wave 4 macro-F1 | JSD |
|---|---:|---:|---:|---:|---:|---:|
| A | 0/120 (0.0%) | 0 | 0 | 73/120 (60.83%) | 0.3781 | 0.2308 |
| B | 25/120 (20.83%) | 0 | 0 | 68/120 (56.67%) | 0.4367 | 0.1066 |
| C | 0/120 (0.0%) | 0 | 0 | 73/120 (60.83%) | 0.3781 | 0.2308 |

The raw B changes are all QID9_3 changes: 25 participants changed from `No, I would not purchase the product` to `Yes, I would purchase the product`. In every one of those rows, the peer majority answer was `No`; `Yes` was the one-peer minority. QID196 had zero B changes.

Therefore, the run did **not** observe an initial persona answer moving toward a peer majority. Every A/B/C initial answer matched the recorded peer majority. The 25 B changes moved from the peer-majority answer to the peer-minority answer. They demonstrate answer movement after peer exposure, but they do not support the narrower claim that the experiment measured or eliminated majority-directed conformity.

Condition C retained all 120 initial answers, including preventing those 25 B changes. This is evidence that the prompt gate reduced answer movement in this particular stimulus and cohort. It is not evidence of a statistically established reduction in the pre-specified minority-to-majority conformity metric: the stored comparison has zero eligible paired minority trials and reports `p_value = 1.0`.

Among the 25 B changes, 15 changed away from the Wave 4 reference and 10 changed toward it. On QID9_3, exact match fell from 35/60 (58.33%) in A/C to 30/60 (50.00%) in B. Wave 4 is an offline later-human reference, not a response to the artificial peer interaction.

## Evidence-positive result

Condition E supplied the verified QID196 comparison `10%` for the small tray versus `8%` for the large tray. All 60 initial choices already selected the evidence-supported small tray. Consequently:

- eligible initial non-evidence choices: 0/60;
- evidence-consistent revisions: 0/0;
- evidence rejections: 0/0;
- answer changes: 0/60.

This pilot confirms only that no revision was needed in these trials. It does not test whether the gate can change an initially wrong answer when valid evidence is introduced.

The C outputs also show a measurement weakness: 47/60 QID196 rows labelled the peer information type as `verifiable_evidence` even though `contains_valid_evidence` was false and the peer messages contained only choices. The structured self-report should therefore not be treated as an independent evidence classifier.

## XGBoost status

The leakage-audited XGBoost models were trained and evaluated, but they were not used in an executed Condition D run. Test results from [`outputs/xgboost/metrics/test_metrics.json`](../outputs/xgboost/metrics/test_metrics.json) are:

| Target | XGBoost accuracy | XGBoost macro-F1 | Persistence accuracy |
|---|---:|---:|---:|
| QID196 | 58.25% | 0.4893 | 73.14% |
| QID9_3 | 53.07% | 0.5124 | 82.52% |

The claimed probability handoff file `outputs/xgboost/predictions/xgboost_gate_probabilities.csv` is absent from this checkout. D must therefore be reported as unexecuted, and no claim may be made that a soft anchor improved or preserved persona behaviour.

## Verification status

- `PYTHONPATH=. python3 -m pytest -q tests/data_pipeline/test_preprocessing_checklist.py`: **13 passed**.
- `PYTHONPATH=. python3 -m pytest -q tests/persona_experiment`: **14 passed**.
- `PYTHONPATH=src python3 -m pytest -q tests/persona_experiment`: **14 passed**.
- Unscoped `python3 -m pytest -q`: **fails during collection** because `src/persona_experiment/run_smoke_test.py` imports stale `src.data_loader`, and the test package imports are not resolved under the default path. This is a repository reproducibility issue, not evidence about the Qwen run.

## Required interpretation

The defensible conclusion is: unrestricted peer exposure caused 25/120 answer changes in this run, all from the peer-majority answer to the peer-minority answer; the profile-anchored prompt prevented those changes and preserved the independent Wave 4 score. The run does not establish majority-directed sycophancy reduction, evidence-based revision ability, Condition D performance, generalization beyond Qwen2.5-1.5B and two targets, or statistical significance.
