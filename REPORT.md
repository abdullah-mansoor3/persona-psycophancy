# Profile-Anchored Revision for Synthetic Personas

## Research summary

### Abstract

This project studies a narrow, observable form of LLM sycophancy-related behaviour: whether a profile-grounded synthetic persona changes its survey answer after seeing peer choices, and whether a profile-anchored revision prompt limits changes that are not supported by new evidence. The design was motivated by the attached literature review, which separates factual correction from social accommodation, warns that average persona similarity can hide weak individual fidelity, and treats multi-agent agreement as an outcome that needs explanation rather than automatic proof of quality.

The repository uses the Twin-2K-500 data, two binary target questions, participant-disjoint preprocessing, and a single-round Qwen2.5-1.5B-Instruct experiment. The archived run contains 60 held-out participants, 120 A/B/C question instances per condition, and 60 QID196 evidence-positive trials. All 420 archived generations parsed successfully. In unrestricted peer interaction, 25/120 instances changed answer, all on QID9_3. Every initial answer already matched the constructed peer majority, and all 25 changes moved to the peer minority. The profile-anchored condition produced no changes and matched the independent baseline on Wave 4 exact match (60.83%).

These results show that the gate reduced answer movement in this cohort and stimulus. They do not establish that the experiment observed majority-directed sycophancy, that the gate improves evidence-based revision, that Condition D works, or that the effect is statistically generalizable. The evidence-positive pilot had no initially unsupported choices, so its revision ability was not tested.

This project is about LLM sycophancy and synthetic personas. It does not operationalize or measure psychopathy.

## 1. Motivation and literature connection

The literature review frames three related problems: user-facing and social sycophancy, fidelity of synthetic respondents to human response patterns, and behaviour that emerges when several language-model agents interact. Its central methodological point is important for this project: an answer change can be a useful correction, a response to valid evidence, or harmful social accommodation. These cases should not be counted as one construct.

The project therefore uses the following operational distinction:

- **Answer movement:** the final response differs from the persona's initial response.
- **Unsupported conformity:** movement toward the peer majority without relevant new evidence.
- **Evidence-based revision:** movement supported by relevant, valid information.
- **Fidelity:** agreement with the participant's later Wave 4 response, used only as an offline reference.

The experiment measures outputs under controlled prompts. It does not infer an internal psychological state.

## 2. Research question and hypotheses

The intended question was:

> Can a profile-anchored revision protocol reduce unsupported peer-majority conformity in synthetic personas without blocking justified revision or damaging fidelity to later human responses?

The executed evidence supports only a narrower version:

> In a 60-participant Qwen2.5-1.5B-Instruct run on two target items, does a profile-anchored prompt reduce answer movement after a fixed three-peer exposure?

The majority-directed conformity hypothesis could not be tested properly because every initial A/B/C answer matched the peer majority. The evidence-positive hypothesis could not be tested because all initial QID196 answers already matched the verified mathematical answer.

## 3. Data and preprocessing

The data pipeline was built from Twin-2K-500 `wave_split` and verified in the processed-data artifacts.

| Split | Participants | Role |
|---|---:|---|
| Train | 1,440 | peer pool and model training |
| Validation | 309 | model selection |
| Test | 309 | held-out evaluation |
| Total | 2,058 | participant universe |

The split is participant-disjoint, uses seed 42, and is stratified on the joint Wave 4 labels for QID196 and QID9_3. The selected targets are:

1. **QID196:** a constant-wording marble-tray choice: 1/10 black marbles versus 8/100. The small tray has the higher probability, 10% versus 8%.
2. **QID9_3:** a binary purchase decision with a participant-specific product/price passage. There are 434 distinct passages, so the exact wording is loaded per participant from `target_question_variants.csv`.

Both targets are complete in the earlier and Wave 4 data for all 2,058 participants. The profile text excludes the target questions and target answer payloads under the phrase-level leakage scan. Peer groups use only training participants, exclude the focal participant, contain two majority and one minority response, and reuse the same group for Conditions B and C.

Evidence: [`data/preprocessing_manifest.json`](data/preprocessing_manifest.json), [`src/data_pipeline/build_processed_dataset.py`](src/data_pipeline/build_processed_dataset.py), [`notebooks/data-audit.ipynb`](notebooks/data-audit.ipynb), and [`context/10_empirical_results_audit.md`](context/10_empirical_results_audit.md).

## 4. Experimental method

### Conditions

| Condition | Procedure | Executed? |
|---|---|---|
| A: Independent | Profile and exact question only; produce an initial choice. | Yes |
| B: Unrestricted | Show the initial choice and three peer choices; allow free reconsideration. | Yes |
| C: Profile-anchored gate | Show the same initial choice and peer group; require an evidence-versus-social-signal audit. | Yes |
| D: Gate + XGBoost | Add XGBoost probabilities as a soft reference. | No |
| E: Evidence-positive | Add a verified QID196 probability comparison. | Yes, QID196 only |

The runner reuses the same A-generated initial answer for B, C, D, and E. The peer messages are generated from the same processed peer group, and the prompts present peer choices without fabricated rationales. The C prompt explicitly states that a numerical majority is not evidence and that revision is allowed when relevant, verifiable evidence disproves the initial reasoning.

The stored archive contains run records and evaluator outputs, but not the full prompts or complete per-row decoding metadata. The notebook records a Google Colab Tesla T4 run using `Qwen/Qwen2.5-1.5B-Instruct` on 2026-11-06. No fine-tuning was used.

## 5. Results

### 5.1 Run coverage

| Condition | Question instances | Valid parses | Change count |
|---|---:|---:|---:|
| A | 120 | 120 (100%) | 0 |
| B | 120 | 120 (100%) | 25 |
| C | 120 | 120 (100%) | 0 |
| D | 0 | 0 | Not run |
| E | 60 QID196 | 60 (100%) | 0 |

The archive therefore contains 420 valid generations, not 420 comparable A/B/C/D/E instances.

### 5.2 Answer movement and peer direction

| Metric | A | B | C |
|---|---:|---:|---:|
| Answer changes | 0/120 (0.0%) | 25/120 (20.83%) | 0/120 (0.0%) |
| Initial minority cases | 0 | 0 | 0 |
| Minority-to-majority switches | 0 | 0 | 0 |
| Stored unsupported-conformity metric | 0.0% | 0.0% | 0.0% |

The raw direction matters more than the aggregate change rate. All 25 B changes occurred on QID9_3 and changed from `No, I would not purchase the product` to `Yes, I would purchase the product`. In every changed row, `No` was the peer-majority answer and `Yes` was the one-peer minority answer. QID196 had no B changes.

Thus, B exposed answer movement after peer choices, but it did not produce an observed movement toward the peer majority. The current stored statistical comparison correctly found zero eligible initial-minority pairs and reports no inferential result (`p_value = 1.0`). The earlier claim of a McNemar result with 25 majority-directed switches is not supported by the archived data and has been removed.

Condition C prevented all 25 B changes. This is a descriptive paired contrast showing that the gate made this model/cohort stable under the supplied prompt. It should not be described as a confirmed reduction in majority-directed sycophancy.

### 5.3 Fidelity to Wave 4

Wave 4 is a later human response and test-retest reference. It is not a human response to the artificial peer interaction.

| Condition | Exact matches | Exact-match rate | Macro-F1 | JSD | Human earlier-to-Wave 4 match |
|---|---:|---:|---:|---:|---:|
| A | 73/120 | 60.83% | 0.3781 | 0.2308 | 75.00% |
| B | 68/120 | 56.67% | 0.4367 | 0.1066 | 75.00% |
| C | 73/120 | 60.83% | 0.3781 | 0.2308 | 75.00% |

By target, QID196 exact match was 38/60 (63.33%) in all A/B/C conditions. QID9_3 exact match was 35/60 (58.33%) in A and C, and 30/60 (50.00%) in B. Of the 25 B changes, 15 moved away from the Wave 4 answer and 10 moved toward it. This means unrestricted peer exposure reduced the observed QID9_3 fidelity in this run, while C preserved the A result.

The lower JSD for B is not automatically better: it reflects the generated distribution becoming closer to the Wave 4 aggregate distribution while individual exact match declined. This is consistent with the literature review's warning that population-level similarity and individual fidelity are different goals.

### 5.4 Evidence-positive pilot

Condition E supplied the correct QID196 comparison: 10% for the small tray versus 8% for the large tray. All 60 initial answers already selected the small tray. The result was therefore 0 eligible revisions, 0 evidence-consistent revisions, and 0 evidence rejections. The pilot did not test whether the gate can correct an initially wrong answer.

There is an additional measurement warning: 47/60 QID196 C outputs labelled the peer information type as `verifiable_evidence` even though the stored `contains_valid_evidence` flag was false and the peer messages contained only choices. The model's self-reported evidence category is not an independent validation of the stimulus.

### 5.5 XGBoost baseline and Condition D

The leakage-audited XGBoost models were trained, but D was not executed. The probability handoff file referenced by the report is absent from the checkout. The available held-out test metrics are:

| Target | Accuracy | Macro-F1 | Balanced accuracy | Persistence accuracy |
|---|---:|---:|---:|---:|
| QID196 | 58.25% | 0.4893 | 51.01% | 73.14% |
| QID9_3 | 53.07% | 0.5124 | 51.49% | 82.52% |

The baseline is useful as an audit artifact, but it underperforms the persistence baseline on both targets. No conclusion about a soft XGBoost anchor can be drawn.

## 6. Discussion

1. **The model was item-sensitive.** QID196 showed no answer movement. QID9_3 showed 25 unrestricted changes. This is a difference between two items and their prompt content, not a general estimate of model sycophancy.
2. **The peer-direction result is the central correction.** The experiment's constructed peer majority was already the model's initial answer for all A/B/C rows. The observed B switches went to the peer minority, so the run did not test the planned minority-to-majority mechanism.
3. **The gate reduced movement.** C retained the initial answer in all 120 cases and preserved the A Wave 4 score. This supports a stability effect under the exact prompt and sample, not a general mitigation claim.
4. **Stability and rationality are different.** The E pilot had no initially wrong cases. The gate's ability to accept valid evidence remains untested.
5. **Aggregate distribution metrics need context.** B had lower JSD but lower exact match. A synthetic population can look more like the human aggregate while becoming less faithful at the individual level.
6. **The XGBoost component remains incomplete.** The model artifacts exist, but the D interaction was not run and the model did not beat simple persistence on held-out accuracy.

## 7. Limitations

- **Construct validity:** the run did not contain initial minority answers, so majority-directed conformity was not observed. Answer movement alone cannot be relabelled as sycophancy.
- **Evidence-positive validity:** no trial required an evidence-based change. The gate may still be overly stubborn.
- **Small and selective evaluation:** only the first 60 sorted test participants were run, not all 309 test participants.
- **Model coverage:** only Qwen2.5-1.5B-Instruct was evaluated. Results may change with model size, family, decoding, or provider.
- **Item coverage:** only QID196 and QID9_3 were tested. QID9_3 has randomized passages, and the two items differ sharply in content and response type.
- **Single round:** the protocol does not model multi-round persuasion, coalition formation, or long-term opinion dynamics.
- **Evaluation reference:** Wave 4 is useful for test-retest-style fidelity, but it is not a counterfactual post-peer human response.
- **Automated/self-reported labels:** evidence type and revision justification are generated by the model and were not independently judged for every row.
- **Metadata and reproducibility:** the archive lacks full prompts and complete decoding/model revision metadata. The local CLI also has stale import/path assumptions; targeted suites pass, but an unscoped root pytest fails during collection.
- **No Condition D:** the missing probability handoff prevents any claim about XGBoost-assisted gating.
- **No psychopathy construct:** despite the broader course framing, this repository contains no psychopathy scale, behavioural diagnosis, or psychopathy-specific experiment.

## 8. Key points for the research paper

These are the defensible points to state:

1. We built a leakage-controlled pipeline for profile-grounded synthetic survey personas using 2,058 Twin-2K-500 participants and participant-disjoint train/validation/test splits.
2. We selected two complete binary targets, including a constant-wording probability item and a participant-specific purchase item with 434 wording variants.
3. We compared independent, unrestricted peer-exposure, and profile-anchored revision prompts using the same held-out participants and fixed 2:1 peer groups.
4. In the archived Qwen2.5-1.5B run, unrestricted peer exposure caused 25/120 answer changes, all on QID9_3.
5. All initial answers already matched the constructed peer majority; all 25 changes moved to the peer minority. Therefore, the run demonstrates peer-exposure sensitivity, not majority-directed sycophancy reduction.
6. The profile-anchored prompt prevented all 25 observed changes and preserved the independent Wave 4 exact-match rate of 60.83%.
7. The unrestricted condition reduced QID9_3 Wave 4 exact match from 58.33% to 50.00%; this is an association in a small controlled run, not a causal population estimate.
8. The evidence-positive pilot was inconclusive because all 60 initial answers were already correct; evidence-based revision was not tested.
9. The XGBoost baseline did not beat persistence and was not integrated into an executed D condition.
10. The main contribution is a reproducible evaluation design that separates answer movement, social conformity, evidence-based revision, and human-response fidelity.

## 9. Reproducibility and artifact map

- Raw Qwen records and stored metrics: [`Member_3_Real_Qwen_Results.zip`](Member_3_Real_Qwen_Results.zip), [`results/metrics/condition_metrics_summary.json`](results/metrics/condition_metrics_summary.json), [`results/metrics/statistical_comparison_b_vs_c.json`](results/metrics/statistical_comparison_b_vs_c.json), [`results/metrics/evidence_metrics.json`](results/metrics/evidence_metrics.json).
- Data construction and audit: [`src/data_pipeline/build_processed_dataset.py`](src/data_pipeline/build_processed_dataset.py), [`data/preprocessing_manifest.json`](data/preprocessing_manifest.json), [`notebooks/data-audit.ipynb`](notebooks/data-audit.ipynb).
- Persona prompts and runner: [`src/persona_experiment/prompt_builder.py`](src/persona_experiment/prompt_builder.py), [`src/persona_experiment/experiment_runner.py`](src/persona_experiment/experiment_runner.py), [`src/persona_experiment/revision_gate.py`](src/persona_experiment/revision_gate.py).
- Prediction baseline: [`outputs/xgboost/metrics/test_metrics.json`](outputs/xgboost/metrics/test_metrics.json), [`outputs/xgboost/audit/leakage_check.txt`](outputs/xgboost/audit/leakage_check.txt).
- Evidence-backed conflict register: [`context/10_empirical_results_audit.md`](context/10_empirical_results_audit.md).

The focused verification commands and their results are recorded in the audit file. The attached literature review should be cited for background claims, while all project-result claims should be tied to the repository artifacts above.
