# Profile-Anchored Revision for Synthetic Personas

## Research question
Can a profile-anchored revision protocol reduce unsupported peer-majority conformity in LLM-based synthetic personas without preventing evidence-based revision or substantially damaging fidelity to later human responses?

This project operationalizes one sycophancy-related behaviour: changing toward a peer majority without relevant new evidence. Agreement and answer changes are not automatically sycophancy. The intervention must permit justified revision.

## Research basis
1. Toubia et al. (2025), *Twin-2K-500*, Marketing Science 44(6), 1446–1455. DOI: https://doi.org/10.1287/mksc.2025.0262. Dataset: https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500
2. Choi et al. (2025), *An Empirical Study of Group Conformity in Multi-Agent Systems*, Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.265/
3. Pitre et al. (2025), *CONSENSAGENT*, Findings of ACL 2025. https://aclanthology.org/2025.findings-acl.1141/

See `context/02_research_background.md` for what is borrowed and what is not.

## Conditions
| Condition | Procedure |
|---|---|
| A: Independent | Persona answers from profile, without peer opinions. |
| B: Unrestricted | Same initial answer; expose controlled peer majority; allow free reconsideration. |
| C: Revision gate | Same initial answer and peer messages as B; require distinguishing evidence from social pressure. |
| D: Optional ablation | C plus an XGBoost probability distribution as a soft anchor. |
| Evidence-positive pilot | Add valid, independently verified evidence to test whether justified revisions remain possible. |

## Critical data rules
- Inspect the official dataset configuration and actual schema first. Never assume field names, question IDs, item mappings, response types, or sample counts.
- Never expose a focal participant's held-out target answer or Wave 4 answer to the persona.
- Never use the focal participant as their own peer.
- Keep participant-level train/validation/test splits disjoint.
- Match exact question wording, response options, and experimental variants.
- Wave 4 is a later human response/test-retest reference, **not** a response to our artificial peer interaction.
- Peer answers are sourced from other participants' earlier responses. A 2:1 majority is an experimental manipulation, not a natural population estimate.

## Stack
Python; Hugging Face Datasets/Transformers; Qwen2.5-1.5B-Instruct (fallback 0.5B); XGBoost (optional); pandas, NumPy, scikit-learn, SciPy, Matplotlib. No fine-tuning initially. Test inference on Colab/Kaggle; free GPU availability is not guaranteed.

## Setup
```bash
git clone <YOUR_REPOSITORY_URL>
cd persona-sycophancy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Copy `configs/experiment.example.yaml` to `configs/experiment.yaml` after Member 1 verifies the data and question IDs. **This has been done: the two verified targets (`QID196`, `QID9_3`) are already set in the example config.**

### Data access
All files under `data/` are gitignored. Fetch and build once per machine:

```bash
pip install huggingface_hub pandas pyarrow scikit-learn
python scripts/download_raw_data.py                      # raw Twin-2K-500 files -> data/raw/
python src/data_pipeline/build_processed_dataset.py      # processed files    -> data/processed/
python -m pytest tests/data_pipeline/test_preprocessing_checklist.py   # 13 checks, all must pass
```

Column meaning, design rules and usage recipes for every processed file:
**`context/data_dictionary.md`.** Integrity facts (audit): `notebooks/data-audit.ipynb`.

## Processed dataset summary (verified, Member 1 handoff v1)
- Dataset `LLM-Digital-Twin/Twin-2K-500`, config `wave_split`; 2,058 unique participants.
- Core targets: **QID196** (marble-tray question, constant wording) and **QID9_3**
  (purchase decision, **per-participant randomized passage**, 434 distinct variants —
  use each participant's own passage from `data/processed/target_question_variants.csv`).
- Splits: seed 42, stratified 70/15/15 → 1,440 / 309 / 309 (participant-disjoint).
- Peer groups: train-only pool, exactly 2 majority : 1 minority, same group reused for B and C.
- XGBoost: keep profile-only (634 columns) and expanded (+126) feature sets separate;
  per-model exclusion lists are in `data/preprocessing_manifest.json`.
- Full details: `context/data_dictionary.md`.

## Repository structure and ownership
```text
persona-sycophancy/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/experiment.example.yaml
├── context/
│   ├── 00_project_overview.md
│   ├── 01_theory_and_definitions.md
│   ├── 02_research_background.md
│   ├── 03_data_member.md
│   ├── 04_prediction_member.md
│   ├── 05_agent_experiment_member.md
│   ├── 06_shared_contracts.md
│   ├── 07_experiment_protocol.md
│   ├── 08_progress_tracker.md
│   ├── 09_team_workflow.md
│   ├── data_dictionary.md            # processed-data usage doc (read this first)
│   └── agents/
│       ├── shared_agent_context.md
│       ├── member_1_data_agent.md
│       ├── member_2_prediction_agent.md
│       └── member_3_agent_experiment_agent.md
├── data/
│   ├── raw/                          # gitignored; scripts/download_raw_data.py
│   ├── preprocessing_manifest.json   # seed / mappings / column roles / checksums
│   └── processed/                    # gitignored; src/data_pipeline/build_processed_dataset.py
│       ├── participant_splits.csv
│       ├── persona_profiles.csv
│       ├── target_questions.csv
│       ├── target_question_variants.csv
│       ├── human_answers.csv
│       ├── peer_responses.csv
│       ├── xgboost_features.csv
│       └── xgboost_targets.csv
├── notebooks/dataset_audit.ipynb     # executed audit (data-audit.ipynb)
├── src/
│   ├── common/                 # shared contracts/config only
│   ├── data_pipeline/          # Member 1 owns this
│   ├── prediction/             # Member 2 owns this
│   ├── persona_experiment/     # Member 3 owns this
│   └── evaluation/             # Member 3 owns this
├── tests/{data_pipeline,prediction,persona_experiment,integration}/
├── scripts/
│   └── download_raw_data.py
└── results/{agent_outputs,predictions,metrics,figures}/
```

| Member | Owns | Handoff |
|---|---|---|
| 1: Data | `src/data_pipeline/` and its tests | Verified processed CSVs + audit manifest |
| 2: Prediction | `src/prediction/` and its tests | XGBoost metrics + optional probability CSV |
| 3: Agents/integration | `src/persona_experiment/`, `src/evaluation/` and tests | Agent runs + experiment metrics |

Do not import another member's private helper functions. Integrate through the files and contracts in `context/06_shared_contracts.md`. Shared contract changes require team agreement.

## Execution sequence
1. Member 1 audits Twin-2K-500, maps repeated questions, selects eligible items, creates participant splits and exports profiles/targets/peer pool.
2. Member 2 trains an optional XGBoost baseline using only permitted features and participant splits.
3. Member 3 implements A/B/C conditions, then optionally integrates D and the evidence-positive pilot.
4. Run a 5-participant pilot; check leakage, answer parsing, question mapping, and runtime.
5. Set final N from actual eligibility and runtime. A previously discussed 60 participants × 2 questions is only a planning estimate, not a confirmed sample.
6. Freeze settings and run final evaluation by Day 4. Day 5 is documentation only.

## Metrics
XGBoost: accuracy, macro-F1, log-loss, confusion matrix, majority-class baseline.
Persona: answer-change rate, minority-to-majority switch rate, unsupported conformity rate, evidence-consistent revision rate, Wave 4 exact match, macro-F1, Jensen-Shannon divergence, and human test-retest match reference.

Do not call a lower answer-change rate a success by itself. Report denominators, exclusions, class balance, failed generations, runtime, and null results. Do not claim statistical significance without a valid test.

## Day plan
| Day | Work |
|---|---|
| 1 | M1 data audit/splits; M3 model smoke test and repo setup |
| 2 | M1 data handoff; M2 XGBoost; M3 A/B pipeline |
| 3 | M3 revision gate, integration, evidence-positive pilot |
| 4 | Pilot fixes, final runs, metrics, plots, freeze |
| 5 | Documentation only |

## Integrity
This is a focused empirical project, not a claim to solve sycophancy generally. Do not invent data fields, sample sizes, results, rationales, evidence, or citations. Record limitations and dropped components.
