# Member 1 — Dataset and Data Pipeline

## Ownership
Own `src/data_pipeline/`, `tests/data_pipeline/`, dataset audit notebook, and data documentation. Do not edit prediction or agent internals.

## Objective
Export verified, leakage-safe data files so other members consume files rather than your private functions.

## Dataset
Twin-2K-500: https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500
Paper: https://doi.org/10.1287/mksc.2025.0262
Use official configuration/data as source of truth.

## Tasks
1. Load and inspect actual configs, fields, types, IDs, row counts, question text/options, response encodings, wave mappings, missingness, and task variants.
2. Identify earlier non-target information permitted in persona profiles.
3. Select two target questions only if exact earlier-wave peer responses and Wave 4 repeated-item mappings are verified and answer options are stable.
4. Preserve condition/variant mappings. Do not mix task conditions.
5. Create participant-disjoint train/validation/test splits (70/15/15 is a proposal, not mandatory).
6. Exclude target and Wave 4 target answers from persona profiles.
7. Build peer pool from other participants; focal participant must never be their own peer.
8. Export files defined in `06_shared_contracts.md` and a data audit manifest.

## Deliverables
Actual repository exports are `target_questions.csv`, `participant_splits.csv`, `persona_profiles.csv`, `human_answers.csv`, `peer_responses.csv`, `target_question_variants.csv`, `xgboost_features.csv`, `xgboost_targets.csv`, and `data/preprocessing_manifest.json`.

## Done when
- [x] Actual schema and wave mappings documented.
- [x] Target questions and variants justified.
- [x] Participant splits reproducible and disjoint.
- [x] Leakage/self-peer checks pass.
- [x] Output files validate against shared contracts.
- [x] Counts, missingness, exclusions, IDs, seed, and caveats recorded in progress tracker.

Raw data should not be committed unless license and team policy permit it.
