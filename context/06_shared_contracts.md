# Shared Contracts

These are proposed project interfaces, not claims about the dataset's existing columns. Member 1 must inspect the source and finalize actual mappings. Do not change a shared field without team agreement.

## General
Use UTF-8 CSV for tabular handoffs and JSON/JSONL for manifests and runs. Use stable project IDs; never rely on row order. Preserve source item IDs, wave, condition/variant, original answer labels, and normalized values. Do not include personal identifiers in prompts or committed results.

## Proposed processed files

### `questions.csv`
`question_id`, `source_question_id`, `wave_source`, `question_text`, `options_json`, `answer_type`, `condition_id`, `wave4_question_id`, `eligible_for_core`, `selection_notes`.

### `participant_splits.csv`
`participant_id`, `split`, `split_seed`, `split_version`. Every participant belongs to exactly one partition. If a peer reference pool is separate, document its disjointness.

### `persona_profiles.csv`
`participant_id`, `profile_json`, `profile_source_waves`, `profile_version`. Must exclude focal target and Wave 4 target answers.

### `human_ground_truth.csv`
`participant_id`, `question_id`, `source_question_id`, `condition_id`, `wave4_answer`, `wave4_answer_normalized`, `is_missing`. Evaluation-only; never prompt input.

### `peer_responses.csv`
`peer_participant_id`, `question_id`, `source_question_id`, `condition_id`, `peer_answer`, `peer_answer_normalized`, `source_wave`. Same item/condition; focal participant excluded during sampling.

### `data_audit.json`
Dataset/config/version, actual source fields/types, counts, missingness, selected items/mappings, answer options, split seed/counts, leakage checks, exclusions, limitations.

## Prediction output
Suggested `results/predictions/xgb_probabilities.csv`:
`participant_id`, `question_id`, `model_version`, `option_id`, `probability`, `split`.
One row per option. Probabilities per participant/question sum approximately to 1. Option IDs match `questions.csv`.

## Agent output
Suggested JSONL `results/agent_outputs/agent_runs.jsonl`, one object per run:
`run_id`, `participant_id`, `question_id`, `condition`, `model_id`, `model_revision`, `decoding_config`, `initial_answer`, `initial_rationale`, `peer_answers`, `final_answer`, `final_rationale`, `revision_reason`, `evidence_classification`, `raw_output`, `parse_valid`, `error`, `prompt_version`.

## Code boundaries
Shared code in `src/common/` should contain only small shared dataclasses, validators, and config readers. Each member module should run independently via CLI and consume files, not private functions from another member. Avoid circular imports. Contract changes require agreement and tests.
