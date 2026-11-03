# Shared AI-Agent Context

You assist one member of a three-person research project. Read this and the relevant member context before editing.

## Goal
Test whether a Profile-Anchored Revision Gate reduces unsupported peer-majority conformity in LLM synthetic personas while preserving evidence-based revision and later human-response fidelity.

## Construct
We measure answer movement toward a peer majority without relevant new evidence. Do not label every agreement/change as sycophancy. The gate must not force stubbornness.

## Sources
Twin-2K-500: https://doi.org/10.1287/mksc.2025.0262
Dataset: https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500
Choi et al.: https://aclanthology.org/2025.findings-acl.265/
CONSENSAGENT: https://aclanthology.org/2025.findings-acl.1141/

## Critical constraints
- Inspect real schema before field-dependent code.
- Never guess IDs, mappings, response types, sample counts, or results.
- Wave 4 is later human response/test-retest reference, not post-peer human behaviour.
- No target/Wave 4 leakage into prompts/features.
- No self-peers; participant-level splits disjoint.
- Peer ratio is experimental manipulation.
- XGBoost is optional and not a sycophancy detector.
- Do not fabricate evidence, rationales, citations, metrics, or significance.

## Coding rules
Respect ownership in `context/09_team_workflow.md` and interfaces in `context/06_shared_contracts.md`. Do not edit another member's module or shared contracts unilaterally. Prefer small changes and tests. Avoid unnecessary frameworks/dependencies. Keep raw data/model caches out of Git.

## Before editing
Inspect repository/tests, identify allowed files, verify input schema/contracts, state assumptions.

## After editing
Run relevant tests; list changed files, commands/results, unresolved issues. Never claim unexecuted work is complete. Synthetic fixtures must be labelled as test fixtures.
