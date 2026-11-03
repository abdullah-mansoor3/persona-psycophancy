# Team Workflow and Change Control

## Branches
- `member-1/data-pipeline`
- `member-2/xgboost`
- `member-3/persona-experiments`
- `docs/integration`

Work on branches; merge through PRs.

## Ownership
M1: `src/data_pipeline/`, its tests and data docs.
M2: `src/prediction/`, its tests and prediction docs.
M3: `src/persona_experiment/`, `src/evaluation/`, their tests.
All: context docs, README, agreed config, shared contracts.

## PR checklist
State what changed, affected contract/output, tests run, dependency changes, cross-module impact, and evidence for any result.

## Integration rules
Consume files/contracts, not another member's private functions. Keep raw data and large model caches out of Git. Use reproducible seeds and write config/metadata beside every run.

## Shared changes
Propose change, identify affected modules, get owner agreement, update contract/version, update tests and consumers together.

## AI coding workflow
Give the agent `context/agents/shared_agent_context.md`, its member-specific context, the task, and current progress. Restrict edits to owned paths. Require tests and a change summary. Agent must inspect existing code first and must not invent schema, results, sample sizes, or citations.
