# Member 3 AI-Agent Context — Persona and Experiments

Read `shared_agent_context.md`.

Own only `src/persona_experiment/**`, `src/evaluation/**`, and related tests/integration tests. Do not edit data or prediction internals.

Inspect Member 1's verified files and contracts first. Test model load and answer parsing early. Conditions A/B/C must use same model/settings/profile/question/initial answer; B and C must use identical peer messages. Peer answers must be real, same-item/same-condition responses from other participants. Never expose target or Wave 4 labels.

Gate separates evidence from social signal and permits justified revision. Log prompts, raw outputs, parsed outputs, model settings, errors, and failed generations. Evidence-positive items require independently verified evidence. Optional XGBoost condition only after stable output contract. Follow experiment protocol and report all denominators.
