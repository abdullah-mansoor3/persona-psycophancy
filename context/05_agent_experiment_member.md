# Member 3 — Persona, Peer Interaction, Revision Gate, Integration

## Ownership
Own `src/persona_experiment/`, `src/evaluation/`, related tests, and integration. Do not edit Member 1 or Member 2 internals.

## Model
Primary: `Qwen/Qwen2.5-1.5B-Instruct`; fallback: `Qwen/Qwen2.5-0.5B-Instruct`. Test immediately in Colab/Kaggle; free GPU is not guaranteed. No fine-tuning initially.

## Persona
Input: allowed non-target profile + exact question/options. Output: normalized answer, concise rationale, model/decoding metadata, parse status. Never include focal target or Wave 4 answer.

## Conditions
A. Independent: profile + question, no peers.
B. Unrestricted: same initial answer; controlled peer opinions; free reconsideration.
C. Revision gate: same initial answer and identical peer messages as B; apply evidence-versus-social-signal checklist.
D. Optional: C plus XGBoost probability distribution as soft anchor, only if stable.

## Peer rules
Use actual earlier answers from other participants to the same item and condition. A 2:1 majority is a controlled manipulation. Exclude focal participant. Messages state selected option only; no fabricated rationales. Keep peer set/order identical in B and C.

## Evidence-positive pilot
Dataset provides responses, not ready-made arguments. Team must author and independently verify evidence. Use only exact items with an unambiguous relevant principle. A conjunction probability constraint may fit a suitable Linda item, but verify wording/options. If one item only, label it proof-of-concept; if none can be verified, omit and document.

## Controls and logging
Same model/version/settings/profile/question/initial answer/peer messages across conditions; one revision round. Save prompts, raw outputs, parsed answers, rationale, condition, run ID, model revision, decoding settings, parse errors. Do not silently discard failures.

## Done when
- [ ] Model loads and structured output validates.
- [ ] A/B/C run end-to-end.
- [ ] B/C receive identical peer messages.
- [ ] No target/Wave 4 leakage.
- [ ] Gate allows justified revisions.
- [ ] Pilot and final outputs/metrics are reproducible.
