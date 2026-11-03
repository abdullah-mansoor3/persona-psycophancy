# Project Overview

## Aim
Test whether a **Profile-Anchored Revision Gate** reduces unsupported peer-majority conformity in profile-grounded synthetic personas while preserving evidence-based revision and fidelity to later human responses.

## Research question
Can a profile-anchored revision protocol reduce unsupported conformity in LLM-based synthetic human personas without preventing evidence-based revision or substantially damaging fidelity to later human responses?

## Definitions
- Synthetic persona/digital twin: an LLM prompted with participant-specific information to approximate that participant's responses.
- Unsupported conformity: moving toward a peer majority without relevant new evidence.
- Evidence-based revision: changing because relevant, valid information warrants a different answer.
- Revision gate: a structured prompt procedure that asks the persona to distinguish evidence from social pressure.
- Fidelity: similarity between persona answers and the participant's later recorded answers.

## Scope
Core: construct personas; record initial answers; expose controlled peer answers; compare unrestricted and gated revision; evaluate conformity and Wave 4 fidelity.
Optional: XGBoost soft anchor; evidence-positive pilot.
Out of scope: foundation-model training, production app, claim of solving sycophancy generally.

## Team
- Member 1: dataset audit, cleaning, splits, persona profiles, peer pool.
- Member 2: optional XGBoost baseline and evaluation.
- Member 3: persona model, peer interaction, revision gate, experiment runner, integration.

## Non-negotiable rules
No guessed schema; no target/Wave 4 leakage; no self-peers; participant-disjoint splits; no fabricated data/results; Wave 4 is not post-peer human behaviour.
