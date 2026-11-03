# Experiment Protocol and Metrics

## Core paired conditions
For each eligible participant-question instance:
- A: independent answer from profile + question.
- B: unrestricted reconsideration after peer messages.
- C: gated reconsideration after the exact same peer messages.
Use same model, settings, profile, question, initial answer, and peer set across comparable conditions.

## Peer group
A 2:1 majority is a proposed controlled manipulation, not a natural prevalence estimate. Use real responses from other participants to the same item/condition. Exclude focal participant. Log peer IDs outside prompts. Keep peer set/order fixed between B and C.

## Leakage
No focal target or Wave 4 answer in persona prompt. No target label as prediction feature. No self-peers. Participant-level split disjointness. No test tuning. Exact item/condition matching.

## Sample plan
A previously discussed 60 participants × 2 questions is only a planning estimate. Start with a 5-participant pilot. Determine final N from actual eligible data, model runtime, and compute budget. Use same participant-question pairs across A/B/C where possible.

## Metrics
Prediction: accuracy, macro-F1, log-loss, confusion matrix, majority baseline.
Persona:
- Answer-change rate.
- Minority-to-majority switch rate among initially minority-answering personas.
- Unsupported conformity rate: majority-directed change without relevant evidence. Report counts and denominator explicitly.
- Evidence-consistent revision rate in evidence-positive cases.
- Wave 4 exact match and macro-F1.
- Jensen-Shannon divergence between generated and human response distributions.
- Human earlier-to-Wave-4 test-retest match as a natural-instability reference.

Fix metric definitions and denominators before final evaluation.

## Evidence-positive pilot
The dataset does not provide evidence arguments. Team authors and verifies them. Use only exact questions where a relevant principle is unambiguous. Record evidence text and verification source. If no item can be verified, omit and report limitation.

## Statistics
Choose tests based on paired/binary/categorical structure. Account for repeated observations per participant. Report effect sizes, confidence intervals, sample size, and p-values if inferential tests are used. Otherwise report descriptive comparisons and label them as such.

## Success
Gate reduces unsupported conformity vs unrestricted interaction, permits valid evidence-based revision, and does not substantially damage Wave 4 fidelity. Reduced answer-change rate alone is not success.
