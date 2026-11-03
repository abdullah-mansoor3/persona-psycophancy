# Research Background

## 1. Twin-2K-500
Toubia et al. (2025), “Twin-2K-500: A Data Set for Building Digital Twins of over 2,000 People Based on Their Answers to over 500 Questions,” *Marketing Science*, 44(6), 1446–1455. DOI: https://doi.org/10.1287/mksc.2025.0262

- Dataset: https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500
- Authors' code: https://github.com/tianyipeng-lab/Digital-Twin-Simulation

Use: participant survey information to ground personas; selected repeated Wave 4 tasks as later human response reference. Not every item is necessarily repeated. Inspect official configuration, exact item mapping, question variants, and schema. Wave 4 is not a response to our experimental peers.

## 2. Group conformity in multi-agent systems
Choi et al. (2025), “An Empirical Study of Group Conformity in Multi-Agent Systems,” *Findings of ACL 2025*, 5123–5139. https://aclanthology.org/2025.findings-acl.265/

Use: empirical precedent for measuring stance alignment with a numerical majority. Our adaptation uses participant-grounded personas and controlled peer answers; we do not reproduce their debate setup.

## 3. CONSENSAGENT
Pitre et al. (2025), “CONSENSAGENT: Towards Efficient and Effective Consensus in Multi-Agent LLM Interactions Through Sycophancy Mitigation,” *Findings of ACL 2025*, 22112–22133. https://aclanthology.org/2025.findings-acl.1141/

Use: motivation for structured interaction/prompt-level mitigation. We do not reproduce their dynamic prompt-refinement framework.

## Defensible contribution
Combined controlled evaluation of profile-grounded personas, peer answers sourced from other human participants, numerical-majority pressure, an evidence-gated revision protocol, and joint measurement of conformity and later human-response fidelity.

Do not claim invention of digital twins, conformity experiments, or sycophancy mitigation. Verify all bibliographic details and paper-specific claims against the official publications before submission.
