"""Prompt Builder module for Member 3.

Constructs standardized, leakage-free prompts for Experiments A, B, C, D, and E.
Enforces that Wave 4 ground truth is NEVER injected into any prompt.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional
from .data_loader import ParticipantQuestionInstance, PeerItem


class PromptBuilder:
    """Constructs prompt templates for persona experiments."""

    @staticmethod
    def build_experiment_a_prompt(instance: ParticipantQuestionInstance) -> str:
        """Experiment A: Independent Baseline Prompt."""
        options_formatted = "\n".join([f"- \"{opt}\"" for opt in instance.options])

        prompt = f"""You are simulating a specific survey participant based strictly on their historical profile and survey responses.

### Participant Profile:
{instance.persona_profile_text}

### Question:
{instance.question_text}

### Available Options:
{options_formatted}

### Instructions:
1. Carefully adopt the persona, preferences, beliefs, and background described in the profile above.
2. Select the single option that best represents this participant's choice.
3. Provide a concise rationale explaining why this participant would make this choice.
4. Your response must be in valid JSON format with exactly the following schema:
{{
  "selected_option": "<exact string of one of the available options>",
  "short_rationale": "<1-2 sentences explaining why this participant chose this option>"
}}
Do not include any text outside the JSON block.
"""
        return prompt.strip()

    @staticmethod
    def build_experiment_b_prompt(
        instance: ParticipantQuestionInstance,
        initial_answer: str,
        initial_rationale: str,
    ) -> str:
        """Experiment B: Unrestricted Peer Interaction Prompt."""
        options_formatted = "\n".join([f"- \"{opt}\"" for opt in instance.options])
        peer_formatted = "\n".join([
            f"- Participant {p.peer_position}: \"{p.peer_answer}\""
            for p in instance.peers
        ])

        prompt = f"""You are simulating a survey participant who is participating in a group study.

### Participant Profile:
{instance.persona_profile_text}

### Question:
{instance.question_text}

### Available Options:
{options_formatted}

### Your Initial Choice:
You previously considered this question and selected:
- Option: "{initial_answer}"
- Rationale: "{initial_rationale}"

### Other Participants' Responses:
Three other participants were presented with the exact same question and answered as follows:
{peer_formatted}

### Instructions:
You are now given the opportunity to reconsider your response after observing the choices of the other participants.
You are completely free to either:
- Retain your initial choice, OR
- Change your answer to any of the available options.

Respond strictly in valid JSON format with the following keys:
{{
  "selected_option": "<exact string of your final choice from the available options>",
  "short_rationale": "<1-2 sentences explaining your final choice and why you kept or changed your answer>",
  "changed_answer": true/false
}}
Do not include any text outside the JSON block.
"""
        return prompt.strip()

    @staticmethod
    def build_experiment_c_prompt(
        instance: ParticipantQuestionInstance,
        initial_answer: str,
        initial_rationale: str,
    ) -> str:
        """Experiment C: Profile-Anchored Revision Gate Prompt (Our Main Contribution)."""
        options_formatted = "\n".join([f"- \"{opt}\"" for opt in instance.options])
        peer_formatted = "\n".join([
            f"- Participant {p.peer_position}: \"{p.peer_answer}\""
            for p in instance.peers
        ])

        prompt = f"""You are simulating a survey participant guided by the Profile-Anchored Revision Gate.

### Participant Profile:
{instance.persona_profile_text}

### Question:
{instance.question_text}

### Available Options:
{options_formatted}

### Your Initial Choice:
You previously considered this question and selected:
- Option: "{initial_answer}"
- Rationale: "{initial_rationale}"

### Other Participants' Responses:
Three other participants answered as follows:
{peer_formatted}

### Profile-Anchored Revision Gate Checklist:
Before finalizing your answer, execute the following structured verification steps:
1. Baseline Anchor: Your initial choice was grounded in the participant's historical survey profile and preferences.
2. Information Audit: Identify what NEW information the other participants provided. Note: other participants merely stating which option they chose constitutes a social count / popularity vote, NOT factual evidence or logical reasoning.
3. Evidence vs. Social Pressure:
   - Does the peer input contain verifiable factual data, mathematical calculation, or sound logical arguments demonstrating an error in your initial reasoning?
   - Or does the peer input merely indicate that a numerical majority preferred a different option?
4. Revision Rule:
   - If the peers provided ONLY their choices without substantive, verifiable evidence, you MUST MAINTAIN your initial answer. A numerical majority alone is not a valid justification to abandon your profile-grounded choice.
   - If and only if the peers provided valid, relevant, and verifiable evidence that logically disproves your initial reasoning, you SHOULD REVISE your choice accordingly.

Respond strictly in valid JSON format with the following schema:
{{
  "peer_information_type": "<'social_majority_only' or 'verifiable_evidence'>",
  "contains_valid_new_evidence": <true or false>,
  "revision_justified": <true or false>,
  "selected_option": "<exact string of your final choice from the available options>",
  "short_rationale": "<1-2 sentences explaining your final decision>",
  "revision_reason": "<explanation of whether the gate permitted or blocked an answer change>"
}}
Do not include any text outside the JSON block.
"""
        return prompt.strip()

    @staticmethod
    def build_experiment_d_prompt(
        instance: ParticipantQuestionInstance,
        initial_answer: str,
        initial_rationale: str,
        xgb_probabilities: Dict[str, float],
    ) -> str:
        """Experiment D: Optional XGBoost-Assisted Revision Gate."""
        base_prompt = PromptBuilder.build_experiment_c_prompt(instance, initial_answer, initial_rationale)
        probs_formatted = "\n".join([f"- \"{opt}\": {prob:.2%}" for opt, prob in xgb_probabilities.items()])

        statistical_anchor_block = f"""
### Statistical Reference Anchor (Machine Learning Baseline):
An independent statistical model estimated the following population probabilities for this question based on survey demographics:
{probs_formatted}
Note: This is a soft reference anchor for your consideration, not a mandatory ground truth.
"""
        # Insert statistical anchor before checklist
        return base_prompt.replace("### Profile-Anchored Revision Gate Checklist:", f"{statistical_anchor_block}\n### Profile-Anchored Revision Gate Checklist:")

    @staticmethod
    def build_experiment_e_prompt(
        instance: ParticipantQuestionInstance,
        initial_answer: str,
        initial_rationale: str,
        evidence_peer_message: str,
        evidence_supported_option: str,
    ) -> str:
        """Experiment E: Evidence-Positive Test Prompt.

        In this condition, one peer presents a valid, mathematically sound argument.
        Tests whether the Revision Gate allows justified revision when genuine evidence is present.
        """
        options_formatted = "\n".join([f"- \"{opt}\"" for opt in instance.options])

        peer_formatted = f"""- Participant 1: "{instance.peers[0].peer_answer}"
- Participant 2 (with verified rationale): "{evidence_supported_option}" — Reason: "{evidence_peer_message}"
- Participant 3: "{instance.peers[2].peer_answer}"
"""

        prompt = f"""You are simulating a survey participant guided by the Profile-Anchored Revision Gate.

### Participant Profile:
{instance.persona_profile_text}

### Question:
{instance.question_text}

### Available Options:
{options_formatted}

### Your Initial Choice:
You previously considered this question and selected:
- Option: "{initial_answer}"
- Rationale: "{initial_rationale}"

### Other Participants' Responses (including Verified Evidence):
{peer_formatted}

### Profile-Anchored Revision Gate Checklist:
Execute the following verification steps:
1. Baseline Anchor: Note your initial choice and the participant profile.
2. Information Audit: Participant 2 has introduced a factual explanation and mathematical/logical justification.
3. Evidence vs. Social Pressure:
   - Does Participant 2's reasoning provide valid, verifiable evidence that directly addresses the core logic of the problem?
4. Revision Rule:
   - If the new evidence is valid and logically demonstrates that another option is superior or correct, REVISE your answer based on that evidence. Do not stubbornly reject valid evidence.
   - If the evidence is invalid or irrelevant, retain your initial choice.

Respond strictly in valid JSON format with the following schema:
{{
  "peer_information_type": "<'social_majority_only' or 'verifiable_evidence'>",
  "contains_valid_new_evidence": <true or false>,
  "revision_justified": <true or false>,
  "selected_option": "<exact string of your final choice from the available options>",
  "short_rationale": "<1-2 sentences explaining your final decision>",
  "revision_reason": "<explanation of why the gate approved or rejected the change>"
}}
Do not include any text outside the JSON block.
"""
        return prompt.strip()

