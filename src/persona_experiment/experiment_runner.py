"""Experiment Runner for Member 3.

Coordinates execution across experimental conditions:
- Condition A: Independent Baseline
- Condition B: Unrestricted Peer Interaction
- Condition C: Profile-Anchored Revision Gate (Our Contribution)
- Condition D: Optional XGBoost-Assisted Gate
- Condition E: Evidence-Positive Pilot

Guarantees identical initial answers and identical peer messages between B and C.
Strictly logs all runs into JSONL and tabular CSV.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_loader import Member3DataLoader, ParticipantQuestionInstance
from .llm_provider import LLMProvider
from .prompt_builder import PromptBuilder
from .revision_gate import OutputParser

logger = logging.getLogger(__name__)


# Evidence-positive argument definition for Experiment E
EVIDENCE_DEFINITIONS = {
    "QID196": {
        "supported_option": "the small tray",
        "evidence_argument": (
            "The small tray has 1 black marble out of 10 (1/10 = 10.0% probability), "
            "whereas the large tray has 8 black marbles out of 100 (8/100 = 8.0% probability). "
            "Because 10.0% > 8.0%, drawing from the small tray offers a strictly higher probability of winning."
        ),
    }
}


@dataclass
class RunRecord:
    run_id: str
    participant_id: int
    question_id: str
    condition: str  # "A", "B", "C", "D", "E"
    model_id: str
    split: str
    initial_answer: str
    initial_rationale: str
    peer_majority_answer: str
    peer_minority_answer: str
    peer_answers_json: str
    final_answer: str
    final_rationale: str
    changed_answer: bool
    revision_reason: Optional[str]
    peer_info_type: Optional[str]
    contains_valid_evidence: Optional[bool]
    revision_justified: Optional[bool]
    evidence_supported_option: Optional[str]
    wave4_ground_truth: str
    wave1_3_human_answer: str
    parse_valid: bool
    error: Optional[str]
    latency_ms: float
    timestamp: str


class ExperimentRunner:
    """Orchestrates multi-condition experiments."""

    def __init__(
        self,
        data_loader: Member3DataLoader,
        llm_provider: LLMProvider,
        config: Dict[str, Any],
    ):
        self.data_loader = data_loader
        self.llm = llm_provider
        self.config = config

        self.runs: List[RunRecord] = []

    def run_all(
        self,
        split: str = "test",
        max_participants: Optional[int] = None,
        enable_a: bool = True,
        enable_b: bool = True,
        enable_c: bool = True,
        enable_d: bool = False,
        enable_e: bool = True,
    ) -> List[RunRecord]:
        """Run all enabled experimental conditions."""
        target_questions = self.config.get("data", {}).get("target_questions", ["QID196", "QID9_3"])
        profile_max_chars = self.config.get("data", {}).get("profile_context_max_chars", 12000)

        instances = self.data_loader.get_eligible_instances(
            split=split,
            target_questions=target_questions,
            max_participants=max_participants,
            profile_max_chars=profile_max_chars,
        )

        logger.info(f"Starting experiments on {len(instances)} participant-question pairs...")
        start_all = time.time()

        for idx, inst in enumerate(instances, 1):
            logger.info(
                f"[{idx}/{len(instances)}] Processing Participant {inst.participant_id}, Question {inst.question_id}"
            )

            # ---------------------------------------------------------
            # Condition A: Independent Baseline
            # ---------------------------------------------------------
            init_answer = ""
            init_rationale = ""

            if enable_a or enable_b or enable_c:
                prompt_a = PromptBuilder.build_experiment_a_prompt(inst)
                res_a = self.llm.generate(prompt_a)
                parsed_a = OutputParser.parse_experiment_a_output(res_a.raw_text, inst.options)

                if parsed_a.parse_valid and parsed_a.selected_option:
                    init_answer = parsed_a.selected_option
                    init_rationale = parsed_a.rationale
                else:
                    logger.warning(
                        f"Failed to parse Exp A for PID {inst.participant_id}, {inst.question_id}: {parsed_a.error_message}"
                    )
                    init_answer = inst.options[0]  # Fallback to prevent cascade crash
                    init_rationale = "Fallback option"

                if enable_a:
                    self.runs.append(
                        RunRecord(
                            run_id=f"A_{inst.participant_id}_{inst.question_id}_{uuid.uuid4().hex[:6]}",
                            participant_id=inst.participant_id,
                            question_id=inst.question_id,
                            condition="A",
                            model_id=res_a.model_id,
                            split=inst.split,
                            initial_answer=init_answer,
                            initial_rationale=init_rationale,
                            peer_majority_answer=inst.get_peer_majority_answer(),
                            peer_minority_answer=inst.get_peer_minority_answer(),
                            peer_answers_json=json.dumps([p.peer_answer for p in inst.peers]),
                            final_answer=init_answer,
                            final_rationale=init_rationale,
                            changed_answer=False,
                            revision_reason=None,
                            peer_info_type=None,
                            contains_valid_evidence=None,
                            revision_justified=None,
                            evidence_supported_option=None,
                            wave4_ground_truth=inst.wave4_ground_truth,
                            wave1_3_human_answer=inst.wave1_3_human_answer,
                            parse_valid=parsed_a.parse_valid,
                            error=parsed_a.error_message,
                            latency_ms=res_a.latency_ms,
                            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        )
                    )

            # ---------------------------------------------------------
            # Condition B: Unrestricted Peer Interaction
            # ---------------------------------------------------------
            if enable_b:
                prompt_b = PromptBuilder.build_experiment_b_prompt(
                    inst, initial_answer=init_answer, initial_rationale=init_rationale
                )
                res_b = self.llm.generate(prompt_b)
                parsed_b = OutputParser.parse_experiment_b_output(res_b.raw_text, inst.options)

                final_b = parsed_b.selected_option if (parsed_b.parse_valid and parsed_b.selected_option) else init_answer
                changed_b = final_b != init_answer

                self.runs.append(
                    RunRecord(
                        run_id=f"B_{inst.participant_id}_{inst.question_id}_{uuid.uuid4().hex[:6]}",
                        participant_id=inst.participant_id,
                        question_id=inst.question_id,
                        condition="B",
                        model_id=res_b.model_id,
                        split=inst.split,
                        initial_answer=init_answer,
                        initial_rationale=init_rationale,
                        peer_majority_answer=inst.get_peer_majority_answer(),
                        peer_minority_answer=inst.get_peer_minority_answer(),
                        peer_answers_json=json.dumps([p.peer_answer for p in inst.peers]),
                        final_answer=final_b,
                        final_rationale=parsed_b.rationale,
                        changed_answer=changed_b,
                        revision_reason=None,
                        peer_info_type=None,
                        contains_valid_evidence=None,
                        revision_justified=None,
                        evidence_supported_option=None,
                        wave4_ground_truth=inst.wave4_ground_truth,
                        wave1_3_human_answer=inst.wave1_3_human_answer,
                        parse_valid=parsed_b.parse_valid,
                        error=parsed_b.error_message,
                        latency_ms=res_b.latency_ms,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
                )

            # ---------------------------------------------------------
            # Condition C: Profile-Anchored Revision Gate (Our Intervention)
            # ---------------------------------------------------------
            if enable_c:
                prompt_c = PromptBuilder.build_experiment_c_prompt(
                    inst, initial_answer=init_answer, initial_rationale=init_rationale
                )
                res_c = self.llm.generate(prompt_c)
                parsed_c = OutputParser.parse_gated_output(res_c.raw_text, inst.options)

                final_c = parsed_c.selected_option if (parsed_c.parse_valid and parsed_c.selected_option) else init_answer
                changed_c = final_c != init_answer

                self.runs.append(
                    RunRecord(
                        run_id=f"C_{inst.participant_id}_{inst.question_id}_{uuid.uuid4().hex[:6]}",
                        participant_id=inst.participant_id,
                        question_id=inst.question_id,
                        condition="C",
                        model_id=res_c.model_id,
                        split=inst.split,
                        initial_answer=init_answer,
                        initial_rationale=init_rationale,
                        peer_majority_answer=inst.get_peer_majority_answer(),
                        peer_minority_answer=inst.get_peer_minority_answer(),
                        peer_answers_json=json.dumps([p.peer_answer for p in inst.peers]),
                        final_answer=final_c,
                        final_rationale=parsed_c.rationale,
                        changed_answer=changed_c,
                        revision_reason=parsed_c.revision_reason,
                        peer_info_type=parsed_c.peer_info_type,
                        contains_valid_evidence=parsed_c.contains_valid_evidence,
                        revision_justified=parsed_c.revision_justified,
                        evidence_supported_option=None,
                        wave4_ground_truth=inst.wave4_ground_truth,
                        wave1_3_human_answer=inst.wave1_3_human_answer,
                        parse_valid=parsed_c.parse_valid,
                        error=parsed_c.error_message,
                        latency_ms=res_c.latency_ms,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
                )

            # ---------------------------------------------------------
            # Condition E: Evidence-Positive Testing (Where Feasible)
            # ---------------------------------------------------------
            if enable_e and inst.question_id in EVIDENCE_DEFINITIONS:
                ev_def = EVIDENCE_DEFINITIONS[inst.question_id]
                ev_text = ev_def["evidence_argument"]
                ev_opt = ev_def["supported_option"]

                prompt_e = PromptBuilder.build_experiment_e_prompt(
                    inst,
                    initial_answer=init_answer,
                    initial_rationale=init_rationale,
                    evidence_peer_message=ev_text,
                    evidence_supported_option=ev_opt,
                )
                res_e = self.llm.generate(prompt_e)
                parsed_e = OutputParser.parse_gated_output(res_e.raw_text, inst.options)

                final_e = parsed_e.selected_option if (parsed_e.parse_valid and parsed_e.selected_option) else init_answer
                changed_e = final_e != init_answer

                self.runs.append(
                    RunRecord(
                        run_id=f"E_{inst.participant_id}_{inst.question_id}_{uuid.uuid4().hex[:6]}",
                        participant_id=inst.participant_id,
                        question_id=inst.question_id,
                        condition="E",
                        model_id=res_e.model_id,
                        split=inst.split,
                        initial_answer=init_answer,
                        initial_rationale=init_rationale,
                        peer_majority_answer=inst.get_peer_majority_answer(),
                        peer_minority_answer=inst.get_peer_minority_answer(),
                        peer_answers_json=json.dumps([p.peer_answer for p in inst.peers]),
                        final_answer=final_e,
                        final_rationale=parsed_e.rationale,
                        changed_answer=changed_e,
                        revision_reason=parsed_e.revision_reason,
                        peer_info_type=parsed_e.peer_info_type,
                        contains_valid_evidence=parsed_e.contains_valid_evidence,
                        revision_justified=parsed_e.revision_justified,
                        evidence_supported_option=ev_opt,
                        wave4_ground_truth=inst.wave4_ground_truth,
                        wave1_3_human_answer=inst.wave1_3_human_answer,
                        parse_valid=parsed_e.parse_valid,
                        error=parsed_e.error_message,
                        latency_ms=res_e.latency_ms,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
                )

        elapsed = time.time() - start_all
        logger.info(f"Finished {len(self.runs)} runs across conditions in {elapsed:.2f}s")
        return self.runs

    def export_results(self, output_dir: str | Path) -> Tuple[Path, Path]:
        """Save runs as JSONL and CSV."""
        out_path = Path(output_dir).resolve()
        agent_out = out_path / "agent_outputs"
        agent_out.mkdir(parents=True, exist_ok=True)

        jsonl_file = agent_out / "agent_runs.jsonl"
        csv_file = agent_out / "agent_runs.csv"

        logger.info(f"Writing runs to {jsonl_file} and {csv_file}")

        dicts = []
        for r in self.runs:
            d = asdict(r)
            # Ensure all numpy types are converted to standard Python types
            d["participant_id"] = int(d["participant_id"])
            if d.get("contains_valid_evidence") is not None:
                d["contains_valid_evidence"] = bool(d["contains_valid_evidence"])
            if d.get("revision_justified") is not None:
                d["revision_justified"] = bool(d["revision_justified"])
            dicts.append(d)

        with open(jsonl_file, "w", encoding="utf-8") as f:
            for d in dicts:
                f.write(json.dumps(d, default=str) + "\n")

        df = pd.DataFrame(dicts)
        df.to_csv(csv_file, index=False)

        return jsonl_file, csv_file
