"""Profile-Anchored Revision Gate implementation and output parsing for Member 3.

Handles JSON extraction, option normalization, and verification of gate decisions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ParsedAnswer:
    selected_option: Optional[str]
    rationale: str
    parse_valid: bool
    error_message: Optional[str] = None
    # Gate-specific fields
    peer_info_type: Optional[str] = None
    contains_valid_evidence: Optional[bool] = None
    revision_justified: Optional[bool] = None
    revision_reason: Optional[str] = None
    raw_json: Optional[Dict[str, Any]] = None


class OutputParser:
    """Robust parser for LLM-generated JSON outputs."""

    @staticmethod
    def extract_json_block(raw_text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON object from raw response text."""
        if not raw_text or not raw_text.strip():
            return None

        text = raw_text.strip()

        # Handle ```json ... ``` code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try parsing whole text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try finding outer braces
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except json.JSONDecodeError:
                pass

        return None

    @classmethod
    def match_option(cls, candidate: str, valid_options: List[str]) -> Optional[str]:
        """Normalize candidate string to exact matching option from valid_options."""
        if not candidate:
            return None

        candidate_clean = candidate.strip()

        # 1. Exact match
        for opt in valid_options:
            if candidate_clean == opt:
                return opt

        # 2. Case-insensitive exact match
        for opt in valid_options:
            if candidate_clean.lower() == opt.lower():
                return opt

        # 3. Substring match (e.g. "small tray" in "the small tray")
        for opt in valid_options:
            if opt.lower() in candidate_clean.lower() or candidate_clean.lower() in opt.lower():
                return opt

        return None

    @classmethod
    def parse_experiment_a_output(cls, raw_text: str, valid_options: List[str]) -> ParsedAnswer:
        """Parse Experiment A output."""
        data = cls.extract_json_block(raw_text)
        if not data:
            return ParsedAnswer(
                selected_option=None,
                rationale="",
                parse_valid=False,
                error_message="Failed to parse valid JSON from model output.",
            )

        raw_opt = str(data.get("selected_option", ""))
        matched_opt = cls.match_option(raw_opt, valid_options)

        if not matched_opt:
            return ParsedAnswer(
                selected_option=None,
                rationale=str(data.get("short_rationale", "")),
                parse_valid=False,
                error_message=f"Model selected invalid option: '{raw_opt}'. Valid: {valid_options}",
                raw_json=data,
            )

        return ParsedAnswer(
            selected_option=matched_opt,
            rationale=str(data.get("short_rationale", "")),
            parse_valid=True,
            raw_json=data,
        )

    @classmethod
    def parse_experiment_b_output(cls, raw_text: str, valid_options: List[str]) -> ParsedAnswer:
        """Parse Experiment B output."""
        data = cls.extract_json_block(raw_text)
        if not data:
            return ParsedAnswer(
                selected_option=None,
                rationale="",
                parse_valid=False,
                error_message="Failed to parse valid JSON from model output.",
            )

        raw_opt = str(data.get("selected_option", ""))
        matched_opt = cls.match_option(raw_opt, valid_options)

        if not matched_opt:
            return ParsedAnswer(
                selected_option=None,
                rationale=str(data.get("short_rationale", "")),
                parse_valid=False,
                error_message=f"Model selected invalid option: '{raw_opt}'. Valid: {valid_options}",
                raw_json=data,
            )

        return ParsedAnswer(
            selected_option=matched_opt,
            rationale=str(data.get("short_rationale", "")),
            parse_valid=True,
            raw_json=data,
        )

    @classmethod
    def parse_gated_output(cls, raw_text: str, valid_options: List[str]) -> ParsedAnswer:
        """Parse Experiment C, D, or E output (Revision Gate)."""
        data = cls.extract_json_block(raw_text)
        if not data:
            return ParsedAnswer(
                selected_option=None,
                rationale="",
                parse_valid=False,
                error_message="Failed to parse valid JSON from model output.",
            )

        raw_opt = str(data.get("selected_option", ""))
        matched_opt = cls.match_option(raw_opt, valid_options)

        if not matched_opt:
            return ParsedAnswer(
                selected_option=None,
                rationale=str(data.get("short_rationale", "")),
                parse_valid=False,
                error_message=f"Model selected invalid option: '{raw_opt}'. Valid: {valid_options}",
                raw_json=data,
            )

        return ParsedAnswer(
            selected_option=matched_opt,
            rationale=str(data.get("short_rationale", "")),
            parse_valid=True,
            peer_info_type=data.get("peer_information_type"),
            contains_valid_evidence=bool(data.get("contains_valid_new_evidence")),
            revision_justified=bool(data.get("revision_justified")),
            revision_reason=str(data.get("revision_reason", "")),
            raw_json=data,
        )

