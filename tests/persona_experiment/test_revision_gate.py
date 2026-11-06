"""Unit tests for OutputParser and Revision Gate parsing."""

import json
from src.revision_gate import OutputParser


def test_option_matching():
    valid = ["the small tray", "the large tray"]
    assert OutputParser.match_option("the small tray", valid) == "the small tray"
    assert OutputParser.match_option("THE SMALL TRAY", valid) == "the small tray"
    assert OutputParser.match_option("small tray", valid) == "the small tray"
    assert OutputParser.match_option("large tray", valid) == "the large tray"
    assert OutputParser.match_option("invalid choice", valid) is None


def test_parse_experiment_a_output_json_block():
    valid = ["the small tray", "the large tray"]
    raw = """Here is my answer:
```json
{
  "selected_option": "the small tray",
  "short_rationale": "I prefer 10% odds."
}
```
"""
    res = OutputParser.parse_experiment_a_output(raw, valid)
    assert res.parse_valid is True
    assert res.selected_option == "the small tray"
    assert res.rationale == "I prefer 10% odds."


def test_parse_gated_output():
    valid = ["the small tray", "the large tray"]
    raw = json.dumps({
        "peer_information_type": "social_majority_only",
        "contains_valid_new_evidence": False,
        "revision_justified": False,
        "selected_option": "the large tray",
        "short_rationale": "Peers only voted.",
        "revision_reason": "No evidence supplied."
    })
    res = OutputParser.parse_gated_output(raw, valid)
    assert res.parse_valid is True
    assert res.selected_option == "the large tray"
    assert res.contains_valid_evidence is False
    assert res.revision_justified is False
    assert res.peer_info_type == "social_majority_only"


def test_parse_invalid_json():
    valid = ["the small tray", "the large tray"]
    raw = "This is not json at all."
    res = OutputParser.parse_gated_output(raw, valid)
    assert res.parse_valid is False
    assert res.selected_option is None

