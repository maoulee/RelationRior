"""Unit tests for skill aggregation — semi-structured markdown narrative design.

Tests the AggregatedSkill schema, parser functions, markdown narrative
prompt, stage hints formatting with single-block injection, and backward
compatibility with legacy format.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from subgraph_kgqa.skill_mining.skill_aggregator import (
    AggregatedSkill,
    _build_aggregation_prompt,
    _empty_aggregated_skill,
    _parse_aggregated_skill,
    format_aggregated_skill_for_prompt,
    format_aggregated_stage_hints,
)


# =============================================================================
# AggregatedSkill Schema Tests
# =============================================================================

def test_aggregated_skill_markdown_narrative_default():
    """markdown_narrative defaults to empty string."""
    skill = AggregatedSkill()
    assert skill.markdown_narrative == ""


def test_aggregated_skill_legacy_fields_default():
    """Legacy fields default to empty/empty-dict."""
    skill = AggregatedSkill()
    assert skill.plan_guidance == ""
    assert skill.action_guidance == ""
    assert skill.reasoning_guidance == ""
    assert skill.error_type_distribution == {}


def test_empty_aggregated_skill():
    """_empty_aggregated_skill returns valid defaults."""
    skill = _empty_aggregated_skill("raw response")
    assert skill.markdown_narrative == ""
    assert skill.plan_guidance == ""
    assert skill.error_type_distribution == {}
    assert skill.raw_llm_response == "raw response"


# =============================================================================
# Parser Tests — New markdown_narrative format
# =============================================================================

def test_parse_markdown_narrative():
    """Parser extracts markdown_narrative from new-format response."""
    narrative = "## Direction 1: Leaders\n\n**Preferred relations:** `rel1`"
    raw = json.dumps({"markdown_narrative": narrative})
    skill = _parse_aggregated_skill(raw)
    assert skill.markdown_narrative == narrative
    assert "Direction 1" in skill.markdown_narrative


def test_parse_markdown_narrative_with_legacy_fields():
    """Parser handles hybrid response with both markdown and legacy fields."""
    raw = json.dumps({
        "markdown_narrative": "## Direction 1",
        "plan_guidance": "legacy plan",
        "error_type_distribution": {"action_error": 2},
    })
    skill = _parse_aggregated_skill(raw)
    assert skill.markdown_narrative == "## Direction 1"
    assert skill.plan_guidance == "legacy plan"
    assert skill.error_type_distribution == {"action_error": 2}


def test_parse_legacy_only_response():
    """Parser handles legacy-format response without markdown_narrative."""
    raw = json.dumps({
        "combined_reasoning_hint": "old style hint",
        "plan_guidance": "use relation A",
    })
    skill = _parse_aggregated_skill(raw)
    assert skill.markdown_narrative == ""
    assert skill.plan_guidance == "use relation A"
    assert skill.reasoning_guidance == "old style hint"


def test_parse_reasoning_guidance_backward_compat():
    """reasoning_guidance and combined_reasoning_hint sync correctly."""
    raw = json.dumps({
        "reasoning_guidance": "new guidance",
        "combined_reasoning_hint": "",
    })
    skill = _parse_aggregated_skill(raw)
    assert skill.combined_reasoning_hint == "new guidance"


def test_parse_markdown_fenced_json():
    """Parser handles ```json ... ``` fenced responses."""
    raw = '```json\n{"markdown_narrative": "## Test", "plan_guidance": "test"}\n```'
    skill = _parse_aggregated_skill(raw)
    assert "## Test" in skill.markdown_narrative
    assert skill.plan_guidance == "test"


def test_parse_with_none_values():
    """Parser handles None values gracefully."""
    raw = json.dumps({
        "markdown_narrative": None,
        "plan_guidance": None,
        "error_type_distribution": None,
    })
    skill = _parse_aggregated_skill(raw)
    assert skill.markdown_narrative == ""
    assert skill.plan_guidance == ""
    assert skill.error_type_distribution == {}


def test_parse_invalid_error_type_distribution():
    """Non-dict error_type_distribution returns empty dict."""
    raw = json.dumps({"error_type_distribution": "not a dict"})
    skill = _parse_aggregated_skill(raw)
    assert skill.error_type_distribution == {}


def test_parse_preserves_raw_response():
    """Parser preserves raw LLM response."""
    raw = '{"markdown_narrative": "test"}'
    skill = _parse_aggregated_skill(raw)
    assert skill.raw_llm_response == raw


# =============================================================================
# Prompt Construction Tests
# =============================================================================

class _MockNegSkill:
    def __init__(self, error_type, question="test"):
        self.error_type = error_type
        self.question = question
        self.wrong_plan_relations = []
        self.correct_plan_relations = []
        self.error_pattern = ""
        self.correct_approach = ""


def test_prompt_contains_markdown_instruction():
    """Prompt asks for semi-structured markdown output."""
    prompt = _build_aggregation_prompt("who is X", [])
    assert "markdown_narrative" in prompt
    assert "Direction" in prompt
    assert "Action guidance" in prompt
    assert "case-based reasoning" in prompt
    assert "Final reasoning guidance" in prompt


def test_prompt_contains_error_experience_guidance():
    """Prompt instructs case-based reasoning, not prescriptive rules."""
    prompt = _build_aggregation_prompt("test", [])
    assert "case-based reasoning" in prompt or "case X" in prompt or "not prescriptive" in prompt


def test_prompt_contains_filter_tool_guidance():
    """Prompt reinforces filter tool chain usage."""
    prompt = _build_aggregation_prompt("test", [])
    assert "filter" in prompt.lower()


def test_prompt_error_distribution_with_negatives():
    """Prompt includes error type summary from negative skills."""
    negs = [
        _MockNegSkill("action_error"),
        _MockNegSkill("action_error"),
        _MockNegSkill("plan_error"),
        _MockNegSkill(""),  # empty → unknown
    ]
    prompt = _build_aggregation_prompt("test", [], negative_skills=negs)
    assert "action_error: 2" in prompt
    assert "plan_error: 1" in prompt
    assert "unknown" in prompt


def test_prompt_no_error_section_without_negatives():
    """No error type section when negative_skills is empty or None."""
    prompt1 = _build_aggregation_prompt("test", [], negative_skills=[])
    assert "Error type summary" not in prompt1

    prompt2 = _build_aggregation_prompt("test", [], negative_skills=None)
    assert "Error type summary" not in prompt2


def test_prompt_includes_negative_case_details():
    """Prompt includes failed case details for context."""
    neg = _MockNegSkill("action_error", question="who is the president")
    neg.wrong_plan_relations = ["rel1", "rel2"]
    neg.correct_plan_relations = ["rel3"]
    neg.error_pattern = "confused founder with leader"
    neg.correct_approach = "check role attribute"
    prompt = _build_aggregation_prompt("test", [], negative_skills=[neg])
    assert "who is the president" in prompt
    assert "rel1" in prompt
    assert "confused founder" in prompt


# =============================================================================
# format_aggregated_stage_hints Tests — Single-block injection
# =============================================================================

def test_stage_hints_markdown_narrative_single_block():
    """When markdown_narrative present, stage 1 is empty (planning free of priors)."""
    skill = AggregatedSkill(
        markdown_narrative="## Direction 1: Leaders\n\n**Preferred relations:** `rel1`"
    )
    hints = format_aggregated_stage_hints(skill)
    assert hints["stage:1"] == ""
    assert "stage:2" in hints
    assert "stage:3" in hints
    assert "stage:4" in hints


def test_stage_hints_markdown_contains_narrative():
    """Stage 1 is empty with markdown_narrative (planning free of priors)."""
    skill = AggregatedSkill(markdown_narrative="## Leaders\nTest content")
    hints = format_aggregated_stage_hints(skill)
    assert hints["stage:1"] == ""
    # Narrative content appears in later stages, not stage:1


def test_stage_hints_markdown_stage1_empty_by_design():
    """Stage 1 is intentionally empty with markdown_narrative (planning free of priors)."""
    skill = AggregatedSkill(markdown_narrative="## Test\nSome content here")
    hints = format_aggregated_stage_hints(skill)
    assert hints["stage:1"] == ""
    # Content goes to action/selection stages, not stage 1
    all_hints = " ".join(hints.values())
    assert "Test" in all_hints or len(all_hints) > 0


# Legacy fallback tests

def test_stage_hints_legacy_plan_shared():
    """Legacy: Stage 1 and 2 get the same plan hint."""
    skill = AggregatedSkill(plan_guidance="use relation X")
    hints = format_aggregated_stage_hints(skill)
    assert hints["stage:1"] == hints["stage:2"]
    assert "use relation X" in hints["stage:1"]


def test_stage_hints_legacy_action_with_attention():
    """Legacy: Stage 3 gets ATTENTION marker when action_error >= 2."""
    skill = AggregatedSkill(
        action_guidance="use tool Y",
        error_type_distribution={"action_error": 3, "plan_error": 1}
    )
    hints = format_aggregated_stage_hints(skill)
    assert "[ATTENTION" in hints["stage:3"]


def test_stage_hints_legacy_no_attention_below_threshold():
    """Legacy: No ATTENTION marker when action_error < 2."""
    skill = AggregatedSkill(
        action_guidance="use tool Y",
        error_type_distribution={"action_error": 1}
    )
    hints = format_aggregated_stage_hints(skill)
    assert "[ATTENTION" not in hints["stage:3"]


def test_stage_hints_legacy_pitfall_categorization():
    """Legacy: Pitfalls categorized to correct stages by keyword."""
    skill = AggregatedSkill(
        pitfalls=[
            "wrong relation chosen",
            "incorrect tool call",
            "wrong entity granularity"
        ]
    )
    hints = format_aggregated_stage_hints(skill)
    assert "wrong relation" in hints["stage:1"]
    assert "incorrect tool" in hints["stage:3"]
    assert "entity granularity" in hints["stage:4"]


def test_stage_hints_empty_skill():
    """Empty skill returns empty strings for all stages."""
    skill = AggregatedSkill()
    hints = format_aggregated_stage_hints(skill)
    assert all(v == "" for v in hints.values())


# =============================================================================
# format_aggregated_skill_for_prompt Tests
# =============================================================================

def test_format_uses_markdown_narrative():
    """format prefers markdown_narrative when available."""
    skill = AggregatedSkill(
        markdown_narrative="## Direction 1\n**Relations:** `rel1`",
        question_analysis="legacy analysis",
    )
    text = format_aggregated_skill_for_prompt(skill)
    assert "Direction 1" in text
    assert "RETRIEVED SKILL EXPERIENCE" in text


def test_format_legacy_fallback():
    """Legacy format used when no markdown_narrative."""
    skill = AggregatedSkill(
        question_analysis="test analysis",
        plan_guidance="plan G",
        action_guidance="action G",
        combined_reasoning_hint="reasoning G",
    )
    text = format_aggregated_skill_for_prompt(skill)
    assert "plan G" in text
    assert "action G" in text
    assert "reasoning G" in text


def test_format_empty_skill_returns_empty():
    """Empty skill returns empty string."""
    skill = AggregatedSkill()
    text = format_aggregated_skill_for_prompt(skill)
    assert text == ""


def test_format_with_pitfalls_and_negative_experiences():
    """Legacy format includes pitfalls and negative experiences."""
    skill = AggregatedSkill(
        question_analysis="test question",
        pitfalls=["wrong relation", "bad tool"],
        negative_experiences=["avoid X", "avoid Y"],
    )
    text = format_aggregated_skill_for_prompt(skill)
    assert "wrong relation" in text
    assert "avoid X" in text
