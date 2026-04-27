"""Audit agent for KGQA final-answer quality evaluation.

After the model produces a final answer, the audit agent checks for defects
across 5 binary dimensions (issue-found / no-issue) and can suggest websearch
when graph evidence is insufficient.

Feature-gated by KGQA_ENABLE_AUDIT_AGENT env var (default: off).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional


def is_audit_agent_enabled() -> bool:
    """Check whether the audit agent feature is enabled."""
    return os.getenv("KGQA_ENABLE_AUDIT_AGENT", "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AuditResult:
    """Structured result from the audit agent evaluation."""
    passed: bool
    any_issue: bool                   # True if any defect was detected
    issues: List[str]
    feedback: str                     # Detailed feedback for re-reasoning
    suggest_websearch: bool           # Whether websearch might help
    websearch_queries: List[str]      # Suggested search queries
    raw_audit_response: str = ""
    audit_failed: bool = False        # True when audit itself errored (not a real pass)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _graceful_pass_result(raw_audit_response: str = "", audit_failed: bool = False) -> AuditResult:
    """Return a passing result for graceful degradation on errors."""
    return AuditResult(
        passed=True,
        any_issue=False,
        issues=[],
        feedback="",
        suggest_websearch=False,
        websearch_queries=[],
        raw_audit_response=raw_audit_response,
        audit_failed=audit_failed,
    )


def _extract_json_block(raw: str) -> str:
    """Extract a JSON object from raw LLM output (handles markdown fences)."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty audit response")

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Find outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in audit response")
    return text[start:end + 1]


def _coerce_str_list(items: object) -> List[str]:
    """Coerce a value to a list of non-empty strings."""
    if not isinstance(items, list):
        return []
    return [str(v).strip() for v in items if str(v).strip()]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_audit_prompt(
    question: str,
    final_answer: List[str],
    candidates: List[str],
    raw_response: str,
    conversation_history: List[Dict[str, str]],
    web_search_results: Optional[str] = None,
) -> str:
    """Build the audit evaluation prompt."""
    history_lines: List[str] = []
    for turn in conversation_history:
        role = str(turn.get("role", "")).strip() or "unknown"
        content = str(turn.get("content", "")).strip()
        history_lines.append(f"{role}: {content}")

    final_answer_str = json.dumps(final_answer, ensure_ascii=False, indent=2)
    candidates_str = json.dumps(candidates, ensure_ascii=False, indent=2) if candidates else "N/A"
    history_str = "\n".join(history_lines) if history_lines else "(none)"

    web_search_section = ""
    if web_search_results:
        web_search_section = (
            "WEB SEARCH RESULTS (retrieved because model flagged graph evidence as insufficient):\n"
            f"{web_search_results}\n\n"
            "Use the web search results above as supplementary evidence when evaluating "
            "dimensions 4 (Graph Evidence) and 5 (Knowledge Gap). "
            "If web evidence confirms or contradicts the answer, note it in your evaluation.\n\n"
        )

    return f"""\
[AUDIT VERIFICATION]
You are auditing a knowledge-graph question answering result.
For each dimension below, determine whether a DEFECT exists (binary: issue_found or not).

DIMENSIONS:

1. **Entity Granularity Match** — Is the answer entity granularity appropriate for the question?
   E.g., "what country does X play for" needs "Brazil national football team" not "Brazil".
   Flag if the answer entity is too coarse or too fine-grained for what the question asks.

2. **Answer Set Cardinality** — Does answer count match question intent?
   Flag if the question expects a single answer but multiple are given (or vice versa),
   or if obviously missing/extra entities are present.

3. **Semantic Scope Match** — Is the answer at the right semantic level?
   Flag if the answer addresses a different scope than the question (e.g., city when
   country is asked, or a subtype when the supertype is intended).

4. **Graph Evidence Verification** — Can each answer entity be confirmed by specific graph evidence?
   If you cannot cite specific graph evidence for an answer entity, flag it.  Check the
   CANDIDATES and RAW RESPONSE for supporting triples or paths.

5. **Knowledge Gap Detection** — Would external search help fill evidence gaps?
   Flag when the graph evidence is clearly insufficient to answer confidently and a
   websearch could resolve the ambiguity.

EVIDENCE-BASED REVIEW INSTRUCTIONS:
For each dimension, only flag issue_found=true when you can identify a concrete,
specific problem with supporting evidence from the graph results or answer structure.
Do NOT flag based on hypothetical alternatives or "what-if" scenarios.
If the answer is consistent with the question and supported by graph evidence, mark issue_found=false.

QUESTION: {question}

FINAL ANSWER: {final_answer_str}

CANDIDATES: {candidates_str}

RAW RESPONSE: {raw_response}

CONVERSATION HISTORY:
{history_str}

{web_search_section}Output JSON only:
{{
  "defects": {{
    "granularity": {{"issue_found": <bool>, "description": "<description if issue, else empty string>"}},
    "cardinality": {{"issue_found": <bool>, "description": "<description if issue, else empty string>"}},
    "scope": {{"issue_found": <bool>, "description": "<description if issue, else empty string>"}},
    "evidence": {{"issue_found": <bool>, "description": "<description if issue, else empty string>"}},
    "knowledge_gap": {{"issue_found": <bool>, "description": "<description if issue, else empty string>"}}
  }},
  "any_issue": <bool>,
  "issues": ["concrete issue 1", "concrete issue 2"],
  "reasoning": "Detailed feedback explaining what should change in the next reasoning pass.",
  "suggest_websearch": <bool>,
  "websearch_queries": ["query1", "query2"]
}}

Rules:
- Each defect is binary: issue_found is true or false.
- description must be non-empty when issue_found is true.
- any_issue must be true if ANY defect has issue_found=true.
- issues list must contain one concise, actionable string per defect found.
- Set suggest_websearch to true ONLY when external search is likely to resolve insufficient graph evidence.
- When suggest_websearch is true, provide up to 5 precise search queries.
- reasoning must explain what should change in the next reasoning pass.
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_audit_response(raw: str) -> AuditResult:
    """Parse the LLM audit response into an AuditResult."""
    json_str = _extract_json_block(raw)
    data = json.loads(json_str)

    # Extract per-dimension defects
    defects_data = data.get("defects") or {}
    defect_keys = ("granularity", "cardinality", "scope", "evidence", "knowledge_gap")
    issue_descriptions: List[str] = []
    any_defect = False
    for key in defect_keys:
        entry = defects_data.get(key)
        if not isinstance(entry, dict):
            continue
        if bool(entry.get("issue_found", False)):
            any_defect = True
            desc = str(entry.get("description", "")).strip()
            if desc:
                issue_descriptions.append(f"[{key}] {desc}")

    # any_issue: prefer explicit field, else derive from defects
    any_issue = bool(data.get("any_issue", any_defect))
    passed = not any_issue

    # Collect issues list
    issues = _coerce_str_list(data.get("issues"))
    # Also include structured defect descriptions if not already covered
    for desc in issue_descriptions:
        if desc not in issues:
            issues.append(desc)

    feedback = str(data.get("reasoning", "")).strip()
    suggest_websearch = bool(data.get("suggest_websearch", False))
    websearch_queries = _coerce_str_list(data.get("websearch_queries"))

    if not suggest_websearch:
        websearch_queries = []

    return AuditResult(
        passed=passed,
        any_issue=any_issue,
        issues=issues,
        feedback=feedback,
        suggest_websearch=suggest_websearch,
        websearch_queries=websearch_queries,
        raw_audit_response=raw,
    )


# ---------------------------------------------------------------------------
# Default LLM call (mirrors intent_analyzer.py pattern)
# ---------------------------------------------------------------------------

async def _default_call_llm(messages: List[Dict[str, str]]) -> str:
    """Default LLM call delegating to the shared client."""
    from ..llm_client import call_llm

    return await call_llm(messages, max_tokens=1024, temperature=0.1, timeout_seconds=30.0, retries=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def audit_final_answer(
    question: str,
    final_answer: List[str],
    candidates: List[str],
    raw_response: str,
    conversation_history: List[Dict[str, str]],
    call_llm: Optional[Callable[[List[Dict[str, str]]], Awaitable[str]]] = None,
    web_search_results: Optional[str] = None,
) -> AuditResult:
    """Audit the final answer produced by the KGQA pipeline.

    Parameters
    ----------
    question : str
        The original user question.
    final_answer : List[str]
        The final answer strings produced by the pipeline.
    candidates : List[str]
        The candidate entities retrieved from the graph.
    raw_response : str
        The raw LLM reasoning response.
    conversation_history : List[Dict[str, str]]
        Conversation turns (role/content) for context.
    call_llm : async callable, optional
        LLM invocation function. If None, uses internal _default_call_llm.

    Returns
    -------
    AuditResult
        Structured audit result with defects, issues, and websearch suggestions.
        On any error, returns a passing result (graceful degradation).
    """
    if not is_audit_agent_enabled():
        return _graceful_pass_result()

    prompt = _build_audit_prompt(
        question=question,
        final_answer=final_answer,
        candidates=candidates,
        raw_response=raw_response,
        conversation_history=conversation_history,
        web_search_results=web_search_results,
    )
    messages = [{"role": "user", "content": prompt}]

    llm_caller = call_llm or _default_call_llm

    try:
        raw_audit_response = await llm_caller(messages)
        if not isinstance(raw_audit_response, str):
            raw_audit_response = str(raw_audit_response)
        return _parse_audit_response(raw_audit_response)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Audit agent failed; assuming pass: %s", exc)
        return _graceful_pass_result(audit_failed=True)


def format_audit_feedback_for_rereasoning(audit: AuditResult) -> str:
    """Format audit result as a re-reasoning prompt for the model.

    Parameters
    ----------
    audit : AuditResult
        The audit result to format.

    Returns
    -------
    str
        Formatted feedback string suitable for injection into a re-reasoning
        prompt.
    """
    lines = [
        "The previous answer did not pass audit. Re-reason carefully.",
    ]

    if audit.issues:
        lines.append("Issues:")
        lines.extend(f"- {issue}" for issue in audit.issues)

    if audit.feedback:
        lines.append("Detailed feedback:")
        lines.append(audit.feedback)

    if audit.suggest_websearch:
        lines.append(
            "Web search may be needed because graph evidence appears insufficient."
        )
        if audit.websearch_queries:
            lines.append("Suggested search queries:")
            lines.extend(f"- {query}" for query in audit.websearch_queries)

    lines.append("Revise the answer to address every issue explicitly.")
    return "\n".join(lines)
