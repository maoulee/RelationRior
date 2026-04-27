"""Env-gated decision-consistency mechanism for inference-time model decisions.

When KGQA_ENABLE_DECISION_CONSISTENCY=1 is set:

Modes (controlled by KGQA_CONSISTENCY_MODE env var):

- "global" — apply consistency to every turn >= min_turn:
  1. Generate 2 candidates in parallel.
  2. If signatures agree -> choose first.
  3. If disagree -> 3rd sample, majority vote.

- "selective" — issue-presence based critique verification (zero trajectory drift):
  1. Normal LLM call (= baseline).
  2. Parse -> no final_answer -> return as-is (identical to baseline).
  3. Has final_answer -> 2nd LLM call for defect detection.
  4. No issues found -> pass (verification_agreed=True).
  5. Issues found -> return with critique_recommended=True,
     critique_feedback.  Caller injects feedback for re-reasoning.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .parser import InferenceOutputParser

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """Return True when the decision-consistency gate is active."""
    return os.getenv("KGQA_ENABLE_DECISION_CONSISTENCY", "0").strip().lower() in {"1", "true", "yes", "on"}


def _min_turn() -> int:
    try:
        return max(1, int(os.getenv("KGQA_DECISION_CONSISTENCY_MIN_TURN", "2")))
    except Exception:
        return 2


def _consistency_mode() -> str:
    """Return the consistency mode string.

    Values:
        "global"    — consistency applied to every eligible turn (original)
        "selective" — consistency only on final-stage, risk-gated turns

    Backward-compatibility: when ``KGQA_CONSISTENCY_MODE`` is *not* set but
    ``KGQA_ENABLE_DECISION_CONSISTENCY=1`` is active, the mode defaults to
    ``"global"`` so existing deployments keep their behaviour.
    """
    raw = os.getenv("KGQA_CONSISTENCY_MODE", "").strip().lower()
    if raw in ("global", "selective"):
        return raw
    # Default to "global" when the env var is absent but the feature is enabled,
    # preserving backward compatibility.  If the feature is *not* enabled the
    # mode value is irrelevant but we still return "global".
    return "global"


# (Risk-signal gating removed — selective mode now triggers purely on
#  final_answer presence, no probe-first architecture.)


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------

def compute_signature(parsed: Dict[str, Any]) -> str:
    """Build a deterministic, compact decision signature from parsed output.

    Priority order:
    1. final_answer  -> ordered tuple fingerprint
    2. queries       -> normalised tool-name + stable JSON args
    3. candidates    -> ordered tuple fingerprint
    4. fallback      -> MD5 of normalised raw reasoning text
    """
    final_answer = parsed.get("final_answer", [])
    if final_answer:
        return "fa:" + json.dumps(sorted(final_answer), separators=(",", ":"))

    queries = parsed.get("queries", [])
    if queries:
        normalised: list[str] = []
        for query in queries:
            tool = query.get("tool_name", "")
            args = query.get("arguments", {})
            normalised.append(f"{tool}({json.dumps(args, sort_keys=True, separators=(',' , ':'))})")
        return "q:" + "|".join(sorted(normalised))

    candidates = parsed.get("candidates")
    if candidates is not None:
        return "c:" + json.dumps(sorted(candidates), separators=(",", ":"))

    raw = parsed.get("reasoning", "") or ""
    digest = hashlib.md5(" ".join(raw.lower().split())[:500].encode()).hexdigest()
    return "h:" + digest


def _compare_signatures(sig_a: str, sig_b: str) -> bool:
    """Compare two decision signatures with prefix-aware normalisation.

    * ``fa:`` — compare using only the answer tuple (already the full sig).
    * ``q:``  — normalise the tool name but hash only the stable args so that
                trivially different arg serialisations don't cause disagreement.
    * Other prefixes — exact string equality.
    """
    if sig_a == sig_b:
        return True

    prefix_a, _, body_a = sig_a.partition(":")
    prefix_b, _, body_b = sig_b.partition(":")

    # Different prefix families can never match.
    if prefix_a != prefix_b:
        return False

    if prefix_a == "fa":
        # fa: bodies are already sorted JSON — direct comparison.
        return body_a == body_b

    if prefix_a == "q":
        return _q_signatures_match(body_a, body_b)

    return False


def _q_signatures_match(body_a: str, body_b: str) -> bool:
    """Compare ``q:`` signature bodies with stable-arg hashing.

    Each body is ``tool(args_json)`` entries joined by ``|``.  We re-serialise
    the args JSON with a stable hash so that semantically identical but
    textually different arg dicts are treated as equal.
    """
    parts_a = sorted(body_a.split("|"))
    parts_b = sorted(body_b.split("|"))
    if len(parts_a) != len(parts_b):
        return False

    for pa, pb in zip(parts_a, parts_b):
        norm_a = _normalise_q_part(pa)
        norm_b = _normalise_q_part(pb)
        if norm_a != norm_b:
            return False
    return True


def _normalise_q_part(part: str) -> str:
    """Normalise a single ``tool(args_json)`` entry by hashing stable args."""
    idx = part.find("(")
    if idx == -1:
        return part
    tool = part[:idx]
    args_str = part[idx + 1 : -1] if part.endswith(")") else part[idx + 1 :]
    try:
        args_obj = json.loads(args_str)
        stable = json.dumps(args_obj, sort_keys=True, separators=(",", ":"))
        stable_hash = hashlib.sha256(stable.encode()).hexdigest()[:16]
    except (json.JSONDecodeError, TypeError):
        stable_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]
    return f"{tool}({stable_hash})"


# ---------------------------------------------------------------------------
# Critique-based verification helpers (issue-presence based)
# ---------------------------------------------------------------------------

def _build_critique_prompt(raw_response: str, answer: list, candidates: list) -> str:
    """Build a defect-detection critique prompt for issue-based verification."""
    answer_str = ", ".join(str(a) for a in answer[:10])
    candidates_str = ", ".join(str(c) for c in candidates[:10]) if candidates else "N/A"

    return f"""\
[DEFECT DETECTION VERIFICATION]
You just produced a final answer. Detect defects in your reasoning using binary checks.

FINAL ANSWER: {answer_str}
AVAILABLE CANDIDATES: {candidates_str}

Before evaluating, consider:
1. If the question asked for a different entity type, would your answer still be appropriate?
2. Is there any candidate that would be a BETTER answer?
3. What specific graph evidence confirms each answer entity?

Check each dimension for defects (true = defect found):

1. **Entity Granularity Match** — Is the answer entity granularity appropriate for the question?
   E.g., "what country does X play for" needs "Brazil national football team" not "Brazil"

2. **Answer Set Cardinality** — Does answer count match question intent (single vs multiple)?

3. **Semantic Scope Match** — Is the answer at the right semantic level (role vs title vs instance)?

4. **Graph Path Correctness** — Does the relation path from anchor to answer semantically match the question?

5. **Candidate-Answer Consistency** — Are all answer entities present in candidates? Is the answer empty but candidates exist?

Output JSON only:
{{"defects": {{"granularity": {{"issue_found": false, "description": ""}}, "cardinality": {{"issue_found": false, "description": ""}}, "scope": {{"issue_found": false, "description": ""}}, "path": {{"issue_found": false, "description": ""}}, "candidate_consistency": {{"issue_found": false, "description": ""}}}}, "any_issue": false, "summary": "brief explanation"}}
"""


def _parse_critique_response(raw: str) -> tuple:
    """Parse defect-detection critique response.

    Returns (any_issue: bool, issues: list[str]).
    """
    import json as _json

    # Extract JSON from response — strip markdown fences, then find outermost braces
    text = (raw or "").strip()
    if not text:
        return False, []

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return False, []

    try:
        data = _json.loads(text[start:end + 1])
    except _json.JSONDecodeError:
        return False, []

    # Collect issue descriptions from defects dict
    defects = data.get("defects", {})
    issues: list[str] = []
    for dim_name, dim_data in defects.items():
        if isinstance(dim_data, dict) and dim_data.get("issue_found"):
            desc = dim_data.get("description", "").strip()
            if desc:
                issues.append(f"[{dim_name}] {desc}")
            else:
                issues.append(f"[{dim_name}] issue found (no description)")

    any_issue_derived = bool(issues)
    any_issue = any_issue_derived or bool(data.get("any_issue", False))
    return any_issue, issues


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def consistent_call(
    call_llm: Callable[[List[Dict[str, str]]], Awaitable[str]],
    messages: List[Dict[str, str]],
    *,
    parse_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    turn_number: int = 1,
    trajectory_context: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Optionally wrap a single model turn with decision consistency.

    Returns (raw_response, metadata_dict).
    When the feature is disabled the metadata dict is empty.

    Parameters
    ----------
    call_llm : async callable
        The underlying LLM invocation function.
    messages : list[dict]
        Chat messages passed to the LLM.
    parse_fn : callable, optional
        Parser used to extract structured fields from raw LLM output.
    turn_number : int
        Current turn index (1-based).
    trajectory_context : dict, optional
        Contextual signals for risk-gating in selective mode.  May include
        ``frontend_errors`` (int), ``candidates`` (list), and any other
        trajectory-level metadata.  Ignored in ``global`` mode.

    Backward compatibility
    ----------------------
    ``trajectory_context`` defaults to ``None`` so existing call-sites
    continue to work without modification.
    """
    # ---- Feature disabled or below min_turn: pass through ----
    if not is_enabled() or turn_number < _min_turn():
        raw = await call_llm(messages)
        return raw, {
            "consistency_used": False,
            "consistency_turn_number": turn_number,
        }

    mode = _consistency_mode()

    if parse_fn is None:
        parse_fn = InferenceOutputParser.parse

    # ---- Helper: safe parse with error metadata ----
    def _safe_parse(raw_text: str) -> Tuple[Dict[str, Any], bool]:
        """Parse *raw_text* via *parse_fn*, catching errors.

        Returns (parsed_dict, parse_error_flag).  On failure the parsed dict
        is a minimal stub so downstream signature logic can still proceed.
        """
        try:
            return parse_fn(raw_text), False  # type: ignore[misc]
        except Exception:
            return {"raw_fallback": raw_text}, True

    # ---- Selective mode: critique-based verification (zero drift on non-final turns) ----
    if mode == "selective":
        # Step 1: Normal call — identical to baseline.
        raw_1 = await call_llm(messages)
        parsed_1, parse1_err = _safe_parse(raw_1)

        if parse1_err:
            return raw_1, {
                "consistency_used": False,
                "consistency_turn_number": turn_number,
                "consistency_mode": "selective",
                "consistency_parse_error": True,
            }

        # Step 2: Only critique when we have a final_answer.
        final_answer_1 = parsed_1.get("final_answer")
        if not final_answer_1:
            # No final answer — return baseline result unchanged (zero drift).
            return raw_1, {
                "consistency_used": False,
                "consistency_turn_number": turn_number,
                "consistency_mode": "selective",
            }

        # Step 3: Critique call — ask model to evaluate its own reasoning.
        answer_list = list(final_answer_1) if isinstance(final_answer_1, (list, tuple)) else [str(final_answer_1)]
        candidates = parsed_1.get("candidates", [])

        critique_prompt = _build_critique_prompt(raw_1, answer_list, candidates)
        critique_messages = messages + [{"role": "assistant", "content": raw_1}, {"role": "user", "content": critique_prompt}]
        raw_critique = await call_llm(critique_messages)

        # Step 4: Parse critique response.
        any_issue, issues = _parse_critique_response(raw_critique)

        if not any_issue:
            # No defects found — answer is verified.
            return raw_1, {
                "consistency_used": True,
                "consistency_turn_number": turn_number,
                "consistency_mode": "selective",
                "verification_agreed": True,
                "consistency_agreed_initially": True,
                "critique_any_issue": False,
                "critique_issues": issues,
            }

        # Defects found — recommend re-reasoning with feedback.
        return raw_1, {
            "consistency_used": True,
            "consistency_turn_number": turn_number,
            "consistency_mode": "selective",
            "verification_agreed": False,
            "critique_recommended": True,
            "critique_any_issue": True,
            "critique_issues": issues,
            "critique_feedback": raw_critique,
            "consistency_agreed_initially": False,
        }

    # ---- Global mode (original behaviour) ----
    # Generate two independent candidates.
    raw_1, raw_2 = await asyncio.gather(call_llm(messages), call_llm(messages))
    parsed_1, parse1_err = _safe_parse(raw_1)
    parsed_2, parse2_err = _safe_parse(raw_2)

    if parse1_err:
        return raw_1, {
            "consistency_used": False,
            "consistency_turn_number": turn_number,
            "consistency_mode": "global",
            "consistency_parse_error": True,
        }
    if parse2_err:
        return raw_2, {
            "consistency_used": False,
            "consistency_turn_number": turn_number,
            "consistency_mode": "global",
            "consistency_parse_error": True,
        }

    sig_1, sig_2 = compute_signature(parsed_1), compute_signature(parsed_2)

    if _compare_signatures(sig_1, sig_2):
        return raw_1, {
            "consistency_used": True,
            "consistency_turn_number": turn_number,
            "consistency_mode": "global",
            "consistency_votes": {sig_1: 2},
            "consistency_agreed_initially": True,
            "consistency_chosen_signature": sig_1,
            "consistency_candidate_signatures": [sig_1, sig_2],
        }

    # Tie-breaker: third sample.
    raw_3 = await call_llm(messages)
    parsed_3, parse3_err = _safe_parse(raw_3)
    if parse3_err:
        return raw_3, {
            "consistency_used": False,
            "consistency_turn_number": turn_number,
            "consistency_mode": "global",
            "consistency_parse_error": True,
        }
    sig_3 = compute_signature(parsed_3)

    # Majority vote (first-wins on ties).
    signature_to_indices: Dict[str, List[int]] = {}
    for idx, sig in enumerate((sig_1, sig_2, sig_3)):
        signature_to_indices.setdefault(sig, []).append(idx)

    winner_sig = max(signature_to_indices, key=lambda s: len(signature_to_indices[s]))
    winner_idx = signature_to_indices[winner_sig][0]
    chosen_raw = (raw_1, raw_2, raw_3)[winner_idx]

    all_different = len(set((sig_1, sig_2, sig_3))) == 3

    return chosen_raw, {
        "consistency_used": True,
        "consistency_turn_number": turn_number,
        "consistency_mode": "global",
        "consistency_votes": {sig: len(idxs) for sig, idxs in signature_to_indices.items()},
        "consistency_agreed_initially": False,
        "consistency_chosen_signature": winner_sig,
        "consistency_candidate_signatures": [sig_1, sig_2, sig_3],
        **({"consistency_was_tie": True} if all_different else {}),
    }
