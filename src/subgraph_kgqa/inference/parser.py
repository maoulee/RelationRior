from __future__ import annotations

import ast
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class ParseErrorType(str, Enum):
    """Structured error types for LLM output parsing failures."""

    MISSING_ACT_TAG = "missing_act_tag"
    MISSING_QUERY_TAG = "missing_query_tag"
    INVALID_FUNCTION_SYNTAX = "invalid_function_syntax"
    MISSING_REASONING_TAG = "missing_reasoning_tag"
    TRUNCATED_OUTPUT = "truncated_output"


class ParseError:
    """Describes a single parse failure with position, context, and a fix hint."""

    __slots__ = ("error_type", "position", "expected", "found", "suggestion", "raw_segment")

    def __init__(
        self,
        *,
        error_type: ParseErrorType,
        position: int = -1,
        expected: str = "",
        found: str = "",
        suggestion: str = "",
        raw_segment: str = "",
    ) -> None:
        self.error_type = error_type
        self.position = position
        self.expected = expected
        self.found = found
        self.suggestion = suggestion
        self.raw_segment = raw_segment

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "position": self.position,
            "expected": self.expected,
            "found": self.found,
            "suggestion": self.suggestion,
            "raw_segment": self.raw_segment[:200],
        }

    def __repr__(self) -> str:
        return (
            f"ParseError({self.error_type.value}, pos={self.position}, "
            f"expected={self.expected!r}, found={self.found!r})"
        )


def _detect_truncation_errors(text: str) -> List[ParseError]:
    """Check for signs that the LLM output was cut off mid-tag or mid-expression."""
    errors: List[ParseError] = []
    stripped = text.strip()

    # Unclosed tags
    open_tags = ["<act>", "<query>", "<answer>", "<reasoning>", "<candidates>"]
    for tag in open_tags:
        close_tag = tag.replace("<", "</")
        if tag in stripped and close_tag not in stripped:
            pos = stripped.find(tag)
            errors.append(
                ParseError(
                    error_type=ParseErrorType.TRUNCATED_OUTPUT,
                    position=pos,
                    expected=close_tag,
                    found=stripped[-40:] if len(stripped) > 40 else stripped,
                    suggestion=f"Output appears truncated: {tag} was opened but {close_tag} was never closed. "
                    "Shorten the response or reduce reasoning length.",
                    raw_segment=stripped[pos : pos + 80],
                )
            )

    # Partial tag at end
    last_80 = stripped[-80:] if len(stripped) > 80 else stripped
    if re.search(r"<[a-zA-Z]?$", last_80):
        errors.append(
            ParseError(
                error_type=ParseErrorType.TRUNCATED_OUTPUT,
                position=len(stripped),
                expected="complete tag",
                found=stripped[-20:],
                suggestion="Output ends with a partial tag. Provide a complete response.",
                raw_segment=stripped[-40:],
            )
        )

    # Unclosed parentheses inside query
    open_parens = stripped.count("(") - stripped.count(")")
    if open_parens > 0 and "<query>" in stripped:
        errors.append(
            ParseError(
                error_type=ParseErrorType.TRUNCATED_OUTPUT,
                position=stripped.rfind("("),
                expected="closing parenthesis",
                found=stripped[-40:],
                suggestion="Function call appears truncated. Ensure all parentheses are closed.",
                raw_segment=stripped[-60:],
            )
        )

    return errors


def _extract_ast_call_args(call_node: ast.Call) -> Dict[str, Any]:
    """Safely extract keyword arguments from an AST Call node."""
    arguments: Dict[str, Any] = {}
    for kw in call_node.keywords:
        if kw.arg is None:
            continue  # skip **kwargs
        try:
            arguments[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            # For complex expressions (list of dicts, etc.), try deeper extraction
            arguments[kw.arg] = _safe_eval_node(kw.value)
    return arguments


def _safe_eval_node(node: ast.AST) -> Any:
    """Recursively evaluate AST nodes that ast.literal_eval can't handle."""
    if isinstance(node, ast.List):
        return [_safe_eval_node(el) for el in node.elts]
    if isinstance(node, ast.Dict):
        result = {}
        for key, value in zip(node.keys, node.values):
            k = ast.literal_eval(key) if key else None
            result[k] = _safe_eval_node(value)
        return result
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        # Handle True, False, None
        name_map = {"True": True, "False": False, "None": None}
        return name_map.get(node.id, node.id)
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _parse_args_regex(args_str: str) -> Dict[str, Any]:
    """
    Fallback regex-based argument parser.
    Handles: key="value", key=[...], key=value patterns.
    """
    arguments: Dict[str, Any] = {}
    # Match key=value pairs where value can be quoted string or bracket list or bare word
    # Pattern handles: key="val", key=[...], key=True/False/None, key=123
    pattern = r'(\w+)\s*=\s*'
    pos = 0
    for m in re.finditer(pattern, args_str):
        key = m.group(1)
        val_start = m.end()
        if val_start >= len(args_str):
            continue

        # Determine value boundaries
        char = args_str[val_start]
        if char in ('"', "'"):
            # Quoted string — find matching close
            quote = char
            end = args_str.find(quote, val_start + 1)
            if end == -1:
                end = len(args_str)
            arguments[key] = args_str[val_start + 1:end]
            pos = end + 1
        elif char == '[':
            # List — find matching ]
            depth = 1
            end = val_start + 1
            while end < len(args_str) and depth > 0:
                if args_str[end] == '[':
                    depth += 1
                elif args_str[end] == ']':
                    depth -= 1
                end += 1
            list_str = args_str[val_start:end]
            try:
                arguments[key] = ast.literal_eval(list_str)
            except (ValueError, SyntaxError):
                arguments[key] = list_str
            pos = end
        else:
            # Bare value — read until comma or end
            end = args_str.find(',', val_start)
            if end == -1:
                end = len(args_str)
            val = args_str[val_start:end].strip()
            # Convert booleans/None
            if val == "True":
                arguments[key] = True
            elif val == "False":
                arguments[key] = False
            elif val == "None":
                arguments[key] = None
            else:
                arguments[key] = val
            pos = end

    return arguments


class InferenceOutputParser:
    """Standalone parser for inference-time model outputs."""

    @staticmethod
    def parse(text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "reasoning": "",
            "queries": [],
            "final_answer": [],
            "parse_errors": [],
            "stage_decision": "",
        }
        parse_errors: List[ParseError] = []

        # --- Truncation detection ---
        truncation_errors = _detect_truncation_errors(text)
        parse_errors.extend(truncation_errors)

        # --- Reasoning ---
        reasoning_match = re.search(
            r"<reasoning>(.*?)</reasoning>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        else:
            # Check if reasoning was expected: look for <think/<reasoning> open without close
            for tag_name in ("reasoning", "think", "reason"):
                open_tag = f"<{tag_name}>"
                close_tag = f"</{tag_name}>"
                if open_tag in text.lower() and close_tag not in text.lower():
                    pos = text.lower().find(open_tag)
                    parse_errors.append(
                        ParseError(
                            error_type=ParseErrorType.MISSING_REASONING_TAG,
                            position=pos,
                            expected=close_tag,
                            found=text[pos : pos + 60],
                            suggestion=f"The <{tag_name}> tag was opened but never closed. "
                            "Ensure the closing tag is present.",
                            raw_segment=text[pos : pos + 120],
                        )
                    )
                    break

        text_clean = re.sub(
            r"<(?:reasoning|think|reason)>.*?</(?:reasoning|think|reason)>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # --- Answer ---
        answer_block_match = re.search(
            r"<answer>(.*?)</answer>",
            text_clean,
            re.DOTALL | re.IGNORECASE,
        )
        answer_text = answer_block_match.group(1) if answer_block_match else text_clean
        boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", answer_text)
        if boxed_matches:
            cleaned_answers = []
            seen = set()
            for boxed in boxed_matches:
                answer = boxed.strip()
                if not answer or answer in {"...", "etc."}:
                    continue
                if answer.endswith(".") and not re.search(r"\b[A-Z]\.$", answer):
                    answer = answer[:-1]
                if answer and answer not in seen:
                    cleaned_answers.append(answer)
                    seen.add(answer)
            result["final_answer"] = cleaned_answers

        # --- Act / Queries with graceful partial parsing ---
        act_match = re.search(r"<act>(.*?)</act>", text, re.DOTALL | re.IGNORECASE)
        act_content: Optional[str] = None
        if act_match:
            act_content = act_match.group(1)
        else:
            # Check for tool-call indicators without <act> wrapper
            tool_indicators = [
                re.search(r"<query>", text, re.IGNORECASE),
                re.search(r"\b(plan|action|filter|search|web_search|check_entities|explore_schema|match_pattern)\s*\(", text, re.IGNORECASE),
            ]
            if any(tool_indicators):
                pos = min(m.start() for m in tool_indicators if m)
                parse_errors.append(
                    ParseError(
                        error_type=ParseErrorType.MISSING_ACT_TAG,
                        position=pos,
                        expected="<act>...</act>",
                        found=text[pos : pos + 60],
                        suggestion="Tool-call content found but <act> wrapper is missing. "
                        "Wrap tool calls in <act><query>...</query></act>.",
                        raw_segment=text[pos : pos + 120],
                    )
                )
                # Try to extract queries anyway from raw text for graceful partial parsing
                query_strings = re.findall(
                    r"<query>(.*?)</query>",
                    text,
                    re.DOTALL | re.IGNORECASE,
                )
                act_content = "\n".join(query_strings) if query_strings else None

        if act_content is not None:
            query_strings = re.findall(
                r"<query>(.*?)</query>",
                act_content,
                re.DOTALL | re.IGNORECASE,
            )
            if not query_strings and "<query>" in act_content:
                # <query> opened but not closed — partial parse
                pos = act_content.lower().find("<query>")
                parse_errors.append(
                    ParseError(
                        error_type=ParseErrorType.MISSING_QUERY_TAG,
                        position=pos,
                        expected="</query>",
                        found=act_content[pos : pos + 80],
                        suggestion="<query> tag opened but not closed. "
                        "Ensure each <query> has a matching </query>.",
                        raw_segment=act_content[pos : pos + 120],
                    )
                )
                # Attempt to recover: grab text from <query> to end
                partial = re.findall(r"<query>(.*)", act_content, re.DOTALL | re.IGNORECASE)
                query_strings = [p.strip() for p in partial if p.strip()]

            for i, query_str in enumerate(query_strings):
                stripped_query = query_str.strip()
                parsed_query = InferenceOutputParser._parse_function_call(stripped_query)
                if parsed_query:
                    result["queries"].append(parsed_query)
                else:
                    # Report invalid function syntax but continue parsing others
                    pos_in_act = act_content.find(query_str) if act_content else -1
                    parse_errors.append(
                        ParseError(
                            error_type=ParseErrorType.INVALID_FUNCTION_SYNTAX,
                            position=pos_in_act,
                            expected="function_name(key=value, ...)",
                            found=stripped_query[:100],
                            suggestion="Query does not match expected function call syntax. "
                            "Ensure it follows tool_name(param1=\"val1\", param2=\"val2\") format.",
                            raw_segment=stripped_query[:200],
                        )
                    )

        # --- Candidates ---
        candidates = re.findall(r"<candidate>(.*?)</candidate>", text, re.DOTALL | re.IGNORECASE)
        candidate_list = [candidate.strip() for candidate in candidates if candidate.strip()]
        candidate_tags_found = bool(candidates)

        for tag_name in ("candidates", "candidate_entities"):
            candidates_block = re.search(
                rf"<{tag_name}>(.*?)</{tag_name}>",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if not candidates_block:
                continue
            candidate_tags_found = True
            for line in candidates_block.group(1).splitlines():
                stripped = line.strip()
                if stripped.startswith("- ") or stripped.startswith("* "):
                    candidate = stripped[2:].strip()
                    if candidate:
                        candidate_list.append(candidate)

        if candidate_list:
            result["candidates"] = candidate_list
        elif candidate_tags_found:
            result["candidates"] = []

        decision_match = re.search(
            r"<decision>\s*(retry_action|proceed|continue)\s*</decision>",
            text,
            re.IGNORECASE,
        )
        if decision_match:
            result["stage_decision"] = decision_match.group(1).strip().lower()
        else:
            for token in ("retry_action", "proceed", "continue"):
                if re.search(rf"<{token}\s*/>", text, re.IGNORECASE):
                    result["stage_decision"] = token
                    break

        result["parsed_plans"] = [
            query["arguments"]
            for query in result["queries"]
            if query["tool_name"] in {"plan", "plan_subquestion"}
        ]

        # Attach parse errors
        result["parse_errors"] = [e.to_dict() for e in parse_errors]
        return result

    @staticmethod
    def _parse_function_call(call_str: str) -> Optional[Dict[str, Any]]:
        match = re.match(r"^([a-zA-Z0-9_]+)\s*\((.*)\)$", call_str, re.DOTALL)
        if not match:
            return None

        tool_name = match.group(1)
        args_str = match.group(2).strip()
        if tool_name == "search":
            tool_name = "web_search"

        if not args_str:
            return {"tool_name": tool_name, "arguments": {}}

        # Try safe AST parsing first
        try:
            tree = ast.parse(call_str, mode='eval')
            call_node = tree.body
            if isinstance(call_node, ast.Call):
                arguments = _extract_ast_call_args(call_node)
                return {"tool_name": tool_name, "arguments": arguments}
        except (SyntaxError, ValueError):
            pass

        # Fallback: regex-based key=value parsing for robustness
        arguments = _parse_args_regex(args_str)
        return {"tool_name": tool_name, "arguments": arguments}
