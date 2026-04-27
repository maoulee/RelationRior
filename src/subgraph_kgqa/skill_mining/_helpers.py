"""Shared helper functions for skill_mining sub-package.

Consolidates utilities that were previously duplicated across
retriever.py, case_skill.py, render.py, extractor.py, replay_runner.py.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _to_list(value: Any) -> List[str]:
    """Normalize a value to a list of non-empty strings."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _utc_now() -> str:
    """Return current UTC time as compact ISO string."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(value: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "unknown"


def _extract_question_surface(question_text: str) -> str:
    """Extract the question surface text from a prompt that may contain phase markers."""
    text = str(question_text or "")
    match = re.search(r"\nQuestion:\n(.*?)(?:\n\s*\n\[PHASE 1\]|\Z)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from LLM output.

    Handles both fenced code blocks (```json ... ```) and raw brace-delimited JSON.
    """
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            return None
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None
