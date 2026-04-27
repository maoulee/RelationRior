from __future__ import annotations

import re
from typing import Dict, List, Set


def normalize(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch.isalnum() or ch in " .-_")
    return " ".join(s.split())


def calculate_f1(pred: List[str], gt: List[str]) -> float:
    if not pred or not gt:
        return 0.0
    p_set = {normalize(x) for x in pred}
    g_set = {normalize(x) for x in gt}
    if not p_set or not g_set:
        return 0.0
    common = p_set & g_set
    if not common:
        return 0.0
    precision = len(common) / len(p_set)
    recall = len(common) / len(g_set)
    return 2 * precision * recall / (precision + recall)


def compress_with_structured_data(
    tool_structured_results: List[Dict], candidate_entities: Set[str], fallback_text: str = ""
) -> str:
    """
    Lightweight inference-time fallback.

    The full legacy implementation depends on plug_v11.py, which in turn imports
    old swift.plugin symbols that are not required for standalone prompt-tuning
    / inference evaluation. For those flows, returning the uncompressed fallback
    text is safe and avoids pulling training-time dependencies.
    """

    return fallback_text


__all__ = [
    "normalize",
    "calculate_f1",
    "compress_with_structured_data",
]
