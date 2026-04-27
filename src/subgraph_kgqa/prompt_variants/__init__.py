"""
Prompt variants for Stage 5 (Final Reasoning) actionspace-style tuning.

Each variant is a dict with keys:
  - core_protocol_override: str  — replaces the CORE PROTOCOL block
  - reflection_override: str     — replaces the REFLECTION CHECKPOINT block
  - output_format_override: str  — replaces the OUTPUT FORMAT block
  - reasoning_template_override: str — replaces the <reasoning> template

If a key is empty string, the current default is used (no override).

Usage:
    from subgraph_kgqa.prompt_variants import get_variant, list_variants
    variant = get_variant("v2_no_unsupported_collapse")
    core_protocol = variant["core_protocol_override"]
"""

from .variants import VARIANTS, DEFAULT_NAME


def list_variants():
    """Return available variant names."""
    return sorted(VARIANTS.keys())


def get_variant(name: str) -> dict:
    """Return a variant dict by name. Falls back to DEFAULT_NAME if not found."""
    return VARIANTS.get(name, VARIANTS[DEFAULT_NAME])


def has_variant(name: str) -> bool:
    return name in VARIANTS


def describe_variants() -> dict:
    return {name: payload.get("description", "") for name, payload in VARIANTS.items()}
