"""Background search agent for insufficient result detection.

When graph queries return insufficient results repeatedly, this module triggers
a background web search to find missing attributes or alternative approaches.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SearchAgentConfig:
    """Configuration for background search agent.

    The search agent triggers when:
    - Insufficient result streak >= threshold (default: 2)
    - Search hasn't been injected yet for this session
    - enabled is True (default: off, requires KGQA_ENABLE_BACKGROUND_SEARCH=1)
    """

    insufficient_threshold: int = 2
    max_search_results: int = 5
    enable_wikidata_probe: bool = True
    enabled: bool = False

    @classmethod
    def from_env(cls) -> SearchAgentConfig:
        """Create configuration from environment variables."""
        enabled = os.getenv("KGQA_ENABLE_BACKGROUND_SEARCH", "0").strip().lower() in {"1", "true", "yes", "on"}
        return cls(
            insufficient_threshold=int(os.getenv("KGQA_INSUFFICIENT_THRESHOLD", "2")),
            max_search_results=int(os.getenv("KGQA_MAX_SEARCH_RESULTS", "5")),
            enable_wikidata_probe=os.getenv("KGQA_ENABLE_WIKIDATA_PROBE", "1").strip().lower()
            in {"1", "true", "yes", "on"},
            enabled=enabled,
        )

    def __post_init__(self):
        """Set enabled from environment if not explicitly provided."""
        if not self.enabled:
            self.enabled = os.getenv("KGQA_ENABLE_BACKGROUND_SEARCH", "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class SearchResult:
    """Structured search result from web search."""

    title: str
    url: str
    snippet: str
    source: str = "web"  # "web" or "wikidata"

    # Optional structured data for wikidata results
    qid: str = ""
    property_id: str = ""
    snapshot_status: str = ""  # ACTIVE_AT_SNAPSHOT, ENDED_BEFORE_SNAPSHOT, etc.


class SearchResultFormatter:
    """Formats search results for injection into model context."""

    @staticmethod
    def format_for_context(
        results: List[SearchResult],
        question: str,
        candidates: Optional[List[str]] = None,
    ) -> str:
        """Format search results for model consumption.

        Args:
            results: List of search results to format
            question: Original question being answered
            candidates: Current graph candidates (for comparison)

        Returns:
            Formatted text block ready for injection
        """
        if not results:
            return ""

        lines = [
            "[BACKGROUND SEARCH RESULTS]",
            f"Question: {question}",
        ]

        if candidates:
            lines.extend([
                "",
                "Current graph candidates under comparison:",
            ])
            for candidate in candidates[:8]:
                lines.append(f"- {candidate}")

        lines.extend([
            "",
            "Search found the following information:",
        ])

        # Group by source
        web_results = [r for r in results if r.source == "web"]
        wikidata_results = [r for r in results if r.source == "wikidata"]

        if wikidata_results:
            lines.append("")
            lines.append("[Wikidata Structured Data]")
            for result in wikidata_results[:5]:
                status_note = ""
                if result.snapshot_status == "ACTIVE_AT_SNAPSHOT":
                    status_note = " ✓ Active at snapshot time"
                elif result.snapshot_status == "ENDED_BEFORE_SNAPSHOT":
                    status_note = " ✗ Ended before snapshot"
                elif result.snapshot_status == "STARTS_AFTER_SNAPSHOT":
                    status_note = " ✗ Started after snapshot"

                lines.append(f"- {result.title}{status_note}")
                if result.property_id:
                    lines.append(f"  Property: {result.property_id} ({result.qid})")

        if web_results:
            lines.append("")
            for idx, result in enumerate(web_results[:5], 1):
                lines.append(f"{idx}. {result.title}")
                if result.url:
                    lines.append(f"   URL: {result.url}")
                if result.snippet:
                    lines.append(f"   Snippet: {result.snippet}")

        lines.extend([
            "",
            "Use these results to identify missing attributes or alternative approaches.",
            "The final answer must still be an exact graph entity string if candidates exist.",
        ])

        return "\n".join(lines)


@dataclass
class BackgroundSearchAgent:
    """Agent that triggers background search when results are insufficient."""

    config: SearchAgentConfig = field(default_factory=SearchAgentConfig.from_env)

    def should_trigger_search(
        self,
        state: Dict[str, Any],
        backend_results: List[Any],
    ) -> bool:
        """Determine if background search should be triggered.

        Args:
            state: Current inference state
            backend_results: Latest backend execution results

        Returns:
            True if search should be triggered
        """
        # Don't trigger if already injected
        if state.get("search_injected", False):
            return False

        # Don't trigger if no backend results
        if not backend_results:
            return False

        # Check if results indicate insufficient data
        has_insufficient = any(
            r.status in ("NO_PATH", "NO_MATCHES", "ENTITY_ERROR")
            for r in backend_results
            if hasattr(r, "status")
        )

        if not has_insufficient:
            return False

        # Check if streak threshold is met
        current_streak = state.get("insufficient_streak", 0)
        return current_streak >= self.config.insufficient_threshold

    def build_search_query(
        self,
        state: Dict[str, Any],
        backend_results: List[Any],
    ) -> Optional[str]:
        """Build a search query based on what's missing.

        Args:
            state: Current inference state
            backend_results: Latest backend execution results

        Returns:
            Search query string or None if can't build one
        """
        # Extract missing attributes from failed results
        missing_attrs = self._extract_missing_attributes(state, backend_results)
        if not missing_attrs:
            return None

        # Get core entities for context
        core_entities = list(
            state.get("prompt_context", {}).get("core_entities", set())
        ) or list(state.get("verified_entities", set()))[:2]

        if not core_entities:
            return None

        # Build query: "entity1 entity2 missing_attribute"
        entity_part = " ".join(core_entities[:2])
        attr_part = " ".join(missing_attrs[:2])

        # Add temporal context if available
        snapshot_date = os.getenv("KGQA_GRAPH_SNAPSHOT_DATE", "").strip()
        if snapshot_date and re.match(r"^\d{4}", snapshot_date):
            year = snapshot_date[:4]
            return f"{entity_part} {attr_part} {year}"

        return f"{entity_part} {attr_part}"

    def _extract_missing_attributes(
        self,
        state: Dict[str, Any],
        backend_results: List[Any],
    ) -> List[str]:
        """Extract missing attributes from failed backend results.

        Args:
            state: Current inference state
            backend_results: Latest backend execution results

        Returns:
            List of missing attribute names/relation hints
        """
        missing = []

        for result in backend_results:
            if not hasattr(result, "status") or result.status not in (
                "NO_PATH",
                "NO_MATCHES",
                "ENTITY_ERROR",
            ):
                continue

            # Check response text for hints about what's missing
            response_text = getattr(result, "response_text", "") or ""

            # Look for relation names in error messages
            relation_pattern = r'relation[s]?\s+"([^"]+)"'
            for match in re.finditer(relation_pattern, response_text):
                missing.append(match.group(1))

            # Look for attribute mentions
            attr_keywords = ("attribute", "property", "field", "not found", "no data")
            for keyword in attr_keywords:
                if keyword.lower() in response_text.lower():
                    # Extract the context around the keyword
                    context_match = re.search(
                        rf'.{{0,50}}{keyword}.{{0,50}}',
                        response_text,
                        re.IGNORECASE,
                    )
                    if context_match:
                        missing.append(context_match.group(0).strip())

        return missing[:5]  # Limit to 5 most relevant

    async def execute_search(
        self,
        query: str,
        candidates: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Execute web search and return structured results.

        Args:
            query: Search query string
            candidates: Current graph candidates (for wikidata probe)

        Returns:
            List of structured search results
        """
        # This would integrate with the existing web_search infrastructure
        # For now, return empty list - actual implementation would call
        # GraphBackendClient._execute_web_search_sync() and parse results
        return []


def trigger_background_search(
    state: Dict[str, Any],
    backend_results: List[Any],
    config: Optional[SearchAgentConfig] = None,
) -> Optional[str]:
    """Convenience function to check and trigger background search.

    Args:
        state: Current inference state
        backend_results: Latest backend execution results
        config: Optional search configuration

    Returns:
        Formatted search results for injection, or None if not triggered
    """
    agent = BackgroundSearchAgent(config=config or SearchAgentConfig.from_env())

    if not agent.should_trigger_search(state, backend_results):
        return None

    query = agent.build_search_query(state, backend_results)
    if not query:
        return None

    # Mark as injected in state
    state["search_injected"] = True

    # Return query placeholder - actual search would be async
    return f"[BACKGROUND SEARCH TRIGGERED]\nQuery: {query}\n(Async search pending...)"
