"""
Background Search Agent for KGQA Pipeline.
Activates when model outputs consecutive "information insufficient"
at Stage 5 without backtrack action. Max 3 rounds of search+analyze.

US-004: Search is only available AFTER a selected action-space and graph
candidates already exist (Stage 5 only).  Search invocation requires an
explicit model request; it is never automatic.  Final answers MUST still
come from graph candidates -- search cannot introduce new answers.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SearchAgentConfig:
    """Configuration for background search agent."""
    enabled: bool = False
    max_rounds: int = 3
    search_provider: str = "duckduckgo"
    max_results_per_query: int = 5
    timeout_seconds: int = 10

    def __post_init__(self):
        """Load configuration from environment variables."""
        self.enabled = os.getenv("KGQA_ENABLE_BACKGROUND_SEARCH", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.max_rounds = int(os.getenv("KGQA_BACKGROUND_SEARCH_MAX_ROUNDS", "3"))
        self.search_provider = os.getenv("KGQA_BACKGROUND_SEARCH_PROVIDER", "duckduckgo")
        self.max_results_per_query = int(os.getenv("KGQA_BACKGROUND_SEARCH_MAX_RESULTS", "5"))
        self.timeout_seconds = int(os.getenv("KGQA_BACKGROUND_SEARCH_TIMEOUT", "10"))


@dataclass
class SearchResult:
    """Individual search result."""
    title: str
    snippet: str
    url: str
    source: str = "web"


@dataclass
class SearchRequest:
    """Structured request for a Stage-5-only web search.

    Search is only valid after ``candidates_collected`` is True and at
    least two graph candidates exist.  The caller (model) must supply
    the current candidate set so that results can be mapped back to
    those candidates.
    """

    query: str  # The search query string
    candidates: List[str]  # Current graph candidates to compare
    top_k: int = 5  # Max results to return (1-10)
    action_space_summary: str = ""  # Optional: the selected action space description
    missing_attribute: str = ""  # What specific info is missing


@dataclass
class StructuredSearchResult:
    """Structured evidence from a Stage-5 disambiguation search.

    All evidence is organised *per candidate* so that the model can
    compare candidates without introducing new answer strings that are
    not already in the graph candidate set.
    """

    query: str  # Original query that was searched
    per_candidate_evidence: Dict[str, List[str]] = field(
        default_factory=dict
    )  # candidate -> list of evidence snippets
    source_urls: List[str] = field(default_factory=list)
    date_relevance: Optional[str] = None  # Whether dates match graph snapshot
    summary: str = ""  # Overall summary of findings
    graph_candidate_supported: Optional[str] = (
        None  # Which candidate (if any) is supported
    )


class SearchStageGuard:
    """Validates that search is only invoked at Stage 5.

    Preconditions:
    - ``candidates_collected`` must be True.
    - At least 2 graph candidates must exist.
    - Web search must be enabled via the ``KGQA_ENABLE_WEB_SEARCH`` env var.
    """

    @staticmethod
    def validate(
        *,
        candidates_collected: bool,
        candidate_count: int,
        web_search_enabled: bool,
    ) -> Optional[str]:
        """Return an error string if preconditions are not met, else ``None``."""
        if not web_search_enabled:
            return "web_search is disabled (KGQA_ENABLE_WEB_SEARCH not set)."
        if not candidates_collected:
            return (
                "web_search is only available at Stage 5 after graph candidates "
                "are collected (candidates_collected is False)."
            )
        if candidate_count < 2:
            return (
                f"web_search requires at least 2 graph candidates for "
                f"disambiguation, but only {candidate_count} available."
            )
        return None


@dataclass
class AggregatedSearchResults:
    """Aggregated results from multiple search rounds."""
    candidate_attributes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    total_rounds: int = 0
    completion_reason: str = ""
    queries: List[str] = field(default_factory=list)
    results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackgroundSearchAgent:
    """
    Background search agent for KGQA pipeline.

    Performs autonomous web search when model outputs consecutive
    "information insufficient" at Stage 5 without backtracking.
    """

    def __init__(self, config: Optional[SearchAgentConfig] = None):
        """
        Initialize the search agent.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or SearchAgentConfig()
        self._ddgs = None

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Perform a simple async search query.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.
            timeout: Timeout in seconds.

        Returns:
            List of SearchResult objects.
        """
        if not self.config.enabled:
            logger.debug("Search is disabled in config")
            return []

        max_results = max_results or self.config.max_results_per_query
        timeout = timeout or self.config.timeout_seconds

        try:
            results = await asyncio.to_thread(
                self._sync_search,
                query,
                max_results,
                timeout,
            )
            return results
        except Exception as exc:
            logger.warning("Search failed for query '%s': %s", query, exc)
            return []

    def _sync_search(self, query: str, max_results: int, timeout: int) -> List[SearchResult]:
        """Synchronous wrapper for DDGS search (run in thread pool)."""
        if self.config.search_provider != "duckduckgo":
            return []

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo_search not installed")
            return []

        try:
            ddgs = DDGS()
            search_results = ddgs.text(query, max_results=max_results, timeout=timeout)

            results = []
            for item in search_results or []:
                if isinstance(item, dict):
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        snippet=item.get("body", ""),
                        url=item.get("link", ""),
                        source="duckduckgo",
                    ))
            return results
        except Exception as exc:
            logger.error("DDGS search error: %s", exc)
            return []

    async def search_structured(
        self,
        request: SearchRequest,
    ) -> StructuredSearchResult:
        """Perform a structured Stage-5 disambiguation search.

        This is the US-004 entry point.  It reuses the multi-round search
        logic from ``search_kgqa`` but returns a ``StructuredSearchResult``
        whose evidence is organised *per candidate*.

        Args:
            request: A validated ``SearchRequest``.

        Returns:
            ``StructuredSearchResult`` with per-candidate evidence.
        """
        # Reuse existing multi-round logic
        aggregated = await self.search_kgqa(
            question=request.query,
            candidates=set(request.candidates),
            missing_attribute=request.missing_attribute or "",
            reasoning_summary=request.action_space_summary or "",
        )

        # Build per-candidate evidence mapping
        per_candidate: Dict[str, List[str]] = {}
        for candidate in request.candidates:
            attrs = aggregated.candidate_attributes.get(candidate, {})
            snippets: List[str] = []
            for attr_name, attr_value in attrs.items():
                snippets.append(f"{attr_name}: {attr_value}")
            per_candidate[candidate] = snippets

        # Collect source URLs
        seen = set()
        source_urls = []
        for result in aggregated.results:
            if result.url not in seen:
                seen.add(result.url)
                source_urls.append(result.url)

        # Determine which candidate (if any) has the most evidence
        graph_candidate_supported: Optional[str] = None
        best_count = 0
        for candidate, evidence in per_candidate.items():
            if len(evidence) > best_count:
                best_count = len(evidence)
                graph_candidate_supported = candidate
        if best_count == 0:
            graph_candidate_supported = None

        # Build summary
        parts: List[str] = []
        if aggregated.completion_reason == "disabled":
            parts.append("Search is disabled.")
        elif aggregated.completion_reason == "no_candidates":
            parts.append("No candidates provided.")
        elif aggregated.completion_reason == "no_results":
            parts.append("No search results found.")
        else:
            parts.append(
                f"Completed {aggregated.total_rounds} round(s), "
                f"reason: {aggregated.completion_reason}."
            )
        summary = " ".join(parts)

        # Date relevance (placeholder; populated when graph snapshot date
        # info becomes available via environment variable)
        snapshot_date = os.getenv("KGQA_GRAPH_SNAPSHOT_DATE", "").strip()
        date_relevance: Optional[str] = None
        if snapshot_date:
            date_relevance = f"Graph snapshot date: {snapshot_date}"

        return StructuredSearchResult(
            query=request.query,
            per_candidate_evidence=per_candidate,
            source_urls=source_urls,
            date_relevance=date_relevance,
            summary=summary,
            graph_candidate_supported=graph_candidate_supported,
        )

    async def search_kgqa(
        self,
        question: str,
        candidates: Set[str],
        missing_attribute: str,
        reasoning_summary: str,
    ) -> AggregatedSearchResults:
        """
        Perform background search with multiple rounds.

        Args:
            question: The original question being asked
            candidates: Set of candidate entities
            missing_attribute: The attribute that's missing from the graph
            reasoning_summary: Summary of model's reasoning

        Returns:
            AggregatedSearchResults with collected information
        """
        if not self.config.enabled:
            return AggregatedSearchResults(
                candidate_attributes={},
                total_rounds=0,
                completion_reason="disabled",
                metadata={"search_provider": self.config.search_provider},
            )

        candidate_list = sorted(list(candidates))
        if not candidate_list:
            return AggregatedSearchResults(
                candidate_attributes={},
                total_rounds=0,
                completion_reason="no_candidates",
                metadata={"search_provider": self.config.search_provider},
            )

        candidate_attributes: Dict[str, Dict[str, str]] = {}
        total_rounds = 0

        # Round 1: Broad search
        try:
            round1_results = await self._search_round(
                query=self._build_broad_query(question, missing_attribute, candidate_list[:2]),
                candidates=candidate_list,
            )
            for candidate, attrs in round1_results.items():
                if candidate not in candidate_attributes:
                    candidate_attributes[candidate] = {}
                candidate_attributes[candidate].update(attrs)
            total_rounds += 1
        except Exception:
            # Continue to next round if this one fails
            pass

        # Rounds 2-3: Targeted searches for candidates still missing info
        candidates_missing = [
            c for c in candidate_list
            if not candidate_attributes.get(c) or missing_attribute.lower() not in " ".join(candidate_attributes.get(c, {}).values()).lower()
        ]

        for round_num in range(2, self.config.max_rounds + 1):
            if not candidates_missing or total_rounds >= self.config.max_rounds:
                break

            try:
                round_results = await self._search_round(
                    query=self._build_targeted_query(candidates_missing[0], missing_attribute),
                    candidates=candidates_missing,
                )
                for candidate, attrs in round_results.items():
                    if candidate not in candidate_attributes:
                        candidate_attributes[candidate] = {}
                    candidate_attributes[candidate].update(attrs)
                total_rounds += 1

                # Update candidates still missing info
                candidates_missing = [
                    c for c in candidates_missing
                    if not candidate_attributes.get(c) or missing_attribute.lower() not in " ".join(candidate_attributes.get(c, {}).values()).lower()
                ]
            except Exception:
                # Continue to next round if this one fails
                continue

        # Determine completion reason
        if total_rounds >= self.config.max_rounds:
            completion_reason = "max_rounds_reached"
        elif candidate_attributes:
            completion_reason = "sufficient"
        else:
            completion_reason = "no_results"

        return AggregatedSearchResults(
            candidate_attributes=candidate_attributes,
            total_rounds=total_rounds,
            completion_reason=completion_reason,
            metadata={"search_provider": self.config.search_provider},
        )

    async def _search_round(self, query: str, candidates: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Perform a single round of search.

        Args:
            query: Search query string
            candidates: List of candidate entities to look for

        Returns:
            Dict mapping candidate names to their extracted attributes
        """
        search_results = await self.search(query)

        # Extract attributes for each candidate mentioned in results
        candidate_attributes: Dict[str, Dict[str, str]] = {}
        for result in search_results:
            # Check which candidates are mentioned
            text = f"{result.title} {result.snippet}".lower()
            for candidate in candidates:
                if candidate.lower() in text:
                    if candidate not in candidate_attributes:
                        candidate_attributes[candidate] = {}

                    # Extract attribute values using simple patterns
                    extracted = self._extract_attributes_from_snippet(result.snippet, candidate)
                    for attr_name, attr_value in extracted.items():
                        if attr_value:
                            candidate_attributes[candidate][attr_name] = attr_value

        return candidate_attributes

    def _build_broad_query(self, question: str, missing_attribute: str, top_candidates: List[str]) -> str:
        """Build a broad search query covering multiple candidates."""
        candidates_str = " ".join(top_candidates[:2])
        # Clean and construct query
        question_clean = re.sub(r'[^\w\s]', ' ', question).strip()
        missing_clean = re.sub(r'[^\w\s]', ' ', missing_attribute).strip()
        return f"{question_clean} {missing_clean} {candidates_str}".strip()

    def _build_targeted_query(self, candidate: str, missing_attribute: str) -> str:
        """Build a targeted search query for a single candidate."""
        candidate_clean = re.sub(r'[^\w\s]', ' ', candidate).strip()
        missing_clean = re.sub(r'[^\w\s]', ' ', missing_attribute).strip()
        return f"{candidate_clean} {missing_clean}".strip()

    def _extract_attributes_from_snippet(self, snippet: str, candidate: str) -> Dict[str, str]:
        """
        Extract attribute values from search snippet.

        Uses simple regex patterns to find common patterns like:
        - "X is Y" or "X: Y"
        - "current team: Y"
        - "plays for Y"
        """
        if not snippet or not isinstance(snippet, str):
            return {}
        extracted: Dict[str, str] = {}

        # Pattern: "candidate is/has/was/plays [attribute]"
        patterns = [
            rf"{re.escape(candidate)}\s+(?:is|has|was|plays for)\s+([^.]+)",
            rf"{re.escape(candidate)}:\s+([^.]+)",
            r"(?:current|latest)?\s*(?:team|club|coach|manager|position):\s*([^.]+)",
        ]

        snippet_lower = snippet.lower()
        for pattern in patterns:
            match = re.search(pattern, snippet_lower, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Clean up the value
                value = re.sub(r'\s+', ' ', value)
                if len(value) > 2 and len(value) < 100:
                    # Infer attribute name from context
                    if "team" in snippet_lower or "club" in snippet_lower:
                        extracted["current team"] = value
                    elif "coach" in snippet_lower or "manager" in snippet_lower:
                        extracted["current coach"] = value
                    else:
                        extracted["attribute"] = value
                    break

        return extracted


class SearchResultFormatter:
    """Format search results for model consumption."""

    @staticmethod
    def format_for_model(results: AggregatedSearchResults, question: str) -> str:
        """
        Format aggregated search results for the model.

        Args:
            results: Aggregated search results
            question: Original question

        Returns:
            Formatted string for model consumption
        """
        if not results.candidate_attributes or results.total_rounds == 0:
            return ""

        lines = [
            "[BACKGROUND SEARCH RESULTS]",
            f"Question: {question}",
            f"Rounds: {results.total_rounds}",
            "",
            "[CANDIDATE INFORMATION]",
        ]

        # Sort candidates alphabetically
        for candidate in sorted(results.candidate_attributes.keys()):
            lines.append(f"{candidate}:")
            for attr_name, attr_value in results.candidate_attributes[candidate].items():
                lines.append(f"  - {attr_name}: {attr_value}")

        lines.extend([
            "",
            "Use this supplementary information to re-evaluate your answer.",
            "Final answer must STILL be an exact graph string from original candidates.",
        ])

        return "\n".join(lines)

    @staticmethod
    def format_structured(result: StructuredSearchResult) -> str:
        """Format a ``StructuredSearchResult`` for model consumption.

        The output emphasises per-candidate comparison and reminds the
        model that the final answer must still be an exact graph
        candidate string.
        """
        has_evidence = any(
            ev for ev in result.per_candidate_evidence.values()
        )
        if not has_evidence and not result.summary:
            return ""

        lines = [
            "[STRUCTURED SEARCH EVIDENCE]",
            f"Query: {result.query}",
        ]
        if result.date_relevance:
            lines.append(f"Date context: {result.date_relevance}")
        lines.append("")

        # Per-candidate evidence
        lines.append("[PER-CANDIDATE COMPARISON]")
        for candidate in sorted(result.per_candidate_evidence.keys()):
            evidence = result.per_candidate_evidence[candidate]
            lines.append(f"{candidate}:")
            if evidence:
                for snippet in evidence:
                    lines.append(f"  - {snippet}")
            else:
                lines.append("  (no evidence found)")

        if result.graph_candidate_supported:
            lines.append(
                f"\nBest supported candidate: {result.graph_candidate_supported}"
            )

        if result.source_urls:
            lines.append(f"\nSources: {', '.join(result.source_urls[:5])}")

        if result.summary:
            lines.append(f"\nSummary: {result.summary}")

        lines.extend([
            "",
            "IMPORTANT: Final answer must STILL be an exact graph string "
            "from the current candidate set. Search evidence is supplementary.",
        ])

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

async def quick_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """Quick search function with default configuration.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Search results
    """
    agent = BackgroundSearchAgent()
    return await agent.search(query, max_results=max_results)


async def quick_search_formatted(query: str, max_results: int = 5) -> str:
    """Quick search with formatted results ready for LLM consumption.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Formatted search results
    """
    agent = BackgroundSearchAgent()
    results_list = await agent.search(query, max_results=max_results)

    aggregated = AggregatedSearchResults()
    aggregated.add_results(query, results_list)

    return SearchResultFormatter.format_for_model(aggregated, max_results=max_results)
