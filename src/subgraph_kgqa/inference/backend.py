from __future__ import annotations

import asyncio
import datetime as dt
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import requests


def _default_kg_api_url() -> str:
    return os.getenv("KGQA_KG_API_URL", "http://localhost:8001")


def _web_search_enabled() -> bool:
    return os.getenv("KGQA_ENABLE_WEB_SEARCH", "0").strip().lower() in {"1", "true", "yes", "on"}


def _graph_snapshot_date() -> str:
    return os.getenv("KGQA_GRAPH_SNAPSHOT_DATE", "").strip()


def _graph_snapshot_year() -> str:
    snapshot_date = _graph_snapshot_date()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", snapshot_date):
        return snapshot_date[:4]
    if re.match(r"^\d{4}$", snapshot_date):
        return snapshot_date
    return ""


def _wikidata_first_enabled() -> bool:
    return os.getenv("KGQA_WEB_SEARCH_WIKIDATA_FIRST", "1").strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_ENDPOINTS = {
    "find_entities": f"{_default_kg_api_url()}/v2/find_entities",
    "check_entities": f"{_default_kg_api_url()}/v2/find_entities",
    "explore_schema": f"{_default_kg_api_url()}/v2/explore_schema",
    "plan_subquestion": f"{_default_kg_api_url()}/v2/plan_subquestion",
    "plan": f"{_default_kg_api_url()}/v2/plan_subquestion",
    "find_logical_path_with_relation": f"{_default_kg_api_url()}/v2/find_logical_path_with_relation",
    "match_pattern": f"{_default_kg_api_url()}/v2/match_pattern",
    "action": f"{_default_kg_api_url()}/v2/match_pattern",
    "get_neighbors": f"{_default_kg_api_url()}/v2/get_neighbors",
    "filter": f"{_default_kg_api_url()}/v2/filter",
}


class GraphBackendClient:
    """Inference-time backend client decoupled from the evaluation runner."""

    def __init__(
        self,
        endpoints: Optional[Dict[str, str]] = None,
        *,
        default_max_hops: int = 4,
        default_path_limit: int = 3,
    ) -> None:
        self.endpoints = endpoints or DEFAULT_ENDPOINTS
        self.default_max_hops = default_max_hops
        self.default_path_limit = default_path_limit

    async def execute_tool(
        self,
        session: aiohttp.ClientSession,
        query: Dict[str, Any],
        sample_id: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tool_name = query.get("tool_name")
        args = dict(query.get("arguments", {}))
        if tool_name in {"web_search", "search"}:
            return await self._execute_web_search(args, state or {})

        url = self.endpoints.get(tool_name)
        if not url:
            return self._error_result("KG_ERROR", f"Error: Tool '{tool_name}' not supported.")

        payload = {**args, "sample_id": sample_id}

        if tool_name == "filter":
            scope = args.get("scope", "selected")
            if scope == "all":
                candidates = list((state or {}).get("all_leaf_entities", set()))
                if not candidates:
                    return self._error_result(
                        "KG_ERROR",
                        "[Filter Error] No leaf entities available for scope='all'. Run action() first.",
                    )
                payload["candidates"] = candidates
            else:
                candidates = list((state or {}).get("retrieved_candidates", set()))
                if not candidates:
                    return self._error_result(
                        "KG_ERROR",
                        "[Filter Error] No candidates available. Run action() first to collect candidates.",
                    )
                payload["candidates"] = candidates

        if tool_name in {"match_pattern", "action"}:
            if "anchor" not in args and "start_entity" in args:
                args["anchor"] = args["start_entity"]
            if isinstance(args.get("anchor"), list) and args["anchor"]:
                args["anchor"] = args["anchor"][0]
            payload = {**args, "sample_id": sample_id}

        if tool_name == "plan":
            raw_anchor = args.get("anchor") or args.get("anchor_entity") or args.get("start_entity")
            if isinstance(raw_anchor, list):
                anchor_str = raw_anchor[0] if raw_anchor else ""
            else:
                anchor_str = str(raw_anchor) if raw_anchor else ""
            payload = {
                "sample_id": sample_id,
                "question": args.get("question") or args.get("subquestion") or "",
                "anchor": anchor_str,
                "related": args.get("related") or [],
                "maybe_related": args.get("maybe_related") or [],
                "constraint_relations": args.get("constraint_relations") or args.get("select_relations") or [],
                "constraint_entities": args.get("constraint_entities") or args.get("select_entities") or [],
                "max_hops": args.get("max_hops", self.default_max_hops),
                "path_limit": args.get("path_limit", self.default_path_limit),
            }

        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    return self._error_result(
                        "HTTP_ERROR",
                        f"HTTP Error {response.status}: {await response.text()}",
                    )
                return await response.json()
        except Exception as exc:
            return self._error_result("EXCEPTION", f"Client Exception: {exc}")

    async def _execute_web_search(self, args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        if not _web_search_enabled():
            return self._error_result(
                "KG_ERROR",
                "[Web Search Disabled] web_search is disabled. Set KGQA_ENABLE_WEB_SEARCH=1 to enable it.",
            )
        try:
            return await asyncio.to_thread(self._execute_web_search_sync, args, state)
        except Exception as exc:
            return self._error_result("KG_ERROR", f"[Web Search Error] {exc}")

    @staticmethod
    def _load_search_client():
        try:
            from ddgs import DDGS  # type: ignore

            return DDGS
        except Exception:
            try:
                from duckduckgo_search import DDGS  # type: ignore

                return DDGS
            except Exception:
                return None

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()

    @staticmethod
    def _parse_wikidata_time(value: str) -> Optional[dt.date]:
        text = str(value or "").strip()
        if not text:
            return None
        match = re.match(r"^([+-]?\d{4})-(\d{2})-(\d{2})", text)
        if not match:
            return None
        year = int(match.group(1))
        if year <= 0:
            return None
        try:
            return dt.date(year, int(match.group(2)), int(match.group(3)))
        except Exception:
            return None

    @classmethod
    def _snapshot_status(
        cls,
        start_raw: str,
        end_raw: str,
        snapshot_date: Optional[dt.date],
    ) -> str:
        if snapshot_date is None:
            return "UNKNOWN"
        start = cls._parse_wikidata_time(start_raw)
        end = cls._parse_wikidata_time(end_raw)
        if start and start > snapshot_date:
            return "STARTS_AFTER_SNAPSHOT"
        if end and end < snapshot_date:
            return "ENDED_BEFORE_SNAPSHOT"
        if (start is None or start <= snapshot_date) and (end is None or end >= snapshot_date):
            return "ACTIVE_AT_SNAPSHOT"
        return "UNKNOWN"

    @classmethod
    def _wikidata_search_entity(cls, name: str) -> Optional[Dict[str, str]]:
        if not name:
            return None
        response = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "format": "json",
                "language": "en",
                "type": "item",
                "search": name,
                "limit": 5,
            },
            headers={"User-Agent": "subgraph-codex/1.0"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        target_norm = cls._normalize_text(name)
        hits = data.get("search", []) or []
        for item in hits:
            label = str(item.get("label", "") or "")
            if cls._normalize_text(label) == target_norm:
                return {
                    "id": str(item.get("id", "") or ""),
                    "label": label,
                    "description": str(item.get("description", "") or ""),
                }
        if not hits:
            return None
        best = hits[0]
        return {
            "id": str(best.get("id", "") or ""),
            "label": str(best.get("label", "") or ""),
            "description": str(best.get("description", "") or ""),
        }

    @staticmethod
    def _infer_wikidata_property(query: str) -> Optional[Tuple[str, str]]:
        lowered = str(query or "").lower()
        if any(token in lowered for token in ("wife", "husband", "spouse", "married")):
            return ("P26", "spouse")
        if (
            "play for" in lowered
            or "plays for" in lowered
            or "team" in lowered
            or "signed with" in lowered
        ):
            return ("P54", "member of sports team")
        return None

    @classmethod
    def _fetch_wikidata_statement_rows(cls, anchor_qid: str, property_id: str) -> List[Dict[str, str]]:
        sparql = f"""
SELECT ?obj ?objLabel ?start ?end WHERE {{
  wd:{anchor_qid} p:{property_id} ?st.
  ?st ps:{property_id} ?obj.
  OPTIONAL {{ ?st pq:P580 ?start. }}
  OPTIONAL {{ ?st pq:P582 ?end. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY DESC(?start)
"""
        response = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": sparql},
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "subgraph-codex/1.0",
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        rows: List[Dict[str, str]] = []
        for row in data.get("results", {}).get("bindings", []) or []:
            rows.append(
                {
                    "obj_qid": str(row.get("obj", {}).get("value", "")).rsplit("/", 1)[-1],
                    "obj_label": str(row.get("objLabel", {}).get("value", "") or ""),
                    "start": str(row.get("start", {}).get("value", "") or ""),
                    "end": str(row.get("end", {}).get("value", "") or ""),
                }
            )
        return rows

    @classmethod
    def _build_wikidata_probe(
        cls,
        *,
        query: str,
        candidates: List[str],
        state: Dict[str, Any],
        snapshot_date_text: str,
    ) -> Optional[Dict[str, Any]]:
        if not _wikidata_first_enabled():
            return None
        property_spec = cls._infer_wikidata_property(query)
        if not property_spec:
            return None

        prompt_context = state.get("prompt_context", {}) or {}
        anchors = [str(x).strip() for x in prompt_context.get("core_entities", set()) if str(x).strip()]
        if not anchors:
            anchors = [str(x).strip() for x in state.get("verified_entities", set()) if str(x).strip()]
        if not anchors:
            return None
        anchor_name = anchors[0]
        anchor_hit = cls._wikidata_search_entity(anchor_name)
        if not anchor_hit or not anchor_hit.get("id"):
            return None

        property_id, property_label = property_spec
        rows = cls._fetch_wikidata_statement_rows(anchor_hit["id"], property_id)
        if not rows:
            return None

        candidate_map = {cls._normalize_text(candidate): candidate for candidate in candidates}
        snapshot_date = cls._parse_wikidata_time(snapshot_date_text) if snapshot_date_text else None
        filtered_rows: List[Dict[str, str]] = []
        for row in rows:
            row_norm = cls._normalize_text(row["obj_label"])
            chosen_candidate = candidate_map.get(row_norm)
            if candidates and not chosen_candidate:
                continue
            status = cls._snapshot_status(row["start"], row["end"], snapshot_date)
            filtered_rows.append(
                {
                    "candidate": chosen_candidate or row["obj_label"],
                    "wikidata_label": row["obj_label"],
                    "wikidata_qid": row["obj_qid"],
                    "start": row["start"],
                    "end": row["end"],
                    "snapshot_status": status,
                }
            )

        if not filtered_rows:
            return None

        status_priority = {
            "ACTIVE_AT_SNAPSHOT": 0,
            "UNKNOWN": 1,
            "ENDED_BEFORE_SNAPSHOT": 2,
            "STARTS_AFTER_SNAPSHOT": 3,
        }
        filtered_rows.sort(key=lambda item: (status_priority.get(item["snapshot_status"], 9), item["candidate"]))

        lines = [
            "[WIKIDATA STRUCTURED EVIDENCE]",
            f"Anchor: {anchor_hit['label']} ({anchor_hit['id']})",
            f"Property: {property_label} ({property_id})",
        ]
        if snapshot_date_text:
            lines.append(f"Graph snapshot date: {snapshot_date_text}")
            lines.append("Interpret CURRENT / LATEST relative to this snapshot date, not real-world today.")
        for row in filtered_rows[:10]:
            start = row["start"] or "?"
            end = row["end"] or "open"
            lines.append(
                f"- {row['candidate']} | start={start} | end={end} | snapshot_status={row['snapshot_status']}"
            )
        return {
            "text": "\n".join(lines),
            "structured": {
                "provider": "wikidata",
                "anchor": anchor_hit,
                "property_id": property_id,
                "property_label": property_label,
                "snapshot_date": snapshot_date_text,
                "rows": filtered_rows,
            },
        }

    @classmethod
    def _build_default_search_query(cls, state: Dict[str, Any], candidates: List[str]) -> str:
        prompt_context = state.get("prompt_context", {}) or {}
        question = str(prompt_context.get("original_question", "") or "").strip()
        if not question:
            question = str(prompt_context.get("question", "") or "").strip()
        candidate_part = " ".join(candidates[:4]).strip()
        action_part = ""
        action_id = state.get("last_selected_action_id")
        action_hint = (state.get("action_id_map", {}) or {}).get(action_id, {}) if action_id else {}
        signature = action_hint.get("signature", []) if isinstance(action_hint, dict) else []
        if signature:
            rel_tokens: List[str] = []
            for step in signature:
                if not isinstance(step, dict):
                    continue
                relation = str(step.get("relation", "") or "").strip()
                if not relation:
                    continue
                rel_tokens.append(relation.split(".")[-1].replace("_", " "))
            if rel_tokens:
                action_part = " ".join(rel_tokens[:3]).strip()
        query = " ".join(part for part in [question, candidate_part, action_part] if part).strip()
        return query or candidate_part or question

    @classmethod
    def _execute_web_search_sync(cls, args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        ddgs_cls = cls._load_search_client()
        if ddgs_cls is None:
            return cls._error_result(
                "KG_ERROR",
                "[Web Search Error] Neither `ddgs` nor `duckduckgo_search` is available in the environment.",
            )

        candidates = args.get("candidates", []) or []
        if isinstance(candidates, str):
            candidates = [candidates]
        candidates = [str(item).strip() for item in candidates if str(item).strip()]
        if not candidates:
            state_candidates = list(state.get("retrieved_candidates", set()) or [])
            candidates = [str(item).strip() for item in state_candidates if str(item).strip()][:8]

        query = str(args.get("query", "") or "").strip()
        if not query:
            query = cls._build_default_search_query(state, candidates)
        if not query:
            return cls._error_result("KG_ERROR", "[Web Search Error] Could not build a search query from the current question and candidates.")

        try:
            top_k = int(args.get("top_k", 5) or 5)
        except Exception:
            top_k = 5
        top_k = max(1, min(top_k, 10))

        snapshot_date_text = _graph_snapshot_date()
        snapshot_year = _graph_snapshot_year()
        effective_query = query
        if snapshot_year and not re.search(r"\b(19|20)\d{2}\b", query):
            effective_query = f"{query} as of {snapshot_year}"

        wikidata_probe = cls._build_wikidata_probe(
            query=query,
            candidates=candidates,
            state=state,
            snapshot_date_text=snapshot_date_text,
        )

        results_raw = ddgs_cls().text(effective_query, max_results=top_k)
        results: List[Dict[str, str]] = []
        for item in results_raw[:top_k]:
            title = str(item.get("title", "") or "").strip()
            url = str(item.get("href", "") or item.get("url", "") or "").strip()
            snippet = str(item.get("body", "") or item.get("snippet", "") or "").strip()
            if not (title or url or snippet):
                continue
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )

        if not results:
            return {
                "status": "KG_NO_MATCHES",
                "response_text": "[Web Search] No search results found for the current disambiguation query.",
                "found_end_entities": [],
                "structured_data": {
                    "query": query,
                    "web_results": [],
                    "candidate_mentions": {},
                },
            }

        mention_counts = cls._count_candidate_mentions(results, candidates)
        response_lines = [
            "[WEB SEARCH RESULTS]",
            f"Query: {query}",
        ]
        if effective_query != query:
            response_lines.append(f"Effective query with snapshot hint: {effective_query}")
        if snapshot_date_text:
            response_lines.append(f"Graph snapshot date: {snapshot_date_text}")
            response_lines.append(
                "When resolving CURRENT / LATEST, prefer evidence that is active on or before the graph snapshot date."
            )
            response_lines.append(
                "Do not let later real-world evidence override graph-time semantics."
            )
        if wikidata_probe:
            response_lines.append("")
            response_lines.append(wikidata_probe["text"])
        if candidates:
            response_lines.append("Current graph candidates under comparison:")
            for candidate in candidates[:8]:
                response_lines.append(f"- {candidate}")
        response_lines.append("")
        for idx, item in enumerate(results, 1):
            response_lines.append(f"{idx}. {item['title']}")
            if item["url"]:
                response_lines.append(f"   URL: {item['url']}")
            if item["snippet"]:
                response_lines.append(f"   Snippet: {item['snippet']}")
            response_lines.append("")
        if mention_counts:
            response_lines.append("[CANDIDATE MENTION COUNTS]")
            for candidate, count in mention_counts.items():
                response_lines.append(f"- {candidate}: {count}")
            response_lines.append("")
        response_lines.append(
            "Use these results only to distinguish among the CURRENT graph candidates. Do not introduce a new answer."
        )

        return {
            "status": "KG_SUCCESS",
            "response_text": "\n".join(response_lines).strip(),
            "found_end_entities": [],
            "structured_data": {
                "query": query,
                "effective_query": effective_query,
                "snapshot_date": snapshot_date_text,
                "web_results": results,
                "candidate_mentions": mention_counts,
                "wikidata_probe": (wikidata_probe or {}).get("structured", {}),
            },
        }

    @staticmethod
    def _count_candidate_mentions(results: List[Dict[str, str]], candidates: List[str]) -> Dict[str, int]:
        mention_counts: Dict[str, int] = {}
        for candidate in candidates:
            pattern = re.compile(re.escape(candidate), re.IGNORECASE)
            count = 0
            for item in results:
                haystack = " ".join(
                    part for part in (item.get("title", ""), item.get("snippet", "")) if part
                )
                if pattern.search(haystack):
                    count += 1
            mention_counts[candidate] = count
        return mention_counts

    @staticmethod
    def _error_result(status: str, message: str) -> Dict[str, Any]:
        return {
            "status": status,
            "response_text": message,
            "found_end_entities": [],
            "structured_data": {},
        }
