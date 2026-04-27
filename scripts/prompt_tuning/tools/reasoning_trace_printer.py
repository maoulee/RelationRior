#!/usr/bin/env python3
"""Print detailed reasoning traces from existing test results.

For each case (or filtered subset), prints a turn-by-turn trace showing
reasoning, tool calls, backend results, consistency metadata, and frontend
errors. Designed for debugging the selective decision consistency mechanism.

Usage:
    PYTHONPATH=/zhaoshu/subgraph/src python3 scripts/prompt_tuning/tools/reasoning_trace_printer.py \
      --results /path/to/results.json \
      --focus fe \
      --compact

    PYTHONPATH=/zhaoshu/subgraph/src python3 scripts/prompt_tuning/tools/reasoning_trace_printer.py \
      --results /path/to/results.json \
      --case-ids WebQTest-301 WebQTest-1458
"""
import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Raw-response parsing helpers
# ---------------------------------------------------------------------------

def extract_reasoning(raw: str) -> str:
    """Extract the <reasoning>...</reasoning> block from raw_response."""
    m = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: look for [PHASE ...] markers
    if "[PHASE" in raw:
        return raw.strip()
    return ""


def extract_action(raw: str) -> str:
    """Extract the <act>...</act> or <answer>...</answer> block."""
    m = re.search(r"<act>(.*?)</act>", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def extract_tool_from_query(q: dict) -> str:
    """Summarise a parsed query as a one-liner."""
    tool = q.get("tool_name", "?")
    args = q.get("arguments", {})
    parts = []
    for k, v in args.items():
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v[:4])
            if len(args[k]) > 4:
                v += ", ..."
        elif isinstance(v, dict):
            v = json.dumps(v, ensure_ascii=False)[:80]
        parts.append(f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}")
    return f'{tool}({", ".join(parts)})'


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int = 200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _format_fe_errors(errors) -> list[str]:
    """Format frontend_errors list into readable strings."""
    if not isinstance(errors, list):
        return []
    lines = []
    for e in errors:
        msg = e.get("message", str(e))
        # Try to extract structured fields from the message string
        code = e.get("code", "")
        etype = ""
        reason = ""
        actual = ""
        # Parse FrontendError(...) string
        m = re.search(r"error_type='(\w+)'", msg)
        if m:
            etype = m.group(1)
        m = re.search(r"reason='(.*?)'", msg)
        if m:
            reason = m.group(1)
        m = re.search(r"actual_value='(.*?)'", msg)
        if m:
            actual = m.group(1)
        if etype:
            summary = f"[{etype}]"
            if actual:
                summary += f" value='{actual}'"
            if reason:
                summary += f" -- {reason}"
            lines.append(summary)
        else:
            lines.append(msg[:300])
    return lines


def _format_backend_summary(results) -> str:
    """Summarise backend_results."""
    if not isinstance(results, list) or not results:
        return "(none)"
    parts = []
    for r in results[:3]:
        tool = r.get("tool_name", "?")
        ok = r.get("is_success", r.get("status") == "SUCCESS")
        status = "OK" if ok else "FAIL"
        resp = r.get("response_text", "")
        # Extract key info from response
        extra = ""
        if "Relation Candidates" in resp or "relation" in resp.lower():
            m = re.search(r"(\d+)\s+relation", resp)
            if m:
                extra = f", {m.group(1)} relations"
        elif "Entity Candidates" in resp:
            m = re.search(r"(\d+)\s+candidat", resp, re.I)
            if m:
                extra = f", {m.group(1)} candidates"
        elif len(resp) > 0:
            # Count lines
            nlines = resp.count("\n") + 1
            extra = f", {nlines} lines"
        parts.append(f"{tool}:{status}{extra}")
    if len(results) > 3:
        parts.append(f"... +{len(results)-3} more")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Case printing
# ---------------------------------------------------------------------------

def print_case_trace(case: dict, compact: bool = False, file=None):
    """Print detailed trace for a single case."""
    out = file or sys.stdout
    cid = case.get("case_id", "?")
    f1 = case.get("f1", 0)
    fe_total = case.get("frontend_errors", 0)
    if isinstance(fe_total, list):
        fe_total = len(fe_total)
    turns_n = case.get("turns", len(case.get("trajectory", [])))
    question = case.get("question", "")
    gt = case.get("ground_truth", [])
    pred = case.get("predicted", [])
    consistency_summary = case.get("consistency", {})

    # Header
    out.write(f"\n=== {cid} (F1={f1:.2f}, FE={fe_total}, Turns={turns_n}) ===\n")
    out.write(f"Question: {question}\n")
    out.write(f"GT: {', '.join(str(x) for x in gt) if isinstance(gt, list) else gt}\n")
    out.write(f"Predicted: {', '.join(str(x) for x in pred) if isinstance(pred, list) else pred}\n")

    # Consistency summary
    used = consistency_summary.get("used_turns", 0)
    disagreed = consistency_summary.get("disagreement_turns", 0)
    out.write(f"Consistency: used={used}, disagreements={disagreed}\n")

    trajectory = case.get("trajectory", [])

    for step in trajectory:
        turn_num = step.get("turn", "?")
        cons = step.get("consistency", {})
        cons_used = cons.get("consistency_used", False)
        cons_mode = cons.get("consistency_mode", "")
        critique_any_issue = cons.get("critique_any_issue", None)
        critique_recommended = cons.get("critique_recommended", False)
        critique_issues = cons.get("critique_issues", [])

        raw = step.get("raw_response", "")
        parsed = step.get("parsed_output", {}) or {}
        queries = parsed.get("queries", [])
        candidates = parsed.get("candidates", [])
        final_answer = parsed.get("final_answer", [])

        executed = step.get("executed_queries", [])
        backend = step.get("backend_results", [])
        fe_list = step.get("frontend_errors", [])
        fe_count = len(fe_list) if isinstance(fe_list, list) else fe_list

        reasoning = extract_reasoning(raw)
        action = extract_action(raw)

        # Consistency tag
        cons_tag = "Y" if cons_used else "N"
        probe_tag = ""
        if critique_any_issue is not None:
            probe_tag = f", critique={'FAIL' if critique_any_issue else 'OK'}"
        if critique_recommended:
            probe_tag += ", re-reason"

        if compact:
            # One-line per turn
            q_str = "; ".join(extract_tool_from_query(q) for q in queries) if queries else "(none)"
            fa_str = ", ".join(str(x) for x in final_answer[:4]) if final_answer else "-"
            fe_str = f" FE={fe_count}" if fe_count else ""
            cons_extra = ""
            if cons_used:
                cons_extra = " [CONSISTENCY]"
                chosen = cons.get("consistency_chosen_signature", "")
                if chosen:
                    cons_extra += f" sig={chosen[:60]}"
            out.write(
                f"  T{turn_num} [c={cons_tag}{probe_tag}]: "
                f"queries={len(queries)} cands={len(candidates)} fa=[{fa_str}]{fe_str}{cons_extra}\n"
            )
            continue

        # Full trace
        out.write(f"\n  Turn {turn_num} [consistency={cons_tag}{probe_tag}]:\n")

        # Consistency details
        if cons_used:
            votes = cons.get("consistency_votes", {})
            agreed = cons.get("consistency_agreed_initially", "")
            chosen = cons.get("consistency_chosen_signature", "")
            sigs = cons.get("consistency_candidate_signatures", [])
            out.write(f"    Consistency triggered: YES (reason: {cons.get('consistency_mode', 'unknown')})\n")
            out.write(f"    Agreed initially: {agreed}\n")
            out.write(f"    Votes: {json.dumps(votes)}\n")
            if sigs:
                match = "MATCH" if len(set(sigs)) == 1 else "MISS"
                for i, s in enumerate(sigs[:3]):
                    out.write(f"    sig{i+1}={s[:80]}\n")
                out.write(f"    Signatures: {match}\n")
            if chosen:
                out.write(f"    Chosen: {chosen[:100]}\n")

        # Reasoning (first 400 chars)
        if reasoning:
            out.write(f"    Reasoning: {_truncate(reasoning, 400)}\n")

        # Action
        if action:
            out.write(f"    Action: {_truncate(action, 300)}\n")

        # Queries / tool calls
        if queries:
            out.write("    Queries:\n")
            for q in queries:
                out.write(f"      {extract_tool_from_query(q)}\n")
        elif executed:
            out.write("    Executed queries:\n")
            for q in executed:
                out.write(f"      {extract_tool_from_query(q)}\n")
        else:
            out.write("    Queries: (none)\n")

        # Backend results
        out.write(f"    Backend: {_format_backend_summary(backend)}\n")

        # Candidates
        if candidates:
            cands_str = ", ".join(str(c)[:60] for c in candidates[:6])
            if len(candidates) > 6:
                cands_str += f", ... +{len(candidates)-6}"
            out.write(f"    Candidates: [{cands_str}]\n")
        else:
            out.write("    Candidates: (none)\n")

        # Final answer
        if final_answer:
            fa_str = ", ".join(str(x)[:60] for x in final_answer[:6])
            out.write(f"    Final answer: {fa_str}\n")
        else:
            out.write("    Final answer: (none)\n")

        # Frontend errors
        if fe_count:
            fe_formatted = _format_fe_errors(fe_list)
            out.write(f"    FE: {fe_count}\n")
            for fe_line in fe_formatted:
                out.write(f"      {fe_line}\n")
        else:
            out.write("    FE: 0\n")

    out.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print detailed reasoning traces from test results"
    )
    parser.add_argument(
        "--results", required=True, type=Path,
        help="Path to results.json file"
    )
    parser.add_argument(
        "--focus", choices=["fe", "low-f1", "all"], default="all",
        help="Filter: 'fe' = only cases with frontend errors, "
             "'low-f1' = only F1=0 cases, 'all' = everything"
    )
    parser.add_argument(
        "--case-ids", nargs="*", default=None,
        help="Specific case IDs to print (e.g. WebQTest-301 WebQTest-1458)"
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="One-line per turn summary"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Also save output as .md file next to results.json"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of cases to print"
    )
    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        data = json.load(f)

    # Build case lookup
    cases_by_id = {c["case_id"]: c for c in data}

    # Select cases
    if args.case_ids:
        selected = []
        for cid in args.case_ids:
            if cid in cases_by_id:
                selected.append(cases_by_id[cid])
            else:
                print(f"WARNING: case {cid} not found in results", file=sys.stderr)
    else:
        selected = list(data)

    # Apply focus filter
    if args.focus == "fe":
        selected = [
            c for c in selected
            if (len(c["frontend_errors"]) if isinstance(c["frontend_errors"], list)
                else c.get("frontend_errors", 0)) > 0
        ]
    elif args.focus == "low-f1":
        selected = [c for c in selected if c.get("f1", 1) == 0]

    # Apply limit
    if args.limit:
        selected = selected[:args.limit]

    # Sort by FE descending (most broken first)
    def _fe_count(c):
        fe = c.get("frontend_errors", 0)
        return len(fe) if isinstance(fe, list) else fe
    selected.sort(key=_fe_count, reverse=True)

    if not selected:
        print("No cases matched the filter criteria.", file=sys.stderr)
        return

    print("# Reasoning Trace Report", file=sys.stderr)
    print(f"# Source: {args.results}", file=sys.stderr)
    print(f"# Cases: {len(selected)}", file=sys.stderr)
    print(f"# Mode: {'compact' if args.compact else 'full'}", file=sys.stderr)

    # Collect output
    import io
    buf = io.StringIO()

    for case in selected:
        print_case_trace(case, compact=args.compact, file=buf)

    output = buf.getvalue()
    sys.stdout.write(output)
    sys.stdout.flush()

    # Optionally save
    if args.save:
        out_path = args.results.with_suffix(".trace.md")
        with open(out_path, "w") as f:
            f.write("# Reasoning Trace Report\n\n")
            f.write(f"Source: `{args.results}`\n\n")
            f.write(f"Cases: {len(selected)} | Mode: {'compact' if args.compact else 'full'}\n\n")
            f.write("```\n")
            f.write(output)
            f.write("```\n")
        print(f"\n# Saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
