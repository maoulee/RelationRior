#!/usr/bin/env python3
"""Test question decomposition ability of the local LLM on failed CWQ cases."""

import argparse
import json
import os
import re
import sys
import time

import requests


DECOMPOSITION_PROMPT = """You are a KGQA question decomposition expert. Given a natural language question, decompose it into structured components for knowledge graph traversal.

Question: {question}

Decompose into:
1. ANSWER_TYPE: What TYPE of entity is the answer? (Person/Location/Date/Organization/etc)
2. ANCHOR: Which entity mentioned in the question is the KNOWN starting point? (The entity we traverse FROM)
3. CORE_DOMAIN: Which knowledge domain contains the core relations? (government/people/location/sports/etc)
4. CONSTRAINT_ENTITIES: Which OTHER entities constrain/filter the answer? (NOT the anchor)

RULES:
- ANCHOR is the entity we START from in the graph
- CONSTRAINT_ENTITIES are used to FILTER results AFTER traversal
- ANCHOR ≠ CONSTRAINT_ENTITIES
- For "X of Y" patterns: Y is usually the anchor, X is the answer type
- For "who/what did X" patterns: X is usually the anchor

Output format (JSON only, no other text):
{{"answer_type": "...", "anchor": "...", "core_domain": "...", "constraint_entities": [...]}}"""


LLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen35-9b-local"


def call_llm(question: str, max_retries: int = 2) -> str:
    """Send decomposition prompt to local LLM and return raw response text."""
    prompt = DECOMPOSITION_PROMPT.format(question=question)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(LLM_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, KeyError, IndexError) as e:
            if attempt < max_retries:
                time.sleep(2)
            else:
                return f"ERROR: {e}"


def parse_decomposition(raw: str) -> dict:
    """Try to extract JSON from the LLM response."""
    # Try direct JSON parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in the text
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'(\{[^{}]*"answer_type"[^{}]*\})',
        r'(\{.*\})',
    ]
    for pattern in json_patterns:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return {"raw_text": raw}


def main():
    parser = argparse.ArgumentParser(description="Test LLM question decomposition on failed CWQ cases")
    parser.add_argument("--results", default="reports/stage_pipeline_test/cwq_100_format_fix/results.json",
                        help="Path to results.json with pipeline test results")
    parser.add_argument("--output-dir", default="reports/stage_pipeline_test/cwq_decomposition_test",
                        help="Directory to save decomposition results")
    parser.add_argument("--f1-threshold", type=float, default=0.5,
                        help="Only test cases with F1 below this threshold")
    parser.add_argument("--max-cases", type=int, default=100,
                        help="Maximum number of cases to test")
    args = parser.parse_args()

    # Load results
    with open(args.results, "r") as f:
        all_cases = json.load(f)

    # Filter to failed cases
    failed_cases = [c for c in all_cases if c["f1"] < args.f1_threshold]
    failed_cases = failed_cases[: args.max_cases]

    print(f"Loaded {len(all_cases)} total cases, {len(failed_cases)} failed (F1 < {args.f1_threshold})")
    print(f"LLM endpoint: {LLM_URL}")
    print(f"Model: {MODEL}")
    print("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    parse_successes = 0

    for i, case in enumerate(failed_cases):
        case_id = case["case_id"]
        question = case["question"]
        ground_truth = case["ground_truth"]
        f1 = case["f1"]

        # Call LLM
        raw_response = call_llm(question)
        parsed = parse_decomposition(raw_response)
        parse_ok = "raw_text" not in parsed
        if parse_ok:
            parse_successes += 1

        result = {
            "case_id": case_id,
            "question": question,
            "ground_truth": ground_truth,
            "f1": f1,
            "model_decomposition": parsed if parse_ok else {"raw_text": raw_response},
            "parse_success": parse_ok,
        }
        results.append(result)

        # Print progress
        if parse_ok:
            d = parsed
            decomp_str = (
                f"{d.get('answer_type', '?')} | "
                f"anchor={d.get('anchor', '?')} | "
                f"domain={d.get('core_domain', '?')} | "
                f"constraints={d.get('constraint_entities', [])}"
            )
        else:
            decomp_str = f"PARSE FAILED: {raw_response[:100]}..."

        print(f"[{i+1}/{len(failed_cases)}] {case_id} F1={f1:.2f}")
        print(f"  Q: {question}")
        print(f"  GT: {ground_truth}")
        print(f"  Decomposition: {decomp_str}")
        print()

    # Save results
    output_path = os.path.join(args.output_dir, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    print("=" * 80)
    print(f"Tested: {len(results)} cases")
    print(f"Parse success: {parse_successes}/{len(results)} ({parse_successes/len(results)*100:.1f}%)")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
