#!/usr/bin/env python3
"""Regenerate all skill cards with new action-space-oriented prompts (deterministic fallback)."""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from subgraph_kgqa.skill_mining.case_skill import (
    _fallback_case_skill_from_dataset_case,
    render_case_skill_markdown,
)
from subgraph_kgqa.skill_mining.schemas import write_json

TRAIN_DATA = ROOT / "data" / "webqsp" / "webqsp_train.jsonl"
OUTPUT_ROOT = ROOT / "skills" / "webqsp_train_case_skills_v3"


def main():
    print(f"Loading training data from {TRAIN_DATA}...")
    cases = []
    with TRAIN_DATA.open() as handle:
        for line in handle:
            cases.append(json.loads(line))
    print(f"Loaded {len(cases)} training cases")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    written = 0
    errors = 0
    start = datetime.now()
    index = []

    for i, case in enumerate(cases):
        case_id = str(case.get("id", ""))
        if not case_id:
            continue
        try:
            card = _fallback_case_skill_from_dataset_case(case)
            json_path = OUTPUT_ROOT / f"{card.case_id}.json"
            md_path = OUTPUT_ROOT / f"{card.case_id}.md"
            write_json(json_path, card)
            md_path.write_text(render_case_skill_markdown(card), encoding="utf-8")
            index.append({
                "case_id": card.case_id,
                "question": card.question,
                "question_type": card.question_type,
                "core_relations": card.core_relations,
                "core_relation_domains": card.core_relation_domains,
                "action_space_experience": card.action_space_experience,
                "final_selection_experience": card.final_selection_experience,
            })
            written += 1
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  ERROR {case_id}: {exc}")

        if (i + 1) % 500 == 0:
            elapsed = (datetime.now() - start).total_seconds()
            print(f"  [{i+1}/{len(cases)}] {written} cards, {errors} errors, {elapsed:.0f}s")

    index_path = OUTPUT_ROOT / "index.json"
    write_json(index_path, index)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\nDone: {written} cards written, {errors} errors in {elapsed:.1f}s")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
