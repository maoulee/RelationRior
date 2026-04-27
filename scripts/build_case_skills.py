from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subgraph_kgqa.skill_mining.case_skill import build_case_skill_outputs, sample_case_ids


def _load_dataset_lookup(path: Path) -> dict:
    lookup = {}
    if not path.exists():
        return lookup
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            lookup[row["id"]] = row
    return lookup


async def main() -> int:
    parser = argparse.ArgumentParser(description="Build single-case skill cards from source/atomic cards.")
    parser.add_argument("--skills-root", default="skills")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-path", default="data/webqsp/webqsp_train.jsonl")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    case_ids = args.case_id or []
    if args.sample > 0 and not case_ids:
        case_ids = sample_case_ids(Path(args.skills_root), sample_size=args.sample, seed=args.seed)
        print(f"sampled_case_ids={case_ids}")

    dataset_lookup = _load_dataset_lookup(Path(args.data_path))

    result = await build_case_skill_outputs(
        Path(args.skills_root),
        use_llm=not args.skip_llm,
        case_ids=case_ids or None,
        dataset_lookup=dataset_lookup,
        concurrency=args.concurrency,
    )
    print(result["index_path"])
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
