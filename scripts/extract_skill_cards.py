from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subgraph_kgqa.skill_mining import SkillMiningExtractor


async def main() -> int:
    parser = argparse.ArgumentParser(description="Extract updateable skill raw materials and atomic cards from existing reports.")
    parser.add_argument(
        "--batch-report",
        default="reports/inference_runtime_batch/prompt_branch30/protocol_guard_action_id_experiment.md",
    )
    parser.add_argument("--data-path", default="data/webqsp/webqsp_train.jsonl")
    parser.add_argument("--skills-root", default="skills")
    parser.add_argument("--smoke-reports-dir", default="reports")
    parser.add_argument("--kg-api-url", default="http://127.0.0.1:8001")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    extractor = SkillMiningExtractor(
        output_root=Path(args.skills_root),
        data_path=Path(args.data_path),
        batch_report_path=Path(args.batch_report),
        smoke_reports_dir=Path(args.smoke_reports_dir),
        kg_api_url=args.kg_api_url,
        use_llm=not args.skip_llm,
    )
    summary = await extractor.run(limit=args.limit, concurrency=args.concurrency)
    print(Path(args.skills_root) / "extraction_summary.json")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
