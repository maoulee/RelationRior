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

from subgraph_kgqa.skill_mining.render import render_readable_outputs


async def main() -> int:
    parser = argparse.ArgumentParser(description="Render human-readable skill cards from atomic/source cards.")
    parser.add_argument("--skills-root", default="skills")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()

    result = await render_readable_outputs(
        Path(args.skills_root),
        use_llm=not args.skip_llm,
        case_ids=args.case_id or None,
    )
    print(result["summary_path"])
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
