from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

for path in (SRC, ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from subgraph_kgqa.testing.pipeline import run


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        print("Usage: python evaluate.py")
        print("")
        print("Runs the offline KGQA evaluation pipeline using the unified project entrypoint.")
        raise SystemExit(0)
    run()
