from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

for path in (SRC, ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from subgraph_kgqa.rl.plugin import *  # noqa: F401,F403
