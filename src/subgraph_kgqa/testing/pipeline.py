from __future__ import annotations

import asyncio

from subgraph_kgqa.legacy import load_repo_module


def run() -> None:
    module = load_repo_module("test_pipe6", "test_pipe6.py")
    runner = module.TestRunner()
    asyncio.run(runner.run())
