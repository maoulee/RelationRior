from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from subgraph_kgqa.paths import project_root


DEFAULT_BACKEND_CANDIDATES = (
    lambda root: root / "vendor" / "retrieve" / "graph_server.py",
    lambda _root: Path("/zhaoshu/SubgraphRAG-main/retrieve/graph_server.py"),
)


def resolve_backend_source(explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Backend source not found: {path}")

    env_path = os.getenv("SUBGRAPH_GRAPH_SERVER_SOURCE")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Backend source from SUBGRAPH_GRAPH_SERVER_SOURCE not found: {path}")

    root = project_root()
    for candidate_factory in DEFAULT_BACKEND_CANDIDATES:
        candidate = candidate_factory(root)
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Unable to locate graph_server.py. Set SUBGRAPH_GRAPH_SERVER_SOURCE to the backend file path."
    )


def run_backend(extra_args: Sequence[str], backend_source: str | None = None) -> int:
    source = resolve_backend_source(backend_source)
    env = os.environ.copy()
    pythonpath_parts = [str(project_root() / "src"), str(project_root())]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    result = subprocess.run([sys.executable, str(source), *extra_args], env=env, check=False)
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the KG graph backend.")
    parser.add_argument("--dataset", default=os.getenv("GRAPH_SERVER_DATASET", "webqsp"))
    parser.add_argument("--split", default=os.getenv("GRAPH_SERVER_SPLIT", "train"))
    parser.add_argument("--port", default=os.getenv("GRAPH_SERVER_PORT", "8001"))
    parser.add_argument(
        "--backend-source",
        default=None,
        help="Optional path to graph_server.py. Defaults to SUBGRAPH_GRAPH_SERVER_SOURCE or known local paths.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    extra_args = ["--dataset", str(args.dataset), "--split", str(args.split), "--port", str(args.port)]
    return run_backend(extra_args, backend_source=args.backend_source)


if __name__ == "__main__":
    raise SystemExit(main())
