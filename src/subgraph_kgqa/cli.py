from __future__ import annotations

import argparse

from subgraph_kgqa.backend.server import main as graph_server_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for the Subgraph KGQA project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    graph_server_parser = subparsers.add_parser("graph-server", help="Run the graph backend.")
    graph_server_parser.add_argument("--dataset", default="webqsp")
    graph_server_parser.add_argument("--split", default="train")
    graph_server_parser.add_argument("--port", default="8001")
    graph_server_parser.add_argument("--backend-source", default=None)

    subparsers.add_parser("evaluate", help="Run the offline evaluation pipeline.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "graph-server":
        return graph_server_main(
            [
                "--dataset",
                str(args.dataset),
                "--split",
                str(args.split),
                "--port",
                str(args.port),
                *(["--backend-source", args.backend_source] if args.backend_source else []),
            ]
        )

    if args.command == "evaluate":
        from subgraph_kgqa.testing.pipeline import run as run_evaluation

        run_evaluation()
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
