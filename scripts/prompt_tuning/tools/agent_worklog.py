#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _jsonl_path(log_arg: str) -> Path:
    path = Path(log_arg)
    if path.suffix != ".jsonl":
        path = path.with_suffix(".jsonl")
    return path


def _md_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".md")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _append_md(path: Path, lines: List[str]) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _safe_label(text: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip())
    return sanitized[:80] or "step"


def _base_event(kind: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "timestamp": _utc_now(),
        "event": kind,
        "cwd": os.getcwd(),
    }
    payload.update(extra)
    return payload


def cmd_start(args: argparse.Namespace) -> int:
    jsonl = _jsonl_path(args.log)
    md = _md_path(jsonl)
    payload = _base_event("start", title=args.title)
    _append_jsonl(jsonl, payload)
    _append_md(
        md,
        [
            f"# Worklog: {args.title}",
            "",
            f"- Started: `{payload['timestamp']}`",
            f"- CWD: `{payload['cwd']}`",
            "",
        ],
    )
    return 0


def cmd_note(args: argparse.Namespace) -> int:
    jsonl = _jsonl_path(args.log)
    md = _md_path(jsonl)
    payload = _base_event("note", message=args.message)
    _append_jsonl(jsonl, payload)
    _append_md(md, [f"- `{payload['timestamp']}` NOTE: {args.message}"])
    return 0


def cmd_finish(args: argparse.Namespace) -> int:
    jsonl = _jsonl_path(args.log)
    md = _md_path(jsonl)
    payload = _base_event("finish", status=args.status)
    _append_jsonl(jsonl, payload)
    _append_md(md, ["", f"- Finished: `{payload['timestamp']}`", f"- Status: `{args.status}`", ""])
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.command:
        raise SystemExit("run requires a command after '--'")

    jsonl = _jsonl_path(args.log)
    md = _md_path(jsonl)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    step_id = f"{timestamp}_{_safe_label(args.label)}"
    stdout_path = jsonl.parent / f"{step_id}.stdout.log"
    stderr_path = jsonl.parent / f"{step_id}.stderr.log"

    start_payload = _base_event(
        "run_start",
        label=args.label,
        argv=args.command,
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
    )
    _append_jsonl(jsonl, start_payload)
    _append_md(
        md,
        [
            "",
            f"## {args.label}",
            f"- Started: `{start_payload['timestamp']}`",
            f"- Command: `{json.dumps(args.command, ensure_ascii=True)}`",
            f"- stdout: `{stdout_path}`",
            f"- stderr: `{stderr_path}`",
        ],
    )

    start_time = time.time()
    _ensure_parent(stdout_path)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        proc = subprocess.Popen(
            args.command,
            cwd=os.getcwd(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        return_code = proc.wait()
    duration_sec = round(time.time() - start_time, 3)

    end_payload = _base_event(
        "run_end",
        label=args.label,
        argv=args.command,
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
        exit_code=return_code,
        duration_sec=duration_sec,
    )
    _append_jsonl(jsonl, end_payload)
    _append_md(
        md,
        [
            f"- Finished: `{end_payload['timestamp']}`",
            f"- Exit code: `{return_code}`",
            f"- Duration sec: `{duration_sec}`",
        ],
    )
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log experiment-agent work as JSONL + markdown.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--log", required=True, help="Path to JSONL worklog.")
    start.add_argument("--title", required=True, help="Human-readable session title.")
    start.set_defaults(func=cmd_start)

    note = subparsers.add_parser("note")
    note.add_argument("--log", required=True, help="Path to JSONL worklog.")
    note.add_argument("--message", required=True, help="Short note.")
    note.set_defaults(func=cmd_note)

    run = subparsers.add_parser("run")
    run.add_argument("--log", required=True, help="Path to JSONL worklog.")
    run.add_argument("--label", required=True, help="Short step label.")
    run.add_argument("command", nargs=argparse.REMAINDER, help="Command after '--'.")
    run.set_defaults(func=cmd_run)

    finish = subparsers.add_parser("finish")
    finish.add_argument("--log", required=True, help="Path to JSONL worklog.")
    finish.add_argument("--status", required=True, help="Final status text.")
    finish.set_defaults(func=cmd_finish)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    command = getattr(args, "command", None)
    if isinstance(command, list) and command and command[0] == "--":
        args.command = command[1:]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
