#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _tail_lines(path: Path, n: int = 20) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    return lines[-n:]


def _extract_latest_progress(log_lines: list[str]) -> tuple[int, int, str]:
    latest_done = 0
    latest_total = 0
    latest_line = ""
    pattern = re.compile(r"^\[(\d+)/(\d+)\]\s+(\S+)\s+(ok|error)\s*(.*)$")
    for line in log_lines:
        m = pattern.match(line.strip())
        if not m:
            continue
        latest_done = int(m.group(1))
        latest_total = int(m.group(2))
        latest_line = line.strip()
    return latest_done, latest_total, latest_line


def _load_report_stats(path: Path) -> dict:
    if not path.exists():
        return {
            "exists": False,
            "count": 0,
            "ok": 0,
            "error": 0,
            "last_case": "",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "exists": True,
            "count": 0,
            "ok": 0,
            "error": 0,
            "last_case": "",
        }
    count = len(data) if isinstance(data, list) else 0
    ok = sum(1 for item in data if isinstance(item, dict) and item.get("status") == "ok")
    error = sum(1 for item in data if isinstance(item, dict) and item.get("status") == "error")
    last_case = ""
    if isinstance(data, list) and data:
        last = data[-1]
        if isinstance(last, dict):
            last_case = str(last.get("case_id", ""))
    return {
        "exists": True,
        "count": count,
        "ok": ok,
        "error": error,
        "last_case": last_case,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Show progress for rebuild_case_skills_v3 runs.")
    parser.add_argument("--log", required=True, help="Path to the rebuild log file")
    parser.add_argument("--report", required=True, help="Path to the rebuild report json")
    parser.add_argument("--tail", type=int, default=12, help="How many log lines to show")
    args = parser.parse_args()

    log_path = Path(args.log)
    report_path = Path(args.report)

    log_lines = _tail_lines(log_path, n=max(1, args.tail))
    done, total, latest_line = _extract_latest_progress(log_lines)
    stats = _load_report_stats(report_path)

    print(f"log: {log_path}")
    print(f"report: {report_path}")
    print(f"log_exists: {log_path.exists()}")
    print(f"report_exists: {stats['exists']}")
    print(f"latest_progress: {done}/{total}" if total else "latest_progress: unknown")
    print(f"report_count: {stats['count']}")
    print(f"report_ok: {stats['ok']}")
    print(f"report_error: {stats['error']}")
    if stats["last_case"]:
        print(f"last_case: {stats['last_case']}")
    if latest_line:
        print(f"latest_log_line: {latest_line}")

    print("\nlast_log_lines:")
    for line in log_lines:
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
