#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "reports" / "variant_matrix"
DEFAULT_PROBLEM_CASE_FILE = PROJECT_ROOT / "reports" / "prompt_variants" / "problem_cases_20260328.json"


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _check_ready(url: str) -> bool:
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(url, timeout=5) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except URLError:
        return False


def _load_case_ids(path: Path, source: str) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return sorted({str(item).strip() for item in payload if str(item).strip()})

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported case file format: {path}")

    collected = []
    if source != "all":
        buckets = {source: payload.get(source, [])}
    else:
        buckets = payload

    for items in buckets.values():
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    case_id = str(item.get("case_id", "")).strip()
                else:
                    case_id = str(item).strip()
                if case_id:
                    collected.append(case_id)
    return sorted(set(collected))


def _parse_overall_metric(text: str, key: str) -> str:
    match = re.search(rf"\*\*{re.escape(key)}\*\*: ([^\n]+)", text)
    return match.group(1).strip() if match else "NA"


def _parse_first_metric(text: str, keys: List[str]) -> str:
    for key in keys:
        value = _parse_overall_metric(text, key)
        if value != "NA":
            return value
    return "NA"


def _parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_report(report_path: Path) -> Dict:
    text = report_path.read_text(encoding="utf-8", errors="replace")
    case_scores = {}
    pattern = re.compile(
        r"### .*?Case ID: ([^\n]+)\n(.*?)(?=\n---\n### |\Z)",
        re.DOTALL,
    )
    for case_id, section in pattern.findall(text):
        f1_match = re.search(r"\*\*F1 Stats\*\*: Max=([0-9.]+)", section)
        if f1_match:
            case_scores[case_id.strip()] = float(f1_match.group(1))

    return {
        "avg_best_f1": _parse_float(_parse_overall_metric(text, "Avg Best F1")),
        "hit_at_1": _parse_float(
            _parse_first_metric(text, ["Hit@1 (first pred in GT)", "Hit@1 (Best F1>=0.5)"])
        ),
        "total_turns": _parse_int(_parse_overall_metric(text, "Total Turns")),
        "frontend_validation_errors": _parse_int(
            _parse_overall_metric(text, "Frontend Validation Errors")
        ),
        "total_cases": _parse_int(_parse_overall_metric(text, "Total Cases")),
        "case_scores": case_scores,
    }


def _mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _stdev(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())


async def _run_job(job: Dict, semaphore: asyncio.Semaphore, delay_sec: float, resume: bool) -> Dict:
    async with semaphore:
        report_path = Path(job["report_path"])
        if resume and report_path.exists():
            result = dict(job)
            result["return_code"] = 0
            result["runtime_sec"] = 0.0
            result["started_at"] = ""
            result["finished_at"] = ""
            result["ok"] = True
            result["resumed"] = True
            result["metrics"] = _parse_report(report_path)
            return result

        if delay_sec > 0:
            await asyncio.sleep(delay_sec)

        env = os.environ.copy()
        env.update(job["env"])
        stdout_path = Path(job["stdout_path"])
        stderr_path = Path(job["stderr_path"])

        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            started_at = datetime.utcnow()
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(PROJECT_ROOT / "test_pipe6.py"),
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
            )
            return_code = await process.wait()
            finished_at = datetime.utcnow()

        result = dict(job)
        result["return_code"] = return_code
        result["runtime_sec"] = (finished_at - started_at).total_seconds()
        result["started_at"] = started_at.isoformat() + "Z"
        result["finished_at"] = finished_at.isoformat() + "Z"
        result["resumed"] = False
        result["ok"] = return_code == 0 and Path(job["report_path"]).exists()
        if result["ok"]:
            result["metrics"] = _parse_report(Path(job["report_path"]))
        else:
            result["metrics"] = {}
        return result


def _build_summary(results: List[Dict], label: str, result_dir: Path, config: Dict) -> Tuple[Dict, str]:
    grouped = defaultdict(list)
    for result in results:
        grouped[result["variant"]].append(result)

    aggregate = {
        "label": label,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": config,
        "variants": {},
    }

    lines = [
        f"# Prompt Variant Matrix Summary: {label}",
        "",
        "## Config",
        f"- Variants: {', '.join(config['variants'])}",
        f"- Repeats: {config['repeats']}",
        f"- Limit Cases: {config['limit_cases'] if config['limit_cases'] is not None else 'all'}",
        f"- Target Case Count: {config['target_case_count']}",
        f"- Max Parallel Cases / job: {config['max_parallel_cases']}",
        f"- Max Concurrent Jobs: {config['max_concurrent_jobs']}",
        f"- LLM Max Tokens: {config['llm_max_tokens']}",
        f"- Sampling: temp={config['temperature']}, top_p={config['top_p']}, top_k={config['top_k']}, "
        f"presence_penalty={config['presence_penalty']}, repetition_penalty={config['repetition_penalty']}",
        "",
        "## Aggregate",
        "",
        "| Variant | Runs | Mean F1 | Std F1 | Mean Hit@1 | Mean Turns | Mean Frontend Errors | Mean Runtime (s) | Best Run | Reports |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    ranking = []
    for variant, items in sorted(grouped.items()):
        ok_items = [item for item in items if item.get("ok")]
        metrics = [item["metrics"] for item in ok_items]
        mean_f1 = _mean([m["avg_best_f1"] for m in metrics])
        std_f1 = _stdev([m["avg_best_f1"] for m in metrics])
        mean_hit = _mean([m["hit_at_1"] for m in metrics])
        mean_turns = _mean([m["total_turns"] for m in metrics])
        mean_frontend = _mean([m["frontend_validation_errors"] for m in metrics])
        mean_runtime = _mean([item["runtime_sec"] for item in ok_items])
        best_run = max(ok_items, key=lambda item: item["metrics"]["avg_best_f1"], default=None)

        per_case = defaultdict(list)
        for item in ok_items:
            for case_id, score in item["metrics"]["case_scores"].items():
                per_case[case_id].append(score)

        aggregate["variants"][variant] = {
            "runs": [
                {
                    "repeat_index": item["repeat_index"],
                    "report_path": item["report_path"],
                    "stdout_path": item["stdout_path"],
                    "stderr_path": item["stderr_path"],
                    "return_code": item["return_code"],
                    "runtime_sec": item["runtime_sec"],
                    "started_at": item["started_at"],
                    "finished_at": item["finished_at"],
                    "resumed": item.get("resumed", False),
                    "metrics": item.get("metrics", {}),
                }
                for item in items
            ],
            "mean_avg_best_f1": mean_f1,
            "std_avg_best_f1": std_f1,
            "mean_hit_at_1": mean_hit,
            "mean_total_turns": mean_turns,
            "mean_frontend_validation_errors": mean_frontend,
            "mean_runtime_sec": mean_runtime,
            "per_case_mean_f1": {
                case_id: _mean(scores) for case_id, scores in sorted(per_case.items())
            },
            "best_run_report": best_run["report_path"] if best_run else "",
        }

        report_links = ", ".join(
            f"[r{item['repeat_index']}]({item['report_path']})" for item in ok_items
        ) or "NA"
        best_run_label = (
            f"r{best_run['repeat_index']}={best_run['metrics']['avg_best_f1']:.4f}" if best_run else "NA"
        )
        lines.append(
            f"| {variant} | {len(ok_items)}/{len(items)} | {mean_f1:.4f} | {std_f1:.4f} | "
            f"{mean_hit:.4f} | {mean_turns:.1f} | {mean_frontend:.1f} | {mean_runtime:.1f} | "
            f"{best_run_label} | {report_links} |"
        )
        ranking.append((mean_f1, mean_hit, -mean_frontend, variant))

    ranking.sort(reverse=True)
    if ranking:
        best_variant = ranking[0][3]
        aggregate["best_variant"] = best_variant
        lines.extend(
            [
                "",
                "## Winner",
                f"- Best variant by mean F1: `{best_variant}`",
                f"- Summary JSON: [{result_dir / 'summary.json'}]({result_dir / 'summary.json'})",
            ]
        )

    return aggregate, "\n".join(lines) + "\n"


def _build_jobs(args, result_dir: Path, target_case_ids: List[str]) -> List[Dict]:
    jobs = []
    common_env = {
        "KGQA_LLM_API_URL": args.llm_api_url,
        "KGQA_LLM_API_KEY": args.llm_api_key,
        "KGQA_MODEL_NAME": args.model_name,
        "KGQA_KG_API_URL": args.kg_api_url,
        "KGQA_LLM_MAX_TOKENS": str(args.llm_max_tokens),
        "KGQA_LLM_TEMPERATURE": str(args.temperature),
        "KGQA_LLM_TOP_P": str(args.top_p),
        "KGQA_LLM_TOP_K": str(args.top_k),
        "KGQA_LLM_MIN_P": str(args.min_p),
        "KGQA_LLM_PRESENCE_PENALTY": str(args.presence_penalty),
        "KGQA_LLM_REPETITION_PENALTY": str(args.repetition_penalty),
        "KGQA_ENABLE_THINKING": "1" if args.enable_thinking else "0",
        "KGQA_STRIP_HISTORY_REASONING": "1" if args.strip_history_reasoning else "0",
        "KGQA_MAX_PARALLEL_CASES": str(args.max_parallel_cases),
        "KGQA_BEST_OF_N": str(args.best_of_n),
        "KGQA_MAX_TURNS": str(args.max_turns),
        "KGQA_LLM_TIMEOUT_SEC": str(args.llm_timeout_sec),
        "KGQA_TEST_DATA_PATH": args.test_data_path,
    }

    if target_case_ids:
        common_env["KGQA_TARGET_CASE_IDS"] = ",".join(target_case_ids)
        common_env["KGQA_LIMIT_CASES"] = ""
    elif args.limit_cases is not None:
        common_env["KGQA_LIMIT_CASES"] = str(args.limit_cases)

    for variant in args.variants:
        for repeat_index in range(1, args.repeats + 1):
            report_path = result_dir / f"test_{_safe_name(variant)}_r{repeat_index:02d}.md"
            stdout_path = result_dir / "logs" / f"{_safe_name(variant)}_r{repeat_index:02d}.stdout.log"
            stderr_path = result_dir / "logs" / f"{_safe_name(variant)}_r{repeat_index:02d}.stderr.log"
            env = dict(common_env)
            env["KGQA_PROMPT_VARIANT"] = variant
            env["KGQA_REPORT_FILE"] = str(report_path)
            jobs.append(
                {
                    "variant": variant,
                    "repeat_index": repeat_index,
                    "report_path": str(report_path),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "env": env,
                }
            )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt variants in a concurrent matrix.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=_split_csv(
            os.getenv(
                "KGQA_MATRIX_VARIANTS",
                "original,short_reasoning,repair_friendly,short_reasoning_repair",
            )
        ),
        help="Prompt variants to evaluate.",
    )
    parser.add_argument("--repeats", type=int, default=int(os.getenv("KGQA_MATRIX_REPEATS", "3")))
    parser.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=int(os.getenv("KGQA_MATRIX_MAX_CONCURRENT_JOBS", "3")),
        help="How many evaluation processes to run at once.",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=int(os.getenv("KGQA_LIMIT_CASES", "50")),
        help="Case limit when target case IDs are not provided.",
    )
    parser.add_argument(
        "--target-case-ids",
        default=os.getenv("KGQA_TARGET_CASE_IDS", ""),
        help="Comma-separated case IDs to run.",
    )
    parser.add_argument(
        "--target-case-file",
        default=os.getenv("KGQA_MATRIX_TARGET_CASE_FILE", ""),
        help="Optional JSON file containing case IDs or the existing problem-case bucket JSON.",
    )
    parser.add_argument(
        "--target-case-source",
        default=os.getenv("KGQA_MATRIX_TARGET_CASE_SOURCE", "all"),
        help="If target-case-file is a dict, use this bucket key or 'all' for union.",
    )
    parser.add_argument(
        "--use-problem-cases",
        action="store_true",
        default=_env_bool("KGQA_MATRIX_USE_PROBLEM_CASES", False),
        help="Use reports/prompt_variants/problem_cases_20260328.json as the target case source.",
    )
    parser.add_argument("--label", default=os.getenv("KGQA_MATRIX_LABEL", datetime.utcnow().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--llm-api-url", default=os.getenv("KGQA_LLM_API_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--llm-api-key", default=os.getenv("KGQA_LLM_API_KEY", "EMPTY"))
    parser.add_argument("--model-name", default=os.getenv("KGQA_MODEL_NAME", "qwen35-9b-local"))
    parser.add_argument("--kg-api-url", default=os.getenv("KGQA_KG_API_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--llm-max-tokens", type=int, default=int(os.getenv("KGQA_LLM_MAX_TOKENS", "2048")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("KGQA_LLM_TEMPERATURE", "0.7")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("KGQA_LLM_TOP_P", "0.8")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("KGQA_LLM_TOP_K", "20")))
    parser.add_argument("--min-p", type=float, default=float(os.getenv("KGQA_LLM_MIN_P", "0.0")))
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=float(os.getenv("KGQA_LLM_PRESENCE_PENALTY", "1.5")),
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=float(os.getenv("KGQA_LLM_REPETITION_PENALTY", "1.0")),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=_env_bool("KGQA_ENABLE_THINKING", False),
    )
    parser.add_argument(
        "--strip-history-reasoning",
        action="store_true",
        default=_env_bool("KGQA_STRIP_HISTORY_REASONING", True),
    )
    parser.add_argument("--max-parallel-cases", type=int, default=int(os.getenv("KGQA_MAX_PARALLEL_CASES", "32")))
    parser.add_argument("--best-of-n", type=int, default=int(os.getenv("KGQA_BEST_OF_N", "1")))
    parser.add_argument("--max-turns", type=int, default=int(os.getenv("KGQA_MAX_TURNS", "6")))
    parser.add_argument("--llm-timeout-sec", type=float, default=float(os.getenv("KGQA_LLM_TIMEOUT_SEC", "180")))
    parser.add_argument(
        "--test-data-path",
        default=os.getenv(
            "KGQA_TEST_DATA_PATH",
            str(PROJECT_ROOT / "data" / "webqsp" / "webqsp_train.jsonl"),
        ),
    )
    parser.add_argument(
        "--result-root",
        default=os.getenv("KGQA_MATRIX_RESULT_ROOT", str(DEFAULT_RESULT_ROOT)),
    )
    parser.add_argument(
        "--start-stagger-sec",
        type=float,
        default=float(os.getenv("KGQA_MATRIX_START_STAGGER_SEC", "0.5")),
        help="Small stagger between jobs to avoid synchronized startup spikes.",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=float(os.getenv("KGQA_MATRIX_HEARTBEAT_SEC", "30")),
        help="Progress heartbeat interval in seconds.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode and force rerunning jobs even if report files already exist.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    result_dir = Path(args.result_root) / _safe_name(args.label)
    result_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    target_case_ids = _split_csv(args.target_case_ids)
    if args.use_problem_cases:
        target_case_ids = _load_case_ids(DEFAULT_PROBLEM_CASE_FILE, args.target_case_source)
    elif args.target_case_file:
        target_case_ids = _load_case_ids(Path(args.target_case_file), args.target_case_source)

    if not _check_ready(f"{args.llm_api_url}/models"):
        print(f"LLM server is not ready at {args.llm_api_url}", file=sys.stderr)
        return 1
    if not _check_ready(f"{args.kg_api_url}/health"):
        print(f"Graph server is not ready at {args.kg_api_url}", file=sys.stderr)
        return 1

    jobs = _build_jobs(args, result_dir, target_case_ids)
    print(f"Matrix result dir: {result_dir}", flush=True)
    print(
        f"Launching {len(jobs)} jobs: variants={','.join(args.variants)} repeats={args.repeats} "
        f"limit_cases={'all' if target_case_ids else args.limit_cases} target_case_count={len(target_case_ids)}"
    , flush=True)
    print(
        f"Runtime settings: max_parallel_cases/job={args.max_parallel_cases}, "
        f"max_concurrent_jobs={args.max_concurrent_jobs}, heartbeat={args.heartbeat_sec}s, resume={resume}"
    , flush=True)
    effective_case_concurrency = args.max_parallel_cases * args.max_concurrent_jobs
    if effective_case_concurrency >= 64:
        print(
            f"Warning: effective case concurrency is {effective_case_concurrency}. "
            "This may cause long stalls or LLM timeouts on the 9B server."
        , flush=True)

    semaphore = asyncio.Semaphore(args.max_concurrent_jobs)
    tasks = []
    for index, job in enumerate(jobs):
        print(
            f"Queued {job['variant']} r{job['repeat_index']} -> {job['report_path']}"
        , flush=True)
        tasks.append(
            asyncio.create_task(
                _run_job(
                    job,
                    semaphore,
                    delay_sec=index * args.start_stagger_sec,
                    resume=resume,
                )
            )
        )

    pending = set(tasks)
    results = []
    completed = 0
    total = len(tasks)
    while pending:
        done, pending = await asyncio.wait(
            pending,
            timeout=args.heartbeat_sec,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            print(f"[heartbeat] completed {completed}/{total}; still running {len(pending)} jobs", flush=True)
            continue
        for task in done:
            result = task.result()
            results.append(result)
            completed += 1
            status = "resumed" if result.get("resumed") else ("ok" if result.get("ok") else "failed")
            f1 = result.get("metrics", {}).get("avg_best_f1")
            f1_text = f"{f1:.4f}" if isinstance(f1, float) else "NA"
            print(
                f"[done {completed}/{total}] {result['variant']} r{result['repeat_index']} "
                f"status={status} f1={f1_text} runtime={result.get('runtime_sec', 0.0):.1f}s"
            , flush=True)

    config = {
        "variants": args.variants,
        "repeats": args.repeats,
        "max_concurrent_jobs": args.max_concurrent_jobs,
        "limit_cases": None if target_case_ids else args.limit_cases,
        "target_case_count": len(target_case_ids),
        "target_case_ids": target_case_ids,
        "llm_api_url": args.llm_api_url,
        "kg_api_url": args.kg_api_url,
        "model_name": args.model_name,
        "llm_max_tokens": args.llm_max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "enable_thinking": args.enable_thinking,
        "strip_history_reasoning": args.strip_history_reasoning,
        "max_parallel_cases": args.max_parallel_cases,
        "best_of_n": args.best_of_n,
        "max_turns": args.max_turns,
        "llm_timeout_sec": args.llm_timeout_sec,
        "test_data_path": args.test_data_path,
    }

    summary_json, summary_md = _build_summary(results, args.label, result_dir, config)
    (result_dir / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (result_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    failed_jobs = [result for result in results if not result.get("ok")]
    print(f"Saved matrix summary to {result_dir / 'summary.md'}", flush=True)
    if failed_jobs:
        print(f"{len(failed_jobs)} job(s) failed. See per-run stderr logs under {result_dir / 'logs'}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
