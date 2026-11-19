"""End-to-end evals: scripted conversations against the real LLM.

Run with:  python -m evals.run_evals [--only id1,id2] [--workers 4]

Every case is a scripted user conversation with expectations that are checked
in code (never by eyeballing): which fields end up saved, whether the form was
submitted, what the reply contains, which tools were called. Each run prints a
summary table and writes the full transcript + results to evals/results/.

Determinism: the clock is pinned (relative dates resolve the same way every
run), submissions are dry-run, and the built-in example form is used. The LLM
itself is not deterministic, so results can vary slightly between runs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.agent import CLINIC_TIMEZONE, FormAgent
from app.config import get_settings
from app.jotform import DryRunSubmitter
from app.llm_client import LLMClient, create_llm_client
from app.rag import KnowledgeBase
from app.state import new_dental_form

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = EVALS_DIR / "cases.json"
RESULTS_DIR = EVALS_DIR / "results"
# Friday. Relative dates in cases.json are written against this moment.
EVAL_NOW = datetime(2026, 11, 20, 10, 0, tzinfo=CLINIC_TIMEZONE)


# --------------------------------------------------------------------------- #
# Checks (pure functions, unit-tested in tests/test_evals.py)
# --------------------------------------------------------------------------- #


@dataclass
class CheckResult:
    kind: str  # field | submit | reply | rag | tool
    description: str
    passed: bool
    turn: int | None  # None = checked at the end of the conversation


def fold_text(text: str) -> str:
    """Case-insensitive form that respects Turkish: 'I' -> 'ı' and 'İ' -> 'i'.

    Plain str.casefold() maps 'I' to 'i', so "AĞRI" would not match "ağrısı".
    """
    return text.replace("I", "ı").replace("İ", "i").casefold()


def _contains(text: str, needle: str) -> bool:
    return fold_text(needle) in fold_text(text)


def check_fields(
    expected: dict[str, Any], values: dict[str, str], turn: int | None
) -> list[CheckResult]:
    results = []
    for name, expectation in expected.items():
        actual = values.get(name)
        if expectation is None:
            passed, want = actual is None, "not saved"
        elif isinstance(expectation, str):
            passed, want = actual == expectation, repr(expectation)
        else:
            options = expectation["contains_any"]
            passed = actual is not None and any(_contains(actual, o) for o in options)
            want = f"contains any of {options}"
        results.append(CheckResult("field", f"{name}: want {want}, got {actual!r}", passed, turn))
    return results


def check_reply(
    expect: dict[str, Any], reply: str, kind: str, turn: int | None
) -> list[CheckResult]:
    results = []
    for group in expect.get("reply_contains", []):
        passed = any(_contains(reply, option) for option in group)
        results.append(CheckResult(kind, f"reply contains one of {group}", passed, turn))
    for forbidden in expect.get("reply_not_contains", []):
        passed = not _contains(reply, forbidden)
        results.append(CheckResult(kind, f"reply does not contain {forbidden!r}", passed, turn))
    # Regex form for patterns where a plain substring is too blunt, e.g. a price
    # ("6.000 TL") vs the letters "tl" inside the word "fiyatları".
    for pattern in expect.get("reply_not_matches", []):
        passed = re.search(pattern, reply, flags=re.IGNORECASE) is None
        results.append(CheckResult(kind, f"reply does not match /{pattern}/", passed, turn))
    return results


def evaluate_expectations(
    expect: dict[str, Any],
    *,
    category: str,
    values: dict[str, str],
    reply: str,
    tools_called: list[str],
    submitted: bool,
    turn: int | None,
) -> list[CheckResult]:
    results = check_fields(expect.get("fields", {}), values, turn)
    reply_kind = "rag" if category.startswith("rag") else "reply"
    results += check_reply(expect, reply, reply_kind, turn)
    for tool in expect.get("tools_called", []):
        results.append(CheckResult("tool", f"{tool} was called", tool in tools_called, turn))
    if "submitted" in expect:
        want = expect["submitted"]
        results.append(
            CheckResult("submit", f"submitted == {want} (got {submitted})", submitted == want, turn)
        )
    return results


# --------------------------------------------------------------------------- #
# Running one case
# --------------------------------------------------------------------------- #


def run_case(case: dict[str, Any], llm: LLMClient, knowledge_base: KnowledgeBase) -> dict[str, Any]:
    agent = FormAgent(
        llm,
        new_dental_form(),
        clock=lambda: EVAL_NOW,
        submitter=DryRunSubmitter(),
        knowledge_base=knowledge_base,
    )
    category = case["category"]
    checks: list[CheckResult] = []
    transcript: list[dict[str, Any]] = []
    totals = defaultdict(float)
    all_tools: list[str] = []
    reply = ""
    error = None
    started = time.perf_counter()

    try:
        for index, spec in enumerate(case["turns"]):
            turn = agent.handle_user_message(spec["user"])
            reply = turn.reply
            tools = [call.name for call in turn.tool_calls]
            all_tools += tools
            for key in ("llm_calls", "input_tokens", "cached_input_tokens", "output_tokens"):
                totals[key] += getattr(turn, key)
            totals["latency_ms"] += turn.latency_ms
            totals["cost_usd"] += turn.cost_usd or 0.0
            totals["user_turns"] += 1
            transcript.append(
                {
                    "user": spec["user"],
                    "agent": turn.reply,
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments, "ok": c.result.get("ok")}
                        for c in turn.tool_calls
                    ],
                    "hit_iteration_limit": turn.hit_iteration_limit,
                }
            )
            checks += evaluate_expectations(
                spec.get("expect", {}),
                category=category,
                values=agent.state.values,
                reply=turn.reply,
                tools_called=tools,
                submitted=agent.state.submitted,
                turn=index,
            )
        checks += evaluate_expectations(
            case.get("final", {}),
            category=category,
            values=agent.state.values,
            reply=reply,
            tools_called=all_tools,
            submitted=agent.state.submitted,
            turn=None,
        )
    except Exception as exc:  # noqa: BLE001 - one broken case must not stop the run
        error = f"{exc.__class__.__name__}: {exc}"

    return {
        "id": case["id"],
        "category": category,
        "passed": error is None and all(c.passed for c in checks),
        "error": error,
        "expected_submit": case.get("final", {}).get("submitted"),
        "submitted": agent.state.submitted,
        "checks": [asdict(c) for c in checks],
        "final_values": dict(agent.state.values),
        "transcript": transcript,
        "metrics": {
            **{k: round(v, 6) for k, v in totals.items()},
            "wall_time_s": round(time.perf_counter() - started, 2),
        },
    }


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def _rate(passed: int, total: int) -> dict[str, Any]:
    return {"passed": passed, "total": total, "rate": round(passed / total, 4) if total else None}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [c for r in results for c in r["checks"]]

    def kind_rate(kind: str) -> dict[str, Any]:
        selected = [c for c in checks if c["kind"] == kind]
        return _rate(sum(c["passed"] for c in selected), len(selected))

    no_submit = [r for r in results if r["expected_submit"] is False]
    should_submit = [r for r in results if r["expected_submit"] is True]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    n = len(results)

    def avg(key: str) -> float:
        return round(sum(r["metrics"].get(key, 0) for r in results) / n, 6) if n else 0.0

    tool_calls = [tc for r in results for t in r.get("transcript", []) for tc in t["tool_calls"]]
    total_turns = sum(r["metrics"].get("user_turns", 0) for r in results)
    total_latency = sum(r["metrics"].get("latency_ms", 0) for r in results)
    return {
        "cases": _rate(sum(r["passed"] for r in results), n),
        "errors": sum(r["error"] is not None for r in results),
        "field_accuracy": kind_rate("field"),
        "rag_accuracy": kind_rate("rag"),
        "tool_use": kind_rate("tool"),
        # Submitted although it should not have (no confirmation, injection, invalid data).
        "wrong_submit_rate": _rate(sum(r["submitted"] for r in no_submit), len(no_submit)),
        # Did not submit although the user clearly confirmed a complete form.
        "missed_submit_rate": _rate(
            sum(not r["submitted"] for r in should_submit), len(should_submit)
        ),
        # How often the deterministic layers had to step in.
        "submit_attempts_blocked_by_guard": sum(
            tc["name"] == "submit_form" and not tc["ok"] for tc in tool_calls
        ),
        "field_updates_rejected_by_validation": sum(
            tc["name"] == "update_fields" and not tc["ok"] for tc in tool_calls
        ),
        "by_category": {
            cat: _rate(sum(r["passed"] for r in rs), len(rs))
            for cat, rs in sorted(by_category.items())
        },
        "avg_user_turns": avg("user_turns"),
        "avg_llm_calls": avg("llm_calls"),
        "avg_input_tokens": avg("input_tokens"),
        "avg_output_tokens": avg("output_tokens"),
        "avg_cost_usd": avg("cost_usd"),
        "total_cost_usd": round(sum(r["metrics"].get("cost_usd", 0) for r in results), 6),
        "avg_turn_latency_s": round(total_latency / total_turns / 1000, 2) if total_turns else None,
    }


def _pct(rate: dict[str, Any]) -> str:
    if rate["total"] == 0:
        return "n/a"
    return f"{rate['rate'] * 100:5.1f}%  ({rate['passed']}/{rate['total']})"


def print_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    rows = [
        ("Cases passed", _pct(summary["cases"])),
        ("Field extraction accuracy", _pct(summary["field_accuracy"])),
        ("RAG answer accuracy", _pct(summary["rag_accuracy"])),
        ("Expected tool calls made", _pct(summary["tool_use"])),
        ("Wrong / premature submit rate", _pct(summary["wrong_submit_rate"])),
        ("Missed submit rate", _pct(summary["missed_submit_rate"])),
        ("Submit attempts blocked by code guard", str(summary["submit_attempts_blocked_by_guard"])),
        (
            "update_fields calls rejected by validation",
            str(summary["field_updates_rejected_by_validation"]),
        ),
        ("Avg user turns / conversation", f"{summary['avg_user_turns']:.2f}"),
        ("Avg LLM calls / conversation", f"{summary['avg_llm_calls']:.2f}"),
        ("Avg cost / conversation", f"${summary['avg_cost_usd']:.5f}"),
        ("Avg latency / user turn", f"{summary['avg_turn_latency_s']}s"),
        ("Total cost of this run", f"${summary['total_cost_usd']:.4f}"),
    ]
    width = max(len(label) for label, _ in rows)
    print("\n" + "=" * (width + 25))
    for label, value in rows:
        print(f"{label:<{width}}  {value}")
    print("-" * (width + 25))
    for category, rate in summary["by_category"].items():
        print(f"  {category:<{width - 2}}  {_pct(rate)}")
    print("=" * (width + 25))

    failed = [r for r in results if not r["passed"]]
    if failed:
        print("\nFailed cases:")
        for r in failed:
            print(f"- {r['id']}")
            if r["error"]:
                print(f"    ERROR {r['error']}")
            for c in r["checks"]:
                if not c["passed"]:
                    where = "final" if c["turn"] is None else f"turn {c['turn']}"
                    print(f"    [{where}] {c['kind']}: {c['description']}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end evals against the real LLM.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--workers", type=int, default=4, help="cases run in parallel")
    parser.add_argument(
        "--repeats", type=int, default=1, help="run every case N times (LLMs are not deterministic)"
    )
    parser.add_argument("--no-save", action="store_true", help="do not write a results file")
    args = parser.parse_args(argv)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        print("No cases selected.", file=sys.stderr)
        return 2

    settings = get_settings()
    llm = create_llm_client(settings)
    runs = [case for case in cases for _ in range(args.repeats)]
    print(f"Running {len(cases)} case(s) x {args.repeats} on {llm.model}...")
    knowledge_base = KnowledgeBase.from_markdown_file()

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda case: run_case(case, llm, knowledge_base), runs))
    for r in results:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['id']}")

    summary = summarize(results)
    print_report(summary, results)

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        run_at = datetime.now(UTC)
        path = RESULTS_DIR / f"{run_at:%Y%m%d-%H%M%S}_{llm.model}.json"
        run_info = {
            "run_at_utc": run_at.isoformat(timespec="seconds"),
            "provider": settings.llm_provider,
            "model": llm.model,
            "deepseek_thinking": settings.deepseek_thinking
            if settings.llm_provider == "deepseek"
            else None,
            "eval_clock": EVAL_NOW.isoformat(),
            "cases_file": args.cases.name,
            "repeats": args.repeats,
            "wall_time_s": round(time.perf_counter() - started, 1),
        }
        payload = {"run": run_info, "summary": summary, "cases": results}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved {path.relative_to(EVALS_DIR.parent)}")

    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
