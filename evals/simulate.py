"""Simulated-user evals: an LLM plays a patient and talks to the agent.

Run with:  python -m evals.simulate [--repeats 3] [--only id1,id2]

The scripted evals (run_evals.py) use fixed user messages, so the agent only ever
sees conversations someone wrote down. Here a second LLM plays the user, with a
persona (hurried, scattered, careless with numbers, tries an injection, ...) and
a hidden goal (name, phone, date, complaint). The conversation goes wherever the
two models take it, which finds behaviour no fixed script triggers.

Outcomes are still checked in code: the values saved at the end must equal the
persona's goal (dates are pinned by the eval clock), and the form must be
submitted exactly when the persona actually confirms. The simulator's own LLM
calls are not traced, so the trace contains only the agent's work. Results use
the same file format as run_evals.py, so the eval platform can compare, judge and
gate them like any other run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.agent import FormAgent
from app.config import get_settings
from app.factory import create_knowledge_base, create_traced_llm, create_tracer
from app.jotform import DryRunSubmitter
from app.llm_client import HistoryItem, LLMClient, create_llm_client
from app.pricing import estimate_cost_usd
from app.rag import KnowledgeBase
from app.state import new_dental_form
from app.tracing import Tracer
from evals.run_evals import EVAL_NOW, EVALS_DIR, RESULTS_DIR, CheckResult, check_fields, summarize

DEFAULT_PERSONAS_PATH = EVALS_DIR / "personas.json"
END_MARKER = "[END]"
MAX_USER_TURNS = 10
START = "(The chat window just opened. Write your first message to the clinic's assistant.)"


def simulator_prompt(persona: dict[str, Any]) -> str:
    goal = "\n".join(f"- {key}: {value}" for key, value in persona["goal"].items())
    return f"""You are role-playing a patient who is chatting with a dental clinic's booking
assistant. Stay in character and write only the patient's next message.

Persona: {persona["persona"]}
Write in {persona["language"]}. Keep messages short, like real chat messages.

Your goal (the assistant does not know it; reveal it as your persona would):
{goal}

Rules:
- Never invent details outside your goal. If asked for something you do not have,
  say you do not know.
- When the assistant shows a summary, compare it with your goal. Correct anything
  wrong. Confirm only if everything matches and your persona would confirm.
- When your booking is confirmed as done, or your persona has decided to leave,
  end your message with {END_MARKER}. You may write {END_MARKER} alone.
- Do not mention that you are an AI, a simulation or a test."""


class SimulatedUser:
    """The patient. Sees the assistant's replies as 'user' turns and answers them."""

    def __init__(self, llm: LLMClient, persona: dict[str, Any]) -> None:
        self.llm = llm
        self.system = simulator_prompt(persona)
        self.history: list[HistoryItem] = []
        self.cost_usd = 0.0
        self.llm_calls = 0

    def reply(self, assistant_text: str | None) -> tuple[str, bool]:
        """Next patient message and whether the patient ended the conversation."""
        self.history.append(self.llm.user_message(assistant_text or START))
        response = self.llm.chat(self.history, system=self.system)
        self.llm_calls += 1
        cost = estimate_cost_usd(
            response.model,
            response.input_tokens,
            response.cached_input_tokens,
            response.output_tokens,
        )
        self.cost_usd += cost or 0.0
        text = response.text.strip()
        self.history.append(self.llm.assistant_message(text))
        ended = END_MARKER in text
        return text.replace(END_MARKER, "").strip(), ended


def run_persona(
    persona: dict[str, Any],
    agent_llm: LLMClient,
    simulator_llm: LLMClient,
    knowledge_base: KnowledgeBase | None,
    tracer: Tracer,
    *,
    max_user_turns: int = MAX_USER_TURNS,
) -> dict[str, Any]:
    agent = FormAgent(
        agent_llm,
        new_dental_form(),
        clock=lambda: EVAL_NOW,
        submitter=DryRunSubmitter(),
        knowledge_base=knowledge_base,
        tracer=tracer,
    )
    user = SimulatedUser(simulator_llm, persona)
    transcript: list[dict[str, Any]] = []
    totals: dict[str, float] = defaultdict(float)
    error = None
    ended = False
    started = time.perf_counter()
    reply: str | None = None
    try:
        while len(transcript) < max_user_turns:
            message, ended = user.reply(reply)
            if not message:
                break
            turn = agent.handle_user_message(message)
            reply = turn.reply
            for key in ("llm_calls", "input_tokens", "cached_input_tokens", "output_tokens"):
                totals[key] += getattr(turn, key)
            totals["latency_ms"] += turn.latency_ms
            totals["cost_usd"] += turn.cost_usd or 0.0
            totals["user_turns"] += 1
            transcript.append(
                {
                    "user": message,
                    "agent": turn.reply,
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments, "ok": c.result.get("ok")}
                        for c in turn.tool_calls
                    ],
                    "hit_iteration_limit": turn.hit_iteration_limit,
                }
            )
            if ended:
                break
    except Exception as exc:  # noqa: BLE001 - one broken conversation must not stop the run
        error = f"{exc.__class__.__name__}: {exc}"

    checks = check_fields(persona["expected"], agent.state.values, None)
    want = persona["expect_submit"]
    checks.append(
        CheckResult(
            "submit",
            f"submitted == {want} (got {agent.state.submitted})",
            agent.state.submitted == want,
            None,
        )
    )
    checks.append(
        CheckResult(
            "flow",
            f"conversation ended within {max_user_turns} user turns",
            ended,
            None,
        )
    )
    return {
        "id": persona["id"],
        "category": "simulated",
        "conversation_id": agent.conversation_id,
        "passed": error is None and all(c.passed for c in checks),
        "error": error,
        "expected_submit": want,
        "submitted": agent.state.submitted,
        "checks": [asdict(c) for c in checks],
        "final_values": dict(agent.state.values),
        "transcript": transcript,
        "metrics": {
            **{k: round(v, 6) for k, v in totals.items()},
            "wall_time_s": round(time.perf_counter() - started, 2),
            "simulator_llm_calls": user.llm_calls,
            "simulator_cost_usd": round(user.cost_usd, 6),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulated-user evals against the real LLM.")
    parser.add_argument("--personas", type=Path, default=DEFAULT_PERSONAS_PATH)
    parser.add_argument("--only", help="comma-separated persona ids")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--simulator-model", default="deepseek-flash")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    personas = json.loads(args.personas.read_text(encoding="utf-8"))["personas"]
    if args.only:
        wanted = set(args.only.split(","))
        personas = [p for p in personas if p["id"] in wanted]
    if not personas:
        print("No personas selected.", file=sys.stderr)
        return 2

    settings = get_settings()
    run_at = datetime.now(UTC)
    trace_file = f"eval_{run_at:%Y%m%d-%H%M%S}.jsonl"
    tracer = create_tracer(settings, file_name=trace_file, include_content=True)
    agent_llm = create_traced_llm(settings, tracer)
    # The patient is a separate, untraced client: the trace holds only the agent.
    simulator_llm = create_llm_client(
        replace(settings, deepseek_model=args.simulator_model, deepseek_thinking="disabled")
    )
    knowledge_base = create_knowledge_base(settings)
    runs = [p for p in personas for _ in range(args.repeats)]
    print(f"Simulating {len(personas)} persona(s) x {args.repeats} with {agent_llm.model}...")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(
            pool.map(
                lambda p: run_persona(p, agent_llm, simulator_llm, knowledge_base, tracer), runs
            )
        )
    for r in results:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['id']}  ({len(r['transcript'])} turns)")
        for c in r["checks"]:
            if not c["passed"]:
                print(f"        {c['kind']}: {c['description']}")
        if r["error"]:
            print(f"        ERROR {r['error']}")

    summary = summarize(results)
    simulator_cost = sum(r["metrics"]["simulator_cost_usd"] for r in results)
    print(
        f"\nPassed {summary['cases']['passed']}/{summary['cases']['total']}. "
        f"Agent cost ${summary['total_cost_usd']:.4f}, simulator cost ${simulator_cost:.4f}."
    )
    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"{run_at:%Y%m%d-%H%M%S}_{agent_llm.model}_simulated.json"
        run_info = {
            "run_at_utc": run_at.isoformat(timespec="seconds"),
            "provider": settings.llm_provider,
            "model": agent_llm.model,
            "deepseek_thinking": settings.deepseek_thinking
            if settings.llm_provider == "deepseek"
            else None,
            "simulator_model": args.simulator_model,
            "eval_clock": EVAL_NOW.isoformat(),
            "cases_file": args.personas.name,
            "repeats": args.repeats,
            "trace_file": f"{settings.trace_dir}/{trace_file}" if settings.trace_enabled else None,
            "wall_time_s": round(time.perf_counter() - started, 1),
            "simulator_cost_usd": round(simulator_cost, 6),
        }
        payload = {"run": run_info, "summary": summary, "cases": results}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {path.relative_to(EVALS_DIR.parent)}")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
