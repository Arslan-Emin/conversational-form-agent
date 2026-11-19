"""Tracing: structured events for every LLM call, tool call and agent turn.

This is the hook the companion eval & observability platform (project 2) plugs
into. Every event is a flat JSON-serializable dict with a `type`, an `event_id`,
a UTC `timestamp` and the ids of the conversation and turn it belongs to.

    tracer = Tracer(listeners=[JsonlFileSink(path)])
    tracer.add_listener(my_exporter)            # e.g. send events to another service

Event types:
- `llm_call`   one request to the LLM provider: model, tokens, cost, latency, tool names
- `tool_call`  one tool execution: name, ok, duration
- `agent_turn` one user message handled end to end: LLM calls, tokens, cost, latency

Privacy: message text, tool arguments and tool results can contain personal data,
so they are only included when `include_content=True` (off by default; the evals
turn it on because they only use fake data).

Run `python -m app.tracing summary logs/traces/<file>.jsonl` for cost/latency stats.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.llm_client import HistoryItem, LLMClient, LLMResponse, ToolSpec
from app.pricing import estimate_cost_usd

SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)

TraceEvent = dict[str, Any]
TraceListener = Callable[[TraceEvent], None]

# Fields (conversation_id, turn_index, ...) attached to every event emitted while
# they are set. contextvars keep them separate per thread / per Streamlit session.
_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "trace_context", default=None
)


@contextmanager
def trace_context(**fields: Any) -> Iterator[None]:
    token = _context.set({**(_context.get() or {}), **fields})
    try:
        yield
    finally:
        _context.reset(token)


class Tracer:
    def __init__(
        self, listeners: list[TraceListener] | None = None, *, include_content: bool = False
    ) -> None:
        self._listeners = list(listeners or [])
        self.include_content = include_content

    def add_listener(self, listener: TraceListener) -> None:
        self._listeners.append(listener)

    def emit(self, event_type: str, **fields: Any) -> TraceEvent:
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_id": uuid.uuid4().hex,
            "type": event_type,
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            **(_context.get() or {}),
            **fields,
        }
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - observability must never break the app
                logger.exception("Trace listener failed")
        return event


NULL_TRACER = Tracer()


class JsonlFileSink:
    """Appends one JSON object per line. Thread-safe; creates the folder if needed."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def __call__(self, event: TraceEvent) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")


class InMemorySink:
    """Collects events in a list (tests, or a UI that wants to show them)."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def __call__(self, event: TraceEvent) -> None:
        self.events.append(event)


# --------------------------------------------------------------------------- #
# The LLM wrapper: every call made by the app goes through here.
# --------------------------------------------------------------------------- #


class TracedLLMClient:
    """Wraps any `LLMClient` and emits an `llm_call` event per request.

    A decorator rather than code inside each provider class: a new provider gets
    tracing, cost and latency tracking for free.
    """

    def __init__(self, inner: LLMClient, tracer: Tracer, provider: str) -> None:
        self._inner = inner
        self._tracer = tracer
        self.provider = provider

    @property
    def model(self) -> str:
        return self._inner.model

    def user_message(self, text: str) -> HistoryItem:
        return self._inner.user_message(text)

    def assistant_message(self, text: str) -> HistoryItem:
        return self._inner.assistant_message(text)

    def tool_result(self, call_id: str, output: str) -> HistoryItem:
        return self._inner.tool_result(call_id, output)

    def chat(
        self,
        history: list[HistoryItem],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = self._inner.chat(history, system=system, tools=tools)
        except Exception as error:
            self._tracer.emit(
                "llm_call",
                provider=self.provider,
                model=self.model,
                ok=False,
                error=error.__class__.__name__,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            raise

        fields: dict[str, Any] = {
            "provider": self.provider,
            "model": response.model,
            "ok": True,
            "input_tokens": response.input_tokens,
            "cached_input_tokens": response.cached_input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": estimate_cost_usd(
                response.model,
                response.input_tokens,
                response.cached_input_tokens,
                response.output_tokens,
            ),
            "latency_ms": round(response.latency_ms, 1),
            "history_items": len(history),
            "tools_offered": [tool.name for tool in tools or []],
            "tool_calls": [call.name for call in response.tool_calls],
        }
        if self._tracer.include_content:
            fields["output_text"] = response.text
            fields["tool_call_arguments"] = [call.arguments for call in response.tool_calls]
        self._tracer.emit("llm_call", **fields)
        return response


# --------------------------------------------------------------------------- #
# Reading traces back: cost and latency summary
# --------------------------------------------------------------------------- #


def load_events(path: Path | str) -> list[TraceEvent]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; enough for latency reporting without numpy."""
    if not values:
        return None
    ordered = sorted(values)
    # ceil, not round: Python's round() is banker's rounding (round(2.5) == 2).
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize_events(events: list[TraceEvent]) -> dict[str, Any]:
    calls = [e for e in events if e["type"] == "llm_call"]
    turns = [e for e in events if e["type"] == "agent_turn"]
    tools = [e for e in events if e["type"] == "tool_call"]
    ok_calls = [e for e in calls if e.get("ok")]
    call_latency = [e["latency_ms"] for e in ok_calls]
    turn_latency = [e["latency_ms"] for e in turns]
    known_costs = [e["cost_usd"] for e in ok_calls if e.get("cost_usd") is not None]

    tool_stats: dict[str, dict[str, int]] = {}
    for event in tools:
        stats = tool_stats.setdefault(event["name"], {"calls": 0, "failed": 0})
        stats["calls"] += 1
        stats["failed"] += not event.get("ok")

    return {
        "conversations": len({e.get("conversation_id") for e in events} - {None}),
        "llm_calls": len(calls),
        "llm_errors": len(calls) - len(ok_calls),
        "models": sorted({e["model"] for e in calls}),
        "input_tokens": sum(e.get("input_tokens", 0) for e in ok_calls),
        "cached_input_tokens": sum(e.get("cached_input_tokens", 0) for e in ok_calls),
        "output_tokens": sum(e.get("output_tokens", 0) for e in ok_calls),
        "cost_usd": round(sum(known_costs), 6),
        "calls_without_price": len(ok_calls) - len(known_costs),
        "llm_latency_ms": {
            "p50": percentile(call_latency, 50),
            "p95": percentile(call_latency, 95),
            "max": max(call_latency, default=None),
        },
        "turns": len(turns),
        "turn_latency_ms": {
            "p50": percentile(turn_latency, 50),
            "p95": percentile(turn_latency, 95),
        },
        "tools": tool_stats,
    }


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2 or args[0] != "summary":
        print("Usage: python -m app.tracing summary <trace.jsonl>", file=sys.stderr)
        return 2
    print(json.dumps(summarize_events(load_events(args[1])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
