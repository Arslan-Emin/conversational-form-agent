"""The agent loop.

One user message can need several LLM calls: the model may call a tool, read
its result, call another tool, and only then answer. The loop is:

    send history + tools to the LLM
    if the reply has tool calls:
        run each tool, append the results to history, and loop again
    else:
        the reply text is the final answer for this turn

`max_iterations` caps the number of LLM calls per user message, so a confused
model can never loop forever (or burn money forever).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.jotform import DryRunSubmitter, FormSubmitter
from app.llm_client import HistoryItem, LLMClient
from app.pricing import estimate_cost_usd
from app.prompts import build_system_prompt
from app.rag import KnowledgeBase
from app.state import FormState, new_dental_form
from app.tools import ToolExecutor, to_tool_output
from app.tracing import NULL_TRACER, Tracer, trace_context

CLINIC_TIMEZONE = ZoneInfo("Europe/Istanbul")
DEFAULT_MAX_ITERATIONS = 5
ITERATION_LIMIT_REPLY = (
    "Sorry, I couldn't complete that step. Could you rephrase? / "
    "Üzgünüm, bu adımı tamamlayamadım. Tekrar ifade edebilir misiniz?"
)


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any] | None
    result: dict[str, Any]


@dataclass
class AgentTurn:
    """Everything that happened while answering one user message (used by UI and evals)."""

    reply: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    llm_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    # None if the model's price is unknown (see app/pricing.py).
    cost_usd: float | None = 0.0
    latency_ms: float = 0.0
    hit_iteration_limit: bool = False


class FormAgent:
    def __init__(
        self,
        llm: LLMClient,
        state: FormState | None = None,
        *,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        clock: Callable[[], datetime] | None = None,
        submitter: FormSubmitter | None = None,
        knowledge_base: KnowledgeBase | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.llm = llm
        self.tracer = tracer or NULL_TRACER
        # Groups all trace events of this conversation.
        self.conversation_id = uuid.uuid4().hex
        self._turn_index = 0
        self.state = state or new_dental_form()
        # Injectable clock: evals pin "today" so relative dates resolve the same way every run.
        self._clock = clock or (lambda: datetime.now(CLINIC_TIMEZONE))
        # Default is dry-run: nothing is sent anywhere unless a real submitter is passed in.
        self.tools = ToolExecutor(
            self.state, self._clock, submitter or DryRunSubmitter(), knowledge_base
        )
        self.history: list[HistoryItem] = []
        self.max_iterations = max_iterations

    def handle_user_message(self, text: str) -> AgentTurn:
        """Answer one user message. On an LLM/API error, history is rolled back and the
        error re-raised, so the caller can simply let the user retry."""
        history_checkpoint = len(self.history)
        started = time.perf_counter()
        with trace_context(conversation_id=self.conversation_id, turn_index=self._turn_index):
            self.tools.observe_user_message(text)
            try:
                turn = self._run_turn(text)
            except Exception as error:
                del self.history[history_checkpoint:]
                self.tracer.emit(
                    "agent_turn",
                    ok=False,
                    error=error.__class__.__name__,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                )
                raise
            # The reply the user is reading now is what they can confirm next turn. The
            # prompt makes the model show a summary once the form is complete.
            self.tools.confirmable_version = (
                self.state.version if self.state.is_complete() else None
            )
            self._emit_turn(turn, text, started)
        self._turn_index += 1
        return turn

    def _emit_turn(self, turn: AgentTurn, user_text: str, started: float) -> None:
        fields: dict[str, Any] = {
            "ok": True,
            "llm_calls": turn.llm_calls,
            "input_tokens": turn.input_tokens,
            "cached_input_tokens": turn.cached_input_tokens,
            "output_tokens": turn.output_tokens,
            "cost_usd": turn.cost_usd,
            "latency_ms": round(turn.latency_ms, 1),  # time spent waiting for the LLM
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),  # wall clock
            "tool_calls": [call.name for call in turn.tool_calls],
            "hit_iteration_limit": turn.hit_iteration_limit,
            "fields_filled": len(self.state.values),
            "fields_total": len(self.state.fields),
            "submitted": self.state.submitted,
        }
        if self.tracer.include_content:
            fields["user_text"] = user_text
            fields["reply"] = turn.reply
        self.tracer.emit("agent_turn", **fields)

    def _run_turn(self, text: str) -> AgentTurn:
        self.history.append(self.llm.user_message(text))
        system = build_system_prompt(self.state, self._clock())
        turn = AgentTurn(reply="")

        for _ in range(self.max_iterations):
            response = self.llm.chat(self.history, system=system, tools=self.tools.specs)
            turn.llm_calls += 1
            turn.input_tokens += response.input_tokens
            turn.cached_input_tokens += response.cached_input_tokens
            turn.output_tokens += response.output_tokens
            call_cost = estimate_cost_usd(
                response.model,
                response.input_tokens,
                response.cached_input_tokens,
                response.output_tokens,
            )
            turn.cost_usd = (
                None if call_cost is None or turn.cost_usd is None else turn.cost_usd + call_cost
            )
            turn.latency_ms += response.latency_ms
            self.history.extend(response.history_items)

            if not response.tool_calls:
                turn.reply = response.text
                return turn

            for call in response.tool_calls:
                tool_started = time.perf_counter()
                result = self.tools.execute(call.name, call.arguments)
                self._emit_tool_call(call.name, call.arguments, result, tool_started)
                turn.tool_calls.append(ToolCallRecord(call.name, call.arguments, result))
                self.history.append(self.llm.tool_result(call.id, to_tool_output(result)))

        # Limit reached while the model still wanted tools. Close the turn with an
        # assistant message so the history stays well-formed for the next request.
        turn.hit_iteration_limit = True
        turn.reply = ITERATION_LIMIT_REPLY
        self.history.append(self.llm.assistant_message(ITERATION_LIMIT_REPLY))
        return turn

    def _emit_tool_call(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        result: dict[str, Any],
        started: float,
    ) -> None:
        fields: dict[str, Any] = {
            "name": name,
            "ok": bool(result.get("ok")),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if not result.get("ok"):
            # Validation errors quote the rejected value (user data), so only the
            # field names are logged unless content logging is on.
            if result.get("errors"):
                fields["invalid_fields"] = sorted(result["errors"])
            if result.get("error"):
                fields["error"] = result["error"]
        if self.tracer.include_content:
            fields["arguments"] = arguments
            fields["result"] = result
        self.tracer.emit("tool_call", **fields)
