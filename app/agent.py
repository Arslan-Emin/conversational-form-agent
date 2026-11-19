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
    ) -> None:
        self.llm = llm
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
        try:
            turn = self._run_turn(text)
        except Exception:
            del self.history[history_checkpoint:]
            raise
        # The reply the user is reading now is what they can confirm next turn. The
        # prompt makes the model show a summary once the form is complete.
        self.tools.confirmable_version = self.state.version if self.state.is_complete() else None
        return turn

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
                result = self.tools.execute(call.name, call.arguments)
                turn.tool_calls.append(ToolCallRecord(call.name, call.arguments, result))
                self.history.append(self.llm.tool_result(call.id, to_tool_output(result)))

        # Limit reached while the model still wanted tools. Close the turn with an
        # assistant message so the history stays well-formed for the next request.
        turn.hit_iteration_limit = True
        turn.reply = ITERATION_LIMIT_REPLY
        self.history.append(self.llm.assistant_message(ITERATION_LIMIT_REPLY))
        return turn
