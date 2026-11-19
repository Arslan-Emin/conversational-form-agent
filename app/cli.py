"""Terminal chat with the form agent.

Run with:  python -m app.cli
  --debug    show every tool call and its result
  --dry-run  build the Jotform submission but do not send it

If JOTFORM_API_KEY and JOTFORM_FORM_ID are set, the form schema is fetched from
Jotform and confirmed submissions are sent there. Otherwise a built-in example
form is used and submissions are always dry-run.
"""

from __future__ import annotations

import argparse
import json
import sys

from openai import OpenAIError

from app.agent import AgentTurn, FormAgent
from app.factory import create_traced_llm, create_tracer, load_form_setup
from app.jotform import JotformError
from app.rag import KnowledgeBase
from app.state import FormState

EXIT_COMMANDS = {"exit", "quit", "q", "çıkış"}


def format_state(state: FormState) -> str:
    lines = []
    for field in state.fields:
        value = state.values.get(field.name)
        mark = "x" if value else " "
        lines.append(f"    [{mark}] {field.name}: {value or '-'}")
    if state.submitted:
        lines.append(f"    SUBMITTED (id: {state.submission_id or 'dry-run'})")
    return "\n".join(lines)


def print_turn(turn: AgentTurn, state: FormState, debug: bool) -> None:
    if debug:
        for call in turn.tool_calls:
            args = json.dumps(call.arguments, ensure_ascii=False)
            result = json.dumps(call.result, ensure_ascii=False)
            print(f"  [tool] {call.name}({args})\n         -> {result}")
    print(f"Agent: {turn.reply}")
    print(
        f"  [{turn.llm_calls} LLM call(s) | in={turn.input_tokens} out={turn.output_tokens} "
        f"tokens | {turn.latency_ms:.0f} ms]"
    )
    print("  Form state:")
    print(format_state(state) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the form agent.")
    parser.add_argument("--debug", action="store_true", help="show tool calls and results")
    parser.add_argument("--dry-run", action="store_true", help="do not send submissions")
    parser.add_argument("--no-rag", action="store_true", help="disable FAQ search")
    args = parser.parse_args(argv)

    try:
        tracer = create_tracer()
        llm = create_traced_llm(tracer=tracer)
        setup = load_form_setup()
    except (RuntimeError, JotformError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    knowledge_base = None
    if not args.no_rag:
        print("Loading FAQ knowledge base (first run downloads the embedding model)...")
        knowledge_base = KnowledgeBase.from_markdown_file()

    live = setup.can_submit_live and not args.dry_run
    agent = FormAgent(
        llm,
        setup.new_state(),
        submitter=setup.create_submitter(live=live),
        knowledge_base=knowledge_base,
        tracer=tracer,
    )
    mode = "LIVE submissions" if live else "dry-run"
    print(f"Form agent ({llm.model}) | {setup.description}, {mode}. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            return 0

        try:
            turn = agent.handle_user_message(user_input)
        except OpenAIError as error:
            print(f"[LLM error] {error}\n", file=sys.stderr)
            continue
        print_turn(turn, agent.state, args.debug)


if __name__ == "__main__":
    raise SystemExit(main())
