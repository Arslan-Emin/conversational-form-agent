"""Streamlit demo: chat with the form agent and watch the form fill itself in.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os

import streamlit as st
from openai import OpenAIError

from app.agent import AgentTurn, FormAgent
from app.budget import DailyBudget
from app.config import get_settings
from app.factory import (
    FormSetup,
    create_knowledge_base,
    create_traced_llm,
    create_tracer,
    load_form_setup,
)
from app.jotform import JotformError
from app.llm_client import LLMClient
from app.rag import KnowledgeBase
from app.tracing import Tracer

EXAMPLE_PROMPTS = [
    "Merhaba, yarın öğleden sonra dişim için randevu almak istiyorum",
    "İptal edersem ücret alınır mı?",
    "Hi, I'd like to book a check-up next Monday at 10:00",
]
MAX_MESSAGE_CHARS = 500

st.set_page_config(page_title="Form Agent", page_icon="🦷", layout="wide")


def load_secrets_into_env() -> None:
    """Streamlit Community Cloud provides secrets via st.secrets; the app reads
    settings from environment variables (app/config.py). Copy top-level secrets
    over, without overriding variables that are already set (e.g. from .env)."""
    try:
        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets file locally is normal
        return
    for key, value in secrets.items():
        if isinstance(value, str | int | float | bool) and key not in os.environ:
            os.environ[key] = str(value)


load_secrets_into_env()


# --------------------------------------------------------------------------- #
# Shared resources: created once per server process, reused by every session.
# --------------------------------------------------------------------------- #


@st.cache_resource
def get_budget() -> DailyBudget | None:
    limit = get_settings().daily_budget_usd
    return DailyBudget(limit) if limit else None


@st.cache_resource
def get_tracer() -> Tracer:
    tracer = create_tracer()
    if (budget := get_budget()) is not None:
        tracer.add_listener(budget)  # every llm_call's cost counts towards today's total
    return tracer


@st.cache_resource(show_spinner="Connecting to the LLM...")
def get_llm() -> LLMClient:
    return create_traced_llm(tracer=get_tracer())


@st.cache_resource(show_spinner="Loading the form...", ttl=600)
def get_form_setup() -> FormSetup:
    return load_form_setup()


@st.cache_resource(show_spinner="Loading FAQ knowledge base (first run downloads a model)...")
def get_knowledge_base() -> KnowledgeBase:
    return create_knowledge_base()


# --------------------------------------------------------------------------- #
# Per-session state: each browser tab gets its own agent and form state.
# --------------------------------------------------------------------------- #


def start_conversation() -> None:
    setup = get_form_setup()
    st.session_state.agent = FormAgent(
        get_llm(),
        setup.new_state(),
        submitter=setup.create_submitter(live=False),
        knowledge_base=get_knowledge_base(),
        tracer=get_tracer(),
    )
    st.session_state.messages = []  # [{"role", "content", "turn"}]


def render_tool_calls(turn: AgentTurn) -> None:
    with st.expander(
        f"🔧 {len(turn.tool_calls)} tool call(s) · {turn.llm_calls} LLM call(s) · "
        f"{turn.input_tokens + turn.output_tokens} tokens · {turn.latency_ms / 1000:.1f}s"
    ):
        if not turn.tool_calls:
            st.caption("No tools were called.")
        for call in turn.tool_calls:
            status = "✅" if call.result.get("ok") else "❌"
            st.markdown(f"{status} **`{call.name}`**")
            st.code(json.dumps(call.arguments, ensure_ascii=False, indent=2), language="json")
            st.code(json.dumps(call.result, ensure_ascii=False, indent=2), language="json")


def render_sidebar(agent: FormAgent, setup: FormSetup) -> bool:
    """Draw the live form panel. Returns whether tool details should be shown."""
    state = agent.state
    with st.sidebar:
        st.header("📋 Form state")
        st.caption(f"Source: {setup.description}")

        filled = sum(1 for field in state.fields if field.name in state.values)
        st.progress(filled / len(state.fields), text=f"{filled}/{len(state.fields)} fields")

        for field in state.fields:
            value = state.values.get(field.name)
            icon = "✅" if value else ("⬜" if field.required else "▫️")
            st.markdown(f"{icon} **{field.label}**  \n`{value or '—'}`")

        st.caption("Only values that passed validation appear here.")

        if state.submitted:
            if state.submission_id:
                st.success(f"Submitted to Jotform · ID {state.submission_id}")
            else:
                st.info("Submitted (dry-run: nothing was sent)")

        st.divider()
        st.header("⚙️ Settings")
        live = st.toggle(
            "Send to Jotform (live)",
            value=False,
            disabled=not setup.can_submit_live or state.submitted,
            help="Off = dry-run: the submission payload is built but not sent.",
        )
        agent.tools.submitter = setup.create_submitter(live=live)
        show_tools = st.toggle("Show tool calls", value=True)

        turns = [m["turn"] for m in st.session_state.messages if m.get("turn")]
        if turns:
            tokens = sum(t.input_tokens + t.output_tokens for t in turns)
            latency = sum(t.latency_ms for t in turns) / len(turns) / 1000
            col1, col2 = st.columns(2)
            col1.metric("Tokens", f"{tokens:,}")
            col2.metric("Avg. turn", f"{latency:.1f}s")

        if st.button("🔄 New conversation", use_container_width=True):
            start_conversation()
            st.rerun()
    return show_tools


def demo_limit_message() -> str | None:
    """Why the demo can't take another message right now, or None if it can."""
    budget = get_budget()
    if budget is not None and budget.exhausted:
        return (
            "The demo has reached its daily usage limit. Please come back tomorrow, "
            "or run it locally (see the README)."
        )
    limit = get_settings().demo_max_messages
    if limit is not None and st.session_state.get("messages_sent", 0) >= limit:
        return f"This demo allows {limit} messages per session. Reload the page to start over."
    return None


def handle_message(text: str) -> bool:
    """Run one agent turn. Returns False (and shows an error) if the LLM call failed."""
    # Counted per browser session, across "New conversation" resets.
    st.session_state.messages_sent = st.session_state.get("messages_sent", 0) + 1
    st.session_state.messages.append({"role": "user", "content": text})
    try:
        with st.spinner("Thinking..."):
            turn = st.session_state.agent.handle_user_message(text)
    except (OpenAIError, JotformError) as error:
        st.session_state.messages.pop()
        st.error(f"Something went wrong, please try again. ({error.__class__.__name__})")
        return False
    st.session_state.messages.append({"role": "assistant", "content": turn.reply, "turn": turn})
    return True


def main() -> None:
    try:
        setup = get_form_setup()
        if "agent" not in st.session_state:
            start_conversation()
    except (RuntimeError, JotformError) as error:
        st.error(f"Setup error: {error}")
        st.stop()

    agent: FormAgent = st.session_state.agent
    st.title("🦷 Dental clinic booking agent")
    st.caption(
        "Book an appointment by chatting instead of filling in a form. "
        "Ask about prices, hours or cancellation at any time. Demo data only."
    )

    blocked = demo_limit_message()
    clicked = None
    if not st.session_state.messages and not blocked:
        st.markdown("**Try one of these:**")
        columns = st.columns(len(EXAMPLE_PROMPTS))
        for column, prompt in zip(columns, EXAMPLE_PROMPTS, strict=True):
            if column.button(prompt, use_container_width=True):
                clicked = prompt

    typed = st.chat_input(
        "Type your message...",
        max_chars=MAX_MESSAGE_CHARS,
        disabled=agent.state.submitted or blocked is not None,
    )
    if (message := (typed or clicked)) and handle_message(message):
        st.rerun()  # redraw from the top so the sidebar shows the updated form state

    show_tools = render_sidebar(agent, setup)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if show_tools and message.get("turn"):
                render_tool_calls(message["turn"])

    if agent.state.submitted:
        st.success("This conversation is complete. Start a new one from the sidebar.")
    if blocked:
        st.info(blocked)


main()
