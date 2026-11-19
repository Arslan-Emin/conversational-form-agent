# Conversational Form Agent

[![CI](https://github.com/Arslan-Emin/conversational-form-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Arslan-Emin/conversational-form-agent/actions/workflows/ci.yml)

Fill in a form by **chatting** instead of clicking through fields. The agent extracts
field values from the conversation, asks for what is missing, answers questions from a
FAQ document (RAG), and submits the form to the **real Jotform API** once the user has
confirmed a summary.

Built from scratch (no agent framework) as a small version of what products like
Jotform AI Agents do, with an **evaluation suite and tracing** that measure whether it
actually works. Turkish and English.

![Streamlit demo: chat on the right, live form state on the left](docs/screenshot.jpg)

```text
User:  Merhaba, ben Deniz Test, 0532 123 45 67. Pazartesi 14:30 için randevu istiyorum.
       Bu arada iptal edersem ücret var mı?
Agent: Bilgilerinizi kaydettim. İptal politikası: Randevu saatinden 24 saat öncesine
       kadar yapılan iptaller ücretsizdir...                   ← answered from the FAQ (RAG)
       Randevu için tek eksik: şikayetiniz nedir?
User:  Sol alt dişim ağrıyor
Agent: İşte randevu özetiniz: Deniz Test · 0532 123 45 67 · Pazartesi, 23 Kasım 2026,
       14:30 · Sol alt diş ağrısı. Onaylıyor musunuz?
User:  Evet
Agent: Randevunuz başarıyla alındı ✅                          ← Jotform API submission
```
*(Shortened from a real session. The user writes Turkish, so the agent does too.)*

> Independent portfolio project, not affiliated with Jotform. It only uses Jotform's
> public API. The clinic, its FAQ, prices and all names in the examples are fictional.

## Results at a glance

From the latest eval run: 26 scripted conversations × 3 repeats = 78 runs against
`deepseek-flash` ([raw results](evals/results/20261119-231456_deepseek-flash.json),
[trace summary](evals/results/20261119-231456_trace_summary.json)).

| Metric | Result |
|---|---|
| Conversations passing every check | **78 / 78** |
| Field extraction accuracy | 168 / 168 checks |
| RAG answer accuracy (in-doc and not-in-doc questions) | 36 / 36 checks |
| Wrong or premature submissions (no confirmation, prompt injection, invalid data) | **0 / 33** |
| Missed submissions (user confirmed a complete form) | 0 / 45 |
| Invalid values caught by code validation | 11 |
| Avg. cost per conversation | **$0.00067** (upper bound, peak pricing) |
| LLM call latency p50 / p95 | 0.94 s / 1.61 s |
| Time per user message p50 / p95 | 1.91 s / 2.72 s |
| Input tokens served from the prompt cache | 89% |

A perfect score mostly says the suite is now too easy for this model, see
[Limitations](#limitations-and-future-work). The more useful part is that the suite
found real problems on the way, below.

## Architecture

```mermaid
flowchart LR
    user([User]) --> ui[Streamlit UI / CLI]
    ui --> agent[FormAgent<br/>agent loop]
    agent <--> llm[TracedLLMClient]
    llm <--> provider[(DeepSeek / OpenAI)]

    agent --> tools[ToolExecutor]
    tools --> uf[update_fields] --> val[Deterministic<br/>validation] --> state[(FormState)]
    tools --> kb[search_knowledge_base] --> rag[Local embeddings<br/>+ cosine search] --> faq[[clinic_faq.md]]
    tools --> sf[submit_form] --> guard{Complete and<br/>confirmed?} --> jot[(Jotform API<br/>or dry-run)]

    llm -. llm_call events .-> tracer[Tracer]
    agent -. tool_call / agent_turn events .-> tracer
    tracer -.-> jsonl[(JSONL traces)]
```

**Agent loop** ([app/agent.py](app/agent.py)). Each user message starts a loop: send
the history and the tool list to the LLM; if it asks for tools, run them, append the
results and ask again; if it answers with text, that is the reply. `max_iterations`
caps the number of LLM calls per message, so a confused model cannot loop forever.

**Form state** ([app/state.py](app/state.py)). Field values live in code, not in the
model's memory. The model can only change them through `update_fields`, and everything
else (validation, submission, UI, evals) reads them from there.

**Validation** ([app/validation.py](app/validation.py)). Every value is checked and
normalized in plain Python before it is saved: Turkish phone formats (`+90 532…`,
`(0532) 123-45-67` → `0532 123 45 67`), dates in the past, opening hours, 30-minute
slots, booking window, names. Invalid values are not saved; the reason goes back to
the model, which explains it to the user.

**RAG** ([app/rag.py](app/rag.py)). The FAQ is split at its `##` headings (one
question and answer per chunk), embedded locally with a multilingual model through
fastembed (ONNX, no PyTorch, no API cost), and searched with cosine similarity on an
in-memory numpy matrix. The model gets the top 3 passages and must answer only from
them, or say it doesn't know.

**Jotform** ([app/jotform.py](app/jotform.py)). The form schema is fetched from
`GET /v1/form/{id}/questions` and mapped to fields by question type, so the agent
works with other forms too. Values are turned into `submission[qid][subfield]`
payloads, including Jotform's phone input masks. `--dry-run` builds the payload
without sending it; tests mock all HTTP.

**Tracing** ([app/tracing.py](app/tracing.py)). Every LLM call goes through one
wrapper that emits a structured event (model, tokens, cost, latency, tools), plus
events for each tool call and turn, tagged with conversation and turn ids. Events go
to JSONL for now; `tracer.add_listener()` is the hook for an external eval and
observability platform.

## Evaluation

```bash
python -m evals.run_evals --repeats 3      # 26 conversations x 3 against the real LLM (~$0.05)
python -m evals.retrieval_benchmark        # retrieval only, no LLM calls
python -m app.tracing summary logs/traces/<file>.jsonl
```

[evals/cases.json](evals/cases.json) holds scripted conversations in 8 categories:
normal flow, missing info, invalid info, change of mind, FAQ questions with and
without an answer in the document, no confirmation, and prompt injection. Every check
is done in code: which fields are saved (or must *not* be saved, e.g. no time may be
guessed from "afternoon"), whether the form was submitted, what the reply contains,
that no price is invented, which tools were called. The clock is pinned and
submissions are dry-run, so relative dates and results are reproducible; `--repeats`
deals with the LLM itself not being deterministic.

### What the evals caught

| Run | Change | Passing | Missed submits |
|---|---|---|---|
| [baseline](evals/results/20261119-225827_deepseek-flash_baseline.json) | none | 72 / 75 | 3 / 42 |
| [fix 1](evals/results/20261119-225955_deepseek-flash.json) | Saved values are final in the prompt; re-saving an identical value no longer invalidates the user's confirmation | 75 / 75 | 0 / 42 |
| [fix 2](evals/results/20261119-231456_deepseek-flash.json) | New per-turn checks + a date table in the system prompt | 78 / 78 | 0 / 45 |

Single runs passed. Only with repeats did a pattern show up: the model asked
*"which Monday do you mean?"* instead of saving the date, so the summary came one turn
late. Traces and transcripts of later failures showed the root cause: weekday names
felt ambiguous to the model, so it postponed saving them. Fix 2 added eval checks
that a value given early is saved *in that same turn*, and moved calendar arithmetic
into code: the system prompt now lists the next seven days with their dates, and the
model only looks them up.

Two bugs in the eval code itself were found by unit tests: a `"TL"` (lira) check
matched the letters inside *fiya**tl**arı* ("prices"), and Python's `casefold()` maps
Turkish `İ`/`I` incorrectly, so `"AĞRI"` did not match `"ağrı"`.

### Retrieval benchmark

18 Turkish and English questions with a known answer, 6 without
([results](evals/results/retrieval_20261119-230030.json)):

| Embedding model | Correct chunk 1st | Correct chunk in top 3 | Size |
|---|---|---|---|
| paraphrase-multilingual-**mpnet**-base-v2 (used) | 15 / 18 | **18 / 18** | ~1 GB |
| paraphrase-multilingual-MiniLM-L12-v2 | 14 / 18 | 16 / 18 | ~220 MB |

The agent receives the top 3 passages, so top-3 recall is what counts. With mpnet the
lowest-scoring correct chunk (0.60) scores *below* the best match for an out-of-scope
question (0.68), so no similarity threshold can decide "the answer is not in the
document". That decision is left to the LLM, which is told to answer only from the
passages; the not-in-doc eval cases check that it does.

## Design decisions

- **Own agent loop instead of a framework.** The loop is about 50 lines, every step is
  visible and testable, and a scripted fake LLM can drive it in unit tests.
- **Deterministic code for anything that must be right.** Validation, date
  arithmetic, form state and the submission rule are code. The LLM handles language;
  code handles facts.
- **The confirmation rule is enforced in code, not just in the prompt.**
  `submit_form` is rejected unless all required fields are valid *and* the user saw
  exactly these values in the previous turn (a state version counter). A prompt
  injection cannot talk its way past it. In the eval runs the model never tried, so
  this layer is defense in depth.
- **Validation inside `update_fields`, not as a separate tool.** A separate
  `validate_field` tool would be optional for the model. Here saving and validating
  are the same step.
- **Tool errors go back to the model instead of raising.** Invalid JSON, unknown
  tools or rejected values become structured tool results, so the model can recover
  and ask the user again.
- **Thin provider abstraction.** The agent only depends on an `LLMClient` protocol;
  there are Chat Completions (DeepSeek, any OpenAI-compatible API) and OpenAI
  Responses implementations. Conversation history stays in our code.
- **Local embeddings and a numpy "vector store".** The FAQ has 23 chunks; a database
  would add a dependency and no value. Embeddings cost nothing and need no API key.
- **Privacy by default in traces.** Message text and tool arguments are only logged
  with `TRACE_INCLUDE_CONTENT=true`; values rejected by validation are never logged,
  only the field names. Evals turn content on because they use fake data only.
- **Dry-run by default.** CLI, UI, tests and evals never submit for real unless asked.

## Getting started

Requires Python 3.11+.

```bash
git clone https://github.com/Arslan-Emin/conversational-form-agent.git
cd conversational-form-agent
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # then fill in at least DEEPSEEK_API_KEY
```

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `deepseek` (default) or `openai` |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `DEEPSEEK_THINKING` | DeepSeek settings (default model `deepseek-flash`, thinking `disabled`) |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | Only for `LLM_PROVIDER=openai` |
| `JOTFORM_API_KEY`, `JOTFORM_FORM_ID`, `JOTFORM_BASE_URL` | Optional. Without them a built-in example form is used and submissions are dry-run |
| `TRACE_ENABLED`, `TRACE_DIR`, `TRACE_INCLUDE_CONTENT` | Tracing (on, `logs/traces`, content off) |

```bash
streamlit run streamlit_app.py          # web UI
python -m app.cli --debug --dry-run     # terminal chat, shows tool calls
python -m app.jotform questions         # inspect your Jotform form's questions
pytest                                  # 126 tests, no network, no API keys
ruff check . && ruff format --check .
```

The first start downloads the embedding model (~1 GB) into `.cache/`.

## Project structure

```text
app/
  agent.py          agent loop, turn bookkeeping, trace events
  tools.py          update_fields, submit_form, search_knowledge_base
  state.py          form schema and state (pydantic)
  validation.py     deterministic validators, opening hours
  prompts.py        system prompt, upcoming-days date table
  rag.py            chunking, embeddings, retrieval
  jotform.py        Jotform client, schema mapping, submitters
  llm_client.py     provider-agnostic LLM interface, DeepSeek/OpenAI clients
  tracing.py        tracer, JSONL sink, traced LLM wrapper, trace summaries
  pricing.py        token prices and cost estimates
  factory.py        wiring shared by CLI and UI
  cli.py            terminal chat
streamlit_app.py    web UI
data/clinic_faq.md  RAG source document (fictional)
evals/              scripted conversations, runner, retrieval benchmark, results
tests/              unit tests (fake LLM, mocked HTTP, fake embedder)
```

## Limitations and future work

- **The eval suite is small and scripted.** 26 conversations with fixed user messages
  are far less varied than real users, and a perfect score means the suite needs
  harder cases. Next: an LLM-simulated user and adversarial cases.
- **Reply checks are keyword based.** "Does the answer contain *24 saat*?" can miss
  a correct paraphrase or accept a wrong one. Next: LLM-as-judge with a rubric,
  calibrated against human labels.
- **One model measured.** All numbers are for `deepseek-flash` with thinking off.
  Comparing models, thinking on/off and prompt versions is exactly what an eval
  platform should make easy.
- **Costs are estimates** from a price table, using DeepSeek's peak price, so they are
  an upper bound.
- **No persistence or auth.** Conversations live in memory; the UI is a demo.
- **Validation rules are specific to the dental example** (Turkish phone numbers,
  clinic hours). Other forms reuse the generic text checks.
- **Not tested live:** Jotform phone fields without an input mask and AM/PM time
  fields (the masked phone and 24-hour date/time fields were verified with real test
  submissions that were deleted afterwards).

Next step: a separate **LLM eval and observability platform** that consumes this
project's trace events, replays the eval suite across models and prompt versions, and
tracks the metrics above over time.
