"""Retrieval-only benchmark for embedding models (no LLM calls, no cost).

Run with:
    python -m evals.retrieval_benchmark
    python -m evals.retrieval_benchmark --models <model-a> <model-b>   (fastembed model ids)

For each model it reports:
- top-1 / top-3 hit rate: is the correct FAQ chunk ranked first / in the top 3?
  (The agent receives the top 3, so top-3 is what matters most.)
- the lowest score of a correct chunk vs the highest score of an out-of-scope
  query. If the second is higher, no score threshold can separate "answer
  exists" from "answer missing", and that decision has to be left to the LLM.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

from app.rag import DEFAULT_EMBEDDING_MODEL, FastEmbedEmbedder, KnowledgeBase
from evals.run_evals import fold_text

EVALS_DIR = Path(__file__).resolve().parent
CASES_PATH = EVALS_DIR / "retrieval_cases.json"
RESULTS_DIR = EVALS_DIR / "results"


def benchmark(model: str, cases: dict) -> dict:
    load_started = time.perf_counter()
    kb = KnowledgeBase.from_markdown_file(embedder=FastEmbedEmbedder(model))
    load_s = time.perf_counter() - load_started

    top1 = top3 = 0
    correct_scores, misses = [], []
    query_started = time.perf_counter()
    for case in cases["in_scope"]:
        results = kb.search(case["query"], top_k=3, min_score=-1.0)
        hits = [fold_text(case["expected"]) in fold_text(r.chunk.title) for r in results]
        top1 += hits[0]
        top3 += any(hits)
        correct_scores.append(next((r.score for r, h in zip(results, hits, strict=True) if h), 0.0))
        if not hits[0]:
            misses.append({"query": case["query"], "got": results[0].chunk.title})
    out_scores = [kb.search(q, top_k=1, min_score=-1.0)[0].score for q in cases["out_of_scope"]]
    n_queries = len(cases["in_scope"]) + len(cases["out_of_scope"])

    return {
        "model": model,
        "top1": f"{top1}/{len(cases['in_scope'])}",
        "top3": f"{top3}/{len(cases['in_scope'])}",
        "min_correct_score_in_top3": round(min(correct_scores), 3),
        "max_out_of_scope_score": round(max(out_scores), 3),
        "load_s": round(load_s, 1),
        "ms_per_query": round((time.perf_counter() - query_started) / n_queries * 1000, 1),
        "top1_misses": misses,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=[DEFAULT_EMBEDDING_MODEL])
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    reports = []
    for model in args.models:
        report = benchmark(model, cases)
        reports.append(report)
        print(f"\n{model}")
        for key in ("top1", "top3", "min_correct_score_in_top3", "max_out_of_scope_score"):
            print(f"  {key:<27} {report[key]}")
        for miss in report["top1_misses"]:
            print(f"  top-1 miss: {miss['query']!r} -> {miss['got']!r}")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"retrieval_{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {path.relative_to(EVALS_DIR.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
