"""RAG tests with a deterministic fake embedder: no model download, runs in CI."""

from __future__ import annotations

import re

import numpy as np
import pytest

from app.rag import DEFAULT_FAQ_PATH, Chunk, KnowledgeBase, chunk_markdown

VOCABULARY = ["iptal", "ücret", "saat", "pazar", "otopark", "fiyat"]


class KeywordEmbedder:
    """Bag-of-words vectors over a tiny vocabulary: similar words -> similar vectors."""

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            words = re.findall(r"\w+", text.lower())
            rows.append([sum(word.startswith(v) for word in words) for v in VOCABULARY])
        return np.array(rows, dtype=np.float32)


MARKDOWN = """# Title

> Intro note that is not a chunk.

## Randevumu iptal edersem ücret alınır mı?
24 saat öncesine kadar iptal ücretsizdir.

## Otopark var mı?
Yan sokakta
ücretli otopark var.

## Boş başlık
"""


def test_chunks_follow_headings_and_skip_preamble_and_empty_sections() -> None:
    chunks = chunk_markdown(MARKDOWN)

    assert [c.title for c in chunks] == [
        "Randevumu iptal edersem ücret alınır mı?",
        "Otopark var mı?",
    ]
    # Line breaks inside an answer are collapsed.
    assert chunks[1].text == "Yan sokakta ücretli otopark var."


def test_real_faq_document_chunks() -> None:
    chunks = chunk_markdown(DEFAULT_FAQ_PATH.read_text(encoding="utf-8"))

    assert len(chunks) >= 20
    assert all(c.title.endswith("?") for c in chunks)
    cancellation = next(c for c in chunks if "iptal edersem" in c.title)
    assert "24 saat" in cancellation.text


def test_search_returns_best_match_first() -> None:
    kb = KnowledgeBase(chunk_markdown(MARKDOWN), KeywordEmbedder())

    results = kb.search("iptal ücreti", top_k=2, min_score=0.0)

    assert results[0].chunk.title.startswith("Randevumu iptal")
    assert results[0].score > results[1].score


def test_search_drops_results_below_min_score() -> None:
    kb = KnowledgeBase(chunk_markdown(MARKDOWN), KeywordEmbedder())

    assert kb.search("pazar", min_score=0.3) == []


def test_empty_knowledge_base_is_rejected() -> None:
    with pytest.raises(ValueError):
        KnowledgeBase([], KeywordEmbedder())


def test_embedding_text_includes_title() -> None:
    chunk = Chunk(id=0, title="Otopark var mı?", text="Evet.")
    assert chunk.embedding_text == "Otopark var mı?\nEvet."
