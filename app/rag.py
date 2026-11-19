"""Retrieval-augmented generation (RAG) over the clinic FAQ.

Pipeline:
    markdown document -> chunks -> embeddings -> in-memory matrix
    query -> embedding -> cosine similarity -> top-k chunks above a threshold

The LLM never sees the whole document. It calls `search_knowledge_base`, gets
back only the most relevant chunks, and is instructed to answer from them only.

Why so simple:
- Chunking follows the document structure: every `## ` heading is one
  question/answer pair, so a chunk never cuts an answer in half.
- The FAQ has a few dozen chunks, so a numpy matrix and one dot product is all
  the "vector database" we need. Chroma or similar would add a dependency and a
  persistence layer for no benefit at this size.
- Embeddings run locally (fastembed + ONNX, no PyTorch, no API key, no cost).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FAQ_PATH = PROJECT_ROOT / "data" / "clinic_faq.md"
# Chosen by a small retrieval benchmark on this FAQ (18 TR/EN questions): the correct
# chunk was in the top 3 for 18/18 questions, vs 16/18 for the 220 MB MiniLM model.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_CACHE_DIR = PROJECT_ROOT / ".cache" / "fastembed"

DEFAULT_TOP_K = 3
# Only filters clearly unrelated chunks. In the benchmark, out-of-scope questions
# ("do you do braces?") still scored up to 0.68 while some correct chunks scored
# 0.66, so no threshold can decide "not in the document" on its own. That decision
# is left to the LLM, which is told to answer only from the returned text.
DEFAULT_MIN_SCORE = 0.3


@dataclass(frozen=True)
class Chunk:
    id: int
    title: str
    text: str

    @property
    def embedding_text(self) -> str:
        # The question heading carries most of the meaning, so embed it with the answer.
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one embedding row per text."""
        ...


class FastEmbedEmbedder:
    """Local multilingual embeddings via fastembed (downloads the model once, ~1 GB)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        from fastembed import TextEmbedding  # imported lazily: heavy, and tests don't need it

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(EMBEDDING_CACHE_DIR))

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts)), dtype=np.float32)


def chunk_markdown(markdown: str) -> list[Chunk]:
    """Split on `## ` headings. Text before the first heading (title, notes) is skipped."""
    sections = re.split(r"^## +", markdown, flags=re.MULTILINE)[1:]
    chunks = []
    for index, section in enumerate(sections):
        title, _, body = section.partition("\n")
        text = " ".join(body.split())
        if title.strip() and text:
            chunks.append(Chunk(id=index, title=title.strip(), text=text))
    return chunks


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("Knowledge base needs at least one chunk.")
        self.chunks = chunks
        self._embedder = embedder
        # Normalize once, so cosine similarity becomes a plain dot product at query time.
        self._matrix = _normalize(embedder.embed([c.embedding_text for c in chunks]))
        # One knowledge base is shared by all Streamlit sessions and parallel eval
        # runs; the embedding model is not guaranteed to be thread-safe.
        self._lock = threading.Lock()

    @classmethod
    def from_markdown_file(
        cls, path: Path = DEFAULT_FAQ_PATH, embedder: Embedder | None = None
    ) -> KnowledgeBase:
        text = Path(path).read_text(encoding="utf-8")
        return cls(chunk_markdown(text), embedder or FastEmbedEmbedder())

    def search(
        self, query: str, top_k: int = DEFAULT_TOP_K, min_score: float = DEFAULT_MIN_SCORE
    ) -> list[SearchResult]:
        with self._lock:
            query_vector = _normalize(self._embedder.embed([query]))[0]
        scores = self._matrix @ query_vector
        best = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(self.chunks[i], float(scores[i])) for i in best if scores[i] >= min_score
        ]
