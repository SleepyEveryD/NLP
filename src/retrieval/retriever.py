"""RAG retrieval. RAW evidence only this returns -- never a generated answer (a hard rule, it is).

FAISS over a local corpus, or a free search API returning HTML/PDF -- both honor the rule.
The retrieved text feeds the prompt as context; the LLM still reasons, it does.
"""
from __future__ import annotations

from schemas import Question, RetrievedDoc


class Retriever:
    """query -> top-k raw chunks. Embedding + FAISS, the default backend is."""

    def __init__(self, index_path: str, embedder: str = "intfloat/multilingual-e5-small", top_k: int = 3):
        # Phase 3: load the FAISS index and the embedding model, here.
        raise NotImplementedError("Phase 3: FAISS index + multilingual-e5 embedder, build here you must.")

    def retrieve(self, question: Question) -> list[RetrievedDoc]:
        # The most relevant RAW chunks, return them we do -- their source, always recorded it is.
        raise NotImplementedError
