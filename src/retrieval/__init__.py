"""retrieval package. A module of the PoliMillionaire system, this is.

Two backends honor the RAW-evidence rule (D-008):
- `WikipediaRetriever` -- the live, free Wikipedia API (Phase 4 default; no index to build).
- `Retriever`          -- a local FAISS/hnswlib corpus (the stubbed alternative).
"""
from .wikipedia import WikipediaRetriever

__all__ = ["WikipediaRetriever"]
