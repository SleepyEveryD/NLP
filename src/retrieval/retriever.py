"""RAG retrieval. RAW evidence only this returns -- never a generated answer (a hard rule, D-008 it is).

Three backends, this module gives, and one facade that routes between them:
  * WikipediaRetriever   -- live MediaWiki API; free, no key, RAW extracts. Knowledge topics, it serves.
  * WebSearchRetriever   -- live DuckDuckGo search; RAW result snippets. Post-cutoff NEWS, it serves.
  * FaissRetriever       -- local corpus (Simple Wikipedia), dense vectors. Course-aligned RAG, it is.
  * Retriever (facade)   -- per QUESTION it routes: News -> web, else -> dense/wikipedia.

Why per-question routing, not per-competition? One pipeline ALL six games plays (`run_all_competitions`),
so a single Retriever instance every topic must serve. The News questions a very distinctive shape have
("According to the article published on 2026-05-..", a Guardian byline, an ISO date) -- on that we route,
far more reliably than the generic topic classifier (which `adapt_question` leaves unset in live play).

The retrieved text feeds the prompt as RAW context (`prompting.builder._build_context_block`); the LLM
still reasons over it, it does. No backend ever an answer generates -- raw chunks only, always.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

import requests

from schemas import Question, RetrievedDoc
from retrieval.wikipedia import WikipediaRetriever   # the polished live backend, reuse it we do.


# --------------------------------------------------------------------------- #
# News detection -- THIS game's News questions a tell-tale signature carry.
# --------------------------------------------------------------------------- #

# An ISO date (2026-05-15), an "according to .. article" lead-in, a "published on", a Guardian byline --
# any one of these, a News question it betrays. On the offline mix AND live play alike, the text the same is.
_NEWS_SIGNATURE = re.compile(
    r"\b20\d{2}-\d{2}-\d{2}\b"               # An ISO date, the strongest tell it is.
    r"|according\s+to\s+(?:the|a|an)\b.*\barticle\b"
    r"|\bpublished\s+on\b"
    r"|\bthe\s+guardian\b",
    re.IGNORECASE | re.DOTALL,
)


def _looks_like_news(question: Question) -> bool:
    """True when a post-cutoff NEWS question this is -- to the live web, route it we should."""
    # The topic, when the caller set it (offline dataset / a future client enrichment), trust we do.
    if question.topic and "news" in question.topic.lower():
        return True
    # Else the text's own signature, read we do -- reliable for this game's News, it is.
    return bool(_NEWS_SIGNATURE.search(question.text or ""))


def _query_from_question(question: Question, max_chars: int = 300) -> str:
    """A clean search query, from the question text distil it we do.

    The "According to the article published on <date>," boilerplate, strip it we do -- noise for a
    search engine it is, the real entities it buries. Whitespace collapsed, length capped, the rest is.
    """
    text = (question.text or "").strip()
    # The dated lead-in clause, drop it we do -- "According to .., " up to the first comma, gone it is.
    text = re.sub(
        r"^\s*(?:on\s+)?(?:according\s+to\b[^,]*,\s*)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^\s*on\s+20\d{2}-\d{2}-\d{2}\s*,\s*", "", text, flags=re.IGNORECASE)
    # Whitespace, collapse it we do; the length, cap it we must (a search box, finite it is).
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


# --------------------------------------------------------------------------- #
# Backend 1 -- live Wikipedia: reused from `retrieval.wikipedia.WikipediaRetriever`.
# Entity-first search, a 429 retry, a shared session -- already polished it is, so duplicate it we do not.
# (Imported at the top.) Its knobs: top_k, lang, timeout, chars_per_doc, search_limit.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Backend 2 -- live web search (DuckDuckGo HTML; for post-cutoff NEWS).
# --------------------------------------------------------------------------- #

# A result snippet in DDG's HTML lite endpoint, this matches -- the <a class="result__snippet">..</a> text.
_DDG_SNIPPET = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")  # Inner HTML tags (bold highlights), strip them we do.


def _unescape_html(s: str) -> str:
    import html as _html
    return _html.unescape(s)


class WebSearchRetriever:
    """query -> top-k RAW web result snippets. DuckDuckGo's keyless HTML endpoint, this scrapes.

    For NEWS only this is -- the post-cutoff events Wikipedia cannot know (a Malian minister killed on a
    2026 date, a whale named Timmy). RAW snippets we return, the rule honouring -- no answer we synthesise.

    Brittle, web scraping inherently is (a layout change, a bot block, a 429). So crash-safe entirely it
    stays: empty list on ANY failure. A `search_fn` injection point we expose -- a different free source
    (a news RSS, a search API you name in the video) drop in here you can, without touching the routing.
    """

    _URL = "https://html.duckduckgo.com/html/"
    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    }

    def __init__(
        self,
        top_k: int = 3,
        char_limit: int = 400,
        timeout_s: float = 6.0,
        search_fn: Optional[Callable[[str, int], list[RetrievedDoc]]] = None,
    ):
        self.top_k = top_k
        self.char_limit = char_limit
        self.timeout_s = timeout_s
        # An override hook -- when given, OURS it replaces (a Guardian RSS, a NewsAPI, your choice).
        self._search_fn = search_fn

    def retrieve(self, question: Question) -> list[RetrievedDoc]:
        query = _query_from_question(question)
        if not query:
            return []
        try:
            if self._search_fn is not None:
                return self._search_fn(query, self.top_k)
            return self._ddg_search(query)
        except Exception:
            return []

    # -- internals --

    def _ddg_search(self, query: str) -> list[RetrievedDoc]:
        resp = requests.post(
            self._URL, data={"q": query}, headers=self._HEADERS, timeout=self.timeout_s,
        )
        resp.raise_for_status()
        docs: list[RetrievedDoc] = []
        for i, raw in enumerate(_DDG_SNIPPET.findall(resp.text)):
            text = _unescape_html(_TAG.sub("", raw)).strip()
            if not text:
                continue
            docs.append(RetrievedDoc(
                doc_id=f"ddg:{i}",
                text=text[: self.char_limit],
                source="duckduckgo",
                score=0.0,
            ))
            if len(docs) >= self.top_k:
                break
        return docs


# --------------------------------------------------------------------------- #
# Backend 3 -- local FAISS over a corpus (the course's dense RAG).
# --------------------------------------------------------------------------- #

class FaissRetriever:
    """query -> top-k RAW chunks from a LOCAL corpus. multilingual-e5 + FAISS, this is.

    The index a directory is: `<index_path>/index.faiss` + `<index_path>/docs.jsonl` (one
    `{doc_id, text, source}` per line, row-aligned to the FAISS vectors). Build it with
    `src/retrieval/build_index.py`, you do (on Colab, once per corpus).

    e5 a prefix convention has: PASSAGES "passage: " at index time, QUERIES "query: " at search time.
    Honour it we must, or the cosine scores meaningless they are. Normalised embeddings + inner-product
    index => cosine similarity, this gives. Heavy deps (faiss, sentence-transformers) LAZILY loaded they
    are -- importing this module, a GPU it must never wake.
    """

    def __init__(
        self,
        index_path: str,
        embedder: str = "intfloat/multilingual-e5-small",
        top_k: int = 3,
        char_limit: int = 600,
    ):
        self.index_path = Path(index_path)
        self.embedder_name = embedder
        self.top_k = top_k
        self.char_limit = char_limit
        self._model = None     # Lazily loaded, the SentenceTransformer is.
        self._index = None     # Lazily loaded, the FAISS index is.
        self._docs: list[dict] = []

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        idx_file = self.index_path / "index.faiss"
        docs_file = self.index_path / "docs.jsonl"
        if not idx_file.exists() or not docs_file.exists():
            raise FileNotFoundError(
                f"No FAISS index at {self.index_path} -- build it with build_index.py, you must."
            )
        self._index = faiss.read_index(str(idx_file))
        self._docs = [
            json.loads(line)
            for line in docs_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._model = SentenceTransformer(self.embedder_name)

    def retrieve(self, question: Question) -> list[RetrievedDoc]:
        try:
            self._ensure_loaded()
        except Exception:
            # No index / a load failure -- unaided the model answers. The turn, crash it must not.
            return []
        try:
            query = "query: " + _query_from_question(question)   # the e5 query prefix, mandatory it is.
            vec = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            scores, idxs = self._index.search(vec, self.top_k)
            docs: list[RetrievedDoc] = []
            for score, i in zip(scores[0], idxs[0]):
                if i < 0 or i >= len(self._docs):
                    continue
                d = self._docs[i]
                docs.append(RetrievedDoc(
                    doc_id=str(d.get("doc_id", i)),
                    text=str(d.get("text", ""))[: self.char_limit],
                    source=str(d.get("source", "corpus")),
                    score=float(score),
                ))
            return docs
        except Exception:
            return []


# --------------------------------------------------------------------------- #
# The facade -- per-question routing between the backends.
# --------------------------------------------------------------------------- #

class Retriever:
    """The one retriever the pipeline holds -- per QUESTION, the right backend it picks.

    News question?  -> the live web (`WebSearchRetriever`), with Wikipedia as a safety net.
    Anything else?  -> dense FAISS over the local corpus, OR live Wikipedia when no index there is.

    `source` the strategy chooses (from `RetrievalConfig.source`):
      "routed" (default) -- News->web(+wiki fallback), else->faiss-or-wikipedia.  <- "both", the user picked.
      "wikipedia"        -- always live Wikipedia (the existing live.yaml default).
      "web"              -- always DuckDuckGo web search.
      "faiss"            -- always the local corpus (an `index_path` it needs).

    Backends LAZILY constructed they are (and FAISS even more lazily loads its model) -- so
    `Retriever(...)`, cheap and side-effect-free it stays until the first `retrieve`.
    Signature back-compatible with the notebook's `Retriever(top_k=...)` call, it remains.
    """

    def __init__(
        self,
        top_k: int = 3,
        source: str = "routed",
        index_path: Optional[str] = None,
        embedder: str = "intfloat/multilingual-e5-small",
        char_limit: int = 600,
        timeout_s: float = 6.0,
    ):
        self.top_k = top_k
        self.source = (source or "routed").lower()
        self.index_path = index_path
        self.embedder = embedder
        self.char_limit = char_limit
        self.timeout_s = timeout_s
        # The backends, on first use built they are -- a dict of name -> instance, cached here.
        self._cache: dict[str, object] = {}

    # -- backend builders (memoised) --

    def _wikipedia(self) -> WikipediaRetriever:
        if "wikipedia" not in self._cache:
            self._cache["wikipedia"] = WikipediaRetriever(
                top_k=self.top_k, chars_per_doc=self.char_limit, timeout=self.timeout_s,
            )
        return self._cache["wikipedia"]  # type: ignore[return-value]

    def _web(self) -> WebSearchRetriever:
        if "web" not in self._cache:
            self._cache["web"] = WebSearchRetriever(
                top_k=self.top_k, char_limit=min(self.char_limit, 400), timeout_s=self.timeout_s,
            )
        return self._cache["web"]  # type: ignore[return-value]

    def _faiss(self) -> Optional[FaissRetriever]:
        # No index path -- a FAISS backend, build it we cannot. None we return, and the caller falls back.
        if not self.index_path:
            return None
        if "faiss" not in self._cache:
            self._cache["faiss"] = FaissRetriever(
                index_path=self.index_path, embedder=self.embedder,
                top_k=self.top_k, char_limit=self.char_limit,
            )
        return self._cache["faiss"]  # type: ignore[return-value]

    # -- the public route --

    def retrieve(self, question: Question) -> list[RetrievedDoc]:
        """Per the source strategy, the right backend dispatch -- RAW docs out, always."""
        if self.source == "wikipedia":
            return self._wikipedia().retrieve(question)
        if self.source == "web":
            return self._web().retrieve(question)
        if self.source == "faiss":
            faiss_be = self._faiss()
            return faiss_be.retrieve(question) if faiss_be else []

        # "routed" / "both" / "hybrid" -- the default: by the question, decide we do.
        if _looks_like_news(question):
            # NEWS -- the live web first, for the post-cutoff facts Wikipedia cannot hold.
            docs = self._web().retrieve(question)
            if docs:
                return docs
            # The web blocked us (a 429, a layout shift) -- Wikipedia, a best-effort net it casts.
            return self._wikipedia().retrieve(question)

        # KNOWLEDGE -- the local corpus when we have one, else live Wikipedia.
        faiss_be = self._faiss()
        if faiss_be is not None:
            docs = faiss_be.retrieve(question)
            if docs:
                return docs
        return self._wikipedia().retrieve(question)


def build_retriever(retrieval_cfg, **overrides) -> Optional[Retriever]:
    """A `RetrievalConfig` -> a wired `Retriever` (or None when disabled). The factory, this is.

    From the config the strategy and knobs it reads; `**overrides` win, for a notebook ablation handy.
    `enabled=False` -> None, so the pipeline its retrieval stage skips entirely.
    """
    if not getattr(retrieval_cfg, "enabled", False):
        return None
    return Retriever(
        top_k=overrides.get("top_k", getattr(retrieval_cfg, "top_k", 3)),
        source=overrides.get("source", getattr(retrieval_cfg, "source", "routed")),
        index_path=overrides.get("index_path", getattr(retrieval_cfg, "index_path", None)),
        embedder=overrides.get("embedder", getattr(retrieval_cfg, "embedder", "intfloat/multilingual-e5-small")),
    )
