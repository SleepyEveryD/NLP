"""BM25 retrieval over the local enwiki dump -- the offline, rate-limit-free knowledge backend.

`retrieve(question) -> [RetrievedDoc]`, the same shape every backend speaks. The question text is the
BM25 query (we do NOT AND the options in -- that pulls all four answers' topics in as noise, the News
lesson); BM25's IDF down-weights the scaffolding words on its own. Each matched ARTICLE is then trimmed
to the answer with the shared option-term FOCUS (retrieval._focus), so the model gets a tight excerpt --
the lead plus the windows where the option terms appear -- not a 50KB article.

The `BM25Index` is heavy to construct, so it is LAZY: built on the first retrieve (or injected, for tests).
Crash-safe ALWAYS -- a missing index / load error / query slip returns [] and the caller falls back to the
live Wikipedia net (so a half-built index never sinks a live turn).
"""
from __future__ import annotations

import re

from retrieval._focus import FOCUS_STOP
from retrieval._focus import focus as _focus_text
from retrieval._focus import option_terms as _option_terms_of
from schemas import Question, RetrievedDoc

# Question words to ignore when measuring how on-topic a retrieved article is (scaffolding, not content).
_QUERY_STOP: frozenset[str] = FOCUS_STOP | frozenset({
    "describe", "describes", "consider", "considered", "primary", "reason", "relate",
    "relates", "connection", "term", "image", "career", "role", "release", "strategy",
})


class BM25Retriever:
    """A question -> top-k FOCUSED local-corpus excerpts (BM25 ranked). Lazy index, crash-safe.

    A RELEVANCE GATE guards the local-first routing: BM25 always returns *some* best match, even for a
    question whose article is NOT in the dump (a post-cutoff / niche topic). Returning that off-topic
    article would BLOCK the live-Wikipedia fallback and feed the model junk. The gate keys on the article
    TITLE, not the body: a long biography incidentally mentions "France"/"Alexander" in its TEXT, but its
    TITLE names what it is ABOUT. So a hit is kept only if its title shares a DISTINCTIVE question term
    (a proper noun, a quoted-title word, or a number); when none do, we return [] and the caller drops to
    live -- exactly the post-cutoff / absent-topic questions that need it.
    """

    def __init__(
        self,
        index_dir: str | None = None,
        backend=None,                 # an object with .search(query, k) -- inject for tests; else BM25Index.
        top_k: int = 3,
        char_limit: int = 500,        # the lead (article head) kept as the topic anchor.
        chars_focus: int = 1100,      # total kept per doc (lead + answer-term windows).
        focus_window: int = 180,      # chars either side of an option-term hit.
        query_chars: int = 300,       # the question, capped to this as the BM25 query.
    ):
        self.index_dir = index_dir
        self._backend = backend       # None until first use (or test-injected).
        self.top_k = top_k
        self.char_limit = char_limit
        self.chars_focus = chars_focus
        self.focus_window = focus_window
        self.query_chars = query_chars

    def _get_backend(self):
        """The BM25Index, built lazily on first use -- a load failure raises (caught by `retrieve`)."""
        if self._backend is None:
            from retrieval.bm25_index import BM25Index  # imported late: bm25s is an optional dep.
            self._backend = BM25Index(self.index_dir)
        return self._backend

    @staticmethod
    def _distinctive_terms(text: str) -> list[str]:
        """The question's DISTINCTIVE terms -- proper nouns (capitalised, non-sentence-initial), words
        inside quotes (a quoted film/song title), and numbers. These name the ENTITY the question is
        about, so the right article's TITLE should carry one of them. Lowercased, scaffolding dropped."""
        terms: set[str] = set()
        text = text or ""
        for num in re.findall(r"\d[\d.,/]*\d|\d", text):
            terms.add(num)
        # words inside quotes -- a named work ('Pulp Fiction', 'woman yelling at a cat').
        for span in re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,60})[\"'“”‘’]", text):
            for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", span):
                if w.lower() not in _QUERY_STOP:
                    terms.add(w.lower())
        # proper nouns -- capitalised tokens, skipping the sentence-initial word (capitalised regardless).
        for tok in text.split()[1:]:
            w = re.sub(r"['’]s$", "", tok.strip(".,?!:;()\"'“”‘’"))
            if w[:1].isupper() and len(w) >= 3 and w.lower() not in _QUERY_STOP:
                terms.add(w.lower())
        # LONG content words (>=6 chars, e.g. "photosynthesis", "neorealism") -- the anchor for questions
        # with NO proper noun at all ("which gas do plants absorb during photosynthesis?"), so the gate
        # still has a key and a missing-topic question falls through to live instead of keeping junk.
        for w in re.findall(r"[A-Za-z]{6,}", text):
            if w.lower() not in _QUERY_STOP:
                terms.add(w.lower())
        return list(terms)

    def _on_topic(self, title: str, qterms: list[str]) -> bool:
        """True when the article TITLE shares a distinctive question term -- i.e. the article is ABOUT the
        entity asked. Empty qterms (no entity in the question) -> do not block. The title (not the body) is
        the signal: a biography's body name-drops many entities, but its title says what it is about."""
        if not qterms:
            return True
        low = (title or "").lower()
        return any(t in low for t in qterms)

    def retrieve(self, question: Question) -> list[RetrievedDoc]:
        """Top-k BM25 articles, each focused to the answer window. [] on ANY failure (live net catches)."""
        try:
            query = (question.text or "").strip()[: self.query_chars]
            if not query:
                return []
            hits = self._get_backend().search(query, k=self.top_k)
            if not hits:
                return []
            qterms = self._distinctive_terms(question.text or "")
            terms = _option_terms_of(question)
            docs: list[RetrievedDoc] = []
            for doc_id, text, source, score in hits:
                # RELEVANCE GATE (title-keyed): a hit whose TITLE shares no distinctive question term is
                # off-topic (BM25 returns SOMETHING even for an absent article) -- drop it so live answers.
                if not self._on_topic(doc_id, qterms):
                    continue
                focused = _focus_text(
                    text, terms, lead=text,
                    chars_focus=self.chars_focus, focus_window=self.focus_window,
                    chars_lead=self.char_limit,
                )
                docs.append(RetrievedDoc(
                    doc_id=str(doc_id),
                    text=focused,
                    source=str(source),
                    score=float(score),
                ))
            return docs
        except Exception:
            return []  # missing/half-built index, query slip -> no evidence; the caller falls back to live.
