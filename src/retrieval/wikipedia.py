"""Live Wikipedia retrieval -- RAW article text only, never a generated answer (D-008).

The free Wikipedia Action API we query (no key; raw extracts it returns) -- the assignment's RAG rule it
honors: NOT a paid API, RAW non-generated content, and in the video named it must be. Per question we
search Wikipedia, the top pages' intro extracts fetch, and as RetrievedDoc chunks return them. Evidence
ONLY we feed -- the LLM still reasons.

Graceful ALWAYS: a network slip / timeout / parse error -> `[]` we return, so the 30s live turn it never
crashes (the model, unaided, still answers).
"""
from __future__ import annotations

import re
import time

import requests

from schemas import Question, RetrievedDoc

_API = "https://{lang}.wikipedia.org/w/api.php"
# A descriptive User-Agent, Wikipedia asks for (a bare python-requests UA, sometimes blocked it is).
_UA = "PoliMillionaire-NLP-Assignment/1.0 (educational; Politecnico di Milano NLP course)"


class WikipediaRetriever:
    """query -> top-k RAW Wikipedia intro extracts. The LIVE API the backend is -- no local index, none."""

    def __init__(
        self,
        top_k: int = 3,
        lang: str = "en",
        timeout: float = 5.0,
        chars_per_doc: int = 700,
        search_limit: int = 5,
    ):
        self.top_k = top_k
        self.lang = lang
        self.timeout = timeout
        self.chars_per_doc = chars_per_doc
        self.search_limit = max(search_limit, top_k)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _UA})

    def retrieve(self, question: Question) -> list[RetrievedDoc]:
        """The question -> up to top_k raw Wikipedia extracts. On ANY failure, `[]` (the turn we never sink)."""
        try:
            # ENTITY-FIRST: a quoted title / proper noun ('Marriage Story', M3GAN) the sharpest hit it gives;
            # only then the full natural-language question (which sometimes the keyword search dilutes).
            candidates: list[str] = []
            salient = self._salient_terms(question.text or "")
            if salient:
                candidates.append(salient)
            full = self._build_query(question)
            if full and full not in candidates:
                candidates.append(full)

            titles: list[str] = []
            for query in candidates:
                titles = self._search(query)
                if titles:
                    break
            if not titles:
                return []
            return self._fetch_extracts(titles)[: self.top_k]
        except Exception:
            return []  # No evidence -> the model unaided answers; a live turn we must never crash.

    # ----------------------------------------------------------------- internals

    def _build_query(self, question: Question) -> str:
        # The question text, the query it is -- natural language Wikipedia search handles. Capped, it stays.
        return (question.text or "").strip()[:300]

    def _salient_terms(self, text: str) -> str:
        """The proper nouns + quoted strings, a tighter keyword query they make (the entity, search it we do).

        Abstract questions ('the fundamental principle that drives M3GAN...') the full search misses;
        the entity ('M3GAN') alone, Wikipedia finds. The sentence-initial word, skip it we do (always
        capitalised it is); a trailing possessive ''s', trim it we do ("M3GAN's" -> "M3GAN").
        """
        quoted = re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,40})[\"'“”‘’]", text)
        caps: list[str] = []
        for tok in text.split()[1:]:  # skip the first token -- sentence start, capitalised regardless it is.
            w = re.sub(r"['’]s$", "", tok.strip(".,?!:;()\"'"))
            if w[:1].isupper() and len(w) >= 3:
                caps.append(w)
        return " ".join(dict.fromkeys(quoted + caps))[:300]

    def _get(self, params: dict) -> dict:
        """One API GET, with a SINGLE short retry on 429 (rate limit) -- then give up, graceful we stay.

        The 30s wall is real, so the back-off we cap (<=2s). Still 429? raise -> `retrieve` returns []
        (no evidence, the model unaided answers). The sweep hammering Wikipedia, this softens.
        """
        url = _API.format(lang=self.lang)
        r = self._session.get(url, params=params, timeout=self.timeout)
        if r.status_code == 429:  # Too many requests -- a brief, capped wait, then ONE retry.
            try:
                wait = min(float(r.headers.get("Retry-After", "1") or 1), 2.0)
            except ValueError:
                wait = 1.0
            time.sleep(wait)
            r = self._session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _search(self, query: str) -> list[str]:
        if not query:
            return []
        data = self._get({
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": self.search_limit,
        })
        hits = data.get("query", {}).get("search", [])
        return [h["title"] for h in hits if h.get("title")]

    def _fetch_extracts(self, titles: list[str]) -> list[RetrievedDoc]:
        # The intro extracts for ALL titles, in ONE call we fetch (plain text, no HTML).
        data = self._get({
            "action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
            "redirects": 1, "titles": "|".join(titles[: self.search_limit]), "format": "json",
        })
        pages = data.get("query", {}).get("pages", {})
        # The pages dict, by pageid keyed (unordered) -> the search rank, restore it via the titles order.
        rank = {t: i for i, t in enumerate(titles)}
        docs: list[RetrievedDoc] = []
        for page in pages.values():
            title = page.get("title", "")
            extract = (page.get("extract") or "").strip()
            if not extract:
                continue  # A page without an extract (e.g. a disambiguation), skip it we do.
            docs.append(
                RetrievedDoc(
                    doc_id=title,
                    text=extract[: self.chars_per_doc],
                    source=f"https://{self.lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    score=1.0 / (rank.get(title, 99) + 1),  # Higher for the better search hits.
                )
            )
        docs.sort(key=lambda d: d.score, reverse=True)
        return docs
