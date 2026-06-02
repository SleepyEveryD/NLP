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

# Words too generic to anchor a body window on -- option boilerplate / question scaffold, these are.
_FOCUS_STOP: frozenset[str] = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "into", "their", "they",
    "was", "were", "are", "his", "her", "its", "which", "what", "who", "when", "where",
    "film", "films", "movie", "album", "song", "band", "show", "series", "none", "both",
    "all", "above", "following", "other", "than", "more", "most", "first", "best",
})


class WikipediaRetriever:
    """query -> top-k RAW Wikipedia extracts, FOCUSED on the answer. The LIVE API the backend is.

    Two upgrades over plain intro-extracts (2026-06, Entertainment evidence):
      * ENTITY-FIRST query: a quoted film/album title, searched ALONE first -- so 'Who's That Knocking
        at My Door' lands its OWN page, not the diluted '...Scorsese...Door...' combined query that
        returned the Scorsese main page instead.
      * BODY + FOCUS, not just the 700-char intro: the answer often sits DEEP in the article ("Barry
        Lyndon ... Carl Zeiss 50mm f/0.7 ... NASA"; "premiere at the Chicago International Film Festival
        1967"). So we fetch the full plaintext body of the top pages and keep the lead PLUS windows
        around where the OPTION terms / numbers appear -- bounded by `chars_focus` so latency/tokens stay
        in budget. News does NOT use this class (it has its own web retriever), so this is isolated to the
        knowledge races (Entertainment / History / Science / Philosophy).
    """

    def __init__(
        self,
        top_k: int = 3,
        lang: str = "en",
        timeout: float = 5.0,
        chars_per_doc: int = 700,
        search_limit: int = 5,
        chars_focus: int = 1100,    # total kept per doc once focused (lead + answer-term windows).
        focus_window: int = 180,    # chars kept either side of an option-term / number match in the body.
        deepen_top: int = 2,        # how many top pages to body-fetch+focus (the rest keep their intro).
    ):
        self.top_k = top_k
        self.lang = lang
        self.timeout = timeout
        self.chars_per_doc = chars_per_doc
        self.search_limit = max(search_limit, top_k)
        self.chars_focus = chars_focus
        self.focus_window = focus_window
        self.deepen_top = deepen_top
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _UA})

    def retrieve(self, question: Question) -> list[RetrievedDoc]:
        """The question -> up to top_k FOCUSED Wikipedia extracts. On ANY failure, `[]` (the turn we never sink)."""
        try:
            # ENTITY-FIRST: the proper-noun / quoted salient query (the sharpest hit), then the full
            # natural-language question (which the keyword search sometimes dilutes). (A quoted-title-ALONE
            # candidate was tried and REVERTED: question titles wrap in single quotes and a title's own
            # apostrophe -- "Who's That Knocking" -- truncated the span to "Who", whose junk hits then
            # pre-empted the good combined query. The body-focus below is the real recall win.)
            candidates: list[str] = []
            salient = self._salient_terms(question.text or "")
            if salient and salient not in candidates:
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
            # Intro extracts (ONE call) give the lead + the search ranking -- always-available baseline.
            intro_docs = self._fetch_extracts(titles)[: self.top_k]
            # Then DEEPEN the TOP `deepen_top` pages only (the answer page is ~always rank 1-2): fetch the
            # full body and focus it on the option terms. Capping the body-fetches bounds the extra API
            # load (one call each) so a fast sweep does not rate-limit (429) -- the lower-ranked docs keep
            # their intro. Body-fetch failures fall back to the intro that doc already carries (per-doc safe).
            option_terms = self._option_terms(question)
            return [
                self._deepen(doc, option_terms) if i < self.deepen_top else doc
                for i, doc in enumerate(intro_docs)
            ]
        except Exception:
            return []  # No evidence -> the model unaided answers; a live turn we must never crash.

    # ----------------------------------------------------------------- internals

    def _deepen(self, doc: RetrievedDoc, option_terms: list[str]) -> RetrievedDoc:
        """One intro doc -> a body-FOCUSED doc (lead + windows around the option terms). Intro on failure."""
        body = self._full_body(doc.doc_id)
        if not body:
            return doc  # body fetch failed -> the intro extract it already holds, keep it.
        focused = self._focus(body, option_terms, lead=doc.text)
        return RetrievedDoc(doc_id=doc.doc_id, text=focused, source=doc.source, score=doc.score)

    def _full_body(self, title: str) -> str:
        """The full plaintext article (no HTML) for one title -- '' on any failure (graceful)."""
        try:
            data = self._get({
                "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
                "exsectionformat": "plain", "titles": title, "format": "json",
            })
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                return (page.get("extract") or "").strip()
        except Exception:
            pass
        return ""

    def _option_terms(self, question: Question) -> list[str]:
        """The DISTINCTIVE tokens of the options (proper-ish words >=4 chars + numbers) -- the body
        windows we anchor on these. Generic scaffold words (`_FOCUS_STOP`) dropped they are."""
        terms: set[str] = set()
        for val in (question.options or {}).values():
            s = str(val)
            for num in re.findall(r"\d[\d.,/]*\d|\d", s):   # "0.7", "360", "1967", a bare "3".
                terms.add(num)
            for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", s):
                if w.lower() not in _FOCUS_STOP:
                    terms.add(w)
        return [t for t in terms if len(t) >= 2]

    def _focus(self, body: str, terms: list[str], lead: str) -> str:
        """Full body -> lead + bounded windows around where the option terms/numbers appear.

        The answer sentence often sits mid-article; keep the lead (topic anchor) plus a ~window around
        each option-term hit, merge overlaps, dedupe against the lead, cap at `chars_focus`. No hit ->
        the lead alone (the intro behaviour, preserved). RAW article text throughout (D-008)."""
        body = re.sub(r"\s+", " ", body).strip()
        lead_keep = lead.strip()[: self.chars_per_doc]
        if not terms or not body:
            return lead_keep or body[: self.chars_focus]

        low = body.lower()
        spans: list[tuple[int, int]] = []
        for term in terms:
            for m in re.finditer(re.escape(term.lower()), low):
                spans.append((max(0, m.start() - self.focus_window),
                              min(len(body), m.end() + self.focus_window)))
        if not spans:
            return lead_keep or body[: self.chars_focus]

        spans.sort()
        merged: list[list[int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        parts: list[str] = [lead_keep] if lead_keep else []
        used = len(lead_keep)
        lead_low = lead_keep.lower()
        for s, e in merged:
            if used >= self.chars_focus:
                break
            frag = body[s:e].strip()
            if not frag or frag.lower() in lead_low:   # already covered by the lead.
                continue
            if used + len(frag) > self.chars_focus:
                frag = frag[: max(0, self.chars_focus - used)]
            parts.append("… " + frag)
            used += len(frag)
        return " ".join(parts)

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
