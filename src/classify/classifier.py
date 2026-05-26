"""Question classifier. Route each question, this decides.

Type (mcq/open), topic, language and 'is-this-maths?' -- cheap signals, these are.
Rule-based first (fast, transparent); a learned classifier later, perhaps.
Comfortably under the latency budget, this must fit.
"""
from __future__ import annotations

import dataclasses
import re

from schemas import Question, QuestionType


# ---------------------------------------------------------------------------
# Italian stopwords/diacritics, the language heuristic uses.
# Common Italian function words, chosen they are for high frequency and
# low chance of collision with English text.
# ---------------------------------------------------------------------------
_IT_STOPWORDS: frozenset[str] = frozenset({
    "il", "lo", "la", "i", "gli", "le",
    "un", "uno", "una",
    "di", "da", "in", "con", "su", "per", "tra", "fra",
    "del", "della", "dello", "dei", "delle", "degli",
    "al", "alla", "allo", "ai", "alle", "agli",
    "dal", "dalla", "dallo", "dai", "dalle", "dagli",
    "nel", "nella", "nello", "nei", "nelle", "negli",
    "sul", "sulla", "sullo", "sui", "sulle", "sugli",
    "che", "chi", "cui", "quale", "quali",
    "non", "più", "già", "anche", "come", "dove", "quando",
    "sono", "è", "ha", "hanno", "era", "erano", "stato", "stati",
    "questo", "questa", "questi", "queste",
    "quello", "quella", "quelli", "quelle",
    "quale", "quali", "quanto", "quanta",
    "si", "ci", "ne", "lo", "li",
    "molto", "molti", "molte", "poco", "pochi",
    "primo", "prima", "secondo", "seconda",
    "anno", "anni", "secolo", "secoli",
    "cosa", "cose", "modo", "parte",
})

# Italian diacritics pattern -- accented vowels common in Italian, these are.
_IT_DIACRITICS_RE = re.compile(r"[àèéìòùÀÈÉÌÒÙ]")

# ---------------------------------------------------------------------------
# Topic keyword maps -- the 6 canonical competition labels, only these exist.
# Confident match needed; fire on ambiguous terms, we must not.
# ---------------------------------------------------------------------------
_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Maths": [
        r"\bmath(?:s|ematics)?\b",
        r"\balgebra\b",
        r"\bgeometr(?:y|ia)\b",
        r"\bcalcul(?:us|ation|ator)\b",
        r"\bequation\b",
        r"\bformula\b",
        r"\bintegral\b",
        r"\bderivative\b",
        r"\btrigonometr",
        r"\bstatistic",
        r"\bprobabilit",
        r"\bprime\s+number",
        r"\bfibonacci\b",
        r"\bpythagor",
        r"\beuclid",
    ],
    "Science and Nature": [
        r"\bphysics\b",
        r"\bchemist(?:ry|ry)\b",
        r"\bbiology\b",
        r"\bastronom",
        r"\bplanet\b",
        r"\bgravit",
        r"\bevolut",
        r"\bspecies\b",
        r"\bdna\b",
        r"\batom\b",
        r"\bmolecul",
        r"\bperiodic\s+table\b",
        r"\belement\b",
        r"\bquantum\b",
        r"\beinstein\b",
        r"\bnewton\b",
        r"\bdarwin\b",
        r"\bphotosynthes",
        r"\becosystem\b",
        r"\bclimate\b",
        r"\bvaccine\b",
        r"\bgene(?:tic)?\b",
    ],
    "Ancient History and Politics": [
        r"\brome\b",
        r"\broman\b",
        r"\bgreece\b",
        r"\bgreek\b",
        r"\bangient\b",
        r"\bancient\b",
        r"\brepublic\b",
        r"\bdemocrac",
        r"\bsenate\b",
        r"\bcaesar\b",
        r"\bsocrates\b",
        r"\bplato\b",
        r"\baristotle\b",
        r"\bpericles\b",
        r"\bpharaoh\b",
        r"\begypt(?:ian)?\b",
        r"\bmesopotami",
        r"\bparliam",
        r"\bconstitut",
        r"\belection\b",
        r"\bmonarch",
        r"\bempire\b",
        r"\bwar(?:\s+of)?\b",
        r"\btreat(?:y|ies)\b",
        r"\bpolitics\b",
        r"\bpolitical\b",
    ],
    "Philosophy and Psychology": [
        r"\bphilosoph",
        r"\bpsycholog",
        r"\bethics\b",
        r"\bmoral",
        r"\bexistential",
        r"\bkant\b",
        r"\bnietzsche\b",
        r"\bdescartes\b",
        r"\bhume\b",
        r"\bfreud\b",
        r"\bjung\b",
        r"\bcogniti",
        r"\bconsciousness\b",
        r"\bontolog",
        r"\bepistemo",
        r"\bmetaphys",
        r"\bdialect",
        r"\bsyllogism\b",
        r"\bbehavioris",
        r"\bpavlov\b",
        r"\bmaslow\b",
    ],
    "Entertainment": [
        r"\bmovie\b",
        r"\bfilm\b",
        r"\bactor\b",
        r"\bactress\b",
        r"\bdirector\b",
        r"\bmusic\b",
        r"\bsong\b",
        r"\balbum\b",
        r"\bband\b",
        r"\bsinger\b",
        r"\btelevision\b",
        r"\bseries\b",
        r"\bshow\b",
        r"\bbook\b",
        r"\bnovel\b",
        r"\bauthor\b",
        r"\baward\b",
        r"\boscar\b",
        r"\bgram(?:my|mies)\b",
        r"\bpop\s+culture\b",
        r"\bsport(?:s)?\b",
        r"\bfootball\b",
        r"\bbasketball\b",
        r"\bolympics\b",
        r"\bworld\s+cup\b",
    ],
    "News": [
        r"\bcurrent\s+event",
        r"\bbreaking\b",
        r"\brecent(?:ly)?\b",
        r"\b202[0-9]\b",      # Years 2020-2029 hint at recent news.
        r"\bnews\b",
        r"\bpresident\b",
        r"\bprime\s+minister\b",
        r"\bgovernment\b",
        r"\bunited\s+nations\b",
        r"\bun\s+report\b",
        r"\bwho\s+declared\b",
        r"\bpandemic\b",
        r"\bwar\s+in\b",
        r"\binvasion\b",
        r"\belection\s+\d",
    ],
}

# Precompile topic patterns -- once at class definition, not per call.
_TOPIC_COMPILED: dict[str, list[re.Pattern[str]]] = {
    topic: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for topic, pats in _TOPIC_KEYWORDS.items()
}

# ---------------------------------------------------------------------------
# Calculator trigger -- arithmetic cues combined with at least one digit or
# a numeric word, these require.
# Fire we must NOT on bare years like "In 1492..." -- context words needed.
# ---------------------------------------------------------------------------

# Arithmetic operator tokens, these are.
_CALC_OPERATORS_RE = re.compile(
    r"(?:"
    r"[+\-*/×÷]"          # Symbol operators, these are.
    r"|\bx\b"             # Multiplication 'x', as a word boundary.
    r"|\bplus\b"
    r"|\bminus\b"
    r"|\btimes\b"
    r"|\bdivide(?:d\s+by)?\b"
    r"|\bmultipl(?:y|ied)\b"
    r"|\bover\b"          # 'over' as division signal.
    r"|\bsquared\b"
    r"|\bcubed\b"
    r"|\bsquare\s+root\b"
    r"|\bcube\s+root\b"
    r"|\bto\s+the\s+power\b"
    r"|\braised\s+to\b"
    r")",
    re.IGNORECASE,
)

# Arithmetic WORD cues (no digit required alongside these).
_CALC_WORD_CUES_RE = re.compile(
    r"\b(?:"
    r"sum\s+of"
    r"|product\s+of"
    r"|quotient\s+of"
    r"|remainder\s+of"
    r"|average\s+of"
    r"|mean\s+of"
    r"|total\s+of"
    r"|percent(?:age)?\s+of"
    r"|how\s+many\s+(?:\w+\s+){0,3}(?:are|is|remain|left|total)"
    r"|how\s+much\s+(?:\w+\s+){0,3}(?:is|are|does)"
    r"|what\s+is\s+\d"       # "What is 3 + 4?" pattern.
    r"|calculate\b"
    r"|compute\b"
    r")",
    re.IGNORECASE,
)

# A bare digit sequence not likely to be just a year (>4 digits or decimal).
_NON_YEAR_NUMBER_RE = re.compile(
    r"\b\d{5,}\b"           # 5+ digit number.
    r"|\b\d+\.\d+\b"        # Decimal number.
    r"|\b\d+/\d+\b"         # Fraction.
    r"|\b\d+\s*%"           # Percentage.
)

# ---------------------------------------------------------------------------
# Retrieval cue patterns -- factual/knowledge-heavy signals, these are.
# ---------------------------------------------------------------------------
_RETRIEVAL_FACTUAL_RE = re.compile(
    r"\b(?:"
    r"who\s+(?:was|is|were|are|invented|discovered|wrote|created|founded|led|won|lost)"
    r"|when\s+(?:was|did|were|is)"
    r"|where\s+(?:was|did|is|are)"
    r"|which\s+year\b"
    r"|what\s+year\b"
    r"|capital\s+of\b"
    r"|author\s+of\b"
    r"|inventor\s+of\b"
    r"|founded\s+(?:by|in)\b"
    r"|discovered\s+by\b"
    r"|named\s+after\b"
    r"|first\s+(?:person|country|president|woman|man|team)\b"
    r"|born\s+in\b"
    r"|died\s+in\b"
    r")",
    re.IGNORECASE,
)

# A recent-news / "according to the article" signature -- the live News competition's tell, this is.
# In live play `adapt_question` the topic leaves UNSET, so on the TEXT we must fall back to fire retrieval
# for these post-cutoff questions (else the model, unaided and blind to 2026 events, it answers). The same
# signal the router (`retrieval.retriever._looks_like_news`) reads -- gate and route, in step they stay.
_RETRIEVAL_NEWS_RE = re.compile(
    r"\b20\d{2}-\d{2}-\d{2}\b"                       # an ISO date (2026-05-15), the strongest tell.
    r"|according\s+to\s+(?:the|a|an)\b.*\barticle\b"
    r"|\bpublished\s+on\b",
    re.IGNORECASE | re.DOTALL,
)

# Topics that almost always benefit from retrieval, these do.
_HIGH_RETRIEVAL_TOPICS: frozenset[str] = frozenset({
    "News",
    "Ancient History and Politics",
})


class QuestionClassifier:
    """Cheap routing signals, this produces -- they decide tools, retrieval and prompt choice downstream."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, question: Question) -> Question:
        """Fill topic / language / type, and the enriched question return this does.

        Decide downstream from these: calculator for maths, retrieval on/off, which prompt.

        Mutate the original, we do NOT. A fresh copy via dataclasses.replace, we return.
        Already-set fields (language, topic), preserve we do -- the caller knows best.
        """
        # Question type: options present means MCQ; otherwise OPEN it is.
        qtype = QuestionType.MCQ if question.options else QuestionType.OPEN

        # Language: keep if already known; detect cheaply otherwise, we do.
        language = question.language if question.language else self._detect_language(question.text)

        # Topic: keep if already set (live game knows it); infer only when None.
        topic = question.topic if question.topic is not None else self._infer_topic(question.text)

        # An enriched copy, return we do -- immutability, respect we must.
        return dataclasses.replace(
            question,
            qtype=qtype,
            language=language,
            topic=topic,
        )

    def needs_calculator(self, question: Question) -> bool:
        """True when arithmetic computation the question requires.

        Numbers and operators, look for we do -- but years alone, enough they are not.
        A combination of numeric content AND an arithmetic cue, needed this is.
        Word-cue phrases that imply computation, trigger on their own they may.
        """
        text = question.text

        # Strong word-cue phrases -- arithmetic intent without ambiguity they signal.
        if _CALC_WORD_CUES_RE.search(text):
            return True

        # Operator present AND a number (non-year) also present -- both required they are.
        if _CALC_OPERATORS_RE.search(text) and _NON_YEAR_NUMBER_RE.search(text):
            return True

        # Percent symbol with a digit -- calculation implied, it is.
        if re.search(r"\d\s*%", text):
            return True

        # Operator AND at least two separate digit groups -- e.g. "3 × 4", "12 + 8".
        if _CALC_OPERATORS_RE.search(text):
            digits_found = re.findall(r"\b\d+\b", text)
            # Two or more distinct numbers alongside an operator -- compute, we must.
            if len(digits_found) >= 2:
                return True

        # Square root / power phrases with any digit, sufficient these are.
        if re.search(r"\bsquare\s+root\s+of\s+\d", text, re.IGNORECASE):
            return True
        if re.search(r"\d\s+(?:squared|cubed)\b", text, re.IGNORECASE):
            return True
        if re.search(r"\braised\s+to\s+the\s+\w+\s+power\b", text, re.IGNORECASE):
            return True

        # No arithmetic signal found; calculator, not needed it is.
        return False

    def needs_retrieval(self, question: Question) -> bool:
        """True when external evidence likely helps answer the question.

        Knowledge-heavy topics, factual cue patterns -- signals these are.
        Gate retrieval this does; off in the baseline it is, but the flag remains.

        Topic alone sometimes enough is (News, Ancient History). Factual patterns
        like 'who invented', 'capital of', 'which year' also fire, they do.
        """
        # Topic in the high-retrieval set -- almost always retrieve, we should.
        if question.topic in _HIGH_RETRIEVAL_TOPICS:
            return True

        # A dated "according to the article.." News question -- live play leaves topic unset, so on the
        # text's recency signature we fire; the post-cutoff facts, ONLY retrieval can supply them.
        if _RETRIEVAL_NEWS_RE.search(question.text or ""):
            return True

        # Factual cue words in the question text, search for we do.
        if _RETRIEVAL_FACTUAL_RE.search(question.text):
            return True

        # Named-entity density proxy: many capitalised words (>= 3) hint at factual content.
        # Title-cased tokens (not start of sentence), count we do.
        tokens = question.text.split()
        # Skip the very first token -- sentence start, capitalised it always is.
        capitalised = [
            t for t in tokens[1:]
            if t and t[0].isupper() and t.isalpha()
        ]
        if len(capitalised) >= 3:
            return True

        # No retrieval signal found; skip retrieval, we can.
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_language(self, text: str) -> str:
        """Italian or English, decide we must -- cheap and transparent, keep it we do.

        Count Italian stopwords (whole-word, case-insensitive) and diacritics.
        Score above threshold → 'it'; default → 'en'.

        Note: langdetect would be more accurate.
        Import it we cannot (no new deps); a future upgrade, it is.
        """
        # Lowercase tokens, extract we do.
        tokens = re.findall(r"\b[a-zA-ZàèéìòùÀÈÉÌÒÙ]+\b", text.lower())

        if not tokens:
            # Empty or purely numeric text -- default language, return we do.
            return "en"

        # Italian stopword hits, count we do.
        stopword_hits = sum(1 for t in tokens if t in _IT_STOPWORDS)

        # Diacritic characters -- strongly Italian they are.
        diacritic_hits = len(_IT_DIACRITICS_RE.findall(text))

        # Combined score: stopwords weighted 1, diacritics weighted 2 (stronger signal).
        score = stopword_hits + diacritic_hits * 2

        # Threshold: at least 2 signals AND > 10% of tokens are Italian stopwords.
        stopword_ratio = stopword_hits / len(tokens)
        if score >= 2 and stopword_ratio >= 0.10:
            return "it"

        # Unsure or English-leaning -- default to 'en', the safer choice.
        return "en"

    def _infer_topic(self, text: str) -> str | None:
        """One of the 6 competition labels, guess we do -- only when confident.

        Keyword patterns for each topic, match against the text we do.
        Count hits per topic; the winner (if clearly ahead), return we do.
        Tie or zero hits → None, safe to be it is.
        """
        # Hits per topic, tally we do.
        hit_counts: dict[str, int] = {}
        for topic, patterns in _TOPIC_COMPILED.items():
            hits = sum(1 for pat in patterns if pat.search(text))
            if hits > 0:
                hit_counts[topic] = hits

        if not hit_counts:
            # No keyword matched; topic unknown, leave it we do.
            return None

        # The topic with the most hits, find we do.
        best_topic = max(hit_counts, key=lambda t: hit_counts[t])
        best_score = hit_counts[best_topic]

        # Ties: two topics with equal hits -- confident we are not; None return.
        tied = [t for t, s in hit_counts.items() if s == best_score]
        if len(tied) > 1:
            return None

        # At least 2 hits for a confident single winner, require we do.
        if best_score < 2:
            return None

        return best_topic
