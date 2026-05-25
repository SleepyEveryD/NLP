"""Question classifier. Route each question, this decides.

Type (mcq/open), topic, language and 'is-this-maths?' -- cheap signals, these are.
Rule-based first (fast, transparent); a learned classifier later, perhaps.
Comfortably under the latency budget, this must fit.
"""
from __future__ import annotations

from schemas import Question


class QuestionClassifier:
    """Cheap routing signals, this produces -- they decide tools, retrieval and prompt choice downstream."""

    def classify(self, question: Question) -> Question:
        """Fill topic / language / type, and the enriched question return this does.

        Decide downstream from these: calculator for maths, retrieval on/off, which prompt.
        """
        raise NotImplementedError("Phase 0/2: rule-based start (regex for numbers, langdetect).")

    def needs_calculator(self, question: Question) -> bool:
        # Numbers and arithmetic words, look for we do.
        raise NotImplementedError

    def needs_retrieval(self, question: Question) -> bool:
        # Knowledge-heavy and factual, is the question? Then retrieve, we should.
        raise NotImplementedError
