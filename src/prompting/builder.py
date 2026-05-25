"""Prompt construction. The lever of prompt engineering, this module is.

Zero-shot, few-shot, chain-of-thought -- templates we swap, and benchmark each we do.
Per question-type and difficulty, specialise the prompt we may (an open question the rubric asks).
"""
from __future__ import annotations

from schemas import Question, RetrievedDoc


class PromptBuilder:
    """A named strategy -> a prompt string. By name, strategies are registered.

    In memory/prompts.md the human-readable record lives; here, its code form.
    Drift between the two, allow we must not.
    """

    def __init__(self, strategy: str = "zero_shot_v1"):
        self.strategy = strategy

    def build(self, question: Question, context: list[RetrievedDoc] | None = None) -> str:
        """A question (and optional raw evidence) -> the final prompt string, this turns.

        Context, when given, as RAW retrieved text it is injected -- never a generated answer.
        """
        raise NotImplementedError("Phase 1/2: the strategy registry, implement here you must.")
