"""Prompt construction. The lever of prompt engineering, this module is.

Zero-shot, few-shot, chain-of-thought -- templates we swap, and benchmark each we do.
Per question-type and difficulty, specialise the prompt we may (an open question the rubric asks).
"""
from __future__ import annotations

from schemas import Question, QuestionType, RetrievedDoc


# --- Strategy implementations ---

def _build_context_block(context: list[RetrievedDoc]) -> str:
    """A list of retrieved docs -> a formatted evidence block, this turns.

    Raw text only, injected it is -- never an LLM-generated answer (D-008).
    """
    # Each doc's raw text, on its own numbered line it goes.
    lines = ["Referenced knowledge:"]
    for i, doc in enumerate(context, start=1):
        lines.append(f"[{i}] {doc.text.strip()}")
    lines.append("")  # A blank line, separation it provides.
    return "\n".join(lines)


def _zero_shot_v1(question: Question, context: list[RetrievedDoc] | None) -> str:
    """The zero-shot baseline prompt, this builds.

    A single user-turn string returned it is -- the chat template, applied elsewhere it must be.
    """
    parts: list[str] = []

    # Context block, only when evidence exists, prepend we do.
    if context:
        parts.append(_build_context_block(context))

    # The question text, first it comes.
    parts.append(f"Question: {question.text.strip()}")

    if question.qtype == QuestionType.OPEN or not question.options:
        # An open question, a short free-text answer it expects.
        parts.append("Answer briefly in one or two sentences.")
    else:
        # The options, in A-D order they appear -- missing letters, skipped they are.
        # A single letter, demand we must.
        option_lines: list[str] = []
        for letter in ("A", "B", "C", "D"):
            if letter in question.options:
                option_lines.append(f"{letter}) {question.options[letter]}")
        parts.append("\n".join(option_lines))
        parts.append(
            "Reply with ONLY the letter of the correct option (A, B, C, or D). "
            "No explanation, no punctuation -- the letter alone."
        )

    return "\n".join(parts)


# --- Strategy registry ---

# A name -> builder function, this dict is.
# To add a new strategy (few_shot_v1, cot_v1), register it here you must.
_REGISTRY: dict[str, object] = {
    "zero_shot_v1": _zero_shot_v1,
}


class PromptBuilder:
    """A named strategy -> a prompt string. By name, strategies are registered.

    In memory/prompts.md the human-readable record lives; here, its code form.
    Drift between the two, allow we must not.
    """

    def __init__(self, strategy: str = "zero_shot_v1"):
        # The chosen strategy name, stored it is -- validated at build time.
        self.strategy = strategy

    def build(self, question: Question, context: list[RetrievedDoc] | None = None) -> str:
        """A question (and optional raw evidence) -> the final prompt string, this turns.

        Context, when given, as RAW retrieved text it is injected -- never a generated answer.
        The chat template, NOT applied here it is -- that, the inference engine does (D-003).
        """
        # The strategy function, looked up it is -- unknown names, loudly rejected they are.
        builder_fn = _REGISTRY.get(self.strategy)
        if builder_fn is None:
            known = ", ".join(sorted(_REGISTRY))
            raise ValueError(
                f"Unknown prompt strategy: {self.strategy!r}. "
                f"Known strategies, these are: {known}"
            )
        return builder_fn(question, context)  # type: ignore[operator]
