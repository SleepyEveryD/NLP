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


def _render_mcq(text: str, options: dict[str, str]) -> str:
    """A question and its options -> the 'Question/A)/B)...' block, this renders.

    In A-D order the options appear; missing letters, skipped they are.
    """
    lines = [f"Question: {text.strip()}"]
    for letter in ("A", "B", "C", "D"):
        if letter in options:
            lines.append(f"{letter}) {options[letter]}")
    return "\n".join(lines)


# A few solved exemplars -- the answer FORMAT they teach, and arithmetic they prime.
# From OUTSIDE the dev set drawn they are -- leakage into the benchmark, avoid we must.
_FEW_SHOT_EXAMPLES = [
    ("What is the capital of France?",
     {"A": "Madrid", "B": "Paris", "C": "Rome", "D": "Berlin"}, "B"),
    ("Which gas do plants absorb from the air for photosynthesis?",
     {"A": "Oxygen", "B": "Hydrogen", "C": "Carbon dioxide", "D": "Nitrogen"}, "C"),
    ("What is 6 multiplied by 7?",
     {"A": "42", "B": "36", "C": "48", "D": "49"}, "A"),
]


def _few_shot_v1(question: Question, context: list[RetrievedDoc] | None) -> str:
    """Solved exemplars before the target, this prepends -- the output format, they anchor.

    A single user-turn string returned it is -- the chat template, applied elsewhere it must be.
    """
    parts: list[str] = []

    # Context block, only when evidence exists, prepend we do.
    if context:
        parts.append(_build_context_block(context))

    # The worked examples, first they come -- 'Answer: X' each one ends with.
    for ex_text, ex_opts, ex_gold in _FEW_SHOT_EXAMPLES:
        parts.append(_render_mcq(ex_text, ex_opts))
        parts.append(f"Answer: {ex_gold}")
        parts.append("")  # A blank line between examples, separation it gives.

    # The real question, last it stands.
    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append("Answer briefly in one or two sentences.")
    else:
        parts.append(_render_mcq(question.text, question.options))
        parts.append("Answer with ONLY the letter (A, B, C, or D).")

    return "\n".join(parts)


def _cot_v1(question: Question, context: list[RetrievedDoc] | None) -> str:
    """Brief step-by-step reasoning, then 'Answer: X', this asks for.

    For arithmetic especially, helpful it is -- compute before committing, the model can.
    The parser keys on the 'Answer: X' marker (its highest-confidence pattern, that is).
    """
    parts: list[str] = []

    # Context block, only when evidence exists, prepend we do.
    if context:
        parts.append(_build_context_block(context))

    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append("Think briefly, then answer in one or two sentences.")
    else:
        parts.append(_render_mcq(question.text, question.options))
        parts.append(
            "Think step by step briefly (one or two short sentences). "
            "Then on a new line, write your final choice as 'Answer: X', "
            "where X is one of A, B, C, or D."
        )

    return "\n".join(parts)


def _cot_v2(question: Question, context: list[RetrievedDoc] | None) -> str:
    """cot_v1 + an OPTION-MATCHING check + a HARD brevity cap, this is.

    Two motivating failures, this prompt answers:
      * run #7, qid 6702 (the OPTION-MATCHING slip): on a t-test MCQ the model REASONED correctly
        (df=17, ±2.110 -- option C's content) yet wrote 'Answer: B', whose only flaw was 'df=18'.
        B and C SHARED the conclusion ('do not reject'); the model matched the conclusion alone and
        never cross-checked the buried number against its own work. So: verify the chosen option
        matches EVERY computed detail -- not the conclusion only.
      * run #9, qid 6706 (the TRUNCATION loss): the model's set-up was CORRECT (z=-0.524/-0.842,
        the two equations) but it wrote ~5 paragraphs of LaTeX and hit the 256-token cap BEFORE the
        'Answer:' line -- so the parser fell back to a blind 'A'. At ~11 tok/s the 256 cap ≈ the 25s
        wall, so MORE tokens would only time out. The cure is FEWER tokens to the answer: cap the
        steps, BAN LaTeX (the token hog -- \\frac/\\mu/$...$ tripled the length), and DEMAND the
        'Answer:' line always be reached.
    For open questions, identical to cot_v1 it stays (no options).
    """
    parts: list[str] = []

    # Context block, only when evidence exists, prepend we do.
    if context:
        parts.append(_build_context_block(context))

    if question.qtype == QuestionType.OPEN or not question.options:
        parts.append(f"Question: {question.text.strip()}")
        parts.append("Think briefly, then answer in one or two sentences.")
    else:
        parts.append(_render_mcq(question.text, question.options))
        parts.append(
            "Solve in AT MOST 3 very short steps. Plain numbers ONLY -- NO LaTeX, no \\frac, "
            "no \\mu/\\sigma, no $...$; write 'mu'/'sigma' as words and keep each step under ~12 "
            "words. When two options share the same conclusion, pick the one whose numbers (values, "
            "signs, degrees of freedom) match your result EXACTLY -- not just the conclusion. You "
            "MUST end on a new line with 'Answer: X' (X = A, B, C, or D) -- always reach that line."
        )

    return "\n".join(parts)


# --- Strategy registry ---

# A name -> builder function, this dict is.
# To add a new strategy, register it here you must (the human record, in memory/prompts.md it lives).
_REGISTRY: dict[str, object] = {
    "zero_shot_v1": _zero_shot_v1,
    "few_shot_v1": _few_shot_v1,
    "cot_v1": _cot_v1,
    "cot_v2": _cot_v2,
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
