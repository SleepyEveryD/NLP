"""The QA pipeline -- the orchestrator the notebook calls, this is.

Question -> classify -> (retrieve?) -> prompt -> generate -> (tool?) -> parse -> Prediction.
The single seam between 'the system' and 'the experiment harness', it forms.
Against the 30s budget, every stage is timed.
"""
from __future__ import annotations

import re

from classify.classifier import QuestionClassifier
from inference.engine import LLMEngine
from prompting.builder import PromptBuilder
from schemas import Prediction, Question
from utils.timing import LatencyGuard, stopwatch


class QAPipeline:
    """Compose the modules, this does -- but own none of their internals, it must.

    Injected dependencies, all of them are -- so swap, mock and benchmark each part, we can.
    """

    def __init__(
        self,
        engine: LLMEngine,
        prompt_builder: PromptBuilder,
        classifier: QuestionClassifier | None = None,
        retriever=None,
        tools=None,
        latency_budget_s: float = 30.0,
    ):
        self.engine = engine
        self.prompt_builder = prompt_builder
        self.classifier = classifier
        self.retriever = retriever
        self.tools = tools
        self.latency_budget_s = latency_budget_s

    def answer(self, question: Question) -> Prediction:
        """One question in, one Prediction out -- the whole system, in a single call this is.

        Seven stages in sequence, the pipeline runs:
          classify -> retrieve -> prompt -> generate -> tool (Phase 3) -> parse.
        Optional via DI, every collaborator is -- dormant they stay when None they are.
        Crash-safe the whole body is -- live play must always submit something, even on error.
        """
        # The latency guard and per-stage breakdown, created at the top they must be.
        guard = LatencyGuard(self.latency_budget_s)
        breakdown: dict[str, float] = {}

        # Retrieval state, initialised here -- dormant until a retriever exists it stays.
        retrieval_used: bool = False
        docs = None
        retrieved_doc_ids: list[str] = []

        # Tool state, initialised here -- Phase 3 hook awaits, for now None it is.
        tool_used: str | None = None

        # Raw generation output -- empty string a safe default is.
        raw: str = ""

        try:
            # --- Stage: classify ---
            # The question, enriched with topic/type/language -- only when a classifier present is.
            with stopwatch(breakdown, "classify"):
                if self.classifier:
                    question = self.classifier.classify(question)

            # --- Stage: retrieve ---
            # Evidence, fetched only when a retriever exists AND retrieval needed it is.
            with stopwatch(breakdown, "retrieve"):
                if self.retriever and (
                    not self.classifier
                    or self.classifier.needs_retrieval(question)
                ):
                    # Raw document chunks, retrieved here they are -- never an LLM answer (D-008).
                    docs = self.retriever.retrieve(question)
                    retrieval_used = True
                    # The doc_id of each retrieved chunk, collected for the EvalRecord it is.
                    retrieved_doc_ids = [
                        doc.doc_id for doc in docs if hasattr(doc, "doc_id")
                    ]

            # --- Stage: prompt ---
            # The user-turn string, built from question and optional evidence context it is.
            with stopwatch(breakdown, "prompt"):
                prompt: str = self.prompt_builder.build(question, docs)

            # --- Stage: generate ---
            # The model, called with engine defaults (Phase 1 -- max_new_tokens/temperature not overridden).
            with stopwatch(breakdown, "generate"):
                raw = self.engine.generate(prompt)

            # --- Stage: tool ---
            # Phase 3 hook: the calculator loop lives here -- dormant for now, tool_used stays None.
            # When tools present AND the classifier says arithmetic needed -- dispatch the tool we will.
            with stopwatch(breakdown, "tool"):
                if self.tools and self.classifier and self.classifier.needs_calculator(question):
                    # Phase 3: the JSON-tool dispatch pattern (D-013), implement here we must.
                    # For now, a placeholder -- does nothing, tool_used remains None.
                    pass  # TODO (Phase 3): the calculator tool loop, wire here you must.

            # --- Stage: parse ---
            # The chosen letter and confidence, extracted from messy model output they are.
            with stopwatch(breakdown, "parse"):
                ans, conf = QAPipeline.parse_answer(raw, question)

        except Exception as e:
            # A crash, caught here it is -- the live game must never go silent.
            # The first available option letter, a safe fallback answer it makes.
            fallback_ans = sorted(question.options)[0] if question.options else ""
            return Prediction(
                qid=question.qid,
                answer=fallback_ans,
                confidence=0.0,
                raw_output=raw,
                model=getattr(self.engine, "name", ""),
                prompt_strategy=getattr(self.prompt_builder, "strategy", ""),
                retrieval_used=retrieval_used,
                retrieved_doc_ids=retrieved_doc_ids,
                tool_used=tool_used,
                latency_s=guard.elapsed(),
                tokens_in=getattr(self.engine, "last_tokens_in", 0),
                tokens_out=getattr(self.engine, "last_tokens_out", 0),
                error=str(e),
            )

        # The completed Prediction, assembled and returned it is.
        # Note: breakdown not stored on Prediction (schema is frozen); the runner re-times at its level.
        return Prediction(
            qid=question.qid,
            answer=ans,
            confidence=conf,
            raw_output=raw,
            model=getattr(self.engine, "name", ""),
            prompt_strategy=getattr(self.prompt_builder, "strategy", ""),
            retrieval_used=retrieval_used,
            retrieved_doc_ids=retrieved_doc_ids,
            tool_used=tool_used,
            latency_s=guard.elapsed(),
            tokens_in=getattr(self.engine, "last_tokens_in", 0),
            tokens_out=getattr(self.engine, "last_tokens_out", 0),
            error=None,
        )

    @staticmethod
    def parse_answer(raw_output: str, question: Question) -> tuple[str, float]:
        """The model's text -> (answer, confidence). Robust to chatter, this must be.

        From the raw generation, a single valid option letter we extract.
        A rich set of patterns -- 'Answer: B', '(C)', 'Option D', bare 'a' -- handled they all are.
        Restricted to the letters actually present in question.options, the result is.
        For an open question with no options, the cleaned raw text and low confidence returned are.
        """
        # The valid letters, from the question's options determined they are.
        valid_letters: set[str] = set(question.options.keys()) if question.options else set()

        # An open question with no options -- return cleaned text and low confidence we do.
        if not valid_letters:
            return raw_output.strip(), 0.3

        # The text to search, stripped of leading/trailing whitespace it is.
        text = raw_output.strip()

        # Helper: a candidate letter, validated against valid_letters it is.
        def _valid(letter: str) -> str | None:
            # Uppercased and checked against the allowed set, the letter is.
            up = letter.upper()
            return up if up in valid_letters else None

        # --- Pattern group 1: Explicit answer markers (highest confidence) ---
        # Patterns like "Answer: B", "The answer is C", "answer is: D." caught here they are.
        _EXPLICIT = re.compile(
            r"\b(?:the\s+)?answer\s+(?:is\s*:?\s*|:\s*)([A-Da-d])\b",
            re.IGNORECASE,
        )
        m = _EXPLICIT.search(text)
        if m:
            letter = _valid(m.group(1))
            if letter:
                return letter, 1.0

        # --- Pattern group 2: Parenthesised letter -- "(B)" or "(b)" ---
        # A single letter in parentheses, a strong answer signal it is.
        _PAREN = re.compile(r"\(([A-Da-d])\)", re.IGNORECASE)
        m = _PAREN.search(text)
        if m:
            letter = _valid(m.group(1))
            if letter:
                return letter, 1.0

        # --- Pattern group 3: "Option X" or "option X" ---
        # The word 'Option' followed by a letter, matched here it is.
        _OPTION_WORD = re.compile(r"\boption\s+([A-Da-d])\b", re.IGNORECASE)
        m = _OPTION_WORD.search(text)
        if m:
            letter = _valid(m.group(1))
            if letter:
                return letter, 1.0

        # --- Pattern group 4: Letter followed by closing paren -- "B)" or "b)" ---
        # Common in enumerated list continuations, this form is.
        _LETTER_PAREN = re.compile(r"\b([A-Da-d])\)", re.IGNORECASE)
        m = _LETTER_PAREN.search(text)
        if m:
            letter = _valid(m.group(1))
            if letter:
                return letter, 0.9

        # --- Pattern group 5: Letter followed by option text (e.g. "B Rome") ---
        # The model echoes "B Rome" or "C: Paris" -- the letter we take, the text we ignore.
        if valid_letters and question.options:
            for letter_key in sorted(valid_letters):
                # The option text for this letter, escaped for regex it is.
                opt_text = re.escape(question.options[letter_key][:20].strip())
                _ECHO = re.compile(
                    rf"\b({re.escape(letter_key)})[):\s]+{opt_text}",
                    re.IGNORECASE,
                )
                m = _ECHO.search(text)
                if m:
                    letter = _valid(m.group(1))
                    if letter:
                        return letter, 0.9

        # --- Pattern group 6: Standalone letter on its own line or at start of text ---
        # A bare "A" or "b" as the sole token on a line -- the cleanest single-letter output it is.
        _STANDALONE_LINE = re.compile(
            r"(?:^|\n)\s*([A-Da-d])\s*(?:\n|$)",
            re.IGNORECASE,
        )
        m = _STANDALONE_LINE.search(text)
        if m:
            letter = _valid(m.group(1))
            if letter:
                return letter, 1.0

        # --- Pattern group 7: Any isolated letter bounded by non-alpha chars ---
        # The first letter in the text that is surrounded by word boundaries, grabbed we do.
        # Lower confidence: context around it, we cannot verify.
        _ISOLATED = re.compile(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", re.IGNORECASE)
        matches = _ISOLATED.findall(text)
        # Only valid candidates, kept they are.
        valid_matches = [_valid(ltr) for ltr in matches if _valid(ltr)]
        if valid_matches:
            # The first valid match, returned with medium confidence -- ambiguous it may be.
            return valid_matches[0], 0.5

        # --- Fallback: no letter found -- the first available option, submitted we must ---
        # Live play must always answer something; silent submission, allowed it is not.
        fallback = sorted(valid_letters)[0]
        return fallback, 0.0
