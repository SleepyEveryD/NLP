"""The QA pipeline -- the orchestrator the notebook calls, this is.

Question -> classify -> (retrieve?) -> prompt -> generate -> (tool?) -> parse -> Prediction.
The single seam between 'the system' and 'the experiment harness', it forms.
Against the 30s budget, every stage is timed.
"""
from __future__ import annotations

import json
import re

from agent.voting import majority_vote
from classify.classifier import QuestionClassifier
from inference.engine import LLMEngine
from prompting.builder import PromptBuilder
from schemas import Prediction, Question
from utils.timing import LatencyGuard, stopwatch


def _render_mcq_block(question: Question) -> str:
    """A Question -> the 'Question/A)/B)...' block, for the tool prompts this renders (A-D order)."""
    lines = [f"Question: {question.text.strip()}"]
    for letter in ("A", "B", "C", "D"):
        if letter in question.options:
            lines.append(f"{letter}) {question.options[letter]}")
    return "\n".join(lines)


def _extract_first_json(text: str) -> dict | None:
    """The first balanced {...} object in the text, parse it we do -- chatter around it, tolerate we must.

    Brace-counting (not a greedy regex), so trailing prose after the JSON, never swallow it we do.
    None it returns when no parseable object there is.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        break  # This candidate failed -- the next '{' try we do.
        start = text.find("{", start + 1)
    return None


class QAPipeline:
    """Compose the modules, this does -- but own none of their internals, it must.

    Injected dependencies, all of them are -- so swap, mock and benchmark each part, we can.
    """

    # Below this many seconds left, no time for another ~CoT sampling pass there is -- stop, and the
    # votes we already have, keep them. (cot_v1 a sample ~4s costs; 5s of margin, comfortably safe it is.)
    _SC_MIN_MARGIN_S: float = 5.0

    def __init__(
        self,
        engine: LLMEngine,
        prompt_builder: PromptBuilder,
        classifier: QuestionClassifier | None = None,
        retriever=None,
        tools=None,
        self_consistency_n: int = 1,
        self_consistency_temperature: float = 0.7,
        latency_budget_s: float = 30.0,
    ):
        self.engine = engine
        self.prompt_builder = prompt_builder
        self.classifier = classifier
        self.retriever = retriever
        self.tools = tools
        # Self-consistency: N>1 -> sample N CoT chains and majority-vote (Phase 5 / the Maths bet).
        # n=1 (the default) -> a single greedy pass: ZERO behaviour change for every existing pipeline.
        self.self_consistency_n = max(1, int(self_consistency_n))
        self.self_consistency_temperature = self_consistency_temperature
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

        # The parsed answer/confidence -- set by the parse stage (single pass) OR the vote (self-consistency).
        ans: str = ""
        conf: float = 0.0

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
            # n=1 (default): ONE pass, engine defaults (max_new_tokens/temperature not overridden).
            # n>1 (self-consistency): N SAMPLED chains, parsed and majority-voted -- for the hard
            # reasoning of Maths, more robust than one greedy pass it is (the occasional good chain,
            # the vote surfaces it; the confidence the VOTE SHARE becomes -- a real calibration signal).
            with stopwatch(breakdown, "generate"):
                if self.self_consistency_n > 1:
                    ans, conf, raw = self._self_consistency_answer(prompt, question, guard)
                else:
                    raw = self.engine.generate(prompt)

            # --- Stage: tool ---
            # The calculator loop (D-013): only when tools present AND the classifier says arithmetic
            # is needed. The model a JSON call emits, run it we do, then with the number re-answer it does.
            # On any failure the plain-generate `raw` above stands -- crash-safe the maths path stays.
            with stopwatch(breakdown, "tool"):
                # Skip the tool when retrieval already fired (D-NEWS): the calculator re-answer prompt
                # carries only the bare MCQ -- NOT the retrieved docs -- so letting it override an
                # evidence-grounded answer would silently discard the web snippets (the News-clobber bug).
                # Skip too under self-consistency: the N CoT chains compute their arithmetic INLINE and
                # vote on it -- the tool's single context-free re-answer would fight the vote (and a wrong
                # SET-UP the calculator cannot save anyway -- our Maths logs, repeatedly this they showed).
                if (
                    self.self_consistency_n <= 1
                    and self.tools
                    and self.classifier
                    and not retrieval_used
                    and self.classifier.needs_calculator(question)
                ):
                    tool_raw, used = self._run_calculator_tool(question, guard)
                    if tool_raw is not None:
                        raw = tool_raw      # The tool-grounded answer, the plain one it replaces.
                        tool_used = used

            # --- Stage: parse ---
            # Single-pass mode parses the raw here; self-consistency already voted (ans/conf set above).
            with stopwatch(breakdown, "parse"):
                if self.self_consistency_n <= 1:
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

    def _self_consistency_answer(
        self, prompt: str, question: Question, guard: LatencyGuard
    ) -> tuple[str, float, str]:
        """N sampled chains of the SAME prompt -> a majority vote (the self-consistency technique).

        Each sample at `self_consistency_temperature` we draw (so the chains DIVERGE -- a vote of N
        identical greedy passes, meaningless it would be). Each we parse to a letter; over those
        letters `majority_vote` decides. Budget-aware: a further sample we skip once under
        `_SC_MIN_MARGIN_S` the guard falls -- but at LEAST one we always keep (the loop's first pass,
        the margin check never blocks).

        Returns: (answer, confidence, raw_output) -- the confidence the VOTE SHARE is (winners/total),
        and the raw the most-confident winning chain's text (a representative the log keeps).
        """
        candidates: list[Prediction] = []
        for _ in range(self.self_consistency_n):
            # Time for another ~CoT pass, is there? Below the margin -> the votes we have, settle for them.
            if candidates and guard.remaining() < self._SC_MIN_MARGIN_S:
                break
            sample = self.engine.generate(
                prompt, temperature=self.self_consistency_temperature
            )
            a, c = QAPipeline.parse_answer(sample, question)
            candidates.append(
                Prediction(qid=question.qid, answer=a, confidence=c, raw_output=sample)
            )
        winner = majority_vote(candidates)  # >=1 candidate always -- the empty-list guard never trips.
        return winner.answer, winner.confidence, winner.raw_output

    def _run_calculator_tool(
        self, question: Question, guard: LatencyGuard
    ) -> tuple[str | None, str | None]:
        """The JSON-tool dispatch (D-013), single-turn: the model a calculator call emits, run it we do,
        then with the number it re-answers. The course's LangChain JSON pattern, over our LOCAL engine.

        Returns:
            (final_raw, "calculator") -- when the model a valid call made AND the re-answer succeeded.
            (None, None)              -- otherwise, so the plain-generate answer untouched it stays.
        Crash-safe entirely: a bad JSON, a forbidden expression, a calc error -- all swallowed, None returned.
        """
        calc = self.tools.get("calculator") if isinstance(self.tools, dict) else None
        if calc is None:
            return None, None
        # Two more generations the tool costs -- below the wall too close, then skip it we do.
        if guard.remaining() < 4.0:
            return None, None

        try:
            mcq = _render_mcq_block(question)

            # Turn 1: the tool offer. ONLY-JSON or ONLY-a-letter, the model we ask for.
            offer = (
                f"{mcq}\n\n"
                "A calculator tool you may use for arithmetic. To call it, reply with ONLY this JSON "
                "and nothing else:\n"
                '{"name": "calculator", "arguments": {"expression": "<arithmetic expression>"}}\n'
                "Pure arithmetic the expression must be (digits and + - * / ** % ( ) only).\n"
                "No calculation needed? Then reply with ONLY the letter (A, B, C, or D)."
            )
            first = self.engine.generate(offer)
            call = _extract_first_json(first)
            if not call or call.get("name") != "calculator":
                return None, None  # No tool call -- the plain answer, let it stand.

            args = call.get("arguments") or {}
            expr = str(args.get("expression", "")).strip()
            if not expr:
                return None, None
            result = calc(expr)  # safe-AST calculate(); on anything non-arithmetic, it raises.

            # Turn 2: the result fed back, the final letter we now demand.
            followup = (
                f"{mcq}\n\n"
                f"A calculator computed: {expr} = {result}\n"
                "Using this result, answer with ONLY the letter (A, B, C, or D)."
            )
            final = self.engine.generate(followup)
            return final, "calculator"
        except Exception:
            # Any slip -- the maths path must never crash the turn; the plain answer, fall back to it we do.
            return None, None

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
        # Patterns like "Answer: B", "Answer:B", "The answer is C", "answer is: D" caught here they are.
        # The old regex a SPACE before the colon demanded -- so "Answer: B" (cot's own format!) it MISSED,
        # and the prose fell to pattern-7, the article "a" as "A" grabbing (P2-bug). Two clean branches now:
        # a colon straight after "answer" (space optional), OR " is" then an optional colon. Bare "answer a
        # question", never the article it snags -- a separator (':' or 'is'), each branch demands.
        _EXPLICIT = re.compile(
            r"\b(?:the\s+)?answer\b(?:\s*:\s*|\s+is\s*:?\s*)([A-Da-d])\b",
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
