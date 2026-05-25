"""The QA pipeline -- the orchestrator the notebook calls, this is.

Question -> classify -> (retrieve?) -> prompt -> generate -> (tool?) -> parse -> Prediction.
The single seam between 'the system' and 'the experiment harness', it forms.
Against the 30s budget, every stage is timed.
"""
from __future__ import annotations

from classify.classifier import QuestionClassifier
from inference.engine import LLMEngine
from prompting.builder import PromptBuilder
from schemas import Prediction, Question


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
        """One question in, one Prediction out -- the whole system, in a single call this is."""
        # Phase 1: wire classify -> (retrieve) -> prompt -> generate -> (tool) -> parse, here.
        raise NotImplementedError("Phase 1: the orchestration, implement here you must.")

    @staticmethod
    def parse_answer(raw_output: str, question: Question) -> tuple[str, float]:
        """The model's text -> (answer, confidence). Robust to chatter, this must be."""
        # A standalone A/B/C/D, find we must -- lowercase and 'Option X', handle them too.
        raise NotImplementedError
