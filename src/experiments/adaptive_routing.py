"""The adaptive prompt-routing experiment -- four conditions, one dataset, one log, this runs.

The research question: for a small model (Qwen2.5-7B), does routing each question to a prompt chosen
for its REASONING SHAPE beat forcing one universal prompt on everything? And the deeper one: WHEN does
explicit reasoning help, and when does it hurt?

To answer cleanly, ONE variable we isolate -- the prompt strategy. So this harness deliberately does
NOT retrieve and does NOT call the calculator (the live `QAPipeline` does both); every condition sees
the SAME question with only the PROMPT changed. Four conditions, the ablation they form:

    A_universal     -- one universal prompt for all (the production `few_shot_v1`).
    B_generic_cot   -- always plain "think step by step".
    C_structured    -- always structured enumeration.
    D_adaptive      -- the ReasoningRouter picks per question.

Every prediction one `ExperimentRecord` becomes (strategy chosen, router verdict, correctness, latency,
tokens, reasoning length, raw output) -- written append-style to one JSONL, so `analysis.py` reads it
back without re-running the model. Reproducible the science stays.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from agent.pipeline import QAPipeline
from classify.reasoning_router import ReasoningRouter
from prompting.builder import PromptBuilder
from schemas import Question


# --------------------------------------------------------------------------- #
# The dataset loader -- reads the reasoning eval set AND its gold category labels.
# (`evaluation.dataset.load_questions` drops the `category` field; here we keep it.)
# --------------------------------------------------------------------------- #

def load_reasoning_eval(
    path: str = "data/reasoning_eval.jsonl",
) -> tuple[list[Question], dict[str, str]]:
    """The eval set -> (questions, qid->true-category). Both, the experiment needs.

    The `category` field is the HAND-LABELLED reasoning shape -- the ground truth against which the
    router's classification is scored, and the key the simulated engine's skill table reads.
    """
    questions: list[Question] = []
    categories: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = Question(
                qid=rec["qid"],
                text=rec["text"],
                options=rec.get("options", {}),
                qtype=rec.get("qtype", "mcq"),  # str ok; QAPipeline only reads .options for MCQ.
                level=rec.get("level"),
                topic=rec.get("topic"),
                language=rec.get("language"),
                gold=rec.get("gold"),
            )
            questions.append(q)
            categories[q.qid] = rec.get("category", "commonsense")
    return questions, categories


# --------------------------------------------------------------------------- #
# The conditions -- the four arms of the ablation.
# --------------------------------------------------------------------------- #

@dataclass
class Condition:
    """One arm of the ablation. `strategy=None` means ADAPTIVE (the router picks)."""
    name: str
    description: str
    strategy: Optional[str]   # a fixed strategy name, or None for adaptive routing.


DEFAULT_CONDITIONS: list[Condition] = [
    Condition("A_universal", "Single universal prompt (production few_shot_v1)", "few_shot_v1"),
    Condition("B_generic_cot", "Always generic chain-of-thought", "generic_cot"),
    Condition("C_structured", "Always structured enumeration", "structured_enumeration_cot"),
    Condition("D_adaptive", "Adaptive prompt routing (ReasoningRouter)", None),
]


# --------------------------------------------------------------------------- #
# The record -- everything the rubric asks us to save, one row holds.
# --------------------------------------------------------------------------- #

@dataclass
class ExperimentRecord:
    timestamp: float
    condition: str               # which arm (A_universal / ... / D_adaptive).
    qid: str
    question_text: str
    true_category: str           # the hand-labelled reasoning shape (ground truth).
    routed_category: str         # what the ReasoningClassifier inferred (for routing-accuracy analysis).
    routing_evidence: str        # WHY it inferred that -- the cue that fired.
    strategy: str                # the prompt strategy ACTUALLY used this row.
    predicted_answer: str
    gold_answer: Optional[str]
    correct: Optional[bool]
    confidence: float
    latency_s: float
    tokens_in: int
    tokens_out: int
    reasoning_chars: int         # length of the raw generation in characters -- the "how verbose?" metric.
    reasoning_lines: int         # and in non-empty lines.
    raw_output: str
    error: Optional[str] = None
    options: dict[str, str] = field(default_factory=dict)


class AdaptiveRoutingExperiment:
    """Run the four conditions over the eval set; one JSONL of `ExperimentRecord`, the output it is.

    Injected the engine is (real `TransformersEngine` on Colab, or `SimulatedReasoningEngine` locally),
    so the SAME harness both a GPU run and a CPU smoke test drives.
    """

    def __init__(
        self,
        engine,
        router: ReasoningRouter | None = None,
        conditions: list[Condition] | None = None,
        out_dir: str = "experiments/adaptive_routing",
    ):
        self.engine = engine
        self.router = router or ReasoningRouter()
        self.conditions = conditions or DEFAULT_CONDITIONS
        self.out_dir = Path(out_dir)
        # PromptBuilders, cached per strategy -- cheap, but reuse them we may as well.
        self._builders: dict[str, PromptBuilder] = {}

    def _builder(self, strategy: str) -> PromptBuilder:
        if strategy not in self._builders:
            self._builders[strategy] = PromptBuilder(strategy)
        return self._builders[strategy]

    def _answer_one(
        self, question: Question, true_category: str, condition: Condition
    ) -> ExperimentRecord:
        """One question under one condition -> one record. Isolated: no retrieval, no tools."""
        # The router ALWAYS runs (even in fixed conditions) -- so its verdict we can log and later
        # score its classification accuracy, regardless of whether this arm acted on it.
        signal, routed_strategy = self.router.route(question)

        # Fixed strategy when the condition names one; else the router's pick (adaptive).
        strategy = condition.strategy if condition.strategy is not None else routed_strategy

        prompt = self._builder(strategy).build(question, None)

        raw = ""
        error: Optional[str] = None
        start = time.perf_counter()
        try:
            raw = self.engine.generate(prompt)
        except Exception as e:   # A generation crash -- the row we still write, the error noted.
            error = f"{type(e).__name__}: {e}"
        elapsed = time.perf_counter() - start

        # The simulated engine reports its OWN (deterministic) latency; the real engine does not, so
        # for it we fall back to the measured wall-clock.
        latency = float(getattr(self.engine, "last_latency_s", None) or elapsed)

        ans, conf = QAPipeline.parse_answer(raw, question)
        gold = question.gold
        correct = (ans.strip().upper() == gold.strip().upper()) if gold else None

        nonblank_lines = [ln for ln in raw.splitlines() if ln.strip()]
        return ExperimentRecord(
            timestamp=time.time(),
            condition=condition.name,
            qid=question.qid,
            question_text=question.text,
            true_category=true_category,
            routed_category=signal.category.value,
            routing_evidence=signal.evidence,
            strategy=strategy,
            predicted_answer=ans,
            gold_answer=gold,
            correct=correct,
            confidence=conf,
            latency_s=latency,
            tokens_in=int(getattr(self.engine, "last_tokens_in", 0)),
            tokens_out=int(getattr(self.engine, "last_tokens_out", 0)),
            reasoning_chars=len(raw),
            reasoning_lines=len(nonblank_lines),
            raw_output=raw,
            error=error,
            options=question.options,
        )

    def run(
        self,
        questions: list[Question],
        categories: dict[str, str],
        *,
        verbose: bool = True,
    ) -> str:
        """All conditions x all questions, run and log. The path to the combined JSONL, return it does."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        records_path = self.out_dir / "records.jsonl"

        total = len(self.conditions) * len(questions)
        done = 0
        with records_path.open("w", encoding="utf-8") as fh:
            for condition in self.conditions:
                if verbose:
                    print(f"\n=== Condition {condition.name}: {condition.description} ===")
                for question in questions:
                    rec = self._answer_one(question, categories.get(question.qid, "commonsense"), condition)
                    fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    fh.flush()
                    done += 1
                    if verbose:
                        mark = "ok " if rec.correct else ("ERR" if rec.error else "  X")
                        print(
                            f"  [{done}/{total}] {rec.qid:<7} {condition.name:<14} "
                            f"strat={rec.strategy:<27} -> {rec.predicted_answer} "
                            f"(gold {rec.gold_answer}) [{mark}]"
                        )

        # A tiny meta sidecar -- the engine and condition set, for the record.
        (self.out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "engine": getattr(self.engine, "name", "unknown"),
                    "n_questions": len(questions),
                    "conditions": [asdict(c) for c in self.conditions],
                    "routing_policy": {
                        c.name: (c.strategy or "ADAPTIVE") for c in self.conditions
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if verbose:
            print(f"\nWrote {done} records -> {records_path}")
        return str(records_path)
