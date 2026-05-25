"""Core data contracts shared across all modules.

The single source of truth for the shapes that flow through the pipeline, this file is.
Change a shape here you must, when an interface evolves -- so drift between modules, prevent we do.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QuestionType(str, Enum):
    # The kinds of questions, these are.
    MCQ = "mcq"          # Multiple choice, one correct letter it has.
    OPEN = "open"        # Free-text short answer, this is.
    UNKNOWN = "unknown"


@dataclass
class Question:
    """A single quiz question, as received from the game or a dev set.

    From the game API or an offline dataset, populated this is.
    """
    qid: str
    text: str
    options: dict[str, str] = field(default_factory=dict)  # {"A": "...", ...}; empty for open, it is.
    option_ids: dict[str, int] = field(default_factory=dict)  # {"A": 101, ...}; the server's Option.id per letter.
    qtype: QuestionType = QuestionType.MCQ
    level: Optional[int] = None        # Difficulty rung 1..15, the game may tell us.
    topic: Optional[str] = None        # Filled by the classifier later, it is.
    language: Optional[str] = None     # "en" / "it", detected or known.
    gold: Optional[str] = None         # The truth -- known only for dev sets, None in live play it is.


@dataclass
class RetrievedDoc:
    # A raw chunk of evidence, retrieved this was -- never an LLM answer, it must be.
    doc_id: str
    text: str
    source: str          # URL or corpus name, the origin is.
    score: float = 0.0


@dataclass
class Prediction:
    """What a single model produced for one question.

    The atomic unit of a prediction, this is. Over many of these, the ensemble votes.
    """
    qid: str
    answer: str                        # "A".."D" for MCQ, or text for open.
    confidence: float = 0.0            # 0..1, self-reported or derived it is.
    raw_output: str = ""               # The unparsed generation -- for debugging, kept it is.
    model: str = ""
    prompt_strategy: str = ""
    retrieval_used: bool = False
    retrieved_doc_ids: list[str] = field(default_factory=list)
    tool_used: Optional[str] = None    # e.g. "calculator"; None if no tool called, it is.
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None


@dataclass
class EvalRecord:
    """One row of the benchmark log -- everything the rubric asks us to track, it holds.

    Append-only to JSONL these go. Reproduce any experiment from them, we can.
    """
    run_id: str
    timestamp: float
    qid: str
    question_text: str
    qtype: str
    topic: Optional[str]
    level: Optional[int]
    language: Optional[str]
    model: str
    prompt_strategy: str
    retrieval_used: bool
    retrieved_doc_ids: list[str]
    tool_used: Optional[str]
    predicted_answer: str
    gold_answer: Optional[str]
    correct: Optional[bool]            # None when the gold is unknown (live game), it is.
    confidence: float
    latency_s: float
    latency_breakdown: dict[str, float]
    tokens_in: int
    tokens_out: int
    raw_output: str
    error: Optional[str] = None

    @staticmethod
    def now() -> float:
        # The current wall-clock time, this gives.
        return time.time()
