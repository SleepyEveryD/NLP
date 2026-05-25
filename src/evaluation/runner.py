"""Benchmark runner. A config + a dataset -> a logged run, this turns.

Over the questions loop, call the pipeline, log every EvalRecord -- the reproducible experiment loop, it is.
One run = one config; the config itself, into meta.json we write.
"""
from __future__ import annotations

from agent.pipeline import QAPipeline
from config import RunConfig


class BenchmarkRunner:
    def __init__(self, pipeline: QAPipeline, config: RunConfig, log_root: str = "experiments/runs"):
        self.pipeline = pipeline
        self.config = config
        self.log_root = log_root

    def run(self, questions: list) -> str:
        """Over a dataset run the pipeline and log. The run path, return it does."""
        # Phase 1: for each question -> pipeline.answer -> build EvalRecord -> logger.log, here.
        raise NotImplementedError("Phase 1: the loop + ExperimentLogger wiring, implement here you must.")
