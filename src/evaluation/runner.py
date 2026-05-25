"""Benchmark runner. A config + a dataset -> a logged run, this turns.

Over the questions loop, call the pipeline, log every EvalRecord -- the reproducible experiment loop, it is.
One run = one config; the config itself, into meta.json we write.
"""
from __future__ import annotations

import datetime
import subprocess

from agent.pipeline import QAPipeline
from config import RunConfig
from evaluation.logger import ExperimentLogger
from schemas import EvalRecord


class BenchmarkRunner:
    def __init__(self, pipeline: QAPipeline, config: RunConfig, log_root: str = "experiments/runs"):
        self.pipeline = pipeline
        self.config = config
        self.log_root = log_root

    def run(self, questions: list) -> str:
        """Over a dataset run the pipeline and log. The run path, return it does."""

        # --- Meta dict: best-effort probes, crash-safe they must be ---

        # The config's plain dict, the foundation it is.
        meta = self.config.to_dict()

        # The current git commit, best-effort we fetch.
        git_commit = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Stripped of whitespace, the commit hash is.
                git_commit = result.stdout.strip()
        except Exception:
            # Silently, None we keep -- crash we must not.
            git_commit = None

        # The CUDA device name, best-effort we probe.
        hardware = None
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                # The GPU's name, from device 0 it comes.
                hardware = torch.cuda.get_device_name(0)
        except Exception:
            # No torch, or no CUDA -- None is fine, it is.
            hardware = None

        # The wall-clock timestamp in ISO format, readable it stays.
        meta["git_commit"] = git_commit
        meta["hardware"] = hardware
        meta["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"

        # --- Logger open; in a finally block, close it we do ---
        logger = ExperimentLogger(self.log_root, self.config.run_id, meta=meta)
        try:
            # Over each question, the pipeline we call.
            for i, question in enumerate(questions, start=1):
                # Progress: one line per question, print we do.
                print(f"[{i}/{len(questions)}] qid={question.qid} ...")

                # The pipeline's answer, from the question we extract.
                pred = self.pipeline.answer(question)

                # The correctness flag: None for live play, bool for dev sets.
                correct: bool | None
                if question.gold is not None:
                    # Case-insensitive letter comparison, fair it is.
                    correct = pred.answer.strip().upper() == question.gold.strip().upper()
                else:
                    # Live game: the truth unknown, None it stays.
                    correct = None

                # qtype: a str the EvalRecord wants; an enum the Question may hold.
                qtype_str: str
                try:
                    # If an Enum it is, .value gives the string we need.
                    qtype_str = question.qtype.value
                except AttributeError:
                    # Already a string, or something else -- str() is safe.
                    qtype_str = str(question.qtype)

                # The EvalRecord, fully populated it must be.
                record = EvalRecord(
                    run_id=self.config.run_id,
                    timestamp=EvalRecord.now(),
                    qid=question.qid,
                    question_text=question.text,
                    qtype=qtype_str,
                    topic=question.topic,
                    level=question.level,
                    language=question.language,
                    model=pred.model,
                    prompt_strategy=pred.prompt_strategy,
                    retrieval_used=pred.retrieval_used,
                    retrieved_doc_ids=pred.retrieved_doc_ids,
                    tool_used=pred.tool_used,
                    predicted_answer=pred.answer,
                    gold_answer=question.gold,
                    correct=correct,
                    confidence=pred.confidence,
                    latency_s=pred.latency_s,
                    # No breakdown the Prediction carries; empty dict we pass.
                    latency_breakdown={},
                    tokens_in=pred.tokens_in,
                    tokens_out=pred.tokens_out,
                    raw_output=pred.raw_output,
                    error=pred.error,
                )

                # One line, one prediction -- logged it is.
                logger.log(record)

        finally:
            # Always close the logger; a crash mid-loop, data we must not lose.
            logger.close()

        # The run directory path, as a string we return.
        return str(logger.dir)
