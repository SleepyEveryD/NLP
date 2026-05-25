"""The experiment logger. Append-only JSONL, every prediction becomes.

Model, prompt, latency, accuracy, retrieval and tool usage -- all of it the rubric demands, so log we do.
Reproducible the science stays, only when written down it is.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from schemas import EvalRecord


class ExperimentLogger:
    """One run, one JSONL file -- plus a metadata sidecar, written it is.

    Usage:
        logger = ExperimentLogger("experiments/runs", run_id="baseline_qwen", meta={...})
        logger.log(eval_record)
        logger.close()
    """

    def __init__(self, root: str, run_id: str, meta: Optional[dict] = None):
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.dir / "records.jsonl"
        self._fh = self.records_path.open("a", encoding="utf-8")
        if meta:
            # The run's fingerprint -- config, model, git commit, hardware -- here it lives.
            (self.dir / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    def log(self, record: EvalRecord) -> None:
        # One line, one prediction. Flush immediately we do -- so a Colab crash, nothing it loses.
        self._fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
