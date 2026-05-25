"""Metrics & breakdowns. The investigation's evidence, these compute.

Accuracy overall and by topic/level/type, latency percentiles, overconfidence (confidence vs correctness).
From the JSONL logs all of it derives -- so re-analyse without re-running models, we can.
"""
from __future__ import annotations

import pandas as pd


def load_runs(run_dirs: list[str]) -> "pd.DataFrame":
    # Each records.jsonl into a tidy DataFrame, read it we do.
    raise NotImplementedError("Phase 1: read records.jsonl into a DataFrame, implement here you must.")


def accuracy_by(df: "pd.DataFrame", column: str) -> "pd.DataFrame":
    # Grouped accuracy (by topic / level / model), this gives.
    raise NotImplementedError


def latency_summary(df: "pd.DataFrame") -> dict:
    # Median, p95, max, and budget violations -- the 30s story, this tells.
    raise NotImplementedError
