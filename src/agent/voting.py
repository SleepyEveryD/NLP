"""Ensemble voting. Many Predictions into one, this combines (Phase 5, latency-permitting).

Majority vote, or confidence-weighted -- more reliable answers, the rubric hopes for.
Only if the 30s budget the extra forward passes allows, run this we will.
"""
from __future__ import annotations

from schemas import Prediction


def majority_vote(predictions: list[Prediction]) -> Prediction:
    # The most-voted answer wins; ties, by mean confidence break we do.
    raise NotImplementedError("Phase 5: majority / confidence-weighted voting, implement here you must.")
