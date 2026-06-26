"""Reward computation: weighted average of check scores."""

from __future__ import annotations

from .trace import EvaluationResult


def compute_reward(evaluations: list[EvaluationResult]) -> float:
    """Return ``sum(score * weight) / sum(weights)``.

    Returns ``0.0`` when there are no checks or all weights are zero so an
    empty test never silently "passes".
    """

    total_weight = sum(e.weight for e in evaluations)
    if total_weight <= 0:
        return 0.0
    weighted = sum(e.score * e.weight for e in evaluations)
    return weighted / total_weight
