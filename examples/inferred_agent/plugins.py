"""Example custom evaluator registered via the agentgrade plugin API.

This module is referenced from ``examples/inferred_agent/agentgrade.yaml`` via the
top-level ``plugins:`` key. agentgrade imports ``register`` before running checks,
which adds a deterministic, offline ``min_length`` check type.
"""

from __future__ import annotations

from agentgrade.config import CheckConfig
from agentgrade.evaluators import EvaluationResult, register_evaluator
from agentgrade.trace import AgentTrace


def eval_min_length(output: str, trace: AgentTrace, check: CheckConfig) -> EvaluationResult:
    minimum = int(getattr(check, "min_chars", getattr(check, "value", 0)) or 0)
    actual = len(output)
    passed = actual >= minimum
    return EvaluationResult(
        check_name=f"min_length:{minimum}",
        passed=passed,
        score=1.0 if passed else 0.0,
        weight=check.weight,
        message=(
            f"output length {actual} >= {minimum}"
            if passed
            else f"output length {actual} below minimum {minimum}"
        ),
    )


def register() -> None:
    register_evaluator("min_length", eval_min_length)
