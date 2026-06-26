"""agentgrade — pytest for multi-agent systems."""

from __future__ import annotations

from .trace import AgentStep, AgentTrace, EvaluationResult, TestResult

__version__ = "0.1.0"

__all__ = [
    "AgentStep",
    "AgentTrace",
    "EvaluationResult",
    "TestResult",
    "__version__",
]
