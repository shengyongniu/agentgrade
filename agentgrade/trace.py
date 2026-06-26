"""Core data models for agentgrade traces and results."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentStep(BaseModel):
    """A single step taken by a named agent within a workflow."""

    agent_name: str
    input: str
    output: str
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_output: str | None = None
    error: str | None = None
    latency_ms: int = 0
    cost_usd: float | None = None
    timestamp: str = Field(default_factory=_now_iso)


class AgentTrace(BaseModel):
    """A full execution trace produced by an agent workflow for one test."""

    test_name: str
    steps: list[AgentStep] = Field(default_factory=list)
    final_output: str = ""
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0
    error: str | None = None

    def add_step(self, step: AgentStep) -> AgentStep:
        self.steps.append(step)
        self.total_latency_ms += step.latency_ms
        if step.cost_usd:
            self.total_cost_usd += step.cost_usd
        return step

    def agent_names(self) -> list[str]:
        seen: list[str] = []
        for step in self.steps:
            if step.agent_name not in seen:
                seen.append(step.agent_name)
        return seen


class EvaluationResult(BaseModel):
    """Outcome of a single check/evaluator."""

    check_name: str
    passed: bool
    score: float
    weight: float
    message: str


class TestResult(BaseModel):
    """Aggregate result for one test case."""

    name: str
    input: str
    output: str
    reward: float
    passed: bool
    evaluations: list[EvaluationResult] = Field(default_factory=list)
    trace: AgentTrace
    credit_assignment: dict = Field(default_factory=dict)
    suggested_patches: list[str] = Field(default_factory=list)
