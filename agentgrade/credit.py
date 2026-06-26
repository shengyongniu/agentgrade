"""Credit assignment: map failures back to the agent/tool that caused them.

This is the headline feature of agentgrade. Given the failed checks and the
execution trace, deterministic heuristics decide which named agent, tool, or
dimension (performance/cost) most likely caused each failure.

Credit is *inferred from the trace* by default: for required-content checks we
blame the most downstream agent that still failed to include the value; for
forbidden-content checks we blame the earliest agent that introduced it; for
latency/cost we blame the slowest/most-expensive step's agent. An explicit
``agent_name`` on the check always overrides inference (back-compat).
"""

from __future__ import annotations

import re

from .config import CheckConfig
from .trace import AgentStep, AgentTrace, EvaluationResult

REQUIRED_CONTENT_TYPES = {"contains", "regex", "exact_match"}

MAX_REGEX_INPUT_CHARS = 1_000_000
"""Cap on output length fed to ``re.search`` (mirrors evaluators)."""


def _check_value(check: CheckConfig) -> str:
    return str(getattr(check, "value", getattr(check, "pattern", "")))


def _output_satisfies(check_type: str, output: str, value: str) -> bool:
    """Whether ``output`` already contains the required value/pattern."""

    if not value:
        return True
    if check_type == "regex":
        try:
            return re.search(value, output[:MAX_REGEX_INPUT_CHARS]) is not None
        except re.error:
            # An invalid pattern can't be satisfied; treat as not-satisfied
            # rather than crashing credit inference.
            return False
    if check_type == "exact_match":
        return output.strip() == value.strip()
    return value in output


def _infer_required_culprit(check: CheckConfig, trace: AgentTrace) -> str | None:
    """Blame the most downstream agent that failed to include the required value.

    We walk the steps in order and remember the last step whose ``output`` does
    not satisfy the check. That is the final agent that had the output in hand
    and still shipped it missing the required content.
    """

    value = _check_value(check)
    culprit: str | None = None
    for step in trace.steps:
        if not _output_satisfies(check.type, step.output, value):
            culprit = step.agent_name
    return culprit


def _infer_forbidden_culprit(check: CheckConfig, trace: AgentTrace) -> str | None:
    """Blame the earliest agent whose output introduced the forbidden value."""

    value = _check_value(check)
    if not value:
        return None
    for step in trace.steps:
        if value in step.output:
            return step.agent_name
    return None


def _slowest_step(trace: AgentTrace) -> AgentStep | None:
    if not trace.steps:
        return None
    return max(trace.steps, key=lambda s: s.latency_ms)


def _costliest_step(trace: AgentTrace) -> AgentStep | None:
    steps = [s for s in trace.steps if s.cost_usd is not None]
    if not steps:
        return None
    return max(steps, key=lambda s: s.cost_usd or 0.0)


def assign_credit(
    evaluations: list[EvaluationResult],
    checks: list[CheckConfig],
    trace: AgentTrace,
) -> dict[str, list[str]]:
    """Return a mapping of ``culprit -> [reasons]`` for failed checks.

    Heuristics:
    - Required-content (``contains``/``regex``/``exact_match``) failures are
      attributed to the last agent whose output still lacked the value.
    - ``not_contains`` failures are attributed to the first agent that
      introduced the forbidden value.
    - Latency/cost failures are attributed to the slowest/most-expensive step's
      agent (with a performance/cost-flavoured reason).
    - A tool error in the trace is blamed on that tool/agent.
    - An explicit ``agent_name`` on a check always overrides inference.
    """

    candidates: dict[str, list[str]] = {}

    def add(culprit: str, reason: str) -> None:
        candidates.setdefault(culprit, [])
        if reason not in candidates[culprit]:
            candidates[culprit].append(reason)

    final_agent = trace.agent_names()[-1] if trace.agent_names() else "FinalAgent"

    for evaluation, check in zip(evaluations, checks):
        if evaluation.passed:
            continue

        if check.agent_name:
            add(check.agent_name, evaluation.message)
            continue

        if check.type in REQUIRED_CONTENT_TYPES:
            culprit = _infer_required_culprit(check, trace) or final_agent
            add(culprit, f"{evaluation.message} (inferred: last agent to leave it out)")
        elif check.type == "not_contains":
            culprit = _infer_forbidden_culprit(check, trace) or final_agent
            add(culprit, f"{evaluation.message} (inferred: first agent to introduce it)")
        elif check.type == "max_latency":
            step = _slowest_step(trace)
            if step is not None:
                add(
                    step.agent_name,
                    f"Performance: {evaluation.message} "
                    f"(slowest step: {step.agent_name} at {step.latency_ms}ms)",
                )
            else:
                add("Performance", evaluation.message)
        elif check.type == "max_cost":
            step = _costliest_step(trace)
            if step is not None:
                add(
                    step.agent_name,
                    f"Cost: {evaluation.message} "
                    f"(most expensive step: {step.agent_name} at ${step.cost_usd:.4f})",
                )
            else:
                add("Cost", evaluation.message)
        else:
            add(final_agent, evaluation.message)

    for step in trace.steps:
        if step.error:
            culprit = step.tool_name or step.agent_name
            add(culprit, f"tool/agent error: {step.error}")

    return candidates


def format_credit(candidates: dict[str, list[str]]) -> list[str]:
    """Render credit assignment as human-readable lines."""

    lines: list[str] = []
    for culprit, reasons in candidates.items():
        for reason in reasons:
            lines.append(f"{culprit}: {reason}")
    return lines
