"""Generic, framework-agnostic trace recorder.

Use this helper inside any plain Python agent callable to record steps for
each named agent/tool. It works with no vendor lock-in and no API keys.
"""

from __future__ import annotations

import time

from ..trace import AgentStep, AgentTrace


class TraceRecorder:
    """Accumulates :class:`AgentStep` objects into an :class:`AgentTrace`.

    Example::

        rec = TraceRecorder(test_name="my_test")
        out = rec.step("CoderAgent", input=task, output=code, cost_usd=0.01)
        return code, rec.finalize(final_output=code)
    """

    def __init__(self, test_name: str = "") -> None:
        self.trace = AgentTrace(test_name=test_name)

    def step(
        self,
        agent_name: str,
        *,
        input: str = "",
        output: str = "",
        tool_name: str | None = None,
        tool_input: dict | None = None,
        tool_output: str | None = None,
        error: str | None = None,
        latency_ms: int = 0,
        cost_usd: float | None = None,
    ) -> AgentStep:
        step = AgentStep(
            agent_name=agent_name,
            input=input,
            output=output,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            error=error,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )
        return self.trace.add_step(step)

    def finalize(self, final_output: str) -> AgentTrace:
        self.trace.final_output = final_output
        return self.trace
