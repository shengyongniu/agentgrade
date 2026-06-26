"""A scripted Draft -> Refine pipeline that demonstrates *inferred* credit.

Unlike ``examples/simple_agent``, the checks in this example's ``agentgrade.yaml``
carry **no** ``agent_name`` annotations. agentgrade therefore has to infer the
culprit purely from the trace:

- The ``DraftAgent`` writes a first-pass summarizer function.
- The ``RefineAgent`` rewrites it, adding input validation, but in doing so it
  drops the required ``return`` of a structured ``{"summary": ...}`` payload —
  so the final output is missing the keyword ``"summary"``. Because the draft
  *did* contain it and the refined output does not, the most downstream agent
  to leave it out is ``RefineAgent``.
- The ``DraftAgent`` also leaves a ``TODO`` marker that the refine step keeps,
  tripping a ``not_contains`` check — the first agent to introduce it is
  ``DraftAgent``.

Fully deterministic and offline; no API keys.
"""

from __future__ import annotations

from agentgrade.integrations import TraceRecorder
from agentgrade.trace import AgentTrace


def _draft_agent(task: str) -> str:
    return (
        "def summarize(text):\n"
        "    # TODO: handle empty input\n"
        "    words = text.split()\n"
        "    head = ' '.join(words[:20])\n"
        '    return {"summary": head, "length": len(words)}\n'
    )


def _refine_agent(task: str, draft: str) -> str:
    return (
        "def summarize(text):\n"
        "    if not text:\n"
        "        raise ValueError('text must be non-empty')\n"
        "    # TODO: handle empty input\n"
        "    words = text.split()\n"
        "    head = ' '.join(words[:20])\n"
        "    return head\n"
    )


def run_agent(task: str) -> tuple[str, AgentTrace]:
    rec = TraceRecorder(test_name="summarizer")

    draft = _draft_agent(task)
    rec.step(
        "DraftAgent",
        input=task,
        output=draft,
        tool_name="codegen",
        tool_input={"task": task},
        tool_output="drafted summarizer",
        latency_ms=80,
        cost_usd=0.002,
    )

    refined = _refine_agent(task, draft)
    rec.step(
        "RefineAgent",
        input=draft,
        output=refined,
        tool_name="refactor",
        tool_input={"draft_len": len(draft)},
        tool_output="added validation, simplified return",
        latency_ms=300,
        cost_usd=0.006,
    )

    return refined, rec.finalize(final_output=refined)
