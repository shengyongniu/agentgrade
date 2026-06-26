"""LangGraph adapter for agentgrade.

This module turns a LangGraph graph's streamed node updates into agentgrade's
``(final_output, AgentTrace)`` shape so an existing graph can be dropped into a
test entrypoint callable.

It is intentionally **duck-typed**: there is no top-level ``langgraph`` or
``langchain_core`` import, so this module imports cleanly even when those
packages are not installed. Messages are recognised by their class name
(``AIMessage`` / ``ToolMessage`` / ``HumanMessage``) and accessed via
``getattr`` for ``content`` / ``tool_calls``, which keeps the adapter framework
friendly.

The adapter drives ``graph.stream(input, config=config, stream_mode="updates")``
and emits one :class:`~agentgrade.trace.AgentStep` per node update (one per tool
call when a node's ``AIMessage`` carries several). Per-node ``latency_ms`` is the
wall-clock time between streamed updates; for graphs with parallel/fan-out nodes
those per-step latencies are therefore approximate.
"""

from __future__ import annotations

import time
from typing import Any

from ..trace import AgentStep, AgentTrace

_MESSAGE_CLASS_NAMES = {"AIMessage", "ToolMessage", "HumanMessage"}


def _class_name(obj: Any) -> str:
    return type(obj).__name__


def _is_message(obj: Any) -> bool:
    return _class_name(obj) in _MESSAGE_CLASS_NAMES


def _msg_text(message: Any) -> str:
    """Return the text content of a LangChain-style message.

    Handles both plain string ``content`` and list-of-content-blocks content
    (each block is a dict with a ``text`` key or a plain string).
    """

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _messages_from_update(update: Any, messages_key: str) -> list[Any]:
    """Extract the list of messages from a node's state update."""

    if isinstance(update, dict):
        value = update.get(messages_key)
        if isinstance(value, list):
            return value
        if value is not None:
            return [value]
    return []


def _last_message_of(update: Any, messages_key: str, class_name: str) -> Any | None:
    for message in reversed(_messages_from_update(update, messages_key)):
        if _class_name(message) == class_name:
            return message
    return None


def _stringify_update(update: Any) -> str:
    if isinstance(update, dict):
        return str(update)
    return str(update)


def _tool_calls(message: Any) -> list[dict]:
    calls = getattr(message, "tool_calls", None)
    if not calls:
        return []
    normalised: list[dict] = []
    for call in calls:
        if isinstance(call, dict):
            normalised.append(call)
        else:
            normalised.append(
                {
                    "name": getattr(call, "name", None),
                    "args": getattr(call, "args", None),
                }
            )
    return normalised


def _last_message(update: Any, messages_key: str) -> Any | None:
    messages = _messages_from_update(update, messages_key)
    return messages[-1] if messages else None


def _node_output(update: Any, messages_key: str) -> str:
    ai = _last_message_of(update, messages_key, "AIMessage")
    if ai is not None:
        text = _msg_text(ai)
        if text:
            return text
    last = _last_message(update, messages_key)
    if last is not None:
        text = _msg_text(last)
        if text:
            return text
    return _stringify_update(update)


def _final_output(last_update: Any, messages_key: str, output_key: str | None) -> str:
    if output_key is not None and isinstance(last_update, dict):
        if output_key in last_update:
            value = last_update[output_key]
            if _is_message(value):
                return _msg_text(value)
            return str(value)
    return _node_output(last_update, messages_key)


def trace_langgraph(
    graph: Any,
    input: Any,
    *,
    test_name: str = "",
    config: dict | None = None,
    messages_key: str = "messages",
    output_key: str | None = None,
) -> tuple[str, AgentTrace]:
    """Run a LangGraph graph and return ``(final_output, AgentTrace)``.

    Drives ``graph.stream(input, config=config, stream_mode="updates")``. Each
    streamed event is ``{node_name: state_update}``; one :class:`AgentStep` is
    emitted per node update (one per tool call when a node's ``AIMessage`` has
    several). The first step seeds ``input=str(input)``.

    A mid-run exception is captured on ``trace.error`` and the partial trace is
    returned alongside the last reconstructed output.
    """

    trace = AgentTrace(test_name=test_name)
    final_output = ""
    last_update: Any = None
    last_clock = time.perf_counter()
    seeded_input = False

    try:
        for event in graph.stream(input, config=config, stream_mode="updates"):
            if not isinstance(event, dict):
                continue
            for node_name, update in event.items():
                now = time.perf_counter()
                latency_ms = int((now - last_clock) * 1000)
                last_clock = now
                last_update = update

                step_input = str(input) if not seeded_input else ""
                seeded_input = True

                output = _node_output(update, messages_key)
                final_output = output

                ai = _last_message_of(update, messages_key, "AIMessage")
                tool_msg = _last_message_of(update, messages_key, "ToolMessage")
                tool_output = _msg_text(tool_msg) if tool_msg is not None else None
                tool_error = (
                    tool_output
                    if tool_msg is not None
                    and getattr(tool_msg, "status", None) == "error"
                    else None
                )

                calls = _tool_calls(ai) if ai is not None else []
                if len(calls) > 1:
                    for index, call in enumerate(calls):
                        trace.add_step(
                            AgentStep(
                                agent_name=node_name,
                                input=step_input if index == 0 else "",
                                output=output,
                                tool_name=call.get("name"),
                                tool_input=call.get("args"),
                                tool_output=tool_output,
                                error=tool_error,
                                latency_ms=latency_ms if index == 0 else 0,
                            )
                        )
                else:
                    call = calls[0] if calls else None
                    trace.add_step(
                        AgentStep(
                            agent_name=node_name,
                            input=step_input,
                            output=output,
                            tool_name=call.get("name") if call else None,
                            tool_input=call.get("args") if call else None,
                            tool_output=tool_output,
                            error=tool_error,
                            latency_ms=latency_ms,
                        )
                    )
    except Exception as exc:  # noqa: BLE001 - capture graph crashes onto the trace
        trace.error = f"{type(exc).__name__}: {exc}"
        trace.final_output = final_output
        return final_output, trace

    if last_update is not None:
        final_output = _final_output(last_update, messages_key, output_key)
    trace.final_output = final_output
    return final_output, trace
