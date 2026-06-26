"""Offline unit tests for the LangGraph adapter.

These use FAKE graph/message objects (no ``langgraph`` install required) to
prove the streamed-update -> AgentTrace mapping.
"""

from agentgrade.integrations.langgraph import _msg_text, trace_langgraph


class AIMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class ToolMessage:
    def __init__(self, content, status="success"):
        self.content = content
        self.status = status


class HumanMessage:
    def __init__(self, content):
        self.content = content


class FakeGraph:
    """A stub exposing the ``.stream`` surface the adapter relies on."""

    def __init__(self, events):
        self._events = events
        self.calls = []

    def stream(self, input, config=None, stream_mode="updates"):
        self.calls.append((input, config, stream_mode))
        for event in self._events:
            yield event


def test_msg_text_handles_str_and_blocks():
    assert _msg_text(AIMessage("hello")) == "hello"
    blocks = [{"type": "text", "text": "a"}, "b", {"text": "c"}]
    assert _msg_text(AIMessage(blocks)) == "abc"


def test_basic_two_node_mapping():
    events = [
        {"planner": {"messages": [AIMessage("drafted a plan")]}},
        {"executor": {"messages": [AIMessage("final answer")]}},
    ]
    graph = FakeGraph(events)

    final_output, trace = trace_langgraph(
        graph, {"messages": [("user", "do it")]}, test_name="lg"
    )

    assert trace.test_name == "lg"
    assert len(trace.steps) == 2
    assert trace.agent_names() == ["planner", "executor"]
    assert trace.steps[0].output == "drafted a plan"
    assert trace.steps[0].input == str({"messages": [("user", "do it")]})
    assert trace.steps[1].input == ""
    assert final_output == "final answer"
    assert trace.final_output == "final answer"
    assert graph.calls[0][2] == "updates"


def test_tool_call_and_tool_message_fields():
    events = [
        {
            "agent": {
                "messages": [
                    AIMessage(
                        "calling search",
                        tool_calls=[{"name": "search", "args": {"q": "ddp"}}],
                    )
                ]
            }
        },
        {"tools": {"messages": [ToolMessage("search results here")]}},
    ]
    graph = FakeGraph(events)

    final_output, trace = trace_langgraph(graph, {"messages": []}, test_name="lg")

    assert len(trace.steps) == 2
    agent_step = trace.steps[0]
    assert agent_step.tool_name == "search"
    assert agent_step.tool_input == {"q": "ddp"}
    tool_step = trace.steps[1]
    assert tool_step.tool_output == "search results here"
    assert tool_step.error is None
    assert final_output == "search results here"


def test_tool_error_status_sets_error():
    events = [
        {"tools": {"messages": [ToolMessage("boom", status="error")]}},
    ]
    final_output, trace = trace_langgraph(FakeGraph(events), {"messages": []})
    assert trace.steps[0].error == "boom"


def test_multiple_tool_calls_emit_one_step_each():
    events = [
        {
            "agent": {
                "messages": [
                    AIMessage(
                        "parallel tools",
                        tool_calls=[
                            {"name": "search", "args": {"q": "a"}},
                            {"name": "lookup", "args": {"q": "b"}},
                        ],
                    )
                ]
            }
        },
    ]
    final_output, trace = trace_langgraph(FakeGraph(events), {"messages": []})

    assert len(trace.steps) == 2
    assert [s.tool_name for s in trace.steps] == ["search", "lookup"]
    assert all(s.agent_name == "agent" for s in trace.steps)


def test_output_key_reconstructs_final_output():
    events = [
        {"finish": {"messages": [AIMessage("ignored")], "result": "the result"}},
    ]
    final_output, trace = trace_langgraph(
        FakeGraph(events), {"messages": []}, output_key="result"
    )
    assert final_output == "the result"


def test_exception_mid_stream_is_captured():
    class BoomGraph:
        def stream(self, input, config=None, stream_mode="updates"):
            yield {"planner": {"messages": [AIMessage("partial")]}}
            raise RuntimeError("graph exploded")

    final_output, trace = trace_langgraph(BoomGraph(), {"messages": []})

    assert len(trace.steps) == 1
    assert final_output == "partial"
    assert trace.error is not None
    assert "graph exploded" in trace.error


def test_non_message_update_stringifies():
    events = [{"node": {"messages": []}}]
    final_output, trace = trace_langgraph(FakeGraph(events), {"messages": []})
    assert len(trace.steps) == 1
    assert trace.steps[0].output == str({"messages": []})
