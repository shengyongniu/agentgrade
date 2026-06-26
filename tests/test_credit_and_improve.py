from agentgrade.config import CheckConfig
from agentgrade.credit import assign_credit, format_credit
from agentgrade.improve import build_patch_markdown, suggest_patches
from agentgrade.trace import AgentStep, AgentTrace, EvaluationResult


def _trace_with_agents() -> AgentTrace:
    t = AgentTrace(test_name="t")
    t.add_step(AgentStep(agent_name="CoderAgent", input="", output="", latency_ms=10))
    t.add_step(AgentStep(agent_name="CriticAgent", input="", output="", latency_ms=10))
    return t


def test_credit_blames_named_agent():
    trace = _trace_with_agents()
    checks = [CheckConfig(type="contains", value="DistributedSampler", weight=0.2, agent_name="CoderAgent")]
    evals = [EvaluationResult(check_name="contains:DistributedSampler", passed=False, score=0.0, weight=0.2, message="missing")]
    credit = assign_credit(evals, checks, trace)
    assert "CoderAgent" in credit


def test_credit_latency_and_cost_dimensions():
    trace = _trace_with_agents()
    checks = [
        CheckConfig(type="max_latency", seconds=0, weight=0.1),
        CheckConfig(type="max_cost", usd=0, weight=0.1),
    ]
    evals = [
        EvaluationResult(check_name="max_latency", passed=False, score=0.0, weight=0.1, message="slow"),
        EvaluationResult(check_name="max_cost", passed=False, score=0.0, weight=0.1, message="expensive"),
    ]
    credit = assign_credit(evals, checks, trace)
    flat = format_credit(credit)
    assert any("Performance" in line for line in flat)
    assert any("Cost" in line for line in flat)


def test_credit_tool_error():
    t = AgentTrace(test_name="t")
    t.add_step(AgentStep(agent_name="CoderAgent", input="", output="", tool_name="codegen", error="boom"))
    credit = assign_credit([], [], t)
    assert "codegen" in credit


def test_format_credit():
    lines = format_credit({"CoderAgent": ["missing X"]})
    assert lines == ["CoderAgent: missing X"]


def test_suggest_patches():
    checks = [CheckConfig(type="contains", value="DistributedSampler", weight=0.2, agent_name="CoderAgent")]
    evals = [EvaluationResult(check_name="contains:DistributedSampler", passed=False, score=0.0, weight=0.2, message="missing")]
    patches = suggest_patches(evals, checks)
    assert patches and "CoderAgent" in patches[0] and "DistributedSampler" in patches[0]


def test_build_patch_markdown():
    checks = [CheckConfig(type="contains", value="DistributedSampler", weight=0.2, agent_name="CoderAgent")]
    evals = [EvaluationResult(check_name="contains:DistributedSampler", passed=False, score=0.0, weight=0.2, message="missing")]
    md = build_patch_markdown("ddp", evals, checks)
    assert "CoderAgent" in md and "DistributedSampler" in md and "```diff" in md
