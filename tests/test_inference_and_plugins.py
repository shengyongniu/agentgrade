from pathlib import Path

import pytest

from agentgrade.config import CheckConfig, load_config
from agentgrade.credit import assign_credit
from agentgrade.evaluators import (
    EVALUATORS,
    evaluator,
    register_evaluator,
    run_check,
)
from agentgrade.runner import run_suite
from agentgrade.trace import AgentStep, AgentTrace, EvaluationResult


def _two_step_trace(coder_out: str, critic_out: str) -> AgentTrace:
    t = AgentTrace(test_name="t")
    t.add_step(AgentStep(agent_name="DraftAgent", input="", output=coder_out, latency_ms=10, cost_usd=0.001))
    t.add_step(AgentStep(agent_name="RefineAgent", input="", output=critic_out, latency_ms=20, cost_usd=0.002))
    return t


def _fail(check_name: str, message: str, weight: float = 1.0) -> EvaluationResult:
    return EvaluationResult(check_name=check_name, passed=False, score=0.0, weight=weight, message=message)


def test_required_content_blames_downstream_agent_without_annotation():
    trace = _two_step_trace(coder_out="has keyword KW here", critic_out="rewrote without it")
    checks = [CheckConfig(type="contains", value="KW", weight=1.0)]
    evals = [_fail("contains:KW", "output is missing 'KW'")]
    credit = assign_credit(evals, checks, trace)
    assert "RefineAgent" in credit
    assert "DraftAgent" not in credit


def test_not_contains_blames_introducing_agent():
    trace = _two_step_trace(coder_out="contains BAD token", critic_out="still has BAD token")
    checks = [CheckConfig(type="not_contains", value="BAD", weight=1.0)]
    evals = [_fail("not_contains:BAD", "output unexpectedly contains 'BAD'")]
    credit = assign_credit(evals, checks, trace)
    assert "DraftAgent" in credit
    assert "RefineAgent" not in credit


def test_latency_blames_slowest_step_agent():
    trace = _two_step_trace(coder_out="", critic_out="")  # RefineAgent is slower (20ms)
    checks = [CheckConfig(type="max_latency", seconds=0, weight=1.0)]
    evals = [_fail("max_latency", "latency 30ms exceeds 0ms")]
    credit = assign_credit(evals, checks, trace)
    assert "RefineAgent" in credit
    assert any("Performance" in r and "RefineAgent" in r for r in credit["RefineAgent"])


def test_cost_blames_most_expensive_step_agent():
    trace = _two_step_trace(coder_out="", critic_out="")  # RefineAgent costs more (0.002)
    checks = [CheckConfig(type="max_cost", usd=0, weight=1.0)]
    evals = [_fail("max_cost", "cost $0.0030 exceeds $0.00")]
    credit = assign_credit(evals, checks, trace)
    assert "RefineAgent" in credit
    assert any("Cost" in r for r in credit["RefineAgent"])


def test_agent_name_override_wins_over_inference():
    trace = _two_step_trace(coder_out="has KW", critic_out="dropped it")
    checks = [CheckConfig(type="contains", value="KW", weight=1.0, agent_name="ManuallyBlamed")]
    evals = [_fail("contains:KW", "output is missing 'KW'")]
    credit = assign_credit(evals, checks, trace)
    assert "ManuallyBlamed" in credit
    assert credit["ManuallyBlamed"] == ["output is missing 'KW'"]
    assert "RefineAgent" not in credit


def test_register_evaluator_and_run_check():
    def eval_min_length(output, trace, check):
        minimum = int(getattr(check, "value", 0) or 0)
        passed = len(output) >= minimum
        return EvaluationResult(
            check_name="min_length",
            passed=passed,
            score=1.0 if passed else 0.0,
            weight=check.weight,
            message="ok" if passed else "too short",
        )

    register_evaluator("min_length_test", eval_min_length)
    assert "min_length_test" in EVALUATORS
    trace = AgentTrace(test_name="t")
    ok = run_check("abcdef", trace, CheckConfig(type="min_length_test", value=3, weight=1.0))
    assert ok.passed
    bad = run_check("ab", trace, CheckConfig(type="min_length_test", value=3, weight=1.0))
    assert not bad.passed


def test_evaluator_decorator_registers():
    @evaluator("always_pass_test")
    def _always_pass(output, trace, check):
        return EvaluationResult(
            check_name="always_pass", passed=True, score=1.0, weight=check.weight, message="ok"
        )

    assert "always_pass_test" in EVALUATORS
    trace = AgentTrace(test_name="t")
    res = run_check("anything", trace, CheckConfig(type="always_pass_test", weight=1.0))
    assert res.passed


def test_inferred_agent_example_infers_credit(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    cfg = load_config(repo_root / "examples" / "inferred_agent" / "agentgrade.yaml")
    results = run_suite(cfg)

    result = results[0]
    assert not result.passed
    assert set(result.trace.agent_names()) == {"DraftAgent", "RefineAgent"}
    assert "RefineAgent" in result.credit_assignment
    assert any("summary" in r for r in result.credit_assignment["RefineAgent"])
    assert "DraftAgent" in result.credit_assignment
    assert any("TODO" in r for r in result.credit_assignment["DraftAgent"])
    # No check in this example declares agent_name.
    for test in cfg.tests:
        for check in test.checks:
            assert check.agent_name is None


def test_config_driven_plugin_loading_registers_min_length(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    EVALUATORS.pop("min_length", None)
    cfg = load_config(repo_root / "examples" / "inferred_agent" / "agentgrade.yaml")
    results = run_suite(cfg)

    assert "min_length" in EVALUATORS
    min_length_evals = [
        ev for ev in results[0].evaluations if ev.check_name.startswith("min_length")
    ]
    assert min_length_evals and min_length_evals[0].passed
