from agentgrade.config import CheckConfig
from agentgrade.evaluators import run_check
from agentgrade.rewards import compute_reward
from agentgrade.trace import AgentStep, AgentTrace


def _trace(latency_ms: int = 100, cost: float = 0.01) -> AgentTrace:
    t = AgentTrace(test_name="t")
    t.add_step(AgentStep(agent_name="A", input="", output="", latency_ms=latency_ms, cost_usd=cost))
    return t


def test_contains_pass_and_fail():
    trace = _trace()
    ok = run_check("hello world", trace, CheckConfig(type="contains", value="world", weight=1.0))
    assert ok.passed and ok.score == 1.0
    bad = run_check("hello world", trace, CheckConfig(type="contains", value="mars", weight=1.0))
    assert not bad.passed and bad.score == 0.0


def test_not_contains():
    trace = _trace()
    ok = run_check("safe", trace, CheckConfig(type="not_contains", value="danger", weight=1.0))
    assert ok.passed


def test_regex():
    trace = _trace()
    ok = run_check("use torchrun now", trace, CheckConfig(type="regex", value="torchrun|foo", weight=1.0))
    assert ok.passed


def test_exact_match():
    trace = _trace()
    ok = run_check(" abc ", trace, CheckConfig(type="exact_match", value="abc", weight=1.0))
    assert ok.passed


def test_max_latency():
    trace = _trace(latency_ms=5000)
    ok = run_check("", trace, CheckConfig(type="max_latency", seconds=30, weight=1.0))
    assert ok.passed
    bad = run_check("", trace, CheckConfig(type="max_latency", seconds=1, weight=1.0))
    assert not bad.passed


def test_max_cost():
    trace = _trace(cost=0.5)
    ok = run_check("", trace, CheckConfig(type="max_cost", usd=1.0, weight=1.0))
    assert ok.passed
    bad = run_check("", trace, CheckConfig(type="max_cost", usd=0.1, weight=1.0))
    assert not bad.passed


def test_unknown_check_fails_safely():
    trace = _trace()
    res = run_check("x", trace, CheckConfig(type="nope", weight=1.0))
    assert not res.passed


def test_compute_reward_weighted_average():
    from agentgrade.trace import EvaluationResult

    evals = [
        EvaluationResult(check_name="a", passed=True, score=1.0, weight=0.2, message=""),
        EvaluationResult(check_name="b", passed=False, score=0.0, weight=0.2, message=""),
        EvaluationResult(check_name="c", passed=True, score=1.0, weight=0.6, message=""),
    ]
    assert abs(compute_reward(evals) - 0.8) < 1e-9


def test_compute_reward_empty():
    assert compute_reward([]) == 0.0
