from pathlib import Path

from agentgrade.config import load_config
from agentgrade.runner import run_suite


def test_simple_agent_fails_with_credit(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    cfg = load_config(repo_root / "examples" / "simple_agent" / "agentgrade.yaml")
    results = run_suite(cfg)

    assert len(results) >= 1
    result = results[0]
    assert not result.passed
    assert 0.4 <= result.reward <= 0.61
    # Credit assignment must name a specific agent.
    assert "CoderAgent" in result.credit_assignment
    assert any("DistributedSampler" in r for r in result.credit_assignment["CoderAgent"])
    # Multiple distinct agents recorded in the trace.
    assert set(result.trace.agent_names()) == {"CoderAgent", "CriticAgent"}


def test_ddp_coding_agent_passes(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    cfg = load_config(repo_root / "examples" / "ddp_coding_agent" / "agentgrade.yaml")
    results = run_suite(cfg)

    assert len(results) >= 1
    assert results[0].passed
    assert results[0].reward >= 0.99
