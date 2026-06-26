"""Tests for the security/robustness hardening fixes (F1–F12)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agentgrade.config import CheckConfig, Settings, load_config
from agentgrade.config import TestConfig as _TestConfig
from agentgrade.credit import _output_satisfies, assign_credit
from agentgrade.evaluators import eval_max_cost, eval_max_latency, run_check
from agentgrade.runner import (
    FixtureContainmentError,
    MalformedReplayFixture,
    _fixture_path,
    _load_entrypoint,
    load_plugins,
    run_suite,
    run_test,
)
from agentgrade.trace import AgentStep, AgentTrace, EvaluationResult

REPO_ROOT = Path(__file__).resolve().parents[1]


def _trace(latency_ms: int = 100, cost: float = 0.01) -> AgentTrace:
    t = AgentTrace(test_name="t")
    t.add_step(AgentStep(agent_name="A", input="", output="", latency_ms=latency_ms, cost_usd=cost))
    return t


# --- F1: import warning -----------------------------------------------------


def test_entrypoint_emits_import_warning(monkeypatch, capsys):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.delenv("AGENTGRADE_NO_WARN", raising=False)
    _load_entrypoint("examples.simple_agent.agent:run_agent")
    err = capsys.readouterr().err
    assert "agentgrade: importing agent entrypoint 'examples.simple_agent.agent'" in err


def test_import_warning_silenced_by_env(monkeypatch, capsys):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("AGENTGRADE_NO_WARN", "1")
    _load_entrypoint("examples.simple_agent.agent:run_agent")
    assert capsys.readouterr().err == ""


def test_plugins_warning_lists_modules(monkeypatch, capsys):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.delenv("AGENTGRADE_NO_WARN", raising=False)
    load_plugins(["examples.inferred_agent.plugins:register"])
    err = capsys.readouterr().err
    assert "importing plugin module(s)" in err
    assert "examples.inferred_agent.plugins" in err


# --- F2: cwd appended, not prepended ---------------------------------------


def test_cwd_appended_not_prepended(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    cwd = str(REPO_ROOT)
    # Remove any existing entry to observe insertion position.
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != cwd])
    _load_entrypoint("examples.simple_agent.agent:run_agent")
    assert cwd in sys.path
    # Appended -> not at index 0, so it cannot shadow installed packages.
    assert sys.path[0] != cwd


def test_examples_still_import_with_append(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    func = _load_entrypoint("examples.simple_agent.agent:run_agent")
    assert callable(func)


# --- F3: unit_tests subprocess hardening ------------------------------------


def test_unit_tests_passing_file(tmp_path):
    test_file = tmp_path / "test_trivial.py"
    test_file.write_text("def test_ok():\n    assert True\n")
    res = run_check("", _trace(), CheckConfig(type="unit_tests", path=str(test_file), weight=1.0))
    assert res.passed
    assert res.score == 1.0


def test_unit_tests_uses_sys_executable(tmp_path, monkeypatch):
    test_file = tmp_path / "test_x.py"
    test_file.write_text("def test_ok():\n    assert True\n")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

        class P:
            returncode = 0

        return P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_check("", _trace(), CheckConfig(type="unit_tests", path=str(test_file), weight=1.0))
    assert captured["cmd"][0] == sys.executable
    assert "python" != captured["cmd"][0] or captured["cmd"][0].endswith("python")
    assert captured["kwargs"]["timeout"] == 120
    assert captured["kwargs"]["shell"] is False


def test_unit_tests_timeout_branch(tmp_path, monkeypatch):
    test_file = tmp_path / "test_slow.py"
    test_file.write_text("def test_ok():\n    assert True\n")

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = run_check(
        "", _trace(), CheckConfig(type="unit_tests", path=str(test_file), weight=1.0, timeout_s=5)
    )
    assert not res.passed
    assert "timed out" in res.message


# --- F4: fixture path traversal containment ---------------------------------


def test_fixture_path_rejects_traversal(tmp_path):
    settings = Settings(output_dir=str(tmp_path), fixtures_dir=str(tmp_path / "fixtures"))
    with pytest.raises(FixtureContainmentError):
        _fixture_path(settings, "../evil")
    with pytest.raises(FixtureContainmentError):
        _fixture_path(settings, "sub/evil")
    with pytest.raises(FixtureContainmentError):
        _fixture_path(settings, "..")


def test_fixture_path_normal_name_contained(tmp_path):
    settings = Settings(output_dir=str(tmp_path), fixtures_dir=str(tmp_path / "fixtures"))
    path = _fixture_path(settings, "my_test")
    assert path.name == "my_test.json"
    assert path.resolve().is_relative_to((tmp_path / "fixtures").resolve())


def test_bad_test_name_replay_fails_per_test(tmp_path):
    settings = Settings(
        output_dir=str(tmp_path), fixtures_dir=str(tmp_path / "fixtures"), replay=True
    )
    result = run_test(None, _TestConfig(name="../evil", input=""), settings, replay=True)
    assert not result.passed
    assert "ReplayFixture" in result.credit_assignment
    # Nothing was written/read outside the base dir.
    assert not (tmp_path / "evil.json").exists()


# --- F5 + F11: regex ReDoS / invalid pattern --------------------------------


def test_invalid_regex_yields_failed_check_not_crash():
    res = run_check("hello", _trace(), CheckConfig(type="regex", value="(", weight=1.0))
    assert not res.passed
    assert "invalid regex" in res.message


def test_regex_backcompat_torchrun_unchanged():
    res = run_check(
        "use torchrun now",
        _trace(),
        CheckConfig(type="regex", value="torchrun|python -m torch.distributed.run", weight=1.0),
    )
    assert res.passed and res.score == 1.0
    miss = run_check(
        "no launcher here",
        _trace(),
        CheckConfig(type="regex", value="torchrun|python -m torch.distributed.run", weight=1.0),
    )
    assert not miss.passed


def test_credit_output_satisfies_invalid_regex_not_satisfied():
    assert _output_satisfies("regex", "anything", "(") is False


# --- F6: malformed replay fixture fails per-test ----------------------------


def test_corrupt_fixture_fails_only_that_test(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "bad.json").write_text("{ this is not valid json ")
    settings = Settings(
        output_dir=str(tmp_path), fixtures_dir=str(fixtures), replay=True
    )
    result = run_test(None, _TestConfig(name="bad", input=""), settings, replay=True)
    assert not result.passed
    assert "ReplayFixture" in result.credit_assignment
    assert "corrupt" in result.credit_assignment["ReplayFixture"][0]


def test_corrupt_fixture_does_not_abort_suite(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "bad.json").write_text("not json")
    good = AgentTrace(test_name="good", final_output="ok")
    good.add_step(AgentStep(agent_name="A", input="", output="ok"))
    (fixtures / "good.json").write_text(good.model_dump_json())
    settings = Settings(output_dir=str(tmp_path), fixtures_dir=str(fixtures), replay=True)

    bad = run_test(None, _TestConfig(name="bad", input=""), settings, replay=True)
    ok = run_test(
        None,
        _TestConfig(name="good", input="", checks=[CheckConfig(type="contains", value="ok")]),
        settings,
        replay=True,
    )
    assert not bad.passed
    assert ok.passed


def test_malformed_fixture_exception_message(tmp_path):
    err = MalformedReplayFixture("t", tmp_path / "t.json", ValueError("boom"))
    assert "corrupt or invalid" in str(err)


# --- F12: numeric coercion guards -------------------------------------------


def test_max_latency_non_numeric_fails_cleanly():
    res = eval_max_latency("", _trace(), CheckConfig(type="max_latency", seconds="soon", weight=1.0))
    assert not res.passed
    assert "invalid max_latency" in res.message


def test_max_cost_non_numeric_fails_cleanly():
    res = eval_max_cost("", _trace(), CheckConfig(type="max_cost", usd="lots", weight=1.0))
    assert not res.passed
    assert "invalid max_cost" in res.message


def test_max_latency_numeric_string_still_works():
    res = eval_max_latency(
        "", _trace(latency_ms=100), CheckConfig(type="max_latency", seconds="30", weight=1.0)
    )
    assert res.passed


# --- F7: improve patch parsing robustness -----------------------------------


def test_group_patches_by_agent_normal():
    from agentgrade.cli import _group_patches_by_agent

    grouped = _group_patches_by_agent(["[CoderAgent] Ensure X.", "[CriticAgent] Ensure Y."])
    assert grouped == {"CoderAgent": ["Ensure X."], "CriticAgent": ["Ensure Y."]}


def test_group_patches_by_agent_malformed_edges():
    from agentgrade.cli import _group_patches_by_agent

    grouped = _group_patches_by_agent(
        [
            "no brackets here",
            "[] empty agent",
            123,
            None,
            "[OnlyOpen unterminated",
            "[A] item with ] bracket inside",
        ]
    )
    assert "Agent" in grouped
    assert "no brackets here" in grouped["Agent"]
    assert "empty agent" in grouped["Agent"]
    assert "[OnlyOpen unterminated" in grouped["Agent"]
    assert grouped["A"] == ["item with ] bracket inside"]


# --- F1: simple_agent back-compat reward ------------------------------------


def test_simple_agent_reward_unchanged(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    cfg = load_config(REPO_ROOT / "examples" / "simple_agent" / "agentgrade.yaml")
    result = run_suite(cfg)[0]
    assert result.reward == pytest.approx(0.60)
    assert "CoderAgent" in result.credit_assignment
    assert "CriticAgent" in result.credit_assignment
