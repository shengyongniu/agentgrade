import json
import subprocess
from pathlib import Path

import pytest
import yaml

from agentgrade.config import CheckConfig, load_config
from agentgrade.improve import (
    PATCH_SUGGESTERS,
    build_patch_markdown,
    register_patch_suggester,
    suggest_patches,
)
from agentgrade.runner import record_suite, run_suite
from agentgrade.trace import EvaluationResult

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_CLI = REPO_ROOT / ".venv" / "bin" / "agentgrade"


def _simple_config(tmp_path: Path, *, entrypoint: str | None = None, replay: bool = False) -> Path:
    """Write a copy of the simple_agent config into ``tmp_path`` with overrides."""

    base = yaml.safe_load(
        (REPO_ROOT / "examples" / "simple_agent" / "agentgrade.yaml").read_text()
    )
    if entrypoint is not None:
        base["agent"]["entrypoint"] = entrypoint
    base["settings"]["output_dir"] = str(tmp_path / ".agentgrade")
    base["settings"]["replay"] = replay
    cfg_path = tmp_path / "agentgrade.yaml"
    cfg_path.write_text(yaml.safe_dump(base))
    return cfg_path


def test_record_writes_fixture_and_replay_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)

    cfg_path = _simple_config(tmp_path)
    cfg = load_config(cfg_path)

    paths = record_suite(cfg)
    assert len(paths) == 1
    fixture = paths[0]
    assert fixture.exists()
    assert fixture.name == "ddp_training_script.json"

    real = run_suite(cfg)[0]
    assert real.reward == pytest.approx(0.60)

    # Point the entrypoint at a broken module: replay must NOT import it.
    replay_cfg = load_config(
        _simple_config(tmp_path, entrypoint="this.module:does_not_exist", replay=True)
    )
    replayed = run_suite(replay_cfg)[0]

    assert replayed.reward == pytest.approx(real.reward)
    assert replayed.credit_assignment == real.credit_assignment
    assert set(replayed.trace.agent_names()) == {"CoderAgent", "CriticAgent"}
    assert not replayed.passed


def test_replay_flag_overrides_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    cfg = load_config(_simple_config(tmp_path))
    record_suite(cfg)

    broken = load_config(_simple_config(tmp_path, entrypoint="nope.broken:missing"))
    replayed = run_suite(broken, replay=True)[0]
    assert replayed.reward == pytest.approx(0.60)


def test_missing_fixture_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    cfg = load_config(_simple_config(tmp_path, replay=True))

    results = run_suite(cfg)
    assert len(results) == 1
    result = results[0]
    assert not result.passed
    assert result.reward == 0.0
    assert "ReplayFixture" in result.credit_assignment
    msg = result.credit_assignment["ReplayFixture"][0]
    assert "no replay fixture" in msg
    assert "ddp_training_script" in msg


def test_register_patch_suggester_hook():
    PATCH_SUGGESTERS.pop("json_schema", None)

    @register_patch_suggester("json_schema")
    def _suggest(check: CheckConfig) -> str:
        return f"Ensure the output validates against schema `{getattr(check, 'schema_path', '?')}`."

    assert "json_schema" in PATCH_SUGGESTERS

    checks = [CheckConfig(type="json_schema", weight=1.0, agent_name="SchemaAgent", schema_path="user.json")]
    evals = [
        EvaluationResult(
            check_name="json_schema", passed=False, score=0.0, weight=1.0, message="invalid"
        )
    ]

    patches = suggest_patches(evals, checks)
    assert patches and "SchemaAgent" in patches[0] and "user.json" in patches[0]

    md = build_patch_markdown("custom", evals, checks)
    assert "SchemaAgent" in md and "user.json" in md and "```diff" in md


def test_builtin_suggesters_unchanged_for_custom_types():
    checks = [CheckConfig(type="totally_unknown_type", weight=1.0)]
    evals = [
        EvaluationResult(
            check_name="x", passed=False, score=0.0, weight=1.0, message="nope"
        )
    ]
    assert suggest_patches(evals, checks) == []


def test_json_cli_output_matches_latest_json(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    cfg_path = _simple_config(tmp_path)
    cfg = load_config(cfg_path)

    proc = subprocess.run(
        [str(VENV_CLI), "test", "--config", str(cfg_path), "--json"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 1

    payload = json.loads(proc.stdout)
    assert isinstance(payload, list) and payload
    assert payload[0]["name"] == "ddp_training_script"
    assert not payload[0]["passed"]

    latest = json.loads(
        (Path(cfg.settings.output_dir) / "results" / "latest.json").read_text()
    )
    assert payload == latest
