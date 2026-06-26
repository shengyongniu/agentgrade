"""Test runner: imports the configured agent and runs each test case.

Security note: loading the agent entrypoint and any configured plugins
*imports and executes* the referenced Python modules. agentgrade must only be
pointed at code you trust. See the "Security model / trust boundary" section of
the README. A one-line warning naming the imported module(s) is printed to
stderr on each run; set ``AGENTGRADE_NO_WARN=1`` to silence it.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from .config import AgentGradeConfig, Settings, TestConfig
from .credit import assign_credit
from .evaluators import load_entry_point_evaluators, run_check
from .improve import suggest_patches
from .rewards import compute_reward
from .trace import AgentStep, AgentTrace, TestResult


class ReplayFixtureMissing(Exception):
    """Raised when a replay run cannot find a recorded fixture for a test."""

    def __init__(self, test_name: str, path: Path) -> None:
        self.test_name = test_name
        self.path = path
        super().__init__(
            f"no replay fixture for {test_name!r}; run `agentgrade record` first "
            f"(expected {path})"
        )


class FixtureContainmentError(Exception):
    """Raised when a test name resolves to a path outside the fixtures dir."""


class MalformedReplayFixture(Exception):
    """Raised when a replay fixture exists but cannot be parsed/validated."""

    def __init__(self, test_name: str, path: Path, cause: Exception) -> None:
        self.test_name = test_name
        self.path = path
        self.cause = cause
        super().__init__(
            f"replay fixture for {test_name!r} is corrupt or invalid "
            f"({path}): {type(cause).__name__}: {cause}"
        )


def _warn(message: str) -> None:
    """Print a single concise warning to stderr unless silenced via env var."""

    if os.environ.get("AGENTGRADE_NO_WARN") == "1":
        return
    print(f"agentgrade: {message}", file=sys.stderr)


def _ensure_cwd_on_path() -> None:
    """Append (not prepend) the cwd to ``sys.path``.

    Appending means installed packages keep precedence, so a repo file named
    like a stdlib/third-party module cannot hijack imports.
    """

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.append(cwd)


def _load_entrypoint(entrypoint: str) -> Callable[..., Any]:
    """Import a callable from a ``module.path:function`` entrypoint string."""

    if ":" not in entrypoint:
        raise ValueError(
            f"invalid entrypoint {entrypoint!r}; expected 'module.path:function'"
        )
    module_path, func_name = entrypoint.split(":", 1)

    _ensure_cwd_on_path()

    _warn(
        f"importing agent entrypoint {module_path!r} "
        "— this executes code from your config"
    )
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"entrypoint {entrypoint!r} is not callable")
    return func


def load_plugins(plugins: list[str]) -> None:
    """Import config-declared plugins so they register custom evaluators.

    Each entry is ``module.path:function`` (the function is imported and
    called to perform registration) or a bare ``module.path`` (importing the
    module is enough, e.g. it uses the ``@evaluator`` decorator at import time).
    The current working directory is appended to ``sys.path`` so local plugin
    modules resolve, mirroring the entrypoint loader.
    """

    if not plugins:
        return

    _ensure_cwd_on_path()

    module_names = [spec.split(":", 1)[0] for spec in plugins]
    _warn(
        "importing plugin module(s) "
        + ", ".join(repr(name) for name in module_names)
        + " — this executes code from your config"
    )

    for spec in plugins:
        if ":" in spec:
            module_path, func_name = spec.split(":", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            if callable(func):
                func()
        else:
            importlib.import_module(spec)


def _coerce_trace(test_name: str, raw_output: Any) -> tuple[str, AgentTrace]:
    """Normalise an agent's return value into ``(final_output, AgentTrace)``.

    Supports:
    - ``(final_output, trace)`` tuple where trace is an ``AgentTrace`` or dict
    - a bare string (wrapped in a single-step trace)
    """

    final_output = ""
    trace: AgentTrace | None = None

    if isinstance(raw_output, tuple) and len(raw_output) == 2:
        final_output, maybe_trace = raw_output
        if isinstance(maybe_trace, AgentTrace):
            trace = maybe_trace
        elif isinstance(maybe_trace, dict):
            trace = AgentTrace.model_validate(maybe_trace)
    elif isinstance(raw_output, AgentTrace):
        trace = raw_output
        final_output = raw_output.final_output
    else:
        final_output = str(raw_output)

    if trace is None:
        trace = AgentTrace(test_name=test_name, final_output=str(final_output))
        trace.add_step(
            AgentStep(
                agent_name="FinalAgent",
                input="",
                output=str(final_output),
            )
        )
    else:
        trace.test_name = test_name
        if not final_output:
            final_output = trace.final_output
        trace.final_output = str(final_output)

    return str(final_output), trace


def _sanitize_test_name(test_name: str) -> str:
    """Reject test names that would escape the fixtures directory.

    Path separators, ``..`` segments, leading separators, and control
    characters are not allowed in a fixture file stem. We keep the historical
    behaviour (the name is used verbatim as ``<name>.json``) for well-behaved
    names and only raise on clearly unsafe ones, so existing fixtures are
    unaffected.
    """

    if not test_name:
        raise FixtureContainmentError("test name must not be empty")
    if "/" in test_name or "\\" in test_name or "\x00" in test_name:
        raise FixtureContainmentError(
            f"test name {test_name!r} contains path separators or control chars"
        )
    if any(ord(ch) < 32 for ch in test_name):
        raise FixtureContainmentError(
            f"test name {test_name!r} contains control characters"
        )
    if test_name in {".", ".."} or test_name.startswith(".."):
        raise FixtureContainmentError(
            f"test name {test_name!r} resolves outside the fixtures directory"
        )
    return test_name


def _fixture_path(settings: Settings, test_name: str) -> Path:
    """Return the fixture path for ``test_name``, contained within the base dir.

    Sanitizes the name and asserts (via ``Path.resolve`` + ``is_relative_to``)
    that the final path cannot escape the resolved fixtures directory.
    """

    safe_name = _sanitize_test_name(test_name)
    base = settings.resolved_fixtures_dir()
    base_resolved = base.resolve()
    candidate = (base / f"{safe_name}.json").resolve()
    if not candidate.is_relative_to(base_resolved):
        raise FixtureContainmentError(
            f"fixture path for {test_name!r} escapes {base_resolved}"
        )
    return candidate


def _acquire_run(
    agent: Callable[..., Any] | None,
    test: TestConfig,
    settings: Settings,
    replay: bool,
) -> tuple[str, AgentTrace, str | None, int]:
    """Obtain ``(final_output, trace, error, wall_ms)`` for one test.

    In replay mode the real agent is never invoked; the recorded fixture is
    loaded and reconstructed instead. Otherwise the agent entrypoint is called
    as usual and its return value coerced into a trace.
    """

    if replay:
        path = _fixture_path(settings, test.name)
        if not path.exists():
            raise ReplayFixtureMissing(test.name, path)
        try:
            trace = AgentTrace.model_validate_json(path.read_text())
        except (ValidationError, ValueError, OSError) as exc:
            raise MalformedReplayFixture(test.name, path, exc) from exc
        trace.test_name = test.name
        return trace.final_output, trace, trace.error, trace.total_latency_ms

    assert agent is not None
    start = time.perf_counter()
    error: str | None = None
    try:
        raw = agent(test.input)
    except Exception:  # noqa: BLE001 - capture agent crashes as a failed test
        error = traceback.format_exc()
        raw = ""
    wall_ms = int((time.perf_counter() - start) * 1000)

    final_output, trace = _coerce_trace(test.name, raw)
    return final_output, trace, error, wall_ms


def run_test(
    agent: Callable[..., Any] | None,
    test: TestConfig,
    settings: Settings,
    replay: bool = False,
) -> TestResult:
    try:
        final_output, trace, error, wall_ms = _acquire_run(agent, test, settings, replay)
    except ReplayFixtureMissing as exc:
        empty = AgentTrace(test_name=test.name, error=str(exc))
        return TestResult(
            name=test.name,
            input=test.input,
            output="",
            reward=0.0,
            passed=False,
            evaluations=[],
            trace=empty,
            credit_assignment={"ReplayFixture": [str(exc)]},
            suggested_patches=[],
        )
    except (MalformedReplayFixture, FixtureContainmentError) as exc:
        empty = AgentTrace(test_name=test.name, error=str(exc))
        return TestResult(
            name=test.name,
            input=test.input,
            output="",
            reward=0.0,
            passed=False,
            evaluations=[],
            trace=empty,
            credit_assignment={"ReplayFixture": [str(exc)]},
            suggested_patches=[],
        )

    if trace.total_latency_ms == 0:
        # No per-step latency was recorded (steps reported 0ms each), so fall
        # back to measured wall-clock time. A trace that reports any nonzero
        # per-step latency is trusted as-is and keeps its summed total.
        trace.total_latency_ms = wall_ms
    if error:
        trace.error = error

    evaluations = [run_check(final_output, trace, check) for check in test.checks]
    reward = compute_reward(evaluations)
    passed = error is None and reward >= settings.fail_below_reward

    credit = assign_credit(evaluations, test.checks, trace)
    patches = suggest_patches(evaluations, test.checks)

    return TestResult(
        name=test.name,
        input=test.input,
        output=final_output,
        reward=reward,
        passed=passed,
        evaluations=evaluations,
        trace=trace,
        credit_assignment={k: v for k, v in credit.items()},
        suggested_patches=patches,
    )


def run_suite(config: AgentGradeConfig, replay: bool | None = None) -> list[TestResult]:
    load_entry_point_evaluators()
    load_plugins(config.plugins)
    use_replay = config.settings.replay if replay is None else replay
    agent = None if use_replay else _load_entrypoint(config.agent.entrypoint)
    return [run_test(agent, test, config.settings, use_replay) for test in config.tests]


def record_suite(config: AgentGradeConfig) -> list[Path]:
    """Run the real agent and persist each test's trace as a replay fixture.

    Returns the list of fixture paths written.
    """

    load_entry_point_evaluators()
    load_plugins(config.plugins)
    agent = _load_entrypoint(config.agent.entrypoint)

    fixtures_dir = config.settings.resolved_fixtures_dir()
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for test in config.tests:
        final_output, trace, error, wall_ms = _acquire_run(
            agent, test, config.settings, replay=False
        )
        if trace.total_latency_ms == 0:
            trace.total_latency_ms = wall_ms
        if error:
            trace.error = error
        trace.final_output = final_output
        path = _fixture_path(config.settings, test.name)
        path.write_text(trace.model_dump_json(indent=2))
        written.append(path)
    return written
