"""Evaluators (checks) for agentgrade.

Each evaluator takes the agent output, the trace, and the check config, and
returns an :class:`EvaluationResult` with a score in the range 0.0-1.0.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .config import CheckConfig
from .trace import AgentTrace, EvaluationResult

MAX_REGEX_INPUT_CHARS = 1_000_000
"""Cap on output length fed to ``re.search`` to bound ReDoS exposure.

User-supplied patterns from a trusted config are run against agent output;
capping the searched text limits worst-case backtracking on large outputs.
"""

DEFAULT_UNIT_TESTS_TIMEOUT_S = 120


def _result(name: str, passed: bool, score: float, weight: float, message: str) -> EvaluationResult:
    return EvaluationResult(
        check_name=name, passed=passed, score=score, weight=weight, message=message
    )


def eval_contains(output: str, trace: AgentTrace, check: CheckConfig) -> EvaluationResult:
    value = str(getattr(check, "value", ""))
    passed = value in output
    return _result(
        f"contains:{value}",
        passed,
        1.0 if passed else 0.0,
        check.weight,
        f"output contains {value!r}" if passed else f"output is missing {value!r}",
    )


def eval_not_contains(output: str, trace: AgentTrace, check: CheckConfig) -> EvaluationResult:
    value = str(getattr(check, "value", ""))
    passed = value not in output
    return _result(
        f"not_contains:{value}",
        passed,
        1.0 if passed else 0.0,
        check.weight,
        f"output correctly omits {value!r}" if passed else f"output unexpectedly contains {value!r}",
    )


def eval_regex(output: str, trace: AgentTrace, check: CheckConfig) -> EvaluationResult:
    pattern = str(getattr(check, "value", getattr(check, "pattern", "")))
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return _result(
            f"regex:{pattern}",
            False,
            0.0,
            check.weight,
            f"invalid regex /{pattern}/: {exc}",
        )
    passed = compiled.search(output[:MAX_REGEX_INPUT_CHARS]) is not None
    return _result(
        f"regex:{pattern}",
        passed,
        1.0 if passed else 0.0,
        check.weight,
        f"output matches /{pattern}/" if passed else f"output does not match /{pattern}/",
    )


def eval_exact_match(output: str, trace: AgentTrace, check: CheckConfig) -> EvaluationResult:
    value = str(getattr(check, "value", ""))
    passed = output.strip() == value.strip()
    return _result(
        "exact_match",
        passed,
        1.0 if passed else 0.0,
        check.weight,
        "output exactly matches expected" if passed else "output does not exactly match expected",
    )


def eval_max_latency(output: str, trace: AgentTrace, check: CheckConfig) -> EvaluationResult:
    try:
        limit_s = float(getattr(check, "seconds", 0) or 0)
    except (TypeError, ValueError):
        return _result(
            "max_latency",
            False,
            0.0,
            check.weight,
            f"invalid max_latency 'seconds' value: {getattr(check, 'seconds', None)!r}",
        )
    limit_ms = limit_s * 1000.0
    actual_ms = trace.total_latency_ms
    passed = actual_ms <= limit_ms
    return _result(
        "max_latency",
        passed,
        1.0 if passed else 0.0,
        check.weight,
        f"latency {actual_ms}ms <= {limit_ms:.0f}ms"
        if passed
        else f"latency {actual_ms}ms exceeds {limit_ms:.0f}ms",
    )


def eval_max_cost(output: str, trace: AgentTrace, check: CheckConfig) -> EvaluationResult:
    try:
        limit = float(getattr(check, "usd", 0) or 0)
    except (TypeError, ValueError):
        return _result(
            "max_cost",
            False,
            0.0,
            check.weight,
            f"invalid max_cost 'usd' value: {getattr(check, 'usd', None)!r}",
        )
    actual = trace.total_cost_usd
    passed = actual <= limit
    return _result(
        "max_cost",
        passed,
        1.0 if passed else 0.0,
        check.weight,
        f"cost ${actual:.4f} <= ${limit:.2f}"
        if passed
        else f"cost ${actual:.4f} exceeds ${limit:.2f}",
    )


def eval_python_import_check(output: str, trace: AgentTrace, check: CheckConfig) -> EvaluationResult:
    module = str(getattr(check, "module", getattr(check, "value", "")))
    try:
        importlib.import_module(module)
        passed = True
        message = f"module {module!r} importable"
    except Exception as exc:  # noqa: BLE001 - report any import failure
        passed = False
        message = f"module {module!r} not importable: {exc}"
    return _result(
        f"python_import_check:{module}", passed, 1.0 if passed else 0.0, check.weight, message
    )


def eval_unit_tests(output: str, trace: AgentTrace, check: CheckConfig) -> EvaluationResult:
    path = getattr(check, "path", None)
    if not path or not Path(path).exists():
        return _result(
            "unit_tests",
            False,
            0.0,
            check.weight,
            f"test path {path!r} not found",
        )
    try:
        timeout_s = float(getattr(check, "timeout_s", DEFAULT_UNIT_TESTS_TIMEOUT_S) or DEFAULT_UNIT_TESTS_TIMEOUT_S)
    except (TypeError, ValueError):
        timeout_s = float(DEFAULT_UNIT_TESTS_TIMEOUT_S)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(path), "-q"],
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return _result(
            "unit_tests",
            False,
            0.0,
            check.weight,
            f"unit tests timed out after {timeout_s:.0f}s",
        )
    passed = proc.returncode == 0
    return _result(
        "unit_tests",
        passed,
        1.0 if passed else 0.0,
        check.weight,
        "unit tests passed" if passed else f"unit tests failed (exit {proc.returncode})",
    )


Evaluator = Callable[[str, AgentTrace, CheckConfig], EvaluationResult]

EVALUATORS: dict[str, Evaluator] = {}


def register_evaluator(name: str, fn: Evaluator) -> Evaluator:
    """Register ``fn`` as the evaluator for check type ``name``.

    Returns ``fn`` so it can be used both as a plain call and inside a
    decorator. Later registrations override earlier ones, letting plugins
    replace built-ins intentionally.
    """

    if not callable(fn):
        raise TypeError(f"evaluator for {name!r} must be callable")
    EVALUATORS[name] = fn
    return fn


def evaluator(name: str) -> Callable[[Evaluator], Evaluator]:
    """Decorator form of :func:`register_evaluator`.

    Example::

        @evaluator("min_length")
        def eval_min_length(output, trace, check):
            ...
    """

    def decorator(fn: Evaluator) -> Evaluator:
        return register_evaluator(name, fn)

    return decorator


for _name, _fn in {
    "contains": eval_contains,
    "not_contains": eval_not_contains,
    "regex": eval_regex,
    "exact_match": eval_exact_match,
    "max_latency": eval_max_latency,
    "max_cost": eval_max_cost,
    "python_import_check": eval_python_import_check,
    "unit_tests": eval_unit_tests,
}.items():
    register_evaluator(_name, _fn)


def load_entry_point_evaluators() -> None:
    """Auto-register evaluators advertised via the ``agentgrade.evaluators`` group.

    Best-effort: any failure (no metadata, broken plugin) is swallowed so a
    missing or misbehaving plugin never crashes a run.
    """

    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        selected = (
            eps.select(group="agentgrade.evaluators")
            if hasattr(eps, "select")
            else eps.get("agentgrade.evaluators", [])  # type: ignore[attr-defined]
        )
        for ep in selected:
            try:
                ep.load()
            except Exception:  # noqa: BLE001 - one bad plugin must not break others
                continue
    except Exception:  # noqa: BLE001 - importlib metadata unavailable
        return


def run_check(output: str, trace: AgentTrace, check: CheckConfig) -> EvaluationResult:
    evaluator = EVALUATORS.get(check.type)
    if evaluator is None:
        return _result(
            f"unknown:{check.type}",
            False,
            0.0,
            check.weight,
            f"unknown check type {check.type!r}",
        )
    return evaluator(output, trace, check)
