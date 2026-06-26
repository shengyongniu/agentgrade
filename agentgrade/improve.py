"""Deterministic prompt-patch suggestions from failed checks.

No LLM calls: this module inspects which checks failed and turns them into a
copy-pasteable Markdown checklist that can be appended to the responsible
agent's prompt.
"""

from __future__ import annotations

from typing import Callable

from .config import CheckConfig
from .trace import EvaluationResult


PatchSuggester = Callable[[CheckConfig], "str | None"]

PATCH_SUGGESTERS: dict[str, PatchSuggester] = {}


def register_patch_suggester(
    check_type: str, fn: PatchSuggester | None = None
) -> PatchSuggester | Callable[[PatchSuggester], PatchSuggester]:
    """Register ``fn`` to supply patch-suggestion text for ``check_type``.

    Mirrors :func:`agentgrade.evaluators.register_evaluator`: plugins can teach
    ``improve`` how to phrase a prompt patch for their custom check types.
    Later registrations override earlier ones. Usable directly
    (``register_patch_suggester("t", fn)``) or as a decorator
    (``@register_patch_suggester("t")``).
    """

    def _register(target: PatchSuggester) -> PatchSuggester:
        if not callable(target):
            raise TypeError(f"patch suggester for {check_type!r} must be callable")
        PATCH_SUGGESTERS[check_type] = target
        return target

    if fn is None:
        return _register
    return _register(fn)


def _checklist_item(check: CheckConfig) -> str | None:
    value = getattr(check, "value", None)
    if check.type == "contains" and value:
        return f"Ensure the output includes `{value}`."
    if check.type == "not_contains" and value:
        return f"Ensure the output never includes `{value}`."
    if check.type == "regex" and value:
        return f"Ensure the output matches the pattern `{value}` (e.g. include a launch command)."
    if check.type == "exact_match" and value:
        return f"Ensure the output exactly equals the expected value."
    suggester = PATCH_SUGGESTERS.get(check.type)
    if suggester is not None:
        return suggester(check)
    return None


def suggest_patches(
    evaluations: list[EvaluationResult], checks: list[CheckConfig]
) -> list[str]:
    """Return a flat list of checklist suggestions for failed text checks."""

    suggestions: list[str] = []
    for evaluation, check in zip(evaluations, checks):
        if evaluation.passed:
            continue
        item = _checklist_item(check)
        if item:
            agent = check.agent_name or "Agent"
            suggestions.append(f"[{agent}] {item}")
    return suggestions


def build_patch_markdown(
    test_name: str,
    evaluations: list[EvaluationResult],
    checks: list[CheckConfig],
) -> str:
    """Build a Markdown patch document grouped by responsible agent."""

    by_agent: dict[str, list[str]] = {}
    for evaluation, check in zip(evaluations, checks):
        if evaluation.passed:
            continue
        item = _checklist_item(check)
        if not item:
            continue
        agent = check.agent_name or "Agent"
        by_agent.setdefault(agent, []).append(item)

    lines = [f"# Suggested prompt patch for `{test_name}`", ""]
    if not by_agent:
        lines.append("No deterministic text-check failures to patch.")
        return "\n".join(lines) + "\n"

    lines.append(
        "Append the following checklist items to the corresponding agent prompts:"
    )
    lines.append("")
    for agent, items in by_agent.items():
        lines.append(f"## {agent}")
        lines.append("")
        lines.append("```diff")
        lines.append("  You are a careful engineer. Before returning, verify:")
        for item in items:
            lines.append(f"+ - [ ] {item}")
        lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"
