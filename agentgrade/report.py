"""Report rendering: Rich terminal output, JSON, and Markdown."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Settings
from .credit import format_credit
from .trace import TestResult


def save_json(results: list[TestResult], settings: Settings) -> Path:
    out_dir = Path(settings.output_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "latest.json"
    payload = [r.model_dump() for r in results]
    path.write_text(json.dumps(payload, indent=2))
    return path


def _result_markdown(result: TestResult) -> list[str]:
    status = "PASS" if result.passed else "FAIL"
    lines = [
        f"## {result.name} — {status} (reward {result.reward:.2f})",
        "",
        "### Checks",
        "",
    ]
    for ev in result.evaluations:
        mark = "x" if ev.passed else " "
        lines.append(f"- [{mark}] `{ev.check_name}` (weight {ev.weight}) — {ev.message}")
    lines.append("")
    if result.credit_assignment:
        lines.append("### Root cause candidates")
        lines.append("")
        for culprit, reasons in result.credit_assignment.items():
            for reason in reasons:
                lines.append(f"- **{culprit}** → {reason}")
        lines.append("")
    if result.suggested_patches:
        lines.append("### Suggested patches")
        lines.append("")
        for patch in result.suggested_patches:
            lines.append(f"- {patch}")
        lines.append("")
    return lines


def save_markdown(results: list[TestResult], settings: Settings) -> Path:
    out_dir = Path(settings.output_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "latest.md"

    passed = sum(1 for r in results if r.passed)
    lines = [
        "# agentgrade Report",
        "",
        f"{passed}/{len(results)} tests passed.",
        "",
    ]
    for result in results:
        lines.extend(_result_markdown(result))
    path.write_text("\n".join(lines) + "\n")
    return path


def render_terminal(results: list[TestResult], settings: Settings, console: Console | None = None) -> None:
    console = console or Console()

    for result in results:
        status = "[bold green]PASS[/]" if result.passed else "[bold red]FAIL[/]"
        title = f"{status}  [bold]{result.name}[/]  reward=[bold]{result.reward:.2f}[/] (threshold {settings.fail_below_reward})"

        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("Check")
        table.add_column("Result", justify="center")
        table.add_column("Weight", justify="right")
        table.add_column("Detail")
        for ev in result.evaluations:
            mark = "[green]pass[/]" if ev.passed else "[red]fail[/]"
            table.add_row(ev.check_name, mark, f"{ev.weight}", ev.message)

        console.print(Panel(table, title=title, border_style="green" if result.passed else "red"))

        if not result.passed and result.credit_assignment:
            credit_lines = format_credit(result.credit_assignment)
            body = "\n".join(f"[yellow]•[/] [bold]{line}" for line in credit_lines)
            console.print(
                Panel(body, title="[bold yellow]Root cause candidates[/]", border_style="yellow")
            )

        if not result.passed and result.suggested_patches:
            patch_body = "\n".join(f"[cyan]+[/] {p}" for p in result.suggested_patches)
            console.print(
                Panel(patch_body, title="[bold cyan]Suggested prompt patch[/]", border_style="cyan")
            )

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    summary_style = "green" if passed == total else "red"
    console.print(
        Panel(
            f"[bold]{passed}/{total}[/] tests passed",
            border_style=summary_style,
            title="[bold]agentgrade summary[/]",
        )
    )
